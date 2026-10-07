//===- PlanLP6.cpp - hyb-plan-lp6 --------------------------------------------===//
//
// Runs on the bufferized module. Every buffer of the step function (arguments, allocations,
// returned values) lives in LP6:
//   * read by a PIM phase (the INT8 activation of a GEMV group)  -> PIM activation packets,
//     one copy per partition that runs a stream of the group, replicated to the 8 PUs;
//   * written by a PIM phase (INT32 GEMV results)                  -> PIM output tiles;
//   * everything else (inputs, KV cache, 1-D weights, phase-crossing FP16/INT8 values,
//     outputs) and the named constants (per-row weight scales)    -> dense, row-major.
// The dense region starts after a zero page and the PIM kernel rows follow it. The PIM tile
// plan (npu/Decode/Target/PimLayout.h) is stored on the module as JSON; each buffer gets
// `hyb.lp6 = {addr, bytes, layout, ...}`.
//===----------------------------------------------------------------------===//

#include "npu/Decode/Support/TargetSpec.h"
#include "npu/Decode/PimMetadata.h"
#include "npu/Decode/Transforms/Passes.h"
#include "mlir/IR/SymbolTable.h"
#include <limits>
#include <optional>
#include <set>
#include "llvm/ADT/StringSet.h"
#include <map>
#include "llvm/Support/raw_ostream.h"

namespace mlir::hybrid {
#define GEN_PASS_DEF_HYBPLANLP6
#include "npu/Decode/Transforms/Passes.h.inc"
} // namespace mlir::hybrid

using namespace mlir;
using namespace mlir::hybrid;
namespace pim = ::npu::pim;

namespace {

int64_t bytesOf(Type t) {
  auto m = cast<MemRefType>(t);
  int64_t eb = std::max<int64_t>(1, m.getElementTypeBitWidth() / 8);
  return m.getNumElements() * eb;
}

struct HybPlanLP6Pass : mlir::hybrid::impl::HybPlanLP6Base<HybPlanLP6Pass> {
  using Base::Base;
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    Builder b(ctx);
    std::string err;
    auto specOr = TargetSpec::load(spec, err);
    if (failed(specOr)) {
      module.emitError(err);
      return signalPassFailure();
    }
    func::FuncOp step = getStepFunction(module);
    SymbolTable symbols(module);
    pim::Program program;
    if (banksetPingPong < 0 || banksetPingPong > 63) {
      module.emitError("bankset-pingpong must be 0..63 (1 selects lead 12)");
      return signalPassFailure();
    }
    program.banksetPingPongLead = banksetPingPong == 1 ? 12 : banksetPingPong;
    program.channels = specOr->pim.channels;
    program.partitions = specOr->pim.partitions;
    if (program.channels < 1 || program.partitions < 1 || program.partitions > program.channels * pim::kRanks) {
      module.emitError() << "pim.partitions " << program.partitions << " must be 1 .. pim.channels x " << pim::kRanks;
      return signalPassFailure();
    }
    const bool fixedStorage = module->hasAttr("hyb.pim_weight_storage_requested");
    auto packetRequest = module->getAttrOfType<StringAttr>("hyb.pim_packet_npu_requested");
    const bool packetRequested = bool(packetRequest);
    if (module->hasAttr("hyb.pim_packet_npu_requested") &&
        (!packetRequest || packetRequest.getValue() != "g4")) {
      module.emitError("packet NPU requires the driver-owned g4 request");
      return signalPassFailure();
    }
    std::optional<::npu::decode::PimWeightStorage> importedStorage;
    if (fixedStorage && (specOr->pim.pus != pim::kPUs || specOr->pim.lanes != pim::kLanes ||
                         specOr->pim.ranks != pim::kRanks || (program.channels != 2 && program.channels != 8))) {
      module.emitError("PIM weight storage requires the supported PU8/lane4/rank2, SC2/8 hardware");
      return signalPassFailure();
    }
    if (auto input = module->getAttrOfType<StringAttr>("hyb.pim_weight_storage_import")) {
      auto json = llvm::json::parse(input.getValue());
      if (!json) {
        llvm::consumeError(json.takeError());
        module.emitError("invalid PIM weight storage import JSON");
        return signalPassFailure();
      }
      auto parsed = ::npu::decode::parsePimWeightStorage(*json);
      if (!parsed) { module.emitError(parsed.error); return signalPassFailure(); }
      importedStorage = std::move(parsed.value);
      if (importedStorage->capacityBytes != specOr->lp6.capacityBytes) {
        module.emitError("PIM weight storage target capacity differs from import");
        return signalPassFailure();
      }
    }
    if (packetRequested && (!fixedStorage || !importedStorage || specOr->npu.cores != 4 ||
                            specOr->npu.saRows != 32 || specOr->npu.saCols != 32 ||
                            program.channels != 8 || program.banksetPingPongLead != 0)) {
      module.emitError("packet NPU g4 requires imported PIM storage, 4 cores, a 32x32 SA, 8 SC and even bankset");
      return signalPassFailure();
    }

    DenseMap<Value, DictionaryAttr> pimBuffers;   // PIM-laid-out buffers
    int64_t records = 0;
    int64_t packetPhases = 0;
    bool placementOK = true;
    step.walk([&](func::CallOp call) {
      if (!placementOK) return;
      auto callee = symbols.lookup<func::FuncOp>(call.getCallee());
      if (!callee || !callee->hasAttr(kPhase)) return;
      const StringRef phase = cast<StringAttr>(callee->getAttr(kPhase)).getValue();
      if (auto kernel = callee->getAttrOfType<StringAttr>("hyb.npu_kernel")) {
        auto fail = [&](StringRef message) { call.emitError(message); placementOK = false; };
        if (kernel.getValue() != "pim_packet_g4" || phase != "npu" || !packetRequested || ++packetPhases != 1) {
          fail("packet NPU supports exactly one driver-requested pim_packet_g4 NPU phase");
          return;
        }
        SmallVector<hyb::QMatmulOp> ops;
        callee.walk([&](hyb::QMatmulOp q) { ops.push_back(q); });
        if (ops.size() != 1) { fail("packet NPU phase must contain exactly one qmatmul"); return; }
        hyb::QMatmulOp q = ops.front();
        auto kind = q->getAttrOfType<StringAttr>("hyb.npu_packet");
        auto x = dyn_cast<BlockArgument>(q.getInput()), out = dyn_cast<BlockArgument>(q.getInit());
        auto xt = dyn_cast<MemRefType>(q.getInput().getType()), yt = dyn_cast<MemRefType>(q.getInit().getType());
        if (!kind || kind.getValue() != "g4" || !q.getTarget() || *q.getTarget() != "npu" ||
            !x || !out || x.getOwner() != &callee.getBody().front() || out.getOwner() != x.getOwner() || x == out ||
            !xt || !yt || xt.getRank() != 2 || yt.getRank() != 2 || !xt.hasStaticShape() || !yt.hasStaticShape() ||
            !xt.getElementType().isInteger(8) || !yt.getElementType().isInteger(32)) {
          fail("packet NPU needs direct phase arguments I8[M,K] input and I32[M,N] output");
          return;
        }
        const int64_t M = xt.getDimSize(0), K = xt.getDimSize(1), N = yt.getDimSize(1);
        if (M < 1 || M > 8 || yt.getDimSize(0) != M || N <= 0 || N % 4096 || K <= 0 || K % 256) {
          fail("packet NPU g4 requires M in 1..8, N % 4096 == 0 and K % 256 == 0");
          return;
        }
        // Reuse the physical weight descriptor without creating a PIM invocation:
        // no activation IDs, output tiles, phase range or executable PIM primitive.
        std::map<int, int64_t> loads;
        int64_t pi = program.addProjection(q.getWeight().str(), N, K, {}, loads);
        program.projections[size_t(pi)].computeFirst = records;
        program.projections[size_t(pi)].computeCount = 0;
        q->setAttr("hyb.packet_projection", b.getI64IntegerAttr(pi));
        return;
      }
      if (phase != "pim") return;
      // The group's INT8 input: one activation vector per token.
      Value qx = call.getOperand(0);
      auto qxT = cast<MemRefType>(qx.getType());
      int64_t M = qxT.getDimSize(0), K = qxT.getDimSize(1);
      std::vector<int64_t> acts;
      for (int64_t m = 0; m < M; ++m) acts.push_back(program.addActivation(K));
      std::map<int, int64_t> loads;
      if (!specOr->pim.concurrentRanks && program.partitions > program.channels) {
        // One rank per phase: the rank holding fewer weight tiles; the others' partitions are closed.
        auto stored = [&](int rk) {
          int64_t t = 0;
          for (int c = 0; c < program.partitions; ++c)
            if (program.partRank(c) == rk && c < int(program.nextWeight.size())) t += program.nextWeight[size_t(c)];
          return t;
        };
        int rank = 0;
        for (int rk = 1; rk <= program.partRank(program.partitions - 1); ++rk)
          if (stored(rk) < stored(rank)) rank = rk;
        for (int c = 0; c < program.partitions; ++c)
          if (program.partRank(c) != rank) loads[c] = std::numeric_limits<int64_t>::max() / 4;
      }
      int64_t first = records;
      callee.walk([&](hyb::QMatmulOp q) {
        auto out = cast<BlockArgument>(q.getInit());
        auto outT = cast<MemRefType>(out.getType());
        int64_t pi = program.addProjection(q.getWeight().str(), outT.getDimSize(1), K, acts, loads);
        const pim::Projection &p = program.projections[pi];
        program.projections[pi].computeFirst = records;
        int64_t count = p.commands(); // final executable primitive count
        program.projections[pi].computeCount = count;
        records += count;
        pimBuffers[call.getOperand(out.getArgNumber())] = b.getDictionaryAttr(
            {b.getNamedAttr("layout", b.getStringAttr("pim_out")), b.getNamedAttr("proj", b.getI64IntegerAttr(pi))});
      });
      program.phases.push_back({first, records - first});
      call->setAttr("hyb.pim_phase", b.getI64IntegerAttr(int64_t(program.phases.size()) - 1));
      pimBuffers[qx] = b.getDictionaryAttr({b.getNamedAttr("layout", b.getStringAttr("pim_act")),
                                            b.getNamedAttr("acts", b.getI64ArrayAttr(acts))});
    });
    if (!placementOK) return signalPassFailure();
    if (packetRequested && (packetPhases != 1 || program.projections.size() != 1 || !program.phases.empty() ||
                            !program.activations.empty() || program.nextActivation || program.nextOutput || records)) {
      module.emitError("packet NPU first version needs one storage-only projection and no PIM invocation");
      return signalPassFailure();
    }

    std::set<std::string> fixedNames;
    if (fixedStorage) {
      for (const auto &p : program.projections)
        if (!fixedNames.insert(p.name).second || p.n <= 0 || p.k <= 0 || p.k % pim::kKPerSweep) {
          module.emitError("PIM weight storage needs unique projection names, positive shapes and K % 256 == 0");
          return signalPassFailure();
        }
      if (fixedNames.empty()) {
        module.emitError("PIM weight storage requires at least one PIM projection");
        return signalPassFailure();
      }
    }
    if (importedStorage) {
      auto bound = ::npu::decode::bindPimWeightStorage(program, *importedStorage);
      if (!bound) { module.emitError(bound.error); return signalPassFailure(); }
      if (!specOr->pim.concurrentRanks)
        for (auto [first, count] : program.phases) {
          std::set<int> ranks;
          for (const auto &p : program.projections)
            if (p.computeCount > 0 && p.computeFirst >= first && p.computeFirst < first + count)
              for (int part : p.streamPart) ranks.insert(program.partRank(part));
          if (ranks.size() > 1) {
            module.emitError("PIM weight storage import makes a phase span non-concurrent ranks");
            return signalPassFailure();
          }
        }
    }
    if (packetRequested) {
      std::set<int> channels;
      for (int part : program.projections.front().streamPart) {
        if (program.partRank(part) != 0) {
          module.emitError("packet NPU first version only supports imported rank 0 storage");
          return signalPassFailure();
        }
        channels.insert(program.partChannel(part));
      }
      if (channels.size() != 8) {
        module.emitError("packet NPU first version needs all 8 subchannels in imported storage");
        return signalPassFailure();
      }
    }

    // Dense region.
    uint64_t cursor = 0x10000;
    auto place = [&](int64_t bytes) {
      uint64_t a = (cursor + 255) / 256 * 256;
      cursor = a + bytes;
      return a;
    };
    auto denseAttr = [&](int64_t bytes, StringRef kind, StringRef name) {
      uint64_t a = place(bytes);
      SmallVector<NamedAttribute> attrs = {b.getNamedAttr("layout", b.getStringAttr("dense")),
                                           b.getNamedAttr("addr", b.getI64IntegerAttr(int64_t(a))),
                                           b.getNamedAttr("bytes", b.getI64IntegerAttr(bytes)),
                                           b.getNamedAttr("kind", b.getStringAttr(kind))};
      if (!name.empty()) attrs.push_back(b.getNamedAttr("name", b.getStringAttr(name)));
      return b.getDictionaryAttr(attrs);
    };
    for (BlockArgument a : step.getArguments()) {
      auto kind = step.getArgAttrOfType<StringAttr>(a.getArgNumber(), kKind);
      auto name = step.getArgAttrOfType<StringAttr>(a.getArgNumber(), kName);
      step.setArgAttr(a.getArgNumber(), kLP6, denseAttr(bytesOf(a.getType()), kind ? kind.getValue() : "input",
                                                        name ? name.getValue() : ""));
    }
    // Returned buffers are outputs.
    auto ret = cast<func::ReturnOp>(step.getBody().front().getTerminator());
    DenseMap<Value, std::string> outputs;
    for (auto [i, v] : llvm::enumerate(ret.getOperands())) {
      auto name = step.getResultAttrOfType<StringAttr>(i, kName);
      outputs[v] = name ? name.str() : ("out" + std::to_string(i));
    }
    step.walk([&](memref::AllocOp alloc) {
      Value v = alloc.getResult();
      if (pimBuffers.count(v)) return;
      auto it = outputs.find(v);
      alloc->setAttr(kLP6, denseAttr(bytesOf(v.getType()), it != outputs.end() ? "output" : "activation",
                                     it != outputs.end() ? StringRef(it->second) : StringRef()));
    });
    // Weights read by NPU phases (hyb.weight): one LP6 copy per (name, layout). Small ones (row
    // scales) stay in the dense region; NPU GEMV blocks follow the NPU constant-pool reserve.
    struct W {
      std::string name, layout;
      int64_t bytes;
      SmallVector<int64_t> shape;
      uint64_t addr = 0;
    };
    std::vector<W> weights;
    std::map<std::pair<std::string, std::string>, size_t> weightIndex;
    SmallVector<hyb::WeightOp> weightOps;
    module.walk([&](hyb::WeightOp w) {
      weightOps.push_back(w);
      auto key = std::make_pair(w.getName().str(), w.getLayout().str());
      if (weightIndex.count(key)) return;
      auto t = cast<ShapedType>(w.getType());
      int64_t eb = std::max<int64_t>(1, t.getElementTypeBitWidth() / 8);
      weightIndex[key] = weights.size();
      weights.push_back({key.first, key.second, t.getNumElements() * eb, SmallVector<int64_t>(t.getShape())});
    });
    if (fixedStorage) {
      std::set<std::string> scaleNames;
      for (const W &w : weights) {
        if (w.layout != "row_scale" || !fixedNames.count(w.name)) {
          module.emitError("PIM weight storage only supports PIM matrices and their row_scale; npu_nk32 is unsupported");
          return signalPassFailure();
        }
        scaleNames.insert(w.name);
      }
      if (scaleNames != fixedNames) {
        module.emitError("PIM weight storage needs a row_scale for every projection");
        return signalPassFailure();
      }
    }
    const uint64_t mutableEnd = cursor;
    for (W &w : weights)
      if (w.layout == "row_scale") {
        if (!importedStorage) { w.addr = place(w.bytes); continue; }
        const auto it = std::find_if(importedStorage->weights.begin(), importedStorage->weights.end(),
                                    [&](const auto &e) { return e.name == w.name; });
        if (it == importedStorage->weights.end() || w.shape.size() != 1 || w.shape[0] != it->n ||
            uint64_t(w.bytes) != it->scaleBytes || mutableEnd > it->scaleAddress) {
          module.emitError("PIM weight storage scale shape mismatch or overlap with mutable dense buffers");
          return signalPassFailure();
        }
        w.addr = it->scaleAddress;
        cursor = std::max(cursor, w.addr + it->scaleBytes);
      }
    uint64_t denseEnd = cursor;
    // Room for the NPU code generator's constant pools (one small blob per phase).
    uint64_t extraLimit = (denseEnd + 255) / 256 * 256 + (uint64_t(1) << 20);
    cursor = extraLimit;
    for (W &w : weights)
      if (w.layout != "row_scale") {
        cursor = (cursor + 4095) / 4096 * 4096;
        w.addr = place(w.bytes);
      }
    uint64_t npuWeightEnd = cursor;
    for (hyb::WeightOp w : weightOps) {
      const W &info = weights[weightIndex[{w.getName().str(), w.getLayout().str()}]];
      w->setAttr(kLP6, b.getDictionaryAttr({b.getNamedAttr("layout", b.getStringAttr("dense")),
                                           b.getNamedAttr("addr", b.getI64IntegerAttr(int64_t(info.addr))),
                                           b.getNamedAttr("bytes", b.getI64IntegerAttr(info.bytes))}));
    }
    llvm::json::Array wjson;
    for (const W &w : weights)
      wjson.push_back(llvm::json::Object{{"name", w.name}, {"layout", w.layout}, {"addr", int64_t(w.addr)},
                                         {"bytes", w.bytes}, {"shape", llvm::json::Array(w.shape)}});
    if (!importedStorage) {
      program.kernelRow = int64_t((npuWeightEnd + program.rowBytes() - 1) / program.rowBytes()) + 1;
    } else if (npuWeightEnd > uint64_t(program.kernelRow) * uint64_t(program.rowBytes())) {
      module.emitError("PIM weight storage constant reserve/dense buffers overlap fixed kernel rows");
      return signalPassFailure();
    }
    for (auto &[v, attr] : pimBuffers) {
      SmallVector<NamedAttribute> attrs(attr.getValue());
      attrs.push_back(b.getNamedAttr("bytes", b.getI64IntegerAttr(bytesOf(v.getType()))));
      v.getDefiningOp()->setAttr(kLP6, b.getDictionaryAttr(attrs));
    }
    uint64_t end = std::max<uint64_t>(program.endAddress(), npuWeightEnd);
    if (int64_t(end) > specOr->lp6.capacityBytes) {
      module.emitError() << "LP6 plan needs " << end << " bytes, capacity " << specOr->lp6.capacityBytes;
      return signalPassFailure();
    }
    // Every rank holds its PIM rows at the same row numbers, so they must end inside one rank.
    if (program.rankEndAddress() > uint64_t(pim::kRankBytes)) {
      module.emitError() << "PIM rows end at byte " << program.rankEndAddress() << " of a rank, past "
                         << pim::kRankBytes;
      return signalPassFailure();
    }
    std::string json;
    llvm::raw_string_ostream os(json);
    os << ::npu::decode::toJSON(program);
    module->setAttr("hyb.pim_plan", b.getStringAttr(json));
    {
      std::string wj;
      llvm::raw_string_ostream wos(wj);
      wos << llvm::json::Value(std::move(wjson));
      module->setAttr("hyb.weights", b.getStringAttr(wj));
    }
    module->setAttr("hyb.lp6_dense_end", b.getI64IntegerAttr(int64_t(denseEnd)));
    module->setAttr("hyb.lp6_extra_limit", b.getI64IntegerAttr(int64_t(extraLimit)));
    module->setAttr("hyb.lp6_end", b.getI64IntegerAttr(int64_t(end)));
    module->setAttr("hyb.pim_records", b.getI64IntegerAttr(records));
  }
};
} // namespace
