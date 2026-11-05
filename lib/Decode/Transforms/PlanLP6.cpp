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

#include "npu/Decode/PimMetadata.h"
#include "npu/Decode/Support/PimCost.h"
#include "npu/Decode/Transforms/NpuCost.h"
#include "npu/Decode/Transforms/PassSupport.h"
#include <cmath>
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>
#include <map>
#include <optional>
#include <set>

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
    auto specOr = loadTargetSpec(module, spec);
    if (failed(specOr)) return signalPassFailure();
    func::FuncOp step = getStepFunction(module);
    SymbolTable symbols(module);
    pim::Program program;
    if (banksetPingPong < -1 || banksetPingPong > 63) {
      module.emitError("bankset-pingpong must be -1 (target) or 0..63 (1 selects lead 12)");
      return signalPassFailure();
    }
    program.banksetPingPongLead = banksetPingPong < 0 ? specOr->pim.banksetPingPongLead
                                  : banksetPingPong == 1 ? 12 : banksetPingPong;
    program.channels = specOr->pim.channels;
    program.partitions = specOr->pim.partitions;
    program.commonLayout = commonLayout;
    program.commonTurnSweeps = commonTurnSweeps;
    if (commonLayout && (program.channels != pim::kPUs || specOr->pim.concurrentRanks)) {
      module.emitError("common-layout needs 8 subchannels and pim.concurrent_ranks = false");
      return signalPassFailure();
    }
    if (program.channels < 1 || program.partitions < 1 || program.partitions > program.channels * pim::kRanks) {
      module.emitError() << "pim.partitions " << program.partitions << " must be 1 .. pim.channels x " << pim::kRanks;
      return signalPassFailure();
    }
    // When the plan does not fit, every PIM phase reuses the same activation and output tiles (below),
    // unless a PIM output is a step result (it must survive to the end) or the driver collects
    // intermediate buffers. Plans that fit keep distinct tiles: reused rows made a decode layer's QKV
    // phase 42.2 -> 47.1 us on Ramulator (2-layer fixture, layer 1).
    bool mayReuse = reusePimBuffers;
    step.walk([&](func::CallOp call) {
      auto callee = symbols.lookup<func::FuncOp>(call.getCallee());
      if (!callee || !callee->hasAttr(kPhase) || cast<StringAttr>(callee->getAttr(kPhase)).getValue() != "pim") return;
      callee.walk([&](hyb::QMatmulOp q) {
        for (Operation *user : call.getOperand(cast<BlockArgument>(q.getInit()).getArgNumber()).getUsers())
          if (isa<func::ReturnOp>(user)) mayReuse = false;
      });
    });
    DenseMap<Value, DictionaryAttr> pimBuffers;   // PIM-laid-out buffers
    int64_t records = 0;
    bool placementOK = true;
    step.walk([&](func::CallOp call) {
      if (!placementOK) return;
      auto callee = symbols.lookup<func::FuncOp>(call.getCallee());
      if (!callee || !callee->hasAttr(kPhase)) return;
      const StringRef phase = cast<StringAttr>(callee->getAttr(kPhase)).getValue();
      if (phase != "pim") return;
      // The group's INT8 input: one activation vector per token.
      Value qx = call.getOperand(0);
      auto qxT = cast<MemRefType>(qx.getType());
      int64_t M = qxT.getDimSize(0), K = qxT.getDimSize(1);
      // A sweep loads 256 inputs. The primitives count whole sweeps, so another K would drop inputs and
      // produce wrong sums without any other sign.
      if (K % pim::kKPerSweep) {
        call.emitError() << "PIM GEMV needs K to be a multiple of " << pim::kKPerSweep << ", got " << K
                         << " (compile this group for the NPU with --map or --force=npu)";
        placementOK = false;
        return;
      }
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
        // The dense LP6 region (step inputs such as the KV cache, NPU activations) is in rank 0. When the
        // next NPU phase reads more than 64 KiB of step inputs, this phase runs on the last rank, so those
        // reads can overlap its PIM work (codegen orders only the GDMAs that touch the PIM rank after it).
        int64_t nextDense = 0;
        for (Operation *op = call->getNextNode(); op; op = op->getNextNode()) {
          auto next = dyn_cast<func::CallOp>(op);
          auto nextFn = next ? symbols.lookup<func::FuncOp>(next.getCallee()) : func::FuncOp();
          if (!nextFn || !nextFn->hasAttr(kPhase)) continue;
          if (cast<StringAttr>(nextFn->getAttr(kPhase)).getValue() != "npu") break;
          for (Value v : next.getOperands())
            if (auto arg = dyn_cast<BlockArgument>(v); arg && arg.getOwner() == &step.getBody().front())
              nextDense += bytesOf(v.getType());
          break;
        }
        const int last = program.partRank(program.partitions - 1);
        if (nextDense > (64 << 10) && last > 0) rank = last;
        cost::recordDecision(callee, "pim rank", "next NPU phase reads " + std::to_string(nextDense) + " B of step inputs",
                             "rank " + std::to_string(rank), {{"next_dense_bytes", double(nextDense)}});
        for (int c = 0; c < program.partitions; ++c)
          if (program.partRank(c) != rank) loads[c] = std::numeric_limits<int64_t>::max() / 4;
      }
      int64_t first = records;
      // Stream layout of the phase's projections (they share the input): the fastest estimate over the
      // target's accumulator counts and split-K pieces. A split-K output is a set of partial sums that
      // only the NPU gather adds, so an INT32 result the step returns as is stays whole.
      SmallVector<hyb::QMatmulOp> group;
      callee.walk([&](hyb::QMatmulOp q) { group.push_back(q); });
      std::vector<int64_t> rows;
      bool allowSplit = true;
      for (hyb::QMatmulOp q : group) {
        auto out = cast<BlockArgument>(q.getInit());
        rows.push_back(cast<MemRefType>(out.getType()).getDimSize(1));
        for (Operation *user : call.getOperand(out.getArgNumber()).getUsers())
          allowSplit &= !isa<func::ReturnOp>(user);
      }
      int open = 0;
      for (int c = 0; c < program.partitions; ++c) open += loads.count(c) == 0;
      std::vector<pimcost::Shape> shapes(group.size());
      if (commonLayout && !group.empty()) {
        int rank = 0;
        for (int c = 0; c < program.partitions; ++c)
          if (!loads.count(c)) { rank = program.partRank(c); break; }
        std::vector<std::pair<std::string, int64_t>> segments;
        std::string label;
        for (auto [i, q] : llvm::enumerate(group)) {
          label += (i ? ";" : "") + q.getWeight().str();
          segments.push_back({q.getWeight().str(), rows[i]});
        }
        const int64_t pi = program.addCommonProjection(label, segments, K, acts, rank, allowSplit);
        pim::Projection &p = program.projections[size_t(pi)];
        p.computeFirst = records;
        p.computeCount = p.commands();
        int64_t base = 0;
        for (auto [i, q] : llvm::enumerate(group)) {
          auto out = cast<BlockArgument>(q.getInit());
          pimBuffers[call.getOperand(out.getArgNumber())] = b.getDictionaryAttr(
              {b.getNamedAttr("layout", b.getStringAttr("pim_out")), b.getNamedAttr("proj", b.getI64IntegerAttr(pi)),
               b.getNamedAttr("seg_base", b.getI64IntegerAttr(base))});
          base += rows[i];
        }
        records += p.computeCount;
        cost::recordDecision(callee, "pim common layout", label,
                             "A=" + std::to_string(p.accs) + " S=" + std::to_string(p.splitK) + " tail " +
                                 std::to_string(p.tailBlocks) + "xA=" + std::to_string(p.tailAccs),
                             {{"rows", double(p.n)}, {"k", double(K)}});
      } else if (!group.empty()) {
        pimcost::Estimate best, plain = pimcost::estimate(*specOr, M, K, rows, shapes, open);
        shapes = pimcost::choose(*specOr, M, K, rows, open, allowSplit, best);
        std::string label, choice;
        for (auto [i, q] : llvm::enumerate(group)) {
          label += (i ? ";" : "") + q.getWeight().str();
          choice += (i ? ", " : "") + ("A=" + std::to_string(shapes[i].accs) + " S=" + std::to_string(shapes[i].splitK));
          if (shapes[i].tailBlocks)
            choice += " tail " + std::to_string(shapes[i].tailBlocks) + "xA=" + std::to_string(shapes[i].tailAccs);
        }
        cost::recordDecision(callee, "pim stream layout", label, choice,
                             {{"tokens", double(M)}, {"k", double(K)}, {"partitions", double(open)},
                              {"pim_us", best.pimUs}, {"gather_us", best.gatherUs}, {"split_sum_us", best.sumUs},
                              {"pim_us_512_rows", plain.pimUs}, {"gather_us_512_rows", plain.gatherUs}});
      }
      size_t qi = 0;
      if (!commonLayout) callee.walk([&](hyb::QMatmulOp q) {
        auto out = cast<BlockArgument>(q.getInit());
        auto outT = cast<MemRefType>(out.getType());
        const pimcost::Shape shape = shapes[qi++];
        const int64_t pieceSweeps = K / pim::kKPerSweep / shape.splitK;
        auto streamLoad = [&](int accs) { return pimcost::streamCycles(*specOr, M, pieceSweeps, accs); };
        // Both ranks open and an even K split: each rank computes its own K pieces and needs only their input.
        const int pieceRanks = open == program.channels * pim::kRanks && shape.splitK % pim::kRanks == 0 ? pim::kRanks : 1;
        int64_t pi = program.addProjection(q.getWeight().str(), outT.getDimSize(1), K, acts, loads, shape.accs,
                                           shape.splitK, streamLoad, shape.tailAccs, shape.tailBlocks, pieceRanks);
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
      // PIM records of the phase. The command processor reads one record before it starts, so a long record delays
      // the phase; the later records of several are read while the earlier ones compute. As many as keep the first
      // read below the target's bound, cut at K sweeps (the partition with the fewest sets the limit).
      if (!group.empty()) {
        std::map<int, int64_t> sweepsOf;
        for (int64_t pi = int64_t(program.projections.size()) - (commonLayout ? 1 : int64_t(group.size()));
             pi < int64_t(program.projections.size()); ++pi) {
          const pim::Projection &p = program.projections[size_t(pi)];
          for (int part : p.streamPart) sweepsOf[part] += int64_t(p.tokens()) * p.pieceSweeps();
        }
        int64_t rounds = std::numeric_limits<int64_t>::max();
        for (auto [part, sweeps] : sweepsOf) rounds = std::min(rounds, sweeps);
        const double fetchUs = double(records - first) * pim::kRecordWordsPerPrimitive / specOr->npu.commandFetchWordsPerCycle / specOr->npu.clockMHz;
        const int64_t wanted = int64_t(std::ceil(fetchUs / specOr->pim.recordFetchUs));
        const int64_t count = std::clamp<int64_t>(wanted, 1, std::min<int64_t>(specOr->pim.maxRecords, rounds));
        call->setAttr("hyb.pim_records", b.getI64IntegerAttr(count));
        cost::recordDecision(callee, "pim records", "PIM phase of " + std::to_string(records - first) + " primitives",
                             std::to_string(count), {{"fetch_us", fetchUs}, {"sweep_rounds", double(rounds)}});
      }
      pimBuffers[qx] = b.getDictionaryAttr({b.getNamedAttr("layout", b.getStringAttr("pim_act")),
                                            b.getNamedAttr("acts", b.getI64ArrayAttr(acts))});
    });
    if (!placementOK) return signalPassFailure();

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
      ArrayAttr segments;
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
      weights.push_back({key.first, key.second, t.getNumElements() * eb, SmallVector<int64_t>(t.getShape()),
                         0, w->getAttrOfType<ArrayAttr>("hyb.weight_segments")});
    });
    for (W &w : weights)
      if (w.layout == "row_scale") {
        w.addr = place(w.bytes);
      }
    uint64_t denseEnd = cursor;
    // Room for the NPU code generator's constant pools (one small blob per phase).
    uint64_t extraLimit = (denseEnd + 255) / 256 * 256 + (uint64_t(1) << 20);
    cursor = extraLimit;
    // Common layout: NPU GEMV blocks keep a logical address for planning (out of the DRAM range); their rows go to DRAM rows below.
    uint64_t virtualCursor = uint64_t(1) << 40;
    for (W &w : weights)
      if (w.layout != "row_scale") {
        if (commonLayout) {
          virtualCursor = (virtualCursor + 4095) / 4096 * 4096;
          w.addr = virtualCursor;
          virtualCursor += uint64_t(w.bytes);
          continue;
        }
        cursor = (cursor + 4095) / 4096 * 4096;
        w.addr = place(w.bytes);
      }
    uint64_t npuWeightEnd = cursor;
    if (commonLayout) {
      llvm::json::Array cw;
      int64_t row = int64_t((npuWeightEnd + program.rowBytes() - 1) / program.rowBytes()) + 2;
      row += row & 1;
      for (W &w : weights) {
        if (w.layout == "row_scale") continue;
        pim::CommonWeight c;
        c.logical = w.addr;
        c.bytes = uint64_t(w.bytes);
        c.k = w.shape[1];
        c.n = w.shape[0] * 32;
        c.rowBase = row;
        c.paired = w.segments && w.segments.size() == 2;
        c.kHalves = commonLayout == 2 && c.sweeps() % 2 == 0;
        row += c.rowsUsed() + 2;
        row += row & 1;
        cw.push_back(llvm::json::Object{{"logical", int64_t(c.logical)}, {"bytes", int64_t(c.bytes)}, {"n", c.n}, {"k", c.k},
                                        {"row", c.rowBase}, {"paired", c.paired}, {"k_halves", c.kHalves}});
      }
      npuWeightEnd = std::max<uint64_t>(npuWeightEnd, uint64_t(row) * uint64_t(program.rowBytes()));
      std::string cj;
      llvm::raw_string_ostream cos(cj);
      cos << llvm::json::Value(std::move(cw));
      module->setAttr("hyb.common_weights", b.getStringAttr(cj));
    }
    for (hyb::WeightOp w : weightOps) {
      const W &info = weights[weightIndex[{w.getName().str(), w.getLayout().str()}]];
      w->setAttr(kLP6, b.getDictionaryAttr({b.getNamedAttr("layout", b.getStringAttr("dense")),
                                           b.getNamedAttr("addr", b.getI64IntegerAttr(int64_t(info.addr))),
                                           b.getNamedAttr("bytes", b.getI64IntegerAttr(info.bytes))}));
    }
    llvm::json::Array wjson;
    for (const W &w : weights) {
      if (!w.segments) {
        wjson.push_back(llvm::json::Object{{"name", w.name}, {"layout", w.layout}, {"addr", int64_t(w.addr)},
                                           {"bytes", w.bytes}, {"shape", llvm::json::Array(w.shape)}});
        continue;
      }
      // Virtual concatenation: allocate one tensor but write each checkpoint
      // matrix into its own disjoint subrange. No runtime gather or duplicate.
      bool scale = w.layout == "row_scale";
      if ((scale && w.shape.size() != 1) || (!scale && (w.layout != "npu_nk32" ||
          w.shape.size() != 3 || w.shape[2] != 32))) {
        module.emitError("invalid fused weight layout"); return signalPassFailure();
      }
      uint64_t cursor = w.addr;
      for (Attribute attr : w.segments) {
        auto segment = dyn_cast<DictionaryAttr>(attr);
        auto name = segment ? segment.getAs<StringAttr>("name") : StringAttr();
        auto rows = segment ? segment.getAs<IntegerAttr>("rows") : IntegerAttr();
        if (!name || !rows || rows.getInt() <= 0 || rows.getInt() % 32) {
          module.emitError("invalid fused weight segment"); return signalPassFailure();
        }
        int64_t n = rows.getInt(), bytes = scale ? n*4 : n*w.shape[1];
        if (uint64_t(bytes) > w.addr + w.bytes - cursor) {
          module.emitError("fused weight segment exceeds allocation"); return signalPassFailure();
        }
        SmallVector<int64_t> shape = scale ? SmallVector<int64_t>{n} : SmallVector<int64_t>{n/32,w.shape[1],32};
        wjson.push_back(llvm::json::Object{{"name", name.str()}, {"layout", w.layout}, {"addr", int64_t(cursor)},
                                           {"bytes", bytes}, {"shape", llvm::json::Array(shape)}});
        cursor += bytes;
      }
      if (cursor != w.addr + w.bytes) {
        module.emitError("fused weight segments do not cover allocation"); return signalPassFailure();
      }
    }
    program.kernelRow = int64_t((npuWeightEnd + program.rowBytes() - 1) / program.rowBytes()) + 1;
    for (auto &[v, attr] : pimBuffers) {
      SmallVector<NamedAttribute> attrs(attr.getValue());
      attrs.push_back(b.getNamedAttr("bytes", b.getI64IntegerAttr(bytesOf(v.getType()))));
      v.getDefiningOp()->setAttr(kLP6, b.getDictionaryAttr(attrs));
    }
    uint64_t end = std::max<uint64_t>(program.endAddress(), npuWeightEnd);
    if (int64_t(end) > specOr->lp6.capacityBytes && mayReuse && program.reuseBuffersAcrossPhases()) {
      const uint64_t without = end;
      end = std::max<uint64_t>(program.endAddress(), npuWeightEnd);
      // Listed with the first PIM phase (io.json phases[].decisions).
      func::FuncOp first;
      step.walk([&](func::CallOp call) {
        auto callee = symbols.lookup<func::FuncOp>(call.getCallee());
        if (!first && callee && callee->hasAttr(kPhase) &&
            cast<StringAttr>(callee->getAttr(kPhase)).getValue() == "pim")
          first = callee;
      });
      if (first)
        cost::recordDecision(first, "pim buffer reuse", "PIM activation/output tiles of every PIM phase",
                             "one set (plan exceeds LP6 capacity)",
                             {{"bytes_without", double(without)}, {"bytes_with", double(end)},
                              {"capacity", double(specOr->lp6.capacityBytes)}});
    }
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
    module->setAttr("hyb.pim_records", b.getI64IntegerAttr(records));
  }
};
} // namespace
