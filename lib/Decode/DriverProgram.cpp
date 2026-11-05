//===- DriverProgram.cpp - decode step -> one PLENA program ----------------===//
//
// The records of the step program: every NPU phase (NpuCodegen or the PIM packet kernel), one
// PIM record (PIM primitives) per PIM phase, and a final "collect" phase that gathers the step
// outputs into L1.
//===----------------------------------------------------------------------===//

#include <optional>
#include "DriverProgram.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include "DriverPackage.h"
#include "npu/Decode/PimMetadata.h"
#include "npu/Decode/Transforms/Passes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <map>
#include <set>

using namespace mlir;
using namespace mlir::hybrid;
namespace json = llvm::json;

namespace {

DictionaryAttr lp6Of(func::FuncOp step, Value v) {
  if (auto ba = dyn_cast<BlockArgument>(v)) return step.getArgAttrOfType<DictionaryAttr>(ba.getArgNumber(), kLP6);
  if (Operation *d = v.getDefiningOp()) return d->getAttrOfType<DictionaryAttr>(kLP6);
  return {};
}

/// The cost decisions recorded on a phase (NpuCost.h recordDecision, PimCost.h), with their numbers.
std::optional<json::Array> decisionsJson(func::FuncOp phase) {
  auto decisions = phase->getAttrOfType<ArrayAttr>("hyb.decisions");
  if (!decisions) return std::nullopt;
  json::Array list;
  for (Attribute a : decisions) {
    json::Object entry;
    for (NamedAttribute f : cast<DictionaryAttr>(a)) {
      if (auto v = dyn_cast<StringAttr>(f.getValue())) entry[f.getName().str()] = v.getValue().str();
      else if (auto v = dyn_cast<FloatAttr>(f.getValue())) entry[f.getName().str()] = v.getValueAsDouble();
      else if (auto v = dyn_cast<IntegerAttr>(f.getValue())) entry[f.getName().str()] = v.getInt();
    }
    list.push_back(std::move(entry));
  }
  return list;
}

/// io.json entry of one NPU phase: size of the emitted code, SRAM peaks, weight streams, how often
/// each NPU pass and backend schedule applied, and the cost decisions with their numbers.
json::Object npuPhaseReport(const std::string &name, func::FuncOp phase, const NpuPhaseStats &stats,
                            const TargetSpec &spec, const plena::ProgramBuilder &program) {
  auto count = [&](StringRef attr) -> int64_t {
    auto value = phase->getAttrOfType<IntegerAttr>(attr);
    return value ? value.getInt() : 0;
  };
  json::Array streams;
  for (const auto &s : stats.weightStreams)
    streams.push_back(json::Object{{"tile_bytes", s.tileBytes}, {"iterations", s.iterations},
        {"slots", s.slots}, {"batch_tiles", s.batch}, {"prefetch", s.prefetch},
        {"cross_batch", s.crossBatch}, {"reason", s.reason}, {"initial_slot", s.initialSlot},
        {"release_on_load", s.prefetch}});
  json::Object report{{"phase", name}, {"kind", "npu"}, {"core_words", stats.coreWords},
      {"commands", stats.commands}, {"l1_peak", stats.l1Peak}, {"cores", stats.cores},
      {"l2_staging_bytes_per_core", stats.l2StagingPerCore}, {"weight_streams", std::move(streams)}};

  // Passes, in pipeline order.
  report["projection_fusion"] = json::Object{{"groups", count("hyb.projection_fusions")}};
  if (auto attr = phase->getAttrOfType<DictionaryAttr>("hyb.flash_attention")) {
    json::Object flash;
    for (NamedAttribute field : attr) {
      if (auto value = dyn_cast<IntegerAttr>(field.getValue())) flash[field.getName().str()] = value.getInt();
      else if (auto value = dyn_cast<StringAttr>(field.getValue()))
        flash[field.getName().str()] = value.getValue().str();
    }
    report["flash_attention"] = std::move(flash);
  }
  json::Object attention;
  if (auto attr = phase->getAttrOfType<DictionaryAttr>("hyb.attention_fusion"))
    for (NamedAttribute field : attr)
      if (auto value = dyn_cast<IntegerAttr>(field.getValue())) attention[field.getName().str()] = value.getInt();
  report["attention_fusion"] = std::move(attention);
  report["gemv_epilogue_fusion"] = json::Object{{"applied", count("hyb.gemv_epilogue_fusions")}};
  report["elementwise_distribution"] = json::Object{
      {"groups", count("hyb.elementwise_distributions")},
      {"candidates", count("hyb.elementwise_candidates")},
      {"below_threshold", count("hyb.elementwise_small")},
      {"unsupported", count("hyb.elementwise_unsupported")},
      {"capacity_rejected", count("hyb.elementwise_capacity")},
      {"indivisible", count("hyb.elementwise_indivisible")},
      {"transposes", count("hyb.transposes_distributed")}};
  report["rope_split"] = json::Object{{"applied", count("hyb.rope_splits")}};
  report["gate_up_pair"] = json::Object{{"consumers_pipelined", count("hyb.epilogue_consumers")}};
  report["activation_residency"] =
      json::Object{{"private_result_destinations", count("hyb.activation_residency_private_results")}};
  report["matrix_init_elision"] = json::Object{{"applied", count("hyb.matrix_init_elisions")}};
  report["matmul_rhs_packing"] = json::Object{{"applied", count("hyb.matmul_rhs_packings")},
                                              {"transpose_output", count("hyb.matmul_output_transposes")}};
  report["input_tile_pipeline"] = json::Object{
      {"groups", count("hyb.input_pipeline_groups")}, {"tiles", count("hyb.input_pipeline_tiles")},
      {"columns", spec.npu.inputTileColumns}};

  // Choices made by comparing costs (NpuCost.h), with the numbers behind each.
  if (auto decisions = decisionsJson(phase)) report["decisions"] = std::move(*decisions);

  // Backend scheduling.
  report["async_output_stores"] = json::Object{
      {"issued", stats.asyncOutputStores},
      {"stores_with_vector_work_before_wait", stats.outputStoresWithVectorWork}};
  report["dependency_schedule"] = json::Object{{"staging_release_cuts", stats.stagingReleaseCuts}};
  report["vector_schedule"] = json::Object{
      {"scalar_reciprocals_hoisted", stats.vectorScalarReciprocals},
      {"scalar_reciprocal_reuses", stats.vectorScalarReuses},
      {"pipeline_groups", stats.vectorPipelineGroups},
      {"pipelined_chunks", stats.vectorPipelinedChunks}};
  if (spec.npu.gdmaLookahead) report["gdma_admission"] = json::Object{{"lookahead", spec.npu.gdmaLookahead}};
  if (spec.npu.stagingMiscBytes) report["staging_misc_bytes"] = stats.stagingMiscBytes;
  report["command_schedule"] = json::Object{
      {"commands", int64_t(program.lastSchedule.commands)},
      {"reordered", int64_t(program.lastSchedule.reordered)},
      {"redundant_edges_removed", int64_t(program.lastSchedule.redundantEdges)},
      {"encoding_fallback", program.lastSchedule.encodingFallback}};
  return report;
}

} // namespace

/// Cuts a PIM phase's primitives into `records` records. Each partition keeps its own order and its share of every
/// record is about the same number of K sweeps, so all partitions advance together. The command processor reads a
/// record only when the one before is issued: the later records are read while the earlier ones compute, and only the
/// first record's read stays in front of the phase.
static std::vector<std::vector<pim::Primitive>> splitPimRecords(const std::vector<pim::Primitive> &all, int records) {
  std::map<uint32_t, std::vector<pim::Primitive>> byPartition;
  for (const pim::Primitive &p : all) byPartition[p.partition].push_back(p);
  std::vector<std::vector<pim::Primitive>> out(size_t(std::max(1, records)));
  for (auto &[partition, list] : byPartition) {
    int64_t sweeps = 0, done = 0;
    for (const pim::Primitive &p : list) sweeps += p.opcode == pim::Opcode::GemvStream;
    size_t record = 0;
    for (size_t i = 0; i < list.size(); ++i) {
      out[record].push_back(list[i]);
      if (list[i].opcode == pim::Opcode::GemvStream) ++done;
      const bool storeNext = i + 1 < list.size() && list[i + 1].opcode == pim::Opcode::StoreOutput;
      if (!storeNext && record + 1 < out.size() && done * int64_t(out.size()) >= sweeps * int64_t(record + 1)) ++record;
    }
  }
  llvm::erase_if(out, [](const auto &r) { return r.empty(); });
  return out;
}

bool driver::emitPimCommands(pim::Program &prog, std::vector<pim::Primitive> &pimCommands) {
  for (size_t pi = 0; pi < prog.projections.size(); ++pi) {
    int64_t expect = prog.projections[pi].computeFirst;
    auto emitted = prog.emitCompute(int64_t(pi), pimCommands);
    if (!emitted) {
      llvm::errs() << emitted.error << '\n';
      return false;
    }
    if (prog.projections[pi].computeFirst != expect) {
      llvm::errs() << "PIM command order differs from the plan: " << prog.projections[pi].name << " planned "
                   << expect << ", emitted at " << prog.projections[pi].computeFirst << "\n";
      return false;
    }
  }
  return true;
}

void driver::setPimKernel(plena::ProgramBuilder &program, const pim::Program &prog) {
  json::Object kernel{{"channel", 0},        {"rank", 0},           {"row", prog.kernelRow},
                      {"start_column", 0},   {"bank_set", 1},       {"weight_bits", 8},
                      {"activation_bits", 8}, {"scale_bits", 0},    {"input_register_count", 8},
                      {"output_register_count", 8}, {"subchannels", prog.channels},
                      {"weight_layout", prog.weightLayout()},
                      {"bankset_pingpong_lead", prog.banksetPingPongLead}};
  program.pimKernelJson = llvm::formatv("{0}", json::Value(std::move(kernel))).str();
}

bool driver::emitPhases(ModuleOp mod, func::FuncOp step, const pim::Program &prog,
                        const std::vector<pim::Primitive> &pimCommands, const TargetSpec &spec,
                        plena::ProgramBuilder &program, LP6Extra &extra,
                        std::vector<int> &exit, int &logicalCores, json::Array &report, std::string &err) {
  program.gdmaLookahead = uint32_t(spec.npu.gdmaLookahead);
  program.l1Bytes = uint64_t(spec.npu.l1Bytes);
  program.commandSchedule = true;
  CarriedAccesses carried;   // the last NPU phase's L2 / LP6 accesses (the next phase waits only on these)
  struct { std::vector<int> records, ranks, before; } pimBefore;   // the PIM phase just before
  for (Operation &op : step.getBody().front()) {
    auto call = dyn_cast<func::CallOp>(op);
    if (!call) continue;
    auto callee = mod.lookupSymbol<func::FuncOp>(call.getCallee());
    StringRef kind = callee->getAttrOfType<StringAttr>(kPhase).getValue();
    std::string name = callee.getSymName().str();
    if (kind == "pim") {
      int64_t idx = call->getAttrOfType<IntegerAttr>("hyb.pim_phase").getInt();
      auto [first, count] = prog.phases[size_t(idx)];
      std::vector<pim::Primitive> primitives(pimCommands.begin() + first, pimCommands.begin() + first + count);
      pimBefore.before = exit;
      pimBefore.records.clear();
      auto recordsAttr = call->getAttrOfType<IntegerAttr>("hyb.pim_records");
      const int records = recordsAttr ? int(recordsAttr.getInt()) : 1;
      bool samePhase = false;
      if (prog.commonLayout == 2) {
        // Research option --common-layout 2: a subchannel computes on one rank at a time. Each run of primitives on one rank per
        // subchannel waits for the run before it. Inside a run the records are cut as usual, so only its first read is exposed.
        std::vector<std::vector<pim::Primitive>> runs;
        std::map<int, int> rankOf;
        for (const pim::Primitive &q : primitives) {
          const int sc = prog.partChannel(int(q.partition)), rk = prog.partRank(int(q.partition));
          auto it = rankOf.find(sc);
          if (runs.empty() || (it != rankOf.end() && it->second != rk)) {
            runs.emplace_back();
            rankOf.clear();
          }
          rankOf[sc] = rk;
          runs.back().push_back(q);
        }
        std::vector<int> before = pimBefore.before;
        for (const auto &run : runs) {
          const double fetchUs = double(run.size()) * pim::kRecordWordsPerPrimitive / spec.npu.commandFetchWordsPerCycle / spec.npu.clockMHz;
          std::map<uint32_t, int64_t> sweeps;
          for (const pim::Primitive &q : run) sweeps[q.partition] += q.opcode == pim::Opcode::GemvStream;
          int64_t rounds = std::numeric_limits<int64_t>::max();
          for (auto [partition, count] : sweeps) rounds = std::min(rounds, count);
          const int cut = int(std::clamp<int64_t>(int64_t(std::ceil(fetchUs / spec.pim.recordFetchUs)), 1, std::min<int64_t>(spec.pim.maxRecords, rounds)));
          std::vector<int> ids;
          for (const auto &part : splitPimRecords(run, cut)) {
            ids.push_back(program.pimRecord(name, before, part, samePhase));
            samePhase = true;
          }
          before = ids;
          pimBefore.records.insert(pimBefore.records.end(), ids.begin(), ids.end());
        }
      } else
      for (const auto &part : splitPimRecords(primitives, records)) {
        pimBefore.records.push_back(program.pimRecord(name, pimBefore.before, part, samePhase));
        samePhase = true;
      }
      exit = pimBefore.records;
      pimBefore.ranks.clear();
      for (const auto &p : prog.projections)
        if (p.computeCount > 0 && p.computeFirst >= first && p.computeFirst < first + count)
          for (int part : p.streamPart)
            if (!llvm::is_contained(pimBefore.ranks, prog.partRank(part))) pimBefore.ranks.push_back(prog.partRank(part));
      json::Object pimReport{{"phase", name}, {"kind", "pim"}, {"commands", count}};
      if (auto decisions = decisionsJson(callee)) pimReport["decisions"] = std::move(*decisions);
      report.push_back(std::move(pimReport));
      // The NPU phase before keeps its carried accesses: the PIM phase touches no L2, and the next
      // NPU phase orders its GDMAs against the PIM ranks itself (NpuPhaseInput::pimRecord).
      continue;
    }
    NpuPhaseInput in;
    in.phase = callee;
    for (Value v : call.getOperands()) {
      DictionaryAttr a = lp6Of(step, v);
      if (!a) {
        llvm::errs() << "operand of " << name << " has no LP6 placement\n";
        return false;
      }
      in.argLP6.push_back(a);
    }
    in.pim = &prog;
    in.spec = &spec;
    in.program = &program;
    in.carried = &carried;
    in.pimRecords = pimBefore.records;
    in.pimRanks = pimBefore.ranks;
    in.beforePim = pimBefore.before;
    in.commonWeights = extra.commonWeights.empty() ? nullptr : &extra.commonWeights;
    in.commonChunkRows = extra.commonChunkRows;
    pimBefore = {};
    program.beginPhase(name, exit);
    NpuPhaseStats stats;
    LogicalResult emitted = emitNpuPhase(in, extra, stats, err);
    if (failed(emitted)) {
      llvm::errs() << "NPU codegen failed in " << name << ": " << err << "\n";
      return false;
    }
    exit = program.endPhase();
    for (auto &a : carried.l2) a.id = program.scheduledId(a.id);
    for (auto &a : carried.lp6) a.id = program.scheduledId(a.id);
    logicalCores = std::max(logicalCores, int(stats.cores));
    if (extra.cursor > extra.limit) {
      llvm::errs() << "NPU constant pools overflow the reserved LP6 region\n";
      return false;
    }
    json::Object phaseReport = npuPhaseReport(name, callee, stats, spec, program);
    report.push_back(std::move(phaseReport));
  }
  return true;
}

// --debug-collect also gathers every buffer passed between phases (dense activations, the
// INT8 PIM inputs from partition parts[0] / PU 0, the INT32 PIM outputs in [M, nPad] rows).
bool driver::emitCollect(func::FuncOp step, const pim::Program &prog, plena::ProgramBuilder &program,
                         const std::vector<int> &exit, bool noOutputCollect, bool debugCollect,
                         const std::string &debugPhases, json::Array &outputs, int64_t l2Bytes,
                         std::vector<std::string> &uncollected) {
  program.beginPhase("collect", exit);
  plena::ProgramBuilder &w = program;
  std::map<int, uint32_t> cfg;
  auto load = [&](uint32_t gp, uint32_t v) {
    w.append(plena::OP_LUI | gp << 6 | (v >> 12) << 10);
    if (v & 0xFFF) w.append(plena::OP_ADDI | gp << 6 | gp << 10 | (v & 0xFFF) << 14);
  };
  auto ldma = [&](uint64_t l1, uint32_t bytes) {
    if (!cfg.count(7) || cfg[7] != bytes) {
      load(15, bytes);
      w.append(plena::rform(plena::OP_C_SET, 15, 0, 0, 0, 7));
      cfg[7] = bytes;
    }
    load(1, uint32_t(l1));
    load(2, uint32_t(l1));
    w.append(plena::rform(plena::OP_LDMA, 1, 2, 0, 0, 0));
  };
  uint64_t l1 = 4096;
  bool overflow = false;
  auto item = [&](const std::string &name, Value v, int64_t bytes, json::Array shape, StringRef layout) {
    if (l1 + uint64_t(bytes) > (4u << 20)) {
      overflow = true;
      return uint64_t(0);
    }
    uint64_t at = l1;
    outputs.push_back(json::Object{{"name", name}, {"l1", int64_t(at)}, {"bytes", bytes},
                                   {"dtype", dtypeName(cast<ShapedType>(v.getType()).getElementType())},
                                   {"shape", std::move(shape)}, {"layout", layout}});
    l1 = (l1 + uint64_t(bytes) + 63) / 64 * 64;
    return at;
  };
  auto collectDense = [&](const std::string &name, Value v, DictionaryAttr a) {
    uint64_t addr = uint64_t(a.getAs<IntegerAttr>("addr").getInt());
    uint64_t bytes = uint64_t(a.getAs<IntegerAttr>("bytes").getInt());
    uint64_t at = item(name, v, int64_t(bytes), shapeOf(v.getType()), "dense");
    if (overflow) return;
    constexpr uint64_t chunk = (1 << 20) - 64;
    for (uint64_t o = 0; o < bytes; o += chunk) {
      uint32_t n = uint32_t(std::min(chunk, bytes - o));
      w.gdmaLoad(addr + o, at + o, n);
      ldma(at + o, n);
    }
  };
  // Step outputs go to core 0's L1 (staged through L2 [0, 4 MiB)), then to the upper half of L2,
  // which the simulator also dumps. An output that fits neither (a model's recurrent states can be
  // tens of MiB) stays in LP6 and is listed as uncollected.
  const uint64_t l2Limit = uint64_t(std::max<int64_t>(4 << 20, l2Bytes));
  uint64_t l2 = 4u << 20;
  auto collectToL2 = [&](const std::string &name, Value v, DictionaryAttr a) {
    uint64_t addr = uint64_t(a.getAs<IntegerAttr>("addr").getInt());
    uint64_t bytes = uint64_t(a.getAs<IntegerAttr>("bytes").getInt());
    if (l2 + bytes > l2Limit) return false;
    outputs.push_back(json::Object{{"name", name}, {"memory", "l2"}, {"l2", int64_t(l2)}, {"bytes", int64_t(bytes)},
                                   {"dtype", dtypeName(cast<ShapedType>(v.getType()).getElementType())},
                                   {"shape", shapeOf(v.getType())}, {"layout", "dense"}});
    constexpr uint64_t chunk = (1 << 20) - 64;
    for (uint64_t o = 0; o < bytes; o += chunk) w.gdmaLoad(addr + o, l2 + o, uint32_t(std::min(chunk, bytes - o)));
    l2 = (l2 + bytes + 63) / 64 * 64;
    return true;
  };
  auto ret = cast<func::ReturnOp>(step.getBody().front().getTerminator());
  if (!noOutputCollect)
    for (auto [i, v] : llvm::enumerate(ret.getOperands())) {
      auto name = step.getResultAttrOfType<StringAttr>(unsigned(i), kName);
      std::string label = name ? name.str() : "out" + std::to_string(i);
      DictionaryAttr placement = lp6Of(step, v);
      collectDense(label, v, placement);
      if (!overflow) continue;
      overflow = false;
      if (!collectToL2(label, v, placement)) uncollected.push_back(label);
    }
  if (debugCollect) {
    // A buffer is named after its producer: the first call that has it as an operand.
    std::set<std::string> only;
    for (StringRef p : llvm::split(StringRef(debugPhases), ','))
      if (!p.empty()) only.insert(p.str());
    DenseMap<Value, std::string> producer;
    DenseSet<Value> selected;
    for (Operation &op : step.getBody().front())
      if (auto call = dyn_cast<func::CallOp>(op))
        for (auto [j, v] : llvm::enumerate(call.getOperands())) {
          if (!producer.count(v)) producer[v] = (call.getCallee() + ".arg" + std::to_string(j)).str();
          if (only.empty() || only.count(call.getCallee().str())) selected.insert(v);
        }
    DenseSet<Value> seen(ret.getOperands().begin(), ret.getOperands().end());
    for (Operation &op : step.getBody().front()) {
      auto call = dyn_cast<func::CallOp>(op);
      if (!call) continue;
      for (auto [j, v] : llvm::enumerate(call.getOperands())) {
        if (isa<BlockArgument>(v) || !selected.count(v) || !seen.insert(v).second) continue;
        DictionaryAttr a = lp6Of(step, v);
        std::string name = producer[v];
        StringRef layout = a.getAs<StringAttr>("layout").getValue();
        auto ty = cast<ShapedType>(v.getType());
        if (layout == "dense") {
          collectDense(name, v, a);
        } else if (layout == "pim_act") {
          int64_t K = ty.getDimSize(1);
          auto acts = a.getAs<ArrayAttr>("acts");
          uint64_t at = item(name, v, ty.getNumElements(), shapeOf(ty), layout);
          if (overflow) break;
          for (auto [m, ai] : llvm::enumerate(acts)) {
            const pim::Activation &act = prog.activations[size_t(cast<IntegerAttr>(ai).getInt())];
            for (int64_t first = 0; first < act.packets();) {
              int64_t pk = act.packetBase() + first;
              int64_t n = std::min(act.packets() - first, pim::kPacketsPerRow - pk % pim::kPacketsPerRow);
              uint64_t dst = at + uint64_t(int64_t(m) * K + first * 32);
              w.gdmaLoad(prog.activationPacketAddress(act, first, 0, act.parts[0]), dst, 32, uint32_t(n),
                         uint32_t(prog.colBytes()), 32);
              ldma(dst, uint32_t(32 * n));
              first += n;
            }
          }
        } else if (layout == "pim_out") {
          const pim::Projection &p = prog.projections[size_t(a.getAs<IntegerAttr>("proj").getInt())];
          if (p.splitK != 1 || p.common) {   // its K pieces are partial sums that only the NPU's gather adds
            uncollected.push_back(name);
            continue;
          }
          json::Array shape{ty.getDimSize(0), p.nPad};
          uint64_t at = item(name, v, ty.getDimSize(0) * p.nPad * 4, std::move(shape), layout);
          if (overflow) break;
          const int64_t T = p.tilesPerPU();
          auto partEnd = [&](int64_t t) {
            int64_t b = p.blockOfTile(t);
            const int part = p.streamPart[size_t(b)];
            while (b < p.blocks() && p.streamPart[size_t(b)] == part) ++b;
            return p.blockTile(b);
          };
          for (int m = 0; m < p.tokens(); ++m)
            for (int pu = 0; pu < pim::kPUs; ++pu)
              for (int64_t first = 0; first < T;) {
                const int64_t g = p.streamOfTile(first, 0);
                const int part = p.streamPart[size_t(g)];
                const int64_t tile = p.outputTile(m, 0, g, int(first - p.blockTile(p.blockOfTile(first))));
                int64_t n = std::min({T - first, pim::kPacketsPerRow - tile % pim::kPacketsPerRow, partEnd(first) - first});
                uint64_t dst = at + uint64_t((int64_t(m) * p.nPad + pu * p.rowsPerPU() + first * 8) * 4);
                w.gdmaLoad(prog.outputTileAddress(tile, pu, part), dst, 32, uint32_t(n), uint32_t(prog.colBytes()), 32);
                ldma(dst, uint32_t(32 * n));
                first += n;
              }
        }
      }
    }
  }
  if (overflow) {
    llvm::errs() << "collected buffers do not fit L1 (" << l1 << " bytes)\n";
    return false;
  }
  return true;
}
