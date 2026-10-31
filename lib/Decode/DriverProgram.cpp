//===- DriverProgram.cpp - decode step -> one PLENA program ----------------===//
//
// The records of the step program: every NPU phase (NpuCodegen or the PIM packet kernel), one
// PIM record (PIM primitives) per PIM phase, and a final "collect" phase that gathers the step
// outputs into L1.
//===----------------------------------------------------------------------===//

#include "DriverProgram.h"
#include "DriverPackage.h"
#include "npu/Decode/PimMetadata.h"
#include "npu/Decode/Target/PacketMatmul.h"
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

/// io.json entry of one NPU phase: size of the emitted code, SRAM peaks, weight streams and, for
/// every enabled optional pass or backend schedule, how often it applied.
json::Object npuPhaseReport(const std::string &name, func::FuncOp phase, const NpuPhaseStats &stats,
                            const TargetSpec &spec, const plena::ProgramBuilder &program) {
  auto count = [&](StringRef attr) -> int64_t {
    auto value = phase->getAttrOfType<IntegerAttr>(attr);
    return value ? value.getInt() : 0;
  };
  json::Array streams;
  for (const auto &s : stats.weightStreams) {
    json::Object stream{{"tile_bytes", s.tileBytes}, {"iterations", s.iterations},
        {"slots", s.slots}, {"batch_tiles", s.batch}, {"prefetch", s.prefetch},
        {"cross_batch", s.crossBatch}, {"reason", s.reason}};
    if (spec.npu.weightStreamRotateSlots) stream["initial_slot"] = s.initialSlot;
    if (spec.npu.weightStreamEarlyRelease) stream["release_on_load"] = s.prefetch;
    streams.push_back(std::move(stream));
  }
  json::Object report{{"phase", name}, {"kind", "npu"}, {"core_words", stats.coreWords},
      {"commands", stats.commands}, {"l1_peak", stats.l1Peak}, {"cores", stats.cores},
      {"l2_staging_bytes_per_core", stats.l2StagingPerCore}, {"weight_streams", std::move(streams)}};

  // Optional passes, in pipeline order.
  if (spec.npu.projectionFusion)
    report["projection_fusion"] = json::Object{{"groups", count("hyb.projection_fusions")}};
  if (spec.npu.flashAttention) {
    json::Object flash{{"flash_decoding", spec.npu.flashDecoding}, {"kv_slicing", spec.npu.flashKvSlicing},
                       {"query_tile", spec.npu.attentionQueryTile}, {"key_tile", spec.npu.attentionKeyTile},
                       {"splits", spec.npu.attentionSplits}};
    if (auto attr = phase->getAttrOfType<DictionaryAttr>("hyb.flash_attention"))
      for (NamedAttribute field : attr) {
        if (auto value = dyn_cast<IntegerAttr>(field.getValue())) flash[field.getName().str()] = value.getInt();
        else if (auto value = dyn_cast<StringAttr>(field.getValue()))
          flash[field.getName().str()] = value.getValue().str();
      }
    report["flash_attention"] = std::move(flash);
  }
  if (spec.npu.attentionFusion) {
    json::Object counts;
    if (auto attr = phase->getAttrOfType<DictionaryAttr>("hyb.attention_fusion"))
      for (NamedAttribute field : attr)
        if (auto value = dyn_cast<IntegerAttr>(field.getValue())) counts[field.getName().str()] = value.getInt();
    report["attention_fusion"] = std::move(counts);
  }
  if (spec.npu.gemvEpilogueFusion)
    report["gemv_epilogue_fusion"] = json::Object{{"applied", count("hyb.gemv_epilogue_fusions")}};
  if (spec.npu.elementwiseDistribution)
    report["elementwise_distribution"] = json::Object{
        {"groups", count("hyb.elementwise_distributions")},
        {"candidates", count("hyb.elementwise_candidates")},
        {"below_threshold", count("hyb.elementwise_small")},
        {"unsupported", count("hyb.elementwise_unsupported")},
        {"capacity_rejected", count("hyb.elementwise_capacity")},
        {"indivisible", count("hyb.elementwise_indivisible")},
        {"min_elements", spec.npu.distributeMinElements}};
  if (spec.npu.elementwiseDistribution && spec.npu.distributeTransposes)
    (*report["elementwise_distribution"].getAsObject())["transposes"] = count("hyb.transposes_distributed");
  if (spec.npu.ropeSplit) report["rope_split"] = json::Object{{"applied", count("hyb.rope_splits")}};
  if (spec.npu.pairGateUp)
    report["gate_up_pair"] = json::Object{{"consumer_pipeline", spec.npu.pairConsumerPipeline},
                                          {"consumers_pipelined", count("hyb.epilogue_consumers")}};
  if (spec.npu.activationResidency)
    report["activation_residency"] =
        json::Object{{"private_result_destinations", count("hyb.activation_residency_private_results")}};
  if (spec.npu.matrixInitElision)
    report["matrix_init_elision"] = json::Object{{"applied", count("hyb.matrix_init_elisions")}};
  if (spec.npu.matmulRhsPacking || spec.npu.matmulTransposeOutput)
    report["matmul_rhs_packing"] = json::Object{{"applied", count("hyb.matmul_rhs_packings")},
                                                {"transpose_output", count("hyb.matmul_output_transposes")}};
  if (spec.npu.inputTilePipeline)
    report["input_tile_pipeline"] = json::Object{
        {"groups", count("hyb.input_pipeline_groups")}, {"tiles", count("hyb.input_pipeline_tiles")},
        {"columns", spec.npu.inputTileColumns}, {"async_input_loads", spec.npu.asyncInputLoads}};

  // Choices made by comparing costs (NpuCost.h), with the numbers behind each.
  if (auto decisions = phase->getAttrOfType<ArrayAttr>("hyb.decisions")) {
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
    report["decisions"] = std::move(list);
  }

  // Backend scheduling.
  if (spec.npu.asyncOutputStores)
    report["async_output_stores"] = json::Object{
        {"issued", stats.asyncOutputStores},
        {"vector_overlap_enabled", spec.npu.asyncOutputStoreVectorOverlap},
        {"stores_with_vector_work_before_wait", stats.outputStoresWithVectorWork}};
  if (spec.npu.stagingEarlyRelease || spec.npu.memoryDependencySchedule)
    report["dependency_schedule"] = json::Object{{"staging_release_cuts", stats.stagingReleaseCuts},
                                                 {"matrix_dma_range_guards", spec.npu.memoryDependencySchedule}};
  if (spec.npu.vectorScalarReuse || spec.npu.vectorChunkPipeline)
    report["vector_schedule"] = json::Object{
        {"scalar_reuse_enabled", spec.npu.vectorScalarReuse},
        {"chunk_pipeline_enabled", spec.npu.vectorChunkPipeline},
        {"scalar_reciprocals_hoisted", stats.vectorScalarReciprocals},
        {"scalar_reciprocal_reuses", stats.vectorScalarReuses},
        {"pipeline_groups", stats.vectorPipelineGroups},
        {"pipelined_chunks", stats.vectorPipelinedChunks}};
  if (spec.npu.gdmaLookahead)
    report["gdma_admission"] = json::Object{{"lookahead", spec.npu.gdmaLookahead},
                                            {"streams_only", spec.npu.gdmaLookaheadStreamsOnly}};
  if (spec.npu.stagingMiscBytes) report["staging_misc_bytes"] = stats.stagingMiscBytes;
  if (spec.npu.crossPhasePrefetch) report["cross_phase_prefetch"] = true;
  if (spec.npu.quantFusedMax || spec.npu.vectorLoops)
    report["vector_code"] = json::Object{{"quant_fused_max", spec.npu.quantFusedMax},
                                         {"vector_loops", spec.npu.vectorLoops}};
  if (spec.npu.commandSchedule)
    report["command_schedule"] = json::Object{
        {"commands", int64_t(program.lastSchedule.commands)},
        {"reordered", int64_t(program.lastSchedule.reordered)},
        {"redundant_edges_removed", int64_t(program.lastSchedule.redundantEdges)},
        {"encoding_fallback", program.lastSchedule.encodingFallback}};
  return report;
}

} // namespace

bool driver::emitPimCommands(pim::Program &prog, std::vector<pim::Primitive> &pimCommands) {
  for (size_t pi = 0; pi < prog.projections.size(); ++pi) {
    int64_t expect = prog.projections[pi].computeFirst;
    auto emitted = prog.emitCompute(int64_t(pi), pimCommands);
    if (!emitted) {
      llvm::errs() << emitted.error << '\n';
      return false;
    }
    if (prog.projections[pi].computeFirst != expect) {
      llvm::errs() << "PIM command order differs from the plan\n";
      return false;
    }
  }
  return true;
}

void driver::setPimKernel(plena::ProgramBuilder &program, const pim::Program &prog, bool fixedStorage,
                          uint32_t pimCommandCount) {
  json::Object kernel{{"channel", 0},        {"rank", 0},           {"row", prog.kernelRow},
                      {"start_column", 0},   {"bank_set", 1},       {"weight_bits", 8},
                      {"activation_bits", 8}, {"scale_bits", 0},    {"input_register_count", 8},
                      {"output_register_count", 8}, {"subchannels", prog.channels},
                      {"weight_layout", prog.weightLayout()},
                      {"bankset_pingpong_lead", prog.banksetPingPongLead}};
  // Weight placement is an input in this mode. Inferring runtime bases from
  // only the selected executable primitives can move activations/outputs.
  if (fixedStorage && pimCommandCount != 0)
    kernel["storage_layout"] = ::npu::decode::pimRuntimeStorageLayout(prog);
  program.pimKernelJson = llvm::formatv("{0}", json::Value(std::move(kernel))).str();
}

bool driver::emitPhases(ModuleOp mod, func::FuncOp step, const pim::Program &prog,
                        const std::vector<pim::Primitive> &pimCommands, const TargetSpec &spec,
                        plena::ProgramBuilder &program, LP6Extra &extra, bool weightPrefetch, int pimActivationCores,
                        std::vector<int> &exit, int &logicalCores, json::Array &report, std::string &err) {
  program.gdmaLookahead = uint32_t(spec.npu.gdmaLookahead);
  program.commandSchedule = spec.npu.commandSchedule;
  int parallelActivationPhases = 0;
  CarriedAccesses carried;   // npu.cross_phase_prefetch: the last NPU phase's L2 / LP6 accesses
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
      exit = {program.pimRecord(name, exit, primitives)};
      report.push_back(json::Object{{"phase", name}, {"kind", "pim"}, {"commands", count}});
      carried = {};
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
    in.weightPrefetch = weightPrefetch;
    in.pimActivationCores = pimActivationCores;
    const bool packet = callee->hasAttr("hyb.npu_kernel");
    if (packet) carried = {};
    else in.carried = &carried;
    program.beginPhase(name, exit);
    NpuPhaseStats stats;
    LogicalResult emitted = packet ? emitPimPacketNpuPhase(in, stats, err) : emitNpuPhase(in, extra, stats, err);
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
    if (auto kernel = callee->getAttrOfType<StringAttr>("hyb.npu_kernel")) {
      phaseReport["kernel"] = kernel.getValue().str();
      auto plan = callee->getAttrOfType<StringAttr>("hyb.packet_plan");
      if (!plan) { llvm::errs() << "packet kernel did not report its checked memory plan\n"; return false; }
      auto value = json::parse(plan.getValue());
      if (!value) { llvm::errs() << llvm::toString(value.takeError()) << '\n'; return false; }
      phaseReport["packet_plan"] = std::move(*value);
    }
    if (stats.pimActivationCores > 1) {
      if (stats.pimActivationCores != pimActivationCores || stats.pimActivationPlanJson.empty()) {
        llvm::errs() << "parallel PIM activation feed did not report its checked plan\n";
        return false;
      }
      auto value = json::parse(stats.pimActivationPlanJson);
      if (!value) { llvm::errs() << llvm::toString(value.takeError()) << '\n'; return false; }
      phaseReport["pim_activation_cores"] = stats.pimActivationCores;
      phaseReport["pim_activation_plan"] = std::move(*value);
      ++parallelActivationPhases;
    }
    report.push_back(std::move(phaseReport));
  }
  if (pimActivationCores == 4 && parallelActivationPhases == 0) {
    llvm::errs() << "--pim-activation-cores=4 did not find a supported isolated PIM activation-feed phase\n";
    return false;
  }
  return true;
}

// --debug-collect also gathers every buffer passed between phases (dense activations, the
// INT8 PIM inputs from partition parts[0] / PU 0, the INT32 PIM outputs in [M, nPad] rows).
bool driver::emitCollect(func::FuncOp step, const pim::Program &prog, plena::ProgramBuilder &program,
                         const std::vector<int> &exit, bool noOutputCollect, bool debugCollect,
                         const std::string &debugPhases, json::Array &outputs) {
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
  auto ret = cast<func::ReturnOp>(step.getBody().front().getTerminator());
  if (!noOutputCollect)
    for (auto [i, v] : llvm::enumerate(ret.getOperands())) {
      auto name = step.getResultAttrOfType<StringAttr>(unsigned(i), kName);
      collectDense(name ? name.str() : "out" + std::to_string(i), v, lp6Of(step, v));
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
          json::Array shape{ty.getDimSize(0), p.nPad};
          uint64_t at = item(name, v, ty.getDimSize(0) * p.nPad * 4, std::move(shape), layout);
          if (overflow) break;
          int64_t T = p.tilesPerPU();
          auto partEnd = [&](int64_t t) {
            int64_t g = t / 8;
            while (g < p.streams() && p.streamPart[size_t(g)] == p.streamPart[size_t(t / 8)]) ++g;
            return g * 8;
          };
          for (int m = 0; m < p.tokens(); ++m)
            for (int pu = 0; pu < pim::kPUs; ++pu)
              for (int64_t first = 0; first < T;) {
                int64_t tile = p.tokenOutputTile(m, first);
                int64_t pk = p.outputBase + tile;
                int64_t n = std::min({T - first, pim::kPacketsPerRow - pk % pim::kPacketsPerRow, partEnd(first) - first});
                uint64_t dst = at + uint64_t((int64_t(m) * p.nPad + pu * p.rowsPerPU() + first * 8) * 4);
                w.gdmaLoad(prog.outputTileAddress(p, tile, pu), dst, 32, uint32_t(n), uint32_t(prog.colBytes()), 32);
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
