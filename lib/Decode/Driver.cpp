//===- Driver.cpp - decode step -> one PLENA program, LP6 image -------------------===//
//
//   npu-compile --pipeline=decode step.mlir --manifest manifest.json --spec target.json -o DIR [--model DIR]
//
// Runs the tensor and memory pipelines, then emits into DIR:
//   compiled.mlir          the planned, bufferized module
//   program.bin            the whole decode step as one ISA ver 2.0 program: the records of every
//                          NPU phase, one PIM record (PIM primitives) per PIM phase, and a
//                          final "collect" phase that gathers the step outputs into L1
//   system.json            its sidecar: placement, PIM kernel, command symbols <phase>.<record>
//   io.json                LP6 addresses of inputs / states / weights, L1 offsets of outputs
//   lp6.bin                LP6 image: FP16 dense weights, per-row weight scales, PIM weight
//                          packets (INT8, per-row symmetric), NPU constant pools
//===----------------------------------------------------------------------===//

#include "DriverImage.h"
#include "DriverPackage.h"
#include "DriverStorage.h"
#include "npu/Decode/Dialect/Hyb/HybDialect.h"
#include "npu/Decode/Support/SafeTensors.h"
#include "npu/Decode/Support/TargetSpec.h"
#include "npu/Decode/Target/NpuCodegen.h"
#include "npu/Decode/Target/PacketMatmul.h"
#include "npu/Target/IsaV2.h"
#include "npu/Decode/PimMetadata.h"
#include "npu/Decode/Target/PlenaProgram.h"
#include "npu/Decode/Transforms/Passes.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <atomic>
#include <chrono>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

using namespace mlir;
using namespace mlir::hybrid;
namespace cl = llvm::cl;
namespace json = llvm::json;

static cl::opt<std::string> inputFile(cl::Positional, cl::desc("<step.mlir>"), cl::Required);
static cl::opt<std::string> manifestFile("manifest", cl::desc("frontend manifest.json"), cl::Required);
static cl::opt<std::string> specFile("spec", cl::desc("target spec JSON"), cl::Required);
static cl::opt<std::string> outDir("o", cl::desc("output directory"), cl::Required);
static cl::opt<std::string> modelDir("model", cl::desc("checkpoint directory (default: the module's hyb.model)"),
                                     cl::init(""));
static cl::opt<std::string> forceTarget("force", cl::desc("npu | pim: override the partition"), cl::init(""));
static cl::list<std::string> mapOverrides("map", cl::desc("substring:target, e.g. --map=o_proj:npu (repeatable)"));
static cl::opt<bool> noImage("no-image", cl::desc("do not write lp6.bin"), cl::init(false));
static cl::opt<std::string> weightStorageExport("pim-weight-storage-export",
    cl::desc("export verified PIM-only weight/scale storage (requires image generation)"), cl::init(""));
static cl::opt<std::string> weightStorageImport("pim-weight-storage-import",
    cl::desc("fix PIM weight/scale PA from a verified manifest; omitted ping-pong option follows manifest"), cl::init(""));
static cl::opt<std::string> packetNpu("pim-packet-npu",
    cl::desc("g4: opt-in NPU packet kernel; requires --force=npu and imported PIM weight storage"), cl::init(""));
static cl::opt<int> pimActivationCores("pim-activation-cores",
    cl::desc("1 default; 4 opts into checked isolated quant/PIM activation-feed parallelism with --force=pim"),
    cl::init(1));
static cl::opt<bool> weightPrefetch("npu-weight-prefetch",
    cl::desc("overlap streamed GEMV weight LDMA with matrix compute (false for serial A/B)"), cl::init(true));
static cl::opt<int> banksetPingPong("pim-bankset-pingpong",
    cl::desc("research PIM weight-row ping-pong: 0 off, 1 default lead 12, 2..63 lead"), cl::init(0));
static cl::opt<bool> debugCollect("debug-collect", cl::desc("also collect every buffer passed between phases"),
                                  cl::init(false));
static cl::opt<bool> noOutputCollect("no-output-collect",
    cl::desc("skip final output readback into L1; output remains in LP6"), cl::init(false));
static cl::opt<std::string> debugPhases("debug-phases",
                                        cl::desc("with --debug-collect: only the operands of these phases (a,b,...)"),
                                        cl::init(""));

namespace {

double seconds(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

DictionaryAttr lp6Of(func::FuncOp step, Value v) {
  if (auto ba = dyn_cast<BlockArgument>(v)) return step.getArgAttrOfType<DictionaryAttr>(ba.getArgNumber(), kLP6);
  if (Operation *d = v.getDefiningOp()) return d->getAttrOfType<DictionaryAttr>(kLP6);
  return {};
}

} // namespace

int compileDecode(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  registerAllPasses();
  hybrid::registerHybridPasses();
  cl::ParseCommandLineOptions(argc, argv, "hybrid NPU + LPDDR6-PIM compiler\n");
  auto t0 = std::chrono::steady_clock::now();
  // The import may be this output directory's previous verified sidecar. Keep
  // its bytes before unlinking the old artifact; validate them after cleanup.
  std::unique_ptr<llvm::MemoryBuffer> importedStorageBuffer;
  std::error_code importedStorageReadError;
  if (!weightStorageImport.empty()) {
    auto file = llvm::MemoryBuffer::getFile(weightStorageImport);
    if (file) importedStorageBuffer = std::move(*file);
    else importedStorageReadError = file.getError();
  }
  // A reused output directory must not retain a previous content-verification
  // sidecar when this invocation fails even before reading its input graph.
  if (auto ec = llvm::sys::fs::remove(outDir + "/pim_weight_storage_verified.json")) {
    llvm::errs() << "cannot remove previous PIM weight storage verification: " << ec.message() << '\n';
    return 1;
  }
  const bool fixedStorage = !weightStorageExport.empty() || !weightStorageImport.empty();
  std::optional<::npu::decode::PimWeightStorage> importedStorage;
  if ((pimActivationCores != 1 && pimActivationCores != 4) ||
      (pimActivationCores == 4 &&
       (forceTarget != "pim" || !mapOverrides.empty() || !packetNpu.empty()))) {
    llvm::errs() << "PIM activation cores must be 1 or 4; 4 requires --force=pim, no --map overrides, and no --pim-packet-npu\n";
    return 1;
  }
  if (!packetNpu.empty() &&
      (packetNpu != "g4" || weightStorageImport.empty() || forceTarget != "npu" || !mapOverrides.empty())) {
    llvm::errs() << "PIM packet NPU requires g4, --force=npu, --pim-weight-storage-import, and no --map overrides\n";
    return 1;
  }
  if (!packetNpu.empty() && !weightPrefetch) {
    llvm::errs() << "PIM packet NPU g4 currently requires --npu-weight-prefetch=true\n";
    return 1;
  }
  if (fixedStorage && noImage) {
    llvm::errs() << "PIM weight storage import/export is incompatible with --no-image: content verification is required\n";
    return 1;
  }
  if (!weightStorageImport.empty()) {
    importedStorage = driver::parsePimWeightStorageImport(importedStorageBuffer.get(), importedStorageReadError,
                                                          banksetPingPong, banksetPingPong.getNumOccurrences());
    if (!importedStorage) return 1;
    banksetPingPong = importedStorage->banksetPingPongLead;
  }

  DialectRegistry registry;
  registerAllDialects(registry);
  registerAllExtensions(registry);
  registry.insert<hyb::HybDialect>();
  hyb::registerBufferizableOpInterfaceExternalModels(registry);
  MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();

  llvm::sys::fs::create_directories(outDir);
  driver::removeStaleOutputs(outDir);
  llvm::SourceMgr sm;
  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(inputFile, sm, &ctx);
  if (!module) return 1;
  bool reservedStorageAttr = false;
  module->getOperation()->walk([&](Operation *op) {
    for (StringRef name : {"hyb.pim_weight_storage_requested", "hyb.pim_weight_storage_import",
                          "hyb.pim_packet_npu_requested", "hyb.npu_packet", "hyb.npu_kernel",
                          "hyb.packet_projection", "hyb.packet_plan", "hyb.pim_activation_plan"})
      reservedStorageAttr |= op->hasAttr(name);
  });
  if (reservedStorageAttr) {
    llvm::errs() << "reserved PIM weight storage IR attributes are driver-owned; use the CLI with a frontend graph\n";
    return 1;
  }
  if (fixedStorage) module->getOperation()->setAttr("hyb.pim_weight_storage_requested", UnitAttr::get(&ctx));
  if (importedStorage)
    module->getOperation()->setAttr("hyb.pim_weight_storage_import",
        StringAttr::get(&ctx, llvm::formatv("{0}", ::npu::decode::toJSON(*importedStorage)).str()));
  if (!packetNpu.empty())
    module->getOperation()->setAttr("hyb.pim_packet_npu_requested", StringAttr::get(&ctx, packetNpu.getValue()));

  std::string overrideOpt;
  for (const std::string &m : mapOverrides) overrideOpt += (overrideOpt.empty() ? " overrides=" : ";") + m;
  std::string pipeline =
      "hyb-annotate{manifest=" + manifestFile + "}," +
      "func.func(hyb-simplify,canonicalize,cse,hyb-raise-linear,canonicalize,hyb-quantize,cse,hyb-fold-transpose,"
      "canonicalize),hyb-storage-f16,func.func(hyb-group-linear),hyb-partition{spec=" + specFile + " report=" +
      outDir + "/partition.json" + (forceTarget.empty() ? "" : " force=" + forceTarget) + overrideOpt + "}," +
      "func.func(hyb-lower-npu-gemv),hyb-outline-phases,canonicalize,hyb-npu-tile{spec=" + specFile + "},cse,canonicalize,"
      "one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map},"
      "buffer-results-to-out-params{hoist-static-allocs},canonicalize,cse,hyb-plan-lp6{spec=" + specFile +
      " bankset-pingpong=" + std::to_string(banksetPingPong) + "},"
      "loop-invariant-code-motion,hyb-npu-place,hyb-npu-promote,hyb-npu-memplan{spec=" + specFile + "}";
  PassManager pm(&ctx);
  if (failed(parsePassPipeline(pipeline, pm, llvm::errs()))) return 1;
  if (failed(pm.run(*module))) {
    llvm::errs() << "pipeline failed\n";
    return 1;
  }
  {
    std::error_code ec;
    llvm::raw_fd_ostream os(outDir + "/compiled.mlir", ec);
    module->print(os, OpPrintingFlags().elideLargeElementsAttrs());
  }
  llvm::outs() << llvm::formatv("pipeline: {0:f1} s\n", seconds(t0));

  std::string err;
  FailureOr<TargetSpec> spec = TargetSpec::load(specFile, err);
  if (failed(spec)) {
    llvm::errs() << err << "\n";
    return 1;
  }
  ModuleOp mod = *module;
  func::FuncOp step = getStepFunction(mod);
  auto planJson = json::parse(mod->getAttrOfType<StringAttr>("hyb.pim_plan").getValue());
  if (!planJson) {
    llvm::errs() << "bad hyb.pim_plan\n";
    return 1;
  }
  pim::Program prog = ::npu::decode::fromJSON(*planJson);
  if (!packetNpu.empty()) {
    int packetPhases = 0;
    for (auto fn : mod.getOps<func::FuncOp>()) packetPhases += fn->hasAttr("hyb.npu_kernel");
    if (packetPhases != 1 || prog.projections.size() != 1 || !prog.activations.empty() || !prog.phases.empty()) {
      llvm::errs() << "PIM packet NPU g4 currently supports one projection with storage-only PIM metadata\n";
      return 1;
    }
  }
  uint64_t denseEnd = uint64_t(mod->getAttrOfType<IntegerAttr>("hyb.lp6_dense_end").getInt());
  uint64_t extraLimit = uint64_t(mod->getAttrOfType<IntegerAttr>("hyb.lp6_extra_limit").getInt());
  LP6Extra extra{denseEnd, extraLimit, {}};

  // ---- PIM commands --------------------------------------------------------------------------
  std::vector<pim::Primitive> pimCommands;
  for (size_t pi = 0; pi < prog.projections.size(); ++pi) {
    int64_t expect = prog.projections[pi].computeFirst;
    auto emitted = prog.emitCompute(int64_t(pi), pimCommands);
    if (!emitted) {
      llvm::errs() << emitted.error << '\n';
      return 1;
    }
    if (prog.projections[pi].computeFirst != expect) {
      llvm::errs() << "PIM command order differs from the plan\n";
      return 1;
    }
  }
  const uint32_t pimCommandCount = uint32_t(pimCommands.size());
  if (!packetNpu.empty() && pimCommandCount != 0) {
    llvm::errs() << "PIM packet NPU storage-only plan unexpectedly emitted PIM commands\n";
    return 1;
  }
  if (!driver::writePimPrimitives(outDir, pimCommands)) return 1;

  // ---- NPU / PIM phases in program order, one program --------------------------------------
  plena::ProgramBuilder program;
  {
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
  std::vector<int> exit;           // records the next phase follows
  int logicalCores = 1;
  int parallelActivationPhases = 0;
  json::Array report;
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
      continue;
    }
    NpuPhaseInput in;
    in.phase = callee;
    for (Value v : call.getOperands()) {
      DictionaryAttr a = lp6Of(step, v);
      if (!a) {
        llvm::errs() << "operand of " << name << " has no LP6 placement\n";
        return 1;
      }
      in.argLP6.push_back(a);
    }
    in.pim = &prog;
    in.spec = &*spec;
    in.program = &program;
    in.weightPrefetch = weightPrefetch;
    in.pimActivationCores = pimActivationCores;
    program.beginPhase(name, exit);
    NpuPhaseStats stats;
    LogicalResult emitted = callee->hasAttr("hyb.npu_kernel")
        ? emitPimPacketNpuPhase(in, stats, err) : emitNpuPhase(in, extra, stats, err);
    if (failed(emitted)) {
      llvm::errs() << "NPU codegen failed in " << name << ": " << err << "\n";
      return 1;
    }
    exit = program.endPhase();
    logicalCores = std::max(logicalCores, int(stats.cores));
    if (extra.cursor > extra.limit) {
      llvm::errs() << "NPU constant pools overflow the reserved LP6 region\n";
      return 1;
    }
    json::Array streams;
    for (const auto &s : stats.weightStreams)
      streams.push_back(json::Object{{"tile_bytes", s.tileBytes}, {"iterations", s.iterations},
          {"slots", s.slots}, {"batch_tiles", s.batch}, {"prefetch", s.prefetch},
          {"cross_batch", s.crossBatch}, {"reason", s.reason}});
    json::Object phaseReport{{"phase", name}, {"kind", "npu"}, {"core_words", stats.coreWords},
        {"commands", stats.commands}, {"l1_peak", stats.l1Peak}, {"cores", stats.cores},
        {"l2_staging_bytes_per_core", stats.l2StagingPerCore}, {"weight_streams", std::move(streams)}};
    if (auto kernel = callee->getAttrOfType<StringAttr>("hyb.npu_kernel")) {
      phaseReport["kernel"] = kernel.getValue().str();
      auto plan = callee->getAttrOfType<StringAttr>("hyb.packet_plan");
      if (!plan) { llvm::errs() << "packet kernel did not report its checked memory plan\n"; return 1; }
      auto value = json::parse(plan.getValue());
      if (!value) { llvm::errs() << llvm::toString(value.takeError()) << '\n'; return 1; }
      phaseReport["packet_plan"] = std::move(*value);
    }
    if (stats.pimActivationCores > 1) {
      if (stats.pimActivationCores != pimActivationCores || stats.pimActivationPlanJson.empty()) {
        llvm::errs() << "parallel PIM activation feed did not report its checked plan\n";
        return 1;
      }
      auto value = json::parse(stats.pimActivationPlanJson);
      if (!value) { llvm::errs() << llvm::toString(value.takeError()) << '\n'; return 1; }
      phaseReport["pim_activation_cores"] = stats.pimActivationCores;
      phaseReport["pim_activation_plan"] = std::move(*value);
      ++parallelActivationPhases;
    }
    report.push_back(std::move(phaseReport));
  }
  if (pimActivationCores == 4 && parallelActivationPhases == 0) {
    llvm::errs() << "--pim-activation-cores=4 did not find a supported isolated PIM activation-feed phase\n";
    return 1;
  }

  // ---- collect: the step outputs into L1 (read back from the SRAM dump) --------------------
  // --debug-collect also gathers every buffer passed between phases (dense activations, the
  // INT8 PIM inputs from partition parts[0] / PU 0, the INT32 PIM outputs in [M, nPad] rows).
  json::Array outputs;
  {
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
                                     {"dtype", driver::dtypeName(cast<ShapedType>(v.getType()).getElementType())},
                                     {"shape", std::move(shape)}, {"layout", layout}});
      l1 = (l1 + uint64_t(bytes) + 63) / 64 * 64;
      return at;
    };
    auto collectDense = [&](const std::string &name, Value v, DictionaryAttr a) {
      uint64_t addr = uint64_t(a.getAs<IntegerAttr>("addr").getInt());
      uint64_t bytes = uint64_t(a.getAs<IntegerAttr>("bytes").getInt());
      uint64_t at = item(name, v, int64_t(bytes), driver::shapeOf(v.getType()), "dense");
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
            uint64_t at = item(name, v, ty.getNumElements(), driver::shapeOf(ty), layout);
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
      return 1;
    }
  }
  if (!program.finish(outDir, logicalCores)) {
    llvm::errs() << "cannot write " << outDir << "/program.bin\n";
    return 1;
  }

  // ---- io.json -------------------------------------------------------------------------------
  uint64_t imageBytes = std::max(prog.endAddress(), extra.limit);
  json::Array args = driver::argumentsJSON(step);
  std::string model = modelDir.empty() ? mod->getAttrOfType<StringAttr>("hyb.model").str() : modelDir.getValue();
  driver::writeIO(outDir, manifestFile, mod, model, imageBytes, extra, prog, pimCommandCount, std::move(args),
                  std::move(outputs), std::move(report));
  llvm::outs() << llvm::formatv("codegen: {0:f1} s, {1} PIM commands, LP6 image {2:f2} GB\n", seconds(t0),
                                pimCommandCount, double(imageBytes) / 1e9);
  // This checked plan is produced during ISA emission, after the initial IR dump.
  // Preserve it even without a storage manifest or with --no-image.
  if (pimActivationCores > 1) {
    std::error_code ec;
    llvm::raw_fd_ostream os(outDir + "/compiled.mlir", ec);
    if (ec) { llvm::errs() << ec.message() << '\n'; return 1; }
    mod.print(os, OpPrintingFlags().elideLargeElementsAttrs());
  }
  if (noImage) return 0;

  // ---- LP6 image -----------------------------------------------------------------------------
  ::npu::decode::PimWeightStorage storage;
  size_t quantized = 0;
  if (!driver::writeImage(model, outDir, imageBytes, mod, step, prog, *spec, extra, fixedStorage, importedStorage,
                          storage, quantized, err))
    return 1;
  if (fixedStorage &&
      !driver::finishPimWeightStorage(storage, importedStorage, mod, ctx, outDir, weightStorageExport, err))
    return 1;
  llvm::outs() << llvm::formatv("image: {0:f1} s ({1} quantized weights)\n", seconds(t0), quantized);
  return 0;
}
