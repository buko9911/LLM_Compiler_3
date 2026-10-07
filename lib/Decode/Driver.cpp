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
// This file holds the options and the order of the steps; DriverProgram.cpp emits the program,
// DriverPackage.cpp the metadata files, DriverImage.cpp the image, and DriverStorage.cpp the
// fixed PIM weight storage import / export.
//===----------------------------------------------------------------------===//

#include "DriverImage.h"
#include "DriverPackage.h"
#include "DriverProgram.h"
#include "DriverStorage.h"
#include "npu/Decode/Dialect/Hyb/HybDialect.h"
#include "npu/Decode/Support/TargetSpec.h"
#include "npu/Decode/Target/NpuCodegen.h"
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
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <chrono>
#include <optional>

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

/// Combinations of the experimental PIM options that the driver supports.
bool checkOptions(bool fixedStorage) {
  if ((pimActivationCores != 1 && pimActivationCores != 4) ||
      (pimActivationCores == 4 &&
       (forceTarget != "pim" || !mapOverrides.empty() || !packetNpu.empty()))) {
    llvm::errs() << "PIM activation cores must be 1 or 4; 4 requires --force=pim, no --map overrides, and no --pim-packet-npu\n";
    return false;
  }
  if (!packetNpu.empty() &&
      (packetNpu != "g4" || weightStorageImport.empty() || forceTarget != "npu" || !mapOverrides.empty())) {
    llvm::errs() << "PIM packet NPU requires g4, --force=npu, --pim-weight-storage-import, and no --map overrides\n";
    return false;
  }
  if (!packetNpu.empty() && !weightPrefetch) {
    llvm::errs() << "PIM packet NPU g4 currently requires --npu-weight-prefetch=true\n";
    return false;
  }
  if (fixedStorage && noImage) {
    llvm::errs() << "PIM weight storage import/export is incompatible with --no-image: content verification is required\n";
    return false;
  }
  return true;
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
  if (!checkOptions(fixedStorage)) return 1;
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
  if (!driver::emitPimCommands(prog, pimCommands)) return 1;
  const uint32_t pimCommandCount = uint32_t(pimCommands.size());
  if (!packetNpu.empty() && pimCommandCount != 0) {
    llvm::errs() << "PIM packet NPU storage-only plan unexpectedly emitted PIM commands\n";
    return 1;
  }
  if (!driver::writePimPrimitives(outDir, pimCommands)) return 1;

  // ---- NPU / PIM phases in program order, one program --------------------------------------
  plena::ProgramBuilder program;
  driver::setPimKernel(program, prog, fixedStorage, pimCommandCount);
  std::vector<int> exit;           // records the next phase follows
  int logicalCores = 1;
  json::Array report;
  if (!driver::emitPhases(mod, step, prog, pimCommands, *spec, program, extra, weightPrefetch, pimActivationCores,
                          exit, logicalCores, report, err))
    return 1;

  // ---- collect: the step outputs into L1 (read back from the SRAM dump) --------------------
  json::Array outputs;
  if (!driver::emitCollect(step, prog, program, exit, noOutputCollect, debugCollect, debugPhases, outputs))
    return 1;
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
