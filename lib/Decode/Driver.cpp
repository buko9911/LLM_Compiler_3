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
// DriverPackage.cpp the metadata files and DriverImage.cpp the image.
//===----------------------------------------------------------------------===//

#include "DriverImage.h"
#include "DriverPackage.h"
#include "DriverProgram.h"
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
#include <cstdlib>
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
static cl::opt<std::string> flashAttention("flash-attention",
    cl::desc("auto (default): online-softmax attention only where the full-row attention does not fit "
             "L2 / L1; on: every attention; off: none (compile error where the full row does not fit)"),
    cl::init("auto"));
static cl::list<std::string> mapOverrides("map", cl::desc("substring:target, e.g. --map=o_proj:npu (repeatable)"));
static cl::opt<bool> noImage("no-image", cl::desc("do not write lp6.bin"), cl::init(false));
static cl::opt<int> banksetPingPong("pim-bankset-pingpong",
    cl::desc("PIM weight rows on alternating bank sets: -1 the target's pim.bankset_pingpong_lead (default), "
             "0 off, 1 lead 12, 2..63 lead"), cl::init(-1));
static cl::opt<int> commonLayout("common-layout",
    cl::desc("research: one weight placement that PIM computes on and the NPU reads through a byte transposer (PimLayout.h, "
             "commonShape): 0 off (default), 1 one rank per PIM phase, 2 K halves on the two ranks, computed in turn"), cl::init(0));
static cl::opt<int> commonTurn("common-rank-turn",
    cl::desc("research, with --common-layout=2: the two ranks take strict turns of at most this many K sweeps, cutting longer K halves; 0 whole K halves in merged turns"), cl::init(0));
static cl::opt<std::string> commonOnly("common-only",
    cl::desc("research, with --common-layout: only weights whose name contains this string use the common placement"), cl::init(""));
static cl::opt<int> commonChunk("common-chunk-rows",
    cl::desc("research, with --common-layout: an NPU weight tile is read in GDMAs of this many 128-byte rows"), cl::init(8));
static cl::opt<int> collectLogitRows("collect-logit-rows",
    cl::desc("research: collect only this many logits rows when all of them fit no SRAM dump (0: all)"), cl::init(0));
static cl::opt<std::string> kv8Mode("kv8",
    cl::desc("research: the K/V cache in INT8, per (row, KV head). dequant: read back to FP16 next to the FP16 matmuls; int8: QK^T and PV in INT8 on the SA; "
             "pim: QK^T and PV of the past rows on PIM (one token). Default off"), cl::init(""));
static cl::opt<std::string> kv8Scale("kv8-scale", cl::desc("research, with --kv8: row (default): a scale per (row, KV head); head: a fixed scale per (layer, KV head)"), cl::init("row"));
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

bool checkOptions() {
  if (flashAttention != "auto" && flashAttention != "on" && flashAttention != "off") {
    llvm::errs() << "--flash-attention must be auto, on or off\n";
    return false;
  }
  return true;
}

/// Attributes the compiler itself attaches. A frontend graph must not contain them: they would
/// bypass the checks of the CLI options that set them or pretend a pass already ran.
constexpr llvm::StringLiteral kDriverOwnedAttributes[] = {
    // hyb-npu-projection-fusion
    "hyb.weight_segments", "hyb.projection_fusions",
    // hyb-npu-flash-attention
    "hyb.flash_attention", "hyb.flash_tile", "hyb.flash_decoding", "hyb.flash_math",
    // hyb-npu-attention-fusion
    "hyb.attention_fusion", "hyb.attention_fused",
    // hyb-npu-gemv-epilogue
    "hyb.gemv_epilogue", "hyb.gemv_epilogue_fusions", "hyb.gemv_pair",
    // hyb-npu-elementwise-distribute
    "hyb.elementwise_distributed", "hyb.elementwise_axis", "hyb.elementwise_distributions",
    "hyb.elementwise_candidates", "hyb.elementwise_small",
    "hyb.elementwise_unsupported", "hyb.elementwise_capacity", "hyb.elementwise_indivisible",
    "hyb.row_chain", "hyb.row_distributions", "hyb.row_fused", "hyb.transposes_distributed",
    // hyb-npu-rope
    "hyb.rope_split", "hyb.rope_splits",
    // hyb-npu-activation-residency
    "hyb.activation_residency_private_results",
    // hyb-npu-promote
    "hyb.shared_copy_staging",
    // hyb-npu-matrix-init
    "hyb.elide_matrix_init", "hyb.matrix_init_elisions",
    // hyb-npu-matmul-layout
    "hyb.matmul_rhs_pack", "hyb.matmul_rhs_packed", "hyb.matmul_rhs_packings", "hyb.matmul_transpose_output",
    "hyb.matmul_output_transposes",
    // hyb-npu-input-pipeline
    "hyb.input_pipeline_copy", "hyb.input_pipeline_tile", "hyb.input_pipeline_groups", "hyb.input_pipeline_tiles",
    // hyb-npu-epilogue-pipeline
    "hyb.epilogue_pipeline", "hyb.epilogue_consumer", "hyb.epilogue_consumers",
    // NpuCost.h: choices and their numbers
    "hyb.decisions",
};

/// The MLIR pass pipeline, in five stages. Each stage hands the next one a more concrete program:
/// math -> device and phases -> core tiles -> buffers -> addresses. Every pass runs; each one
/// decides from the shapes and the target's hardware numbers where its rewrite applies.
std::string decodePipeline(const std::string &overrideOpt) {
  const std::string spec = "{spec=" + specFile + "}";
  // 1. Tensor math: name the arguments, clean up the import, W8A8 quantization, FP16 storage.
  const std::string tensor =
      std::string("hyb-annotate{manifest=") + manifestFile + "},"  // argument kinds and checkpoint names
      "func.func(hyb-simplify,canonicalize,cse,"              // import clean-up, elementwise fusion
      "hyb-raise-linear,canonicalize,"                        // matmul(x, W^T) -> hyb.linear
      "hyb-quantize,cse,"                                     // linear -> quant + qmatmul + dequant
      "hyb-fold-transpose,canonicalize),"                     // transpose(K) into the batch_matmul map
      "hyb-storage-f16," +                                    // FP16 storage, FP32 reductions
      (kv8Mode.empty() ? std::string() : "hyb-kv8{mode=" + kv8Mode + " scale=" + kv8Scale + " spec=" + specFile + "},canonicalize,cse,");
  // 2. Device and phases: NPU or PIM per GEMV group, NPU GEMV layout, one function per phase.
  const std::string placement =
      "hyb-partition{spec=" + specFile + " report=" + outDir + "/partition.json" +
      (forceTarget.empty() ? "" : " force=" + forceTarget) + overrideOpt + "},"
      "func.func(hyb-lower-npu-gemv),"                        // INT8 contraction on [N/32, K, 32] weights
      "hyb-outline-phases,canonicalize,";
  // 3. Fusion and tiling (tensor level): fuse first, then distribute over cores and tile for L1.
  const std::string fusion =
      "hyb-npu-projection-fusion" + spec + ","                // q/k/v and gate/up -> one GEMV each
      "hyb-npu-rope" + spec + ","                             // rotate-half RoPE per core, head-major
      "hyb-npu-flash-attention{spec=" + specFile + " mode=" + flashAttention + "},"   // online softmax (--flash-attention)
      "hyb-npu-attention-fusion" + spec + ","                 // QK..PV of a head on one core
      "hyb-npu-state-fusion" + spec + ","                     // a recurrent state's updates by head blocks
      "hyb-npu-gemv-epilogue" + spec + ","                    // dequantize on the GEMV's own core
      "hyb-npu-tile" + spec + ","                             // scf.forall over cores, scf.for over tiles
      "hyb-npu-elementwise-distribute" + spec + ",cse,canonicalize,";
  // 4. Ownership and bufferization. No CSE between privatization and bufferization: it would merge
  //    the private tensor.empty values again and put phase-local activations back into LP6 outputs.
  const std::string ownership =
      "hyb-private-phase-results,"                            // PIM-read results get their own buffer
      "hyb-npu-activation-residency" + spec + ","             // phase outputs get their own buffer
      "one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map},"
      "buffer-results-to-out-params{hoist-static-allocs},canonicalize,cse,";
  // 5. Memory: LP6 addresses, L1/L2 placement, explicit copies, layout rewrites, SRAM offsets.
  const std::string memory =
      "hyb-plan-lp6{spec=" + specFile + " bankset-pingpong=" + std::to_string(banksetPingPong) +
      (commonLayout ? " common-layout=" + std::to_string(commonLayout) : std::string()) +
      (commonTurn ? " common-rank-turn=" + std::to_string(commonTurn) : std::string()) +
      (commonOnly.empty() ? std::string() : " common-only=" + commonOnly.getValue()) +
      " reuse-pim-buffers=" + (debugCollect ? "false" : "true") + "},"
      "loop-invariant-code-motion,"
      "hyb-npu-place,"                                        // L1 (one core) or L2 (shared)
      "hyb-npu-promote,"                                      // compute reads L1 only: insert copies
      "hyb-npu-matrix-init" + spec + ","                      // skip zero fills a matrix INIT overwrites
      "hyb-npu-matmul-layout" + spec + ","                    // fill SA tiles for transpose-B matmuls
      "hyb-npu-input-pipeline" + spec + ","                   // overlap the next input copy with compute
      "hyb-npu-epilogue-pipeline" + spec + ","                // dequantize GEMV tiles behind the next MMA
      "hyb-npu-memplan" + spec + ","                          // liveness-based L1/L2 offsets
      "hyb-pim-boundary" + spec;                              // read order and input feed around PIM phases
  return tensor + placement + fusion + ownership + memory;
}

} // namespace

int compileDecode(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  registerAllPasses();
  hybrid::registerHybridPasses();
  cl::ParseCommandLineOptions(argc, argv, "hybrid NPU + LPDDR6-PIM compiler\n");
  auto t0 = std::chrono::steady_clock::now();
  if (!checkOptions()) return 1;

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
  bool driverOwnedAttr = false;
  module->getOperation()->walk([&](Operation *op) {
    for (StringRef name : kDriverOwnedAttributes) driverOwnedAttr |= op->hasAttr(name);
  });
  if (driverOwnedAttr) {
    llvm::errs() << "reserved compiler IR attributes are driver-owned; use the CLI with a frontend graph\n";
    return 1;
  }
  std::string overrideOpt;
  for (const std::string &m : mapOverrides) overrideOpt += (overrideOpt.empty() ? " overrides=" : ";") + m;
  std::string pipeline = decodePipeline(overrideOpt);
  PassManager pm(&ctx);
  if (failed(parsePassPipeline(pipeline, pm, llvm::errs()))) return 1;
  if (failed(pm.run(*module))) {
    llvm::errs() << "pipeline failed\n";
    return 1;
  }
  {
    std::error_code ec;
    llvm::raw_fd_ostream os(outDir + "/pipeline.txt", ec);
    if (ec) { llvm::errs() << ec.message() << '\n'; return 1; }
    os << "builtin.module(" << pipeline << ")\n";
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
  uint64_t denseEnd = uint64_t(mod->getAttrOfType<IntegerAttr>("hyb.lp6_dense_end").getInt());
  uint64_t extraLimit = uint64_t(mod->getAttrOfType<IntegerAttr>("hyb.lp6_extra_limit").getInt());
  LP6Extra extra{denseEnd, extraLimit, {}};
  uint64_t commonEnd = 0;
  if (auto cw = mod->getAttrOfType<StringAttr>("hyb.common_weights")) {
    auto parsed = json::parse(cw.getValue());
    if (!parsed) {
      llvm::errs() << "bad hyb.common_weights\n";
      return 1;
    }
    for (const json::Value &v : *parsed->getAsArray()) {
      const json::Object *o = v.getAsObject();
      pim::CommonWeight c;
      c.logical = uint64_t(*o->getInteger("logical"));
      c.bytes = uint64_t(*o->getInteger("bytes"));
      c.n = *o->getInteger("n");
      c.k = *o->getInteger("k");
      c.rowBase = *o->getInteger("row");
      c.paired = *o->getBoolean("paired");
      c.kHalves = *o->getBoolean("k_halves");
      commonEnd = std::max<uint64_t>(commonEnd, (c.kHalves ? uint64_t(pim::kRankBytes) : 0) +
                                                    uint64_t(c.rowBase + c.rowsUsed() + 2) * uint64_t(prog.rowBytes()));
      extra.commonWeights.push_back(c);
    }
    extra.commonChunkRows = commonChunk;
  }

  // ---- PIM commands --------------------------------------------------------------------------
  std::vector<pim::Primitive> pimCommands;
  if (!driver::emitPimCommands(prog, pimCommands)) return 1;
  const uint32_t pimCommandCount = uint32_t(pimCommands.size());
  if (!driver::writePimPrimitives(outDir, pimCommands)) return 1;

  // ---- NPU / PIM phases in program order, one program --------------------------------------
  plena::ProgramBuilder program;
  program.traceOperations = std::getenv("HYBRID_PROFILE_OPERATIONS") != nullptr;
  driver::setPimKernel(program, prog);
  program.relayFarEdges = !extra.commonWeights.empty();
  std::vector<int> exit;           // records the next phase follows
  int logicalCores = 1;
  json::Array report;
  if (!driver::emitPhases(mod, step, prog, pimCommands, *spec, program, extra,
                          exit, logicalCores, report, err))
    return 1;

  // ---- collect: the step outputs into L1 (read back from the SRAM dump) --------------------
  json::Array outputs;
  std::vector<std::string> uncollected;
  if (!driver::emitCollect(step, prog, program, exit, noOutputCollect, debugCollect, debugPhases, outputs,
                           spec->npu.l2Bytes, uncollected, uint64_t(spec->npu.l1BaseBytes), uint64_t(spec->npu.l2BaseBytes),
                           uint64_t(spec->npu.l1Bytes), collectLogitRows))
    return 1;
  for (const std::string &name : uncollected)
    llvm::errs() << "output " << name << " does not fit the collected SRAM; it stays in LP6 (io.json uncollected)\n";
  if (!program.finish(outDir, logicalCores)) {
    llvm::errs() << "cannot write " << outDir << "/program.bin\n";
    return 1;
  }

  // ---- io.json -------------------------------------------------------------------------------
  uint64_t imageBytes = std::max({prog.endAddress(), extra.limit, commonEnd});
  json::Array args = driver::argumentsJSON(step);
  std::string model = modelDir.empty() ? mod->getAttrOfType<StringAttr>("hyb.model").str() : modelDir.getValue();
  json::Array missing;
  for (const std::string &name : uncollected) missing.push_back(name);
  driver::writeIO(outDir, manifestFile, mod, model, imageBytes, extra, prog, pimCommandCount, std::move(args),
                  std::move(outputs), std::move(report), std::move(missing));
  llvm::outs() << llvm::formatv("codegen: {0:f1} s, {1} PIM commands, LP6 image {2:f2} GB\n", seconds(t0),
                                pimCommandCount, double(imageBytes) / 1e9);
  if (noImage) return 0;

  // ---- LP6 image -----------------------------------------------------------------------------
  size_t quantized = 0;
  if (!driver::writeImage(model, outDir, imageBytes, mod, step, prog, *spec, extra, quantized, err)) return 1;
  llvm::outs() << llvm::formatv("image: {0:f1} s ({1} quantized weights)\n", seconds(t0), quantized);
  return 0;
}
