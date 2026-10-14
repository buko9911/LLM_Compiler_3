//===----------------------------------------------------------------------===//
// Passes.cpp — the compiler stages as MLIR passes
//
// Each stage function (legalizeGraph, tileGraph, placeGraph, lowerToHardwareOps,
// emitTarget)
// keeps its implementation; this file gives it a Pass so the driver and
// npu-opt schedule it with mlir::PassManager.
//===----------------------------------------------------------------------===//
#include "npu/Transforms/Passes.h"
#include "npu/Transforms/Hybrid.h"
#include "npu/Transforms/Pipeline.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include <optional>

using namespace mlir;
namespace npu {
namespace {
// Declares every dialect and interface model a stage may create, so the pass
// manager loads them before the pass runs even when the input never used them.
void dependOnPipeline(DialectRegistry &registry) { registerPipelineDialects(registry); }

// Stages 2 and 3 size tiles and memory from the hardware file. A pass built by
// npu-compile carries the configuration the driver already parsed; one built
// from a pipeline string reads the TOML named by `settings`. Either way the
// module's fingerprint must match, as it does when the driver resumes --from.
template <typename Derived>
class HardwarePass : public PassWrapper<Derived, OperationPass<ModuleOp>> {
public:
  HardwarePass() = default;
  HardwarePass(const HardwarePass &other)
      : PassWrapper<Derived, OperationPass<ModuleOp>>(other), hardware(other.hardware) {}
  explicit HardwarePass(HardwareConfig config) : hardware(std::move(config)) {}
  void getDependentDialects(DialectRegistry &registry) const override { dependOnPipeline(registry); }
  void setSettings(const std::string &path) { settings = path; }

protected:
  Pass::Option<std::string> settings{*this, "settings",
                                     llvm::cl::desc("hardware settings .toml (the simulator's)")};
  std::optional<HardwareConfig> hardware;

  const HardwareConfig *resolve() {
    ModuleOp module = this->getOperation();
    if (!hardware) {
      if (settings.empty()) {
        module.emitError("requires the hardware configuration: settings=<hardware.toml>");
        return nullptr;
      }
      auto loaded = loadHardwareConfig(settings);
      if (!loaded) {
        module.emitError(loaded.error);
        return nullptr;
      }
      hardware = std::move(loaded.value);
    }
    auto fingerprint = module->getAttrOfType<StringAttr>("npu.hardware");
    if (fingerprint && fingerprint.getValue() != hardware->fingerprint) {
      module.emitError("module hardware fingerprint differs from settings");
      return nullptr;
    }
    return &*hardware;
  }
};

// The driver's command line as module attributes (the "명령줄을 IR 속성으로"
// section of npu-compile), for pipelines that start from a captured graph.
class AttachHardwarePass : public HardwarePass<AttachHardwarePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AttachHardwarePass)
  using HardwarePass::HardwarePass;
  // Options register themselves with *this, so a copy builds fresh ones;
  // the pass manager then copies their values over.
  AttachHardwarePass(const AttachHardwarePass &other) : HardwarePass(other) {}
  StringRef getArgument() const final { return "npu-attach-hardware"; }
  StringRef getName() const override { return "npu-attach-hardware"; }
  StringRef getDescription() const final {
    return "Record the hardware capacities and numerical policy on a graph module";
  }
  Option<bool> fp16{*this, "fp16", llvm::cl::desc("allow intermediate FP16 rounding"),
                    llvm::cl::init(false)};
  Option<bool> reciprocalDivision{*this, "reciprocal-division",
                                  llvm::cl::desc("lower a/b as a * (1/b)"), llvm::cl::init(false)};
  Option<bool> readback{*this, "readback", llvm::cl::desc("copy outputs back into L2 at the end"),
                        llvm::cl::init(false)};
  Option<std::string> quantize{*this, "quantize",
                               llvm::cl::desc("w8a8: INT8 weight GEMVs, applied by npu-legalize")};
  Option<bool> dramIntermediates{*this, "dram-intermediates",
                                 llvm::cl::desc("keep intermediate tensors in DRAM instead of L2"),
                                 llvm::cl::init(false)};
  void runOnOperation() override {
    auto *hw = resolve();
    if (!hw) return signalPassFailure();
    ModuleOp module = getOperation();
    auto *context = module.getContext();
    attachHardware(module, *hw);
    module->setAttr("npu.stage", StringAttr::get(context, "graph"));
    if (fp16) module->setAttr("npu.numerical", StringAttr::get(context, "fp16"));
    if (reciprocalDivision) module->setAttr("npu.division", StringAttr::get(context, "reciprocal"));
    if (readback) module->setAttr("npu.readback", UnitAttr::get(context));
    if (dramIntermediates) module->setAttr("npu.dram_intermediates", UnitAttr::get(context));
    if (!quantize.empty()) module->setAttr("npu.quantize", StringAttr::get(context, quantize));
  }
};

// Maps every INT8 GEMV group to the NPU or the PIM (npu.target, npu.group) and
// keeps the cost report on the module as npu.partition.
class PartitionPass : public PassWrapper<PartitionPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PartitionPass)
  PartitionPass() = default;
  PartitionPass(const PartitionPass &other) : PassWrapper(other) {}
  StringRef getArgument() const final { return "npu-partition"; }
  StringRef getName() const override { return "npu-partition"; }
  StringRef getDescription() const final {
    return "Map each W8A8 GEMV group to the NPU or the LPDDR6-PIM with the hybrid cost model";
  }
  Option<std::string> config{*this, "config", llvm::cl::desc("cost model JSON (configs/hybrid.json)")};
  Option<std::string> policy{*this, "policy", llvm::cl::desc("cost | npu | pim, overrides the config")};
  void getDependentDialects(DialectRegistry &registry) const override { dependOnPipeline(registry); }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto model = HybridCostModel::load(config);
    if (!model) {
      module.emitError(model.error);
      return signalPassFailure();
    }
    if (!policy.empty()) model.value.policy = policy;
    auto report = partitionGraph(module, model.value, {});
    if (failed(report)) return signalPassFailure();
    module->setAttr("npu.partition", StringAttr::get(module.getContext(), *report));
  }
};

class LegalizePass : public PassWrapper<LegalizePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizePass)
  StringRef getArgument() const final { return "npu-legalize"; }
  StringRef getName() const override { return "npu-legalize"; }
  StringRef getDescription() const final {
    return "Normalize a captured linalg graph and reject what the target cannot run (Stage 1)";
  }
  void getDependentDialects(DialectRegistry &registry) const override { dependOnPipeline(registry); }
  void runOnOperation() override {
    if (failed(legalizeGraph(getOperation()))) signalPassFailure();
  }
};

class TilePass : public HardwarePass<TilePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TilePass)
  using HardwarePass::HardwarePass;
  StringRef getArgument() const final { return "npu-tile"; }
  StringRef getName() const override { return "npu-tile"; }
  StringRef getDescription() const final {
    return "Choose tiling plans and build tile loops with the core split (Stage 2)";
  }
  void runOnOperation() override {
    auto *hw = resolve();
    if (!hw || failed(tileGraph(getOperation(), *hw))) signalPassFailure();
  }
};

class PlacePass : public HardwarePass<PlacePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PlacePass)
  using HardwarePass::HardwarePass;
  StringRef getArgument() const final { return "npu-place"; }
  StringRef getName() const override { return "npu-place"; }
  StringRef getDescription() const final {
    return "Bufferize and decide each buffer's memory space; FlashAttention state in L2 (Stage 3)";
  }
  void runOnOperation() override {
    auto *hw = resolve();
    if (!hw || failed(placeGraph(getOperation(), *hw))) signalPassFailure();
  }
};

class HardwareOpsPass : public PassWrapper<HardwareOpsPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HardwareOpsPass)
  StringRef getArgument() const final { return "npu-hw-ops"; }
  StringRef getName() const override { return "npu-hw-ops"; }
  StringRef getDescription() const final {
    return "Rewrite placed linalg ops as NPU hardware ops: npu.mma, npu.acc_writeout, "
           "npu.local_copy/global_copy, scf.forall mapped to #npu.core (Stage 4)";
  }
  void getDependentDialects(DialectRegistry &registry) const override { dependOnPipeline(registry); }
  void runOnOperation() override {
    if (failed(lowerToHardwareOps(getOperation()))) signalPassFailure();
  }
};

class PipelineAttentionPass : public PassWrapper<PipelineAttentionPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PipelineAttentionPass)
  StringRef getArgument() const final { return "npu-pipeline-attention"; }
  StringRef getName() const override { return "npu-pipeline-attention"; }
  StringRef getDescription() const final {
    return "Software-pipeline FlashAttention KV loops: QK of block j+1 beside softmax of "
           "block j, with per-block buffers (Stage 4b)";
  }
  void getDependentDialects(DialectRegistry &registry) const override { dependOnPipeline(registry); }
  void runOnOperation() override {
    if (failed(pipelineFlashAttention(getOperation()))) signalPassFailure();
  }
};

class AssignAddressesPass : public HardwarePass<AssignAddressesPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AssignAddressesPass)
  using HardwarePass::HardwarePass;
  StringRef getArgument() const final { return "npu-assign-addresses"; }
  StringRef getName() const override { return "npu-assign-addresses"; }
  StringRef getDescription() const final {
    return "Compute buffer liveness and assign physical L1/L2/DRAM addresses (Stage 4b)";
  }
  void runOnOperation() override {
    auto *hw = resolve();
    if (!hw || failed(assignAddresses(getOperation(), *hw))) signalPassFailure();
  }
};

class EmitPass : public PassWrapper<EmitPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EmitPass)
  StringRef getArgument() const final { return "npu-emit"; }
  StringRef getName() const override { return "npu-emit"; }
  StringRef getDescription() const final {
    return "Unroll the hardware-op IR into npu.core_block and npu.dma target IR (Stage 5)";
  }
  void getDependentDialects(DialectRegistry &registry) const override { dependOnPipeline(registry); }
  void runOnOperation() override {
    if (failed(emitTarget(getOperation()))) signalPassFailure();
  }
};

struct PipelineOptions : PassPipelineOptions<PipelineOptions> {
  Option<std::string> settings{*this, "settings", llvm::cl::desc("hardware settings .toml")};
  Option<bool> fp16{*this, "fp16", llvm::cl::desc("allow intermediate FP16 rounding"),
                    llvm::cl::init(false)};
  Option<bool> reciprocalDivision{*this, "reciprocal-division",
                                  llvm::cl::desc("lower a/b as a * (1/b)"), llvm::cl::init(false)};
  Option<bool> readback{*this, "readback", llvm::cl::desc("copy outputs back into L2 at the end"),
                        llvm::cl::init(false)};
  Option<std::string> quantize{*this, "quantize", llvm::cl::desc("w8a8: INT8 weight GEMVs")};
  Option<bool> dramIntermediates{*this, "dram-intermediates",
                                 llvm::cl::desc("keep intermediate tensors in DRAM instead of L2"),
                                 llvm::cl::init(false)};
};

template <typename P>
std::unique_ptr<P> withSettings(const std::string &settings) {
  auto pass = std::make_unique<P>();
  pass->setSettings(settings);
  return pass;
}
} // namespace

std::unique_ptr<Pass> createLegalizePass() { return std::make_unique<LegalizePass>(); }
std::unique_ptr<Pass> createTilePass(HardwareConfig hardware) {
  return std::make_unique<TilePass>(std::move(hardware));
}
std::unique_ptr<Pass> createPlacePass(HardwareConfig hardware) {
  return std::make_unique<PlacePass>(std::move(hardware));
}
std::unique_ptr<Pass> createHardwareOpsPass() { return std::make_unique<HardwareOpsPass>(); }
std::unique_ptr<Pass> createPipelineAttentionPass() { return std::make_unique<PipelineAttentionPass>(); }
std::unique_ptr<Pass> createAssignAddressesPass(HardwareConfig hardware) {
  return std::make_unique<AssignAddressesPass>(std::move(hardware));
}
std::unique_ptr<Pass> createEmitPass() { return std::make_unique<EmitPass>(); }

void registerPasses() {
  PassRegistration<AttachHardwarePass>();
  PassRegistration<LegalizePass>();
  PassRegistration<PartitionPass>();
  PassRegistration<TilePass>();
  PassRegistration<PlacePass>();
  PassRegistration<HardwareOpsPass>();
  PassRegistration<PipelineAttentionPass>();
  PassRegistration<AssignAddressesPass>();
  PassRegistration<EmitPass>();
  registerPass([] { return createEncodePass(); });
  // The whole of npu-compile --from=graph, minus writing the package files.
  PassPipelineRegistration<PipelineOptions>(
      "npu-pipeline", "Captured linalg graph to an encoded Unified Program (ISA ver 2.0)",
      [](OpPassManager &pm, const PipelineOptions &options) {
        auto attach = withSettings<AttachHardwarePass>(options.settings);
        attach->fp16 = options.fp16;
        attach->reciprocalDivision = options.reciprocalDivision;
        attach->readback = options.readback;
        attach->quantize = options.quantize;
        attach->dramIntermediates = options.dramIntermediates;
        pm.addPass(std::move(attach));
        pm.addPass(createLegalizePass());
        pm.addPass(withSettings<TilePass>(options.settings));
        pm.addPass(withSettings<PlacePass>(options.settings));
        pm.addPass(createHardwareOpsPass());
        pm.addPass(createPipelineAttentionPass());
        pm.addPass(withSettings<AssignAddressesPass>(options.settings));
        pm.addPass(createEmitPass());
        pm.addPass(createEncodePass());
      });
}
} // namespace npu
