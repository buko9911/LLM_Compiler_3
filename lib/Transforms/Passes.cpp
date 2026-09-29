//===----------------------------------------------------------------------===//
// Passes.cpp — the compiler stages as MLIR passes
//
// Each stage function (legalizeGraph, tileGraph, placeGraph, lowerPlacedGraph)
// keeps its implementation; this file gives it a Pass so the driver and
// plena-opt schedule it with mlir::PassManager.
//===----------------------------------------------------------------------===//
#include "plena/Transforms/Passes.h"
#include "plena/Transforms/Pipeline.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include <optional>

using namespace mlir;
namespace plena {
namespace {
// Declares every dialect and interface model a stage may create, so the pass
// manager loads them before the pass runs even when the input never used them.
void dependOnPipeline(DialectRegistry &registry) { registerPipelineDialects(registry); }

// Stages 2 and 3 size tiles and memory from the hardware file. A pass built by
// plena-compile carries the configuration the driver already parsed; one built
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
    auto fingerprint = module->getAttrOfType<StringAttr>("plena.hardware");
    if (fingerprint && fingerprint.getValue() != hardware->fingerprint) {
      module.emitError("module hardware fingerprint differs from settings");
      return nullptr;
    }
    return &*hardware;
  }
};

// The driver's command line as module attributes (the "명령줄을 IR 속성으로"
// section of plena-compile), for pipelines that start from a captured graph.
class AttachHardwarePass : public HardwarePass<AttachHardwarePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AttachHardwarePass)
  using HardwarePass::HardwarePass;
  // Options register themselves with *this, so a copy builds fresh ones;
  // the pass manager then copies their values over.
  AttachHardwarePass(const AttachHardwarePass &other) : HardwarePass(other) {}
  StringRef getArgument() const final { return "plena-attach-hardware"; }
  StringRef getName() const override { return "plena-attach-hardware"; }
  StringRef getDescription() const final {
    return "Record the hardware capacities and numerical policy on a graph module";
  }
  Option<bool> fp16{*this, "fp16", llvm::cl::desc("allow intermediate FP16 rounding"),
                    llvm::cl::init(false)};
  Option<bool> reciprocalDivision{*this, "reciprocal-division",
                                  llvm::cl::desc("lower a/b as a * (1/b)"), llvm::cl::init(false)};
  Option<bool> readback{*this, "readback", llvm::cl::desc("copy outputs back into L2 at the end"),
                        llvm::cl::init(false)};
  void runOnOperation() override {
    auto *hw = resolve();
    if (!hw) return signalPassFailure();
    ModuleOp module = getOperation();
    auto *context = module.getContext();
    attachHardware(module, *hw);
    module->setAttr("plena.stage", StringAttr::get(context, "graph"));
    if (fp16) module->setAttr("plena.numerical", StringAttr::get(context, "fp16"));
    if (reciprocalDivision) module->setAttr("plena.division", StringAttr::get(context, "reciprocal"));
    if (readback) module->setAttr("plena.readback", UnitAttr::get(context));
  }
};

class LegalizePass : public PassWrapper<LegalizePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizePass)
  StringRef getArgument() const final { return "plena-legalize"; }
  StringRef getName() const override { return "plena-legalize"; }
  StringRef getDescription() const final {
    return "Normalize a captured linalg graph and reject what the target cannot run (1막)";
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
  StringRef getArgument() const final { return "plena-tile"; }
  StringRef getName() const override { return "plena-tile"; }
  StringRef getDescription() const final {
    return "Choose tiling plans and build tile loops with the core split (2막)";
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
  StringRef getArgument() const final { return "plena-place"; }
  StringRef getName() const override { return "plena-place"; }
  StringRef getDescription() const final {
    return "Bufferize, compute buffer liveness and assign addresses (3막)";
  }
  void runOnOperation() override {
    auto *hw = resolve();
    if (!hw || failed(placeGraph(getOperation(), *hw))) signalPassFailure();
  }
};

class LowerPass : public PassWrapper<LowerPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerPass)
  StringRef getArgument() const final { return "plena-lower"; }
  StringRef getName() const override { return "plena-lower"; }
  StringRef getDescription() const final {
    return "Lower a placed graph to plena.core_block and plena.dma target IR (4막)";
  }
  void getDependentDialects(DialectRegistry &registry) const override { dependOnPipeline(registry); }
  void runOnOperation() override {
    if (failed(lowerPlacedGraph(getOperation()))) signalPassFailure();
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
std::unique_ptr<Pass> createLowerPass() { return std::make_unique<LowerPass>(); }

void registerPasses() {
  PassRegistration<AttachHardwarePass>();
  PassRegistration<LegalizePass>();
  PassRegistration<TilePass>();
  PassRegistration<PlacePass>();
  PassRegistration<LowerPass>();
  registerPass([] { return createEncodePass(); });
  // The whole of plena-compile --from=graph, minus writing the package files.
  PassPipelineRegistration<PipelineOptions>(
      "plena-pipeline", "Captured linalg graph to an encoded Unified Program (ISA ver 1.0)",
      [](OpPassManager &pm, const PipelineOptions &options) {
        auto attach = withSettings<AttachHardwarePass>(options.settings);
        attach->fp16 = options.fp16;
        attach->reciprocalDivision = options.reciprocalDivision;
        attach->readback = options.readback;
        pm.addPass(std::move(attach));
        pm.addPass(createLegalizePass());
        pm.addPass(withSettings<TilePass>(options.settings));
        pm.addPass(withSettings<PlacePass>(options.settings));
        pm.addPass(createLowerPass());
        pm.addPass(createEncodePass());
      });
}
} // namespace plena
