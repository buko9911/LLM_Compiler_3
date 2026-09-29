#ifndef PLENA_TRANSFORMS_PASSES_H
#define PLENA_TRANSFORMS_PASSES_H
#include "plena/Target/HardwareConfig.h"
#include "mlir/Pass/Pass.h"
#include <memory>
namespace plena {
// Each compiler stage is a ModuleOp pass, so plena-compile and plena-opt run the
// same code through mlir::PassManager (IR verification after every pass,
// timing, crash reproducers, textual pipelines).
//
//   plena-legalize   1막  normalize the captured graph, reject what cannot lower
//   plena-tile       2막  tile loops and core split          (needs hardware)
//   plena-place      3막  bufferize, liveness, address plan  (needs hardware)
//   plena-lower      4막  plena.core_block / plena.dma target IR
//   encode-plena     5막  Unified Program (ISA ver 1.0) words
//
// The hardware stages take the parsed configuration from the driver, or read
// the TOML named by their `settings` option when built from a pipeline string.
std::unique_ptr<mlir::Pass> createLegalizePass();
std::unique_ptr<mlir::Pass> createTilePass(HardwareConfig hardware);
std::unique_ptr<mlir::Pass> createPlacePass(HardwareConfig hardware);
std::unique_ptr<mlir::Pass> createLowerPass();
std::unique_ptr<mlir::Pass> createEncodePass();
// Registers every pass above and the `plena-pipeline{settings=...}` pipeline.
void registerPasses();
} // namespace plena
#endif
