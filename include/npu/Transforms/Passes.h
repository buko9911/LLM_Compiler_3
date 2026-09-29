#ifndef NPU_TRANSFORMS_PASSES_H
#define NPU_TRANSFORMS_PASSES_H
#include "npu/Target/HardwareConfig.h"
#include "mlir/Pass/Pass.h"
#include <memory>
namespace npu {
// Each compiler stage is a ModuleOp pass, so npu-compile and npu-opt run the
// same code through mlir::PassManager (IR verification after every pass,
// timing, crash reproducers, textual pipelines).
//
//   npu-legalize   Stage 1  normalize the captured graph, reject what cannot lower
//   npu-tile       Stage 2  tile loops and core split          (needs hardware)
//   npu-place      Stage 3  bufferize, liveness, address plan  (needs hardware)
//   npu-lower      Stage 4  npu.core_block / npu.dma target IR
//   encode-npu     Stage 5  Unified Program (ISA ver 1.0) words
//
// The hardware stages take the parsed configuration from the driver, or read
// the TOML named by their `settings` option when built from a pipeline string.
std::unique_ptr<mlir::Pass> createLegalizePass();
std::unique_ptr<mlir::Pass> createTilePass(HardwareConfig hardware);
std::unique_ptr<mlir::Pass> createPlacePass(HardwareConfig hardware);
std::unique_ptr<mlir::Pass> createLowerPass();
std::unique_ptr<mlir::Pass> createEncodePass();
// Registers every pass above and the `npu-pipeline{settings=...}` pipeline.
void registerPasses();
} // namespace npu
#endif
