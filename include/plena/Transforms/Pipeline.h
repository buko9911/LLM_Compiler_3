#ifndef PLENA_TRANSFORMS_PIPELINE_H
#define PLENA_TRANSFORMS_PIPELINE_H
#include "plena/Target/HardwareConfig.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
namespace plena {
// Global DRAM-to-DRAM copies reserve this L2 window in every placed graph.
inline constexpr uint64_t kDRAMStagingBytes = 64 * 1024;
void registerPipelineDialects(mlir::DialectRegistry &registry);
void attachHardware(mlir::ModuleOp module, const HardwareConfig &hardware);
mlir::LogicalResult legalizeGraph(mlir::ModuleOp module);
mlir::LogicalResult tileGraph(mlir::ModuleOp module, const HardwareConfig &hardware);
mlir::LogicalResult placeGraph(mlir::ModuleOp module, const HardwareConfig &hardware);
mlir::LogicalResult lowerPlacedGraph(mlir::ModuleOp module);
mlir::LogicalResult cleanIR(mlir::ModuleOp module);
// One table serves the legality check and ISA selection, so no scalar operation
// is ever admitted without a matching VPU instruction. Null means unsupported.
bool isSupportedElementwiseScalar(mlir::Operation *op);
const char *elementwiseInstruction(mlir::Operation *op);
// Non-null when the op is a recognised row reduction; names its V_REDUCE variant.
const char *reductionInstruction(mlir::Operation *op);
// Non-null when the scalar op can apply a broadcast FP32 scalar to an FP16 stream.
const char *broadcastInstruction(mlir::Operation *op);
// A width cast names an existing stream: FP16 streams and FP32 scalars are what
// the hardware already holds, so --fp16 admits the cast without an instruction.
bool isElementwiseWidthCast(mlir::Operation *op);
// Non-null for a rank-1 FP32 lane operation; reciprocal is recognised separately
// because it takes one operand where arith.divf carries two.
const char *scalarLaneInstruction(mlir::Operation *op);
bool isScalarLaneReciprocal(mlir::Operation *op);
// True when the scalar unit has no such instruction and the value has to make a
// one-element trip through the vector unit instead.
bool isScalarLaneVectorDetour(mlir::Operation *op);
// True when a/b is admitted as a * (1/b). Two instructions, never one.
bool isReciprocalDivision(mlir::Operation *op);
} // namespace plena
#endif
