#ifndef NPU_TRANSFORMS_PIPELINE_H
#define NPU_TRANSFORMS_PIPELINE_H
#include "npu/Target/HardwareConfig.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
namespace npu {
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
// V_DOT_F16_F32 means a sum of squares (the stream dotted with itself), and
// kAbsMaxReduction the largest magnitude, V_ABS_F16 then V_REDUCE_MAX_F16_F32.
const char *reductionInstruction(mlir::Operation *op);
inline constexpr const char *kAbsMaxReduction = "V_ABS_F16+V_REDUCE_MAX_F16_F32";
// Bytes per element of the types a placed buffer may hold (f16, f32, i8, i32);
// 0 for any other type.
unsigned elementBytes(mlir::Type type);
// W8A8 (npu.quantize = "w8a8"): every matmul whose B operand is an entry
// argument becomes per-row activation quantization, an INT8 x INT8 -> INT32
// matmul on a per-output-channel INT8 weight, and a dequantization. Runs inside
// legalizeGraph, after normalisation.
mlir::LogicalResult quantizeGraph(mlir::ModuleOp module);
// The two elementwise generics quantizeGraph emits, recognised by their bodies:
// FP16 [M,K] and per-row FP32 scale -> INT8 (V_QUANT_F16_I8), and INT32 [M,N]
// with a per-row and a per-channel FP32 scale -> FP16 (V_DEQUANT_I32_F16).
bool isQuantizeGeneric(mlir::Operation *op);
bool isDequantizeGeneric(mlir::Operation *op);
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
} // namespace npu
#endif
