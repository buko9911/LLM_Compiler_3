// Shared operation eligibility for backend source maps and scheduling provenance.
#ifndef NPU_DECODE_SUPPORT_NPUOPERATIONTRACE_H
#define NPU_DECODE_SUPPORT_NPUOPERATIONTRACE_H
#include "npu/Decode/Dialect/Hyb/HybDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir::hybrid {
// This is the backend's metadata/index-evaluation contract, NOT a generic
// side-effect analysis. In particular dealloc is not safe to move across.
inline bool isNpuTraceMetadataOnly(Operation &op) {
  if (isa<arith::ConstantOp, memref::AllocOp, memref::DeallocOp, hyb::WeightOp,
          affine::AffineApplyOp, affine::AffineMinOp, affine::AffineMaxOp,
          memref::SubViewOp, memref::ExpandShapeOp, memref::CollapseShapeOp,
          memref::CastOp>(op)) return true;
  return op.getDialect() && op.getDialect()->getNamespace() == "arith" &&
         llvm::none_of(op.getOperandTypes(), [](Type type) { return isa<MemRefType>(type); });
}
} // namespace mlir::hybrid
#endif
