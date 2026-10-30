//===- TensorDestination.h - tensor result destination tracing ---*- C++ -*-===//
//
// Destination privatization must run after the final tensor CSE and before
// bufferization. Cloning an empty destination changes storage ownership, not
// arithmetic, reduction order, or the phase's externally visible tensor value.
//===----------------------------------------------------------------------===//
#ifndef NPU_DECODE_SUPPORT_TENSORDESTINATION_H
#define NPU_DECODE_SUPPORT_TENSORDESTINATION_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

namespace mlir::hybrid {

/// Find the operand that first supplies tensor.empty storage to a returned
/// tensor. Follow only explicitly tied destinations and shape-only views;
/// unknown producers, block arguments, and untied results are not candidates.
inline OpOperand *findEmptyDestination(Value value) {
  while (auto result = dyn_cast<OpResult>(value)) {
    Operation *op = result.getOwner();
    OpOperand *tied = nullptr;
    if (auto dps = dyn_cast<DestinationStyleOpInterface>(op))
      tied = dps.getTiedOpOperand(result);
    else if (auto forall = dyn_cast<scf::ForallOp>(op))
      tied = forall.getTiedOpOperand(result);
    else if (auto loop = dyn_cast<LoopLikeOpInterface>(op))
      tied = loop.getTiedLoopInit(result);
    else if (isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp, tensor::CastOp>(op))
      tied = &op->getOpOperand(0);
    if (!tied) return nullptr;
    if (tied->get().getDefiningOp<tensor::EmptyOp>()) return tied;
    value = tied->get();
  }
  return nullptr;
}

} // namespace mlir::hybrid
#endif
