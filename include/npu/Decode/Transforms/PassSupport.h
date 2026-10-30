//===- PassSupport.h - helpers shared by the decode passes ------*- C++ -*-===//
//
// Small utilities that most NPU passes need: the target spec of a pass's `spec` option, the NPU
// phase functions to transform, the buffer behind a memref view, and core-count divisors.
//===----------------------------------------------------------------------===//
#ifndef NPU_DECODE_TRANSFORMS_PASSSUPPORT_H
#define NPU_DECODE_TRANSFORMS_PASSSUPPORT_H

#include "npu/Decode/Support/TargetSpec.h"
#include "npu/Decode/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include <algorithm>

namespace mlir::hybrid {

/// Loads the target spec named by a pass option. On failure the error is emitted on `op`.
inline FailureOr<TargetSpec> loadTargetSpec(Operation *op, StringRef path) {
  std::string error;
  FailureOr<TargetSpec> spec = TargetSpec::load(path, error);
  if (failed(spec)) op->emitError(error);
  return spec;
}

/// The NPU phase functions the NPU passes transform: outlined functions with hyb.phase = "npu",
/// except the isolated PIM packet kernel (hyb.npu_kernel), which owns a separately checked plan.
inline SmallVector<func::FuncOp> getNpuPhases(ModuleOp module) {
  SmallVector<func::FuncOp> phases;
  for (auto fn : module.getOps<func::FuncOp>()) {
    auto phase = fn->getAttrOfType<StringAttr>(kPhase);
    if (phase && phase.getValue() == "npu" && !fn.isExternal() && !fn->hasAttr("hyb.npu_kernel"))
      phases.push_back(fn);
  }
  return phases;
}

/// Whether the module has a PIM phase. The NPU-only optional passes leave such modules alone.
inline bool hasPimPhase(ModuleOp module) {
  return llvm::any_of(module.getOps<func::FuncOp>(), [](func::FuncOp fn) {
    auto phase = fn->getAttrOfType<StringAttr>(kPhase);
    return phase && phase.getValue() == "pim";
  });
}

/// Ops that only reinterpret a memref (no data movement).
inline bool isViewOp(Operation *op) {
  return isa<memref::SubViewOp, memref::ExpandShapeOp, memref::CollapseShapeOp, memref::CastOp,
             memref::ReinterpretCastOp>(op);
}

/// The buffer a memref value views: the allocation or block argument behind its view ops.
inline Value viewRoot(Value value) {
  while (Operation *op = value.getDefiningOp()) {
    if (!isViewOp(op)) break;
    value = op->getOperand(0);
  }
  return value;
}

/// The largest divisor of `n` that is at most `limit`, or 1 when there is none above 1. Used to
/// pick how many cores share a dimension evenly.
inline int64_t largestDivisorAtMost(int64_t n, int64_t limit) {
  for (int64_t d = std::min(n, limit); d > 1; --d)
    if (n % d == 0) return d;
  return 1;
}

} // namespace mlir::hybrid
#endif
