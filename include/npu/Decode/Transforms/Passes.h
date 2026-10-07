#ifndef HYBRID_TRANSFORMS_PASSES_H
#define HYBRID_TRANSFORMS_PASSES_H

#include "npu/Decode/Dialect/Hyb/HybDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"

namespace mlir::hybrid {
#define GEN_PASS_DECL
#include "npu/Decode/Transforms/Passes.h.inc"
#define GEN_PASS_REGISTRATION
#include "npu/Decode/Transforms/Passes.h.inc"

/// Attribute names shared by the passes and the backends.
inline constexpr llvm::StringLiteral kKind = "hyb.kind";
inline constexpr llvm::StringLiteral kName = "hyb.name";
inline constexpr llvm::StringLiteral kPhase = "hyb.phase";   // "npu" | "pim" on outlined functions
inline constexpr llvm::StringLiteral kLP6 = "hyb.lp6";       // placement of an LP6 buffer

/// The decode step function (the only public function after import).
func::FuncOp getStepFunction(ModuleOp module);
} // namespace mlir::hybrid

#endif
