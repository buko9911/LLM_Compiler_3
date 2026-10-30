//===- NpuElementwiseDistribution.cpp - hyb-npu-elementwise-distribute ----===//
//
// Split a static, top-level, all-parallel FP16 linalg.generic with an identity output map over
// the cores, without changing its scalar body. Reductions and attention are left to the other
// passes. hyb-npu-place / -promote then give each core private L1 tiles and a shared L2 result.
//===----------------------------------------------------------------------===//

#include "npu/Decode/Transforms/PassSupport.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <algorithm>
#include <optional>

namespace mlir::hybrid {
#define GEN_PASS_DEF_HYBNPUELEMENTWISEDISTRIBUTE
#include "npu/Decode/Transforms/Passes.h.inc"
}
using namespace mlir;
using namespace mlir::hybrid;

namespace {
constexpr llvm::StringLiteral marker = "hyb.elementwise_distributed";

bool supported(linalg::GenericOp op, func::FuncOp fn) {
  if (op->getBlock() != &fn.front() || !op.hasPureTensorSemantics() ||
      op.getNumResults() != 1 || op.getNumDpsInits() != 1 ||
      op.getNumLoops() == 0 || op.getNumReductionLoops() ||
      op->hasAttr("hyb.npu_gemv") || op->hasAttr("hyb.flash_math") || op->hasAttr(marker)) return false;
  // A distributed shared destination must begin as an uninitialized fresh
  // tensor. Preserving an initialized phase argument can make bufferization
  // require an LP6 -> LP6 copy, which the current DMA backend cannot lower.
  // Such operations retain their original single-core schedule.
  if (!op.getDpsInits()[0].getDefiningOp<tensor::EmptyOp>() ||
      !op.getRegion().front().getArguments().back().use_empty()) return false;
  auto output = dyn_cast<RankedTensorType>(op.getResult(0).getType());
  if (!output || !output.hasStaticShape() || !output.getElementType().isF16() ||
      output.getRank() != op.getNumLoops()) return false;
  auto maps = op.getIndexingMapsArray();
  if (!maps.back().isIdentity()) return false;
  for (auto [value, map] : llvm::zip(op->getOperands(), maps)) {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    if (!type || !type.hasStaticShape() ||
        (!type.getElementType().isF16() && !type.getElementType().isF32()) ||
        map.getNumSymbols() || map.getNumResults() != unsigned(type.getRank())) return false;
    // Broadcast dimensions are omitted; repeated dimensions, arithmetic
    // indexing and transposed output writes are intentionally not admitted.
    SmallVector<bool> seen(op.getNumLoops(), false);
    bool vector = false;
    for (auto [axis, expression] : llvm::enumerate(map.getResults())) {
      if (auto dim = dyn_cast<AffineDimExpr>(expression)) {
        if (seen[dim.getPosition()] || type.getDimSize(axis) != output.getDimSize(dim.getPosition()))
          return false;
        seen[dim.getPosition()] = true;
        if (dim.getPosition() + 1 == op.getNumLoops()) {
          if (axis + 1 != map.getNumResults()) return false;
          vector = true;
        }
      } else if (auto constant = dyn_cast<AffineConstantExpr>(expression)) {
        if (constant.getValue() != 0 || type.getDimSize(axis) != 1) return false;
      } else return false;
    }
    // The vectorizer reads contiguous FP16 vector operands and FP32 row scalars (online attention
    // math uses its own typed path and is excluded above).
    if (vector ? !type.getElementType().isF16() : !type.getElementType().isF32()) return false;
  }
  for (Operation &scalar : op.getRegion().front()) {
    if (isa<linalg::YieldOp>(scalar)) continue;
    StringRef dialect = scalar.getName().getDialectNamespace();
    if ((dialect != "arith" && dialect != "math") || scalar.getNumRegions() ||
        !isMemoryEffectFree(&scalar)) return false;
    // A linalg.index, region-bearing op, or an external computed scalar may
    // depend on the original loop position or on unsupported register state.
    for (Value operand : scalar.getOperands()) {
      if (operand.getParentBlock() == &op.getRegion().front()) continue;
      if (!operand.getDefiningOp<arith::ConstantOp>()) return false;
    }
  }
  return llvm::all_of(output.getShape(), [](int64_t d) { return d > 0; });
}

struct Partition { unsigned axis; int64_t cores; };

std::optional<Partition> partition(linalg::GenericOp op, int64_t limit) {
  auto type = cast<RankedTensorType>(op.getResult(0).getType());
  // Prefer an outer axis: full contiguous vector rows then retain their exact
  // 32-lane chunk boundaries. If only the inner axis is usable (e.g. M=1),
  // require each partition to contain a multiple of 32 elements as well.
  for (unsigned axis = 0; axis < unsigned(type.getRank()); ++axis) {
    int64_t length = type.getDimSize(axis);
    for (int64_t cores = std::min(limit, length); cores > 1; --cores) {
      if (length % cores) continue;
      if (axis + 1 == unsigned(type.getRank()) && (length / cores) % 32) continue;
      return Partition{axis, cores};
    }
  }
  return std::nullopt;
}

bool fitsLocalTile(linalg::GenericOp op, Partition part, const TargetSpec &target) {
  // This bounds the new tile's own operand buffers, not the whole phase's live
  // set. The normal L1/L2 planner remains authoritative after bufferization.
  if (target.npu.l1Bytes <= target.npu.l1ReservedBytes) return false;
  uint64_t available = target.npu.l1Bytes - target.npu.l1ReservedBytes;
  uint64_t used = 0;
  auto maps = op.getIndexingMapsArray();
  for (auto [value, map] : llvm::zip(op->getOperands(), maps)) {
    auto type = cast<RankedTensorType>(value.getType());
    uint64_t bytes = type.getElementTypeBitWidth() / 8;
    for (auto [axis, length] : llvm::enumerate(type.getShape())) {
      int64_t dimension = length;
      if (map.getResult(axis) == getAffineDimExpr(part.axis, op.getContext())) dimension /= part.cores;
      if (dimension <= 0 || bytes > available / uint64_t(dimension)) return false;
      bytes *= uint64_t(dimension);
    }
    if (bytes > available - used) return false;
    used += bytes;
  }
  return true;
}

struct HybNpuElementwiseDistributePass
    : mlir::hybrid::impl::HybNpuElementwiseDistributeBase<HybNpuElementwiseDistributePass> {
  using Base::Base;

  void runOnOperation() override {
    auto target = loadTargetSpec(getOperation(), spec);
    if (failed(target)) return signalPassFailure();
    // An NPU-only option, like hyb-npu-attention-fusion.
    if (!target->npu.elementwiseDistribution || hasPimPhase(getOperation())) return;
    for (func::FuncOp fn : getNpuPhases(getOperation())) {
      int64_t count = 0, candidates = 0, small = 0, unsupported = 0, capacity = 0, indivisible = 0;
      fn.walk([&](scf::ForallOp loop) { count += loop->hasAttr(marker); });
      SmallVector<linalg::GenericOp> work;
      for (Operation &op : fn.front()) if (auto generic = dyn_cast<linalg::GenericOp>(op)) work.push_back(generic);
      IRRewriter rewriter(&getContext());
      for (linalg::GenericOp generic : work) {
        if (!supported(generic, fn)) { ++unsupported; continue; }
        ++candidates;
        auto type = cast<RankedTensorType>(generic.getResult(0).getType());
        if (type.getNumElements() < target->npu.distributeMinElements) { ++small; continue; }
        auto part = partition(generic, target->npu.cores);
        if (!part) { ++indivisible; continue; }
        if (!fitsLocalTile(generic, *part, *target)) { ++capacity; continue; }
        scf::SCFTilingOptions options;
        options.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
        SmallVector<OpFoldResult> threads(type.getRank(), rewriter.getIndexAttr(0));
        threads[part->axis] = rewriter.getIndexAttr(part->cores);
        options.setNumThreads(threads);
        // Cloned scalar code keeps this marker and its local shape in the
        // operation map; no source ordinal is assumed stable across passes.
        generic->setAttr(marker, rewriter.getUnitAttr());
        generic->setAttr("hyb.elementwise_original_shape", rewriter.getDenseI64ArrayAttr(type.getShape()));
        rewriter.setInsertionPoint(generic);
        auto result = scf::tileUsingSCF(rewriter, cast<TilingInterface>(generic.getOperation()), options);
        if (failed(result) || result->loops.size() != 1) {
          generic.emitError("cannot distribute the admitted independent elementwise tensor");
          return signalPassFailure();
        }
        auto loop = result->loops.front();
        loop->setAttr("hyb.cores", rewriter.getI64IntegerAttr(part->cores));
        loop->setAttr(marker, rewriter.getUnitAttr());
        loop->setAttr("hyb.elementwise_axis", rewriter.getI64IntegerAttr(part->axis));
        loop->setAttr("hyb.elementwise_original_shape", rewriter.getDenseI64ArrayAttr(type.getShape()));
        rewriter.replaceOp(generic, result->replacements);
        ++count;
      }
      RewritePatternSet cleanup(&getContext());
      tensor::populateFoldTensorEmptyPatterns(cleanup);
      for (Dialect *dialect : getContext().getLoadedDialects()) dialect->getCanonicalizationPatterns(cleanup);
      for (RegisteredOperationName op : getContext().getRegisteredOperations())
        op.getCanonicalizationPatterns(cleanup, &getContext());
      if (failed(applyPatternsGreedily(fn, std::move(cleanup)))) return signalPassFailure();
      auto record = [&](StringRef name, int64_t value) { fn->setAttr(name, rewriter.getI64IntegerAttr(value)); };
      record("hyb.elementwise_distributions", count);
      record("hyb.elementwise_candidates", candidates);
      record("hyb.elementwise_small", small);
      record("hyb.elementwise_unsupported", unsupported);
      record("hyb.elementwise_capacity", capacity);
      record("hyb.elementwise_indivisible", indivisible);
    }
  }
};
} // namespace
