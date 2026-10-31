//===- NpuElementwiseDistribution.cpp - hyb-npu-elementwise-distribute ----===//
//
// Split a static, top-level, all-parallel FP16 linalg.generic with an identity output map over
// the cores, without changing its scalar body. Attention is left to the other passes.
// hyb-npu-place / -promote then give each core private L1 tiles and a shared L2 result.
//
// With npu.row_distribution, work on whole token rows is split by rows as well: every hyb.quant
// (its scale is a maximum over the row) and every distributed elementwise op whose operand comes
// from a row reduction (RMSNorm, q/k norm). Their single-use row-wise producers (the reduction,
// its zero fill, the normalization) are fused into the same core loop, so a core reads its rows
// once and no intermediate goes through L2.
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

/// A top-level static linalg.generic whose leading loop is parallel and indexes only the leading
/// axis of the output and of the operands that depend on it: row r of the result needs only row r
/// of those operands. Trailing loops may be reductions (sum of squares of a row).
bool rowWise(linalg::GenericOp op, func::FuncOp fn) {
  if (op->getBlock() != &fn.front() || !op.hasPureTensorSemantics() || op.getNumResults() != 1 ||
      op.getNumDpsInits() != 1 || op.getNumLoops() == 0 || op->hasAttr("hyb.npu_gemv") ||
      op->hasAttr("hyb.flash_math") || op->hasAttr(marker) ||
      op.getIteratorTypesArray().front() != utils::IteratorType::parallel)
    return false;
  AffineExpr row = getAffineDimExpr(0, op.getContext());
  for (auto [value, map] : llvm::zip(op->getOperands(), op.getIndexingMapsArray())) {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    if (!type || !type.hasStaticShape() || map.getNumSymbols()) return false;
    for (auto [axis, expression] : llvm::enumerate(map.getResults()))
      if (!isa<AffineDimExpr, AffineConstantExpr>(expression) || (expression == row && axis != 0)) return false;
  }
  AffineMap output = op.getIndexingMapsArray().back();
  if (!output.getNumResults() || output.getResult(0) != row) return false;
  for (Operation &scalar : op.getRegion().front()) {
    if (isa<linalg::YieldOp>(scalar)) continue;
    StringRef dialect = scalar.getName().getDialectNamespace();
    if ((dialect != "arith" && dialect != "math") || scalar.getNumRegions() || !isMemoryEffectFree(&scalar))
      return false;
  }
  return true;
}

/// The operands of a row-chain op whose leading axis is the token row.
SmallVector<Value> rowOperands(Operation *op) {
  if (auto quant = dyn_cast<hyb::QuantOp>(op)) return {quant.getInput()};
  auto generic = cast<linalg::GenericOp>(op);
  SmallVector<Value> operands;
  AffineExpr row = getAffineDimExpr(0, op->getContext());
  for (OpOperand &operand : generic->getOpOperands()) {
    AffineMap map = generic.getMatchingIndexingMap(&operand);
    if (map.getNumResults() && map.getResult(0) == row) operands.push_back(operand.get());
  }
  return operands;
}

/// The producers fused into a row chain ending at `root`: single-use row-wise generics reached
/// through row operands, and the zero fill that initializes a fused reduction.
llvm::SetVector<Operation *> rowProducers(Operation *root, func::FuncOp fn) {
  llvm::SetVector<Operation *> chain;
  SmallVector<Operation *> work{root};
  while (!work.empty()) {
    for (Value value : rowOperands(work.pop_back_val())) {
      Operation *producer = value.getDefiningOp();
      if (!producer || !value.hasOneUse() || producer->getBlock() != &fn.front() || chain.contains(producer)) continue;
      if (isa<linalg::FillOp>(producer)) {
        chain.insert(producer);
      } else if (auto generic = dyn_cast<linalg::GenericOp>(producer); generic && rowWise(generic, fn)) {
        chain.insert(producer);
        work.push_back(producer);
      }
    }
  }
  return chain;
}

/// L1 bytes of one core's row tiles of every operand in the chain (an upper bound: an operand
/// read by two chain ops is counted twice).
bool chainFits(Operation *root, const llvm::SetVector<Operation *> &chain, int64_t cores, const TargetSpec &target) {
  if (target.npu.l1Bytes <= target.npu.l1ReservedBytes) return false;
  uint64_t available = target.npu.l1Bytes - target.npu.l1ReservedBytes, used = 0;
  SmallVector<Operation *> ops(chain.begin(), chain.end());
  ops.push_back(root);
  for (Operation *op : ops) {
    if (isa<linalg::FillOp>(op)) continue;      // its tile is a row operand of the op it initializes
    SmallVector<Value> rows = rowOperands(op);
    for (Value value : op->getOperands()) {
      auto type = dyn_cast<RankedTensorType>(value.getType());
      if (!type) continue;
      uint64_t bytes = uint64_t(type.getNumElements()) * std::max<unsigned>(1, type.getElementTypeBitWidth() / 8);
      if (isa<hyb::QuantOp>(op) || llvm::is_contained(rows, value)) bytes /= uint64_t(cores);
      used += bytes;
      if (used > available) return false;
    }
  }
  return true;
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

/// npu.row_distribution: splits every row chain over the cores by token rows. Roots are hyb.quant
/// ops and admitted elementwise generics that read a row reduction; each root is tiled with its
/// row producers fused. The last root goes first, so a quant takes in the normalization (itself
/// a root) and the reduction before them.
LogicalResult distributeRows(func::FuncOp fn, const TargetSpec &target, IRRewriter &rewriter, int64_t &chains,
                             int64_t &fused) {
  auto readsReduction = [&](Operation *op) {
    return llvm::any_of(rowProducers(op, fn), [](Operation *producer) {
      auto generic = dyn_cast<linalg::GenericOp>(producer);
      return generic && generic.getNumReductionLoops();
    });
  };
  SmallVector<Operation *> roots;
  for (Operation &op : fn.front()) {
    auto generic = dyn_cast<linalg::GenericOp>(op);
    if (isa<hyb::QuantOp>(op) || (generic && supported(generic, fn) && readsReduction(&op))) roots.push_back(&op);
  }
  for (Operation *root : llvm::reverse(roots)) {
    if (root->getBlock() != &fn.front() || root->use_empty()) continue;     // fused into a later chain
    auto type = dyn_cast<RankedTensorType>(root->getResult(0).getType());
    if (!type || !type.hasStaticShape() || type.getRank() < 2 ||
        type.getNumElements() < target.npu.distributeMinElements)
      continue;
    int64_t cores = largestDivisorAtMost(type.getDimSize(0), target.npu.cores);
    llvm::SetVector<Operation *> chain = rowProducers(root, fn);
    if (cores < 2 || !chainFits(root, chain, cores, target)) continue;
    root->setAttr(marker, rewriter.getUnitAttr());
    root->setAttr("hyb.elementwise_original_shape", rewriter.getDenseI64ArrayAttr(type.getShape()));
    for (Operation *op : chain)
      if (!isa<linalg::FillOp>(op)) op->setAttr(marker, rewriter.getUnitAttr());
    SmallVector<OpFoldResult> threads(cast<TilingInterface>(root).getLoopIteratorTypes().size(),
                                      rewriter.getIndexAttr(0));
    threads[0] = rewriter.getIndexAttr(cores);
    scf::SCFTileAndFuseOptions options;
    options.tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
    options.tilingOptions.setNumThreads(threads);
    options.setFusionControlFn([&](tensor::ExtractSliceOp, OpResult producer,
                                   bool) -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
      if (chain.contains(producer.getOwner())) return scf::SCFTileAndFuseOptions::ControlFnResult{};
      return std::nullopt;
    });
    rewriter.setInsertionPoint(root);
    auto result = scf::tileConsumerAndFuseProducersUsingSCF(rewriter, cast<TilingInterface>(root), options);
    if (failed(result) || result->loops.size() != 1) return root->emitError("cannot distribute the row chain");
    for (auto &[original, replacement] : result->replacements) rewriter.replaceAllUsesWith(original, replacement);
    Operation *loop = result->loops.front();
    loop->setAttr("hyb.cores", rewriter.getI64IntegerAttr(cores));
    loop->setAttr(marker, rewriter.getUnitAttr());
    loop->setAttr("hyb.elementwise_axis", rewriter.getI64IntegerAttr(0));
    loop->setAttr("hyb.elementwise_original_shape", rewriter.getDenseI64ArrayAttr(type.getShape()));
    int64_t producers = llvm::count_if(chain, [](Operation *op) { return !isa<linalg::FillOp>(op); });
    loop->setAttr("hyb.row_chain", rewriter.getI64IntegerAttr(producers));
    // Consumers precede their producers in `chain`, so each one is dead when it is reached. Fusion
    // leaves the slice it replaced in the loop; it is unused.
    rewriter.eraseOp(root);
    for (Operation *op : chain) {
      for (Operation *user : llvm::make_early_inc_range(op->getUsers()))
        if (isa<tensor::ExtractSliceOp>(user) && isOpTriviallyDead(user)) rewriter.eraseOp(user);
      if (!isOpTriviallyDead(op)) return op->emitError("a fused row producer is still used outside its chain");
      rewriter.eraseOp(op);
    }
    ++chains;
    fused += producers;
  }
  return success();
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
      IRRewriter rewriter(&getContext());
      int64_t rowChains = 0, rowFused = 0;
      if (target->npu.rowDistribution && failed(distributeRows(fn, *target, rewriter, rowChains, rowFused)))
        return signalPassFailure();
      count += rowChains;
      SmallVector<linalg::GenericOp> work;
      for (Operation &op : fn.front()) if (auto generic = dyn_cast<linalg::GenericOp>(op)) work.push_back(generic);
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
      // npu.distribute_transposes: a top-level FP16 transpose (a strided copy) over the cores by
      // its leading output axis; each core moves its own output rows.
      int64_t transposes = 0;
      SmallVector<linalg::TransposeOp> layouts;
      if (target->npu.distributeTransposes)
        for (Operation &op : fn.front())
          if (auto t = dyn_cast<linalg::TransposeOp>(op)) layouts.push_back(t);
      for (linalg::TransposeOp t : layouts) {
        auto type = dyn_cast<RankedTensorType>(t->getResult(0).getType());
        if (!t.hasPureTensorSemantics() || !type || !type.hasStaticShape() || !type.getElementType().isF16() ||
            type.getNumElements() < target->npu.distributeMinElements ||
            !t.getInit().getDefiningOp<tensor::EmptyOp>())
          continue;
        int64_t cores = largestDivisorAtMost(type.getDimSize(0), target->npu.cores);
        uint64_t tileBytes = 2 * uint64_t(type.getNumElements()) * 2 / uint64_t(std::max<int64_t>(cores, 1));
        if (cores < 2 || target->npu.l1Bytes <= target->npu.l1ReservedBytes ||
            tileBytes > uint64_t(target->npu.l1Bytes - target->npu.l1ReservedBytes))
          continue;
        scf::SCFTilingOptions options;
        options.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
        SmallVector<OpFoldResult> threads(type.getRank(), rewriter.getIndexAttr(0));
        threads[0] = rewriter.getIndexAttr(cores);
        options.setNumThreads(threads);
        rewriter.setInsertionPoint(t);
        auto result = scf::tileUsingSCF(rewriter, cast<TilingInterface>(t.getOperation()), options);
        if (failed(result) || result->loops.size() != 1) {
          t.emitError("cannot distribute the transpose");
          return signalPassFailure();
        }
        auto loop = result->loops.front();
        loop->setAttr("hyb.cores", rewriter.getI64IntegerAttr(cores));
        loop->setAttr(marker, rewriter.getUnitAttr());
        loop->setAttr("hyb.elementwise_axis", rewriter.getI64IntegerAttr(0));
        loop->setAttr("hyb.elementwise_original_shape", rewriter.getDenseI64ArrayAttr(type.getShape()));
        rewriter.replaceOp(t, result->replacements);
        ++transposes;
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
      if (target->npu.rowDistribution) {
        record("hyb.row_distributions", rowChains);
        record("hyb.row_fused", rowFused);
      }
      if (target->npu.distributeTransposes) record("hyb.transposes_distributed", transposes);
    }
  }
};
} // namespace
