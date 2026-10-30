//===- NpuAttentionFusion.cpp - preserve head-local attention across views --===//
//
// The exported GQA graph reshapes scores [H, G*M, S] to [H, G, M, S] for the
// per-token mask/softmax, then collapses them before P*V. Those views used to
// stop the ordinary linalg producer-fusion walk: QK and PV were distributed,
// but softmax remained a full-tensor operation on core 0.
//
// Tile only the independent H dimension and bubble each head slice through
// the intervening reshape operations. All of a head's QK, mask, softmax and
// PV then execute in one core-local region. The key dimension, scalar bodies,
// intermediate element types and reduction ordering are unchanged. This is
// attention producer fusion, NOT online/FlashAttention or key-block tiling.
//===----------------------------------------------------------------------===//

#include "npu/Decode/Support/TargetSpec.h"
#include "npu/Decode/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/CSE.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir::hybrid {
#define GEN_PASS_DEF_HYBNPUATTENTIONFUSION
#include "npu/Decode/Transforms/Passes.h.inc"
} // namespace mlir::hybrid

using namespace mlir;
using namespace mlir::hybrid;

namespace {

int64_t headCores(int64_t heads, int64_t limit) {
  for (int64_t n = std::min(heads, limit); n > 1; --n)
    if (heads % n == 0) return n;
  return 1;
}

bool staticTensor(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type && type.hasStaticShape();
}

/// A deliberately bounded attention subgraph. Do not fuse arbitrary producers
/// of Q/K/V, other projections, or a shared softmax that has another consumer.
struct AttentionChain {
  linalg::BatchMatmulOp pv, qk;
  llvm::SetVector<Operation *> producers;
  int64_t heads = 0;
  unsigned reshapes = 0;
  bool hasExp = false, hasDivide = false, hasMax = false, hasSum = false;

  bool visit(Value value, unsigned depth = 64) {
    Operation *op = value.getDefiningOp();
    if (!op) return true; // Function argument / input boundary.
    if (!depth || op->getBlock() != pv->getBlock()) return false;
    if (producers.contains(op)) return true;
    if (isa<tensor::EmptyOp, arith::ConstantOp>(op)) return true;
    if (!staticTensor(value)) return false;
    auto type = cast<RankedTensorType>(value.getType());
    if (!type.getRank() || type.getDimSize(0) != heads) return false;

    if (auto mm = dyn_cast<linalg::BatchMatmulOp>(op)) {
      if (qk && qk != mm) return false;
      qk = mm;
      producers.insert(op);
      return true; // Q and K stay outside this fusion region.
    }
    if (auto collapse = dyn_cast<tensor::CollapseShapeOp>(op)) {
      // A head-axis slice is contiguous only when the head itself is not
      // merged with another dimension. This also prevents changing GQA maps.
      if (collapse.getReassociationIndices().front() != ReassociationIndices{0}) return false;
      ++reshapes;
      producers.insert(op);
      return visit(collapse.getSrc(), depth - 1);
    }
    if (auto expand = dyn_cast<tensor::ExpandShapeOp>(op)) {
      if (expand.getReassociationIndices().front() != ReassociationIndices{0}) return false;
      ++reshapes;
      producers.insert(op);
      return visit(expand.getSrc(), depth - 1);
    }
    if (isa<linalg::FillOp>(op)) {
      producers.insert(op);
      return true;
    }
    auto generic = dyn_cast<linalg::GenericOp>(op);
    if (!generic || generic.getNumResults() != 1 || !generic.hasPureTensorSemantics()) return false;
    // Every scalar region is preserved, not rewritten/reassociated. Exclude
    // hidden effects and index-dependent bodies from this supported pattern.
    for (Operation &scalar : generic.getRegion().front())
      if (!isa<linalg::YieldOp>(scalar) &&
          (scalar.getName().getDialectNamespace() != "arith" &&
           scalar.getName().getDialectNamespace() != "math")) return false;
    const auto iterators = generic.getIteratorTypesArray();
    if (iterators.empty() || iterators.front() != utils::IteratorType::parallel) return false;
    if (generic.getNumReductionLoops() &&
        (generic.getNumReductionLoops() != 1 || iterators.back() != utils::IteratorType::reduction))
      return false;
    // To tile H without changing any row, every result carries that parallel
    // dimension unchanged as its leading axis.
    auto maps = generic.getIndexingMapsArray();
    auto outputMap = maps.back();
    if (outputMap.getNumResults() == 0 || outputMap.getResult(0) != getAffineDimExpr(0, op->getContext()))
      return false;
    generic.walk([&](Operation *scalar) {
      hasExp |= isa<math::ExpOp>(scalar);
      hasDivide |= isa<arith::DivFOp>(scalar);
      hasMax |= generic.getNumReductionLoops() && isa<arith::MaximumFOp>(scalar);
      hasSum |= generic.getNumReductionLoops() && isa<arith::AddFOp>(scalar);
    });
    producers.insert(op);
    for (Value input : generic.getDpsInputs()) {
      // Inputs such as [M,S] masks do not carry H and are simply replicated
      // reads. Only traverse producers whose result has the head dimension.
      auto inputType = dyn_cast<RankedTensorType>(input.getType());
      Operation *def = input.getDefiningOp();
      if (!def || isa<tensor::EmptyOp, arith::ConstantOp>(def)) continue;
      if (!inputType || !inputType.hasStaticShape() || !inputType.getRank()) return false;
      if (inputType.getDimSize(0) == heads && !visit(input, depth - 1)) return false;
    }
    return true;
  }

  bool match(linalg::BatchMatmulOp candidate) {
    pv = candidate;
    if (!pv.hasPureTensorSemantics() || pv->getParentOfType<scf::ForallOp>() ||
        pv->getParentOfType<scf::ForOp>()) return false;
    if (!llvm::all_of(pv->getOperands(), staticTensor)) return false;
    auto probability = cast<RankedTensorType>(pv.getDpsInputs()[0].getType());
    if (probability.getRank() != 3 || probability.getDimSize(0) <= 0) return false;
    heads = probability.getDimSize(0);
    if (!visit(pv.getDpsInputs()[0]) || !qk || !hasExp || !hasDivide || !hasMax || !hasSum)
      return false;
    auto score = dyn_cast<RankedTensorType>(qk.getResult(0).getType());
    if (!score || score.getShape() != probability.getShape()) return false;
    for (Operation *producer : producers)
      for (Operation *user : producer->getUsers())
        if (user != pv && !producers.contains(user)) return false;
    return true;
  }
};

struct HybNpuAttentionFusionPass
    : mlir::hybrid::impl::HybNpuAttentionFusionBase<HybNpuAttentionFusionPass> {
  using Base::Base;

  void runOnOperation() override {
    std::string error;
    auto target = TargetSpec::load(spec, error);
    if (failed(target)) {
      getOperation().emitError(error);
      return signalPassFailure();
    }
    if (!target->npu.attentionFusion) return;
    // This policy is deliberately an NPU-only baseline option. A mixed
    // NPU/PIM program can have different phase ownership and memory pressure;
    // do not silently retune its NPU epilogues when the option is present.
    for (func::FuncOp fn : getOperation().getOps<func::FuncOp>())
      if (auto phase = fn->getAttrOfType<StringAttr>(kPhase); phase && phase.getValue() == "pim") return;
    for (func::FuncOp fn : getOperation().getOps<func::FuncOp>()) {
      auto phase = fn->getAttrOfType<StringAttr>(kPhase);
      if (!phase || phase.getValue() != "npu" || fn->hasAttr("hyb.npu_kernel")) continue;
      int64_t matched = 0, fused = 0, direct = 0, unsupported = 0, singleCore = 0;
      int64_t existing = 0;
      fn.walk([&](scf::ForallOp loop) { existing += loop->hasAttr("hyb.attention_fused"); });
      llvm::SmallPtrSet<Operation *, 8> consumedProducts;
      SmallVector<linalg::BatchMatmulOp> products;
      fn.walk([&](linalg::BatchMatmulOp mm) { products.push_back(mm); });
      IRRewriter rw(&getContext());
      // Reverse traversal sees PV before QK. Delete dead producers only after
      // all candidates have been considered, keeping this worklist valid.
      for (linalg::BatchMatmulOp mm : llvm::reverse(products)) {
        if (mm->use_empty() || mm->getParentOfType<scf::ForallOp>() || consumedProducts.contains(mm)) continue;
        AttentionChain chain;
        if (!chain.match(mm)) { ++unsupported; continue; }
        ++matched;
        if (!chain.reshapes) { ++direct; continue; } // Existing tiler already handles this chain.
        int64_t cores = headCores(chain.heads, target->npu.cores);
        if (cores < 2) { ++singleCore; continue; }
        scf::SCFTileAndFuseOptions options;
        options.tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
        SmallVector<OpFoldResult> threads;
        for (int64_t n : {cores, int64_t(0), int64_t(0), int64_t(0)})
          threads.push_back(rw.getIndexAttr(n));
        options.tilingOptions.setNumThreads(threads);
        options.setFusionControlFn([&](tensor::ExtractSliceOp, OpResult result, bool)
            -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
          Operation *producer = result.getOwner();
          if (isa<linalg::FillOp>(producer) || chain.producers.contains(producer))
            return scf::SCFTileAndFuseOptions::ControlFnResult{};
          return std::nullopt;
        });
        RewritePatternSet cleanup(&getContext());
        tensor::populateBubbleUpExtractSliceOpPatterns(cleanup);
        tensor::populateFoldTensorEmptyPatterns(cleanup);
        options.cleanupPatterns.emplace(std::move(cleanup));
        rw.setInsertionPoint(mm);
        auto result = scf::tileConsumerAndFuseProducersUsingSCF(rw, cast<TilingInterface>(mm.getOperation()), options);
        if (failed(result) || result->loops.empty() || !result->fusedProducers.contains(chain.qk)) {
          mm.emitError("cannot fuse the supported attention chain through its head-preserving reshapes");
          return signalPassFailure();
        }
        for (auto &[original, replacement] : result->replacements)
          rw.replaceAllUsesWith(original, replacement);
        auto loop = result->loops.front();
        loop->setAttr("hyb.cores", rw.getI64IntegerAttr(cores));
        loop->setAttr("hyb.attention_fused", rw.getUnitAttr());
        consumedProducts.insert(chain.qk);
        ++fused;
      }
      if (fused) {
        RewritePatternSet cleanup(&getContext());
        tensor::populateFoldTensorEmptyPatterns(cleanup);
        for (Dialect *dialect : getContext().getLoadedDialects()) dialect->getCanonicalizationPatterns(cleanup);
        for (RegisteredOperationName op : getContext().getRegisteredOperations())
          op.getCanonicalizationPatterns(cleanup, &getContext());
        if (failed(applyPatternsGreedily(fn, std::move(cleanup)))) return signalPassFailure();
        // Slice bubbling can introduce linearize/delinearize index operations.
        // Lower them with the standard affine utility before the decode
        // backend, which evaluates affine.apply and integer index arithmetic.
        SmallVector<Operation *> indexOps;
        fn.walk([&](Operation *op) {
          if (isa<affine::AffineLinearizeIndexOp, affine::AffineDelinearizeIndexOp>(op)) indexOps.push_back(op);
        });
        for (Operation *op : indexOps) {
          rw.setInsertionPoint(op);
          LogicalResult lowered = isa<affine::AffineLinearizeIndexOp>(op)
              ? affine::lowerAffineLinearizeIndexOp(rw, cast<affine::AffineLinearizeIndexOp>(op))
              : affine::lowerAffineDelinearizeIndexOp(rw, cast<affine::AffineDelinearizeIndexOp>(op));
          if (failed(lowered)) return signalPassFailure();
        }
        // Multiple consumers (row max, exponent and row sum) can cause the
        // tiler to clone an identical QK slice. Deduplicate before bufferizing
        // so one head tile actually performs QK once, not once per use.
        DominanceInfo dominance(fn);
        eliminateCommonSubExpressions(rw, dominance, fn);
      }
      fn->setAttr("hyb.attention_fusion", rw.getDictionaryAttr({
          rw.getNamedAttr("matched_chains", rw.getI64IntegerAttr(matched + existing)),
          rw.getNamedAttr("fused_chains", rw.getI64IntegerAttr(fused + existing)),
          rw.getNamedAttr("already_direct", rw.getI64IntegerAttr(direct)),
          rw.getNamedAttr("unsupported_products", rw.getI64IntegerAttr(unsupported)),
          rw.getNamedAttr("single_core", rw.getI64IntegerAttr(singleCore))}));
    }
  }
};

} // namespace
