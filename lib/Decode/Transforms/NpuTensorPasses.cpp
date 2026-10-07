//===- NpuTensorPasses.cpp - hyb-lower-npu-gemv, hyb-npu-tile ---------------===//
//
// Tensor-level NPU passes that run after partitioning:
//   hyb-lower-npu-gemv  qmatmul{target = npu} -> INT8 contraction on the NPU weight layout
//   hyb-npu-tile        core distribution (scf.forall) and L1-sized loop tiles (scf.for)
//===----------------------------------------------------------------------===//

#include "npu/Decode/Support/TargetSpec.h"
#include "npu/Decode/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::hybrid {
#define GEN_PASS_DEF_HYBLOWERNPUGEMV
#define GEN_PASS_DEF_HYBNPUTILE
#include "npu/Decode/Transforms/Passes.h.inc"
} // namespace mlir::hybrid

using namespace mlir;
using namespace mlir::hybrid;

namespace {

//===----------------------------------------------------------------------===//
// hyb-lower-npu-gemv
//===----------------------------------------------------------------------===//
struct LowerNpuGemv : OpRewritePattern<hyb::QMatmulOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(hyb::QMatmulOp op, PatternRewriter &rewriter) const override {
    if (!op.getTarget() || *op.getTarget() != "npu" || op->getNumResults() != 1 ||
        op->hasAttr("hyb.npu_packet")) return failure();
    auto qT = cast<RankedTensorType>(op.getInput().getType());
    auto accT = cast<RankedTensorType>(op.getResult(0).getType());
    int64_t M = qT.getDimSize(0), K = qT.getDimSize(1), N = accT.getDimSize(1);
    if (N % 32 != 0) return rewriter.notifyMatchFailure(op, "N must be a multiple of 32");
    Location loc = op.getLoc();
    MLIRContext *ctx = getContext();
    Type i8 = rewriter.getI8Type(), i32 = rewriter.getI32Type();
    Value w = hyb::WeightOp::create(rewriter, loc, RankedTensorType::get({N / 32, K, 32}, i8), op.getWeightAttr(),
                                    rewriter.getStringAttr("npu_nk32"));
    Value empty = tensor::EmptyOp::create(rewriter, loc, ArrayRef<int64_t>{M, N / 32, 32}, i32);
    Value zero = arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(0));
    Value init = linalg::FillOp::create(rewriter, loc, zero, empty).getResult(0);
    AffineExpr m, nb, c, k;
    bindDims(ctx, m, nb, c, k);
    SmallVector<AffineMap> maps = {AffineMap::get(4, 0, {m, k}, ctx), AffineMap::get(4, 0, {nb, k, c}, ctx),
                                   AffineMap::get(4, 0, {m, nb, c}, ctx)};
    SmallVector<utils::IteratorType> iters(3, utils::IteratorType::parallel);
    iters.push_back(utils::IteratorType::reduction);
    auto g = linalg::GenericOp::create(
        rewriter, loc, TypeRange{init.getType()}, ValueRange{op.getInput(), w}, ValueRange{init}, maps, iters,
        [&](OpBuilder &b, Location l, ValueRange args) {
          Value a = arith::ExtSIOp::create(b, l, i32, args[0]);
          Value x = arith::ExtSIOp::create(b, l, i32, args[1]);
          Value p = arith::MulIOp::create(b, l, a, x);
          linalg::YieldOp::create(b, l, ValueRange{arith::AddIOp::create(b, l, args[2], p).getResult()});
        });
    g->setAttr("hyb.npu_gemv", op.getWeightAttr());
    if (auto grp = op.getGroupAttr()) g->setAttr("hyb.group", grp);
    SmallVector<ReassociationIndices> re = {{0}, {1, 2}};
    Value acc = tensor::CollapseShapeOp::create(rewriter, loc, accT, g.getResult(0), re);
    rewriter.replaceOp(op, acc);
    return success();
  }
};

struct HybLowerNpuGemvPass : mlir::hybrid::impl::HybLowerNpuGemvBase<HybLowerNpuGemvPass> {
  void runOnOperation() override {
    auto module = getOperation()->getParentOfType<ModuleOp>();
    if (auto requested = module->getAttrOfType<StringAttr>("hyb.pim_packet_npu_requested")) {
      bool invalid = requested.getValue() != "g4";
      getOperation().walk([&](hyb::QMatmulOp q) {
        if (!q.getTarget() || *q.getTarget() != "npu") invalid = true;
        else q->setAttr("hyb.npu_packet", requested);
      });
      if (invalid) {
        getOperation().emitError("PIM packet NPU requires g4 and NPU qmatmul targets");
        return signalPassFailure();
      }
    }
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerNpuGemv>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) signalPassFailure();
  }
};

//===----------------------------------------------------------------------===//
// hyb-npu-tile
//===----------------------------------------------------------------------===//
int64_t largestDivisorAtMost(int64_t n, int64_t limit) {
  for (int64_t d = std::min(n, limit); d > 1; --d)
    if (n % d == 0) return d;
  return 1;
}

bool insideForall(Operation *op) { return op->getParentOfType<scf::ForallOp>() != nullptr; }

SmallVector<OpFoldResult> indexAttrs(OpBuilder &b, ArrayRef<int64_t> v) {
  SmallVector<OpFoldResult> r;
  for (int64_t x : v) r.push_back(b.getIndexAttr(x));
  return r;
}

struct HybNpuTilePass : mlir::hybrid::impl::HybNpuTileBase<HybNpuTilePass> {
  using Base::Base;

  TargetSpec target;

  void runOnOperation() override {
    std::string err;
    auto specOr = TargetSpec::load(spec, err);
    if (failed(specOr)) {
      getOperation().emitError(err);
      return signalPassFailure();
    }
    target = *specOr;
    for (auto fn : getOperation().getOps<func::FuncOp>()) {
      auto phase = fn->getAttrOfType<StringAttr>(kPhase);
      if (!phase || phase.getValue() != "npu") continue;
      if (fn->hasAttr("hyb.npu_kernel")) continue; // checked isolated packet plan owns its tiling
      if (failed(tileGemvs(fn)) || failed(tileAttention(fn)) || failed(tileDequants(fn))) return signalPassFailure();
      cleanup(fn);
    }
  }

  /// extract_slice(tensor.empty) -> tensor.empty, so a tile's temporaries are private to it.
  void cleanup(func::FuncOp fn) {
    RewritePatternSet patterns(&getContext());
    tensor::populateFoldTensorEmptyPatterns(patterns);
    for (Dialect *d : getContext().getLoadedDialects()) d->getCanonicalizationPatterns(patterns);
    for (RegisteredOperationName op : getContext().getRegisteredOperations())
      op.getCanonicalizationPatterns(patterns, &getContext());
    (void)applyPatternsGreedily(fn, std::move(patterns));
  }

  void markCores(LoopLikeOpInterface loop, int64_t cores) {
    loop->setAttr("hyb.cores", IntegerAttr::get(IntegerType::get(&getContext(), 64), cores));
  }

  /// Tiles `op` (scf.forall with `v` threads, or scf.for with tile sizes `v`) and fuses the
  /// producers of its tiles: linalg.fill always (accumulator inits), every linalg op when
  /// `fuseAll`. Uses of the original results are replaced.
  FailureOr<scf::SCFTileAndFuseResult> tileAndFuse(IRRewriter &rw, Operation *op, bool forall, ArrayRef<int64_t> v,
                                                   bool fuseAll) {
    scf::SCFTileAndFuseOptions opts;
    if (forall) {
      opts.tilingOptions.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
      opts.tilingOptions.setNumThreads(indexAttrs(rw, v));
    } else {
      opts.tilingOptions.setTileSizes(indexAttrs(rw, v));
    }
    opts.setFusionControlFn([fuseAll](tensor::ExtractSliceOp, OpResult producer,
                                      bool) -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
      Operation *o = producer.getOwner();
      if (isa<linalg::FillOp>(o) || (fuseAll && isa<linalg::LinalgOp>(o)))
        return scf::SCFTileAndFuseOptions::ControlFnResult{};
      return std::nullopt;
    });
    rw.setInsertionPoint(op);
    FailureOr<scf::SCFTileAndFuseResult> r = scf::tileConsumerAndFuseProducersUsingSCF(rw, cast<TilingInterface>(op), opts);
    if (succeeded(r))
      for (auto &[orig, repl] : r->replacements) rw.replaceAllUsesWith(orig, repl);
    return r;
  }

  static Operation *findGemv(const llvm::SetVector<Operation *> &ops) {
    for (Operation *o : ops)
      if (o->hasAttr("hyb.npu_gemv")) return o;
    return nullptr;
  }

  /// NPU GEMV: output blocks over the cores, then weight tiles of at most gemv_tile_bytes.
  LogicalResult tileGemvs(func::FuncOp fn) {
    SmallVector<linalg::GenericOp> gemvs;
    fn.walk([&](linalg::GenericOp g) {
      if (g->hasAttr("hyb.npu_gemv") && !insideForall(g)) gemvs.push_back(g);
    });
    IRRewriter rewriter(&getContext());
    for (linalg::GenericOp g : gemvs) {
      SmallVector<int64_t> dom = g.getStaticLoopRanges();       // (m, nb, c, k)
      int64_t NB = dom[1], K = dom[3];
      int64_t cores = largestDivisorAtMost(NB, target.npu.cores);
      Operation *tiled = g;
      if (cores > 1) {
        auto r = tileAndFuse(rewriter, g, /*forall=*/true, {0, cores, 0, 0}, /*fuseAll=*/false);
        if (failed(r)) return g.emitError("cannot distribute the GEMV over the cores");
        markCores(r->loops.front(), cores);
        tiled = findGemv(r->tiledAndFusedOps);
        if (!tiled) return g.emitError("tiled GEMV not found");
      }
      int64_t perCore = NB / cores;
      int64_t tile = largestDivisorAtMost(perCore, std::max<int64_t>(1, target.npu.gemvTileBytes / (K * 32)));
      if (tile < perCore) {
        auto r = tileAndFuse(rewriter, tiled, /*forall=*/false, {0, tile, 0, 0}, /*fuseAll=*/false);
        if (failed(r)) return tiled->emitError("cannot tile the GEMV weights");
        r->loops.front()->setAttr("hyb.stream", UnitAttr::get(&getContext()));
      }
    }
    return success();
  }

  /// Attention: the last batch_matmul of a chain over its batch dim, producers fused.
  LogicalResult tileAttention(func::FuncOp fn) {
    SmallVector<linalg::BatchMatmulOp> bmms;
    fn.walk([&](linalg::BatchMatmulOp b) { bmms.push_back(b); });
    IRRewriter rewriter(&getContext());
    for (linalg::BatchMatmulOp b : llvm::reverse(bmms)) {
      if (!b->getBlock() || insideForall(b)) continue;       // fused into a later one
      int64_t B = b.getStaticLoopRanges()[0];
      int64_t cores = largestDivisorAtMost(B, target.npu.cores);
      if (cores < 2) continue;
      auto r = tileAndFuse(rewriter, b, /*forall=*/true, {cores, 0, 0, 0}, /*fuseAll=*/true);
      if (failed(r)) return b.emitError("cannot distribute attention over the cores");
      markCores(r->loops.front(), cores);
    }
    // Producers fused into the loop are dead outside now (identical fused copies are merged by
    // the CSE that follows this pass).
    SmallVector<Operation *> dead;
    fn.walk([&](Operation *op) {
      if (isa<linalg::LinalgOp>(op) && !insideForall(op) && isOpTriviallyDead(op)) dead.push_back(op);
    });
    for (Operation *op : llvm::reverse(dead)) op->erase();
    return success();
  }

  /// Large dequant (the lm_head logits) split by columns.
  LogicalResult tileDequants(func::FuncOp fn) {
    SmallVector<hyb::DequantOp> deqs;
    fn.walk([&](hyb::DequantOp d) {
      if (!insideForall(d)) deqs.push_back(d);
    });
    IRRewriter rewriter(&getContext());
    for (hyb::DequantOp d : deqs) {
      auto t = cast<ShapedType>(d.getInit().getType());
      int64_t N = t.getDimSize(1);
      if (t.getNumElements() < target.npu.distributeMinElements) continue;
      int64_t cores = largestDivisorAtMost(N / 32, target.npu.cores);
      if (cores < 2) continue;
      scf::SCFTilingOptions opts;
      opts.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
      opts.setNumThreads(indexAttrs(rewriter, {0, cores}));
      rewriter.setInsertionPoint(d);
      FailureOr<scf::SCFTilingResult> r = scf::tileUsingSCF(rewriter, cast<TilingInterface>(d.getOperation()), opts);
      if (failed(r)) return d.emitError("cannot distribute the dequant over the cores");
      rewriter.replaceOp(d, r->replacements);
      markCores(r->loops.front(), cores);
    }
    return success();
  }
};

} // namespace
