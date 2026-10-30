//===- NpuGemvEpilogue.cpp - hyb-npu-gemv-epilogue -------------------------===//
//
// Split a single-consumer GEMV + dequant by output columns over the cores, so each core keeps
// its INT32 partition local through dequantization and publishes only FP16. hyb-npu-tile still
// tiles the weights inside each partition.
//===----------------------------------------------------------------------===//

#include "npu/Decode/Transforms/PassSupport.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"

namespace mlir::hybrid {
#define GEN_PASS_DEF_HYBNPUGEMVEPILOGUE
#include "npu/Decode/Transforms/Passes.h.inc"
}
using namespace mlir;
using namespace mlir::hybrid;

namespace {
SmallVector<OpFoldResult> attrs(OpBuilder &b, ArrayRef<int64_t> values) {
  SmallVector<OpFoldResult> result;
  for (int64_t v : values) result.push_back(b.getIndexAttr(v));
  return result;
}
Value slice(OpBuilder &b, Location loc, Value v, ArrayRef<OpFoldResult> offsets, ArrayRef<int64_t> sizes) {
  return tensor::ExtractSliceOp::create(b, loc, v, offsets, attrs(b, sizes),
                                      attrs(b, SmallVector<int64_t>(sizes.size(), 1)));
}

struct HybNpuGemvEpiloguePass : mlir::hybrid::impl::HybNpuGemvEpilogueBase<HybNpuGemvEpiloguePass> {
  using Base::Base;
  void runOnOperation() override {
    auto target = loadTargetSpec(getOperation(), spec);
    if (failed(target)) return signalPassFailure();
    if (!target->npu.gemvEpilogueFusion) return;
    for (func::FuncOp fn : getNpuPhases(getOperation())) {
      SmallVector<hyb::DequantOp> candidates;
      for (Operation &op : fn.front())
        if (auto d = dyn_cast<hyb::DequantOp>(op)) candidates.push_back(d);
      int64_t count = 0;
      for (auto d : candidates)
        if (fuse(d, *target)) ++count;
      fn->setAttr("hyb.gemv_epilogue_fusions", IntegerAttr::get(IntegerType::get(&getContext(), 64), count));
    }
  }

  bool fuse(hyb::DequantOp d, const TargetSpec &target) {
    if (d->getNumResults() != 1) return false;
    auto collapse = d.getAcc().getDefiningOp<tensor::CollapseShapeOp>();
    auto g = collapse ? collapse.getSrc().getDefiningOp<linalg::GenericOp>() : linalg::GenericOp();
    if (!g || !g->hasAttr("hyb.npu_gemv") || g->getBlock() != d->getBlock() ||
        g.getNumDpsInputs() != 2 || g.getNumDpsInits() != 1 ||
        !g.getResult(0).hasOneUse() || !collapse.getResult().hasOneUse() ||
        g->getAttrOfType<StringAttr>("hyb.npu_gemv") != d.getWeightAttr() ||
        collapse.getReassociationIndices() != SmallVector<ReassociationIndices>{{0}, {1, 2}}) return false;
    auto q = dyn_cast<RankedTensorType>(g.getDpsInputs()[0].getType());
    auto w = dyn_cast<RankedTensorType>(g.getDpsInputs()[1].getType());
    auto c = dyn_cast<RankedTensorType>(g.getResult(0).getType());
    auto y = dyn_cast<RankedTensorType>(d.getInit().getType());
    auto s = dyn_cast<RankedTensorType>(d.getScale().getType());
    auto ws = dyn_cast<RankedTensorType>(d.getWscale().getType());
    if (!q || !w || !c || !y || !s || !ws || !q.hasStaticShape() || !w.hasStaticShape() ||
        !c.hasStaticShape() || !y.hasStaticShape() || !s.hasStaticShape() || !ws.hasStaticShape() ||
        q.getRank() != 2 || w.getRank() != 3 || c.getRank() != 3 || y.getRank() != 2 ||
        s.getRank() != 2 || ws.getRank() != 1 || !q.getElementType().isInteger(8) ||
        !w.getElementType().isInteger(8) || !c.getElementType().isInteger(32) ||
        !y.getElementType().isF16() || !s.getElementType().isF32() || !ws.getElementType().isF32()) return false;
    int64_t m = q.getDimSize(0), k = q.getDimSize(1), nb = w.getDimSize(0), n = nb * 32;
    if (m <= 0 || k <= 0 || nb <= 0 || w.getShape() != ArrayRef<int64_t>({nb, k, 32}) ||
        c.getShape() != ArrayRef<int64_t>({m, nb, 32}) || y.getShape() != ArrayRef<int64_t>({m, n}) ||
        s.getShape() != ArrayRef<int64_t>({m, 1}) || ws.getDimSize(0) != n) return false;
    auto fill = g.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
    Attribute zero;
    if (!fill || !matchPattern(fill.getInputs()[0], m_Constant(&zero)) ||
        !isa<IntegerAttr>(zero) || !cast<IntegerAttr>(zero).getValue().isZero()) return false;
    MLIRContext *ctx = &getContext();
    AffineExpr a, b, col, r;
    bindDims(ctx, a, b, col, r);
    SmallVector<AffineMap> expected = {AffineMap::get(4, 0, {a, r}, ctx),
        AffineMap::get(4, 0, {b, r, col}, ctx), AffineMap::get(4, 0, {a, b, col}, ctx)};
    if (g.getIndexingMapsArray() != expected || g.getIteratorTypesArray() !=
        SmallVector<utils::IteratorType>{utils::IteratorType::parallel, utils::IteratorType::parallel,
                                        utils::IteratorType::parallel, utils::IteratorType::reduction}) return false;
    int64_t cores = largestDivisorAtMost(nb, target.npu.cores), part = nb / cores, columns = part * 32;
    // A whole column partition is kept local. Conservative admission; the
    // authoritative liveness planner still accounts for surrounding buffers.
    int64_t tile = largestDivisorAtMost(part, std::max<int64_t>(1, target.npu.gemvTileBytes / (k * 32)));
    int64_t bytes = m * columns * 6 + columns * 4 + m * k + m * 4 + 2 * tile * k * 32 +
                    2 * m * tile * 32 * 4 + target.npu.l1ReservedBytes + 4096;
    if (bytes > target.npu.l1Bytes) return false;

    OpBuilder builder(d);
    Location loc = d.getLoc();
    auto loop = scf::ForallOp::create(builder, loc, attrs(builder, {cores}), ValueRange{d.getInit()}, std::nullopt);
    loop->setAttr("hyb.cores", builder.getI64IntegerAttr(cores));
    loop->setAttr("hyb.gemv_epilogue", builder.getUnitAttr());
    builder.setInsertionPointToStart(loop.getBody());
    auto index = [&](int64_t v) -> Value { return arith::ConstantIndexOp::create(builder, loc, v); };
    Value base = arith::MulIOp::create(builder, loc, loop.getInductionVar(0), index(part));
    Value startColumn = arith::MulIOp::create(builder, loc, base, index(32));
    OpFoldResult z = builder.getIndexAttr(0);
    Value weights = slice(builder, loc, g.getDpsInputs()[1], {base, z, z}, {part, k, 32});
    Value scale = slice(builder, loc, d.getWscale(), {startColumn}, {columns});
    Value empty = tensor::EmptyOp::create(builder, loc, ArrayRef<int64_t>{m, part, 32}, builder.getI32Type());
    Value init = linalg::FillOp::create(builder, loc, fill.getInputs()[0], empty).getResult(0);
    IRMapping mapping;
    mapping.map(g.getDpsInputs()[1], weights);
    mapping.map(g.getDpsInits()[0], init);
    auto local = cast<linalg::GenericOp>(builder.clone(*g.getOperation(), mapping));
    local.getResult(0).setType(init.getType());
    local->setAttr("hyb.gemv_epilogue", builder.getUnitAttr());
    Value acc = tensor::CollapseShapeOp::create(builder, loc,
        RankedTensorType::get({m, columns}, builder.getI32Type()), local.getResult(0),
        SmallVector<ReassociationIndices>{{0}, {1, 2}});
    Value output = tensor::EmptyOp::create(builder, loc, ArrayRef<int64_t>{m, columns}, builder.getF16Type());
    auto deq = hyb::DequantOp::create(builder, loc, TypeRange{output.getType()}, acc, d.getScale(), scale,
                                    output, d.getWeightAttr());
    builder.setInsertionPointToEnd(&loop.getTerminator().getRegion().front());
    tensor::ParallelInsertSliceOp::create(builder, loc, deq.getResult(0), loop.getRegionOutArgs()[0],
        SmallVector<OpFoldResult>{z, startColumn}, attrs(builder, {m, columns}), attrs(builder, {1, 1}));
    d.getResult(0).replaceAllUsesWith(loop.getResult(0));
    d.erase();
    collapse.erase();
    g.erase();
    return true;
  }
};
}
