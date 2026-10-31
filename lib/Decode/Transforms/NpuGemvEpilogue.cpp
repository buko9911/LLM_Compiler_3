//===- NpuGemvEpilogue.cpp - hyb-npu-gemv-epilogue -------------------------===//
//
// Split a single-consumer GEMV + dequant by output columns over the cores, so each core keeps
// its INT32 partition local through dequantization and publishes only FP16. hyb-npu-tile still
// tiles the weights inside each partition.
//
// npu.pair_gate_up: a two-segment fused GEMV (gate;up) whose halves feed one elementwise op
// (SiLU(gate) * up) is split so that core c computes the same column range of both halves and
// applies the op itself; only the op's result is published (no FP16 gate/up round trip through
// L2 and no row redistribution for the op).
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
        if ((target->npu.pairGateUp && fusePaired(d, *target)) || fuse(d, *target)) ++count;
      fn->setAttr("hyb.gemv_epilogue_fusions", IntegerAttr::get(IntegerType::get(&getContext(), 64), count));
    }
  }

  /// fill -> GEMV (hyb.npu_gemv) -> collapse -> dequant, with its static sizes.
  struct Chain {
    linalg::GenericOp gemv;
    tensor::CollapseShapeOp collapse;
    linalg::FillOp fill;
    int64_t m, k, nb, n;
  };
  std::optional<Chain> chain(hyb::DequantOp d) {
    if (d->getNumResults() != 1) return std::nullopt;
    auto collapse = d.getAcc().getDefiningOp<tensor::CollapseShapeOp>();
    auto g = collapse ? collapse.getSrc().getDefiningOp<linalg::GenericOp>() : linalg::GenericOp();
    if (!g || !g->hasAttr("hyb.npu_gemv") || g->getBlock() != d->getBlock() ||
        g.getNumDpsInputs() != 2 || g.getNumDpsInits() != 1 ||
        !g.getResult(0).hasOneUse() || !collapse.getResult().hasOneUse() ||
        g->getAttrOfType<StringAttr>("hyb.npu_gemv") != d.getWeightAttr() ||
        collapse.getReassociationIndices() != SmallVector<ReassociationIndices>{{0}, {1, 2}}) return std::nullopt;
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
        !y.getElementType().isF16() || !s.getElementType().isF32() || !ws.getElementType().isF32())
      return std::nullopt;
    int64_t m = q.getDimSize(0), k = q.getDimSize(1), nb = w.getDimSize(0), n = nb * 32;
    if (m <= 0 || k <= 0 || nb <= 0 || w.getShape() != ArrayRef<int64_t>({nb, k, 32}) ||
        c.getShape() != ArrayRef<int64_t>({m, nb, 32}) || y.getShape() != ArrayRef<int64_t>({m, n}) ||
        s.getShape() != ArrayRef<int64_t>({m, 1}) || ws.getDimSize(0) != n) return std::nullopt;
    auto fill = g.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
    Attribute zero;
    if (!fill || !matchPattern(fill.getInputs()[0], m_Constant(&zero)) ||
        !isa<IntegerAttr>(zero) || !cast<IntegerAttr>(zero).getValue().isZero()) return std::nullopt;
    MLIRContext *ctx = &getContext();
    AffineExpr a, b, col, r;
    bindDims(ctx, a, b, col, r);
    SmallVector<AffineMap> expected = {AffineMap::get(4, 0, {a, r}, ctx),
        AffineMap::get(4, 0, {b, r, col}, ctx), AffineMap::get(4, 0, {a, b, col}, ctx)};
    if (g.getIndexingMapsArray() != expected || g.getIteratorTypesArray() !=
        SmallVector<utils::IteratorType>{utils::IteratorType::parallel, utils::IteratorType::parallel,
                                        utils::IteratorType::parallel, utils::IteratorType::reduction})
      return std::nullopt;
    return Chain{g, collapse, fill, m, k, nb, n};
  }

  /// The local GEMV + dequant of weight blocks [block, block + blocks) on one core.
  Value localGemv(OpBuilder &builder, Location loc, hyb::DequantOp d, Chain ch, Value block,
                  int64_t blocks) {
    auto index = [&](int64_t v) -> Value { return arith::ConstantIndexOp::create(builder, loc, v); };
    OpFoldResult z = builder.getIndexAttr(0);
    int64_t columns = blocks * 32;
    Value column = arith::MulIOp::create(builder, loc, block, index(32));
    Value weights = slice(builder, loc, ch.gemv.getDpsInputs()[1], {block, z, z}, {blocks, ch.k, 32});
    Value scale = slice(builder, loc, d.getWscale(), {column}, {columns});
    Value empty = tensor::EmptyOp::create(builder, loc, ArrayRef<int64_t>{ch.m, blocks, 32}, builder.getI32Type());
    Value init = linalg::FillOp::create(builder, loc, ch.fill.getInputs()[0], empty).getResult(0);
    IRMapping mapping;
    mapping.map(ch.gemv.getDpsInputs()[1], weights);
    mapping.map(ch.gemv.getDpsInits()[0], init);
    auto local = cast<linalg::GenericOp>(builder.clone(*ch.gemv.getOperation(), mapping));
    local.getResult(0).setType(init.getType());
    local->setAttr("hyb.gemv_epilogue", builder.getUnitAttr());
    Value acc = tensor::CollapseShapeOp::create(builder, loc,
        RankedTensorType::get({ch.m, columns}, builder.getI32Type()), local.getResult(0),
        SmallVector<ReassociationIndices>{{0}, {1, 2}});
    Value output = tensor::EmptyOp::create(builder, loc, ArrayRef<int64_t>{ch.m, columns}, builder.getF16Type());
    return hyb::DequantOp::create(builder, loc, TypeRange{output.getType()}, acc, d.getScale(), scale, output,
                                  d.getWeightAttr()).getResult(0);
  }

  /// npu.pair_gate_up: dequant(fused[a;b]) used only as [:, :n/2] and [:, n/2:] by one
  /// all-parallel identity-map generic. Core c computes columns [c w, (c + 1) w) of both halves
  /// and the generic on them.
  bool fusePaired(hyb::DequantOp d, const TargetSpec &target) {
    auto ch = chain(d);
    if (!ch || ch->nb % 2) return false;
    auto weight = ch->gemv.getDpsInputs()[1].getDefiningOp<hyb::WeightOp>();
    auto segments = weight ? weight->getAttrOfType<ArrayAttr>("hyb.weight_segments") : ArrayAttr();
    if (!segments || segments.size() != 2) return false;
    int64_t half = ch->n / 2, halfBlocks = ch->nb / 2;
    for (Attribute s : segments) {
      auto rows = cast<DictionaryAttr>(s).getAs<IntegerAttr>("rows");
      if (!rows || rows.getInt() != half) return false;
    }
    // The two halves, each a full-row slice used once by the same generic.
    tensor::ExtractSliceOp parts[2];
    for (Operation *user : d->getUsers()) {
      auto s = dyn_cast<tensor::ExtractSliceOp>(user);
      if (!s || !s.hasUnitStride() || !s->hasOneUse() || s.getStaticSizes() != ArrayRef<int64_t>({ch->m, half}))
        return false;
      auto offsets = s.getStaticOffsets();
      if (offsets[0] != 0 || (offsets[1] != 0 && offsets[1] != half)) return false;
      tensor::ExtractSliceOp &slot = parts[offsets[1] ? 1 : 0];
      if (slot) return false;
      slot = s;
    }
    if (!parts[0] || !parts[1]) return false;
    auto op = dyn_cast<linalg::GenericOp>(*parts[0]->getUsers().begin());
    if (!op || op != dyn_cast<linalg::GenericOp>(*parts[1]->getUsers().begin()) || op->getBlock() != d->getBlock() ||
        op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1 || op.getNumReductionLoops() ||
        !op.hasPureTensorSemantics() || !op.getDpsInits()[0].getDefiningOp<tensor::EmptyOp>() ||
        !llvm::all_of(op.getIndexingMapsArray(), [](AffineMap m) { return m.isIdentity(); }) ||
        op.getResult(0).getType() != RankedTensorType::get({ch->m, half}, Float16Type::get(&getContext())))
      return false;
    int64_t cores = largestDivisorAtMost(halfBlocks, target.npu.cores), part = halfBlocks / cores,
            columns = part * 32;
    int64_t tile = largestDivisorAtMost(part, std::max<int64_t>(1, target.npu.gemvTileBytes / (ch->k * 32)));
    // Both partitions (INT32 + FP16 each), the op's output, the input and the weight tiles.
    int64_t bytes = 2 * ch->m * columns * 6 + ch->m * columns * 2 + 2 * columns * 4 + ch->m * ch->k + ch->m * 4 +
                    2 * tile * ch->k * 32 + 2 * ch->m * tile * 32 * 4 + target.npu.l1ReservedBytes + 4096;
    if (bytes > target.npu.l1Bytes) return false;

    OpBuilder builder(op);
    Location loc = op.getLoc();
    auto loop = scf::ForallOp::create(builder, loc, attrs(builder, {cores}), ValueRange{op.getDpsInits()[0]},
                                      std::nullopt);
    loop->setAttr("hyb.cores", builder.getI64IntegerAttr(cores));
    loop->setAttr("hyb.gemv_epilogue", builder.getUnitAttr());
    loop->setAttr("hyb.gemv_pair", builder.getUnitAttr());
    builder.setInsertionPointToStart(loop.getBody());
    auto index = [&](int64_t v) -> Value { return arith::ConstantIndexOp::create(builder, loc, v); };
    Value base = arith::MulIOp::create(builder, loc, loop.getInductionVar(0), index(part));
    Value first = localGemv(builder, loc, d, *ch, base, part);
    Value second = localGemv(builder, loc, d, *ch,
                             arith::AddIOp::create(builder, loc, base, index(halfBlocks)), part);
    Value output = tensor::EmptyOp::create(builder, loc, ArrayRef<int64_t>{ch->m, columns}, builder.getF16Type());
    IRMapping mapping;
    mapping.map(parts[0].getResult(), first);
    mapping.map(parts[1].getResult(), second);
    mapping.map(op.getDpsInits()[0], output);
    auto local = cast<linalg::GenericOp>(builder.clone(*op.getOperation(), mapping));
    local.getResult(0).setType(output.getType());
    OpFoldResult z = builder.getIndexAttr(0);
    Value startColumn = arith::MulIOp::create(builder, loc, base, index(32));
    builder.setInsertionPointToEnd(&loop.getTerminator().getRegion().front());
    tensor::ParallelInsertSliceOp::create(builder, loc, local.getResult(0), loop.getRegionOutArgs()[0],
        SmallVector<OpFoldResult>{z, startColumn}, attrs(builder, {ch->m, columns}), attrs(builder, {1, 1}));
    op.getResult(0).replaceAllUsesWith(loop.getResult(0));
    op.erase();
    parts[0].erase();
    parts[1].erase();
    d.erase();
    ch->collapse.erase();
    ch->gemv.erase();
    return true;
  }

  bool fuse(hyb::DequantOp d, const TargetSpec &target) {
    auto ch = chain(d);
    if (!ch) return false;
    linalg::GenericOp g = ch->gemv;
    tensor::CollapseShapeOp collapse = ch->collapse;
    linalg::FillOp fill = ch->fill;
    int64_t m = ch->m, k = ch->k, nb = ch->nb, n = ch->n;
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
