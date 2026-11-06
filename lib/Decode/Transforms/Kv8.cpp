//===- Kv8.cpp - hyb-kv8 ----------------------------------------------------===//
//
// Research option --kv8: the K/V cache in INT8 (W8A8KV8). Runs right after hyb-storage-f16.
//
// Quant rule, --kv8-scale=row (default): symmetric INT8 per (token row, KV head): s = max|row| / 127 in FP32, q = round_even(row / s).
// A row's bytes depend on nothing but the row, so a token computed alone or in a block stores the same row. A new row is
// quantized before any row of the step reads it, itself included, so a block sees what token-by-token decode reads from the cache.
// Step inputs: past_k.i, past_v.i INT8 in the same layouts; past_ks.i, past_vs.i FP32 [KVH, P], the scale of every cached
// row. Step outputs: new_k.i, new_v.i INT8; new_ks.i, new_vs.i FP32.
// --kv8-scale=head: one scale per (layer, KV head) fixed by the host (past_ks.i, past_vs.i FP32 [KVH]): no scale per row to store, no
// P * V scale before PV, and the row's bytes depend only on the row and that constant; values beyond 127 steps saturate.
//
//   mode=dequant  K and V dequantized to FP16 next to the FP16 matmuls (Q and P stay FP16).
//   mode=int8     QK^T and PV on the SA in INT8: Q and P quantized per row, P after it was multiplied by the V scales, and
//                 the INT32 results dequantized with both scales.
//   mode=pim      as int8, with QK^T and PV of the past rows on PIM (one token, see lowerPimAttention).
//
// The new ops are linalg.generics marked hyb.kv_op, so tiling and fusion treat them as any other generic; their bodies
// state the math and the backend lowers them to the conversion instructions (CodegenOps.cpp lowerKv).
//===----------------------------------------------------------------------===//

#include "npu/Decode/Transforms/PassSupport.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "npu/Target/PimLayout.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/StringExtras.h"
#include <functional>
#include <map>

namespace mlir::hybrid {
#define GEN_PASS_DEF_HYBKV8
#include "npu/Decode/Transforms/Passes.h.inc"
} // namespace mlir::hybrid

using namespace mlir;
using namespace mlir::hybrid;

namespace {

Value emptyOf(OpBuilder &b, Location loc, ArrayRef<int64_t> shape, Type et) {
  return tensor::EmptyOp::create(b, loc, shape, et);
}

RankedTensorType tensorOf(Value v) { return cast<RankedTensorType>(v.getType()); }

/// s[.., r] = max_k |x[.., r, k]| (the backend also divides by 127, the hyb.quant rule).
Value rowScale(OpBuilder &b, Location loc, Value x) {
  auto t = tensorOf(x);
  int64_t rank = t.getRank();
  MLIRContext *ctx = b.getContext();
  SmallVector<AffineExpr> keep;
  for (int64_t d = 0; d + 1 < rank; ++d) keep.push_back(getAffineDimExpr(d, ctx));
  SmallVector<AffineMap> maps = {AffineMap::getMultiDimIdentityMap(rank, ctx), AffineMap::get(rank, 0, keep, ctx)};
  SmallVector<utils::IteratorType> iters(rank - 1, utils::IteratorType::parallel);
  iters.push_back(utils::IteratorType::reduction);
  Value init = emptyOf(b, loc, t.getShape().drop_back(), b.getF32Type());
  auto g = linalg::GenericOp::create(b, loc, TypeRange{init.getType()}, ValueRange{x}, ValueRange{init}, maps, iters,
      [&](OpBuilder &bb, Location l, ValueRange a) {
        Value e = arith::ExtFOp::create(bb, l, bb.getF32Type(), a[0]);
        Value m = math::AbsFOp::create(bb, l, e);
        linalg::YieldOp::create(bb, l, ValueRange{m});
      });
  g->setAttr("hyb.kv_op", b.getStringAttr("row_scale"));
  return g.getResult(0);
}

/// q[.., r, k] = round_even(x[.., r, k] / s[.., r]).
Value quantRows(OpBuilder &b, Location loc, Value x, Value s) {
  auto t = tensorOf(x);
  int64_t rank = t.getRank();
  MLIRContext *ctx = b.getContext();
  SmallVector<AffineExpr> keep;
  for (int64_t d = 0; d + 1 < rank; ++d) keep.push_back(getAffineDimExpr(d, ctx));
  AffineMap id = AffineMap::getMultiDimIdentityMap(rank, ctx);
  SmallVector<AffineMap> maps = {id, AffineMap::get(rank, 0, keep, ctx), id};
  SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
  Value init = emptyOf(b, loc, t.getShape(), b.getI8Type());
  auto g = linalg::GenericOp::create(b, loc, TypeRange{init.getType()}, ValueRange{x, s}, ValueRange{init}, maps, iters,
      [&](OpBuilder &bb, Location l, ValueRange a) {
        Value e = arith::ExtFOp::create(bb, l, bb.getF32Type(), a[0]);
        Value d = arith::DivFOp::create(bb, l, e, a[1]);
        Value r = math::RoundEvenOp::create(bb, l, d);
        Value q = arith::FPToSIOp::create(bb, l, bb.getI8Type(), r);
        linalg::YieldOp::create(bb, l, ValueRange{q});
      });
  g->setAttr("hyb.kv_op", b.getStringAttr("quant"));
  return g.getResult(0);
}

/// q[h, r, k] = round_even(x[h, r, k] / s[h]), the scale fixed per KV head.
Value quantFixed(OpBuilder &b, Location loc, Value x, Value s) {
  auto t = tensorOf(x);
  MLIRContext *ctx = b.getContext();
  AffineMap id = AffineMap::getMultiDimIdentityMap(3, ctx);
  SmallVector<AffineMap> maps = {id, AffineMap::get(3, 0, {getAffineDimExpr(0, ctx)}, ctx), id};
  SmallVector<utils::IteratorType> iters(3, utils::IteratorType::parallel);
  Value init = emptyOf(b, loc, t.getShape(), b.getI8Type());
  auto g = linalg::GenericOp::create(b, loc, TypeRange{init.getType()}, ValueRange{x, s}, ValueRange{init}, maps, iters,
      [&](OpBuilder &bb, Location l, ValueRange a) {
        Value e = arith::ExtFOp::create(bb, l, bb.getF32Type(), a[0]);
        Value d = arith::DivFOp::create(bb, l, e, a[1]);
        Value r = math::RoundEvenOp::create(bb, l, d);
        Value q = arith::FPToSIOp::create(bb, l, bb.getI8Type(), r);
        linalg::YieldOp::create(bb, l, ValueRange{q});
      });
  g->setAttr("hyb.kv_op", b.getStringAttr("quant"));
  return g.getResult(0);
}

/// y[h, r, n] = acc[h, r, n] * s0[h, r] (* s1[h, n] when given), FP16. acc is INT32 ("dequant"), or INT8 ("dequant_i8", a
/// cached K or V read back to FP16, with the scale on one axis only: rowScaled [H, P, D], otherwise [H, D, P]).
Value dequantOf(OpBuilder &b, Location loc, Value acc, Value s0, Value s1, StringRef kind, Value sHead = Value()) {
  auto t = tensorOf(acc);
  MLIRContext *ctx = b.getContext();
  AffineExpr h = getAffineDimExpr(0, ctx), r = getAffineDimExpr(1, ctx), n = getAffineDimExpr(2, ctx);
  AffineMap id = AffineMap::getMultiDimIdentityMap(3, ctx);
  SmallVector<Value> ins = {acc};
  SmallVector<AffineMap> maps = {id};
  if (s0) {
    ins.push_back(s0);
    maps.push_back(AffineMap::get(3, 0, {h, r}, ctx));
  }
  if (s1) {
    ins.push_back(s1);
    maps.push_back(AffineMap::get(3, 0, {h, n}, ctx));
  }
  if (sHead) {       // a scale per KV head, s[h]
    ins.push_back(sHead);
    maps.push_back(AffineMap::get(3, 0, {h}, ctx));
  }
  maps.push_back(id);
  SmallVector<utils::IteratorType> iters(3, utils::IteratorType::parallel);
  Value init = emptyOf(b, loc, t.getShape(), b.getF16Type());
  auto g = linalg::GenericOp::create(b, loc, TypeRange{init.getType()}, ins, ValueRange{init}, maps, iters,
      [&](OpBuilder &bb, Location l, ValueRange a) {
        Value y = arith::SIToFPOp::create(bb, l, bb.getF32Type(), a[0]);
        for (size_t i = 1; i + 1 < a.size(); ++i) y = arith::MulFOp::create(bb, l, y, a[i]);
        Value f = arith::TruncFOp::create(bb, l, bb.getF16Type(), y);
        linalg::YieldOp::create(bb, l, ValueRange{f});
      });
  g->setAttr("hyb.kv_op", b.getStringAttr(kind));
  return g.getResult(0);
}

/// P[h, r, n] * s[h, n]: an ordinary generic of FP16 and FP32 (the backend's typed path).
Value scaleColumns(OpBuilder &b, Location loc, Value p, Value s) {
  auto t = tensorOf(p);
  MLIRContext *ctx = b.getContext();
  AffineExpr h = getAffineDimExpr(0, ctx), n = getAffineDimExpr(2, ctx);
  AffineMap id = AffineMap::getMultiDimIdentityMap(3, ctx);
  SmallVector<AffineMap> maps = {id, AffineMap::get(3, 0, {h, n}, ctx), id};
  SmallVector<utils::IteratorType> iters(3, utils::IteratorType::parallel);
  Value init = emptyOf(b, loc, t.getShape(), b.getF16Type());
  return linalg::GenericOp::create(b, loc, TypeRange{init.getType()}, ValueRange{p, s}, ValueRange{init}, maps, iters,
      [&](OpBuilder &bb, Location l, ValueRange a) {
        Value e = arith::ExtFOp::create(bb, l, bb.getF32Type(), a[0]);
        Value m = arith::MulFOp::create(bb, l, e, a[1]);
        Value f = arith::TruncFOp::create(bb, l, bb.getF16Type(), m);
        linalg::YieldOp::create(bb, l, ValueRange{f});
      }).getResult(0);
}

Value transposed(OpBuilder &b, Location loc, Value x, ArrayRef<int64_t> perm) {
  auto t = tensorOf(x);
  SmallVector<int64_t> shape;
  for (int64_t p : perm) shape.push_back(t.getDimSize(p));
  Value init = emptyOf(b, loc, shape, t.getElementType());
  return linalg::TransposeOp::create(b, loc, x, init, perm)->getResult(0);
}

Value i32Zeros(OpBuilder &b, Location loc, ArrayRef<int64_t> shape) {
  Value zero = arith::ConstantOp::create(b, loc, b.getI32IntegerAttr(0));
  Value init = emptyOf(b, loc, shape, b.getI32Type());
  return linalg::FillOp::create(b, loc, zero, init).getResult(0);
}

Value int8Matmul(OpBuilder &b, Location loc, Value a, Value w) {
  auto ta = tensorOf(a), tw = tensorOf(w);
  Value init = i32Zeros(b, loc, {ta.getDimSize(0), ta.getDimSize(1), tw.getDimSize(2)});
  return linalg::BatchMatmulOp::create(b, loc, TypeRange{init.getType()}, ValueRange{a, w}, ValueRange{init}).getResult(0);
}

/// The new rows [H, M, D] quantized over the cores when there are several tokens, with K's layout copy [H, D, M]. One token: on the first core
/// (a few KB, and a core loop costs more than it saves).
struct QuantizedRows { Value scale, rows, columns; };
QuantizedRows quantizeNewRows(OpBuilder &b, Location loc, Value rows, bool wantColumns, int64_t cores, Value fixed = Value()) {
  auto t = tensorOf(rows);
  const int64_t H = t.getDimSize(0), M = t.getDimSize(1), D = t.getDimSize(2);
  if (M == 1 || cores < 2 || H % cores) {
    Value s = fixed ? Value() : rowScale(b, loc, rows), q = fixed ? quantFixed(b, loc, rows, fixed) : quantRows(b, loc, rows, s);
    return {s, q, wantColumns ? transposed(b, loc, q, {0, 2, 1}) : Value()};
  }
  const int64_t part = H / cores;
  auto attrs = [&](ArrayRef<int64_t> v) {
    SmallVector<OpFoldResult> r;
    for (int64_t x : v) r.push_back(b.getIndexAttr(x));
    return r;
  };
  SmallVector<Value> outs = {emptyOf(b, loc, {H, M}, b.getF32Type()), emptyOf(b, loc, {H, M, D}, b.getI8Type())};
  if (fixed) outs.front() = outs[1], outs.pop_back();
  if (wantColumns) outs.push_back(emptyOf(b, loc, {H, D, M}, b.getI8Type()));
  auto loop = scf::ForallOp::create(b, loc, attrs({cores}), outs, std::nullopt);
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(loop.getBody());
  Value first = arith::MulIOp::create(b, loc, loop.getInductionVar(0), arith::ConstantIndexOp::create(b, loc, part));
  OpFoldResult zero = b.getIndexAttr(0);
  Value x = tensor::ExtractSliceOp::create(b, loc, rows, SmallVector<OpFoldResult>{first, zero, zero}, attrs({part, M, D}), attrs({1, 1, 1}));
  Value fixedSlice = fixed ? Value(tensor::ExtractSliceOp::create(b, loc, fixed, SmallVector<OpFoldResult>{first}, attrs({part}), attrs({1}))) : Value();
  Value s = fixed ? Value() : rowScale(b, loc, x), q = fixed ? quantFixed(b, loc, x, fixedSlice) : quantRows(b, loc, x, s);
  Value c = wantColumns ? transposed(b, loc, q, {0, 2, 1}) : Value();
  b.setInsertionPointToEnd(&loop.getTerminator().getRegion().front());
  unsigned at = 0;
  if (!fixed)
    tensor::ParallelInsertSliceOp::create(b, loc, s, loop.getRegionOutArgs()[at++], SmallVector<OpFoldResult>{first, zero}, attrs({part, M}), attrs({1, 1}));
  tensor::ParallelInsertSliceOp::create(b, loc, q, loop.getRegionOutArgs()[at++], SmallVector<OpFoldResult>{first, zero, zero}, attrs({part, M, D}), attrs({1, 1, 1}));
  if (wantColumns)
    tensor::ParallelInsertSliceOp::create(b, loc, c, loop.getRegionOutArgs()[at++], SmallVector<OpFoldResult>{first, zero, zero}, attrs({part, D, M}), attrs({1, 1, 1}));
  loop->setAttr("hyb.cores", b.getI64IntegerAttr(cores));
  loop->setAttr("hyb.elementwise_distributed", b.getUnitAttr());
  at = 0;
  Value rs = fixed ? Value() : loop.getResult(at++);
  Value rq = loop.getResult(at++);
  return {rs, rq, wantColumns ? loop.getResult(at) : Value()};
}

struct Layer {
  int index = 0;
  BlockArgument pastK, pastV;
  tensor::ConcatOp kc, vc;
  linalg::BatchMatmulOp qk, pv;
};

/// y = a + b, INT32 (the partial sums of a split attention).
Value addInt32(OpBuilder &b, Location loc, Value x, Value y) {
  auto t = tensorOf(x);
  AffineMap id = AffineMap::getMultiDimIdentityMap(t.getRank(), b.getContext());
  Value init = emptyOf(b, loc, t.getShape(), b.getI32Type());
  auto g = linalg::GenericOp::create(b, loc, TypeRange{init.getType()}, ValueRange{x, y}, ValueRange{init}, SmallVector<AffineMap>{id, id, id},
      SmallVector<utils::IteratorType>(t.getRank(), utils::IteratorType::parallel),
      [&](OpBuilder &bb, Location l, ValueRange a) {
        linalg::YieldOp::create(bb, l, ValueRange{arith::AddIOp::create(bb, l, a[0], a[1]).getResult()});
      });
  g->setAttr("hyb.kv_op", b.getStringAttr("add_i32"));
  return g.getResult(0);
}

/// mode=pim, one token. QK^T and PV of the past rows run on PIM: the K (V) matrix of a layer is a stored matrix with one KV head per PU, so a stream
/// of a subchannel works on all KV heads at once, each PU on its own query head's input. The own row goes through the SA in INT8, softmax stays on
/// the NPU. Phases of the attention:
///   NPU  Q quantized, own row's score, input for PIM     PIM  QK^T [G, H P]
///   NPU  scores gathered per core, softmax, P quantized, input for PIM     PIM  PV [G, H D]
///   NPU  results gathered per core, own row added, dequantized
LogicalResult lowerPimAttention(OpBuilder &b, Location loc, int layer, linalg::BatchMatmulOp qk, linalg::BatchMatmulOp pv, Value k8T, Value v8, Value skArg, Value skNew,
                                Value svArg, Value svNew, int64_t cores, bool headScale) {
  Value q = qk.getDpsInputs()[0], qkResult = qk.getResult(0), p = pv.getDpsInputs()[0];
  auto qT = tensorOf(q);
  const int64_t H = qT.getDimSize(0), G = qT.getDimSize(1), D = qT.getDimSize(2), N = tensorOf(qkResult).getDimSize(2), P = N - 1;
  if (H % cores || H != ::npu::pim::kPUs) return qk.emitError("--kv8=pim needs 8 KV heads and a core count that divides them");
  if (tensorOf(k8T).getDimSize(2) != 1) return qk.emitError("--kv8=pim works on one token");
  const int64_t part = H / cores;
  auto attrs = [&](ArrayRef<int64_t> v) {
    SmallVector<OpFoldResult> r;
    for (int64_t x : v) r.push_back(b.getIndexAttr(x));
    return r;
  };
  auto i32 = b.getI32Type();
  b.setInsertionPoint(qk);
  // Q and the own row's score on the first core, the input of the PIM phase in [G, H D].
  Value sq = rowScale(b, loc, q), q8 = quantRows(b, loc, q, sq);
  auto own = linalg::BatchMatmulOp::create(b, loc, TypeRange{RankedTensorType::get({H, G, int64_t(1)}, i32)}, ValueRange{q8, k8T},
                                           ValueRange{i32Zeros(b, loc, {H, G, 1})});
  own->setAttr("hyb.kv_own", b.getUnitAttr());
  Value qp = transposed(b, loc, q8, {1, 0, 2});
  Value qAct = tensor::CollapseShapeOp::create(b, loc, RankedTensorType::get({G, H * D}, b.getI8Type()), qp, SmallVector<ReassociationIndices>{{0}, {1, 2}});
  Value accInit = emptyOf(b, loc, {G, H * P}, i32);
  Value acc = hyb::QMatmulOp::create(b, loc, TypeRange{accInit.getType()}, qAct, accInit, b.getStringAttr("kv8.k." + std::to_string(layer)), IntegerAttr(), StringAttr()).getResult(0);
  // Scores to P, per core. The chain of generics between QK^T and PV is cloned on the core's heads.
  SmallVector<Operation *> chain;
  {
    DenseSet<Operation *> fwd;
    SmallVector<Value> work{qkResult};
    while (!work.empty()) {
      Value v = work.pop_back_val();
      for (Operation *u : v.getUsers())
        if (u != pv.getOperation() && fwd.insert(u).second)
          for (Value r : u->getResults()) work.push_back(r);
    }
    for (Operation &op : *qk->getBlock())
      if (fwd.contains(&op)) chain.push_back(&op);
    for (Operation *op : chain)
      if (!isa<linalg::GenericOp>(op)) return op->emitError("--kv8=pim: the softmax chain has an op that is not a generic");
  }
  OpFoldResult zero = b.getIndexAttr(0);
  b.setInsertionPoint(pv);    // after what the chain reads (the mask)
  // In the phase that reads them: a concat of LP6 buffers cannot cross a phase.
  auto f32Rows = RankedTensorType::get({H, N}, b.getF32Type());
  Value sk = headScale ? skArg : Value(tensor::ConcatOp::create(b, loc, f32Rows, 1, ValueRange{skArg, skNew}));
  Value sv = headScale ? svArg : Value(tensor::ConcatOp::create(b, loc, f32Rows, 1, ValueRange{svArg, svNew}));
  Value pOutInit = emptyOf(b, loc, {H, G, N}, b.getI8Type()), spOutInit = emptyOf(b, loc, {H, G}, b.getF32Type());
  auto loop = scf::ForallOp::create(b, loc, attrs({cores}), ValueRange{pOutInit, spOutInit}, std::nullopt);
  loop->setAttr("hyb.cores", b.getI64IntegerAttr(cores));
  loop->setAttr("hyb.attention_fused", b.getUnitAttr());
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(loop.getBody());
    Value first = arith::MulIOp::create(b, loc, loop.getInductionVar(0), arith::ConstantIndexOp::create(b, loc, part));
    Value accC = emptyOf(b, loc, {part, G, N}, i32);
    for (int64_t h = 0; h < part; ++h) {
      Value head = arith::AddIOp::create(b, loc, first, arith::ConstantIndexOp::create(b, loc, h));
      Value at = arith::MulIOp::create(b, loc, head, arith::ConstantIndexOp::create(b, loc, P));
      Value slice = tensor::ExtractSliceOp::create(b, loc, acc, SmallVector<OpFoldResult>{zero, at}, attrs({G, P}), attrs({1, 1}));
      accC = tensor::InsertSliceOp::create(b, loc, slice, accC, attrs({h, 0, 0}), attrs({1, G, P}), attrs({1, 1, 1}));
    }
    Value ownC = tensor::ExtractSliceOp::create(b, loc, own.getResult(0), SmallVector<OpFoldResult>{first, zero, zero}, attrs({part, G, 1}), attrs({1, 1, 1}));
    accC = tensor::InsertSliceOp::create(b, loc, ownC, accC, attrs({0, 0, P}), attrs({part, G, 1}), attrs({1, 1, 1}));
    Value sqC = tensor::ExtractSliceOp::create(b, loc, sq, SmallVector<OpFoldResult>{first, zero}, attrs({part, G}), attrs({1, 1}));
    Value skC = headScale ? Value(tensor::ExtractSliceOp::create(b, loc, sk, SmallVector<OpFoldResult>{first}, attrs({part}), attrs({1})))
                          : Value(tensor::ExtractSliceOp::create(b, loc, sk, SmallVector<OpFoldResult>{first, zero}, attrs({part, N}), attrs({1, 1})));
    Value scC = headScale ? dequantOf(b, loc, accC, sqC, Value(), "dequant", skC) : dequantOf(b, loc, accC, sqC, skC, "dequant");
    IRMapping map;
    map.map(qkResult, scC);
    // A fresh init on the core's heads: tensor.empty, its reshapes, and fills of them.
    std::function<Value(Value)> sliced = [&](Value v) -> Value {
      if (Value m = map.lookupOrNull(v)) return m;
      auto t = dyn_cast<RankedTensorType>(v.getType());
      Operation *def = v.getDefiningOp();
      if (!t || !t.getRank() || t.getDimSize(0) != H) return v;
      SmallVector<int64_t> shape(t.getShape());
      shape[0] = part;
      if (isa_and_nonnull<tensor::EmptyOp, tensor::CollapseShapeOp, tensor::ExpandShapeOp>(def)) return emptyOf(b, loc, shape, t.getElementType());
      if (auto fill = dyn_cast_or_null<linalg::FillOp>(def)) return linalg::FillOp::create(b, loc, fill.getInputs()[0], sliced(fill.getOutputs()[0])).getResult(0);
      SmallVector<OpFoldResult> offs(t.getRank(), zero), sizes = attrs(shape);
      offs[0] = first;
      return tensor::ExtractSliceOp::create(b, loc, v, offs, sizes, SmallVector<OpFoldResult>(t.getRank(), b.getIndexAttr(1)));
    };
    for (Operation *op : chain) {
      IRMapping m2 = map;
      for (Value o : op->getOperands()) m2.map(o, sliced(o));
      Operation *c = b.clone(*op, m2);
      for (auto [r, nr] : llvm::zip(op->getResults(), c->getResults())) {
        SmallVector<int64_t> shape(tensorOf(r).getShape());
        shape[0] = part;
        nr.setType(RankedTensorType::get(shape, tensorOf(r).getElementType()));
        map.map(r, nr);
      }
    }
    Value pC = map.lookup(p);
    Value ps = headScale ? pC : scaleColumns(b, loc, pC, tensor::ExtractSliceOp::create(b, loc, sv, SmallVector<OpFoldResult>{first, zero}, attrs({part, N}), attrs({1, 1})));
    Value spC = rowScale(b, loc, ps), p8C = quantRows(b, loc, ps, spC);
    b.setInsertionPointToEnd(&loop.getTerminator().getRegion().front());
    tensor::ParallelInsertSliceOp::create(b, loc, p8C, loop.getRegionOutArgs()[0], SmallVector<OpFoldResult>{first, zero, zero}, attrs({part, G, N}), attrs({1, 1, 1}));
    tensor::ParallelInsertSliceOp::create(b, loc, spC, loop.getRegionOutArgs()[1], SmallVector<OpFoldResult>{first, zero}, attrs({part, G}), attrs({1, 1}));
  }
  Value p8 = loop.getResult(0), sp = loop.getResult(1);
  // The past columns of P as the PIM input [G, H P]; the own column stays on the NPU.
  Value p8Past = tensor::ExtractSliceOp::create(b, loc, p8, attrs({0, 0, 0}), attrs({H, G, P}), attrs({1, 1, 1}));
  Value p8Own = tensor::ExtractSliceOp::create(b, loc, p8, attrs({0, 0, P}), attrs({H, G, 1}), attrs({1, 1, 1}));
  Value ppT = transposed(b, loc, p8Past, {1, 0, 2});
  Value pAct = tensor::CollapseShapeOp::create(b, loc, RankedTensorType::get({G, H * P}, b.getI8Type()), ppT, SmallVector<ReassociationIndices>{{0}, {1, 2}});
  Value acc2Init = emptyOf(b, loc, {G, H * D}, i32);
  Value acc2 = hyb::QMatmulOp::create(b, loc, TypeRange{acc2Init.getType()}, pAct, acc2Init, b.getStringAttr("kv8.v." + std::to_string(layer)), IntegerAttr(), StringAttr()).getResult(0);
  auto own2 = linalg::BatchMatmulOp::create(b, loc, TypeRange{RankedTensorType::get({H, G, D}, i32)}, ValueRange{p8Own, v8}, ValueRange{i32Zeros(b, loc, {H, G, D})});
  own2->setAttr("hyb.kv_own", b.getUnitAttr());
  Value oInit = emptyOf(b, loc, {H, G, D}, b.getF16Type());
  auto loop2 = scf::ForallOp::create(b, loc, attrs({cores}), ValueRange{oInit}, std::nullopt);
  loop2->setAttr("hyb.cores", b.getI64IntegerAttr(cores));
  loop2->setAttr("hyb.attention_fused", b.getUnitAttr());
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(loop2.getBody());
    Value first = arith::MulIOp::create(b, loc, loop2.getInductionVar(0), arith::ConstantIndexOp::create(b, loc, part));
    Value accC = emptyOf(b, loc, {part, G, D}, i32);
    for (int64_t h = 0; h < part; ++h) {
      Value head = arith::AddIOp::create(b, loc, first, arith::ConstantIndexOp::create(b, loc, h));
      Value at = arith::MulIOp::create(b, loc, head, arith::ConstantIndexOp::create(b, loc, D));
      Value slice = tensor::ExtractSliceOp::create(b, loc, acc2, SmallVector<OpFoldResult>{zero, at}, attrs({G, D}), attrs({1, 1}));
      accC = tensor::InsertSliceOp::create(b, loc, slice, accC, attrs({h, 0, 0}), attrs({1, G, D}), attrs({1, 1, 1}));
    }
    Value ownC = tensor::ExtractSliceOp::create(b, loc, own2.getResult(0), SmallVector<OpFoldResult>{first, zero, zero}, attrs({part, G, D}), attrs({1, 1, 1}));
    Value sum = addInt32(b, loc, accC, ownC);
    Value spC = tensor::ExtractSliceOp::create(b, loc, sp, SmallVector<OpFoldResult>{first, zero}, attrs({part, G}), attrs({1, 1}));
    Value svH = headScale ? Value(tensor::ExtractSliceOp::create(b, loc, sv, SmallVector<OpFoldResult>{first}, attrs({part}), attrs({1}))) : Value();
    Value oC = dequantOf(b, loc, sum, spC, Value(), "dequant", svH);
    b.setInsertionPointToEnd(&loop2.getTerminator().getRegion().front());
    tensor::ParallelInsertSliceOp::create(b, loc, oC, loop2.getRegionOutArgs()[0], SmallVector<OpFoldResult>{first, zero, zero}, attrs({part, G, D}), attrs({1, 1, 1}));
  }
  pv.getResult(0).replaceAllUsesWith(loop2.getResult(0));
  pv.erase();
  for (Operation *op : llvm::reverse(chain)) op->erase();
  qk.erase();
  return success();
}

struct HybKv8Pass : mlir::hybrid::impl::HybKv8Base<HybKv8Pass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    func::FuncOp step = getStepFunction(module);
    if (mode != "dequant" && mode != "int8" && mode != "pim") {
      module.emitError() << "unknown --kv8 mode '" << mode << "'";
      return signalPassFailure();
    }
    auto target = loadTargetSpec(module, spec);
    if (failed(target)) return signalPassFailure();
    const int64_t cores = target->npu.cores;
    const bool headScale = scale == "head";
    if (!headScale && scale != "row") {
      module.emitError() << "--kv8-scale must be row or head";
      return signalPassFailure();
    }
    std::map<int, Layer> layers;
    Location loc = step.getLoc();
    auto parse = [&](StringRef name, StringRef prefix) -> int {
      int n = -1;
      if (name.starts_with(prefix) && !name.drop_front(prefix.size()).getAsInteger(10, n)) return n;
      return -1;
    };
    for (BlockArgument a : step.getArguments()) {
      auto name = step.getArgAttrOfType<StringAttr>(a.getArgNumber(), kName);
      if (!name) continue;
      if (int i = parse(name.getValue(), "past_k."); i >= 0) { layers[i].index = i; layers[i].pastK = a; }
      if (int i = parse(name.getValue(), "past_v."); i >= 0) { layers[i].index = i; layers[i].pastV = a; }
    }
    if (layers.empty()) {
      module.emitError("--kv8 needs a step with a K/V cache");
      return signalPassFailure();
    }
    Block &body = step.getBody().front();
    auto ret = cast<func::ReturnOp>(body.getTerminator());
    OpBuilder b(&getContext());
    Type f32 = b.getF32Type(), i8 = b.getI8Type();
    // Find the chains first: retyping an argument makes its old uses invalid.
    for (auto &[i, l] : layers) {
      auto single = [&](Value v, auto &out) {
        for (Operation *u : v.getUsers())
          if (auto c = dyn_cast<std::remove_reference_t<decltype(out)>>(u)) out = c;
        return bool(out);
      };
      if (!l.pastK || !l.pastV || !single(l.pastK, l.kc) || !single(l.pastV, l.vc) ||
          !single(l.kc.getResult(), l.qk) || !single(l.vc.getResult(), l.pv)) {
        module.emitError() << "--kv8: layer " << i << " has no cache concat and matmuls of the expected shape";
        return signalPassFailure();
      }
    }
    // Scale arguments after the K/V arguments.
    unsigned insertAt = 0;
    for (BlockArgument a : step.getArguments()) {
      auto name = step.getArgAttrOfType<StringAttr>(a.getArgNumber(), kName);
      if (name && (name.getValue().starts_with("past_k.") || name.getValue().starts_with("past_v.")))
        insertAt = a.getArgNumber() + 1;
    }
    std::map<int, std::pair<Value, Value>> scales;
    SmallVector<std::pair<std::string, Value>> newOutputs;
    llvm::BitVector erase(step.getNumArguments());
    for (auto &[i, l] : layers) {
      auto kT = tensorOf(l.pastK), vT = tensorOf(l.pastV);
      const int64_t H = kT.getDimSize(0), P = kT.getDimSize(2);
      auto sT = headScale ? RankedTensorType::get({H}, f32) : RankedTensorType::get({H, P}, f32);
      auto attrs = [&](const std::string &name) {
        return b.getDictionaryAttr({b.getNamedAttr(kKind, b.getStringAttr("state")), b.getNamedAttr(kName, b.getStringAttr(name)),
                                    b.getNamedAttr("hyb.layer", b.getI64IntegerAttr(i)), b.getNamedAttr("hyb.dtype", b.getStringAttr("f32"))});
      };
      if (failed(step.insertArgument(insertAt, sT, attrs("past_ks." + std::to_string(i)), loc))) return signalPassFailure();
      Value sk = body.getArgument(insertAt++);
      if (failed(step.insertArgument(insertAt, sT, attrs("past_vs." + std::to_string(i)), loc))) return signalPassFailure();
      Value sv = body.getArgument(insertAt++);
      scales[i] = {sk, sv};
      (void)vT;
    }
    // The step's outputs by name.
    auto resultNamed = [&](const std::string &name) -> int {
      for (unsigned r = 0; r < step.getNumResults(); ++r)
        if (auto n = step.getResultAttrOfType<StringAttr>(r, kName); n && n.getValue() == name) return int(r);
      return -1;
    };
    for (auto &[i, l] : layers) {
      auto [skArg, svArg] = scales[i];
      b.setInsertionPoint(l.kc);
      Value kNew = l.kc.getInputs()[1], vNew = l.vc.getInputs()[1];     // [H, D, M], [H, M, D]
      if (l.vc->isBeforeInBlock(l.kc)) b.setInsertionPoint(l.kc);
      Value kRows = transposed(b, loc, kNew, {0, 2, 1});
      auto [skNew, k8, k8T] = quantizeNewRows(b, loc, kRows, true, cores, headScale ? skArg : Value());
      auto kcT = tensorOf(l.kc.getResult());
      Value kc8 = mode == "pim" ? Value() : Value(tensor::ConcatOp::create(b, loc, RankedTensorType::get(kcT.getShape(), i8), 2, ValueRange{l.pastK, k8T}));
      Value sk = mode == "pim" || headScale ? Value() : Value(tensor::ConcatOp::create(b, loc, RankedTensorType::get({kcT.getDimSize(0), kcT.getDimSize(2)}, f32), 1, ValueRange{skArg, skNew}));
      // V may be defined after the K concat in the block: keep the order of the original concat.
      if (l.kc->isBeforeInBlock(l.vc)) b.setInsertionPointAfter(l.vc.getOperation());
      auto [svNew, v8, unused] = quantizeNewRows(b, loc, vNew, false, cores, headScale ? svArg : Value());
      auto vcT = tensorOf(l.vc.getResult());
      Value vc8 = mode == "pim" ? Value() : Value(tensor::ConcatOp::create(b, loc, RankedTensorType::get(vcT.getShape(), i8), 1, ValueRange{l.pastV, v8}));
      Value sv = mode == "pim" || headScale ? Value() : Value(tensor::ConcatOp::create(b, loc, RankedTensorType::get({vcT.getDimSize(0), vcT.getDimSize(1)}, f32), 1, ValueRange{svArg, svNew}));
      // The chains.
      b.setInsertionPoint(l.qk);
      Value q = l.qk.getDpsInputs()[0];
      if (mode == "pim") {
        if (failed(lowerPimAttention(b, loc, i, l.qk, l.pv, k8T, v8, skArg, skNew, svArg, svNew, cores, headScale))) return signalPassFailure();
      } else if (mode == "int8") {
        Value sq = rowScale(b, loc, q);
        sq.getDefiningOp()->setAttr("hyb.kv_q", b.getUnitAttr());
        Value q8 = quantRows(b, loc, q, sq);
        q8.getDefiningOp()->setAttr("hyb.kv_q", b.getUnitAttr());
        Value acc = int8Matmul(b, loc, q8, kc8);
        Value sc = headScale ? dequantOf(b, loc, acc, sq, Value(), "dequant", skArg) : dequantOf(b, loc, acc, sq, sk, "dequant");
        l.qk.getResult(0).replaceAllUsesWith(sc);
        l.qk.erase();
        b.setInsertionPoint(l.pv);
        Value p = l.pv.getDpsInputs()[0];
        Value ps = headScale ? p : scaleColumns(b, loc, p, sv);
        Value sp = rowScale(b, loc, ps), p8 = quantRows(b, loc, ps, sp);
        Value acc2 = int8Matmul(b, loc, p8, vc8);
        Value o = dequantOf(b, loc, acc2, sp, Value(), "dequant", headScale ? svArg : Value());
        o.getDefiningOp()->setAttr("hyb.kv_attention", b.getUnitAttr());
        l.pv.getResult(0).replaceAllUsesWith(o);
        l.pv.erase();
      } else {
        Value kf = headScale ? dequantOf(b, loc, kc8, Value(), Value(), "dequant_i8", skArg) : dequantOf(b, loc, kc8, Value(), sk, "dequant_i8");
        l.qk->setOperand(1, kf);
        l.qk->setAttr("hyb.kv_q", b.getUnitAttr());
        b.setInsertionPoint(l.pv);
        // [H, P, D]: the scale of a cached row sits on the row axis.
        Value vf = headScale ? dequantOf(b, loc, vc8, Value(), Value(), "dequant_i8", svArg) : dequantOf(b, loc, vc8, sv, Value(), "dequant_i8");
        l.pv->setOperand(1, vf);
        l.pv->setAttr("hyb.kv_attention", b.getUnitAttr());
      }
      l.kc.erase();
      l.vc.erase();
      l.pastK.setType(RankedTensorType::get(tensorOf(l.pastK).getShape(), i8));
      l.pastV.setType(RankedTensorType::get(tensorOf(l.pastV).getShape(), i8));
      if (mode == "pim") {
        erase.resize(step.getNumArguments());
        erase.set(l.pastK.getArgNumber());
        erase.set(l.pastV.getArgNumber());
      }
      // Outputs. A drafter returns the rows of its context only: the first rows of the new ones.
      b.setInsertionPoint(ret);
      auto output = [&](const char *prefix, const char *scalePrefix, Value rows, Value s) {
        const std::string name = std::string(prefix) + std::to_string(i);
        int r = resultNamed(name);
        if (r < 0) return;
        auto wanted = tensorOf(ret.getOperand(r));
        auto have = tensorOf(rows);
        if (wanted.getDimSize(1) != have.getDimSize(1)) {
          SmallVector<OpFoldResult> zero3(3, b.getIndexAttr(0)), one3(3, b.getIndexAttr(1));
          SmallVector<OpFoldResult> size3 = {b.getIndexAttr(have.getDimSize(0)), b.getIndexAttr(wanted.getDimSize(1)), b.getIndexAttr(have.getDimSize(2))};
          rows = tensor::ExtractSliceOp::create(b, loc, rows, zero3, size3, one3);
          SmallVector<OpFoldResult> zero2(2, b.getIndexAttr(0)), one2(2, b.getIndexAttr(1));
          SmallVector<OpFoldResult> size2 = {b.getIndexAttr(have.getDimSize(0)), b.getIndexAttr(wanted.getDimSize(1))};
          if (s) s = tensor::ExtractSliceOp::create(b, loc, s, zero2, size2, one2);
        }
        ret->setOperand(r, rows);
        if (s) newOutputs.push_back({std::string(scalePrefix) + std::to_string(i), s});
      };
      output("new_k.", "new_ks.", k8, skNew);
      output("new_v.", "new_vs.", v8, svNew);
    }
    if (mode == "pim") {
      erase.resize(step.getNumArguments());
      if (failed(step.eraseArguments(erase))) return signalPassFailure();
    }
    // New results follow the old ones.
    unsigned nr = step.getNumResults();
    SmallVector<Type> resultTypes(ret.getOperandTypes());
    SmallVector<Value> operands(ret.getOperands());
    SmallVector<DictionaryAttr> attrs;
    for (unsigned r = 0; r < nr; ++r) attrs.push_back(step.getResultAttrDict(r));
    for (auto &[name, v] : newOutputs) {
      operands.push_back(v);
      resultTypes.push_back(v.getType());
      attrs.push_back(b.getDictionaryAttr({b.getNamedAttr(kKind, b.getStringAttr("output")), b.getNamedAttr(kName, b.getStringAttr(name))}));
    }
    ret->setOperands(operands);
    SmallVector<Type> argTypes(body.getArgumentTypes());
    step.setType(FunctionType::get(&getContext(), argTypes, resultTypes));
    step.setAllResultAttrs(attrs);
    module->setAttr("hyb.kv8", b.getStringAttr(mode));
    module->setAttr("hyb.kv8_scale", b.getStringAttr(scale));
  }
};
} // namespace
