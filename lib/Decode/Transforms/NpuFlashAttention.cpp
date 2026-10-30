//===- NpuFlashAttention.cpp - online attention and split-K decode ---------===//
//
// Replace a supported, private QK -> softmax -> PV graph with a tiled online
// recurrence. Scores/probabilities have query_tile x key_tile lifetime; the
// running maximum, denominator and unnormalised output are FP32. The existing
// FP16 SA consumes a rounded FP16 probability tile and writes an FP32 product.
//
// Flash-Decoding additionally partitions the key sequence across real cores.
// Each partition exports (m, l, O) to shared SRAM, then a second core region
// performs a stable merge. All transfers and the merge use ordinary lowering.
// No fictitious attention instruction, free communication or extra SRAM.
//===----------------------------------------------------------------------===//

#include "npu/Decode/Support/TargetSpec.h"
#include "npu/Decode/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SetVector.h"
#include <limits>

namespace mlir::hybrid {
#define GEN_PASS_DEF_HYBNPUFLASHATTENTION
#include "npu/Decode/Transforms/Passes.h.inc"
} // namespace mlir::hybrid

using namespace mlir;
using namespace mlir::hybrid;

namespace {

Value withoutCasts(Value v) {
  while (Operation *op = v.getDefiningOp()) {
    if (!isa<arith::ExtFOp, arith::TruncFOp>(op)) break;
    v = op->getOperand(0);
  }
  return v;
}

Value yielded(linalg::GenericOp g) {
  return withoutCasts(cast<linalg::YieldOp>(g.getRegion().front().getTerminator()).getValues()[0]);
}

bool hasExpAncestor(Value value, unsigned depth = 32) {
  if (!depth) return false;
  Operation *op = value.getDefiningOp();
  if (!op) return false;
  if (auto g = dyn_cast<linalg::GenericOp>(op)) {
    bool exp = false;
    g.walk([&](math::ExpOp) { exp = true; });
    if (exp) return true;
    for (Value input : g.getDpsInputs())
      if (hasExpAncestor(input, depth - 1)) return true;
  } else if (isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(op)) {
    return hasExpAncestor(op->getOperand(0), depth - 1);
  }
  return false;
}

bool hasMatmulAncestor(Value value, unsigned depth = 32) {
  if (!depth) return false;
  Operation *op = value.getDefiningOp();
  if (!op) return false;
  if (isa<linalg::BatchMatmulOp>(op)) return true;
  if (auto g = dyn_cast<linalg::GenericOp>(op)) {
    for (Value input : g.getDpsInputs())
      if (hasMatmulAncestor(input, depth - 1)) return true;
  } else if (isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(op)) {
    return hasMatmulAncestor(op->getOperand(0), depth - 1);
  }
  return false;
}

bool staticF16(Value value) {
  auto t = dyn_cast<RankedTensorType>(value.getType());
  return t && t.hasStaticShape() && t.getElementType().isF16();
}

std::optional<APFloat> fillConstant(Value value) {
  auto fill = value.getDefiningOp<linalg::FillOp>();
  auto constant = fill ? fill.getInputs()[0].getDefiningOp<arith::ConstantOp>() : nullptr;
  auto number = constant ? dyn_cast<FloatAttr>(constant.getValue()) : FloatAttr();
  if (!number) return std::nullopt;
  return number.getValue();
}

bool zeroFilled(Value value) {
  auto number = fillConstant(value);
  return number && number->isZero();
}

/// Unit-dimension folding can remove H=1 around softmax without removing the
/// batch dimension of QK/PV. Accept only GQA row reshaping and such unit-head
/// views; a reshape that merges two non-unit heads is not equivalent tiling.
bool supportedScoreView(Operation *op) {
  SmallVector<ReassociationIndices> reassociation;
  RankedTensorType high, low;
  if (auto expand = dyn_cast<tensor::ExpandShapeOp>(op)) {
    reassociation = expand.getReassociationIndices();
    high = expand.getResultType();
    low = expand.getSrcType();
  } else if (auto collapse = dyn_cast<tensor::CollapseShapeOp>(op)) {
    reassociation = collapse.getReassociationIndices();
    high = collapse.getSrcType();
    low = collapse.getResultType();
  } else return false;
  if (!high.hasStaticShape() || !low.hasStaticShape()) return false;
  if (high.getRank() == 4 && low.getRank() == 3 &&
      reassociation == SmallVector<ReassociationIndices>{{0}, {1, 2}, {3}}) return true;
  if (high.getDimSize(0) != 1) return false;
  if (high.getRank() == 3 && low.getRank() == 2)
    return reassociation == SmallVector<ReassociationIndices>{{0, 1}, {2}};
  if (high.getRank() == 4 && low.getRank() == 3)
    return reassociation == SmallVector<ReassociationIndices>{{0, 1}, {2}, {3}};
  if (high.getRank() == 4 && low.getRank() == 2)
    return (high.getDimSize(1) == 1 || high.getDimSize(2) == 1) &&
           reassociation == SmallVector<ReassociationIndices>{{0, 1, 2}, {3}};
  return false;
}

/// The matcher checks the actual scalar/dataflow contract, not merely the
/// presence of one exp and one reduction somewhere in an arbitrary graph.
struct Attention {
  linalg::BatchMatmulOp pv, qk;
  Value score;
  llvm::SetVector<Operation *> chain;
  DenseMap<Operation *, unsigned> scoreInput;
  int64_t heads = 0, groups = 1, tokens = 0, rows = 0, keys = 0, depth = 0, valueDepth = 0;
  bool transposedKey = false;
  std::string error;

  bool reject(StringRef reason) { error = reason.str(); return false; }

  bool maps(linalg::GenericOp g, unsigned rank, bool reduction, unsigned inputs) {
    if (!g || !g.hasPureTensorSemantics() || g.getNumDpsInputs() != inputs || g.getNumDpsInits() != 1 ||
        g.getNumLoops() != rank) return false;
    for (Operation &scalar : g.getRegion().front()) {
      StringRef dialect = scalar.getName().getDialectNamespace();
      if (!isa<linalg::YieldOp>(scalar) && dialect != "arith" && dialect != "math") return false;
    }
    auto iterators = g.getIteratorTypesArray();
    for (unsigned i = 0; i < rank; ++i)
      if (iterators[i] != ((reduction && i + 1 == rank) ? utils::IteratorType::reduction
                                                                   : utils::IteratorType::parallel)) return false;
    auto id = AffineMap::getMultiDimIdentityMap(rank, g.getContext());
    SmallVector<AffineExpr> dims;
    for (unsigned i = 0; i + 1 < rank; ++i) dims.push_back(getAffineDimExpr(i, g.getContext()));
    auto row = AffineMap::get(rank, 0, dims, g.getContext());
    auto indexing = g.getIndexingMapsArray();
    if (indexing[0] != id || indexing.back() != (reduction ? row : id)) return false;
    return inputs != 2 || indexing[1] == row;
  }

  bool reduction(linalg::GenericOp g, Value input, unsigned rank, bool maximum) {
    if (!maps(g, rank, true, 1) || g.getDpsInputs()[0] != input) return false;
    auto initial = fillConstant(g.getDpsInits()[0]);
    if (!initial || (maximum ? !(initial->isInfinity() && initial->isNegative()) : !initial->isZero()))
      return false;
    auto body = &g.getRegion().front();
    Operation *op = yielded(g).getDefiningOp();
    if (!op || (maximum ? !isa<arith::MaximumFOp>(op) : !isa<arith::AddFOp>(op))) return false;
    Value a = withoutCasts(op->getOperand(0)), b = withoutCasts(op->getOperand(1));
    return (a == body->getArgument(0) && b == body->getArgument(1)) ||
           (b == body->getArgument(0) && a == body->getArgument(1));
  }

  bool visitScore(Value value, unsigned depthLimit = 32) {
    if (!depthLimit || !staticF16(value)) return reject("score preprocessing must have static FP16 storage");
    Operation *op = value.getDefiningOp();
    if (!op || op->getBlock() != pv->getBlock()) return reject("score preprocessing leaves the attention block");
    chain.insert(op);
    if (auto mm = dyn_cast<linalg::BatchMatmulOp>(op)) {
      if (qk && qk != mm) return reject("more than one QK product");
      qk = mm;
      return true;
    }
    if (auto expand = dyn_cast<tensor::ExpandShapeOp>(op)) {
      if (!supportedScoreView(op)) return reject("unsupported score expand_shape");
      return visitScore(expand.getSrc(), depthLimit - 1);
    }
    if (auto collapse = dyn_cast<tensor::CollapseShapeOp>(op)) {
      if (!supportedScoreView(op)) return reject("unsupported score collapse_shape");
      return visitScore(collapse.getSrc(), depthLimit - 1);
    }
    auto generic = dyn_cast<linalg::GenericOp>(op);
    if (!generic || generic.getNumDpsInits() != 1 || generic.getNumReductionLoops() ||
        !generic.getIndexingMapsArray().back().isIdentity()) return reject("unsupported score preprocessing");
    for (Operation &scalar : generic.getRegion().front()) {
      StringRef dialect = scalar.getName().getDialectNamespace();
      if (!isa<linalg::YieldOp>(scalar) && dialect != "arith" && dialect != "math")
        return reject("index-dependent or effectful score preprocessing");
    }
    if (!generic.getRegion().front().getArguments().back().use_empty())
      return reject("score preprocessing reads its destination");
    unsigned found = 0, inputIndex = 0;
    for (auto [i, input] : llvm::enumerate(generic.getDpsInputs())) {
      if (hasMatmulAncestor(input)) { ++found; inputIndex = i; }
      auto type = dyn_cast<RankedTensorType>(input.getType());
      if (!type || !type.hasStaticShape()) return reject("dynamic score preprocessing input");
      for (AffineExpr expr : generic.getIndexingMapsArray()[i].getResults()) {
        if (isa<AffineDimExpr>(expr)) continue;
        auto constant = dyn_cast<AffineConstantExpr>(expr);
        // The selected operand is sliced to local coordinates, while the
        // cloned scalar generic retains its map. Only zero broadcasts remain
        // valid without remapping a nonzero constant into that local slice.
        if (!constant || constant.getValue() != 0)
          return reject("score preprocessing needs projected indexing maps with zero broadcasts");
      }
    }
    if (found != 1) return reject("score preprocessing must have exactly one QK dependency");
    if (!generic.getIndexingMapsArray()[inputIndex].isIdentity())
      return reject("score preprocessing must preserve the score indexing map");
    scoreInput[op] = inputIndex;
    return visitScore(generic.getDpsInputs()[inputIndex], depthLimit - 1);
  }

  bool match(linalg::BatchMatmulOp candidate) {
    pv = candidate;
    Value p = pv.getDpsInputs()[0];
    while (Operation *view = p.getDefiningOp()) {
      if (!isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp>(view)) break;
      if (!supportedScoreView(view)) return reject("unsupported probability reshape");
      chain.insert(view);
      p = view->getOperand(0);
    }
    auto probability = p.getDefiningOp<linalg::GenericOp>();
    if (!probability) return reject("probability is not an explicit softmax");
    auto shape = dyn_cast<RankedTensorType>(p.getType());
    if (!shape || !shape.hasStaticShape() || shape.getRank() < 2 || shape.getRank() > 4)
      return reject("attention needs static rank-2/3 or GQA rank-4 scores");
    unsigned rank = shape.getRank();
    if (!maps(probability, rank, false, 2)) return reject("unsupported softmax normalization maps");
    auto div = yielded(probability).getDefiningOp<arith::DivFOp>();
    auto &pbody = probability.getRegion().front();
    if (!div || withoutCasts(div.getLhs()) != pbody.getArgument(0) ||
        withoutCasts(div.getRhs()) != pbody.getArgument(1)) return reject("unsupported softmax normalization");
    Value exponentials = probability.getDpsInputs()[0];
    auto exp = exponentials.getDefiningOp<linalg::GenericOp>();
    if (!maps(exp, rank, false, 2)) return reject("unsupported softmax exponent maps");
    auto e = yielded(exp).getDefiningOp<math::ExpOp>();
    auto sub = e ? withoutCasts(e.getOperand()).getDefiningOp<arith::SubFOp>() : nullptr;
    if (!sub || withoutCasts(sub.getLhs()) != exp.getRegion().front().getArgument(0) ||
        withoutCasts(sub.getRhs()) != exp.getRegion().front().getArgument(1))
      return reject("unsupported softmax exponent");
    score = exp.getDpsInputs()[0];
    auto maximum = exp.getDpsInputs()[1].getDefiningOp<linalg::GenericOp>();
    auto sum = probability.getDpsInputs()[1].getDefiningOp<linalg::GenericOp>();
    if (!reduction(maximum, score, rank, true) || !reduction(sum, exponentials, rank, false))
      return reject("unsupported softmax max/sum reduction");
    for (Operation *op : {probability.getOperation(), exp.getOperation(), maximum.getOperation(), sum.getOperation()})
      chain.insert(op);
    if (!visitScore(score) || !qk) return false;
    if (!zeroFilled(qk.getDpsInits()[0]) || !zeroFilled(pv.getDpsInits()[0]))
      return reject("QK/PV must have zero-filled destinations");
    auto qtype = dyn_cast<RankedTensorType>(qk.getDpsInputs()[0].getType());
    auto ktype = dyn_cast<RankedTensorType>(qk.getDpsInputs()[1].getType());
    auto vtype = dyn_cast<RankedTensorType>(pv.getDpsInputs()[1].getType());
    auto otype = dyn_cast<RankedTensorType>(pv.getResult(0).getType());
    if (!qtype || !ktype || !vtype || !otype || !staticF16(qk.getDpsInputs()[0]) ||
        !staticF16(qk.getDpsInputs()[1]) || !staticF16(pv.getDpsInputs()[1]) || !staticF16(pv.getResult(0)) ||
        qtype.getRank() != 3 || ktype.getRank() != 3 || vtype.getRank() != 3 || otype.getRank() != 3)
      return reject("attention operands/output must be static rank-3 FP16");
    heads = qtype.getDimSize(0);
    rows = qtype.getDimSize(1);
    keys = vtype.getDimSize(1);
    groups = 1;
    tokens = rows;
    // A frontend may collapse the GQA view again before softmax. Retain the
    // token/group interpretation of any rank-4 score preprocessing, otherwise
    // slicing a token-specific mask as if it had G*M rows would be invalid.
    bool hasGroupView = false;
    for (Operation *op : chain) {
      if (op->getNumResults() != 1) continue;
      auto viewType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
      if (!viewType || viewType.getRank() != 4 || !viewType.getElementType().isF16()) continue;
      if (viewType.getDimSize(0) != heads || viewType.getDimSize(3) != keys ||
          viewType.getDimSize(1) * viewType.getDimSize(2) != rows)
        return reject("incompatible GQA preprocessing views");
      if (hasGroupView && (groups != viewType.getDimSize(1) || tokens != viewType.getDimSize(2)))
        return reject("changing the GQA group/token interpretation is unsupported");
      groups = viewType.getDimSize(1);
      tokens = viewType.getDimSize(2);
      hasGroupView = true;
    }
    for (Operation *op : chain) {
      if (op->getNumResults() != 1) continue;
      auto viewType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
      if (!viewType || !viewType.getElementType().isF16()) continue;
      auto dims = viewType.getShape();
      bool supported = dims == ArrayRef<int64_t>({heads, rows, keys}) ||
                       dims == ArrayRef<int64_t>({heads, groups, tokens, keys});
      if (heads == 1)
        supported |= dims == ArrayRef<int64_t>({rows, keys}) ||
                     dims == ArrayRef<int64_t>({groups, tokens, keys});
      if (!supported) return reject("score views do not preserve head/query/key dimensions");
    }
    depth = qtype.getDimSize(2);
    valueDepth = vtype.getDimSize(2);
    if (heads <= 0 || rows <= 0 || keys <= 0 || depth <= 0 || valueDepth <= 0)
      return reject("empty attention dimensions are unsupported");
    auto *ctx = pv.getContext();
    AffineExpr h, m, n, k;
    bindDims(ctx, h, m, n, k);
    auto qmap = AffineMap::get(4, 0, {h, m, k}, ctx);
    auto outmap = AffineMap::get(4, 0, {h, m, n}, ctx);
    auto keymap = AffineMap::get(4, 0, {h, k, n}, ctx);
    auto transmap = AffineMap::get(4, 0, {h, n, k}, ctx);
    auto qm = qk.getIndexingMapsArray(), pm = pv.getIndexingMapsArray();
    if (qm[0] != qmap || qm[2] != outmap || (qm[1] != keymap && qm[1] != transmap) ||
        pm[0] != qmap || pm[1] != keymap || pm[2] != outmap)
      return reject("unsupported QK/PV indexing maps");
    transposedKey = qm[1] == transmap;
    if (qtype.getShape() != ArrayRef<int64_t>({heads, rows, depth}) ||
        vtype.getShape() != ArrayRef<int64_t>({heads, keys, valueDepth}) ||
        otype.getShape() != ArrayRef<int64_t>({heads, rows, valueDepth}) ||
        ktype.getShape() != (transposedKey ? ArrayRef<int64_t>({heads, keys, depth})
                                         : ArrayRef<int64_t>({heads, depth, keys})))
      return reject("Q/K/V dimensions do not agree with the GQA score reshape");
    for (Operation *op : chain)
      for (Operation *user : op->getUsers())
        if (user != pv && !chain.contains(user)) return reject("attention intermediate has an external consumer");
    return true;
  }
};

struct State { Value maximum, sum, output; };

struct KVSegment {
  Value key, value;
  int64_t length, maskOffset;
};

class OnlineEmitter {
public:
  OnlineEmitter(IRRewriter &rewriter, Attention &attention, const TargetSpec &target)
      : b(rewriter), a(attention), t(target), loc(a.pv.getLoc()), ctx(b.getContext()) {}

  Value emit() {
    splits = t.npu.flashDecoding ? t.npu.attentionSplits : 1;
    segments.push_back({a.qk.getDpsInputs()[1], a.pv.getDpsInputs()[1], a.keys, 0});
    if (t.npu.flashKvSlicing) findKVSegments();
    headCores = std::min(a.heads, t.npu.cores / splits);
    while (headCores > 1 && a.heads % headCores) --headCores;
    headsPerCore = a.heads / headCores;
    int64_t workers = headCores * splits;
    Value result = empty({a.heads, a.rows, a.valueDepth}, b.getF16Type());
    if (splits == 1) {
      auto parallel = forall(headCores, {result}, false);
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPoint(parallel.getBody()->getTerminator());
        Value worker = parallel.getInductionVar(0);
        forTiles(worker, [&](Value head, int64_t, int64_t group, int64_t query, int64_t count) {
          State state = initial(count);
          for (const KVSegment &segment : segments)
            state = run(segment, head, group, query, count, index(0), segment.length, state);
          publish(parallel, finish(state), parallel.getRegionOutArgs()[0],
                  {head, index(group * a.tokens + query), index(0)}, {1, count, a.valueDepth});
        });
      }
      return finishForall(parallel).front();
    }

    // Equal nonempty ranges permit one statically shaped core body. Fewer than
    // `splits` residual keys are evaluated after the partial states are merged.
    const KVSegment &prefix = segments.front();
    int64_t partition = prefix.length / splits;
    Value partialMax = empty({workers, headsPerCore, a.rows}, b.getF32Type());
    Value partialSum = empty({workers, headsPerCore, a.rows}, b.getF32Type());
    Value partialOutput = empty({workers, headsPerCore, a.rows, a.valueDepth}, b.getF32Type());
    auto partial = forall(workers, {partialMax, partialSum, partialOutput}, true);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPoint(partial.getBody()->getTerminator());
      Value worker = partial.getInductionVar(0);
      Value headWorker = arith::DivUIOp::create(b, loc, worker, index(splits));
      Value split = arith::RemUIOp::create(b, loc, worker, index(splits));
      Value start = multiply(split, partition);
      forTiles(headWorker, [&](Value head, int64_t local, int64_t group, int64_t query, int64_t count) {
        State state = run(prefix, head, group, query, count, start, partition, initial(count));
        auto offset = SmallVector<Value>{worker, index(local), index(group * a.tokens + query)};
        publish(partial, state.maximum, partial.getRegionOutArgs()[0], offset, {1, 1, count});
        publish(partial, state.sum, partial.getRegionOutArgs()[1], offset, {1, 1, count});
        offset.push_back(index(0));
        publish(partial, state.output, partial.getRegionOutArgs()[2], offset, {1, 1, count, a.valueDepth});
      });
    }
    b.setInsertionPointAfter(partial);
    auto merge = forall(headCores, {result}, true);
    merge->setAttr("hyb.flash_merge", b.getUnitAttr());
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPoint(merge.getBody()->getTerminator());
      Value worker = merge.getInductionVar(0);
      forTiles(worker, [&](Value head, int64_t local, int64_t group, int64_t query, int64_t count) {
        State state = initial(count);
        for (int64_t s = 0; s < splits; ++s) {
          Value slot = add(multiply(worker, splits), s);
          auto offset = SmallVector<Value>{slot, index(local), index(group * a.tokens + query)};
          State other;
          other.maximum = slice(partial.getResult(0), offset, {1, 1, count}, {1, count});
          other.sum = slice(partial.getResult(1), offset, {1, 1, count}, {1, count});
          offset.push_back(index(0));
          other.output = slice(partial.getResult(2), offset, {1, 1, count, a.valueDepth}, {1, count, a.valueDepth});
          state = combine(state, other);
        }
        int64_t covered = partition * splits;
        if (covered < prefix.length)
          state = run(prefix, head, group, query, count, index(covered), prefix.length - covered, state);
        for (const KVSegment &segment : ArrayRef<KVSegment>(segments).drop_front())
          state = run(segment, head, group, query, count, index(0), segment.length, state);
        publish(merge, finish(state), merge.getRegionOutArgs()[0],
                {head, index(group * a.tokens + query), index(0)}, {1, count, a.valueDepth});
      });
    }
    return finishForall(merge).front();
  }

  bool slicedKV() const { return segments.size() == 2; }

private:
  IRRewriter &b;
  Attention &a;
  const TargetSpec &t;
  Location loc;
  MLIRContext *ctx;
  int64_t splits = 1, headCores = 1, headsPerCore = 1;
  SmallVector<KVSegment, 2> segments;

  void findKVSegments() {
    auto key = a.qk.getDpsInputs()[1].getDefiningOp<tensor::ConcatOp>();
    auto value = a.pv.getDpsInputs()[1].getDefiningOp<tensor::ConcatOp>();
    int64_t keyAxis = a.transposedKey ? 1 : 2;
    // Only a private, static, two-part cache append is folded. Other concat
    // users keep their original materialisation; no cache ABI is changed.
    if (!key || !value || !key.getResult().hasOneUse() || !value.getResult().hasOneUse() ||
        key.getDim() != uint64_t(keyAxis) || value.getDim() != 1 ||
        key.getInputs().size() != 2 || value.getInputs().size() != 2) return;
    SmallVector<KVSegment, 2> candidate;
    int64_t offset = 0;
    for (unsigned i = 0; i < 2; ++i) {
      Value k = key.getInputs()[i], v = value.getInputs()[i];
      if (!staticF16(k) || !staticF16(v)) return;
      auto kt = cast<RankedTensorType>(k.getType()), vt = cast<RankedTensorType>(v.getType());
      if (kt.getRank() != 3 || vt.getRank() != 3 || kt.getDimSize(0) != a.heads || vt.getDimSize(0) != a.heads ||
          kt.getDimSize(a.transposedKey ? 2 : 1) != a.depth || vt.getDimSize(2) != a.valueDepth ||
          kt.getDimSize(keyAxis) != vt.getDimSize(1)) return;
      int64_t length = vt.getDimSize(1);
      if (!length) return;
      candidate.push_back({k, v, length, offset});
      offset += length;
    }
    if (offset != a.keys || candidate.front().length < splits) return;
    segments = std::move(candidate);
  }

  Value index(int64_t v) { return arith::ConstantIndexOp::create(b, loc, v); }
  Value add(Value v, int64_t amount) { return arith::AddIOp::create(b, loc, v, index(amount)); }
  Value multiply(Value v, int64_t amount) { return arith::MulIOp::create(b, loc, v, index(amount)); }
  SmallVector<OpFoldResult> attrs(ArrayRef<int64_t> numbers) {
    SmallVector<OpFoldResult> result;
    for (int64_t n : numbers) result.push_back(b.getIndexAttr(n));
    return result;
  }
  Value empty(ArrayRef<int64_t> shape, Type type) { return tensor::EmptyOp::create(b, loc, shape, type); }
  Value fill(ArrayRef<int64_t> shape, float number) {
    Value zero = arith::ConstantOp::create(b, loc, b.getF32FloatAttr(number));
    return linalg::FillOp::create(b, loc, zero, empty(shape, b.getF32Type())).getResult(0);
  }
  Value slice(Value source, ArrayRef<Value> offsets, ArrayRef<int64_t> sizes, ArrayRef<int64_t> shape = {}) {
    SmallVector<OpFoldResult> off;
    for (Value v : offsets) off.push_back(v);
    auto element = cast<RankedTensorType>(source.getType()).getElementType();
    auto type = RankedTensorType::get(shape.empty() ? sizes : shape, element);
    return tensor::ExtractSliceOp::create(b, loc, type, source, off, attrs(sizes),
                                           attrs(SmallVector<int64_t>(sizes.size(), 1)));
  }
  scf::ForallOp forall(int64_t count, ValueRange outputs, bool decode) {
    auto loop = scf::ForallOp::create(b, loc, attrs({count}), outputs, std::nullopt);
    loop->setAttr("hyb.cores", b.getI64IntegerAttr(count));
    loop->setAttr("hyb.flash_tile", b.getUnitAttr());
    if (decode) loop->setAttr("hyb.flash_decoding", b.getUnitAttr());
    return loop;
  }
  SmallVector<Value> finishForall(scf::ForallOp loop) {
    if (loop->getAttrOfType<IntegerAttr>("hyb.cores").getInt() != 1) {
      b.setInsertionPointAfter(loop);
      return SmallVector<Value>(loop.getResults().begin(), loop.getResults().end());
    }
    // The generic single-trip promotion assumes one parallel_insert_slice per
    // output. Our worker can publish several disjoint head/query slices. Chain
    // those updates explicitly for a one-worker region before canonicalizing.
    b.setInsertionPoint(loop);
    IRMapping mapping;
    mapping.map(loop.getInductionVar(0), index(0));
    mapping.map(loop.getRegionOutArgs(), loop.getOutputs());
    for (Operation &op : loop.getBody()->without_terminator()) b.clone(op, mapping);
    auto remap = [&](ArrayRef<OpFoldResult> mixed) {
      SmallVector<OpFoldResult> result;
      for (OpFoldResult part : mixed) {
        if (auto value = dyn_cast<Value>(part)) result.push_back(mapping.lookupOrDefault(value));
        else result.push_back(part);
      }
      return result;
    };
    for (Operation &op : loop.getTerminator().getRegion().front()) {
      auto insert = cast<tensor::ParallelInsertSliceOp>(op);
      Value destination = mapping.lookupOrDefault(insert.getDest());
      Value updated = tensor::InsertSliceOp::create(b, loc, mapping.lookupOrDefault(insert.getSource()), destination,
          remap(insert.getMixedOffsets()), remap(insert.getMixedSizes()), remap(insert.getMixedStrides()));
      mapping.map(insert.getDest(), updated);
    }
    SmallVector<Value> results;
    for (Value output : loop.getRegionOutArgs()) results.push_back(mapping.lookup(output));
    b.replaceOp(loop, results);
    b.setInsertionPointAfter(results.back().getDefiningOp());
    return results;
  }
  void publish(scf::ForallOp loop, Value value, Value output, ArrayRef<Value> offsets, ArrayRef<int64_t> sizes) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToEnd(&loop.getTerminator().getRegion().front());
    SmallVector<OpFoldResult> off;
    for (Value v : offsets) off.push_back(v);
    tensor::ParallelInsertSliceOp::create(b, loc, value, output, off, attrs(sizes),
                                           attrs(SmallVector<int64_t>(sizes.size(), 1)));
  }
  void forTiles(Value worker, llvm::function_ref<void(Value, int64_t, int64_t, int64_t, int64_t)> fn) {
    Value first = multiply(worker, headsPerCore);
    for (int64_t h = 0; h < headsPerCore; ++h) {
      // Pack complete GQA groups when the query sequence is short. The score
      // view remains [1, Gtile, M, Ktile], so one token mask is broadcast to
      // every group while the SA sees Gtile*M rows rather than tiny M rows.
      if (a.groups > 1 && a.tokens <= t.npu.attentionQueryTile) {
        int64_t groupsPerTile = std::max<int64_t>(1, t.npu.attentionQueryTile / a.tokens);
        for (int64_t group = 0; group < a.groups; group += groupsPerTile) {
          int64_t groupCount = std::min(groupsPerTile, a.groups - group);
          fn(add(first, h), h, group, 0, groupCount * a.tokens);
        }
      } else {
        for (int64_t group = 0; group < a.groups; ++group)
          for (int64_t query = 0; query < a.tokens; query += t.npu.attentionQueryTile)
            fn(add(first, h), h, group, query, std::min(t.npu.attentionQueryTile, a.tokens - query));
      }
    }
  }
  AffineMap identity(unsigned rank) { return AffineMap::getMultiDimIdentityMap(rank, ctx); }
  AffineMap rowMap() { return AffineMap::get(3, 0, {getAffineDimExpr(0, ctx), getAffineDimExpr(1, ctx)}, ctx); }
  using Body = llvm::function_ref<Value(OpBuilder &, Location, ValueRange)>;
  Value generic(ValueRange inputs, ArrayRef<int64_t> shape, Type element, ArrayRef<AffineMap> maps, Body body,
                Value init = {}, bool reduction = false) {
    if (!init) init = empty(shape, element);
    SmallVector<utils::IteratorType> iterators(maps.back().getNumDims(), utils::IteratorType::parallel);
    if (reduction) iterators.back() = utils::IteratorType::reduction;
    auto op = linalg::GenericOp::create(b, loc, TypeRange{init.getType()}, inputs, ValueRange{init}, maps, iterators,
        [&](OpBuilder &nested, Location l, ValueRange args) {
          linalg::YieldOp::create(nested, l, body(nested, l, args));
        });
    op->setAttr("hyb.flash_math", b.getUnitAttr());
    return op.getResult(0);
  }
  Value elementwise(ValueRange inputs, ArrayRef<AffineMap> maps, ArrayRef<int64_t> shape, Body body,
                    Type element = {}, Value init = {}) {
    return generic(inputs, shape, element ? element : b.getF32Type(), maps, body, init);
  }
  static Value f32(OpBuilder &b, Location loc, Value v) {
    return v.getType().isF32() ? v : arith::ExtFOp::create(b, loc, b.getF32Type(), v);
  }
  static Value constant(OpBuilder &b, Location loc, float x) {
    return arith::ConstantOp::create(b, loc, b.getF32FloatAttr(x));
  }
  SmallVector<int64_t> shapeOf(Value value) { return llvm::to_vector(cast<RankedTensorType>(value.getType()).getShape()); }
  State initial(int64_t count) {
    return {fill({1, count}, -std::numeric_limits<float>::infinity()), fill({1, count}, 0),
            fill({1, count, a.valueDepth}, 0)};
  }
  Value maximum(Value x, Value y) {
    return elementwise({x, y}, {identity(2), identity(2), identity(2)}, shapeOf(x),
        [](OpBuilder &b, Location l, ValueRange v) { return arith::MaximumFOp::create(b, l, v[0], v[1]); },
        b.getF32Type(), x);
  }
  Value rescale(Value ownMaximum, Value otherMaximum) {
    return elementwise({ownMaximum, otherMaximum}, {identity(2), identity(2), identity(2)}, shapeOf(ownMaximum),
        [](OpBuilder &b, Location l, ValueRange v) -> Value {
          Value next = arith::MaximumFOp::create(b, l, v[0], v[1]);
          Value equal = arith::CmpFOp::create(b, l, arith::CmpFPredicate::OEQ, v[0], next);
          Value delta = arith::SubFOp::create(b, l, v[0], next);
          Value safe = arith::SelectOp::create(b, l, equal, constant(b, l, 0), delta);
          return math::ExpOp::create(b, l, safe);
        });
  }
  State combine(State x, State y) {
    Value alpha = rescale(x.maximum, y.maximum), beta = rescale(y.maximum, x.maximum);
    Value m = maximum(x.maximum, y.maximum);
    auto body = [](OpBuilder &b, Location l, ValueRange v) -> Value {
      Value left = arith::MulFOp::create(b, l, v[0], v[1]);
      Value right = arith::MulFOp::create(b, l, v[2], v[3]);
      return arith::AddFOp::create(b, l, left, right);
    };
    Value sum = elementwise({x.sum, alpha, y.sum, beta},
        {identity(2), identity(2), identity(2), identity(2), identity(2)}, shapeOf(x.sum), body,
        b.getF32Type(), x.sum);
    Value output = elementwise({x.output, alpha, y.output, beta},
        {identity(3), rowMap(), identity(3), rowMap(), identity(3)}, shapeOf(x.output), body,
        b.getF32Type(), x.output);
    return {m, sum, output};
  }
  Value finish(State state) {
    return elementwise({state.output, state.sum}, {identity(3), rowMap(), identity(3)}, shapeOf(state.output),
        [](OpBuilder &b, Location l, ValueRange v) -> Value {
          Value zero = constant(b, l, 0), one = constant(b, l, 1);
          Value nonempty = arith::CmpFOp::create(b, l, arith::CmpFPredicate::OGT, v[1], zero);
          Value divisor = arith::SelectOp::create(b, l, nonempty, v[1], one);
          Value normalized = arith::DivFOp::create(b, l, v[0], divisor);
          Value safe = arith::SelectOp::create(b, l, nonempty, normalized, zero);
          return arith::TruncFOp::create(b, l, b.getF16Type(), safe);
        }, b.getF16Type());
  }

  /// Clone only the score's elementwise preprocessing and its GQA views. The
  /// source body and FP16 materialisation boundaries remain unchanged.
  Value scoreTile(Value value, Value scores, Value head, int64_t group, int64_t query, int64_t count,
                  Value key, int64_t keyCount) {
    if (value == a.qk.getResult(0)) return scores;
    Operation *op = value.getDefiningOp();
    auto type = cast<RankedTensorType>(value.getType());
    int64_t groupCount = count > a.tokens ? count / a.tokens : 1;
    int64_t tokenCount = count / groupCount;
    SmallVector<int64_t> shape;
    SmallVector<Value> offsets;
    if (type.getRank() == 4) {
      shape = {1, groupCount, tokenCount, keyCount};
      offsets = {head, index(group), index(query), key};
    } else if (type.getRank() == 2) {
      shape = {count, keyCount};
      offsets = {index(group * a.tokens + query), key};
    } else if (type.getShape() == ArrayRef<int64_t>({a.heads, a.rows, a.keys})) {
      shape = {1, count, keyCount};
      offsets = {head, index(group * a.tokens + query), key};
    } else {
      // H=1 was folded, leaving the GQA [G,M,K] view.
      shape = {groupCount, tokenCount, keyCount};
      offsets = {index(group), index(query), key};
    }
    if (auto expand = dyn_cast<tensor::ExpandShapeOp>(op)) {
      Value input = scoreTile(expand.getSrc(), scores, head, group, query, count, key, keyCount);
      return tensor::ExpandShapeOp::create(b, loc, RankedTensorType::get(shape, type.getElementType()), input,
                                             expand.getReassociationIndices());
    }
    if (auto collapse = dyn_cast<tensor::CollapseShapeOp>(op)) {
      Value input = scoreTile(collapse.getSrc(), scores, head, group, query, count, key, keyCount);
      return tensor::CollapseShapeOp::create(b, loc, RankedTensorType::get(shape, type.getElementType()), input,
                                               collapse.getReassociationIndices());
    }
    auto source = cast<linalg::GenericOp>(op);
    unsigned scoreIndex = a.scoreInput.lookup(op);
    auto maps = source.getIndexingMapsArray();
    SmallVector<Value> inputs;
    for (auto [i, input] : llvm::enumerate(source.getDpsInputs())) {
      if (i == scoreIndex) {
        inputs.push_back(scoreTile(input, scores, head, group, query, count, key, keyCount));
        continue;
      }
      SmallVector<Value> inputOffsets;
      SmallVector<int64_t> inputSizes;
      for (AffineExpr expr : maps[i].getResults()) {
        if (auto dim = dyn_cast<AffineDimExpr>(expr)) {
          inputOffsets.push_back(offsets[dim.getPosition()]);
          inputSizes.push_back(shape[dim.getPosition()]);
        } else {
          inputOffsets.push_back(index(cast<AffineConstantExpr>(expr).getValue()));
          inputSizes.push_back(1);
        }
      }
      inputs.push_back(slice(input, inputOffsets, inputSizes));
    }
    Value init = empty(shape, type.getElementType());
    auto cloned = linalg::GenericOp::create(b, loc, TypeRange{init.getType()}, inputs, ValueRange{init}, maps,
        source.getIteratorTypesArray(), [&](OpBuilder &nested, Location l, ValueRange args) {
          IRMapping mapping;
          mapping.map(source.getRegion().front().getArguments(), args);
          for (Operation &scalar : source.getRegion().front()) nested.clone(scalar, mapping);
        });
    return cloned.getResult(0);
  }

  State tile(const KVSegment &segment, Value head, int64_t group, int64_t query, int64_t count,
             Value key, int64_t keyCount, State old) {
    Value q = slice(a.qk.getDpsInputs()[0], {head, index(group * a.tokens + query), index(0)}, {1, count, a.depth});
    Value k = a.transposedKey ? slice(segment.key, {head, key, index(0)}, {1, keyCount, a.depth})
                             : slice(segment.key, {head, index(0), key}, {1, a.depth, keyCount});
    Value zero16 = arith::ConstantOp::create(b, loc, b.getF16FloatAttr(0));
    Value scoreInit = linalg::FillOp::create(b, loc, zero16, empty({1, count, keyCount}, b.getF16Type())).getResult(0);
    auto qk = linalg::BatchMatmulOp::create(b, loc, ValueRange{q, k}, ValueRange{scoreInit});
    qk->setAttr("indexing_maps", b.getAffineMapArrayAttr(a.qk.getIndexingMapsArray()));
    qk->setAttr("hyb.flash_tile", b.getUnitAttr());
    Value maskKey = add(key, segment.maskOffset);
    Value scores = scoreTile(a.score, qk.getResult(0), head, group, query, count, maskKey, keyCount);
    if (cast<RankedTensorType>(scores.getType()).getRank() == 4)
      scores = tensor::CollapseShapeOp::create(b, loc, scores, SmallVector<ReassociationIndices>{{0}, {1, 2}, {3}});
    auto scoreType = cast<RankedTensorType>(scores.getType());
    if (scoreType.getRank() == 3 && scoreType.getDimSize(0) != 1)
      scores = tensor::CollapseShapeOp::create(b, loc, scores, SmallVector<ReassociationIndices>{{0, 1}, {2}});
    if (cast<RankedTensorType>(scores.getType()).getRank() == 2)
      scores = tensor::ExpandShapeOp::create(b, loc, RankedTensorType::get({1, count, keyCount}, b.getF16Type()),
                                               scores, SmallVector<ReassociationIndices>{{0, 1}, {2}});
    Value tileMax = generic({scores}, {1, count}, b.getF32Type(), {identity(3), rowMap()},
        [](OpBuilder &b, Location l, ValueRange v) -> Value {
          return arith::MaximumFOp::create(b, l, f32(b, l, v[0]), v[1]);
        }, fill({1, count}, -std::numeric_limits<float>::infinity()), true);
    // Compute alpha before updating m in place; all three yielded states then
    // remain equivalent to the corresponding scf.for iteration destinations.
    Value alpha = rescale(old.maximum, tileMax), m = maximum(old.maximum, tileMax);
    Value p32 = elementwise({scores, m}, {identity(3), rowMap(), identity(3)}, {1, count, keyCount},
        [](OpBuilder &b, Location l, ValueRange v) -> Value {
          Value noKeys = arith::CmpFOp::create(b, l, arith::CmpFPredicate::OEQ, v[1],
                                               constant(b, l, -std::numeric_limits<float>::infinity()));
          Value delta = arith::SubFOp::create(b, l, f32(b, l, v[0]), v[1]);
          Value safeDelta = arith::SelectOp::create(b, l, noKeys, constant(b, l, 0), delta);
          Value exponential = math::ExpOp::create(b, l, safeDelta);
          return arith::SelectOp::create(b, l, noKeys, constant(b, l, 0), exponential);
        });
    Value blockSum = generic({p32}, {1, count}, b.getF32Type(), {identity(3), rowMap()},
        [](OpBuilder &b, Location l, ValueRange v) { return arith::AddFOp::create(b, l, v[0], v[1]); },
        fill({1, count}, 0), true);
    Value p16 = elementwise({p32}, {identity(3), identity(3)}, {1, count, keyCount},
        [](OpBuilder &b, Location l, ValueRange v) { return arith::TruncFOp::create(b, l, b.getF16Type(), v[0]); },
        b.getF16Type());
    Value v = slice(segment.value, {head, key, index(0)}, {1, keyCount, a.valueDepth});
    auto pv = linalg::BatchMatmulOp::create(b, loc, ValueRange{p16, v},
                                            ValueRange{fill({1, count, a.valueDepth}, 0)});
    pv->setAttr("hyb.flash_tile", b.getUnitAttr());
    Value product = pv.getResult(0);
    auto accumulation = [](OpBuilder &b, Location l, ValueRange v) -> Value {
      Value old = arith::MulFOp::create(b, l, v[0], v[1]);
      return arith::AddFOp::create(b, l, old, v[2]);
    };
    Value sum = elementwise({old.sum, alpha, blockSum},
        {identity(2), identity(2), identity(2), identity(2)}, {1, count}, accumulation,
        b.getF32Type(), old.sum);
    Value output = elementwise({old.output, alpha, product},
        {identity(3), rowMap(), identity(3), identity(3)}, {1, count, a.valueDepth}, accumulation,
        b.getF32Type(), old.output);
    return {m, sum, output};
  }

  State run(const KVSegment &segment, Value head, int64_t group, int64_t query, int64_t count,
            Value start, int64_t length, State state) {
    int64_t tileSize = t.npu.attentionKeyTile;
    int64_t full = length / tileSize;
    if (full) {
      auto loop = scf::ForOp::create(b, loc, index(0), index(full), index(1),
          ValueRange{state.maximum, state.sum, state.output},
          [&](OpBuilder &nested, Location l, Value iv, ValueRange args) {
            OpBuilder::InsertionGuard guard(b);
            b.setInsertionPointToStart(nested.getInsertionBlock());
            Value at = arith::AddIOp::create(b, loc, start, multiply(iv, tileSize));
            State next = tile(segment, head, group, query, count, at, tileSize, {args[0], args[1], args[2]});
            scf::YieldOp::create(b, l, ValueRange{next.maximum, next.sum, next.output});
          });
      loop->setAttr("hyb.flash_tile", b.getUnitAttr());
      state = {loop.getResult(0), loop.getResult(1), loop.getResult(2)};
    }
    if (int64_t tail = length % tileSize)
      state = tile(segment, head, group, query, count, add(start, full * tileSize), tail, state);
    return state;
  }
};

struct HybNpuFlashAttentionPass
    : mlir::hybrid::impl::HybNpuFlashAttentionBase<HybNpuFlashAttentionPass> {
  using Base::Base;

  void runOnOperation() override {
    std::string error;
    auto target = TargetSpec::load(spec, error);
    if (failed(target)) { getOperation().emitError(error); return signalPassFailure(); }
    if (!target->npu.flashAttention) return;
    for (auto fn : getOperation().getOps<func::FuncOp>())
      if (auto phase = fn->getAttrOfType<StringAttr>(kPhase); phase && phase.getValue() != "npu") {
        fn.emitError("FlashAttention is currently supported only for NPU-only programs");
        return signalPassFailure();
      }
    for (auto fn : getOperation().getOps<func::FuncOp>()) {
      auto phase = fn->getAttrOfType<StringAttr>(kPhase);
      if (!phase || phase.getValue() != "npu") continue;
      if (fn->hasAttr("hyb.npu_kernel")) {
        fn.emitError("FlashAttention is unsupported in isolated PIM packet kernels");
        return signalPassFailure();
      }
      SmallVector<linalg::BatchMatmulOp> candidates;
      fn.walk([&](linalg::BatchMatmulOp mm) {
        if (!mm->hasAttr("hyb.flash_tile") && !mm->getParentOfType<scf::ForallOp>() &&
            !mm->getParentOfType<scf::ForOp>() &&
            (hasExpAncestor(mm.getDpsInputs()[0]) || hasMatmulAncestor(mm.getDpsInputs()[0]))) candidates.push_back(mm);
      });
      IRRewriter rewriter(&getContext());
      int64_t transformed = 0, segmented = 0;
      for (auto candidate : llvm::reverse(candidates)) {
        Attention attention;
        if (!attention.match(candidate)) {
          candidate.emitError("unsupported requested FlashAttention chain: ") << attention.error;
          return signalPassFailure();
        }
        if (target->npu.flashDecoding && attention.keys < target->npu.attentionSplits) {
          candidate.emitError("Flash-Decoding needs at least one key per requested split");
          return signalPassFailure();
        }
        rewriter.setInsertionPoint(candidate);
        OnlineEmitter emitter(rewriter, attention, *target);
        rewriter.replaceOp(candidate, emitter.emit());
        segmented += emitter.slicedKV();
        ++transformed;
      }
      if (!transformed) continue; // Repeat application leaves metadata and IR unchanged.
      RewritePatternSet cleanup(&getContext());
      for (Dialect *dialect : getContext().getLoadedDialects()) dialect->getCanonicalizationPatterns(cleanup);
      for (RegisteredOperationName op : getContext().getRegisteredOperations())
        op.getCanonicalizationPatterns(cleanup, &getContext());
      if (failed(applyPatternsGreedily(fn, std::move(cleanup)))) return signalPassFailure();
      fn->setAttr("hyb.flash_attention", rewriter.getDictionaryAttr({
          rewriter.getNamedAttr("algorithm", rewriter.getStringAttr(target->npu.flashDecoding ? "flash_decoding" : "flash_attention")),
          rewriter.getNamedAttr("transformed_chains", rewriter.getI64IntegerAttr(transformed)),
          rewriter.getNamedAttr("segmented_kv_chains", rewriter.getI64IntegerAttr(segmented)),
          rewriter.getNamedAttr("query_tile", rewriter.getI64IntegerAttr(target->npu.attentionQueryTile)),
          rewriter.getNamedAttr("key_tile", rewriter.getI64IntegerAttr(target->npu.attentionKeyTile)),
          rewriter.getNamedAttr("splits", rewriter.getI64IntegerAttr(target->npu.flashDecoding ? target->npu.attentionSplits : 1))}));
    }
  }
};

} // namespace
