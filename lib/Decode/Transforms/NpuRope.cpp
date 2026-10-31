//===- NpuRope.cpp - hyb-npu-rope ------------------------------------------===//
//
// The exported rotate-half RoPE is
//   out = x * cos + concat(-x[..., D/2:], x[..., :D/2]) * sin
// whose concatenation bufferizes to an L2 buffer filled by single-core copies; attention then
// reads the result through a [1, 0, 2] transpose into the head-major layout, another single-core
// pass over L2. Each output half needs only the two halves of x:
//   out[..., :D/2] = x_lo * cos_lo - x_hi * sin_lo,   out[..., D/2:] = x_hi * cos_hi + x_lo * sin_hi
// with the same rounding (a + (-b) is a - b, and (-p) * q is -(p * q)). Both halves become
// generics over a per-core range of heads; when the transpose is the only user they write the
// head-major layout directly.
//===----------------------------------------------------------------------===//

#include "npu/Decode/Transforms/PassSupport.h"
#include "npu/Decode/Transforms/NpuCost.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir::hybrid {
#define GEN_PASS_DEF_HYBNPUROPE
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

/// The block argument behind `v`, looking through one arith.extf.
BlockArgument argument(Value v) {
  if (auto ext = v.getDefiningOp<arith::ExtFOp>()) v = ext.getIn();
  return dyn_cast<BlockArgument>(v);
}

/// tensor.extract_slice of `x` [T, H, D] at last-axis offset `at`, size D/2, unit strides.
bool isHalf(tensor::ExtractSliceOp s, Value x, int64_t at) {
  auto t = cast<RankedTensorType>(x.getType());
  return s && s.getSource() == x && s.hasUnitStride() &&
         s.getStaticOffsets() == ArrayRef<int64_t>({0, 0, at}) &&
         s.getStaticSizes() == ArrayRef<int64_t>({t.getDimSize(0), t.getDimSize(1), t.getDimSize(2) / 2});
}

/// An all-parallel identity-map generic that only negates its input (with FP16 <-> FP32 casts).
bool isNegation(linalg::GenericOp g) {
  if (g.getNumDpsInputs() != 1 || g.getNumDpsInits() != 1 || g.getNumReductionLoops() ||
      !llvm::all_of(g.getIndexingMapsArray(), [](AffineMap m) { return m.isIdentity(); }))
    return false;
  Block &body = g.getRegion().front();
  Value v = cast<linalg::YieldOp>(body.getTerminator()).getValues()[0];
  if (auto t = v.getDefiningOp<arith::TruncFOp>()) v = t.getIn();
  auto neg = v.getDefiningOp<arith::NegFOp>();
  if (!neg || argument(neg.getOperand()) != body.getArgument(0)) return false;
  return llvm::count_if(body, [](Operation &op) { return !isa<arith::ExtFOp, arith::TruncFOp>(op); }) == 2;
}

struct Rope {
  linalg::GenericOp rope, negation;
  tensor::ConcatOp concat;
  Value x, cos, sin;
  linalg::TransposeOp transpose;      // the [1, 0, 2] sole user, if any
};

/// rope = generic(x, cos, concat(neg(x[.., D/2:]), x[.., :D/2]), sin) { x * cos + r * sin } with
/// x [T, H, D] and the tables [T, D] read by (t, h, d) -> (t, d), in any operand order.
std::optional<Rope> match(linalg::GenericOp g) {
  if (g.getNumDpsInputs() != 4 || g.getNumDpsInits() != 1 || g.getNumLoops() != 3 ||
      g.getNumReductionLoops() || !g.hasPureTensorSemantics() ||
      !g.getDpsInits()[0].getDefiningOp<tensor::EmptyOp>())
    return std::nullopt;
  auto out = dyn_cast<RankedTensorType>(g.getResult(0).getType());
  if (!out || !out.hasStaticShape() || !out.getElementType().isF16() || out.getDimSize(2) % 2) return std::nullopt;
  int64_t T = out.getDimSize(0), D = out.getDimSize(2);
  MLIRContext *ctx = g.getContext();
  AffineMap identity = AffineMap::getMultiDimIdentityMap(3, ctx);
  AffineMap table = AffineMap::get(3, 0, {getAffineDimExpr(0, ctx), getAffineDimExpr(2, ctx)}, ctx);
  auto maps = g.getIndexingMapsArray();
  if (maps.back() != identity) return std::nullopt;
  int concatAt = -1, xAt = -1;
  SmallVector<int> tables;
  for (int i = 0; i < 4; ++i) {
    Value v = g.getDpsInputs()[i];
    if (maps[i] == identity && v.getType() == out) {
      int &slot = v.getDefiningOp<tensor::ConcatOp>() ? concatAt : xAt;
      if (slot >= 0) return std::nullopt;
      slot = i;
    } else if (maps[i] == table && v.getType() == RankedTensorType::get({T, D}, out.getElementType())) {
      tables.push_back(i);
    } else {
      return std::nullopt;
    }
  }
  if (concatAt < 0 || xAt < 0 || tables.size() != 2) return std::nullopt;
  Value x = g.getDpsInputs()[xAt];
  auto concat = g.getDpsInputs()[concatAt].getDefiningOp<tensor::ConcatOp>();
  if (concat.getDim() != 2 || concat.getInputs().size() != 2 || !concat->hasOneUse()) return std::nullopt;
  auto negation = concat.getInputs()[0].getDefiningOp<linalg::GenericOp>();
  auto low = concat.getInputs()[1].getDefiningOp<tensor::ExtractSliceOp>();
  if (!negation || !negation->hasOneUse() || !isNegation(negation) || !isHalf(low, x, 0)) return std::nullopt;
  auto high = negation.getDpsInputs()[0].getDefiningOp<tensor::ExtractSliceOp>();
  if (!isHalf(high, x, D / 2)) return std::nullopt;
  // Body: yield(truncf?(addf(mulf(x, c), mulf(r, s)))), operands of each product in any order.
  Block &body = g.getRegion().front();
  Value y = cast<linalg::YieldOp>(body.getTerminator()).getValues()[0];
  if (auto t = y.getDefiningOp<arith::TruncFOp>()) y = t.getIn();
  auto add = y.getDefiningOp<arith::AddFOp>();
  if (!add || llvm::count_if(body, [](Operation &op) { return !isa<arith::ExtFOp, arith::TruncFOp>(op); }) != 4)
    return std::nullopt;
  auto inputOf = [&](Value v) -> int {
    BlockArgument a = argument(v);
    return a && a.getOwner() == &body && a.getArgNumber() < 4 ? int(a.getArgNumber()) : -1;
  };
  auto product = [&](Value v) -> std::optional<std::pair<int, int>> {
    auto m = v.getDefiningOp<arith::MulFOp>();
    if (!m) return std::nullopt;
    return std::make_pair(inputOf(m.getLhs()), inputOf(m.getRhs()));
  };
  auto first = product(add.getLhs()), second = product(add.getRhs());
  if (!first || !second) return std::nullopt;
  auto other = [](std::pair<int, int> p, int known) { return p.first == known ? p.second : p.second == known ? p.first : -1; };
  if (other(*first, xAt) < 0) std::swap(first, second);     // IEEE addition is commutative
  int cosAt = other(*first, xAt), sinAt = other(*second, concatAt);
  if (cosAt < 0 || sinAt < 0 || cosAt == sinAt || !llvm::is_contained(tables, cosAt) ||
      !llvm::is_contained(tables, sinAt))
    return std::nullopt;
  Rope r{g, negation, concat, x, g.getDpsInputs()[cosAt], g.getDpsInputs()[sinAt], {}};
  if (g->hasOneUse())
    if (auto t = dyn_cast<linalg::TransposeOp>(*g->getUsers().begin());
        t && t.getPermutation() == ArrayRef<int64_t>({1, 0, 2}) && t.getInit().getDefiningOp<tensor::EmptyOp>())
      r.transpose = t;
  return r;
}

void rewrite(Rope &r, int64_t coreLimit) {
  OpBuilder b(r.rope);
  Location loc = r.rope.getLoc();
  MLIRContext *ctx = b.getContext();
  auto xt = cast<RankedTensorType>(r.x.getType());
  int64_t T = xt.getDimSize(0), H = xt.getDimSize(1), D = xt.getDimSize(2), half = D / 2;
  int64_t cores = largestDivisorAtMost(H, coreLimit), heads = H / cores;
  bool headMajor = bool(r.transpose);
  Type f16 = xt.getElementType(), f32 = b.getF32Type();
  SmallVector<int64_t> full = headMajor ? SmallVector<int64_t>{H, T, D} : SmallVector<int64_t>{T, H, D};
  SmallVector<int64_t> tile = headMajor ? SmallVector<int64_t>{heads, T, half} : SmallVector<int64_t>{T, heads, half};
  Value dest = tensor::EmptyOp::create(b, loc, full, f16);
  auto loop = scf::ForallOp::create(b, loc, attrs(b, {cores}), ValueRange{dest}, std::nullopt);
  loop->setAttr("hyb.cores", b.getI64IntegerAttr(cores));
  loop->setAttr("hyb.rope_split", b.getUnitAttr());
  b.setInsertionPointToStart(loop.getBody());
  Value first = arith::MulIOp::create(b, loc, loop.getInductionVar(0), arith::ConstantIndexOp::create(b, loc, heads));
  OpFoldResult zero = b.getIndexAttr(0);
  auto xHalf = [&](int64_t at) -> Value {
    return tensor::ExtractSliceOp::create(b, loc, r.x, SmallVector<OpFoldResult>{zero, first, b.getIndexAttr(at)},
                                          attrs(b, {T, heads, half}), attrs(b, {1, 1, 1}));
  };
  auto tableHalf = [&](Value t, int64_t at) -> Value {
    return tensor::ExtractSliceOp::create(b, loc, t, SmallVector<OpFoldResult>{zero, b.getIndexAttr(at)},
                                          attrs(b, {T, half}), attrs(b, {1, 1}));
  };
  // Loops (h, t, j) for the head-major tile, (t, h, j) otherwise; the output map is the identity.
  AffineExpr d0, d1, d2;
  bindDims(ctx, d0, d1, d2);
  AffineMap xMap = headMajor ? AffineMap::get(3, 0, {d1, d0, d2}, ctx) : AffineMap::getMultiDimIdentityMap(3, ctx);
  AffineMap tMap = AffineMap::get(3, 0, {headMajor ? d1 : d0, d2}, ctx);
  AffineMap oMap = AffineMap::getMultiDimIdentityMap(3, ctx);
  auto halfRope = [&](Value a, Value c, Value p, Value s, bool add) -> Value {
    Value init = tensor::EmptyOp::create(b, loc, tile, f16);
    auto g = linalg::GenericOp::create(
        b, loc, TypeRange{init.getType()}, ValueRange{a, c, p, s}, ValueRange{init},
        ArrayRef<AffineMap>{xMap, tMap, xMap, tMap, oMap},
        SmallVector<utils::IteratorType>(3, utils::IteratorType::parallel),
        [&](OpBuilder &nb, Location l, ValueRange args) {
          auto wide = [&](Value v) -> Value { return arith::ExtFOp::create(nb, l, f32, v); };
          Value direct = arith::MulFOp::create(nb, l, wide(args[0]), wide(args[1]));
          Value rotated = arith::MulFOp::create(nb, l, wide(args[2]), wide(args[3]));
          Value sum = add ? Value(arith::AddFOp::create(nb, l, direct, rotated))
                          : Value(arith::SubFOp::create(nb, l, direct, rotated));
          linalg::YieldOp::create(nb, l, ValueRange{arith::TruncFOp::create(nb, l, f16, sum)});
        });
    return g.getResult(0);
  };
  Value lo = xHalf(0), hi = xHalf(half);
  Value outLo = halfRope(lo, tableHalf(r.cos, 0), hi, tableHalf(r.sin, 0), false);
  Value outHi = halfRope(hi, tableHalf(r.cos, half), lo, tableHalf(r.sin, half), true);
  b.setInsertionPointToEnd(&loop.getTerminator().getRegion().front());
  auto at = [&](int64_t column) -> SmallVector<OpFoldResult> {
    if (headMajor) return {first, zero, b.getIndexAttr(column)};
    return {zero, first, b.getIndexAttr(column)};
  };
  for (auto [value, column] : {std::make_pair(outLo, int64_t(0)), std::make_pair(outHi, half)})
    tensor::ParallelInsertSliceOp::create(b, loc, value, loop.getRegionOutArgs()[0], at(column), attrs(b, tile),
                                          attrs(b, {1, 1, 1}));
  Operation *replaced = headMajor ? r.transpose.getOperation() : r.rope.getOperation();
  replaced->getResult(0).replaceAllUsesWith(loop.getResult(0));
  if (headMajor) r.transpose.erase();
  r.rope.erase();
  r.concat.erase();
  r.negation.erase();
}

/// The split rule of NpuCost.h on the work the rewrite distributes: the RoPE generic, the negation,
/// the rotate-half concat and (when folded) the head-major transpose, all on one core otherwise.
/// npu.rope_split_min_elements is a floor kept for experiments (0 by default).
bool pays(func::FuncOp fn, const Rope &r, const TargetSpec &target) {
  auto x = cast<RankedTensorType>(r.x.getType());
  const double elements = double(x.getNumElements());
  const double copy = elements * cost::copyCyclesPerElement(target.npu, 2);
  double cycles = cost::genericCycles(target.npu, r.rope) + cost::genericCycles(target.npu, r.negation) + copy;
  if (r.transpose) cycles += copy;
  int64_t cores = largestDivisorAtMost(x.getDimSize(1), target.npu.cores);
  bool split = cost::splitPays(target.npu, cycles, cores) && x.getNumElements() >= target.npu.ropeSplitMinElements;
  std::string label;
  llvm::raw_string_ostream os(label);
  os << "rope " << r.x.getType();
  cost::recordDecision(fn, "split rope", label, split ? "split" : "one core",
                       {{"cores", double(cores)}, {"one_core_us", cycles / cost::cyclesPerUs(target.npu)},
                        {"saving_us", cost::splitSavingUs(target.npu, cycles, cores)},
                        {"split_cost_us", target.npu.coreSplitUs}});
  return split;
}

struct HybNpuRopePass : mlir::hybrid::impl::HybNpuRopeBase<HybNpuRopePass> {
  using Base::Base;
  void runOnOperation() override {
    auto target = loadTargetSpec(getOperation(), spec);
    if (failed(target)) return signalPassFailure();
    if (!target->npu.ropeSplit || hasPimPhase(getOperation())) return;
    for (func::FuncOp fn : getNpuPhases(getOperation())) {
      SmallVector<Rope> found;
      for (Operation &op : fn.front())
        if (auto g = dyn_cast<linalg::GenericOp>(op))
          if (auto r = match(g); r && pays(fn, *r, *target)) found.push_back(*r);
      for (Rope &r : found) rewrite(r, target->npu.cores);
      fn->setAttr("hyb.rope_splits", IntegerAttr::get(IntegerType::get(&getContext(), 64), int64_t(found.size())));
    }
  }
};
} // namespace
