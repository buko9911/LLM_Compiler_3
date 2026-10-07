//===- CodegenGeneric.cpp - NPU codegen: linalg.generic vectorizer ----------===//
//
// Body values are Const (folded), Scalar (row-invariant: per-row f32 operands and
// expressions of them, computed once per row on the scalar unit) or Vector (depend on the
// innermost loop; FP16 lanes, 32 per register). extf / truncf of a vector alias its
// register (the vector unit is FP16). A reduction over the innermost loop folds each tile
// with V_REDUCE (FP16 -> FP32; a product of two vectors uses the dot mode, so x*x never
// rounds to FP16) and an FP32 scalar accumulator.
//===----------------------------------------------------------------------===//

#include "CodegenImpl.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include <algorithm>
#include <functional>

namespace mlir::hybrid::codegen {

LogicalResult Codegen::lowerGeneric(linalg::GenericOp g, std::string &err) {
  auto iters = g.getIteratorTypesArray();
  unsigned nLoops = g.getNumLoops();
  SmallVector<int64_t> ranges = g.getStaticLoopRanges();
  if (nLoops == 0) {
    err = "rank-0 generic";
    return failure();
  }
  unsigned inner = nLoops - 1;
  for (unsigned d = 0; d < inner; ++d)
    if (iters[d] != utils::IteratorType::parallel) {
      err = "reduction over a non-innermost loop";
      return failure();
    }
  bool reduction = iters[inner] == utils::IteratorType::reduction;
  MLIRContext *ctx = g.getContext();
  SmallVector<View> views;
  SmallVector<AffineMap> maps;
  SmallVector<bool> vec;
  for (OpOperand &o : g->getOpOperands()) {
    FailureOr<View> v = viewOf(o.get(), err);
    if (failed(v)) return failure();
    AffineMap map = g.getMatchingIndexingMap(&o);
    bool isVec = false;
    for (unsigned r = 0; r < map.getNumResults(); ++r)
      if (map.getResult(r) == getAffineDimExpr(inner, ctx)) {
        if (r != map.getNumResults() - 1 || v->strides.back() != 1) {
          err = "innermost loop is not the contiguous dim of an operand";
          return failure();
        }
        isVec = true;
      }
    if (isVec && !v->et.isF16()) {
      err = "vector operand must be f16";
      return failure();
    }
    views.push_back(*v);
    maps.push_back(map);
    vec.push_back(isVec);
  }
  Block &body = g.getRegion().front();
  unsigned nIn = g.getNumDpsInputs();
  if (g.getNumDpsInits() != 1) {
    err = "generic with several results";
    return failure();
  }
  BlockArgument outArg = body.getArgument(nIn);
  Value result = cast<linalg::YieldOp>(body.getTerminator()).getValues()[0];

  // Reduction: the ops reached from the accumulator are the combiner and casts.
  DenseSet<Operation *> skip;
  Value pre, dotA, dotB;
  uint32_t redFunct = 1, combFunct = 1;
  if (reduction) {
    if (!views[nIn].et.isF32()) {
      err = "reduction accumulator must be f32";
      return failure();
    }
    SmallVector<Value> work{outArg};
    Operation *comb = nullptr;
    while (!work.empty()) {
      Value v = work.pop_back_val();
      for (Operation *u : v.getUsers()) {
        if (isa<linalg::YieldOp>(u) || !skip.insert(u).second) continue;
        if (isa<arith::AddFOp, arith::MaximumFOp, arith::MaxNumFOp>(u)) {
          if (comb) {
            err = "several reduction combiners";
            return failure();
          }
          comb = u;
        } else if (!isa<arith::ExtFOp, arith::TruncFOp>(u)) {
          err = "unsupported use of the reduction accumulator";
          return failure();
        }
        work.push_back(u->getResult(0));
      }
    }
    if (!comb) {
      err = "reduction without a combiner";
      return failure();
    }
    bool isMax = !isa<arith::AddFOp>(comb);
    redFunct = isMax ? 2 : 1;
    combFunct = isMax ? 4 : 1;
    for (Value o : comb->getOperands()) {
      Operation *d = o.getDefiningOp();
      if (!(d && skip.count(d)) && o != outArg) pre = o;
    }
    if (!pre) {
      err = "reduction without an input";
      return failure();
    }
  }

  // Classification.
  DenseMap<Value, K> kind;
  DenseMap<Value, double> cval;
  for (unsigned i = 0; i < nIn; ++i) kind[body.getArgument(i)] = vec[i] ? K::Vector : K::Scalar;
  if (!reduction) kind[outArg] = K::Vector;
  std::function<K(Value)> kindOf = [&](Value v) -> K {
    auto it = kind.find(v);
    if (it != kind.end()) return it->second;
    if (auto c = constValue(v)) {
      cval[v] = *c;
      return kind[v] = K::Const;
    }
    return K::Scalar;
  };
  for (Operation &op : body.without_terminator()) {
    if (skip.count(&op) || op.getNumResults() != 1) continue;
    K k = K::Const;
    for (Value o : op.getOperands()) k = std::max(k, kindOf(o));
    if (k == K::Const) {
      if (auto c = constValue(op.getResult(0))) cval[op.getResult(0)] = *c;
      else k = K::Scalar;
    }
    kind[op.getResult(0)] = k;
  }
  if (reduction)
    if (auto mul = pre.getDefiningOp<arith::MulFOp>();
        mul && redFunct == 1 && pre.hasOneUse() && kind.lookup(mul.getLhs()) == K::Vector &&
        kind.lookup(mul.getRhs()) == K::Vector) {
      dotA = mul.getLhs();
      dotB = mul.getRhs();
      skip.insert(mul);
      redFunct = 3;
    }
  for (unsigned i = 0; i < nIn; ++i)
    if (!vec[i] && !body.getArgument(i).use_empty() && !views[i].et.isF32()) {
      err = "row-invariant operand must be f32";
      return failure();
    }

  // Vector liveness: alias roots and the index of the last use.
  SmallVector<Operation *> vops;
  for (Operation &op : body.without_terminator())
    if (!skip.count(&op) && kind.lookup(op.getResult(0)) == K::Vector) vops.push_back(&op);
  DenseMap<Value, Value> aliasOf;
  auto rootOfV = [&](Value v) {
    auto it = aliasOf.find(v);
    return it == aliasOf.end() ? v : it->second;
  };
  for (Operation *op : vops)
    if (isa<arith::ExtFOp, arith::TruncFOp>(op)) aliasOf[op->getResult(0)] = rootOfV(op->getOperand(0));
  DenseMap<Value, int> lastUse;
  int endIndex = int(vops.size());
  for (auto [i, op] : llvm::enumerate(vops))
    for (Value o : op->getOperands())
      if (kind.lookup(o) == K::Vector) lastUse[rootOfV(o)] = std::max(lastUse.lookup(rootOfV(o)), int(i));
  if (reduction) {
    for (Value v : {pre, dotA, dotB})
      if (v) lastUse[rootOfV(v)] = endIndex;
  } else {
    lastUse[rootOfV(result)] = endIndex;
  }

  // Scalar registers: s1, s3 scratch; s11 the accumulator; the rest for body scalars.
  static constexpr uint32_t kPool[] = {12, 13, 14, 15, 4, 5, 6, 7, 8, 9, 10, 0, 2};
  const uint32_t accReg = 11, tmpReg = 3;
  DenseMap<Value, uint32_t> constReg, rowReg;
  unsigned nextPool = 0;
  auto take = [&]() -> FailureOr<uint32_t> {
    if (nextPool >= std::size(kPool)) {
      err = "too many scalar values in a generic body";
      return failure();
    }
    return kPool[nextPool++];
  };
  SmallVector<uint64_t> base(views.size(), 0);
  std::function<FailureOr<uint32_t>(Value)> scalarOf = [&](Value v) -> FailureOr<uint32_t> {
    if (auto it = constReg.find(v); it != constReg.end()) return it->second;
    if (auto it = rowReg.find(v); it != rowReg.end()) return it->second;
    K k = kindOf(v);
    if (k == K::Const) {
      auto r = take();
      if (failed(r)) return failure();
      sload(*r, uint64_t(poolL1 + scalarConst(float(cval.lookup(v)))));
      return constReg[v] = *r;
    }
    if (auto ba = dyn_cast<BlockArgument>(v)) {
      auto r = take();
      if (failed(r)) return failure();
      sload(*r, base[ba.getArgNumber()]);
      return rowReg[v] = *r;
    }
    Operation *d = v.getDefiningOp();
    if (isa<arith::ExtFOp, arith::TruncFOp>(d)) {
      auto r = scalarOf(d->getOperand(0));
      if (succeeded(r)) rowReg[v] = *r;
      return r;
    }
    auto a = scalarOf(d->getOperand(0));
    if (failed(a)) return failure();
    FailureOr<uint32_t> b = uint32_t(0);
    if (d->getNumOperands() > 1) b = scalarOf(d->getOperand(1));
    if (failed(b)) return failure();
    auto r = take();
    if (failed(r)) return failure();
    if (isa<arith::AddFOp>(d)) emit(rform(OP_S_ALU, *r, *a, *b, 0, 1));
    else if (isa<arith::SubFOp>(d)) emit(rform(OP_S_ALU, *r, *a, *b, 0, 2));
    else if (isa<arith::MulFOp>(d)) emit(rform(OP_S_ALU, *r, *a, *b, 0, 3));
    else if (isa<arith::MaximumFOp, arith::MaxNumFOp>(d)) emit(rform(OP_S_ALU, *r, *a, *b, 0, 4));
    else if (isa<arith::MinimumFOp, arith::MinNumFOp>(d)) emit(rform(OP_S_ALU, *r, *a, *b, 0, 5));
    else if (isa<arith::DivFOp>(d)) {
      emit(rform(OP_S_ALU, 1, *b, 0, 0, 6));        // s1 = 1 / b
      emit(rform(OP_S_ALU, *r, *a, 1, 0, 3));
    } else if (isa<math::SqrtOp>(d)) emit(rform(OP_S_ALU, *r, *a, 0, 0, 7));
    else if (isa<math::RsqrtOp>(d)) emit(rform(OP_S_ALU, *r, *a, 0, 0, 8));
    else if (isa<arith::NegFOp>(d)) {
      sload(1, uint64_t(poolL1 + scalarConst(0.0f)));
      emit(rform(OP_S_ALU, *r, 1, *a, 0, 2));
    } else {
      err = "unsupported scalar op " + d->getName().getStringRef().str();
      return failure();
    }
    return rowReg[v] = *r;
  };

  // Constants first (once per op), then the row scalars reuse the registers after them.
  std::function<LogicalResult(Value)> loadConsts = [&](Value v) -> LogicalResult {
    K k = kindOf(v);
    if (k == K::Const) return failed(scalarOf(v)) ? failure() : success();
    if (k != K::Scalar || isa<BlockArgument>(v)) return success();
    for (Value o : v.getDefiningOp()->getOperands())
      if (failed(loadConsts(o))) return failure();
    return success();
  };
  for (Operation *op : vops)
    for (Value o : op->getOperands())
      if (failed(loadConsts(o))) return failure();
  const unsigned constMark = nextPool;

  SmallVector<int64_t> outer(ranges.begin(), ranges.begin() + inner);
  int64_t n = ranges[inner];
  LogicalResult status = success();
  forEachOuter(outer, 0, [&](ArrayRef<int64_t> idx) {
    if (failed(status)) return;
    for (size_t i = 0; i < views.size(); ++i) {
      int64_t off = 0;
      for (unsigned r = 0; r < maps[i].getNumResults(); ++r) {
        AffineExpr e = maps[i].getResult(r);
        int64_t at = 0;
        if (auto de = dyn_cast<AffineDimExpr>(e)) at = de.getPosition() < inner ? idx[de.getPosition()] : 0;
        else if (auto ce = dyn_cast<AffineConstantExpr>(e)) at = ce.getValue();
        off += at * views[i].strides[r];
      }
      base[i] = addr(views[i], off);
    }
    rowReg.clear();
    nextPool = constMark;
    for (Operation *op : vops)
      for (Value o : op->getOperands()) {
        K k = kind.lookup(o);
        if ((k == K::Scalar || k == K::Const) && failed(scalarOf(o))) {
          status = failure();
          return;
        }
      }
    if (reduction) sload(accReg, base[nIn]);
    if (failed(emitTiles(body, nIn, vec, kind, vops, aliasOf, lastUse, base, n, reduction, pre, dotA, dotB,
                         redFunct, combFunct, result, constReg, rowReg, accReg, tmpReg, err))) {
      status = failure();
      return;
    }
    if (reduction) sstore(base[nIn], accReg);
  });
  return status;
}

LogicalResult Codegen::emitTiles(Block &body, unsigned nIn, ArrayRef<bool> vec, DenseMap<Value, K> &kind,
                                 ArrayRef<Operation *> vops, DenseMap<Value, Value> &aliasOf,
                                 DenseMap<Value, int> &lastUse, ArrayRef<uint64_t> base, int64_t n, bool reduction,
                                 Value pre, Value dotA, Value dotB, uint32_t redFunct, uint32_t combFunct, Value result,
                                 DenseMap<Value, uint32_t> &constReg, DenseMap<Value, uint32_t> &rowReg,
                                 uint32_t accReg, uint32_t tmpReg, std::string &err) {
  auto rootOfV = [&](Value v) {
    auto it = aliasOf.find(v);
    return it == aliasOf.end() ? v : it->second;
  };
  auto sreg = [&](Value v) -> uint32_t {
    if (auto it = constReg.find(v); it != constReg.end()) return it->second;
    return rowReg.lookup(v);
  };
  for (int64_t s = 0; s < n; s += 32) {
    setVL(uint32_t(std::min<int64_t>(32, n - s)));
    DenseMap<Value, uint32_t> vreg;
    std::array<bool, 16> used{};
    auto takeV = [&]() -> int {
      for (int r = 0; r < 16; ++r)
        if (!used[r]) {
          used[r] = true;
          return r;
        }
      return -1;
    };
    auto regOf = [&](Value v) { return vreg.lookup(rootOfV(v)); };
    unsigned nArgs = reduction ? nIn : nIn + 1;
    for (unsigned i = 0; i < nArgs; ++i) {
      BlockArgument a = body.getArgument(i);
      if ((i < nIn && !vec[i]) || !lastUse.count(a)) continue;
      int r = takeV();
      if (r < 0) {
        err = "out of vector registers";
        return failure();
      }
      vload(uint32_t(r), base[i] + uint64_t(s) * 2, 1);
      vreg[a] = uint32_t(r);
    }
    auto release = [&](int index) {
      for (auto &[v, r] : vreg)
        if (lastUse.lookup(v) == index) used[r] = false;
    };
    for (auto [index, op] : llvm::enumerate(vops)) {
      Value res = op->getResult(0);
      if (aliasOf.count(res)) continue;
      int r = takeV();
      if (r < 0) {
        err = "out of vector registers";
        return failure();
      }
      uint32_t d = uint32_t(r);
      auto vk = [&](Value v) { return kind.lookup(v) == K::Vector; };
      Value a0 = op->getNumOperands() > 0 ? op->getOperand(0) : Value();
      Value a1 = op->getNumOperands() > 1 ? op->getOperand(1) : Value();
      auto vv = [&](uint32_t f) { emit(rform(OP_V_BIN, d, regOf(a0), regOf(a1), 0, f)); };
      auto vs = [&](Value v, Value sc, uint32_t f) { emit(rform(OP_V_SCALAR, d, regOf(v), sreg(sc), 0, f)); };
      if (isa<arith::AddFOp, arith::MulFOp, arith::MaximumFOp, arith::MaxNumFOp, arith::MinimumFOp,
              arith::MinNumFOp>(op)) {
        uint32_t f = isa<arith::AddFOp>(op) ? 1 : isa<arith::MulFOp>(op) ? 3
                     : isa<arith::MaximumFOp, arith::MaxNumFOp>(op) ? 4 : 5;
        if (vk(a0) && vk(a1)) vv(f);
        else if (vk(a0)) vs(a0, a1, f);
        else vs(a1, a0, f);
      } else if (isa<arith::SubFOp>(op)) {
        if (vk(a0) && vk(a1)) vv(2);
        else if (vk(a0)) vs(a0, a1, 2);
        else {                                       // s - v = (-v) + s
          emit(rform(OP_V_UN, d, regOf(a1), 0, 0, 1));
          emit(rform(OP_V_SCALAR, d, d, sreg(a0), 0, 1));
        }
      } else if (isa<arith::DivFOp>(op)) {
        if (vk(a0) && vk(a1)) {
          emit(rform(OP_V_UN, d, regOf(a1), 0, 0, 3));
          emit(rform(OP_V_BIN, d, regOf(a0), d, 0, 3));
        } else if (vk(a0)) {                         // v / s = v * (1 / s)
          emit(rform(OP_S_ALU, tmpReg, sreg(a1), 0, 0, 6));
          emit(rform(OP_V_SCALAR, d, regOf(a0), tmpReg, 0, 3));
        } else {                                     // s / v = rcp(v) * s
          emit(rform(OP_V_UN, d, regOf(a1), 0, 0, 3));
          emit(rform(OP_V_SCALAR, d, d, sreg(a0), 0, 3));
        }
      } else if (isa<arith::NegFOp>(op)) emit(rform(OP_V_UN, d, regOf(a0), 0, 0, 1));
      else if (isa<math::ExpOp>(op)) emit(rform(OP_V_UN, d, regOf(a0), 0, 0, 2));
      else if (isa<math::SqrtOp>(op)) emit(rform(OP_V_UN, d, regOf(a0), 0, 0, 4));
      else if (isa<math::RsqrtOp>(op)) emit(rform(OP_V_UN, d, regOf(a0), 0, 0, 5));
      else if (isa<math::AbsFOp>(op)) emit(rform(OP_V_UN, d, regOf(a0), 0, 0, 6));
      else {
        err = "unsupported vector op " + op->getName().getStringRef().str();
        return failure();
      }
      vreg[res] = d;
      release(int(index));
    }
    if (reduction) {
      if (dotA) emit(rform(OP_V_RED, tmpReg, regOf(dotA), regOf(dotB), 0, 3));
      else emit(rform(OP_V_RED, tmpReg, regOf(pre), 0, 0, redFunct));
      emit(rform(OP_S_ALU, accReg, accReg, tmpReg, 0, combFunct));
    } else {
      vstore(base[nIn] + uint64_t(s) * 2, regOf(result), 2);
    }
  }
  return success();
}

} // namespace mlir::hybrid::codegen
