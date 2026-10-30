//===- CodegenGeneric.cpp - NPU codegen: linalg.generic vectorizer ----------===//
//
// In the legacy lowering, body values are Const (folded), Scalar (row-invariant: per-row f32 operands and
// expressions of them, computed once per row on the scalar unit) or Vector (depend on the
// innermost loop; FP16 lanes, 32 per register). extf / truncf of a vector alias its
// register (the vector unit is FP16). A reduction over the innermost loop folds each tile
// with V_REDUCE (FP16 -> FP32; a product of two vectors uses the dot mode, so x*x never
// rounds to FP16) and an FP32 scalar accumulator.
// Generics explicitly marked hyb.flash_math use the separate typed path below:
// FP32 vector state, real casts, comparisons and masked selection. Keeping the
// dispatch explicit preserves the numerical and instruction contract of old
// programs while online attention opts into the existing typed ISA.
//===----------------------------------------------------------------------===//

#include "CodegenImpl.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include <algorithm>
#include <functional>

namespace mlir::hybrid::codegen {

/// Typed vector lowering for online-attention state. This is deliberately
/// separate from the legacy FP16 vectorizer: an extf in that path is a register
/// alias, while here it must be an actual F16 -> F32 conversion. The existing
/// ISA supports every operation below; no host arithmetic or timing shortcut
/// participates in execution. Sixteen lanes fit both FP32 and FP16 values.
LogicalResult Codegen::lowerTypedGeneric(linalg::GenericOp g, std::string &err) {
  auto fail = [&](StringRef message) {
    err = ("typed generic: " + message).str();
    return failure();
  };
  const unsigned loops = g.getNumLoops(), inputs = g.getNumDpsInputs();
  if (!loops || g.getNumDpsInits() != 1) return fail("requires loops and one output");
  const unsigned inner = loops - 1;
  auto iterators = g.getIteratorTypesArray();
  auto ranges = g.getStaticLoopRanges();
  if (llvm::any_of(ranges, [](int64_t n) { return n <= 0; })) return fail("requires positive static ranges");
  for (unsigned d = 0; d < inner; ++d)
    if (iterators[d] != utils::IteratorType::parallel) return fail("only the innermost axis may reduce");
  const bool reduction = iterators[inner] == utils::IteratorType::reduction;
  SmallVector<View> views;
  SmallVector<AffineMap> maps;
  SmallVector<bool> contiguous;
  for (OpOperand &operand : g->getOpOperands()) {
    auto view = viewOf(operand.get(), err);
    if (failed(view)) return failure();
    if (!view->et.isF16() && !view->et.isF32()) return fail("memory operands must be F16 or F32");
    AffineMap map = g.getMatchingIndexingMap(&operand);
    bool vector = false;
    for (auto [axis, expression] : llvm::enumerate(map.getResults())) {
      auto dimension = dyn_cast<AffineDimExpr>(expression);
      if (!dimension && !isa<AffineConstantExpr>(expression)) return fail("requires projected dimension maps");
      if (dimension && dimension.getPosition() == inner) {
        if (axis + 1 != map.getNumResults() || view->strides.back() != 1)
          return fail("innermost vector axis must be contiguous");
        vector = true;
      }
    }
    views.push_back(*view);
    maps.push_back(map);
    contiguous.push_back(vector);
  }
  if (reduction && (contiguous.back() || !views.back().et.isF32()))
    return fail("reduction output must be an FP32 row scalar");
  if (!reduction && !contiguous.back()) return fail("parallel output must contain the innermost axis");

  Block &body = g.getRegion().front();
  BlockArgument accumulator = body.getArgument(inputs);
  Value yielded = cast<linalg::YieldOp>(body.getTerminator()).getValues()[0];
  Value pre;
  uint32_t combine = 1;
  DenseSet<Operation *> skip;
  if (reduction) {
    auto stripCasts = [&](Value value) {
      while (Operation *op = value.getDefiningOp()) {
        if (!isa<arith::ExtFOp, arith::TruncFOp>(op)) break;
        skip.insert(op);
        value = op->getOperand(0);
      }
      return value;
    };
    Value combined = stripCasts(yielded);
    Operation *op = combined.getDefiningOp();
    if (!op || !isa<arith::AddFOp, arith::MaximumFOp, arith::MaxNumFOp>(op))
      return fail("reduction requires one sum or maximum combiner");
    auto isAccumulator = [&](Value value) {
      while (Operation *cast = value.getDefiningOp()) {
        if (!isa<arith::ExtFOp, arith::TruncFOp>(cast)) break;
        value = cast->getOperand(0);
      }
      return value == accumulator;
    };
    const bool first = isAccumulator(op->getOperand(0)), second = isAccumulator(op->getOperand(1));
    if (first == second) return fail("reduction combiner must consume its accumulator once");
    stripCasts(op->getOperand(first ? 0 : 1));
    pre = op->getOperand(first ? 1 : 0);
    combine = isa<arith::AddFOp>(op) ? 1 : 4;
    skip.insert(op);
    for (Operation *user : accumulator.getUsers())
      if (!skip.contains(user)) return fail("unsupported additional accumulator use");
  }

  SmallVector<Operation *> operations;
  for (Operation &op : body.without_terminator()) {
    if (skip.contains(&op) || isa<arith::ConstantOp>(op)) continue;
    if (op.getNumResults() != 1) return fail("scalar body operation must have one result");
    operations.push_back(&op);
  }
  DenseMap<Value, int> lastUse;
  for (auto [index, op] : llvm::enumerate(operations))
    for (Value operand : op->getOperands()) lastUse[operand] = int(index);
  lastUse[reduction ? pre : yielded] = int(operations.size());

  SmallVector<int64_t> outer(ranges.begin(), ranges.begin() + inner);
  SmallVector<uint64_t> bases(views.size());
  LogicalResult status = success();
  forEachOuter(outer, 0, [&](ArrayRef<int64_t> index) {
    if (failed(status)) return;
    for (size_t i = 0; i < views.size(); ++i) {
      int64_t offset = 0;
      for (auto [axis, expression] : llvm::enumerate(maps[i].getResults())) {
        int64_t coordinate = 0;
        if (auto d = dyn_cast<AffineDimExpr>(expression))
          coordinate = d.getPosition() < inner ? index[d.getPosition()] : 0;
        else coordinate = cast<AffineConstantExpr>(expression).getValue();
        offset += coordinate * views[i].strides[axis];
      }
      bases[i] = addr(views[i], offset);
    }
    if (reduction) sload(11, bases.back());
    for (int64_t start = 0; start < ranges[inner]; start += 16) {
      const uint32_t lanes = uint32_t(std::min<int64_t>(16, ranges[inner] - start));
      setVL(lanes);
      std::array<bool, 16> used{};
      DenseMap<Value, uint32_t> registers;
      auto allocate = [&]() -> FailureOr<uint32_t> {
        for (uint32_t r = 0; r < used.size(); ++r)
          if (!used[r]) { used[r] = true; return r; }
        err = "typed generic: out of vector registers";
        return failure();
      };
      auto vector = [&](uint32_t op, uint32_t dtype, uint32_t dst,
                        uint32_t a, uint32_t b = 0, uint32_t c = 0) {
        rec(typedVector(op, dtype, dst, a, b, c, lanes));
      };
      auto typeCode = [&](Type type) { return type.isInteger(1) ? uint32_t(DT_I8) : uint32_t(dtypeOf(type)); };
      std::function<FailureOr<uint32_t>(Value)> materialize = [&](Value value) -> FailureOr<uint32_t> {
        if (auto found = registers.find(value); found != registers.end()) return found->second;
        auto reg = allocate();
        if (failed(reg)) return failure();
        if (auto arg = dyn_cast<BlockArgument>(value)) {
          if (arg.getOwner() != &body || arg.getArgNumber() >= views.size()) {
            err = "typed generic: unsupported captured block argument";
            return failure();
          }
          const unsigned n = arg.getArgNumber();
          uint32_t loadKind = views[n].et.isF32() ? 3 : 1;
          if (contiguous[n]) vload(*reg, bases[n] + uint64_t(start * views[n].eb), loadKind);
          else {
            setVL(1);
            vload(*reg, bases[n], loadKind);
            setVL(lanes);
            vector(T_BCAST, typeCode(views[n].et), *reg, *reg);
          }
        } else if (auto constant = constValue(value)) {
          uint32_t dtype = typeCode(value.getType());
          uint32_t loadKind = dtype == DT_F32 ? 3 : dtype == DT_I8 ? 5 : 1;
          vload(*reg, uint64_t(poolL1 + vectorConst(*constant, dtype)), loadKind);
        } else {
          err = "typed generic: operand was not emitted in SSA order";
          return failure();
        }
        registers[value] = *reg;
        return *reg;
      };
      auto release = [&](int instruction) {
        SmallVector<Value> dead;
        for (auto [value, reg] : registers)
          if (!lastUse.contains(value) || lastUse.lookup(value) <= instruction) {
            used[reg] = false;
            dead.push_back(value);
          }
        for (Value value : dead) registers.erase(value);
      };
      for (auto [instruction, op] : llvm::enumerate(operations)) {
        SmallVector<uint32_t> args;
        for (Value operand : op->getOperands()) {
          auto reg = materialize(operand);
          if (failed(reg)) { status = failure(); return; }
          args.push_back(*reg);
        }
        auto dst = allocate();
        if (failed(dst)) { status = failure(); return; }
        uint32_t dtype = typeCode(op->getOperand(0).getType());
        auto binary = [&](uint32_t code) { vector(code, dtype, *dst, args[0], args[1]); };
        auto unary = [&](uint32_t code) { vector(code, dtype, *dst, args[0]); };
        if (isa<arith::AddFOp>(op)) binary(T_ADD);
        else if (isa<arith::SubFOp>(op)) binary(T_SUB);
        else if (isa<arith::MulFOp>(op)) binary(T_MUL);
        else if (isa<arith::MaximumFOp, arith::MaxNumFOp>(op)) binary(T_MAX);
        else if (isa<arith::MinimumFOp, arith::MinNumFOp>(op)) binary(T_MIN);
        else if (isa<arith::DivFOp>(op)) {
          vector(T_RCP, dtype, *dst, args[1]);
          vector(T_MUL, dtype, *dst, args[0], *dst);
        } else if (isa<arith::NegFOp>(op)) unary(T_NEG);
        else if (isa<math::AbsFOp>(op)) unary(T_ABS);
        else if (isa<math::ExpOp>(op)) unary(T_EXP);
        else if (isa<math::SqrtOp>(op)) unary(T_SQRT);
        else if (isa<math::RsqrtOp>(op)) unary(T_RSQRT);
        else if (isa<arith::ExtFOp>(op)) unary(T_CAST_F32);
        else if (isa<arith::TruncFOp>(op)) unary(T_CAST_F16);
        else if (auto compare = dyn_cast<arith::CmpFOp>(op)) {
          switch (compare.getPredicate()) {
          case arith::CmpFPredicate::OEQ: binary(T_CMP_EQ); break;
          case arith::CmpFPredicate::OLT: binary(T_CMP_LT); break;
          case arith::CmpFPredicate::OLE: binary(T_CMP_LE); break;
          case arith::CmpFPredicate::OGT: vector(T_CMP_LT, dtype, *dst, args[1], args[0]); break;
          case arith::CmpFPredicate::OGE: vector(T_CMP_LE, dtype, *dst, args[1], args[0]); break;
          default: status = fail("unsupported floating-point predicate"); return;
          }
        } else if (isa<arith::SelectOp>(op)) {
          vector(T_SELECT, typeCode(op->getResult(0).getType()), *dst, args[1], args[2], args[0]);
        } else {
          err = "typed generic: unsupported operation " + op->getName().getStringRef().str();
          status = failure(); return;
        }
        registers[op->getResult(0)] = *dst;
        release(int(instruction));
      }
      auto result = materialize(reduction ? pre : yielded);
      if (failed(result)) { status = failure(); return; }
      if (reduction) {
        bool f32 = pre.getType().isF32();
        uint32_t function = combine == 1 ? (f32 ? 4 : 1) : (f32 ? 5 : 2);
        emit(rform(OP_V_RED, 3, *result, 0, 0, function));
        emit(rform(OP_S_ALU, 11, 11, 3, 0, combine));
      } else {
        vstore(bases.back() + uint64_t(start * views.back().eb), *result, views.back().et.isF32() ? 4 : 2);
      }
    }
    if (reduction) sstore(bases.back(), 11);
  });
  return status;
}

LogicalResult Codegen::lowerGeneric(linalg::GenericOp g, std::string &err) {
  if (g->hasAttr("hyb.flash_math")) return lowerTypedGeneric(g, err);
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

  // Interleave only independent output coordinates. Exact in-place maps are
  // safe; shifted/uncertain aliases and overlapping destination rows are not.
  // This is checked against the final physical views, not operand names.
  bool pipelineSafe = in.spec->npu.vectorChunkPipeline && !reduction && vec.back();
  uint64_t outputSpan = views.back().eb;
  for (int d = int(views.back().shape.size()) - 1; pipelineSafe && d >= 0; --d) {
    const View &out = views.back();
    if (out.shape[d] <= 0 || out.strides[d] <= 0 ||
        uint64_t(out.strides[d] * out.eb) < outputSpan) pipelineSafe = false;
    else outputSpan += uint64_t((out.shape[d] - 1) * out.strides[d] * out.eb);
  }
  for (unsigned i = 0; pipelineSafe && i < nIn; ++i) {
    if (body.getArgument(i).use_empty()) continue;
    const View &src = views[i], &dst = views.back();
    bool sameCoordinates = addr(src) == addr(dst) && src.shape == dst.shape &&
        src.strides == dst.strides && src.eb == dst.eb && maps[i] == maps.back();
    if (src.space == dst.space && !sameCoordinates &&
        overlaps(addr(src), addr(src) + spanBytes(src), addr(dst), addr(dst) + spanBytes(dst)))
      pipelineSafe = false;
  }

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
    // Scalar reciprocals of row-invariant divisors are immutable within this
    // row. Use dedicated scalar registers, never the reduction scratch. When
    // the finite register pool is exhausted the original per-chunk code stays.
    DenseMap<Value, uint32_t> reciprocals;
    if (in.spec->npu.vectorScalarReuse && n > 32) {
      DenseMap<Value, int64_t> uses;
      SmallVector<Value> divisors;
      for (Operation *op : vops)
        if (auto div = dyn_cast<arith::DivFOp>(op);
            div && kind.lookup(div.getLhs()) == K::Vector && kind.lookup(div.getRhs()) != K::Vector) {
          if (!uses.count(div.getRhs())) divisors.push_back(div.getRhs());
          ++uses[div.getRhs()];
        }
      for (Value divisor : divisors) {
        if (nextPool == std::size(kPool)) break;
        uint32_t source = constReg.contains(divisor) ? constReg.lookup(divisor) : rowReg.lookup(divisor);
        uint32_t destination = kPool[nextPool++];
        emit(rform(OP_S_ALU, destination, source, 0, 0, 6));
        reciprocals[divisor] = destination;
        ++vectorScalarReciprocals;
        vectorScalarReuses += uses.lookup(divisor) * ((n + 31) / 32) - 1;
      }
    }
    if (reduction) sload(accReg, base[nIn]);
    if (failed(emitTiles(body, nIn, vec, kind, vops, aliasOf, lastUse, base, n, reduction, pre, dotA, dotB,
                         redFunct, combFunct, result, constReg, rowReg, reciprocals, pipelineSafe,
                         accReg, tmpReg, err))) {
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
                                 DenseMap<Value, uint32_t> &reciprocals, bool pipelineSafe,
                                 uint32_t accReg, uint32_t tmpReg, std::string &err) {
  auto rootOfV = [&](Value v) {
    auto it = aliasOf.find(v);
    return it == aliasOf.end() ? v : it->second;
  };
  auto sreg = [&](Value v) -> uint32_t {
    if (auto it = constReg.find(v); it != constReg.end()) return it->second;
    return rowReg.lookup(v);
  };
  // Prove each chunk fits in one half of the existing 16-register file. A
  // separate register namespace makes interleaving legal without renaming
  // hardware or hidden storage. No arithmetic or reduction order is changed.
  unsigned nArgs = reduction ? nIn : nIn + 1;
  DenseSet<Value> live;
  for (unsigned i = 0; i < nArgs; ++i)
    if (!(i < nIn && !vec[i]) && lastUse.count(body.getArgument(i))) live.insert(body.getArgument(i));
  size_t peak = live.size();
  for (auto [index, op] : llvm::enumerate(vops)) {
    if (aliasOf.count(op->getResult(0))) continue;
    live.insert(op->getResult(0));
    peak = std::max(peak, size_t(live.size()));
    SmallVector<Value> expired;
    for (Value value : live)
      if (lastUse.lookup(value) == int(index)) expired.push_back(value);
    for (Value value : expired) live.erase(value);
    // A per-chunk scalar scratch reciprocal would be overwritten by the
    // other chunk before its vector user. Only immutable cached scalars may
    // be shared across the interleaved instruction sequences.
    if (auto div = dyn_cast<arith::DivFOp>(op);
        div && kind.lookup(div.getLhs()) == K::Vector && kind.lookup(div.getRhs()) != K::Vector &&
        !reciprocals.count(div.getRhs())) pipelineSafe = false;
  }
  pipelineSafe &= !reduction && peak <= 8;
  const int64_t batch = pipelineSafe ? 2 : 1;
  struct Chunk {
    uint32_t lanes;
    SmallVector<std::function<void()>> instructions;
  };
  for (int64_t first = 0; first < n; first += batch * 32) {
    SmallVector<Chunk, 2> chunks;
    const int64_t count = std::min<int64_t>(batch, (n - first + 31) / 32);
    for (int64_t chunk = 0; chunk < count; ++chunk) {
      const int64_t s = first + chunk * 32;
      chunks.push_back({uint32_t(std::min<int64_t>(32, n - s)), {}});
      auto &instructions = chunks.back().instructions;
      auto word = [&](uint32_t value) { instructions.push_back([this, value] { emit(value); }); };
      DenseMap<Value, uint32_t> vreg;
      std::array<bool, 16> used{};
      auto takeV = [&]() -> int {
        const int begin = pipelineSafe ? int(chunk) * 8 : 0;
        const int end = pipelineSafe ? begin + 8 : 16;
        for (int r = begin; r < end; ++r)
          if (!used[r]) {
            used[r] = true;
            return r;
          }
        return -1;
      };
      auto regOf = [&](Value v) { return vreg.lookup(rootOfV(v)); };
      for (unsigned i = 0; i < nArgs; ++i) {
        BlockArgument a = body.getArgument(i);
        if ((i < nIn && !vec[i]) || !lastUse.count(a)) continue;
        int r = takeV();
        if (r < 0) {
          err = "out of vector registers";
          return failure();
        }
        const uint64_t address = base[i] + uint64_t(s) * 2;
        instructions.push_back([this, r, address] { vload(uint32_t(r), address, 1); });
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
        auto vv = [&](uint32_t f) { word(rform(OP_V_BIN, d, regOf(a0), regOf(a1), 0, f)); };
        auto vs = [&](Value v, Value sc, uint32_t f) { word(rform(OP_V_SCALAR, d, regOf(v), sreg(sc), 0, f)); };
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
            word(rform(OP_V_UN, d, regOf(a1), 0, 0, 1));
            word(rform(OP_V_SCALAR, d, d, sreg(a0), 0, 1));
          }
        } else if (isa<arith::DivFOp>(op)) {
          if (vk(a0) && vk(a1)) {
            word(rform(OP_V_UN, d, regOf(a1), 0, 0, 3));
            word(rform(OP_V_BIN, d, regOf(a0), d, 0, 3));
          } else if (vk(a0)) {                         // v / s = v * (1 / s)
            if (auto inverse = reciprocals.find(a1); inverse != reciprocals.end())
              word(rform(OP_V_SCALAR, d, regOf(a0), inverse->second, 0, 3));
            else {
              word(rform(OP_S_ALU, tmpReg, sreg(a1), 0, 0, 6));
              word(rform(OP_V_SCALAR, d, regOf(a0), tmpReg, 0, 3));
            }
          } else {                                     // s / v = rcp(v) * s
            word(rform(OP_V_UN, d, regOf(a1), 0, 0, 3));
            word(rform(OP_V_SCALAR, d, d, sreg(a0), 0, 3));
          }
        } else if (isa<arith::NegFOp>(op)) word(rform(OP_V_UN, d, regOf(a0), 0, 0, 1));
        else if (isa<math::ExpOp>(op)) word(rform(OP_V_UN, d, regOf(a0), 0, 0, 2));
        else if (isa<math::SqrtOp>(op)) word(rform(OP_V_UN, d, regOf(a0), 0, 0, 4));
        else if (isa<math::RsqrtOp>(op)) word(rform(OP_V_UN, d, regOf(a0), 0, 0, 5));
        else if (isa<math::AbsFOp>(op)) word(rform(OP_V_UN, d, regOf(a0), 0, 0, 6));
        else {
          err = "unsupported vector op " + op->getName().getStringRef().str();
          return failure();
        }
        vreg[res] = d;
        release(int(index));
      }
      if (reduction) {
        if (dotA) word(rform(OP_V_RED, tmpReg, regOf(dotA), regOf(dotB), 0, 3));
        else word(rform(OP_V_RED, tmpReg, regOf(pre), 0, 0, redFunct));
        word(rform(OP_S_ALU, accReg, accReg, tmpReg, 0, combFunct));
      } else {
        const uint64_t address = base[nIn] + uint64_t(s) * 2;
        const uint32_t value = regOf(result);
        instructions.push_back([this, address, value] { vstore(address, value, 2); });
      }
    }
    if (count == 2) { ++vectorPipelineGroups; vectorPipelinedChunks += 2; }
    // Corresponding instructions in each chunk have identical dependences.
    // Alternate them while preserving the original order within each chunk.
    // Input loads precede output stores; stores retain increasing chunk order.
    for (size_t instruction = 0; instruction < chunks.front().instructions.size(); ++instruction)
      for (auto &chunk : chunks) {
        setVL(chunk.lanes);
        chunk.instructions[instruction]();
      }
  }
  return success();
}

} // namespace mlir::hybrid::codegen
