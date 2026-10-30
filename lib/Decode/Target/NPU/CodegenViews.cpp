//===- CodegenViews.cpp - NPU codegen: index evaluation and views -----------===//
//
// Codegen members that evaluate index values (loop induction variables are known while loops
// are unrolled) and resolve memref values to (space, base, offset, strides) views.
//===----------------------------------------------------------------------===//

#include "CodegenImpl.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/SmallBitVector.h"

namespace mlir::hybrid::codegen {

namespace {

int64_t elemBytes(Type t) { return std::max<int64_t>(1, int64_t(t.getIntOrFloatBitWidth()) / 8); }

} // namespace

// ---- index evaluation and views --------------------------------------------------------
FailureOr<int64_t> Codegen::evalIndex(Value v, std::string &err) {
  if (auto it = env.find(v); it != env.end()) return it->second;
  if (auto c = getConstantIntValue(v)) return *c;
  Operation *d = v.getDefiningOp();
  if (!d) {
    err = "index value without a known value";
    return failure();
  }
  auto bin = [&](auto fn) -> FailureOr<int64_t> {
    auto a = evalIndex(d->getOperand(0), err), b = evalIndex(d->getOperand(1), err);
    if (failed(a) || failed(b)) return failure();
    return fn(*a, *b);
  };
  if (isa<arith::AddIOp>(d)) return bin([](int64_t a, int64_t b) { return a + b; });
  if (isa<arith::SubIOp>(d)) return bin([](int64_t a, int64_t b) { return a - b; });
  if (isa<arith::MulIOp>(d)) return bin([](int64_t a, int64_t b) { return a * b; });
  if (isa<arith::DivSIOp, arith::DivUIOp, arith::FloorDivSIOp>(d)) return bin([](int64_t a, int64_t b) { return a / b; });
  if (isa<arith::CeilDivSIOp>(d)) return bin([](int64_t a, int64_t b) { return (a + b - 1) / b; });
  if (isa<arith::RemSIOp, arith::RemUIOp>(d)) return bin([](int64_t a, int64_t b) { return a % b; });
  if (isa<arith::MinSIOp>(d)) return bin([](int64_t a, int64_t b) { return std::min(a, b); });
  if (isa<arith::MaxSIOp>(d)) return bin([](int64_t a, int64_t b) { return std::max(a, b); });
  if (isa<arith::IndexCastOp>(d)) return evalIndex(d->getOperand(0), err);
  auto evalMap = [&](AffineMap map, ValueRange operands) -> FailureOr<SmallVector<int64_t>> {
    SmallVector<int64_t> vals;
    for (Value o : operands) {
      auto x = evalIndex(o, err);
      if (failed(x)) return failure();
      vals.push_back(*x);
    }
    SmallVector<int64_t> out;
    for (AffineExpr e : map.getResults()) {
      auto r = evalAffine(e, ArrayRef<int64_t>(vals).take_front(map.getNumDims()),
                          ArrayRef<int64_t>(vals).drop_front(map.getNumDims()));
      if (!r) {
        err = "cannot evaluate affine expression";
        return failure();
      }
      out.push_back(*r);
    }
    return out;
  };
  if (auto a = dyn_cast<affine::AffineApplyOp>(d)) {
    auto r = evalMap(a.getAffineMap(), a.getMapOperands());
    if (failed(r)) return failure();
    return (*r)[0];
  }
  if (auto a = dyn_cast<affine::AffineMinOp>(d)) {
    auto r = evalMap(a.getAffineMap(), a.getMapOperands());
    if (failed(r)) return failure();
    return *std::min_element(r->begin(), r->end());
  }
  if (auto a = dyn_cast<affine::AffineMaxOp>(d)) {
    auto r = evalMap(a.getAffineMap(), a.getMapOperands());
    if (failed(r)) return failure();
    return *std::max_element(r->begin(), r->end());
  }
  err = "unsupported index op " + d->getName().getStringRef().str();
  return failure();
}

std::optional<int64_t> Codegen::evalAffine(AffineExpr e, ArrayRef<int64_t> dims, ArrayRef<int64_t> syms) {
  if (auto c = dyn_cast<AffineConstantExpr>(e)) return c.getValue();
  if (auto d = dyn_cast<AffineDimExpr>(e)) return dims[d.getPosition()];
  if (auto s = dyn_cast<AffineSymbolExpr>(e)) return syms[s.getPosition()];
  auto b = dyn_cast<AffineBinaryOpExpr>(e);
  if (!b) return std::nullopt;
  auto l = evalAffine(b.getLHS(), dims, syms), r = evalAffine(b.getRHS(), dims, syms);
  if (!l || !r) return std::nullopt;
  switch (e.getKind()) {
  case AffineExprKind::Add: return *l + *r;
  case AffineExprKind::Mul: return *l * *r;
  case AffineExprKind::Mod: return ((*l % *r) + *r) % *r;
  case AffineExprKind::FloorDiv: return *l >= 0 ? *l / *r : -((-*l + *r - 1) / *r);
  case AffineExprKind::CeilDiv: return (*l + *r - 1) / *r;
  default: return std::nullopt;
  }
}

/// Iteration index of the innermost scf.for around `op` (double-buffer slot selection).
int64_t Codegen::iterationOf(Operation *op) {
  auto f = op->getParentOfType<scf::ForOp>();
  if (!f) return 0;
  auto it = env.find(f.getInductionVar());
  auto b = loopBounds.find(f);
  if (it == env.end() || b == loopBounds.end()) return 0;
  return (it->second - b->second.first) / b->second.second;
}

FailureOr<View> Codegen::viewOf(Value v, std::string &err) {
  auto mt = dyn_cast<MemRefType>(v.getType());
  if (!mt) {
    err = "not a memref";
    return failure();
  }
  View r;
  r.et = mt.getElementType();
  r.eb = elemBytes(r.et);
  r.space = spaceOf(mt);
  auto identity = [&](View &x) {
    x.shape.assign(mt.getShape().begin(), mt.getShape().end());
    x.strides.assign(x.shape.size(), 1);
    for (int i = int(x.shape.size()) - 2; i >= 0; --i) x.strides[i] = x.strides[i + 1] * x.shape[i + 1];
  };
  if (auto ba = dyn_cast<BlockArgument>(v)) {
    if (ba.getOwner()->getParentOp() != func::FuncOp(in.phase).getOperation()) {
      err = "memref block argument of a nested region";
      return failure();
    }
    r.lp6 = in.argLP6[ba.getArgNumber()];
    r.layout = r.lp6.getAs<StringAttr>("layout").getValue().str();
    if (r.layout == "dense") r.base = uint64_t(r.lp6.getAs<IntegerAttr>("addr").getInt());
    identity(r);
    return r;
  }
  Operation *d = v.getDefiningOp();
  if (auto a = dyn_cast<memref::AllocOp>(d)) {
    // Planned allocations have dense physical storage. Non-identity layout
    // roots must not silently acquire dense strides: that would invalidate
    // both address generation and every downstream alias/lifetime proof.
    if (!mt.getLayout().isIdentity()) {
      err = "non-identity allocation layout is unsupported; use an identity allocation with explicit views";
      return failure();
    }
    auto off = a->getAttrOfType<IntegerAttr>("hyb.offset");
    if (!off || (r.space != kL1 && r.space != kL2)) {
      err = "buffer without an L1 / L2 offset";
      return failure();
    }
    r.base = uint64_t(off.getInt());
    if (auto n = a->getAttrOfType<IntegerAttr>("hyb.buffers"))
      r.base += uint64_t(iterationOf(a) % n.getInt()) * uint64_t(a->getAttrOfType<IntegerAttr>("hyb.slot_bytes").getInt());
    identity(r);
    return r;
  }
  if (auto wop = dyn_cast<hyb::WeightOp>(d)) {
    auto p = wop->getAttrOfType<DictionaryAttr>("hyb.lp6");
    if (!p) {
      err = "weight without an LP6 address";
      return failure();
    }
    r.base = uint64_t(p.getAs<IntegerAttr>("addr").getInt());
    identity(r);
    return r;
  }
  if (auto sv = dyn_cast<memref::SubViewOp>(d)) {
    FailureOr<View> s = viewOf(sv.getSource(), err);
    if (failed(s)) return failure();
    View x = *s;
    SmallVector<int64_t> shape, strides;
    int64_t off = s->offset;
    auto offs = sv.getMixedOffsets(), sizes = sv.getMixedSizes(), steps = sv.getMixedStrides();
    llvm::SmallBitVector dropped = sv.getDroppedDims();
    for (size_t i = 0; i < offs.size(); ++i) {
      auto value = [&](OpFoldResult f) -> FailureOr<int64_t> {
        if (auto a = dyn_cast<Attribute>(f)) return cast<IntegerAttr>(a).getInt();
        return evalIndex(cast<Value>(f), err);
      };
      auto o = value(offs[i]), n = value(sizes[i]), st = value(steps[i]);
      if (failed(o) || failed(n) || failed(st)) return failure();
      off += *o * s->strides[i];
      if (dropped.test(i)) continue;
      shape.push_back(*n);
      strides.push_back(s->strides[i] * *st);
    }
    x.offset = off;
    x.shape = shape;
    x.strides = strides;
    return x;
  }
  if (auto ex = dyn_cast<memref::ExpandShapeOp>(d)) {
    FailureOr<View> s = viewOf(ex.getSrc(), err);
    if (failed(s)) return failure();
    View x = *s;
    x.shape.assign(mt.getShape().begin(), mt.getShape().end());
    x.strides.assign(x.shape.size(), 0);
    for (auto [i, group] : llvm::enumerate(ex.getReassociationIndices())) {
      int64_t stride = s->strides[i];
      for (int j = int(group.size()) - 1; j >= 0; --j) {
        x.strides[group[j]] = stride;
        stride *= x.shape[group[j]];
      }
    }
    return x;
  }
  if (auto co = dyn_cast<memref::CollapseShapeOp>(d)) {
    FailureOr<View> s = viewOf(co.getSrc(), err);
    if (failed(s)) return failure();
    View x = *s;
    x.shape.clear();
    x.strides.clear();
    for (auto group : co.getReassociationIndices()) {
      int64_t size = 1, stride = -1, prevStride = 0, prevSize = 0;
      for (int j = int(group.size()) - 1; j >= 0; --j) {
        int64_t dim = group[j];
        if (s->shape[dim] != 1) {
          if (stride < 0) stride = s->strides[dim];
          else if (s->strides[dim] != prevStride * prevSize) {
            err = "collapse of non-contiguous dims";
            return failure();
          }
          prevStride = s->strides[dim];
          prevSize = s->shape[dim];
        }
        size *= s->shape[dim];
      }
      if (stride < 0) stride = group.empty() ? 1 : s->strides[group.back()];
      x.shape.push_back(size);
      x.strides.push_back(stride);
    }
    return x;
  }
  if (auto c = dyn_cast<memref::CastOp>(d)) return viewOf(c.getSource(), err);
  err = "unsupported memref producer " + d->getName().getStringRef().str();
  return failure();
}

} // namespace mlir::hybrid::codegen
