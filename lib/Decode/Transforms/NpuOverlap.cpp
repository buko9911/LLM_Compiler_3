// Independent dequant/GEMV scheduling at the promoted memref boundary.
// Input copies stay in place: the subsequent GEMV may reuse their L2 source.
// Arithmetic moves after the forall in IR; memplan extends all four operands
// across that forall, and codegen interleaves bounded chunks on the home core.
#include "npu/Decode/Support/TargetSpec.h"
#include "npu/Decode/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir::hybrid {
#define GEN_PASS_DEF_HYBNPUOVERLAP
#include "npu/Decode/Transforms/Passes.h.inc"
}
using namespace mlir;
using namespace mlir::hybrid;

namespace {
bool view(Operation *o) {
  return isa<memref::SubViewOp, memref::ExpandShapeOp, memref::CollapseShapeOp,
             memref::CastOp, memref::ReinterpretCastOp>(o);
}
Value root(Value v) {
  while (Operation *o = v.getDefiningOp()) {
    if (!view(o)) break;
    v = o->getOperand(0);
  }
  return v;
}
bool pure(Operation *o) {
  return view(o) || isa<arith::ConstantOp, memref::AllocOp, affine::AffineApplyOp,
                        affine::AffineMinOp, affine::AffineMaxOp>(o);
}
struct HybNpuOverlapPass : mlir::hybrid::impl::HybNpuOverlapBase<HybNpuOverlapPass> {
  using Base::Base;
  void runOnOperation() override {
    std::string error;
    auto target = TargetSpec::load(spec, error);
    if (failed(target)) { getOperation().emitError(error); return signalPassFailure(); }
    if (!target->npu.overlapDequantGemv) return;
    int64_t id = 0;
    for (auto fn : getOperation().getOps<func::FuncOp>()) {
      auto phase = fn->getAttrOfType<StringAttr>(kPhase);
      if (!phase || phase.getValue() != "npu" || fn.empty()) continue;
      SmallVector<hyb::DequantOp> candidates;
      for (Operation &o : fn.front()) if (auto d = dyn_cast<hyb::DequantOp>(o)) candidates.push_back(d);
      int64_t count = 0;
      for (auto d : candidates) {
        if (d->getNumResults()) continue; // memref form only
        SmallVector<Value> roots;
        bool safe = true;
        for (Value v : d->getOperands()) {
          auto mt = dyn_cast<MemRefType>(v.getType());
          auto a = mt ? dyn_cast_or_null<IntegerAttr>(mt.getMemorySpace()) : IntegerAttr();
          SmallVector<int64_t> strides;
          int64_t offset;
          if (!mt || !mt.hasStaticShape() || !a || a.getInt() != 1 ||
              failed(mt.getStridesAndOffset(strides, offset)) || strides.empty() || strides.back() != 1 ||
              !root(v).getDefiningOp<memref::AllocOp>()) { safe = false; break; }
          roots.push_back(root(v));
        }
        if (!safe || roots.size() != 4) continue;
        auto acc = cast<MemRefType>(d.getAcc().getType());
        auto scale = cast<MemRefType>(d.getScale().getType());
        auto weightScale = cast<MemRefType>(d.getWscale().getType());
        auto output = cast<MemRefType>(d.getInit().getType());
        if (acc.getRank() != 2 || scale.getRank() != 2 || weightScale.getRank() != 1 || output.getRank() != 2 ||
            !acc.getElementType().isInteger(32) || !scale.getElementType().isF32() ||
            !weightScale.getElementType().isF32() || !output.getElementType().isF16()) continue;
        int64_t m = output.getDimSize(0), n = output.getDimSize(1);
        if (m <= 0 || n <= 0 || acc.getShape() != output.getShape() ||
            scale.getDimSize(0) != m || scale.getDimSize(1) != 1 || weightScale.getDimSize(0) != n) continue;
        if (llvm::is_contained(ArrayRef<Value>(roots).take_front(3), roots.back())) continue;
        Operation *next = d->getNextNode();
        while (next && pure(next)) next = next->getNextNode();
        auto f = dyn_cast_or_null<scf::ForallOp>(next);
        if (!f || f.getInductionVars().size() != 1 || f->hasAttr("hyb.overlap_id")) continue;
        int gemvs = 0;
        f.walk([&](Operation *o) {
          if (o == f || o->hasTrait<OpTrait::IsTerminator>() || pure(o)) return;
          if (auto g = dyn_cast<linalg::GenericOp>(o); g && g->hasAttr("hyb.npu_gemv")) {
            ++gemvs;
          } else if (o->getParentOfType<linalg::GenericOp>()) {
            return; // scalar body of the recognized GEMV
          } else if (!isa<scf::ForOp, memref::CopyOp, linalg::FillOp>(o)) safe = false;
          for (Value v : o->getOperands())
            if (isa<MemRefType>(v.getType()) && llvm::is_contained(roots, root(v))) safe = false;
        });
        if (!safe || gemvs != 1) continue;
        auto attr = IntegerAttr::get(IntegerType::get(&getContext(), 64), ++id);
        f->setAttr("hyb.overlap_id", attr);
        d->setAttr("hyb.overlap_id", attr);
        d->moveAfter(f);
        ++count;
      }
      fn->setAttr("hyb.overlap_pairs", IntegerAttr::get(IntegerType::get(&getContext(), 64), count));
    }
  }
};
}
