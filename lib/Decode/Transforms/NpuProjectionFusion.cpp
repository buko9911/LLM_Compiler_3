// Same-input horizontal fusion. Concatenate output columns, preserving every
// reduction and scale; the image planner places original weights only once.
#include "npu/Decode/Support/TargetSpec.h"
#include "npu/Decode/Transforms/Passes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"

namespace mlir::hybrid {
#define GEN_PASS_DEF_HYBNPUPROJECTIONFUSION
#include "npu/Decode/Transforms/Passes.h.inc"
}
using namespace mlir;
using namespace mlir::hybrid;

namespace {
struct Projection {
  hyb::DequantOp dequant;
  tensor::CollapseShapeOp collapse;
  linalg::GenericOp gemv;
  hyb::WeightOp weight, scale;
  linalg::FillOp fill;
  int64_t m, n, k;
};

std::optional<Projection> match(hyb::DequantOp d) {
  if (d->getNumResults() != 1) return std::nullopt;
  auto c = d.getAcc().getDefiningOp<tensor::CollapseShapeOp>();
  auto g = c ? c.getSrc().getDefiningOp<linalg::GenericOp>() : linalg::GenericOp();
  if (!g || !g->hasAttr("hyb.npu_gemv") || g->getBlock() != d->getBlock() ||
      g.getNumDpsInputs() != 2 || g.getNumDpsInits() != 1 ||
      !g.getResult(0).hasOneUse() || !c.getResult().hasOneUse() ||
      c.getReassociationIndices() != SmallVector<ReassociationIndices>{{0}, {1, 2}})
    return std::nullopt;
  auto w = g.getDpsInputs()[1].getDefiningOp<hyb::WeightOp>();
  auto s = d.getWscale().getDefiningOp<hyb::WeightOp>();
  auto f = g.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
  Attribute zero;
  if (!w || !s || !f || !w.getResult().hasOneUse() || !s.getResult().hasOneUse() ||
      w.getLayout() != "npu_nk32" || s.getLayout() != "row_scale" ||
      w.getName() != s.getName() || w.getNameAttr() != d.getWeightAttr() ||
      g->getAttrOfType<StringAttr>("hyb.npu_gemv") != d.getWeightAttr() ||
      w->hasAttr("hyb.weight_segments") || s->hasAttr("hyb.weight_segments") ||
      !matchPattern(f.getInputs()[0], m_Constant(&zero)) || !isa<IntegerAttr>(zero) ||
      !cast<IntegerAttr>(zero).getValue().isZero()) return std::nullopt;
  auto q = dyn_cast<RankedTensorType>(g.getDpsInputs()[0].getType());
  auto wt = dyn_cast<RankedTensorType>(w.getType());
  auto a = dyn_cast<RankedTensorType>(g.getResult(0).getType());
  auto y = dyn_cast<RankedTensorType>(d.getInit().getType());
  auto sc = dyn_cast<RankedTensorType>(d.getScale().getType());
  auto ws = dyn_cast<RankedTensorType>(s.getType());
  if (!q || !wt || !a || !y || !sc || !ws || !q.hasStaticShape() || !wt.hasStaticShape() ||
      !y.hasStaticShape() || q.getRank() != 2 || wt.getRank() != 3 || y.getRank() != 2 ||
      !q.getElementType().isInteger(8) || !wt.getElementType().isInteger(8) ||
      !a.getElementType().isInteger(32) || !y.getElementType().isF16() ||
      !sc.getElementType().isF32() || !ws.getElementType().isF32()) return std::nullopt;
  int64_t m = q.getDimSize(0), k = q.getDimSize(1), n = y.getDimSize(1);
  if (m <= 0 || k <= 0 || n <= 0 || n % 32 ||
      wt.getShape() != ArrayRef<int64_t>({n/32,k,32}) ||
      a.getShape() != ArrayRef<int64_t>({m,n/32,32}) ||
      y.getShape() != ArrayRef<int64_t>({m,n}) ||
      sc.getShape() != ArrayRef<int64_t>({m,1}) || ws.getShape() != ArrayRef<int64_t>({n}))
    return std::nullopt;
  // Require the canonical contraction, including its scalar payload. Cloning
  // an arbitrary annotated generic would not prove identical arithmetic.
  AffineExpr x,b,col,r;
  bindDims(g.getContext(), x,b,col,r);
  SmallVector<AffineMap> maps = {AffineMap::get(4,0,{x,r},g.getContext()),
      AffineMap::get(4,0,{b,r,col},g.getContext()), AffineMap::get(4,0,{x,b,col},g.getContext())};
  if (g.getIndexingMapsArray() != maps || g.getIteratorTypesArray() !=
      SmallVector<utils::IteratorType>{utils::IteratorType::parallel,utils::IteratorType::parallel,
          utils::IteratorType::parallel,utils::IteratorType::reduction}) return std::nullopt;
  auto &body = g.getRegion().front();
  if (body.getOperations().size() != 5) return std::nullopt;
  auto it = body.begin();
  auto ex = dyn_cast<arith::ExtSIOp>(&*it++), ew = dyn_cast<arith::ExtSIOp>(&*it++);
  auto mul = dyn_cast<arith::MulIOp>(&*it++);
  auto add = dyn_cast<arith::AddIOp>(&*it++);
  auto yield = dyn_cast<linalg::YieldOp>(&*it);
  if (!ex || !ew || !mul || !add || !yield || ex.getIn() != body.getArgument(0) ||
      ew.getIn() != body.getArgument(1) || mul.getLhs() != ex || mul.getRhs() != ew ||
      add.getLhs() != body.getArgument(2) || add.getRhs() != mul || yield.getOperand(0) != add)
    return std::nullopt;
  return Projection{d,c,g,w,s,f,m,n,k};
}

struct HybNpuProjectionFusionPass : mlir::hybrid::impl::HybNpuProjectionFusionBase<HybNpuProjectionFusionPass> {
  using Base::Base;
  void runOnOperation() override {
    std::string error;
    auto target = TargetSpec::load(spec,error);
    if (failed(target)) { getOperation().emitError(error); return signalPassFailure(); }
    if (!target->npu.projectionFusion) return;
    // A fixed PIM image is a different contract; never repack its weights.
    if (getOperation()->hasAttr("hyb.pim_weight_storage_requested")) return;
    for (auto fn : getOperation().getOps<func::FuncOp>()) {
      auto phase = fn->getAttrOfType<StringAttr>(kPhase);
      if (!phase || phase.getValue() != "npu" || fn.empty() || fn->hasAttr("hyb.npu_kernel")) continue;
      SmallVector<SmallVector<Projection>> groups;
      for (Operation &op : fn.front()) {
        auto d = dyn_cast<hyb::DequantOp>(op);
        auto p = d ? match(d) : std::nullopt;
        if (!p) continue;
        auto group = llvm::find_if(groups,[&](auto &g) {
          return g.front().gemv.getDpsInputs()[0] == p->gemv.getDpsInputs()[0] &&
                 g.front().dequant.getScale() == d.getScale();
        });
        if (group == groups.end()) groups.push_back({*p});
        else group->push_back(*p);
      }
      int64_t count = 0;
      for (auto &group : groups) if (group.size() > 1 && fuse(group,*target)) ++count;
      fn->setAttr("hyb.projection_fusions", IntegerAttr::get(IntegerType::get(&getContext(),64),count));
    }
  }

  bool fuse(SmallVector<Projection> &ps, const TargetSpec &target) {
    // Place before the earliest contraction so even early Q-norm users remain
    // dominated. All new runtime inputs must already be available there.
    llvm::sort(ps,[](auto &a,auto &b) { return a.gemv->isBeforeInBlock(b.gemv); });
    auto &first = ps.front();
    DominanceInfo dom;
    if (!dom.dominates(first.dequant.getScale(),first.gemv)) return false;
    int64_t n = 0;
    for (auto &p : ps) n += p.n;
    // Bound private accumulator/output growth before authoritative memplan.
    int64_t cores = std::min<int64_t>(n/32,target.npu.cores);
    while (cores > 1 && (n/32) % cores) --cores;
    int64_t local = first.m * (n/cores) * 6 + first.m*first.k + n/cores*4 +
                    2*std::max<int64_t>(target.npu.gemvTileBytes,first.k*32) +
                    target.npu.l1ReservedBytes + 4096;
    if (local > target.npu.l1Bytes || first.m*n*6 > target.npu.l2Bytes) return false;
    OpBuilder b(first.gemv);
    Location loc = first.gemv.getLoc();
    SmallVector<Attribute> segments;
    std::string name = "fused[";
    for (auto &p : ps) {
      if (segments.size()) name += ";";
      name += p.weight.getName().str();
      segments.push_back(b.getDictionaryAttr({b.getNamedAttr("name",p.weight.getNameAttr()),
                                              b.getNamedAttr("rows",b.getI64IntegerAttr(p.n))}));
    }
    name += "]";
    auto segmentAttr = b.getArrayAttr(segments);
    auto wt = RankedTensorType::get({n/32,first.k,32},b.getI8Type());
    auto w = hyb::WeightOp::create(b,loc,wt,b.getStringAttr(name),b.getStringAttr("npu_nk32"));
    auto s = hyb::WeightOp::create(b,loc,RankedTensorType::get({n},b.getF32Type()),
                                 b.getStringAttr(name),b.getStringAttr("row_scale"));
    w->setAttr("hyb.weight_segments",segmentAttr);
    s->setAttr("hyb.weight_segments",segmentAttr);
    Value e = tensor::EmptyOp::create(b,loc,ArrayRef<int64_t>{first.m,n/32,32},b.getI32Type());
    Value z = arith::ConstantOp::create(b,loc,b.getI32IntegerAttr(0));
    Value init = linalg::FillOp::create(b,loc,z,e).getResult(0);
    IRMapping mapping;
    mapping.map(first.weight.getResult(),w.getResult());
    mapping.map(first.gemv.getDpsInits()[0],init);
    auto g = cast<linalg::GenericOp>(b.clone(*first.gemv,mapping));
    g.getResult(0).setType(init.getType());
    g->setAttr("hyb.npu_gemv",b.getStringAttr(name));
    Value acc = tensor::CollapseShapeOp::create(b,loc,RankedTensorType::get({first.m,n},b.getI32Type()),
        g.getResult(0),SmallVector<ReassociationIndices>{{0},{1,2}});
    Value out = tensor::EmptyOp::create(b,loc,ArrayRef<int64_t>{first.m,n},b.getF16Type());
    auto d = hyb::DequantOp::create(b,loc,TypeRange{out.getType()},acc,first.dequant.getScale(),s,out,b.getStringAttr(name));
    int64_t offset = 0;
    for (auto &p : ps) {
      Value slice = tensor::ExtractSliceOp::create(b,loc,d.getResult(0),
          SmallVector<OpFoldResult>{b.getIndexAttr(0),b.getIndexAttr(offset)},
          SmallVector<OpFoldResult>{b.getIndexAttr(p.m),b.getIndexAttr(p.n)},
          SmallVector<OpFoldResult>{b.getIndexAttr(1),b.getIndexAttr(1)});
      p.dequant.getResult(0).replaceAllUsesWith(slice);
      offset += p.n;
    }
    for (auto &p : ps) {
      p.dequant.erase(); p.collapse.erase(); p.gemv.erase();
    }
    return true;
  }
};
}
