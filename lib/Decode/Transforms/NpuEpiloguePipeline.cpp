//===- NpuEpiloguePipeline.cpp - hyb-npu-epilogue-pipeline ----------------===//
//
// After hyb-npu-gemv-epilogue a core computes its GEMV column partition in a weight-streaming
// loop and dequantizes the whole partition afterwards, so the VPU work runs after the last MMA.
// This pass prepares that dequant to run tile by tile inside the loop: the backend issues the
// dequant of the previous tile's columns between the current tile's MMAs (SA and VPU are
// separate engines). Here, before memory planning, the dequant's scale copies move in front of
// the loop and the dequant and the loop get the same id; hyb-npu-memplan then keeps every
// dequant buffer live from the start of the loop. The IR still says "dequantize after the loop";
// the arithmetic and the instruction count are unchanged.
//===----------------------------------------------------------------------===//

#include "npu/Decode/Transforms/PassSupport.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::hybrid {
#define GEN_PASS_DEF_HYBNPUEPILOGUEPIPELINE
#include "npu/Decode/Transforms/Passes.h.inc"
} // namespace mlir::hybrid

using namespace mlir;
using namespace mlir::hybrid;

namespace {

/// The streamed GEMV loop of a gemv-epilogue core body and the dequant that consumes it.
struct Candidate {
  scf::ForOp loop;
  hyb::DequantOp dequant;
  SmallVector<Operation *> hoist;   // copies (and their destination allocs) the dequant reads
};

/// Whether `op` writes memory whose root is `root`.
bool writesRoot(Operation *op, Value root) {
  bool writes = false;
  op->walk([&](Operation *inner) {
    if (auto copy = dyn_cast<memref::CopyOp>(inner)) {
      writes |= viewRoot(copy.getTarget()) == root;
    } else if (auto dps = dyn_cast<DestinationStyleOpInterface>(inner)) {
      for (OpOperand &init : dps.getDpsInitsMutable())
        writes |= isa<MemRefType>(init.get().getType()) && viewRoot(init.get()) == root;
    }
  });
  return writes;
}

std::optional<Candidate> match(scf::ForallOp forall) {
  Block &body = forall.getRegion().front();
  scf::ForOp loop;
  for (Operation &op : body)
    if (auto f = dyn_cast<scf::ForOp>(op); f && f->hasAttr("hyb.stream")) {
      if (loop) return std::nullopt;     // one streamed GEMV per core body
      loop = f;
    }
  if (!loop) return std::nullopt;
  linalg::GenericOp gemv;
  loop.walk([&](linalg::GenericOp g) {
    if (g->hasAttr("hyb.npu_gemv")) gemv = g;
  });
  if (!gemv) return std::nullopt;
  Value acc = viewRoot(gemv.getDpsInits()[0]);
  if (!acc.getDefiningOp<memref::AllocOp>() || acc.getDefiningOp()->getBlock() != &body) return std::nullopt;

  // The dequant of the whole accumulator, after the loop in the same block. Between the loop
  // and the dequant only views, allocations and the copies that feed the dequant may appear.
  hyb::DequantOp dequant;
  SmallVector<Operation *> between;
  for (Operation *op = loop->getNextNode(); op; op = op->getNextNode()) {
    if (auto d = dyn_cast<hyb::DequantOp>(op)) {
      dequant = d;
      break;
    }
    between.push_back(op);
  }
  if (!dequant || viewRoot(dequant.getAcc()) != acc) return std::nullopt;
  Value scale = viewRoot(dequant.getScale()), wscale = viewRoot(dequant.getWscale());
  // The backend splits the dequant by loop tile: equal column blocks, all buffers in L1.
  auto tiles = constantTripCount(loop);
  auto init = cast<MemRefType>(dequant.getInit().getType());
  if (!tiles || *tiles < 2 || init.getRank() != 2 || init.getDimSize(1) % *tiles ||
      init.getDimSize(1) / *tiles % 16)
    return std::nullopt;
  for (Value v : dequant->getOperands())
    if (cast<MemRefType>(v.getType()).getMemorySpaceAsInt() != 1) return std::nullopt;
  Candidate c{loop, dequant, {}};
  for (Operation *op : between) {
    if (isViewOp(op) || isa<memref::AllocOp>(op) || isMemoryEffectFree(op)) continue;
    auto copy = dyn_cast<memref::CopyOp>(op);
    if (!copy) return std::nullopt;
    Value target = viewRoot(copy.getTarget());
    if (target != scale && target != wscale) return std::nullopt;
    // The copy may move before the loop only if the loop does not write its source and its
    // destination is a fresh allocation used by nothing but this copy and the dequant.
    if (writesRoot(loop, viewRoot(copy.getSource()))) return std::nullopt;
    auto alloc = target.getDefiningOp<memref::AllocOp>();
    if (!alloc || alloc->getBlock() != &body) return std::nullopt;
    for (Operation *user : alloc->getUsers())
      if (user != copy && user != dequant && !isViewOp(user)) return std::nullopt;
    c.hoist.push_back(alloc);
    c.hoist.push_back(copy);
  }
  // Every value the moved copies read must already be available before the loop.
  for (Operation *op : c.hoist)
    for (Value v : op->getOperands())
      if (Operation *def = v.getDefiningOp(); def && def->getBlock() == &body && loop->isBeforeInBlock(def) &&
                                                 !llvm::is_contained(c.hoist, def))
        return std::nullopt;
  return c;
}

struct HybNpuEpiloguePipelinePass
    : mlir::hybrid::impl::HybNpuEpiloguePipelineBase<HybNpuEpiloguePipelinePass> {
  using Base::Base;
  void runOnOperation() override {
    auto target = loadTargetSpec(getOperation(), spec);
    if (failed(target)) return signalPassFailure();
    if (!target->npu.gemvEpiloguePipeline) return;
    Builder b(&getContext());
    int64_t id = 0;
    for (func::FuncOp fn : getNpuPhases(getOperation())) {
      int64_t count = 0;
      SmallVector<scf::ForallOp> foralls;
      fn.walk([&](scf::ForallOp f) {
        if (f->hasAttr("hyb.gemv_epilogue")) foralls.push_back(f);
      });
      for (scf::ForallOp f : foralls) {
        auto c = match(f);
        if (!c) continue;
        for (Operation *op : c->hoist) op->moveBefore(c->loop);
        c->loop->setAttr("hyb.epilogue_pipeline", b.getI64IntegerAttr(id));
        c->dequant->setAttr("hyb.epilogue_pipeline", b.getI64IntegerAttr(id));
        ++id;
        ++count;
      }
      fn->setAttr("hyb.epilogue_pipelines", b.getI64IntegerAttr(count));
    }
  }
};

} // namespace
