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
#include "npu/Decode/Transforms/NpuCost.h"
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

std::optional<Candidate> match(scf::ForallOp forall, scf::ForOp loop) {
  Block &body = forall.getRegion().front();
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
    // Other dequants of the body (the second GEMV of a gate/up pair) may read the same scale copy.
    for (Operation *user : alloc->getUsers())
      if (user != copy && user != dequant && !isViewOp(user) && !isa<hyb::DequantOp>(user)) return std::nullopt;
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

/// The all-parallel identity-map 2-D generic right after the second dequant that reads both
/// dequant outputs (and nothing else; an output may be read twice) and writes a fresh L1 buffer
/// of the same shape: it can run on each finished tile's rows and columns.
linalg::GenericOp pairConsumer(hyb::DequantOp first, hyb::DequantOp second) {
  auto g = dyn_cast_or_null<linalg::GenericOp>(second->getNextNode());
  if (!g || g.getNumDpsInputs() < 2 || g.getNumDpsInits() != 1 || g.getNumLoops() != 2 ||
      g.getNumReductionLoops() || !llvm::all_of(g.getIndexingMapsArray(), [](AffineMap m) { return m.isIdentity(); }))
    return {};
  Value a = viewRoot(first.getInit()), b = viewRoot(second.getInit());
  bool readsA = false, readsB = false;
  for (Value in : g.getDpsInputs()) {
    Value root = viewRoot(in);
    if (root != a && root != b) return {};
    (root == a ? readsA : readsB) = true;
  }
  if (!readsA || !readsB) return {};
  auto out = g.getDpsInits()[0];
  auto shape = cast<MemRefType>(out.getType()).getShape();
  for (Value v : g->getOperands())
    if (cast<MemRefType>(v.getType()).getMemorySpaceAsInt() != 1 ||
        cast<MemRefType>(v.getType()).getShape() != shape)
      return {};
  if (cast<MemRefType>(second.getInit().getType()).getShape() != shape) return {};
  auto alloc = viewRoot(out).getDefiningOp<memref::AllocOp>();
  if (!alloc || alloc->getBlock() != g->getBlock()) return {};
  return g;
}

/// Whether the gate/up consumer should run tile by tile inside the up GEMV loop. A core issues its
/// instructions in order, so VPU work placed between two tiles delays the next tile's operand load
/// and MMA. Per tile (NpuCost.h): the weight stream time of the tile against the SA time (operand
/// loads + MMA chain) plus the dequant already placed there. When the stream is longer (the loop
/// waits for weights), the consumer goes inside: up to the slack it is hidden, beyond it it costs
/// no more than after the loop. When the SA side is longer, the loop is compute bound and the
/// consumer stays after it (inside, every tile would wait for it). Measured on Qwen3-4B one-layer
/// fixtures: inside is faster at M = 8 and 16, after at M = 32 and 64.
bool consumerFits(func::FuncOp fn, scf::ForOp loop, linalg::GenericOp consumer, int64_t cores,
                  const TargetSpec &target) {
  linalg::GenericOp gemv;
  loop.walk([&](linalg::GenericOp g) {
    if (g->hasAttr("hyb.npu_gemv")) gemv = g;
  });
  if (!gemv) return false;
  SmallVector<int64_t> dom = gemv.getStaticLoopRanges();      // (m, blocks, 32, k) of one tile
  const int64_t m = dom[0], columns = dom[1] * dom[2], k = dom[3];
  const auto &npu = target.npu;
  const double stream = cost::streamCycles(npu, columns * k, cores);
  const double compute = cost::gemvTileCycles(npu, m, k, columns);
  const double dequant = double(m * columns) * cost::dequantCyclesPerElement(npu);
  const double work = double(m * columns) * cost::vectorCyclesPerElement(npu, consumer);
  const double slack = stream - compute - dequant;
  bool inside = slack >= 0;
  std::string label;
  llvm::raw_string_ostream os(label);
  os << "gate/up consumer, tile " << m << "x" << columns << " k " << k;
  const double us = cost::cyclesPerUs(npu);
  cost::recordDecision(fn, "consumer per tile", label, inside ? "inside the GEMV loop" : "after the loop",
                       {{"stream_us", stream / us}, {"sa_us", compute / us}, {"dequant_us", dequant / us},
                        {"consumer_us", work / us}, {"slack_us", slack / us}});
  return inside;
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
      int64_t count = 0, consumers = 0;
      SmallVector<scf::ForallOp> foralls;
      fn.walk([&](scf::ForallOp f) {
        if (f->hasAttr("hyb.gemv_epilogue")) foralls.push_back(f);
      });
      for (scf::ForallOp f : foralls) {
        // Every streamed GEMV of the core body (two for a gate/up pair) with the dequant after it.
        SmallVector<scf::ForOp> loops;
        for (Operation &op : f.getRegion().front())
          if (auto loop = dyn_cast<scf::ForOp>(op); loop && loop->hasAttr("hyb.stream")) loops.push_back(loop);
        if (loops.size() > 1 && !f->hasAttr("hyb.gemv_pair")) continue;
        SmallVector<hyb::DequantOp> pipelined;
        for (scf::ForOp loop : loops) {
          auto c = match(f, loop);
          if (!c) continue;
          for (Operation *op : c->hoist) op->moveBefore(c->loop);
          c->loop->setAttr("hyb.epilogue_pipeline", b.getI64IntegerAttr(id));
          c->dequant->setAttr("hyb.epilogue_pipeline", b.getI64IntegerAttr(id));
          pipelined.push_back(c->dequant);
          ++id;
          ++count;
        }
        // Gate/up pair: the op on both halves runs tile by tile behind the second GEMV as well.
        int64_t cores = f->hasAttr("hyb.cores") ? f->getAttrOfType<IntegerAttr>("hyb.cores").getInt() : 1;
        if (target->npu.pairConsumerPipeline && f->hasAttr("hyb.gemv_pair") && pipelined.size() == 2)
          if (auto consumer = pairConsumer(pipelined[0], pipelined[1]);
              consumer && consumerFits(fn, loops.back(), consumer, cores, *target)) {
            consumer->setAttr("hyb.epilogue_consumer", pipelined[1]->getAttr("hyb.epilogue_pipeline"));
            ++consumers;
          }
      }
      fn->setAttr("hyb.epilogue_pipelines", b.getI64IntegerAttr(count));
      if (consumers) fn->setAttr("hyb.epilogue_consumers", b.getI64IntegerAttr(consumers));
    }
  }
};

} // namespace
