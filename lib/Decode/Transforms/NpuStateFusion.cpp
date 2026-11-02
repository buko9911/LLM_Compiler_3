//===- NpuStateFusion.cpp - hyb-npu-state-fusion ---------------------------===//
//
// A recurrent state, an FP32 phase argument such as a Gated DeltaNet layer's [heads, dv, dk]
// state, is updated token by token by elementwise and row-reduction generics that are all
// parallel along its leading (head) axis: the decay, the key read S k, the rank-1 update and the
// query read S q of every token. Left alone, each of them is a separate op over the whole state
// (the whole state in one core's L1, or a round trip through L2 between ops). This pass moves the
// chain, from the state argument to the ops that read the new state, into one scf.forall over the
// cores. Each core owns a block of heads: its state slice is read once, stays in its L1 for every
// token and is written once; the per-token inputs (decay, beta, q, k, v rows) are sliced to its
// heads. The scalar bodies, element types and reduction order are unchanged.
//===----------------------------------------------------------------------===//

#include "npu/Decode/Transforms/NpuCost.h"
#include "npu/Decode/Transforms/PassSupport.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::hybrid {
#define GEN_PASS_DEF_HYBNPUSTATEFUSION
#include "npu/Decode/Transforms/Passes.h.inc"
} // namespace mlir::hybrid

using namespace mlir;
using namespace mlir::hybrid;

namespace {

/// The result position at which `map` reads loop 0 as a plain dimension.
std::optional<unsigned> headPosition(AffineMap map) {
  for (auto [i, e] : llvm::enumerate(map.getResults()))
    if (auto d = dyn_cast<AffineDimExpr>(e); d && d.getPosition() == 0) return unsigned(i);
  return std::nullopt;
}

/// A generic is part of the chain when it is parallel along loop 0 with the state's head count,
/// reads every chain value and writes its result with loop 0 as dimension 0, and reads its other
/// operands either along loop 0 as a plain dimension (sliced per core) or not at all (shared).
bool eligible(linalg::GenericOp g, const DenseSet<Value> &chain, int64_t heads) {
  if (!g.hasPureTensorSemantics() || g.getNumDpsInits() != 1 || g->getNumResults() != 1) return false;
  if (g.getNumLoops() == 0 || g.getIteratorTypesArray()[0] != utils::IteratorType::parallel ||
      g.getStaticLoopRanges()[0] != heads)
    return false;
  if (headPosition(g.getMatchingIndexingMap(g.getDpsInitOperand(0))) != 0u) return false;
  for (OpOperand *in : g.getDpsInputOperands()) {
    AffineMap map = g.getMatchingIndexingMap(in);
    if (chain.contains(in->get()) ? headPosition(map) != 0u : map.isFunctionOfDim(0) && !headPosition(map))
      return false;
  }
  return llvm::none_of(g.getRegion().front(), [](Operation &op) { return isa<linalg::IndexOp>(op); });
}

/// The generics reachable from `state` through eligible users, in program order.
SmallVector<Operation *> collect(Value state, int64_t heads) {
  DenseSet<Value> chain{state};
  llvm::SetVector<Operation *> ops;
  SmallVector<Value> work{state};
  Block *block = state.getParentBlock();
  while (!work.empty()) {
    Value v = work.pop_back_val();
    for (Operation *user : v.getUsers()) {
      auto g = dyn_cast<linalg::GenericOp>(user);
      if (!g || g->getBlock() != block || ops.contains(g) || !eligible(g, chain, heads)) continue;
      ops.insert(g);
      chain.insert(g->getResult(0));
      work.push_back(g->getResult(0));
    }
  }
  SmallVector<Operation *> ordered(ops.begin(), ops.end());
  llvm::sort(ordered, [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
  return ordered;
}

struct HybNpuStateFusionPass : mlir::hybrid::impl::HybNpuStateFusionBase<HybNpuStateFusionPass> {
  using Base::Base;

  void runOnOperation() override {
    auto target = loadTargetSpec(getOperation(), spec);
    if (failed(target)) return signalPassFailure();
    for (func::FuncOp fn : getNpuPhases(getOperation())) {
      int64_t fused = 0;
      for (BlockArgument state : fn.getArguments()) {
        auto type = dyn_cast<RankedTensorType>(state.getType());
        if (!type || type.getRank() < 2 || !type.hasStaticShape() || !type.getElementType().isF32()) continue;
        SmallVector<Operation *> ops = collect(state, type.getDimSize(0));
        if (ops.empty()) continue;
        fused += fuse(fn, state, ops, *target);
      }
      if (fused) fn->setAttr("hyb.state_fusions", IntegerAttr::get(IntegerType::get(&getContext(), 64), fused));
    }
  }

  bool fuse(func::FuncOp fn, Value state, ArrayRef<Operation *> ops, const TargetSpec &target) {
    const int64_t heads = cast<RankedTensorType>(state.getType()).getDimSize(0);
    const int64_t cores = largestDivisorAtMost(heads, target.npu.cores), block = heads / std::max<int64_t>(cores, 1);
    llvm::SmallPtrSet<Operation *, 32> inChain(ops.begin(), ops.end());
    // The chain's values read after it: the new state and the per-token reads.
    SmallVector<Value> outputs;
    Operation *last = ops.back();
    for (Operation *op : ops)
      for (Operation *user : op->getResult(0).getUsers())
        if (!inChain.contains(user) && !llvm::is_contained(outputs, op->getResult(0)))
          outputs.push_back(op->getResult(0));
    // Readers of chain values that sit between chain ops (e.g. a view of a per-token read) move
    // after the forall, with every op before it that uses their results. None of them may feed the
    // chain back.
    llvm::SetVector<Operation *> moved;
    SmallVector<Operation *> work;
    for (Value v : outputs)
      for (Operation *user : v.getUsers())
        if (!inChain.contains(user) && user->getBlock() == last->getBlock() && user->isBeforeInBlock(last))
          work.push_back(user);
    while (!work.empty()) {
      Operation *op = work.pop_back_val();
      if (!moved.insert(op)) continue;
      for (Value r : op->getResults())
        for (Operation *user : r.getUsers()) {
          if (inChain.contains(user)) return false;
          if (user->getBlock() == last->getBlock() && user->isBeforeInBlock(last)) work.push_back(user);
        }
    }
    std::string label;
    llvm::raw_string_ostream os(label);
    os << "state " << state.getType() << ", " << ops.size() << " ops";
    const double sliceKiB = double(cast<RankedTensorType>(state.getType()).getNumElements() / heads * block * 4) / 1024.0;
    cost::recordDecision(fn, "fuse state chain", label, cores > 1 ? "head blocks over cores" : "one core",
                         {{"heads", double(heads)}, {"cores", double(cores)}, {"slice_kib", sliceKiB},
                          {"outputs", double(outputs.size())}});
    if (cores < 2 || outputs.empty()) return false;

    OpBuilder b(last->getContext());
    b.setInsertionPointAfter(last);
    Location loc = last->getLoc();
    SmallVector<Value> inits;
    for (Value v : outputs) {
      auto t = cast<RankedTensorType>(v.getType());
      inits.push_back(tensor::EmptyOp::create(b, loc, t.getShape(), t.getElementType()));
    }
    auto forall = scf::ForallOp::create(b, loc, ArrayRef<OpFoldResult>{b.getIndexAttr(cores)}, inits, std::nullopt);
    b.setInsertionPoint(forall.getTerminator());
    Value offset = arith::MulIOp::create(b, loc, forall.getInductionVar(0),
                                         arith::ConstantIndexOp::create(b, loc, block));
    auto sliceArgs = [&](RankedTensorType t, unsigned dim) {
      SmallVector<OpFoldResult> offsets(t.getRank(), b.getIndexAttr(0)), sizes, strides(t.getRank(), b.getIndexAttr(1));
      for (int64_t d : t.getShape()) sizes.push_back(b.getIndexAttr(d));
      offsets[dim] = offset;
      sizes[dim] = b.getIndexAttr(block);
      return std::make_tuple(offsets, sizes, strides);
    };
    DenseMap<std::pair<Value, unsigned>, Value> slices;
    auto slice = [&](Value v, unsigned dim) {
      auto [it, inserted] = slices.try_emplace({v, dim});
      if (inserted) {
        auto [offsets, sizes, strides] = sliceArgs(cast<RankedTensorType>(v.getType()), dim);
        it->second = tensor::ExtractSliceOp::create(b, loc, v, offsets, sizes, strides);
      }
      return it->second;
    };
    DenseMap<Value, Value> mapped{{state, slice(state, 0)}};
    for (Operation *op : ops) {
      auto g = cast<linalg::GenericOp>(op);
      SmallVector<Value> operands;
      for (OpOperand &operand : g->getOpOperands()) {
        Value v = operand.get();
        if (auto it = mapped.find(v); it != mapped.end()) operands.push_back(it->second);
        else if (auto pos = headPosition(g.getMatchingIndexingMap(&operand))) operands.push_back(slice(v, *pos));
        else operands.push_back(v);
      }
      Operation *clone = b.clone(*op);
      clone->setOperands(operands);
      clone->getResult(0).setType(cast<linalg::GenericOp>(clone).getDpsInits()[0].getType());
      mapped[op->getResult(0)] = clone->getResult(0);
    }
    b.setInsertionPointToStart(forall.getTerminator().getBody());
    for (auto [v, out] : llvm::zip(outputs, forall.getRegionOutArgs())) {
      auto [offsets, sizes, strides] = sliceArgs(cast<RankedTensorType>(v.getType()), 0);
      tensor::ParallelInsertSliceOp::create(b, loc, mapped[v], out, offsets, sizes, strides);
    }
    for (auto [v, result] : llvm::zip(outputs, forall.getResults()))
      v.replaceUsesWithIf(result, [&](OpOperand &use) { return !inChain.contains(use.getOwner()); });
    SmallVector<Operation *> order(moved.begin(), moved.end());
    llvm::sort(order, [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
    Operation *after = forall;
    for (Operation *op : order) {
      op->moveAfter(after);
      after = op;
    }
    for (Operation *op : llvm::reverse(ops)) op->erase();
    forall->setAttr("hyb.cores", b.getI64IntegerAttr(cores));
    forall->setAttr("hyb.state_fused", b.getUnitAttr());
    return true;
  }
};

} // namespace
