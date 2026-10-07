//===- StorageF16.cpp - hyb-storage-f16 ------------------------------------===//
//
// Storage precision of the NPU: activations, KV and 1-D weights in FP16. A value stays f32
// only when it is a row-reduction result, or computed from such values alone (the per-row
// scalar chain of RMSNorm / softmax). linalg bodies extend their inputs to f32 and truncate
// the yielded value, so reductions still accumulate in FP32.
//===----------------------------------------------------------------------===//

#include "npu/Decode/Transforms/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/EquivalenceClasses.h"

namespace mlir::hybrid {
#define GEN_PASS_DEF_HYBSTORAGEF16
#include "npu/Decode/Transforms/Passes.h.inc"
} // namespace mlir::hybrid

using namespace mlir;
using namespace mlir::hybrid;

namespace {

bool isF32Tensor(Type t) {
  auto rt = dyn_cast<RankedTensorType>(t);
  return rt && rt.getElementType().isF32();
}

Type toF16(Type t) {
  auto rt = cast<RankedTensorType>(t);
  return RankedTensorType::get(rt.getShape(), Float16Type::get(t.getContext()));
}

bool isReduction(linalg::GenericOp g) {
  return llvm::any_of(g.getIteratorTypesArray(),
                      [](utils::IteratorType t) { return t == utils::IteratorType::reduction; });
}

struct HybStorageF16Pass : mlir::hybrid::impl::HybStorageF16Base<HybStorageF16Pass> {
  void runOnOperation() override {
    func::FuncOp step = getStepFunction(getOperation());
    Block &body = step.getBody().front();
    OpBuilder b(&getContext());

    // Weight arguments consumed by quantized ops are referenced by name: drop them.
    llvm::BitVector erase(step.getNumArguments());
    for (BlockArgument a : step.getArguments()) {
      auto kind = step.getArgAttrOfType<StringAttr>(a.getArgNumber(), kKind);
      if (kind && kind.getValue() == "weight" && a.use_empty()) erase.set(a.getArgNumber());
    }
    if (failed(step.eraseArguments(erase))) return signalPassFailure();

    // Every tensor.empty / linalg.fill feeds exactly one op afterwards, so an init can take
    // the precision of the op it initializes.
    SmallVector<Operation *> inits;
    body.walk([&](Operation *op) {
      if (isa<tensor::EmptyOp, linalg::FillOp>(op)) inits.push_back(op);
    });
    for (Operation *op : llvm::reverse(inits)) {
      SmallVector<OpOperand *> uses;
      for (OpOperand &u : op->getResult(0).getUses()) uses.push_back(&u);
      for (size_t i = 1; i < uses.size(); ++i) {
        b.setInsertionPoint(uses[i]->getOwner());
        Operation *c = b.clone(*op);
        uses[i]->set(c->getResult(0));
      }
    }

    // Decide which f32 values stay f32. Values that must share an element type (reshape
    // source/result, DPS init/result, fill init/result, concat operands/result) form one class;
    // a class stays f32 when it holds a reduction result, the scale of hyb.quant, or the result
    // of an elementwise op whose f32 inputs all stay f32. Iterate to a fixed point.
    llvm::EquivalenceClasses<Value> classes;
    auto unite = [&](Value a, Value c) {
      if (isF32Tensor(a.getType()) && isF32Tensor(c.getType())) classes.unionSets(a, c);
    };
    for (BlockArgument a : step.getArguments())
      if (isF32Tensor(a.getType())) classes.insert(a);
    body.walk([&](Operation *op) {
      for (Value r : op->getResults())
        if (isF32Tensor(r.getType())) classes.insert(r);
      if (isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp, tensor::ExtractSliceOp, tensor::CastOp>(op))
        unite(op->getOperand(0), op->getResult(0));
      if (auto c = dyn_cast<tensor::ConcatOp>(op))
        for (Value in : c.getInputs()) unite(in, c.getResult());
      if (auto dps = dyn_cast<DestinationStyleOpInterface>(op))
        for (auto [i, init] : llvm::enumerate(dps.getDpsInitsMutable()))
          if (i < op->getNumResults()) unite(init.get(), op->getResult(i));
    });
    DenseSet<Value> keptLeaders;
    auto kept = [&](Value v) {
      return isF32Tensor(v.getType()) && classes.findLeader(v) != classes.member_end() &&
             keptLeaders.contains(classes.getLeaderValue(v));
    };
    auto markKept = [&](Value v) {
      if (!isF32Tensor(v.getType()) || classes.findLeader(v) == classes.member_end()) return false;
      return keptLeaders.insert(classes.getLeaderValue(v)).second;
    };
    for (bool changed = true; changed;) {
      changed = false;
      body.walk([&](Operation *op) {
        if (auto g = dyn_cast<linalg::GenericOp>(op)) {
          bool any = false, all = true;
          for (Value v : g.getDpsInputs()) {
            if (!isF32Tensor(v.getType())) continue;
            any = true;
            all &= kept(v);
          }
          if (isReduction(g) || (any && all))
            for (Value r : g->getResults()) changed |= markKept(r);
        } else if (auto q = dyn_cast<hyb::QuantOp>(op)) {
          changed |= markKept(q->getResult(1));
        } else if (isa<hyb::WeightOp>(op)) {      // weights keep their checkpoint layout type
          changed |= markKept(op->getResult(0));
        }
      });
    }
    DenseSet<Value> keep;
    body.walk([&](Operation *op) {
      for (Value r : op->getResults())
        if (kept(r)) keep.insert(r);
    });

    // Retype block arguments and results.
    for (BlockArgument a : step.getArguments())
      if (isF32Tensor(a.getType())) a.setType(toF16(a.getType()));
    SmallVector<Operation *> ops;
    body.walk([&](Operation *op) {
      if (op->getParentOp() == step.getOperation()) ops.push_back(op);
    });
    for (Operation *op : ops) {
      b.setInsertionPoint(op);
      // Named linalg ops carry a region typed at construction: rebuild them.
      if (auto fill = dyn_cast<linalg::FillOp>(op)) {
        Value init = fill.getOutputs()[0];
        Type t = init.getType();
        if (isF32Tensor(fill.getResult(0).getType()) && !keep.contains(fill.getResult(0))) {
          init.setType(toF16(init.getType()));
          t = init.getType();
        }
        Type et = cast<RankedTensorType>(t).getElementType();
        Value v = fill.getInputs()[0];
        if (v.getType() != et) v = arith::TruncFOp::create(b, fill.getLoc(), et, v);
        auto nf = linalg::FillOp::create(b, fill.getLoc(), v, init);
        fill.getResult(0).replaceAllUsesWith(nf.getResult(0));
        fill.erase();
        continue;
      }
      if (isa<linalg::BatchMatmulOp, linalg::MatmulOp, linalg::TransposeOp>(op)) {
        auto lop = cast<linalg::LinalgOp>(op);
        Value out = lop.getDpsInits()[0];
        Value res = op->getResult(0);
        bool toHalf = isF32Tensor(res.getType()) && !keep.contains(res);
        if (toHalf && isF32Tensor(out.getType())) out.setType(toF16(out.getType()));
        Operation *nop;
        if (auto tr = dyn_cast<linalg::TransposeOp>(op)) {
          nop = linalg::TransposeOp::create(b, op->getLoc(), tr.getInput(), out, tr.getPermutation());
        } else if (auto bm = dyn_cast<linalg::BatchMatmulOp>(op)) {
          nop = linalg::BatchMatmulOp::create(b, op->getLoc(), TypeRange{out.getType()}, bm.getDpsInputs(), ValueRange{out});
          if (bm.hasUserDefinedMaps()) nop->setAttr("indexing_maps", bm.getIndexingMaps());
        } else {
          auto mm = cast<linalg::MatmulOp>(op);
          nop = linalg::MatmulOp::create(b, op->getLoc(), TypeRange{out.getType()}, mm.getDpsInputs(), ValueRange{out});
          if (mm.hasUserDefinedMaps()) nop->setAttr("indexing_maps", mm.getIndexingMaps());
        }
        res.replaceAllUsesWith(nop->getResult(0));
        op->erase();
        continue;
      }
      for (Value r : op->getResults())
        if (isF32Tensor(r.getType()) && !keep.contains(r)) r.setType(toF16(r.getType()));
      if (auto g = dyn_cast<linalg::GenericOp>(op)) fixGenericBody(g, b);
    }
    // Function type follows the return operands.
    auto ret = cast<func::ReturnOp>(body.getTerminator());
    step.setType(FunctionType::get(&getContext(), body.getArgumentTypes(), ret.getOperandTypes()));
  }

  /// Block arguments follow the operand element types: an f16 input is extended to f32 where
  /// the body used it; an f16 output gets its yielded value truncated.
  void fixGenericBody(linalg::GenericOp g, OpBuilder &b) {
    Block &blk = g.getRegion().front();
    for (OpOperand &operand : g->getOpOperands()) {
      BlockArgument arg = blk.getArgument(operand.getOperandNumber());
      Type et = getElementTypeOrSelf(operand.get().getType());
      if (arg.getType() == et) continue;
      Type old = arg.getType();
      arg.setType(et);
      if (!arg.use_empty()) {
        b.setInsertionPointToStart(&blk);
        Value ext = arith::ExtFOp::create(b, g.getLoc(), old, arg);
        arg.replaceAllUsesExcept(ext, ext.getDefiningOp());
      }
    }
    auto yield = cast<linalg::YieldOp>(blk.getTerminator());
    for (auto [i, v] : llvm::enumerate(yield.getValues())) {
      Type et = getElementTypeOrSelf(g->getResult(i).getType());
      if (v.getType() == et) continue;
      b.setInsertionPoint(yield);
      Value t = arith::TruncFOp::create(b, g.getLoc(), et, v);
      yield->setOperand(i, t);
    }
  }
};
} // namespace
