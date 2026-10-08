//===- OutlinePhases.cpp - hyb-outline-phases, hyb-private-phase-results ----===//
//
// The decode step runs as a sequence of phases: a PIM phase is one group of qmatmuls with
// target = pim (one PIM submission), an NPU phase a maximal run of the other ops. Every phase
// is outlined into a private function tagged `hyb.phase`, and the step function becomes the
// list of calls. Constants and tensor.empty are cloned into each phase that uses them.
// Before bufferization, every NPU phase result read by a PIM phase gets a destination of its
// own, so its PIM-layout LP6 buffer never also holds a value used only inside the phase.
//===----------------------------------------------------------------------===//

#include "npu/Decode/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/SetVector.h"
#include <functional>

namespace mlir::hybrid {
#define GEN_PASS_DEF_HYBOUTLINEPHASES
#define GEN_PASS_DEF_HYBPRIVATEPHASERESULTS
#include "npu/Decode/Transforms/Passes.h.inc"
} // namespace mlir::hybrid

using namespace mlir;
using namespace mlir::hybrid;

namespace {

bool isPimOp(Operation *op) {
  auto q = dyn_cast<hyb::QMatmulOp>(op);
  return q && q.getTarget() && *q.getTarget() == "pim";
}

/// Ops cloned into every phase that uses them instead of producing a cross-phase value:
/// constants, tensor.empty, constant fills, and reshapes of step arguments (views of LP6 buffers).
bool isFree(Operation *op) {
  if (isa<arith::ConstantOp, tensor::EmptyOp, hyb::WeightOp>(op)) return true;
  if (isa<linalg::FillOp>(op))                    // fill of a constant into a fresh tensor
    return llvm::all_of(op->getOperands(), [](Value v) { return v.getDefiningOp() && isFree(v.getDefiningOp()); });
  if (isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp>(op)) {
    Value src = op->getOperand(0);
    if (isa<BlockArgument>(src)) return true;
    return src.getDefiningOp() && isFree(src.getDefiningOp());
  }
  return false;
}

struct HybOutlinePhasesPass : mlir::hybrid::impl::HybOutlinePhasesBase<HybOutlinePhasesPass> {
  using Base::Base;
  void runOnOperation() override {
    ModuleOp module = getOperation();
    func::FuncOp step = getStepFunction(module);
    Block &body = step.getBody().front();
    OpBuilder b(&getContext());

    // Members of a PIM group run in one submission: move them next to the first member.
    DenseMap<int64_t, Operation *> last;
    SmallVector<Operation *> ops;
    for (Operation &op : body.without_terminator()) ops.push_back(&op);
    for (Operation *op : ops) {
      if (!isPimOp(op)) continue;
      int64_t g = cast<hyb::QMatmulOp>(op).getGroup().value_or(-1);
      auto it = last.find(g);
      if (it == last.end()) {
        last[g] = op;
        continue;
      }
      auto q = cast<hyb::QMatmulOp>(op);
      op->moveAfter(it->second);
      b.setInsertionPoint(op);
      auto ty = cast<RankedTensorType>(q.getInit().getType());
      q.getInitMutable().assign(tensor::EmptyOp::create(b, op->getLoc(), ty.getShape(), ty.getElementType()));
      it->second = op;
    }

    // Segments in program order.
    struct Segment {
      std::string kind;
      int64_t group = -1;
      SmallVector<Operation *> ops;
      std::optional<int64_t> gemvLayer;     // layer of the NPU GEMVs in this segment
      bool packetKernel = false;          // a single packet qmatmul owns a separate NPU phase
    };
    auto layerOf = [](StringRef weight) -> int64_t {
      size_t at = weight.find("layers.");
      if (at == StringRef::npos) return -1;
      int64_t n = -1;
      if (weight.drop_front(at + 7).take_while(llvm::isDigit).getAsInteger(10, n)) return -1;
      return n;
    };
    SmallVector<Segment> segments;
    for (Operation &op : body.without_terminator()) {
      if (isFree(&op)) continue;
      std::string kind = isPimOp(&op) ? "pim" : "npu";
      bool packetKernel = op.hasAttr("hyb.npu_packet");
      int64_t g = kind == "pim" ? cast<hyb::QMatmulOp>(op).getGroup().value_or(-1) : -1;
      std::optional<int64_t> layer;
      if (auto w = op.getAttrOfType<StringAttr>("hyb.npu_gemv")) layer = layerOf(w.getValue());
      bool split = npuSplit == "layer" && layer && !segments.empty() && segments.back().kind == "npu" &&
                   segments.back().gemvLayer && *segments.back().gemvLayer != *layer;
      if (segments.empty() || segments.back().kind != kind || segments.back().group != g || split ||
          packetKernel || segments.back().packetKernel)
        segments.push_back({kind, g, {}, std::nullopt, packetKernel});
      segments.back().ops.push_back(&op);
      if (layer) segments.back().gemvLayer = layer;
    }

    int index = 0;
    for (Segment &seg : segments) {
      llvm::SmallPtrSet<Operation *, 32> inSeg(seg.ops.begin(), seg.ops.end());
      auto inside = [&](Operation *op) {
        for (; op; op = op->getParentOp())
          if (inSeg.contains(op)) return true;
        return false;
      };
      llvm::SetVector<Value> inputs, freeValues;
      std::function<void(Value)> addFree = [&](Value v) {
        if (!freeValues.insert(v)) return;
        for (Value o : v.getDefiningOp()->getOperands()) {
          if (Operation *d = o.getDefiningOp(); d && isFree(d)) addFree(o);
          else inputs.insert(o);
        }
      };
      for (Operation *op : seg.ops)
        op->walk([&](Operation *nested) {
          for (Value v : nested->getOperands()) {
            Operation *def = v.getDefiningOp();
            if (def && inside(def)) continue;
            if (auto ba = dyn_cast<BlockArgument>(v); ba && inside(ba.getOwner()->getParentOp())) continue;
            if (def && isFree(def)) addFree(v);
            else inputs.insert(v);
          }
        });
      llvm::SetVector<Value> outputs;
      for (Operation *op : seg.ops)
        for (Value r : op->getResults())
          for (Operation *user : r.getUsers())
            if (!inside(user)) outputs.insert(r);

      std::string name = seg.kind + "_" + std::to_string(index++);
      b.setInsertionPoint(step);
      auto fnType = b.getFunctionType(TypeRange(inputs.getArrayRef()), TypeRange(outputs.getArrayRef()));
      auto fn = func::FuncOp::create(b, step.getLoc(), name, fnType);
      fn.setPrivate();
      fn->setAttr(kPhase, b.getStringAttr(seg.kind));
      if (seg.packetKernel) fn->setAttr("hyb.npu_kernel", b.getStringAttr("pim_packet_g4"));
      if (seg.kind == "pim") fn->setAttr("hyb.group", b.getI64IntegerAttr(seg.group));
      Block *entry = fn.addEntryBlock();
      OpBuilder fb = OpBuilder::atBlockBegin(entry);
      IRMapping map;
      for (auto [v, a] : llvm::zip(inputs, entry->getArguments())) map.map(v, a);
      // Free values may depend on step arguments (a reshape of an argument): those arguments
      // become phase inputs, and free ops are cloned in dependence order.
      std::function<Value(Value)> materialize = [&](Value v) -> Value {
        if (Value m = map.lookupOrNull(v)) return m;
        Operation *def = v.getDefiningOp();
        for (Value o : def->getOperands()) materialize(o);
        return fb.clone(*def, map)->getResult(cast<OpResult>(v).getResultNumber());
      };
      for (Value v : freeValues) materialize(v);
      for (Operation *op : seg.ops) fb.clone(*op, map);
      SmallVector<Value> rets;
      for (Value v : outputs) rets.push_back(map.lookup(v));
      func::ReturnOp::create(fb, step.getLoc(), rets);

      b.setInsertionPoint(seg.ops.front());
      auto call = func::CallOp::create(b, seg.ops.front()->getLoc(), fn, inputs.getArrayRef());
      for (auto [v, r] : llvm::zip(outputs, call.getResults())) {
        Value out = v;
        out.replaceUsesWithIf(r, [&](OpOperand &u) { return !inside(u.getOwner()); });
      }
      for (Operation *op : llvm::reverse(seg.ops)) op->erase();
    }
    // Free ops left in the step are dead now.
    SmallVector<Operation *> dead;
    for (Operation &op : body.without_terminator())
      if (isFree(&op)) dead.push_back(&op);
    for (Operation *op : llvm::reverse(dead))
      if (op->use_empty()) op->erase();
  }
};

//===----------------------------------------------------------------------===//
// hyb-private-phase-results
//===----------------------------------------------------------------------===//

/// The operand through which `v` is computed into a tensor.empty, following destination
/// operands, loop inits and reshapes; null when the chain starts elsewhere.
OpOperand *emptyDestination(Value v) {
  while (auto r = dyn_cast<OpResult>(v)) {
    Operation *op = r.getOwner();
    OpOperand *tied = nullptr;
    if (auto dps = dyn_cast<DestinationStyleOpInterface>(op)) tied = dps.getTiedOpOperand(r);
    else if (auto forall = dyn_cast<scf::ForallOp>(op)) tied = forall.getTiedOpOperand(r);
    else if (auto loop = dyn_cast<LoopLikeOpInterface>(op)) tied = loop.getTiedLoopInit(r);
    else if (isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp, tensor::CastOp>(op)) tied = &op->getOpOperand(0);
    if (!tied) return nullptr;
    if (tied->get().getDefiningOp<tensor::EmptyOp>()) return tied;
    v = tied->get();
  }
  return nullptr;
}

bool isPhase(func::FuncOp fn, StringRef kind) {
  auto k = fn ? fn->getAttrOfType<StringAttr>(kPhase) : StringAttr();
  return k && k.getValue() == kind;
}

struct HybPrivatePhaseResultsPass
    : mlir::hybrid::impl::HybPrivatePhaseResultsBase<HybPrivatePhaseResultsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SymbolTable symbols(module);
    getStepFunction(module).walk([&](func::CallOp call) {
      auto fn = symbols.lookup<func::FuncOp>(call.getCallee());
      if (!isPhase(fn, "npu")) return;
      auto ret = cast<func::ReturnOp>(fn.getBody().front().getTerminator());
      for (OpResult r : call.getResults()) {
        bool toPim = llvm::any_of(r.getUsers(), [&](Operation *user) {
          auto use = dyn_cast<func::CallOp>(user);
          return use && isPhase(symbols.lookup<func::FuncOp>(use.getCallee()), "pim");
        });
        OpOperand *dest = toPim ? emptyDestination(ret.getOperand(r.getResultNumber())) : nullptr;
        if (!dest) continue;
        auto empty = dest->get().getDefiningOp<tensor::EmptyOp>();
        if (empty->hasOneUse()) continue;
        OpBuilder b(empty);
        b.setInsertionPointAfter(empty);
        dest->set(b.clone(*empty.getOperation())->getResult(0));
      }
    });
  }
};
} // namespace
