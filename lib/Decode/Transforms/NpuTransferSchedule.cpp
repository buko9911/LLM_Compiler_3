// Bounded input-transfer scheduling using SSA and declared memory effects.
// Compute names and arithmetic patterns are deliberately not inspected here.
#include "npu/Decode/Support/TargetSpec.h"
#include "npu/Decode/Support/NpuOperationTrace.h"
#include "npu/Decode/Transforms/Passes.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::hybrid {
#define GEN_PASS_DEF_HYBNPUTRANSFERSCHEDULE
#include "npu/Decode/Transforms/Passes.h.inc"
}
using namespace mlir;
using namespace mlir::hybrid;

namespace {
bool inputCopy(memref::CopyOp copy) {
  auto source = dyn_cast<MemRefType>(copy.getSource().getType());
  auto target = dyn_cast<MemRefType>(copy.getTarget().getType());
  return source && target && target.getMemorySpaceAsInt() == 1 &&
         (source.getMemorySpaceAsInt() == 0 || source.getMemorySpaceAsInt() == 2);
}

bool scheduled(Operation *op) {
  return op->hasAttr("hyb.stream") || op->hasAttr("hyb.overlap_id") ||
         op->hasAttr("hyb.input_pipeline_copy") || op->hasAttr("hyb.input_pipeline_tile") ||
         op->hasAttr("hyb.schedule_barrier");
}

bool ownedSchedule(Operation *op) {
  for (Operation *parent = op; parent && !isa<func::FuncOp>(parent); parent = parent->getParentOp())
    if (scheduled(parent)) return true;
  return false;
}

bool knownEffects(Operation *op, SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  if (scheduled(op) || op->getNumSuccessors() ||
      op->hasTrait<OpTrait::IsTerminator>()) return false;
  auto interface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!interface) return false;
  // A region-bearing compute operation may declare its outer buffer effects
  // (e.g. a structured generic). Require its entire body to be memory-free.
  // Unknown/control-flow regions and hidden calls are opaque scheduling walls.
  for (Region &region : op->getRegions()) {
    if (!region.hasOneBlock()) return false;
    for (Operation &nested : region.front()) {
      if (nested.getNumRegions() || !isMemoryEffectFree(&nested)) return false;
    }
  }
  interface.getEffects(effects);
  return true;
}

bool conflict(memref::CopyOp copy, const MemoryEffects::EffectInstance &effect,
              AliasAnalysis &aliases) {
  Value value = effect.getValue();
  if (!value || !isa<BaseMemRefType>(value.getType())) return true;
  auto *kind = effect.getEffect();
  bool read = isa<MemoryEffects::Read>(kind);
  bool write = isa<MemoryEffects::Write, MemoryEffects::Allocate, MemoryEffects::Free>(kind);
  if (!read && !write) return true;
  // The copy reads source and writes target. Read/read may reorder; any
  // unknown alias on a pair containing a write is a real scheduling barrier.
  if (write && !aliases.alias(value, copy.getSource()).isNo()) return true;
  return (read || write) && !aliases.alias(value, copy.getTarget()).isNo();
}

int64_t schedule(memref::CopyOp copy, AliasAnalysis &aliases) {
  if (!inputCopy(copy) || copy->hasAttr("hyb.transfer_scheduled")) return 0;
  auto targetAlloc = copy.getTarget().getDefiningOp<memref::AllocOp>();
  if (targetAlloc && (targetAlloc->getBlock() != copy->getBlock() ||
                     !targetAlloc.getType().hasStaticShape() || targetAlloc->getNumOperands()))
    targetAlloc = {};
  Operation *insertion = copy;
  bool moveAllocation = false;
  int64_t crossedMemoryOps = 0;
  for (Operation *previous = copy->getPrevNode(); previous; previous = previous->getPrevNode()) {
    if (previous == targetAlloc.getOperation()) {
      moveAllocation = true;
      continue;
    }
    // Keep the transfer stream ordered. The purpose is to expose independent
    // work in this local window, not to flood L2 with phase-wide prefetches.
    if (auto transfer = dyn_cast<memref::CopyOp>(previous); transfer && inputCopy(transfer)) break;
    if (llvm::any_of(copy->getOperands(), [&](Value operand) {
          return operand.getDefiningOp() == previous;
        })) break;
    SmallVector<MemoryEffects::EffectInstance> effects;
    if (!knownEffects(previous, effects) ||
        llvm::any_of(effects, [&](const auto &effect) { return conflict(copy, effect, aliases); })) break;
    if (llvm::any_of(effects, [](const auto &effect) {
          return isa<MemoryEffects::Read, MemoryEffects::Write>(effect.getEffect());
        })) ++crossedMemoryOps;
    insertion = previous;
  }
  // Moving only through declarations/constants cannot expose useful work.
  // Do not perturb allocation placement or annotate a no-op transformation.
  if (!crossedMemoryOps || insertion == copy.getOperation()) return 0;
  if (moveAllocation) targetAlloc->moveBefore(insertion);
  copy->moveBefore(insertion);
  copy->setAttr("hyb.transfer_scheduled", UnitAttr::get(copy.getContext()));
  return crossedMemoryOps;
}

struct HybNpuTransferSchedulePass
    : mlir::hybrid::impl::HybNpuTransferScheduleBase<HybNpuTransferSchedulePass> {
  using Base::Base;
  void runOnOperation() override {
    std::string error;
    auto target = TargetSpec::load(spec, error);
    if (failed(target)) { getOperation().emitError(error); return signalPassFailure(); }
    if (!target->npu.transferSchedule) return;
    for (auto fn : getOperation().getOps<func::FuncOp>()) {
      auto phase = fn->getAttrOfType<StringAttr>(kPhase);
      if (!phase || phase.getValue() != "npu" || fn->hasAttr("hyb.npu_kernel")) continue;
      // Assign stable phase-local identities before reordering. Re-running the
      // pass preserves origin identities instead of renumbering the new order.
      if (!fn->hasAttr("hyb.schedule_origin_schema")) {
        int64_t origin = 0;
        fn.walk([&](Operation *op) {
          if (op == fn.getOperation() || op->hasTrait<OpTrait::IsTerminator>() ||
              isNpuTraceMetadataOnly(*op)) return;
          op->setAttr("hyb.schedule_origin", IntegerAttr::get(IntegerType::get(&getContext(), 64), origin++));
        });
        fn->setAttr("hyb.schedule_origin_schema", StringAttr::get(&getContext(), "npu-trace-phase-ordinal-v1"));
        fn->setAttr("hyb.schedule_origin_count", IntegerAttr::get(IntegerType::get(&getContext(), 64), origin));
      }
      AliasAnalysis aliases(fn);
      SmallVector<memref::CopyOp> copies;
      fn.walk([&](memref::CopyOp copy) { if (inputCopy(copy)) copies.push_back(copy); });
      int64_t moved = 0, crossed = 0, owned = 0;
      for (auto copy : copies) {
        if (ownedSchedule(copy)) { ++owned; continue; }
        int64_t count = schedule(copy, aliases);
        if (count) { ++moved; crossed += count; }
      }
      auto i64 = IntegerType::get(&getContext(), 64);
      fn->setAttr("hyb.transfer_schedule_candidates", IntegerAttr::get(i64, copies.size()));
      fn->setAttr("hyb.transfer_schedule_moved", IntegerAttr::get(i64, moved));
      fn->setAttr("hyb.transfer_schedule_crossed", IntegerAttr::get(i64, crossed));
      fn->setAttr("hyb.transfer_schedule_owned", IntegerAttr::get(i64, owned));
    }
  }
};
} // namespace
