//===- NpuActivationResidency.cpp - isolate escaping destinations ----------===//
//
// Tensor CSE can give an exported activation and earlier phase-local tensors
// the same tensor.empty destination. Bufferization then reuses the phase's
// DRAM out-parameter even for QKV / Gate-Up inputs consumed inside that phase.
//
// Give each escaping result a private empty destination before bufferization.
// The original shared destination stays phase-local; hyb-npu-place puts it in
// shared L2 when several cores consume it. No producer/consumer is reordered,
// no tensor value is approximated, and final phase outputs retain the DRAM ABI.
// This is activation residency, not weight fusion or cross-phase SRAM linking.
//===----------------------------------------------------------------------===//

#include "npu/Decode/Support/TargetSpec.h"
#include "npu/Decode/Support/TensorDestination.h"
#include "npu/Decode/Transforms/Passes.h"

namespace mlir::hybrid {
#define GEN_PASS_DEF_HYBNPUACTIVATIONRESIDENCY
#include "npu/Decode/Transforms/Passes.h.inc"
} // namespace mlir::hybrid

using namespace mlir;
using namespace mlir::hybrid;

namespace {

bool isNpuOnly(ModuleOp module) {
  if (module->hasAttr("hyb.pim_packet_npu_requested")) return false;
  bool hasNpu = false;
  for (auto fn : module.getOps<func::FuncOp>()) {
    auto phase = fn->getAttrOfType<StringAttr>(kPhase);
    if (!phase) continue;
    if (phase.getValue() != "npu" || fn->hasAttr("hyb.npu_kernel")) return false;
    hasNpu = true;
  }
  return hasNpu;
}

struct HybNpuActivationResidencyPass
    : mlir::hybrid::impl::HybNpuActivationResidencyBase<HybNpuActivationResidencyPass> {
  using Base::Base;

  void runOnOperation() override {
    std::string error;
    auto target = TargetSpec::load(spec, error);
    if (failed(target)) {
      getOperation().emitError(error);
      return signalPassFailure();
    }
    if (!target->npu.activationResidency || !isNpuOnly(getOperation())) return;

    for (auto fn : getOperation().getOps<func::FuncOp>()) {
      auto phase = fn->getAttrOfType<StringAttr>(kPhase);
      // OutlinePhases emits one straight-line block. Unknown control flow is
      // left untouched rather than treating one return as the function exit.
      if (!phase || phase.getValue() != "npu" || !llvm::hasSingleElement(fn.getBody())) continue;
      auto ret = dyn_cast<func::ReturnOp>(fn.front().getTerminator());
      if (!ret) continue;
      int64_t count = 0;
      for (Value value : ret.getOperands()) {
        if (!isa<RankedTensorType>(value.getType())) continue;
        OpOperand *destination = findEmptyDestination(value);
        if (!destination) continue;
        auto empty = destination->get().getDefiningOp<tensor::EmptyOp>();
        if (empty->hasOneUse()) continue;
        OpBuilder builder(empty);
        builder.setInsertionPointAfter(empty);
        destination->set(builder.clone(*empty.getOperation())->getResult(0));
        ++count;
      }
      // Preserve the recorded transformation count when the pass is repeated.
      // This records privatized destinations, not assumed DRAM bytes saved.
      auto previous = fn->getAttrOfType<IntegerAttr>("hyb.activation_residency_private_results");
      fn->setAttr("hyb.activation_residency_private_results",
          IntegerAttr::get(IntegerType::get(&getContext(), 64), count + (previous ? previous.getInt() : 0)));
    }
  }
};

} // namespace
