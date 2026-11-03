//===- NpuActivationResidency.cpp - isolate escaping destinations ----------===//
//
// Tensor CSE can give an exported activation and earlier phase-local tensors
// the same tensor.empty destination. Bufferization then reuses the phase's
// DRAM out-parameter even for QKV / Gate-Up inputs consumed inside that phase.
//
// Give each escaping result a private empty destination before bufferization.
// The original shared destination stays phase-local; hyb-npu-place puts it in
// shared L2 when several cores consume it. Phase outputs keep their LP6 buffers.
//===----------------------------------------------------------------------===//

#include "npu/Decode/Support/TensorDestination.h"
#include "npu/Decode/Transforms/PassSupport.h"

namespace mlir::hybrid {
#define GEN_PASS_DEF_HYBNPUACTIVATIONRESIDENCY
#include "npu/Decode/Transforms/Passes.h.inc"
} // namespace mlir::hybrid

using namespace mlir;
using namespace mlir::hybrid;

namespace {

/// The packet-NPU research kernel (--pim-packet-npu) keeps its own buffer contract. Every other NPU
/// phase qualifies, also in a program with PIM phases (their LP6 inputs are phase outputs, which
/// keep their LP6 buffers).
bool packetKernel(ModuleOp module) { return module->hasAttr("hyb.pim_packet_npu_requested"); }

struct HybNpuActivationResidencyPass
    : mlir::hybrid::impl::HybNpuActivationResidencyBase<HybNpuActivationResidencyPass> {
  using Base::Base;

  void runOnOperation() override {
    auto target = loadTargetSpec(getOperation(), spec);
    if (failed(target)) return signalPassFailure();
    if (packetKernel(getOperation())) return;

    for (func::FuncOp fn : getNpuPhases(getOperation())) {
      // hyb-outline-phases emits one straight-line block; leave anything else untouched.
      if (!llvm::hasSingleElement(fn.getBody())) continue;
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
      // Accumulate when the pass is run more than once.
      auto previous = fn->getAttrOfType<IntegerAttr>("hyb.activation_residency_private_results");
      fn->setAttr("hyb.activation_residency_private_results",
          IntegerAttr::get(IntegerType::get(&getContext(), 64), count + (previous ? previous.getInt() : 0)));
    }
  }
};

} // namespace
