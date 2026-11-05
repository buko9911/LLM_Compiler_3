//===- PimBoundary.cpp - hyb-pim-boundary ---------------------------------===//
//
// Choices at the boundary between an NPU phase and the PIM phases around it, made from tensor sizes and the
// target's hardware numbers and recorded with their numbers (io.json, phases[].decisions).
//  * Read order. The GDMA read engine serves its queue in order. A large read of LP6 data with no hazard (the KV
//    cache) is issued at once and every later read waits behind it, also the small INT32 PIM results the phase needs
//    first. A read larger than the PIM results read before it gets `hyb.read_after_gather`: codegen makes it wait for
//    those reads, so the results arrive first and the compute that needs them starts while the large read runs.
//  * Input feed. Writing an activation to every PU's bank needs 32 copies of each 32-byte packet in L2 (the bank
//    groups x subchannels of one DRAM column). One core copies at the local DMA rate of one core. The copy gets
//    `hyb.feed_cores`, the number of cores that share it, when publishing the input to the others costs less than it
//    saves.
//  * Quant of one row with that feed. The scale is the maximum over the row, so the quant runs on one core and its input
//    and output cross L2. It gets `hyb.quant_cores` when the cores exchanging their maxima and quantizing and feeding
//    their own columns costs less.
//===----------------------------------------------------------------------===//

#include "npu/Decode/Transforms/NpuCost.h"
#include "npu/Decode/Transforms/PassSupport.h"
#include "npu/Target/PimLayout.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/STLExtras.h"
#include <map>

namespace mlir::hybrid {
#define GEN_PASS_DEF_HYBPIMBOUNDARY
#include "npu/Decode/Transforms/Passes.h.inc"
}
using namespace mlir;
using namespace mlir::hybrid;

namespace {
namespace pim = ::npu::pim;

Value rootOf(Value v) {
  while (Operation *d = v.getDefiningOp()) {
    if (auto view = dyn_cast<ViewLikeOpInterface>(d)) v = view.getViewSource();
    else break;
  }
  return v;
}

int64_t bytesOf(Type t) {
  auto m = dyn_cast<MemRefType>(t);
  if (!m || !m.hasStaticShape()) return 0;
  return m.getNumElements() * std::max<int64_t>(1, m.getElementTypeBitWidth() / 8);
}

struct HybPimBoundaryPass : mlir::hybrid::impl::HybPimBoundaryBase<HybPimBoundaryPass> {
  using Base::Base;

  /// [x(L2) -> L1 row] [hyb.quant] [Q -> activation, shared by the cores]: split when, per the numbers below, it saves
  /// time. One core: the L2 row to L1, the VPU, Q to L2 for the others. Split: each step on K / cores, plus the exchange.
  void splitQuant(func::FuncOp fn, memref::CopyOp feed, int64_t k, const TargetSpec &target) {
    hyb::QuantOp quant;
    for (Operation *op = feed->getPrevNode(); op && !quant; op = op->getPrevNode())
      if (auto q = dyn_cast<hyb::QuantOp>(op)) quant = q;
    if (!quant || quant.getQInit() != feed.getSource()) return;
    memref::CopyOp input;
    for (Operation *op = quant->getPrevNode(); op && !input; op = op->getPrevNode())
      if (auto c = dyn_cast<memref::CopyOp>(op); c && c.getTarget() == quant.getInput()) input = c;
    auto source = input ? dyn_cast<MemRefType>(input.getSource().getType()) : MemRefType();
    const int cores = target.npu.cores;
    if (!source || source.getMemorySpaceAsInt() != 2 || k % cores || (k / cores) % 32) return;
    const double ldma = target.npu.ldmaGBpsPerCore * 1e3, vpu = double(k) * cost::quantCyclesPerElement(target.npu) / target.npu.clockMHz;
    const double oneCoreUs = double(k) * 2 / ldma + vpu + double(k) / ldma;
    const double splitUs = (double(k) * 2 / ldma + vpu) / cores + cost::splitCostUs(target.npu, fn);
    const bool split = splitUs < oneCoreUs;
    cost::recordDecision(fn, "split quant", "row K " + std::to_string(k), split ? "split" : "one core",
                         {{"cores", double(split ? cores : 1)}, {"one_core_us", oneCoreUs}, {"split_us", splitUs}});
    if (!split) return;
    quant->setAttr("hyb.quant_cores", IntegerAttr::get(IntegerType::get(&getContext(), 64), cores));
    input->setAttr("hyb.quant_source", UnitAttr::get(&getContext()));
    feed->setAttr("hyb.quant_feeds", UnitAttr::get(&getContext()));
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto target = loadTargetSpec(module, spec);
    if (failed(target)) return signalPassFailure();
    func::FuncOp step = getStepFunction(module);
    if (!step) return;
    SymbolTable symbols(module);
    step.walk([&](func::CallOp call) {
      auto fn = symbols.lookup<func::FuncOp>(call.getCallee());
      if (!fn || !fn->hasAttr(kPhase) || cast<StringAttr>(fn->getAttr(kPhase)).getValue() != "npu") return;
      auto layoutOf = [&](Value v) -> StringRef {
        auto arg = dyn_cast<BlockArgument>(rootOf(v));
        if (!arg || arg.getOwner() != &fn.getBody().front()) return {};
        Value outer = call.getOperand(arg.getArgNumber());
        DictionaryAttr lp6;
        if (auto ba = dyn_cast<BlockArgument>(outer)) lp6 = step.getArgAttrOfType<DictionaryAttr>(ba.getArgNumber(), kLP6);
        else if (Operation *d = outer.getDefiningOp()) lp6 = d->getAttrOfType<DictionaryAttr>(kLP6);
        if (!lp6) return {};
        auto layout = lp6.getAs<StringAttr>("layout");
        return layout ? layout.getValue() : StringRef("dense");
      };
      int64_t gathered = 0;
      fn.walk([&](memref::CopyOp copy) {
        StringRef from = layoutOf(copy.getSource()), to = layoutOf(copy.getTarget());
        const int64_t bytes = bytesOf(copy.getSource().getType());
        if (from == "pim_out") {
          gathered += bytes;
          return;
        }
        if (from == "dense" && gathered > 0 && bytes > gathered) {
          const double readUs = double(bytes) / (target->npu.gdmaReadGBps * 1e3);
          copy->setAttr("hyb.read_after_gather", UnitAttr::get(&getContext()));
          cost::recordDecision(fn, "read order", "LP6 read of " + std::to_string(bytes) + " B",
                               "after the PIM results", {{"read_bytes", double(bytes)}, {"pim_result_bytes", double(gathered)},
                                                         {"read_us", readUs}});
          return;
        }
        if (to != "pim_act") return;
        auto type = dyn_cast<MemRefType>(copy.getSource().getType());
        if (!type || !type.hasStaticShape() || type.getRank() < 1) return;
        // Inside a core loop the quant was split by token rows: each core already feeds its own rows.
        if (copy->getParentOfType<scf::ForallOp>()) return;
        const int64_t k = type.getShape().back(), rows = type.getNumElements() / std::max<int64_t>(1, k);
        const double ldma = target->npu.ldmaGBpsPerCore * 1e3, l1Copy = double(target->npu.l1WriteBytesPerCycle) * target->npu.clockMHz;
        const int cores = target->npu.cores;
        const double replicaBytes = double(rows * k) * 32;
        const double oneCoreUs = replicaBytes / l1Copy + replicaBytes / ldma;
        const double splitUs = double(rows * k) / ldma * (1.0 + 1.0 / cores) + oneCoreUs / cores + cost::splitCostUs(target->npu, fn);
        const bool split = cores > 1 && splitUs < oneCoreUs && k <= target->npu.stagingMiscBytes / 2;
        if (split) copy->setAttr("hyb.feed_cores", IntegerAttr::get(IntegerType::get(&getContext(), 64), cores));
        cost::recordDecision(fn, "input feed", "activation K " + std::to_string(k), split ? "split" : "one core",
                             {{"cores", double(split ? cores : 1)}, {"one_core_us", oneCoreUs}, {"split_us", splitUs}});
        if (split && rows == 1) splitQuant(fn, copy, k, *target);
      });
    });
  }
};
} // namespace
