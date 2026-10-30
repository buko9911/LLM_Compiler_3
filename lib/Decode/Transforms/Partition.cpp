//===- Partition.cpp - hyb-partition ----------------------------------------===//
//
// For every group of qmatmuls reading one quantized input, compare the PIM cost (LdMAC
// streams balanced over the partitions, plus the NPU boundary work: activation written in PIM
// layout, INT32 outputs read back) with the NPU cost (weights streamed from LP6) and set the
// `target` attribute. The group is the decision unit because its members share one
// activation broadcast (PIM) or one weight stream pass (NPU).
//===----------------------------------------------------------------------===//

#include "npu/Decode/Transforms/PassSupport.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <map>

namespace mlir::hybrid {
#define GEN_PASS_DEF_HYBPARTITION
#include "npu/Decode/Transforms/Passes.h.inc"
} // namespace mlir::hybrid

using namespace mlir;
using namespace mlir::hybrid;

namespace {
struct HybPartitionPass : mlir::hybrid::impl::HybPartitionBase<HybPartitionPass> {
  using Base::Base;
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto specOr = loadTargetSpec(module, spec);
    if (failed(specOr)) return signalPassFailure();
    for (StringRef item : llvm::split(StringRef(overrides), ';')) {
      auto [key, t] = item.split(':');
      if (!key.empty()) specOr->mapping.overrides.push_back({key.str(), t.str()});
    }
    auto tokensAttr = module->getAttrOfType<IntegerAttr>("hyb.tokens");
    int64_t M = tokensAttr ? tokensAttr.getInt() : 1;
    std::map<int64_t, SmallVector<hyb::QMatmulOp>> groups;
    // Grouping is part of partitioning, not an execution-fusion pass. Derive
    // IDs from SSA inputs across the module (function-local counters collide).
    DenseMap<Value,int64_t> groupOf;
    module.walk([&](hyb::QMatmulOp op) {
      auto [it, inserted] = groupOf.try_emplace(op.getInput(),groupOf.size());
      op.setGroupAttr(IntegerAttr::get(IntegerType::get(&getContext(),64),it->second));
      groups[it->second].push_back(op);
    });
    llvm::json::Array table;
    Builder b(&getContext());
    int pim = 0, npu = 0;
    for (auto &[g, ops] : groups) {
      SmallVector<GemvShape> shapes;
      llvm::json::Array names, dims;
      for (hyb::QMatmulOp op : ops) {
        auto t = cast<ShapedType>(op.getInit().getType());
        auto in = cast<ShapedType>(op.getInput().getType());
        shapes.push_back({t.getDimSize(1), in.getDimSize(1)});
        names.push_back(op.getWeight().str());
        dims.push_back(llvm::json::Array{t.getDimSize(1), in.getDimSize(1)});
      }
      GemvCost c = estimateGemvGroup(*specOr, M, shapes);
      // Precedence: --force, then the longest matching override, then the policy; "cost"
      // compares the two estimates.
      std::string target = c.pimUs + c.pimBoundaryUs <= c.npuUs ? "pim" : "npu", why = "cost";
      if (specOr->mapping.policy != "cost") {
        target = specOr->mapping.policy;
        why = "policy";
      }
      size_t best = 0;
      for (auto &[key, t] : specOr->mapping.overrides)
        for (hyb::QMatmulOp op : ops)
          if (key.size() > best && op.getWeight().contains(key)) {
            best = key.size();
            target = t;
            why = "override '" + key + "'";
          }
      if (!force.empty()) {
        target = force;
        why = "force";
      }
      if (target != "npu" && target != "pim") {
        module.emitError() << "unknown target '" << target << "'";
        return signalPassFailure();
      }
      (target == "pim" ? pim : npu)++;
      for (hyb::QMatmulOp op : ops) op.setTargetAttr(b.getStringAttr(target));
      table.push_back(llvm::json::Object{{"group", g}, {"weights", std::move(names)}, {"pim_us", c.pimUs},
                                         {"pim_boundary_us", c.pimBoundaryUs}, {"npu_us", c.npuUs},
                                         {"target", target}, {"reason", why}, {"shapes", std::move(dims)}});
    }
    module->setAttr("hyb.partition", b.getStringAttr(llvm::formatv("{0} groups on PIM, {1} on NPU", pim, npu).str()));
    if (!report.empty()) {
      std::error_code ec;
      llvm::raw_fd_ostream os(report, ec);
      if (!ec) os << llvm::formatv("{0:2}", llvm::json::Value(llvm::json::Object{{"tokens", M}, {"groups", std::move(table)}}));
    }
  }
};
} // namespace
