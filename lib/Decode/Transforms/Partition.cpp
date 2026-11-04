//===- Partition.cpp - hyb-partition ----------------------------------------===//
//
// For every group of qmatmuls reading one quantized input, compare the PIM cost (the fastest
// stream layout of npu/Decode/Support/PimCost.h, plus the NPU boundary work: activation written
// in PIM layout, INT32 outputs and split-K partial sums read back) with the NPU cost (weights
// streamed from LP6 at the sustained GDMA rate) and set the `target` attribute. The group is the
// decision unit because its members share one activation broadcast (PIM) or one weight stream
// pass (NPU). The groups of one layer are chosen together: a PIM group between two NPU groups of
// the layer splits the NPU weight stream, which costs npu_run_split_us (measured).
//===----------------------------------------------------------------------===//

#include "npu/Decode/Transforms/PassSupport.h"
#include "llvm/ADT/STLExtras.h"
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
    // One decision per group, made per layer: the groups of a layer whose target comes from the cost
    // model are chosen together, because a PIM group between two NPU groups of the same layer splits that
    // layer's NPU weight stream into two phases. Each extra NPU run of a layer costs npu_run_split_us,
    // measured: NPU work at the split that its own weight stream no longer hides.
    auto layerOf = [](StringRef weight) -> int64_t {
      size_t at = weight.find("layers.");
      if (at == StringRef::npos) return -1;
      int64_t n = -1;
      if (weight.drop_front(at + 7).take_while(llvm::isDigit).getAsInteger(10, n)) return -1;
      return n;
    };
    struct Choice {
      int64_t id, rows, layer;
      GemvCost c;
      std::string target, why;
      llvm::json::Array names, dims;
    };
    std::vector<Choice> list;
    for (auto &[g, ops] : groups) {
      SmallVector<GemvShape> shapes;
      Choice x;
      x.id = g;
      for (hyb::QMatmulOp op : ops) {
        auto t = cast<ShapedType>(op.getInit().getType());
        auto in = cast<ShapedType>(op.getInput().getType());
        shapes.push_back({t.getDimSize(1), in.getDimSize(1)});
        x.names.push_back(op.getWeight().str());
        x.dims.push_back(llvm::json::Array{t.getDimSize(1), in.getDimSize(1)});
      }
      x.layer = layerOf(ops.front().getWeight());
      // Tokens of this group: the rows of its input (the LM head of a prefill reads one row).
      x.rows = cast<ShapedType>(ops.front().getInput().getType()).getDimSize(0);
      if (x.rows <= 0) x.rows = M;
      x.c = estimateGemvGroup(*specOr, x.rows, shapes);
      // Precedence: --force, then the longest matching override, then the policy; "cost"
      // compares the two estimates.
      x.target = x.c.pimUs + x.c.pimBoundaryUs <= x.c.npuUs ? "pim" : "npu";
      x.why = "cost";
      if (specOr->mapping.policy != "cost") {
        x.target = specOr->mapping.policy;
        x.why = "policy";
      }
      size_t best = 0;
      for (auto &[key, t] : specOr->mapping.overrides)
        for (hyb::QMatmulOp op : ops)
          if (key.size() > best && op.getWeight().contains(key)) {
            best = key.size();
            x.target = t;
            x.why = "override '" + key + "'";
          }
      if (!force.empty()) {
        x.target = force;
        x.why = "force";
      }
      if (x.target != "npu" && x.target != "pim") {
        module.emitError() << "unknown target '" << x.target << "'";
        return signalPassFailure();
      }
      list.push_back(std::move(x));
    }
    std::map<int64_t, SmallVector<size_t>> byLayer;   // program order inside a layer
    for (size_t i = 0; i < list.size(); ++i)
      if (list[i].layer >= 0) byLayer[list[i].layer].push_back(i);
    double splitUs = 0;
    for (auto &[layer, members] : byLayer) {
      SmallVector<size_t> free;
      for (size_t i : members)
        if (list[i].why == "cost") free.push_back(i);
      if (free.empty() || free.size() > 12) continue;
      splitUs = specOr->pim.npuRunSplitUs;
      auto layerUs = [&](uint32_t mask) {
        double us = 0;
        int runs = 0;
        bool prevNpu = false;
        for (size_t i : members) {
          auto at = llvm::find(free, i);
          bool npu = at != free.end() ? !((mask >> (at - free.begin())) & 1) : list[i].target == "npu";
          us += npu ? list[i].c.npuUs : list[i].c.pimUs + list[i].c.pimBoundaryUs;
          if (npu && !prevNpu) runs++;
          prevNpu = npu;
        }
        return us + splitUs * std::max(0, runs - 1);
      };
      uint32_t incumbent = 0;   // bit set = PIM, start from the per-group answer
      for (size_t f = 0; f < free.size(); ++f)
        if (list[free[f]].target == "pim") incumbent |= 1u << f;
      uint32_t chosen = incumbent;
      double bestUs = layerUs(incumbent);
      for (uint32_t mask = 0; mask < (1u << free.size()); ++mask)
        if (double us = layerUs(mask); us < bestUs * (1 - 1e-9)) {
          bestUs = us;
          chosen = mask;
        }
      for (size_t f = 0; f < free.size(); ++f) {
        std::string t = (chosen >> f) & 1 ? "pim" : "npu";
        if (t != list[free[f]].target) list[free[f]].why = "cost, layer (NPU run split)";
        list[free[f]].target = t;
      }
    }
    llvm::json::Array table;
    Builder b(&getContext());
    int pim = 0, npu = 0;
    for (Choice &x : list) {
      (x.target == "pim" ? pim : npu)++;
      for (hyb::QMatmulOp op : groups[x.id]) op.setTargetAttr(b.getStringAttr(x.target));
      table.push_back(llvm::json::Object{{"group", x.id}, {"weights", std::move(x.names)}, {"tokens", x.rows},
                                         {"layer", x.layer}, {"pim_us", x.c.pimUs},
                                         {"pim_boundary_us", x.c.pimBoundaryUs}, {"npu_us", x.c.npuUs},
                                         {"target", x.target}, {"reason", x.why}, {"shapes", std::move(x.dims)}});
    }
    module->setAttr("hyb.partition", b.getStringAttr(llvm::formatv("{0} groups on PIM, {1} on NPU", pim, npu).str()));
    if (!report.empty()) {
      std::error_code ec;
      llvm::raw_fd_ostream os(report, ec);
      if (!ec)
        os << llvm::formatv("{0:2}", llvm::json::Value(llvm::json::Object{
                                         {"tokens", M}, {"npu_run_split_us", splitUs}, {"groups", std::move(table)}}));
    }
  }
};
} // namespace
