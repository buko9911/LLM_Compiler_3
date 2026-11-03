#include "npu/Decode/Support/PimCost.h"
#include <algorithm>
#include <functional>
#include <limits>

namespace mlir::hybrid::pimcost {

int phasePartitions(const TargetSpec &spec) { return spec.pim.parallelPartitions(); }

int64_t paddedRows(const TargetSpec &spec, int64_t n, int accs) {
  const int64_t rows = int64_t(spec.pim.pus) * spec.pim.lanes * accs;
  return (n + rows - 1) / rows * rows;
}

int64_t streamCycles(const TargetSpec &spec, int64_t tokens, int64_t pieceSweeps, int accs) {
  const auto &p = spec.pim;
  const double sweep = 32.0 * accs * p.ldmacCK + p.sweepCK;
  return int64_t(double(tokens) * (double(pieceSweeps) * sweep + p.streamCK) + 0.5);
}

Estimate estimate(const TargetSpec &spec, int64_t tokens, int64_t k, llvm::ArrayRef<int64_t> n,
                  llvm::ArrayRef<Shape> shapes, int partitions) {
  Estimate e;
  std::vector<int64_t> load(size_t(std::max(1, partitions)), 0);
  const int64_t sweeps = k / 256;
  for (size_t i = 0; i < n.size(); ++i) {
    const Shape s = shapes[i];
    const int64_t nPad = paddedRows(spec, n[i], s.accs);
    const int64_t blocks = nPad / (int64_t(spec.pim.pus) * spec.pim.lanes * s.accs);
    const int64_t cycles = streamCycles(spec, tokens, sweeps / s.splitK, s.accs);
    for (int64_t g = 0; g < blocks * s.splitK; ++g) *std::min_element(load.begin(), load.end()) += cycles;
    e.gatherUs += double(tokens) * s.splitK *
                  (double(nPad) * 4 / (spec.pim.readbackGBps * 1e3) + spec.pim.readbackPieceUs);
    e.sumUs += double(tokens) * (s.splitK - 1) * double(nPad) / 16 * 4 /
               (spec.npu.clockMHz * std::max(1, spec.npu.cores));
  }
  e.pimUs = double(*std::max_element(load.begin(), load.end())) / spec.pim.clockMHz;
  return e;
}

std::vector<Shape> choose(const TargetSpec &spec, int64_t tokens, int64_t k, llvm::ArrayRef<int64_t> n,
                          int partitions, bool allowSplit, Estimate &best) {
  const int64_t sweeps = k / 256;
  std::vector<Shape> options;
  for (int accs : spec.pim.streamAccs)
    for (int split = 1; split <= (allowSplit ? spec.pim.maxSplitK : 1); ++split)
      if (sweeps % split == 0) options.push_back({accs, split});
  std::vector<Shape> current(n.size()), chosen(n.size(), Shape{});
  double bestUs = std::numeric_limits<double>::infinity();
  int64_t bestCommands = std::numeric_limits<int64_t>::max();
  // Exhaustive over the group (at most a few projections share one input).
  std::function<void(size_t)> visit = [&](size_t i) {
    if (i == n.size()) {
      Estimate e = estimate(spec, tokens, k, n, current, partitions);
      int64_t commands = 0;
      for (size_t j = 0; j < n.size(); ++j) {
        const int64_t blocks = paddedRows(spec, n[j], current[j].accs) /
                               (int64_t(spec.pim.pus) * spec.pim.lanes * current[j].accs);
        const int64_t pieceSweeps = sweeps / current[j].splitK;
        commands += blocks * current[j].splitK * (current[j].accs * 3 / 2 + pieceSweeps * 9);
      }
      // A relative margin keeps the simpler layout on estimates within rounding of each other.
      const double us = e.total();
      if (us < bestUs * (1 - 1e-9) || (us <= bestUs * (1 + 1e-9) && commands < bestCommands)) {
        bestUs = us;
        bestCommands = commands;
        chosen = current;
        best = e;
      }
      return;
    }
    for (const Shape &s : options) {
      current[i] = s;
      visit(i + 1);
    }
  };
  if (options.empty()) {
    best = estimate(spec, tokens, k, n, chosen, partitions);
    return chosen;
  }
  visit(0);
  return chosen;
}

} // namespace mlir::hybrid::pimcost
