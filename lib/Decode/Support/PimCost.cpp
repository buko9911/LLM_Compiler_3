#include "npu/Decode/Support/PimCost.h"
#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <tuple>

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

std::vector<int> shapeBlocks(const TargetSpec &spec, int64_t n, const Shape &shape) {
  const int64_t unit = int64_t(spec.pim.pus) * spec.pim.lanes;
  std::vector<int> blocks;
  if (shape.tailBlocks > 0) {
    const int64_t tailRows = shape.tailBlocks * unit * shape.tailAccs, rows = unit * shape.accs;
    const int64_t main = std::max<int64_t>(0, (n - tailRows + rows - 1) / rows);
    blocks.assign(size_t(main), shape.accs);
    blocks.insert(blocks.end(), size_t(shape.tailBlocks), shape.tailAccs);
  } else {
    blocks.assign(size_t(paddedRows(spec, n, shape.accs) / (unit * shape.accs)), shape.accs);
  }
  return blocks;
}

Estimate estimate(const TargetSpec &spec, int64_t tokens, int64_t k, llvm::ArrayRef<int64_t> n,
                  llvm::ArrayRef<Shape> shapes, int partitions) {
  Estimate e;
  std::vector<int64_t> load(size_t(std::max(1, partitions)), 0);
  const int64_t sweeps = k / spec.pim.kPerSweep, unit = int64_t(spec.pim.pus) * spec.pim.lanes;
  // Both ranks in use: a projection with an even K split keeps each piece on one rank (pim::Program::addProjection),
  // and a rank needs only the input packets of its pieces, when every projection of the group splits so.
  const bool twoRanks = partitions > spec.pim.channels, allOpen = partitions == spec.pim.channels * spec.pim.ranks;
  bool pieceRanks = allOpen;
  for (size_t i = 0; i < n.size(); ++i) pieceRanks &= shapes[i].splitK % spec.pim.ranks == 0;
  for (size_t i = 0; i < n.size(); ++i) {
    const Shape s = shapes[i];
    const std::vector<int> blocks = shapeBlocks(spec, n[i], s);
    const bool onePerRank = allOpen && s.splitK % spec.pim.ranks == 0;
    int64_t nPad = 0;
    for (int a : blocks) nPad += unit * a;
    for (int64_t piece = 0; piece < s.splitK; ++piece)
      for (int a : blocks) {
        const int64_t lo = onePerRank ? (piece % spec.pim.ranks) * spec.pim.channels : 0;
        const int64_t hi = onePerRank ? lo + spec.pim.channels : int64_t(load.size());
        *std::min_element(load.begin() + lo, load.begin() + hi) += streamCycles(spec, tokens, sweeps / s.splitK, a);
      }
    e.gatherUs += double(tokens) * s.splitK * double(nPad) * 4 / (spec.pim.readbackGBps * 1e3);
    e.sumUs += double(tokens) * (s.splitK - 1) * double(nPad) / 16 * 4 /
               (spec.npu.clockMHz * std::max(1, spec.npu.cores));
  }
  e.pimUs = double(*std::max_element(load.begin(), load.end())) / spec.pim.clockMHz;
  if (twoRanks) e.pimUs *= spec.pim.twoRankFactor;
  // Input written as whole columns to every PU of the ranks that read it.
  const double ranks = twoRanks ? (pieceRanks ? 1.0 : double(spec.pim.ranks)) : 1.0;
  e.scatterUs = double(tokens) * double(k) * spec.pim.pus * spec.pim.channels * ranks / (spec.pim.actWriteGBps * 1e3);
  return e;
}

std::vector<Shape> choose(const TargetSpec &spec, int64_t tokens, int64_t k, llvm::ArrayRef<int64_t> n,
                          int partitions, bool allowSplit, Estimate &best) {
  // Equal groups (every layer of a model) get equal answers; the search is exhaustive.
  using Key = std::tuple<int64_t, int64_t, std::vector<int64_t>, int, bool, const TargetSpec *>;
  static std::map<Key, std::pair<std::vector<Shape>, Estimate>> cache;
  Key key{tokens, k, std::vector<int64_t>(n.begin(), n.end()), partitions, allowSplit, &spec};
  if (auto it = cache.find(key); it != cache.end()) {
    best = it->second.second;
    return it->second.first;
  }
  const int64_t sweeps = k / spec.pim.kPerSweep, unit = int64_t(spec.pim.pus) * spec.pim.lanes;
  std::vector<Shape> options;
  for (int accs : spec.pim.streamAccs)
    for (int split = 1; split <= (allowSplit ? spec.pim.maxSplitK : 1); ++split)
      if (sweeps % split == 0) options.push_back({accs, split});
  // The last projection may end in a tail of smaller streams instead of its last 1..partitions blocks.
  std::vector<Shape> lastOptions = options;
  if (!n.empty())
    for (const Shape &base : options) {
      const int64_t rows = unit * base.accs, blocks = (n.back() + rows - 1) / rows;
      for (int64_t drop = 1; drop <= std::min<int64_t>(blocks, partitions); ++drop)
        for (int tail : spec.pim.streamAccs) {
          if (tail >= base.accs) continue;
          const int64_t rest = n.back() - (blocks - drop) * rows;
          if (rest <= 0) continue;
          Shape s = base;
          s.tailAccs = tail;
          s.tailBlocks = (rest + unit * tail - 1) / (unit * tail);
          lastOptions.push_back(s);
        }
    }
  std::vector<Shape> current(n.size()), chosen(n.size(), Shape{});
  double bestUs = std::numeric_limits<double>::infinity();
  int64_t bestCommands = std::numeric_limits<int64_t>::max();
  std::function<void(size_t)> visit = [&](size_t i) {
    if (i == n.size()) {
      Estimate e = estimate(spec, tokens, k, n, current, partitions);
      int64_t commands = 0;
      for (size_t j = 0; j < n.size(); ++j)
        for (int a : shapeBlocks(spec, n[j], current[j]))
          commands += current[j].splitK * (a * 3 / 2 + sweeps / current[j].splitK * 9);
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
    for (const Shape &s : (i + 1 == n.size() ? lastOptions : options)) {
      current[i] = s;
      visit(i + 1);
    }
  };
  if (options.empty()) best = estimate(spec, tokens, k, n, chosen, partitions);
  else visit(0);
  cache[key] = {chosen, best};
  return chosen;
}

} // namespace mlir::hybrid::pimcost
