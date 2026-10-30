//===- CommandSchedule.cpp - system command order of one NPU phase --------===//
//
// ProgramBuilder::schedulePhase (npu.command_schedule), run after the address-range dependencies
// of a phase are built:
//   1. list-schedule the phase's commands in dependency order, preferring the command whose
//      nearest core-block consumer has the smallest accumulated instruction depth (core blocks
//      before DMA on ties); keep the original order if an edge would no longer fit the 16-bit
//      ISA 2.0 dependency distance;
//   2. drop a dependency only when another dependency of the same command provably reaches it.
//===----------------------------------------------------------------------===//

#include "npu/Decode/Target/PlenaProgram.h"
#include <algorithm>
#include <cassert>
#include <limits>
#include <queue>
#include <tuple>

using namespace mlir::hybrid::plena;

void ProgramBuilder::schedulePhase() {
  const size_t count = cmds.size() - phaseStart;
  if (count < 2) return;
  lastSchedule.commands = count;
  std::vector<uint64_t> depth(count, 0), consumer(count, std::numeric_limits<uint64_t>::max());
  std::vector<std::vector<size_t>> successors(count);
  std::vector<size_t> pending(count, 0);
  // Instruction depth is only an ordering heuristic, not a cycle estimate.
  // Include all original edges, including the bounded GDMA admission chains.
  for (size_t i = 0; i < count; ++i) {
    const auto &c = cmds[phaseStart + i];
    for (int d : c.deps) {
      assert(d >= 0 && size_t(d) < phaseStart + i);
      if (size_t(d) < phaseStart) continue;
      size_t parent = size_t(d) - phaseStart;
      depth[i] = std::max(depth[i], depth[parent]);
      successors[parent].push_back(i);
      ++pending[i];
    }
    if (c.kind == Kind::CoreBlock) {
      depth[i] += std::max<int64_t>(1, c.isaCount);
      consumer[i] = depth[i];
    }
  }
  for (size_t i = count; i-- > 0;)
    for (size_t next : successors[i]) consumer[i] = std::min(consumer[i], consumer[next]);
  using Ready = std::tuple<uint64_t, unsigned, size_t>;
  std::priority_queue<Ready, std::vector<Ready>, std::greater<Ready>> ready;
  auto enqueue = [&](size_t i) {
    const auto kind = cmds[phaseStart + i].kind;
    ready.emplace(consumer[i], kind == Kind::CoreBlock ? 0 : 1, i);
  };
  for (size_t i = 0; i < count; ++i) if (!pending[i]) enqueue(i);
  std::vector<size_t> order;
  while (!ready.empty()) {
    auto [priority, kind, i] = ready.top();
    ready.pop();
    order.push_back(i);
    for (size_t next : successors[i]) if (!--pending[next]) enqueue(next);
  }
  assert(order.size() == count && order.back() == count - 1);
  std::vector<int> remap(count);
  for (size_t i = 0; i < count; ++i) remap[order[i]] = int(phaseStart + i);
  // ISA v2 encodes a dependency as a 16-bit backward command distance.
  // Reordering can lengthen an otherwise valid edge. Keep the original DAG
  // if any proposed edge becomes unencodable, before moving any command.
  for (size_t i = 0; i < count; ++i)
    for (int d : cmds[phaseStart + order[i]].deps) {
      size_t mapped = size_t(d) < phaseStart ? size_t(d) : size_t(remap[size_t(d) - phaseStart]);
      if (phaseStart + i - mapped >= (size_t(1) << 16)) {
        lastSchedule.encodingFallback = true;
        return;
      }
    }
  lastRemap = remap;
  std::vector<Command> sorted;
  sorted.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    if (order[i] != i) ++lastSchedule.reordered;
    auto c = std::move(cmds[phaseStart + order[i]]);
    for (int &d : c.deps) if (size_t(d) >= phaseStart) d = remap[size_t(d) - phaseStart];
    std::sort(c.deps.begin(), c.deps.end());
    sorted.push_back(std::move(c));
  }
  std::move(sorted.begin(), sorted.end(), cmds.begin() + phaseStart);

  // Drop an edge only after proving another dependency reaches it. The search is bounded; past
  // the budget the edge is kept.
  std::vector<unsigned> seen(cmds.size(), 0);
  unsigned generation = 0;
  for (size_t id = phaseStart; id < cmds.size(); ++id) {
    auto &deps = cmds[id].deps;
    if (deps.size() < 2) continue;
    ++generation;
    std::vector<int> keep, stack;
    size_t visited = 0;
    const int minimum = deps.front();
    for (auto it = deps.rbegin(); it != deps.rend(); ++it) {
      int dependency = *it;
      if (seen[size_t(dependency)] == generation) {
        ++lastSchedule.redundantEdges;
        continue;
      }
      keep.push_back(dependency);
      stack.push_back(dependency);
      while (!stack.empty() && visited < 4096) {
        int node = stack.back(); stack.pop_back();
        if (seen[size_t(node)] == generation) continue;
        seen[size_t(node)] = generation;
        ++visited;
        for (int ancestor : cmds[size_t(node)].deps)
          if (ancestor >= minimum && seen[size_t(ancestor)] != generation) stack.push_back(ancestor);
      }
      stack.clear();
    }
    std::reverse(keep.begin(), keep.end());
    deps = std::move(keep);
  }
}
