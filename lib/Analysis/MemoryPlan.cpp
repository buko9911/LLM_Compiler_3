#include "npu/Analysis/MemoryPlan.h"
#include <algorithm>
#include <limits>
#include <set>

namespace npu {
namespace {
bool overlap(uint64_t a, uint64_t ae, uint64_t b, uint64_t be) { return a < be && b < ae; }
uint64_t capacity(MemorySpace s, const MemoryLimits &l) {
  return s == MemorySpace::L1 ? l.l1 : s == MemorySpace::L2 ? l.l2 : s == MemorySpace::DRAM ? l.dram : 0;
}
bool sameStorage(const BufferPlacement &a, const BufferPlacement &b) {
  return a.space == b.space && (a.space != MemorySpace::L1 || a.requirement.core == b.requirement.core);
}
std::optional<uint64_t> findAddress(const BufferPlacement &candidate,
                                   const MemoryPlan &plan, uint64_t cap) {
  const auto &r = candidate.requirement;
  std::vector<std::pair<uint64_t,uint64_t>> occupied;
  for (const auto &p : plan.buffers) {
    if (p.action == BufferPlacement::Action::Rematerialize || !sameStorage(p, candidate)) continue;
    if (overlap(r.begin, r.end, p.requirement.begin, p.requirement.end))
      occupied.push_back({p.address, p.address + p.requirement.bytes});
  }
  std::sort(occupied.begin(), occupied.end());
  uint64_t address = 0;
  auto aligned = [&](uint64_t n) -> std::optional<uint64_t> {
    if (n > UINT64_MAX - (r.alignment - 1)) return {};
    return (n + r.alignment - 1) & ~(r.alignment - 1);
  };
  for (auto range : occupied) {
    auto a = aligned(address); if (!a) return {};
    if (*a <= range.first && r.bytes <= range.first - *a) return a;
    address = std::max(address, range.second);
  }
  auto a = aligned(address);
  if (a && *a <= cap && r.bytes <= cap - *a) return a;
  return {};
}
uint64_t spillCost(const BufferRequirement &r, const MemoryLimits &l) {
  auto satAdd = [](uint64_t a, uint64_t b) { return a > UINT64_MAX - b ? UINT64_MAX : a + b; };
  auto count = satAdd(r.reloads, r.stores);
  auto cycles = satAdd(r.bytes / l.dramBytesPerCycle + (r.bytes % l.dramBytesPerCycle != 0), l.dmaSetupCycles);
  if (cycles && count > UINT64_MAX / cycles) return UINT64_MAX;
  return count * cycles;
}
}
Result<MemoryPlan> planMemory(std::vector<BufferRequirement> requirements, const MemoryLimits &limits) {
  using R = Result<MemoryPlan>;
  if (!limits.cores || limits.cores > 65536 || !limits.dramBytesPerCycle) return R::failure("invalid memory limits");
  MemoryPlan plan; plan.l1HighWater.resize(limits.cores);
  std::set<std::string> names;
  for (const auto &r : requirements) {
    if (r.name.empty() || !names.insert(r.name).second) return R::failure("buffer names must be unique and nonempty");
    if (!r.bytes || r.begin >= r.end || !r.alignment || (r.alignment & (r.alignment-1)) || r.core >= limits.cores)
      return R::failure("invalid buffer requirement: " + r.name);
    if (r.space == MemorySpace::Accumulator || unsigned(r.space) > 3)
      return R::failure("accumulator is dedicated state, not an allocatable buffer");
  }
  std::stable_sort(requirements.begin(), requirements.end(), [](const auto &a, const auto &b) {
    return a.begin < b.begin;
  });
  for (const auto &r : requirements) {
    BufferPlacement p{r, r.space, 0, BufferPlacement::Action::Resident};
    auto address = findAddress(p, plan, capacity(p.space, limits));
    if (!address) {
      bool remat = r.pure && r.inputsAvailable && r.preservesNumerics;
      bool spill = r.spillable && r.space != MemorySpace::DRAM;
      if (remat && (!spill || r.rematerializeCycles < spillCost(r, limits))) {
        p.action = BufferPlacement::Action::Rematerialize;
      } else if (spill) {
        p.space = MemorySpace::DRAM;
        address = findAddress(p, plan, limits.dram);
        if (address) p.action = BufferPlacement::Action::Spill;
        else if (remat) p.action = BufferPlacement::Action::Rematerialize;
        else return R::failure("DRAM spill capacity exhausted: " + r.name);
      } else return R::failure("memory capacity exhausted: " + r.name);
    }
    if (p.action != BufferPlacement::Action::Rematerialize) {
      p.address = *address;
      auto end = p.address + r.bytes;
      if (p.space == MemorySpace::L1) plan.l1HighWater[r.core] = std::max(plan.l1HighWater[r.core], end);
      else if (p.space == MemorySpace::L2) plan.l2HighWater = std::max(plan.l2HighWater, end);
      else plan.dramHighWater = std::max(plan.dramHighWater, end);
    }
    plan.buffers.push_back(std::move(p));
  }
  auto valid = verifyMemoryPlan(plan, limits);
  if (!valid) return R::failure(valid.error);
  return R::success(std::move(plan));
}
Result<bool> verifyMemoryPlan(const MemoryPlan &plan, const MemoryLimits &limits) {
  using R = Result<bool>;
  if (plan.l1HighWater.size() != limits.cores) return R::failure("missing per-core high water marks");
  uint64_t l2HighWater = 0, dramHighWater = 0;
  std::vector<uint64_t> l1HighWater(limits.cores);
  std::set<std::string> names;
  for (size_t i = 0; i < plan.buffers.size(); ++i) {
    const auto &a = plan.buffers[i]; const auto &r = a.requirement;
    if (r.name.empty() || !names.insert(r.name).second) return R::failure("invalid buffer identity");
    if (!r.bytes || !r.alignment || (r.alignment & (r.alignment-1)) || r.begin >= r.end || r.core >= limits.cores)
      return R::failure("invalid buffer contract");
    if (a.action == BufferPlacement::Action::Rematerialize) {
      if (!r.pure || !r.inputsAvailable || !r.preservesNumerics) return R::failure("unsafe rematerialization");
      continue;
    }
    if (a.action == BufferPlacement::Action::Spill && (!r.spillable || a.space != MemorySpace::DRAM))
      return R::failure("invalid spill placement");
    if (a.action == BufferPlacement::Action::Resident && a.space != r.space)
      return R::failure("resident buffer changed memory space");
    auto cap = capacity(a.space, limits);
    if (a.address % r.alignment || a.address > cap || r.bytes > cap - a.address)
      return R::failure("misaligned or out-of-capacity placement: " + r.name);
    uint64_t end = a.address + r.bytes;
    if (a.space == MemorySpace::L1) l1HighWater[r.core] = std::max(l1HighWater[r.core], end);
    else if (a.space == MemorySpace::L2) l2HighWater = std::max(l2HighWater, end);
    else dramHighWater = std::max(dramHighWater, end);
    for (size_t j = 0; j < i; ++j) {
      const auto &b = plan.buffers[j];
      if (b.action == BufferPlacement::Action::Rematerialize || !sameStorage(a,b)) continue;
      if (overlap(r.begin,r.end,b.requirement.begin,b.requirement.end) &&
          overlap(a.address,a.address+r.bytes,b.address,b.address+b.requirement.bytes))
        return R::failure("simultaneously live buffers overlap: " + r.name + ", " + b.requirement.name);
    }
  }
  if (l1HighWater != plan.l1HighWater || l2HighWater != plan.l2HighWater || dramHighWater != plan.dramHighWater)
    return R::failure("incorrect high water marks");
  return R::success(true);
}
} // namespace npu
