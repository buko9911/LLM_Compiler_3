#ifndef PLENA_ANALYSIS_MEMORYPLAN_H
#define PLENA_ANALYSIS_MEMORYPLAN_H
#include "plena/Target/Program.h"
#include <array>
#include <optional>
#include <string>
#include <vector>

namespace plena {
// Event times are in a proven execution order. For asynchronous accesses end
// must be the completion event, never just the issue event. Lifetime is [begin,end).
struct BufferRequirement {
  std::string name;
  MemorySpace space = MemorySpace::L2;
  unsigned core = 0;
  uint64_t bytes = 0, alignment = 64, begin = 0, end = 0;
  bool spillable = false;
  bool pure = false, inputsAvailable = false, preservesNumerics = false;
  uint64_t reloads = 1, stores = 1;
  uint64_t rematerializeCycles = 0;
};
struct MemoryLimits {
  uint64_t l1 = 0, l2 = 0, dram = 0;
  unsigned cores = 1;
  uint64_t dramBytesPerCycle = 1, dmaSetupCycles = 0;
};
struct BufferPlacement {
  enum class Action { Resident, Spill, Rematerialize };
  BufferRequirement requirement;
  MemorySpace space = MemorySpace::L2;
  uint64_t address = 0;
  Action action = Action::Resident;
};
struct MemoryPlan {
  std::vector<BufferPlacement> buffers;
  uint64_t l2HighWater = 0, dramHighWater = 0;
  std::vector<uint64_t> l1HighWater;
};
// Allocation and fallback selection are one decision. Actions still need to be
// realized by the placement pass; this analysis alone does not insert transfers.
Result<MemoryPlan> planMemory(std::vector<BufferRequirement> requirements,
                              const MemoryLimits &limits);
Result<bool> verifyMemoryPlan(const MemoryPlan &, const MemoryLimits &);
} // namespace plena
#endif
