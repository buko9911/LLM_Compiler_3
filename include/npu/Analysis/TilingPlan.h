#ifndef NPU_ANALYSIS_TILINGPLAN_H
#define NPU_ANALYSIS_TILINGPLAN_H
#include "npu/Analysis/MemoryPlan.h"
#include "npu/Target/HardwareConfig.h"
#include <vector>
namespace npu {
enum class Schedule { ResidentPanels, Hybrid, Streaming, PartialSumVPU };
enum class NumericalPolicy { Preserve, AllowFP16Rounding };
struct MatmulShape { uint64_t m = 0, n = 0, k = 0; };
struct Tile { uint64_t m = 0, n = 0, k = 0; };
struct TilingPlan {
  Schedule schedule = Schedule::ResidentPanels;
  Tile l2, l1, array;
  unsigned coreSplit = 1, stages = 1;
  uint64_t estimatedCycles = 0, dramTrafficBytes = 0, localTrafficBytes = 0;
  NumericalPolicy numerical = NumericalPolicy::Preserve;
  std::vector<BufferRequirement> memory;
  MemoryPlan placement;
};
struct TilingRequest {
  MatmulShape shape;
  HardwareConfig hardware;
  uint64_t graphResidentBytes = 0;
  NumericalPolicy numerical = NumericalPolicy::Preserve;
  // How many L2 stagings are live at once. Two lets the transfer for the next
  // panel run while the array is still on this one, and costs twice the room.
  unsigned stages = 1;
  // Rank single staging alongside the requested count, rather than only using
  // it as a capacity fallback. Graph lowering currently requests at most two.
  bool searchStages = false;
};
// FP16 A/B, FP32 accumulator, one final FP16 conversion. No C initial value.
// These are planning candidates, not an implicit permission to change source
// arithmetic. Graph legalization must prove this numerical contract separately.
Result<std::vector<TilingPlan>> enumerateTilingPlans(const TilingRequest &request);
Result<TilingPlan> chooseTilingPlan(const TilingRequest &request);
std::string describe(const TilingPlan &plan);
} // namespace npu
#endif
