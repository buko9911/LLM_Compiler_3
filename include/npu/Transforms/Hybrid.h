#ifndef NPU_TRANSFORMS_HYBRID_H
#define NPU_TRANSFORMS_HYBRID_H
// NPU + LPDDR6-PIM hybrid compilation (tools/npu-hybrid).
//
// After W8A8 legalization every weight GEMV is an INT8 x INT8 -> INT32 matmul.
// GEMVs that read one quantized activation form a group (q/k/v, gate/up, o,
// down, lm_head). partitionGraph decides per group whether the NPU or the PIM
// computes it; outlinePhases then cuts the step into NPU phases and PIM phases
// that run one after another on one LP6 memory:
//
//   NPU 0 | PIM 0 | NPU 1 | PIM 1 | ... | PIM G-1 | NPU G
//
// PIM group g runs between NPU phase g and g+1. A value that crosses a phase
// boundary lives in DRAM at an address every phase agrees on (a "global").
#include "npu/Target/Result.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include <string>
#include <utility>
#include <vector>

namespace npu {
// Mapping policy and the cost model it uses. Loaded from JSON (configs/hybrid.json);
// every field has the default below.
struct HybridCostModel {
  // NPU GEMV: the weight streamed from LP6 once, or the array, whichever is slower.
  double npuClockMHz = 800, npuReadGBps = 22.51, npuGemvFixedUs = 2.0;
  int npuCores = 4;
  // PIM GEMV: LdMAC streams balanced over the partitions that compute. The LP6 is
  // four LPDDR6 channels: 8 subchannels x 2 ranks (configs/hybrid_1ch.json: one).
  double pimClockMHz = 2400, pimEfficiency = 0.5725, pimPhaseFixedUs = 1.0;
  int pimPUs = 8, pimLanes = 4, pimChannels = 8, pimRanks = 2, pimPartitions = 16, pimLdmacCK = 10;
  // Whether partitions on different ranks compute at the same time. The draft
  // defines no rank-to-rank rule for PIM compute, so a phase uses one rank.
  bool pimConcurrentRanks = false;
  // NPU work at a PIM boundary: the activation written in PIM layout, the INT32
  // results read back.
  double actWriteFixedUs = 1.5, actWriteUsPerKiB = 0.9, readbackFixedUs = 1.5, readbackUsPerKiB = 0.35;
  // "cost" compares the estimates; "npu" or "pim" maps every group there.
  std::string policy = "cost";
  // Weight name substring -> "npu" | "pim"; the longest match wins over the policy.
  std::vector<std::pair<std::string, std::string>> overrides;

  int parallelPartitions() const {
    return pimConcurrentRanks ? pimPartitions : std::min(pimPartitions, pimChannels);
  }
  static Result<HybridCostModel> load(const std::string &path);
};

struct GemvGroupCost { double npuUs = 0, pimUs = 0, boundaryUs = 0; };
// shapes: (N, K) of each GEMV of the group.
GemvGroupCost estimateGemvGroup(const HybridCostModel &model, int64_t tokens,
                                const std::vector<std::pair<int64_t, int64_t>> &shapes);

// Sets npu.target = "npu" | "pim" and npu.group = g on every INT8 matmul.
// names[i] names entry argument i (for overrides and the report). Returns the
// report as JSON text: one entry per group with both estimates and the choice.
mlir::FailureOr<std::string> partitionGraph(mlir::ModuleOp module, const HybridCostModel &model,
                                            const std::vector<std::string> &names);

// A tensor that crosses a phase boundary, or an entry argument.
struct HybridGlobal {
  enum class Kind { Argument, Value, PimOutput };
  Kind kind = Kind::Value;
  std::string name;
  mlir::Type type;          // ranked tensor type
  int64_t argument = -1;    // Argument: the entry argument index
  int64_t producer = -1;    // Value: its NPU phase; PimOutput: its PIM group
  uint64_t bytes = 0, address = 0;
  int64_t packedPanel = 0;  // set when the consuming NPU phase asked for panel packing
};
struct PimGemv {
  int64_t weight = -1;      // entry argument holding the INT8 weight [K, N]
  int64_t n = 0, k = 0;
  std::string name;
  int64_t result = -1;      // global receiving the INT32 [M, N]
};
struct PimGroup {
  int64_t input = -1;       // global holding the INT8 activation [M, K]
  int64_t tokens = 0, k = 0;
  std::vector<PimGemv> gemvs;
};
struct HybridPlan {
  // npuPhases[i] runs, then pimGroups[pimAfter[i]] when that is not -1 (a
  // split only), then npuPhases[i + 1]. Each phase module holds one function
  // whose arguments and results carry npu.global.
  std::vector<mlir::OwningOpRef<mlir::ModuleOp>> npuPhases;
  std::vector<int64_t> pimAfter;
  std::vector<PimGroup> pimGroups;
  std::vector<HybridGlobal> globals;
  std::vector<int64_t> outputs;   // the global of each entry result
};
// Cuts a partitioned, legal module into phases at the PIM groups. names as for
// partitionGraph. splitEvery > 0 also starts a new NPU phase at every
// splitEvery-th GEMV group (group g with g % splitEvery == 0, g > 0) that stays
// on the NPU, e.g. one phase per decoder layer: a simulator program holds at
// most 65,536 commands.
mlir::FailureOr<HybridPlan> outlinePhases(mlir::ModuleOp module, const std::vector<std::string> &names,
                                          int64_t splitEvery = 0);
} // namespace npu
#endif
