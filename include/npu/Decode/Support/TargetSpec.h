#ifndef HYBRID_SUPPORT_TARGETSPEC_H
#define HYBRID_SUPPORT_TARGETSPEC_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace mlir::hybrid {

/// Target description (configs/target.json): NPU (PLENA) and PIM (LPDDR6-PIM) parameters,
/// plus the calibration constants of the cost model.
struct TargetSpec {
  struct Npu {
    double clockMHz = 800;
    int cores = 4, saRows = 32, saCols = 32;
    int64_t l1Bytes = 4 << 20, l2Bytes = 8 << 20;
    double lp6ReadGBps = 19.0, gemvFixedUs = 2.0;
    // Memory plan: L1 [4 KiB, 4 KiB + l1Reserved) holds the backend's constant pool and scratch;
    // L2 [0, cores x l2StagingPerCore) holds each core's LP6 <-> L1 staging window.
    // Zero selects a per-phase window from the space left after planning live L2 buffers.
    int64_t l1ReservedBytes = 256 << 10, l2StagingPerCore = 512 << 10;
    // Zero removes the software ring cap (capacity and loop trip count still bound it).
    int64_t weightStreamMaxSlots = 8;
    // Zero uses half the ring; positive values bound the number of GEMV tiles per block.
    int64_t weightPrefetchBatchTiles = 0;
    bool weightPrefetchCrossBatch = false;
    // Tiling: weight bytes per GEMV loop tile; smallest op split over the cores.
    int64_t gemvTileBytes = 128 << 10, distributeMinElements = 16384;
    // Commands per program (baseline_settings completion_event_slots).
    int64_t completionEventSlots = 65536;
  } npu;
  struct Pim {
    double clockMHz = 2400;
    int pus = 8, lanes = 4, channels = 2, ranks = 2, partitions = 4, nlldCK = 10;
    /// Whether PIM partitions on different ranks compute at the same time. The LPDDR6-PIM draft
    /// defines no rank-to-rank rule for PIM compute commands; when false, each PIM phase runs on
    /// the subchannels of one rank and the other rank only holds data.
    bool concurrentRanks = true;
    /// Partitions that compute at the same time.
    int parallelPartitions() const { return concurrentRanks ? partitions : std::min(partitions, channels); }
    double efficiency = 0.485, phaseFixedUs = 1.0;
    double actWriteUsPerKiB = 0.9, actWriteFixedUs = 1.5, readbackUsPerKiB = 0.35, readbackFixedUs = 1.5;
  } pim;
  struct Lp6 {
    int64_t capacityBytes = int64_t(16) << 30, rowBytes = 65536;
  } lp6;
  /// GEMV mapping: "cost" (cost model per group), "npu" or "pim" (every group); overrides map a
  /// substring of a weight name to a target and win over the policy (first match).
  struct Mapping {
    std::string policy = "cost";
    std::vector<std::pair<std::string, std::string>> overrides;
  } mapping;

  static FailureOr<TargetSpec> load(llvm::StringRef path, std::string &error);
};

/// One projection of a GEMV group: N output rows, K inputs, INT8 weights.
struct GemvShape {
  int64_t n, k;
};

/// Cost model (microseconds) of one group of GEMVs sharing an input, M tokens.
struct GemvCost {
  double pimUs, pimBoundaryUs, npuUs;
};
GemvCost estimateGemvGroup(const TargetSpec &spec, int64_t tokens, llvm::ArrayRef<GemvShape> group);

} // namespace mlir::hybrid

#endif
