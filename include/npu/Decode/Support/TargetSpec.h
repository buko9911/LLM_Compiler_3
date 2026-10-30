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
    // Start a weight ring at an untouched L2 slot when earlier scratch accesses
    // still protect slot zero. Capacity and all address hazards are unchanged.
    bool weightStreamRotateSlots = false;
    // Retire each staged read before its tile's compute, then refill the slot.
    // Uses existing block completion events; no new DMA/ISA resource.
    bool weightStreamEarlyRelease = false;
    // Tiling: weight bytes per GEMV loop tile; smallest op split over the cores.
    int64_t gemvTileBytes = 128 << 10, distributeMinElements = 16384;
    // Independent dequant distribution threshold; zero inherits distributeMinElements.
    // This permits PIM epilogue tuning without changing GEMV core partitioning.
    int64_t dequantMinElements = 0;
    // Keep phase-local activations separate from externally visible results.
    // Run destination privatization immediately before bufferization, after CSE.
    bool activationResidency = false;
    // Normalize attention reshape boundaries so head-local producer fusion applies.
    // This preserves the full softmax reduction; it is not online FlashAttention.
    bool attentionFusion = false;
    // Preserve INT32 tiles on their producing core until FP16 dequantization.
    bool gemvEpilogueFusion = false;
    // Concatenate compatible same-input projections offline, then slice results.
    bool projectionFusion = false;
    // Prove that a matrix INIT overwrites a private zero-filled destination.
    bool matrixInitElision = false;
    // One event-tagged L1 -> L2 store per core; independent vector work may
    // continue until a source overwrite, another DMA, matrix work or block exit.
    bool asyncOutputStores = false;
    // Common L2 -> L1 event scheduling, guarded by physical destination ranges.
    bool asyncInputLoads = false;
    // Dependency-driven copy motion before memory planning; not operator-name based.
    bool transferSchedule = false;
    // End a normal staged input-copy block before unrelated following compute.
    // CORE_BEGIN resets registers, so this is legal only at buffer-op boundaries.
    bool stagingEarlyRelease = false;
    // Use actual L1 read/write ranges for matrix/output-DMA hazards.
    bool memoryDependencySchedule = false;
    // Split sufficiently large independent elementwise tensors across existing
    // cores. Tensor tiling preserves each element's arithmetic and rounding.
    bool elementwiseDistribution = false;
    // Reuse a row-invariant scalar reciprocal across legacy vector chunks.
    bool vectorScalarReuse = false;
    // Interleave independent legacy vector chunks within the existing register
    // file; unsafe aliases, reductions and register pressure retain serial code.
    bool vectorChunkPipeline = false;
    // Per-core, phase-local GDMA admission cap; 0 retains unlimited lookahead.
    int64_t gdmaLookahead = 0;
    // Order system records by their first consumer; retain all memory hazards.
    bool commandSchedule = false;
    // Common producer/consumer tiling; full K reduction remains unchanged.
    bool inputTilePipeline = false;
    int64_t inputTileColumns = 128;
    // Pack transpose-B matmul RHS into an SA-friendly KxN layout.
    bool matmulRhsPacking = false;
    // Prefer packing A and transposing the output when it moves fewer elements.
    bool matmulTransposeOutput = false;
    // False is an immediate-wait control with the same async DMA transport.
    bool asyncInputLoadOverlap = true;
    // Early-release stream diagnostic: omit next-tile prefetch/dependencies.
    bool weightPrefetchComputeFirst = false;
    // Diagnostic ablation: false waits immediately after each async store,
    // separating asynchronous DMA implementation/timing from VPU overlap.
    bool asyncOutputStoreVectorOverlap = true;
    // Online attention uses bounded Q/K tiles and FP32 normalization/output state.
    // Flash-Decoding additionally distributes KV splits and merges their states.
    // Opt-in: its rounding differs from materialized FP16 softmax.
    bool flashAttention = false, flashDecoding = false;
    // Stream compatible [past KV, new KV] concatenations without a full L2 copy.
    bool flashKvSlicing = false;
    int64_t attentionQueryTile = 32, attentionKeyTile = 128, attentionSplits = 1;
    // Schedule independent dequant chunks inside subsequent GEMV issue windows.
    bool overlapDequantGemv = false;
    int64_t overlapDequantChunkElements = 128;
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
