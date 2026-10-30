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
    // ---- hardware -------------------------------------------------------------------------------
    double clockMHz = 800;
    int cores = 4, saRows = 32, saCols = 32;
    int64_t l1Bytes = 4 << 20, l2Bytes = 8 << 20;
    int64_t completionEventSlots = 65536;  // commands per program (simulator completion events)
    // Cost model of hyb-partition: sustained LP6 read bandwidth and a fixed cost per GEMV.
    double lp6ReadGBps = 19.0, gemvFixedUs = 2.0;

    // ---- memory plan (hyb-npu-memplan, backend) ---------------------------------------------------
    // L1 [4 KiB, 4 KiB + l1Reserved) holds the backend's constant pool and scratch.
    // L2 [0, cores x l2StagingPerCore) holds each core's LP6 <-> L1 staging window; zero sizes the
    // window per phase from the L2 space left after the live L2 buffers.
    int64_t l1ReservedBytes = 256 << 10, l2StagingPerCore = 512 << 10;

    // ---- tiling (hyb-npu-tile and the passes that size tiles) ------------------------------------
    // Weight bytes per GEMV loop tile; ops smaller than distributeMinElements stay on one core.
    int64_t gemvTileBytes = 128 << 10, distributeMinElements = 16384;
    // Dequant distribution threshold; zero inherits distributeMinElements.
    int64_t dequantMinElements = 0;

    // ---- GEMV weight streaming (backend, CodegenLoops.cpp) -----------------------------------------
    // Ring slots in the L2 staging window; zero removes the cap (capacity and trip count still bound it).
    int64_t weightStreamMaxSlots = 8;
    // GEMV tiles per core block; zero uses half the ring.
    int64_t weightPrefetchBatchTiles = 0;
    // Prefetch the first tile of the next batch while the last tile of this batch computes.
    bool weightPrefetchCrossBatch = false;
    // Start the ring at an L2 slot that no earlier access still protects.
    bool weightStreamRotateSlots = false;
    // Release each staged slot as soon as its tile is in L1, then refill it.
    bool weightStreamEarlyRelease = false;

    // ---- optional passes (off by default; each pass is a no-op when its switch is off) -------------
    bool projectionFusion = false;         // hyb-npu-projection-fusion
    bool attentionFusion = false;          // hyb-npu-attention-fusion
    bool gemvEpilogueFusion = false;       // hyb-npu-gemv-epilogue
    bool elementwiseDistribution = false;  // hyb-npu-elementwise-distribute
    bool activationResidency = false;      // hyb-npu-activation-residency
    bool matrixInitElision = false;        // hyb-npu-matrix-init
    bool matmulRhsPacking = false;         // hyb-npu-matmul-layout: pack the transposed RHS
    bool matmulTransposeOutput = false;    // hyb-npu-matmul-layout: transpose A and the output instead
    bool inputTilePipeline = false;        // hyb-npu-input-pipeline
    int64_t inputTileColumns = 128;

    // ---- backend scheduling (off by default) -------------------------------------------------------
    // L2 -> L1 loads and L1 -> L2 stores complete on a local event; the core waits only when an
    // instruction touches the destination (loads) or the source (stores).
    bool asyncInputLoads = false;
    bool asyncOutputStores = false;
    // false: wait right after every async store (same transport, no vector work in between).
    bool asyncOutputStoreVectorOverlap = true;
    // End a staged input-copy block before unrelated compute so its staging slot frees early.
    bool stagingEarlyRelease = false;
    // Use exact L1 read/write ranges for matrix / output-DMA hazards.
    bool memoryDependencySchedule = false;
    // Reuse a row-invariant scalar reciprocal across vector chunks; interleave independent chunks.
    bool vectorScalarReuse = false, vectorChunkPipeline = false;
    // Per-core GDMA commands issued ahead within a phase; zero is unlimited.
    int64_t gdmaLookahead = 0;
    // Reorder system commands by their first consumer, keeping every dependency.
    bool commandSchedule = false;
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
