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

    // ---- timing of the NPU itself (NpuCost.h; the simulator's model, docs/architecture/NPU_SIMULATOR.md) --
    // The passes compare these costs to choose between equivalent programs; every choice is a function
    // of the tensor shapes and these numbers, never of the step kind.
    int64_t l1ReadBytesPerCycle = 128;     // L1: 32 banks x 4 B read ports (one shared by M_LD and V_LD)
    int64_t l1WriteBytesPerCycle = 128;    // and 32 x 4 B write ports
    int64_t mmaKPerCycle = 2;              // INT8 MACs per PE per cycle: a 32-deep K piece feeds in 16 cycles
    int64_t mmaPipelineCycles = 2;         // INT8 pipeline latency; a chain drains in rows + cols - 3 + this
    int64_t vectorLanes = 16;              // 32-bit lanes: FP32 elements per cycle of a lane ALU op
    int64_t vectorExpPerCycle = 4;         // shared SFUs: EXP, reciprocal, (r)sqrt elements per cycle
    int64_t vectorRcpPerCycle = 2, vectorSqrtPerCycle = 2;
    // Sustained LP6 -> L2 read of all cores together: a program's GDMA reads replayed without compute.
    double gdmaReadGBps = 92.9;
    // Fixed cost of running one op on several cores instead of one (dispatch, the cross-core events
    // and the per-core copies of its operands): measured, see npu.core_split_us in the target files.
    double coreSplitUs = 0.5;

    // ---- memory plan (hyb-npu-memplan, backend) ---------------------------------------------------
    // L1 [4 KiB, 4 KiB + l1Reserved) holds the backend's constant pool and scratch.
    // L2 [0, cores x l2StagingPerCore) holds each core's LP6 <-> L1 staging window; zero sizes the
    // window per phase from the L2 space left after the live L2 buffers.
    int64_t l1ReservedBytes = 256 << 10, l2StagingPerCore = 512 << 10;

    // ---- tiling (hyb-npu-tile and the passes that size tiles) ------------------------------------
    // Weight bytes per GEMV loop tile (at least one SA column block: K x saCols bytes).
    int64_t gemvTileBytes = 128 << 10;
    // Whether an op is split over the cores is a cost comparison (NpuCost.h, coreSplitUs). These
    // are only floors for experiments: ops with fewer elements stay on one core (0: no floor;
    // dequantMinElements 0 inherits distributeMinElements).
    int64_t distributeMinElements = 0, dequantMinElements = 0;

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
    // hyb-npu-gemv-epilogue: a two-segment fused GEMV (gate;up) whose halves feed one elementwise
    // op (SiLU(gate) * up) is split so each core computes the same columns of both halves and
    // applies that op itself; neither half goes through L2.
    bool pairGateUp = false;
    // hyb-npu-epilogue-pipeline: that op may also run tile by tile behind the second GEMV. It is
    // placed there only when the loop waits for weights (NpuCost.h): a core issues in order, so in
    // a compute-bound loop VPU work between tiles delays the next tile's MMA.
    bool pairConsumerPipeline = false;
    bool elementwiseDistribution = false;  // hyb-npu-elementwise-distribute
    // hyb-npu-elementwise-distribute also splits token-row work over the cores: hyb.quant and
    // row reductions (RMSNorm, q/k norm), with their single-use row-wise producers fused.
    bool rowDistribution = false;
    // hyb-npu-elementwise-distribute also splits top-level FP16 transposes over the cores by their
    // leading output axis (the V and attention-output layouts around attention).
    bool distributeTransposes = false;
    // hyb-npu-rope: a rotate-half RoPE becomes two half-width generics on a per-core range of heads,
    // written straight into the head-major layout when that transpose is its only user.
    bool ropeSplit = false;
    // Applied when splitting pays (NpuCost.h); ropeSplitMinElements is a floor for experiments.
    int64_t ropeSplitMinElements = 0;
    bool activationResidency = false;      // hyb-npu-activation-residency
    bool matrixInitElision = false;        // hyb-npu-matrix-init
    bool matmulRhsPacking = false;         // hyb-npu-matmul-layout: pack the transposed RHS
    bool matmulTransposeOutput = false;    // hyb-npu-matmul-layout: transpose A and the output instead
    bool inputTilePipeline = false;        // hyb-npu-input-pipeline
    bool gemvEpiloguePipeline = false;     // hyb-npu-epilogue-pipeline
    int64_t inputTileColumns = 128;
    // hyb-npu-flash-attention: online softmax over key tiles with FP32 running max / sum / output.
    // Flash-Decoding additionally splits the keys over cores and merges the states; KV slicing
    // streams [past KV, new KV] without a full copy. Its rounding differs from the FP16 softmax.
    bool flashAttention = false, flashDecoding = false, flashKvSlicing = false;
    int64_t attentionQueryTile = 32, attentionKeyTile = 128, attentionSplits = 1;

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
    // hyb.quant: keep max|x| of a row in a vector register (one V_MAX per chunk) instead of storing
    // |x| to L1 and reducing it again. Max is exact in any order, so the scales are the same.
    bool quantFusedMax = false;
    // Emit the repeated 32-lane chunks of a row (linalg.generic, hyb.quant) as one hardware loop
    // (C_LOOP) instead of unrolling them: the same instructions run, far fewer are fetched.
    bool vectorLoops = false;
    // Per-core GDMA commands issued ahead within a phase; zero is unlimited.
    int64_t gdmaLookahead = 0;
    // Count only weight-stream GDMAs against gdmaLookahead; small staged loads are not held back.
    bool gdmaLookaheadStreamsOnly = false;
    // Bytes at the end of each staging window for small LP6 <-> L1 transfers, used as a byte
    // ring; the weight ring stays below them. Zero stages every transfer at the window start.
    int64_t stagingMiscBytes = 0;
    // Shrink that area to the window space no weight ring of the phase can use, so no ring loses a
    // slot (transfers that no longer fit go to the window start).
    bool stagingMiscFitRings = false;
    // GDMA loads of an NPU phase wait for the earlier phase's accesses to the same L2 / LP6
    // bytes instead of its exit barrier, so the next layer's weights stream in during its tail.
    bool crossPhasePrefetch = false;
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
