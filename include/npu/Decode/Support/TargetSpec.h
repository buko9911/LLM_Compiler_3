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

    // ---- GEMV weight streaming (backend, CodegenLoops.cpp) -----------------------------------------
    // Ring slots in the L2 staging window; zero removes the cap (capacity and trip count still bound it).
    int64_t weightStreamMaxSlots = 8;
    int64_t inputTileColumns = 128;        // hyb-npu-input-pipeline: columns per pipelined input tile

    // ---- backend scheduling -------------------------------------------------------------------------
    // Per-core weight-stream GDMA commands in flight within a phase; zero is unlimited. Small staged
    // loads and stores are not counted.
    int64_t gdmaLookahead = 0;
    // Bytes at the end of each staging window for small LP6 <-> L1 transfers, used as a byte
    // ring; the weight ring stays below them. Zero stages every transfer at the window start.
    int64_t stagingMiscBytes = 0;
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
    // ---- stream layout cost (npu/Decode/Support/PimCost.h), measured on the simulator ----------
    /// PIM cycles per LdMAC in a long stream, row activation and precharge included (nLLD when 0).
    double ldmacCK = 0;
    /// Fixed PIM cycles per stream sweep (its activation register loads) and per stream (clear, store).
    double sweepCK = 0, streamCK = 0;
    /// NPU gather of INT32 PIM outputs (per K piece) and scatter of INT8 activations (per token).
    double readbackGBps = 32, readbackPieceUs = 0, actWriteGBps = 96, actWritePieceUs = 0;
    /// Accumulator counts a stream may use (even, 2..16) and the largest split-K.
    std::vector<int> streamAccs = {2, 4, 6, 8, 10, 12, 14, 16};
    int maxSplitK = 8;
    /// Default weight row layout: 0 even bank set, else alternate bank sets with this LdMAC lead.
    int banksetPingPongLead = 0;
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
