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
/// plus the calibration constants of the cost model. Every key must be in the file (TargetSpec::load).
struct TargetSpec {
  struct Npu {
    // ---- hardware -------------------------------------------------------------------------------
    double clockMHz = 800;
    int cores = 4, saRows = 32, saCols = 32;
    int64_t l1Bytes = 4 << 20, l2Bytes = 8 << 20;
    // Research: the program uses L1 [l1BaseBytes, l1BaseBytes + l1Bytes) and L2 [l2BaseBytes, l2BaseBytes + l2Bytes), so that two programs merged
    // into one file (research/race_decoding/drafter_race) do not share memory. Optional keys, 0 when absent.
    int64_t l1BaseBytes = 0, l2BaseBytes = 0;
    // Cost model of hyb-partition: a fixed cost per NPU GEMV.
    double gemvFixedUs = 2.0;

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
    /// Sustained L1 <-> L2 local DMA of one core, GB/s (measured).
    double ldmaGBpsPerCore = 0;
    /// Words of system commands the command processor reads per cycle.
    int64_t commandFetchWordsPerCycle = 0;
    // Fixed cost of running one op on several cores instead of one (dispatch, the cross-core events
    // and the per-core copies of its operands): measured, see npu.core_split_us in the target files.
    double coreSplitUs = 0.5, coreSplitNoStreamUs = 0.2;

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
    int64_t gdmaChainPerRank = 0;  // research: one GDMA chain per rank
    // Bytes at the end of each staging window for small LP6 <-> L1 transfers, used as a byte
    // ring; the weight ring stays below them. Zero stages every transfer at the window start.
    int64_t stagingMiscBytes = 0;
  } npu;
  struct Pim {
    /// DRAM CK of the LPDDR6-PIM model (core/ramulator2/src/ramulator/pim/runtime_profile.h, tCK 0.375 ns).
    /// The *CK numbers below are in these cycles.
    double clockMHz = 2666.667;
    /// The geometry fixed by the LPDDR6-PIM draft. npu/Target/PimLayout.h is written for it, so
    /// TargetSpec::load refuses other values instead of laying out a program the hardware cannot run.
    int pus = 8, lanes = 4, accumulators = 16, kPerSweep = 256, channels = 8, ranks = 2, partitions = 16;
    /// Minimum LdMAC spacing (draft Table 51; 10 CK at 9600-10667 Mbps). A fitted ldmac_ck cannot be below it.
    int nlldCK = 10;
    /// Whether the partitions of different ranks compute at the same time. False runs each PIM phase on
    /// the subchannels of one rank; the other rank only holds data and serves normal reads meanwhile.
    bool concurrentRanks = true;
    /// Partitions that compute at the same time, and the ranks one PIM phase uses.
    int parallelPartitions() const { return concurrentRanks ? partitions : std::min(partitions, channels); }
    // ---- stream layout cost (npu/Decode/Support/PimCost.h), measured on the simulator ----------
    /// DRAM CK per LdMAC in a long stream, row activation and precharge included.
    double ldmacCK = 0;
    /// Fixed DRAM CK per stream sweep (its activation register loads) and per stream (clear, store).
    double sweepCK = 0, streamCK = 0;
    /// A phase on both ranks takes this much longer than the one-rank cost: the ranks of a subchannel
    /// share its command bus and DQ (measured ratio).
    double twoRankFactor = 1;
    /// The first PIM record of a phase is read before the phase computes: at most this long, in at most maxRecords records.
    double recordFetchUs = 0;
    int64_t maxRecords = 1;
    /// NPU gather of INT32 PIM outputs and scatter of INT8 activations.
    double readbackGBps = 32, actWriteGBps = 96;
    /// NPU time per PIM output per token to bring it back and dequantize it (hyb-partition; the NPU
    /// path dequantizes behind its own GEMV, so this is extra PIM-side work).
    double readbackNsPerOutput = 0;
    /// hyb-partition: NPU time lost when a PIM group splits a layer's NPU weight stream into two runs
    /// (the NPU work next to the split is no longer hidden behind weight reads).
    double npuRunSplitUs = 0;
    /// A layer with any group on PIM: the time its NPU work (attention, norms, SiLU) takes in the open,
    /// which an all-NPU layer hides under its weight reads (measured).
    double layerExposedUs = 0;
    /// Accumulator counts a stream may use (even, 2..16) and the largest split-K.
    std::vector<int> streamAccs = {2, 4, 6, 8, 10, 12, 14, 16};
    int maxSplitK = 8;
    /// Default weight row layout: 0 even bank set, else alternate bank sets with this LdMAC lead.
    int banksetPingPongLead = 0;
  } pim;
  struct Lp6 {
    int64_t capacityBytes = int64_t(16) << 30;
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
