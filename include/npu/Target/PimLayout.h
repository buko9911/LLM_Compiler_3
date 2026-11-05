#ifndef NPU_TARGET_PIMLAYOUT_H
#define NPU_TARGET_PIMLAYOUT_H
// LPDDR6-PIM W8A8 GEMV layout: which tiles a GEMV uses, where their packets
// live in LP6, and the compiler-lowered primitives that run it.
//
// It mirrors the PIM runtime lowering of core/ramulator2/src/ramulator/pim/lp6_lowering (PIMRuntimeLowering::
// BuildRuntimeTileLayout) for W8A8 GEMV streams without a block scale:
//  * one LdMAC drives the 8 PUs of the even bank set; a PU packet holds 4 lanes
//    x 8 INT8 weights; an activation packet holds 32 INT8 values, 4 input
//    register slices of 8;
//  * a stream covers 32 x A output rows (8 PUs x A accumulators x 4 lanes; A even, 2..16, chosen
//    per projection, 16 = 512 rows by default) and 256 inputs per sweep (8 input registers x 4
//    slices x 8); a longer K is several sweeps;
//  * split-K: a projection may cut K into S equal pieces of whole sweeps. Each piece is a stream
//    set of its own with its own INT32 output tiles; the NPU adds the S partial sums when it
//    gathers the result (integer sums, so the result is exact);
//  * a partition p is subchannel p % C of rank p / C, C = LP6 subchannels (8 for
//    four LPDDR6 channels, 2 for one);
//  * tile ids are dense per kind; the runtime places weight rows from the
//    kernel row, then activation rows, then output rows (no scale rows).
//
// LP6 byte addresses follow core/ramulator2/configs/lp6/LPDDR6_PIM_Mobile_4CH.ini (C = 8),
// address_mapping = rarobacobgch (MSB -> LSB: rank, row, bank, column, bank group,
// subchannel above the 32 B offset): the strides below the rank grow with C; a rank is 8 GiB.
#include "npu/Target/Result.h"
#include "npu/Target/PimPrimitive.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace npu::pim {

inline constexpr int kPUs = 8, kLanes = 4, kAccs = 16, kGroup = 8, kSlices = 4, kInTiles = 32;
inline constexpr int64_t kKPerSweep = 256, kRowsPerStream = 512, kPacketsPerRow = 64;
inline constexpr int kRanks = 2;
inline constexpr int64_t kChBytes = 32, kRankBytes = int64_t(1) << 33;
// ISA 2.0 words of one primitive in a PIM record, on average (LOAD 3, GEMV_STREAM 5 and so on, measured on Qwen3-4B).
inline constexpr double kRecordWordsPerPrimitive = 3.2;

// One physical 32-byte weight packet, described without reading weight data.
// Lane l stores W[inputBegin + e, outputChannels[l]], e=0..7, in the
// mathematical X[M,K] * W[K,N] convention. Channels >= Projection::n are
// padding lanes and must remain present as zeros in the packed image.
struct WeightPacketLocation {
  uint64_t address = 0;
  int64_t inputBegin = 0;
  std::array<int64_t, kLanes> outputChannels{};
};

// One INT8 activation vector of one token.
struct Activation {
  int64_t k = 0, base = 0;   // base: first activation tile id
  std::vector<int> parts;    // partitions whose PUs read it
  int64_t packets() const { return k / 32; }
  int64_t packetBase() const { return base / kSlices; }
};

// One GEMV, a weight of N output channels and K inputs, for M tokens.
// Stream g = piece * blocks() + block: one block of output rows, spread over the 8 PUs, and K sweeps
// [piece * pieceSweeps(), (piece + 1) * pieceSweeps()). Block b has blockAccs(b) accumulators, i.e.
// 32 x blockAccs(b) rows (4 x blockAccs(b) per PU): the first mainBlocks() blocks `accs`, the last
// `tailBlocks` blocks `tailAccs` (a smaller stream that evens out the partitions' loads).
struct Projection {
  std::string name;
  int64_t n = 0, k = 0, nPad = 0;           // nPad: N rounded up to whole blocks
  int accs = kAccs;                         // accumulators per stream of the main blocks (even, 2..16)
  int splitK = 1;                           // K pieces (sweeps() % splitK == 0)
  int tailAccs = 0;                         // accumulators of the tail blocks (0: no tail)
  int64_t tailBlocks = 0;
  int64_t outputBase = 0;
  std::vector<int64_t> streamWeightBase;    // first weight tile of each stream, in its partition
  std::vector<int64_t> activations;         // activation per token
  std::vector<int> streamPart;              // partition of each stream (sorted inside each piece)
  // Common layout (--common-layout, research option; see commonShape below): the GEMVs of a group are one projection whose
  // outputs follow `segments` (checkpoint tensor, rows); `streamOrder` is the emission order of the streams.
  bool common = false;
  std::vector<std::pair<std::string, int64_t>> segments;
  std::vector<int64_t> streamOrder;
  int64_t computeFirst = 0, computeCount = 0;   // executable primitive range
  int64_t mainBlocks() const {
    return (nPad - tailBlocks * int64_t(kPUs) * kLanes * tailAccs) / (int64_t(kPUs) * kLanes * accs);
  }
  int64_t blocks() const { return mainBlocks() + tailBlocks; }
  int blockAccs(int64_t b) const { return b < mainBlocks() ? accs : tailAccs; }
  int64_t blockTile(int64_t b) const {     // first PU-local output tile of block b
    const int64_t main = mainBlocks();
    return b <= main ? b * (accs / 2) : main * (accs / 2) + (b - main) * (tailAccs / 2);
  }
  int64_t blockOfTile(int64_t t) const {
    const int64_t mainTiles = mainBlocks() * (accs / 2);
    return t < mainTiles ? t / (accs / 2) : mainBlocks() + (t - mainTiles) / (tailAccs / 2);
  }
  int64_t streams() const { return blocks() * splitK; }
  int64_t sweeps() const { return k / kKPerSweep; }
  int64_t pieceSweeps() const { return sweeps() / splitK; }
  int64_t rowsPerPU() const { return nPad / kPUs; }
  int64_t tilesPerPU() const { return nPad / (kPUs * 8); }   // output tiles of 8 per PU, per piece and token
  int streamAccs(int64_t g) const { return blockAccs(g % blocks()); }
  int64_t tilesPerStream(int64_t g) const { return streamAccs(g) / 2; }   // per PU
  // Weight tiles per sweep. Common layout: an empty row per sweep when a sweep takes an even number of rows, so that the rows of one
  // accumulator pair in consecutive sweeps alternate bank sets (the NPU can open the next one early).
  int64_t sweepTiles(int accs) const {
    return accs * kInTiles + (common && (accs / 2) % 2 == 0 ? kPacketsPerRow : 0);
  }
  int64_t streamTiles(int64_t g) const { return pieceSweeps() * sweepTiles(streamAccs(g)); }
  // Common layout: outputs of two equal segments (gate, up) lie on the odd and even PUs of a bank group, others in PU order.
  bool pairedSegments() const { return segments.size() == 2 && segments[0].second == segments[1].second; }
  int commonPU(int index) const { return pairedSegments() ? (index < 4 ? 2 * index : 2 * (index - 4) + 1) : index; }
  int commonPUIndex(int pu) const { return pairedSegments() ? (pu % 2 ? 4 + pu / 2 : pu / 2) : pu; }
  int tokens() const { return int(activations.size()); }
  // Output tiles. For each token and K piece, every partition numbers the tiles of its streams from
  // the same id: stream g's register r is tile outputTile(m, piece, g, r). One LP6 column (one tile
  // id) then holds that id's tile of every subchannel, so the NPU gathers whole bank-group slices
  // (all subchannels at once) instead of 32-byte tiles from one bank at a time.
  int64_t outputSlots = 0;            // tile ids per (token, piece): the most tiles one partition stores
  std::vector<int64_t> streamSlot;    // first tile slot of each stream inside its partition and piece
  void assignOutputSlots();           // from streamPart; addProjection and plan loading call it
  int64_t outputTile(int m, int piece, int64_t g, int reg) const {
    return outputBase + (int64_t(m) * splitK + piece) * outputSlots + streamSlot[size_t(g)] + reg;
  }
  // The stream that stores PU-local output tile t (rows t * 8 .. t * 8 + 7 of each PU) of a piece.
  int64_t streamOfTile(int64_t t, int piece) const { return piece * blocks() + blockOfTile(t); }
  // Current compiler schedule: clear, load/stream per K sweep, store.
  int64_t commands() const {
    int64_t perToken = 0;
    for (int64_t b = 0; b < blocks(); ++b)
      perToken += splitK * (blockAccs(b) + pieceSweeps() * (kInTiles / kSlices + 1) + blockAccs(b) / 2);
    return int64_t(tokens()) * perToken;
  }
};

// All PIM work of one program: one kernel, one tile-id space.
struct Program {
  int channels = 2;     // LP6 subchannels of the address decode
  int partitions = 4;   // at most channels x kRanks
  // 0: even banks; 2..63: alternate weight rows across bank sets with this lead.
  int banksetPingPongLead = 0;
  // Research option --common-layout: 0 off, 1 one rank per phase, 2 K halves on the two ranks, computed in turn (see commonShape).
  int commonLayout = 0;
  const char *weightLayout() const {
    return banksetPingPongLead ? "bankset-row-alternating-v1" : "even-v1";
  }
  int64_t kernelRow = 0;
  std::vector<int64_t> nextWeight = std::vector<int64_t>(4, 0);  // weight tile ids per partition
  int64_t nextActivation = 0, nextOutput = 0;
  // A PIM phase's activation and output tiles are dead once the NPU phase after it has read them (the
  // phases run one after another). With reuseBuffers every PIM phase numbers them from 0 and the regions
  // are sized by the largest phase (activationPeak, outputPeak).
  bool reuseBuffers = false;
  int64_t activationPeak = 0, outputPeak = 0;
  int64_t activationTiles() const { return reuseBuffers ? activationPeak : nextActivation; }
  int64_t outputTiles() const { return reuseBuffers ? outputPeak : nextOutput; }
  // Renumber the activation and output tiles of every PIM phase from 0 and set reuseBuffers. False (and no
  // change) when an activation is read by two phases.
  bool reuseBuffersAcrossPhases();
  std::vector<Activation> activations;
  std::vector<Projection> projections;
  std::vector<std::pair<int64_t, int64_t>> phases; // first command, command count

  int partChannel(int p) const { return p % channels; }
  int partRank(int p) const { return p / channels; }
  int64_t bgBytes() const { return kChBytes * channels; }
  int64_t colBytes() const { return 4 * bgBytes(); }
  int64_t bankBytes() const { return kPacketsPerRow * colBytes(); }
  int64_t rowBytes() const { return 4 * bankBytes(); }
  int64_t addActivation(int64_t k);
  // Adds a GEMV reading `activations` (one per token). Each stream goes to the
  // least loaded partition of `loads`, shared by the GEMVs of one PIM phase; a
  // partition that must not compute carries a huge load. Equal loads are broken
  // by resident weight storage, including streams assigned in this projection.
  // `accs` and `splitK` set the stream shape (default: 512-row streams over all of K).
  // `streamLoad` is the load one stream adds (0: its LdMAC count / 32, sweeps x accumulators x tokens).
  // `tailAccs` / `tailBlocks`: the last blocks use fewer accumulators. `streamLoad(accs)` is the load
  // one stream adds (empty: its LdMAC count / 32, sweeps x accumulators x tokens).
  // `pieceRanks` > 1: K piece p stays on rank p % pieceRanks, so that rank needs only those input packets.
  int64_t addProjection(const std::string &name, int64_t n, int64_t k, const std::vector<int64_t> &activations,
                        std::map<int, int64_t> &loads, int accs = kAccs, int splitK = 1,
                        std::function<int64_t(int)> streamLoad = {}, int tailAccs = 0, int64_t tailBlocks = 0,
                        int pieceRanks = 1);
  // Packet ranges of activation `index` that the streams on rank `rank` read (sorted, merged). A rank that computes
  // only some K pieces needs only their packets.
  std::vector<std::pair<int64_t, int64_t>> activationPackets(int64_t index, int rank) const;
  // Common layout: one projection for a group of GEMVs. `rank` is the rank of this phase when commonLayout == 1.
  int64_t addCommonProjection(const std::string &name, const std::vector<std::pair<std::string, int64_t>> &segments, int64_t k,
                              const std::vector<int64_t> &activations, int rank, bool mayCutK);
  // Weight tiles already stored in the partitions of rank `rank`.
  int64_t storedTiles(int rank) const;
  // Output tile ids of every projection again, in order (after the streams were rebound).
  void assignOutputs();
  int64_t phaseOf(const Projection &p) const;   // PIM phase holding p's commands, -1 for none
  // Lower a projection into final executable PIM IR, before serialization.
  Result<bool> emitCompute(int64_t index, std::vector<Primitive> &commands);

  struct BaseRows { int64_t weight, activation, output; };
  BaseRows baseRows() const;
  uint64_t packetAddress(int64_t baseRow, int64_t packet, int pu, int part) const;
  uint64_t weightPacketAddress(int64_t tile, int pu, int part) const {
    return packetAddress(baseRows().weight, tile, pu, part) +
           (banksetPingPongLead && (tile / kPacketsPerRow) % 2 ? bankBytes() : 0);
  }
  uint64_t activationPacketAddress(const Activation &a, int64_t packet, int pu, int part) const {
    return packetAddress(baseRows().activation, a.packetBase() + packet, pu, part);
  }
  uint64_t outputTileAddress(int64_t tile, int pu, int part) const {
    return packetAddress(baseRows().output, tile, pu, part);
  }
  // Logical output rows of stored tile t of PU pu: 8 rows, in packet word order.
  // Visits the same packets, in the same order, as forEachWeightPacket, but
  // needs no model bytes. Address analysis and consumers of the stored weight
  // share the packer's logical-to-physical mapping through this callback.
  // This has the existing layout's K-multiple-of-256 contract; no K-tail
  // support is introduced. Padding output channels are not filtered out.
  template <typename Fn> void forEachWeightPacketLocation(const Projection &p, Fn fn) const;
  // Visits every weight packet of p: its LP6 address and its 32 bytes, built
  // from q [N][K] (rows >= N are zero).
  template <typename Fn> void forEachWeightPacket(const Projection &p, const int8_t *q, Fn fn) const;
  // First LP6 byte after the output rows inside one rank, and of the whole PIM
  // region (the last used rank starts at rank * kRankBytes).
  uint64_t rankEndAddress() const;
  uint64_t endAddress() const;
};

// Common layout, the research option --common-layout (2026-11-05, research/race_decoding/common_layout). One placement of the weights
// that PIM computes on directly and the NPU reads through a byte transposer on its read path:
//   output n of the group = PU index x rows + 64 R + 8 s + 2 l + x
//   R register of the PU (one row per sweep, stream after stream), s subchannel, l lane, x accumulator parity.
// So a register holds 8 consecutive outputs and a column of a PU holds 64 consecutive ones over the 8 subchannels. An NPU tile of 32
// outputs is 4 subchannels x 2 accumulators of one PU. K is whole per output on a rank; commonLayout 2 puts K halves on the two ranks.
struct CommonShape {
  int accs = 16, tailAccs = 0;
  int64_t mainBlocks = 0, tailBlocks = 0, nPad = 0;
};
inline CommonShape commonShape(int64_t n) {
  CommonShape c;
  const int64_t rowsPerRegister = 512;        // over the 8 subchannels and 8 PUs
  const int64_t registers = (n + rowsPerRegister - 1) / rowsPerRegister;
  if (registers <= 8) {
    c.accs = int(2 * registers);
    c.mainBlocks = kPUs;
  } else {
    c.accs = kAccs;
    c.mainBlocks = (registers / 8) * kPUs;
    const int64_t rest = registers % 8;
    if (rest) {
      c.tailAccs = int(2 * rest);
      const int64_t rows = n - c.mainBlocks * 32 * c.accs;
      c.tailBlocks = (rows + 32 * c.tailAccs - 1) / (32 * c.tailAccs);
    } else if (n % rowsPerRegister) {
      c.mainBlocks -= kPUs;
      c.tailAccs = kAccs;
      c.tailBlocks = (n - c.mainBlocks * 32 * c.accs + 32 * kAccs - 1) / (32 * kAccs);
    }
  }
  c.nPad = c.mainBlocks * 32 * c.accs + c.tailBlocks * 32 * c.tailAccs;
  return c;
}

// The same placement seen from the NPU: where the 32 B row k of tile j of an NPU weight goes. The NPU reads a row as one burst; the
// bursts of a tile in a sweep are the PIM packets of that tile and sweep (test/Decode/pim_layout.cpp). Fetch order per sweep:
// 64 columns x 4 subchannels. rowBase: first row in every PU, the same on both ranks.
struct CommonWeight {
  uint64_t logical = 0, bytes = 0;     // the weight's logical address range, which the compiler plans with; never reaches DRAM
  int64_t n = 0, k = 0, rowBase = 0;
  bool paired = false, kHalves = false;
  int64_t sweeps() const { return k / kKPerSweep; }
  int64_t pieceSweeps() const { return kHalves ? sweeps() / 2 : sweeps(); }
  int64_t rowsUsed() const {
    const CommonShape c = commonShape(n);
    const auto step = [](int a) { return int64_t(a / 2 + ((a / 2) % 2 == 0 ? 1 : 0)); };
    return (c.mainBlocks / kPUs) * pieceSweeps() * step(c.accs) + (c.tailBlocks ? pieceSweeps() * step(c.tailAccs) : 0) + 2;
  }
  // PA of the 32-byte row k of tile j, for 8 subchannels (rarobacobgch).
  uint64_t address(int64_t j, int64_t row) const {
    const CommonShape c = commonShape(n);
    const int64_t rows = c.nPad / kPUs, n0 = j * 32;
    const int index = int(n0 / rows);
    const int pu = paired ? (index < 4 ? 2 * index : 2 * (index - 4) + 1) : index;
    const int64_t local = n0 - index * rows, R = local / 64, half = (local % 64) / 32;
    const int64_t regsMain = (c.mainBlocks / kPUs) * (c.accs / 2);
    const auto step = [](int a) { return int64_t(a / 2 + ((a / 2) % 2 == 0 ? 1 : 0)); };
    const int64_t v = row / kKPerSweep, inner = row % kKPerSweep, piece = v / pieceSweeps(), vv = v % pieceSweeps();
    int64_t dramRow = rowBase;
    if (R < regsMain) {
      dramRow += (R / (c.accs / 2)) * pieceSweeps() * step(c.accs) + vv * step(c.accs) + R % (c.accs / 2);
    } else {
      dramRow += (c.mainBlocks / kPUs) * pieceSweeps() * step(c.accs) + vv * step(c.tailAccs) + (R - regsMain);
    }
    const int rank = kHalves ? int(piece) : 0;
    const uint64_t bank = uint64_t(2 * (pu % 2)) + uint64_t(dramRow & 1);
    const uint64_t column = uint64_t(inner / 4), sc = uint64_t(4 * half + inner % 4), group = uint64_t(pu / 2);
    return uint64_t(rank) * uint64_t(kRankBytes) + uint64_t(dramRow) * (uint64_t(1) << 18) + bank * (uint64_t(1) << 16) +
           column * (uint64_t(1) << 10) + group * (uint64_t(1) << 8) + sc * 32;
  }
};

template <typename Fn>
void Program::forEachWeightPacketLocation(const Projection &p, Fn fn) const {
  const int64_t streams = p.streams(), sweeps = p.pieceSweeps(), rows = p.rowsPerPU(), blocks = p.blocks();
  for (int64_t g = 0; g < streams; ++g) {
    const int64_t block = g % blocks, piece = g / blocks, tile0 = p.blockTile(block);
    const int accs = p.blockAccs(block);
    for (int64_t s = 0; s < sweeps; ++s)
      for (int o = 0; o < accs; ++o)
        for (int i = 0; i < kInTiles; ++i) {
          int64_t tile = p.streamWeightBase[size_t(g)] + s * p.sweepTiles(accs) + o * kInTiles + i;
          for (int pu = 0; pu < kPUs; ++pu) {
            WeightPacketLocation location;
            location.address = weightPacketAddress(tile, pu, p.streamPart[size_t(g)]);
            location.inputBegin = ((piece * sweeps + s) * kInTiles + i) * kGroup;
            for (int l = 0; l < kLanes; ++l)
              location.outputChannels[size_t(l)] =
                  p.common ? p.commonPUIndex(pu) * rows +
                                 64 * (p.streamSlot[size_t(g)] + o / 2) + 8 * partChannel(p.streamPart[size_t(g)]) + 2 * l + (o % 2)
                           : pu * rows + (tile0 + o / 2) * 8 + 2 * l + (o % 2);
            fn(location);
          }
        }
  }
}

template <typename Fn>
void Program::forEachWeightPacket(const Projection &p, const int8_t *q, Fn fn) const {
  std::array<uint8_t, 32> packet;
  forEachWeightPacketLocation(p, [&](const WeightPacketLocation &location) {
    for (int l = 0; l < kLanes; ++l) {
      int64_t row = location.outputChannels[size_t(l)];
      for (int e = 0; e < kGroup; ++e)
        packet[size_t(l * kGroup + e)] = row < p.n ? uint8_t(q[row * p.k + location.inputBegin + e]) : 0;
    }
    fn(location.address, packet);
  });
}
} // namespace npu::pim
#endif
