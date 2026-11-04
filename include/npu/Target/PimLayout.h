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
  int64_t streamTiles(int64_t g) const { return pieceSweeps() * streamAccs(g) * kInTiles; }
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
  int64_t firstRow(const Projection &p, int pu, int64_t t) const { return pu * p.rowsPerPU() + t * 8; }
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

template <typename Fn>
void Program::forEachWeightPacketLocation(const Projection &p, Fn fn) const {
  const int64_t streams = p.streams(), sweeps = p.pieceSweeps(), rows = p.rowsPerPU(), blocks = p.blocks();
  for (int64_t g = 0; g < streams; ++g) {
    const int64_t block = g % blocks, piece = g / blocks, tile0 = p.blockTile(block);
    const int accs = p.blockAccs(block);
    for (int64_t s = 0; s < sweeps; ++s)
      for (int o = 0; o < accs; ++o)
        for (int i = 0; i < kInTiles; ++i) {
          int64_t tile = p.streamWeightBase[size_t(g)] + (s * accs + o) * kInTiles + i;
          for (int pu = 0; pu < kPUs; ++pu) {
            WeightPacketLocation location;
            location.address = weightPacketAddress(tile, pu, p.streamPart[size_t(g)]);
            location.inputBegin = ((piece * sweeps + s) * kInTiles + i) * kGroup;
            for (int l = 0; l < kLanes; ++l)
              location.outputChannels[size_t(l)] = pu * rows + (tile0 + o / 2) * 8 + 2 * l + (o % 2);
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
