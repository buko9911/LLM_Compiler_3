#ifndef NPU_TARGET_PIMLAYOUT_H
#define NPU_TARGET_PIMLAYOUT_H
// LPDDR6-PIM W8A8 GEMV layout: which tiles a GEMV uses, where their packets
// live in LP6, and the PIM_GEMV commands that run it.
//
// It mirrors the lp6-pim-main runtime lowering (PIMRuntimeLowering::
// BuildRuntimeTileLayout) for W8A8 GEMV streams without a block scale:
//  * one LdMAC drives the 8 PUs of the even bank set; a PU packet holds 4 lanes
//    x 8 INT8 weights; an activation packet holds 32 INT8 values, 4 input
//    register slices of 8;
//  * a GEMV_STREAM covers 512 output rows (8 PUs x 16 accumulators x 4 lanes)
//    and 256 inputs (8 input registers x 4 slices x 8); a longer K is several
//    sweeps;
//  * a partition p is subchannel p % 2 of rank p / 2;
//  * tile ids are dense per kind; the runtime places weight rows from the
//    kernel row, then activation rows, then output rows (no scale rows).
//
// LP6 byte addresses follow lp6-pim-main configs/LPDDR6_PIM_Mobile_2CH.ini,
// address_mapping = rarobacobgch (MSB -> LSB: rank, row, bank, column, bank
// group, subchannel above the 32 B offset).
#include "npu/Target/Result.h"
#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace npu::pim {

inline constexpr int kPUs = 8, kLanes = 4, kAccs = 16, kGroup = 8, kSlices = 4, kInTiles = 32;
inline constexpr int64_t kKPerSweep = 256, kRowsPerStream = 512, kPacketsPerRow = 64;
inline constexpr int kChannels = 2, kRanks = 2;
inline constexpr int64_t kChBytes = 32, kBgBytes = 64, kColBytes = 256, kBankBytes = 16384,
                         kRowBytes = 65536, kRankBytes = int64_t(1) << 33;

// One INT8 activation vector of one token.
struct Activation {
  int64_t k = 0, base = 0;   // base: first activation tile id
  std::vector<int> parts;    // partitions whose PUs read it
  int64_t packets() const { return k / 32; }
  int64_t packetBase() const { return base / kSlices; }
};

// One GEMV, a weight of N output channels and K inputs, for M tokens.
struct Projection {
  std::string name;
  int64_t n = 0, k = 0, nPad = 0;           // nPad: N rounded up to whole streams
  int64_t outputBase = 0;
  std::vector<int64_t> streamWeightBase;    // first weight tile of each stream, in its partition
  std::vector<int64_t> activations;         // activation per token
  std::vector<int> streamPart;              // partition of each stream
  int64_t computeFirst = 0, computeCount = 0;   // PIM_GEMV command range
  int64_t streams() const { return nPad / kRowsPerStream; }
  int64_t sweeps() const { return k / kKPerSweep; }
  int64_t rowsPerPU() const { return nPad / kPUs; }
  int64_t tilesPerPU() const { return nPad / (kPUs * 8); }
  int64_t streamTiles() const { return sweeps() * kAccs * kInTiles; }
  int tokens() const { return int(activations.size()); }
  int tilePart(int64_t t) const { return streamPart[size_t((t % (streams() * 8)) / 8)]; }
  int64_t tokenOutputTile(int m, int64_t t) const { return int64_t(m) * streams() * 8 + t; }
  // PIM_GEMV commands emitCompute appends for this projection.
  int64_t commands() const { return int64_t(tokens()) * streams(); }
};

// All PIM work of one program: one kernel, one tile-id space.
struct Program {
  int partitions = 4;
  int64_t kernelRow = 0;
  std::vector<int64_t> nextWeight = std::vector<int64_t>(4, 0);  // weight tile ids per partition
  int64_t nextActivation = 0, nextOutput = 0;
  std::vector<Activation> activations;
  std::vector<Projection> projections;

  int partChannel(int p) const { return p % kChannels; }
  int partRank(int p) const { return p / kChannels; }
  int64_t addActivation(int64_t k);
  // Adds a GEMV reading `activations` (one per token). Each stream goes to the
  // least loaded partition of `loads`, shared by the GEMVs of one PIM phase; a
  // partition that must not compute carries a huge load.
  int64_t addProjection(const std::string &name, int64_t n, int64_t k, const std::vector<int64_t> &activations,
                        std::map<int, int64_t> &loads);
  // Weight tiles already stored in the partitions of rank `rank`.
  int64_t storedTiles(int rank) const;
  // PIM_GEMV commands of projection `index`, one per token and stream, appended
  // to `commands` (two words each).
  Result<bool> emitCompute(int64_t index, std::vector<uint32_t> &commands);

  struct BaseRows { int64_t weight, activation, output; };
  BaseRows baseRows() const;
  uint64_t packetAddress(int64_t baseRow, int64_t packet, int pu, int part) const;
  uint64_t activationPacketAddress(const Activation &a, int64_t packet, int pu, int part) const {
    return packetAddress(baseRows().activation, a.packetBase() + packet, pu, part);
  }
  uint64_t outputTileAddress(const Projection &p, int64_t tile, int pu) const {
    return packetAddress(baseRows().output, p.outputBase + tile, pu, p.tilePart(tile));
  }
  // Logical output rows of stored tile t of PU pu: 8 rows, in packet word order.
  int64_t firstRow(const Projection &p, int pu, int64_t t) const { return pu * p.rowsPerPU() + t * 8; }
  // Visits every weight packet of p: its LP6 address and its 32 bytes, built
  // from q [N][K] (rows >= N are zero).
  template <typename Fn> void forEachWeightPacket(const Projection &p, const int8_t *q, Fn fn) const;
  // First LP6 byte after the output rows inside one rank, and of the whole PIM
  // region (the last used rank starts at rank * kRankBytes).
  uint64_t rankEndAddress() const;
  uint64_t endAddress() const;
};

template <typename Fn>
void Program::forEachWeightPacket(const Projection &p, const int8_t *q, Fn fn) const {
  const int64_t streams = p.streams(), sweeps = p.sweeps(), K = p.k, rows = p.rowsPerPU();
  const int64_t weightRow = baseRows().weight;
  std::array<uint8_t, 32> packet;
  for (int64_t g = 0; g < streams; ++g)
    for (int64_t s = 0; s < sweeps; ++s)
      for (int o = 0; o < kAccs; ++o)
        for (int i = 0; i < kInTiles; ++i) {
          int64_t tile = p.streamWeightBase[size_t(g)] + (s * kAccs + o) * kInTiles + i;
          for (int pu = 0; pu < kPUs; ++pu) {
            for (int l = 0; l < kLanes; ++l) {
              int64_t row = pu * rows + (g * 8 + o / 2) * 8 + 2 * l + (o % 2);
              for (int e = 0; e < kGroup; ++e) {
                int64_t column = (s * kInTiles + i) * kGroup + e;
                packet[size_t(l * kGroup + e)] = row < p.n ? uint8_t(q[row * K + column]) : 0;
              }
            }
            fn(packetAddress(weightRow, tile, pu, p.streamPart[size_t(g)]), packet);
          }
        }
}
} // namespace npu::pim
#endif
