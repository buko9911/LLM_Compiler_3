// Storage ties must not accumulate phase tails on a fixed subchannel.
#include "npu/Decode/PimMetadata.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>

using namespace ::npu::pim;

static void require(bool ok, const char *message) {
  if (!ok) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

static std::map<int, int64_t> openRank(const Program &p, int rank) {
  std::map<int, int64_t> loads;
  for (int c = 0; c < p.partitions; ++c)
    if (p.partRank(c) != rank) loads[c] = std::numeric_limits<int64_t>::max() / 4;
  return loads;
}

// Consumers must be able to recover logical indices without reading weights.
// Check bijective coverage including N padding, both ranks/channel counts,
// row-alternating banks, and the byte/order contract with the image writer.
static void packetLocations() {
  for (int channels : {2, 8})
    for (int rank : {0, 1})
      for (int ping : {0, 12}) {
        Program plan;
        plan.channels = channels;
        plan.partitions = channels * kRanks;
        plan.kernelRow = 17;
        plan.banksetPingPongLead = ping;
        auto loads = openRank(plan, rank);
        auto act = plan.addActivation(512);
        auto pi = plan.addProjection("padded", 513, 512, {act}, loads);
        const auto &p = plan.projections[pi];
        std::vector<WeightPacketLocation> locations;
        std::vector<uint8_t> seen(size_t(p.nPad * p.k), 0);
        std::set<uint64_t> addresses;
        plan.forEachWeightPacketLocation(p, [&](const WeightPacketLocation &l) {
          require(l.address % 32 == 0 && l.address / kRankBytes == uint64_t(rank),
                  "packet location lost alignment or rank");
          require(l.inputBegin >= 0 && l.inputBegin + kGroup <= p.k && l.inputBegin % kGroup == 0,
                  "packet location has invalid K interval");
          require(addresses.insert(l.address).second, "packet locations alias");
          for (auto n : l.outputChannels) {
            require(n >= 0 && n < p.nPad, "packet location has invalid output channel");
            for (int e = 0; e < kGroup; ++e)
              require(seen[size_t(n * p.k + l.inputBegin + e)]++ == 0, "logical weight is visited twice");
          }
          locations.push_back(l);
        });
        require(std::all_of(seen.begin(), seen.end(), [](uint8_t n) { return n == 1; }),
                "packet locations omit weight or padding lanes");
        require(locations.front().outputChannels == std::array<int64_t, 4>{0, 2, 4, 6} &&
                    locations.front().inputBegin == 0, "first packet changed lane order");
        require(locations.front().address == uint64_t(rank) * kRankBytes +
                    uint64_t(17) * channels * 32768, "first packet changed physical address");
        std::vector<int8_t> weights(size_t(p.n * p.k));
        for (size_t i = 0; i < weights.size(); ++i) weights[i] = int8_t((i * 37 + i / 512 * 11) % 255 - 127);
        size_t cursor = 0;
        plan.forEachWeightPacket(p, weights.data(), [&](uint64_t address, const std::array<uint8_t, 32> &bytes) {
          require(cursor < locations.size(), "writer emits additional packets");
          const auto &l = locations[cursor++];
          require(address == l.address, "writer and location iteration order differ");
          for (int lane = 0; lane < kLanes; ++lane)
            for (int e = 0; e < kGroup; ++e) {
              auto n = l.outputChannels[size_t(lane)];
              auto expected = n < p.n ? uint8_t(weights[size_t(n * p.k + l.inputBegin + e)]) : uint8_t(0);
              require(bytes[size_t(lane * kGroup + e)] == expected, "packet payload disagrees with logical indices");
            }
        });
        require(cursor == locations.size(), "writer omits packets");
      }
}

int main() {
  packetLocations();
  Program tails;
  tails.channels = 8;
  tails.partitions = 16;
  for (int phase = 0; phase < 8; ++phase) {
    auto loads = openRank(tails, 0);
    auto ai = tails.addActivation(256);
    auto pi = tails.addProjection("tail", 512, 256, {ai}, loads);
    require(tails.projections[pi].streamPart[0] < 8, "closed rank was selected");
    require(tails.projections[pi].streamWeightBase[0] == 0, "phase tails were not spread over free partitions");
  }
  require(*std::min_element(tails.nextWeight.begin(), tails.nextWeight.begin() + 8) ==
              *std::max_element(tails.nextWeight.begin(), tails.nextWeight.begin() + 8),
          "equal phase tails must balance resident weight storage");

  // Compute load has priority over storage, including a multi-token submission.
  auto loads = openRank(tails, 0);
  for (int c = 0; c < 8; ++c) loads[c] = 20;
  loads[3] = 0;
  tails.nextWeight[3] += 100000;
  auto ai = tails.addActivation(512);
  auto pi = tails.addProjection("priority", 512, 512, {ai, ai}, loads);
  require(tails.projections[pi].streamPart[0] == 3 && loads[3] == 4, "storage displaced phase compute priority");

  // Include streams assigned by the current projection when breaking storage ties.
  Program uneven;
  uneven.channels = uneven.partitions = 2;
  uneven.nextWeight = {0, 512};
  ai = uneven.addActivation(256);
  loads.clear();
  pi = uneven.addProjection("odd", 1536, 256, {ai}, loads);
  require(uneven.nextWeight[0] == uneven.nextWeight[1], "pending streams were omitted from storage tie-break");
  require(loads[0] == 2 && loads[1] == 1, "phase load exceeded its optimal maximum");

  // Full Llama 3.1 8B W8A8 pressure: 32 layers plus the untied LM head.
  // The old lowest-index tie-break exceeded both the 16 GiB image and rank bounds.
  Program llama;
  llama.channels = 8;
  llama.partitions = 16;
  llama.kernelRow = 162;
  int phaseCount = 0;
  auto phase = [&](int64_t k, std::initializer_list<int64_t> ns) {
    auto stored = [&](int rank) {
      int64_t sum = 0;
      for (int c = 0; c < int(llama.nextWeight.size()); ++c)
        if (llama.partRank(c) == rank) sum += llama.nextWeight[c];
      return sum;
    };
    int rank = stored(1) < stored(0) ? 1 : 0;
    auto ls = openRank(llama, rank);
    auto a = llama.addActivation(k);
    int64_t streams = 0;
    for (auto n : ns) {
      auto i = llama.addProjection("weight", n, k, {a}, ls);
      const auto &p = llama.projections[i];
      streams += p.streams();
      for (auto c : p.streamPart) require(llama.partRank(c) == rank, "phase spans closed ranks");
      require(std::is_sorted(p.streamPart.begin(), p.streamPart.end()), "partition output tiles not contiguous");
    }
    int64_t maxLoad = 0;
    for (int c = 0; c < llama.partitions; ++c)
      if (llama.partRank(c) == rank) maxLoad = std::max(maxLoad, ls[c]);
    require(maxLoad == (streams + 7) / 8 * (k / 256), "storage balance increased phase critical work");
    ++phaseCount;
  };
  for (int layer = 0; layer < 32; ++layer) {
    phase(4096, {4096, 1024, 1024});
    phase(4096, {4096});
    phase(4096, {14336, 14336});
    phase(14336, {4096});
  }
  phase(4096, {128256});
  require(phaseCount == 129, "missing Llama phase");
  require(llama.endAddress() <= (uint64_t(1) << 34), "Llama image exceeds 16 GiB");
  require(llama.rankEndAddress() <= uint64_t(kRankBytes), "PIM rows exceed one rank");
  // Each partition uses a dense, non-overlapping weight tile sequence.
  std::vector<int64_t> cursor(llama.partitions, 0);
  for (const auto &p : llama.projections)
    for (int64_t s = 0; s < p.streams(); ++s) {
      int c = p.streamPart[s];
      require(p.streamWeightBase[s] == cursor[c], "partition weight tiles overlap or have gaps");
      cursor[c] += p.streamTiles();
    }
  require(cursor == llama.nextWeight, "resident storage differs from stream allocation");
  auto copy = ::npu::decode::fromJSON(::npu::decode::toJSON(llama));
  require(copy.endAddress() == llama.endAddress() && copy.nextWeight == llama.nextWeight, "layout JSON lost allocation");
  for (size_t i = 0; i < llama.projections.size(); ++i) {
    require(copy.projections[i].streamPart == llama.projections[i].streamPart, "JSON lost partition assignment");
    require(copy.projections[i].streamWeightBase == llama.projections[i].streamWeightBase, "JSON lost weight addresses");
  }
  // Both layouts occupy the same reserved rows. Only weight row parity
  // selects the other bank of each PU; kernelRow must not affect parity.
  Program ping = llama;
  ping.kernelRow = 163;
  ping.banksetPingPongLead = 12;
  Program even = ping;
  even.banksetPingPongLead = 0;
  for (int part = 0; part < 16; ++part)
    for (int pu = 0; pu < 8; ++pu)
      for (int tile : {0, 63, 64, 127, 128, 191, 192}) {
        auto before = even.weightPacketAddress(tile, pu, part);
        auto after = ping.weightPacketAddress(tile, pu, part);
        require(after - before == uint64_t((tile / 64) % 2) * even.bankBytes(),
                "ping-pong weight packet selects the wrong bank");
        require(after < ping.endAddress(), "ping-pong packet exceeds image bounds");
      }
  require(ping.endAddress() == even.endAddress(), "ping-pong changed reserved image bounds");
  require(ping.activationPacketAddress(ping.activations[0], 0, 7, 15) ==
              even.activationPacketAddress(even.activations[0], 0, 7, 15), "ping-pong moved activations");
  require(ping.outputTileAddress(ping.projections[0], 0, 7) ==
              even.outputTileAddress(even.projections[0], 0, 7), "ping-pong moved outputs");
  auto pingCopy = ::npu::decode::fromJSON(::npu::decode::toJSON(ping));
  require(pingCopy.banksetPingPongLead == 12 &&
              pingCopy.weightPacketAddress(64, 7, 15) == ping.weightPacketAddress(64, 7, 15),
          "JSON lost ping-pong weight layout");
  std::cout << "PIM layout PASS; Llama image " << llama.endAddress() << " bytes, 129 optimal-load phases\n";
}
