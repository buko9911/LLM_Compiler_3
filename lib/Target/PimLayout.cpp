#include "npu/Target/PimLayout.h"
#include "npu/Target/ISAv2.h"
#include <algorithm>

namespace npu::pim {
namespace {
int64_t rowsFor(int64_t tiles) { return tiles == 0 ? 0 : (tiles + kPacketsPerRow - 1) / kPacketsPerRow; }
} // namespace

int64_t Program::addActivation(int64_t k) {
  Activation a;
  a.k = k;
  a.base = nextActivation;
  nextActivation += k / kGroup;
  activations.push_back(a);
  return int64_t(activations.size()) - 1;
}

int64_t Program::addProjection(const std::string &name, int64_t n, int64_t k,
                               const std::vector<int64_t> &inputs, std::map<int, int64_t> &loads) {
  Projection p;
  p.name = name;
  p.n = n;
  p.k = k;
  p.nPad = (n + kRowsPerStream - 1) / kRowsPerStream * kRowsPerStream;
  p.activations = inputs;
  p.outputBase = nextOutput;
  // Least-loaded partition per stream, then regrouped so the streams of one
  // partition are contiguous and its stored output tiles stay contiguous.
  std::vector<int64_t> count(size_t(partitions), 0);
  if (int(nextWeight.size()) < partitions) nextWeight.resize(size_t(partitions), 0);
  auto stored = [&](int c) { return nextWeight[size_t(c)] + count[size_t(c)] * p.streamTiles(); };
  for (int64_t s = 0; s < p.streams(); ++s) {
    int best = 0;
    for (int c = 1; c < partitions; ++c)
      if (loads[c] < loads[best] || (loads[c] == loads[best] && stored(c) < stored(best))) best = c;
    count[size_t(best)]++;
    loads[best] += p.sweeps() * int64_t(inputs.size());
  }
  for (int c = 0; c < partitions; ++c)
    for (int64_t i = 0; i < count[size_t(c)]; ++i) p.streamPart.push_back(c);
  // Weight tile ids are per partition: a stream's packets live only in its
  // partition's banks, and the runtime addresses a tile inside the partition
  // of the GEMV_STREAM.
  for (int c : p.streamPart) {
    p.streamWeightBase.push_back(nextWeight[size_t(c)]);
    nextWeight[size_t(c)] += p.streamTiles();
  }
  nextOutput += p.streams() * 8 * int64_t(inputs.size());
  for (int64_t a : inputs)
    for (int c : p.streamPart)
      if (!std::count(activations[size_t(a)].parts.begin(), activations[size_t(a)].parts.end(), c))
        activations[size_t(a)].parts.push_back(c);
  for (int64_t a : inputs) std::sort(activations[size_t(a)].parts.begin(), activations[size_t(a)].parts.end());
  projections.push_back(p);
  return int64_t(projections.size()) - 1;
}

int64_t Program::storedTiles(int rank) const {
  int64_t tiles = 0;
  for (int c = 0; c < partitions && c < int(nextWeight.size()); ++c)
    if (partRank(c) == rank) tiles += nextWeight[size_t(c)];
  return tiles;
}

Result<bool> Program::emitCompute(int64_t index, std::vector<Primitive> &commands) {
  // Scheduling is a compiler decision. The runtime receives this exact
  // primitive sequence; it does not reconstruct a fixed GEMV macro.
  Projection &p = projections[size_t(index)];
  p.computeFirst = int64_t(commands.size());
  const int64_t streams = p.streams();
  for (int m = 0; m < p.tokens(); ++m) {
    const Activation &a = activations[size_t(p.activations[size_t(m)])];
    for (int64_t g = 0; g < streams; ++g) {
      auto make = [&](Opcode opcode) {
        Primitive q;
        q.opcode = opcode;
        q.partition = uint32_t(p.streamPart[size_t(g)]);
        return q;
      };
      for (int acc = 0; acc < kAccs; ++acc) {
        auto q = make(Opcode::ClearOutput); q.accumulator = acc;
        commands.push_back(q);
      }
      for (int64_t sweep = 0; sweep < p.sweeps(); ++sweep) {
        int64_t firstA = a.base + sweep * kInTiles;
        int64_t firstW = p.streamWeightBase[size_t(g)] + sweep * kAccs * kInTiles;
        if (firstA + kInTiles > INT32_MAX || firstW + kAccs * kInTiles > INT32_MAX)
          return Result<bool>::failure(p.name + ": PIM tile ID exceeds INT32_MAX");
        // A load fills a whole 32-byte input register (four logical K=8
        // slices). Emit it once per packet; all slices remain addressable.
        for (int i = 0; i < kInTiles; i += kSlices) {
          auto q = make(Opcode::LoadActivation); q.activation = int32_t(firstA + i);
          commands.push_back(q);
        }
        auto q = make(Opcode::GemvStream);
        q.weight = int32_t(firstW); q.activation = int32_t(firstA); q.accumulator = 0;
        q.inputs = kInTiles; q.outputs = kAccs;
        q.weightInputStride = 1; q.weightOutputStride = kInTiles;
        q.activationInputStride = 1; q.accumulatorOutputStride = 1;
        commands.push_back(q);
      }
      int64_t firstO = p.outputBase + (int64_t(m) * streams + g) * (kAccs / 2);
      if (firstO + kAccs / 2 > INT32_MAX)
        return Result<bool>::failure(p.name + ": PIM output tile ID exceeds INT32_MAX");
      for (int reg = 0; reg < kAccs / 2; ++reg) {
        auto q = make(Opcode::StoreOutput);
        q.accumulator = 2 * reg; q.output = int32_t(firstO + reg);
        commands.push_back(q);
      }
    }
  }
  p.computeCount = int64_t(commands.size()) - p.computeFirst;
  return Result<bool>::success(true);
}

Program::BaseRows Program::baseRows() const {
  int64_t weight = kernelRow, most = 0;
  for (int64_t n : nextWeight) most = std::max(most, n);
  int64_t activation = weight + rowsFor(most);
  // The runtime reserves activation rows by activation tile count.
  int64_t output = activation + rowsFor(nextActivation);
  return {weight, activation, output};
}

uint64_t Program::packetAddress(int64_t baseRow, int64_t packet, int pu, int part) const {
  int channel = partChannel(part), rank = partRank(part);
  int64_t row = baseRow + packet / kPacketsPerRow, column = packet % kPacketsPerRow;
  int bank = pu * 2;   // the even bank set
  int group = bank / 4, inGroup = bank % 4;
  return uint64_t(rank) * uint64_t(kRankBytes) + uint64_t(row) * uint64_t(rowBytes()) +
         uint64_t(inGroup) * uint64_t(bankBytes()) + uint64_t(column) * uint64_t(colBytes()) +
         uint64_t(group) * uint64_t(bgBytes()) + uint64_t(channel) * kChBytes;
}

uint64_t Program::rankEndAddress() const {
  return uint64_t(baseRows().output + rowsFor(nextOutput)) * uint64_t(rowBytes());
}

uint64_t Program::endAddress() const {
  uint64_t end = rankEndAddress();
  if (end == uint64_t(kernelRow) * uint64_t(rowBytes())) return end;   // no PIM rows at all
  return uint64_t(partRank(partitions - 1)) * uint64_t(kRankBytes) + end;
}
} // namespace npu::pim
