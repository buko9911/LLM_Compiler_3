#include "npu/Target/PimLayout.h"
#include "npu/Target/ISAv2.h"
#include <algorithm>
#include <map>

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
                               const std::vector<int64_t> &inputs, std::map<int, int64_t> &loads, int accs,
                               int splitK, std::function<int64_t(int)> streamLoad, int tailAccs,
                               int64_t tailBlocks, int pieceRanks) {
  Projection p;
  p.name = name;
  p.n = n;
  p.k = k;
  p.accs = accs;
  p.splitK = splitK;
  p.tailAccs = tailBlocks > 0 ? tailAccs : 0;
  p.tailBlocks = tailBlocks > 0 ? tailBlocks : 0;
  const int64_t rows = int64_t(kPUs) * kLanes * accs, tailRows = int64_t(kPUs) * kLanes * p.tailAccs;
  if (p.tailBlocks) {
    // Main blocks cover what the tail does not; the tail rounds the rest up.
    const int64_t mainBlocks = std::max<int64_t>(0, (n - p.tailBlocks * tailRows + rows - 1) / rows);
    p.nPad = mainBlocks * rows + p.tailBlocks * tailRows;
  } else {
    p.nPad = (n + rows - 1) / rows * rows;
  }
  p.activations = inputs;
  p.outputBase = nextOutput;
  // Least-loaded partition per stream. The streams of one partition are then made contiguous inside
  // each K piece, main blocks and tail blocks separately (so every partition keeps the blocks, and the
  // load, the greedy pass gave it). A stream's default load is its LdMAC count (sweeps x accumulators).
  if (int(nextWeight.size()) < partitions) nextWeight.resize(size_t(partitions), 0);
  std::vector<int64_t> count(size_t(partitions), 0);
  auto stored = [&](int c) { return nextWeight[size_t(c)] + count[size_t(c)] * p.pieceSweeps() * accs * kInTiles; };
  auto load = [&](int a) {
    return streamLoad ? streamLoad(a) : p.pieceSweeps() * a * int64_t(inputs.size());
  };
  for (int64_t piece = 0; piece < splitK; ++piece) {
    std::vector<int64_t> main(size_t(partitions), 0), tail(size_t(partitions), 0);
    for (int64_t b = 0; b < p.blocks(); ++b) {
      const int rank = int(piece) % pieceRanks;
      const bool oneRank = pieceRanks > 1;
      int best = oneRank ? rank * channels : 0;
      for (int c = best + 1; c < (oneRank ? (rank + 1) * channels : partitions); ++c)
        if (loads[c] < loads[best] || (loads[c] == loads[best] && stored(c) < stored(best))) best = c;
      count[size_t(best)]++;
      (b < p.mainBlocks() ? main : tail)[size_t(best)]++;
      loads[best] += load(p.blockAccs(b));
    }
    for (const auto *here : {&main, &tail})
      for (int c = 0; c < partitions; ++c)
        for (int64_t i = 0; i < (*here)[size_t(c)]; ++i) p.streamPart.push_back(c);
  }
  // Weight tile ids are per partition: a stream's packets live only in its
  // partition's banks, and the runtime addresses a tile inside the partition
  // of the GEMV_STREAM.
  for (size_t g = 0; g < p.streamPart.size(); ++g) {
    const int c = p.streamPart[g];
    p.streamWeightBase.push_back(nextWeight[size_t(c)]);
    nextWeight[size_t(c)] += p.streamTiles(int64_t(g));
  }
  p.assignOutputSlots();
  nextOutput += int64_t(inputs.size()) * splitK * p.outputSlots;
  for (int64_t a : inputs)
    for (int c : p.streamPart)
      if (!std::count(activations[size_t(a)].parts.begin(), activations[size_t(a)].parts.end(), c))
        activations[size_t(a)].parts.push_back(c);
  for (int64_t a : inputs) std::sort(activations[size_t(a)].parts.begin(), activations[size_t(a)].parts.end());
  projections.push_back(p);
  return int64_t(projections.size()) - 1;
}

int64_t Program::addCommonProjection(const std::string &name, const std::vector<std::pair<std::string, int64_t>> &segments, int64_t k,
                                     const std::vector<int64_t> &inputs, int rank, bool mayCutK) {
  Projection p;
  p.name = name;
  p.common = true;
  p.segments = segments;
  for (const auto &seg : segments) p.n += seg.second;
  p.k = k;
  const CommonShape c = commonShape(p.n);
  p.accs = c.accs;
  p.tailAccs = c.tailAccs;
  p.tailBlocks = c.tailBlocks;
  p.nPad = c.nPad;
  p.splitK = commonLayout == 2 && mayCutK && p.sweeps() % 2 == 0 ? 2 : 1;
  p.activations = inputs;
  p.outputBase = nextOutput;
  if (int(nextWeight.size()) < partitions) nextWeight.resize(size_t(partitions), 0);
  const int64_t blocks = p.blocks();
  // Blocks go round the subchannels, so stream i of every subchannel holds the same registers. Piece p stays on rank p.
  auto partOf = [&](int64_t piece, int64_t b) {
    const int64_t inSet = b < p.mainBlocks() ? b : b - p.mainBlocks();
    const int rk = p.splitK > 1 ? int(piece) : rank;
    return rk * channels + int(inSet % kPUs);
  };
  p.streamPart.resize(size_t(blocks * p.splitK));
  for (int64_t piece = 0; piece < p.splitK; ++piece)
    for (int64_t b = 0; b < blocks; ++b) p.streamPart[size_t(piece * blocks + b)] = partOf(piece, b);
  // Emission: round by round over the subchannels, K pieces in turn; on odd rounds the second piece first, so a rank stays busy for
  // two streams in a row and the rank changes half as often.
  const int64_t rounds = (blocks + kPUs - 1) / kPUs + (p.tailBlocks % kPUs ? 1 : 0);
  (void)rounds;
  std::vector<std::vector<int64_t>> byRound;
  for (int64_t b = 0; b < blocks; ++b) {
    const size_t round = size_t(b < p.mainBlocks() ? b / kPUs : p.mainBlocks() / kPUs + (b - p.mainBlocks()) / kPUs);
    if (byRound.size() <= round) byRound.resize(round + 1);
    byRound[round].push_back(b);
  }
  for (size_t round = 0; round < byRound.size(); ++round)
    for (int64_t step = 0; step < p.splitK; ++step) {
      const int64_t piece = (round & 1) ? p.splitK - 1 - step : step;
      for (int64_t b : byRound[round]) p.streamOrder.push_back(piece * blocks + b);
    }
  p.streamWeightBase.assign(p.streamPart.size(), 0);
  for (int64_t g = 0; g < int64_t(p.streamPart.size()); ++g) {
    const int part = p.streamPart[size_t(g)];
    p.streamWeightBase[size_t(g)] = nextWeight[size_t(part)];
    nextWeight[size_t(part)] += p.streamTiles(g);
  }
  // Keep the partitions of the set level: a tail on fewer subchannels would shift the next projection's rows.
  for (int rk = 0; rk * channels < partitions; ++rk) {
    int64_t most = 0;
    for (int c2 = rk * channels; c2 < (rk + 1) * channels && c2 < partitions; ++c2) most = std::max(most, nextWeight[size_t(c2)]);
    for (int c2 = rk * channels; c2 < (rk + 1) * channels && c2 < partitions; ++c2)
      if (most) nextWeight[size_t(c2)] = (most + kPacketsPerRow * 2 - 1) / (kPacketsPerRow * 2) * (kPacketsPerRow * 2);
  }
  p.assignOutputSlots();
  nextOutput += int64_t(inputs.size()) * p.splitK * p.outputSlots;
  for (int64_t a : inputs)
    for (int part : p.streamPart)
      if (!std::count(activations[size_t(a)].parts.begin(), activations[size_t(a)].parts.end(), part))
        activations[size_t(a)].parts.push_back(part);
  for (int64_t a : inputs) std::sort(activations[size_t(a)].parts.begin(), activations[size_t(a)].parts.end());
  projections.push_back(p);
  return int64_t(projections.size()) - 1;
}

std::vector<std::pair<int64_t, int64_t>> Program::activationPackets(int64_t index, int rank) const {
  std::vector<std::pair<int64_t, int64_t>> ranges;
  for (const Projection &p : projections) {
    if (!std::count(p.activations.begin(), p.activations.end(), index)) continue;
    const int64_t packetsPerSweep = kInTiles / kSlices;
    for (int64_t g = 0; g < p.streams(); ++g) {
      if (partRank(p.streamPart[size_t(g)]) != rank) continue;
      const int64_t piece = g / p.blocks();
      ranges.push_back({piece * p.pieceSweeps() * packetsPerSweep, (piece + 1) * p.pieceSweeps() * packetsPerSweep});
    }
  }
  std::sort(ranges.begin(), ranges.end());
  std::vector<std::pair<int64_t, int64_t>> merged;
  for (auto r : ranges) {
    if (!merged.empty() && r.first <= merged.back().second) merged.back().second = std::max(merged.back().second, r.second);
    else merged.push_back(r);
  }
  return merged;
}

void Projection::assignOutputSlots() {
  streamSlot.assign(size_t(streams()), 0);
  outputSlots = 0;
  for (int64_t piece = 0; piece < splitK; ++piece) {
    std::map<int, int64_t> used;
    for (int64_t b = 0; b < blocks(); ++b) {
      const int64_t g = piece * blocks() + b;
      int64_t &slot = used[streamPart[size_t(g)]];
      streamSlot[size_t(g)] = slot;
      slot += tilesPerStream(g);
      outputSlots = std::max(outputSlots, slot);
    }
  }
}

int64_t Program::phaseOf(const Projection &p) const {
  if (p.computeCount <= 0) return -1;
  for (size_t i = 0; i < phases.size(); ++i)
    if (p.computeFirst >= phases[i].first && p.computeFirst < phases[i].first + phases[i].second) return int64_t(i);
  return -1;
}

void Program::assignOutputs() {
  nextOutput = outputPeak = 0;
  int64_t phase = -1;
  for (Projection &p : projections) {
    if (reuseBuffers && p.computeCount > 0 && phaseOf(p) != phase) {
      phase = phaseOf(p);
      nextOutput = 0;
    }
    p.outputBase = nextOutput;
    p.assignOutputSlots();
    nextOutput += int64_t(p.tokens()) * p.splitK * p.outputSlots;
    outputPeak = std::max(outputPeak, nextOutput);
  }
}

bool Program::reuseBuffersAcrossPhases() {
  std::map<int64_t, int64_t> owner;   // activation -> phase
  for (const Projection &p : projections) {
    const int64_t ph = phaseOf(p);
    if (ph < 0) continue;
    for (int64_t a : p.activations) {
      auto [it, fresh] = owner.emplace(a, ph);
      if (!fresh && it->second != ph) return false;
    }
  }
  std::map<int64_t, int64_t> actLo, outLo;
  for (auto [a, ph] : owner) {
    auto [it, fresh] = actLo.emplace(ph, activations[size_t(a)].base);
    if (!fresh) it->second = std::min(it->second, activations[size_t(a)].base);
  }
  for (const Projection &p : projections)
    if (int64_t ph = phaseOf(p); ph >= 0) {
      auto [it, fresh] = outLo.emplace(ph, p.outputBase);
      if (!fresh) it->second = std::min(it->second, p.outputBase);
    }
  activationPeak = outputPeak = 0;
  for (auto [a, ph] : owner) {
    Activation &x = activations[size_t(a)];
    x.base -= actLo[ph];
    activationPeak = std::max(activationPeak, x.base + x.k / kGroup);
  }
  for (Projection &p : projections) {
    const int64_t ph = phaseOf(p);
    p.outputBase = ph < 0 ? 0 : p.outputBase - outLo[ph];
    outputPeak = std::max(outputPeak, p.outputBase + int64_t(p.tokens()) * p.splitK * p.outputSlots);
  }
  reuseBuffers = true;
  return true;
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
  const int64_t streams = p.streams(), blocks = p.blocks();
  auto validAccs = [](int a) { return a >= 2 && a <= kAccs && a % 2 == 0; };
  if (!validAccs(p.accs) || (p.tailBlocks && !validAccs(p.tailAccs)) || p.splitK < 1 || p.sweeps() % p.splitK)
    return Result<bool>::failure(p.name + ": PIM stream needs an even accumulator count 2..16 and whole-sweep K pieces");
  for (int m = 0; m < p.tokens(); ++m) {
    const Activation &a = activations[size_t(p.activations[size_t(m)])];
    for (int64_t order = 0; order < streams; ++order) {
      const int64_t g = p.streamOrder.empty() ? order : p.streamOrder[size_t(order)];
      const int64_t block = g % blocks, piece = g / blocks;
      const int accs = p.blockAccs(block);
      auto make = [&](Opcode opcode) {
        Primitive q;
        q.opcode = opcode;
        q.partition = uint32_t(p.streamPart[size_t(g)]);
        return q;
      };
      for (int acc = 0; acc < accs; ++acc) {
        auto q = make(Opcode::ClearOutput); q.accumulator = acc;
        commands.push_back(q);
      }
      for (int64_t sweep = 0; sweep < p.pieceSweeps(); ++sweep) {
        int64_t firstA = a.base + (piece * p.pieceSweeps() + sweep) * kInTiles;
        int64_t firstW = p.streamWeightBase[size_t(g)] + sweep * p.sweepTiles(accs);
        if (firstA + kInTiles > INT32_MAX || firstW + accs * kInTiles > INT32_MAX)
          return Result<bool>::failure(p.name + ": PIM tile ID exceeds INT32_MAX");
        // A load fills a whole 32-byte input register (four logical K=8
        // slices). Emit it once per packet; all slices remain addressable.
        for (int i = 0; i < kInTiles; i += kSlices) {
          auto q = make(Opcode::LoadActivation); q.activation = int32_t(firstA + i);
          commands.push_back(q);
        }
        auto q = make(Opcode::GemvStream);
        q.weight = int32_t(firstW); q.activation = int32_t(firstA); q.accumulator = 0;
        q.inputs = kInTiles; q.outputs = accs;
        q.weightInputStride = 1; q.weightOutputStride = kInTiles;
        q.activationInputStride = 1; q.accumulatorOutputStride = 1;
        commands.push_back(q);
      }
      int64_t firstO = p.outputTile(m, int(piece), g, 0);

      if (firstO + accs / 2 > INT32_MAX)
        return Result<bool>::failure(p.name + ": PIM output tile ID exceeds INT32_MAX");
      for (int reg = 0; reg < accs / 2; ++reg) {
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
  int64_t output = activation + rowsFor(activationTiles());
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
  return uint64_t(baseRows().output + rowsFor(outputTiles())) * uint64_t(rowBytes());
}

uint64_t Program::endAddress() const {
  uint64_t end = rankEndAddress();
  if (end == uint64_t(kernelRow) * uint64_t(rowBytes())) return end;   // no PIM rows at all
  return uint64_t(partRank(partitions - 1)) * uint64_t(kRankBytes) + end;
}
} // namespace npu::pim
