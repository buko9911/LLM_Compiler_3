//===- CodegenPim.cpp - NPU codegen: PIM boundary ---------------------------===//
//
// Codegen members that move data across the PIM boundary: INT32 PIM outputs gathered from
// their tiles, INT8 activations scattered to every consuming PU bank, and the checked 4-core
// activation feed of an isolated quant phase.
//===----------------------------------------------------------------------===//

#include "CodegenImpl.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include <algorithm>
#include <limits>
#include <set>

namespace mlir::hybrid::codegen {

/// INT32 GEMV results [rows, cols] of a PIM output -> contiguous L1 rows. A stored tile holds
/// 8 consecutive outputs of one PU. For each token, K piece and PU, every partition keeps its
/// tiles under the same ids (Projection::outputTile), so one bank group's part of a column (all
/// subchannels, 32 B each) holds that PU's tile of every partition: those slices are gathered with
/// GDMAs that read every subchannel at once, then each partition's run of tiles is moved to its
/// logical place by one strided LDMA. A split-K projection stores one partial sum per K piece:
/// piece 0 lands in the destination, every further piece in the L1 scratch, and the VPU adds it
/// (INT32, so the sum is exact and independent of the split).
LogicalResult Codegen::gatherPimOut(const View &s, const View &d, std::string &err) {
  const pim::Projection &p = in.pim->projections[size_t(s.lp6.getAs<IntegerAttr>("proj").getInt())];
  if (p.common) return gatherPimCommon(s, d, err);
  if (s.shape.size() != 2 || s.strides[1] != 1 || d.space != kL1 || d.strides.back() != 1) {
    err = "PIM output copy must be [rows, cols] into contiguous L1 rows";
    return failure();
  }
  int64_t N = s.strides[0], m0 = s.offset / N, n0 = s.offset % N, rows = s.shape[0], cols = s.shape[1];
  if (n0 % 8 || cols % 8) {
    err = "PIM output columns must be whole tiles of 8";
    return failure();
  }
  int64_t dRow = d.shape.size() == 2 ? d.strides[0] : cols;
  const int64_t T = p.tilesPerPU(), npu = p.rowsPerPU(), blocks = p.blocks();
  const int channels = in.pim->channels;
  const int64_t slice = int64_t(pim::kChBytes) * channels, colBytes = in.pim->colBytes();
  // First PU-local tile after t whose stream lives in another partition (inside one K piece).
  auto partEnd = [&](int64_t t, int piece) {
    int64_t b = p.blockOfTile(t);
    const int part = p.streamPart[size_t(piece * blocks + b)];
    while (b < blocks && p.streamPart[size_t(piece * blocks + b)] == part) ++b;
    return p.blockTile(b);
  };
  const uint64_t win = windowBase(curCore), winEnd = win + uint64_t(window());
  int64_t chunk = window() / 4 / 8 * 8;
  const uint64_t partial = scratchBase;
  if (p.splitK > 1) {
    const int64_t scratch = (in.spec->npu.l1ReservedBytes - kPoolMax) / 4 / 16 * 16;
    if (scratch < 16) {
      err = "split-K PIM output needs L1 scratch for one partial-sum chunk";
      return failure();
    }
    chunk = std::min(chunk, scratch);
  }
  struct Run { int64_t t, n, id; int part; };
  uint64_t &stage = gatherStage.try_emplace(curCore, win).first->second;
  for (int64_t r = 0; r < rows; ++r) {
    int m = int(m0 + r);
    for (int64_t c0 = n0; c0 < n0 + cols; c0 += chunk) {
      int64_t c1 = std::min(n0 + cols, c0 + chunk);
      const uint64_t dst = addr(d, r * dRow + (c0 - n0));
      for (int piece = 0; piece < p.splitK; ++piece) {
        const uint64_t target = piece == 0 ? dst : partial;
        for (int pu = 0; pu < pim::kPUs; ++pu) {
          int64_t lo = std::max(c0, int64_t(pu) * npu), hi = std::min(c1, int64_t(pu + 1) * npu);
          if (lo >= hi) continue;
          int64_t ta = (lo - pu * npu) / 8, tb = std::min(T, (hi - pu * npu) / 8);
          SmallVector<Run> runs;
          std::set<int> ranks;
          for (int64_t t = ta; t < tb;) {
            const int64_t g = p.streamOfTile(t, piece);
            const int part = p.streamPart[size_t(g)];
            const int64_t n = std::min(tb - t, partEnd(t, piece) - t);
            runs.push_back({t, n, p.outputTile(m, piece, g, int(t - p.blockTile(p.blockOfTile(t)))), part});
            ranks.insert(in.pim->partRank(part));
            t += n;
          }
          for (int rank : ranks) {
            int64_t idLo = std::numeric_limits<int64_t>::max(), idHi = 0;
            for (const Run &run : runs)
              if (in.pim->partRank(run.part) == rank) {
                idLo = std::min(idLo, run.id);
                idHi = std::max(idHi, run.id + run.n);
              }
            const uint64_t need = uint64_t(idHi - idLo) * uint64_t(slice);
            if (need > uint64_t(window())) {
              err = "PIM output gather: one PU's tile slices exceed the L2 staging window";
              return failure();
            }
            if (stage + need > winEnd) stage = win;   // reuse waits for the earlier readers (L2 hazards)
            const uint64_t base = stage;
            stage += need;
            for (int64_t id = idLo; id < idHi;) {
              const int64_t n = std::min(idHi - id, pim::kPacketsPerRow - id % pim::kPacketsPerRow);
              lastGatherRead[curCore] =
                  gdmaLoad(in.pim->outputTileAddress(id, pu, rank * channels), base + uint64_t((id - idLo) * slice),
                           uint32_t(slice), uint32_t(n), uint32_t(colBytes), uint32_t(slice));
              id += n;
            }
            for (const Run &run : runs)
              if (in.pim->partRank(run.part) == rank)
                ldma(target + uint64_t((pu * npu + run.t * 8 - c0) * 4),
                     base + uint64_t((run.id - idLo) * slice + in.pim->partChannel(run.part) * pim::kChBytes),
                     uint32_t(pim::kChBytes), uint32_t(run.n), uint32_t(slice), uint32_t(pim::kChBytes), true);
          }
        }
        if (piece == 0) continue;
        for (int64_t e = 0; e < c1 - c0; e += 16) {
          const uint32_t lanes = uint32_t(std::min<int64_t>(16, c1 - c0 - e));
          setVL(lanes);
          vload(0, dst + uint64_t(e * 4), 7);        // I32
          vload(1, partial + uint64_t(e * 4), 7);
          rec(typedVector(T_ADD, DT_I32, 2, 0, 1, 0, lanes));
          vstore(dst + uint64_t(e * 4), 2, 8);
        }
      }
    }
  }
  return success();
}

/// Research option --common-layout (PimLayout.h, commonShape). A column of a PU holds 64 consecutive outputs over the 8 subchannels, so
/// one GDMA per run of columns and one contiguous LDMA per PU put a K piece in order. Further K pieces are added as in gatherPimOut.
LogicalResult Codegen::gatherPimCommon(const View &s, const View &d, std::string &err) {
  const pim::Projection &p = in.pim->projections[size_t(s.lp6.getAs<IntegerAttr>("proj").getInt())];
  if (s.shape.size() != 2 || s.strides[1] != 1 || d.space != kL1 || d.strides.back() != 1) {
    err = "PIM output copy must be [rows, cols] into contiguous L1 rows";
    return failure();
  }
  const int64_t segBase = s.lp6.getAs<IntegerAttr>("seg_base").getInt();
  int64_t N = s.strides[0], m0 = s.offset / N, n0 = s.offset % N, rows = s.shape[0], cols = s.shape[1];
  int64_t dRow = d.shape.size() == 2 ? d.strides[0] : cols;
  const int channels = in.pim->channels;
  const int64_t slice = int64_t(pim::kChBytes) * channels, colBytes = in.pim->colBytes(), perPU = p.rowsPerPU();
  const uint64_t win = windowBase(curCore), winEnd = win + uint64_t(window());
  int64_t chunk = window() / 4 / 64 * 64;
  const uint64_t partial = scratchBase;
  if (p.splitK > 1) {
    const int64_t scratch = (in.spec->npu.l1ReservedBytes - kPoolMax) / 4 / 64 * 64;
    if (scratch < 64) {
      err = "split-K PIM output needs L1 scratch for one partial-sum chunk";
      return failure();
    }
    chunk = std::min(chunk, scratch);
  }
  uint64_t &stage = gatherStage.try_emplace(curCore, win).first->second;
  for (int64_t r = 0; r < rows; ++r) {
    const int m = int(m0 + r);
    for (int64_t c0 = n0; c0 < n0 + cols; c0 += chunk) {
      const int64_t c1 = std::min(n0 + cols, c0 + chunk), f0 = segBase + c0, f1 = segBase + c1;
      const uint64_t dst = addr(d, r * dRow + (c0 - n0));
      for (int piece = 0; piece < p.splitK; ++piece) {
        const uint64_t target = piece == 0 ? dst : partial;
        const int rank = in.pim->partRank(p.streamPart[size_t(piece * p.blocks())]);
        for (int index = 0; index < pim::kPUs; ++index) {
          const int pu = p.commonPU(index);
          const int64_t lo = std::max(f0, index * perPU), hi = std::min(f1, (index + 1) * perPU);
          if (lo >= hi) continue;
          const int64_t nl = lo - index * perPU, nh = hi - index * perPU, Ra = nl / 64, Rb = (nh + 63) / 64;
          const uint64_t need = uint64_t(Rb - Ra) * uint64_t(slice);
          if (need > uint64_t(window())) {
            err = "PIM output gather: one PU's columns exceed the L2 staging window";
            return failure();
          }
          if (stage + need > winEnd) stage = win;   // reuse waits for the earlier readers (L2 hazards)
          const uint64_t base = stage;
          stage += need;
          const int64_t idBase = p.outputBase + (int64_t(m) * p.splitK + piece) * p.outputSlots;
          for (int64_t R = Ra; R < Rb;) {
            const int64_t id = idBase + R, n = std::min(Rb - R, pim::kPacketsPerRow - id % pim::kPacketsPerRow);
            lastGatherRead[curCore] = gdmaLoad(in.pim->outputTileAddress(id, pu, rank * channels), base + uint64_t((R - Ra) * slice),
                                               uint32_t(slice), uint32_t(n), uint32_t(colBytes), uint32_t(slice));
            R += n;
          }
          ldma(target + uint64_t((lo - f0) * 4), base + uint64_t((nl - Ra * 64) * 4), uint32_t((nh - nl) * 4), 1, 0, 0, true);
        }
        if (piece == 0) continue;
        for (int64_t e = 0; e < c1 - c0; e += 16) {
          const uint32_t lanes = uint32_t(std::min<int64_t>(16, c1 - c0 - e));
          setVL(lanes);
          vload(0, dst + uint64_t(e * 4), 7);        // I32
          vload(1, partial + uint64_t(e * 4), 7);
          rec(typedVector(T_ADD, DT_I32, 2, 0, 1, 0, lanes));
          vstore(dst + uint64_t(e * 4), 2, 8);
        }
      }
    }
  }
  return success();
}

/// Research option --common-layout: the read of one NPU weight tile, in GDMAs of commonChunkRows x 128 B per 4 subchannels. The tile's
/// subchannel group has its own admission chain, so the tiles on the two groups are read at once, and the chains of the cores interleave.
/// Only the first two GDMAs wait for the slot's earlier users and the last two guard the slot: inside a chain each follows the one two back.
bool Codegen::commonWeightLoad(uint64_t logical, uint64_t slot, uint32_t bytes) {
  if (!in.commonWeights) return false;
  const pim::CommonWeight *cw = nullptr;
  for (const pim::CommonWeight &c : *in.commonWeights)
    if (logical >= c.logical && logical < c.logical + c.bytes) { cw = &c; break; }
  if (!cw) return false;
  const int64_t tile = int64_t(logical - cw->logical) / (cw->k * 32), firstRow = (int64_t(logical - cw->logical) % (cw->k * 32)) / 32;
  const int64_t rowEnd = firstRow + int64_t(bytes) / 32;
  if (firstRow % 4 || rowEnd % 4 || rowEnd > cw->k) llvm::report_fatal_error("common layout: a weight read must cover whole groups of 4 K rows of one tile");
  const uint64_t end = slot + bytes;
  for (const Acc &a : l2acc)
    if (a.id < 0 && a.core == curCore && overlaps(a.s, a.e, slot, end)) {
      flush();
      break;
    }
  std::vector<int> deps;
  for (const Acc &a : l2acc)
    if (overlaps(a.s, a.e, slot, end) && a.id >= 0) deps.push_back(a.id);
  if (readAfterGather)
    for (auto [core, id] : lastGatherRead) deps.push_back(id);
  reachableDependencies(deps);
  std::sort(deps.begin(), deps.end());
  deps.erase(std::unique(deps.begin(), deps.end()), deps.end());
  const pim::CommonShape shape = pim::commonShape(cw->n);
  const int64_t local = (tile * 32) % (shape.nPad / pim::kPUs);
  w.gdmaChainGroup = 1 + int((local % 64) / 32);
  const int chunkRows = std::max(1, in.commonChunkRows);
  int last = -1, previous = -1, count = 0;
  for (int64_t row = firstRow; row < rowEnd; row += 4 * chunkRows) {
    // A chunk stays inside one sweep: the columns of a DRAM row.
    const int64_t inner = row % pim::kKPerSweep, rows4 = std::min({int64_t(chunkRows), (pim::kKPerSweep - inner) / 4, (rowEnd - row) / 4});
    const int id = w.appendGdma(ProgramBuilder::Kind::GdmaLoad, count < 2 ? deps : std::vector<int>{}, cw->address(tile, row),
                                slot + uint64_t(row - firstRow) * 32, 128, uint32_t(rows4), 1024, 128, true);
    previous = last;
    last = id;
    ++count;
    if (rows4 != chunkRows) row += 4 * (rows4 - chunkRows);
  }
  w.gdmaChainGroup = 0;
  llvm::erase_if(l2acc, [&](const Acc &a) { return a.s >= slot && a.e <= end; });
  l2acc.push_back({slot, end, last, curCore, GW});
  if (previous >= 0) l2acc.push_back({slot, end, previous, curCore, GW});
  return true;
}

/// L1 scratch columns for `packets` activation packets at `src`: column p (colBytes() = 4 bank groups x
/// C subchannels x 32 B) holds 4 C copies of packet p. The first copy is placed, then the filled part of
/// every column is doubled. Rows of 32 B with the column stride all fall on the same 8 of the 32 SRAM
/// banks (one row per cycle); the doubling copies 64..colBytes/2-byte rows, which use every bank.
void Codegen::replicateColumns(uint64_t dup, uint64_t src, int64_t packets) {
  const int64_t colBytes = in.pim->colBytes();
  l1Copy(dup, src, 32, uint32_t(packets), 32, uint32_t(colBytes));
  for (int64_t have = 32; have < colBytes; have *= 2)
    l1Copy(dup + uint64_t(have), dup, uint32_t(std::min(have, colBytes - have)), uint32_t(packets),
           uint32_t(colBytes), uint32_t(colBytes));
}

/// Stores activation packets [first, first + count), replicated at L2 `l2`: for every rank the packets that rank's
/// streams read, one linear store per bank set. A rank that computes only some K pieces takes only their packets.
void Codegen::storeActivationPackets(int64_t actIndex, uint64_t l2, int64_t first, int64_t count) {
  const pim::Activation &a = in.pim->activations[size_t(actIndex)];
  const int64_t colBytes = in.pim->colBytes();
  const int channels = in.pim->channels;
  std::set<int> ranks;
  for (int part : a.parts) ranks.insert(in.pim->partRank(part));
  for (int rk : ranks)
    for (auto [lo, hi] : in.pim->activationPackets(actIndex, rk)) {
      const int64_t from = std::max(lo, first), to = std::min(hi, first + count);
      for (int64_t p = from; p < to;) {
        const int64_t pk = a.packetBase() + p;
        const int64_t n = std::min(to - p, pim::kPacketsPerRow - pk % pim::kPacketsPerRow);
        for (int parity = 0; parity < 2; ++parity)   // bank 0 (PUs 0,2,4,6) then bank 2 (PUs 1,3,5,7)
          gdmaStore(l2 + uint64_t((p - first) * colBytes), in.pim->activationPacketAddress(a, p, parity, rk * channels),
                    uint32_t(colBytes * n), 1, uint32_t(colBytes * n), uint32_t(colBytes * n));
        p += n;
      }
    }
}

/// The feed of scatterPimAct shared among `feedCores` cores (hyb-pim-boundary chose it): the core that quantized the
/// vector publishes it to its small-transfer staging, every other core copies its share of the packets to the L1
/// scratch, and each core replicates and stores its own packets. Other cores touch only their L1 scratch and their L2
/// staging window, which the phase reserves for them.
LogicalResult Codegen::scatterPimActCores(const View &s, const View &d, int feedCores, std::string &err) {
  constexpr int64_t kFeedBatchesPerCore = 4;
  auto acts = d.lp6.getAs<ArrayAttr>("acts");
  const int64_t K = d.shape[1], sRow = s.shape.size() == 2 ? s.strides[0] : K;
  if (s.space != kL1 || s.strides.back() != 1 || d.strides.size() != 2 || d.offset || d.shape[0] != int64_t(acts.size())) {
    err = "shared PIM activation feed needs whole contiguous L1 rows of the activation buffer";
    return failure();
  }
  const int source = curCore;
  const int64_t colBytes = in.pim->colBytes();
  const int64_t scratch = in.spec->npu.l1ReservedBytes - kPoolMax;
  std::map<int, uint64_t> stages;   // per core: next free bytes of the staging window, across the tokens
  for (int64_t m = 0; m < d.shape[0]; ++m) {
    const int64_t actIndex = cast<IntegerAttr>(acts[size_t(m)]).getInt();
    const pim::Activation &a = in.pim->activations[size_t(actIndex)];
    const uint64_t row = addr(s, m * sRow);
    // Publish the vector to L2. The block that ends here is what the other cores wait for.
    const uint64_t published = stagingFor(uint64_t(K));
    ldma(row, published, uint32_t(K), 1, 0, 0, false);
    const int publish = flush();
    if (publish < 0) {
      err = "PIM activation publication emitted no core block";
      return failure();
    }
    const int64_t packets = a.packets();
    for (int core = 0; core < feedCores; ++core) {
      const int64_t begin = packets * core / feedCores, end = packets * (core + 1) / feedCores;
      if (begin == end) continue;
      const int c = (source + core) % feedCores;
      startCore(c, c == source ? std::vector<int>{} : std::vector<int>{publish});
      uint64_t src = row + uint64_t(begin * 32);
      const uint64_t dup = scratchBase, win = windowBase(c);
      const int64_t partBytes = (end - begin) * 32;
      int64_t budget = scratch;
      if (c != source) {
        // The packets sit at the end of the L1 scratch, the replicas grow from its start.
        src = dup + uint64_t((scratch - partBytes) / 64 * 64);
        budget = (scratch - partBytes) / 64 * 64;
        ldma(src, published + uint64_t(begin * 32), uint32_t(partBytes), 1, 0, 0, true);
      }
      // A few batches per core, so the stores of one run while the next is copied to L2.
      const int64_t pipelined = std::max<int64_t>(1, (end - begin + kFeedBatchesPerCore - 1) / kFeedBatchesPerCore);
      const int64_t batch = std::min({pipelined, budget / colBytes, (int64_t(ringWindow()) - 32 * pim::kPacketsPerRow) / colBytes});
      if (batch < 1) {
        err = "shared PIM activation feed needs one column of L1 scratch and L2 staging";
        return failure();
      }
      uint64_t &stage = stages.try_emplace(c, win).first->second;
      const uint64_t windowEnd = win + uint64_t(ringWindow());
      for (int64_t first = begin; first < end; first += batch) {
        const int64_t count = std::min(batch, end - first);
        if (stage + uint64_t(colBytes * count) > windowEnd) stage = win;
        replicateColumns(dup, src + uint64_t((first - begin) * 32), count);
        ldma(dup, stage, uint32_t(colBytes * count), 1, 0, 0, false);
        storeActivationPackets(actIndex, stage, first, count);
        stage += uint64_t(colBytes * count);
      }
      endCore();
    }
    curCore = source;
    w.logicalCore = source;
    w.frontier = last[size_t(source)];
  }
  return success();
}

/// hyb-pim-boundary chose to split this quant of one row over the cores, with the feed of its output. The scale is the
/// maximum over the whole row, so the cores first publish the maximum of their columns, then each takes all maxima,
/// quantizes its columns and replicates and stores its own packets. Every core works in its L1 scratch and its L2 staging
/// window (the phase reserves them for hyb.feed_cores); core `source` also keeps the scale for the copy after the op.
LogicalResult Codegen::lowerQuantCores(hyb::QuantOp q, std::string &err) {
  FailureOr<View> Q = viewOf(q.getQInit(), err), S = viewOf(q.getScaleInit(), err);
  if (failed(Q) || failed(S)) return failure();
  const int cores = int(q->getAttrOfType<IntegerAttr>("hyb.quant_cores").getInt());
  memref::CopyOp feed;
  for (Operation *op = q->getNextNode(); op && !feed; op = op->getNextNode())
    if (auto c = dyn_cast<memref::CopyOp>(op); c && c->hasAttr("hyb.quant_feeds")) feed = c;
  FailureOr<View> D = feed ? viewOf(feed.getTarget(), err) : FailureOr<View>(failure());
  if (failed(D)) {
    if (err.empty()) err = "a quant split by columns needs the feed of its output";
    return failure();
  }
  auto acts = D->lp6.getAs<ArrayAttr>("acts");
  const int64_t K = Q->shape.back(), part = K / cores;
  if (!quantSourceL2 || !acts || acts.size() != 1 || K % cores || part % 32) {
    err = "a quant split by columns needs its L2 row, one activation and whole packets per core";
    return failure();
  }
  const pim::Activation &a = in.pim->activations[size_t(cast<IntegerAttr>(acts[0]).getInt())];
  if (a.packets() * 32 != K) {
    err = "a quant split by columns needs one packet per 32 elements";
    return failure();
  }
  const uint64_t x = *quantSourceL2;
  quantSourceL2.reset();
  const int source = curCore;
  const int64_t colBytes = in.pim->colBytes(), scratch = in.spec->npu.l1ReservedBytes - kPoolMax;
  const uint64_t xL1 = scratchBase, qL1 = xL1 + uint64_t((part * 2 + 63) / 64 * 64), maxL1 = qL1 + uint64_t((part + 63) / 64 * 64),
                 dup = maxL1 + 64;
  const int64_t budget = scratch - int64_t(dup - scratchBase);
  const uint64_t published = stagingFor(uint64_t(cores) * 16);
  std::vector<int> maxima;
  for (int core = 0; core < cores; ++core) {
    const int c = (source + core) % cores;
    startCore(c, {});
    ldma(xL1, x + uint64_t(c * part * 2), uint32_t(part * 2), 1, 0, 0, true);
    quantMax(xL1, part);
    sstore(maxL1, 0);
    ldma(maxL1, published + uint64_t(c) * 16, 4, 1, 0, 0, false);
    endCore();
    maxima.insert(maxima.end(), last[size_t(c)].begin(), last[size_t(c)].end());
  }
  const int64_t packets = a.packets();
  std::map<int, uint64_t> stages;
  for (int core = 0; core < cores; ++core) {
    const int c = (source + core) % cores;
    const int64_t begin = packets * c / cores, end = packets * (c + 1) / cores;
    startCore(c, maxima);
    ldma(maxL1, published, uint32_t(cores) * 16, 1, 0, 0, true);
    sload(0, maxL1);
    for (int other = 1; other < cores; ++other) {
      sload(1, maxL1 + uint64_t(other) * 16);
      emit(rform(OP_S_ALU, 0, 0, 1, 0, 4));        // s0 = max(s0, s1)
    }
    sload(1, uint64_t(poolL1 + scalarConst(1.0f / 127.0f)));
    emit(rform(OP_S_ALU, 0, 0, 1, 0, 3));          // s0 = max / 127
    if (c == source) sstore(addr(*S), 0);
    quantApply(xL1, qL1, part);
    constexpr int64_t kFeedBatchesPerCore = 4;
    const int64_t pipelined = std::max<int64_t>(1, (end - begin + kFeedBatchesPerCore - 1) / kFeedBatchesPerCore);
    const int64_t batch = std::min({pipelined, budget / colBytes, (int64_t(ringWindow()) - 32 * pim::kPacketsPerRow) / colBytes});
    if (batch < 1) {
      err = "a quant split by columns needs one column of L1 scratch and L2 staging";
      return failure();
    }
    const uint64_t win = windowBase(c), windowEnd = win + uint64_t(ringWindow());
    uint64_t &stage = stages.try_emplace(c, win).first->second;
    for (int64_t first = begin; first < end; first += batch) {
      const int64_t count = std::min(batch, end - first);
      if (stage + uint64_t(colBytes * count) > windowEnd) stage = win;
      replicateColumns(dup, qL1 + uint64_t((first - begin) * 32), count);
      ldma(dup, stage, uint32_t(colBytes * count), 1, 0, 0, false);
      storeActivationPackets(cast<IntegerAttr>(acts[0]).getInt(), stage, first, count);
      stage += uint64_t(colBytes * count);
    }
    endCore();
  }
  curCore = source;
  w.logicalCore = source;
  w.frontier = last[size_t(source)];
  return success();
}

/// INT8 activation rows (L1) -> every consuming PU's bank.
/// Subchannel and bank group sit below the column in the address, so one column of the
/// 4 bank groups x C subchannels of a bank is colBytes() = 128 C contiguous bytes: each packet is
/// copied 4 C times in L1 scratch and, for every rank that reads the vector, the even-bank PUs
/// (0,2,4,6) and odd-bank PUs (1,3,5,7) take one linear store each.
LogicalResult Codegen::scatterPimAct(const View &s, const View &d, std::string &err) {
  constexpr int64_t kFeedBatches = 4;
  if (s.space != kL1 || s.strides.back() != 1) {
    err = "PIM activation copy needs contiguous L1 rows";
    return failure();
  }
  auto acts = d.lp6.getAs<ArrayAttr>("acts");
  int64_t K = d.shape[1], sRow = s.shape.size() == 2 ? s.strides[0] : K;
  // The copy may cover a row range of the activation buffer (a core's rows after the quant was
  // split by token rows): row r of the source is token row0 + r, one activation vector each.
  const int64_t row0 = d.strides.size() == 2 && d.strides[0] ? d.offset / d.strides[0] : 0;
  const int64_t rows = d.shape.size() == 2 ? d.shape[0] : 1;
  if (d.strides.size() != 2 || d.offset % std::max<int64_t>(1, d.strides[0]) || row0 + rows > int64_t(acts.size())) {
    err = "PIM activation copy must cover whole token rows of the activation buffer";
    return failure();
  }
  uint64_t dup = scratchBase, win = windowBase(curCore), stage = win;
  const int64_t colBytes = in.pim->colBytes();
  // Replication is scratch, not a planned L1 allocation. Bound it by both the
  // L1 reserve and L2 window; a wide FFN vector otherwise overwrites live L1
  // activation/scale buffers before they are stored (e.g. Llama's K=14336).
  const int64_t batchPackets = std::min({K / 32,
      (in.spec->npu.l1ReservedBytes - kPoolMax) / colBytes,
      (int64_t(window()) - 32 * pim::kPacketsPerRow) / colBytes});
  if (batchPackets < 1) {
    err = "PIM activation replication needs one column of L1 scratch and L2 staging";
    return failure();
  }
  for (int64_t m = 0; m < rows; ++m) {
    const pim::Activation &a = in.pim->activations[size_t(cast<IntegerAttr>(acts[size_t(row0 + m)]).getInt())];
    uint64_t row = addr(s, m * sRow);
    // Every rank with a consuming partition takes the whole-column store, also when only some of
    // its subchannels read the vector: an activation's packet rows are reserved in every partition
    // (the same rows, PimLayout::baseRows), so the extra copies land in unused slots. Per-PU stores
    // move 32 B at a time and were 17x slower per byte (Qwen3-4B O and down inputs, 5 of 8
    // subchannels: 389 KB in 71 us, against 164 KB in 1.7 us for a full rank; measured 2026-11-03).
    const int64_t actIndex = cast<IntegerAttr>(acts[size_t(row0 + m)]).getInt();
    if (!a.parts.empty()) {
      // A few batches, in different staging bytes, so the stores of one run while the next is copied to L2.
      const int64_t batch = std::min(batchPackets, std::max<int64_t>(1, (a.packets() + kFeedBatches - 1) / kFeedBatches));
      const uint64_t windowEnd = win + uint64_t(ringWindow());
      for (int64_t begin = 0; begin < a.packets(); begin += batch) {
        const int64_t count = std::min(batch, a.packets() - begin);
        if (stage + uint64_t(colBytes * count) > windowEnd) stage = win;
        replicateColumns(dup, row + uint64_t(begin * 32), count);
        ldma(dup, stage, uint32_t(colBytes * count), 1, 0, 0, false);
        storeActivationPackets(actIndex, stage, begin, count);
        stage += uint64_t(colBytes * count);
      }
    }
  }
  return success();
}

/// --kv8=pim: M input vectors, [M, heads x K] in contiguous L1 rows, where a PU reads the slice of its KV head. One column of a bank
/// holds the packet of one head per bank group, the same bytes over the subchannels: each packet is copied per bank group and doubled to the
/// subchannels, then one store per bank (even PUs, odd PUs).
LogicalResult Codegen::scatterPimKv(const View &s, const View &d, std::string &err) {
  if (s.space != kL1 || s.strides.back() != 1) {
    err = "PIM K/V input copy needs contiguous L1 rows";
    return failure();
  }
  auto acts = d.lp6.getAs<ArrayAttr>("acts");
  const int64_t rows = s.shape[0], sRow = s.strides[0], colBytes = in.pim->colBytes(), bgBytes = in.pim->bgBytes();
  const int channels = in.pim->channels;
  const uint64_t dup = scratchBase, win = windowBase(curCore), windowEnd = win + uint64_t(ringWindow());
  uint64_t stage = win;
  for (int64_t m = 0; m < rows; ++m) {
    const int64_t actIndex = cast<IntegerAttr>(acts[size_t(m)]).getInt();
    const pim::Activation &a = in.pim->activations[size_t(actIndex)];
    const uint64_t row = addr(s, m * sRow);
    const int64_t batch = std::min({std::max<int64_t>(1, (a.packets() + 3) / 4), (in.spec->npu.l1ReservedBytes - kPoolMax) / (2 * colBytes),
                                    (int64_t(ringWindow()) - 32 * pim::kPacketsPerRow) / (2 * colBytes)});
    if (batch < 1) {
      err = "PIM K/V input needs two columns of L1 scratch and L2 staging";
      return failure();
    }
    for (int64_t begin = 0; begin < a.packets(); begin += batch) {
      const int64_t count = std::min(batch, a.packets() - begin);
      if (stage + uint64_t(2 * colBytes * count) > windowEnd) stage = win;
      for (int parity = 0; parity < 2; ++parity)
        for (int group = 0; group < 4; ++group) {
          const uint64_t first = dup + uint64_t(parity * colBytes + group * bgBytes);
          const int head = 2 * group + parity;
          l1Copy(first, row + uint64_t(head * a.k + begin * 32), 32, uint32_t(count), 32, uint32_t(2 * colBytes));
          for (int64_t have = 32; have < bgBytes; have *= 2)
            l1Copy(first + uint64_t(have), first, uint32_t(have), uint32_t(count), uint32_t(2 * colBytes), uint32_t(2 * colBytes));
        }
      ldma(dup, stage, uint32_t(2 * colBytes * count), 1, 0, 0, false);
      std::set<int> ranks;
      for (int part : a.parts) ranks.insert(in.pim->partRank(part));
      for (int rk : ranks)
        for (int parity = 0; parity < 2; ++parity)
          for (int64_t p = begin; p < begin + count;) {
            const int64_t n = std::min(begin + count - p, pim::kPacketsPerRow - (a.packetBase() + p) % pim::kPacketsPerRow);
            gdmaStore(stage + uint64_t((p - begin) * 2 * colBytes + parity * colBytes), in.pim->activationPacketAddress(a, p, parity, rk * channels),
                      uint32_t(colBytes), uint32_t(n), uint32_t(2 * colBytes), uint32_t(colBytes));
            p += n;
          }
      stage += uint64_t(2 * colBytes * count);
    }
  }
  return success();
}

/// --kv8=pim: writing a new K/V row of the step (the copy to its output new_k.i, new_v.i) also writes the row into the stored matrix: one burst per
/// packet that holds a byte of the row. The slot is a compile-time number (timing): the bursts are real, their other lanes are not preserved, so the
/// host writes the matrices again before the next step (research/race_decoding/kv8/run_kv8.py) and no later read of this step sees them.
void Codegen::storeKvRow(const View &s, const View &d) {
  if (!in.pim || d.space != kLP6 || !d.lp6 || s.space != kL1) return;
  auto name = d.lp6.getAs<StringAttr>("name");
  if (!name) return;
  const bool isK = name.getValue().starts_with("new_k."), isV = name.getValue().starts_with("new_v.");
  if (!isK && !isV) return;
  const std::string key = std::string(isK ? "kv8.k." : "kv8.v.") + name.getValue().drop_front(6).str();
  const pim::Projection *p = nullptr;
  for (const pim::Projection &x : in.pim->projections)
    if (x.name == key) p = &x;
  if (!p) return;
  const int64_t slot = in.kvSlot, heads = pim::kPUs;
  std::map<int64_t, std::vector<uint64_t>> byHead;
  in.pim->forEachWeightPacketLocation(*p, [&](const pim::WeightPacketLocation &l) {
    if (isK) {
      // K: rows n = head x capacity + position, inputs = head_dim. All packets of the row, one lane each.
      const int64_t perHead = p->n / heads;
      for (int64_t ch : l.outputChannels)
        if (ch < p->n && ch % perHead == slot) byHead[ch / perHead].push_back(l.address);
    } else if (slot >= l.inputBegin && slot < l.inputBegin + pim::kGroup) {
      // V: rows n = head x head_dim + column, inputs = position. The packet that holds this position, for each column.
      for (int64_t ch : l.outputChannels)
        if (ch < p->n) { byHead[ch / (p->n / heads)].push_back(l.address); break; }
    }
  });
  for (auto &[head, list] : byHead) {
    std::sort(list.begin(), list.end());
    list.erase(std::unique(list.begin(), list.end()), list.end());
    for (uint64_t address : list) copyL1ToLP6(addr(s, 0) + uint64_t(head) * 128, address, 32, 1, 32, 32);
  }
}

} // namespace mlir::hybrid::codegen
