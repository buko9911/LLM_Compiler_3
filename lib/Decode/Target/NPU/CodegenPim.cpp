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

} // namespace mlir::hybrid::codegen
