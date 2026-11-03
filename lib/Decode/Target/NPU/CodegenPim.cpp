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
  uint64_t stage = win;
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

/// Admit only an isolated quant/feed phase. Other cores' L1 and the staging
/// windows are phase-owned here; a generic mixed phase may have live values
/// there and must not silently borrow those resources.
LogicalResult Codegen::planParallelActivationFeed(func::FuncOp fn, std::string &err) {
  if (in.pimActivationCores == 1) return success();
  auto fail = [&](StringRef reason) -> LogicalResult {
    err = "4-core PIM activation feed: " + reason.str();
    return failure();
  };
  if (in.pimActivationCores != 4) return fail("only 1 or 4 cores are supported");
  bool hasActivation = llvm::any_of(in.argLP6, [](DictionaryAttr a) {
    auto layout = a.getAs<StringAttr>("layout");
    return layout && layout.getValue() == "pim_act";
  });
  if (!hasActivation) return success(); // unrelated dequant/attention phases keep their lowering
  if (!in.pim || in.spec->npu.cores != 4 || in.pim->channels != 8)
    return fail("requires a 4-core target and all 8 subchannels");
  if (fn.getBody().getBlocks().size() != 1) return fail("requires one isolated quant/feed block");
  if (auto peak = fn->getAttrOfType<IntegerAttr>("hyb.l2_peak"); peak && peak.getInt() != 0)
    return fail("isolated quant/feed phase cannot have live L2 allocations");

  hyb::QuantOp quant;
  SmallVector<memref::CopyOp> copies;
  SmallVector<memref::AllocOp> allocs;
  for (Operation &op : fn.getBody().front()) {
    if (auto q = dyn_cast<hyb::QuantOp>(op)) {
      if (quant) return fail("isolated quant/feed phase requires exactly one quant");
      quant = q;
    } else if (auto copy = dyn_cast<memref::CopyOp>(op)) copies.push_back(copy);
    else if (auto alloc = dyn_cast<memref::AllocOp>(op)) allocs.push_back(alloc);
    else if (!isa<arith::ConstantOp, memref::CastOp, memref::DeallocOp, func::ReturnOp>(op))
      return fail("isolated quant/feed phase does not support subviews, loops or other compute");
  }
  if (!quant || copies.size() != 3 || allocs.size() != 3)
    return fail("isolated quant/feed phase needs one quant and its three allocation/copy pairs");
  memref::CopyOp input, output, scale;
  for (auto copy : copies) {
    if (copy.getTarget() == quant.getInput()) input = copy;
    else if (copy.getSource() == quant.getQInit()) output = copy;
    else if (copy.getSource() == quant.getScaleInit()) scale = copy;
    else return fail("copy is not a direct quant input, full Q output or row-scale output");
  }
  if (!input || !output || !scale || !input->isBeforeInBlock(quant) ||
      !quant->isBeforeInBlock(output) || !quant->isBeforeInBlock(scale))
    return fail("quant input/output copy order is not supported");
  auto directArgument = [&](Value v) {
    auto a = dyn_cast<BlockArgument>(v);
    return a && a.getOwner() == &fn.getBody().front();
  };
  if (!directArgument(input.getSource()) || !directArgument(output.getTarget()) ||
      !directArgument(scale.getTarget()))
    return fail("requires full phase arguments, without destination or input subviews");
  for (auto alloc : allocs)
    if (!alloc.getType().hasStaticShape() || spaceOf(alloc.getType()) != kL1 ||
        (alloc.getResult() != quant.getInput() && alloc.getResult() != quant.getQInit() &&
         alloc.getResult() != quant.getScaleInit()) || alloc->hasAttr("hyb.buffers"))
      return fail("requires exactly the quant's three static private L1 allocations");

  auto x = viewOf(quant.getInput(), err), q = viewOf(quant.getQInit(), err);
  auto s = viewOf(quant.getScaleInit(), err), dst = viewOf(output.getTarget(), err);
  auto src = viewOf(input.getSource(), err), scaleDst = viewOf(scale.getTarget(), err);
  if (failed(x) || failed(q) || failed(s) || failed(dst) || failed(src) || failed(scaleDst)) return failure();
  if (q->shape.size() != 2 || q->shape[0] < 1 || q->shape[1] < 1 || q->shape[1] % 256 ||
      !q->et.isInteger(8) || !x->et.isF16() || !s->et.isF32() ||
      x->shape != q->shape || dst->shape != q->shape || src->shape != q->shape ||
      s->shape != SmallVector<int64_t>({q->shape[0], 1}) || scaleDst->shape != s->shape ||
      src->layout != "dense" || scaleDst->layout != "dense" || dst->layout != "pim_act")
    return fail("requires full FP16 X[M,K], INT8 Q[M,K], FP32 scale[M,1], and K % 256 == 0");
  for (const View *v : {&*x, &*q, &*s, &*dst, &*src, &*scaleDst})
    if (v->offset != 0 || v->strides.size() != 2 || v->strides[1] != 1 || v->strides[0] != v->shape[1])
      return fail("all quant/feed views must be full contiguous rows");

  ParallelActivationFeed plan;
  plan.copy = output;
  plan.m = q->shape[0]; plan.k = q->shape[1];
  plan.qBytes = uint64_t(plan.m) * uint64_t(plan.k);
  plan.qL1 = addr(*q); plan.scaleL1 = addr(*s);
  auto ids = dst->lp6.getAs<ArrayAttr>("acts");
  if (!ids || int64_t(ids.size()) != plan.m) return fail("activation IDs must match every full Q row");
  std::set<int64_t> uniqueIds;
  std::vector<std::pair<int64_t, int64_t>> packetRanges;
  for (auto id : ids) {
    auto integer = dyn_cast<IntegerAttr>(id);
    if (!integer || integer.getInt() < 0 || size_t(integer.getInt()) >= in.pim->activations.size() ||
        !uniqueIds.insert(integer.getInt()).second)
      return fail("activation IDs must be distinct valid rows");
    const auto &a = in.pim->activations[size_t(integer.getInt())];
    if (a.k != plan.k || a.base < 0 || a.base % pim::kSlices ||
        (a.parts.size() != 8 && a.parts.size() != 16))
      return fail("each activation must contain a complete row consumed by one or two full ranks");
    std::set<int> parts(a.parts.begin(), a.parts.end());
    if (parts.size() != a.parts.size()) return fail("activation partitions must be distinct");
    std::set<int> ranks;
    for (int part : parts) {
      if (part < 0 || part >= in.pim->partitions || in.pim->partRank(part) >= pim::kRanks)
        return fail("activation partition is outside the configured ranks");
      ranks.insert(in.pim->partRank(part));
    }
    for (int rank : ranks)
      for (int ch = 0; ch < 8; ++ch)
        if (!parts.count(rank * 8 + ch)) return fail("activation partitions must include all 8 subchannels of every used rank");
    std::vector<int> rowRanks(ranks.begin(), ranks.end());
    if (!plan.activationIds.empty() && plan.ranks != rowRanks)
      return fail("activation rows must use the same rank set");
    plan.ranks = std::move(rowRanks);
    plan.activationIds.push_back(integer.getInt());
    packetRanges.push_back({a.packetBase(), a.packetBase() + a.packets()});
  }
  std::sort(packetRanges.begin(), packetRanges.end());
  for (size_t i = 1; i < packetRanges.size(); ++i)
    if (packetRanges[i].first < packetRanges[i - 1].second) return fail("activation rows overlap");

  const int64_t reserved = in.spec->npu.l1ReservedBytes - kPoolMax;
  if (reserved <= 0 || 2 * plan.k + 4 > reserved)
    return fail("quant and feed scratch exceed the L1 reserve");
  const uint64_t l1ReservedEnd = uint64_t(kL1Base + in.spec->npu.l1ReservedBytes);
  std::vector<std::pair<uint64_t, uint64_t>> live;
  for (const View *v : {&*x, &*q, &*s}) {
    uint64_t begin = addr(*v), end = begin + uint64_t(v->elements() * v->eb);
    if (begin < l1ReservedEnd || end > uint64_t(in.spec->npu.l1Bytes))
      return fail("quant's live L1 buffers overlap scratch or exceed capacity");
    live.push_back({begin, end});
    plan.l1Peak = std::max(plan.l1Peak, end);
  }
  std::sort(live.begin(), live.end());
  for (size_t i = 1; i < live.size(); ++i)
    if (live[i].first < live[i - 1].second) return fail("quant's live L1 buffers alias");
  if (window() <= 0 || window() % 64 || uint64_t(window()) * 4 > uint64_t(in.spec->npu.l2Bytes))
    return fail("requires four bounded aligned L2 staging windows");
  uint64_t alignedQ = (plan.qBytes + 63) / 64 * 64;
  if (alignedQ >= uint64_t(window())) return fail("L2 window cannot hold shared Q and feed scratch");
  plan.sharedQ = uint64_t(window()) - alignedQ;
  if (uint64_t(plan.m) * 4 > plan.sharedQ)
    return fail("remaining core0 scale store would overlap shared Q");
  const int64_t available = int64_t(plan.sharedQ) - 32 * pim::kPacketsPerRow;
  plan.batchPackets = std::min({plan.k / 32, reserved / in.pim->colBytes(), available / in.pim->colBytes()});
  if (available < in.pim->colBytes() || plan.batchPackets < 1)
    return fail("L2 window cannot hold shared Q and one replicated column");
  plan.dupBytes = uint64_t(plan.batchPackets * in.pim->colBytes());
  parallelActivationFeed = std::move(plan);
  return success();
}

LogicalResult Codegen::scatterPimActParallel(const View &s, const View &, std::string &err) {
  auto &p = *parallelActivationFeed;
  if (curCore != 0 || cores != 4) {
    err = "4-core PIM activation feed must follow the isolated core0 quant";
    return failure();
  }
  // Publishing Q is paid on chip. This range is outside every feed window's
  // live duplicate region and remains immutable until all worker loads end.
  constexpr uint64_t chunk = 128 << 10;
  for (uint64_t offset = 0; offset < p.qBytes; offset += chunk)
    ldma(addr(s) + offset, p.sharedQ + offset, uint32_t(std::min(chunk, p.qBytes - offset)), 1, 0, 0, false);
  p.publishEvent = flush();
  if (p.publishEvent < 0) { err = "PIM activation Q publication did not emit a core event"; return failure(); }
  for (int core = 0; core < 4; ++core) {
    if (core >= p.m) continue;
    startCore(core, {p.publishEvent});
    for (int64_t m = core; m < p.m; m += 4) {
      const auto &a = in.pim->activations[size_t(p.activationIds[size_t(m)])];
      const uint64_t row = addr(s, m * s.strides[0]);
      if (core != 0) ldma(row, p.sharedQ + uint64_t(m * p.k), uint32_t(p.k), 1, 0, 0, true);
      for (int64_t begin = 0; begin < a.packets(); begin += p.batchPackets) {
        int64_t count = std::min(p.batchPackets, a.packets() - begin);
        replicateColumns(scratchBase, row + uint64_t(begin * 32), count);
        const uint64_t win = windowBase(core);
        // Existing L2 hazard tracking waits for the preceding GDMA readers
        // before this core overwrites its window for another batch/token.
        ldma(scratchBase, win, uint32_t(in.pim->colBytes() * count), 1, 0, 0, false);
        for (int64_t first = begin; first < begin + count;) {
          int64_t pk = a.packetBase() + first;
          int64_t n = std::min(begin + count - first, pim::kPacketsPerRow - pk % pim::kPacketsPerRow);
          // The same L2 replica feeds every rank. Each store registers an
          // independent GR hazard, so its completion is required before
          // any following batch/token overwrites this window.
          for (int rank : p.ranks)
            for (int parity = 0; parity < 2; ++parity)
              p.storeEvents.push_back(gdmaStore(win + uint64_t((first - begin) * in.pim->colBytes()),
                  in.pim->activationPacketAddress(a, first, parity, rank * 8),
                  uint32_t(in.pim->colBytes() * n), 1, uint32_t(in.pim->colBytes() * n),
                  uint32_t(in.pim->colBytes() * n)));
          first += n;
        }
      }
    }
    endCore();
  }
  // Only the original scale copy can remain in this admitted phase. It uses
  // core0's preserved L1 scale and disjoint window prefix; normal hazards
  // protect its reuse. endPhase joins ALL worker/store sinks before PIM.
  curCore = 0; w.logicalCore = 0; w.frontier = last[0];
  return success();
}

std::string Codegen::parallelActivationPlanJSON() const {
  const auto &p = *parallelActivationFeed;
  namespace json = llvm::json;
  json::Array windows, rows, stores;
  for (int core = 0; core < 4; ++core) {
    windows.push_back(json::Object{{"core", core}, {"offset", int64_t(windowBase(core))},
                                   {"bytes", int64_t(p.dupBytes)}});
    json::Array tokens;
    for (int64_t m = core; m < p.m; m += 4) tokens.push_back(m);
    rows.push_back(json::Object{{"core", core}, {"tokens", std::move(tokens)}});
  }
  for (int event : p.storeEvents) stores.push_back(event);
  json::Object object{{"kernel", "pim_activation_scatter_4core_v1"}, {"cores", 4}, {"quant_cores", 1},
    {"isolated_quant_feed_phase_checked", true}, {"M", p.m}, {"K", p.k}, {"rank", p.ranks.front()}, {"subchannels", 8},
    {"q_source_l1", int64_t(p.qL1)}, {"scale_source_l1", int64_t(p.scaleL1)},
    {"q_shared_l2", int64_t(p.sharedQ)}, {"q_bytes", int64_t(p.qBytes)},
    {"q_publish_bytes", int64_t(p.qBytes)}, {"q_worker_load_bytes", (p.m - (p.m + 3) / 4) * p.k},
    {"core0_reuses_original_Q", true}, {"batch_packets", p.batchPackets}, {"duplication_bytes", int64_t(p.dupBytes)},
    {"replica_l1_to_l2_bytes", int64_t(p.qBytes * 32)}, {"lp6_activation_write_bytes", int64_t(p.qBytes * 64 * p.ranks.size())},
    {"l1_scratch_offset", int64_t(scratchBase)}, {"l1_scratch_bytes", int64_t(p.dupBytes)},
    {"l1_peak_bytes", int64_t(p.l1Peak)}, {"l2_window_bytes_per_core", window()},
    {"l2_capacity_bytes", in.spec->npu.l2Bytes}, {"replica_l2_ranges", std::move(windows)},
    {"token_assignment", std::move(rows)}, {"q_ready_event", p.publishEvent},
    {"activation_store_events", std::move(stores)}, {"phase_exit_joins_all_sinks", true},
    {"weight_activation_scale_addresses_unchanged", true}};
  // Preserve the existing one-rank report as well as its emitted program.
  if (p.ranks.size() > 1) {
    object.erase("rank");
    json::Array ranks;
    for (int rank : p.ranks) ranks.push_back(rank);
    object["ranks"] = std::move(ranks);
  }
  return llvm::formatv("{0}", json::Value(std::move(object))).str();
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

/// INT8 activation rows (L1) -> every consuming PU's bank.
/// Subchannel and bank group sit below the column in the address, so one column of the
/// 4 bank groups x C subchannels of a bank is colBytes() = 128 C contiguous bytes: each packet is
/// copied 4 C times in L1 scratch and, for every rank that reads the vector, the even-bank PUs
/// (0,2,4,6) and odd-bank PUs (1,3,5,7) take one linear store each.
LogicalResult Codegen::scatterPimAct(const View &s, const View &d, std::string &err) {
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
  uint64_t dup = scratchBase, win = windowBase(curCore);
  const int64_t colBytes = in.pim->colBytes();
  const int channels = in.pim->channels;
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
    std::set<int> pairRanks;
    for (int p : a.parts) pairRanks.insert(in.pim->partRank(p));
    if (!pairRanks.empty()) {
      for (int64_t begin = 0; begin < a.packets(); begin += batchPackets) {
        const int64_t count = std::min(batchPackets, a.packets() - begin);
        replicateColumns(dup, row + uint64_t(begin * 32), count);
        ldma(dup, win, uint32_t(colBytes * count), 1, 0, 0, false);
        for (int rk : pairRanks) {
          int part0 = rk * channels;
          for (int64_t first = begin; first < begin + count;) {
            int64_t pk = a.packetBase() + first;
            int64_t n = std::min(begin + count - first, pim::kPacketsPerRow - pk % pim::kPacketsPerRow);
            for (int parity = 0; parity < 2; ++parity)   // bank 0 (PUs 0,2,4,6) then bank 2 (PUs 1,3,5,7)
              gdmaStore(win + uint64_t((first - begin) * colBytes), in.pim->activationPacketAddress(a, first, parity, part0),
                        uint32_t(colBytes * n), 1, uint32_t(colBytes * n), uint32_t(colBytes * n));
            first += n;
          }
        }
      }
    }
  }
  return success();
}

} // namespace mlir::hybrid::codegen
