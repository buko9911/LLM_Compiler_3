//===- CodegenEmit.cpp - NPU codegen: command streams and DMA ----------------===//
//
// Codegen members that build the per-core command streams: blocks and their completion-event
// dependencies, address-range hazards of the GDMA and core accesses to L2 and LP6, and the
// DMA primitives (LDMA, L1 copy, 2-D GDMA, LP6 <-> L1 through the core's staging window).
//===----------------------------------------------------------------------===//

#include "CodegenImpl.h"
#include <algorithm>

namespace mlir::hybrid::codegen {

// Late, address-aware local DMA scheduling. Event 1 belongs to weight
// prefetch, event 2 to the output store, and event 3 to the common input load.
// Independent vector work may cross either transfer; ready matrix operands
// may cross an input load. Unknown memory instructions and block boundaries
// conservatively retire transfers. No event escapes its producer block.
void Codegen::emit(uint32_t word) {
  switch (word & 63) {
  case OP_V_BIN: case OP_V_UN: case OP_V_SCALAR: case OP_V_RED:
  case OP_S_ALU: case OP_V_CONV:
    noteOutputStoreVectorWork();
    break;
  case OP_LUI: case OP_ADDI: case OP_C_SET: case OP_WAIT_EVENT:
    break;
  default:
    // Vector memory uses the checked helpers below; raw/unknown memory
    // instructions cannot bypass the source-lifetime proof.
    waitOutputStore();
    waitInputLoad();
    break;
  }
  w.append(word);
}

void Codegen::rec(const std::array<uint32_t, 4> &record) {
  uint32_t opcode = record[0] & 63;
  if (in.spec->npu.memoryDependencySchedule &&
      (opcode == OP_M_LOAD || opcode == OP_M_WRITEOUT || opcode == OP_M_LOAD_ACC)) {
    SmallVector<L2Range> ranges;
    bool write = false;
    if (matrixMemoryAccess(record, ranges, write)) guardLocalMemory(ranges, write);
    else { waitInputLoad(); waitOutputStore(); }
    w.appendRecord(record);
    return;
  }
  if ((record[0] & 63) == OP_M_LOAD || (record[0] & 63) == OP_M_WRITEOUT) {
    if (matrixTransferPending(record)) waitInputLoad();
  } else if ((record[0] & 63) != OP_V_TYPED) waitInputLoad();
  if ((record[0] & 63) == OP_V_TYPED) noteOutputStoreVectorWork();
  else waitOutputStore();
  w.appendRecord(record);
}

bool Codegen::inputLoadOverlaps(uint64_t address, uint64_t bytes) const {
  return pendingInputLoad && llvm::any_of(pendingInputLoad->destinationRanges,
      [&](auto r) { return overlaps(address,address+bytes,r.first,r.second); });
}
void Codegen::guardInputLoad(uint64_t address, uint64_t bytes) {
  if (inputLoadOverlaps(address,bytes)) waitInputLoad();
}
void Codegen::waitInputLoad() {
  if (!pendingInputLoad) return;
  int source = w.currentSource;
  w.currentSource = pendingInputLoad->source;
  pendingInputLoad.reset();
  load(10,3);
  emit(rform(OP_WAIT_EVENT,10));
  w.currentSource = source;
}
bool Codegen::matrixMemoryAccess(const std::array<uint32_t,4> &r,
                                SmallVector<L2Range> &ranges, bool &write) const {
  uint32_t opcode = r[0] & 63;
  uint32_t funct = (r[0] >> 22) & 15;
  write = opcode == OP_M_WRITEOUT;
  uint64_t bytes;
  if (opcode == OP_M_LOAD_ACC) {
    if (funct != 1 && funct != 2) return false;
    bytes = 4;
  } else if (write) {
    if (funct != 1 && funct != 2 && funct != 4) return false;
    bytes = funct == 1 ? 2 : 4;
  } else {
    if (opcode != OP_M_LOAD || (funct != 1 && funct != 3 && funct != 5 && funct != 7)) return false;
    bytes = (funct == 1 || funct == 5) ? 1 : 2;
  }
  uint64_t address = w.gpValue((r[0] >> (write ? 6 : 10)) & 15);
  ranges = rowRanges(address,uint64_t(r[2])*bytes,r[1],r[3]);
  return true;
}

void Codegen::guardLocalMemory(ArrayRef<L2Range> ranges, bool write) {
  for (auto [s,e] : ranges) {
    // Outstanding input DMA writes L1: both reads and writes must wait.
    guardInputLoad(s, e-s);
    // Outstanding output DMA reads L1: independent reads may proceed, but
    // overwriting its source must wait. Row ranges retain strided holes.
    if (write && pendingOutputStore &&
        llvm::any_of(pendingOutputStore->sourceRanges,
          [&](auto r) { return overlaps(s,e,r.first,r.second); }))
      waitOutputStore();
  }
}

bool Codegen::matrixTransferPending(const std::array<uint32_t,4> &r) const {
  if (!pendingInputLoad) return false;
  SmallVector<L2Range> ranges;
  bool write = false;
  if (!matrixMemoryAccess(r, ranges, write)) return true;
  for (auto [s,e] : ranges)
    if (inputLoadOverlaps(s,e-s)) return true;
  return false;
}
void Codegen::matrixInputs(const std::array<uint32_t,4> &a, const std::array<uint32_t,4> &b) {
  // Distinct weight/activation operand buffers. Loading the ready operand
  // first hides its SRAM transfer under an unrelated pending input DMA.
  if (pendingInputLoad && matrixTransferPending(a) && !matrixTransferPending(b)) {
    rec(b); rec(a);
  } else { rec(a); rec(b); }
}

void Codegen::waitOutputStore() {
  if (!pendingOutputStore) return;
  if (pendingOutputStore->vectorWork) ++outputStoresWithVectorWork;
  int source = w.currentSource;
  w.currentSource = pendingOutputStore->source;
  pendingOutputStore.reset();
  load(10, 2);
  emit(rform(OP_WAIT_EVENT, 10));
  w.currentSource = source;
}

void Codegen::guardOutputStoreVectorWrite(uint64_t address, uint32_t funct) {
  if (!pendingOutputStore) return;
  // A vector store is at most one 512-bit register. Use its full footprint
  // conservatively: typed-vector records can change VL independently of cfg.
  if (funct != 2 && funct != 4 && funct != 6 && funct != 8 && funct != 10) {
    waitOutputStore();
    return;
  }
  uint64_t bytes = funct == 10 ? 4 : 64;
  for (auto [begin, end] : pendingOutputStore->sourceRanges)
    if (overlaps(address, address + bytes, begin, end)) {
      waitOutputStore();
      return;
    }
}

// ---- command stream per core ------------------------------------------------------------
/// Closes the open block of the current core; later instructions start a new block after it.
int Codegen::flush() {
  waitInputLoad();
  waitOutputStore();
  int id = w.flushBlock(false);
  if (id >= 0) {
    for (Acc &a : l2acc)
      if (a.id < 0 && a.core == curCore) a.id = id;
    last[size_t(curCore)] = {id};
  }
  w.frontier = last[size_t(curCore)];
  return id;
}

/// Starts a new block on the current core that also waits for `deps`.
void Codegen::blockAfter(const std::vector<int> &deps) {
  flush();
  std::vector<int> f = last[size_t(curCore)];
  for (int d : deps)
    if (std::find(f.begin(), f.end(), d) == f.end()) f.push_back(d);
  w.frontier = f;
  for (int d : deps) satisfied[size_t(curCore)].insert(d);
}

void Codegen::startCore(int c, const std::vector<int> &deps) {
  flush();
  curCore = c;
  w.logicalCore = c;
  std::vector<int> f = last[size_t(c)];
  for (int d : deps)
    if (std::find(f.begin(), f.end(), d) == f.end()) f.push_back(d);
  w.frontier = f;
  if (!poolLoaded[size_t(c)] && poolBytes) {
    poolLoaded[size_t(c)] = true;
    copyLP6ToL1(poolLP6, uint64_t(poolL1), uint32_t(poolBytes), 1, uint32_t(poolBytes), uint32_t(poolBytes));
  }
}

void Codegen::endCore() {
  waitInputLoad();
  waitOutputStore();
  int id = w.flushBlock(false);
  if (id >= 0) {
    for (Acc &a : l2acc)
      if (a.id < 0 && a.core == curCore) a.id = id;
    last[size_t(curCore)] = {id};
  } else {
    last[size_t(curCore)] = w.frontier;
  }
}

SmallVector<Codegen::L2Range> Codegen::rowRanges(uint64_t base, uint64_t rowBytes, uint64_t rows, uint64_t stride) {
  if (!rows || !rowBytes) return {};
  if (rows == 1 || stride <= rowBytes || rows > kMaxL2AccessRanges)
    return {{base, base + (rows - 1) * stride + rowBytes}};
  SmallVector<L2Range> ranges;
  ranges.reserve(rows);
  for (uint64_t r = 0; r < rows; ++r)
    ranges.push_back({base + r * stride, base + r * stride + rowBytes});
  return ranges;
}

/// LP6 -> L2 (2-D). Waits for earlier stores to the LP6 range and every earlier access to
/// the L2 range.
int Codegen::gdmaLoad(uint64_t lp6, uint64_t l2, uint32_t rowBytes, uint32_t rows, uint32_t lp6Stride,
                      uint32_t l2Stride) {
  uint64_t le = lp6 + uint64_t(rows - 1) * lp6Stride + rowBytes, l2e = l2 + uint64_t(rows - 1) * l2Stride + rowBytes;
  auto l2Ranges = in.spec->npu.commandSchedule ? rowRanges(l2,rowBytes,rows,l2Stride) : SmallVector<L2Range>{{l2,l2e}};
  auto lp6Ranges = in.spec->npu.commandSchedule ? rowRanges(lp6,rowBytes,rows,lp6Stride) : SmallVector<L2Range>{{lp6,le}};
  auto touches = [&](uint64_t s, uint64_t e, ArrayRef<L2Range> ranges) {
    return llvm::any_of(ranges,[&](auto r) { return overlaps(s,e,r.first,r.second); });
  };
  for (const Acc &a : l2acc)
    if (a.id < 0 && a.core == curCore && touches(a.s,a.e,l2Ranges)) {
      flush();
      break;
    }
  std::vector<int> deps;
  for (const Acc &a : l2acc)
    if (touches(a.s,a.e,l2Ranges) && a.id >= 0) deps.push_back(a.id);
  for (const LAcc &a : lp6acc)
    if (a.store && touches(a.s,a.e,lp6Ranges)) deps.push_back(a.id);
  std::sort(deps.begin(), deps.end());
  deps.erase(std::unique(deps.begin(), deps.end()), deps.end());
  int id = w.appendGdma(ProgramBuilder::Kind::GdmaLoad, deps, lp6, l2, rowBytes, rows, lp6Stride, l2Stride);
  llvm::erase_if(l2acc, [&](const Acc &a) {
    return llvm::any_of(l2Ranges,[&](auto r) { return a.s >= r.first && a.e <= r.second; });
  });
  for (auto [s,e] : l2Ranges) l2acc.push_back({s,e,id,curCore,GW});
  for (auto [s,e] : lp6Ranges) lp6acc.push_back({s,e,id,false});
  return id;
}

/// L2 -> LP6 (2-D). Waits for the writers of the L2 range and earlier accesses to the LP6 range.
int Codegen::gdmaStore(uint64_t l2, uint64_t lp6, uint32_t rowBytes, uint32_t rows, uint32_t l2Stride,
                       uint32_t lp6Stride) {
  uint64_t le = lp6 + uint64_t(rows - 1) * lp6Stride + rowBytes, l2e = l2 + uint64_t(rows - 1) * l2Stride + rowBytes;
  auto l2Ranges = in.spec->npu.commandSchedule ? rowRanges(l2,rowBytes,rows,l2Stride) : SmallVector<L2Range>{{l2,l2e}};
  auto lp6Ranges = in.spec->npu.commandSchedule ? rowRanges(lp6,rowBytes,rows,lp6Stride) : SmallVector<L2Range>{{lp6,le}};
  auto touches = [&](uint64_t s, uint64_t e, ArrayRef<L2Range> ranges) {
    return llvm::any_of(ranges,[&](auto r) { return overlaps(s,e,r.first,r.second); });
  };
  for (const Acc &a : l2acc)
    if (a.id < 0 && a.core == curCore && touches(a.s,a.e,l2Ranges) &&
        (!in.spec->npu.commandSchedule || a.k == CW)) {
      flush();
      break;
    }
  std::vector<int> deps;
  for (const Acc &a : l2acc)
    if (touches(a.s,a.e,l2Ranges) && a.id >= 0 && (a.k == GW || a.k == CW)) deps.push_back(a.id);
  for (const LAcc &a : lp6acc)
    if (touches(a.s,a.e,lp6Ranges)) deps.push_back(a.id);
  std::sort(deps.begin(), deps.end());
  deps.erase(std::unique(deps.begin(), deps.end()), deps.end());
  int id = w.appendGdma(ProgramBuilder::Kind::GdmaStore, deps, lp6, l2, rowBytes, rows, lp6Stride, l2Stride);
  for (auto [s,e] : l2Ranges) l2acc.push_back({s,e,id,curCore,GR});
  llvm::erase_if(lp6acc, [&](const LAcc &a) {
    return llvm::any_of(lp6Ranges,[&](auto r) { return a.s >= r.first && a.e <= r.second; });
  });
  for (auto [s,e] : lp6Ranges) lp6acc.push_back({s,e,id,true});
  return id;
}

/// A core instruction reads or writes L2 [s, e): the block must follow the GDMAs (and other
/// cores' blocks) that touch the range.
void Codegen::coreDependencies(uint64_t s, uint64_t e, bool write, std::vector<int> &need) {
  for (const Acc &a : l2acc) {
    if (!overlaps(a.s, a.e, s, e) || a.id < 0) continue;
    bool other = a.core != curCore && (a.k == CW || a.k == CR);
    if (a.k == GW || (write && a.k == GR) || (other && (write || a.k == CW)))
      if (!satisfied[size_t(curCore)].count(a.id)) need.push_back(a.id);
  }
}

void Codegen::coreAccess(ArrayRef<L2Range> ranges, bool write) {
  std::vector<int> need;
  // A strided DMA is one instruction. Resolve all its hazards before registering
  // any row, otherwise a later row's blockAfter could commit earlier rows to the
  // block preceding the instruction that actually accesses them.
  for (auto [s, e] : ranges) {
    w.reserveL2(s, e - s);
    coreDependencies(s, e, write, need);
  }
  if (!need.empty()) blockAfter(need);
  for (auto [s, e] : ranges) {
    if (write) llvm::erase_if(l2acc, [&](const Acc &a) { return a.s >= s && a.e <= e && a.id >= 0; });
    l2acc.push_back({s, e, -1, curCore, write ? CW : CR});
  }
}

// ---- data movement primitives -----------------------------------------------------------
void Codegen::ldma(uint64_t l1, uint64_t l2, uint32_t rowBytes, uint32_t rows, uint32_t srcStride, uint32_t dstStride,
                   bool toL1) {
  waitInputLoad();
  waitOutputStore(); // single output slot; also protects same-core L2 readers
  coreAccess(rowRanges(l2, rowBytes, rows, toL1 ? srcStride : dstStride), !toL1);
  bool async = !toL1 && in.spec->npu.asyncOutputStores;
  bool inputAsync = toL1 && in.spec->npu.asyncInputLoads;
  if (rows == 1 || (srcStride == rowBytes && dstStride == rowBytes)) {
    setConfig(7, rowBytes * rows);
    load(1, uint32_t(l1));
    load(2, uint32_t(l2));
    if (async || inputAsync) load(10, inputAsync ? 3 : 2);
    emit(rform(OP_LDMA, 1, 2, (async || inputAsync) ? 10 : 0, 0, inputAsync ? 10 : toL1 ? 0 : async ? 11 : 1));
  } else {
    setConfig(7, rowBytes);
    setConfig(0, rows);
    setConfig(1, srcStride);
    setConfig(2, dstStride);
    load(1, uint32_t(l1));
    load(2, uint32_t(l2));
    if (async || inputAsync) load(10, inputAsync ? 3 : 2);
    emit(rform(OP_LDMA, 1, 2, (async || inputAsync) ? 10 : 0, 0, inputAsync ? 12 : toL1 ? 4 : async ? 13 : 5));
  }
  if (inputAsync) {
    pendingInputLoad = PendingInputLoad{rowRanges(l1,rowBytes,rows,dstStride),w.currentSource};
    if (!in.spec->npu.asyncInputLoadOverlap) waitInputLoad();
  }
  if (async) {
    pendingOutputStore = PendingOutputStore{rowRanges(l1, rowBytes, rows, srcStride), false, w.currentSource};
    ++asyncOutputStores;
    if (!in.spec->npu.asyncOutputStoreVectorOverlap) waitOutputStore();
  }
}

/// Contiguous L2 -> L1 weight load. At most one is outstanding; event 1 is reused after
/// its consumer waits. Output stores optionally use the separate event 2.
void Codegen::prefetchWeight(uint64_t l1, uint64_t l2, uint32_t bytes) {
  waitInputLoad();
  waitOutputStore(); // retire before loading GP10 with the weight event
  coreAccess(l2, l2 + bytes, false);
  setConfig(7, bytes);
  load(1, uint32_t(l1));
  load(2, uint32_t(l2));
  load(10, 1);
  emit(rform(OP_LDMA, 1, 2, 10, 0, 10));
}

void Codegen::l1Copy(uint64_t dst, uint64_t src, uint32_t rowBytes, uint32_t rows, uint32_t srcStride,
                     uint32_t dstStride) {
  setConfig(7, rowBytes);
  setConfig(0, rows);
  setConfig(1, srcStride);
  setConfig(2, dstStride);
  load(3, uint32_t(dst));
  load(4, uint32_t(src));
  emit(rform(OP_LDMA, 3, 4, 0, 0, 9));
}

/// LP6 rows -> L1 rows through the core's staging window (chunked to the window).
void Codegen::copyLP6ToL1(uint64_t lp6, uint64_t l1, uint32_t rowBytes, uint32_t rows, uint32_t lp6Stride,
                          uint32_t l1Stride) {
  uint64_t win = windowBase(curCore);
  if (rowBytes > uint64_t(window())) {
    for (uint32_t r = 0; r < rows; ++r)
      for (uint64_t o = 0; o < rowBytes; o += uint64_t(window())) {
        uint32_t n = uint32_t(std::min<uint64_t>(uint64_t(window()), rowBytes - o));
        gdmaLoad(lp6 + uint64_t(r) * lp6Stride + o, win, n, 1, n, n);
        ldma(l1 + uint64_t(r) * l1Stride + o, win, n, 1, n, n, true);
      }
    return;
  }
  uint32_t per = uint32_t(std::max<int64_t>(1, window() / rowBytes));
  for (uint32_t r0 = 0; r0 < rows; r0 += per) {
    uint32_t n = std::min(per, rows - r0);
    gdmaLoad(lp6 + uint64_t(r0) * lp6Stride, win, rowBytes, n, lp6Stride, rowBytes);
    ldma(l1 + uint64_t(r0) * l1Stride, win, rowBytes, n, rowBytes, l1Stride, true);
  }
}

void Codegen::copyL1ToLP6(uint64_t l1, uint64_t lp6, uint32_t rowBytes, uint32_t rows, uint32_t l1Stride,
                          uint32_t lp6Stride) {
  uint64_t win = windowBase(curCore);
  if (rowBytes > uint64_t(window())) {
    for (uint32_t r = 0; r < rows; ++r)
      for (uint64_t o = 0; o < rowBytes; o += uint64_t(window())) {
        uint32_t n = uint32_t(std::min<uint64_t>(uint64_t(window()), rowBytes - o));
        ldma(l1 + uint64_t(r) * l1Stride + o, win, n, 1, n, n, false);
        gdmaStore(win, lp6 + uint64_t(r) * lp6Stride + o, n, 1, n, n);
      }
    return;
  }
  uint32_t per = uint32_t(std::max<int64_t>(1, window() / rowBytes));
  for (uint32_t r0 = 0; r0 < rows; r0 += per) {
    uint32_t n = std::min(per, rows - r0);
    ldma(l1 + uint64_t(r0) * l1Stride, win, rowBytes, n, l1Stride, rowBytes, false);
    gdmaStore(win, lp6 + uint64_t(r0) * lp6Stride, rowBytes, n, rowBytes, lp6Stride);
  }
}

void Codegen::transfer2d(int64_t ss, int64_t ds, uint64_t src, uint64_t dst, uint32_t runBytes, uint32_t rows,
                         uint32_t sStride, uint32_t dStride, std::string &err) {
  if (rows == 1) sStride = dStride = runBytes;
  // ISA v2 GDMA row_bytes has 20 bits. A contiguous tensor copy can be
  // larger even when its complete L2 allocation fits. Split each row into
  // legal, cache-line-aligned segments and retain the original row strides.
  // Each segment goes through the normal hazard tracking below.
  if (((ss == kLP6 && ds == kL2) || (ss == kL2 && ds == kLP6)) && runBytes >= (1u << 20)) {
    constexpr uint32_t chunk = (1u << 20) - 64;
    for (uint64_t offset = 0; offset < runBytes; offset += chunk)
      transfer2d(ss, ds, src + offset, dst + offset,
                 uint32_t(std::min<uint64_t>(chunk, runBytes - offset)), rows,
                 sStride, dStride, err);
    return;
  }
  if (ss == kL1 && ds == kL1) l1Copy(dst, src, runBytes, rows, sStride, dStride);
  else if (ss == kL2 && ds == kL1) ldma(dst, src, runBytes, rows, sStride, dStride, true);
  else if (ss == kL1 && ds == kL2) ldma(src, dst, runBytes, rows, sStride, dStride, false);
  else if (ss == kLP6 && ds == kL2) gdmaLoad(src, dst, runBytes, rows, sStride, dStride);
  else if (ss == kL2 && ds == kLP6) gdmaStore(src, dst, runBytes, rows, sStride, dStride);
  else if (ss == kLP6 && ds == kL1) copyLP6ToL1(src, dst, runBytes, rows, sStride, dStride);
  else if (ss == kL1 && ds == kLP6) copyL1ToLP6(src, dst, runBytes, rows, sStride, dStride);
  else err = "unsupported copy between memory spaces " + std::to_string(ss) + " -> " + std::to_string(ds);
}

} // namespace mlir::hybrid::codegen
