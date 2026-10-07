//===- CodegenEmit.cpp - NPU codegen: command streams and DMA ----------------===//
//
// Codegen members that build the per-core command streams: blocks and their completion-event
// dependencies, address-range hazards of the GDMA and core accesses to L2 and LP6, and the
// DMA primitives (LDMA, L1 copy, 2-D GDMA, LP6 <-> L1 through the core's staging window).
//===----------------------------------------------------------------------===//

#include "CodegenImpl.h"
#include <algorithm>

namespace mlir::hybrid::codegen {

// ---- command stream per core ------------------------------------------------------------
/// Closes the open block of the current core; later instructions start a new block after it.
int Codegen::flush() {
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
  for (const Acc &a : l2acc)
    if (a.id < 0 && a.core == curCore && overlaps(a.s, a.e, l2, l2e)) {
      flush();
      break;
    }
  std::vector<int> deps;
  for (const Acc &a : l2acc)
    if (overlaps(a.s, a.e, l2, l2e) && a.id >= 0) deps.push_back(a.id);
  for (const LAcc &a : lp6acc)
    if (a.store && overlaps(a.s, a.e, lp6, le)) deps.push_back(a.id);
  std::sort(deps.begin(), deps.end());
  deps.erase(std::unique(deps.begin(), deps.end()), deps.end());
  int id = w.appendGdma(ProgramBuilder::Kind::GdmaLoad, deps, lp6, l2, rowBytes, rows, lp6Stride, l2Stride);
  llvm::erase_if(l2acc, [&](const Acc &a) { return a.s >= l2 && a.e <= l2e; });
  l2acc.push_back({l2, l2e, id, curCore, GW});
  lp6acc.push_back({lp6, le, id, false});
  return id;
}

/// L2 -> LP6 (2-D). Waits for the writers of the L2 range and earlier accesses to the LP6 range.
int Codegen::gdmaStore(uint64_t l2, uint64_t lp6, uint32_t rowBytes, uint32_t rows, uint32_t l2Stride,
                       uint32_t lp6Stride) {
  uint64_t le = lp6 + uint64_t(rows - 1) * lp6Stride + rowBytes, l2e = l2 + uint64_t(rows - 1) * l2Stride + rowBytes;
  for (const Acc &a : l2acc)
    if (a.id < 0 && a.core == curCore && overlaps(a.s, a.e, l2, l2e)) {
      flush();
      break;
    }
  std::vector<int> deps;
  for (const Acc &a : l2acc)
    if (overlaps(a.s, a.e, l2, l2e) && a.id >= 0 && (a.k == GW || a.k == CW)) deps.push_back(a.id);
  for (const LAcc &a : lp6acc)
    if (overlaps(a.s, a.e, lp6, le)) deps.push_back(a.id);
  std::sort(deps.begin(), deps.end());
  deps.erase(std::unique(deps.begin(), deps.end()), deps.end());
  int id = w.appendGdma(ProgramBuilder::Kind::GdmaStore, deps, lp6, l2, rowBytes, rows, lp6Stride, l2Stride);
  l2acc.push_back({l2, l2e, id, curCore, GR});
  llvm::erase_if(lp6acc, [&](const LAcc &a) { return a.s >= lp6 && a.e <= le; });
  lp6acc.push_back({lp6, le, id, true});
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
  coreAccess(rowRanges(l2, rowBytes, rows, toL1 ? srcStride : dstStride), !toL1);
  if (rows == 1 || (srcStride == rowBytes && dstStride == rowBytes)) {
    setConfig(7, rowBytes * rows);
    load(1, uint32_t(l1));
    load(2, uint32_t(l2));
    emit(rform(OP_LDMA, 1, 2, 0, 0, toL1 ? 0 : 1));
    return;
  }
  setConfig(7, rowBytes);
  setConfig(0, rows);
  setConfig(1, srcStride);
  setConfig(2, dstStride);
  load(1, uint32_t(l1));
  load(2, uint32_t(l2));
  emit(rform(OP_LDMA, 1, 2, 0, 0, toL1 ? 4 : 5));
}

/// Contiguous L2 -> L1 weight load. At most one is outstanding; event 1 is reused after
/// its consumer waits. Output stores and other copies retain their blocking semantics.
void Codegen::prefetchWeight(uint64_t l1, uint64_t l2, uint32_t bytes) {
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
