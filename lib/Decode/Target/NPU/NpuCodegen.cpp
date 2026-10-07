//===- NpuCodegen.cpp - NPU phase -> PLENA program records ------------------===//
//
// Input: one NPU phase function after hyb-npu-place / -promote / -memplan: every compute op
// works on L1 buffers with planned offsets, data moves only through memref.copy between memory
// spaces (0 LP6, 1 L1, 2 L2), scf.forall distributes over cores, scf.for tiles a loop. This
// backend only translates:
//   * views (subview / expand / collapse / cast) -> (space, base, offset, strides); loop
//     induction variables are evaluated while loops are unrolled;
//   * memref.copy -> DMA by (source space, destination space): L1<->L1 L1_COPY_STRIDED,
//     L2<->L1 strided LDMA, LP6<->L2 2-D GDMA, LP6<->L1 through the core's L2 staging window;
//     PIM layouts: INT32 outputs gathered from their tiles, INT8 inputs scattered to every
//     consuming PU bank (64-byte blocks for rank pairs);
//   * scf.forall -> one command stream per logical core, joined at the end; scf.for ->
//     unrolled; in a weight-streaming loop (hyb.stream) the GDMA of the next tiles is issued
//     ahead into a ring in the staging window. Eligible GEMV loops process several staged tiles
//     per core block and prefetch the next L1 weight slot while the current MMA runs;
//   * compute ops -> vector / scalar / matrix instructions (FP16 vectors, FP32 reductions,
//     FP16 and INT8 MMA, the INT8 quant / dequant contract of the former hybrid_bench).
// Command dependencies come from address ranges: a GDMA waits for earlier accesses to its L2
// and LP6 ranges; a core block that reads or writes L2 waits for the GDMAs touching it.
//===----------------------------------------------------------------------===//

#include "npu/Decode/Target/NpuCodegen.h"
#include "npu/Decode/Dialect/Hyb/HybDialect.h"
#include "npu/Decode/Target/PlenaProgram.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <functional>
#include <map>
#include <set>
#include <unordered_set>

using namespace mlir;
using namespace mlir::hybrid;
using namespace mlir::hybrid::plena;

uint64_t LP6Extra::add(std::vector<uint8_t> bytes) {
  uint64_t a = (cursor + 255) / 256 * 256;
  cursor = a + bytes.size();
  blobs.push_back({a, std::move(bytes)});
  return a;
}

namespace {

constexpr int64_t kLP6 = 0, kL1 = 1, kL2 = 2;
constexpr int64_t kL1Base = 4096, kPoolMax = 32 << 10;

int64_t elemBytes(Type t) { return std::max<int64_t>(1, int64_t(t.getIntOrFloatBitWidth()) / 8); }

uint16_t toHalfBits(float f) {
  // IEEE binary16, round to nearest even.
  uint32_t x;
  std::memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000, mant = x & 0x7FFFFF;
  int32_t exp = int32_t((x >> 23) & 0xFF) - 127 + 15;
  if (((x >> 23) & 0xFF) == 0xFF) return uint16_t(sign | 0x7C00 | (mant ? 0x200 : 0));
  if (exp >= 31) return uint16_t(sign | 0x7C00);
  if (exp <= 0) {
    if (exp < -10) return uint16_t(sign);
    mant |= 0x800000;
    uint32_t shift = uint32_t(14 - exp);
    uint32_t half = mant >> shift, rem = mant & ((1u << shift) - 1), mid = 1u << (shift - 1);
    if (rem > mid || (rem == mid && (half & 1))) half++;
    return uint16_t(sign | half);
  }
  uint32_t half = uint32_t(exp) << 10 | (mant >> 13);
  uint32_t rem = mant & 0x1FFF;
  if (rem > 0x1000 || (rem == 0x1000 && (half & 1))) half++;
  return uint16_t(sign | half);
}

int64_t spaceOf(Type t) {
  auto m = dyn_cast<MemRefType>(t);
  if (!m || !m.getMemorySpace()) return kLP6;
  if (auto i = dyn_cast<IntegerAttr>(m.getMemorySpace())) return i.getInt();
  return kLP6;
}

/// A memref value resolved to addresses: element (i0, i1, ...) is at
/// base + (offset + sum i_k * strides[k]) * eb in `space` (LP6 layouts other than "dense" use
/// logical row-major coordinates instead of addresses).
struct View {
  int64_t space = kLP6;
  uint64_t base = 0;
  int64_t offset = 0;
  SmallVector<int64_t> shape, strides;
  int64_t eb = 2;
  Type et;
  std::string layout = "dense";
  DictionaryAttr lp6;
  int64_t elements() const {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
  }
};

class Codegen {
public:
  Codegen(const NpuPhaseInput &in, LP6Extra &extra) : in(in), extra(extra), w(*in.program) {}

  LogicalResult run(NpuPhaseStats &stats, std::string &err);

private:
  const NpuPhaseInput &in;
  LP6Extra &extra;
  ProgramBuilder &w;
  std::map<int, uint32_t> cfg;
  DenseMap<Value, int64_t> env;                    // loop induction variables while unrolling
  DenseMap<Operation *, std::pair<int64_t, int64_t>> loopBounds;   // scf.for -> (lb, step)
  // Cores.
  int cores = 1, curCore = 0;
  std::vector<std::vector<int>> last;              // per core: the block its next block follows
  std::vector<bool> poolLoaded;
  std::vector<std::unordered_set<int>> satisfied;  // per core: GDMAs its current block waits for
  // Constant pool (L1 at kL1Base on every core, loaded from an LP6 blob).
  std::vector<uint8_t> pool;
  std::map<uint32_t, int64_t> scalarSlot;
  std::map<std::pair<uint32_t, int>, int64_t> vecSlot;
  bool poolFrozen = false;
  int64_t poolL1 = kL1Base, poolBytes = 0;
  uint64_t poolLP6 = 0;
  uint64_t scratchBase = kL1Base + kPoolMax;
  // Zero-filled L1 ranges whose fill was folded into the MMA that overwrites them.
  std::set<std::pair<uint64_t, uint64_t>> zeroed;
  // Address-range hazards.
  enum AccKind { GW, GR, CW, CR };
  struct Acc {
    uint64_t s, e;
    int id, core;
    AccKind k;
  };
  std::vector<Acc> l2acc;
  struct LAcc {
    uint64_t s, e;
    int id;
    bool store;
  };
  std::vector<LAcc> lp6acc;
  std::string opError;

  std::vector<NpuPhaseStats::WeightStream> weightStreams;
  struct ParallelActivationFeed {
    Operation *copy = nullptr;
    int64_t m = 0, k = 0, batchPackets = 0;
    std::vector<int> ranks;
    uint64_t sharedQ = 0, qBytes = 0, dupBytes = 0;
    uint64_t qL1 = 0, scaleL1 = 0, l1Peak = 0;
    int publishEvent = -1;
    std::vector<int> storeEvents;
    std::vector<int64_t> activationIds;
  };
  std::optional<ParallelActivationFeed> parallelActivationFeed;

  int64_t window() const {
    if (auto a = in.phase->getAttrOfType<IntegerAttr>("hyb.l2_staging_per_core")) return a.getInt();
    return in.spec->npu.l2StagingPerCore;
  }
  uint64_t windowBase(int c) const { return uint64_t(c) * uint64_t(window()); }

  // ---- instruction helpers -------------------------------------------------------------
  void emit(uint32_t word) { w.append(word); }
  void rec(const std::array<uint32_t, 4> &r) { w.appendRecord(r); }
  /// Load weight, load activation, then one fixed-tile M_MMA per 32-wide K slice.
  void mma(int64_t k, uint32_t funct) {
    for (uint32_t word : matrixMma(uint32_t(k), funct, false)) w.append(word);
  }
  void load(uint32_t gp, uint32_t v) {
    emit(OP_LUI | gp << 6 | (v >> 12) << 10);
    if (v & 0xFFF) emit(OP_ADDI | gp << 6 | gp << 10 | (v & 0xFFF) << 14);
  }
  void setConfig(uint32_t funct, uint32_t v) {
    auto it = cfg.find(int(funct));
    if (it != cfg.end() && it->second == v) return;
    load(15, v);
    emit(rform(OP_C_SET, 15, 0, 0, 0, funct));
    cfg[int(funct)] = v;
  }
  void setVL(uint32_t n) { setConfig(3, n); }
  void vload(uint32_t v, uint64_t addr, uint32_t funct) {
    load(4, uint32_t(addr));
    emit(rform(OP_V_MEM, v, 4, 0, 0, funct));
  }
  void vstore(uint64_t addr, uint32_t v, uint32_t funct) {
    load(4, uint32_t(addr));
    emit(rform(OP_V_MEM, 4, v, 0, 0, funct));
  }
  void sload(uint32_t s, uint64_t addr) {
    load(5, uint32_t(addr));
    emit(rform(OP_V_MEM, s, 5, 0, 0, 9));
  }
  void sstore(uint64_t addr, uint32_t s) {
    load(5, uint32_t(addr));
    emit(rform(OP_V_MEM, 5, s, 0, 0, 10));
  }

  // ---- command stream per core ------------------------------------------------------------
  /// Closes the open block of the current core; later instructions start a new block after it.
  int flush() {
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
  void blockAfter(const std::vector<int> &deps) {
    flush();
    std::vector<int> f = last[size_t(curCore)];
    for (int d : deps)
      if (std::find(f.begin(), f.end(), d) == f.end()) f.push_back(d);
    w.frontier = f;
    for (int d : deps) satisfied[size_t(curCore)].insert(d);
  }
  void startCore(int c, const std::vector<int> &deps) {
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
  void endCore() {
    int id = w.flushBlock(false);
    if (id >= 0) {
      for (Acc &a : l2acc)
        if (a.id < 0 && a.core == curCore) a.id = id;
      last[size_t(curCore)] = {id};
    } else {
      last[size_t(curCore)] = w.frontier;
    }
  }

  static bool overlaps(uint64_t s1, uint64_t e1, uint64_t s2, uint64_t e2) { return s1 < e2 && s2 < e1; }

  using L2Range = std::pair<uint64_t, uint64_t>;
  // Bound compiler-side hazard metadata for unusually fragmented copies. Above this
  // limit retain the old conservative bounding range; never omit an actual access.
  static constexpr uint64_t kMaxL2AccessRanges = 1024;

  static SmallVector<L2Range> rowRanges(uint64_t base, uint64_t rowBytes, uint64_t rows,
                                      uint64_t stride) {
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
  int gdmaLoad(uint64_t lp6, uint64_t l2, uint32_t rowBytes, uint32_t rows, uint32_t lp6Stride, uint32_t l2Stride) {
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
  int gdmaStore(uint64_t l2, uint64_t lp6, uint32_t rowBytes, uint32_t rows, uint32_t l2Stride, uint32_t lp6Stride) {
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
  void coreDependencies(uint64_t s, uint64_t e, bool write, std::vector<int> &need) {
    for (const Acc &a : l2acc) {
      if (!overlaps(a.s, a.e, s, e) || a.id < 0) continue;
      bool other = a.core != curCore && (a.k == CW || a.k == CR);
      if (a.k == GW || (write && a.k == GR) || (other && (write || a.k == CW)))
        if (!satisfied[size_t(curCore)].count(a.id)) need.push_back(a.id);
    }
  }
  void coreAccess(ArrayRef<L2Range> ranges, bool write) {
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
  void coreAccess(uint64_t s, uint64_t e, bool write) {
    L2Range range{s, e};
    coreAccess(ArrayRef<L2Range>(range), write);
  }

  // ---- data movement primitives -----------------------------------------------------------
  void ldma(uint64_t l1, uint64_t l2, uint32_t rowBytes, uint32_t rows, uint32_t srcStride, uint32_t dstStride,
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
  void prefetchWeight(uint64_t l1, uint64_t l2, uint32_t bytes) {
    coreAccess(l2, l2 + bytes, false);
    setConfig(7, bytes);
    load(1, uint32_t(l1));
    load(2, uint32_t(l2));
    load(10, 1);
    emit(rform(OP_LDMA, 1, 2, 10, 0, 10));
  }
  void l1Copy(uint64_t dst, uint64_t src, uint32_t rowBytes, uint32_t rows, uint32_t srcStride, uint32_t dstStride) {
    setConfig(7, rowBytes);
    setConfig(0, rows);
    setConfig(1, srcStride);
    setConfig(2, dstStride);
    load(3, uint32_t(dst));
    load(4, uint32_t(src));
    emit(rform(OP_LDMA, 3, 4, 0, 0, 9));
  }
  /// LP6 rows -> L1 rows through the core's staging window (chunked to the window).
  void copyLP6ToL1(uint64_t lp6, uint64_t l1, uint32_t rowBytes, uint32_t rows, uint32_t lp6Stride, uint32_t l1Stride) {
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
  void copyL1ToLP6(uint64_t l1, uint64_t lp6, uint32_t rowBytes, uint32_t rows, uint32_t l1Stride, uint32_t lp6Stride) {
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

  // ---- constant pool -------------------------------------------------------------------
  int64_t scalarConst(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    auto it = scalarSlot.find(bits);
    if (it != scalarSlot.end()) return it->second;
    if (poolFrozen) {
      opError = "constant pool is frozen (missed constant)";
      return 0;
    }
    int64_t off = int64_t(pool.size());
    pool.resize(pool.size() + 4);
    std::memcpy(pool.data() + off, &bits, 4);
    scalarSlot[bits] = off;
    return off;
  }
  /// 64-byte vector of one repeated value (dtype: F16, F32, I8, I32).
  int64_t vectorConst(double value, int dtype) {
    uint32_t bits = 0;
    float f = float(value);
    if (dtype == DT_F16) bits = toHalfBits(f);
    else if (dtype == DT_F32) std::memcpy(&bits, &f, 4);
    else bits = uint32_t(int64_t(value));
    auto key = std::make_pair(bits, dtype);
    auto it = vecSlot.find(key);
    if (it != vecSlot.end()) return it->second;
    if (poolFrozen) {
      opError = "constant pool is frozen (missed vector constant)";
      return 0;
    }
    int64_t off = (int64_t(pool.size()) + 63) / 64 * 64;
    pool.resize(off + 64, 0);
    int eb = dtype == DT_F16 ? 2 : dtype == DT_I8 ? 1 : 4;
    for (int i = 0; i < 64 / eb; ++i) std::memcpy(pool.data() + off + i * eb, &bits, eb);
    vecSlot[key] = off;
    return off;
  }
  static int dtypeOf(Type et) {
    return et.isF16() ? DT_F16 : et.isF32() ? DT_F32 : et.isInteger(8) ? DT_I8 : DT_I32;
  }
  std::optional<double> constValue(Value v) {
    Operation *d = v.getDefiningOp();
    if (!d) return std::nullopt;
    if (auto c = dyn_cast<arith::ConstantOp>(d)) {
      if (auto f = dyn_cast<FloatAttr>(c.getValue())) return f.getValueAsDouble();
      if (auto i = dyn_cast<IntegerAttr>(c.getValue())) return double(i.getInt());
      return std::nullopt;
    }
    if (isa<arith::TruncFOp, arith::ExtFOp>(d)) return constValue(d->getOperand(0));
    return std::nullopt;
  }
  void collectConstants(func::FuncOp fn) {
    scalarConst(1.0f / 127.0f);
    scalarConst(0.0f);
    fn.walk([&](Operation *op) {
      if (auto g = dyn_cast<linalg::GenericOp>(op))
        g.getRegion().walk([&](Operation *b) {
          for (Value o : b->getOperands())
            if (auto c = constValue(o)) scalarConst(float(*c));
        });
      if (auto f = dyn_cast<linalg::FillOp>(op))
        if (auto c = constValue(f.getInputs()[0]))
          vectorConst(*c, dtypeOf(getElementTypeOrSelf(f.getOutputs()[0].getType())));
    });
    poolBytes = (int64_t(pool.size()) + 63) / 64 * 64;
    pool.resize(size_t(poolBytes), 0);
    poolFrozen = true;
  }

  // ---- index evaluation and views --------------------------------------------------------
  FailureOr<int64_t> evalIndex(Value v, std::string &err) {
    if (auto it = env.find(v); it != env.end()) return it->second;
    if (auto c = getConstantIntValue(v)) return *c;
    Operation *d = v.getDefiningOp();
    if (!d) {
      err = "index value without a known value";
      return failure();
    }
    auto bin = [&](auto fn) -> FailureOr<int64_t> {
      auto a = evalIndex(d->getOperand(0), err), b = evalIndex(d->getOperand(1), err);
      if (failed(a) || failed(b)) return failure();
      return fn(*a, *b);
    };
    if (isa<arith::AddIOp>(d)) return bin([](int64_t a, int64_t b) { return a + b; });
    if (isa<arith::SubIOp>(d)) return bin([](int64_t a, int64_t b) { return a - b; });
    if (isa<arith::MulIOp>(d)) return bin([](int64_t a, int64_t b) { return a * b; });
    if (isa<arith::DivSIOp, arith::DivUIOp, arith::FloorDivSIOp>(d)) return bin([](int64_t a, int64_t b) { return a / b; });
    if (isa<arith::CeilDivSIOp>(d)) return bin([](int64_t a, int64_t b) { return (a + b - 1) / b; });
    if (isa<arith::RemSIOp, arith::RemUIOp>(d)) return bin([](int64_t a, int64_t b) { return a % b; });
    if (isa<arith::MinSIOp>(d)) return bin([](int64_t a, int64_t b) { return std::min(a, b); });
    if (isa<arith::MaxSIOp>(d)) return bin([](int64_t a, int64_t b) { return std::max(a, b); });
    if (isa<arith::IndexCastOp>(d)) return evalIndex(d->getOperand(0), err);
    auto evalMap = [&](AffineMap map, ValueRange operands) -> FailureOr<SmallVector<int64_t>> {
      SmallVector<int64_t> vals;
      for (Value o : operands) {
        auto x = evalIndex(o, err);
        if (failed(x)) return failure();
        vals.push_back(*x);
      }
      SmallVector<int64_t> out;
      for (AffineExpr e : map.getResults()) {
        auto r = evalAffine(e, ArrayRef<int64_t>(vals).take_front(map.getNumDims()),
                            ArrayRef<int64_t>(vals).drop_front(map.getNumDims()));
        if (!r) {
          err = "cannot evaluate affine expression";
          return failure();
        }
        out.push_back(*r);
      }
      return out;
    };
    if (auto a = dyn_cast<affine::AffineApplyOp>(d)) {
      auto r = evalMap(a.getAffineMap(), a.getMapOperands());
      if (failed(r)) return failure();
      return (*r)[0];
    }
    if (auto a = dyn_cast<affine::AffineMinOp>(d)) {
      auto r = evalMap(a.getAffineMap(), a.getMapOperands());
      if (failed(r)) return failure();
      return *std::min_element(r->begin(), r->end());
    }
    if (auto a = dyn_cast<affine::AffineMaxOp>(d)) {
      auto r = evalMap(a.getAffineMap(), a.getMapOperands());
      if (failed(r)) return failure();
      return *std::max_element(r->begin(), r->end());
    }
    err = "unsupported index op " + d->getName().getStringRef().str();
    return failure();
  }

  static std::optional<int64_t> evalAffine(AffineExpr e, ArrayRef<int64_t> dims, ArrayRef<int64_t> syms) {
    if (auto c = dyn_cast<AffineConstantExpr>(e)) return c.getValue();
    if (auto d = dyn_cast<AffineDimExpr>(e)) return dims[d.getPosition()];
    if (auto s = dyn_cast<AffineSymbolExpr>(e)) return syms[s.getPosition()];
    auto b = dyn_cast<AffineBinaryOpExpr>(e);
    if (!b) return std::nullopt;
    auto l = evalAffine(b.getLHS(), dims, syms), r = evalAffine(b.getRHS(), dims, syms);
    if (!l || !r) return std::nullopt;
    switch (e.getKind()) {
    case AffineExprKind::Add: return *l + *r;
    case AffineExprKind::Mul: return *l * *r;
    case AffineExprKind::Mod: return ((*l % *r) + *r) % *r;
    case AffineExprKind::FloorDiv: return *l >= 0 ? *l / *r : -((-*l + *r - 1) / *r);
    case AffineExprKind::CeilDiv: return (*l + *r - 1) / *r;
    default: return std::nullopt;
    }
  }

  static bool isViewOp(Operation *op) {
    return isa<memref::SubViewOp, memref::ExpandShapeOp, memref::CollapseShapeOp, memref::CastOp>(op);
  }

  /// Iteration index of the innermost scf.for around `op` (double-buffer slot selection).
  int64_t iterationOf(Operation *op) {
    auto f = op->getParentOfType<scf::ForOp>();
    if (!f) return 0;
    auto it = env.find(f.getInductionVar());
    auto b = loopBounds.find(f);
    if (it == env.end() || b == loopBounds.end()) return 0;
    return (it->second - b->second.first) / b->second.second;
  }

  FailureOr<View> viewOf(Value v, std::string &err) {
    auto mt = dyn_cast<MemRefType>(v.getType());
    if (!mt) {
      err = "not a memref";
      return failure();
    }
    View r;
    r.et = mt.getElementType();
    r.eb = elemBytes(r.et);
    r.space = spaceOf(mt);
    auto identity = [&](View &x) {
      x.shape.assign(mt.getShape().begin(), mt.getShape().end());
      x.strides.assign(x.shape.size(), 1);
      for (int i = int(x.shape.size()) - 2; i >= 0; --i) x.strides[i] = x.strides[i + 1] * x.shape[i + 1];
    };
    if (auto ba = dyn_cast<BlockArgument>(v)) {
      if (ba.getOwner()->getParentOp() != func::FuncOp(in.phase).getOperation()) {
        err = "memref block argument of a nested region";
        return failure();
      }
      r.lp6 = in.argLP6[ba.getArgNumber()];
      r.layout = r.lp6.getAs<StringAttr>("layout").getValue().str();
      if (r.layout == "dense") r.base = uint64_t(r.lp6.getAs<IntegerAttr>("addr").getInt());
      identity(r);
      return r;
    }
    Operation *d = v.getDefiningOp();
    if (auto a = dyn_cast<memref::AllocOp>(d)) {
      auto off = a->getAttrOfType<IntegerAttr>("hyb.offset");
      if (!off || (r.space != kL1 && r.space != kL2)) {
        err = "buffer without an L1 / L2 offset";
        return failure();
      }
      r.base = uint64_t(off.getInt());
      if (auto n = a->getAttrOfType<IntegerAttr>("hyb.buffers"))
        r.base += uint64_t(iterationOf(a) % n.getInt()) * uint64_t(a->getAttrOfType<IntegerAttr>("hyb.slot_bytes").getInt());
      identity(r);
      return r;
    }
    if (auto wop = dyn_cast<hyb::WeightOp>(d)) {
      auto p = wop->getAttrOfType<DictionaryAttr>("hyb.lp6");
      if (!p) {
        err = "weight without an LP6 address";
        return failure();
      }
      r.base = uint64_t(p.getAs<IntegerAttr>("addr").getInt());
      identity(r);
      return r;
    }
    if (auto sv = dyn_cast<memref::SubViewOp>(d)) {
      FailureOr<View> s = viewOf(sv.getSource(), err);
      if (failed(s)) return failure();
      View x = *s;
      SmallVector<int64_t> shape, strides;
      int64_t off = s->offset;
      auto offs = sv.getMixedOffsets(), sizes = sv.getMixedSizes(), steps = sv.getMixedStrides();
      llvm::SmallBitVector dropped = sv.getDroppedDims();
      for (size_t i = 0; i < offs.size(); ++i) {
        auto value = [&](OpFoldResult f) -> FailureOr<int64_t> {
          if (auto a = dyn_cast<Attribute>(f)) return cast<IntegerAttr>(a).getInt();
          return evalIndex(cast<Value>(f), err);
        };
        auto o = value(offs[i]), n = value(sizes[i]), st = value(steps[i]);
        if (failed(o) || failed(n) || failed(st)) return failure();
        off += *o * s->strides[i];
        if (dropped.test(i)) continue;
        shape.push_back(*n);
        strides.push_back(s->strides[i] * *st);
      }
      x.offset = off;
      x.shape = shape;
      x.strides = strides;
      return x;
    }
    if (auto ex = dyn_cast<memref::ExpandShapeOp>(d)) {
      FailureOr<View> s = viewOf(ex.getSrc(), err);
      if (failed(s)) return failure();
      View x = *s;
      x.shape.assign(mt.getShape().begin(), mt.getShape().end());
      x.strides.assign(x.shape.size(), 0);
      for (auto [i, group] : llvm::enumerate(ex.getReassociationIndices())) {
        int64_t stride = s->strides[i];
        for (int j = int(group.size()) - 1; j >= 0; --j) {
          x.strides[group[j]] = stride;
          stride *= x.shape[group[j]];
        }
      }
      return x;
    }
    if (auto co = dyn_cast<memref::CollapseShapeOp>(d)) {
      FailureOr<View> s = viewOf(co.getSrc(), err);
      if (failed(s)) return failure();
      View x = *s;
      x.shape.clear();
      x.strides.clear();
      for (auto group : co.getReassociationIndices()) {
        int64_t size = 1, stride = -1, prevStride = 0, prevSize = 0;
        for (int j = int(group.size()) - 1; j >= 0; --j) {
          int64_t dim = group[j];
          if (s->shape[dim] != 1) {
            if (stride < 0) stride = s->strides[dim];
            else if (s->strides[dim] != prevStride * prevSize) {
              err = "collapse of non-contiguous dims";
              return failure();
            }
            prevStride = s->strides[dim];
            prevSize = s->shape[dim];
          }
          size *= s->shape[dim];
        }
        if (stride < 0) stride = group.empty() ? 1 : s->strides[group.back()];
        x.shape.push_back(size);
        x.strides.push_back(stride);
      }
      return x;
    }
    if (auto c = dyn_cast<memref::CastOp>(d)) return viewOf(c.getSource(), err);
    err = "unsupported memref producer " + d->getName().getStringRef().str();
    return failure();
  }

  uint64_t addr(const View &v, int64_t elemOffset = 0) const { return v.base + uint64_t((v.offset + elemOffset) * v.eb); }

  /// Calls fn(offsets...) over the index space of `shape` except the last `inner` dims.
  template <typename Fn> static void forEachOuter(ArrayRef<int64_t> shape, int inner, Fn fn) {
    int outer = int(shape.size()) - inner;
    SmallVector<int64_t> idx(std::max(outer, 0), 0);
    int64_t total = 1;
    for (int d = 0; d < outer; ++d) total *= shape[d];
    for (int64_t t = 0; t < total; ++t) {
      fn(ArrayRef<int64_t>(idx));
      for (int d = outer - 1; d >= 0; --d) {
        if (++idx[d] < shape[d]) break;
        idx[d] = 0;
      }
    }
  }

  // ---- data movement ops ---------------------------------------------------------------------
  DenseMap<Operation *, uint64_t> streamSlot;   // stream copy -> L2 ring slot of this iteration
  Operation *prefetchedWeightCopy = nullptr;
  bool prefetchedWeightNeedsWait = true;
  struct WeightPrefetch {
    Operation *gemv;
    uint64_t l1, l2;
    uint32_t bytes;
  };
  std::optional<WeightPrefetch> nextWeight;

  void transfer2d(int64_t ss, int64_t ds, uint64_t src, uint64_t dst, uint32_t runBytes, uint32_t rows,
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

  LogicalResult lowerCopy(Operation *op, Value srcV, Value dstV, ArrayRef<int64_t> perm, std::string &err) {
    if (op == prefetchedWeightCopy) {
      if (prefetchedWeightNeedsWait) {
        load(10, 1);
        emit(rform(OP_WAIT_EVENT, 10));
      }
      prefetchedWeightCopy = nullptr;
      return success();
    }
    FailureOr<View> s0 = viewOf(srcV, err), d0 = viewOf(dstV, err);
    if (failed(s0) || failed(d0)) return failure();
    if (s0->layout == "pim_out") return gatherPimOut(*s0, *d0, err);
    if (d0->layout == "pim_act") {
      if (parallelActivationFeed && parallelActivationFeed->copy == op)
        return scatterPimActParallel(*s0, *d0, err);
      return scatterPimAct(*s0, *d0, err);
    }
    View s = *s0, d = *d0;
    if (!perm.empty()) {    // linalg.transpose: dst dim i = src dim perm[i]
      View p = s;
      for (size_t i = 0; i < perm.size(); ++i) {
        p.shape[i] = s.shape[perm[i]];
        p.strides[i] = s.strides[perm[i]];
      }
      s = p;
    }
    if (s.shape != d.shape) {
      err = "copy shape mismatch";
      return failure();
    }
    for (int i = int(s.shape.size()) - 1; i >= 0 && s.shape.size() > 1; --i)
      if (s.shape[i] == 1) {
        s.shape.erase(s.shape.begin() + i);
        s.strides.erase(s.strides.begin() + i);
        d.shape.erase(d.shape.begin() + i);
        d.strides.erase(d.strides.begin() + i);
      }
    if (s.shape.empty()) {
      s.shape = d.shape = {1};
      s.strides = d.strides = {1};
    }
    int rank = int(d.shape.size());
    int64_t runElems = 1;
    int inner = 0;
    while (inner < rank) {
      int dim = rank - 1 - inner;
      if (s.strides[dim] != runElems || d.strides[dim] != runElems) break;
      runElems *= d.shape[dim];
      ++inner;
    }
    if (inner == 0) {
      err = "copy without a contiguous innermost dim";
      return failure();
    }
    uint32_t runBytes = uint32_t(runElems * d.eb);
    if (auto it = streamSlot.find(op); it != streamSlot.end()) {
      if (inner != rank || d.space != kL1) {
        err = "stream copy must be contiguous into L1";
        return failure();
      }
      ldma(addr(d), it->second, runBytes, 1, runBytes, runBytes, true);
      return success();
    }
    int rowDim = rank - inner - 1;
    std::string terr;
    auto body = [&](ArrayRef<int64_t> idx) {
      int64_t so = 0, dof = 0;
      for (size_t k = 0; k < idx.size(); ++k) {
        so += idx[k] * s.strides[k];
        dof += idx[k] * d.strides[k];
      }
      uint32_t rows = rowDim >= 0 ? uint32_t(d.shape[rowDim]) : 1;
      uint32_t sStride = rowDim >= 0 ? uint32_t(s.strides[rowDim] * s.eb) : runBytes;
      uint32_t dStride = rowDim >= 0 ? uint32_t(d.strides[rowDim] * d.eb) : runBytes;
      transfer2d(s.space, d.space, addr(s, so), addr(d, dof), runBytes, rows, sStride, dStride, terr);
    };
    SmallVector<int64_t> outerShape(d.shape.begin(), d.shape.begin() + std::max(rowDim, 0));
    if (rowDim < 0) body({});
    else forEachOuter(outerShape, 0, body);
    if (!terr.empty()) {
      err = terr;
      return failure();
    }
    return success();
  }

  /// INT32 GEMV results [rows, cols] of a PIM output -> contiguous L1 rows. A stored tile holds
  /// 8 consecutive outputs of one PU; runs split at DRAM rows and partition boundaries
  /// (as the former hybrid_bench read_outputs), gathered in the staging window, one LDMA per chunk.
  LogicalResult gatherPimOut(const View &s, const View &d, std::string &err) {
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
    int64_t T = p.tilesPerPU(), npu = p.rowsPerPU();
    auto partEnd = [&](int64_t t) {
      int64_t g = t / 8;
      while (g < p.streams() && p.streamPart[size_t(g)] == p.streamPart[size_t(t / 8)]) ++g;
      return g * 8;
    };
    uint64_t win = windowBase(curCore);
    int64_t chunk = window() / 4 / 8 * 8;
    for (int64_t r = 0; r < rows; ++r) {
      int m = int(m0 + r);
      for (int64_t c0 = n0; c0 < n0 + cols; c0 += chunk) {
        int64_t c1 = std::min(n0 + cols, c0 + chunk);
        for (int pu = 0; pu < pim::kPUs; ++pu) {
          int64_t lo = std::max(c0, int64_t(pu) * npu), hi = std::min(c1, int64_t(pu + 1) * npu);
          if (lo >= hi) continue;
          int64_t ta = (lo - pu * npu) / 8, tb = (hi - pu * npu) / 8;
          for (int64_t t = ta; t < tb && t < T;) {
            int64_t tile = p.tokenOutputTile(m, t);
            int64_t pk = p.outputBase + tile;
            int64_t n = std::min({tb - t, pim::kPacketsPerRow - pk % pim::kPacketsPerRow, partEnd(t) - t});
            gdmaLoad(in.pim->outputTileAddress(p, tile, pu), win + uint64_t((pu * npu + t * 8 - c0) * 4), 32,
                     uint32_t(n), uint32_t(in.pim->colBytes()), 32);
            t += n;
          }
        }
        ldma(addr(d, r * dRow + (c0 - n0)), win, uint32_t((c1 - c0) * 4), 1, 0, 0, true);
      }
    }
    return success();
  }

  /// Admit only an isolated quant/feed phase. Other cores' L1 and the staging
  /// windows are phase-owned here; a generic mixed phase may have live values
  /// there and must not silently borrow those resources.
  LogicalResult planParallelActivationFeed(func::FuncOp fn, std::string &err) {
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

  LogicalResult scatterPimActParallel(const View &s, const View &, std::string &err) {
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
          for (int bg = 0; bg < 4; ++bg)
            for (int ch = 0; ch < 8; ++ch)
              l1Copy(scratchBase + uint64_t(bg * in.pim->bgBytes() + ch * pim::kChBytes),
                     row + uint64_t(begin * 32), 32, uint32_t(count), 32, uint32_t(in.pim->colBytes()));
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

  std::string parallelActivationPlanJSON() const {
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

  /// INT8 activation rows (L1) -> every consuming PU's bank (as the former hybrid_bench write_activation).
  /// Subchannel and bank group sit below the column in the address, so one column of the
  /// 4 bank groups x C subchannels of a bank is colBytes() = 128 C contiguous bytes: when every
  /// subchannel of a rank reads the vector, each packet is copied 4 C times in L1 scratch and
  /// the even-bank PUs (0,2,4,6) and odd-bank PUs (1,3,5,7) take one linear store each.
  /// Other partitions are written per PU.
  LogicalResult scatterPimAct(const View &s, const View &d, std::string &err) {
    if (s.space != kL1 || s.strides.back() != 1) {
      err = "PIM activation copy needs contiguous L1 rows";
      return failure();
    }
    auto acts = d.lp6.getAs<ArrayAttr>("acts");
    int64_t K = d.shape[1], sRow = s.shape.size() == 2 ? s.strides[0] : K;
    uint64_t dup = scratchBase, win = windowBase(curCore);
    const int64_t colBytes = in.pim->colBytes(), bgBytes = in.pim->bgBytes();
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
    const int64_t dupBytes = colBytes * batchPackets;
    for (auto [m, ai] : llvm::enumerate(acts)) {
      const pim::Activation &a = in.pim->activations[size_t(cast<IntegerAttr>(ai).getInt())];
      uint64_t row = addr(s, int64_t(m) * sRow);
      std::set<int> ranks, pairRanks;
      for (int p : a.parts) ranks.insert(in.pim->partRank(p));
      for (int rk : ranks) {
        bool both = true;
        for (int c = 0; c < channels; ++c)
          both &= std::find(a.parts.begin(), a.parts.end(), rk * channels + c) != a.parts.end();
        if (both) pairRanks.insert(rk);
      }
      if (!pairRanks.empty()) {
        for (int64_t begin = 0; begin < a.packets(); begin += batchPackets) {
          const int64_t count = std::min(batchPackets, a.packets() - begin);
          for (int bg = 0; bg < 4; ++bg)
            for (int ch = 0; ch < channels; ++ch)
              l1Copy(dup + uint64_t(bg * bgBytes + ch * pim::kChBytes), row + uint64_t(begin * 32),
                     32, uint32_t(count), 32, uint32_t(colBytes));
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
      uint64_t single = win + uint64_t(dupBytes);
      for (int part : a.parts) {
        if (pairRanks.count(in.pim->partRank(part))) continue;
        for (int pu = 0; pu < pim::kPUs; ++pu)
          for (int64_t first = 0; first < a.packets();) {
            int64_t pk = a.packetBase() + first;
            int64_t n = std::min(a.packets() - first, pim::kPacketsPerRow - pk % pim::kPacketsPerRow);
            ldma(row + uint64_t(first * 32), single, uint32_t(32 * n), 1, 0, 0, false);
            gdmaStore(single, in.pim->activationPacketAddress(a, first, pu, part), 32, uint32_t(n), 32,
                      uint32_t(colBytes));
            first += n;
          }
      }
    }
    return success();
  }

  // ---- compute ops ----------------------------------------------------------------------------
  static std::pair<uint64_t, uint64_t> rangeOf(const View &v, uint64_t base) {
    int64_t lastOff = 0;
    for (size_t i = 0; i < v.shape.size(); ++i) lastOff += (v.shape[i] - 1) * v.strides[i];
    return {base, base + uint64_t((lastOff + 1) * v.eb)};
  }

  static bool isMatmulInit(Operation *user, Value out) {
    if (isa<linalg::MatmulOp, linalg::BatchMatmulOp>(user) || user->hasAttr("hyb.npu_gemv"))
      return cast<linalg::LinalgOp>(user).getDpsInits()[0] == out;
    return false;
  }

  LogicalResult checkZeroInit(const View &c, std::string &err) {
    auto r = rangeOf(c, addr(c));
    auto it = zeroed.find(r);
    if (it == zeroed.end()) {
      err = "matmul accumulator must be zero-filled right before (the SA overwrites it)";
      return failure();
    }
    zeroed.erase(it);
    return success();
  }

  LogicalResult lowerFill(linalg::FillOp f, std::string &err) {
    Value out = f.getOutputs()[0];
    auto c = constValue(f.getInputs()[0]);
    if (!c) {
      err = "fill of a non-constant";
      return failure();
    }
    FailureOr<View> v = viewOf(out, err);
    if (failed(v)) return failure();
    if (*c == 0.0 && llvm::any_of(out.getUsers(), [&](Operation *u) { return isMatmulInit(u, out); })) {
      zeroed.insert(rangeOf(*v, addr(*v)));
      return success();
    }
    int dt = dtypeOf(v->et);
    int64_t slot = vectorConst(*c, dt);
    uint32_t loadF = dt == DT_F16 ? 1 : dt == DT_F32 ? 3 : dt == DT_I8 ? 5 : 7;
    uint32_t storeF = loadF + 1;
    int64_t lanes = 64 / v->eb;
    int64_t inner = v->shape.empty() ? 1 : v->shape.back();
    if (!v->shape.empty() && v->strides.back() != 1) {
      err = "fill with a strided innermost dim";
      return failure();
    }
    setVL(uint32_t(std::min(lanes, inner)));
    vload(0, uint64_t(poolL1 + slot), loadF);
    if (v->shape.empty()) {
      setVL(1);
      vstore(addr(*v), 0, storeF);
      return success();
    }
    forEachOuter(v->shape, 1, [&](ArrayRef<int64_t> idx) {
      int64_t off = 0;
      for (size_t k = 0; k < idx.size(); ++k) off += idx[k] * v->strides[k];
      for (int64_t s = 0; s < inner; s += lanes) {
        setVL(uint32_t(std::min(lanes, inner - s)));
        vstore(addr(*v, off + s), 0, storeF);
      }
    });
    return success();
  }

  LogicalResult lowerMatmul(linalg::LinalgOp mm, std::string &err) {
    bool batched = isa<linalg::BatchMatmulOp>(mm.getOperation());
    FailureOr<View> A = viewOf(mm.getDpsInputs()[0], err), B = viewOf(mm.getDpsInputs()[1], err),
                    C = viewOf(mm.getDpsInits()[0], err);
    if (failed(A) || failed(B) || failed(C)) return failure();
    if (!A->et.isF16() || !B->et.isF16()) {
      err = "matmul operands must be f16";
      return failure();
    }
    if (failed(checkZeroInit(*C, err))) return failure();
    auto maps = mm.getIndexingMapsArray();
    int nb = batched ? 1 : 0;
    AffineMap bm = maps[1];
    unsigned kDim = maps[0].getNumDims() - 1;
    bool transB = bm.getResult(bm.getNumResults() - 1) == getAffineDimExpr(kDim, mm.getContext());
    int64_t batch = batched ? C->shape[0] : 1;
    int64_t M = C->shape[nb], N = C->shape[nb + 1], K = A->shape[nb + 1];
    uint32_t woF = C->et.isF16() ? 1 : 4;
    int64_t aRow = A->strides[nb] * 2, cRow = C->strides[nb] * C->eb;
    if (A->strides[nb + 1] != 1 || C->strides[nb + 1] != 1 || B->strides[nb + 1] != 1) {
      err = "matmul operands need a contiguous innermost dim";
      return failure();
    }
    int64_t bRow = B->strides[nb] * 2;
    for (int64_t b = 0; b < batch; ++b) {
      int64_t ao = batched ? b * A->strides[0] : 0, bo = batched ? b * B->strides[0] : 0,
              co = batched ? b * C->strides[0] : 0;
      if (!transB) {
        for (int64_t m0 = 0; m0 < M; m0 += 32)
          for (int64_t n0 = 0; n0 < N; n0 += 32) {
            int64_t mm_ = std::min<int64_t>(32, M - m0), nn = std::min<int64_t>(32, N - n0);
            load(7, uint32_t(addr(*B, bo + n0)));
            load(8, uint32_t(addr(*A, ao + m0 * A->strides[nb])));
            rec(matrixLoad(7, uint32_t(K), uint32_t(nn), uint32_t(bRow), 3));
            rec(matrixLoad(8, uint32_t(mm_), uint32_t(K), uint32_t(aRow), 7));
            mma(K, 3);
            load(9, uint32_t(addr(*C, co + m0 * C->strides[nb] + n0)));
            rec(matrixWriteout(9, uint32_t(mm_), uint32_t(nn), uint32_t(cRow), woF));
          }
      } else {
        // C[m, n0:n0+32] = rows n0..n0+31 of B (as the activation) x row m of A (a K x 1 weight).
        for (int64_t m = 0; m < M; ++m)
          for (int64_t n0 = 0; n0 < N; n0 += 32) {
            int64_t nn = std::min<int64_t>(32, N - n0);
            load(7, uint32_t(addr(*A, ao + m * A->strides[nb])));
            load(8, uint32_t(addr(*B, bo + n0 * B->strides[nb])));
            rec(matrixLoad(7, uint32_t(K), 1, 2, 3));
            rec(matrixLoad(8, uint32_t(nn), uint32_t(K), uint32_t(bRow), 7));
            mma(K, 3);
            load(9, uint32_t(addr(*C, co + m * C->strides[nb] + n0)));
            rec(matrixWriteout(9, uint32_t(nn), 1, uint32_t(C->eb), woF));
          }
      }
    }
    return success();
  }

  /// NPU GEMV: acc[m, nb, c] = sum_k q[m, k] w[nb, k, c] (INT8 x INT8 -> INT32), one MMA per
  /// 32-column block (the weight block is the [K, 32] MMA tile as stored).
  LogicalResult lowerGemv(linalg::GenericOp g, std::string &err) {
    FailureOr<View> Q = viewOf(g.getDpsInputs()[0], err), W = viewOf(g.getDpsInputs()[1], err),
                    C = viewOf(g.getDpsInits()[0], err);
    if (failed(Q) || failed(W) || failed(C)) return failure();
    int64_t M = Q->shape[0], K = Q->shape[1], NB = W->shape[0];
    if (Q->strides[1] != 1 || W->strides[2] != 1 || W->strides[1] != 32 || C->strides[2] != 1 || C->strides[1] != 32) {
      err = "GEMV operands not in the MMA layout";
      return failure();
    }
    if (failed(checkZeroInit(*C, err))) return failure();
    for (int64_t j = 0; j < NB; ++j)
      for (int64_t m0 = 0; m0 < M; m0 += 32) {
        int64_t mm_ = std::min<int64_t>(32, M - m0);
        load(7, uint32_t(addr(*W, j * W->strides[0])));
        load(8, uint32_t(addr(*Q, m0 * Q->strides[0])));
        rec(matrixLoad(7, uint32_t(K), 32, 32, 1));
        rec(matrixLoad(8, uint32_t(mm_), uint32_t(K), uint32_t(Q->strides[0]), 5));
        mma(K, 1);
        if (nextWeight && nextWeight->gemv == g.getOperation() && j + 1 == NB && m0 + 32 >= M) {
          const auto p = *nextWeight;
          prefetchWeight(p.l1, p.l2, p.bytes);
          nextWeight.reset();
        }
        load(9, uint32_t(addr(*C, m0 * C->strides[0] + j * C->strides[1])));
        rec(matrixWriteout(9, uint32_t(mm_), 32, uint32_t(C->strides[0] * 4), 2));
      }
    return success();
  }

  LogicalResult lowerQuant(hyb::QuantOp q, std::string &err) {
    FailureOr<View> X = viewOf(q.getInput(), err), Q = viewOf(q.getQInit(), err), S = viewOf(q.getScaleInit(), err);
    if (failed(X) || failed(Q) || failed(S)) return failure();
    int64_t M = X->shape[0], K = X->shape[1];
    uint64_t temp = scratchBase, maxSlot = temp + uint64_t(K * 2);
    for (int64_t m = 0; m < M; ++m) {
      uint64_t src = addr(*X, m * X->strides[0]), dst = addr(*Q, m * Q->strides[0]);
      for (int64_t s = 0; s < K; s += 32) {          // |x|
        setVL(uint32_t(std::min<int64_t>(32, K - s)));
        vload(0, src + s * 2, 1);
        emit(rform(OP_V_UN, 1, 0, 0, 0, 6));
        vstore(temp + s * 2, 1, 2);
      }
      reduceToScalar(temp, K, 2, maxSlot);           // max|x| (FP32)
      sload(0, maxSlot);
      sload(1, uint64_t(poolL1 + scalarConst(1.0f / 127.0f)));
      emit(rform(OP_S_ALU, 0, 0, 1, 0, 3));          // s0 = max / 127
      sstore(addr(*S, m * S->strides[0]), 0);
      for (int64_t s = 0; s < K; s += 32) {
        setVL(uint32_t(std::min<int64_t>(32, K - s)));
        vload(0, src + s * 2, 1);
        emit(rform(OP_V_CONV, 2, 0, 0, 0, 1));       // V_QUANT_F16_I8 with s0
        vstore(dst + s, 2, 6);
      }
    }
    return success();
  }

  /// Pairwise tree of V_REDUCE results over 32-element tiles of an f16 row: funct 1 sum,
  /// 2 max; result stored as f32 at `dst`.
  void reduceToScalar(uint64_t src, int64_t elements, uint32_t funct, uint64_t dst) {
    std::set<int> occupied;
    uint32_t combine = funct == 2 ? 4 : 1;
    for (int64_t start = 0; start < elements; start += 32) {
      int level = 0;
      while (occupied.count(level)) ++level;
      uint32_t s = uint32_t(4 + level);
      setVL(uint32_t(std::min<int64_t>(32, elements - start)));
      vload(0, src + start * 2, 1);
      emit(rform(OP_V_RED, s, 0, 0, 0, funct));
      for (int lower = 0; lower < level; ++lower) {
        emit(rform(OP_S_ALU, s, uint32_t(4 + lower), s, 0, combine));
        occupied.erase(lower);
      }
      occupied.insert(level);
    }
    std::vector<int> levels(occupied.begin(), occupied.end());
    uint32_t s = uint32_t(4 + levels[0]);
    for (size_t i = 1; i < levels.size(); ++i) emit(rform(OP_S_ALU, s, uint32_t(4 + levels[i]), s, 0, combine));
    sstore(dst, s);
  }

  LogicalResult lowerDequant(hyb::DequantOp d, std::string &err) {
    FailureOr<View> A = viewOf(d.getAcc(), err), S = viewOf(d.getScale(), err), B = viewOf(d.getWscale(), err),
                    Y = viewOf(d.getInit(), err);
    if (failed(A) || failed(S) || failed(B) || failed(Y)) return failure();
    int64_t M = Y->shape[0], N = Y->shape[1];
    for (int64_t m = 0; m < M; ++m) {
      sload(0, addr(*S, m * S->strides[0]));
      uint64_t src = addr(*A, m * A->strides[0]), dst = addr(*Y, m * Y->strides[0]);
      for (int64_t s = 0; s < N; s += 16) {
        setVL(uint32_t(std::min<int64_t>(16, N - s)));
        vload(0, src + s * 4, 7);
        vload(1, addr(*B, s), 3);
        emit(rform(OP_V_CONV, 2, 0, 0, 1, 3));        // V_DEQUANT_I32_F16 v2 = v0 * s0 * v1
        vstore(dst + s * 2, 2, 2);
      }
    }
    return success();
  }

  // ---- control flow -----------------------------------------------------------------------------
  LogicalResult lower(Operation *op, std::string &err) {
    for (Value o : op->getOperands())
      if (isa<MemRefType>(o.getType()) && spaceOf(o.getType()) != kL1) {
        err = "compute op with an operand outside L1 (run hyb-npu-promote)";
        return failure();
      }
    if (auto g = dyn_cast<linalg::GenericOp>(op))
      return g->hasAttr("hyb.npu_gemv") ? lowerGemv(g, err) : lowerGeneric(g, err);
    if (auto f = dyn_cast<linalg::FillOp>(op)) return lowerFill(f, err);
    if (isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op)) return lowerMatmul(cast<linalg::LinalgOp>(op), err);
    if (auto t = dyn_cast<linalg::TransposeOp>(op)) return lowerCopy(op, t.getInput(), t.getInit(), t.getPermutation(), err);
    if (auto q = dyn_cast<hyb::QuantOp>(op)) return lowerQuant(q, err);
    if (auto d = dyn_cast<hyb::DequantOp>(op)) return lowerDequant(d, err);
    err = "unsupported op " + op->getName().getStringRef().str();
    return failure();
  }

  static bool isPure(Operation &op) {
    if (isa<arith::ConstantOp, memref::AllocOp, memref::DeallocOp, hyb::WeightOp, affine::AffineApplyOp,
            affine::AffineMinOp, affine::AffineMaxOp>(op) || isViewOp(&op))
      return true;
    return op.getDialect() && op.getDialect()->getNamespace() == "arith" &&
           llvm::none_of(op.getOperandTypes(), [](Type t) { return isa<MemRefType>(t); });
  }

  LogicalResult emitBlock(Block &b, std::string &err) {
    for (Operation &op : b) {
      if (op.hasTrait<OpTrait::IsTerminator>() || isPure(op)) continue;
      LogicalResult r = success();
      if (auto f = dyn_cast<scf::ForOp>(op)) r = emitFor(f, err);
      else if (auto f = dyn_cast<scf::ForallOp>(op)) r = emitForall(f, err);
      else if (auto c = dyn_cast<memref::CopyOp>(op)) r = lowerCopy(&op, c.getSource(), c.getTarget(), {}, err);
      else r = lower(&op, err);
      if (succeeded(r) && !opError.empty()) {
        err = opError;
        r = failure();
      }
      if (failed(r)) {
        if (!isa<scf::ForOp, scf::ForallOp>(op) && err.find(" [at ") == std::string::npos) {
          std::string s;
          llvm::raw_string_ostream os(s);
          op.print(os, OpPrintingFlags().skipRegions());
          err += " [at " + s.substr(0, 240) + "]";
        }
        return failure();
      }
    }
    return success();
  }

  static uint64_t spanBytes(const View &v) {
    int64_t last = 0;
    for (size_t i = 0; i < v.shape.size(); ++i) last += (v.shape[i] - 1) * v.strides[i];
    return uint64_t((last + 1) * v.eb);
  }

  /// The L2 footprint of a view, excluding holes between batch rows unless the
  /// metadata cap requires a conservative span. Keep the weight pipeline's
  /// preflight consistent with the eventual LDMA accesses.
  SmallVector<L2Range> viewRanges(const View &v) const {
    if (llvm::any_of(v.shape, [](int64_t n) { return n == 0; })) return {};
    uint64_t runBytes = uint64_t(v.eb), count = 1;
    int inner = 0;
    for (int d = int(v.shape.size()) - 1; d >= 0; --d) {
      if (v.shape[d] != 1 && uint64_t(v.strides[d] * v.eb) != runBytes) break;
      runBytes *= uint64_t(v.shape[d]);
      ++inner;
    }
    for (int d = 0; d < int(v.shape.size()) - inner; ++d) {
      if (uint64_t(v.shape[d]) > kMaxL2AccessRanges / count)
        return {{addr(v), addr(v) + spanBytes(v)}};
      count *= uint64_t(v.shape[d]);
    }
    SmallVector<L2Range> ranges;
    ranges.reserve(count);
    forEachOuter(v.shape, inner, [&](ArrayRef<int64_t> idx) {
      int64_t offset = 0;
      for (size_t d = 0; d < idx.size(); ++d) offset += idx[d] * v.strides[d];
      uint64_t start = addr(v, offset);
      ranges.push_back({start, start + runBytes});
    });
    std::sort(ranges.begin(), ranges.end());
    size_t merged = 0;
    for (auto [s, e] : ranges) {
      if (merged && s <= ranges[merged - 1].second)
        ranges[merged - 1].second = std::max(ranges[merged - 1].second, e);
      else ranges[merged++] = {s, e};
    }
    ranges.resize(merged);
    return ranges;
  }

  /// Only pipeline the isolated fill -> weight copy -> GEMV -> L2 output pattern. Other
  /// loops keep the original schedule; moving arbitrary copies would require alias analysis.
  bool weightPipeline(scf::ForOp f, memref::CopyOp stream, linalg::GenericOp &gemv,
                      memref::CopyOp &output, std::string &err) {
    linalg::FillOp fill;
    int stage = 0;
    for (Operation &op : f.getBody()->without_terminator()) {
      if (isPure(op)) continue;
      if (auto x = dyn_cast<linalg::FillOp>(op); x && stage == 0) {
        fill = x;
        stage = 1;
      } else if (&op == stream.getOperation() && stage == 1) {
        stage = 2;
      } else if (auto x = dyn_cast<linalg::GenericOp>(op); x && stage == 2 && x->hasAttr("hyb.npu_gemv")) {
        gemv = x;
        stage = 3;
      } else if (auto x = dyn_cast<memref::CopyOp>(op); x && stage == 3) {
        output = x;
        stage = 4;
      } else {
        return false;
      }
    }
    if (stage != 4 || f.getNumRegionIterArgs() || gemv.getDpsInputs()[1] != stream.getTarget() ||
        fill.getOutputs()[0] != gemv.getDpsInits()[0] || output.getSource() != gemv.getDpsInits()[0] ||
        constValue(fill.getInputs()[0]) != std::optional<double>(0.0)) return false;
    // The activation is loop invariant; both weight slots must be distinct from it and all
    // output slots. Allocation metadata makes this check independent of the iteration.
    Operation *qdef = gemv.getDpsInputs()[0].getDefiningOp();
    if (qdef && f->isProperAncestor(qdef)) return false;
    auto wa = stream.getTarget().getDefiningOp<memref::AllocOp>();
    auto ca = output.getSource().getDefiningOp<memref::AllocOp>();
    if (!wa || !ca || wa->getParentOp() != f || ca->getParentOp() != f) return false;
    auto buffers = wa->getAttrOfType<IntegerAttr>("hyb.buffers");
    auto slot = wa->getAttrOfType<IntegerAttr>("hyb.slot_bytes");
    auto cb = ca->getAttrOfType<IntegerAttr>("hyb.buffers");
    auto cs = ca->getAttrOfType<IntegerAttr>("hyb.slot_bytes");
    if (!buffers || buffers.getInt() != 2 || !slot || !cb || !cs) return false;
    auto W = viewOf(stream.getTarget(), err), Q = viewOf(gemv.getDpsInputs()[0], err),
         C = viewOf(output.getSource(), err), O = viewOf(output.getTarget(), err);
    if (failed(W) || failed(Q) || failed(C) || failed(O)) return false;
    uint64_t wb = uint64_t(wa->getAttrOfType<IntegerAttr>("hyb.offset").getInt()),
             cbeg = uint64_t(ca->getAttrOfType<IntegerAttr>("hyb.offset").getInt()),
             we = wb + 2 * uint64_t(slot.getInt());
    return W->space == kL1 && Q->space == kL1 && C->space == kL1 && O->space == kL2 &&
           spanBytes(*W) == uint64_t(W->elements() * W->eb) && slot.getInt() >= W->elements() * W->eb &&
           !overlaps(wb, we, addr(*Q), addr(*Q) + spanBytes(*Q)) &&
           !overlaps(wb, we, cbeg, cbeg + uint64_t(cb.getInt() * cs.getInt())) &&
           addr(*O) >= uint64_t(cores) * uint64_t(window());
  }

  LogicalResult emitFor(scf::ForOp f, std::string &err) {
    auto lb = evalIndex(f.getLowerBound(), err), ub = evalIndex(f.getUpperBound(), err),
         step = evalIndex(f.getStep(), err);
    if (failed(lb) || failed(ub) || failed(step)) return failure();
    loopBounds[f] = {*lb, *step};
    int64_t n = (*ub - *lb + *step - 1) / *step;
    Value iv = f.getInductionVar();
    // Weight streaming: the GDMA of the next tiles runs ahead into a ring of staging slots.
    memref::CopyOp stream;
    uint64_t slotBytes = 0, nslots = 0;
    if (f->hasAttr("hyb.stream"))
      for (Operation &op : f.getBody()->without_terminator())
        if (auto c = dyn_cast<memref::CopyOp>(op)) {
          Value r = c.getSource();
          while (Operation *d = r.getDefiningOp()) {
            if (!isViewOp(d)) break;
            r = d->getOperand(0);
          }
          if (r.getDefiningOp<hyb::WeightOp>() && spaceOf(c.getTarget().getType()) == kL1) {
            env[iv] = *lb;
            FailureOr<View> v = viewOf(c.getSource(), err);
            if (failed(v)) return failure();
            slotBytes = uint64_t(v->elements() * v->eb + 63) / 64 * 64;
            nslots = uint64_t(window()) / std::max<uint64_t>(1, slotBytes);
            if (in.spec->npu.weightStreamMaxSlots)
              nslots = std::min(nslots, uint64_t(in.spec->npu.weightStreamMaxSlots));
            if (nslots >= 1) stream = c;
            break;
          }
        }
    linalg::GenericOp gemv;
    memref::CopyOp output;
    // Two staged weights are needed for L1 ping/pong, plus a spare L2 slot so the next
    // GDMA can run during this batch. With only two, batching can slow the DRAM pipeline.
    bool pipeline = in.weightPrefetch && stream && n > 1 && nslots >= 3 &&
                    weightPipeline(f, stream, gemv, output, err);
    if (!err.empty()) return failure();
    // Keep at least half the L2 ring available for future GDMA where possible. Wait for
    // the whole small batch up front, so its LDMA operations cannot split the core block.
    int64_t batch = pipeline ? std::max<int64_t>(2, int64_t(nslots / 2)) : 1;
    bool crossBatch = pipeline && in.spec->npu.weightPrefetchCrossBatch;
    if (pipeline && in.spec->npu.weightPrefetchBatchTiles)
      batch = std::min<int64_t>(in.spec->npu.weightPrefetchBatchTiles, int64_t(nslots));
    if (crossBatch) batch = std::min(batch, int64_t(nslots) - 1);
    if (f->hasAttr("hyb.stream")) {
      std::string reason = pipeline ? "enabled" : !in.weightPrefetch ? "disabled" :
          !stream ? "no_staging_slot" : n <= 1 ? "single_iteration" :
          nslots < 3 ? "fewer_than_three_slots" : "unsupported_loop_pattern";
      weightStreams.push_back({int64_t(slotBytes), n, int64_t(nslots), batch, pipeline, crossBatch, reason});
    }
    int64_t issued = -1;
    uint64_t win = windowBase(curCore);
    for (int64_t i = 0; i < n; i += batch) {
      int64_t end = std::min(n, i + batch);
      if (stream) {
        while (issued + 1 < n && issued + 1 <= i + int64_t(nslots) - 1) {
          ++issued;
          env[iv] = *lb + issued * *step;
          FailureOr<View> v = viewOf(stream.getSource(), err);
          if (failed(v)) return failure();
          gdmaLoad(addr(*v), win + uint64_t(issued % int64_t(nslots)) * slotBytes, uint32_t(v->elements() * v->eb), 1, 0, 0);
        }
      }
      if (pipeline) {
        std::vector<int> deps;
        // A carried tile is loaded while this batch's last MMA runs. Its GDMA must
        // already be complete before entering the block, just like all other loads.
        for (int64_t j = i; j < std::min(n, end + int64_t(crossBatch)); ++j) {
          uint64_t slot = win + uint64_t(j % int64_t(nslots)) * slotBytes;
          coreDependencies(slot, slot + slotBytes, false, deps);
          if (j < end) {
            env[iv] = *lb + j * *step;
            auto o = viewOf(output.getTarget(), err);
            if (failed(o)) return failure();
            for (auto [s, e] : viewRanges(*o)) coreDependencies(s, e, true, deps);
          }
        }
        std::sort(deps.begin(), deps.end());
        deps.erase(std::unique(deps.begin(), deps.end()), deps.end());
        blockAfter(deps);
      }
      for (int64_t j = i; j < end; ++j) {
        if (stream) streamSlot[stream] = win + uint64_t(j % int64_t(nslots)) * slotBytes;
        if (pipeline && j + 1 < (crossBatch ? n : end)) {
          env[iv] = *lb + (j + 1) * *step;
          auto dst = viewOf(stream.getTarget(), err);
          if (failed(dst)) return failure();
          nextWeight = WeightPrefetch{gemv, addr(*dst),
              win + uint64_t((j + 1) % int64_t(nslots)) * slotBytes, uint32_t(dst->elements() * dst->eb)};
        }
        prefetchedWeightCopy = pipeline && (j > i || (crossBatch && i > 0)) ? stream.getOperation() : nullptr;
        prefetchedWeightNeedsWait = j > i;
        env[iv] = *lb + j * *step;
        if (failed(emitBlock(*f.getBody(), err))) return failure();
      }
      if (crossBatch && end < n) {
        // Local events are reset at block entry. Consume the event in its producer
        // block; the next block inherits ready L1 data through its block dependency.
        load(10, 1);
        emit(rform(OP_WAIT_EVENT, 10));
      }
      if (pipeline) flush();  // release the batch's L2 read ranges before ring-slot reuse
    }
    if (stream) streamSlot.erase(stream);
    env.erase(iv);
    loopBounds.erase(f);
    return success();
  }

  LogicalResult emitForall(scf::ForallOp f, std::string &err) {
    int64_t n = 1;
    for (OpFoldResult ub : f.getMixedUpperBound()) {
      auto a = dyn_cast<Attribute>(ub);
      if (!a) {
        err = "scf.forall with a dynamic bound";
        return failure();
      }
      n *= cast<IntegerAttr>(a).getInt();
    }
    if (f.getInductionVars().size() != 1 || n > cores) {
      err = "scf.forall must be 1-D over at most the core count";
      return failure();
    }
    int home = curCore;
    flush();
    std::vector<int> start = last[size_t(home)];
    Value iv = f.getInductionVars()[0];
    for (int64_t c = 0; c < n; ++c) {
      startCore(int(c), start);
      env[iv] = c;
      if (failed(emitBlock(*f.getBody(), err))) return failure();
      endCore();
    }
    env.erase(iv);
    std::vector<int> join;
    for (int64_t c = 0; c < n; ++c)
      for (int id : last[size_t(c)])
        if (std::find(join.begin(), join.end(), id) == join.end()) join.push_back(id);
    curCore = home;
    w.logicalCore = home;
    last[size_t(home)] = join;
    w.frontier = join;
    return success();
  }
  // ---- generic vectorizer --------------------------------------------------------------
  // Body values are Const (folded), Scalar (row-invariant: per-row f32 operands and
  // expressions of them, computed once per row on the scalar unit) or Vector (depend on the
  // innermost loop; FP16 lanes, 32 per register). extf / truncf of a vector alias its
  // register (the vector unit is FP16). A reduction over the innermost loop folds each tile
  // with V_REDUCE (FP16 -> FP32; a product of two vectors uses the dot mode, so x*x never
  // rounds to FP16) and an FP32 scalar accumulator.
  enum class K { Const, Scalar, Vector };

  LogicalResult lowerGeneric(linalg::GenericOp g, std::string &err) {
    auto iters = g.getIteratorTypesArray();
    unsigned nLoops = g.getNumLoops();
    SmallVector<int64_t> ranges = g.getStaticLoopRanges();
    if (nLoops == 0) {
      err = "rank-0 generic";
      return failure();
    }
    unsigned inner = nLoops - 1;
    for (unsigned d = 0; d < inner; ++d)
      if (iters[d] != utils::IteratorType::parallel) {
        err = "reduction over a non-innermost loop";
        return failure();
      }
    bool reduction = iters[inner] == utils::IteratorType::reduction;
    MLIRContext *ctx = g.getContext();
    SmallVector<View> views;
    SmallVector<AffineMap> maps;
    SmallVector<bool> vec;
    for (OpOperand &o : g->getOpOperands()) {
      FailureOr<View> v = viewOf(o.get(), err);
      if (failed(v)) return failure();
      AffineMap map = g.getMatchingIndexingMap(&o);
      bool isVec = false;
      for (unsigned r = 0; r < map.getNumResults(); ++r)
        if (map.getResult(r) == getAffineDimExpr(inner, ctx)) {
          if (r != map.getNumResults() - 1 || v->strides.back() != 1) {
            err = "innermost loop is not the contiguous dim of an operand";
            return failure();
          }
          isVec = true;
        }
      if (isVec && !v->et.isF16()) {
        err = "vector operand must be f16";
        return failure();
      }
      views.push_back(*v);
      maps.push_back(map);
      vec.push_back(isVec);
    }
    Block &body = g.getRegion().front();
    unsigned nIn = g.getNumDpsInputs();
    if (g.getNumDpsInits() != 1) {
      err = "generic with several results";
      return failure();
    }
    BlockArgument outArg = body.getArgument(nIn);
    Value result = cast<linalg::YieldOp>(body.getTerminator()).getValues()[0];

    // Reduction: the ops reached from the accumulator are the combiner and casts.
    DenseSet<Operation *> skip;
    Value pre, dotA, dotB;
    uint32_t redFunct = 1, combFunct = 1;
    if (reduction) {
      if (!views[nIn].et.isF32()) {
        err = "reduction accumulator must be f32";
        return failure();
      }
      SmallVector<Value> work{outArg};
      Operation *comb = nullptr;
      while (!work.empty()) {
        Value v = work.pop_back_val();
        for (Operation *u : v.getUsers()) {
          if (isa<linalg::YieldOp>(u) || !skip.insert(u).second) continue;
          if (isa<arith::AddFOp, arith::MaximumFOp, arith::MaxNumFOp>(u)) {
            if (comb) {
              err = "several reduction combiners";
              return failure();
            }
            comb = u;
          } else if (!isa<arith::ExtFOp, arith::TruncFOp>(u)) {
            err = "unsupported use of the reduction accumulator";
            return failure();
          }
          work.push_back(u->getResult(0));
        }
      }
      if (!comb) {
        err = "reduction without a combiner";
        return failure();
      }
      bool isMax = !isa<arith::AddFOp>(comb);
      redFunct = isMax ? 2 : 1;
      combFunct = isMax ? 4 : 1;
      for (Value o : comb->getOperands()) {
        Operation *d = o.getDefiningOp();
        if (!(d && skip.count(d)) && o != outArg) pre = o;
      }
      if (!pre) {
        err = "reduction without an input";
        return failure();
      }
    }

    // Classification.
    DenseMap<Value, K> kind;
    DenseMap<Value, double> cval;
    for (unsigned i = 0; i < nIn; ++i) kind[body.getArgument(i)] = vec[i] ? K::Vector : K::Scalar;
    if (!reduction) kind[outArg] = K::Vector;
    std::function<K(Value)> kindOf = [&](Value v) -> K {
      auto it = kind.find(v);
      if (it != kind.end()) return it->second;
      if (auto c = constValue(v)) {
        cval[v] = *c;
        return kind[v] = K::Const;
      }
      return K::Scalar;
    };
    for (Operation &op : body.without_terminator()) {
      if (skip.count(&op) || op.getNumResults() != 1) continue;
      K k = K::Const;
      for (Value o : op.getOperands()) k = std::max(k, kindOf(o));
      if (k == K::Const) {
        if (auto c = constValue(op.getResult(0))) cval[op.getResult(0)] = *c;
        else k = K::Scalar;
      }
      kind[op.getResult(0)] = k;
    }
    if (reduction)
      if (auto mul = pre.getDefiningOp<arith::MulFOp>();
          mul && redFunct == 1 && pre.hasOneUse() && kind.lookup(mul.getLhs()) == K::Vector &&
          kind.lookup(mul.getRhs()) == K::Vector) {
        dotA = mul.getLhs();
        dotB = mul.getRhs();
        skip.insert(mul);
        redFunct = 3;
      }
    for (unsigned i = 0; i < nIn; ++i)
      if (!vec[i] && !body.getArgument(i).use_empty() && !views[i].et.isF32()) {
        err = "row-invariant operand must be f32";
        return failure();
      }

    // Vector liveness: alias roots and the index of the last use.
    SmallVector<Operation *> vops;
    for (Operation &op : body.without_terminator())
      if (!skip.count(&op) && kind.lookup(op.getResult(0)) == K::Vector) vops.push_back(&op);
    DenseMap<Value, Value> aliasOf;
    auto rootOfV = [&](Value v) {
      auto it = aliasOf.find(v);
      return it == aliasOf.end() ? v : it->second;
    };
    for (Operation *op : vops)
      if (isa<arith::ExtFOp, arith::TruncFOp>(op)) aliasOf[op->getResult(0)] = rootOfV(op->getOperand(0));
    DenseMap<Value, int> lastUse;
    int endIndex = int(vops.size());
    for (auto [i, op] : llvm::enumerate(vops))
      for (Value o : op->getOperands())
        if (kind.lookup(o) == K::Vector) lastUse[rootOfV(o)] = std::max(lastUse.lookup(rootOfV(o)), int(i));
    if (reduction) {
      for (Value v : {pre, dotA, dotB})
        if (v) lastUse[rootOfV(v)] = endIndex;
    } else {
      lastUse[rootOfV(result)] = endIndex;
    }

    // Scalar registers: s1, s3 scratch; s11 the accumulator; the rest for body scalars.
    static constexpr uint32_t kPool[] = {12, 13, 14, 15, 4, 5, 6, 7, 8, 9, 10, 0, 2};
    const uint32_t accReg = 11, tmpReg = 3;
    DenseMap<Value, uint32_t> constReg, rowReg;
    unsigned nextPool = 0;
    auto take = [&]() -> FailureOr<uint32_t> {
      if (nextPool >= std::size(kPool)) {
        err = "too many scalar values in a generic body";
        return failure();
      }
      return kPool[nextPool++];
    };
    SmallVector<uint64_t> base(views.size(), 0);
    std::function<FailureOr<uint32_t>(Value)> scalarOf = [&](Value v) -> FailureOr<uint32_t> {
      if (auto it = constReg.find(v); it != constReg.end()) return it->second;
      if (auto it = rowReg.find(v); it != rowReg.end()) return it->second;
      K k = kindOf(v);
      if (k == K::Const) {
        auto r = take();
        if (failed(r)) return failure();
        sload(*r, uint64_t(poolL1 + scalarConst(float(cval.lookup(v)))));
        return constReg[v] = *r;
      }
      if (auto ba = dyn_cast<BlockArgument>(v)) {
        auto r = take();
        if (failed(r)) return failure();
        sload(*r, base[ba.getArgNumber()]);
        return rowReg[v] = *r;
      }
      Operation *d = v.getDefiningOp();
      if (isa<arith::ExtFOp, arith::TruncFOp>(d)) {
        auto r = scalarOf(d->getOperand(0));
        if (succeeded(r)) rowReg[v] = *r;
        return r;
      }
      auto a = scalarOf(d->getOperand(0));
      if (failed(a)) return failure();
      FailureOr<uint32_t> b = uint32_t(0);
      if (d->getNumOperands() > 1) b = scalarOf(d->getOperand(1));
      if (failed(b)) return failure();
      auto r = take();
      if (failed(r)) return failure();
      if (isa<arith::AddFOp>(d)) emit(rform(OP_S_ALU, *r, *a, *b, 0, 1));
      else if (isa<arith::SubFOp>(d)) emit(rform(OP_S_ALU, *r, *a, *b, 0, 2));
      else if (isa<arith::MulFOp>(d)) emit(rform(OP_S_ALU, *r, *a, *b, 0, 3));
      else if (isa<arith::MaximumFOp, arith::MaxNumFOp>(d)) emit(rform(OP_S_ALU, *r, *a, *b, 0, 4));
      else if (isa<arith::MinimumFOp, arith::MinNumFOp>(d)) emit(rform(OP_S_ALU, *r, *a, *b, 0, 5));
      else if (isa<arith::DivFOp>(d)) {
        emit(rform(OP_S_ALU, 1, *b, 0, 0, 6));        // s1 = 1 / b
        emit(rform(OP_S_ALU, *r, *a, 1, 0, 3));
      } else if (isa<math::SqrtOp>(d)) emit(rform(OP_S_ALU, *r, *a, 0, 0, 7));
      else if (isa<math::RsqrtOp>(d)) emit(rform(OP_S_ALU, *r, *a, 0, 0, 8));
      else if (isa<arith::NegFOp>(d)) {
        sload(1, uint64_t(poolL1 + scalarConst(0.0f)));
        emit(rform(OP_S_ALU, *r, 1, *a, 0, 2));
      } else {
        err = "unsupported scalar op " + d->getName().getStringRef().str();
        return failure();
      }
      return rowReg[v] = *r;
    };

    // Constants first (once per op), then the row scalars reuse the registers after them.
    std::function<LogicalResult(Value)> loadConsts = [&](Value v) -> LogicalResult {
      K k = kindOf(v);
      if (k == K::Const) return failed(scalarOf(v)) ? failure() : success();
      if (k != K::Scalar || isa<BlockArgument>(v)) return success();
      for (Value o : v.getDefiningOp()->getOperands())
        if (failed(loadConsts(o))) return failure();
      return success();
    };
    for (Operation *op : vops)
      for (Value o : op->getOperands())
        if (failed(loadConsts(o))) return failure();
    const unsigned constMark = nextPool;

    SmallVector<int64_t> outer(ranges.begin(), ranges.begin() + inner);
    int64_t n = ranges[inner];
    LogicalResult status = success();
    forEachOuter(outer, 0, [&](ArrayRef<int64_t> idx) {
      if (failed(status)) return;
      for (size_t i = 0; i < views.size(); ++i) {
        int64_t off = 0;
        for (unsigned r = 0; r < maps[i].getNumResults(); ++r) {
          AffineExpr e = maps[i].getResult(r);
          int64_t at = 0;
          if (auto de = dyn_cast<AffineDimExpr>(e)) at = de.getPosition() < inner ? idx[de.getPosition()] : 0;
          else if (auto ce = dyn_cast<AffineConstantExpr>(e)) at = ce.getValue();
          off += at * views[i].strides[r];
        }
        base[i] = addr(views[i], off);
      }
      rowReg.clear();
      nextPool = constMark;
      for (Operation *op : vops)
        for (Value o : op->getOperands()) {
          K k = kind.lookup(o);
          if ((k == K::Scalar || k == K::Const) && failed(scalarOf(o))) {
            status = failure();
            return;
          }
        }
      if (reduction) sload(accReg, base[nIn]);
      if (failed(emitTiles(body, nIn, vec, kind, vops, aliasOf, lastUse, base, n, reduction, pre, dotA, dotB,
                           redFunct, combFunct, result, constReg, rowReg, accReg, tmpReg, err))) {
        status = failure();
        return;
      }
      if (reduction) sstore(base[nIn], accReg);
    });
    return status;
  }

  LogicalResult emitTiles(Block &body, unsigned nIn, ArrayRef<bool> vec, DenseMap<Value, K> &kind,
                          ArrayRef<Operation *> vops, DenseMap<Value, Value> &aliasOf, DenseMap<Value, int> &lastUse,
                          ArrayRef<uint64_t> base, int64_t n, bool reduction, Value pre, Value dotA, Value dotB,
                          uint32_t redFunct, uint32_t combFunct, Value result, DenseMap<Value, uint32_t> &constReg,
                          DenseMap<Value, uint32_t> &rowReg, uint32_t accReg, uint32_t tmpReg, std::string &err) {
    auto rootOfV = [&](Value v) {
      auto it = aliasOf.find(v);
      return it == aliasOf.end() ? v : it->second;
    };
    auto sreg = [&](Value v) -> uint32_t {
      if (auto it = constReg.find(v); it != constReg.end()) return it->second;
      return rowReg.lookup(v);
    };
    for (int64_t s = 0; s < n; s += 32) {
      setVL(uint32_t(std::min<int64_t>(32, n - s)));
      DenseMap<Value, uint32_t> vreg;
      std::array<bool, 16> used{};
      auto takeV = [&]() -> int {
        for (int r = 0; r < 16; ++r)
          if (!used[r]) {
            used[r] = true;
            return r;
          }
        return -1;
      };
      auto regOf = [&](Value v) { return vreg.lookup(rootOfV(v)); };
      unsigned nArgs = reduction ? nIn : nIn + 1;
      for (unsigned i = 0; i < nArgs; ++i) {
        BlockArgument a = body.getArgument(i);
        if ((i < nIn && !vec[i]) || !lastUse.count(a)) continue;
        int r = takeV();
        if (r < 0) {
          err = "out of vector registers";
          return failure();
        }
        vload(uint32_t(r), base[i] + uint64_t(s) * 2, 1);
        vreg[a] = uint32_t(r);
      }
      auto release = [&](int index) {
        for (auto &[v, r] : vreg)
          if (lastUse.lookup(v) == index) used[r] = false;
      };
      for (auto [index, op] : llvm::enumerate(vops)) {
        Value res = op->getResult(0);
        if (aliasOf.count(res)) continue;
        int r = takeV();
        if (r < 0) {
          err = "out of vector registers";
          return failure();
        }
        uint32_t d = uint32_t(r);
        auto vk = [&](Value v) { return kind.lookup(v) == K::Vector; };
        Value a0 = op->getNumOperands() > 0 ? op->getOperand(0) : Value();
        Value a1 = op->getNumOperands() > 1 ? op->getOperand(1) : Value();
        auto vv = [&](uint32_t f) { emit(rform(OP_V_BIN, d, regOf(a0), regOf(a1), 0, f)); };
        auto vs = [&](Value v, Value sc, uint32_t f) { emit(rform(OP_V_SCALAR, d, regOf(v), sreg(sc), 0, f)); };
        if (isa<arith::AddFOp, arith::MulFOp, arith::MaximumFOp, arith::MaxNumFOp, arith::MinimumFOp,
                arith::MinNumFOp>(op)) {
          uint32_t f = isa<arith::AddFOp>(op) ? 1 : isa<arith::MulFOp>(op) ? 3
                       : isa<arith::MaximumFOp, arith::MaxNumFOp>(op) ? 4 : 5;
          if (vk(a0) && vk(a1)) vv(f);
          else if (vk(a0)) vs(a0, a1, f);
          else vs(a1, a0, f);
        } else if (isa<arith::SubFOp>(op)) {
          if (vk(a0) && vk(a1)) vv(2);
          else if (vk(a0)) vs(a0, a1, 2);
          else {                                       // s - v = (-v) + s
            emit(rform(OP_V_UN, d, regOf(a1), 0, 0, 1));
            emit(rform(OP_V_SCALAR, d, d, sreg(a0), 0, 1));
          }
        } else if (isa<arith::DivFOp>(op)) {
          if (vk(a0) && vk(a1)) {
            emit(rform(OP_V_UN, d, regOf(a1), 0, 0, 3));
            emit(rform(OP_V_BIN, d, regOf(a0), d, 0, 3));
          } else if (vk(a0)) {                         // v / s = v * (1 / s)
            emit(rform(OP_S_ALU, tmpReg, sreg(a1), 0, 0, 6));
            emit(rform(OP_V_SCALAR, d, regOf(a0), tmpReg, 0, 3));
          } else {                                     // s / v = rcp(v) * s
            emit(rform(OP_V_UN, d, regOf(a1), 0, 0, 3));
            emit(rform(OP_V_SCALAR, d, d, sreg(a0), 0, 3));
          }
        } else if (isa<arith::NegFOp>(op)) emit(rform(OP_V_UN, d, regOf(a0), 0, 0, 1));
        else if (isa<math::ExpOp>(op)) emit(rform(OP_V_UN, d, regOf(a0), 0, 0, 2));
        else if (isa<math::SqrtOp>(op)) emit(rform(OP_V_UN, d, regOf(a0), 0, 0, 4));
        else if (isa<math::RsqrtOp>(op)) emit(rform(OP_V_UN, d, regOf(a0), 0, 0, 5));
        else if (isa<math::AbsFOp>(op)) emit(rform(OP_V_UN, d, regOf(a0), 0, 0, 6));
        else {
          err = "unsupported vector op " + op->getName().getStringRef().str();
          return failure();
        }
        vreg[res] = d;
        release(int(index));
      }
      if (reduction) {
        if (dotA) emit(rform(OP_V_RED, tmpReg, regOf(dotA), regOf(dotB), 0, 3));
        else emit(rform(OP_V_RED, tmpReg, regOf(pre), 0, 0, redFunct));
        emit(rform(OP_S_ALU, accReg, accReg, tmpReg, 0, combFunct));
      } else {
        vstore(base[nIn] + uint64_t(s) * 2, regOf(result), 2);
      }
    }
    return success();
  }
};

LogicalResult Codegen::run(NpuPhaseStats &stats, std::string &err) {
  func::FuncOp fn = func::FuncOp(in.phase);
  if (failed(planParallelActivationFeed(fn, err))) return failure();
  cores = parallelActivationFeed ? 4 : 1;
  fn.walk([&](scf::ForallOp f) {
    if (auto c = f->getAttrOfType<IntegerAttr>("hyb.cores")) cores = std::max<int>(cores, int(c.getInt()));
  });
  if (cores > in.spec->npu.cores) {
    err = "phase uses more cores than the target has";
    return failure();
  }
  if (window() < 64 || uint64_t(window()) * uint64_t(in.spec->npu.cores) > uint64_t(in.spec->npu.l2Bytes)) {
    err = "missing or invalid planned L2 staging window";
    return failure();
  }
  // Every core's first block follows the phase entry (ProgramBuilder::beginPhase).
  last.assign(size_t(cores), w.frontier);
  const int64_t words0 = w.coreWords();
  const int commands0 = w.numCommands();
  poolLoaded.assign(size_t(cores), false);
  satisfied.assign(size_t(cores), {});
  scratchBase = uint64_t(kL1Base + kPoolMax);
  collectConstants(fn);
  if (poolBytes > kPoolMax || kPoolMax >= in.spec->npu.l1ReservedBytes) {
    err = "constant pool larger than its reserve";
    return failure();
  }
  if (poolBytes) poolLP6 = extra.add(pool);
  startCore(0, {});
  if (failed(emitBlock(fn.getBody().front(), err))) return failure();
  endCore();
  w.flushBlock(false);
  stats.coreWords = w.coreWords() - words0;
  stats.commands = w.numCommands() - commands0;
  if (auto p = fn->getAttrOfType<IntegerAttr>("hyb.l1_peak")) stats.l1Peak = p.getInt();
  stats.cores = cores;
  stats.l2StagingPerCore = window();
  stats.weightStreams = std::move(weightStreams);
  if (parallelActivationFeed) {
    stats.pimActivationCores = 4;
    stats.pimActivationPlanJson = parallelActivationPlanJSON();
    fn->setAttr("hyb.pim_activation_plan", StringAttr::get(fn.getContext(), stats.pimActivationPlanJson));
  }
  return success();
}

} // namespace

LogicalResult mlir::hybrid::emitNpuPhase(const NpuPhaseInput &in, LP6Extra &extra, NpuPhaseStats &stats,
                                        std::string &error) {
  Codegen cg(in, extra);
  return cg.run(stats, error);
}
