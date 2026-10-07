//===- CodegenImpl.h - NpuCodegen internals ---------------------*- C++ -*-===//
//
// Private to lib/Decode/Target/NPU: the Codegen class that lowers one NPU phase. Its member
// functions are defined by responsibility in
//   NpuCodegen.cpp      emitNpuPhase, Codegen::run, the constant pool
//   CodegenEmit.cpp     command stream per core, address-range hazards, DMA primitives
//   CodegenViews.cpp    index evaluation and memref views
//   CodegenOps.cpp      copy, fill, matmul, GEMV, quant and dequant lowering
//   CodegenGeneric.cpp  linalg.generic vectorizer
//   CodegenLoops.cpp    blocks, scf.for (weight streaming and prefetch), scf.forall
//   CodegenPim.cpp      PIM boundary: output gather, activation scatter, 4-core activation feed
//===----------------------------------------------------------------------===//
#ifndef NPU_LIB_DECODE_TARGET_NPU_CODEGENIMPL_H
#define NPU_LIB_DECODE_TARGET_NPU_CODEGENIMPL_H

#include "npu/Decode/Target/NpuCodegen.h"
#include "npu/Decode/Dialect/Hyb/HybDialect.h"
#include "npu/Decode/Target/PlenaProgram.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

namespace mlir::hybrid::codegen {
using namespace mlir::hybrid::plena;

constexpr int64_t kLP6 = 0, kL1 = 1, kL2 = 2;
constexpr int64_t kL1Base = 4096, kPoolMax = 32 << 10;

inline int64_t spaceOf(Type t) {
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

  // ---- command stream per core, hazards (CodegenEmit.cpp) -----------------------------------
  int flush();
  void blockAfter(const std::vector<int> &deps);
  void startCore(int c, const std::vector<int> &deps);
  void endCore();

  static bool overlaps(uint64_t s1, uint64_t e1, uint64_t s2, uint64_t e2) { return s1 < e2 && s2 < e1; }

  using L2Range = std::pair<uint64_t, uint64_t>;
  // Bound compiler-side hazard metadata for unusually fragmented copies. Above this
  // limit retain the old conservative bounding range; never omit an actual access.
  static constexpr uint64_t kMaxL2AccessRanges = 1024;

  static SmallVector<L2Range> rowRanges(uint64_t base, uint64_t rowBytes, uint64_t rows, uint64_t stride);
  int gdmaLoad(uint64_t lp6, uint64_t l2, uint32_t rowBytes, uint32_t rows, uint32_t lp6Stride, uint32_t l2Stride);
  int gdmaStore(uint64_t l2, uint64_t lp6, uint32_t rowBytes, uint32_t rows, uint32_t l2Stride, uint32_t lp6Stride);
  void coreDependencies(uint64_t s, uint64_t e, bool write, std::vector<int> &need);
  void coreAccess(ArrayRef<L2Range> ranges, bool write);
  void coreAccess(uint64_t s, uint64_t e, bool write) {
    L2Range range{s, e};
    coreAccess(ArrayRef<L2Range>(range), write);
  }

  // ---- data movement primitives (CodegenEmit.cpp) -------------------------------------------
  void ldma(uint64_t l1, uint64_t l2, uint32_t rowBytes, uint32_t rows, uint32_t srcStride, uint32_t dstStride,
            bool toL1);
  void prefetchWeight(uint64_t l1, uint64_t l2, uint32_t bytes);
  void l1Copy(uint64_t dst, uint64_t src, uint32_t rowBytes, uint32_t rows, uint32_t srcStride, uint32_t dstStride);
  void copyLP6ToL1(uint64_t lp6, uint64_t l1, uint32_t rowBytes, uint32_t rows, uint32_t lp6Stride, uint32_t l1Stride);
  void copyL1ToLP6(uint64_t l1, uint64_t lp6, uint32_t rowBytes, uint32_t rows, uint32_t l1Stride, uint32_t lp6Stride);

  // ---- constant pool (NpuCodegen.cpp) -------------------------------------------------------
  int64_t scalarConst(float f);
  int64_t vectorConst(double value, int dtype);
  static int dtypeOf(Type et) {
    return et.isF16() ? DT_F16 : et.isF32() ? DT_F32 : et.isInteger(8) ? DT_I8 : DT_I32;
  }
  std::optional<double> constValue(Value v);
  void collectConstants(func::FuncOp fn);

  // ---- index evaluation and views (CodegenViews.cpp) -----------------------------------------
  FailureOr<int64_t> evalIndex(Value v, std::string &err);
  static std::optional<int64_t> evalAffine(AffineExpr e, ArrayRef<int64_t> dims, ArrayRef<int64_t> syms);
  static bool isViewOp(Operation *op) {
    return isa<memref::SubViewOp, memref::ExpandShapeOp, memref::CollapseShapeOp, memref::CastOp>(op);
  }
  int64_t iterationOf(Operation *op);
  FailureOr<View> viewOf(Value v, std::string &err);

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

  // ---- data movement ops (CodegenEmit.cpp, CodegenOps.cpp) ------------------------------------
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
                  uint32_t sStride, uint32_t dStride, std::string &err);
  LogicalResult lowerCopy(Operation *op, Value srcV, Value dstV, ArrayRef<int64_t> perm, std::string &err);

  // ---- PIM boundary (CodegenPim.cpp) ---------------------------------------------------------
  LogicalResult gatherPimOut(const View &s, const View &d, std::string &err);
  LogicalResult planParallelActivationFeed(func::FuncOp fn, std::string &err);
  LogicalResult scatterPimActParallel(const View &s, const View &, std::string &err);
  std::string parallelActivationPlanJSON() const;
  LogicalResult scatterPimAct(const View &s, const View &d, std::string &err);

  // ---- compute ops (CodegenOps.cpp) ----------------------------------------------------------
  static std::pair<uint64_t, uint64_t> rangeOf(const View &v, uint64_t base);
  static bool isMatmulInit(Operation *user, Value out);
  LogicalResult checkZeroInit(const View &c, std::string &err);
  LogicalResult lowerFill(linalg::FillOp f, std::string &err);
  LogicalResult lowerMatmul(linalg::LinalgOp mm, std::string &err);
  LogicalResult lowerGemv(linalg::GenericOp g, std::string &err);
  LogicalResult lowerQuant(hyb::QuantOp q, std::string &err);
  void reduceToScalar(uint64_t src, int64_t elements, uint32_t funct, uint64_t dst);
  LogicalResult lowerDequant(hyb::DequantOp d, std::string &err);
  LogicalResult lower(Operation *op, std::string &err);

  // ---- control flow (CodegenLoops.cpp) -------------------------------------------------------
  static bool isPure(Operation &op);
  LogicalResult emitBlock(Block &b, std::string &err);
  static uint64_t spanBytes(const View &v);
  SmallVector<L2Range> viewRanges(const View &v) const;
  bool weightPipeline(scf::ForOp f, memref::CopyOp stream, linalg::GenericOp &gemv, memref::CopyOp &output,
                      std::string &err);
  LogicalResult emitFor(scf::ForOp f, std::string &err);
  LogicalResult emitForall(scf::ForallOp f, std::string &err);

  // ---- generic vectorizer (CodegenGeneric.cpp) ----------------------------------------------
  // Kind of a body value: folded constant, row-invariant scalar, or FP16 vector.
  enum class K { Const, Scalar, Vector };
  LogicalResult lowerGeneric(linalg::GenericOp g, std::string &err);
  LogicalResult emitTiles(Block &body, unsigned nIn, ArrayRef<bool> vec, DenseMap<Value, K> &kind,
                          ArrayRef<Operation *> vops, DenseMap<Value, Value> &aliasOf, DenseMap<Value, int> &lastUse,
                          ArrayRef<uint64_t> base, int64_t n, bool reduction, Value pre, Value dotA, Value dotB,
                          uint32_t redFunct, uint32_t combFunct, Value result, DenseMap<Value, uint32_t> &constReg,
                          DenseMap<Value, uint32_t> &rowReg, uint32_t accReg, uint32_t tmpReg, std::string &err);
};

} // namespace mlir::hybrid::codegen

#endif
