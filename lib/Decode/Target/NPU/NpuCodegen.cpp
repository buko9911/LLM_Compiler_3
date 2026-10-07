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
// The Codegen class is declared in CodegenImpl.h; this file holds the entry point and the
// constant pool, the Codegen*.cpp files the rest of its members.
//===----------------------------------------------------------------------===//

#include "CodegenImpl.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include <cstring>

using namespace mlir;
using namespace mlir::hybrid;

uint64_t LP6Extra::add(std::vector<uint8_t> bytes) {
  uint64_t a = (cursor + 255) / 256 * 256;
  cursor = a + bytes.size();
  blobs.push_back({a, std::move(bytes)});
  return a;
}

namespace mlir::hybrid::codegen {
namespace {

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

} // namespace

// ---- constant pool -------------------------------------------------------------------
int64_t Codegen::scalarConst(float f) {
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
int64_t Codegen::vectorConst(double value, int dtype) {
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

std::optional<double> Codegen::constValue(Value v) {
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

void Codegen::collectConstants(func::FuncOp fn) {
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

} // namespace mlir::hybrid::codegen

LogicalResult mlir::hybrid::emitNpuPhase(const NpuPhaseInput &in, LP6Extra &extra, NpuPhaseStats &stats,
                                        std::string &error) {
  codegen::Codegen cg(in, extra);
  return cg.run(stats, error);
}
