// Materialize transposed FP16 RHS operands to expose full SA output tiles.
// The backend otherwise computes one output row at a time for transpose-B.
// Packing changes addresses, not K reduction order or accumulator precision.
#include "npu/Decode/Support/TargetSpec.h"
#include "npu/Decode/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include <algorithm>

namespace mlir::hybrid {
#define GEN_PASS_DEF_HYBNPUMATMULLAYOUT
#include "npu/Decode/Transforms/Passes.h.inc"
}
using namespace mlir;
using namespace mlir::hybrid;
namespace {
Value allocationRoot(Value value) {
  while (Operation *op = value.getDefiningOp()) {
    if (!isa<memref::SubViewOp, memref::ExpandShapeOp, memref::CollapseShapeOp,
             memref::CastOp, memref::ReinterpretCastOp>(op)) break;
    value = op->getOperand(0);
  }
  return value;
}
// Bound the new scratch without relying on future physical allocation reuse.
// Every allocation in the current block and enclosing blocks is counted, even
// when dead. Loop allocations reserve both slots required by the memory planner.
bool fitsScratch(Operation *at, uint64_t extra, const TargetSpec &spec) {
  uint64_t used = spec.npu.l1ReservedBytes;
  if (used > uint64_t(spec.npu.l1Bytes) || extra > uint64_t(spec.npu.l1Bytes) - used) return false;
  used += extra;
  for (Block *block = at->getBlock(); block; ) {
    for (Operation &op : *block) {
      auto alloc = dyn_cast<memref::AllocOp>(op);
      if (!alloc || alloc.getType().getMemorySpaceAsInt() != 1) continue;
      auto type = alloc.getType();
      if (!type.hasStaticShape() || !type.getElementType().isIntOrFloat()) return false;
      uint64_t bytes = type.getElementTypeBitWidth() / 8;
      for (int64_t dim : type.getShape()) {
        if (dim < 0 || (dim && bytes > uint64_t(spec.npu.l1Bytes) / uint64_t(dim))) return false;
        bytes *= dim;
      }
      // Use the address span for strided allocations, not just element count.
      SmallVector<int64_t> strides;
      int64_t offset;
      if (failed(type.getStridesAndOffset(strides, offset)) || offset < 0) return false;
      uint64_t span = offset;
      for (auto [dim, stride] : llvm::zip(type.getShape(), strides)) {
        if (stride < 0 || stride == ShapedType::kDynamic) return false;
        if (dim > 0) {
          if (uint64_t(stride) > uint64_t(spec.npu.l1Bytes) / uint64_t(dim)) return false;
          span += uint64_t(dim - 1) * uint64_t(stride);
          if (span > uint64_t(spec.npu.l1Bytes)) return false;
        }
      }
      span = (span + 1) * (type.getElementTypeBitWidth() / 8);
      bytes = std::max(bytes, span);
      if (isa<scf::ForOp>(alloc->getParentOp())) bytes *= 2;
      if (bytes > uint64_t(spec.npu.l1Bytes) - used) return false;
      used += bytes;
    }
    Operation *parent = block->getParentOp();
    if (!parent || isa<func::FuncOp>(parent)) break;
    block = parent->getBlock();
  }
  return true;
}

// The algebraic transpose-output form uses a fresh zero accumulator. Admit
// only a direct dominating zero fill with no intervening access through aliases.
linalg::FillOp zeroInitializer(Value output, Operation *matrix) {
  Value root = allocationRoot(output);
  for (Operation *previous = matrix->getPrevNode(); previous; previous = previous->getPrevNode()) {
    bool touches = false;
    previous->walk([&](Operation *op) {
      for (Value value : op->getOperands())
        if (isa<MemRefType>(value.getType()) && allocationRoot(value) == root) touches = true;
    });
    if (!touches) continue;
    auto fill = dyn_cast<linalg::FillOp>(previous);
    if (!fill || fill.getOutputs()[0] != output) return {};
    Attribute constant;
    if (!matchPattern(fill.getInputs()[0], m_Constant(&constant))) return {};
    auto value = dyn_cast<FloatAttr>(constant);
    return value && value.getValue().isZero() && !value.getValue().isNegative() ? fill : linalg::FillOp{};
  }
  return {};
}

bool transposeOutput(linalg::LinalgOp matrix, Value a, Value b, Value c,
                     const TargetSpec &spec, ArrayRef<AffineMap> maps) {
  if (!spec.npu.matmulTransposeOutput) return false;
  auto originalFill = zeroInitializer(c, matrix);
  if (!originalFill) return false;
  auto at = cast<MemRefType>(a.getType()), bt = cast<MemRefType>(b.getType()), ct = cast<MemRefType>(c.getType());
  unsigned rank = at.getRank();
  if (!ct.getLayout().isIdentity()) return false;
  // All operands must fit before computing element counts; avoid overflow on
  // untrusted static shapes. Compare actual transpose traffic, not op names.
  auto elements = [&](MemRefType type) -> uint64_t {
    uint64_t count = 1;
    for (int64_t dim : type.getShape()) {
      if (count > uint64_t(spec.npu.l1Bytes) / 2 / uint64_t(dim)) return 0;
      count *= uint64_t(dim);
    }
    return count;
  };
  uint64_t ae = elements(at), be = elements(bt), ce = elements(ct);
  if (!ae || !be || !ce || ae + ce >= be) return false;
  uint64_t bytes = 2 * (ae + ce);
  if (isa<scf::ForOp>(matrix->getParentOp())) bytes *= 2;
  if (!fitsScratch(matrix, bytes, spec)) return false;
  SmallVector<int64_t> ashape(at.getShape()), cshape(ct.getShape()), permutation;
  for (unsigned i = 0; i < rank; ++i) permutation.push_back(i);
  std::swap(ashape[rank-2], ashape[rank-1]);
  std::swap(cshape[rank-2], cshape[rank-1]);
  std::swap(permutation[rank-2], permutation[rank-1]);
  OpBuilder builder(matrix);
  auto alloc = [&](ArrayRef<int64_t> shape) {
    auto result = builder.create<memref::AllocOp>(matrix.getLoc(),
      MemRefType::get(shape, at.getElementType(), MemRefLayoutAttrInterface{}, at.getMemorySpace()));
    result->setAttr("hyb.matmul_transpose_scratch", builder.getUnitAttr());
    return result;
  };
  auto packedA = alloc(ashape), temporary = alloc(cshape);
  // No original C read occurs between its proven fill and the complete final
  // transpose overwrite. Preserve the former matrix-init zero-store elision.
  originalFill->setAttr("hyb.elide_matrix_init", builder.getUnitAttr());
  builder.create<linalg::TransposeOp>(matrix.getLoc(), a, packedA.getResult(), permutation);
  auto zero = builder.create<arith::ConstantOp>(matrix.getLoc(), builder.getF16FloatAttr(0.0));
  auto fill = builder.create<linalg::FillOp>(matrix.getLoc(), ValueRange{zero.getResult()}, ValueRange{temporary.getResult()});
  // This newly private D is fully overwritten by the matrix ISA INIT before
  // its sole later reader, the output transpose. Avoid materializing zeros.
  fill->setAttr("hyb.elide_matrix_init", builder.getUnitAttr());
  matrix->setOperand(0, b);
  matrix->setOperand(1, packedA);
  matrix->setOperand(2, temporary);
  matrix->setAttr("indexing_maps", builder.getAffineMapArrayAttr(maps));
  matrix->setAttr("hyb.matmul_transpose_output", builder.getUnitAttr());
  builder.setInsertionPointAfter(matrix);
  builder.create<linalg::TransposeOp>(matrix.getLoc(), temporary.getResult(), c, permutation);
  return true;
}

bool pack(linalg::LinalgOp matrix, const TargetSpec &spec) {
  bool batched = isa<linalg::BatchMatmulOp>(matrix.getOperation());
  if ((!batched && !isa<linalg::MatmulOp>(matrix.getOperation())) ||
      matrix->getNumResults() || matrix.getNumDpsInputs() != 2 || matrix.getNumDpsInits() != 1) return false;
  Value a = matrix.getDpsInputs()[0], b = matrix.getDpsInputs()[1], c = matrix.getDpsInits()[0];
  unsigned rank = batched ? 3 : 2, dims = rank + 1;
  for (Value operand : {a, b, c}) {
    auto type = dyn_cast<MemRefType>(operand.getType());
    if (!type || type.getRank() != rank || !type.hasStaticShape() ||
        !type.getElementType().isF16() || type.getMemorySpaceAsInt() != 1 ||
        llvm::any_of(type.getShape(), [](int64_t d) { return d <= 0; })) return false;
  }
  Value outputRoot = allocationRoot(c);
  if (!outputRoot.getDefiningOp<memref::AllocOp>() ||
      outputRoot == allocationRoot(a) || outputRoot == allocationRoot(b)) return false;
  auto at = cast<MemRefType>(a.getType()), bt = cast<MemRefType>(b.getType()), ct = cast<MemRefType>(c.getType());
  // Both layout variants change output traversal order; overlapping logical
  // output elements are not a legal layout-rewrite candidate.
  if (!ct.getLayout().isIdentity()) return false;
  // Packing only repays the extra movement when multiple output rows can share
  // an SA tile. Single-row products retain the original zero-packing path.
  if (at.getDimSize(rank - 2) <= 1) return false;
  MLIRContext *ctx = matrix.getContext();
  auto d = [&](unsigned n) { return getAffineDimExpr(n, ctx); };
  SmallVector<AffineExpr> am, bm, cm, packedMap;
  if (batched) { am.push_back(d(0)); bm.push_back(d(0)); cm.push_back(d(0)); packedMap.push_back(d(0)); }
  unsigned m = batched ? 1 : 0, n = m + 1, k = m + 2;
  am.append({d(m), d(k)}); bm.append({d(n), d(k)}); cm.append({d(m), d(n)});
  packedMap.append({d(k), d(n)});
  auto maps = matrix.getIndexingMapsArray();
  if (maps.size() != 3 || maps[0] != AffineMap::get(dims,0,am,ctx) ||
      maps[1] != AffineMap::get(dims,0,bm,ctx) || maps[2] != AffineMap::get(dims,0,cm,ctx)) return false;
  if (at.getDimSize(rank-1) != bt.getDimSize(rank-1) ||
      at.getDimSize(rank-2) != ct.getDimSize(rank-2) ||
      bt.getDimSize(rank-2) != ct.getDimSize(rank-1) ||
      (batched && (at.getDimSize(0) != bt.getDimSize(0) || at.getDimSize(0) != ct.getDimSize(0)))) return false;
  auto canonicalMaps = maps;
  canonicalMaps[1] = AffineMap::get(dims, 0, packedMap, ctx);
  if (transposeOutput(matrix, a, b, c, spec, canonicalMaps)) return true;
  if (!spec.npu.matmulRhsPacking) return false;
  uint64_t bytes = 2;
  for (int64_t dim : bt.getShape()) {
    if (bytes > uint64_t(spec.npu.l1Bytes) / uint64_t(dim)) return false;
    bytes *= dim;
  }
  if (isa<scf::ForOp>(matrix->getParentOp())) bytes *= 2;
  if (!fitsScratch(matrix, bytes, spec)) return false;
  SmallVector<int64_t> shape(bt.getShape()), permutation;
  for (unsigned i = 0; i < rank; ++i) permutation.push_back(i);
  std::swap(shape[rank-2], shape[rank-1]);
  std::swap(permutation[rank-2], permutation[rank-1]);
  OpBuilder builder(matrix);
  auto packed = builder.create<memref::AllocOp>(matrix.getLoc(),
      MemRefType::get(shape, bt.getElementType(), MemRefLayoutAttrInterface{}, bt.getMemorySpace()));
  packed->setAttr("hyb.matmul_rhs_pack", builder.getUnitAttr());
  builder.create<linalg::TransposeOp>(matrix.getLoc(), b, packed.getResult(), permutation);
  // The fresh destination cannot alias any source or output. The transpose
  // stays immediately before its consumer and reads the same RHS version.
  matrix->setOperand(1, packed);
  maps[1] = AffineMap::get(dims, 0, packedMap, ctx);
  matrix->setAttr("indexing_maps", builder.getAffineMapArrayAttr(maps));
  matrix->setAttr("hyb.matmul_rhs_packed", builder.getUnitAttr());
  return true;
}

struct HybNpuMatmulLayoutPass : mlir::hybrid::impl::HybNpuMatmulLayoutBase<HybNpuMatmulLayoutPass> {
  using Base::Base;
  void runOnOperation() override {
    std::string error;
    auto target = TargetSpec::load(spec, error);
    if (failed(target)) { getOperation().emitError(error); return signalPassFailure(); }
    if (!target->npu.matmulRhsPacking && !target->npu.matmulTransposeOutput) return;
    for (auto fn : getOperation().getOps<func::FuncOp>()) {
      auto phase = fn->getAttrOfType<StringAttr>(kPhase);
      if (!phase || phase.getValue() != "npu" || fn->hasAttr("hyb.npu_kernel")) continue;
      int64_t count = 0, transposes = 0;
      SmallVector<linalg::LinalgOp> matrices;
      fn.walk([&](linalg::LinalgOp op) {
        if (isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op.getOperation())) matrices.push_back(op);
      });
      for (auto matrix : matrices) {
        if (!pack(matrix, *target)) continue;
        if (matrix->hasAttr("hyb.matmul_transpose_output")) ++transposes;
        else ++count;
      }
      fn->setAttr("hyb.matmul_rhs_packings", IntegerAttr::get(IntegerType::get(&getContext(),64),count));
      fn->setAttr("hyb.matmul_output_transposes", IntegerAttr::get(IntegerType::get(&getContext(),64),transposes));
    }
  }
};
} // namespace
