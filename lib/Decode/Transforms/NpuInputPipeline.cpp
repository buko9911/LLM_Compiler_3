//===- NpuInputPipeline.cpp - hyb-npu-input-pipeline ----------------------===//
//
// Split a private L2 -> L1 input copy and the FP16 matmul that consumes it into batch / N tiles
// (full K per tile), so the next tile's copy can overlap the current tile's compute. Batch tiles
// are disjoint contiguous slices of the same allocation. Column tiles of a matrix with several
// rows are interleaved rows of it: an access is tracked by its address span, so the next tile's
// copy would overlap the current tile's operand load and wait for it. Those tiles go through two
// contiguous L1 buffers instead (ping-pong), which also replace the whole B buffer.
//===----------------------------------------------------------------------===//

#include "npu/Decode/Transforms/PassSupport.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include <functional>

namespace mlir::hybrid {
#define GEN_PASS_DEF_HYBNPUINPUTPIPELINE
#include "npu/Decode/Transforms/Passes.h.inc"
}
using namespace mlir;
using namespace mlir::hybrid;

namespace {
/// The element type of a matrix operand: FP16, or --kv8's INT8 operands and INT32 result.
bool staticElements(Value value, unsigned rank, unsigned space, unsigned intBits = 0) {
  auto type = dyn_cast<MemRefType>(value.getType());
  SmallVector<int64_t> strides;
  int64_t offset;
  return type && type.hasStaticShape() && type.getRank() == rank &&
         (intBits ? type.getElementType().isInteger(intBits) : type.getElementType().isF16()) && type.getMemorySpaceAsInt() == space &&
         llvm::all_of(type.getShape(), [](int64_t dim) { return dim > 0; }) &&
         succeeded(type.getStridesAndOffset(strides, offset)) &&
         llvm::all_of(strides, [](int64_t stride) { return stride > 0; }) &&
         strides.back() == 1;
}

bool canonicalMaps(linalg::LinalgOp op, bool batched) {
  MLIRContext *ctx = op.getContext();
  unsigned dims = batched ? 4 : 3;
  auto d = [&](unsigned i) { return getAffineDimExpr(i, ctx); };
  SmallVector<AffineMap> expected;
  if (batched) {
    expected = {AffineMap::get(dims, 0, {d(0), d(1), d(3)}, ctx),
                AffineMap::get(dims, 0, {d(0), d(3), d(2)}, ctx),
                AffineMap::get(dims, 0, {d(0), d(1), d(2)}, ctx)};
  } else {
    expected = {AffineMap::get(dims, 0, {d(0), d(2)}, ctx),
                AffineMap::get(dims, 0, {d(2), d(1)}, ctx),
                AffineMap::get(dims, 0, {d(0), d(1)}, ctx)};
  }
  return op.getIndexingMapsArray() == expected;
}

Value slice(OpBuilder &builder, Location loc, Value value, bool batched,
            int64_t batch, int64_t column, int64_t columns) {
  auto type = cast<MemRefType>(value.getType());
  SmallVector<OpFoldResult> offsets, sizes, strides;
  for (int64_t size : type.getShape()) {
    offsets.push_back(builder.getIndexAttr(0));
    sizes.push_back(builder.getIndexAttr(size));
    strides.push_back(builder.getIndexAttr(1));
  }
  if (batched) {
    offsets[0] = builder.getIndexAttr(batch);
    sizes[0] = builder.getIndexAttr(1);
  }
  offsets.back() = builder.getIndexAttr(column);
  sizes.back() = builder.getIndexAttr(columns);
  return memref::SubViewOp::create(builder, loc, value, offsets, sizes, strides);
}

// Preserve zero-initialization as one logical fill per output tile. The backend
// proves INIT coverage from these fills; an earlier tile's write must not break
// that proof. Reject output views/reads/escapes before the original matrix.
linalg::FillOp privateZeroInit(Value output, Operation *matrix) {
  if (!output.getDefiningOp<memref::AllocOp>()) return {};
  linalg::FillOp init;
  for (Operation *user : output.getUsers()) {
    if (user == matrix) continue;
    if (user->getBlock() != matrix->getBlock()) return {};
    if (matrix->isBeforeInBlock(user)) continue;
    auto fill = dyn_cast<linalg::FillOp>(user);
    Attribute value;
    if (!fill || init || fill.getOutputs()[0] != output ||
        !matchPattern(fill.getInputs()[0], m_Constant(&value)) ||
        !((isa<FloatAttr>(value) && cast<FloatAttr>(value).getValue().isZero()) ||
          (isa<IntegerAttr>(value) && cast<IntegerAttr>(value).getValue().isZero()))) return {};
    init = fill;
  }
  return init;
}

struct Tile { Value source, weight, activation, output; };

/// Moves the first tile's copy (with the view and allocation ops it needs) above the pure compute
/// ops right before it that touch neither its source nor its destination, so the asynchronous
/// L2 -> L1 load overlaps them (e.g. V under the softmax). It stops at anything else: another copy
/// or matrix op (DMA and SA hazards), a region op, or an op on the same storage. The destination
/// lives longer; hyb-npu-memplan places it against the L1 capacity.
void hoistFirstCopy(memref::CopyOp copy) {
  Value sourceRoot = viewRoot(copy.getSource()), targetRoot = viewRoot(copy.getTarget());
  auto touches = [&](Operation *op) {
    for (Value v : op->getOperands())
      if (isa<MemRefType>(v.getType()) && (viewRoot(v) == sourceRoot || viewRoot(v) == targetRoot)) return true;
    return false;
  };
  // The copy's operand definitions in this block, in order (views and allocations only).
  SmallVector<Operation *> defs;
  std::function<bool(Value)> collect = [&](Value v) -> bool {
    Operation *d = v.getDefiningOp();
    if (!d || d->getBlock() != copy->getBlock()) return true;
    if (!isa<memref::SubViewOp, memref::AllocOp, memref::ExpandShapeOp, memref::CollapseShapeOp, arith::ConstantOp,
             affine::AffineApplyOp>(d))
      return false;
    for (Value o : d->getOperands())
      if (!collect(o)) return false;
    if (!llvm::is_contained(defs, d)) defs.push_back(d);
    return true;
  };
  for (Value v : copy->getOperands())
    if (!collect(v)) return;
  Operation *at = copy.getOperation();
  for (Operation *prev = copy->getPrevNode(); prev; prev = prev->getPrevNode()) {
    if (llvm::is_contained(defs, prev)) continue;
    if (isa<memref::SubViewOp, memref::AllocOp, memref::ExpandShapeOp, memref::CollapseShapeOp, arith::ConstantOp,
            affine::AffineApplyOp>(prev))
      continue;                                    // other views and index math: no effect, pass over them
    if (!isa<linalg::GenericOp, linalg::FillOp>(prev) || prev->getNumRegions() > 1 || touches(prev)) break;
    at = prev;
  }
  if (at == copy.getOperation()) return;
  llvm::sort(defs, [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
  for (Operation *d : defs)
    if (at->isBeforeInBlock(d)) d->moveBefore(at);
  copy->moveBefore(at);
}

// Only an immediately consumed private copy can be moved. A/B/C allocation
// roots must be distinct; later placement then protects their physical ranges.
// Rank, positive strides and complete indexing maps are proved before editing.
int64_t pipeline(memref::CopyOp copy, int64_t tileColumns) {
  auto matrix = dyn_cast_or_null<linalg::LinalgOp>(copy->getNextNode());
  if (!matrix || !isa<linalg::MatmulOp, linalg::BatchMatmulOp>(matrix.getOperation()) ||
      matrix->getNumResults() || matrix.getNumDpsInputs() != 2 || matrix.getNumDpsInits() != 1)
    return 0;
  bool batched = isa<linalg::BatchMatmulOp>(matrix.getOperation());
  unsigned rank = batched ? 3 : 2;
  Value a = matrix.getDpsInputs()[0], b = matrix.getDpsInputs()[1], c = matrix.getDpsInits()[0];
  if (copy.getTarget() != b || !b.getDefiningOp<memref::AllocOp>() ||
      !canonicalMaps(matrix, batched)) return 0;
  const bool int8 = cast<MemRefType>(b.getType()).getElementType().isInteger(8);
  if (!staticElements(a, rank, 1, int8 ? 8 : 0) || !staticElements(b, rank, 1, int8 ? 8 : 0) || !staticElements(c, rank, 1, int8 ? 32 : 0) ||
      !staticElements(copy.getSource(), rank, 2, int8 ? 8 : 0)) return 0;
  if (llvm::any_of(b.getUsers(), [&](Operation *user) {
        return user != copy.getOperation() && user != matrix.getOperation();
      })) return 0;
  Value ar = viewRoot(a), cr = viewRoot(c);
  if (!ar.getDefiningOp<memref::AllocOp>() || !cr.getDefiningOp<memref::AllocOp>() ||
      ar == b || cr == b || ar == cr) return 0;
  // Column/batch tiles must be physically disjoint. Explicit overlapping or
  // exotic allocation layouts cannot establish that with index maps alone.
  if (!cast<MemRefType>(b.getType()).getLayout().isIdentity() ||
      !cast<MemRefType>(c.getType()).getLayout().isIdentity()) return 0;
  auto zeroInit = privateZeroInit(c, matrix.getOperation());
  if (!zeroInit) return 0;
  auto bt = cast<MemRefType>(b.getType());
  int64_t batches = batched ? bt.getDimSize(0) : 1;
  int64_t n = bt.getShape().back();
  int64_t tilesPerBatch = (n + tileColumns - 1) / tileColumns;
  // Bound generated code growth; an unsupported large shape retains its copy.
  if (batches * tilesPerBatch < 2 || batches * tilesPerBatch > 256) return 0;
  OpBuilder builder(copy);
  const int64_t rows = bt.getDimSize(rank - 2);
  const bool pingPong = tilesPerBatch > 1 && rows > 1;
  SmallVector<Value, 2> buffers;
  if (pingPong) {
    SmallVector<int64_t> shape(bt.getShape());
    if (batched) shape[0] = 1;
    shape.back() = std::min(tileColumns, n);
    auto type = MemRefType::get(shape, bt.getElementType(), MemRefLayoutAttrInterface{}, bt.getMemorySpace());
    for (int i = 0; i < 2; ++i) buffers.push_back(memref::AllocOp::create(builder, copy.getLoc(), type));
  }
  SmallVector<Tile> tiles;
  for (int64_t batch = 0; batch < batches; ++batch) {
    Value activation = batched ? slice(builder, copy.getLoc(), a, true, batch, 0,
                                       cast<MemRefType>(a.getType()).getShape().back()) : a;
    for (int64_t column = 0; column < n; column += tileColumns) {
      int64_t size = std::min(tileColumns, n - column);
      Value weight;
      if (pingPong) {
        Value buffer = buffers[tiles.size() % 2];
        weight = size == cast<MemRefType>(buffer.getType()).getShape().back()
                     ? buffer : slice(builder, copy.getLoc(), buffer, batched, 0, 0, size);
      } else {
        weight = slice(builder, copy.getLoc(), b, batched, batch, column, size);
      }
      tiles.push_back({slice(builder, copy.getLoc(), copy.getSource(), batched, batch, column, size), weight,
                       activation, slice(builder, copy.getLoc(), c, batched, batch, column, size)});
    }
  }
  memref::CopyOp first;
  auto emitCopy = [&](const Tile &tile) {
    auto part = memref::CopyOp::create(builder, copy.getLoc(), tile.source, tile.weight);
    part->setAttr("hyb.input_pipeline_copy", builder.getUnitAttr());
    if (!first) first = part;
  };
  emitCopy(tiles[0]);
  for (size_t i = 0; i < tiles.size(); ++i) {
    // The common input event waits for copy i before issuing copy i+1.
    // Matrix i reads a disjoint part of B while that next copy is outstanding.
    if (i + 1 < tiles.size()) emitCopy(tiles[i + 1]);
    linalg::FillOp::create(builder, matrix.getLoc(), zeroInit.getInputs(), ValueRange{tiles[i].output});
    IRMapping mapping;
    mapping.map(a, tiles[i].activation);
    mapping.map(b, tiles[i].weight);
    mapping.map(c, tiles[i].output);
    Operation *part = builder.clone(*matrix.getOperation(), mapping);
    part->setAttr("hyb.input_pipeline_tile", builder.getUnitAttr());
  }
  int64_t count = tiles.size();
  matrix.erase();
  copy.erase();
  zeroInit.erase();
  if (pingPong && b.use_empty()) b.getDefiningOp()->erase();
  hoistFirstCopy(first);
  return count;
}

struct HybNpuInputPipelinePass
    : mlir::hybrid::impl::HybNpuInputPipelineBase<HybNpuInputPipelinePass> {
  using Base::Base;
  void runOnOperation() override {
    auto target = loadTargetSpec(getOperation(), spec);
    if (failed(target)) return signalPassFailure();
    for (func::FuncOp fn : getNpuPhases(getOperation())) {
      SmallVector<memref::CopyOp> copies;
      fn.walk([&](memref::CopyOp copy) { copies.push_back(copy); });
      int64_t groups = 0, tiles = 0;
      for (auto copy : copies) {
        int64_t count = pipeline(copy, target->npu.inputTileColumns);
        if (count) { ++groups; tiles += count; }
      }
      auto i64 = IntegerType::get(&getContext(), 64);
      fn->setAttr("hyb.input_pipeline_groups", IntegerAttr::get(i64, groups));
      fn->setAttr("hyb.input_pipeline_tiles", IntegerAttr::get(i64, tiles));
    }
  }
};
} // namespace
