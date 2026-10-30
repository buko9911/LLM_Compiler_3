//===- NpuInputPipeline.cpp - hyb-npu-input-pipeline ----------------------===//
//
// Split a private L2 -> L1 input copy and the FP16 matmul that consumes it into batch / N tiles
// over disjoint slices of the same allocation (full K per tile), so the next tile's copy can
// overlap the current tile's compute. No extra SRAM; the backend's events enforce readiness.
//===----------------------------------------------------------------------===//

#include "npu/Decode/Transforms/PassSupport.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"

namespace mlir::hybrid {
#define GEN_PASS_DEF_HYBNPUINPUTPIPELINE
#include "npu/Decode/Transforms/Passes.h.inc"
}
using namespace mlir;
using namespace mlir::hybrid;

namespace {
bool staticF16(Value value, unsigned rank, unsigned space) {
  auto type = dyn_cast<MemRefType>(value.getType());
  SmallVector<int64_t> strides;
  int64_t offset;
  return type && type.hasStaticShape() && type.getRank() == rank &&
         type.getElementType().isF16() && type.getMemorySpaceAsInt() == space &&
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
        !isa<FloatAttr>(value) || !cast<FloatAttr>(value).getValue().isZero()) return {};
    init = fill;
  }
  return init;
}

struct Tile { Value source, weight, activation, output; };

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
      !staticF16(a, rank, 1) || !staticF16(b, rank, 1) || !staticF16(c, rank, 1) ||
      !staticF16(copy.getSource(), rank, 2) || !canonicalMaps(matrix, batched)) return 0;
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
  SmallVector<Tile> tiles;
  for (int64_t batch = 0; batch < batches; ++batch) {
    Value activation = batched ? slice(builder, copy.getLoc(), a, true, batch, 0,
                                       cast<MemRefType>(a.getType()).getShape().back()) : a;
    for (int64_t column = 0; column < n; column += tileColumns) {
      int64_t size = std::min(tileColumns, n - column);
      tiles.push_back({slice(builder, copy.getLoc(), copy.getSource(), batched, batch, column, size),
                       slice(builder, copy.getLoc(), b, batched, batch, column, size), activation,
                       slice(builder, copy.getLoc(), c, batched, batch, column, size)});
    }
  }
  auto emitCopy = [&](const Tile &tile) {
    auto part = memref::CopyOp::create(builder, copy.getLoc(), tile.source, tile.weight);
    part->setAttr("hyb.input_pipeline_copy", builder.getUnitAttr());
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
  return count;
}

struct HybNpuInputPipelinePass
    : mlir::hybrid::impl::HybNpuInputPipelineBase<HybNpuInputPipelinePass> {
  using Base::Base;
  void runOnOperation() override {
    auto target = loadTargetSpec(getOperation(), spec);
    if (failed(target)) return signalPassFailure();
    if (!target->npu.inputTilePipeline) return;
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
