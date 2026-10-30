// Elide physical zero stores only after proving the matrix INIT owns the value.
#include "npu/Decode/Support/TargetSpec.h"
#include "npu/Decode/Transforms/Passes.h"
#include "mlir/IR/Matchers.h"

namespace mlir::hybrid {
#define GEN_PASS_DEF_HYBNPUMATRIXINIT
#include "npu/Decode/Transforms/Passes.h.inc"
}
using namespace mlir;
using namespace mlir::hybrid;

namespace {
bool canElide(linalg::FillOp fill) {
  Value out = fill.getOutputs()[0];
  auto alloc = out.getDefiningOp<memref::AllocOp>();
  if (!alloc || alloc->getBlock() != fill->getBlock() || !alloc.getType().hasStaticShape() ||
      alloc.getType().getMemorySpaceAsInt() != 1) return false;
  Attribute value;
  if (!matchPattern(fill.getInputs()[0], m_Constant(&value))) return false;
  bool zero = (isa<IntegerAttr>(value) && cast<IntegerAttr>(value).getValue().isZero()) ||
              (isa<FloatAttr>(value) && cast<FloatAttr>(value).getValue().isZero());
  if (!zero) return false;
  linalg::LinalgOp init;
  for (Operation *user : out.getUsers()) {
    auto matrix = dyn_cast<linalg::LinalgOp>(user);
    if (!matrix || (!isa<linalg::MatmulOp, linalg::BatchMatmulOp>(user) && !user->hasAttr("hyb.npu_gemv"))) continue;
    if (matrix.getNumDpsInits() != 1 || matrix.getDpsInits()[0] != out ||
        user->getBlock() != fill->getBlock() || !fill->isBeforeInBlock(user) ||
        llvm::is_contained(matrix.getDpsInputs(), out)) continue;
    // The supported matrix indexing must cover every output element. A view
    // as the destination would not equal the allocated `out` above.
    auto map = matrix.getMatchingIndexingMap(&matrix.getDpsInitsMutable()[0]);
    if (!map.isProjectedPermutation() || map.getNumResults() != alloc.getType().getRank()) continue;
    auto ranges = matrix.getStaticLoopRanges();
    bool covers = ranges.size() == map.getNumDims();
    for (auto [i, expr] : llvm::enumerate(map.getResults())) {
      auto dim = dyn_cast<AffineDimExpr>(expr);
      if (!dim || ranges[dim.getPosition()] != alloc.getType().getDimSize(i)) covers = false;
    }
    if (!covers || llvm::any_of(ranges, [](int64_t n) { return n <= 0; })) continue;
    if (!init || user->isBeforeInBlock(init)) init = matrix;
  }
  if (!init) return false;
  for (Operation *user : out.getUsers()) {
    if (user == fill || user == init) continue;
    // Even a view before INIT is rejected: its users might read the zero or
    // escape the allocation. Views created after INIT can only see its result.
    if (user->getBlock() != fill->getBlock() || !init->isBeforeInBlock(user)) return false;
  }
  return true;
}

struct HybNpuMatrixInitPass : mlir::hybrid::impl::HybNpuMatrixInitBase<HybNpuMatrixInitPass> {
  using Base::Base;
  void runOnOperation() override {
    std::string error;
    auto target = TargetSpec::load(spec, error);
    if (failed(target)) { getOperation().emitError(error); return signalPassFailure(); }
    if (!target->npu.matrixInitElision) return;
    for (auto fn : getOperation().getOps<func::FuncOp>()) {
      auto phase = fn->getAttrOfType<StringAttr>(kPhase);
      if (!phase || phase.getValue() != "npu" || fn->hasAttr("hyb.npu_kernel")) continue;
      int64_t count = 0;
      fn.walk([&](linalg::FillOp fill) {
        if (canElide(fill)) {
          fill->setAttr("hyb.elide_matrix_init", UnitAttr::get(&getContext()));
          ++count;
        }
      });
      fn->setAttr("hyb.matrix_init_elisions", IntegerAttr::get(IntegerType::get(&getContext(), 64), count));
    }
  }
};
}
