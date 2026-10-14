//===----------------------------------------------------------------------===//
// HardwareOps.cpp — Stage 4 — placed memref IR to NPU hardware ops
//
// Stage 3 leaves linalg ops on placed buffers. Their meaning is the generic one:
// linalg.matmul reads C from memory, adds A·B and writes C back. The hardware
// does something narrower, and this stage writes that down as ops:
//
//   linalg.matmul on an L1 tile      npu.acc_zero / npu.mma / npu.acc_writeout
//                                    (the accumulator is a value, !npu.acc)
//   K-chunk loop of such matmuls     the loop carries the accumulator; one
//                                    writeout after the loop
//   copy with a side in DRAM         npu.global_copy   (central GDMA)
//   copy between L2 and L1           npu.local_copy    (the core's DMA)
//   scf.for {npu.core_split}         scf.forall {mapping = [#npu.core]}
//
// Loops, buffers and addresses stay as they are; Stage 5 (npu-emit) unrolls them
// into instructions. Nothing here changes which instructions come out: the
// writeout is placed exactly where emission used to write the accumulator out.
//===----------------------------------------------------------------------===//
#include "GraphInternal.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
namespace npu {
namespace {
using mlir::npu::AccType;
using mlir::npu::AccZeroOp;
using mlir::npu::MmaOp;
using mlir::npu::AccWriteoutOp;
using mlir::npu::GlobalCopyOp;
using mlir::npu::LocalCopyOp;
using mlir::npu::CoreMappingAttr;

// ── Copies ───────────────────────────────────────────────────────────────────
// Which engine moves the bytes follows from where they are: a DRAM side needs
// the central GDMA, anything else is the core's own DMA. The hints Stage 2
// attached to linalg.copy become declared attributes of the new op.
template <typename CopyOp>
struct CopyToHardware : OpConversionPattern<CopyOp> {
  using OpConversionPattern<CopyOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(CopyOp op, typename CopyOp::Adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value source, target;
    if constexpr (std::is_same_v<CopyOp, linalg::CopyOp>) {
      if (!op.hasPureBufferSemantics()) return failure();
      source = op.getInputs()[0]; target = op.getOutputs()[0];
    } else {
      source = op.getSource(); target = op.getTarget();
    }
    if (mlir::npu::memorySpace(source.getType()) == 0 || mlir::npu::memorySpace(target.getType()) == 0) {
      rewriter.replaceOpWithNewOp<GlobalCopyOp>(op, source, target);
      return success();
    }
    auto local = rewriter.replaceOpWithNewOp<LocalCopyOp>(op, source, target);
    if (op->hasAttr("npu.prefetch_weight")) local.setPrefetch(true);
    if (op->hasAttr("npu.cache_activation")) local.setCacheActivation(true);
    return success();
  }
};

LogicalResult convertCopies(ModuleOp m) {
  ConversionTarget target(*m.getContext());
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  target.addIllegalOp<linalg::CopyOp, memref::CopyOp>();
  RewritePatternSet patterns(m.getContext());
  patterns.add<CopyToHardware<linalg::CopyOp>, CopyToHardware<memref::CopyOp>>(m.getContext());
  return applyPartialConversion(m, target, std::move(patterns));
}

// ── Matmuls ──────────────────────────────────────────────────────────────────
bool isCopy(Operation *op) { return isa<linalg::CopyOp, memref::CopyOp, linalg::TransposeOp>(op); }

// Ops that emit nothing or only move bytes that are not the accumulator's
// tile. Emission writes the accumulator out at the first op that is not one of
// these, so the writeout goes right before it.
bool keepsAccumulatorLive(Operation *op, Value tile) {
  if (isa<memref::AllocOp, memref::SubViewOp, memref::CollapseShapeOp, memref::ExpandShapeOp,
          memref::CastOp, arith::ConstantOp, arith::AddIOp, arith::SubIOp, arith::MulIOp,
          arith::MinSIOp, affine::AffineApplyOp, affine::AffineMinOp>(op))
    return true;
  if (isCopy(op)) return !llvm::is_contained(op->getOperands(), tile);
  return false;
}

// K-chunk loop: the matmul is the only user of its L1 output inside a plain
// scf.for, and the output lives outside the loop. The loop then accumulates
// one tile over all of its iterations.
scf::ForOp accumulatingLoop(linalg::MatmulOp mm, Value tile) {
  auto loop = dyn_cast<scf::ForOp>(mm->getParentOp());
  if (!loop || loop.getNumRegionIterArgs() || loop->hasAttr("npu.core_split") ||
      !mm->hasAttr("npu.accumulate") || !loop.isDefinedOutsideOfLoop(tile))
    return {};
  for (Operation *user : tile.getUsers())
    if (user != mm && loop->isAncestor(user)) return {};
  return loop;
}

LogicalResult convertMatmul(IRRewriter &rewriter, linalg::MatmulOp mm) {
  if (!mm.hasPureBufferSemantics()) return mm.emitError("hardware ops need placed buffers");
  Value a = mm.getDpsInputs()[0], w = mm.getDpsInputs()[1], tile = mm.getDpsInits()[0];
  auto tileType = dyn_cast<MemRefType>(tile.getType());
  if (!tileType || tileType.getRank() != 2 || !tileType.hasStaticShape())
    return mm.emitError("a matmul output must be a static rank-2 tile");
  auto accType = AccType::get(mm.getContext(), tileType.getDimSize(0), tileType.getDimSize(1));
  Location loc = mm.getLoc();

  Operation *chainStart, *chainEnd;
  Value acc;
  if (auto loop = accumulatingLoop(mm, tile)) {
    rewriter.setInsertionPoint(loop);
    auto zero = AccZeroOp::create(rewriter, loc, accType);
    auto carried = scf::ForOp::create(rewriter, loop.getLoc(), loop.getLowerBound(),
                                      loop.getUpperBound(), loop.getStep(), ValueRange{zero});
    carried->setDiscardableAttrs(loop->getDiscardableAttrDictionary());
    Block *from = loop.getBody(), *into = carried.getBody();
    into->getOperations().splice(into->end(), from->getOperations(), from->begin(),
                                 std::prev(from->end()));
    rewriter.replaceAllUsesWith(loop.getInductionVar(), carried.getInductionVar());
    rewriter.setInsertionPoint(mm);
    auto mma = MmaOp::create(rewriter, loc, accType, carried.getRegionIterArgs()[0], a, w);
    rewriter.setInsertionPointToEnd(into);
    scf::YieldOp::create(rewriter, loc, mma.getResult());
    rewriter.eraseOp(mm);
    rewriter.eraseOp(loop);
    chainStart = zero; chainEnd = carried; acc = carried.getResult(0);
  } else {
    rewriter.setInsertionPoint(mm);
    auto zero = AccZeroOp::create(rewriter, loc, accType);
    auto mma = MmaOp::create(rewriter, loc, accType, zero, a, w);
    rewriter.eraseOp(mm);
    chainStart = zero; chainEnd = mma; acc = mma.getResult();
  }

  // The tile's zero fill is what the empty accumulator now says.
  for (Operation *user : llvm::make_early_inc_range(tile.getUsers()))
    if (auto fill = dyn_cast<linalg::FillOp>(user);
        fill && fill->hasAttr("npu.tile_zero") && fill->getBlock() == chainStart->getBlock() &&
        fill->isBeforeInBlock(chainStart))
      rewriter.eraseOp(fill);

  Operation *at = chainEnd->getNextNode();
  while (at && !at->hasTrait<OpTrait::IsTerminator>() && keepsAccumulatorLive(at, tile))
    at = at->getNextNode();
  if (!at) return mm.emitError("no place to write the accumulator out");
  rewriter.setInsertionPoint(at);
  AccWriteoutOp::create(rewriter, loc, acc, tile);
  return success();
}

// ── Core distribution ────────────────────────────────────────────────────────
// scf.for promises an order between iterations; the loop Stage 2 marked for
// core distribution promises the opposite. scf.forall says that, and its
// mapping names the processing unit.
void convertCoreSplit(IRRewriter &rewriter, scf::ForOp loop) {
  rewriter.setInsertionPoint(loop);
  auto mapping = rewriter.getArrayAttr({CoreMappingAttr::get(loop.getContext())});
  auto forall = scf::ForallOp::create(
      rewriter, loop.getLoc(), ArrayRef<OpFoldResult>{getAsOpFoldResult(loop.getLowerBound())},
      ArrayRef<OpFoldResult>{getAsOpFoldResult(loop.getUpperBound())},
      ArrayRef<OpFoldResult>{getAsOpFoldResult(loop.getStep())}, ValueRange{}, mapping);
  Block *from = loop.getBody(), *into = forall.getBody();
  into->getOperations().splice(std::prev(into->end()), from->getOperations(), from->begin(),
                               std::prev(from->end()));
  rewriter.replaceAllUsesWith(loop.getInductionVar(), forall.getInductionVar(0));
  rewriter.eraseOp(loop);
}
} // namespace

LogicalResult lowerToHardwareOps(ModuleOp m) {
  // npu.attention becomes hardware ops directly, in its pipelined order.
  if (failed(expandAttention(m))) return failure();
  IRRewriter rewriter(m.getContext());
  SmallVector<linalg::MatmulOp> matmuls;
  m.walk([&](linalg::MatmulOp mm) { matmuls.push_back(mm); });
  for (auto mm : matmuls)
    if (failed(convertMatmul(rewriter, mm))) return failure();

  if (failed(convertCopies(m))) return failure();

  SmallVector<scf::ForOp> splits;
  m.walk([&](scf::ForOp loop) {
    if (loop->hasAttr("npu.core_split")) splits.push_back(loop);
  });
  for (auto loop : splits) {
    if (loop.getNumRegionIterArgs()) return loop.emitError("a core loop cannot carry values");
    convertCoreSplit(rewriter, loop);
  }
  m->setAttr("npu.stage", StringAttr::get(m.getContext(), "hw"));
  return success();
}
} // namespace npu
