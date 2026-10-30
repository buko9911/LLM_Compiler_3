//===- TensorPasses.cpp - [A] tensor-level passes --------------------------===//
//
// hyb-annotate, hyb-simplify, hyb-raise-linear, hyb-quantize, hyb-fold-transpose,
// Linear raising, quantization and transpose normalization.
//===----------------------------------------------------------------------===//

#include "npu/Decode/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

namespace mlir::hybrid {
#define GEN_PASS_DEF_HYBANNOTATE
#define GEN_PASS_DEF_HYBSIMPLIFY
#define GEN_PASS_DEF_HYBRAISELINEAR
#define GEN_PASS_DEF_HYBQUANTIZE
#define GEN_PASS_DEF_HYBFOLDTRANSPOSE
#include "npu/Decode/Transforms/Passes.h.inc"
} // namespace mlir::hybrid

using namespace mlir;
using namespace mlir::hybrid;

namespace {

//===----------------------------------------------------------------------===//
// hyb-annotate
//===----------------------------------------------------------------------===//
struct HybAnnotatePass : mlir::hybrid::impl::HybAnnotateBase<HybAnnotatePass> {
  using Base::Base;
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    auto buffer = llvm::MemoryBuffer::getFile(manifest);
    if (!buffer) {
      module.emitError() << "cannot read manifest '" << manifest << "'";
      return signalPassFailure();
    }
    auto parsed = llvm::json::parse((*buffer)->getBuffer());
    if (!parsed) {
      module.emitError() << "bad manifest JSON: " << llvm::toString(parsed.takeError());
      return signalPassFailure();
    }
    llvm::json::Object *root = parsed->getAsObject();
    func::FuncOp step = getStepFunction(module);
    if (!step) {
      module.emitError("no decode step function");
      return signalPassFailure();
    }
    auto str = [&](StringRef s) { return StringAttr::get(ctx, s); };
    llvm::json::Array *args = root->getArray("arguments");
    llvm::json::Array *results = root->getArray("results");
    if (!args || !results || args->size() != step.getNumArguments() ||
        results->size() != step.getNumResults()) {
      step.emitError("manifest does not match the function signature");
      return signalPassFailure();
    }
    for (auto [i, a] : llvm::enumerate(*args)) {
      llvm::json::Object *o = a.getAsObject();
      step.setArgAttr(i, kKind, str(*o->getString("kind")));
      step.setArgAttr(i, kName, str(*o->getString("name")));
      if (auto layer = o->getInteger("layer"))
        step.setArgAttr(i, "hyb.layer", IntegerAttr::get(IntegerType::get(ctx, 64), *layer));
    }
    for (auto [i, r] : llvm::enumerate(*results)) {
      llvm::json::Object *o = r.getAsObject();
      step.setResultAttr(i, kKind, str("output"));
      step.setResultAttr(i, kName, str(*o->getString("name")));
    }
    module->setAttr("hyb.model", str(*root->getString("model")));
    module->setAttr("hyb.tokens", IntegerAttr::get(IntegerType::get(ctx, 64), *root->getInteger("tokens")));
    module->setAttr("hyb.capacity", IntegerAttr::get(IntegerType::get(ctx, 64), *root->getInteger("capacity")));
    if (auto mm = root->getNumber("mask_min"))
      module->setAttr("hyb.mask_min", FloatAttr::get(Float64Type::get(ctx), *mm));
  }
};

//===----------------------------------------------------------------------===//
// hyb-simplify
//===----------------------------------------------------------------------===//
struct FPowIToMul : OpRewritePattern<math::FPowIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(math::FPowIOp op, PatternRewriter &rewriter) const override {
    APInt e;
    if (!matchPattern(op.getRhs(), m_ConstantInt(&e)) || e != 2) return failure();
    rewriter.replaceOpWithNewOp<arith::MulFOp>(op, op.getLhs(), op.getLhs());
    return success();
  }
};

struct HybSimplifyPass : mlir::hybrid::impl::HybSimplifyBase<HybSimplifyPass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<FPowIToMul>(ctx);
    linalg::populateEraseUnusedOperandsAndResultsPatterns(patterns);
    linalg::ControlDropUnitDims unitDims;
    linalg::populateFoldUnitExtentDimsPatterns(patterns, unitDims);
    // Fuse elementwise producers with a single use into their consumer (including into
    // reductions), so every row-wise chain becomes one generic without recomputing a shared
    // producer; the NPU vectorizer hoists row-invariant sub-expressions out of the column loop.
    linalg::populateElementwiseOpsFusionPatterns(patterns, [](OpOperand *operand) {
      Operation *producer = operand->get().getDefiningOp();
      return producer && producer->hasOneUse();
    });
    for (Dialect *d : ctx->getLoadedDialects()) d->getCanonicalizationPatterns(patterns);
    for (RegisteredOperationName op : ctx->getRegisteredOperations()) op.getCanonicalizationPatterns(patterns, ctx);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) signalPassFailure();
  }
};

//===----------------------------------------------------------------------===//
// hyb-raise-linear
//===----------------------------------------------------------------------===//
static StringAttr weightNameOf(Value v, StringRef wantKind = "weight") {
  auto arg = dyn_cast<BlockArgument>(v);
  if (!arg) return {};
  auto func = dyn_cast<func::FuncOp>(arg.getOwner()->getParentOp());
  if (!func) return {};
  auto kind = func.getArgAttrOfType<StringAttr>(arg.getArgNumber(), kKind);
  if (!kind || kind.getValue() != wantKind) return {};
  return func.getArgAttrOfType<StringAttr>(arg.getArgNumber(), kName);
}

static bool isZeroFill(Value init) {
  auto fill = init.getDefiningOp<linalg::FillOp>();
  if (!fill) return false;
  FloatAttr f;
  return matchPattern(fill.getInputs()[0], m_Constant(&f)) && f.getValue().isZero();
}

struct RaiseLinear : OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::MatmulOp mm, PatternRewriter &rewriter) const override {
    if (mm.hasUserDefinedMaps() || mm.getNumResults() != 1) return failure();
    Value x = mm.getDpsInputs()[0], w = mm.getDpsInputs()[1];
    auto tr = w.getDefiningOp<linalg::TransposeOp>();
    if (!tr || tr.getPermutation() != ArrayRef<int64_t>({1, 0})) return failure();
    StringAttr name = weightNameOf(tr.getInput());
    if (!name || !isZeroFill(mm.getDpsInits()[0])) return failure();
    rewriter.replaceOpWithNewOp<hyb::LinearOp>(mm, mm.getResult(0).getType(), x, name);
    return success();
  }
};

struct HybRaiseLinearPass : mlir::hybrid::impl::HybRaiseLinearBase<HybRaiseLinearPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<RaiseLinear>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) signalPassFailure();
  }
};

//===----------------------------------------------------------------------===//
// hyb-quantize
//===----------------------------------------------------------------------===//
struct QuantizeLinear : OpRewritePattern<hyb::LinearOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(hyb::LinearOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto xT = cast<RankedTensorType>(op.getInput().getType());
    auto yT = cast<RankedTensorType>(op.getResult().getType());
    if (xT.getRank() != 2) return failure();
    int64_t M = xT.getDimSize(0), K = xT.getDimSize(1), N = yT.getDimSize(1);
    Type i8 = rewriter.getI8Type(), i32 = rewriter.getI32Type(), f32 = rewriter.getF32Type();
    auto qT = RankedTensorType::get({M, K}, i8);
    auto sT = RankedTensorType::get({M, 1}, f32);
    auto accT = RankedTensorType::get({M, N}, i32);
    Value qInit = tensor::EmptyOp::create(rewriter, loc, qT.getShape(), i8);
    Value sInit = tensor::EmptyOp::create(rewriter, loc, sT.getShape(), f32);
    auto quant = hyb::QuantOp::create(rewriter, loc, TypeRange{qT, sT}, op.getInput(), qInit, sInit);
    Value accInit = tensor::EmptyOp::create(rewriter, loc, accT.getShape(), i32);
    auto qmm = hyb::QMatmulOp::create(rewriter, loc, TypeRange{accT}, quant.getResult(0), accInit,
                                      op.getWeightAttr(), IntegerAttr(), StringAttr());
    Value yInit = tensor::EmptyOp::create(rewriter, loc, yT.getShape(), yT.getElementType());
    Value wscale = hyb::WeightOp::create(rewriter, loc, RankedTensorType::get({N}, f32), op.getWeightAttr(),
                                         rewriter.getStringAttr("row_scale"));
    auto deq = hyb::DequantOp::create(rewriter, loc, TypeRange{yT}, qmm.getResult(0), quant.getResult(1), wscale,
                                      yInit, op.getWeightAttr());
    rewriter.replaceOp(op, deq.getResult(0));
    return success();
  }
};

struct HybQuantizePass : mlir::hybrid::impl::HybQuantizeBase<HybQuantizePass> {
  using Base::Base;
  void runOnOperation() override {
    if (scheme != "w8a8-row") {
      getOperation().emitError() << "unsupported quantization scheme " << scheme;
      return signalPassFailure();
    }
    getOperation()->getParentOfType<ModuleOp>()->setAttr("hyb.quant", StringAttr::get(&getContext(), scheme));
    RewritePatternSet patterns(&getContext());
    patterns.add<QuantizeLinear>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) signalPassFailure();
  }
};

//===----------------------------------------------------------------------===//
// hyb-fold-transpose: batch_matmul(A, transpose(B, [0, 2, 1])) -> B read transposed.
//===----------------------------------------------------------------------===//
struct FoldTransposeIntoBatchMatmul : OpRewritePattern<linalg::BatchMatmulOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::BatchMatmulOp op, PatternRewriter &rewriter) const override {
    if (op.hasUserDefinedMaps()) return failure();
    auto tr = op.getDpsInputs()[1].getDefiningOp<linalg::TransposeOp>();
    if (!tr || tr.getPermutation() != ArrayRef<int64_t>({0, 2, 1})) return failure();
    MLIRContext *ctx = getContext();
    AffineExpr b, m, n, k;
    bindDims(ctx, b, m, n, k);
    SmallVector<AffineMap> maps = {AffineMap::get(4, 0, {b, m, k}, ctx), AffineMap::get(4, 0, {b, n, k}, ctx),
                                   AffineMap::get(4, 0, {b, m, n}, ctx)};
    auto fused = linalg::BatchMatmulOp::create(rewriter, op.getLoc(), op.getResultTypes(),
                                               ValueRange{op.getDpsInputs()[0], tr.getInput()}, op.getDpsInits());
    fused->setAttr("indexing_maps", rewriter.getAffineMapArrayAttr(maps));
    rewriter.replaceOp(op, fused->getResults());
    return success();
  }
};

struct HybFoldTransposePass : mlir::hybrid::impl::HybFoldTransposeBase<HybFoldTransposePass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldTransposeIntoBatchMatmul>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) signalPassFailure();
  }
};

} // namespace
