#include "npu/Decode/Dialect/Hyb/HybDialect.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::hyb;

#include "npu/Decode/Dialect/Hyb/HybOpsDialect.cpp.inc"

void HybDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "npu/Decode/Dialect/Hyb/HybOps.cpp.inc"
      >();
}

// Destination-style ops read their inputs and write their inits, like linalg ops.
template <typename OpT>
static void dpsEffects(OpT op, SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  auto dps = cast<DestinationStyleOpInterface>(op.getOperation());
  for (OpOperand *operand : dps.getDpsInputOperands())
    if (isa<MemRefType>(operand->get().getType()))
      effects.emplace_back(MemoryEffects::Read::get(), operand, SideEffects::DefaultResource::get());
  for (OpOperand &operand : dps.getDpsInitsMutable())
    if (isa<MemRefType>(operand.get().getType())) {
      effects.emplace_back(MemoryEffects::Read::get(), &operand, SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), &operand, SideEffects::DefaultResource::get());
    }
}

MutableOperandRange QuantOp::getDpsInitsMutable() { return MutableOperandRange(*this, 1, 2); }
MutableOperandRange QMatmulOp::getDpsInitsMutable() { return getInitMutable(); }
MutableOperandRange DequantOp::getDpsInitsMutable() { return getInitMutable(); }

void QuantOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &e) { dpsEffects(*this, e); }
void QMatmulOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &e) { dpsEffects(*this, e); }
void DequantOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &e) { dpsEffects(*this, e); }

namespace {
/// Bufferization of the destination-style hyb ops: every operand becomes its buffer, the op is
/// recreated without results, the init buffers replace the results.
template <typename OpT>
struct DpsBufferization : bufferization::DstBufferizableOpInterfaceExternalModel<DpsBufferization<OpT>, OpT> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter, const bufferization::BufferizationOptions &options,
                          bufferization::BufferizationState &state) const {
    auto dps = cast<DestinationStyleOpInterface>(op);
    if (dps.hasPureBufferSemantics()) return success();
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    SmallVector<Value> operands;
    for (OpOperand &o : op->getOpOperands()) {
      FailureOr<Value> buffer = bufferization::getBuffer(rewriter, o.get(), options, state);
      if (failed(buffer)) return failure();
      operands.push_back(*buffer);
    }
    SmallVector<Value> outs;
    for (OpOperand &init : dps.getDpsInitsMutable()) outs.push_back(operands[init.getOperandNumber()]);
    OperationState st(op->getLoc(), op->getName(), operands, TypeRange{}, op->getAttrs());
    rewriter.create(st);
    bufferization::replaceOpWithBufferizedValues(rewriter, op, outs);
    return success();
  }
};
/// hyb.weight is a read-only LP6 tensor: it bufferizes to the same op with a memref result.
struct WeightBufferization
    : bufferization::BufferizableOpInterface::ExternalModel<WeightBufferization, WeightOp> {
  bool bufferizesToAllocation(Operation *, Value) const { return false; }
  bool isWritable(Operation *, Value, const bufferization::AnalysisState &) const { return false; }
  bufferization::AliasingOpOperandList getAliasingOpOperands(Operation *, Value,
                                                              const bufferization::AnalysisState &) const {
    return {};
  }
  FailureOr<bufferization::BufferLikeType> getBufferType(Operation *op, Value value,
                                                         const bufferization::BufferizationOptions &,
                                                         const bufferization::BufferizationState &,
                                                         SmallVector<Value> &) const {
    auto t = cast<RankedTensorType>(value.getType());
    return cast<bufferization::BufferLikeType>(MemRefType::get(t.getShape(), t.getElementType()));
  }
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter, const bufferization::BufferizationOptions &,
                          bufferization::BufferizationState &) const {
    auto w = cast<WeightOp>(op);
    auto t = dyn_cast<RankedTensorType>(w.getType());
    if (!t) return success();
    rewriter.setInsertionPoint(op);
    auto m = WeightOp::create(rewriter, op->getLoc(), MemRefType::get(t.getShape(), t.getElementType()), w.getName(),
                              w.getLayout());
    m->setAttrs(op->getAttrs());
    bufferization::replaceOpWithBufferizedValues(rewriter, op, m.getResult());
    return success();
  }
};

/// Tiling of hyb.dequant over its [M, N] iteration space (both parallel).
struct DequantTiling : TilingInterface::ExternalModel<DequantTiling, DequantOp> {
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *) const {
    return {utils::IteratorType::parallel, utils::IteratorType::parallel};
  }
  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    auto t = cast<ShapedType>(cast<DequantOp>(op).getInit().getType());
    SmallVector<Range> r;
    for (int64_t d : t.getShape()) r.push_back(Range{b.getIndexAttr(0), b.getIndexAttr(d), b.getIndexAttr(1)});
    return r;
  }
  FailureOr<TilingResult> getTiledImplementation(Operation *op, OpBuilder &b, ArrayRef<OpFoldResult> offsets,
                                                 ArrayRef<OpFoldResult> sizes) const {
    auto d = cast<DequantOp>(op);
    Location loc = op->getLoc();
    OpFoldResult zero = b.getIndexAttr(0), one = b.getIndexAttr(1);
    auto slice = [&](Value v, ArrayRef<OpFoldResult> off, ArrayRef<OpFoldResult> sz) -> Value {
      SmallVector<OpFoldResult> strides(off.size(), one);
      if (isa<RankedTensorType>(v.getType()))
        return tensor::ExtractSliceOp::create(b, loc, v, off, sz, strides);
      return {};
    };
    Value acc = slice(d.getAcc(), offsets, sizes);
    Value scale = slice(d.getScale(), {offsets[0], zero}, {sizes[0], one});
    Value wscale = slice(d.getWscale(), {offsets[1]}, {sizes[1]});
    Value init = slice(d.getInit(), offsets, sizes);
    if (!acc || !scale || !wscale || !init) return failure();
    auto tiled = DequantOp::create(b, loc, TypeRange{init.getType()}, acc, scale, wscale, init, d.getWeightAttr());
    tiled->setDiscardableAttrs(op->getDiscardableAttrDictionary());
    SmallVector<Operation *> slices;
    for (Value v : {acc, scale, wscale, init}) slices.push_back(v.getDefiningOp());
    return TilingResult{{tiled}, SmallVector<Value>(tiled->getResults()), slices};
  }
  LogicalResult getResultTilePosition(Operation *, OpBuilder &, unsigned, ArrayRef<OpFoldResult> offsets,
                                      ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
                                      SmallVector<OpFoldResult> &resultSizes) const {
    resultOffsets.assign(offsets.begin(), offsets.end());
    resultSizes.assign(sizes.begin(), sizes.end());
    return success();
  }
  FailureOr<TilingResult> generateResultTileValue(Operation *op, OpBuilder &b, unsigned, ArrayRef<OpFoldResult> offsets,
                                                  ArrayRef<OpFoldResult> sizes) const {
    return getTiledImplementation(op, b, offsets, sizes);
  }
  // Variants with a pack/unpack alignment hint (unused here).
  FailureOr<TilingResult> getTiledImplementation(Operation *op, OpBuilder &b, ArrayRef<OpFoldResult> offsets,
                                                 ArrayRef<OpFoldResult> sizes, ArrayRef<InnerTileAlignment>) const {
    return getTiledImplementation(op, b, offsets, sizes);
  }
  FailureOr<TilingResult> generateResultTileValue(Operation *op, OpBuilder &b, unsigned n, ArrayRef<OpFoldResult> offsets,
                                                  ArrayRef<OpFoldResult> sizes, ArrayRef<InnerTileAlignment>) const {
    return generateResultTileValue(op, b, n, offsets, sizes);
  }
};
/// Tiling of hyb.quant over its token rows. The scale is a maximum over the whole row, so the
/// column axis is not part of the iteration domain and a tile always holds complete rows.
struct QuantTiling : TilingInterface::ExternalModel<QuantTiling, QuantOp> {
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *) const {
    return {utils::IteratorType::parallel};
  }
  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    auto t = cast<ShapedType>(cast<QuantOp>(op).getInput().getType());
    return {Range{b.getIndexAttr(0), b.getIndexAttr(t.getDimSize(0)), b.getIndexAttr(1)}};
  }
  FailureOr<TilingResult> getTiledImplementation(Operation *op, OpBuilder &b, ArrayRef<OpFoldResult> offsets,
                                                 ArrayRef<OpFoldResult> sizes) const {
    auto q = cast<QuantOp>(op);
    Location loc = op->getLoc();
    OpFoldResult zero = b.getIndexAttr(0), one = b.getIndexAttr(1);
    auto rows = [&](Value v) -> Value {
      auto t = dyn_cast<RankedTensorType>(v.getType());
      if (!t || t.getRank() != 2) return {};
      return tensor::ExtractSliceOp::create(b, loc, v, ArrayRef<OpFoldResult>{offsets[0], zero},
                                            ArrayRef<OpFoldResult>{sizes[0], b.getIndexAttr(t.getDimSize(1))},
                                            ArrayRef<OpFoldResult>{one, one});
    };
    Value input = rows(q.getInput()), qInit = rows(q.getQInit()), scaleInit = rows(q.getScaleInit());
    if (!input || !qInit || !scaleInit) return failure();
    auto tiled = QuantOp::create(b, loc, TypeRange{qInit.getType(), scaleInit.getType()}, input, qInit, scaleInit);
    tiled->setDiscardableAttrs(op->getDiscardableAttrDictionary());
    SmallVector<Operation *> slices;
    for (Value v : {input, qInit, scaleInit}) slices.push_back(v.getDefiningOp());
    return TilingResult{{tiled}, SmallVector<Value>(tiled->getResults()), slices};
  }
  LogicalResult getResultTilePosition(Operation *op, OpBuilder &b, unsigned result, ArrayRef<OpFoldResult> offsets,
                                      ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
                                      SmallVector<OpFoldResult> &resultSizes) const {
    auto t = cast<ShapedType>(op->getResult(result).getType());
    resultOffsets = {offsets[0], b.getIndexAttr(0)};
    resultSizes = {sizes[0], b.getIndexAttr(t.getDimSize(1))};
    return success();
  }
  FailureOr<TilingResult> generateResultTileValue(Operation *op, OpBuilder &b, unsigned, ArrayRef<OpFoldResult> offsets,
                                                  ArrayRef<OpFoldResult> sizes) const {
    return getTiledImplementation(op, b, offsets.take_front(1), sizes.take_front(1));
  }
  LogicalResult getIterationDomainTileFromResultTile(Operation *, OpBuilder &, unsigned, ArrayRef<OpFoldResult> offsets,
                                                     ArrayRef<OpFoldResult> sizes,
                                                     SmallVectorImpl<OpFoldResult> &domainOffsets,
                                                     SmallVectorImpl<OpFoldResult> &domainSizes) const {
    domainOffsets.assign({offsets[0]});
    domainSizes.assign({sizes[0]});
    return success();
  }
  // Variants with a pack/unpack alignment hint (unused here).
  FailureOr<TilingResult> getTiledImplementation(Operation *op, OpBuilder &b, ArrayRef<OpFoldResult> offsets,
                                                 ArrayRef<OpFoldResult> sizes, ArrayRef<InnerTileAlignment>) const {
    return getTiledImplementation(op, b, offsets, sizes);
  }
  FailureOr<TilingResult> generateResultTileValue(Operation *op, OpBuilder &b, unsigned n, ArrayRef<OpFoldResult> offsets,
                                                  ArrayRef<OpFoldResult> sizes, ArrayRef<InnerTileAlignment>) const {
    return generateResultTileValue(op, b, n, offsets, sizes);
  }
};
} // namespace

void mlir::hyb::registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, HybDialect *) {
    QuantOp::attachInterface<DpsBufferization<QuantOp>>(*ctx);
    QMatmulOp::attachInterface<DpsBufferization<QMatmulOp>>(*ctx);
    DequantOp::attachInterface<DpsBufferization<DequantOp>>(*ctx);
    WeightOp::attachInterface<WeightBufferization>(*ctx);
    DequantOp::attachInterface<DequantTiling>(*ctx);
    QuantOp::attachInterface<QuantTiling>(*ctx);
  });
}

#define GET_OP_CLASSES
#include "npu/Decode/Dialect/Hyb/HybOps.cpp.inc"
