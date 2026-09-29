//===----------------------------------------------------------------------===//
// Quantize.cpp — Stage 1 — W8A8: 가중치 행렬곱을 INT8 로
//
// 가중치(진입 인자)를 B 로 받는 행렬곱 하나가 세 조각이 된다.
//
//   alpha[m] = max_k |x[m,k]| * fl32(1/127)             행마다 활성값 스케일
//   qx[m,k]  = clamp(rne(x[m,k] / alpha[m]), -127, 127)  V_QUANT_F16_I8
//   acc      = qx (INT8) x Wq (INT8) -> INT32            M_MMA_I8I8I32
//   y[m,n]   = fp16((fp32(acc[m,n]) * alpha[m]) * s[n])  V_DEQUANT_I32_F16
//
// Wq 와 s 는 원래 FP16 가중치에서 나온다. 출력 채널 n 마다
//
//   s[n]     = fl32(max_k |W[k,n]|) / 127                (FP32 나눗셈)
//   Wq[k,n]  = clamp(rne(W[k,n] / s[n]), -127, 127)      s[n] = 0 이면 0
//
// 이 계산은 이미지를 만드는 쪽(npu-compile --inputs, lib/Target/Quantize)이
// 한다. 여기서는 인자의 타입을 INT8 로 바꾸고, 스케일 인자를 새로 붙이고,
// 둘 다 어느 원본 인자에서 왔는지 npu.quantized 로 적어 둔다.
//
// 같은 활성값을 읽는 행렬곱들(q/k/v, gate/up)은 alpha 와 qx 를 함께 쓴다.
// 활성값 스케일 계산 규칙은 PIM 과 NPU 가 같은 NPU 명령으로 얻으므로, 어느
// 쪽에서 곱하든 INT32 누적값이 같고 출력이 비트 단위로 같다.
//===----------------------------------------------------------------------===//
#include "GraphInternal.h"

using namespace mlir;
namespace npu {

unsigned elementBytes(Type type) {
  if (type.isF16()) return 2;
  if (type.isF32() || type.isInteger(32)) return 4;
  if (type.isInteger(8)) return 1;
  return 0;
}

namespace {
constexpr float kInverse127 = 1.0f / 127.0f;

bool isIdentity2(AffineMap map) { return map == AffineMap::getMultiDimIdentityMap(2, map.getContext()); }
bool isRowProjection(AffineMap map) {
  auto *context = map.getContext();
  auto row = getAffineDimExpr(0, context);
  return map == AffineMap::get(2, 0, {row}, context) ||
         map == AffineMap::get(2, 0, {row, getAffineConstantExpr(0, context)}, context);
}
bool isColumnProjection(AffineMap map) {
  auto *context = map.getContext();
  return map == AffineMap::get(2, 0, {getAffineDimExpr(1, context)}, context);
}
bool shaped(Value v, unsigned rank, Type element) {
  auto t = dyn_cast<ShapedType>(v.getType());
  return t && t.hasStaticShape() && t.getRank() == rank && t.getElementType() == element;
}
bool parallel(linalg::GenericOp gen) {
  return gen.getNumLoops() == 2 &&
         llvm::all_of(gen.getIteratorTypesArray(),
                      [](utils::IteratorType t) { return t == utils::IteratorType::parallel; });
}
// region 안의 연산을 차례로. 상수는 clamp 의 경계뿐이라 건너뛴다.
SmallVector<Operation *> computations(linalg::GenericOp gen) {
  SmallVector<Operation *> ops;
  for (auto &op : gen.getRegion().front().without_terminator())
    if (!isa<arith::ConstantOp>(op)) ops.push_back(&op);
  return ops;
}
bool isConstant(Value v, double value) {
  auto c = v.getDefiningOp<arith::ConstantOp>();
  auto attr = c ? dyn_cast<FloatAttr>(c.getValue()) : FloatAttr();
  return attr && attr.getValue().convertToDouble() == value;
}
} // namespace

bool isQuantizeGeneric(Operation *op) {
  auto gen = dyn_cast<linalg::GenericOp>(op);
  if (!gen || gen.getNumDpsInputs() != 2 || gen.getNumDpsInits() != 1 || !parallel(gen)) return false;
  auto *context = gen.getContext();
  auto maps = gen.getIndexingMapsArray();
  if (!isIdentity2(maps[0]) || !isRowProjection(maps[1]) || !isIdentity2(maps[2])) return false;
  if (!shaped(gen.getDpsInputs()[0], 2, Float16Type::get(context)) ||
      !isRowShape(gen.getDpsInputs()[1].getType()) ||
      cast<ShapedType>(gen.getDpsInputs()[1].getType()).getElementType() != Float32Type::get(context) ||
      !shaped(gen.getDpsInits()[0], 2, IntegerType::get(context, 8)))
    return false;
  Block &body = gen.getRegion().front();
  auto ops = computations(gen);
  if (ops.size() != 6) return false;
  auto widen = dyn_cast<arith::ExtFOp>(ops[0]);
  auto divide = dyn_cast<arith::DivFOp>(ops[1]);
  auto round = dyn_cast<math::RoundEvenOp>(ops[2]);
  auto low = dyn_cast<arith::MaximumFOp>(ops[3]);
  auto high = dyn_cast<arith::MinimumFOp>(ops[4]);
  auto narrow = dyn_cast<arith::FPToSIOp>(ops[5]);
  return widen && divide && round && low && high && narrow &&
         widen.getIn() == body.getArgument(0) && divide.getLhs() == widen.getResult() &&
         divide.getRhs() == body.getArgument(1) && round.getOperand() == divide.getResult() &&
         low.getLhs() == round.getResult() && isConstant(low.getRhs(), -127.0) &&
         high.getLhs() == low.getResult() && isConstant(high.getRhs(), 127.0) &&
         narrow.getIn() == high.getResult() &&
         cast<linalg::YieldOp>(body.getTerminator()).getValues()[0] == narrow.getResult();
}

bool isDequantizeGeneric(Operation *op) {
  auto gen = dyn_cast<linalg::GenericOp>(op);
  if (!gen || gen.getNumDpsInputs() != 3 || gen.getNumDpsInits() != 1 || !parallel(gen)) return false;
  auto *context = gen.getContext();
  auto maps = gen.getIndexingMapsArray();
  if (!isIdentity2(maps[0]) || !isRowProjection(maps[1]) || !isColumnProjection(maps[2]) ||
      !isIdentity2(maps[3]))
    return false;
  auto f32 = Float32Type::get(context);
  if (!shaped(gen.getDpsInputs()[0], 2, IntegerType::get(context, 32)) ||
      !isRowShape(gen.getDpsInputs()[1].getType()) ||
      cast<ShapedType>(gen.getDpsInputs()[1].getType()).getElementType() != f32 ||
      !shaped(gen.getDpsInputs()[2], 1, f32) || !shaped(gen.getDpsInits()[0], 2, Float16Type::get(context)))
    return false;
  Block &body = gen.getRegion().front();
  auto ops = computations(gen);
  if (ops.size() != 4) return false;
  auto widen = dyn_cast<arith::SIToFPOp>(ops[0]);
  auto byRow = dyn_cast<arith::MulFOp>(ops[1]);
  auto byChannel = dyn_cast<arith::MulFOp>(ops[2]);
  auto narrow = dyn_cast<arith::TruncFOp>(ops[3]);
  return widen && byRow && byChannel && narrow && widen.getIn() == body.getArgument(0) &&
         byRow.getLhs() == widen.getResult() && byRow.getRhs() == body.getArgument(1) &&
         byChannel.getLhs() == byRow.getResult() && byChannel.getRhs() == body.getArgument(2) &&
         narrow.getIn() == byChannel.getResult() &&
         cast<linalg::YieldOp>(body.getTerminator()).getValues()[0] == narrow.getResult();
}

namespace {
struct QuantizedActivation { Value alpha, values; };

// alpha[m] = max_k |x[m,k]| * fl32(1/127), qx = V_QUANT(x, alpha).
QuantizedActivation quantizeActivation(OpBuilder &b, Location loc, Value x) {
  auto *context = b.getContext();
  auto type = cast<RankedTensorType>(x.getType());
  int64_t rows = type.getDimSize(0), cols = type.getDimSize(1);
  auto f32 = Float32Type::get(context);
  auto i8 = IntegerType::get(context, 8);
  auto rowType = RankedTensorType::get({rows}, f32);
  auto identity = AffineMap::getMultiDimIdentityMap(2, context);
  auto projection = AffineMap::get(2, 0, {getAffineDimExpr(0, context)}, context);
  auto lane = AffineMap::getMultiDimIdentityMap(1, context);
  using IT = utils::IteratorType;

  // 행마다 |x| 의 최댓값. 합법성 검사가 최댓값 리덕션의 초기값을 -inf 로 요구한다.
  auto negativeInfinity = b.create<arith::ConstantOp>(
      loc, b.getFloatAttr(f32, APFloat::getInf(APFloat::IEEEsingle(), /*Negative=*/true)));
  Value seed = b.create<linalg::FillOp>(loc, ValueRange{negativeInfinity},
                                        ValueRange{b.create<tensor::EmptyOp>(loc, rowType.getShape(), f32)})
                   .getResult(0);
  Value peak = b.create<linalg::GenericOp>(
                    loc, TypeRange{rowType}, ValueRange{x}, ValueRange{seed},
                    ArrayRef<AffineMap>{identity, projection}, ArrayRef<IT>{IT::parallel, IT::reduction},
                    [&](OpBuilder &nb, Location nl, ValueRange args) {
                      Value magnitude = nb.create<math::AbsFOp>(nl, args[0]);
                      Value wide = nb.create<arith::ExtFOp>(nl, f32, magnitude);
                      nb.create<linalg::YieldOp>(nl, ValueRange{nb.create<arith::MaximumFOp>(nl, wide, args[1])});
                    })
                   .getResult(0);

  // alpha = peak * fl32(1/127). 상수는 합법화 끝에 인자로 올라간다.
  Value inverse = b.create<arith::ConstantOp>(
      loc, DenseElementsAttr::get(rowType, b.getFloatAttr(f32, kInverse127).getValue()));
  Value alpha = b.create<linalg::GenericOp>(
                     loc, TypeRange{rowType}, ValueRange{peak, inverse},
                     ValueRange{b.create<tensor::EmptyOp>(loc, rowType.getShape(), f32)},
                     ArrayRef<AffineMap>{lane, lane, lane}, ArrayRef<IT>{IT::parallel},
                     [&](OpBuilder &nb, Location nl, ValueRange args) {
                       nb.create<linalg::YieldOp>(nl, ValueRange{nb.create<arith::MulFOp>(nl, args[0], args[1])});
                     })
                    .getResult(0);

  // qx = clamp(rne(x / alpha), -127, 127). V_QUANT_F16_I8 은 alpha = 0 이면 0 을 낸다.
  auto quantized = RankedTensorType::get({rows, cols}, i8);
  Value values = b.create<linalg::GenericOp>(
                      loc, TypeRange{quantized}, ValueRange{x, alpha},
                      ValueRange{b.create<tensor::EmptyOp>(loc, quantized.getShape(), i8)},
                      ArrayRef<AffineMap>{identity, projection, identity}, ArrayRef<IT>{IT::parallel, IT::parallel},
                      [&](OpBuilder &nb, Location nl, ValueRange args) {
                        Value low = nb.create<arith::ConstantOp>(nl, nb.getFloatAttr(f32, -127.0));
                        Value high = nb.create<arith::ConstantOp>(nl, nb.getFloatAttr(f32, 127.0));
                        Value wide = nb.create<arith::ExtFOp>(nl, f32, args[0]);
                        Value scaled = nb.create<arith::DivFOp>(nl, wide, args[1]);
                        Value rounded = nb.create<math::RoundEvenOp>(nl, scaled);
                        Value clamped = nb.create<arith::MinimumFOp>(
                            nl, nb.create<arith::MaximumFOp>(nl, rounded, low), high);
                        nb.create<linalg::YieldOp>(nl, ValueRange{nb.create<arith::FPToSIOp>(nl, i8, clamped)});
                      })
                     .getResult(0);
  return {alpha, values};
}
} // namespace

LogicalResult quantizeGraph(ModuleOp m) {
  auto mode = m->getAttrOfType<StringAttr>("npu.quantize");
  if (!mode) return success();
  if (mode.getValue() != "w8a8") return m.emitError("unsupported quantization mode: ") << mode.getValue();
  auto functions = m.getOps<func::FuncOp>();
  if (!llvm::hasSingleElement(functions)) return m.emitError("quantization requires one entry function");
  auto function = *functions.begin();
  auto *context = m.getContext();
  auto f16 = Float16Type::get(context);
  auto f32 = Float32Type::get(context);
  auto i8 = IntegerType::get(context, 8);
  auto i32 = IntegerType::get(context, 32);

  // 가중치: 이 행렬곱 하나만 읽는 FP16 진입 인자. 다른 데서도 읽히는 인자는
  // 원래 값이 필요하므로 FP16 으로 남긴다.
  SmallVector<linalg::MatmulOp> work;
  function.walk([&](linalg::MatmulOp mm) {
    auto weight = dyn_cast<BlockArgument>(mm.getDpsInputOperand(1)->get());
    if (!weight || weight.getOwner()->getParentOp() != function.getOperation() || !weight.hasOneUse()) return;
    if (!shaped(mm.getDpsInputOperand(0)->get(), 2, f16) || !shaped(weight, 2, f16) ||
        !shaped(mm.getDpsInitOperand(0)->get(), 2, f16) || !isZeroFilled(mm.getDpsInitOperand(0)->get()))
      return;
    work.push_back(mm);
  });

  llvm::DenseMap<Value, QuantizedActivation> activations;
  OpBuilder b(context);
  using IT = utils::IteratorType;
  auto identity = AffineMap::getMultiDimIdentityMap(2, context);
  auto projection = AffineMap::get(2, 0, {getAffineDimExpr(0, context)}, context);
  auto column = AffineMap::get(2, 0, {getAffineDimExpr(1, context)}, context);
  for (auto mm : work) {
    Location loc = mm.getLoc();
    Value x = mm.getDpsInputOperand(0)->get();
    auto weight = cast<BlockArgument>(mm.getDpsInputOperand(1)->get());
    auto weightType = cast<RankedTensorType>(weight.getType());
    int64_t rows = cast<RankedTensorType>(x.getType()).getDimSize(0);
    int64_t outputs = weightType.getDimSize(1);

    // 같은 활성값의 첫 행렬곱 앞에서 한 번만 양자화한다. walk 가 프로그램
    // 순서라 뒤의 사용처는 전부 이 자리를 지난다.
    b.setInsertionPoint(mm);
    auto found = activations.find(x);
    if (found == activations.end()) found = activations.try_emplace(x, quantizeActivation(b, loc, x)).first;
    QuantizedActivation q = found->second;

    // 가중치 인자를 INT8 로, 채널별 스케일 인자를 끝에 붙인다.
    unsigned source = weight.getArgNumber();
    weight.setType(RankedTensorType::get(weightType.getShape(), i8));
    unsigned scaleIndex = function.getNumArguments();
    if (failed(function.insertArgument(scaleIndex, RankedTensorType::get({outputs}, f32), {}, loc)))
      return failure();
    Value scale = function.getArgument(scaleIndex);
    auto origin = [&](StringRef part) {
      return DictionaryAttr::get(context, {NamedAttribute(StringAttr::get(context, "source"),
                                                          IntegerAttr::get(IntegerType::get(context, 64), source)),
                                           NamedAttribute(StringAttr::get(context, "part"),
                                                          StringAttr::get(context, part))});
    };
    function.setArgAttr(source, "npu.quantized", origin("weight"));
    function.setArgAttr(scaleIndex, "npu.quantized", origin("scale"));

    auto accumulatorType = RankedTensorType::get({rows, outputs}, i32);
    Value zero = b.create<arith::ConstantOp>(loc, b.getIntegerAttr(i32, 0));
    Value init = b.create<linalg::FillOp>(loc, ValueRange{zero},
                                          ValueRange{b.create<tensor::EmptyOp>(loc, accumulatorType.getShape(), i32)})
                     .getResult(0);
    Value accumulator = b.create<linalg::MatmulOp>(loc, TypeRange{accumulatorType}, ValueRange{q.values, weight},
                                                   ValueRange{init})
                            .getResult(0);
    auto resultType = cast<RankedTensorType>(mm.getResult(0).getType());
    Value y = b.create<linalg::GenericOp>(
                   loc, TypeRange{resultType}, ValueRange{accumulator, q.alpha, scale},
                   ValueRange{b.create<tensor::EmptyOp>(loc, resultType.getShape(), f16)},
                   ArrayRef<AffineMap>{identity, projection, column, identity},
                   ArrayRef<IT>{IT::parallel, IT::parallel},
                   [&](OpBuilder &nb, Location nl, ValueRange args) {
                     Value wide = nb.create<arith::SIToFPOp>(nl, f32, args[0]);
                     Value byRow = nb.create<arith::MulFOp>(nl, wide, args[1]);
                     Value byChannel = nb.create<arith::MulFOp>(nl, byRow, args[2]);
                     nb.create<linalg::YieldOp>(nl, ValueRange{nb.create<arith::TruncFOp>(nl, f16, byChannel)});
                   })
                  .getResult(0);
    mm.getResult(0).replaceAllUsesWith(y);
    mm.erase();
  }
  // setType 은 블록 인자만 바꾼다. 함수 타입은 인자 타입에서 다시 만든다.
  function.setType(FunctionType::get(context, function.getBody().getArgumentTypes(),
                                     function.getFunctionType().getResults()));
  return cleanIR(m);
}
} // namespace npu
