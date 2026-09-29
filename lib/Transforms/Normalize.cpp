//===----------------------------------------------------------------------===//
// Normalize.cpp — Stage 1 전반 — 캡처를 이 컴파일러가 아는 모양으로
//===----------------------------------------------------------------------===//
#include "GraphInternal.h"

using namespace mlir;
namespace npu {
// 그래서 전역 패스 대신 리덕션 자체를 겨냥해 손으로 쓴다.
//===----------------------------------------------------------------------===//
namespace {

bool reducesRows(linalg::GenericOp gen) {
  return gen.getNumDpsInits() == 1 &&
         llvm::is_contained(gen.getIteratorTypesArray(), utils::IteratorType::reduction) &&
         isRowShape(gen.getDpsInitOperand(0)->get().getType());
}
/// region 하나의 타입을 바꿔, FP32 값 아래로 흐르는 것이 전부 FP32 가 되게 한다.
///
/// `widened` 는 블록 인자마다 "이미 FP32 인가"를 담는다. 거기서 시작해 아래로
/// 번져 나간다 — 피연산자 중 하나라도 FP32 면 나머지를 extf 로 올리고 결과도
/// FP32 로 바꾼다.
void widenRegion(OpBuilder &b, Block &body, ArrayRef<bool> widened) {
  auto f32 = b.getF32Type();

  for (auto [index, arg] : llvm::enumerate(body.getArguments()))
    if (index < widened.size() && widened[index])
      arg.setType(f32);

  // 순회 중에 op 을 추가하므로 early_inc_range 로 돈다.
  for (Operation &op : llvm::make_early_inc_range(body.without_terminator())) {
    bool touchesF32 = llvm::any_of(op.getOperands(),
                                   [&](Value v) { return v.getType().isF32(); });
    if (op.getNumResults() != 1 || !touchesF32)
      continue;

    b.setInsertionPoint(&op);
    for (auto &operand : op.getOpOperands())
      if (operand.get().getType().isF16())
        operand.set(b.create<arith::ExtFOp>(op.getLoc(), f32, operand.get()));
    op.getResult(0).setType(f32);
  }
}
/// 리덕션 하나와 그 아래 행 체인 전체를 FP32 로 옮긴다.
/// (widenConsumers 와 서로 부르므로 먼저 선언한다.)
LogicalResult widenRowChain(OpBuilder &b, linalg::GenericOp gen);

/// 넓혀진 값을 쓰는 쪽들을 따라가며 처리한다.
///
/// 소비자도 행 벡터를 내놓으면 체인이 이어지므로 재귀한다. 행과 행렬을 섞는
/// 소비자는 FP32 로 올리는 대신 truncf 를 받는다 — broadcast 경로가 이미
/// 읽을 줄 아는 모양이기 때문이다.
LogicalResult widenConsumers(OpBuilder &b, Value widenedValue) {
  SmallVector<OpOperand *> uses;
  for (OpOperand &use : widenedValue.getUses()) uses.push_back(&use);
  for (auto *use : uses) {
    auto consumer = dyn_cast<linalg::GenericOp>(use->getOwner());
    if (!consumer) return use->getOwner()->emitError("a widened row value reaches an unsupported user");
    unsigned index = use->getOperandNumber();
    if (index >= consumer.getRegion().front().getNumArguments()) continue;
    if (llvm::all_of(consumer->getResultTypes(), isRowShape) && consumer.getNumDpsInits() == 1) {
      if (failed(widenRowChain(b,consumer))) return failure();
      continue;
    }
    auto arg = consumer.getRegion().front().getArgument(index);
    if (arg.getType().isF32()) continue;
    arg.setType(b.getF32Type());
    b.setInsertionPointToStart(&consumer.getRegion().front());
    auto narrowed = b.create<arith::TruncFOp>(consumer.getLoc(),b.getF16Type(),arg);
    arg.replaceAllUsesExcept(narrowed,narrowed);
  }
  return success();
}

LogicalResult widenRowChain(OpBuilder &b, linalg::GenericOp gen) {
  auto init = cast<RankedTensorType>(gen.getDpsInitOperand(0)->get().getType());
  if (init.getElementType().isF32()) return success();
  auto widenedType = RankedTensorType::get(init.getShape(),b.getF32Type());
  auto fill = gen.getDpsInitOperand(0)->get().getDefiningOp<linalg::FillOp>();
  b.setInsertionPoint(gen);
  Value replacement;
  if (fill) {
    auto seed = fill.getInputs()[0].getDefiningOp<arith::ConstantOp>();
    auto attr = seed ? dyn_cast<FloatAttr>(seed.getValue()) : FloatAttr();
    if (!attr) return gen.emitError("a row reduction needs a constant identity to widen");
    APFloat value = attr.getValue();
    bool lost = false;
    value.convert(APFloat::IEEEsingle(),APFloat::rmNearestTiesToEven,&lost);
    auto constant = b.create<arith::ConstantOp>(gen.getLoc(),b.getF32FloatAttr(value.convertToFloat()));
    auto empty = b.create<tensor::EmptyOp>(gen.getLoc(),widenedType.getShape(),b.getF32Type());
    replacement = b.create<linalg::FillOp>(gen.getLoc(),ValueRange{constant},ValueRange{empty}).getResult(0);
  } else {
    replacement = b.create<tensor::EmptyOp>(gen.getLoc(),widenedType.getShape(),b.getF32Type());
  }
  gen.getDpsInitOperand(0)->set(replacement);
  // 피연산자가 이미 FP32 인 블록 인자는 그대로 따라 올라간다. 이 재작성이
  // 넓히는 대상은 누산기 쪽이다.
  SmallVector<bool> widened;
  for (auto value : gen.getDpsInputs())
    widened.push_back(cast<ShapedType>(value.getType()).getElementType().isF32());
  widened.push_back(true);
  widenRegion(b,gen.getRegion().front(),widened);
  gen.getResult(0).setType(widenedType);
  return widenConsumers(b,gen.getResult(0));
}
} // namespace

//===----------------------------------------------------------------------===//
// 누적과 반올림을 한 쌍으로 접기
//
// 캡처는 `matmul(f16, f16) -> f32` 뒤에 f16 으로 내리는 truncf 를 붙여 준다.
// torch 가 뜻하는 바가 그것이기 때문이다.
//
// 그런데 그것이 하드웨어가 하는 일과 정확히 같다 — M_MMA_F16F16F32 가 FP32
// 로 누적하고 M_WRITEOUT_F16 이 내보내며 반올림한다. 그러므로 이 한 쌍을
// 접는 것은 근사가 아니라 **정확한** 변환이다.
//===----------------------------------------------------------------------===//
namespace {

/// `matmul -> f32` + `truncf -> f16` 쌍을 `matmul -> f16` 하나로 접는다.
///
/// 조건이 여럿이고 하나라도 어긋나면 그 행렬곱은 건너뛴다(continue). 접지
/// 못해도 컴파일은 계속되며, 나중 단계가 FP32 중간값을 그대로 다룬다.
template <typename MatmulLike>
LogicalResult foldAccumulateAndRound(OpBuilder &b, ModuleOp m) {
  SmallVector<MatmulLike> work;
  m.walk([&](MatmulLike mm) { work.push_back(mm); });

  for (auto mm : work) {
    // ① 누산기가 FP32 이고, 결과를 쓰는 곳이 하나여야 한다.
    //    쓰는 곳이 여럿이면 truncf 만 떼어낼 수 없다.
    auto init = dyn_cast<RankedTensorType>(mm.getDpsInitOperand(0)->get().getType());
    if (!init || !init.getElementType().isF32() || !mm.getResult(0).hasOneUse())
      continue;

    // ② 두 입력이 FP16 이어야 한다. M_MMA_F16F16F32 가 받는 모양이다.
    bool inputsAreF16 = llvm::all_of(mm.getDpsInputs(), [](Value v) {
      auto t = dyn_cast<RankedTensorType>(v.getType());
      return t && t.getElementType().isF16();
    });
    if (!inputsAreF16)
      continue;

    // ③ 유일한 소비자가 generic 하나여야 한다 — 이것이 반올림 후보다.
    auto round = dyn_cast<linalg::GenericOp>(*mm.getResult(0).getUsers().begin());
    if (!round || round.getNumDpsInputs() != 1 || round.getNumDpsInits() != 1)
      continue;

    // ④ 그 generic 이 같은 모양의 FP16 을 내놓아야 한다.
    auto rounded = dyn_cast<RankedTensorType>(round.getDpsInitOperand(0)->get().getType());
    if (!rounded || !rounded.getElementType().isF16() ||
        rounded.getShape() != init.getShape())
      continue;

    // ⑤ 원소를 자리 그대로 훑어야 한다(항등 맵, 전부 병렬).
    //    자리를 옮기는 generic 이면 단순 반올림이 아니다.
    bool walksInPlace =
        llvm::all_of(round.getIndexingMapsArray(),
                     [](AffineMap map) { return map.isIdentity(); }) &&
        llvm::all_of(round.getIteratorTypesArray(), [](utils::IteratorType t) {
          return t == utils::IteratorType::parallel;
        });
    if (!walksInPlace)
      continue;

    // ⑥ region 안이 truncf 딱 하나여야 한다. 입력을 그대로 받아 내놓는,
    //    다른 계산이 섞이지 않은 순수한 반올림이어야 접을 수 있다.
    Block &body = round.getRegion().front();
    auto ops = llvm::to_vector(
        llvm::map_range(body.without_terminator(), [](Operation &o) { return &o; }));
    if (ops.size() != 1 || !isa<arith::TruncFOp>(ops[0]) ||
        ops[0]->getOperand(0) != body.getArgument(0) ||
        cast<linalg::YieldOp>(body.getTerminator()).getValues()[0] != ops[0]->getResult(0))
      continue;

    // ⑦ 누산기가 0 으로 시작해야 한다. 0 이 아닌 초기값은 하드웨어가 실을
    //    방법이 없다(누산기를 불러올 명령이 없다).
    if (!isZeroFilled(mm.getDpsInitOperand(0)->get()))
      continue;

    // 조건을 다 통과했다 — FP16 을 직접 내놓는 행렬곱으로 바꿔 끼운다.
    b.setInsertionPoint(mm);
    auto zero = b.create<arith::ConstantOp>(mm.getLoc(),b.getF16FloatAttr(0.0f));
    auto empty = b.create<tensor::EmptyOp>(mm.getLoc(),rounded.getShape(),b.getF16Type());
    auto filled = b.create<linalg::FillOp>(mm.getLoc(),ValueRange{zero},ValueRange{empty});
    auto narrow = b.create<MatmulLike>(mm.getLoc(),TypeRange{rounded},
                                       mm.getDpsInputs(),ValueRange{filled.getResult(0)});
    round.getResult(0).replaceAllUsesWith(narrow.getResult(0));
    round.erase();
    mm.erase();
  }
  return success();
}
} // namespace

//===----------------------------------------------------------------------===//
// 행렬 체인을 FP16 으로 좁히기
//
// VPU 에는 FP32 원소별 명령이 없다 — V_ADD_F16 은 있고 V_ADD_F32 는 없다.
// 그래서 캡처가 준 FP32 행렬 체인은 적힌 그대로 돌지 않는다.
//
// `--fp16` 이 바로 "그 중간값들을 FP16 으로 두어도 좋다"는 허가이고, 이
// 재작성이 그것을 실행한다: 넓히는 generic 이 사라지고 소비자들이 FP16
// 원본을 직접 읽는다.
//===----------------------------------------------------------------------===//
namespace {
LogicalResult narrowMatrixChains(OpBuilder &b, ModuleOp m) {
  auto isMatrix = [](Type type) {
    auto t = dyn_cast<RankedTensorType>(type);
    return t && t.hasStaticShape() && t.getRank() == 2 && t.getDimSize(1) > 1;
  };
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<linalg::GenericOp> work;
    m.walk([&](linalg::GenericOp gen) { work.push_back(gen); });
    for (auto gen : work) {
      if (gen.getNumDpsInits() != 1) continue;
      auto init = dyn_cast<RankedTensorType>(gen.getDpsInitOperand(0)->get().getType());
      if (!init || !init.getElementType().isF32() || !isMatrix(init) ||
          llvm::is_contained(gen.getIteratorTypesArray(), utils::IteratorType::reduction))
        continue;
      // 좁히기는 결과 타입을 제자리에서 바꾼다. reshape 는 새 원소 타입을
      // 그대로 통과시키므로, 닿을 수 있는 값 전체를 먼저 계획해 두고 그 밖의
      // 것이 읽으면 포기한다 — return 이나 slice 가 남으면 이미 없는 바이트를
      // 가리키게 된다.
      SmallVector<Value> narrowed_values{gen.getResult(0)};
      bool reshapedOnly = true;
      for (unsigned seen = 0; seen < narrowed_values.size() && reshapedOnly; ++seen)
        for (Operation *user : narrowed_values[seen].getUsers()) {
          if (isa<linalg::GenericOp>(user)) continue;
          if (isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(user)) {
            narrowed_values.push_back(user->getResult(0));
            continue;
          }
          reshapedOnly = false;
          break;
        }
      if (!reshapedOnly) continue;
      // FP32 입력마다 돌아갈 FP16 원본이 이미 있어야 한다.
      SmallVector<Value> sources;
      bool ready = true;
      for (auto value : gen.getDpsInputs()) {
        auto type = dyn_cast<RankedTensorType>(value.getType());
        if (type && type.getElementType().isF16()) { sources.push_back(value); continue; }
        // broadcast 행은 FP32 로 남는다 — 스칼라 유닛과 V_*_SCALAR_F16 이
        // 그 형태를 원한다.
        if (isRowShape(value.getType())) { sources.push_back(value); continue; }
        auto producer = value.getDefiningOp<linalg::GenericOp>();
        Value original;
        if (producer && producer.getNumDpsInputs() == 1) {
          Block &body = producer.getRegion().front();
          auto ops = llvm::to_vector(llvm::map_range(body.without_terminator(),
                                                     [](Operation &o) { return &o; }));
          if (ops.size() == 1 && isa<arith::ExtFOp>(ops[0]) &&
              ops[0]->getOperand(0) == body.getArgument(0))
            original = producer.getDpsInputs()[0];
        }
        if (!original) { ready = false; break; }
        sources.push_back(original);
      }
      if (!ready) continue;
      b.setInsertionPoint(gen);
      auto narrowed = RankedTensorType::get(init.getShape(),b.getF16Type());
      for (auto [operand,source] : llvm::zip(gen.getDpsInputOperands(),sources)) operand->set(source);
      gen.getDpsInitOperand(0)->set(b.create<tensor::EmptyOp>(gen.getLoc(),narrowed.getShape(),b.getF16Type()));
      Block &body = gen.getRegion().front();
      for (auto [index,arg] : llvm::enumerate(body.getArguments()))
        arg.setType(index + 1 == body.getNumArguments()
                        ? Type(b.getF16Type())
                        : cast<ShapedType>(gen->getOperand(index).getType()).getElementType());
      for (Operation &op : llvm::make_early_inc_range(body.without_terminator())) {
        if (auto ext = dyn_cast<arith::ExtFOp>(op); ext && ext.getIn().getType().isF16()) {
          ext.getResult().replaceAllUsesWith(ext.getIn());
          ext.erase();
          continue;
        }
        if (op.getNumResults() != 1 ||
            !llvm::any_of(op.getOperands(), [](Value v) { return v.getType().isF16(); }))
          continue;
        // broadcast 행은 FP32 로 와서 FP16 스트림과 만난다. 여기서 반올림한
        // 모양이 V_*_SCALAR_F16 이 읽는 모양이고, --fp16 이 이미 그것을 허용했다.
        b.setInsertionPoint(&op);
        for (auto &operand : op.getOpOperands())
          if (operand.get().getType().isF32())
            operand.set(b.create<arith::TruncFOp>(op.getLoc(),b.getF16Type(),operand.get()));
        op.getResult(0).setType(b.getF16Type());
      }
      auto yielded = cast<linalg::YieldOp>(body.getTerminator());
      if (yielded.getValues()[0].getType().isF32()) {
        b.setInsertionPoint(yielded);
        yielded.setOperand(0,b.create<arith::TruncFOp>(gen.getLoc(),b.getF16Type(),yielded.getValues()[0]));
      }
      gen.getResult(0).setType(narrowed);
      for (Value value : ArrayRef<Value>(narrowed_values).drop_front()) {
        auto shape = cast<RankedTensorType>(value.getType());
        value.setType(RankedTensorType::get(shape.getShape(), b.getF16Type()));
      }
      // FP32 를 기대하는 소비자는 자기 쪽에서 넓힌다. 리덕션은 이미 그렇게 한다.
      SmallVector<OpOperand *> uses;
      for (Value value : narrowed_values)
        for (OpOperand &use : value.getUses()) uses.push_back(&use);
      for (auto *use : uses) {
        auto consumer = dyn_cast<linalg::GenericOp>(use->getOwner());
        if (!consumer) continue;
        unsigned index = use->getOperandNumber();
        if (index >= consumer.getRegion().front().getNumArguments()) continue;
        auto arg = consumer.getRegion().front().getArgument(index);
        if (!arg.getType().isF32()) continue;
        arg.setType(b.getF16Type());
        b.setInsertionPointToStart(&consumer.getRegion().front());
        auto widened = b.create<arith::ExtFOp>(consumer.getLoc(),b.getF32Type(),arg);
        arg.replaceAllUsesExcept(widened,widened);
      }
      changed = true;
    }
  }
  return success();
}
} // namespace

namespace {
// 여기서 (N) 과 (N,1) 은 같은 행 벡터이므로 그 사이의 reshape 는 아무 정보도
// 나르지 않는다. 떼어내면 형태가 하나로 유지되고, 소비자의 인덱싱 맵만 고치면 된다.
LogicalResult dropRowReshapes(ModuleOp m) {
  auto *context = m.getContext();
  auto row = getAffineDimExpr(0,context), zero = getAffineConstantExpr(0,context);
  auto keepdim = AffineMap::get(2,0,{row,zero},context);
  auto flat = AffineMap::get(2,0,{row},context);
  SmallVector<tensor::ExpandShapeOp> work;
  m.walk([&](tensor::ExpandShapeOp e) { work.push_back(e); });
  for (auto expand : work) {
    if (!isRowShape(expand.getSrc().getType()) || !isRowShape(expand.getResult().getType())) continue;
    SmallVector<OpOperand *> uses;
    for (OpOperand &use : expand.getResult().getUses()) uses.push_back(&use);
    bool rewrote = true;
    for (auto *use : uses) {
      auto consumer = dyn_cast<linalg::GenericOp>(use->getOwner());
      if (!consumer) { rewrote = false; continue; }
      auto maps = consumer.getIndexingMapsArray();
      unsigned index = use->getOperandNumber();
      if (index >= maps.size() || maps[index] != keepdim) { rewrote = false; continue; }
      maps[index] = flat;
      consumer.setIndexingMapsAttr(ArrayAttr::get(context,
          llvm::to_vector(llvm::map_range(maps,[](AffineMap map) -> Attribute {
            return AffineMapAttr::get(map); }))));
      use->set(expand.getSrc());
    }
    if (rewrote && expand.getResult().use_empty()) expand.erase();
  }
  return success();
}
} // namespace

namespace {
// region 안의 스칼라 상수는 놓을 자리가 없다 — ISA 에는 상수를 레지스터로
// 주소 지정 가능한 곳에 쓰는 명령이 없다. 그래서 broadcast 행으로 바꾸고,
// hoistConstants 가 그것을 패키지가 채우는 진입 인자로 올린다.
LogicalResult hoistRegionScalars(OpBuilder &b, ModuleOp m) {
  auto *context = m.getContext();
  SmallVector<linalg::GenericOp> work;
  m.walk([&](linalg::GenericOp gen) { work.push_back(gen); });
  for (auto gen : work) {
    auto result = dyn_cast<RankedTensorType>(gen.getDpsInitOperand(0)->get().getType());
    if (!result || gen.getNumDpsInits() != 1 || result.getRank() < 1 || result.getRank() > 2) continue;
    // 행렬은 행마다 값 하나를 받아 퍼뜨리고, 행 벡터는 자기 모양 그대로 상수를
    // 받는다. 둘 다 이미 있는 경로를 재사용한다.
    bool broadcast = !isRowShape(result);
    // linalg region 은 바깥과 격리되어 있지 않다. 그래서 캡처는 스칼라 상수를
    // 함수 범위에 두고 안에서 참조만 한다.
    SetVector<Value> literals;
    gen.getRegion().walk([&](Operation *op) {
      for (auto &operand : op->getOpOperands()) {
        Value value = operand.get();
        if (!isa<FloatType>(value.getType()) || value.getParentRegion() == &gen.getRegion()) continue;
        // 역수의 분자 1 은 그대로 둔다 — S_RCP_F32 가 1/x 를 정확히 계산하는데,
        // 분자를 밖으로 올리면 매처가 그 형태를 못 알아본다.
        if (operand.getOperandNumber() == 0 && isExactReciprocal(operand.getOwner())) continue;
        literals.insert(value);
      }
    });
    if (literals.empty()) continue;
    b.setInsertionPoint(gen);
    SmallVector<Value> inputs(gen.getDpsInputs());
    SmallVector<AffineMap> maps(gen.getIndexingMapsArray());
    auto rowMap = broadcast ? AffineMap::get(2,0,{getAffineDimExpr(0,context)},context)
                            : AffineMap::getMultiDimIdentityMap(result.getRank(),context);
    auto rowType = broadcast ? RankedTensorType::get({result.getDimSize(0)},b.getF32Type())
                             : RankedTensorType::get(result.getShape(),b.getF32Type());
    bool usable = true;
    for (auto literal : literals) {
      auto constant = literal.getDefiningOp<arith::ConstantOp>();
      auto attr = constant ? dyn_cast<FloatAttr>(constant.getValue()) : FloatAttr();
      if (!attr) { usable = false; break; }
      APFloat value = attr.getValue();
      bool lost = false;
      value.convert(APFloat::IEEEsingle(),APFloat::rmNearestTiesToEven,&lost);
      inputs.push_back(b.create<arith::ConstantOp>(gen.getLoc(),
          DenseElementsAttr::get(rowType,value)));
      maps.insert(maps.begin() + (inputs.size() - 1), rowMap);
    }
    if (!usable) continue;
    auto replacement = b.create<linalg::GenericOp>(gen.getLoc(),gen->getResultTypes(),inputs,
        gen.getDpsInits(),maps,gen.getIteratorTypesArray());
    IRMapping mapping;
    gen.getRegion().cloneInto(&replacement.getRegion(),mapping);
    Block &body = replacement.getRegion().front();
    // 블록 인자는 피연산자 순서를 따른다 — 입력 전부, 그 다음 init. 새로 만든
    // 행들은 입력이므로 맨 뒤가 아니라 init 앞에 들어간다.
    unsigned before = gen.getNumDpsInputs();
    SmallVector<Value> added;
    for (unsigned i = 0; i < literals.size(); ++i)
      added.push_back(body.insertArgument(before + i, b.getF32Type(), gen.getLoc()));
    for (auto [literal,arg] : llvm::zip(literals,added)) {
      b.setInsertionPointToStart(&body);
      Value value = arg;
      if (literal.getType().isF16()) value = b.create<arith::TruncFOp>(gen.getLoc(),b.getF16Type(),arg);
      replacement.getRegion().walk([&,literal=literal](Operation *op) {
        for (auto &operand : op->getOpOperands())
          if (operand.get() == literal) operand.set(value);
      });
    }
    // f64 상수는 f32 피연산자를 가진 truncf 를 남긴다. 같은 타입 사이의 캐스트는
    // 연산이 아니라 이름일 뿐이므로 지운다.
    replacement.getRegion().walk([&](Operation *op) {
      if (!isa<arith::TruncFOp,arith::ExtFOp>(op)) return;
      if (op->getOperand(0).getType() != op->getResult(0).getType()) return;
      op->getResult(0).replaceAllUsesWith(op->getOperand(0));
      op->erase();
    });
    gen.getResult(0).replaceAllUsesWith(replacement.getResult(0));
    gen.erase();
  }
  return success();
}
} // namespace

/// 크기가 1 인 축을 없앤다 — MLIR 의 linalg-fold-unit-extent-dims 를 돌린다.
///
/// 캡처는 배치가 1 이어도 그 축을 들고 다녀서 전부 rank 3 이 된다. 그 1 은
/// 아무 정보도 담고 있지 않으므로 지워도 데이터는 그대로다(1x32x32 와 32x32
/// 는 둘 다 원소 1024 개). 지우고 나면 파이프라인 나머지가 전제하는 rank-2
/// 세계로 돌아온다.
///
/// **이름은 배치를 말하지만 축의 위치를 가리지 않는다** — 32x1x32 의 가운데
/// 1 도 접는다. 반대로 크기가 1 이 아닌 축은 건드리지 않으므로 32x32x32 에는
/// 아무 일도 일어나지 않는다. 그런 것은 원소별이면 flattenElementwise 가
/// 1024x32 로 합치고, batch_matmul 이면 dropUnitBatchMatmuls 가 32 개로 쪼갠다.
///
/// 맨 앞의 가드가 있는 이유: 이 패스는 루프가 나르는 값도 고쳐서 누적이
/// 제자리 버퍼화를 못 하게 만들 수 있다
/// (`Yield operand #0 is not equivalent to the corresponding iter bbArg`).
/// 그래서 지울 축이 실제로 있을 때만 돌린다.
LogicalResult foldBatchDimension(ModuleOp m) {
  bool ranked = false;
  m.walk([&](Operation *op) {
    for (auto type : op->getResultTypes())
      if (auto shaped = dyn_cast<ShapedType>(type); shaped && shaped.getRank() > 2) ranked = true;
  });
  if (!ranked) return success();
  PassManager pm(m.getContext());
  pm.addPass(createLinalgFoldUnitExtentDimsPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  return pm.run(m);
}

namespace {
// 캡처에서 x**2 는 math.fpowi 로 온다. 거듭제곱 명령은 없고, 지수가 작은
// 양의 상수이면 곱셈을 반복한 것이 **같은 값**이다 — 근사가 아니다.
LogicalResult expandIntegerPowers(OpBuilder &b, ModuleOp m) {
  SmallVector<math::FPowIOp> work;
  m.walk([&](math::FPowIOp op) { work.push_back(op); });
  for (auto op : work) {
    APInt exponent;
    if (!matchPattern(op.getRhs(),m_ConstantInt(&exponent)) || exponent.isNegative() ||
        exponent.getZExtValue() < 1 || exponent.getZExtValue() > 4)
      return op.emitError("only a literal exponent of 1..4 is lowered");
    b.setInsertionPoint(op);
    Value value = op.getLhs();
    for (uint64_t i = 1; i < exponent.getZExtValue(); ++i)
      value = b.create<arith::MulFOp>(op.getLoc(),value,op.getLhs());
    op.getResult().replaceAllUsesWith(value);
    op.erase();
  }
  return success();
}

// torch-mlir lowers relu to `select(cmpf ugt x, +0.0), x, +0.0)`. The VPU has no
// compare or select, but that expression is exactly `maximumf(x, +0.0)`: a NaN
// compares unordered and is selected (maximumf propagates it), and -0.0 is not
// greater than +0.0 so +0.0 is selected (maximumf orders -0.0 below +0.0).
// Other predicates or a -0.0 constant differ on one of those inputs and are
// left alone.
struct SelectOfCompareToMaximum : OpRewritePattern<arith::SelectOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::SelectOp select, PatternRewriter &rewriter) const override {
    auto compare = select.getCondition().getDefiningOp<arith::CmpFOp>();
    if (!compare || compare.getPredicate() != arith::CmpFPredicate::UGT) return failure();
    APFloat zero(0.0);
    if (!matchPattern(compare.getRhs(), m_ConstantFloat(&zero)) || !zero.isPosZero())
      return failure();
    if (select.getTrueValue() != compare.getLhs() || select.getFalseValue() != compare.getRhs())
      return failure();
    rewriter.replaceOpWithNewOp<arith::MaximumFOp>(select, compare.getLhs(), compare.getRhs());
    return success();
  }
};

// The FP16 tensor a widening generic (`extf` only, identity maps) reads, or null.
Value widenedSource(Value value) {
  auto gen = value.getDefiningOp<linalg::GenericOp>();
  if (!gen || gen.getNumDpsInputs() != 1 || gen.getNumDpsInits() != 1 ||
      !llvm::all_of(gen.getIndexingMapsArray(), [](AffineMap map) { return map.isIdentity(); }))
    return {};
  Block &body = gen.getRegion().front();
  auto ops = llvm::to_vector(llvm::map_range(body.without_terminator(), [](Operation &o) { return &o; }));
  auto input = gen.getDpsInputs()[0];
  auto type = dyn_cast<RankedTensorType>(input.getType());
  if (ops.size() != 1 || !isa<arith::ExtFOp>(ops[0]) || ops[0]->getOperand(0) != body.getArgument(0) ||
      cast<linalg::YieldOp>(body.getTerminator()).getValues()[0] != ops[0]->getResult(0) ||
      !type || !type.getElementType().isF16())
    return {};
  return input;
}

// torch-mlir lowers an FP16 addmm (Linear with bias) as matmul(extf a, extf b)
// in FP32. Widening FP16 is exact and M_MMA_F16F16F32 multiplies FP16 operands
// into an FP32 accumulator, so reading the FP16 originals computes the same
// products and sums: matmul(a, b) -> f32 is the identical function.
struct MatmulOfWidenedInputs : OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::MatmulOp mm, PatternRewriter &rewriter) const override {
    SmallVector<Value> narrow;
    for (Value input : mm.getDpsInputs()) {
      Value source = widenedSource(input);
      if (!source) return failure();
      narrow.push_back(source);
    }
    rewriter.replaceOpWithNewOp<linalg::MatmulOp>(mm, mm->getResultTypes(), narrow, mm.getDpsInits());
    return success();
  }
};

// Upstream linalg elementwise fusion, restricted to producers that only widen
// or broadcast (a body of extf ops at most) and that are rank-1 or broadcast:
// a bias vector the capture widened and then materialised as a full matrix
// becomes an operand the consumer reads per channel (map (i,j) -> (j)).
// Rank-2 widenings stay separate; narrowMatrixChains looks for them.
void populateBroadcastFusion(RewritePatternSet &patterns) {
  linalg::populateElementwiseOpsFusionPatterns(patterns, [](OpOperand *use) {
    auto producer = use->get().getDefiningOp<linalg::GenericOp>();
    auto consumer = dyn_cast<linalg::GenericOp>(use->getOwner());
    if (!producer || !consumer || producer.getNumDpsInits() != 1) return false;
    auto parallel = [](linalg::GenericOp g) {
      return llvm::all_of(g.getIteratorTypesArray(),
                          [](utils::IteratorType t) { return t == utils::IteratorType::parallel; });
    };
    if (!parallel(producer) || !parallel(consumer)) return false;
    if (!llvm::all_of(producer.getRegion().front().without_terminator(),
                      [](Operation &op) { return isa<arith::ExtFOp>(op); }))
      return false;
    auto result = cast<ShapedType>(producer.getResult(0).getType());
    bool broadcast = llvm::any_of(producer.getIndexingMapsArray(),
                                  [](AffineMap map) { return !map.isIdentity(); });
    return result.getRank() == 1 || broadcast;
  });
}

// --fp16: a matmul whose FP32 result goes on into more arithmetic (a bias add)
// instead of straight into a truncf is rounded at M_WRITEOUT_F16 and widened
// back for its users. That is one intermediate FP16 rounding, the policy --fp16
// grants; narrowMatrixChains then runs the users in FP16.
LogicalResult roundAccumulatorsAtWriteout(OpBuilder &b, ModuleOp m) {
  if (m->getAttrOfType<StringAttr>("npu.numerical") != StringAttr::get(m.getContext(), "fp16"))
    return success();
  SmallVector<linalg::MatmulOp> work;
  m.walk([&](linalg::MatmulOp mm) { work.push_back(mm); });
  for (auto mm : work) {
    auto init = dyn_cast<RankedTensorType>(mm.getDpsInitOperand(0)->get().getType());
    if (!init || !init.hasStaticShape() || !init.getElementType().isF32() ||
        !isZeroFilled(mm.getDpsInitOperand(0)->get()))
      continue;
    if (!llvm::all_of(mm.getDpsInputs(), [](Value v) {
          auto t = dyn_cast<RankedTensorType>(v.getType());
          return t && t.getElementType().isF16();
        }))
      continue;
    auto loc = mm.getLoc();
    b.setInsertionPoint(mm);
    auto zero = b.create<arith::ConstantOp>(loc, b.getF16FloatAttr(0.0f));
    auto empty = b.create<tensor::EmptyOp>(loc, init.getShape(), b.getF16Type());
    auto filled = b.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{empty});
    auto narrow = b.create<linalg::MatmulOp>(loc, TypeRange{empty.getType()}, mm.getDpsInputs(),
                                             ValueRange{filled.getResult(0)});
    auto wide = b.create<tensor::EmptyOp>(loc, init.getShape(), b.getF32Type());
    auto identity = b.getMultiDimIdentityMap(init.getRank());
    SmallVector<utils::IteratorType> parallel(init.getRank(), utils::IteratorType::parallel);
    auto widen = b.create<linalg::GenericOp>(
        loc, TypeRange{wide.getType()}, ValueRange{narrow.getResult(0)}, ValueRange{wide},
        ArrayRef<AffineMap>{identity, identity}, parallel,
        [](OpBuilder &nested, Location at, ValueRange args) {
          Value ext = nested.create<arith::ExtFOp>(at, nested.getF32Type(), args[0]);
          nested.create<linalg::YieldOp>(at, ext);
        });
    mm.getResult(0).replaceAllUsesWith(widen.getResult(0));
    mm.erase();
  }
  return success();
}
} // namespace

namespace {
/// batch_matmul 을 배치 항목마다 matmul 하나로 쪼갠다.
///
/// **이름과 달리 배치 크기를 가리지 않는다.** 배치가 2 면 matmul 이 둘,
/// 32 면 서른둘 나온다. 시스톨릭 배열에는 배치 축이라는 것이 없으므로
/// 어느 쪽이든 결국 따로 돌아야 한다. (이름은 splitBatchMatmuls 가 맞다.)
///
/// 각 항목은 extract_slice 로 rank-2 를 떼어내 계산하고 insert_slice 로
/// 제자리에 돌려놓는다. 어텐션 헤드가 정확히 이 모양이다 — 피연산자 배치를
/// 공유하는 독립된 행렬곱들.
LogicalResult dropUnitBatchMatmuls(OpBuilder &b, ModuleOp m) {
  SmallVector<linalg::BatchMatmulOp> work;
  m.walk([&](linalg::BatchMatmulOp op) { work.push_back(op); });
  for (auto op : work) {
    SmallVector<Value> operands(op.getDpsInputs());
    operands.push_back(op.getDpsInits()[0]);
    auto result = cast<RankedTensorType>(op.getResult(0).getType());
    if (result.getRank() != 3 || !result.hasStaticShape())
      return op.emitError("a batch matmul needs a static rank-3 result");
    int64_t batch = result.getDimSize(0);
    for (Value value : operands) {
      auto type = dyn_cast<RankedTensorType>(value.getType());
      if (!type || type.getRank() != 3 || !type.hasStaticShape() || type.getDimSize(0) != batch)
        return op.emitError("every batch matmul operand needs the same static batch");
    }
    b.setInsertionPoint(op);
    Value assembled = op.getDpsInits()[0];
    for (int64_t item = 0; item < batch; ++item) {
      SmallVector<Value> slices;
      for (Value value : operands) {
        auto type = cast<RankedTensorType>(value.getType());
        SmallVector<OpFoldResult> offsets{b.getIndexAttr(item),b.getIndexAttr(0),b.getIndexAttr(0)};
        SmallVector<OpFoldResult> sizes{b.getIndexAttr(1),b.getIndexAttr(type.getDimSize(1)),
                                        b.getIndexAttr(type.getDimSize(2))};
        SmallVector<OpFoldResult> strides(3,b.getIndexAttr(1));
        auto flat = RankedTensorType::get(type.getShape().drop_front(),type.getElementType());
        slices.push_back(b.create<tensor::ExtractSliceOp>(op.getLoc(),flat,value,offsets,sizes,strides));
      }
      auto flatResult = RankedTensorType::get(result.getShape().drop_front(),result.getElementType());
      auto one = b.create<linalg::MatmulOp>(op.getLoc(),TypeRange{flatResult},
          ValueRange{slices[0],slices[1]},ValueRange{slices[2]});
      SmallVector<OpFoldResult> offsets{b.getIndexAttr(item),b.getIndexAttr(0),b.getIndexAttr(0)};
      SmallVector<OpFoldResult> sizes{b.getIndexAttr(1),b.getIndexAttr(result.getDimSize(1)),
                                      b.getIndexAttr(result.getDimSize(2))};
      SmallVector<OpFoldResult> strides(3,b.getIndexAttr(1));
      assembled = b.create<tensor::InsertSliceOp>(op.getLoc(),one.getResult(0),assembled,
                                                  offsets,sizes,strides);
    }
    op.getResult(0).replaceAllUsesWith(assembled);
    op.erase();
  }
  return success();
}

// torch 는 `linear(x, W)` 를 transpose(W) 뒤에 matmul 로 낮춘다. 가중치를
// 실행 시점에 전치하면 원소마다 스트라이드 전송이 한 번씩 들고, 게다가 그
// 가중치는 패키지가 직접 쓰지 않는 진입 인자다. 그래서 패키지가 **이미 전치된**
// 인자를 요구하게 하고 전치 자체를 없앤다.
LogicalResult pretransposeWeights(OpBuilder &b, ModuleOp m) {
  auto function = *m.getOps<func::FuncOp>().begin();
  SmallVector<int64_t> indices;
  for (auto arg : function.getArguments()) {
    auto type = dyn_cast<RankedTensorType>(arg.getType());
    if (!type || type.getRank() != 2 || !type.hasStaticShape() || arg.use_empty()) continue;
    SmallVector<linalg::TransposeOp> users;
    for (auto *user : arg.getUsers()) {
      auto t = dyn_cast<linalg::TransposeOp>(user);
      // 다른 독자가 하나라도 있으면 전치 안 된 바이트가 여전히 필요하다는 뜻이다.
      if (!t || t.getPermutation() != ArrayRef<int64_t>{1, 0}) { users.clear(); break; }
      users.push_back(t);
    }
    if (users.empty()) continue;
    arg.setType(RankedTensorType::get({type.getDimSize(1), type.getDimSize(0)},
                                      type.getElementType()));
    for (auto t : users) { t->getResult(0).replaceAllUsesWith(arg); t.erase(); }
    indices.push_back(arg.getArgNumber());
  }
  if (indices.empty()) return success();
  SmallVector<Type> inputs(function.getFunctionType().getInputs());
  for (auto i : indices) inputs[i] = function.getArgument(i).getType();
  function.setType(b.getFunctionType(inputs, function.getFunctionType().getResults()));
  m->setAttr("npu.pretransposed", b.getDenseI64ArrayAttr(indices));
  return success();
}

// 타깃 단계는 메모리를 2차원 뷰로 주소 지정하므로, 그보다 높은 차원의 순열은
// 먼저 rank-2 작업으로 바뀌어야 한다. 순열이 제자리에 두는 앞쪽 차원은 루프가
// 되고, (d1,d0,d2) 처럼 행을 통째로 옮기기만 하는 것은 사각형 복사라서 전치
// 경로를 아예 타지 않는다.
LogicalResult decomposeTransposes(OpBuilder &b, ModuleOp m) {
  SmallVector<linalg::TransposeOp> work;
  m.walk([&](linalg::TransposeOp t) { work.push_back(t); });
  while (!work.empty()) {
    auto t = work.pop_back_val();
    auto in = cast<RankedTensorType>(t.getInput().getType());
    auto out = cast<RankedTensorType>(t.getInit().getType());
    ArrayRef<int64_t> perm = t.getPermutation();
    if (in.getRank() <= 2) continue;
    if (!in.hasStaticShape()) return t.emitError("a transpose needs a static shape");
    b.setInsertionPoint(t);
    auto loc = t.getLoc();
    auto one = b.getIndexAttr(1);
    // `axis` 방향 `index` 위치의 초평면 하나를 떼어내면서 그 축을 없앤다.
    auto slice = [&](Value value, int64_t axis, int64_t index) -> Value {
      auto type = cast<RankedTensorType>(value.getType());
      SmallVector<OpFoldResult> offsets(type.getRank(), b.getIndexAttr(0)),
          sizes, strides(type.getRank(), one);
      for (auto d : type.getShape()) sizes.push_back(b.getIndexAttr(d));
      offsets[axis] = b.getIndexAttr(index);
      sizes[axis] = one;
      SmallVector<int64_t> shape(type.getShape());
      shape.erase(shape.begin() + axis);
      return b.create<tensor::ExtractSliceOp>(
          loc, RankedTensorType::get(shape, type.getElementType()), value, offsets, sizes, strides);
    };
    auto place = [&](Value piece, Value into, int64_t axis, int64_t index) -> Value {
      auto type = cast<RankedTensorType>(into.getType());
      SmallVector<OpFoldResult> offsets(type.getRank(), b.getIndexAttr(0)),
          sizes, strides(type.getRank(), one);
      for (auto d : type.getShape()) sizes.push_back(b.getIndexAttr(d));
      offsets[axis] = b.getIndexAttr(index);
      sizes[axis] = one;
      return b.create<tensor::InsertSliceOp>(loc, piece, into, offsets, sizes, strides);
    };
    Value assembled = t.getInit();
    if (perm[0] == 0) {
      // 가장 바깥 축은 제자리에 있으므로, 그 평면들이 각각 따로 전치된다.
      SmallVector<int64_t> inner(perm.drop_front());
      for (auto &p : inner) --p;
      for (int64_t i = 0, e = in.getDimSize(0); i < e; ++i) {
        SmallVector<int64_t> shape(out.getShape().drop_front());
        auto empty = b.create<tensor::EmptyOp>(loc, shape, out.getElementType());
        auto sub = b.create<linalg::TransposeOp>(loc, slice(t.getInput(), 0, i), empty, inner);
        work.push_back(sub);
        assembled = place(sub.getResult()[0], assembled, 0, i);
      }
    } else if (in.getRank() == 3 && perm == ArrayRef<int64_t>{1, 0, 2}) {
      // 행을 보존하는 경우 — 결과의 평면 j 는 원본을 스트라이드로 읽은 것이다.
      for (int64_t j = 0, e = in.getDimSize(1); j < e; ++j)
        assembled = place(slice(t.getInput(), 1, j), assembled, 0, j);
    } else {
      return t.emitError("this transpose permutation is not yet decomposed");
    }
    t.getResult()[0].replaceAllUsesWith(assembled);
    t.erase();
  }
  return success();
}

// 모든 스트림 명령은 2차원 창을 주소 지정한다. 그래서 축이 더 많은 원소별
// 연산은 먼저 reshape 하거나 쪼개야 한다.
//
// 모든 피연산자가 보조를 맞춰 훑는 축은 **접는다** — 비용이 0 이다. 어떤
// 피연산자가 broadcast 하는 축은 옆 축과 함께 접을 수 없으므로 **벗겨낸다**:
// 인덱스마다 op 하나씩 만들고, broadcast 피연산자는 자기 평면 하나에 고정한다.
LogicalResult flattenElementwise(OpBuilder &b, ModuleOp m) {
  SmallVector<linalg::GenericOp> work;
  m.walk([&](linalg::GenericOp gen) { work.push_back(gen); });
  while (!work.empty()) {
    auto gen = work.pop_back_val();
    if (gen.getNumDpsInits() != 1 || gen.getNumResults() != 1) continue;
    auto out = dyn_cast<RankedTensorType>(gen.getResult(0).getType());
    unsigned loops = gen.getNumLoops();
    // 스트림 명령은 전부 루프 두 개짜리 모양이다. 리덕션은 결과의 축보다 루프가
    // 많을 수 있으므로, 판단 기준은 결과가 아니라 루프 개수다.
    if (!out || loops <= 2 || !out.hasStaticShape()) continue;
    if (llvm::any_of(gen->getOperands(), [](Value v) {
          auto t = dyn_cast<RankedTensorType>(v.getType());
          return !t || !t.hasStaticShape();
        }))
      continue;
    if (gen.getIteratorTypesArray()[0] != utils::IteratorType::parallel) continue;
    auto maps = gen.getIndexingMapsArray();
    auto loc = gen.getLoc();
    b.setInsertionPoint(gen);
    // 모든 피연산자가 같은 순서로 훑는 앞쪽 축들은 사실상 한 축이다. 합치는 것은
    // 순수한 reshape 이고, op 을 쪼개지 않고 통째로 남긴다.
    unsigned lead = loops - 1;
    auto iterators = gen.getIteratorTypesArray();
    bool mergeable = llvm::all_of(ArrayRef<utils::IteratorType>(iterators).take_front(lead),
        [](utils::IteratorType it) { return it == utils::IteratorType::parallel; });
    for (auto map : maps) {
      if (map.getNumResults() < lead) { mergeable = false; break; }
      for (unsigned d = 0; d < lead; ++d)
        if (map.getResult(d) != getAffineDimExpr(d, b.getContext())) { mergeable = false; break; }
      if (!mergeable) break;
    }
    if (mergeable) {
      auto grouping = [&](unsigned rank) {
        SmallVector<ReassociationIndices> groups;
        ReassociationIndices merged;
        for (unsigned d = 0; d < lead; ++d) merged.push_back(d);
        groups.push_back(merged);
        for (unsigned d = lead; d < rank; ++d) groups.push_back({int64_t(d)});
        return groups;
      };
      SmallVector<Value> operands;
      SmallVector<AffineMap> flatMaps;
      for (auto [k, value] : llvm::enumerate(gen->getOperands())) {
        unsigned rank = cast<RankedTensorType>(value.getType()).getRank();
        operands.push_back(b.create<tensor::CollapseShapeOp>(loc, value, grouping(rank)));
        SmallVector<AffineExpr> results{getAffineDimExpr(0, b.getContext())};
        for (auto e : maps[k].getResults().drop_front(lead))
          results.push_back(e.shiftDims(loops, unsigned(-int(lead) + 1), lead));
        flatMaps.push_back(AffineMap::get(loops - lead + 1, 0, results, b.getContext()));
      }
      SmallVector<utils::IteratorType> flatIterators{utils::IteratorType::parallel};
      for (auto it : ArrayRef<utils::IteratorType>(iterators).drop_front(lead))
        flatIterators.push_back(it);
      auto flat = cast<RankedTensorType>(operands.back().getType());
      auto one = b.create<linalg::GenericOp>(loc, TypeRange{flat},
          ValueRange(operands).drop_back(), ValueRange{operands.back()}, flatMaps, flatIterators);
      IRMapping mapping;
      gen.getRegion().cloneInto(&one.getRegion(), mapping);
      auto grown = b.create<tensor::ExpandShapeOp>(loc, out, one.getResult(0),
                                                   grouping(unsigned(out.getRank())));
      gen.getResult(0).replaceAllUsesWith(grown);
      gen.erase();
      work.push_back(one);
      continue;
    }
    // 그렇지 않으면 가장 바깥 축을 벗겨낸다. 각 피연산자는 그 축을 훑거나,
    // 그 축의 평면 하나에 고정되거나, 아예 그 축을 보지 않는다.
    auto zero = getAffineConstantExpr(0, b.getContext());
    auto first = getAffineDimExpr(0, b.getContext());
    SmallVector<int64_t> takeAxis(maps.size(), -1);
    for (auto [i, map] : llvm::enumerate(maps)) {
      bool uses = map.isFunctionOfDim(0);
      if (map.getNumResults() && map.getResult(0) == first) takeAxis[i] = 0;
      else if (!uses && map.getNumResults() && map.getResult(0) == zero) takeAxis[i] = 0;
      else if (!uses) takeAxis[i] = -1;
      else return gen.emitError("this elementwise broadcast is not yet flattened");
    }
    if (takeAxis.back() != 0) return gen.emitError("an elementwise result must walk its outer axis");
    SmallVector<AffineMap> inner;
    for (auto [i, map] : llvm::enumerate(maps)) {
      auto results = map.getResults();
      if (takeAxis[i] == 0) results = results.drop_front();
      SmallVector<AffineExpr> shifted;
      for (auto e : results) shifted.push_back(e.shiftDims(loops, unsigned(-1), 1));
      inner.push_back(AffineMap::get(loops - 1, 0, shifted, b.getContext()));
    }
    SmallVector<utils::IteratorType> inner_iterators(
        ArrayRef<utils::IteratorType>(iterators).drop_front());
    auto plane = [&](Value value, int64_t index) -> Value {
      auto type = cast<RankedTensorType>(value.getType());
      SmallVector<OpFoldResult> offsets(type.getRank(), b.getIndexAttr(0)), sizes,
          strides(type.getRank(), b.getIndexAttr(1));
      for (auto d : type.getShape()) sizes.push_back(b.getIndexAttr(d));
      offsets[0] = b.getIndexAttr(index);
      sizes[0] = b.getIndexAttr(1);
      return b.create<tensor::ExtractSliceOp>(loc,
          RankedTensorType::get(type.getShape().drop_front(), type.getElementType()),
          value, offsets, sizes, strides);
    };
    Value assembled = gen.getDpsInits()[0];
    auto innerOut = RankedTensorType::get(out.getShape().drop_front(), out.getElementType());
    for (int64_t i = 0, e = out.getDimSize(0); i < e; ++i) {
      SmallVector<Value> operands;
      for (auto [k, value] : llvm::enumerate(gen->getOperands()))
        operands.push_back(takeAxis[k] == 0
            ? plane(value, maps[k].isFunctionOfDim(0) ? i : 0) : value);
      auto one = b.create<linalg::GenericOp>(loc, TypeRange{innerOut},
          ValueRange(operands).drop_back(), ValueRange{operands.back()}, inner, inner_iterators);
      IRMapping mapping;
      gen.getRegion().cloneInto(&one.getRegion(), mapping);
      SmallVector<OpFoldResult> offsets(out.getRank(), b.getIndexAttr(0)), sizes,
          strides(out.getRank(), b.getIndexAttr(1));
      for (auto d : out.getShape()) sizes.push_back(b.getIndexAttr(d));
      offsets[0] = b.getIndexAttr(i);
      sizes[0] = b.getIndexAttr(1);
      assembled = b.create<tensor::InsertSliceOp>(loc, one.getResult(0), assembled,
                                                  offsets, sizes, strides);
      work.push_back(one);
    }
    gen.getResult(0).replaceAllUsesWith(assembled);
    gen.erase();
  }
  return success();
}

// RoPE 는 회전된 절반을 concat 으로 만든다. 그런 전송 명령은 없다. 대신 앞쪽
// 단계가 각 조각을 결과의 제 슬라이스에 쓰는 법을 이미 알고 있고, 배치 단계도
// 어차피 그 모양을 보고 싶어 한다.
LogicalResult decomposeConcats(ModuleOp m) {
  RewritePatternSet patterns(m.getContext());
  tensor::populateDecomposeTensorConcatPatterns(patterns);
  return applyPatternsGreedily(m, std::move(patterns));
}

// 배열은 출력 타일마다 누산기를 스스로 지우고, V_REDUCE 는 행에 더하는 게
// 아니라 새로 쓴다. 그러므로 **아무도 읽기 전에 모든 바이트가 다시 쓰이는**
// 초기화는 실제로 돌 필요가 없다.
//
// ISA 에는 상수를 메모리에 쓰는 명령이 아예 없다. 그래서 이 증명이야말로
// 그런 fill 을 합법으로 만들어 주는 근거다.
LogicalResult markOverwrittenFills(OpBuilder &b, ModuleOp m) {
  // 하드웨어가 어차피 버리는 초기 바이트를 읽는 경우.
  auto discarded = [](OpOperand &use) {
    Operation *owner = use.getOwner();
    if (auto mm = dyn_cast<linalg::MatmulOp>(owner))
      return mm.getDpsInitOperand(0) == &use;
    auto gen = dyn_cast<linalg::GenericOp>(owner);
    return gen && gen.getNumDpsInits() == 1 && gen.getDpsInitOperand(0) == &use &&
           llvm::is_contained(gen.getIteratorTypesArray(), utils::IteratorType::reduction);
  };
  // 초기 바이트의 슬라이스는 그런 읽기에만 닿을 수 있다.
  std::function<bool(Value)> sliceDiscarded = [&](Value value) {
    for (OpOperand &use : value.getUses()) {
      if (discarded(use)) continue;
      if (isa<tensor::ExtractSliceOp, tensor::CollapseShapeOp, tensor::ExpandShapeOp>(use.getOwner()) &&
          sliceDiscarded(use.getOwner()->getResult(0)))
        continue;
      return false;
    }
    return true;
  };
  // 쓰기를 따라간다. 진짜 읽기라면 가장 바깥 축 전체를 덮는 쓰기들보다 뒤에
  // 와야 한다. 나머지는 쓰기이거나 죽는 값이다.
  std::function<bool(Value, SmallVector<bool>)> covered = [&](Value value,
                                                              SmallVector<bool> written) {
    auto type = cast<RankedTensorType>(value.getType());
    for (OpOperand &use : value.getUses()) {
      Operation *owner = use.getOwner();
      if (discarded(use)) continue;
      auto insert = dyn_cast<tensor::InsertSliceOp>(owner);
      if (insert && insert.getDestMutable().getOperandNumber() == use.getOperandNumber()) {
        auto offsets = insert.getStaticOffsets(), sizes = insert.getStaticSizes(),
             steps = insert.getStaticStrides();
        if (llvm::any_of(steps, [](int64_t s) { return s != 1; })) return false;
        for (unsigned d = 1; d < offsets.size(); ++d)
          if (offsets[d] != 0 || sizes[d] != type.getDimSize(d)) return false;
        if (offsets[0] < 0 || sizes[0] < 0 || offsets[0] + sizes[0] > int64_t(written.size()))
          return false;
        auto next = written;
        for (int64_t i = offsets[0]; i < offsets[0] + sizes[0]; ++i) next[i] = true;
        if (!covered(insert.getResult(), next)) return false;
        continue;
      }
      if (isa<tensor::ExtractSliceOp, tensor::CollapseShapeOp, tensor::ExpandShapeOp>(owner) &&
          sliceDiscarded(owner->getResult(0)))
        continue;
      // 진짜 읽기다 — 그것이 볼 수 있는 모든 바이트가 이미 다시 쓰여 있어야 한다.
      if (!llvm::all_of(written, [](bool w) { return w; })) return false;
    }
    return true;
  };
  m.walk([&](linalg::FillOp fill) {
    auto result = fill.getResult(0);
    auto type = dyn_cast<RankedTensorType>(result.getType());
    if (!type || !type.hasStaticShape() || type.getRank() < 1) return;
    // 행렬곱은 누산기를 지우기만 하므로, 거기서는 0 만 떼어낼 수 있다.
    auto value = fill.getInputs()[0].getDefiningOp<arith::ConstantOp>();
    auto attr = value ? dyn_cast<FloatAttr>(value.getValue()) : FloatAttr();
    bool zero = attr && attr.getValue().isZero();
    bool feedsMatmul = false;
    fill->getParentOp()->walk([&](linalg::MatmulOp mm) {
      if (mm.getDpsInitOperand(0)->get().getDefiningOp() == fill.getOperation()) feedsMatmul = true;
    });
    if (feedsMatmul && !zero) return;
    if (covered(result, SmallVector<bool>(type.getDimSize(0), false)))
      fill->setAttr("npu.tile_zero", b.getUnitAttr());
  });
  return success();
}
} // namespace

/// Stage 1 전반 — Stage 0 캡처를 이 컴파일러가 아는 모양으로 바꾼다.
///
/// **순서가 의미를 가진다.** 한 재작성이 다른 재작성의 전제를 만들거나 깨뜨리고,
/// 그래서 같은 재작성이 여러 번 나오기도 한다. 아래 각 단계의 주석은 "왜 지금
/// 여기인지"를 적은 것이다.
///
/// 이 함수가 남긴 IR 은 곧바로 legalizeGraph 가 검사한다. 거기서 거부되면
/// 단계 파일이 하나도 안 써지므로, 무엇이 나왔는지 보려면 환경변수
/// `NPU_DUMP_NORMALIZED=1` 을 쓴다(맨 아래).
LogicalResult normalizeGraph(ModuleOp m) {
  OpBuilder batched(m.getContext());

  // ── ⓪ FP32 로 넓혀 곱한 행렬곱을 FP16 피연산자로 ──────────────────
  // ①의 누적·반올림 접기는 FP16 피연산자를 요구하므로 그보다 먼저 한다.
  {
    RewritePatternSet widened(m.getContext());
    widened.add<MatmulOfWidenedInputs>(m.getContext());
    if (failed(applyPatternsGreedily(m, std::move(widened)))) return failure();
  }

  // ── ① 누적·반올림 쌍 접기 ─────────────────────────────────────────
  // 무엇이 reshape 하거나 slice 하기 전에 해야 한다. 지금은 그 한 쌍이 아직
  // 나란히 붙어 있고, 뒤에 오는 모든 재작성이 그 사이에 뭔가를 끼워 넣는다.
  if (failed(foldAccumulateAndRound<linalg::BatchMatmulOp>(batched, m))) return failure();
  if (failed(foldAccumulateAndRound<linalg::MatmulOp>(batched, m))) return failure();
  if (failed(cleanIR(m))) return failure();

  // ── ② 배치 차원 정리 ──────────────────────────────────────────────
  // foldBatchDimension 은 **크기가 1 인 축**을 없앤다. 1x32x32 도 32x1x32 도
  // 32x32 가 된다. 크기가 1 이 아닌 축은 건드리지 않으므로 32x32x32 에는
  // 아무 일도 일어나지 않고, op 개수도 바뀌지 않는다.
  //
  // (batch_matmul 을 matmul 로 바꾸는 것은 아래 dropUnitBatchMatmuls 이고,
  //  32x32x32 원소별 연산을 1024x32 로 합치는 것은 ③의 flattenElementwise 다.)
  //
  // 접고 나면 rank 가 맞아져서, 그 전에는 모양이 달라 접지 못하던
  // 누적·반올림 쌍이 접히게 된다. 그래서 ①을 다시 돌린다.
  if (failed(foldBatchDimension(m))) return failure();
  if (failed(foldAccumulateAndRound<linalg::MatmulOp>(batched, m))) return failure();
  if (failed(foldAccumulateAndRound<linalg::BatchMatmulOp>(batched, m))) return failure();
  if (failed(cleanIR(m))) return failure();

  // batch_matmul 을 항목마다 matmul 하나로 쪼갠다. 배치가 1 이든 32 든
  // 마찬가지다 — 시스톨릭 배열에 배치 축이 없으니 어차피 따로 돌아야 한다.
  //
  // 그러고 나서 배치 접기를 한 번 더 부른다. batch_matmul 은 rank-3 피연산자를
  // 요구하므로 그것이 살아 있는 동안에는 주변 단위 축을 접을 수 없었는데,
  // 이제 그 제약이 사라졌기 때문이다.
  if (failed(dropUnitBatchMatmuls(batched, m))) return failure();
  if (failed(foldBatchDimension(m))) return failure();

  // ①이 접지 못한 FP32 누산 결과(뒤에 bias 덧셈 등이 붙은 것)는 --fp16 이면
  // M_WRITEOUT_F16 에서 반올림한다. 그리고 넓혀서 펼쳐 둔 bias 는 소비자가
  // 채널별로 직접 읽게 융합한다(linalg elementwise fusion).
  if (failed(roundAccumulatorsAtWriteout(batched, m))) return failure();
  {
    RewritePatternSet fusion(m.getContext());
    populateBroadcastFusion(fusion);
    if (failed(applyPatternsGreedily(m, std::move(fusion)))) return failure();
  }
  if (failed(cleanIR(m))) return failure();

  // ── ③ 모양을 바꾸는 재작성들 ──────────────────────────────────────
  if (failed(decomposeConcats(m))) return failure();      // concat → 조각별 슬라이스 쓰기
  if (failed(pretransposeWeights(batched, m))) return failure();   // 전치를 미리 접어 없앤다
  if (failed(decomposeTransposes(batched, m))) return failure();   // 남은 전치를 분해
  if (failed(flattenElementwise(batched, m))) return failure();    // 원소별 연산을 rank-2 로
  if (failed(cleanIR(m))) return failure();

  // ── ④ 정수 거듭제곱 전개 ──────────────────────────────────────────
  OpBuilder powers(m.getContext());
  if (failed(expandIntegerPowers(powers, m))) return failure();

  // ── ④' ReLU 를 최댓값으로 ─────────────────────────────────────────
  // ⑦의 상수 올리기가 +0.0 을 broadcast 행으로 바꾸기 전에 해야 한다 —
  // 패턴이 상수 피연산자를 알아봐야 하기 때문이다.
  RewritePatternSet relu(m.getContext());
  relu.add<SelectOfCompareToMaximum>(m.getContext());
  if (failed(applyPatternsGreedily(m, std::move(relu)))) return failure();

  // ── ⑤ 아무도 안 읽는 결과 떼어내기 ────────────────────────────────
  // torch 의 softmax 는 행 최댓값을 "최댓값과 argmax 인덱스를 함께 내놓는
  // generic" 으로 낮춘다. 인덱스는 아무도 읽지 않는다. 쓰이지 않는 결과를
  // 지워야 행 경로가 알아보는 단일 결과 모양이 된다.
  RewritePatternSet patterns(m.getContext());
  linalg::populateEraseUnusedOperandsAndResultsPatterns(patterns);
  if (failed(applyPatternsGreedily(m, std::move(patterns)))) return failure();
  if (failed(cleanIR(m))) return failure();

  // ── ⑥ FP16 행 리덕션을 FP32 로 넓히기 ─────────────────────────────
  // 하드웨어에 FP16 누적 명령이 없다. 먼저 대상을 모아 두고 고친다 —
  // 고치는 도중에 IR 이 바뀌므로 walk 하면서 바로 고칠 수 없다.
  OpBuilder b(m.getContext());
  SmallVector<linalg::GenericOp> reductions;
  m.walk([&](linalg::GenericOp gen) {
    auto init = dyn_cast<RankedTensorType>(gen.getDpsInitOperand(0)->get().getType());
    if (reducesRows(gen) && init && init.getElementType().isF16())
      reductions.push_back(gen);
  });
  for (auto gen : reductions)
    if (failed(widenRowChain(b, gen))) return failure();

  // 행 체인을 넓히면 그 생산자들이 다시 만들어진다. 그 과정에서 축이 둘보다
  // 많아진 것이 있으면, FP16 좁히기가 보기 전에 다시 평탄화한다.
  if (failed(flattenElementwise(b, m))) return failure();
  if (failed(cleanIR(m))) return failure();

  // ── ⑦ FP16 으로 좁히기 ────────────────────────────────────────────
  if (failed(dropRowReshapes(m))) return failure();
  if (failed(cleanIR(m))) return failure();
  if (failed(narrowMatrixChains(b, m))) return failure();
  if (failed(hoistRegionScalars(b, m))) return failure();
  if (failed(cleanIR(m))) return failure();
  if (failed(foldAccumulateAndRound<linalg::MatmulOp>(b, m))) return failure();
  if (failed(cleanIR(m))) return failure();

  // 좁히기는 새 원소 타입을 reshape 를 통해 나르므로 축이 다시 늘어난다.
  // 이것이 마지막 재작성이다 — 여기서 남은 모양이 곧 lowering 이 받는 모양이다.
  if (failed(flattenElementwise(b, m))) return failure();
  if (failed(cleanIR(m))) return failure();

  // ── ⑧ 덮어쓰이는 fill 표시 ────────────────────────────────────────
  if (failed(markOverwrittenFills(b, m))) return failure();

  // legalizeGraph 는 단계 파일이 하나라도 써지기 전에 거부하므로, 정규화가
  // 실제로 무엇을 만들었는지 볼 방법은 이것뿐이다.
  if (const char *dump = getenv("NPU_DUMP_NORMALIZED"); dump && *dump) {
    llvm::errs() << "// --- after normalisation ---\n";
    m.print(llvm::errs());
  }
  return success();
}
} // namespace npu
