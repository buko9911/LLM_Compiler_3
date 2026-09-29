//===----------------------------------------------------------------------===//
// Legalize.cpp — Stage 1 후반 — 받을 수 있는 그래프인지 검사
//===----------------------------------------------------------------------===//
#include "GraphInternal.h"

using namespace mlir;
namespace npu {
namespace {
// 합법성 검사와 ISA 선택이 같은 표 하나를 쓴다. 그래서 대응하는 VPU 명령이
// 없는 스칼라 연산은 통과할 수가 없다.
const char *scalarVectorInstruction(Operation *op) {
  if (isa<arith::MulFOp>(op)) return "V_MUL_SCALAR_F16";
  if (isa<arith::AddFOp>(op)) return "V_ADD_SCALAR_F16";
  if (isa<arith::SubFOp>(op)) return "V_SUB_SCALAR_F16";
  if (isa<arith::MaximumFOp>(op)) return "V_MAX_SCALAR_F16";
  if (isa<arith::MinimumFOp>(op)) return "V_MIN_SCALAR_F16";
  return nullptr;
}
const char *vectorInstruction(Operation *op) {
  if (isa<arith::MulFOp>(op)) return "V_MUL_F16";
  if (isa<arith::AddFOp>(op)) return "V_ADD_F16";
  if (isa<arith::SubFOp>(op)) return "V_SUB_F16";
  if (isa<arith::MaximumFOp>(op)) return "V_MAX_F16";
  if (isa<arith::MinimumFOp>(op)) return "V_MIN_F16";
  if (isa<arith::NegFOp>(op)) return "V_NEG_F16";
  if (isa<math::ExpOp>(op)) return "V_EXP_F16";
  if (isa<math::SqrtOp>(op)) return "V_SQRT_F16";
  if (isa<math::RsqrtOp>(op)) return "V_RSQRT_F16";
  if (isa<math::AbsFOp>(op)) return "V_ABS_F16";
  return nullptr;
}
// 스트림은 처음부터 끝까지 FP16 이고 broadcast 스칼라는 FP32 로 남는다.
// 그래서 region 안의 폭 변환은 명령이 아니라 **이미 있는 값의 다른 이름**이다.
// --fp16 이 허용하는 것이 정확히 이것이고, 없으면 이 캐스트는 거부해야 한다.
bool isStreamWidthCast(Operation *op) { return isa<arith::ExtFOp,arith::TruncFOp>(op); }
// 하드웨어에 나눗셈이 없다. a/b 는 a * (1/b) 가 되는데, 이것은 같은 연산을
// 반올림한 것이 아니라 **다른 연산**이다. 그래서 --fp16 에 묻어가지 않고
// 따로 --reciprocal-division 으로 켜야 한다.
bool divisionIsReciprocal(Operation *op) {
  auto module = op->getParentOfType<ModuleOp>();
  return module && module->getAttrOfType<StringAttr>("npu.division") ==
                       StringAttr::get(op->getContext(),"reciprocal");
}
const char *scalarUnitInstruction(Operation *op) {
  if (isa<arith::AddFOp>(op)) return "S_ADD_F32";
  if (isa<arith::SubFOp>(op)) return "S_SUB_F32";
  if (isa<arith::MulFOp>(op)) return "S_MUL_F32";
  if (isa<arith::MaximumFOp>(op)) return "S_MAX_F32";
  if (isa<arith::MinimumFOp>(op)) return "S_MIN_F32";
  if (isa<math::SqrtOp>(op)) return "S_SQRT_F32";
  if (isa<math::RsqrtOp>(op)) return "S_RSQRT_F32";
  if (isExactReciprocal(op)) return "S_RCP_F32";
  if (isa<arith::DivFOp>(op) && divisionIsReciprocal(op)) return "S_RCP_F32";
  // math.exp 에는 스칼라 유닛 명령이 없다. lowering 은 원소 하나짜리 벡터로
  // 계산한다 — v1 의 FlashAttention 백엔드가 쓰던 것과 같은 우회다.
  if (isa<math::ExpOp>(op)) return "V_EXP_F16";
  return nullptr;
}
} // namespace
bool isSupportedElementwiseScalar(Operation *op) {
  if (isa<arith::DivFOp>(op)) return divisionIsReciprocal(op);
  return vectorInstruction(op) || scalarVectorInstruction(op) || isStreamWidthCast(op);
}
bool isReciprocalDivision(Operation *op) {
  return isa<arith::DivFOp>(op) && divisionIsReciprocal(op);
}
const char *broadcastInstruction(Operation *op) { return scalarVectorInstruction(op); }
const char *scalarLaneInstruction(Operation *op) { return scalarUnitInstruction(op); }
bool isScalarLaneReciprocal(Operation *op) { return isExactReciprocal(op); }
bool isScalarLaneVectorDetour(Operation *op) { return isa<math::ExpOp>(op); }
bool isElementwiseWidthCast(Operation *op) { return isStreamWidthCast(op); }
const char *reductionInstruction(Operation *op) {
  auto gen = dyn_cast<linalg::GenericOp>(op);
  return gen ? reductionOf(gen) : nullptr;
}
const char *elementwiseInstruction(Operation *op) { return vectorInstruction(op); }

// ISA 에는 레지스터를 메모리에 쓰는 방법이 없다. 그래서 컴파일러는 실행 시점에
// 상수를 만들어 낼 수 없다. 상수 텐서는 진입 인자가 되고, 그 바이트는 패키지가
// 실어 날라 이미지를 만드는 쪽에 건넨다.
LogicalResult hoistConstants(ModuleOp m) {
  auto function = *m.getOps<func::FuncOp>().begin();
  SmallVector<arith::ConstantOp> work;
  m.walk([&](arith::ConstantOp c) {
    if (isa<RankedTensorType>(c.getType())) work.push_back(c);
  });
  if (work.empty()) return success();
  OpBuilder b(m.getContext());
  SmallVector<Attribute> values;
  for (auto constant : work) {
    auto type = cast<RankedTensorType>(constant.getType());
    auto dense = dyn_cast<DenseElementsAttr>(constant.getValue());
    if (!dense || !type.hasStaticShape() ||
        !(type.getElementType().isF16() || type.getElementType().isF32()))
      return constant.emitError("constant tensors must be static dense FP16 or FP32");
    unsigned index = function.getNumArguments();
    if (failed(function.insertArgument(index,type,{},constant.getLoc()))) return failure();
    constant.getResult().replaceAllUsesWith(function.getArgument(index));
    constant.erase();
    values.push_back(dense);
  }
  m->setAttr("npu.constants",b.getArrayAttr(values));
  return success();
}

/// Stage 1 후반 — 이 그래프를 받을 수 있는지 검사한다.
///
/// 먼저 normalizeGraph 로 모양을 고친 뒤, 고쳐도 다룰 수 없는 것을 진단과
/// 함께 거부한다. **거부는 실패가 아니라 계약이다** — 여기서 막지 않으면
/// 뒤쪽 단계가 조용히 틀린 코드를 낸다.
///
/// 거부 메시지는 "왜 안 되는지"를 적는다. 하드웨어가 못 하는 것과 아직
/// 구현하지 않은 것은 구분해서 쓴다.
LogicalResult legalizeGraph(ModuleOp m) {
  if (failed(normalizeGraph(m))) return failure();

  // FP16 중간 반올림은 수치 정책이므로 사용자가 --fp16 으로 명시해야 한다.
  // 컴파일러가 마음대로 정밀도를 낮추지 않는다.
  if (m->getAttrOfType<StringAttr>("npu.numerical") != StringAttr::get(m.getContext(), "fp16"))
    return m.emitError("graph lowering requires --fp16 (explicit intermediate rounding policy)");

  unsigned functions = 0, matmuls = 0, generics = 0;
  auto result = m.walk([&](Operation *op) -> WalkResult {
    // ── 함수: 진입점 하나, 블록 하나 ────────────────────────────────
    if (auto f = dyn_cast<func::FuncOp>(op)) {
      ++functions;
      if (f.isExternal() || !llvm::hasSingleElement(f.getBody())) {
        f.emitError("only a single-block defined entry function is supported");
        return WalkResult::interrupt();
      }
    }

    // ── 행렬곱: rank-2 정적 FP16, C 초기값 0 ─────────────────────────
    if (auto mm = dyn_cast<linalg::MatmulOp>(op)) {
      ++matmuls;
      if (!mm.hasPureTensorSemantics()) {
        mm.emitError("graph input must use tensor semantics");
        return WalkResult::interrupt();
      }

      // 크기가 컴파일 시점에 정해져 있어야 타일 계획을 세우고 주소를 배정할
      // 수 있다. 동적 모양은 이 파이프라인이 다루지 않는다.
      for (auto v : mm->getOperands()) {
        auto t = dyn_cast<RankedTensorType>(v.getType());
        if (!t || !t.hasStaticShape() || t.getRank() != 2 ||
            !t.getElementType().isF16() || t.getNumElements() <= 0) {
          mm.emitError("current matmul lowering requires positive static rank-2 FP16 tensors, got ")
              << v.getType();
          return WalkResult::interrupt();
        }
      }

      // 누산기를 불러올 명령이 없으므로 C 는 0 에서 시작해야 한다.
      // 0 채움은 배치 전개가 남긴 슬라이스나 reshape 뒤에 있을 수 있다.
      if (!isZeroFilled(mm.getDpsInitOperand(0)->get())) {
        mm.emitError("nonzero C initialization lowering is not yet implemented (not a hardware impossibility)");
        return WalkResult::interrupt();
      }
    }

    // ── 전치: rank-2 정적 FP16 의 [1,0] 뿐 ──────────────────────────
    // op이 전치라면 → 그 입력 타입을 꺼낸다 → 모양 있는 타입이 아니거나,
    // 2차원이 아니거나, 크기가 미정이거나, FP16이 아니거나, 순열이 [1,0]이 아니면 → 이유를 그 위치에 찍고 순회를 멈춤
    if (auto transpose = dyn_cast<linalg::TransposeOp>(op)) {
      auto in = dyn_cast<ShapedType>(transpose.getInput().getType());
      if (!in || in.getRank() != 2 || !in.hasStaticShape() || !in.getElementType().isF16() ||
          transpose.getPermutation() != ArrayRef<int64_t>{1,0}) {
        transpose.emitError("only a static rank-2 FP16 [1,0] transpose is lowered");
        return WalkResult::interrupt();
      }
    }
    // ── generic: 행 리덕션이거나 원소별 연산 ────────────────────────
    if (auto gen = dyn_cast<linalg::GenericOp>(op)) {
      ++generics;
      if (!gen.hasPureTensorSemantics()) {
        gen.emitError("graph input must use tensor semantics");
        return WalkResult::interrupt();
      }

      if (reductionOf(gen)) {
        // V_REDUCE 는 스트림 전체를 다시 계산한다. 그래서 초기값이 이미
        // 결합 연산의 항등원이어야 결과가 그래프가 말하는 것과 같아진다
        // (합이면 0, 최댓값이면 -inf).
        bool sum = StringRef(reductionOf(gen)) == "V_REDUCE_SUM_F16_F32";
        auto fill = gen.getDpsInitOperand(0)->get().getDefiningOp<linalg::FillOp>();
        auto seed = fill ? fill.getInputs()[0].getDefiningOp<arith::ConstantOp>() : arith::ConstantOp();
        auto attr = seed ? dyn_cast<FloatAttr>(seed.getValue()) : FloatAttr();
        bool identity = attr && (sum ? attr.getValue().isZero()
                                     : attr.getValue().isInfinity() && attr.getValue().isNegative());
        if (!identity) {
          gen.emitError(sum ? "row sum requires a zero-initialised accumulator"
                            : "row max requires a -inf-initialised accumulator");
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (gen.getNumDpsInits() != 1) {
        gen.emitError("elementwise lowering requires tensor semantics and one result"); return WalkResult::interrupt();
      }
      if (!llvm::all_of(gen.getIteratorTypesArray(),
                        [](utils::IteratorType t) { return t == utils::IteratorType::parallel; })) {
        gen.emitError("only fully parallel generics are lowered"); return WalkResult::interrupt();
      }
      auto *context = gen.getContext();
      auto lane = dyn_cast<RankedTensorType>(gen.getDpsInitOperand(0)->get().getType());
      if (lane && isRowShape(lane) && lane.getElementType().isF32()) {
        if (!llvm::all_of(gen.getIndexingMapsArray(),
                          [](AffineMap map) { return map.isIdentity(); })) {
          gen.emitError("scalar-lane generics must be identity-mapped"); return WalkResult::interrupt();
        }
        for (auto v : gen->getOperands()) {
          auto t = dyn_cast<RankedTensorType>(v.getType());
          if (!t || !isRowShape(v.getType()) || !t.getElementType().isF32()) {
            gen.emitError("scalar-lane generics take FP32 row vectors"); return WalkResult::interrupt();
          }
        }
        for (auto &scalar : gen.getRegion().front().without_terminator())
          if (!isa<arith::ConstantOp>(scalar) && !scalarUnitInstruction(&scalar)) {
            scalar.emitError("no scalar unit instruction implements this operation");
            return WalkResult::interrupt();
          }
        return WalkResult::advance();
      }
      auto identity = AffineMap::getMultiDimIdentityMap(2,context);
      auto rowExpr = getAffineDimExpr(0,context);
      auto zeroExpr = getAffineConstantExpr(0,context);
      auto projection = AffineMap::get(2,0,{rowExpr},context);
      auto keepdim = AffineMap::get(2,0,{rowExpr,zeroExpr},context);
      if (gen.getIndexingMapsArray().back() != identity) {
        gen.emitError("the elementwise result must be identity-mapped"); return WalkResult::interrupt();
      }
      for (auto [map,v] : llvm::zip(gen.getIndexingMapsArray(),gen->getOperands())) {
        auto t = dyn_cast<RankedTensorType>(v.getType());
        if (!t || !t.hasStaticShape() || t.getNumElements() <= 0) {
          gen.emitError("elementwise lowering requires positive static tensors"); return WalkResult::interrupt();
        }
        // 행으로 투영하는 것은 리덕션 결과가 되먹임되는 broadcast 이고,
        // 열로 투영하는 것은 채널별 가중치다 — 모든 행에 같은 벡터가 온다.
        auto column = AffineMap::get(2,0,{getAffineDimExpr(1,context)},context);
        bool stream = map == identity && t.getRank() == 2 && t.getElementType().isF16();
        bool shared = map == column && t.getRank() == 1 && t.getElementType().isF16();
        bool scalar = (map == projection || map == keepdim) &&
                      isRowShape(v.getType()) && t.getElementType().isF32();
        if (!stream && !shared && !scalar) {
          gen.emitError("operands must be FP16 matrices, per-channel FP16 vectors, or per-row FP32 scalars");
          return WalkResult::interrupt();
        }
      }
      Block &body = gen.getRegion().front();
      // %out 을 읽는 것이 곧 "이것은 누적이다"라는 선언이고, 루프가 나르는
      // 값도 정확히 그 모양으로 온다. region 이 읽을 때는 승격이 init 을 복사해
      // 넣어 주므로, 타일이 allocator 가 남긴 쓰레기인 경우는 없다.
      for (auto &scalar : body.without_terminator())
        if (!isSupportedElementwiseScalar(&scalar)) {
          scalar.emitError("no VPU instruction implements this scalar operation"); return WalkResult::interrupt();
        }
    }
    // KV 블록 루프는 진행 상태를 나르는 scf.for 로 온다. 범위는 정적이어야
    // 한다 — 반복 횟수가 하드웨어 루프 카운트가 되고, 모든 슬라이스 오프셋이
    // 컴파일 시점 주소로 접혀야 하기 때문이다.
    if (auto loop = dyn_cast<scf::ForOp>(op)) {
      auto bound = [](Value v) { return getConstantIntValue(v).has_value(); };
      if (!bound(loop.getLowerBound()) || !bound(loop.getUpperBound()) || !bound(loop.getStep())) {
        loop.emitError("loop bounds must be compile-time constants"); return WalkResult::interrupt();
      }
      for (auto arg : loop.getInitArgs()) {
        auto t = dyn_cast<RankedTensorType>(arg.getType());
        if (!t || !t.hasStaticShape()) {
          loop.emitError("loop-carried values must be static tensors"); return WalkResult::interrupt();
        }
      }
    }
    if (op->getParentOfType<linalg::LinalgOp>()) return WalkResult::advance();
    if (!isa<ModuleOp,func::FuncOp,func::ReturnOp,linalg::MatmulOp,linalg::GenericOp,linalg::FillOp,
             linalg::CopyOp,linalg::TransposeOp,tensor::EmptyOp,
             tensor::CollapseShapeOp,tensor::ExpandShapeOp,tensor::ExtractSliceOp,tensor::InsertSliceOp,
             scf::ForOp,scf::YieldOp,arith::ConstantOp,arith::MulIOp,arith::AddIOp>(op)) {
      op->emitError("current graph lowering does not implement '") << op->getName() << "'"; return WalkResult::interrupt();
    }
    for (auto type : op->getResultTypes()) if (auto shaped = dyn_cast<ShapedType>(type)) {
      if (!shaped.hasStaticShape()) { op->emitError("dynamic shapes are unsupported"); return WalkResult::interrupt(); }
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) return failure();
  if (functions != 1 || (!matmuls && !generics))
    return m.emitError("requires one entry function containing a matmul or elementwise generic");
  if (failed(hoistConstants(m))) return failure();
  m->setAttr("npu.stage",StringAttr::get(m.getContext(),"legal"));
  return success();
}
} // namespace npu
