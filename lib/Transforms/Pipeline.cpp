//===----------------------------------------------------------------------===//
// Pipeline.cpp — 다이얼렉트 등록, 하드웨어 속성, IR 청소, 공유 헬퍼
//===----------------------------------------------------------------------===//
#include "GraphInternal.h"

using namespace mlir;
namespace plena {
/// 이 파이프라인이 쓸 다이얼렉트와 인터페이스 구현을 장부(DialectRegistry)에
/// 올린다. MLIR 은 라이브러리 묶음이라 기본으로 켜져 있는 것이 없다.
///
/// **인터페이스 등록을 빠뜨리면 빌드는 통과하고 실행 중에 abort 한다.**
/// 진단이 아니라 LLVM ERROR 이므로 위치도 알려주지 않는다. 링크(CMakeLists)와
/// 등록(여기)은 별개이며, 둘 다 해야 cast<...Interface>(op) 가 성공한다.
///
/// 새 MLIR 라이브러리 호출을 추가하면 여기 등록도 함께 본다.
void registerPipelineDialects(DialectRegistry &r) {
  // 파일에 나올 수 있는 방언들. 하나라도 빠지면 그 op 을 파싱조차 못 한다.
  r.insert<mlir::plena::PlenaDialect, affine::AffineDialect, arith::ArithDialect,
           bufferization::BufferizationDialect, func::FuncDialect,
           linalg::LinalgDialect, math::MathDialect, memref::MemRefDialect,
           scf::SCFDialect, tensor::TensorDialect>();

  // ── 3막 버퍼화(tensor → memref)가 묻는 것 ──────────────────────────
  arith::registerBufferizableOpInterfaceExternalModels(r);
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(r);
  linalg::registerBufferizableOpInterfaceExternalModels(r);
  scf::registerBufferizableOpInterfaceExternalModels(r);
  tensor::registerBufferizableOpInterfaceExternalModels(r);

  // 제자리 쓰기를 증명하려고 "이 쓰기가 저 버퍼의 일부인가"를 묻는다.
  linalg::registerSubsetOpInterfaceExternalModels(r);
  tensor::registerSubsetOpInterfaceExternalModels(r);

  // ── 2막 타일링이 묻는 것 ───────────────────────────────────────────
  // tileUsingSCF 는 op 이 무엇인지 모르고 TilingInterface 로만 대화한다.
  // 이것이 없으면 첫 타일링 시도에서 "promised by dialect but never
  // implemented" 로 죽는다.
  linalg::registerTilingInterfaceExternalModels(r);
  tensor::registerTilingInterfaceExternalModels(r);

  // 타일이 둘 이상일 때만 조회된다 — 반복 횟수를 접고 슬라이스가 범위 안에
  // 있음을 증명하는 데 쓴다. 1x1 타일에서는 드러나지 않으므로, 필요할 때
  // 추가하지 않고 값을 만들 수 있는 방언을 전부 미리 등록한다.
  affine::registerValueBoundsOpInterfaceExternalModels(r);
  arith::registerValueBoundsOpInterfaceExternalModels(r);
  linalg::registerValueBoundsOpInterfaceExternalModels(r);
  memref::registerValueBoundsOpInterfaceExternalModels(r);
  scf::registerValueBoundsOpInterfaceExternalModels(r);
  tensor::registerValueBoundsOpInterfaceExternalModels(r);

  // ── 1막 정규화가 묻는 것 ───────────────────────────────────────────
  // 단위 차원을 접으면 텐서 모양이 바뀌므로 새 모양을 되물어야 한다.
  tensor::registerInferTypeOpInterfaceExternalModels(r);
}
/// 하드웨어 설정을 모듈 속성으로 옮긴다.
///
/// 이후 단계는 HardwareConfig 구조체가 아니라 이 속성들을 읽는다. 그래서
/// 중간 단계 .mlir 파일 하나만 있으면 --from=<단계> 로 재진입할 수 있다 —
/// 파일이 자기가 어떤 하드웨어를 전제했는지 들고 다니기 때문이다.
void attachHardware(ModuleOp m, const HardwareConfig &h) {
  OpBuilder b(m.getContext());

  // 설정 파일 전문의 지문. 다른 설정으로 만든 단계 파일을 드라이버가 거부한다.
  m->setAttr("plena.hardware", b.getStringAttr(h.fingerprint));
  m->setAttr("plena.schema", b.getI64IntegerAttr(1));

  struct Capacity {
    const char *attribute;
    uint64_t value;
  };
  const Capacity capacities[] = {
      {"plena.cores", h.cores},
      {"plena.l1_bytes", h.l1Bytes},
      {"plena.l2_bytes", h.l2Bytes},
      {"plena.dram_bytes", h.dramBytes},
  };
  for (const auto &c : capacities)
    m->setAttr(c.attribute, b.getI64IntegerAttr(c.value));
}
/// canonicalize + CSE. 재작성 사이사이에서 부른다 — 한 재작성이 남긴 죽은 op
/// 을 다음 재작성이 패턴으로 잘못 잡는 것을 막고, 모양이 접힌 뒤의 결과를
/// 다음 단계가 상수로 볼 수 있게 한다.
LogicalResult cleanIR(ModuleOp m) {
  PassManager pm(m.getContext());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  return pm.run(m);
}
//===----------------------------------------------------------------------===//
// 행 리덕션을 FP32 로 넓히기
//
// 하드웨어에는 FP16 누적 명령이 없다. FP16 스트림을 FP32 로 줄여 받는 것만
// 할 수 있다. 그래서 캡처가 준 FP16 리덕션은 적힌 그대로는 돌지 않고,
// 거기 매달린 행 벡터 체인 전체가 FP32 로 옮겨간다.
//
// 그리고 keepdim 리덕션이 남긴 단위 차원도 여기서 없앤다 —
// tensor<Nx1xf16> 이 행 경로에 닿기 전에 tensor<Nxf32> 가 되어야 한다.
//
// MLIR 의 `linalg-fold-unit-extent-dims` 로는 안 된다. 모양은 접어 주지만
// 루프가 나르는 값까지 고쳐서, 누적이 제자리 버퍼화를 못 하게 된다
// (`Yield operand #0 is not equivalent to the corresponding iter bbArg`).

/// 행 벡터인가 — 마지막 축을 줄인 리덕션이 남기는 모양. keepdim 캡처가
/// 남긴 단위 차원(Nx1)이 있든 없든 참이다.
///
/// tensor 가 아니라 ShapedType 을 받는 이유: 이 분류기들은 버퍼화 뒤에도
/// 다시 돌고, 그때는 피연산자가 memref 인데 lowering 이 같은 판정을 요구한다.
bool isRowShape(Type type) {
  auto shaped = dyn_cast<ShapedType>(type);
  return shaped && shaped.hasStaticShape() &&
         (shaped.getRank() == 1 || (shaped.getRank() == 2 && shaped.getDimSize(1) == 1));
}

/// 이 값이 0 으로 채워진 것인가. 배치 전개가 남긴 슬라이스나 reshape 뒤에
/// 0 채움이 숨어 있을 수 있으므로 몇 단계 거슬러 올라간다. 그것들은 바이트
/// 이름만 바꾸므로 누산기는 여전히 0 이다.
bool isZeroFilled(Value value) {
  for (unsigned hops = 0; value && hops < 8; ++hops) {
    if (auto fill = value.getDefiningOp<linalg::FillOp>()) {
      auto seed = fill.getInputs()[0].getDefiningOp<arith::ConstantOp>();
      auto attr = seed ? dyn_cast<FloatAttr>(seed.getValue()) : FloatAttr();
      return attr && attr.getValue().isZero();
    }
    auto *op = value.getDefiningOp();
    if (!op) return false;
    if (isa<tensor::ExtractSliceOp,tensor::CollapseShapeOp,tensor::ExpandShapeOp>(op))
      value = op->getOperand(0);
    else if (auto insert = dyn_cast<tensor::InsertSliceOp>(op)) value = insert.getDest();
    else return false;
  }
  return false;
}

// 행마다 하나씩인 FP32 값은 스칼라 슬롯에 산다. 그래서 rank-1 FP32 generic 은
// 스칼라 연산 유닛으로 내려간다. 나눗셈은 분자가 상수 1 일 때만 받는다 —
// 그때 S_RCP_F32 가 근사가 아니라 요청한 값 그대로를 계산한다.
bool isExactReciprocal(Operation *op) {
  auto div = dyn_cast<arith::DivFOp>(op);
  if (!div) return false;
  auto one = div.getLhs().getDefiningOp<arith::ConstantOp>();
  auto attr = one ? dyn_cast<FloatAttr>(one.getValue()) : FloatAttr();
  return attr && attr.getValue().convertToDouble() == 1.0;
}

// 행 리덕션은 이름이 아니라 **모양**으로 알아본다 — 병렬 행 하나, 리덕션 열
// 하나, 항등 입력, 행으로 투영하는 출력, 그리고 region 이 넓혀 쓰는 FP32
// 누산기. ISA 가 가진 결합 연산은 합과 최댓값 둘뿐이다.
const char *reductionOf(linalg::GenericOp gen) {
  if (gen.getNumDpsInputs() != 1 || gen.getNumDpsInits() != 1) return nullptr;
  auto iterators = gen.getIteratorTypesArray();
  if (iterators.size() != 2 || iterators[0] != utils::IteratorType::parallel ||
      iterators[1] != utils::IteratorType::reduction) return nullptr;
  auto *context = gen.getContext();
  auto maps = gen.getIndexingMapsArray();
  auto row = getAffineDimExpr(0, context);
  auto zero = getAffineConstantExpr(0, context);
  // 캡처는 keepdim 이 남긴 단위 차원을 그대로 들고 있어서, 두 투영이 같은 행
  // 벡터를 뜻한다.
  bool projects = maps.size() == 2 &&
                  maps[0] == AffineMap::getMultiDimIdentityMap(2, context) &&
                  (maps[1] == AffineMap::get(2, 0, {row}, context) ||
                   maps[1] == AffineMap::get(2, 0, {row, zero}, context));
  if (!projects) return nullptr;
  // tensor 가 아니라 ShapedType 인 이유: 같은 분류기가 버퍼화 뒤에도 다시
  // 도는데, 그때는 피연산자가 memref 이고 lowering 이 같은 판정을 요구한다.
  auto in = dyn_cast<ShapedType>(gen.getDpsInputOperand(0)->get().getType());
  auto out = dyn_cast<ShapedType>(gen.getDpsInitOperand(0)->get().getType());
  if (!in || !out || !in.hasStaticShape() || !out.hasStaticShape() || in.getRank() != 2 ||
      !isRowShape(gen.getDpsInitOperand(0)->get().getType()) ||
      !in.getElementType().isF16() || !out.getElementType().isF32())
    return nullptr;
  Block &body = gen.getRegion().front();
  auto scalars = llvm::to_vector(llvm::map_range(body.without_terminator(), [](Operation &o) { return &o; }));
  if (scalars.size() != 2 || !isa<arith::ExtFOp>(scalars[0]) ||
      scalars[0]->getOperand(0) != body.getArgument(0)) return nullptr;
  auto *combine = scalars[1];
  if (combine->getNumOperands() != 2 ||
      cast<linalg::YieldOp>(body.getTerminator()).getValues()[0] != combine->getResult(0) ||
      !llvm::is_contained(combine->getOperands(), scalars[0]->getResult(0)) ||
      !llvm::is_contained(combine->getOperands(), body.getArgument(1))) return nullptr;
  if (isa<arith::AddFOp>(combine)) return "V_REDUCE_SUM_F16_F32";
  if (isa<arith::MaximumFOp>(combine)) return "V_REDUCE_MAX_F16_F32";
  return nullptr;
}
} // namespace plena
