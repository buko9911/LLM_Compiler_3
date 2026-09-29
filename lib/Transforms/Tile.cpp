//===----------------------------------------------------------------------===//
// Tile.cpp — 2막 — 타일 계획을 고르고 타일 루프를 세운다
//===----------------------------------------------------------------------===//
#include "GraphInternal.h"
#include <map>
#include <tuple>

using namespace mlir;
namespace plena {
//===----------------------------------------------------------------------===//
// 2막 — 쪼개기
//
// 큰 행렬곱 하나를 타일 루프 중첩으로 바꾼다. 전부 텐서 레벨에서 일어나고,
// memref 와 주소는 3막이 붙인다.
//===----------------------------------------------------------------------===//
namespace {

/// 행렬곱 하나를 `sizes` 크기로 잘라 scf.for 루프를 세운다.
///
/// 잘라서 생긴 **작은** 행렬곱을 돌려준다. tileGraph 가 그것을 다시 이 함수에
/// 넣어 양파 껍질처럼 네 겹을 만든다(패널 → 코어 → 배열 → K).
///
/// `sizes` 에서 0 은 "그 축은 자르지 않는다"는 뜻이다.
///
/// 실제 일은 MLIR 의 scf::tileUsingSCF 가 한다. 그 함수는 linalg.matmul 이
/// 무엇인지 모르고 TilingInterface 로만 대화하므로, 아래 cast 가 성공하려면
/// registerPipelineDialects 의 registerTilingInterfaceExternalModels 가 이미
/// 실행돼 있어야 한다. 빠져 있으면 여기서 abort 한다.
FailureOr<linalg::MatmulOp> tile(IRRewriter &r, linalg::MatmulOp mm, ArrayRef<int64_t> sizes) {
  scf::SCFTilingOptions options;
  SmallVector<OpFoldResult> tileSizes;
  for (auto size : sizes)
    tileSizes.push_back(r.getIndexAttr(size));
  options.setTileSizes(tileSizes);

  auto tiled = scf::tileUsingSCF(r, cast<TilingInterface>(mm.getOperation()), options);
  if (failed(tiled) || tiled->tiledOps.empty()) return failure();

  auto next = dyn_cast<linalg::MatmulOp>(tiled->tiledOps.front());
  if (!next) return failure();

  r.replaceOp(mm, tiled->replacements);
  return next;
}

/// 메모리 공간을 지정한 텐서 자리를 잡는다. space 1 = L2, 2 = L1.
/// 이 공간 표시가 3막에서 memref 타입에 박히고, 그것으로 주소가 정해진다.
Value allocTensor(IRRewriter &r, Location loc, RankedTensorType type, unsigned space) {
  return r.create<bufferization::AllocTensorOp>(loc, type, ValueRange{}, Value{},
                                                r.getI64IntegerAttr(space));
}

Value copyTensor(IRRewriter &r, Location loc, Value src, Value dst) {
  return r.create<linalg::CopyOp>(loc, TypeRange{dst.getType()}, ValueRange{src},
                                  ValueRange{dst})
      .getResult(0);
}

/// 행렬곱의 피연산자를 더 가까운 메모리로 끌어올린다.
///
/// 자르기만 해서는 조각이 어디 있는지 정해지지 않는다. 여기서 "이 조각을
/// space 로 복사해 온 뒤 거기서 계산하라"는 alloc + copy 를 끼워 넣는다.
/// 3막 버퍼화가 그 alloc 을 실제 버퍼로 만들고 주소를 배정한다.
LogicalResult promote(IRRewriter &r, linalg::MatmulOp mm, unsigned space, bool inputs,
                      bool output) {
  r.setInsertionPoint(mm);

  // 입력 A, B 를 각각 복사해 온다.
  if (inputs) {
    for (unsigned i = 0; i < 2; ++i) {
      Value source = mm.getDpsInputOperand(i)->get();
      auto sourceType = dyn_cast<RankedTensorType>(source.getType());
      // 타일 크기가 컴파일 시점 상수여야 버퍼 크기와 주소가 정해진다.
      if (!sourceType || !sourceType.hasStaticShape())
        return mm.emitError("tiling must produce static input tiles");

      Value buffer = allocTensor(r, mm.getLoc(), sourceType, space);
      mm.getDpsInputOperand(i)->set(copyTensor(r, mm.getLoc(), source, buffer));
    }
  }

  // 출력은 방향이 반대다 — 새 자리에서 계산한 뒤 원래 자리로 되돌려 쓴다.
  if (output) {
    Value dest = mm.getDpsInitOperand(0)->get();
    auto destType = dyn_cast<RankedTensorType>(dest.getType());
    if (!destType || !destType.hasStaticShape())
      return mm.emitError("tiling must produce static output tiles");

    Value buffer = allocTensor(r, mm.getLoc(), destType, space);

    // 새로 잡은 자리에는 allocator 가 남긴 값이 들어 있다. 누산기는 0 에서
    // 시작해야 하므로 명시적으로 채운다. plena.tile_zero 표시는 뒤 단계가
    // "이 fill 은 타일 초기화"임을 알아보는 데 쓴다.
    auto zero = r.create<arith::ConstantOp>(mm.getLoc(),
                                            r.getFloatAttr(destType.getElementType(), 0));
    auto filled = r.create<linalg::FillOp>(mm.getLoc(), ValueRange{zero}, ValueRange{buffer});
    filled->setAttr("plena.tile_zero", r.getUnitAttr());
    mm.getDpsInitOperand(0)->set(filled.getResult(0));

    // 계산이 끝난 뒤 원래 자리로 되돌린다. replaceAllUsesExcept 로 방금 만든
    // 복사 자신은 제외해야 자기 결과를 자기 입력으로 삼는 고리가 안 생긴다.
    r.setInsertionPointAfter(mm);
    Value copied = copyTensor(r, mm.getLoc(), mm.getResult(0), dest);
    mm.getResult(0).replaceAllUsesExcept(copied, copied.getDefiningOp());
  }
  return success();
}

/// 코어 하나가 한 번에 맡을 열 개수.
///
/// 적어도 배열 폭(32)은 되어야 하고, 패널을 정확히 나누어떨어뜨려야 한다.
/// 한 구간이 한 코어이므로, 이렇게 나눌 수 없는 패널은 그냥 한 코어에 남는다.
uint64_t coreLane(uint64_t columns, unsigned cores) {
  if (cores < 2) return columns;
  uint64_t want = std::max<uint64_t>(32,columns/cores);
  want -= want % 32;
  for (; want > 32 && columns % want; want -= 32) {}
  return want >= 32 && columns % want == 0 ? want : columns;
}
// `value` 를 만드는 것들이 루프에 전혀 의존하지 않으면 루프 밖으로 옮긴다.
//
// 타일이 곱하는 활성값은 그 타일이 **어느 열에 쓰는지**와 무관하다. 그래서
// 루프 안에서 올리면 같은 바이트를 타일마다 한 번씩 L2 에서 다시 읽게 된다.
// 옮기는 것은 스테이징 op 뿐이고, 나머지는 타일러가 둔 자리에 그대로 둔다.
bool hoistStaging(IRRewriter &r, scf::ForOp loop, Value value) {
  SmallVector<Operation *> chain;
  SmallVector<Value> pending{value};
  DenseSet<Operation *> seen;
  while (!pending.empty()) {
    auto *made = pending.pop_back_val().getDefiningOp();
    if (!made || !loop->isAncestor(made) || !seen.insert(made).second) continue;
    if (!isa<bufferization::AllocTensorOp,tensor::ExtractSliceOp,linalg::CopyOp>(made)) return false;
    chain.push_back(made);
    for (Value operand : made->getOperands()) pending.push_back(operand);
  }
  auto outside = [&](Value operand) {
    if (auto *made = operand.getDefiningOp()) return !loop->isAncestor(made);
    auto block = cast<BlockArgument>(operand).getOwner()->getParentOp();
    return !loop->isAncestor(block) && block != loop.getOperation();
  };
  // 생산자부터 옮긴다 — 자기 입력이 먼저 나가면 그때 옮길 수 있게 되는 것들이다.
  for (bool moving = true; moving;) {
    moving = false;
    for (auto *&made : chain) {
      if (!made || llvm::any_of(made->getOperands(),[&](Value o){ return !outside(o); })) continue;
      r.moveOpBefore(made,loop);
      made = nullptr;
      moving = true;
    }
  }
  auto *made = value.getDefiningOp();
  return made && !loop->isAncestor(made);
}
// 활성값 스테이징은 그 주변의 1회짜리 루프가 사라진 뒤에야 루프 불변이 된다.
// 그래서 타일러가 아직 중첩을 짓는 도중이 아니라 canonicalize 이후에 돈다.
// 코어 분할이 경계다 — L1 은 코어마다 따로이므로 각 코어가 자기 사본을 올려야 한다.
LogicalResult hoistActivations(ModuleOp m, IRRewriter &r) {
  SmallVector<Operation *> staged;
  m.walk([&](Operation *op) { if (op->hasAttr("plena.stage_activation")) staged.push_back(op); });
  for (auto *op : staged) {
    while (auto loop = op->getParentOfType<scf::ForOp>()) {
      if (loop->hasAttr("plena.core_split") || !hoistStaging(r,loop,op->getResult(0))) break;
    }
    op->removeAttr("plena.stage_activation");
  }
  return success();
}
uint64_t divisor(uint64_t n) {
  for (uint64_t t = std::min<uint64_t>(n,32); t; --t) if (n%t == 0) return t;
  return 1;
}
}
namespace {
// 원소별 타일은 용량만이 제약이다. VPU 가 연속된 타일을 통째로 흘려보내므로
// 행은 나누고 열은 통째로 둔다. 약수를 돌려주면 모든 버퍼가 컴파일 시점 상수로
// 남아 행렬곱 경로와 같은 성질을 갖는다.
int64_t elementwiseRows(int64_t rows, uint64_t perRow, uint64_t budget) {
  if (!perRow) return 0;
  int64_t fits = int64_t(std::min<uint64_t>(uint64_t(rows), budget / perRow));
  for (int64_t t = fits; t > 0; --t) if (rows % t == 0) return t;
  return 0;
}
LogicalResult promoteElementwise(IRRewriter &r, linalg::GenericOp gen, unsigned space) {
  r.setInsertionPoint(gen);
  for (auto *operand : gen.getDpsInputOperands()) {
    auto t = dyn_cast<RankedTensorType>(operand->get().getType());
    if (!t || !t.hasStaticShape()) return gen.emitError("tiling must produce static input tiles");
    operand->set(copyTensor(r,gen.getLoc(),operand->get(),allocTensor(r,gen.getLoc(),t,space)));
  }
  auto dest = gen.getDpsInitOperand(0)->get();
  auto t = dyn_cast<RankedTensorType>(dest.getType());
  if (!t || !t.hasStaticShape()) return gen.emitError("tiling must produce static output tiles");
  // 누적은 자기 init 을 읽으므로, 그 타일은 진행 중인 값을 담은 채로 도착해야
  // 한다. 그렇지 않은 경우에는 region 이 채우기 전에 그 타일을 읽는 것이 없다.
  // 리덕션은 예외다 — V_REDUCE 는 스트림 전체를 다시 계산하고 init 을 보지
  // 않는다. 그래서 합법성 검사가 이미 init 을 항등원으로 강제해 두었다.
  bool accumulates = !gen.getRegion().front().getArguments().back().use_empty() && !reductionOf(gen);
  auto tile = allocTensor(r,gen.getLoc(),t,space);
  gen.getDpsInitOperand(0)->set(accumulates ? copyTensor(r,gen.getLoc(),dest,tile) : tile);
  r.setInsertionPointAfter(gen);
  auto copied = copyTensor(r,gen.getLoc(),gen.getResult(0),dest);
  gen.getResult(0).replaceAllUsesExcept(copied,copied.getDefiningOp());
  return success();
}
FailureOr<linalg::GenericOp> tileElementwise(IRRewriter &r, linalg::GenericOp gen, int64_t rows) {
  scf::SCFTilingOptions options;
  options.setTileSizes({r.getIndexAttr(rows), r.getIndexAttr(0)});
  auto tiled = scf::tileUsingSCF(r, cast<TilingInterface>(gen.getOperation()), options);
  if (failed(tiled) || tiled->tiledOps.empty()) return failure();
  auto next = dyn_cast<linalg::GenericOp>(tiled->tiledOps.front());
  if (!next) return failure();
  r.replaceOp(gen,tiled->replacements);
  return next;
}
// V_REDUCE 하나가 스트림 전체를 스칼라 하나로 줄인다. 그래서 가장 안쪽 타일은
// 정확히 한 행이다 — 그 합이 그 행의 합이려면 열 축이 통째로 남아야 한다.
LogicalResult tileReduction(IRRewriter &r, linalg::GenericOp gen, const HardwareConfig &hw,
                            SmallVectorImpl<Attribute> &explanations) {
  auto in = cast<RankedTensorType>(gen.getDpsInputOperand(0)->get().getType());
  int64_t rows = in.getDimSize(0), cols = in.getDimSize(1);
  uint64_t perRow = uint64_t(cols)*2 + 4;
  int64_t panel = int64_t(std::min<uint64_t>(uint64_t(rows), (hw.l2Bytes/2)/perRow));
  for (; panel > 0 && rows % panel; --panel) {}
  if (!panel || perRow > hw.l1Bytes/2)
    return gen.emitError("one reduction row does not fit the configured L1/L2 capacity");
  // 안쪽 타일은 L1 이 허락하는 만큼의 행을 담고, 하드웨어 루프가 그것들을 훑는다.
  int64_t tileRows = int64_t(std::min<uint64_t>(uint64_t(panel), (hw.l1Bytes/2)/perRow));
  for (; tileRows > 0 && panel % tileRows; --tileRows) {}
  if (!tileRows) return gen.emitError("no common reduction row factor for L2 panel and L1 tile");
  explanations.push_back(r.getStringAttr("Reduction rows L2=" + std::to_string(panel) +
                                         " L1=" + std::to_string(tileRows) +
                                         " cols=" + std::to_string(cols)));
  gen->setAttr("plena.fusion_group",r.getI64IntegerAttr(explanations.size()-1));
  auto outer = tileElementwise(r,gen,panel);
  if (failed(outer) || failed(promoteElementwise(r,*outer,1))) return failure();
  auto inner = tileElementwise(r,*outer,tileRows);
  if (failed(inner) || failed(promoteElementwise(r,*inner,2))) return failure();
  return success();
}
LogicalResult tileElementwiseGraph(ModuleOp m, const HardwareConfig &hw, IRRewriter &r,
                                   SmallVectorImpl<Attribute> &explanations) {
  SmallVector<linalg::GenericOp> work;
  m.walk([&](linalg::GenericOp gen) { work.push_back(gen); });
  for (auto gen : work) {
    if (reductionOf(gen)) {
      if (failed(tileReduction(r,gen,hw,explanations))) return failure();
      continue;
    }
    auto lane = cast<RankedTensorType>(gen.getDpsInitOperand(0)->get().getType());
    if (isRowShape(lane)) {
      // 스칼라 슬롯 하나가 레인 하나를 담으므로 타일은 원소 하나다. 남아 있는
      // 단위 차원이 있어도 마찬가지다.
      explanations.push_back(r.getStringAttr("Scalar lane elements=" + std::to_string(lane.getDimSize(0))));
      gen->setAttr("plena.fusion_group",r.getI64IntegerAttr(explanations.size()-1));
      auto outer = tileElementwise(r,gen,lane.getDimSize(0));
      if (failed(outer) || failed(promoteElementwise(r,*outer,1))) return failure();
      auto inner = tileElementwise(r,*outer,1);
      if (failed(inner) || failed(promoteElementwise(r,*inner,2))) return failure();
      continue;
    }
    auto type = cast<RankedTensorType>(gen.getDpsInitOperand(0)->get().getType());
    int64_t rows = type.getDimSize(0), cols = type.getDimSize(1);
    // 피연산자마다 실제 한 행의 크기를 센다 — broadcast 레인은 FP32 스칼라
    // 하나, 행렬 피연산자는 cols 개의 FP16 값이다. 원본 텐서의 전체 행이 아니라
    // **후보 타일에 들어가는 행만** 계산에 넣는다.
    uint64_t perRow = 0;
    for (Value operand : gen->getOperands()) {
      auto t = cast<RankedTensorType>(operand.getType());
      perRow += uint64_t(t.getNumElements() / rows) *
                (t.getElementTypeBitWidth() / 8);
    }
    // 각 메모리 계층의 절반은 다른 살아 있는 버퍼를 위해 남겨 둔다.
    int64_t panel = elementwiseRows(rows,perRow,hw.l2Bytes/2);
    int64_t tileRows = elementwiseRows(rows,perRow,hw.l1Bytes/2);
    if (!panel || !tileRows)
      return gen.emitError("one elementwise row does not fit the configured L1/L2 capacity");
    tileRows = std::min(tileRows,panel);
    for (; tileRows > 0 && panel % tileRows; --tileRows) {}
    if (!tileRows) return gen.emitError("no common elementwise row factor for L2 panel and L1 tile");
    explanations.push_back(r.getStringAttr("Elementwise rows L2=" + std::to_string(panel) +
                                           " L1=" + std::to_string(tileRows) +
                                           " cols=" + std::to_string(cols)));
    gen->setAttr("plena.fusion_group",r.getI64IntegerAttr(explanations.size()-1));
    auto outer = tileElementwise(r,gen,panel);
    if (failed(outer) || failed(promoteElementwise(r,*outer,1))) return failure();
    auto inner = tileElementwise(r,*outer,tileRows);
    if (failed(inner) || failed(promoteElementwise(r,*inner,2))) return failure();
  }
  return success();
}
} // namespace

namespace {
// 전치는 열을 따라 내려가며 읽으므로, 원소 하나가 전송 한 행이 된다.
// 그것을 DRAM 텐서 전체에 대고 하면 원소마다 명령이 하나씩 든다. 반면 L1 타일로
// 하면 타일 행마다 코어 명령 하나로 끝나고 루프로 감쌀 수도 있다.
// 그래서 입력은 L2 로 연속되게 올리고 타일은 L1 에 내린다.
LogicalResult tileTransposes(ModuleOp m, IRRewriter &r, SmallVectorImpl<Attribute> &explanations) {
  SmallVector<linalg::TransposeOp> work;
  m.walk([&](linalg::TransposeOp t) { work.push_back(t); });
  for (auto t : work) {
    auto type = cast<RankedTensorType>(t.getInit().getType());
    int64_t rows = std::min<int64_t>(type.getDimSize(0),32), cols = std::min<int64_t>(type.getDimSize(1),32);
    for (; rows > 1 && type.getDimSize(0) % rows; --rows) {}
    for (; cols > 1 && type.getDimSize(1) % cols; --cols) {}
    explanations.push_back(r.getStringAttr("Transpose tile " + std::to_string(rows) + "x" +
                                           std::to_string(cols)));
    t->setAttr("plena.fusion_group",r.getI64IntegerAttr(explanations.size()-1));
    scf::SCFTilingOptions options;
    options.setTileSizes({r.getIndexAttr(rows), r.getIndexAttr(cols)});
    auto tiled = scf::tileUsingSCF(r, cast<TilingInterface>(t.getOperation()), options);
    if (failed(tiled) || tiled->tiledOps.empty()) return failure();
    auto inner = dyn_cast<linalg::TransposeOp>(tiled->tiledOps.front());
    if (!inner) return failure();
    r.replaceOp(t,tiled->replacements);
    r.setInsertionPoint(inner);
    auto source = cast<RankedTensorType>(inner.getInput().getType());
    inner.getDpsInputOperand(0)->set(
        copyTensor(r,inner.getLoc(),inner.getInput(),allocTensor(r,inner.getLoc(),source,1)));
    auto destination = cast<RankedTensorType>(inner.getInit().getType());
    auto dest = inner.getInit();
    inner.getDpsInitOperand(0)->set(allocTensor(r,inner.getLoc(),destination,2));
    r.setInsertionPointAfter(inner);
    auto result = inner.getResults().front();
    // L1 에서 DRAM 으로 가는 길이 없어서 타일은 먼저 L2 에 내린다. 전치 자체는
    // 이미 끝났고, 이 둘은 연속된 이동일 뿐이다.
    auto staged = copyTensor(r,inner.getLoc(),result,allocTensor(r,inner.getLoc(),destination,1));
    auto copied = copyTensor(r,inner.getLoc(),staged,dest);
    result.replaceAllUsesExcept(copied,staged.getDefiningOp());
  }
  return success();
}
} // namespace

namespace {
// 가중치 패널은 행 우선 가중치의 열 방향 조각이다. 그래서 그것을 가져오는
// 전송은 K 마다 짧은 행 하나씩이 된다. 전송 비용은 바이트가 아니라 **행 개수**가
// 지배하므로, 패키지에게 가중치를 애초에 패널 단위로 늘어놓아 달라고 요구한다.
// 그러면 패널 하나가 통째로 연속된 한 행이 된다.
// 다른 데서 읽지 않는 인자만 이렇게 순서를 바꿀 수 있다.
LogicalResult packWeightPanels(OpBuilder &b, linalg::MatmulOp mm, uint64_t panel) {
  // 통째로 가져가는 뷰만 받는다. 가중치의 슬라이스는 패널의 슬라이스이기도
  // 한데, 모아 놓은 바이트가 가진 모양은 패널뿐이다.
  auto whole = [](Operation *made) {
    auto slice = dyn_cast<tensor::ExtractSliceOp>(made);
    if (!slice) {
      if (!isa<tensor::CollapseShapeOp,tensor::ExpandShapeOp>(made)) return false;
      auto from = dyn_cast<RankedTensorType>(made->getOperand(0).getType());
      auto to = dyn_cast<RankedTensorType>(made->getResult(0).getType());
      return from && to && from.getRank() && to.getRank() &&
             from.getShape().back() == to.getShape().back();
    }
    auto source = slice.getSourceType();
    return llvm::all_of(slice.getStaticOffsets(),[](int64_t o){ return o == 0; }) &&
           llvm::all_of(slice.getStaticStrides(),[](int64_t t){ return t == 1; }) &&
           ArrayRef<int64_t>(slice.getStaticSizes()) == source.getShape();
  };
  Value weight = mm.getDpsInputOperand(1)->get();
  while (auto *made = weight.getDefiningOp()) {
    if (!whole(made)) return success();
    weight = made->getOperand(0);
  }
  auto arg = dyn_cast<BlockArgument>(weight);
  if (!arg) return success();
  auto type = dyn_cast<RankedTensorType>(arg.getType());
  if (!type || !type.hasStaticShape() || type.getDimSize(type.getRank()-1) % int64_t(panel))
    return success();
  // 읽는 쪽은 전부 이 가중치가 거쳐 가는 뷰 중 하나여야 한다.
  SmallVector<Value> reachable{arg};
  for (unsigned seen = 0; seen < reachable.size(); ++seen)
    for (Operation *user : reachable[seen].getUsers()) {
      if (isa<linalg::MatmulOp>(user)) {
        if (user != mm.getOperation() ||
            cast<linalg::MatmulOp>(user).getDpsInputOperand(1)->get() != reachable[seen])
          return success();
        continue;
      }
      if (!whole(user)) return success();
      reachable.push_back(user->getResult(0));
    }
  auto function = cast<func::FuncOp>(arg.getOwner()->getParentOp());
  if (function.getArgAttr(arg.getArgNumber(),"plena.packed_panel")) return success();
  function.setArgAttr(arg.getArgNumber(),"plena.packed_panel",b.getI64IntegerAttr(panel));
  return success();
}
} // namespace

/// 2막 — 타일 계획을 고르고 타일 루프를 세운다.
///
/// L2 패널 → 코어 구간 → L1 패널 → 배열 → K 조각 순으로 타일링한다.
/// 전체 K가 L1에 들어가면 L1 패널의 입력을 여러 배열 타일에서 재사용한다.
/// K를 나눌 때는 배열 타일 하나를 끝까지 누적한 뒤 다음 타일로 넘어간다.
/// 바깥이 큰 단위, 안으로 갈수록 작다:
///
///   ① 패널   L2 에 올릴 덩어리        promote(space 1) → L2
///   ② 코어   패널을 코어별 열 구간으로  (루프만, 복사 없음)
///   ③ 배열   32x32 배열이 한 번에 먹는 타일   promote(space 2) → L1
///   ④ K 조각 누적 축을 나눈다          promote(space 2) → L1
///
/// 크기는 전부 lib/Analysis/TilingPlan.cpp 의 비용 모델이 고른다. 여기서는
/// 고른 계획대로 루프를 세우고, 뒤 단계가 알아볼 표시를 붙이는 일만 한다.
LogicalResult tileGraph(ModuleOp m, const HardwareConfig &hw) {
  // 자르는 도중에 IR 이 바뀌므로 대상을 먼저 모아 둔다.
  SmallVector<linalg::MatmulOp> work;
  m.walk([&](linalg::MatmulOp mm) { work.push_back(mm); });

  IRRewriter r(m.getContext());
  SmallVector<Attribute> explanations;
  // Attention heads repeat identical shapes. Search once per shape in this
  // module; hardware and numerical policy are fixed throughout this pass.
  std::map<std::tuple<uint64_t,uint64_t,uint64_t>,TilingPlan> selectedPlans;

  for (auto mm : work) {
    auto a = cast<RankedTensorType>(mm.getDpsInputOperand(0)->get().getType());
    auto b = cast<RankedTensorType>(mm.getDpsInputOperand(1)->get().getType());

    // ── 계획 고르기 ─────────────────────────────────────────────────
    TilingRequest request;
    request.hardware = hw;
    request.graphResidentBytes = kDRAMStagingBytes;
    request.numerical = NumericalPolicy::AllowFP16Rounding;
    // L2 단일/이중 스테이징을 함께 비교한다. 펼쳐진 루프 안의 L1 버퍼
    // 복사본도 용량 제약에 포함한다.
    request.stages = 2;
    request.searchStages = true;
    request.shape = {uint64_t(a.getDimSize(0)), uint64_t(b.getDimSize(1)),
                     uint64_t(a.getDimSize(1))};

    auto key = std::make_tuple(request.shape.m,request.shape.n,request.shape.k);
    auto found = selectedPlans.find(key);
    if (found == selectedPlans.end()) {
      auto candidates = enumerateTilingPlans(request);
      if (!candidates) return mm.emitError(candidates.error);
      // Static lowering requires exact divisibility at each memory level.
      auto selected = llvm::find_if(candidates.value,[&](const TilingPlan &p) {
        return request.shape.m % p.l2.m == 0 && request.shape.n % p.l2.n == 0 &&
               request.shape.k % p.l1.k == 0;
      });
      if (selected == candidates.value.end())
        return mm.emitError("no static divisible tiling candidate fits");
      found = selectedPlans.emplace(key,std::move(*selected)).first;
    }
    const auto &p = found->second;

    // 고른 계획을 사람이 읽을 문자열로 남긴다 — 단계 파일의 plena.plans.
    explanations.push_back(r.getStringAttr(describe(p)));
    // 이 첫 텐서 조각에서는 융합 그룹 경계가 곧 op 하나다.
    mm->setAttr("plena.fusion_group", r.getI64IntegerAttr(explanations.size() - 1));

    // 원래 누산기 자리를 빈 텐서로 갈아 끼운다. 타일마다 자기 누산기를
    // 새로 잡을 것이므로 여기 있던 0 채움은 더 이상 쓰이지 않는다.
    auto init = mm.getDpsInitOperand(0)->get();
    r.setInsertionPoint(mm);
    auto empty = r.create<tensor::EmptyOp>(
        mm.getLoc(), cast<RankedTensorType>(init.getType()).getShape(), a.getElementType());
    mm.getDpsInitOperand(0)->set(empty);

    // 가중치를 패널 단위로 모아 달라고 표시한다.
    if (failed(packWeightPanels(r, mm, p.l2.n))) return failure();

    // ── ① 패널: L2 에 올릴 덩어리 ───────────────────────────────────
    auto panel = tile(r, mm, {int64_t(p.l2.m), int64_t(p.l2.n), 0});
    if (failed(panel) || failed(promote(r, *panel, 1, true, true))) return failure();
    auto panelLoop = (*panel)->getParentOfType<scf::ForOp>();

    // ── ② 코어: 코어마다 연속된 열 구간 하나 ────────────────────────
    // 구간들은 서로 독립이다 — 같은 L2 패널을 읽고, 그 패널의 겹치지 않는
    // 조각에 쓰고, 나머지는 각자 L1 에 둔다. 그래서 코어를 늘려도 L2 를 더
    // 쓰지 않는다. 라운드로빈이 아니라 연속 구간인 이유는, 그래야 활성값을
    // 구간 안에서 한 번만 올리면 되기 때문이다.
    auto lane = coreLane(p.l2.n, hw.cores);
    const bool residentPanel = p.l1.m > 32 || p.l1.n > 32;
    auto group = tile(r, *panel, {int64_t(residentPanel ? p.l2.m : divisor(p.l2.m)), int64_t(lane), 0});
    if (failed(group)) return failure();
    if (lane != p.l2.n)
      if (auto independent = (*group)->getParentOfType<scf::ForOp>())
        independent->setAttr("plena.core_split", r.getUnitAttr());

    // Full-K L1 panels reuse both operands across several array tiles. Never
    // move a K loop outside these tiles: the machine has one accumulator.
    auto local = group;
    if (residentPanel) {
      local = tile(r, *group, {int64_t(p.l1.m), int64_t(p.l1.n), 0});
      if (failed(local) || failed(promote(r, *local, 2, true, false))) return failure();
    }
    // ── ③ 배열: 32x32 배열이 한 번에 처리하는 타일 ──────────────────
    auto array = tile(r, *local, {int64_t(divisor(p.l1.m)), int64_t(divisor(p.l1.n)), 0});
    if (failed(array) || failed(promote(r, *array, 2, false, true))) return failure();

    // ── ④ K 조각: 누적 축을 나눈다 ──────────────────────────────────
    auto chunk = tile(r, *array, {0, 0, int64_t(p.l1.k)});
    if (failed(chunk) || (!residentPanel && failed(promote(r, *chunk, 2, true, false)))) return failure();
    (*chunk)->setAttr("plena.accumulate", r.getUnitAttr());
    if (!residentPanel) {
      if (auto *weight = (*chunk).getDpsInputOperand(1)->get().getDefiningOp())
        weight->setAttr("plena.prefetch_weight",r.getUnitAttr());
      // 활성값은 타일이 어느 열에 쓰는지와 무관하다. 그래서 타일마다 올리면 같은
      // 바이트를 타일 수만큼 L2 에서 다시 읽게 된다. 코어당 한 번이면 충분하고,
      // 코어가 맡은 열 구간이 정확히 그 범위다.
      if (auto *activation = (*chunk).getDpsInputOperand(0)->get().getDefiningOp()) {
        activation->setAttr("plena.stage_activation",r.getUnitAttr());
        // 패널 안의 재사용은 hoist가 처리한다. 전체 K 활성값이 N 패널 사이에서
        // 반복될 때만 별도 L1 캐시를 사용한다.
        if (request.shape.m <= 32 && p.l1.k == request.shape.k && p.l2.n < request.shape.n)
          activation->setAttr("plena.cache_activation",r.getUnitAttr());
      }
    }
    // 패널 루프를 스테이지 수만큼 펼치는 것이 곧 스테이징을 그 수만큼 존재하게
    // 만드는 일이다. 배치 단계는 루프 본문 하나의 모든 버퍼를 동시에 살아 있는
    // 것으로 보므로 복사본들이 서로 다른 주소에 내리고, 그러면 다음 패널의 전송을
    // 이번 패널의 계산 뒤로 미룰 이유가 사라진다 — 즉 겹쳐서 돈다.
    if (p.stages > 1 && panelLoop && failed(loopUnrollByFactor(panelLoop,p.stages)))
      return mm.emitError("panel loop does not unroll to the planned stages");
  }
  if (failed(cleanIR(m)) || failed(hoistActivations(m,r))) return failure();
  if (failed(tileTransposes(m,r,explanations))) return failure();
  // generic 은 행렬곱 타일링이 끝난 뒤에 모은다. 그래야 그 피연산자가 교체되어
  // 사라진 값이 아니라 잘린 결과를 가리킨다.
  if (failed(tileElementwiseGraph(m,hw,r,explanations))) return failure();
  m->setAttr("plena.plans",r.getArrayAttr(explanations));
  m->setAttr("plena.stage",r.getStringAttr("tiled"));
  return cleanIR(m);
}
} // namespace plena
