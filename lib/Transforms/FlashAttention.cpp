//===----------------------------------------------------------------------===//
// FlashAttention.cpp — Stage 1 끝 — FlashAttention 루프 인식
//
// 이 파이프라인은 텐서를 모두 DRAM 에 두고 타일만 칩 안으로 올린다. 그래서 칩
// 안에서 쓰는 메모리는 타일 크기로 정해지고 텐서 크기와 무관하다. FlashAttention
// 루프는 예외로 다룰 가치가 있다: KV 블록을 도는 루프가 반복 사이에 넘기는
// 상태(행 최댓값 m, 행 합 l, 출력 O)와 반복 안의 중간값(S, P, …)은 크기가
// 루프 본문 모양(Q 블록 × KV 블록)으로 정해지고 반복 횟수(시퀀스 길이)와
// 무관하다. 칩 안에 두어도 "항상 들어간다"는 보장이 깨지지 않는다.
//
// 여기서는 그런 루프를 구조로 알아보고 npu.flash_attention 을 붙인다. Stage 3
// 가 이 표시가 붙은 루프의 버퍼만 L2 에 둔다. 알아보지 못한 루프는 표시 없이
// 기존 경로로 내려가므로, 인식이 빗나가도 결과가 틀리지는 않고 느릴 뿐이다.
//
// 알아보는 모양(online softmax):
//   m_new = max(m, rowmax(S))     S 는 이 루프 안 행렬곱에서 나온다
//   O_new = f(O, P·V)             P 는 exp 에서 나온다
// scale, 마스크, 캐스트 같은 원소별 연산이 사이에 끼어도 된다.
//===----------------------------------------------------------------------===//
#include "GraphInternal.h"

using namespace mlir;
namespace npu {
namespace {
template <typename T> bool contains(linalg::GenericOp g) {
  return !g.getRegion().front().getOps<T>().empty();
}

// v 를 원소별 generic 들을 거슬러 올라가며, `found` 를 만족하는 루프 안의 연산에
// 닿는가. 리덕션이나 루프 밖 값에서 멈춘다.
bool reachesBack(Value v, scf::ForOp loop, llvm::function_ref<bool(Operation *)> found,
                 unsigned depth = 8) {
  Operation *def = v.getDefiningOp();
  if (!def || !loop->isAncestor(def)) return false;
  if (found(def)) return true;
  auto g = dyn_cast<linalg::GenericOp>(def);
  if (!g || !depth || g.getNumReductionLoops() != 0) return false;
  for (Value in : g.getDpsInputs())
    if (reachesBack(in, loop, found, depth - 1)) return true;
  return false;
}

bool isOnlineSoftmaxLoop(scf::ForOp loop) {
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  auto fromMatmul = [&](Value v) {
    return reachesBack(v, loop, [](Operation *op) { return isa<linalg::MatmulOp>(op); });
  };
  auto fromExp = [&](Value v) {
    return reachesBack(v, loop, [](Operation *op) {
      auto g = dyn_cast<linalg::GenericOp>(op);
      return g && contains<math::ExpOp>(g);
    });
  };
  bool runningMax = false, accumulatedOutput = false;
  for (auto [carried, next] : llvm::zip(loop.getRegionIterArgs(), yield.getOperands())) {
    auto type = dyn_cast<RankedTensorType>(carried.getType());
    auto update = next.getDefiningOp<linalg::GenericOp>();
    if (!type || !update || !loop->isAncestor(update) ||
        !llvm::is_contained(update->getOperands(), carried))
      continue;
    // m_new = max(m, rowmax(S))
    if (isRowShape(type) && type.getElementType().isF32() && contains<arith::MaximumFOp>(update))
      for (Value operand : update->getOperands())
        if (auto reduce = operand.getDefiningOp<linalg::GenericOp>();
            reduce && loop->isAncestor(reduce))
          if (const char *name = reductionOf(reduce);
              name && StringRef(name).contains("MAX") && fromMatmul(reduce.getDpsInputs()[0]))
            runningMax = true;
    // O_new = f(O, P·V)
    if (type.getRank() == 2)
      for (Value operand : update->getOperands())
        if (auto pv = operand.getDefiningOp<linalg::MatmulOp>();
            pv && loop->isAncestor(pv) && fromExp(pv.getDpsInputs()[0]))
          accumulatedOutput = true;
  }
  return runningMax && accumulatedOutput;
}
} // namespace

unsigned markFlashAttentionLoops(ModuleOp m) {
  unsigned marked = 0;
  m.walk([&](scf::ForOp loop) {
    if (!isOnlineSoftmaxLoop(loop)) return;
    loop->setAttr("npu.flash_attention", UnitAttr::get(m.getContext()));
    ++marked;
  });
  return marked;
}

//===----------------------------------------------------------------------===//
// Stage 4b — FlashAttention 소프트웨어 파이프라이닝 (npu-pipeline-attention)
//
// 표시된 KV 루프의 한 반복은 두 단계다.
//   stage 0: K_j 를 올리고 S_j = Q·K_jᵀ (Matrix)
//   stage 1: softmax_j (VPU), P_j·V_j (Matrix), m·l·O 갱신 (VPU)
// 루프 그대로면 명령 순서가 QK_0 → softmax_0 → PV_0 → QK_1 → … 이고, 코어가 명령을
// 순서대로 내므로 softmax_j 와 QK_{j+1} 은 데이터 의존이 없는데도 겹치지 못한다.
// 이 pass 가 순서를
//   prologue  QK_0
//   steady    [V_j, K_{j+1} 적재] QK_{j+1} · softmax_j · (S_{j+1} writeout) · PV_j · O 갱신
//   epilogue  softmax_{N-1} · PV_{N-1}
// 로 바꾼다. mlir-air 의 ping-pong 변환(air-unroll-loop-for-pipelining-pattern →
// air-hoist-alloc-in-for-pattern → air-construct-ping-pong-dependency-pattern)과
// 같은 순서의 일이다.
//
//   1. 루프를 펼친다. 본문의 memref.alloc 이 반복마다 복제되므로 S_j 와 S_{j+1} 이
//      서로 다른 버퍼가 된다(버퍼 복제). 주소는 아직 없고, 뒤의
//      npu-assign-addresses 가 생존 구간을 보고 겹치지 않는 S 끼리 주소를 나눠
//      쓰게 한다 — ping/pong 두 주소가 번갈아 쓰인다.
//   2. stage 0 을 한 반복 앞으로 당긴다(QK_{j+1} 을 softmax_j 앞으로).
//   3. QK_{j+1} 의 결과 내보내기(acc_writeout 과 그 결과를 L2 로 옮기는 복사)를
//      그 결과를 쓰는 연산이나 다음 Matrix 연산 직전까지 미룬다. 코어는 M_MMA 를
//      내고 바로 softmax_j 의 VPU 명령을 낸다. 두 유닛이 동시에 돈다.
//   4. stage 1 의 DRAM→L2 적재(V_j)를 stage 0 앞으로 올린다. 그래야 GDMA 가
//      core block 을 반복 중간에서 끊지 않는다.
// 옮길 때마다 버퍼 접근(읽기/쓰기)이 겹치는지 검사한다. AIR 의 토큰 의존성에
// 해당하는 것이 이 검사다. 검사에 걸리거나 모양이 예상과 다르면 그 루프는 펼치기만
// 하고 순서는 바꾸지 않는다(결과는 같고 겹침만 없다).
//===----------------------------------------------------------------------===//
namespace {
using mlir::npu::AccWriteoutOp;
using mlir::npu::AccZeroOp;
using mlir::npu::GlobalCopyOp;
using mlir::npu::LocalCopyOp;
using mlir::npu::MmaOp;

Value rootOf(Value v) {
  while (Operation *def = v.getDefiningOp()) {
    auto view = dyn_cast<ViewLikeOpInterface>(def);
    if (!view) break;
    v = view.getViewSource();
  }
  return v;
}

struct Touch { Value root; bool write; };

// 연산(과 그 안의 연산)이 읽고 쓰는 버퍼.
void touches(Operation *op, SmallVectorImpl<Touch> &out) {
  op->walk([&](Operation *inner) {
    auto add = [&](Value v, bool write) {
      if (isa<BaseMemRefType>(v.getType())) out.push_back({rootOf(v), write});
    };
    if (auto c = dyn_cast<LocalCopyOp>(inner)) { add(c.getSource(), false); add(c.getTarget(), true); }
    else if (auto c = dyn_cast<GlobalCopyOp>(inner)) { add(c.getSource(), false); add(c.getTarget(), true); }
    else if (auto mm = dyn_cast<MmaOp>(inner)) { add(mm.getActivation(), false); add(mm.getWeight(), false); }
    else if (auto w = dyn_cast<AccWriteoutOp>(inner)) add(w.getTarget(), true);
    else if (auto l = dyn_cast<linalg::LinalgOp>(inner)) {
      for (Value v : l.getDpsInputs()) add(v, false);
      for (Value v : l.getDpsInits()) add(v, true);
    } else if (auto c = dyn_cast<memref::CopyOp>(inner)) { add(c.getSource(), false); add(c.getTarget(), true); }
    else if (!isa<ViewLikeOpInterface, memref::AllocOp, memref::DeallocOp>(inner) &&
             !inner->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
      for (Value v : inner->getOperands()) add(v, true);  // 모르는 연산은 읽고 쓴다고 본다
  });
}

bool conflict(ArrayRef<Touch> a, ArrayRef<Touch> b) {
  for (const Touch &x : a)
    for (const Touch &y : b)
      if (x.root == y.root && (x.write || y.write)) return true;
  return false;
}

bool isMatrix(Operation *op) {
  bool found = false;
  op->walk([&](Operation *inner) { if (isa<AccZeroOp, MmaOp, AccWriteoutOp>(inner)) found = true; });
  return found;
}
bool isGlobal(Operation *op) {
  bool found = false;
  op->walk([&](GlobalCopyOp) { found = true; });
  return found;
}

// `group` 을 (순서를 지킨 채) `before` 바로 앞으로 옮겨도 되는가: 사이에 있는 연산과
// 버퍼 접근이 겹치지 않고, 그 연산들이 만든 값을 쓰지 않아야 한다.
bool canHoist(ArrayRef<Operation *> group, Operation *before) {
  SmallVector<Touch> mine;
  DenseSet<Operation *> members(group.begin(), group.end());
  for (Operation *op : group) touches(op, mine);
  // 그룹 구성원이 흩어져 있을 수 있다(예: 앞쪽에 놓인 alloc 과 뒤쪽의 적재).
  // 첫 구성원이 아니라 마지막 구성원까지, 그 사이의 구성원 아닌 연산을 모두 본다.
  Operation *last = group.front();
  for (Operation *op : group)
    if (last->isBeforeInBlock(op)) last = op;
  for (Operation *op = before; op && op != last; op = op->getNextNode()) {
    if (members.contains(op)) continue;
    SmallVector<Touch> theirs;
    touches(op, theirs);
    if (conflict(mine, theirs)) return false;
    for (Operation *member : group)
      for (Value operand : member->getOperands())
        if (operand.getDefiningOp() == op) return false;
  }
  return true;
}

struct Iteration {
  SmallVector<Operation *> ops;   // 펼친 한 반복의 최상위 연산, 순서대로
  size_t stage1 = 0;              // stage 1 의 첫 연산 위치
  AccWriteoutOp writeout;         // S 의 writeout
  Operation *store = nullptr;     // writeout 결과를 L2 로 옮기는 복사
};

// stage 0 = 첫 행렬곱(QK)과 그 결과를 L2 S 로 옮기는 복사까지의 앞부분.
bool splitStages(Iteration &it) {
  for (size_t i = 0; i < it.ops.size(); ++i) {
    Operation *op = it.ops[i];
    if (auto w = dyn_cast<AccWriteoutOp>(op); w && !it.writeout) { it.writeout = w; continue; }
    if (it.writeout) {
      auto copy = dyn_cast<LocalCopyOp>(op);
      if (!copy || copy.getSource() != it.writeout.getTarget()) return false;
      it.store = op; it.stage1 = i + 1;
      break;
    }
    if (!isa<arith::ConstantOp, arith::MulIOp, arith::AddIOp, affine::AffineApplyOp, memref::SubViewOp,
             memref::AllocOp, GlobalCopyOp, LocalCopyOp, AccZeroOp, MmaOp>(op))
      return false;
  }
  return it.store && it.stage1 < it.ops.size();
}

LogicalResult pipelineLoop(scf::ForOp loop) {
  auto constant = [](Value v) { return getConstantIntValue(v); };
  auto lb = constant(loop.getLowerBound()), ub = constant(loop.getUpperBound()), step = constant(loop.getStep());
  if (!lb || !ub || !step || *step <= 0 || loop.getNumRegionIterArgs() != 0 ||
      !isa<func::FuncOp>(loop->getParentOp()))
    return success();  // 이 모양이 아니면 그대로 둔다
  const int64_t trips = (*ub - *lb + *step - 1) / *step;
  if (trips < 2) return success();
  // 1. 펼치기: 반복마다 본문을 복제한다(alloc 도 복제된다).
  OpBuilder b(loop);
  MLIRContext *context = loop.getContext();
  SmallVector<Iteration> iterations(trips);
  for (int64_t i = 0; i < trips; ++i) {
    IRMapping map;
    auto iv = b.create<arith::ConstantIndexOp>(loop.getLoc(), *lb + i * *step);
    iv->setAttr("npu.kv_block", b.getI64IntegerAttr(i));
    iterations[i].ops.push_back(iv);
    map.map(loop.getInductionVar(), iv);
    for (Operation &op : loop.getBody()->without_terminator()) {
      Operation *clone = b.clone(op, map);
      clone->setAttr("npu.kv_block", b.getI64IntegerAttr(i));
      iterations[i].ops.push_back(clone);
    }
  }
  loop->erase();
  for (auto &it : iterations)
    if (!splitStages(it)) {
      // 예상한 모양이 아니다: 펼친 채로 두고 순서는 바꾸지 않는다.
      iterations.front().ops.front()->getParentOfType<func::FuncOp>()->setAttr(
          "npu.pipeline_skipped", UnitAttr::get(context));
      return success();
    }
  unsigned overlapped = 0;
  for (int64_t i = 0; i + 1 < trips; ++i) {
    Iteration &cur = iterations[i], &next = iterations[i + 1];
    Operation *stage1Begin = cur.ops[cur.stage1];
    // 2. stage 0 (QK_{i+1}) 를 stage 1 (softmax_i …) 앞으로.
    ArrayRef<Operation *> stage0(next.ops.data(), next.stage1);
    if (!canHoist(stage0, stage1Begin)) continue;
    for (Operation *op : stage0) op->moveBefore(stage1Begin);
    // 4. stage 1 의 DRAM→L2 적재(V_i)를 stage 0 앞으로. 피연산자를 만드는
    //    subview/alloc 이 뒤에 있으면 함께 옮긴다.
    Operation *front = stage0.front();
    for (size_t k = cur.stage1; k < cur.ops.size(); ++k) {
      auto load = dyn_cast<GlobalCopyOp>(cur.ops[k]);
      if (!load || mlir::npu::memorySpace(load.getSource().getType()) != 0) continue;
      SmallVector<Operation *> group;
      for (Value operand : load->getOperands())
        if (Operation *def = operand.getDefiningOp();
            def && def->getBlock() == load->getBlock() && front->isBeforeInBlock(def) &&
            isa<memref::AllocOp, memref::SubViewOp>(def) &&
            llvm::all_of(def->getOperands(), [&](Value v) {
              Operation *d = v.getDefiningOp();
              return !d || d->getBlock() != load->getBlock() || d->isBeforeInBlock(front);
            }))
          group.push_back(def);
      group.push_back(load);
      if (!canHoist(group, front)) continue;
      for (Operation *op : group) op->moveBefore(front);
      front = group.front();
    }
    // 3. S_{i+1} 내보내기를 그것이 필요해지는 곳까지 미룬다.
    SmallVector<Operation *> drain{next.writeout, next.store};
    SmallVector<Touch> drained;
    for (Operation *op : drain) touches(op, drained);
    Operation *after = next.store;
    for (Operation *op = next.store->getNextNode(); op && !op->hasTrait<OpTrait::IsTerminator>();
         op = op->getNextNode()) {
      SmallVector<Touch> theirs;
      touches(op, theirs);
      if (isMatrix(op) || isGlobal(op) || conflict(drained, theirs)) break;
      after = op;
    }
    if (after != next.store) {
      Operation *insertBefore = after->getNextNode();
      for (Operation *op : drain) op->moveBefore(insertBefore);
      ++overlapped;
    }
  }
  iterations.front().ops.front()->getParentOfType<func::FuncOp>()->setAttr(
      "npu.pipelined_blocks", IntegerAttr::get(IntegerType::get(context, 64), overlapped));
  return success();
}
} // namespace

LogicalResult pipelineFlashAttention(ModuleOp m) {
  SmallVector<scf::ForOp> loops;
  m.walk([&](scf::ForOp loop) { if (loop->hasAttr("npu.flash_attention")) loops.push_back(loop); });
  for (auto loop : loops)
    if (failed(pipelineLoop(loop))) return failure();
  return success();
}
} // namespace npu
