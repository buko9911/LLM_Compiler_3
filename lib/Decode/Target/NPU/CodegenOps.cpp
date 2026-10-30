//===- CodegenOps.cpp - NPU codegen: copy and compute op lowering -----------===//
//
// Codegen members that lower memref.copy / linalg.transpose to DMA and the compute ops to
// vector / scalar / matrix instructions: fill, FP16 matmul, INT8 GEMV, quant and dequant.
//===----------------------------------------------------------------------===//

#include "CodegenImpl.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include <algorithm>
#include <functional>
#include <set>

namespace mlir::hybrid::codegen {

// ---- data movement ops ---------------------------------------------------------------------
LogicalResult Codegen::lowerCopy(Operation *op, Value srcV, Value dstV, ArrayRef<int64_t> perm, std::string &err) {
  if (op == prefetchedWeightCopy) {
    if (prefetchedWeightNeedsWait) {
      load(10, 1);
      emit(rform(OP_WAIT_EVENT, 10));
    }
    prefetchedWeightCopy = nullptr;
    return success();
  }
  FailureOr<View> s0 = viewOf(srcV, err), d0 = viewOf(dstV, err);
  if (failed(s0) || failed(d0)) return failure();
  if (s0->layout == "pim_out") return gatherPimOut(*s0, *d0, err);
  if (d0->layout == "pim_act") {
    if (parallelActivationFeed && parallelActivationFeed->copy == op)
      return scatterPimActParallel(*s0, *d0, err);
    return scatterPimAct(*s0, *d0, err);
  }
  if (s0->layout != "dense" || d0->layout != "dense") {
    err = "copy " + (s0->layout != "dense" ? "from a " + s0->layout : "to a " + d0->layout) +
          " LP6 buffer: PIM layouts are only gathered from (pim_out) or scattered to (pim_act)";
    return failure();
  }
  View s = *s0, d = *d0;
  if (!perm.empty()) {    // linalg.transpose: dst dim i = src dim perm[i]
    View p = s;
    for (size_t i = 0; i < perm.size(); ++i) {
      p.shape[i] = s.shape[perm[i]];
      p.strides[i] = s.strides[perm[i]];
    }
    s = p;
  }
  if (s.shape != d.shape) {
    err = "copy shape mismatch";
    return failure();
  }
  for (int i = int(s.shape.size()) - 1; i >= 0 && s.shape.size() > 1; --i)
    if (s.shape[i] == 1) {
      s.shape.erase(s.shape.begin() + i);
      s.strides.erase(s.strides.begin() + i);
      d.shape.erase(d.shape.begin() + i);
      d.strides.erase(d.strides.begin() + i);
    }
  if (s.shape.empty()) {
    s.shape = d.shape = {1};
    s.strides = d.strides = {1};
  }
  int rank = int(d.shape.size());
  int64_t runElems = 1;
  int inner = 0;
  while (inner < rank) {
    int dim = rank - 1 - inner;
    if (s.strides[dim] != runElems || d.strides[dim] != runElems) break;
    runElems *= d.shape[dim];
    ++inner;
  }
  // A one-column tail can lose its unit innermost dimension above, leaving
  // a strided vector (for example K[:, 64:65] from a 65-column matrix).
  // One scalar is still a contiguous DMA row. With inner == 0 the existing
  // 2-D lowering uses runElems=1, rows=last_dim and each side's real stride.
  // This copies exactly the requested elements, including the gather cost.
  uint32_t runBytes = uint32_t(runElems * d.eb);
  if (auto it = streamSlot.find(op); it != streamSlot.end()) {
    if (inner != rank || d.space != kL1) {
      err = "stream copy must be contiguous into L1";
      return failure();
    }
    ldma(addr(d), it->second, runBytes, 1, runBytes, runBytes, true);
    return success();
  }
  int rowDim = rank - inner - 1;
  std::string terr;
  auto body = [&](ArrayRef<int64_t> idx) {
    int64_t so = 0, dof = 0;
    for (size_t k = 0; k < idx.size(); ++k) {
      so += idx[k] * s.strides[k];
      dof += idx[k] * d.strides[k];
    }
    uint32_t rows = rowDim >= 0 ? uint32_t(d.shape[rowDim]) : 1;
    uint32_t sStride = rowDim >= 0 ? uint32_t(s.strides[rowDim] * s.eb) : runBytes;
    uint32_t dStride = rowDim >= 0 ? uint32_t(d.strides[rowDim] * d.eb) : runBytes;
    transfer2d(s.space, d.space, addr(s, so), addr(d, dof), runBytes, rows, sStride, dStride, terr);
  };
  SmallVector<int64_t> outerShape(d.shape.begin(), d.shape.begin() + std::max(rowDim, 0));
  if (rowDim < 0) body({});
  else forEachOuter(outerShape, 0, body);
  if (!terr.empty()) {
    err = terr;
    return failure();
  }
  if (in.spec->npu.stagingEarlyRelease && s.space == kLP6 && d.space == kL1) {
    // A normal buffer-copy boundary has no live vector/scalar/SA value: other
    // lowerings store their results to L1 before returning. Never place this
    // cut inside a reduction, row loop, or event-1 weight prefetch. The stream,
    // prefetched, and PIM paths above return before reaching this point.
    // Register L2 reads against this load-only completion, so future GDMA
    // overwrites need not wait for unrelated compute in the next block.
    flush();
    ++stagingReleaseCuts;
  }
  return success();
}

// ---- compute ops ----------------------------------------------------------------------------
bool Codegen::isMatmulInit(Operation *user, Value out) {
  if (!isa<linalg::MatmulOp, linalg::BatchMatmulOp>(user) && !user->hasAttr("hyb.npu_gemv")) return false;
  auto matrix = dyn_cast<linalg::LinalgOp>(user);
  // INIT replaces the accumulator, but it does not initialize matrix inputs.
  // When the destination is also an input, that read needs the actual zero
  // bytes. Aliases reached through views already prevent fill elision because
  // the view itself is a non-matrix user of the filled buffer.
  return matrix && matrix.getNumDpsInits() == 1 && matrix.getDpsInits()[0] == out &&
         llvm::none_of(matrix.getDpsInputs(), [&](Value input) { return input == out; });
}

LogicalResult Codegen::checkZeroInit(linalg::LinalgOp operation, std::string &err) {
  // CSE may hoist a zero-filled tensor and bufferization then copies that
  // template into each matrix destination. Prove the logical initialization
  // through those copies instead of remembering physical address ranges:
  // addresses can be recycled and a later write can invalidate such a record.
  auto rootOf = [&](Value value) {
    while (Operation *op = value.getDefiningOp()) {
      if (!isViewOp(op)) break;
      value = op->getOperand(0);
    }
    return value;
  };
  std::function<bool(Operation *, Value)> writes = [&](Operation *op, Value root) {
    if (isViewOp(op) || isa<memref::AllocOp>(op)) return false;
    if (auto copy = dyn_cast<memref::CopyOp>(op)) return rootOf(copy.getTarget()) == root;
    if (auto dps = dyn_cast<DestinationStyleOpInterface>(op))
      return llvm::any_of(dps.getDpsInits(), [&](Value value) { return rootOf(value) == root; });
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (Operation &nested : block)
          if (writes(&nested, root)) return true;
    // Unknown consumers of this allocation are potential writes. In
    // particular, never infer zero across an opaque call or deallocation.
    return llvm::any_of(op->getOperands(), [&](Value value) {
      return isa<MemRefType>(value.getType()) && rootOf(value) == root;
    });
  };
  std::function<bool(Value, Operation *, unsigned)> zeroBefore =
      [&](Value value, Operation *before, unsigned depth) {
    if (!depth) return false;
    Value root = rootOf(value);
    if (!root.getDefiningOp<memref::AllocOp>()) return false;
    for (;;) {
      for (Operation *previous = before->getPrevNode(); previous; previous = previous->getPrevNode()) {
        if (previous == root.getDefiningOp()) return false;
        if (!writes(previous, root)) continue;
        if (auto fill = dyn_cast<linalg::FillOp>(previous)) {
          Value destination = fill.getOutputs()[0];
          return (destination == value || destination == root) &&
                 constValue(fill.getInputs()[0]) == std::optional<double>(0.0);
        }
        if (auto copy = dyn_cast<memref::CopyOp>(previous)) {
          Value destination = copy.getTarget();
          return (destination == value || destination == root) &&
                 zeroBefore(copy.getSource(), copy, depth - 1);
        }
        return false;
      }
      Operation *parent = before->getParentOp();
      if (!isa<scf::ForOp, scf::ForallOp>(parent)) return false;
      // Crossing a loop boundary is valid only for an invariant zero
      // template. A write later in the loop may precede this read on its next
      // iteration (or on another parallel worker).
      if (writes(parent, root)) return false;
      before = parent;
    }
  };
  if (zeroBefore(operation.getDpsInits()[0], operation, 32)) return success();
  err = "matmul accumulator requires a proven zero fill or zero-template copy without intervening writes";
  return failure();
}

LogicalResult Codegen::lowerFill(linalg::FillOp f, std::string &err) {
  Value out = f.getOutputs()[0];
  auto c = constValue(f.getInputs()[0]);
  if (!c) {
    err = "fill of a non-constant";
    return failure();
  }
  FailureOr<View> v = viewOf(out, err);
  if (failed(v)) return failure();
  // hyb-npu-matrix-init proves that the first consumer fully replaces this
  // private destination with an ISA INIT. The logical fill remains in IR so
  // checkZeroInit and later memory planning retain the arithmetic contract.
  if (f->hasAttr("hyb.elide_matrix_init")) return success();
  // A shared zero template may also initialize online state through copies.
  // Those reads need real bytes, even if one later matrix uses the template
  // directly. Only an exclusively matrix-init fill can be elided.
  if (*c == 0.0 && llvm::all_of(out.getUsers(), [&](Operation *u) {
        return u == f.getOperation() || isMatmulInit(u, out);
      })) return success();
  int dt = dtypeOf(v->et);
  int64_t slot = vectorConst(*c, dt);
  uint32_t loadF = dt == DT_F16 ? 1 : dt == DT_F32 ? 3 : dt == DT_I8 ? 5 : 7;
  uint32_t storeF = loadF + 1;
  int64_t lanes = 64 / v->eb;
  int64_t inner = v->shape.empty() ? 1 : v->shape.back();
  if (!v->shape.empty() && v->strides.back() != 1) {
    err = "fill with a strided innermost dim";
    return failure();
  }
  setVL(uint32_t(std::min(lanes, inner)));
  vload(0, uint64_t(poolL1 + slot), loadF);
  if (v->shape.empty()) {
    setVL(1);
    vstore(addr(*v), 0, storeF);
    return success();
  }
  forEachOuter(v->shape, 1, [&](ArrayRef<int64_t> idx) {
    int64_t off = 0;
    for (size_t k = 0; k < idx.size(); ++k) off += idx[k] * v->strides[k];
    for (int64_t s = 0; s < inner; s += lanes) {
      setVL(uint32_t(std::min(lanes, inner - s)));
      vstore(addr(*v, off + s), 0, storeF);
    }
  });
  return success();
}

LogicalResult Codegen::lowerMatmul(linalg::LinalgOp mm, std::string &err) {
  bool batched = isa<linalg::BatchMatmulOp>(mm.getOperation());
  FailureOr<View> A = viewOf(mm.getDpsInputs()[0], err), B = viewOf(mm.getDpsInputs()[1], err),
                  C = viewOf(mm.getDpsInits()[0], err);
  if (failed(A) || failed(B) || failed(C)) return failure();
  if (!A->et.isF16() || !B->et.isF16()) {
    err = "matmul operands must be f16";
    return failure();
  }
  if (failed(checkZeroInit(mm, err))) return failure();
  auto maps = mm.getIndexingMapsArray();
  int nb = batched ? 1 : 0;
  AffineMap bm = maps[1];
  unsigned kDim = maps[0].getNumDims() - 1;
  bool transB = bm.getResult(bm.getNumResults() - 1) == getAffineDimExpr(kDim, mm.getContext());
  int64_t batch = batched ? C->shape[0] : 1;
  int64_t M = C->shape[nb], N = C->shape[nb + 1], K = A->shape[nb + 1];
  uint32_t woF = C->et.isF16() ? 1 : 4;
  int64_t aRow = A->strides[nb] * 2, cRow = C->strides[nb] * C->eb;
  if (A->strides[nb + 1] != 1 || C->strides[nb + 1] != 1 || B->strides[nb + 1] != 1) {
    err = "matmul operands need a contiguous innermost dim";
    return failure();
  }
  int64_t bRow = B->strides[nb] * 2;
  for (int64_t b = 0; b < batch; ++b) {
    int64_t ao = batched ? b * A->strides[0] : 0, bo = batched ? b * B->strides[0] : 0,
            co = batched ? b * C->strides[0] : 0;
    if (!transB) {
      for (int64_t m0 = 0; m0 < M; m0 += 32)
        for (int64_t n0 = 0; n0 < N; n0 += 32) {
          int64_t mm_ = std::min<int64_t>(32, M - m0), nn = std::min<int64_t>(32, N - n0);
          load(7, uint32_t(addr(*B, bo + n0)));
          load(8, uint32_t(addr(*A, ao + m0 * A->strides[nb])));
          matrixInputs(matrixLoad(7, uint32_t(K), uint32_t(nn), uint32_t(bRow), 3),
                       matrixLoad(8, uint32_t(mm_), uint32_t(K), uint32_t(aRow), 7));
          mma(K, 3);
          load(9, uint32_t(addr(*C, co + m0 * C->strides[nb] + n0)));
          rec(matrixWriteout(9, uint32_t(mm_), uint32_t(nn), uint32_t(cRow), woF));
        }
    } else {
      // C[m, n0:n0+32] = rows n0..n0+31 of B (as the activation) x row m of A (a K x 1 weight).
      for (int64_t m = 0; m < M; ++m)
        for (int64_t n0 = 0; n0 < N; n0 += 32) {
          int64_t nn = std::min<int64_t>(32, N - n0);
          load(7, uint32_t(addr(*A, ao + m * A->strides[nb])));
          load(8, uint32_t(addr(*B, bo + n0 * B->strides[nb])));
          matrixInputs(matrixLoad(7, uint32_t(K), 1, 2, 3),
                       matrixLoad(8, uint32_t(nn), uint32_t(K), uint32_t(bRow), 7));
          mma(K, 3);
          load(9, uint32_t(addr(*C, co + m * C->strides[nb] + n0)));
          rec(matrixWriteout(9, uint32_t(nn), 1, uint32_t(C->eb), woF));
        }
    }
  }
  return success();
}

/// NPU GEMV: acc[m, nb, c] = sum_k q[m, k] w[nb, k, c] (INT8 x INT8 -> INT32), one MMA per
/// 32-column block (the weight block is the [K, 32] MMA tile as stored).
LogicalResult Codegen::lowerGemv(linalg::GenericOp g, std::string &err) {
  FailureOr<View> Q = viewOf(g.getDpsInputs()[0], err), W = viewOf(g.getDpsInputs()[1], err),
                  C = viewOf(g.getDpsInits()[0], err);
  if (failed(Q) || failed(W) || failed(C)) return failure();
  int64_t M = Q->shape[0], K = Q->shape[1], NB = W->shape[0];
  if (Q->strides[1] != 1 || W->strides[2] != 1 || W->strides[1] != 32 || C->strides[2] != 1 || C->strides[1] != 32) {
    err = "GEMV operands not in the MMA layout";
    return failure();
  }
  if (failed(checkZeroInit(g, err))) return failure();
  const int64_t mmas = NB * ((M + 31) / 32);
  int64_t issued = 0;
  for (int64_t j = 0; j < NB; ++j)
    for (int64_t m0 = 0; m0 < M; m0 += 32) {
      int64_t mm_ = std::min<int64_t>(32, M - m0);
      load(7, uint32_t(addr(*W, j * W->strides[0])));
      load(8, uint32_t(addr(*Q, m0 * Q->strides[0])));
      matrixInputs(matrixLoad(7, uint32_t(K), 32, 32, 1),
                   matrixLoad(8, uint32_t(mm_), uint32_t(K), uint32_t(Q->strides[0]), 5));
      mma(K, 1);
      // weightPipeline proved that the next ping/pong slot is disjoint from
      // every current weight, activation and accumulator access. Issue after
      // the first MMA, not the last: multi-column/multi-row tiles can hide the
      // next transfer under all their remaining MMAs. Event 1 is retired by
      // the loop consumer before the next iteration reuses it.
      if (nextWeight && nextWeight->gemv == g.getOperation() && j == 0 && m0 == 0) {
        const auto p = *nextWeight;
        prefetchWeight(p.l1, p.l2, p.bytes);
        nextWeight.reset();
      }
      // The VPU dequantizes the previous loop tile while the SA works on this one. Issue it before
      // the writeout: the core holds the writeout until the MMA is done.
      if (failed(emitPendingDequant(++issued, mmas, err))) return failure();
      load(9, uint32_t(addr(*C, m0 * C->strides[0] + j * C->strides[1])));
      rec(matrixWriteout(9, uint32_t(mm_), 32, uint32_t(C->strides[0] * 4), 2));
    }
  return success();
}

LogicalResult Codegen::emitPendingDequant(int64_t done, int64_t total, std::string &err) {
  if (!pendingDequant) return success();
  PendingDequant &p = *pendingDequant;
  int64_t rows = cast<ShapedType>(p.dequant.getInit().getType()).getDimSize(0);
  int64_t due = rows * done / total;              // rows owed after `done` of `total` MMAs
  if (due > p.row) {
    if (failed(lowerDequant(p.dequant, err, p.row, due - p.row, p.col0, p.cols))) return failure();
    p.row = due;
  }
  if (p.row == rows) pendingDequant.reset();
  return success();
}

LogicalResult Codegen::lowerQuant(hyb::QuantOp q, std::string &err) {
  FailureOr<View> X = viewOf(q.getInput(), err), Q = viewOf(q.getQInit(), err), S = viewOf(q.getScaleInit(), err);
  if (failed(X) || failed(Q) || failed(S)) return failure();
  int64_t M = X->shape[0], K = X->shape[1];
  uint64_t temp = scratchBase, maxSlot = temp + uint64_t(K * 2);
  for (int64_t m = 0; m < M; ++m) {
    uint64_t src = addr(*X, m * X->strides[0]), dst = addr(*Q, m * Q->strides[0]);
    for (int64_t s = 0; s < K; s += 32) {          // |x|
      setVL(uint32_t(std::min<int64_t>(32, K - s)));
      vload(0, src + s * 2, 1);
      emit(rform(OP_V_UN, 1, 0, 0, 0, 6));
      vstore(temp + s * 2, 1, 2);
    }
    reduceToScalar(temp, K, 2, maxSlot);           // max|x| (FP32)
    sload(0, maxSlot);
    sload(1, uint64_t(poolL1 + scalarConst(1.0f / 127.0f)));
    emit(rform(OP_S_ALU, 0, 0, 1, 0, 3));          // s0 = max / 127
    sstore(addr(*S, m * S->strides[0]), 0);
    for (int64_t s = 0; s < K; s += 32) {
      setVL(uint32_t(std::min<int64_t>(32, K - s)));
      vload(0, src + s * 2, 1);
      emit(rform(OP_V_CONV, 2, 0, 0, 0, 1));       // V_QUANT_F16_I8 with s0
      vstore(dst + s, 2, 6);
    }
  }
  return success();
}

/// Pairwise tree of V_REDUCE results over 32-element tiles of an f16 row: funct 1 sum,
/// 2 max; result stored as f32 at `dst`.
void Codegen::reduceToScalar(uint64_t src, int64_t elements, uint32_t funct, uint64_t dst) {
  std::set<int> occupied;
  uint32_t combine = funct == 2 ? 4 : 1;
  for (int64_t start = 0; start < elements; start += 32) {
    int level = 0;
    while (occupied.count(level)) ++level;
    uint32_t s = uint32_t(4 + level);
    setVL(uint32_t(std::min<int64_t>(32, elements - start)));
    vload(0, src + start * 2, 1);
    emit(rform(OP_V_RED, s, 0, 0, 0, funct));
    for (int lower = 0; lower < level; ++lower) {
      emit(rform(OP_S_ALU, s, uint32_t(4 + lower), s, 0, combine));
      occupied.erase(lower);
    }
    occupied.insert(level);
  }
  std::vector<int> levels(occupied.begin(), occupied.end());
  uint32_t s = uint32_t(4 + levels[0]);
  for (size_t i = 1; i < levels.size(); ++i) emit(rform(OP_S_ALU, s, uint32_t(4 + levels[i]), s, 0, combine));
  sstore(dst, s);
}

/// y[m, n] = acc[m, n] * scale[m] * wscale[n], 16 lanes per vector (INT32 -> FP16), over rows
/// [row0, row0 + rows) and columns [col0, col0 + cols). The op itself (rows < 0) covers all of
/// it, or only the columns a pipelined GEMV loop left (dequantTail).
LogicalResult Codegen::lowerDequant(hyb::DequantOp d, std::string &err, int64_t row0, int64_t rows, int64_t col0,
                                    int64_t cols) {
  FailureOr<View> A = viewOf(d.getAcc(), err), S = viewOf(d.getScale(), err), B = viewOf(d.getWscale(), err),
                  Y = viewOf(d.getInit(), err);
  if (failed(A) || failed(S) || failed(B) || failed(Y)) return failure();
  int64_t M = Y->shape[0], N = Y->shape[1];
  if (rows < 0) {
    auto tail = dequantTail.find(d);
    col0 = tail == dequantTail.end() ? 0 : tail->second;
    rows = M;
    cols = N - col0;
  }
  for (int64_t m = row0; m < row0 + rows; ++m) {
    sload(0, addr(*S, m * S->strides[0]));
    uint64_t src = addr(*A, m * A->strides[0]), dst = addr(*Y, m * Y->strides[0]);
    for (int64_t s = col0; s < col0 + cols; s += 16) {
      setVL(uint32_t(std::min<int64_t>(16, col0 + cols - s)));
      vload(0, src + s * 4, 7);
      vload(1, addr(*B, s), 3);
      emit(rform(OP_V_CONV, 2, 0, 0, 1, 3));        // V_DEQUANT_I32_F16 v2 = v0 * s0 * v1
      vstore(dst + s * 2, 2, 2);
    }
  }
  return success();
}

LogicalResult Codegen::lower(Operation *op, std::string &err) {
  for (Value o : op->getOperands())
    if (isa<MemRefType>(o.getType()) && spaceOf(o.getType()) != kL1) {
      err = "compute op with an operand outside L1 (run hyb-npu-promote)";
      return failure();
    }
  if (auto g = dyn_cast<linalg::GenericOp>(op))
    return g->hasAttr("hyb.npu_gemv") ? lowerGemv(g, err) : lowerGeneric(g, err);
  if (auto f = dyn_cast<linalg::FillOp>(op)) return lowerFill(f, err);
  if (isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op)) return lowerMatmul(cast<linalg::LinalgOp>(op), err);
  if (auto t = dyn_cast<linalg::TransposeOp>(op)) return lowerCopy(op, t.getInput(), t.getInit(), t.getPermutation(), err);
  if (auto q = dyn_cast<hyb::QuantOp>(op)) return lowerQuant(q, err);
  if (auto d = dyn_cast<hyb::DequantOp>(op)) return lowerDequant(d, err);
  err = "unsupported op " + op->getName().getStringRef().str();
  return failure();
}

} // namespace mlir::hybrid::codegen
