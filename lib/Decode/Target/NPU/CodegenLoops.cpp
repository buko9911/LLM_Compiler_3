//===- CodegenLoops.cpp - NPU codegen: blocks and loops ---------------------===//
//
// Codegen members that walk a block: scf.for is unrolled (a weight-streaming loop issues the
// GDMA of the next tiles ahead into a ring of staging slots and, when eligible, prefetches the
// next L1 weight slot while the current MMA runs); scf.forall becomes one command stream per
// logical core, joined at the end.
//===----------------------------------------------------------------------===//

#include "CodegenImpl.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/ScopeExit.h"
#include <algorithm>

namespace mlir::hybrid::codegen {

/// Ops that only define values the backend evaluates itself (constants, index arithmetic, views,
/// allocations, weight references): they emit no instructions.
bool Codegen::isPure(Operation &op) {
  if (isa<arith::ConstantOp, memref::AllocOp, memref::DeallocOp, hyb::WeightOp, affine::AffineApplyOp,
          affine::AffineMinOp, affine::AffineMaxOp, memref::SubViewOp, memref::ExpandShapeOp,
          memref::CollapseShapeOp, memref::CastOp>(op))
    return true;
  return op.getDialect() && op.getDialect()->getNamespace() == "arith" &&
         llvm::none_of(op.getOperandTypes(), [](Type type) { return isa<MemRefType>(type); });
}

LogicalResult Codegen::emitBlock(Block &b, std::string &err) {
  for (Operation &op : b) {
    if (op.hasTrait<OpTrait::IsTerminator>() || isPure(op)) continue;
    int previousSource = w.currentSource;
    if (w.traceOperations) w.currentSource = sourceIds.lookup(&op);
    auto restoreSource = llvm::scope_exit([&] { w.currentSource = previousSource; });
    LogicalResult r = success();
    if (auto f = dyn_cast<scf::ForOp>(op)) r = emitFor(f, err);
    else if (auto f = dyn_cast<scf::ForallOp>(op)) r = emitForall(f, err);
    else if (auto c = dyn_cast<memref::CopyOp>(op)) r = lowerCopy(&op, c.getSource(), c.getTarget(), {}, err);
    else r = lower(&op, err);
    if (succeeded(r) && !opError.empty()) {
      err = opError;
      r = failure();
    }
    if (failed(r)) {
      if (!isa<scf::ForOp, scf::ForallOp>(op) && err.find(" [at ") == std::string::npos) {
        std::string s;
        llvm::raw_string_ostream os(s);
        op.print(os, OpPrintingFlags().skipRegions());
        err += " [at " + s.substr(0, 240) + "]";
      }
      return failure();
    }
  }
  return success();
}

uint64_t Codegen::spanBytes(const View &v) {
  int64_t last = 0;
  for (size_t i = 0; i < v.shape.size(); ++i) last += (v.shape[i] - 1) * v.strides[i];
  return uint64_t((last + 1) * v.eb);
}

/// The L2 footprint of a view, excluding holes between batch rows unless the
/// metadata cap requires a conservative span. Keep the weight pipeline's
/// preflight consistent with the eventual LDMA accesses.
SmallVector<Codegen::L2Range> Codegen::viewRanges(const View &v) const {
  if (llvm::any_of(v.shape, [](int64_t n) { return n == 0; })) return {};
  uint64_t runBytes = uint64_t(v.eb), count = 1;
  int inner = 0;
  for (int d = int(v.shape.size()) - 1; d >= 0; --d) {
    if (v.shape[d] != 1 && uint64_t(v.strides[d] * v.eb) != runBytes) break;
    runBytes *= uint64_t(v.shape[d]);
    ++inner;
  }
  for (int d = 0; d < int(v.shape.size()) - inner; ++d) {
    if (uint64_t(v.shape[d]) > kMaxL2AccessRanges / count)
      return {{addr(v), addr(v) + spanBytes(v)}};
    count *= uint64_t(v.shape[d]);
  }
  SmallVector<L2Range> ranges;
  ranges.reserve(count);
  forEachOuter(v.shape, inner, [&](ArrayRef<int64_t> idx) {
    int64_t offset = 0;
    for (size_t d = 0; d < idx.size(); ++d) offset += idx[d] * v.strides[d];
    uint64_t start = addr(v, offset);
    ranges.push_back({start, start + runBytes});
  });
  std::sort(ranges.begin(), ranges.end());
  size_t merged = 0;
  for (auto [s, e] : ranges) {
    if (merged && s <= ranges[merged - 1].second)
      ranges[merged - 1].second = std::max(ranges[merged - 1].second, e);
    else ranges[merged++] = {s, e};
  }
  ranges.resize(merged);
  return ranges;
}

/// Pipeline fill -> weight copy -> GEMV -> optional output copy. A fused
/// epilogue writes directly into its core-private accumulator partition.
/// loops keep the original schedule; moving arbitrary copies would require alias analysis.
bool Codegen::weightPipeline(scf::ForOp f, memref::CopyOp stream, linalg::GenericOp &gemv, memref::CopyOp &output,
                             std::string &err) {
  linalg::FillOp fill;
  int stage = 0;
  for (Operation &op : f.getBody()->without_terminator()) {
    if (isPure(op)) continue;
    if (auto x = dyn_cast<linalg::FillOp>(op); x && stage == 0) {
      fill = x;
      stage = 1;
    } else if (&op == stream.getOperation() && stage == 1) {
      stage = 2;
    } else if (auto x = dyn_cast<linalg::GenericOp>(op); x && stage == 2 && x->hasAttr("hyb.npu_gemv")) {
      gemv = x;
      stage = 3;
    } else if (auto x = dyn_cast<memref::CopyOp>(op); x && stage == 3) {
      output = x;
      stage = 4;
    } else {
      return false;
    }
  }
  bool direct = stage == 3 && gemv->hasAttr("hyb.gemv_epilogue");
  if ((!direct && stage != 4) || f.getNumRegionIterArgs() || gemv.getDpsInputs()[1] != stream.getTarget() ||
      fill.getOutputs()[0] != gemv.getDpsInits()[0] ||
      (output && output.getSource() != gemv.getDpsInits()[0]) ||
      constValue(fill.getInputs()[0]) != std::optional<double>(0.0)) return false;
  // The activation is loop invariant; both weight slots must be distinct from it and all
  // output slots. Allocation metadata makes this check independent of the iteration.
  Operation *qdef = gemv.getDpsInputs()[0].getDefiningOp();
  if (qdef && f->isProperAncestor(qdef)) return false;
  auto wa = stream.getTarget().getDefiningOp<memref::AllocOp>();
  Value accumulator = gemv.getDpsInits()[0], root = accumulator;
  if (direct)
    while (Operation *op = root.getDefiningOp()) {
      if (!isViewOp(op)) break;
      root = op->getOperand(0);
    }
  auto ca = root.getDefiningOp<memref::AllocOp>();
  if (!wa || !ca || wa->getParentOp() != f ||
      ca->getParentOp() != (direct ? f->getParentOp() : f.getOperation())) return false;
  auto buffers = wa->getAttrOfType<IntegerAttr>("hyb.buffers");
  auto slot = wa->getAttrOfType<IntegerAttr>("hyb.slot_bytes");
  auto cb = ca->getAttrOfType<IntegerAttr>("hyb.buffers");
  auto cs = ca->getAttrOfType<IntegerAttr>("hyb.slot_bytes");
  if (!buffers || buffers.getInt() != 2 || !slot || (!direct && (!cb || !cs))) return false;
  auto W = viewOf(stream.getTarget(), err), Q = viewOf(gemv.getDpsInputs()[0], err),
       C = viewOf(root, err), O = viewOf(output ? output.getTarget() : accumulator, err);
  if (failed(W) || failed(Q) || failed(C) || failed(O)) return false;
  uint64_t wb = uint64_t(wa->getAttrOfType<IntegerAttr>("hyb.offset").getInt()),
           cbeg = uint64_t(ca->getAttrOfType<IntegerAttr>("hyb.offset").getInt()),
           we = wb + 2 * uint64_t(slot.getInt());
  bool outputSafe = O->space == kL2 ? addr(*O) >= uint64_t(cores) * uint64_t(window()) :
      O->space == kL1 && !overlaps(wb, we, addr(*O), addr(*O) + spanBytes(*O));
  return W->space == kL1 && Q->space == kL1 && C->space == kL1 && outputSafe &&
         spanBytes(*W) == uint64_t(W->elements() * W->eb) && slot.getInt() >= W->elements() * W->eb &&
         !overlaps(wb, we, addr(*Q), addr(*Q) + spanBytes(*Q)) &&
         !overlaps(wb, we, cbeg, cbeg + (direct ? spanBytes(*C) : uint64_t(cb.getInt() * cs.getInt())));
}

LogicalResult Codegen::emitFor(scf::ForOp f, std::string &err) {
  auto lb = evalIndex(f.getLowerBound(), err), ub = evalIndex(f.getUpperBound(), err),
       step = evalIndex(f.getStep(), err);
  if (failed(lb) || failed(ub) || failed(step)) return failure();
  loopBounds[f] = {*lb, *step};
  int64_t n = (*ub - *lb + *step - 1) / *step;
  Value iv = f.getInductionVar();
  // Weight streaming: the GDMA of the next tiles runs ahead into a ring of staging slots.
  memref::CopyOp stream;
  uint64_t slotBytes = 0, nslots = 0;
  if (f->hasAttr("hyb.stream"))
    for (Operation &op : f.getBody()->without_terminator())
      if (auto c = dyn_cast<memref::CopyOp>(op)) {
        Value r = c.getSource();
        while (Operation *d = r.getDefiningOp()) {
          if (!isViewOp(d)) break;
          r = d->getOperand(0);
        }
        if (r.getDefiningOp<hyb::WeightOp>() && spaceOf(c.getTarget().getType()) == kL1) {
          env[iv] = *lb;
          FailureOr<View> v = viewOf(c.getSource(), err);
          if (failed(v)) return failure();
          slotBytes = uint64_t(v->elements() * v->eb + 63) / 64 * 64;
          nslots = uint64_t(window()) / std::max<uint64_t>(1, slotBytes);
          if (in.spec->npu.weightStreamMaxSlots)
            nslots = std::min(nslots, uint64_t(in.spec->npu.weightStreamMaxSlots));
          if (nslots >= 1) stream = c;
          break;
        }
      }
  linalg::GenericOp gemv;
  memref::CopyOp output;
  // Two staged weights are needed for L1 ping/pong, plus a spare L2 slot so the next
  // GDMA can run during this batch. With only two, batching can slow the DRAM pipeline.
  bool pipeline = in.weightPrefetch && stream && n > 1 && nslots >= 3 &&
                  weightPipeline(f, stream, gemv, output, err);
  if (!err.empty()) return failure();
  // Keep at least half the L2 ring available for future GDMA where possible. Wait for
  // the whole small batch up front, so its LDMA operations cannot split the core block.
  int64_t batch = pipeline ? std::max<int64_t>(2, int64_t(nslots / 2)) : 1;
  bool crossBatch = pipeline && in.spec->npu.weightPrefetchCrossBatch;
  if (pipeline && in.spec->npu.weightPrefetchBatchTiles)
    batch = std::min<int64_t>(in.spec->npu.weightPrefetchBatchTiles, int64_t(nslots));
  if (crossBatch) batch = std::min(batch, int64_t(nslots) - 1);
  uint64_t win = windowBase(curCore), initialSlot = 0;
  if (stream && in.spec->npu.weightStreamRotateSlots) {
    for (uint64_t candidate = 0; candidate < nslots; ++candidate) {
      uint64_t begin = win + candidate * slotBytes;
      if (llvm::none_of(l2acc, [&](const Acc &a) { return overlaps(a.s, a.e, begin, begin + slotBytes); })) {
        initialSlot = candidate;
        break;
      }
    }
  }
  auto slotAddress = [&](int64_t tile) {
    return win + ((uint64_t(tile) + initialSlot) % nslots) * slotBytes;
  };
  bool earlyRelease = pipeline && in.spec->npu.weightStreamEarlyRelease;
  if (f->hasAttr("hyb.stream")) {
    std::string reason = pipeline ? "enabled" : !in.weightPrefetch ? "disabled" :
        !stream ? "no_staging_slot" : n <= 1 ? "single_iteration" :
        nslots < 3 ? "fewer_than_three_slots" : "unsupported_loop_pattern";
    weightStreams.push_back({int64_t(slotBytes), n, int64_t(nslots), earlyRelease ? 1 : batch,
                             pipeline, earlyRelease || crossBatch, reason, int64_t(initialSlot)});
  }
  if (earlyRelease) {
    // Global dependencies can observe core-block completion, not a local DMA
    // event. Consume every local event before splitting a block. A slot read
    // therefore belongs to the load/previous-tile block, never to the compute
    // of the tile that just arrived. No hazard is deleted speculatively.
    auto stage = [&](int64_t tile) -> LogicalResult {
      env[iv] = *lb + tile * *step;
      auto src = viewOf(stream.getSource(), err);
      if (failed(src)) return failure();
      gdmaLoad(addr(*src), slotAddress(tile), uint32_t(src->elements()*src->eb), 1, 0, 0);
      return success();
    };
    for (int64_t j = 0; j < std::min<int64_t>(n,nslots); ++j)
      if (failed(stage(j))) return failure();
    for (int64_t j = 0; j < n; ++j) {
      env[iv] = *lb + j * *step;
      streamSlot[stream] = slotAddress(j);
      if (j == 0) {
        if (failed(lowerCopy(stream,stream.getSource(),stream.getTarget(),{},err))) return failure();
        flush(); // first weight is now in L1; release its L2 read before MMA
      }
      if (j + int64_t(nslots) < n && failed(stage(j + int64_t(nslots)))) return failure();
      // Only the next staged load must be ready before this compute block.
      // The current tile was consumed in the preceding block.
      std::vector<int> deps;
      env[iv] = *lb + j * *step;
      if (output) {
        auto out = viewOf(output.getTarget(),err);
        if (failed(out)) return failure();
        if (out->space == kL2)
          for (auto [s,e] : viewRanges(*out)) coreDependencies(s,e,true,deps);
      }
      if (j + 1 < n) {
        uint64_t slot = slotAddress(j+1);
        coreDependencies(slot,slot+slotBytes,false,deps);
        env[iv] = *lb + (j+1)* *step;
        auto dst = viewOf(stream.getTarget(),err);
        if (failed(dst)) return failure();
        nextWeight = WeightPrefetch{gemv,addr(*dst),slot,uint32_t(dst->elements()*dst->eb)};
      }
      std::sort(deps.begin(),deps.end());
      deps.erase(std::unique(deps.begin(),deps.end()),deps.end());
      blockAfter(deps);
      env[iv] = *lb + j * *step;
      prefetchedWeightCopy = stream;
      prefetchedWeightNeedsWait = false;
      if (failed(emitBlock(*f.getBody(),err))) return failure();
      if (j + 1 < n) {
        load(10,1);
        emit(rform(OP_WAIT_EVENT,10));
      }
      flush();
    }
    streamSlot.erase(stream); env.erase(iv); loopBounds.erase(f);
    return success();
  }
  int64_t issued = -1;
  for (int64_t i = 0; i < n; i += batch) {
    int64_t end = std::min(n, i + batch);
    if (stream) {
      while (issued + 1 < n && issued + 1 <= i + int64_t(nslots) - 1) {
        ++issued;
        env[iv] = *lb + issued * *step;
        FailureOr<View> v = viewOf(stream.getSource(), err);
        if (failed(v)) return failure();
        gdmaLoad(addr(*v), slotAddress(issued), uint32_t(v->elements() * v->eb), 1, 0, 0);
      }
    }
    if (pipeline) {
      std::vector<int> deps;
      // A carried tile is loaded while this batch's last MMA runs. Its GDMA must
      // already be complete before entering the block, just like all other loads.
      for (int64_t j = i; j < std::min(n, end + int64_t(crossBatch)); ++j) {
        uint64_t slot = slotAddress(j);
        coreDependencies(slot, slot + slotBytes, false, deps);
        if (j < end && output) {
          env[iv] = *lb + j * *step;
          auto o = viewOf(output.getTarget(), err);
          if (failed(o)) return failure();
          if (o->space == kL2)
            for (auto [s, e] : viewRanges(*o)) coreDependencies(s, e, true, deps);
        }
      }
      std::sort(deps.begin(), deps.end());
      deps.erase(std::unique(deps.begin(), deps.end()), deps.end());
      blockAfter(deps);
    }
    for (int64_t j = i; j < end; ++j) {
      if (stream) streamSlot[stream] = slotAddress(j);
      if (pipeline && j + 1 < (crossBatch ? n : end)) {
        env[iv] = *lb + (j + 1) * *step;
        auto dst = viewOf(stream.getTarget(), err);
        if (failed(dst)) return failure();
        nextWeight = WeightPrefetch{gemv, addr(*dst),
            slotAddress(j + 1), uint32_t(dst->elements() * dst->eb)};
      }
      prefetchedWeightCopy = pipeline && (j > i || (crossBatch && i > 0)) ? stream.getOperation() : nullptr;
      prefetchedWeightNeedsWait = j > i;
      env[iv] = *lb + j * *step;
      if (failed(emitBlock(*f.getBody(), err))) return failure();
    }
    if (crossBatch && end < n) {
      // Local events are reset at block entry. Consume the event in its producer
      // block; the next block inherits ready L1 data through its block dependency.
      load(10, 1);
      emit(rform(OP_WAIT_EVENT, 10));
    }
    if (pipeline) flush();  // release the batch's L2 read ranges before ring-slot reuse
  }
  if (stream) streamSlot.erase(stream);
  env.erase(iv);
  loopBounds.erase(f);
  return success();
}

LogicalResult Codegen::emitForall(scf::ForallOp f, std::string &err) {
  int64_t n = 1;
  for (OpFoldResult ub : f.getMixedUpperBound()) {
    auto a = dyn_cast<Attribute>(ub);
    if (!a) {
      err = "scf.forall with a dynamic bound";
      return failure();
    }
    n *= cast<IntegerAttr>(a).getInt();
  }
  if (f.getInductionVars().size() != 1 || n > cores) {
    err = "scf.forall must be 1-D over at most the core count";
    return failure();
  }
  int home = curCore;
  flush();
  std::vector<int> start = last[size_t(home)];
  Value iv = f.getInductionVars()[0];
  for (int64_t c = 0; c < n; ++c) {
    startCore(int(c), start);
    env[iv] = c;
    if (failed(emitBlock(*f.getBody(), err))) return failure();
    endCore();
  }
  env.erase(iv);
  std::vector<int> join;
  for (int64_t c = 0; c < n; ++c)
    for (int id : last[size_t(c)])
      if (std::find(join.begin(), join.end(), id) == join.end()) join.push_back(id);
  curCore = home;
  w.logicalCore = home;
  last[size_t(home)] = join;
  w.frontier = join;
  return success();
}

} // namespace mlir::hybrid::codegen
