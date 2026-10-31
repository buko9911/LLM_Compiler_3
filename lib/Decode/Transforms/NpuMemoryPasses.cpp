//===- NpuMemoryPasses.cpp - hyb-npu-place, hyb-npu-promote, hyb-npu-memplan -===//
//
// Memory hierarchy of an NPU phase, made explicit on memrefs after bufferization:
//   memory space 0  LP6      phase arguments, weights (hyb.weight)
//   memory space 1  L1       private to a core (4 MiB)
//   memory space 2  L2       shared by the cores (8 MiB)
// Compute ops read and write L1 only; memref.copy between spaces is the DMA the backend emits.
//===----------------------------------------------------------------------===//

#include "npu/Decode/Transforms/PassSupport.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SetVector.h"
#include <algorithm>

namespace mlir::hybrid {
#define GEN_PASS_DEF_HYBNPUPLACE
#define GEN_PASS_DEF_HYBNPUPROMOTE
#define GEN_PASS_DEF_HYBNPUMEMPLAN
#include "npu/Decode/Transforms/Passes.h.inc"
} // namespace mlir::hybrid

using namespace mlir;
using namespace mlir::hybrid;

namespace {

constexpr int64_t kLP6 = 0, kL1 = 1, kL2 = 2;

bool isCompute(Operation *op) {
  return isa<linalg::LinalgOp, hyb::QuantOp, hyb::DequantOp>(op) && !isa<linalg::CopyOp>(op);
}

int64_t spaceOf(Type t) {
  auto m = dyn_cast<MemRefType>(t);
  if (!m || !m.getMemorySpace()) return kLP6;
  if (auto i = dyn_cast<IntegerAttr>(m.getMemorySpace())) return i.getInt();
  return kLP6;
}

MemRefType withSpace(MemRefType t, int64_t space) {
  Attribute ms = space == kLP6 ? Attribute() : IntegerAttr::get(IntegerType::get(t.getContext(), 64), space);
  return MemRefType::get(t.getShape(), t.getElementType(), t.getLayout(), ms);
}

/// Changes the memory space of `v` and of every view derived from it.
void retype(Value v, int64_t space) {
  v.setType(withSpace(cast<MemRefType>(v.getType()), space));
  for (Operation *u : v.getUsers())
    if (isViewOp(u) && u->getOperand(0) == v) retype(u->getResult(0), space);
}

/// Non-view users of `v` and its views.
void forEachUser(Value v, llvm::function_ref<void(Operation *)> fn) {
  for (Operation *u : v.getUsers()) {
    if (isViewOp(u) && u->getOperand(0) == v) forEachUser(u->getResult(0), fn);
    else fn(u);
  }
}

int64_t forallCores(scf::ForallOp f) {
  if (auto c = f->getAttrOfType<IntegerAttr>("hyb.cores")) return c.getInt();
  int64_t n = 1;
  for (OpFoldResult ub : f.getMixedUpperBound())
    if (auto a = dyn_cast<Attribute>(ub)) n *= cast<IntegerAttr>(a).getInt();
  return n;
}

//===----------------------------------------------------------------------===//
// hyb-npu-place
//===----------------------------------------------------------------------===//
struct HybNpuPlacePass : mlir::hybrid::impl::HybNpuPlaceBase<HybNpuPlacePass> {
  void runOnOperation() override {
    for (func::FuncOp fn : getNpuPhases(getOperation())) {
      SmallVector<memref::AllocOp> allocs;
      fn.walk([&](memref::AllocOp a) { allocs.push_back(a); });
      for (memref::AllocOp a : allocs) {
        if (spaceOf(a.getType()) != kLP6) continue;
        bool inForall = a->getParentOfType<scf::ForallOp>() != nullptr;
        bool multi = false;
        forEachUser(a.getResult(), [&](Operation *u) {
          if (auto f = u->getParentOfType<scf::ForallOp>(); f && forallCores(f) > 1) multi = true;
        });
        retype(a.getResult(), inForall || !multi ? kL1 : kL2);
      }
    }
  }
};

//===----------------------------------------------------------------------===//
// hyb-npu-promote
//===----------------------------------------------------------------------===//
bool readsOperand(Operation *op, OpOperand &o) {
  auto dps = dyn_cast<DestinationStyleOpInterface>(op);
  if (!dps || !dps.isDpsInit(&o)) return true;
  if (auto g = dyn_cast<linalg::GenericOp>(op)) return g.payloadUsesValueFromOperand(&o);
  if (isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op)) return true;   // C += A B
  return false;                                                         // fill, transpose, quant, dequant
}

bool writesOperand(Operation *op, OpOperand &o) {
  auto dps = dyn_cast<DestinationStyleOpInterface>(op);
  return dps && dps.isDpsInit(&o);
}

struct Promoter {
  struct Entry {
    Value outer, root, l1;
    bool dirty = false;
  };
  using Cache = SmallVector<Entry>;

  Value allocL1(OpBuilder &b, Location loc, Value v) {
    auto t = cast<MemRefType>(v.getType());
    auto l1 = MemRefType::get(t.getShape(), t.getElementType(), MemRefLayoutAttrInterface(),
                              IntegerAttr::get(IntegerType::get(t.getContext(), 64), kL1));
    auto a = memref::AllocOp::create(b, loc, l1);
    a->setAttr("hyb.promoted", UnitAttr::get(t.getContext()));
    return a;
  }

  void copyOut(OpBuilder &b, Location loc, Entry &e) {
    memref::CopyOp::create(b, loc, e.l1, e.outer);
    e.dirty = false;
  }

  /// Writes back dirty entries of `roots` (all when empty) before `at`; drops entries whose root
  /// is in `invalidate`.
  void flush(Cache &cache, Operation *at, const llvm::SetVector<Value> *roots,
             const llvm::SetVector<Value> *invalidate) {
    OpBuilder b(at);
    for (Entry &e : cache)
      if (e.dirty && (!roots || roots->contains(e.root))) copyOut(b, at->getLoc(), e);
    if (invalidate)
      llvm::erase_if(cache, [&](const Entry &e) { return invalidate->contains(e.root); });
  }

  /// Roots of non-L1 memrefs read / written by the ops nested in `op` (or by `op` itself).
  void accessedRoots(Operation *op, llvm::SetVector<Value> &read, llvm::SetVector<Value> &written) {
    op->walk([&](Operation *inner) {
      for (OpOperand &o : inner->getOpOperands()) {
        if (!isa<MemRefType>(o.get().getType()) || isViewOp(inner)) continue;
        Value r = viewRoot(o.get());
        if (spaceOf(r.getType()) == kL1) continue;
        if (Operation *d = r.getDefiningOp(); d && op->isProperAncestor(d)) continue;
        bool w = false, rd = true;
        if (auto c = dyn_cast<memref::CopyOp>(inner)) {
          w = o.getOperandNumber() == 1;
          rd = !w;
        } else if (isCompute(inner)) {
          w = writesOperand(inner, o);
          rd = readsOperand(inner, o);
        } else {
          w = true;
        }
        if (rd) read.insert(r);
        if (w) written.insert(r);
      }
    });
  }

  void promote(Operation *op, Cache &cache) {
    OpBuilder b(op);
    for (OpOperand &o : op->getOpOperands()) {
      Value v = o.get();
      if (!isa<MemRefType>(v.getType())) continue;
      Value root = viewRoot(v);
      if (spaceOf(root.getType()) == kL1) continue;
      bool rd = readsOperand(op, o), wr = writesOperand(op, o);
      int idx = -1;
      for (auto [i, e] : llvm::enumerate(cache))
        if (e.outer == v) idx = int(i);
      if (idx < 0) {
        for (Entry &e : cache)
          if (e.root == root && e.dirty) copyOut(b, op->getLoc(), e);
        Value l1 = allocL1(b, op->getLoc(), v);
        if (rd) memref::CopyOp::create(b, op->getLoc(), v, l1);
        cache.push_back({v, root, l1, false});
        idx = int(cache.size()) - 1;
      }
      if (wr) {
        Value keep = cache[idx].l1;
        for (Entry &e : cache)
          if (e.l1 != keep && e.root == root && e.dirty) copyOut(b, op->getLoc(), e);
        llvm::erase_if(cache, [&](const Entry &e) { return e.l1 != keep && e.root == root; });
        for (Entry &e : cache)
          if (e.l1 == keep) e.dirty = true;
        o.set(keep);
      } else {
        o.set(cache[idx].l1);
      }
    }
  }

  /// memref.copy and other memref users that stay as they are: make the outer buffers current
  /// first; a copy whose source is cached in L1 reads the L1 copy.
  void other(Operation *op, Cache &cache) {
    llvm::SetVector<Value> read, written;
    accessedRoots(op, read, written);
    llvm::SetVector<Value> all = read;
    all.insert(written.begin(), written.end());
    flush(cache, op, &all, nullptr);
    if (auto c = dyn_cast<memref::CopyOp>(op)) {
      if (c.getSource() == c.getTarget()) {       // left by bufferization of an in-place tile
        op->erase();
        return;
      }
      for (Entry &e : cache)
        if (e.outer == c.getSource()) {
          c.getSourceMutable().assign(e.l1);
          break;
        }
      // Shared results of a distributed producer can feed a strided assembly
      // (e.g. tensor.insert_slice after bufferization). There is no L2 -> L2
      // copy instruction. Express its actual two DMA transfers through a
      // private L1 buffer before memory planning, retaining both logical copy
      // layouts. This is not hidden backend scratch or a new DMA resource.
      if (spaceOf(c.getSource().getType()) == kL2 && spaceOf(c.getTarget().getType()) == kL2 &&
          cast<MemRefType>(c.getSource().getType()).hasStaticShape()) {
        OpBuilder b(c);
        Value staging = allocL1(b, c.getLoc(), c.getSource());
        auto read = memref::CopyOp::create(b, c.getLoc(), c.getSource(), staging);
        auto marker = b.getUnitAttr();
        staging.getDefiningOp()->setAttr("hyb.shared_copy_staging", marker);
        read->setAttr("hyb.shared_copy_staging", marker);
        c->setAttr("hyb.shared_copy_staging", marker);
        c.getSourceMutable().assign(staging);
      }
    }
    llvm::erase_if(cache, [&](const Entry &e) { return written.contains(e.root); });
  }

  /// Read-only operands of compute ops in `loop` that are defined outside it: promoted once in
  /// the parent region.
  void hoist(scf::ForOp loop, Cache &cache) {
    llvm::SetVector<Value> read, written;
    accessedRoots(loop, read, written);
    SmallVector<std::pair<Operation *, unsigned>> uses;
    loop.walk([&](Operation *inner) {
      if (!isCompute(inner)) return;
      for (OpOperand &o : inner->getOpOperands()) {
        Value v = o.get();
        if (!isa<MemRefType>(v.getType())) continue;
        if (Operation *d = v.getDefiningOp(); d && loop->isProperAncestor(d)) continue;
        if (auto ba = dyn_cast<BlockArgument>(v); ba && loop->isAncestor(ba.getOwner()->getParentOp())) continue;
        Value root = viewRoot(v);
        if (spaceOf(root.getType()) == kL1 || written.contains(root) || writesOperand(inner, o)) continue;
        uses.push_back({inner, o.getOperandNumber()});
      }
    });
    OpBuilder b(loop);
    for (auto [inner, n] : uses) {
      Value v = inner->getOperand(n);
      int idx = -1;
      for (auto [i, e] : llvm::enumerate(cache))
        if (e.outer == v) idx = int(i);
      if (idx < 0) {
        Value root = viewRoot(v);
        for (Entry &e : cache)
          if (e.root == root && e.dirty) copyOut(b, loop.getLoc(), e);
        Value l1 = allocL1(b, loop.getLoc(), v);
        memref::CopyOp::create(b, loop.getLoc(), v, l1);
        cache.push_back({v, root, l1, false});
        idx = int(cache.size()) - 1;
      }
      inner->setOperand(n, cache[idx].l1);
    }
  }

  void processBlock(Block &blk) {
    Cache cache;
    SmallVector<Operation *> ops;
    for (Operation &op : blk.without_terminator()) ops.push_back(&op);
    for (Operation *op : ops) {
      if (isViewOp(op) || isa<memref::AllocOp, memref::DeallocOp, hyb::WeightOp>(op) ||
          (op->getNumRegions() == 0 &&
           llvm::none_of(op->getOperandTypes(), [](Type t) { return isa<MemRefType>(t); })))
        continue;
      if (isCompute(op)) {
        promote(op, cache);
        continue;
      }
      if (isa<scf::ForOp, scf::ForallOp>(op)) {
        llvm::SetVector<Value> read, written;
        accessedRoots(op, read, written);
        llvm::SetVector<Value> all = read;
        all.insert(written.begin(), written.end());
        flush(cache, op, &all, &written);
        if (auto f = dyn_cast<scf::ForOp>(op)) hoist(f, cache);
        for (Region &r : op->getRegions())
          for (Block &b : r) processBlock(b);
        continue;
      }
      other(op, cache);
    }
    flush(cache, blk.getTerminator(), nullptr, nullptr);
  }
};

struct HybNpuPromotePass : mlir::hybrid::impl::HybNpuPromoteBase<HybNpuPromotePass> {
  void runOnOperation() override {
    for (func::FuncOp fn : getNpuPhases(getOperation())) {
      Promoter p;
      p.processBlock(fn.getBody().front());
    }
  }
};

//===----------------------------------------------------------------------===//
// hyb-npu-memplan
//===----------------------------------------------------------------------===//
struct HybNpuMemPlanPass : mlir::hybrid::impl::HybNpuMemPlanBase<HybNpuMemPlanPass> {
  using Base::Base;
  void runOnOperation() override {
    auto specOr = loadTargetSpec(getOperation(), spec);
    if (failed(specOr)) return signalPassFailure();
    bool ok = true;
    for (func::FuncOp fn : getNpuPhases(getOperation()))
      if (failed(plan(fn, *specOr))) ok = false;
    if (!ok) signalPassFailure();
  }

  struct Buf {
    memref::AllocOp alloc;
    int64_t space, bytes, slots, first, last, offset = -1;
  };

  LogicalResult plan(func::FuncOp fn, const TargetSpec &t) {
    DenseMap<Operation *, int64_t> pos, end;
    int64_t n = 0;
    fn.walk<WalkOrder::PreOrder>([&](Operation *op) { pos[op] = n++; });
    fn.walk<WalkOrder::PostOrder>([&](Operation *op) {
      int64_t e = pos[op];
      for (Region &r : op->getRegions())
        for (Block &b : r)
          for (Operation &c : b) e = std::max(e, end[&c]);
      end[op] = e;
    });
    // hyb-npu-epilogue-pipeline: the backend runs a marked dequant inside the GEMV loop with the
    // same id, so the dequant's buffers are live from the start of that loop.
    DenseMap<Attribute, Operation *> pipelineLoops;
    fn.walk([&](scf::ForOp f) {
      if (Attribute id = f->getAttr("hyb.epilogue_pipeline")) pipelineLoops[id] = f;
    });
    std::vector<Buf> bufs;
    fn.walk([&](memref::AllocOp a) {
      int64_t space = spaceOf(a.getType());
      if (space != kL1 && space != kL2) return;
      auto mt = a.getType();
      int64_t eb = std::max<int64_t>(1, mt.getElementTypeBitWidth() / 8);
      int64_t bytes = (mt.getNumElements() * eb + 63) / 64 * 64;
      int64_t slots = 1;
      if (auto f = a->getParentOfType<scf::ForOp>(); f && fn->isAncestor(f))
        if (auto tc = constantTripCount(f); tc && *tc > 1) slots = 2;
      // Live from the first to the last use (an alloc holds no data before its first use).
      Buf b{a, space, bytes, slots, INT64_MAX, -1};
      forEachUser(a.getResult(), [&](Operation *u) {
        // A use inside a loop that does not contain the buffer keeps it live for the whole loop.
        Operation *anchor = u;
        for (Operation *p = u->getParentOp(); p && p != fn; p = p->getParentOp())
          if (isa<scf::ForOp, scf::ForallOp>(p) && !p->isAncestor(a)) anchor = p;
        b.first = std::min(b.first, pos[anchor]);
        b.last = std::max(b.last, end[anchor]);
        if (Attribute id = u->getAttr("hyb.epilogue_pipeline"); id && pipelineLoops.count(id))
          b.first = std::min(b.first, pos[pipelineLoops[id]]);
        // An epilogue consumer also runs tile by tile inside that loop.
        if (Attribute id = u->getAttr("hyb.epilogue_consumer"); id && pipelineLoops.count(id))
          b.first = std::min(b.first, pos[pipelineLoops[id]]);
      });
      if (b.last < 0) {
        b.first = b.last = pos[a];
      }
      bufs.push_back(b);
    });
    int64_t l1Base = 4096 + t.npu.l1ReservedBytes, l1Limit = t.npu.l1Bytes;
    int64_t l2Base = t.npu.cores * t.npu.l2StagingPerCore, l2Limit = t.npu.l2Bytes;
    int64_t peak[3] = {0, 0, 0};
    std::vector<size_t> order(bufs.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      return bufs[a].first != bufs[b].first ? bufs[a].first < bufs[b].first : bufs[a].bytes > bufs[b].bytes;
    });
    std::vector<size_t> placed;
    for (size_t i : order) {
      Buf &b = bufs[i];
      int64_t base = b.space == kL1 ? l1Base : l2Base, limit = b.space == kL1 ? l1Limit : l2Limit;
      int64_t size = b.bytes * b.slots;
      std::vector<std::pair<int64_t, int64_t>> busy;
      for (size_t j : placed) {
        const Buf &o = bufs[j];
        if (o.space == b.space && !(o.last < b.first || o.first > b.last))
          busy.push_back({o.offset, o.offset + o.bytes * o.slots});
      }
      std::sort(busy.begin(), busy.end());
      int64_t at = base;
      for (auto [s, e] : busy) {
        if (at + size <= s) break;
        at = std::max(at, e);
      }
      if (at + size > limit)
        return fn.emitError() << (b.space == kL1 ? "L1" : "L2") << " overflow: needs " << at + size << " bytes, limit "
                              << limit;
      b.offset = at;
      peak[b.space] = std::max(peak[b.space], at + size);
      placed.push_back(i);
      Builder bld(fn.getContext());
      b.alloc->setAttr("hyb.offset", bld.getI64IntegerAttr(at));
      if (b.slots > 1) {
        b.alloc->setAttr("hyb.buffers", bld.getI64IntegerAttr(b.slots));
        b.alloc->setAttr("hyb.slot_bytes", bld.getI64IntegerAttr(b.bytes));
      }
    }
    Builder bld(fn.getContext());
    int64_t staging = t.npu.l2StagingPerCore;
    if (staging == 0) {
      // The ordinary L2 buffers were planned from zero. Translate them as a group;
      // their proven lifetimes, relative addresses, and reuse remain unchanged.
      staging = (l2Limit - peak[kL2]) / t.npu.cores / 64 * 64;
      if (staging < 64) return fn.emitError("L2 has no room left for DMA staging after live buffers");
      l2Base = staging * t.npu.cores;
      for (Buf &b : bufs)
        if (b.space == kL2) {
          b.offset += l2Base;
          b.alloc->setAttr("hyb.offset", bld.getI64IntegerAttr(b.offset));
        }
      if (peak[kL2]) peak[kL2] += l2Base;
    }
    fn->setAttr("hyb.l2_staging_per_core", bld.getI64IntegerAttr(staging));
    fn->setAttr("hyb.l1_peak", bld.getI64IntegerAttr(peak[kL1]));
    fn->setAttr("hyb.l2_peak", bld.getI64IntegerAttr(peak[kL2]));
    return success();
  }
};

} // namespace
