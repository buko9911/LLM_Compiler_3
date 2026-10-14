#include "npu/Transforms/Pipeline.h"
#include "npu/Analysis/MemoryPlan.h"
#include "npu/Dialect/NPUDialect.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringMap.h"

using namespace mlir;
namespace npu {
namespace {
// ── FlashAttention state in L2 ───────────────────────────────────────────────
// Bufferization puts every tensor the graph computes in DRAM (memory space 0),
// and tiling stages each operand through an L2 buffer of its own. That keeps
// the on-chip footprint bounded by tile sizes, whatever the tensor sizes are.
// A loop that Stage 1 recognised as FlashAttention (npu.flash_attention) is
// the exception: its carried state (m, l, O) and per-block intermediates are
// sized by the loop body, not by the trip count, so they can stay in L2
// without giving up that bound. Here they otherwise make a full round trip per
// use: L2 -> DRAM by GDMA, DRAM -> L2 by GDMA, and a core block boundary each
// time.
//
// Moving such a buffer to L2 turns its copies with the staging buffers into
// L2 -> L2 copies, which no engine performs; so a buffer moves only if every
// one of those copies can be removed by letting the staging buffer *be* the
// intermediate (copy coalescing).

Value bufferRoot(Value v) {
  while (Operation *def = v.getDefiningOp()) {
    auto view = dyn_cast<ViewLikeOpInterface>(def);
    if (!view) break;
    v = view.getViewSource();
  }
  return v;
}

unsigned spaceOf(Value v) { return cast<BaseMemRefType>(v.getType()).getMemorySpaceAsInt(); }

struct CopyEnds { Value source, target; };
std::optional<CopyEnds> copyEnds(Operation *op) {
  if (auto copy = dyn_cast<memref::CopyOp>(op)) return CopyEnds{copy.getSource(), copy.getTarget()};
  if (auto copy = dyn_cast<linalg::CopyOp>(op); copy && copy.hasPureBufferSemantics())
    return CopyEnds{copy.getInputs()[0], copy.getOutputs()[0]};
  return std::nullopt;
}

struct Access { Operation *op; bool read, write; };

class Residency {
public:
  Residency(func::FuncOp entry, ArrayRef<scf::ForOp> loops)
      : entry(entry), dominance(entry), loops(loops) {}

  // Returns how many DRAM buffers of the marked loops now live in L2.
  unsigned run() {
    // A buffer belongs to a loop if it is allocated in it or touched in it:
    // the carried state is allocated before the loop and copied inside.
    auto inLoop = [&](Operation *op) {
      return llvm::any_of(loops, [&](scf::ForOp loop) { return loop->isAncestor(op); });
    };
    SmallVector<memref::AllocOp> candidates;
    entry.walk([&](memref::AllocOp a) {
      if (spaceOf(a) == 0 && (inLoop(a) || llvm::any_of(a->getUsers(), inLoop)))
        candidates.push_back(a);
    });
    unsigned moved = 0;
    for (auto alloc : candidates)
      if (tryMove(alloc)) ++moved;
    return moved;
  }

private:
  func::FuncOp entry;
  DominanceInfo dominance;
  ArrayRef<scf::ForOp> loops;
  DenseMap<Operation *, uint64_t> order;

  void renumber() {
    order.clear();
    uint64_t next = 0;
    entry.walk<WalkOrder::PreOrder>([&](Operation *op) { order[op] = ++next; });
  }

  // Every operation that reads or writes the bytes of `root`, through any view.
  SmallVector<Access> accesses(Value root) {
    SmallVector<Access> out;
    entry.walk([&](Operation *op) {
      bool read = false, write = false;
      if (auto effects = dyn_cast<MemoryEffectOpInterface>(op)) {
        SmallVector<MemoryEffects::EffectInstance> list;
        effects.getEffects(list);
        for (auto &e : list) {
          Value v = e.getValue();
          if (!v || !isa<BaseMemRefType>(v.getType()) || bufferRoot(v) != root) continue;
          if (isa<MemoryEffects::Read>(e.getEffect())) read = true;
          if (isa<MemoryEffects::Write>(e.getEffect())) write = true;
        }
      } else if (!op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
        for (Value v : op->getOperands())
          if (isa<BaseMemRefType>(v.getType()) && bufferRoot(v) == root) read = write = true;
      }
      if (read || write) out.push_back({op, read, write});
    });
    return out;
  }

  // The buffer is reached only through views, copies and linalg ops, so
  // accesses() sees every use and nothing carries it out of its region.
  static bool plainUses(Value root) {
    SmallVector<Value> work{root};
    while (!work.empty()) {
      Value v = work.pop_back_val();
      for (Operation *user : v.getUsers()) {
        if (auto view = dyn_cast<ViewLikeOpInterface>(user); view && view.getViewSource() == v) {
          for (Value r : user->getResults()) work.push_back(r);
          continue;
        }
        if (!isa<linalg::LinalgOp, memref::CopyOp, memref::DeallocOp>(user)) return false;
      }
    }
    return true;
  }

  static bool isLoop(Operation *op) { return isa<scf::ForOp, scf::ForallOp, scf::ParallelOp>(op); }

  bool tryMove(memref::AllocOp alloc) {
    Value r = alloc.getResult();
    auto type = cast<MemRefType>(r.getType());
    auto l2Type = MemRefType::get(type.getShape(), type.getElementType(), type.getLayout(),
                                  IntegerAttr::get(IntegerType::get(r.getContext(), 64), 1));
    // Only whole-buffer copies: a staging buffer for a slice would need a view
    // in its place, and a function result has to stay where the caller reads it.
    llvm::SetVector<Value> staging;
    DenseSet<Operation *> boundary;
    // The intermediate has to be defined before every staging buffer it takes
    // over. A staging buffer that writes it is often allocated first; a static
    // allocation in the same block can simply move up to it.
    Operation *earliest = nullptr;
    for (Operation *user : r.getUsers()) {
      auto ends = copyEnds(user);
      if (!ends || ends->source == ends->target) return false;
      Value other = ends->source == r ? ends->target : ends->source;
      if (spaceOf(other) == 0) continue;  // a GDMA transfer, still possible from L2
      auto stage = other.getDefiningOp<memref::AllocOp>();
      if (!stage || spaceOf(other) != 1 || other.getType() != l2Type) return false;
      if (!dominance.properlyDominates(alloc.getOperation(), stage.getOperation())) {
        if (stage->getBlock() != alloc->getBlock() || !alloc->getOperands().empty()) return false;
        if (!earliest || stage->isBeforeInBlock(earliest)) earliest = stage;
      }
      if (!plainUses(other)) return false;
      staging.insert(other);
      boundary.insert(user);
    }
    if (staging.empty()) return false;
    renumber();
    // After coalescing, everything a staging buffer did is done to the
    // intermediate, so each staging buffer is checked against the others too.
    DenseMap<Value, SmallVector<Access>> stagingAccesses;
    for (Value s : staging) stagingAccesses[s] = accesses(s);
    SmallVector<Access> own;
    for (const Access &a : accesses(r))
      if (!boundary.contains(a.op)) own.push_back(a);
    for (Value s : staging) {
      SmallVector<Access> others = own;
      for (Value t : staging)
        if (t != s)
          for (const Access &a : stagingAccesses[t])
            if (!boundary.contains(a.op)) others.push_back(a);
      if (!canCoalesce(s, r, stagingAccesses[s], others, boundary)) return false;
    }
    // Commit: the intermediate moves to L2 and absorbs its staging buffers.
    if (earliest) alloc->moveBefore(earliest);
    r.setType(l2Type);
    for (Operation *copy : boundary) copy->erase();
    for (Value s : staging) {
      Operation *stage = s.getDefiningOp();
      for (Operation *user : llvm::make_early_inc_range(s.getUsers()))
        if (isa<memref::DeallocOp>(user)) user->erase();
      s.replaceAllUsesWith(r);
      stage->erase();
    }
    return true;
  }

  // Can staging buffer `s` become intermediate `r`? `mine` are the accesses of
  // s; `others` everything else that will touch r once all coalescing is done.
  bool canCoalesce(Value s, Value r, ArrayRef<Access> mine, ArrayRef<Access> others,
                   const DenseSet<Operation *> &boundary) {
    uint64_t first = UINT64_MAX, last = 0;
    for (const Access &a : mine) { first = std::min(first, order[a.op]); last = std::max(last, order[a.op]); }
    Operation *in = nullptr, *out = nullptr;
    bool writtenInside = false;
    for (const Access &a : mine) {
      if (boundary.contains(a.op)) {
        auto ends = *copyEnds(a.op);
        if (ends.source == r && ends.target == s) {
          if (in || order[a.op] != first) return false;  // the snapshot must come first
          in = a.op;
        } else if (ends.source == s && ends.target == r) {
          if (out || order[a.op] != last) return false;  // the write-back must come last
          out = a.op;
        } else return false;
      } else if (a.write) writtenInside = true;
    }
    // A snapshot that is modified and never written back would, coalesced,
    // modify the intermediate behind its readers' backs.
    if (!out && writtenInside) return false;
    for (const Access &a : others) {
      uint64_t at = order[a.op];
      if (at <= first || at >= last) continue;
      if (a.write) return false;           // would change what s holds
      if (out && a.read) return false;     // would see s half written
    }
    // A staging buffer allocated outside a loop it is used in lives across
    // iterations; any other access of r in that loop is then interleaved with
    // it no matter where it stands in the loop body.
    Operation *stage = s.getDefiningOp();
    for (const Access &a : mine)
      for (Operation *loop = a.op->getParentOp(); loop && loop != entry; loop = loop->getParentOp())
        if (isLoop(loop) && !loop->isAncestor(stage))
          for (const Access &o : others)
            if (loop->isAncestor(o.op)) return false;
    return true;
  }
};

LogicalResult planAddresses(ModuleOp m, const HardwareConfig &hw);
} // namespace

LogicalResult placeGraph(ModuleOp m, const HardwareConfig &hw) {
  auto *context = m.getContext();
  auto attentionLoops = [&] {
    SmallVector<scf::ForOp> loops;
    m.walk([&](scf::ForOp loop) { if (loop->hasAttr("npu.flash_attention")) loops.push_back(loop); });
    return loops;
  };
  const bool resident = !attentionLoops().empty() && !m->hasAttr("npu.dram_intermediates");
  // A FlashAttention loop starts from m, l and O passed in as arguments and
  // would update them in the caller's DRAM buffers. Making those arguments
  // read-only lets bufferization copy them into buffers of their own (one GDMA
  // load each), which can then stay in L2 for the whole loop.
  if (resident)
    for (auto loop : attentionLoops())
      for (Value init : loop.getInitArgs())
        if (auto arg = dyn_cast<BlockArgument>(init))
          if (auto f = dyn_cast<func::FuncOp>(arg.getOwner()->getParentOp()))
            f.setArgAttr(arg.getArgNumber(), "bufferization.writable", BoolAttr::get(context, false));
  bufferization::OneShotBufferizationOptions options;
  options.bufferizeFunctionBoundaries = true;
  options.setFunctionBoundaryTypeConversion(bufferization::LayoutMapOption::IdentityLayoutMap);
  options.defaultMemorySpaceFn = [&](bufferization::TensorLikeType) -> std::optional<Attribute> {
    return IntegerAttr::get(IntegerType::get(context,64),0);
  };
  bufferization::BufferizationState state;
  if (failed(bufferization::runOneShotModuleBufferize(m,options,state)) || failed(cleanIR(m))) return failure();
  for (auto f : m.getOps<func::FuncOp>())
    for (unsigned i = 0; i < f.getNumArguments(); ++i) f.removeArgAttr(i, "bufferization.writable");
  // Stage 3 decides where each buffer lives (memory space), not its address.
  // Addresses are assigned after Stage 4 (npu-assign-addresses), once later
  // transformations such as FlashAttention pipelining have settled which
  // buffers exist and in what order they are used.
  auto placed = [&] {
    m->setAttr("npu.stage", StringAttr::get(context, "placed"));
    return success();
  };
  if (!resident) return placed();
  OwningOpRef<ModuleOp> saved = cast<ModuleOp>(m->clone());
  const auto loops = attentionLoops();
  const unsigned moved = Residency(*m.getOps<func::FuncOp>().begin(), loops).run();
  if (moved) {
    // The loop state is bounded by the loop body, but L2 is shared by every
    // buffer; if it does not fit next to the staging buffers, keep it in DRAM
    // rather than fail the graph. The check is a trial address plan on a copy:
    // the real plan comes later, on the same buffers.
    OwningOpRef<ModuleOp> trial = cast<ModuleOp>(m->clone());
    LogicalResult fits = failure();
    {
      ScopedDiagnosticHandler quiet(context, [](Diagnostic &) { return success(); });
      fits = planAddresses(*trial, hw);
    }
    if (succeeded(fits)) {
      m->setAttr("npu.l2_resident", IntegerAttr::get(IntegerType::get(context, 64), moved));
      return placed();
    }
    m.getBody()->clear();
    m.getBody()->getOperations().splice(m.getBody()->end(), saved->getBody()->getOperations());
    m->setAttrs((*saved)->getAttrDictionary());
    m->setAttr("npu.l2_resident_dropped", IntegerAttr::get(IntegerType::get(context, 64), moved));
  }
  return placed();
}

LogicalResult assignAddresses(ModuleOp m, const HardwareConfig &hw) {
  // Hand-written placed IR (tests that pin a hazard to chosen addresses) may
  // arrive with every buffer addressed already; such a module keeps them.
  auto entry = *m.getOps<func::FuncOp>().begin();
  bool addressed = llvm::all_of(entry.getArguments(), [&](BlockArgument arg) {
    return entry.getArgAttr(arg.getArgNumber(), "npu.address") != nullptr;
  });
  m.walk([&](memref::AllocOp alloc) { addressed &= alloc->hasAttr("npu.address"); });
  if (!addressed && failed(planAddresses(m, hw))) return failure();
  m->setAttr("npu.stage", StringAttr::get(m.getContext(), "scheduled"));
  return success();
}

namespace {
LogicalResult planAddresses(ModuleOp m, const HardwareConfig &hw) {
  OpBuilder b(m.getContext());
  SmallVector<Value> allocations;
  std::vector<BufferRequirement> requirements;
  auto entry = *m.getOps<func::FuncOp>().begin();
  for (auto arg : entry.getArguments()) allocations.push_back(arg);
  m.walk([&](memref::AllocOp alloc) { allocations.push_back(alloc); });
  // One position per statement of the entry block. An operation nested in a loop
  // takes the loop's position, so a buffer written on one iteration and read on
  // the next still covers every point between.
  Block &body = entry.front();
  DenseMap<Operation *, uint64_t> position;
  uint64_t last = 0;
  for (Operation &op : body) position[&op] = ++last;
  ++last;
  // A buffer's bytes live as long as any SSA value that can name them.
  // BufferViewFlowAnalysis gives those values (subviews, reshapes, casts, loop
  // iter_args and results, ...), and mlir::Liveness gives, per value, the last
  // statement of the entry block that still needs it; a use inside a loop or
  // any other nested region counts as a use by the statement containing it.
  // A value returned from the function is used by func.return, the last
  // statement, so it stays live to the end.
  Liveness liveness(entry);
  const LivenessBlockInfo *info = liveness.getLiveness(&body);
  BufferViewFlowAnalysis aliases(entry);
  auto statement = [&](Operation *op) { return body.findAncestorOpInBlock(*op); };
  auto lifetime = [&](Value value) -> std::pair<uint64_t, uint64_t> {
    if (isa<BlockArgument>(value)) return {0, last};
    uint64_t begin = position.lookup(statement(value.getDefiningOp())), end = begin;
    for (Value alias : aliases.resolve(value)) {
      if (alias.getParentBlock() == &body) {
        Operation *endOp = info->getEndOperation(alias, info->getStartOperation(alias));
        end = std::max(end, position.lookup(endOp) + 1);
      } else if (Operation *owner = statement(alias.getParentBlock()->getParentOp())) {
        // A view made inside a nested region cannot leave it except through a
        // region result, which is an alias of its own in the entry block.
        end = std::max(end, position.lookup(owner) + 1);
      }
    }
    return {begin, std::max(end, begin + 1)};
  };
  for (auto value : allocations) {
    auto t = dyn_cast<MemRefType>(value.getType());
    unsigned width = t && t.getElementType().isF16() ? 2 : t && t.getElementType().isF32() ? 4 : 0;
    if (!t || !t.hasStaticShape() || !width || !t.getLayout().isIdentity())
      return m.emitError("placement requires static contiguous FP16/FP32 allocations");
    uint64_t bytes = uint64_t(t.getNumElements())*width;
    BufferRequirement r; r.name = "buffer" + std::to_string(requirements.size());
    r.bytes = bytes; r.space = MemorySpace(t.getMemorySpaceAsInt());
    std::tie(r.begin,r.end) = lifetime(value);
    requirements.push_back(r);
  }
  // Double buffering of L2 panels. Lifetimes above follow the written order, so
  // the next panel's buffer starts after the previous panel's buffer has died
  // and first-fit hands it the same address. The GDMA filling the next panel
  // then has to wait until every core block finished reading the previous one
  // (Stage 6 orders writes after reads of the same bytes), and transfer and
  // compute run one after the other.
  //
  // A buffer that a GDMA fills from DRAM or drains to DRAM is a panel buffer.
  // It is made live from where the previous panel buffer of the same type
  // began, so the two coexist and get different addresses; the command
  // processor can then run the next transfer while the current panel is in use.
  // Only the nearest earlier panel counts, so one role holds at most two
  // buffers. If L2 cannot hold that, placement falls back to plain reuse.
  auto meetsDram = [&](Value value) {
    auto space = [](Value v) { return cast<BaseMemRefType>(v.getType()).getMemorySpaceAsInt(); };
    for (Value alias : aliases.resolve(value))
      for (Operation *user : alias.getUsers()) {
        Value source, target;
        if (auto copy = dyn_cast<memref::CopyOp>(user)) { source = copy.getSource(); target = copy.getTarget(); }
        else if (auto copy = dyn_cast<mlir::npu::GlobalCopyOp>(user)) { source = copy.getSource(); target = copy.getTarget(); }
        else if (auto copy = dyn_cast<linalg::CopyOp>(user);
                 copy && copy.hasPureBufferSemantics()) { source = copy.getInputs()[0]; target = copy.getOutputs()[0]; }
        else continue;
        if (space(source) == 0 || space(target) == 0) return true;
      }
    return false;
  };
  std::vector<size_t> panels;
  for (size_t i = 0; i < allocations.size(); ++i)
    if (!isa<BlockArgument>(allocations[i]) && requirements[i].space == MemorySpace::L2 &&
        meetsDram(allocations[i]))
      panels.push_back(i);
  const std::vector<BufferRequirement> written = requirements;
  std::vector<bool> doubled(requirements.size(), false);
  for (size_t x : panels) {
    std::optional<size_t> previous;
    for (size_t p : panels)
      if (p != x && allocations[p].getType() == allocations[x].getType() &&
          written[p].end <= written[x].begin &&
          (!previous || written[p].end > written[*previous].end))
        previous = p;
    if (previous && written[*previous].begin < requirements[x].begin) {
      requirements[x].begin = written[*previous].begin;
      doubled[x] = true;
    }
  }
  // A GDMA command reaches between DRAM and L2 only, so a transfer that starts
  // and ends in DRAM needs somewhere in L2 to pass through. Reserving it here is
  // what lets the target stage split such a transfer without an allocator.
  BufferRequirement staging;
  staging.name = "staging"; staging.space = MemorySpace::L2;
  staging.bytes = kDRAMStagingBytes; staging.begin = 0; staging.end = std::max<uint64_t>(1,last);
  requirements.push_back(staging);
  // --readback: an L2 window for the outputs, live only after the last
  // statement, so it can share addresses with everything that died before.
  if (m->hasAttr("npu.readback")) {
    uint64_t bytes = 0;
    entry.walk([&](func::ReturnOp ret) {
      for (auto v : ret.getOperands())
        if (auto t = dyn_cast<MemRefType>(v.getType()))
          bytes += uint64_t(t.getNumElements()) * (t.getElementType().isF32() ? 4 : 2);
    });
    if (bytes) {
      BufferRequirement readback;
      readback.name = "readback"; readback.space = MemorySpace::L2;
      readback.bytes = bytes; readback.begin = last; readback.end = last + 1;
      requirements.push_back(readback);
    }
  }
  const MemoryLimits limits{hw.l1Bytes,hw.l2Bytes,hw.dramBytes,1,hw.dramBytesPerCycle,0};
  auto placed = planMemory(requirements,limits);
  // Where L2 runs out, give up double buffering only around that point: the
  // buffers whose lifetime was moved earlier across the statement at which a
  // buffer no longer fits go back to their written lifetime, and placement is
  // tried again. Buffers elsewhere keep their second address.
  int64_t dropped = 0;
  while (!placed && llvm::is_contained(doubled, true)) {
    const std::string marker = ": buffer";
    auto at = placed.error.rfind(marker);
    std::optional<uint64_t> when;
    if (at != std::string::npos) {
      size_t index = 0, digit = at + marker.size();
      while (digit < placed.error.size() && isdigit(static_cast<unsigned char>(placed.error[digit])))
        index = index * 10 + size_t(placed.error[digit++] - '0');
      if (digit > at + marker.size() && index < requirements.size()) when = requirements[index].begin;
    }
    bool released = false;
    for (size_t i = 0; i < doubled.size(); ++i) {
      bool across = when && requirements[i].begin <= *when && *when < written[i].begin;
      if (doubled[i] && (!when || across)) {
        requirements[i] = written[i]; doubled[i] = false; ++dropped; released = true;
      }
    }
    if (!released)  // the shortage is not near a moved lifetime: undo them all
      for (size_t i = 0; i < doubled.size(); ++i)
        if (doubled[i]) { requirements[i] = written[i]; doubled[i] = false; ++dropped; }
    placed = planMemory(requirements,limits);
  }
  if (dropped) m->setAttr("npu.double_buffer_dropped", b.getI64IntegerAttr(dropped));
  if (!placed) return m.emitError(placed.error);
  // planMemory sorts its input by lifetime start, so results are not in call
  // order. Match on the unique name the analysis already validates.
  llvm::StringMap<uint64_t> addressOf;
  for (const auto &p : placed.value.buffers) {
    if (p.action != BufferPlacement::Action::Resident)
      return m.emitError("placement cannot yet realize spill or rematerialization: " + p.requirement.name);
    addressOf[p.requirement.name] = p.address;
  }
  for (size_t i = 0; i < allocations.size(); ++i) {
    auto found = addressOf.find(requirements[i].name);
    if (found == addressOf.end()) return m.emitError("placement lost a buffer requirement");
    auto address = b.getI64IntegerAttr(found->second);
    if (auto arg = dyn_cast<BlockArgument>(allocations[i])) {
      auto f = cast<func::FuncOp>(arg.getOwner()->getParentOp());
      f.setArgAttr(arg.getArgNumber(),"npu.address",address);
    } else {
      // [first statement, one past the last statement] that needs the buffer,
      // as computed above; recorded so 03-placed.mlir shows why two buffers
      // may share an address.
      auto *alloc = allocations[i].getDefiningOp();
      alloc->setAttr("npu.address",address);
      alloc->setAttr("npu.live",b.getDenseI64ArrayAttr({int64_t(requirements[i].begin),
                                                          int64_t(requirements[i].end)}));
      // Live from the previous panel's start rather than its own statement.
      if (doubled[i]) alloc->setAttr("npu.double_buffer",b.getUnitAttr());
    }
  }
  m->setAttr("npu.l1_capacity",b.getI64IntegerAttr(hw.l1Bytes));
  m->setAttr("npu.double_buffered",b.getI64IntegerAttr(llvm::count(doubled, true)));
  if (auto found = addressOf.find("readback"); found != addressOf.end())
    m->setAttr("npu.readback_l2",b.getI64IntegerAttr(found->second));
  m->setAttr("npu.l2_staging",b.getI64IntegerAttr(addressOf["staging"]));
  m->setAttr("npu.l2_staging_bytes",b.getI64IntegerAttr(kDRAMStagingBytes));
  m->setAttr("npu.l1_bytes",b.getI64IntegerAttr(std::max<uint64_t>(64,placed.value.l1HighWater[0])));
  m->setAttr("npu.l2_bytes",b.getI64IntegerAttr(std::max<uint64_t>(64,placed.value.l2HighWater)));
  m->setAttr("npu.dram_bytes",b.getI64IntegerAttr(std::max<uint64_t>(64,placed.value.dramHighWater)));
  return success();
}
} // namespace
} // namespace npu
