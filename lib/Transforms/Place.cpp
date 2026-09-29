#include "npu/Transforms/Pipeline.h"
#include "npu/Analysis/MemoryPlan.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"

using namespace mlir;
namespace npu {
LogicalResult placeGraph(ModuleOp m, const HardwareConfig &hw) {
  bufferization::OneShotBufferizationOptions options;
  options.bufferizeFunctionBoundaries = true;
  options.setFunctionBoundaryTypeConversion(bufferization::LayoutMapOption::IdentityLayoutMap);
  options.defaultMemorySpaceFn = [&](bufferization::TensorLikeType) -> std::optional<Attribute> {
    return IntegerAttr::get(IntegerType::get(m.getContext(),64),0);
  };
  bufferization::BufferizationState state;
  if (failed(bufferization::runOneShotModuleBufferize(m,options,state)) || failed(cleanIR(m))) return failure();
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
  auto placed = planMemory(requirements,{hw.l1Bytes,hw.l2Bytes,hw.dramBytes,1,hw.dramBytesPerCycle,0});
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
    }
  }
  m->setAttr("npu.l1_capacity",b.getI64IntegerAttr(hw.l1Bytes));
  if (auto found = addressOf.find("readback"); found != addressOf.end())
    m->setAttr("npu.readback_l2",b.getI64IntegerAttr(found->second));
  m->setAttr("npu.l2_staging",b.getI64IntegerAttr(addressOf["staging"]));
  m->setAttr("npu.l2_staging_bytes",b.getI64IntegerAttr(kDRAMStagingBytes));
  m->setAttr("npu.l1_bytes",b.getI64IntegerAttr(std::max<uint64_t>(64,placed.value.l1HighWater[0])));
  m->setAttr("npu.l2_bytes",b.getI64IntegerAttr(std::max<uint64_t>(64,placed.value.l2HighWater)));
  m->setAttr("npu.dram_bytes",b.getI64IntegerAttr(std::max<uint64_t>(64,placed.value.dramHighWater)));
  m->setAttr("npu.stage",b.getStringAttr("placed"));
  return success();
}
} // namespace npu
