//===----------------------------------------------------------------------===//
// Hybrid.cpp — NPU + PIM 분할과 페이즈 나누기
//
// W8A8 합법화가 끝난 그래프에서 가중치 GEMV 는 전부 INT8 x INT8 -> INT32
// 행렬곱이다. 같은 양자화 활성값을 읽는 GEMV 들이 한 묶음이다(q/k/v,
// gate/up, o, down, lm_head). 묶음 안에서는 활성값을 한 번만 PIM 뱅크에
// 뿌리거나(PIM) 가중치를 한 번의 스트리밍으로 읽으므로(NPU), 결정 단위가
// 묶음이다.
//
// partitionGraph 는 묶음마다 비용 모델로 NPU 와 PIM 중 하나를 고른다.
// outlinePhases 는 PIM 묶음을 경계로 그래프를 NPU 페이즈들로 자른다. op 은
// 자기 입력이 준비되는 가장 이른 NPU 페이즈에 들어간다. 페이즈를 건너가는
// 값은 모든 페이즈가 같은 DRAM 주소로 부르는 "전역"이 된다.
//===----------------------------------------------------------------------===//
#include "npu/Transforms/Hybrid.h"
#include "GraphInternal.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include <algorithm>
#include <limits>

using namespace mlir;
namespace npu {

Result<HybridCostModel> HybridCostModel::load(const std::string &path) {
  using R = Result<HybridCostModel>;
  HybridCostModel m;
  if (path.empty()) return R::success(m);
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer) return R::failure("cannot read " + path);
  auto parsed = llvm::json::parse((*buffer)->getBuffer());
  if (!parsed) return R::failure(path + ": " + llvm::toString(parsed.takeError()));
  auto *root = parsed->getAsObject();
  if (!root) return R::failure(path + ": not a JSON object");
  auto number = [](llvm::json::Object *o, llvm::StringRef key, auto &field) {
    if (!o) return;
    if (auto v = o->getNumber(key)) field = static_cast<std::remove_reference_t<decltype(field)>>(*v);
  };
  auto *npu = root->getObject("npu"), *pim = root->getObject("pim"), *mapping = root->getObject("mapping");
  number(npu, "clock_mhz", m.npuClockMHz);
  number(npu, "lp6_read_gbps", m.npuReadGBps);
  number(npu, "gemv_fixed_us", m.npuGemvFixedUs);
  number(npu, "cores", m.npuCores);
  number(pim, "clock_mhz", m.pimClockMHz);
  number(pim, "efficiency", m.pimEfficiency);
  number(pim, "phase_fixed_us", m.pimPhaseFixedUs);
  number(pim, "pus", m.pimPUs);
  number(pim, "lanes", m.pimLanes);
  number(pim, "channels", m.pimChannels);
  number(pim, "ranks", m.pimRanks);
  number(pim, "partitions", m.pimPartitions);
  number(pim, "ldmac_ck", m.pimLdmacCK);
  number(pim, "act_write_fixed_us", m.actWriteFixedUs);
  number(pim, "act_write_us_per_kib", m.actWriteUsPerKiB);
  number(pim, "readback_fixed_us", m.readbackFixedUs);
  number(pim, "readback_us_per_kib", m.readbackUsPerKiB);
  if (pim)
    if (auto v = pim->getBoolean("concurrent_ranks")) m.pimConcurrentRanks = *v;
  if (mapping) {
    if (auto policy = mapping->getString("policy")) m.policy = policy->str();
    if (auto *overrides = mapping->getObject("overrides"))
      for (auto &[key, value] : *overrides)
        if (auto target = value.getAsString()) m.overrides.push_back({key.str(), target->str()});
  }
  if (m.policy != "cost" && m.policy != "npu" && m.policy != "pim")
    return R::failure("mapping.policy must be cost, npu or pim");
  for (auto &[key, target] : m.overrides)
    if (target != "npu" && target != "pim") return R::failure("override '" + key + "' must map to npu or pim");
  return R::success(m);
}

GemvGroupCost estimateGemvGroup(const HybridCostModel &model, int64_t tokens,
                                const std::vector<std::pair<int64_t, int64_t>> &shapes) {
  GemvGroupCost cost;
  // PIM: a stream is 512 output rows (8 PUs x 16 accumulators x 4 lanes), one
  // sweep of it 32 LdMACs x 16 outputs over 256 inputs. The streams of the group
  // are balanced over the partitions that compute; every token repeats them.
  const int64_t rowsPerStream = int64_t(model.pimPUs) * 16 * model.pimLanes;
  std::vector<int64_t> load(size_t(std::max(1, model.parallelPartitions())), 0);
  int64_t activationBytes = 0, outputBytes = 0;
  for (auto [n, k] : shapes) {
    int64_t streams = (n + rowsPerStream - 1) / rowsPerStream, sweeps = k / 256;
    for (int64_t s = 0; s < streams; ++s) *std::min_element(load.begin(), load.end()) += sweeps;
    activationBytes = std::max(activationBytes, k);
    outputBytes += n * 4;
  }
  double ldmacs = double(*std::max_element(load.begin(), load.end())) * 16 * 32 * double(tokens);
  cost.pimUs = ldmacs * model.pimLdmacCK / model.pimClockMHz / model.pimEfficiency + model.pimPhaseFixedUs;
  cost.boundaryUs = double(tokens) * (model.actWriteFixedUs + model.actWriteUsPerKiB * double(activationBytes) / 1024 +
                                      model.readbackFixedUs + model.readbackUsPerKiB * double(outputBytes) / 1024);
  // NPU: the weights stream from LP6 once for all tokens (memory bound while M <= 32).
  double bytes = 0, macs = 0;
  for (auto [n, k] : shapes) {
    bytes += double(n) * double(k) + double(n) * 4;
    macs += double(n) * double(k) * double(tokens);
  }
  double stream = bytes / (model.npuReadGBps * 1e3);
  double array = macs / (double(model.npuCores) * 32 * 32 * 2 * model.npuClockMHz);
  cost.npuUs = std::max(stream, array) + model.npuGemvFixedUs * double(shapes.size());
  return cost;
}

namespace {
func::FuncOp entryOf(ModuleOp m) {
  auto functions = m.getOps<func::FuncOp>();
  return llvm::hasSingleElement(functions) ? *functions.begin() : func::FuncOp();
}
bool isInteger8Matmul(linalg::MatmulOp mm) {
  return cast<ShapedType>(mm.getDpsInputOperand(0)->get().getType()).getElementType().isInteger(8);
}
std::string argumentName(int64_t index, const std::vector<std::string> &names) {
  if (index >= 0 && size_t(index) < names.size() && !names[size_t(index)].empty()) return names[size_t(index)];
  return "arg" + std::to_string(index);
}
std::string weightName(linalg::MatmulOp mm, const std::vector<std::string> &names) {
  auto arg = dyn_cast<BlockArgument>(mm.getDpsInputOperand(1)->get());
  return arg ? argumentName(arg.getArgNumber(), names) : "<computed>";
}
struct GemvGroup {
  Value input;
  SmallVector<linalg::MatmulOp> members;
};
// INT8 matmuls grouped by the activation they read, in program order.
SmallVector<GemvGroup> gemvGroups(func::FuncOp f) {
  SmallVector<GemvGroup> groups;
  DenseMap<Value, size_t> index;
  f.walk([&](linalg::MatmulOp mm) {
    if (!isInteger8Matmul(mm)) return;
    Value input = mm.getDpsInputOperand(0)->get();
    auto [it, inserted] = index.try_emplace(input, groups.size());
    if (inserted) groups.push_back({input, {}});
    groups[it->second].members.push_back(mm);
  });
  return groups;
}
} // namespace

FailureOr<std::string> partitionGraph(ModuleOp m, const HybridCostModel &model,
                                      const std::vector<std::string> &names) {
  auto f = entryOf(m);
  if (!f) return m.emitError("partitioning requires one entry function");
  auto *context = m.getContext();
  llvm::json::Array table;
  int onPim = 0, onNpu = 0;
  double estimate = 0;
  for (auto [g, group] : llvm::enumerate(gemvGroups(f))) {
    auto input = cast<ShapedType>(group.input.getType());
    int64_t tokens = input.getDimSize(0), k = input.getDimSize(1);
    std::vector<std::pair<int64_t, int64_t>> shapes;
    llvm::json::Array weights, dims;
    // PIM takes an entry-argument weight (it is laid out in the banks ahead of
    // time) and a K that fills whole 256-input sweeps.
    bool placeable = k % 256 == 0 && !group.input.getDefiningOp<linalg::MatmulOp>();
    for (auto mm : group.members) {
      auto weight = cast<ShapedType>(mm.getDpsInputOperand(1)->get().getType());
      shapes.push_back({weight.getDimSize(1), weight.getDimSize(0)});
      placeable &= isa<BlockArgument>(mm.getDpsInputOperand(1)->get());
      weights.push_back(weightName(mm, names));
      dims.push_back(llvm::json::Array{weight.getDimSize(1), weight.getDimSize(0)});
    }
    GemvGroupCost cost = estimateGemvGroup(model, tokens, shapes);
    // Precedence: an override (longest matching weight name), then the policy,
    // where "cost" compares the two estimates.
    std::string target = cost.pimUs + cost.boundaryUs <= cost.npuUs ? "pim" : "npu", reason = "cost";
    if (model.policy != "cost") {
      target = model.policy;
      reason = "policy";
    }
    size_t longest = 0;
    for (auto &[key, forced] : model.overrides)
      for (auto mm : group.members)
        if (key.size() > longest && weightName(mm, names).find(key) != std::string::npos) {
          longest = key.size();
          target = forced;
          reason = "override '" + key + "'";
        }
    if (target == "pim" && !placeable) {
      target = "npu";
      reason += "; PIM needs an argument weight and K % 256 == 0";
    }
    (target == "pim" ? onPim : onNpu)++;
    estimate += target == "pim" ? cost.pimUs + cost.boundaryUs : cost.npuUs;
    for (auto mm : group.members) {
      mm->setAttr("npu.target", StringAttr::get(context, target));
      mm->setAttr("npu.group", IntegerAttr::get(IntegerType::get(context, 64), int64_t(g)));
    }
    table.push_back(llvm::json::Object{{"group", int64_t(g)},
                                       {"weights", std::move(weights)},
                                       {"shapes_nk", std::move(dims)},
                                       {"tokens", tokens},
                                       {"npu_us", cost.npuUs},
                                       {"pim_us", cost.pimUs},
                                       {"pim_boundary_us", cost.boundaryUs},
                                       {"target", target},
                                       {"reason", reason}});
  }
  llvm::json::Object report{{"policy", model.policy},
                            {"groups_on_pim", onPim},
                            {"groups_on_npu", onNpu},
                            {"estimated_gemv_us", estimate},
                            {"groups", std::move(table)}};
  std::string text;
  llvm::raw_string_ostream os(text);
  os << llvm::formatv("{0:2}", llvm::json::Value(std::move(report)));
  return text;
}

FailureOr<HybridPlan> outlinePhases(ModuleOp m, const std::vector<std::string> &names) {
  auto f = entryOf(m);
  if (!f || !llvm::hasSingleElement(f.getBody())) return m.emitError("outlining requires one single-block entry function");
  Block &body = f.front();
  auto *context = m.getContext();
  HybridPlan plan;

  // PIM groups, in the order their first GEMV appears.
  DenseMap<Operation *, int64_t> groupOf;
  DenseMap<int64_t, int64_t> denseGroup;
  SmallVector<SmallVector<linalg::MatmulOp>> members;
  bool nested = false;
  f.walk([&](linalg::MatmulOp mm) {
    auto target = mm->getAttrOfType<StringAttr>("npu.target");
    if (!target || target.getValue() != "pim") return;
    if (mm->getBlock() != &body) { nested = true; return; }
    int64_t g = mm->getAttrOfType<IntegerAttr>("npu.group").getInt();
    auto [it, inserted] = denseGroup.try_emplace(g, int64_t(members.size()));
    if (inserted) members.emplace_back();
    members[size_t(it->second)].push_back(mm);
    groupOf[mm.getOperation()] = it->second;
  });
  if (nested) return m.emitError("a PIM GEMV inside a loop cannot be outlined");
  const int64_t groups = int64_t(members.size());

  auto operandsOf = [](Operation *op) {
    llvm::SetVector<Value> values(op->operand_begin(), op->operand_end());
    getUsedValuesDefinedAbove(op->getRegions(), values);
    return values;
  };
  // Values that depend on no input: constants, empty tensors, constant fills and
  // views or insertions of those. They carry no data from one phase to another,
  // so every phase that reads one makes its own copy. (The zero fill of a PIM
  // accumulator is one; no NPU phase reads it.)
  DenseSet<Operation *> free;
  for (Operation &op : body.without_terminator()) {
    bool candidate = isa<arith::ConstantOp, tensor::EmptyOp, linalg::FillOp, tensor::ExtractSliceOp,
                         tensor::CollapseShapeOp, tensor::ExpandShapeOp, tensor::InsertSliceOp>(op);
    if (candidate && llvm::all_of(operandsOf(&op), [&](Value v) {
          Operation *producer = v.getDefiningOp();
          return producer && free.count(producer);
        }))
      free.insert(&op);
  }
  // Views move no data either. They too are made again wherever they are read,
  // so what crosses a phase boundary is always a whole buffer.
  auto cloneable = [&](Operation *op) {
    return op && (free.count(op) || isa<tensor::ExtractSliceOp, tensor::CollapseShapeOp, tensor::ExpandShapeOp>(op));
  };

  // The earliest NPU phase in which a value can be read: an argument from the
  // start, a PIM result after its group, a view when what it views is ready,
  // anything else after its producer.
  DenseMap<Operation *, int64_t> phaseOf;
  std::function<int64_t(Value)> available = [&](Value v) -> int64_t {
    if (isa<BlockArgument>(v)) return 0;
    Operation *producer = v.getDefiningOp();
    if (auto it = groupOf.find(producer); it != groupOf.end()) return it->second + 1;
    if (cloneable(producer)) {
      int64_t phase = 0;
      for (Value operand : operandsOf(producer)) phase = std::max(phase, available(operand));
      return phase;
    }
    return phaseOf.lookup(producer);
  };
  for (Operation &op : body.without_terminator()) {
    if (groupOf.count(&op) || cloneable(&op)) continue;
    int64_t phase = 0;
    for (Value v : operandsOf(&op)) phase = std::max(phase, available(v));
    phaseOf[&op] = phase;
  }
  for (int64_t g = 0; g < groups; ++g) {
    Value input = members[size_t(g)].front().getDpsInputOperand(0)->get();
    for (auto mm : members[size_t(g)])
      if (mm.getDpsInputOperand(0)->get() != input)
        return mm.emitError("the GEMVs of one PIM group must read one activation");
    if (available(input) > g) return members[size_t(g)].front().emitError("PIM group reads a value computed after it");
  }

  // Buffers a value is made of, looking through views: block arguments, PIM
  // results and results of NPU ops.
  std::function<void(Value, llvm::SetVector<Value> &)> sources = [&](Value v, llvm::SetVector<Value> &out) {
    Operation *producer = v.getDefiningOp();
    if (cloneable(producer)) {
      for (Value operand : operandsOf(producer)) sources(operand, out);
      return;
    }
    out.insert(v);
  };
  // A returned value or a PIM input that is a view has to become a buffer of its
  // own: the phase where it is first readable copies it.
  SmallVector<Value> materialized;
  DenseMap<Value, int64_t> materializedIn;
  auto needsCopy = [&](Value v) {
    if (!cloneable(v.getDefiningOp()) || materializedIn.count(v)) return;
    materializedIn[v] = available(v);
    materialized.push_back(v);
  };
  for (Value v : body.getTerminator()->getOperands()) needsCopy(v);
  for (int64_t g = 0; g < groups; ++g) needsCopy(members[size_t(g)].front().getDpsInputOperand(0)->get());

  // Globals: entry arguments an NPU phase reads, values that leave their phase,
  // and PIM results.
  DenseMap<Value, int64_t> globalOf;
  auto add = [&](Value v, HybridGlobal global) {
    auto type = cast<RankedTensorType>(v.getType());
    global.type = type;
    global.bytes = uint64_t(type.getNumElements()) * elementBytes(type.getElementType());
    plan.globals.push_back(std::move(global));
    globalOf[v] = int64_t(plan.globals.size()) - 1;
    return globalOf[v];
  };
  auto globalFor = [&](Value v) -> int64_t {
    if (auto it = globalOf.find(v); it != globalOf.end()) return it->second;
    HybridGlobal g;
    if (auto arg = dyn_cast<BlockArgument>(v)) {
      g.kind = HybridGlobal::Kind::Argument;
      g.argument = arg.getArgNumber();
      g.name = argumentName(g.argument, names);
    } else if (auto it = groupOf.find(v.getDefiningOp()); it != groupOf.end()) {
      g.kind = HybridGlobal::Kind::PimOutput;
      g.producer = it->second;
      g.name = "pim" + std::to_string(it->second) + "." + weightName(cast<linalg::MatmulOp>(v.getDefiningOp()), names);
    } else {
      g.kind = HybridGlobal::Kind::Value;
      g.producer = materializedIn.count(v) ? materializedIn.lookup(v) : phaseOf.lookup(v.getDefiningOp());
      g.name = "phase" + std::to_string(g.producer) + ".value" + std::to_string(plan.globals.size());
    }
    return add(v, std::move(g));
  };
  // The statement of the entry block that contains `op`.
  auto statement = [&](Operation *op) { return body.findAncestorOpInBlock(*op); };
  // Whether an NPU result of `phase` is read outside it, following views.
  std::function<bool(Value, int64_t)> leaves = [&](Value v, int64_t phase) {
    for (OpOperand &use : v.getUses()) {
      Operation *user = statement(use.getOwner());
      if (!user) return true;
      if (isa<func::ReturnOp>(user) || groupOf.count(user)) {
        // Read as itself; a view read there is copied where it is first readable.
        if (!materializedIn.count(v) || materializedIn.lookup(v) != phase) return true;
        continue;
      }
      if (cloneable(user)) {
        for (Value result : user->getResults())
          if (leaves(result, phase)) return true;
        if (materializedIn.count(user->getResult(0)) && materializedIn.lookup(user->getResult(0)) != phase)
          return true;
        continue;
      }
      if (auto it = phaseOf.find(user); it != phaseOf.end() && it->second != phase) return true;
    }
    return false;
  };

  const int64_t phases = groups + 1;
  llvm::StringSet<> kept{"npu.cores", "npu.l1_bytes", "npu.l2_bytes", "npu.dram_bytes", "npu.hardware",
                         "npu.schema", "npu.numerical", "npu.division"};
  for (int64_t p = 0; p < phases; ++p) {
    SmallVector<Operation *> ops;
    for (Operation &op : body.without_terminator())
      if (auto it = phaseOf.find(&op); it != phaseOf.end() && it->second == p) ops.push_back(&op);
    SmallVector<Value> copies;
    for (Value v : materialized)
      if (materializedIn.lookup(v) == p) copies.push_back(v);
    // Inputs: what the phase reads from outside itself. Outputs: what it makes
    // for later phases, a PIM group, or the step's result.
    llvm::SetVector<Value> inputs, outputs;
    DenseSet<Operation *> inside(ops.begin(), ops.end());
    auto read = [&](Value v) {
      llvm::SetVector<Value> made;
      sources(v, made);
      for (Value source : made) {
        Operation *producer = source.getDefiningOp();
        if (producer && inside.count(producer)) continue;
        inputs.insert(source);
      }
    };
    for (auto *op : ops) {
      for (Value v : operandsOf(op)) read(v);
      for (Value result : op->getResults())
        if (!materializedIn.count(result) && leaves(result, p)) outputs.insert(result);
    }
    for (Value v : copies) {
      read(v);
      outputs.insert(v);
    }
    OpBuilder b(context);
    auto module = ModuleOp::create(m.getLoc());
    for (auto attr : m->getAttrs())
      if (kept.contains(attr.getName().getValue())) module->setAttr(attr.getName(), attr.getValue());
    module->setAttr("npu.stage", StringAttr::get(context, "legal"));
    module->setAttr("npu.phase", IntegerAttr::get(IntegerType::get(context, 64), p));
    SmallVector<Type> inputTypes, outputTypes;
    for (Value v : inputs) inputTypes.push_back(v.getType());
    for (Value v : outputs) outputTypes.push_back(v.getType());
    b.setInsertionPointToEnd(module.getBody());
    auto function = b.create<func::FuncOp>(f.getLoc(), "phase" + std::to_string(p),
                                           FunctionType::get(context, inputTypes, outputTypes));
    auto global = [&](Value v) { return IntegerAttr::get(IntegerType::get(context, 64), globalFor(v)); };
    for (auto [i, v] : llvm::enumerate(inputs)) function.setArgAttr(unsigned(i), "npu.global", global(v));
    for (auto [i, v] : llvm::enumerate(outputs)) function.setResultAttr(unsigned(i), "npu.global", global(v));
    Block *entry = function.addEntryBlock();
    IRMapping map;
    for (auto [i, v] : llvm::enumerate(inputs)) map.map(v, entry->getArgument(unsigned(i)));
    b.setInsertionPointToEnd(entry);
    // A view or free value is made again with everything it is made of,
    // producers first.
    std::function<void(Value)> materialize = [&](Value v) {
      Operation *producer = v.getDefiningOp();
      if (!producer || !cloneable(producer) || map.contains(v)) return;
      for (Value operand : operandsOf(producer)) materialize(operand);
      b.clone(*producer, map);
    };
    for (auto *op : ops) {
      for (Value v : operandsOf(op)) materialize(v);
      b.clone(*op, map);
    }
    // A view that leaves the phase is copied into a buffer of its own.
    for (Value v : copies) {
      materialize(v);
      auto type = cast<RankedTensorType>(v.getType());
      Value into = b.create<tensor::EmptyOp>(f.getLoc(), type.getShape(), type.getElementType());
      Value copy = b.create<linalg::CopyOp>(f.getLoc(), map.lookup(v), into).getResult(0);
      map.map(v, copy);
    }
    SmallVector<Value> results;
    for (Value v : outputs) results.push_back(map.lookup(v));
    b.create<func::ReturnOp>(f.getLoc(), results);
    plan.npuPhases.push_back(OwningOpRef<ModuleOp>(module));
  }

  for (int64_t g = 0; g < groups; ++g) {
    PimGroup group;
    Value input = members[size_t(g)].front().getDpsInputOperand(0)->get();
    group.input = globalFor(input);
    auto inputType = cast<RankedTensorType>(input.getType());
    group.tokens = inputType.getDimSize(0);
    group.k = inputType.getDimSize(1);
    for (auto mm : members[size_t(g)]) {
      PimGemv gemv;
      auto weight = cast<BlockArgument>(mm.getDpsInputOperand(1)->get());
      auto weightType = cast<RankedTensorType>(weight.getType());
      gemv.weight = weight.getArgNumber();
      gemv.k = weightType.getDimSize(0);
      gemv.n = weightType.getDimSize(1);
      gemv.name = argumentName(gemv.weight, names);
      gemv.result = globalFor(mm.getResult(0));
      group.gemvs.push_back(gemv);
    }
    plan.pimGroups.push_back(std::move(group));
  }
  for (Value v : body.getTerminator()->getOperands()) plan.outputs.push_back(globalFor(v));
  return plan;
}
} // namespace npu
