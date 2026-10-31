//===- NpuCost.cpp - NPU time of the choices the passes make ---------------===//

#include "npu/Decode/Transforms/NpuCost.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseSet.h"
#include <algorithm>
#include <cmath>

using namespace mlir;
using namespace mlir::hybrid;

double cost::gemvTileCycles(const TargetSpec::Npu &npu, int64_t m, int64_t k, int64_t columns) {
  const double feed = std::ceil(double(k) / 32.0) * std::ceil(32.0 / double(npu.mmaKPerCycle));
  const double drain = double(npu.saRows + npu.saCols - 3 + npu.mmaPipelineCycles);
  const int64_t blocks = (columns + npu.saCols - 1) / npu.saCols;
  double cycles = 0;
  for (int64_t row = 0; row < m; row += npu.saRows) {
    const int64_t rows = std::min<int64_t>(npu.saRows, m - row);
    const double load = double(k) * double(npu.saCols + rows) / double(npu.l1ReadBytesPerCycle);
    cycles += double(blocks) * (load + feed + drain);
  }
  return cycles;
}

double cost::streamCycles(const TargetSpec::Npu &npu, int64_t bytes, int64_t streams) {
  const double bytesPerCycle = npu.gdmaReadGBps * 1e3 / npu.clockMHz;   // GB/s -> bytes per NPU cycle
  return double(bytes) * double(std::max<int64_t>(1, streams)) / bytesPerCycle;
}

namespace {
int64_t elementBytes(Type t) { return std::max<int64_t>(1, int64_t(t.getIntOrFloatBitWidth()) / 8); }
} // namespace

namespace {
/// Latency (cycles) of a VPU op on a per-row scalar (NPU_SIMULATOR.md §8): the backend computes a
/// value that does not vary along the vector axis once per row.
double scalarLatency(Operation &op) {
  if (isa<math::ExpOp, math::Exp2Op>(op)) return 6;
  if (isa<arith::DivFOp>(op)) return 5 + 2;                                  // reciprocal, multiply
  if (isa<math::RsqrtOp, math::SqrtOp>(op)) return 8;
  return 2;
}
} // namespace

double cost::genericCycles(const TargetSpec::Npu &npu, linalg::GenericOp g) {
  SmallVector<int64_t> ranges = g.getStaticLoopRanges();
  if (ranges.empty()) return 0;
  int64_t points = 1;
  for (int64_t r : ranges) points *= std::max<int64_t>(1, r);
  const int64_t rows = points / std::max<int64_t>(1, ranges.back());
  // The vector axis is the innermost loop. An operand varies along it if its map uses that loop;
  // such operands are loaded (and the result stored) chunk by chunk on every row, one 512-bit
  // register (vectorLanes x 4 bytes) per cycle.
  const AffineExpr inner = getAffineDimExpr(unsigned(ranges.size() - 1), g.getContext());
  const double vectorBytes = 4.0 * double(npu.vectorLanes);
  llvm::DenseSet<Value> vector;
  double loads = 0, stores = 0;
  for (OpOperand &o : g->getOpOperands()) {
    AffineMap map = g.getMatchingIndexingMap(&o);
    if (!llvm::is_contained(map.getResults(), inner)) continue;
    vector.insert(g.getMatchingBlockArgument(&o));
    double bytes = double(elementBytes(getElementTypeOrSelf(o.get().getType())));
    if (g.isDpsInput(&o)) loads += bytes;
    else stores += bytes;
  }
  const double lane = 1.0 / double(npu.vectorLanes);
  double perPoint = (loads + stores) / vectorBytes, perRow = g.getNumReductionLoops() ? 6.0 : 0.0;
  for (Operation &op : g.getRegion().front()) {
    if (isa<linalg::YieldOp, arith::ConstantOp, linalg::IndexOp>(op)) continue;
    bool isVector = llvm::any_of(op.getOperands(), [&](Value v) { return vector.contains(v); });
    if (isVector)
      for (Value r : op.getResults()) vector.insert(r);
    // FP16 <-> FP32 conversions ride on the vector loads and stores (the backend aliases them).
    if (isa<arith::ExtFOp, arith::TruncFOp>(op)) continue;
    if (!isVector) { perRow += scalarLatency(op); continue; }
    if (isa<math::ExpOp, math::Exp2Op>(op)) perPoint += 1.0 / double(npu.vectorExpPerCycle);
    else if (isa<arith::DivFOp>(op)) perPoint += 1.0 / double(npu.vectorRcpPerCycle) + lane;
    else if (isa<math::RsqrtOp, math::SqrtOp>(op)) perPoint += 1.0 / double(npu.vectorSqrtPerCycle);
    else perPoint += lane;                                                   // lane ALU ops
  }
  return double(points) * perPoint + double(rows) * perRow;
}

double cost::vectorCyclesPerElement(const TargetSpec::Npu &npu, linalg::GenericOp g) {
  int64_t points = 1;
  for (int64_t r : g.getStaticLoopRanges()) points *= std::max<int64_t>(1, r);
  return genericCycles(npu, g) / double(std::max<int64_t>(1, points));
}

double cost::quantCyclesPerElement(const TargetSpec::Npu &npu) {
  return 4.0 / double(npu.vectorLanes) + (2.0 + 1.0) / (4.0 * double(npu.vectorLanes));
}

double cost::dequantCyclesPerElement(const TargetSpec::Npu &npu) {
  return 1.0 / double(npu.vectorLanes) + (4.0 + 4.0 + 2.0) / (4.0 * double(npu.vectorLanes));
}

double cost::copyCyclesPerElement(const TargetSpec::Npu &npu, int64_t bytes) {
  return double(bytes) / double(npu.l1ReadBytesPerCycle) + double(bytes) / double(npu.l1WriteBytesPerCycle);
}

void cost::recordDecision(func::FuncOp phase, StringRef kind, StringRef op, StringRef choice,
                          ArrayRef<std::pair<StringRef, double>> numbers) {
  MLIRContext *ctx = phase.getContext();
  Builder b(ctx);
  SmallVector<NamedAttribute> fields = {b.getNamedAttr("kind", b.getStringAttr(kind)),
                                        b.getNamedAttr("op", b.getStringAttr(op)),
                                        b.getNamedAttr("choice", b.getStringAttr(choice))};
  for (auto [name, value] : numbers)
    fields.push_back(b.getNamedAttr(name, b.getF64FloatAttr(std::round(value * 1000.0) / 1000.0)));
  SmallVector<Attribute> list;
  if (auto old = phase->getAttrOfType<ArrayAttr>("hyb.decisions")) list.append(old.begin(), old.end());
  list.push_back(b.getDictionaryAttr(fields));
  phase->setAttr("hyb.decisions", b.getArrayAttr(list));
}
