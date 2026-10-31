//===- NpuCost.h - NPU time of the choices the passes make ------*- C++ -*-===//
//
// Several passes choose between equivalent programs: run an op on one core or split it over the
// cores, run an elementwise consumer tile by tile inside a GEMV loop or after it. Each choice
// compares the times below. They are computed from the tensor shapes and the target's hardware
// numbers (TargetSpec::Npu timing: the simulator's model, docs/architecture/NPU_SIMULATOR.md
// §6-§8), so a choice never depends on which step (decode, verify, prefill) is compiled: equal
// shapes give equal programs. Every choice is recorded with its numbers (recordDecision) and
// reported in io.json.
//===----------------------------------------------------------------------===//
#ifndef NPU_DECODE_TRANSFORMS_NPUCOST_H
#define NPU_DECODE_TRANSFORMS_NPUCOST_H

#include "npu/Decode/Support/TargetSpec.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include <utility>

namespace mlir::hybrid::cost {

/// NPU cycles per microsecond.
inline double cyclesPerUs(const TargetSpec::Npu &npu) { return npu.clockMHz; }

/// One core's cycles for a weight tile of `columns` output columns (saCols per SA column block),
/// `k` deep, for `m` token rows. Per SA row block (up to saRows rows) and column block: M_LD of the
/// weights (k x saCols bytes) and of the activations (rows x k bytes) at the L1 read rate, then the
/// MMA chain (k / 32 pieces fed at mmaKPerCycle, one drain). The single W/A operand buffer keeps
/// M_LD and MMA apart (ISA §6), so the terms add.
double gemvTileCycles(const TargetSpec::Npu &npu, int64_t m, int64_t k, int64_t columns);

/// Cycles until `bytes` of streamed weights reach one core when `streams` cores stream at once
/// (they share the sustained GDMA read bandwidth).
double streamCycles(const TargetSpec::Npu &npu, int64_t bytes, int64_t streams);

/// VPU cycles of a generic over its loop nest (genericCycles) and per point (vectorCyclesPerElement).
/// The vector axis is the innermost loop. Per point: the lane-ALU ops on values that vary along it
/// (FP32, vectorLanes elements per cycle), its SFU ops (exp, reciprocal, (r)sqrt) at the shared
/// rates, and one 512-bit load or store (vectorLanes x 4 bytes) per cycle for every operand that
/// varies along it; FP16 <-> FP32 conversions ride on those loads and stores. Per row: ops on
/// row-invariant values (computed once as scalars) at their latency, and the horizontal reduce of a
/// reduction. The terms add, as the VPU issues a chunk's ops in order.
double genericCycles(const TargetSpec::Npu &npu, linalg::GenericOp g);
double vectorCyclesPerElement(const TargetSpec::Npu &npu, linalg::GenericOp g);

/// VPU cycles per element of hyb.quant (|x|, max, scale, round to INT8: four lane ops, FP16 load,
/// INT8 store) and hyb.dequant (one fused convert-scale op, INT32 and FP32 loads, FP16 store).
double quantCyclesPerElement(const TargetSpec::Npu &npu);
double dequantCyclesPerElement(const TargetSpec::Npu &npu);

/// Cycles per element of a layout copy of `bytes`-wide elements through L1 (read and write).
double copyCyclesPerElement(const TargetSpec::Npu &npu, int64_t bytes);

/// Splitting work of `cycles` on one core over `cores` cores saves (1 - 1/cores) of it; it pays when
/// that saving exceeds the fixed cost of a split (coreSplitUs).
inline double splitSavingUs(const TargetSpec::Npu &npu, double cycles, int64_t cores) {
  return cores > 1 ? cycles * (1.0 - 1.0 / double(cores)) / cyclesPerUs(npu) : 0.0;
}
inline bool splitPays(const TargetSpec::Npu &npu, double cycles, int64_t cores) {
  return cores > 1 && splitSavingUs(npu, cycles, cores) > npu.coreSplitUs;
}

/// Appends {kind, op, choice, numbers...} to the phase's `hyb.decisions` (io.json phases[].decisions).
void recordDecision(func::FuncOp phase, StringRef kind, StringRef op, StringRef choice,
                    ArrayRef<std::pair<StringRef, double>> numbers);

} // namespace mlir::hybrid::cost

#endif
