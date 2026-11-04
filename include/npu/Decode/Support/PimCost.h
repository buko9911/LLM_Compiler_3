//===- PimCost.h - PIM stream layout and its cost ---------------------------===//
//
// The PIM side of the cost model, shared by hyb-partition (NPU or PIM per GEMV group) and
// hyb-plan-lp6 (the stream layout of each PIM GEMV). All numbers come from the target's pim
// section, measured on the Ramulator LPDDR6-PIM model (configs/baselines/*.json).
//
// A PIM GEMV runs as streams (npu/Target/PimLayout.h). A stream with A accumulators covers 32 A
// output rows and one K piece; split-K cuts K into S pieces whose INT32 partial sums the NPU adds
// while gathering. Per partition, the streams run one after another, each sweep of 256 inputs as
// 32 A LdMACs. All partitions of a phase run at the same time (pim.concurrent_ranks: both ranks or one).
//
//   stream CK  = tokens x (pieceSweeps x (32 A ldmac_ck + sweep_ck) + stream_ck)
//   phase us   = (max over partitions of its streams' CK) / clock_mhz      (CK = DRAM CK, 2666.667 MHz)
//   gather us  = tokens x S x Npad x 4 B / readback_gbps                              (per projection)
//   (a projection may end in a tail of smaller streams: blocks of tailAccs accumulators)
//   sum us     = tokens x (S - 1) x Npad / 16 x 4 NPU cycles / cores                  (split-K adds)
//
// Every layout gives the same INT32 sums, so the choice only moves time.
//===----------------------------------------------------------------------===//
#ifndef NPU_DECODE_SUPPORT_PIMCOST_H
#define NPU_DECODE_SUPPORT_PIMCOST_H

#include "npu/Decode/Support/TargetSpec.h"
#include "llvm/ADT/ArrayRef.h"
#include <cstdint>
#include <vector>

namespace mlir::hybrid::pimcost {

struct Shape {
  int accs = 16, splitK = 1;
  int tailAccs = 0;          // the last tailBlocks blocks use tailAccs accumulators (0: none)
  int64_t tailBlocks = 0;
  bool operator==(const Shape &o) const {
    return accs == o.accs && splitK == o.splitK && tailAccs == o.tailAccs && tailBlocks == o.tailBlocks;
  }
};

struct Estimate {
  double pimUs = 0, gatherUs = 0, sumUs = 0, scatterUs = 0;
  double total() const { return pimUs + gatherUs + sumUs + scatterUs; }
};

/// Output rows of one projection padded to whole streams of `accs` accumulators.
int64_t paddedRows(const TargetSpec &spec, int64_t n, int accs);

/// Blocks (streams per K piece) of a projection of n rows with `shape`, main ones first; each entry is
/// the block's accumulator count. Matches pim::Program::addProjection.
std::vector<int> shapeBlocks(const TargetSpec &spec, int64_t n, const Shape &shape);

/// PIM clock cycles of one stream (all tokens).
int64_t streamCycles(const TargetSpec &spec, int64_t tokens, int64_t pieceSweeps, int accs);

/// One PIM phase of the projections n[i] x k with shapes[i]: streams assigned in order to the least
/// loaded of `partitions` partitions, as pim::Program::addProjection does.
Estimate estimate(const TargetSpec &spec, int64_t tokens, int64_t k, llvm::ArrayRef<int64_t> n,
                  llvm::ArrayRef<Shape> shapes, int partitions);

/// The fastest shapes for the phase over the target's candidates (pim.stream_accs, split-K up to
/// pim.max_split_k with whole sweeps per piece; no split when `allowSplit` is false). The last
/// projection may also replace its last 1..partitions blocks by a tail of smaller streams. Ties keep
/// the fewer PIM commands (more accumulators, fewer pieces, no tail).
std::vector<Shape> choose(const TargetSpec &spec, int64_t tokens, int64_t k, llvm::ArrayRef<int64_t> n,
                          int partitions, bool allowSplit, Estimate &best);

/// Partitions that compute at the same time in one PIM phase.
int phasePartitions(const TargetSpec &spec);

} // namespace mlir::hybrid::pimcost

#endif
