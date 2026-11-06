//===- NpuCodegen.h - NPU phase -> PLENA program records ----------*- C++ -*-===//
#ifndef HYBRID_TARGET_NPUCODEGEN_H
#define HYBRID_TARGET_NPUCODEGEN_H

#include "npu/Decode/Support/TargetSpec.h"
#include "npu/Target/PimLayout.h"
#include "npu/Decode/Target/PlenaProgram.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include <string>
#include <vector>

namespace mlir::hybrid {
namespace pim = ::npu::pim;

/// A blob the image writer places in LP6 (per-phase constant pools).
struct LP6Blob {
  uint64_t addr;
  std::vector<uint8_t> bytes;
};

/// Allocator for LP6 bytes past the planned dense region (below the PIM kernel rows).
struct LP6Extra {
  uint64_t cursor, limit;
  std::vector<LP6Blob> blobs;
  uint64_t add(std::vector<uint8_t> bytes);
  // Research option --common-layout (PimLayout.h): NPU GEMV weights in the common placement, the GDMA size of a tile read.
  std::vector<pim::CommonWeight> commonWeights = {};
  int commonChunkRows = 8;
};

/// L2 and LP6 accesses that an NPU phase leaves for the next NPU phase:
/// the next phase's GDMA loads wait for these instead of the phase's exit barrier.
struct CarriedAccesses {
  struct L2Access {
    uint64_t s, e;
    int id, core, kind;
  };
  struct LP6Access {
    uint64_t s, e;
    int id;
    bool store;
  };
  std::vector<L2Access> l2;
  std::vector<LP6Access> lp6;
};

struct NpuPhaseInput {
  func::FuncOp phase;                     // bufferized phase function
  std::vector<DictionaryAttr> argLP6;     // placement of every argument (from the call site)
  const pim::Program *pim = nullptr;
  const TargetSpec *spec = nullptr;
  plena::ProgramBuilder *program = nullptr;   // the step's single program (phase already begun)
  CarriedAccesses *carried = nullptr;    // in: the previous NPU phase's accesses; out: this phase's
  // NPU+PIM: the PIM phase just before this one, the LP6 ranks it computes on and the records it waited
  // for. A GDMA that touches none of those ranks and has no other hazard follows `beforePim` instead of
  // the phase entry, so its DRAM traffic overlaps the PIM work on the other rank.
  std::vector<int> pimRecords, pimRanks, beforePim;
  // Research option --common-layout: where the NPU GEMV weights really are, and the GDMA size of their rows.
  const std::vector<pim::CommonWeight> *commonWeights = nullptr;
  int commonChunkRows = 8;
  // Research option --kv8=pim: the cache slot a new K/V row is written to. The bursts are real, the slot is fixed at compile time: the last one, which a short
  // test never reads (the program has no indirect addressing; a real one would take the slot from the host).
  int kvSlot = 511;
};

struct NpuPhaseStats {
  int64_t coreWords = 0, commands = 0, l1Peak = 0, cores = 1;
  int64_t l2StagingPerCore = 0;
  int64_t stagingMiscBytes = 0;          // npu.staging_misc_bytes as planned for the phase
  struct WeightStream {
    int64_t tileBytes, iterations, slots, batch;
    bool prefetch, crossBatch;
    std::string reason;
    int64_t initialSlot = 0;
  };
  std::vector<WeightStream> weightStreams;
  int64_t asyncOutputStores = 0, outputStoresWithVectorWork = 0;
  int64_t stagingReleaseCuts = 0;
  int64_t vectorScalarReciprocals = 0, vectorScalarReuses = 0;
  int64_t vectorPipelineGroups = 0, vectorPipelinedChunks = 0;
};

/// Lowers one NPU phase (after hyb-npu-place / -promote / -memplan) into the records of the step
/// program between ProgramBuilder::beginPhase and endPhase.
LogicalResult emitNpuPhase(const NpuPhaseInput &in, LP6Extra &extra, NpuPhaseStats &stats, std::string &error);

} // namespace mlir::hybrid

#endif
