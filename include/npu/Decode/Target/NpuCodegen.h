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
};

struct NpuPhaseInput {
  func::FuncOp phase;                     // bufferized phase function
  std::vector<DictionaryAttr> argLP6;     // placement of every argument (from the call site)
  const pim::Program *pim = nullptr;
  const TargetSpec *spec = nullptr;
  plena::ProgramBuilder *program = nullptr;   // the step's single program (phase already begun)
  bool weightPrefetch = true;            // pipeline streamed GEMV weights through the two L1 slots
  int pimActivationCores = 1;            // explicit bounded isolated quant/activation-feed experiment
};

struct NpuPhaseStats {
  int64_t coreWords = 0, commands = 0, l1Peak = 0, cores = 1;
  int64_t l2StagingPerCore = 0;
  int64_t pimActivationCores = 1;
  std::string pimActivationPlanJson;      // checked scratch/ownership plan when parallel feed is used
  struct WeightStream {
    int64_t tileBytes, iterations, slots, batch;
    bool prefetch, crossBatch;
    std::string reason;
  };
  std::vector<WeightStream> weightStreams;
};

/// Lowers one NPU phase (after hyb-npu-place / -promote / -memplan) into the records of the step
/// program between ProgramBuilder::beginPhase and endPhase.
LogicalResult emitNpuPhase(const NpuPhaseInput &in, LP6Extra &extra, NpuPhaseStats &stats, std::string &error);

} // namespace mlir::hybrid

#endif
