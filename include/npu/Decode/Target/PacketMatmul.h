//===- PacketMatmul.h - Fixed PIM-packet NPU phase ---------------*- C++ -*-===//
#ifndef NPU_DECODE_TARGET_PACKETMATMUL_H
#define NPU_DECODE_TARGET_PACKETMATMUL_H

#include "npu/Decode/Target/NpuCodegen.h"

namespace mlir::hybrid {

/// Lower one storage-preserving INT8 qmatmul with the checked g4 packet plan.
/// The caller owns beginPhase/endPhase. The phase exclusively owns its checked
/// L1/L2 scratch; input and output are dense LP6 I8[M,K] and I32[M,N].
/// Unsupported layouts, shapes, and capacities fail before appending records.
LogicalResult emitPimPacketNpuPhase(const NpuPhaseInput &in, NpuPhaseStats &stats,
                                  std::string &error);

} // namespace mlir::hybrid
#endif
