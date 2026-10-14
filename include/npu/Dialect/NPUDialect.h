#ifndef NPU_DIALECT_NPUDIALECT_H
#define NPU_DIALECT_NPUDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/SCF/IR/DeviceMappingInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "npu/Target/Result.h"
#include <vector>
#include "npu/Dialect/NPUDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "npu/Dialect/NPUTypes.h.inc"
#define GET_ATTRDEF_CLASSES
#include "npu/Dialect/NPUAttrs.h.inc"
#define GET_OP_CLASSES
#include "npu/Dialect/NPUOps.h.inc"

namespace mlir::npu {
// Encodes the body of a npu.core_block (npu.instruction and npu.matrix_*
// ops) into its core ISA words. Shared by the verifier and the encode pass.
::npu::Result<std::vector<uint32_t>> encodeCoreBody(Block &body);
// Rebuilds a core block body from ISA words: one op per record.
LogicalResult buildCoreBody(OpBuilder &builder, Location loc, const std::vector<uint32_t> &words);
// Memory space of a placed buffer: 0 DRAM, 1 L2, 2 L1.
inline unsigned memorySpace(Type type) {
  auto memref = dyn_cast<BaseMemRefType>(type);
  return memref ? memref.getMemorySpaceAsInt() : 0;
}
} // namespace mlir::npu

#endif
