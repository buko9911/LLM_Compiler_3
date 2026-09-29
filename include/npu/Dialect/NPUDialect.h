#ifndef NPU_DIALECT_NPUDIALECT_H
#define NPU_DIALECT_NPUDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "npu/Target/Result.h"
#include <vector>
#include "npu/Dialect/NPUDialect.h.inc"

#define GET_OP_CLASSES
#include "npu/Dialect/NPUOps.h.inc"

namespace mlir::npu {
// Encodes the body of a npu.core_block (npu.instruction and npu.matrix_*
// ops) into its core ISA words. Shared by the verifier and the encode pass.
::npu::Result<std::vector<uint32_t>> encodeCoreBody(Block &body);
// Rebuilds a core block body from ISA words: one op per record.
LogicalResult buildCoreBody(OpBuilder &builder, Location loc, const std::vector<uint32_t> &words);
} // namespace mlir::npu

#endif
