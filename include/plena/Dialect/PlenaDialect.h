#ifndef PLENA_DIALECT_PLENADIALECT_H
#define PLENA_DIALECT_PLENADIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "plena/Target/Result.h"
#include <vector>
#include "plena/Dialect/PlenaDialect.h.inc"

#define GET_OP_CLASSES
#include "plena/Dialect/PlenaOps.h.inc"

namespace mlir::plena {
// Encodes the body of a plena.core_block (plena.instruction and plena.matrix_*
// ops) into its core ISA words. Shared by the verifier and the encode pass.
::plena::Result<std::vector<uint32_t>> encodeCoreBody(Block &body);
// Rebuilds a core block body from ISA words: one op per record.
LogicalResult buildCoreBody(OpBuilder &builder, Location loc, const std::vector<uint32_t> &words);
} // namespace mlir::plena

#endif
