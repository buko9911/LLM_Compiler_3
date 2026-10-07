#ifndef HYBRID_DIALECT_HYB_HYBDIALECT_H
#define HYBRID_DIALECT_HYB_HYBDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "npu/Decode/Dialect/Hyb/HybOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "npu/Decode/Dialect/Hyb/HybOps.h.inc"

namespace mlir::hyb {
/// Registers the external bufferization models of the destination-style hyb ops.
void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry);
} // namespace mlir::hyb

#endif
