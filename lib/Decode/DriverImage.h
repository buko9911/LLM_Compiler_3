//===- DriverImage.h - decode LP6 image writer -------------------*- C++ -*-===//
//
// Private to the decode driver: lp6.bin, the LP6 image of the weights and constant pools.
//===----------------------------------------------------------------------===//
#ifndef NPU_LIB_DECODE_DRIVERIMAGE_H
#define NPU_LIB_DECODE_DRIVERIMAGE_H

#include "npu/Decode/PimMetadata.h"
#include "npu/Decode/Support/TargetSpec.h"
#include "npu/Decode/Target/NpuCodegen.h"
#include "mlir/IR/BuiltinOps.h"
#include <optional>
#include <string>

namespace mlir::hybrid::driver {

/// Writes <outDir>/lp6.bin (`imageBytes`): FP16 dense weights, the NPU constant pools, and the
/// quantized weights (PIM packets, NPU GEMV blocks, per-row scales), all from one per-row INT8
/// quantization of the checkpoint `model`. With fixed storage, `storage` gets the manifest of
/// this compile with the digests of the bytes written, checked per weight against the import.
/// `quantized` is the number of quantized weights.
bool writeImage(const std::string &model, const std::string &outDir, uint64_t imageBytes, ModuleOp mod,
                func::FuncOp step, const pim::Program &prog, const TargetSpec &spec, const LP6Extra &extra,
                bool fixedStorage, const std::optional<::npu::decode::PimWeightStorage> &importedStorage,
                ::npu::decode::PimWeightStorage &storage, size_t &quantized, std::string &err);

} // namespace mlir::hybrid::driver

#endif
