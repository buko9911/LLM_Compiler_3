//===- DriverStorage.h - fixed PIM weight storage import / export -*- C++ -*-===//
//
// Private to the decode driver. --pim-weight-storage-import / -export fix the PIM weight and
// scale placement; the image writer records the content digests of what it writes, and the
// verified manifest is written only after every digest has been checked.
//===----------------------------------------------------------------------===//
#ifndef NPU_LIB_DECODE_DRIVERSTORAGE_H
#define NPU_LIB_DECODE_DRIVERSTORAGE_H

#include "npu/Decode/PimMetadata.h"
#include "npu/Decode/Support/SafeTensors.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/MemoryBuffer.h"
#include <map>
#include <optional>
#include <string>
#include <system_error>

namespace mlir::hybrid::driver {

/// The imported manifest from its bytes (read before the output directory was cleaned). An
/// explicit --pim-bankset-pingpong must match its layout. Prints the error and returns nullopt.
std::optional<::npu::decode::PimWeightStorage>
parsePimWeightStorageImport(const llvm::MemoryBuffer *buffer, std::error_code readError, int banksetPingPong,
                            bool banksetExplicit);

/// The storage manifest of this compile, without content digests, from the PIM plan and the
/// row_scale slots; `storageIndex` maps a weight name to its entry (canonical name order).
bool collectPimWeightStorage(const ::npu::pim::Program &prog,
                             const std::map<std::string, std::pair<uint64_t, uint64_t>> &scaleSlots,
                             int64_t capacityBytes, ::npu::decode::PimWeightStorage &storage,
                             std::map<std::string, size_t> &storageIndex);

/// Records the checkpoint source of one stored weight and checks it against the import;
/// returns the error, empty when it matches.
std::string recordStorageSource(const SafeTensors &st, const std::string &name,
                                ::npu::decode::PimWeightStorageEntry *storageWeight,
                                const ::npu::decode::PimWeightStorageEntry *expectedStorage);

/// After the image: validates the complete manifest, checks it against the import, writes the
/// verified sidecar (and the export), and records its digests in compiled.mlir and io.json.
bool finishPimWeightStorage(const ::npu::decode::PimWeightStorage &storage,
                            const std::optional<::npu::decode::PimWeightStorage> &importedStorage, ModuleOp mod,
                            MLIRContext &ctx, const std::string &outDir, const std::string &exportPath,
                            std::string &err);

} // namespace mlir::hybrid::driver

#endif
