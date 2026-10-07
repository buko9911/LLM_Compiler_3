//===- DriverStorage.cpp - fixed PIM weight storage import / export --------===//
//
// --pim-weight-storage-import pins the PIM weight / scale placement to a verified manifest,
// --pim-weight-storage-export writes one. Both require the image: the manifest is verified
// against the bytes the image writer stores (source, packet and scale digests).
//===----------------------------------------------------------------------===//

#include "DriverStorage.h"
#include "DriverPackage.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::hybrid;
namespace json = llvm::json;

std::optional<::npu::decode::PimWeightStorage>
driver::parsePimWeightStorageImport(const llvm::MemoryBuffer *buffer, std::error_code readError,
                                    int banksetPingPong, bool banksetExplicit) {
  if (!buffer) {
    llvm::errs() << "cannot read PIM weight storage manifest: " << readError.message() << '\n'; return std::nullopt;
  }
  auto value = json::parse(buffer->getBuffer());
  if (!value) {
    llvm::errs() << "invalid PIM weight storage JSON: " << llvm::toString(value.takeError()) << '\n';
    return std::nullopt;
  }
  auto parsed = ::npu::decode::parsePimWeightStorage(*value);
  if (!parsed) { llvm::errs() << parsed.error << '\n'; return std::nullopt; }
  std::optional<::npu::decode::PimWeightStorage> importedStorage = std::move(parsed.value);
  const int normalizedLead = banksetPingPong == 1 ? 12 : int(banksetPingPong);
  if (banksetExplicit && normalizedLead != importedStorage->banksetPingPongLead) {
    llvm::errs() << "PIM weight storage bankset layout differs from explicit --pim-bankset-pingpong\n";
    return std::nullopt;
  }
  return importedStorage;
}

bool driver::collectPimWeightStorage(const ::npu::pim::Program &prog,
                                     const std::map<std::string, std::pair<uint64_t, uint64_t>> &scaleSlots,
                                     int64_t capacityBytes, ::npu::decode::PimWeightStorage &storage,
                                     std::map<std::string, size_t> &storageIndex) {
  storage.channels = prog.channels; storage.partitions = prog.partitions;
  storage.kernelRow = prog.kernelRow; storage.banksetPingPongLead = prog.banksetPingPongLead;
  storage.capacityBytes = capacityBytes;
  for (const auto &p : prog.projections) {
    if (!scaleSlots.count(p.name)) { llvm::errs() << "PIM weight storage missing row_scale: " << p.name << '\n'; return false; }
    ::npu::decode::PimWeightStorageEntry w;
    w.name = p.name; w.n = p.n; w.k = p.k; w.nPad = p.nPad;
    w.streamPart = p.streamPart; w.streamWeightBase = p.streamWeightBase;
    w.scaleAddress = scaleSlots.at(p.name).first; w.scaleBytes = scaleSlots.at(p.name).second;
    storage.weights.push_back(std::move(w));
  }
  std::sort(storage.weights.begin(), storage.weights.end(), [](const auto &a, const auto &b) { return a.name < b.name; });
  auto checked = ::npu::decode::validatePimWeightStorage(storage, false);
  if (!checked) { llvm::errs() << checked.error << '\n'; return false; }
  for (size_t i = 0; i < storage.weights.size(); ++i) storageIndex[storage.weights[i].name] = i;
  return true;
}

std::string driver::recordStorageSource(const SafeTensors &st, const std::string &name,
                                        ::npu::decode::PimWeightStorageEntry *storageWeight,
                                        const ::npu::decode::PimWeightStorageEntry *expectedStorage) {
  const auto *source = st.find(name);
  if (!source || source->shape != std::vector<int64_t>{storageWeight->n, storageWeight->k} ||
      (source->dtype != "F16" && source->dtype != "BF16"))
    return "PIM weight storage source dtype/shape mismatch: " + name;
  ::npu::decode::PimStorageDigest h;
  h.text("lp6-pim-source-v1"); h.text(name); h.text(source->dtype);
  h.integer(storageWeight->n); h.integer(storageWeight->k);
  h.bytes(llvm::ArrayRef<uint8_t>(source->data, size_t(storageWeight->n * storageWeight->k) * 2));
  storageWeight->sourceDtype = source->dtype; storageWeight->sourceSHA256 = h.finish();
  if (expectedStorage && (storageWeight->sourceDtype != expectedStorage->sourceDtype ||
                          storageWeight->sourceSHA256 != expectedStorage->sourceSHA256))
    return "PIM weight storage source SHA256 mismatch: " + name;
  return "";
}

bool driver::finishPimWeightStorage(const ::npu::decode::PimWeightStorage &storage,
                                    const std::optional<::npu::decode::PimWeightStorage> &importedStorage,
                                    ModuleOp mod, MLIRContext &ctx, const std::string &outDir,
                                    const std::string &exportPath, std::string &err) {
  auto checked = ::npu::decode::validatePimWeightStorage(storage, true);
  if (!checked) { llvm::errs() << checked.error << '\n'; return false; }
  const std::string layoutHash = ::npu::decode::pimWeightLayoutDigest(storage);
  const std::string manifestHash = ::npu::decode::pimWeightManifestDigest(storage);
  if (importedStorage && manifestHash != ::npu::decode::pimWeightManifestDigest(*importedStorage)) {
    llvm::errs() << "PIM weight storage verified manifest differs from import\n"; return false;
  }
  auto value = ::npu::decode::toJSON(storage);
  if (!writeAtomicJSON(outDir + "/pim_weight_storage_verified.json", value, err) ||
      (!exportPath.empty() && !writeAtomicJSON(exportPath, value, err))) {
    llvm::errs() << "PIM weight storage export failed: " << err << '\n'; return false;
  }
  mod->setAttr("hyb.pim_weight_storage_layout_sha256", StringAttr::get(&ctx, layoutHash));
  mod->setAttr("hyb.pim_weight_storage_manifest_sha256", StringAttr::get(&ctx, manifestHash));
  {
    std::error_code ec;
    llvm::raw_fd_ostream os(outDir + "/compiled.mlir", ec);
    if (ec) { llvm::errs() << ec.message() << '\n'; return false; }
    mod.print(os, OpPrintingFlags().elideLargeElementsAttrs());
  }
  return addStorageToIO(outDir, layoutHash, manifestHash, err);
}
