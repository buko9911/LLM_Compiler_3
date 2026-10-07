#ifndef NPU_DECODE_PIMMETADATA_H
#define NPU_DECODE_PIMMETADATA_H
#include "npu/Target/PimLayout.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SHA256.h"
namespace npu::decode {
llvm::json::Value toJSON(const pim::Program &program);
pim::Program fromJSON(const llvm::json::Value &value);
// Current W8A8 runtime storage reservations, including tensors not selected for
// PIM execution. Distinct from the reusable weight-only manifest below.
llvm::json::Object pimRuntimeStorageLayout(const pim::Program &program);

// Opt-in storage contract for the existing PIM-only matrix path. Invocation
// fields (M, activations, outputs, primitive ranges) deliberately do not appear.
struct PimWeightStorageEntry {
  std::string name;
  int64_t n = 0, k = 0, nPad = 0;
  std::vector<int> streamPart;
  std::vector<int64_t> streamWeightBase;
  uint64_t scaleAddress = 0, scaleBytes = 0;
  std::string sourceDtype, sourceSHA256, packetSHA256, scaleSHA256;
};
struct PimWeightStorage {
  int channels = 0, partitions = 0, banksetPingPongLead = 0;
  int64_t kernelRow = 0, capacityBytes = 0;
  std::vector<PimWeightStorageEntry> weights; // canonical name order
};

// Canonical digest: uint64 little endian; strings are length-prefixed;
// byte blocks have the lengths established by the relevant schema.
class PimStorageDigest {
public:
  void integer(uint64_t value);
  void text(llvm::StringRef value);
  void bytes(llvm::ArrayRef<uint8_t> value) { hash.update(value); }
  std::string finish();
private:
  llvm::SHA256 hash;
};

Result<bool> validatePimWeightStorage(const PimWeightStorage &, bool requireContent);
Result<PimWeightStorage> parsePimWeightStorage(const llvm::json::Value &);
llvm::json::Value toJSON(const PimWeightStorage &);
std::string pimWeightLayoutDigest(const PimWeightStorage &);
std::string pimWeightManifestDigest(const PimWeightStorage &);
// Keeps the new program's invocation fields; rebuilds activation partition sets.
Result<bool> bindPimWeightStorage(pim::Program &, const PimWeightStorage &);
}
#endif
