#ifndef NPU_DECODE_PIMMETADATA_H
#define NPU_DECODE_PIMMETADATA_H
#include "npu/Target/PimLayout.h"
#include "llvm/Support/JSON.h"
namespace npu::decode {
llvm::json::Value toJSON(const pim::Program &program);
pim::Program fromJSON(const llvm::json::Value &value);
}
#endif
