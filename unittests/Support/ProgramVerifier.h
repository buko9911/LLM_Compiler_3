#ifndef NPU_UNITTESTS_SUPPORT_PROGRAMVERIFIER_H
#define NPU_UNITTESTS_SUPPORT_PROGRAMVERIFIER_H
#include "npu/Target/Result.h"
#include <cstdint>
#include <vector>

namespace npu {
// Test support: structural checks of builder-stream core blocks (operand and accumulator
// lifetimes) and of encoded ISA 2.0 programs (header, records, dependencies).
inline constexpr uint32_t kProgramMagic = 0x414e4c50;
inline constexpr uint32_t kProgramVersion = 0x00020000;
Result<bool> verifyCore(const std::vector<uint32_t> &words);
Result<bool> verifyProgramWords(const std::vector<uint32_t> &words);
} // namespace npu
#endif
