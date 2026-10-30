#ifndef NPU_TARGET_ISAV2_H
#define NPU_TARGET_ISAV2_H
#include "npu/Target/Result.h"
#include "npu/Target/PimPrimitive.h"
#include <cstddef>
#include <cstdint>
#include <vector>

// PLENA ISA ver 2.0, the format NPU_Simulator main executes.
//
// The compiler keeps emitting core code as a *builder stream* in the ISA ver 1.0
// encodings (see Decode/Target/PlenaProgram.h).
// lowerCoreBlock turns one core block of that stream into ver 2.0 words the way
// NPU_Simulator's tools/isa_v2.py (lower_core_block) does, so the two produce
// the same words for the same block:
//   * GP constants are materialised (C_LI, or C_LUI + C_ADDI) only where an
//     instruction reads them, and only if no register already holds the value;
//   * vector and scalar memory use base register + scaled offset;
//   * C_SET takes an immediate and is emitted only when the value changes;
//   * consecutive M_MMA K slices of one rectangle become one M_MMA with a count;
//   * aligned L2 <-> L1 transfers become LDMA.I (all immediates), others LDMA.R.
// Inside a hardware loop every GP and control write is kept, so loop semantics
// do not change. A builder vector V_MEM may carry a byte offset from its address
// register (vectorMemoryOffset); the decode compiler uses it only inside
// hardware loops, where the register value is known to the core alone.
namespace npu::isa2 {
/// Builder-stream V_MEM offset field: bytes / 16 in bits [14, 22) and [26, 32) (14 bits).
inline constexpr uint32_t vectorMemoryOffsetBits(uint64_t bytes) {
  const uint64_t units = bytes / 16;
  return uint32_t(units & 0xFF) << 14 | uint32_t(units >> 8 & 0x3F) << 26;
}
inline constexpr uint64_t vectorMemoryOffset(uint32_t word) {
  return (uint64_t(word >> 14 & 0xFF) | uint64_t(word >> 26 & 0x3F) << 8) * 16;
}
inline constexpr uint64_t kVectorMemoryOffsetLimit = uint64_t(1) << 18;   // exclusive
inline constexpr uint32_t kProgramVersion = 0x00020000;
inline constexpr const char *kProgramSchema = "plena.v2.unified_program.isa_v2.0";
inline constexpr uint32_t kCoreBegin = 0x26, kGdmaLoad = 0x2b, kGdmaStore = 0x2c,
                          kBarrier = 0x2d, kProgramEnd = 0x2f;
inline constexpr unsigned kCoreAny = 31;       // CORE_BEGIN logical core: no fixed core
inline constexpr unsigned kMaxDependencies = 255;
inline constexpr uint32_t kMaxDistance = 0xffff;
inline constexpr uint64_t kMaxGdmaRowBytes = uint64_t(1) << 20;  // exclusive

// One builder-stream (ISA ver 1.0) core block -> ISA ver 2.0 core words.
// Optional diagnostic map: each emitted word's originating builder instruction.
// Coalesced MMA slices identify their first source instruction. Encoding is unchanged.
Result<std::vector<uint32_t>> lowerCoreBlock(const std::vector<uint32_t> &builderWords,
                                           std::vector<size_t> *sourceWords = nullptr);

// Length in words of the ver 2.0 core instruction at words[0], or 0 when the
// opcode is not a ver 2.0 core instruction or the record is truncated.
unsigned instructionWords(const uint32_t *words, size_t available);
// Every word belongs to a valid instruction and the block ends with
// C_SYNC ALL (the old C_FENCE_ALL), which exposes the block's accesses.
Result<bool> verifyCoreWords(const std::vector<uint32_t> &words);

// System records. Dependencies are earlier record ordinals; they are written
// as 16-bit backward distances, two per word.
Result<std::vector<uint32_t>> gdmaRecord(bool load, uint32_t ordinal,
                                         const std::vector<uint32_t> &dependencies,
                                         uint64_t lp6Address, uint64_t l2Address,
                                         uint64_t rowBytes, uint64_t rows,
                                         uint64_t lp6Stride, uint64_t l2Stride);
Result<std::vector<uint32_t>> coreBeginRecord(uint32_t ordinal,
                                              const std::vector<uint32_t> &dependencies,
                                              uint64_t coreWords, unsigned logicalCore,
                                              uint64_t l1Bytes);
Result<std::vector<uint32_t>> barrierRecord(uint32_t ordinal,
                                            const std::vector<uint32_t> &dependencies);
// Record field limits.
inline constexpr uint32_t kMagic = 0x414e4c50, kVersion = 2u << 16;
inline constexpr uint32_t kHeaderWords = 5, OP_PROGRAM_END = 0x2f;
inline constexpr uint64_t kGdmaRowBytes = 1u << 20;   // row bytes < 2^20
inline constexpr uint64_t kGdmaRows = 1u << 16;       // rows - 1 < 2^16
inline constexpr uint64_t kGdmaL2 = 1u << 24;         // L2 byte address < 2^24
inline constexpr uint64_t kGdmaLp6 = uint64_t(1) << 36;
inline constexpr uint64_t kGdmaStride = 1u << 24;
inline constexpr uint64_t kCoreWords = 1u << 24;      // core words < 2^24
inline constexpr uint64_t kL1Units = 1u << 20;        // L1 bytes / 8 < 2^20

// PIM primitive record: W0 opcode 0x2e, dependencies[13:6], primitive count[31:14];
// Zero inline count selects a following 32-bit count word (>= 2^18), before
// dependencies. Then variable-length primitives. 0x27 is retired and
// rejected: old two-word PIM_GEMV packages must be recompiled.
inline constexpr uint32_t OP_PIM_PRIMITIVES = 0x2e;
inline constexpr uint64_t kPimCommands = 1u << 18;
Result<std::vector<uint32_t>> pimRecord(uint32_t ordinal, const std::vector<uint32_t> &deps,
                                       const pim::Primitive *commands, uint32_t count);
Result<std::vector<uint32_t>> pimRecord(uint32_t ordinal, const std::vector<uint32_t> &deps,
                                       const pim::PaPrimitive *commands, uint32_t count);

inline Result<bool> verifyCoreBlock(const uint32_t *words, size_t count) {
  if (!count) return verifyCoreWords({});
  if (!words) return Result<bool>::failure("missing core instruction words");
  return verifyCoreWords(std::vector<uint32_t>(words, words + count));
}
} // namespace npu::isa2
#endif
