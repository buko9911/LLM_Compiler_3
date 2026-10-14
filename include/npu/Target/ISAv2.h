#ifndef NPU_TARGET_ISAV2_H
#define NPU_TARGET_ISAV2_H
#include "npu/Target/Result.h"
#include <cstddef>
#include <cstdint>
#include <vector>

// PLENA ISA ver 2.0, the format NPU_Simulator main executes.
//
// The compiler keeps emitting core code as a *builder stream* in the ISA ver 1.0
// encodings (npu.instruction / npu.matrix_* in 05-target.mlir; see ISA.h).
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
// do not change.
namespace npu::isa2 {
inline constexpr uint32_t kProgramVersion = 0x00020000;
inline constexpr const char *kProgramSchema = "plena.v2.unified_program.isa_v2.0";
inline constexpr uint32_t kCoreBegin = 0x26, kGdmaLoad = 0x2b, kGdmaStore = 0x2c,
                          kBarrier = 0x2d, kProgramEnd = 0x2f;
inline constexpr unsigned kCoreAny = 31;       // CORE_BEGIN logical core: no fixed core
inline constexpr unsigned kMaxDependencies = 255;
inline constexpr uint32_t kMaxDistance = 0xffff;
inline constexpr uint64_t kMaxGdmaRowBytes = uint64_t(1) << 20;  // exclusive

// One builder-stream (ISA ver 1.0) core block -> ISA ver 2.0 core words.
Result<std::vector<uint32_t>> lowerCoreBlock(const std::vector<uint32_t> &builderWords);

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
} // namespace npu::isa2
#endif
