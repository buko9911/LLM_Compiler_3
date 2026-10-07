#ifndef NPU_TARGET_ISAV2_H
#define NPU_TARGET_ISAV2_H
#include "npu/Target/Result.h"
#include "npu/Target/PimPrimitive.h"
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace npu::isa2 {
// PLENA ISA ver 2.0 (IESLAB_NPU_Simulator 75725ad): the core-block lowering,
// the system records and the PIM primitive extension. A port of the simulator's
// tools/isa_v2.py. Both graph and decode pipelines use this implementation.
//
// Lower writes core code as a builder stream in the ISA ver 1.0 encodings of
// ISA.h: every GP value is set with C_LUI/C_ADDI, C_SET reads a GP, vector
// memory addresses are whole GP values, a matrix operand is one four-word
// M_LOAD record and an MMA is one word per 32-element K slice. verifyCore
// checks that stream; lowerCoreBlock turns one core block of it into ver 2.0:
//  * a GP constant is materialized (C_LI, or C_LUI + C_ADDI above 22 bits) only
//    where an instruction reads it and no register holds it yet;
//  * V_LD/V_ST/S_LD/S_ST address base GP + scaled offset (1 MiB windows);
//  * C_SET takes an immediate, just before the first reader, when it changes;
//  * consecutive M_MMA slices of one rectangle become one M_MMA count;
//  * 32-byte-aligned RAW L2 <-> L1 transfers become LDMA.I, the rest LDMA.R;
//  * local LDMA events are renumbered 1..15 within the block.
// Inside a hardware loop every GP and control write stays in the output.

// Record field limits.
inline constexpr uint32_t kMagic = 0x414e4c50, kVersion = 2u << 16;
inline constexpr uint32_t kHeaderWords = 5, OP_PROGRAM_END = 0x2f;
inline constexpr uint32_t kCoreAny = 31;              // CORE_BEGIN logical core: dynamic placement
inline constexpr size_t kMaxDependencies = 255;       // per record; more are joined by a BARRIER
inline constexpr uint64_t kMaxDistance = 0xffff;      // 16-bit backward distance
inline constexpr uint64_t kGdmaRowBytes = 1u << 20;   // row bytes < 2^20
inline constexpr uint64_t kGdmaRows = 1u << 16;       // rows - 1 < 2^16
inline constexpr uint64_t kGdmaL2 = 1u << 24;         // L2 byte address < 2^24
inline constexpr uint64_t kGdmaLp6 = uint64_t(1) << 36;
inline constexpr uint64_t kGdmaStride = 1u << 24;
inline constexpr uint64_t kCoreWords = 1u << 24;      // core words < 2^24
inline constexpr uint64_t kL1Units = 1u << 20;        // L1 bytes / 8 < 2^20

Result<std::vector<uint32_t>> lowerCoreBlock(const std::vector<uint32_t> &builder);
// Words of the ver 2.0 core instruction that starts with `word` (1 or 2).
unsigned instructionWords(uint32_t word);
// Structure of a lowered core block: opcodes, instruction lengths, loops.
Result<bool> verifyCoreBlock(const uint32_t *words, size_t count);

// Dependencies are record ordinals, each written as a 16-bit backward
// distance, two per word. A record's completion event is its ordinal.
std::vector<uint32_t> gdmaRecord(bool load, uint32_t ordinal, const std::vector<uint32_t> &deps,
                                 uint64_t lp6, uint64_t l2, uint32_t rowBytes, uint32_t rows,
                                 uint32_t lp6Stride, uint32_t l2Stride);
std::vector<uint32_t> coreBeginRecord(uint32_t ordinal, const std::vector<uint32_t> &deps,
                                      uint32_t coreWords, uint32_t logicalCore, uint64_t l1Bytes);
std::vector<uint32_t> barrierRecord(uint32_t ordinal, const std::vector<uint32_t> &deps);
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
} // namespace npu::isa2
#endif
