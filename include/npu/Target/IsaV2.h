#ifndef NPU_TARGET_ISAV2_H
#define NPU_TARGET_ISAV2_H
#include "npu/Target/Result.h"
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace npu::isa2 {
// PLENA ISA ver 2.0 (IESLAB_NPU_Simulator 75725ad): the core-block lowering,
// the system records and the PIM_GEMV command. A port of the simulator's
// tools/isa_v2.py, the same as hybrid-compiler lib/Target/NPU/IsaV2.cpp.
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

// pim.bin command stream: "PIMC", command count, two words per command.
inline constexpr uint32_t kPimCommandMagic = 0x434d4950;
// PIM_GEMV (command type 0): one W8A8 GEMV stream on one partition. The PIM
// controller expands it to CLEAR x16, per 256-input sweep LOAD_ACT x32 +
// GEMV_STREAM, then STORE x8 (lp6-pim-main lp6_dramsim3_pim_load_commands).
// Tiles must be multiples of 512 (weight), 32 (activation) and 8 (output).
Result<std::pair<uint32_t, uint32_t>> pimGemv(unsigned partition, uint32_t sweeps, uint32_t weightTile,
                                              uint32_t activationTile, uint32_t outputTile);
} // namespace npu::isa2
#endif
