#ifndef NPU_TARGET_PIMPRIMITIVE_H
#define NPU_TARGET_PIMPRIMITIVE_H

#include "npu/Target/Result.h"
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace npu::pim {
// Final, target-specific PIM IR. Tile IDs use the program's fixed memory
// layout; instruction order is selected by the compiler. STAGE operations
// are functional-memory initialization helpers, not executable instructions.
enum class Opcode : uint32_t {
  LoadActivation = 3, LoadScale = 4, ClearOutput = 5,
  GemvTile = 6, GemvStream = 7, StoreOutput = 8
};

struct Primitive {
  Opcode opcode = Opcode::ClearOutput;
  uint32_t partition = 0, bankSet = 0; // virtual bank set: 0 even, 1 odd
  int32_t bank = -1; // -1: all active PU banks
  int32_t weight = -1, activation = -1, scale = -1, accumulator = -1, output = -1;
  int32_t inputs = 0, outputs = 0;
  int32_t weightInputStride = 0, weightOutputStride = 0, activationInputStride = 0;
  int32_t scaleInputPeriod = 1, scaleInputStride = 0, scaleOutputStride = 0;
  int32_t accumulatorOutputStride = 0;
};

// Primitive ABI v1, independent of the NPU core ISA. W0: version[31:24],
// bank+1[23:18], bank set[17:16], partition[15:8], opcode[7:0]. Only the
// opcode's operands follow: loads/clear 2 words, store 3, tile 5, stream 14.
inline constexpr uint32_t kPrimitiveVersion = 1;
using EncodedPrimitive = std::vector<uint32_t>;
size_t instructionWords(uint32_t header);
Result<bool> verify(const Primitive &p);
Result<EncodedPrimitive> encode(const Primitive &p);
Result<Primitive> decode(const uint32_t *words, size_t count);
const char *opcodeName(Opcode opcode);
std::string print(const Primitive &p);
std::string printProgram(const std::vector<Primitive> &program);

// Address-bearing final IR, ABI v2. PA is a byte address, decoded by the
// common NPU/PIM controller mapping. GEMV inputs name PU register positions;
// LOAD/STORE and weight reads name memory by PA. v1 remains legacy-compatible.
struct PaPrimitive {
  Opcode opcode = Opcode::ClearOutput;
  uint64_t pa = 0;
  bool singleBank = false; // false: decoded bank-set broadcast, PU0 anchor
  uint32_t registerIndex = 0, activationPosition = 0;
  int32_t scalePosition = -1; // current v2 GEMV is scale-less W8A8
  uint32_t accumulator = 0, inputs = 1, outputs = 1;
  uint64_t weightInputByteStride = 0, weightOutputByteStride = 0;
  uint32_t accumulatorOutputStride = 1;
};
inline constexpr uint32_t kPaPrimitiveVersion = 2;
Result<bool> verify(const PaPrimitive &p);
Result<EncodedPrimitive> encode(const PaPrimitive &p);
Result<PaPrimitive> decodePa(const uint32_t *words, size_t count);
std::string print(const PaPrimitive &p);
} // namespace npu::pim
#endif
