#ifndef NPU_TARGET_PROGRAM_H
#define NPU_TARGET_PROGRAM_H
#include "npu/Target/ISAv2.h"
#include "npu/Target/Result.h"
#include <cstdint>
#include <string>
#include <vector>

namespace npu {
// Unified program container of NPU_Simulator main, "ISA ver 2.0". Core blocks
// are built in the ISA ver 1.0 builder stream (ISA.h) and lowered on encoding.
inline constexpr uint32_t kProgramMagic = 0x414e4c50;   // little-endian "PLNA"
inline constexpr uint32_t kProgramVersion = isa2::kProgramVersion;
inline constexpr const char *kProgramSchema = isa2::kProgramSchema;
enum class MemorySpace : unsigned { DRAM = 0, L2 = 1, L1 = 2, Accumulator = 3 };
struct Access {
  MemorySpace space;
  uint64_t offset, bytes;
  bool write;
  uint64_t rows = 1, stride = 0; // bytes is the width of each accessed row
};
struct Command {
  enum class Kind { Core, Load, Store };
  Kind kind = Kind::Core;
  unsigned core = 0;
  uint64_t dramOffset = 0;
  // One command moves a rectangle: `bytes` per row, `rows` of them, each side
  // advancing by its own stride. The hardware descriptor has carried this all
  // along; a row per command burned an event slot for every row of a tile.
  uint32_t l2Offset = 0, bytes = 0, rows = 1, dramStride = 0, l2Stride = 0;
  std::vector<uint32_t> words;
  // Core access summaries must include asynchronous operations through completion.
  std::vector<Access> accesses;
  std::vector<uint32_t> dependencies;
};
struct Program {
  unsigned cores = 1;
  uint32_t l1Bytes = 0, l2Bytes = 0;
  uint64_t dramBytes = 0;
  std::vector<Command> commands;
};
// Adds per-core order and overlapping RAW/WAR/WAW edges. Disjoint DMA is free
// to overlap computation. Dependencies reference command indices.
Result<Program> schedule(Program program);
struct EncodedProgram {
  std::vector<uint32_t> words;
  uint32_t commands = 0;   // system records, including joining BARRIERs
  uint64_t coreWords = 0;  // ISA ver 2.0 core instruction words
};
Result<EncodedProgram> encodeProgramWords(const Program &program);
Result<std::vector<uint8_t>> encodeProgram(const Program &program);
Result<std::string> systemManifest(const Program &program);
// Build metadata for an already successfully encoded program, without scheduling again.
std::string systemManifestForEncoded(const Program &program, const EncodedProgram &encoded);
// Checks one core block in the builder stream (operand buffer, accumulator, loops).
Result<bool> verifyCore(const std::vector<uint32_t> &words);
Result<bool> verifyProgramWords(const std::vector<uint32_t> &words);
} // namespace npu
#endif
