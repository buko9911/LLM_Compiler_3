#ifndef NPU_TARGET_PROGRAM_H
#define NPU_TARGET_PROGRAM_H
#include "npu/Target/Result.h"
#include "npu/Target/PimPrimitive.h"
#include <cstdint>
#include <string>
#include <vector>

namespace npu {
// Unified program container of the PLENA simulator, "ISA ver 2.0". Core
// blocks are built and checked as builder-stream words (ISA.h) and lowered to
// ver 2.0 one block at a time when the program is encoded (ISAv2.h).
inline constexpr uint32_t kProgramMagic = 0x414e4c50;   // little-endian "PLNA"
inline constexpr uint32_t kProgramVersion = 0x00020000; // major << 16 | minor
inline constexpr const char *kProgramSchema = "plena.v2.unified_program.isa_v2.0";
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
// Schedules, lowers every core block and writes the ver 2.0 records. A record
// waits for at most 255 events, so BARRIER records join longer dependency
// lists, and a DMA whose rows do not fit a GDMA record becomes several
// records; the header command count includes both.
struct EncodedProgram {
  std::vector<uint32_t> words;
  uint32_t commands = 0;
  uint64_t coreWords = 0;
};
Result<EncodedProgram> encodeProgramWords(const Program &program);
std::string systemManifestForEncoded(const Program &program, const EncodedProgram &encoded);
Result<std::vector<uint8_t>> encodeProgram(const Program &program);
Result<std::string> systemManifest(const Program &program);
// Build metadata for an already successfully encoded program, without scheduling again.
std::string systemManifestForEncoded(const Program &program, const std::vector<uint8_t> &encoded);
// Builder-stream core block (ISA.h encodings): matrix operand and accumulator
// lifetimes, loops, the closing C_FENCE_ALL.
Result<bool> verifyCore(const std::vector<uint32_t> &words);
// Encoded ver 2.0 program: header, records, dependencies, core block structure.
Result<bool> verifyProgramWords(const std::vector<uint32_t> &words);

// One decode step as one program. A segment is an NPU phase (an
// encoded program) or a PIM phase (final executable primitives), in run
// order. The records of an NPU phase keep their order and dependencies; the
// ones without dependencies wait for the previous segment, and a BARRIER joins
// the phase's last records. A PIM phase is one PIM record after the previous
// segment.
struct Segment {
  std::string name;
  std::vector<uint32_t> program;   // NPU phase: encodeProgram words
  std::vector<pim::Primitive> pim; // PIM phase: compiler-lowered target IR
  std::vector<pim::PaPrimitive> paPim; // PA ABI v2; mutually exclusive with pim/program
};
struct LinkedProgram {
  std::vector<uint32_t> words;
  // Per record: <segment>.<core_block|gdma|barrier>_<ordinal>; a PIM record is
  // named after its segment.
  std::vector<std::string> symbols;
};
Result<LinkedProgram> linkProgram(const std::vector<Segment> &segments);
} // namespace npu
#endif
