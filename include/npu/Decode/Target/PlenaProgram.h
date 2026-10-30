//===- PlenaProgram.h - PLENA unified program builder ------------*- C++ -*-===//
//
// C++ port of IESLAB_NPU_Simulator/transactional_emulator/tools/v2_system_program.py
// (SystemProgramBuilder + encode_unified_program): system records (GDMA load/store,
// barriers, PIM records) and core blocks with numeric completion-event dependencies in one
// 32-bit word stream. A core block starts when its dependencies complete; CORE_BEGIN resets
// the GP and control registers, so the builder replays their tracked values at every new block.
// Core code is written as a builder stream (the encodings below); finish() lowers every core
// block to ISA ver 2.0 (ISAv2.h) and writes ISA ver 2.0 system records.
//
// One decode step is one program: every NPU phase is a stretch of records between
// beginPhase() and endPhase() (a BARRIER that joins the phase), and every PIM phase is one PIM
// record. Each record depends, directly or through earlier records of its phase, on the exit of
// the previous phase, so phases run in program order as the old host schedule did.
//===----------------------------------------------------------------------===//
#ifndef HYBRID_TARGET_PLENAPROGRAM_H
#define HYBRID_TARGET_PLENAPROGRAM_H

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>
#include "npu/Target/PimPrimitive.h"

namespace mlir::hybrid::plena {

class ProgramBuilder {
public:
  enum class Kind { CoreBlock, GdmaLoad, GdmaStore, Barrier, Pim };
  struct Command {
    Kind kind;
    std::vector<int> deps;
    int logicalCore = 0;
    int64_t isaStart = 0, isaCount = 0;
    uint64_t lp6 = 0, l2 = 0;
    uint32_t rowBytes = 0, rows = 1, lp6Stride = 0, l2Stride = 0;
    int64_t pimFirst = 0, pimCount = 0;   // Kind::Pim: indices into pimPrimitives
    int phase = 0;                        // index into phaseNames (symbol prefix)
    int source = -1;                      // optional diagnostic origin, never encoded
  };

  int logicalCore = 0;
  std::vector<int> frontier;
  // Sidecar-only source attribution. It must never change instruction emission.
  bool traceOperations = false;
  int currentSource = -1;
  /// Optional compiler admission window over each core's phase-local GDMA commands.
  /// Zero preserves unrestricted issue. Set before beginPhase; loads and stores
  /// share a core's window. Aggregate admission is bounded by N * logical cores;
  /// histories are separate because core streams are emitted sequentially.
  uint32_t gdmaLookahead = 0;
  bool commandSchedule = false;
  struct ScheduleStats {
    uint64_t commands = 0, reordered = 0, redundantEdges = 0;
    bool encodingFallback = false;
  };
  ScheduleStats lastSchedule;
  std::vector<std::string> sourceOperations;

  /// Starts a phase: fresh GP / control state, and every record of the phase follows `entry`.
  void beginPhase(const std::string &name, std::vector<int> entry);
  /// Ends the phase: a BARRIER (chained when it joins more than 255 records) after every record
  /// of the phase; returns its ordinal, or the phase entry when the phase is empty.
  std::vector<int> endPhase();
  /// A PIM phase: compiler-lowered primitives, ordered after `deps`.
  int pimRecord(const std::string &name, std::vector<int> deps, const std::vector<::npu::pim::Primitive> &commands);
  /// PIM kernel configuration written to system.json when the program has PIM records.
  std::string pimKernelJson;

  /// Appends one core word (not a 4-word record) and tracks GP / control state.
  void append(uint32_t word);
  /// Appends a 4-word record (typed vector, matrix).
  void appendRecord(const std::array<uint32_t, 4> &words);
  /// Closes the current core block; returns its command index or -1 when empty.
  int flushBlock(bool updateFrontier = true);
  int gdmaLoad(uint64_t lp6, uint64_t l2, uint32_t bytes, uint32_t rows = 1, uint32_t lp6Stride = 0,
               uint32_t l2Stride = 0, bool overlapWithCurrentBlock = false);
  int gdmaStore(uint64_t lp6, uint64_t l2, uint32_t bytes, uint32_t rows = 1, uint32_t lp6Stride = 0,
                uint32_t l2Stride = 0, bool allowFollowingOverlap = false);
  /// Raw GDMA with explicit dependencies (frontier untouched). `admitted` GDMAs count against
  /// gdmaLookahead; the others are neither held back nor hold later ones back.
  int appendGdma(Kind kind, std::vector<int> deps, uint64_t lp6, uint64_t l2, uint32_t bytes, uint32_t rows,
                 uint32_t lp6Stride, uint32_t l2Stride, bool admitted = true);
  void reserveL2(uint64_t addr, uint64_t size) { l2HighWater = std::max(l2HighWater, addr + size); }
  /// Writes <dir>/program.bin and <dir>/system.json (with pim_kernel when there are PIM records);
  /// returns false on I/O error or a record that does not fit ISA ver 2.0.
  bool finish(const std::string &dir, int logicalCores);

  int64_t coreWords() const { return int64_t(isa.size() + cur.size()); }
  int numCommands() const { return int(cmds.size()); }
  /// The records every record of the current phase follows (the previous phase's exit).
  const std::vector<int> &phaseEntry() const { return base; }
  /// A command index of the phase that endPhase just closed, after its command_schedule reorder.
  int scheduledId(int id) const {
    return id >= int(lastPhaseStart) && size_t(id) - lastPhaseStart < lastRemap.size()
        ? lastRemap[size_t(id) - lastPhaseStart] : id;
  }
  uint32_t gpValue(unsigned reg) const { return gp.at(reg); }

private:
  void ensureSegment();
  void track(uint32_t word);
  int push(Command c);
  void schedulePhase();

  std::vector<Command> cmds;
  std::vector<uint32_t> isa, cur;
  std::vector<int> isaSources, curSources;
  std::vector<::npu::pim::Primitive> pimPrimitives;
  std::optional<std::vector<int>> segDeps;
  std::array<uint32_t, 16> gp{};
  std::map<int, uint32_t> control;
  bool replay = false;
  uint64_t l2HighWater = 0;
  std::vector<std::string> phaseNames{"main"};
  int phase = 0;
  size_t phaseStart = 0;
  std::map<int, std::vector<int>> phaseGdmas; // logical core -> DMA admission history
  std::vector<int> base;                  // the current phase's entry
  size_t lastPhaseStart = 0;              // the closed phase's first command and its reorder
  std::vector<int> lastRemap;             // (old index - lastPhaseStart) -> new index
};

// ---- builder-stream encodings (lowered to ISA ver 2.0 by ISAv2.h) ------------------------------------------------------
inline uint32_t rform(uint32_t opcode, uint32_t rd = 0, uint32_t rs1 = 0, uint32_t rs2 = 0, uint32_t rs3 = 0,
                      uint32_t funct = 0) {
  return opcode | rd << 6 | rs1 << 10 | rs2 << 14 | rs3 << 18 | funct << 22;
}
enum Op : uint32_t {
  OP_ADDI = 0x22, OP_LOOP_BEGIN = 0x23, OP_LOOP_END = 0x24, OP_LUI = 0x25, OP_WAIT_EVENT = 0x2e,
  OP_V_BIN = 0x30, OP_V_UN = 0x31,
  OP_V_SCALAR = 0x32, OP_V_RED = 0x33, OP_S_ALU = 0x34, OP_V_MEM = 0x35, OP_V_TYPED = 0x36, OP_M_LOAD = 0x37,
  OP_LDMA = 0x38, OP_C_SET = 0x39, OP_M_LOAD_ACC = 0x3a, OP_M_MMA = 0x3b, OP_M_WRITEOUT = 0x3c,
  OP_V_CONV = 0x3d, OP_C_WAIT = 0x3f
};
enum TypedOp : uint32_t {
  T_ADD = 0, T_SUB = 1, T_MUL = 2, T_MAX = 4, T_MIN = 5, T_NEG = 6, T_ABS = 7,
  T_EXP = 8, T_RCP = 9, T_SQRT = 10, T_RSQRT = 11,
  T_CMP_EQ = 13, T_CMP_LT = 14, T_CMP_LE = 15, T_SELECT = 16, T_MOV = 17, T_BCAST = 18, T_SLICE = 19,
  T_AND = 21, T_SHL = 24, T_SHR_L = 25, T_SHR = 26, T_CAST_F16 = 27, T_CAST_F32 = 28, T_CAST_I8 = 29,
  T_CAST_I32 = 30, T_SPLAT = 31, T_BITCAST = 40
};
enum DType : uint32_t { DT_F16 = 0, DT_F32 = 1, DT_I8 = 2, DT_I32 = 3 };
inline std::array<uint32_t, 4> typedVector(uint32_t op, uint32_t dt, uint32_t rd, uint32_t rs1, uint32_t rs2,
                                           uint32_t rs3, uint32_t elements) {
  return {OP_V_TYPED | rd << 6 | rs1 << 10 | rs2 << 14 | rs3 << 18, op, dt, elements};
}
inline std::array<uint32_t, 4> matrixLoad(uint32_t gp, uint32_t rows, uint32_t cols, uint32_t strideBytes,
                                          uint32_t funct) {
  return {rform(OP_M_LOAD, 0, gp, 0, 0, funct), rows, cols, strideBytes};
}
// Builder stream: one M_MMA is a fixed 32x32x32 tile. It consumes the next kMmaTileK elements of the
// loaded W [K,N] / A [M,K] rectangles (the simulator's mma_tile_k); bit 26 selects INIT (0) or ACC (1).
constexpr uint32_t kMmaTileK = 32;
inline uint32_t matrixMmaTile(uint32_t funct, bool accumulate) {
  return rform(OP_M_MMA, 0, 0, 0, 0, funct) | uint32_t(accumulate) << 26;
}
/// The one-word M_MMA sequence that covers one loaded pair of K-long rectangles.
inline std::vector<uint32_t> matrixMma(uint32_t k, uint32_t funct, bool accumulate) {
  std::vector<uint32_t> words{matrixMmaTile(funct, accumulate)};
  for (uint32_t done = kMmaTileK; done < k; done += kMmaTileK) words.push_back(matrixMmaTile(funct, true));
  return words;
}
inline std::array<uint32_t, 4> matrixWriteout(uint32_t gp, uint32_t m, uint32_t n, uint32_t strideBytes,
                                              uint32_t funct) {
  return {rform(OP_M_WRITEOUT, gp, 0, 0, 0, funct), m, n, strideBytes};
}
/// Restores a saved accumulator. funct=2 is INT32; funct=1 is FP32.
/// The existing ISA-v2 lowering handles the four-word builder record.
inline std::array<uint32_t, 4> matrixLoadAccumulator(uint32_t gp, uint32_t m, uint32_t n,
                                                   uint32_t strideBytes, bool integer) {
  return {rform(OP_M_LOAD_ACC, 0, gp, 0, 0, integer ? 2 : 1), m, n, strideBytes};
}

} // namespace mlir::hybrid::plena

#endif
