#include "ProgramVerifier.h"
#include "Assembler.h"
#include "npu/Target/ISAv2.h"
#include <set>

namespace npu {
namespace {
// What one core instruction stream carries across an instruction boundary.
//
// ISA ver 1.0: M_LOAD puts W [K,N] and A [M,K] rectangles into the single
// operand buffer. Each one-word M_MMA consumes the next kMmaTileK K elements;
// the slice that reaches K releases both operands. INIT needs an empty
// accumulator, ACC continues a live one. Mirrors src/accelerator/dispatch.rs.
struct CoreState {
  bool live = false, weight = false, act = false;
  unsigned mode = 0, weightMode = 0, actMode = 0;
  uint32_t weightK = 0, actK = 0, consumedK = 0;
  bool operator==(const CoreState &o) const {
    return live == o.live && weight == o.weight && act == o.act &&
           (!live || mode == o.mode) &&
           (!weight || (weightMode == o.weightMode && weightK == o.weightK)) &&
           (!act || (actMode == o.actMode && actK == o.actK)) && consumedK == o.consumedK;
  }
};
// Steps over whole records: a matrix payload word is data, never an opcode.
Result<size_t> matchingLoopEnd(const std::vector<uint32_t> &words, size_t begin) {
  unsigned depth = 0;
  for (size_t i = begin; i < words.size(); i += recordWords(words[i])) {
    unsigned op = words[i] & 63;
    if (op == 0x23) ++depth;
    else if (op == 0x24 && --depth == 0) return Result<size_t>::success(i);
  }
  return Result<size_t>::failure("C_LOOP_BEGIN without a matching C_LOOP_END");
}
// A loop body runs at least once (the hardware rejects a zero trip count) and
// every iteration starts where the previous one stopped. Verifying the body from
// its entry state and then from the state that produces reaches a fixed point in
// two passes; without one, some iteration would see a state no pass checked.
Result<CoreState> runRange(const std::vector<uint32_t> &words, size_t begin, size_t end,
                           CoreState state, unsigned depth) {
  using R = Result<CoreState>;
  for (size_t i = begin; i < end; i += recordWords(words[i])) {
    uint32_t word = words[i];
    if (!disassembleRecord(words.data() + i, end - i)) return R::failure("invalid core instruction");
    unsigned op = word & 63, fn = (word >> 22) & 15;
    if (op == 0x24) return R::failure("C_LOOP_END without a matching C_LOOP_BEGIN");
    if (op == 0x23) {
      if (depth >= 4) return R::failure("hardware loop nesting exceeds four");
      auto close = matchingLoopEnd(words, i);
      if (!close) return R::failure(close.error);
      if (close.value == i + 1) return R::failure("empty hardware loop body");
      auto first = runRange(words, i + 1, close.value, state, depth + 1);
      if (!first) return R::failure(first.error);
      auto second = runRange(words, i + 1, close.value, first.value, depth + 1);
      if (!second) return R::failure(second.error);
      if (!(second.value == first.value))
        return R::failure("hardware loop body does not reach a fixed point; "
                          "its accumulator or operand state differs between iterations");
      state = first.value;
      i = close.value;
      continue;
    }
    if (op == 0x37) {
      // Weight funct 1/3 = I8/F16 over [K,N]; activation 5/7 over [M,K].
      if (fn <= 3) {
        if (state.weight) return R::failure("operand buffer already holds an unconsumed weight");
        state.weight = true; state.weightMode = fn; state.weightK = words[i + 1];
      } else {
        if (state.act) return R::failure("operand buffer already holds an unconsumed activation");
        state.act = true; state.actMode = fn - 4; state.actK = words[i + 2];
      }
    }
    if (op == 0x3b) {
      const bool accumulate = word >> 26 & 1;
      if (!state.weight || !state.act || state.weightMode != fn || state.actMode != fn)
        return R::failure("MMA requires matching weight and activation in the operand buffer");
      if (state.weightK != state.actK) return R::failure("weight and activation K differ");
      if (!accumulate && state.live) return R::failure("MMA INIT with a live accumulator");
      if (accumulate && (!state.live || state.mode != fn))
        return R::failure("MMA ACC without a live accumulator of the same mode");
      state.live = true; state.mode = fn;
      state.consumedK += kMmaTileK;
      if (state.consumedK >= state.weightK) {
        state.weight = state.act = false; state.consumedK = 0;
      }
    }
    if (op == 0x3c) {
      if (!state.live) return R::failure("writeout without live accumulator");
      if ((state.mode == 1) != (fn == 2)) return R::failure("writeout dtype does not match accumulator");
      state.live = false;
    }
  }
  return R::success(state);
}
} // namespace

Result<bool> verifyCore(const std::vector<uint32_t> &words) {
  using R = Result<bool>;
  if (words.empty()) return R::failure("empty core block");
  auto state = runRange(words, 0, words.size(), CoreState{}, 0);
  if (!state) return R::failure(state.error);
  if (state.value.live) return R::failure("live accumulator crosses CORE boundary");
  if (state.value.weight || state.value.act) return R::failure("operand load crosses CORE boundary");
  if (words.back() != (0x3fu | 4u << 22))
    return R::failure("core block must end with C_FENCE_ALL to expose completed accesses");
  return R::success(true);
}

namespace {
namespace isa2 = ::npu::isa2;
// Dependency list of the record `ordinal` at words[at]: two 16-bit backward
// distances per word, low half first; the unused half of the last word is 0.
Result<bool> readDependencies(const std::vector<uint32_t> &words, size_t &at, uint32_t ordinal, uint32_t count) {
  using R = Result<bool>;
  if ((uint64_t(count) + 1) / 2 > words.size() - at) return R::failure("truncated dependencies");
  std::set<uint32_t> unique;
  for (uint32_t i = 0; i < count; ++i) {
    uint32_t distance = words[at + i / 2] >> (16 * (i % 2)) & 0xffff;
    if (!distance || distance > ordinal || !unique.insert(ordinal - distance).second)
      return R::failure("invalid dependency graph");
  }
  if (count % 2 && words[at + count / 2] >> 16) return R::failure("unused dependency half must be 0");
  at += (size_t(count) + 1) / 2;
  return R::success(true);
}

} // namespace

Result<bool> verifyProgramWords(const std::vector<uint32_t> &words) {
  using R = Result<bool>;
  if (words.size() < 6 || words[0] != kProgramMagic || words[1] != kProgramVersion || words[2] != words.size())
    return R::failure("invalid Unified Program (ISA ver 2.0) header");
  size_t at = 5;
  uint64_t coreWords = 0;
  for (uint32_t id = 0; id < words[3]; ++id) {
    if (at >= words.size()) return R::failure("truncated command record");
    const uint32_t head = words[at];
    Result<bool> deps = R::success(true);
    switch (head & 63) {
    case 0x2d:   // BARRIER
      if (head >> 14) return R::failure("reserved BARRIER bits set");
      ++at;
      deps = readDependencies(words, at, id, head >> 6 & 0xff);
      break;
    case 0x2b:   // GDMA_LOAD
    case 0x2c: { // GDMA_STORE
      if (words.size() - at < 3) return R::failure("truncated GDMA record");
      if (words[at + 2] >> 31) return R::failure("reserved GDMA bit set");
      // Row bytes are record bits [34:15]; bits [31:15] of word 0, [2:0] of word 1.
      const uint32_t rowBytes = (head >> 15 | words[at + 1] << 17) & 0xfffff;
      if (!rowBytes) return R::failure("empty DMA extent in encoded program");
      at += 3;
      if (head >> 6 & 1) {
        if (words.size() - at < 2) return R::failure("truncated GDMA 2-D extension");
        const uint64_t e = words[at] | uint64_t(words[at + 1]) << 32;
        const uint64_t rows = (e & 0xffff) + 1, lp6Stride = e >> 16 & 0xffffff, l2Stride = e >> 40;
        // A 2-D record describes a rectangle of several rows.
        if (rows == 1 || lp6Stride < rowBytes || l2Stride < rowBytes)
          return R::failure("unsupported DMA layout in encoded program");
        at += 2;
      }
      deps = readDependencies(words, at, id, head >> 7 & 0xff);
      break;
    }
    case 0x26: { // CORE_BEGIN
      if (words.size() - at < 2) return R::failure("truncated CORE_BEGIN");
      if (head >> 6 & 1) return R::failure("unsupported CORE_BEGIN extension");
      if ((head >> 7 & 31) == isa2::kCoreAny) return R::failure("CORE_BEGIN without a logical core");
      const uint32_t count = words[at + 1] & 0xffffff;
      at += 2;
      deps = readDependencies(words, at, id, words[at - 1] >> 24);
      if (!deps) return deps;
      if (count > words.size() - at) return R::failure("truncated core instructions");
      if (auto body = isa2::verifyCoreBlock(words.data() + at, count); !body)
        return R::failure("command " + std::to_string(id) + ": " + body.error);
      at += count;
      coreWords += count;
      break;
    }
    case 0x27:
      return R::failure("retired PIM_GEMV record; recompile with the PIM primitive backend");
    case isa2::OP_PIM_PRIMITIVES: {
      uint32_t count = head >> 14;
      ++at;
      if (!count) {
        if (at >= words.size()) return R::failure("truncated PIM extended count");
        count = words[at++];
        if (count < isa2::kPimCommands) return R::failure("noncanonical PIM extended count");
      }
      deps = readDependencies(words, at, id, head >> 6 & 0xff);
      if (!deps) return deps;
      for (uint32_t i = 0; i < count; ++i) {
        if (at >= words.size()) return R::failure("truncated PIM record");
        size_t n = pim::instructionWords(words[at]);
        if (!n || n > words.size() - at) return R::failure("invalid PIM primitive header or truncated operands");
        if (words[at] >> 24 == pim::kPaPrimitiveVersion) {
          auto p = pim::decodePa(words.data() + at, n);
          if (!p) return R::failure("invalid PA PIM primitive: " + p.error);
        } else {
          auto p = pim::decode(words.data() + at, n);
          if (!p) return R::failure("invalid PIM primitive: " + p.error);
        }
        at += n;
      }
      break;
    }
    default:
      return R::failure("unknown program command");
    }
    if (!deps) return deps;
  }
  if (words.size() - at != 1 || words[at] != 0x2f || coreWords != words[4])
    return R::failure("invalid program footer or core instruction count");
  return R::success(true);
}
} // namespace npu
