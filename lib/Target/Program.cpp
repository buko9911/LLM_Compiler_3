#include "plena/Target/Program.h"
#include "plena/Target/ISA.h"
#include <algorithm>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <tuple>

namespace plena {
namespace {
class Frontier {
  struct State {
    std::optional<uint32_t> writer;
    std::set<uint32_t> readers;
    bool operator==(const State &other) const {
      return writer == other.writer && readers == other.readers;
    }
  };
  std::map<uint64_t, State> intervals{{0, {}}};
  auto split(uint64_t at) {
    auto it = intervals.lower_bound(at);
    if (it != intervals.end() && it->first == at) return it;
    return intervals.emplace_hint(it, at, std::prev(it)->second);
  }
public:
  void access(const Access &a, uint32_t id, std::set<uint32_t> &deps) {
    auto end = split(a.offset + a.bytes), begin = split(a.offset);
    for (auto it = begin; it != end; ++it) {
      auto &s = it->second;
      if (s.writer && *s.writer != id) deps.insert(*s.writer);
      if (a.write) {
        for (auto reader : s.readers) if (reader != id) deps.insert(reader);
        s.readers.clear(); s.writer = id;
      } else s.readers.insert(id);
    }
    // Adjacent equal states represent one interval. In particular a full-panel
    // overwrite must retire the thousands of boundaries made by array tiles.
    auto it = begin == intervals.begin() ? begin : std::prev(begin);
    while (it != end) {
      auto next = std::next(it);
      if (next == intervals.end()) break;
      bool last = next == end;
      if (it->second == next->second) intervals.erase(next);
      else it = next;
      if (last) break;
    }
  }
};
bool fits(uint64_t offset, uint64_t bytes, uint64_t capacity) {
  return bytes && offset <= capacity && bytes <= capacity - offset;
}
} // namespace

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

Result<Program> schedule(Program program) {
  using R = Result<Program>;
  if (!program.cores || program.cores > 65536 || program.commands.size() > UINT32_MAX)
    return R::failure("invalid program size or core count");
  Frontier dram, l2;
  std::vector<std::optional<uint32_t>> previous(program.cores);
  for (uint32_t id = 0; id < program.commands.size(); ++id) {
    auto &cmd = program.commands[id];
    std::set<uint32_t> deps(cmd.dependencies.begin(), cmd.dependencies.end());
    if (!deps.empty() && *deps.rbegin() >= id) return R::failure("dependency must reference an earlier command");
    std::vector<Access> accesses = cmd.accesses;
    if (cmd.kind == Command::Kind::Core) {
      if (cmd.core >= program.cores) return R::failure("core outside configured range");
      auto valid = verifyCore(cmd.words);
      if (!valid) return R::failure("command " + std::to_string(id) + ": " + valid.error);
      if (previous[cmd.core]) deps.insert(*previous[cmd.core]);
      previous[cmd.core] = id;
    } else {
      if (!cmd.words.empty() || !cmd.accesses.empty()) return R::failure("DMA command contains core payload");
      bool store = cmd.kind == Command::Kind::Store;
      if (!cmd.rows || !cmd.bytes) return R::failure("DMA command moves nothing");
      if (cmd.rows > 1 && (cmd.dramStride < cmd.bytes || cmd.l2Stride < cmd.bytes))
        return R::failure("DMA rows overlap: a stride is shorter than one row");
      accesses = {{MemorySpace::DRAM, cmd.dramOffset, cmd.bytes, store, cmd.rows, cmd.dramStride},
                  {MemorySpace::L2, cmd.l2Offset, cmd.bytes, !store, cmd.rows, cmd.l2Stride}};
    }
    for (const auto &a : accesses) {
      uint64_t cap = a.space == MemorySpace::DRAM ? program.dramBytes :
                     a.space == MemorySpace::L2 ? program.l2Bytes :
                     a.space == MemorySpace::L1 ? program.l1Bytes : 0;
      if (!a.rows || !fits(a.offset, a.bytes, cap) ||
          (a.rows > 1 && (a.stride < a.bytes ||
                         a.rows - 1 > (cap - a.offset - a.bytes) / a.stride))) return R::failure("memory access outside declared capacity");
    }
    // A core block is an atomic dependency node. Union its repeated/adjacent
    // rectangles before walking the frontier; array tiles often describe the
    // same activation thousands of times. Keep read/write sets separate.
    for (auto &a : accesses) {
      if (a.rows == 1 || a.stride == a.bytes) {
        a.bytes *= a.rows; a.rows = 1; a.stride = 0;
      }
    }
    std::sort(accesses.begin(),accesses.end(),[](const Access &a,const Access &b) {
      return std::tie(a.space,a.write,a.rows,a.stride,a.offset,a.bytes) <
             std::tie(b.space,b.write,b.rows,b.stride,b.offset,b.bytes);
    });
    size_t count = 0;
    for (const auto &a : accesses) {
      if (count) {
        auto &last = accesses[count-1];
        if (last.space == a.space && last.write == a.write && last.rows == a.rows &&
            last.stride == a.stride && a.offset <= last.offset + last.bytes) {
          auto width = std::max(last.bytes,a.offset-last.offset+a.bytes);
          if (last.rows == 1 || width <= last.stride) { last.bytes = width; continue; }
        }
      }
      accesses[count++] = a;
    }
    accesses.resize(count);
    for (const auto &a : accesses) {
      if (a.space == MemorySpace::L1) continue;
      const bool contiguous = a.rows == 1 || a.stride == a.bytes;
      const uint64_t rows = contiguous ? 1 : a.rows;
      for (uint64_t row = 0; row < rows; ++row) {
        Access part{a.space, a.offset + row * a.stride,
                    contiguous ? a.bytes * a.rows : a.bytes, a.write};
        if (a.space == MemorySpace::DRAM) dram.access(part, id, deps);
        else if (a.space == MemorySpace::L2) l2.access(part, id, deps);
      }
      // Private L1 accesses are covered by per-core order and the final fence.
    }
    cmd.dependencies.assign(deps.begin(), deps.end());
  }
  return R::success(std::move(program));
}

Result<bool> verifyProgramWords(const std::vector<uint32_t> &words) {
  using R = Result<bool>;
  if (words.size() < 6 || words[0] != kProgramMagic || words[1] != kProgramVersion || words[2] != words.size())
    return R::failure("invalid Unified Program (ISA ver 1.0) header");
  size_t at = 5;
  uint64_t instructions = 0;
  for (uint32_t id = 0; id < words[3]; ++id) {
    if (words.size() - at < 11) return R::failure("truncated command header");
    auto header = at;
    uint32_t op = words[at], encodedId = words[at+1];
    if (encodedId != id) return R::failure("non-canonical or duplicate command ID");
    uint32_t deps = 0;
    if (op == 0x26) deps = words[at+8];
    else if (op == 0x2b || op == 0x2c) {
      deps = words[at+2];
      // row_bytes, rows, and a stride per side. Rows that step by less than one
      // row would overlap, so the record would not describe a rectangle.
      uint32_t rowBytes = words[at+7], rows = words[at+8];
      uint32_t dramStride = words[at+9], l2Stride = words[at+10];
      if (!rowBytes || !rows) return R::failure("empty DMA extent in encoded program");
      if (rows == 1 ? (dramStride != rowBytes || l2Stride != rowBytes)
                    : (dramStride < rowBytes || l2Stride < rowBytes))
        return R::failure("unsupported DMA layout in encoded program");
    } else return R::failure("unknown program command");
    at += 11;
    if (deps > words.size() - at) return R::failure("truncated dependencies");
    std::set<uint32_t> unique;
    for (uint32_t i = 0; i < deps; ++i) {
      uint32_t d = words[at++];
      if (d >= id || !unique.insert(d).second) return R::failure("invalid dependency graph");
    }
    if (op != 0x26) continue;
    if (words[header+2] > 65535 || words[header+6] != 0 || words[header+9] != 0 || words[header+10] > 1)
      return R::failure("unsupported CORE layout");
    if (words[header+10]) {
      if (words.size() - at < 7) return R::failure("truncated L1 span");
      if (words[at] != 0 || words[at+1] != 0 || words[at+2] != words[header+5] ||
          words[at+3] != 0 || words[at+4] != 64 || words[at+5] != 0 || words[at+6] != UINT32_MAX)
        return R::failure("invalid L1 span");
      at += 7;
    } else if (words[header+5]) return R::failure("missing L1 span");
    uint32_t count = words[header+7];
    if (count > words.size() - at) return R::failure("truncated core instructions");
    auto valid = verifyCore(std::vector<uint32_t>(words.begin()+at,words.begin()+at+count));
    if (!valid) return valid;
    at += count; instructions += count;
    if (words.size() - at < 2 || words[at] != 0x27 || words[at+1] != id)
      return R::failure("invalid CORE_END");
    at += 2;
  }
  if (words.size() - at != 1 || words[at] != 0x2f || instructions != words[4])
    return R::failure("invalid program footer or instruction count");
  return R::success(true);
}

Result<std::vector<uint8_t>> encodeProgram(const Program &program) {
  using R = Result<std::vector<uint8_t>>;
  auto checked = schedule(program);
  if (!checked) return R::failure(checked.error);
  const auto &p = checked.value;
  std::vector<uint32_t> out(5);
  uint64_t coreWords = 0;
  for (uint32_t id = 0; id < p.commands.size(); ++id) {
    const auto &c = p.commands[id];
    if (c.words.size() > UINT32_MAX || c.dependencies.size() > UINT32_MAX)
      return R::failure("command exceeds binary field size");
    auto count = uint32_t(c.dependencies.size());
    if (c.kind == Command::Kind::Core) {
      out.insert(out.end(), {0x26, id, c.core, UINT32_MAX, UINT32_MAX, p.l1Bytes, 0,
                             uint32_t(c.words.size()), count, 0, p.l1Bytes ? 1u : 0u});
      out.insert(out.end(), c.dependencies.begin(), c.dependencies.end());
      if (p.l1Bytes) out.insert(out.end(), {0, 0, p.l1Bytes, 0, 64, 0, UINT32_MAX});
      out.insert(out.end(), c.words.begin(), c.words.end());
      out.insert(out.end(), {0x27, id});
      coreWords += c.words.size();
    } else {
      out.insert(out.end(), {c.kind == Command::Kind::Load ? 0x2bu : 0x2cu, id, count,
                             uint32_t(c.dramOffset), uint32_t(c.dramOffset >> 32),
                             c.l2Offset, 0, c.bytes, c.rows,
                             c.rows > 1 ? c.dramStride : c.bytes,
                             c.rows > 1 ? c.l2Stride : c.bytes});
      out.insert(out.end(), c.dependencies.begin(), c.dependencies.end());
    }
  }
  out.push_back(0x2f);
  if (out.size() > UINT32_MAX || coreWords > UINT32_MAX) return R::failure("program exceeds binary field size");
  out[0] = kProgramMagic; out[1] = kProgramVersion; out[2] = uint32_t(out.size());
  out[3] = uint32_t(p.commands.size()); out[4] = uint32_t(coreWords);
  auto verified = verifyProgramWords(out);
  if (!verified) return R::failure(verified.error);
  std::vector<uint8_t> bytes;
  bytes.reserve(out.size() * 4);
  for (auto w : out) for (unsigned b = 0; b < 4; ++b) bytes.push_back(uint8_t(w >> (8*b)));
  return R::success(std::move(bytes));
}
Result<std::string> systemManifest(const Program &p) {
  auto bytes = encodeProgram(p);
  if (!bytes) return Result<std::string>::failure(bytes.error);
  return Result<std::string>::success(systemManifestForEncoded(p, bytes.value.size() / 4));
}
std::string systemManifestForEncoded(const Program &p, uint64_t wordCount) {
  uint64_t instructions = 0;
  for (const auto &c : p.commands) instructions += c.words.size();
  std::string s = std::string("{\n  \"schema\": \"") + kProgramSchema + "\",\n  \"program\": \"program.bin\",\n  \"scheduling_policy\": \"compiler_static\",\n  \"logical_to_physical\": [";
  for (unsigned i = 0; i < p.cores; ++i) s += (i ? ", " : "") + std::to_string(i);
  return s + "],\n  \"program_word_count\": " + std::to_string(wordCount) +
    ",\n  \"command_count\": " + std::to_string(p.commands.size()) +
    ",\n  \"core_instruction_count\": " + std::to_string(instructions) +
    ",\n  \"l2_bytes_required\": " + std::to_string(p.l2Bytes) +
    ",\n  \"l2_regions\": [{\"name\": \"l2\", \"offset\": 0, \"size\": " +
    std::to_string(p.l2Bytes) + ", \"alignment\": 64}]\n}\n";
}
} // namespace plena
