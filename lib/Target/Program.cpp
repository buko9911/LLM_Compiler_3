#include "npu/Target/Program.h"
#include "npu/Target/ISA.h"
#include "npu/Target/ISAv2.h"
#include <algorithm>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <tuple>

namespace npu {
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

// One ver 2.0 GDMA record: `rows` rows of `rowBytes`, each side at its stride.
struct Transfer { uint64_t lp6, l2, rowBytes, rows, lp6Stride, l2Stride; };

// The smallest divisor of n in [from, 2^16], or 0.
uint64_t divisorFrom(uint64_t n, uint64_t from) {
  for (uint64_t d = std::max<uint64_t>(from, 1); d <= isa2::kGdmaRows && d <= n; ++d)
    if (n % d == 0) return d;
  return 0;
}

// A DMA command as GDMA records. The command's own shape is kept whenever the
// fields hold it. A row of 1 MiB or more does not fit the 20-bit row size: a
// contiguous transfer is cut into the fewest equal rows, 64-byte multiples when
// possible (the simulator then issues the same LP6 blocks in the same order),
// and a strided one is split into column pieces.
Result<std::vector<Transfer>> gdmaTransfers(const Command &c) {
  using R = Result<std::vector<Transfer>>;
  const Transfer t{c.dramOffset, c.l2Offset, c.bytes, c.rows, c.rows > 1 ? c.dramStride : c.bytes,
                   c.rows > 1 ? c.l2Stride : c.bytes};
  std::vector<Transfer> pieces;
  if (t.rowBytes < isa2::kGdmaRowBytes) {
    pieces.push_back(t);
  } else {
    const bool contiguous = t.rows == 1 || (t.lp6Stride == t.rowBytes && t.l2Stride == t.rowBytes);
    const uint64_t total = t.rowBytes * t.rows;
    const uint64_t fewest = (total + isa2::kGdmaRowBytes - 2) / (isa2::kGdmaRowBytes - 1);
    uint64_t rows = 0;
    if (contiguous && total % 64 == 0) rows = divisorFrom(total / 64, fewest);
    if (contiguous && !rows) rows = divisorFrom(total, fewest);
    if (rows) {
      pieces.push_back({t.lp6, t.l2, total / rows, rows, total / rows, total / rows});
    } else {
      constexpr uint64_t kPiece = isa2::kGdmaRowBytes / 2;
      for (uint64_t at = 0; at < t.rowBytes; at += kPiece)
        pieces.push_back({t.lp6 + at, t.l2 + at, std::min(kPiece, t.rowBytes - at), t.rows, t.lp6Stride,
                          t.l2Stride});
    }
  }
  std::vector<Transfer> out;
  for (const auto &piece : pieces)
    for (uint64_t first = 0; first < piece.rows; first += isa2::kGdmaRows) {   // rows - 1 < 2^16
      Transfer q = piece;
      q.rows = std::min(isa2::kGdmaRows, piece.rows - first);
      q.lp6 += first * piece.lp6Stride;
      q.l2 += first * piece.l2Stride;
      if (q.rows == 1) q.lp6Stride = q.l2Stride = q.rowBytes;
      if (q.lp6 >= isa2::kGdmaLp6 || q.l2 >= isa2::kGdmaL2 || q.lp6Stride >= isa2::kGdmaStride ||
          q.l2Stride >= isa2::kGdmaStride)
        return R::failure("DMA does not fit a GDMA record (LP6 address < 2^36, L2 address and strides < 2^24)");
      out.push_back(q);
    }
  return R::success(std::move(out));
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

namespace {
// One record of a verified program: the words before its dependencies, its
// dependencies as ordinals of that program, and the words after them.
struct Record {
  unsigned op;
  std::vector<uint32_t> head, deps, tail;
};
std::vector<Record> readRecords(const std::vector<uint32_t> &words) {
  std::vector<Record> records;
  size_t at = 5;
  for (uint32_t id = 0; id < words[3]; ++id) {
    const uint32_t head = words[at];
    Record r{head & 63, {}, {}, {}};
    size_t headWords = 1, tailWords = 0;
    uint32_t count = 0;
    switch (r.op) {
    case 0x2b: case 0x2c: headWords = head >> 6 & 1 ? 5 : 3; count = head >> 7 & 0xff; break;
    case 0x26: headWords = 2; count = words[at + 1] >> 24; tailWords = words[at + 1] & 0xffffff; break;
    case isa2::OP_PIM_PRIMITIVES: {
      count = head >> 6 & 0xff;
      uint32_t primitives = head >> 14;
      if (!primitives) { headWords = 2; primitives = words[at + 1]; }
      size_t body = at + headWords + (count + 1) / 2;
      for (uint32_t i = 0; i < primitives; ++i)
        tailWords += pim::instructionWords(words[body + tailWords]);
      break;
    }
    default: count = head >> 6 & 0xff; break;   // BARRIER
    }
    r.head.assign(words.begin() + at, words.begin() + at + headWords);
    at += headWords;
    for (uint32_t i = 0; i < count; ++i) r.deps.push_back(id - (words[at + i / 2] >> (16 * (i % 2)) & 0xffff));
    at += (size_t(count) + 1) / 2;
    r.tail.assign(words.begin() + at, words.begin() + at + tailWords);
    at += tailWords;
    records.push_back(std::move(r));
  }
  return records;
}
// Writes `r` as record `ordinal`: its dependency count and backward distances change.
void writeRecord(std::vector<uint32_t> &out, Record r, uint32_t ordinal) {
  const uint32_t n = uint32_t(r.deps.size());
  if (r.op == 0x2b || r.op == 0x2c) r.head[0] = (r.head[0] & ~(0xffu << 7)) | n << 7;
  else if (r.op == 0x26) r.head[1] = (r.head[1] & 0xffffff) | n << 24;
  else r.head[0] = (r.head[0] & ~(0xffu << 6)) | n << 6;
  out.insert(out.end(), r.head.begin(), r.head.end());
  for (size_t i = 0; i < n; i += 2)
    out.push_back((ordinal - r.deps[i]) | (i + 1 < n ? ordinal - r.deps[i + 1] : 0) << 16);
  out.insert(out.end(), r.tail.begin(), r.tail.end());
}
} // namespace

Result<LinkedProgram> linkProgram(const std::vector<Segment> &segments) {
  using R = Result<LinkedProgram>;
  LinkedProgram linked;
  auto &out = linked.words;
  out.assign(5, 0);
  uint32_t next = 0;   // ordinal of the next record
  uint64_t coreWords = 0;
  std::vector<uint32_t> entry;   // exit of the previous segment
  auto near = [&](const std::vector<uint32_t> &deps) {
    return std::all_of(deps.begin(), deps.end(), [&](uint32_t d) { return next - d <= isa2::kMaxDistance; });
  };
  auto tooFar = [](const std::string &name) {
    return R::failure(name + ": a record is more than 65535 records from the phase boundary "
                             "(split NPU phases with --split-every)");
  };
  for (const auto &s : segments) {
    if (!s.pim.empty() || !s.paPim.empty()) {
      if (!s.program.empty() || (!s.pim.empty() && !s.paPim.empty()) ||
          s.pim.size() > UINT32_MAX || s.paPim.size() > UINT32_MAX)
        return R::failure(s.name + ": invalid PIM segment");
      auto record = s.paPim.empty()
        ? isa2::pimRecord(next, entry, s.pim.data(), uint32_t(s.pim.size()))
        : isa2::pimRecord(next, entry, s.paPim.data(), uint32_t(s.paPim.size()));
      if (!record) return R::failure(s.name + ": " + record.error);
      out.insert(out.end(), record.value.begin(), record.value.end());
      linked.symbols.push_back(s.name);
      entry = {next++};
      continue;
    }
    if (auto valid = verifyProgramWords(s.program); !valid) return R::failure(s.name + ": " + valid.error);
    const uint32_t base = next;
    auto records = readRecords(s.program);
    std::vector<bool> waitedFor(records.size(), false);
    for (auto &r : records) {
      for (auto &d : r.deps) { waitedFor[d] = true; d += base; }
      if (r.deps.empty()) r.deps = entry;
      if (!near(r.deps)) return tooFar(s.name);
      const char *kind = r.op == 0x26 ? ".core_block_" : r.op == 0x2d ? ".barrier_" : ".gdma_";
      linked.symbols.push_back(s.name + kind + std::to_string(next));
      if (r.op == 0x26) coreWords += r.tail.size();
      writeRecord(out, std::move(r), next++);
    }
    // The phase ends with a BARRIER after every record nothing in it waits for.
    std::vector<uint32_t> last;
    for (uint32_t i = 0; i < records.size(); ++i)
      if (!waitedFor[i]) last.push_back(base + i);
    if (last.empty()) continue;   // an empty phase passes its entry on
    do {
      std::vector<uint32_t> joined;
      for (size_t i = 0; i < last.size(); i += isa2::kMaxDependencies) {
        std::vector<uint32_t> group(last.begin() + i, last.begin() + std::min(last.size(), i + isa2::kMaxDependencies));
        if (!near(group)) return tooFar(s.name);
        auto barrier = isa2::barrierRecord(next, group);
        if (!barrier) return R::failure(barrier.error);
        out.insert(out.end(), barrier.value.begin(), barrier.value.end());
        linked.symbols.push_back(s.name + ".barrier_" + std::to_string(next));
        joined.push_back(next++);
      }
      last = std::move(joined);
    } while (last.size() > 1);
    entry = last;
  }
  out.push_back(0x2f);
  if (out.size() > UINT32_MAX || coreWords > UINT32_MAX) return R::failure("program exceeds binary field size");
  out[0] = kProgramMagic; out[1] = kProgramVersion; out[2] = uint32_t(out.size());
  out[3] = next; out[4] = uint32_t(coreWords);
  if (auto verified = verifyProgramWords(out); !verified) return R::failure(verified.error);
  return R::success(std::move(linked));
}

Result<EncodedProgram> encodeProgramWords(const Program &program) {
  using R = Result<EncodedProgram>;
  auto checked = schedule(program);
  if (!checked) return R::failure(checked.error);
  const auto &p = checked.value;
  if (p.cores > isa2::kCoreAny) return R::failure("CORE_BEGIN addresses at most 31 logical cores");
  if ((uint64_t(p.l1Bytes) + 7) / 8 >= isa2::kL1Units) return R::failure("L1 size does not fit CORE_BEGIN");
  std::vector<uint32_t> out(5);
  uint32_t next = 0;   // ordinal of the next record, its completion event
  uint64_t coreWords = 0;
  std::vector<std::vector<uint32_t>> records(p.commands.size());   // records that complete each command
  auto fail = [](uint32_t id, const std::string &what) {
    return R::failure("command " + std::to_string(id) + ": " + what);
  };
  auto near = [&](const std::vector<uint32_t> &deps) {
    return deps.empty() || next - deps.front() <= isa2::kMaxDistance;
  };
  for (uint32_t id = 0; id < p.commands.size(); ++id) {
    const auto &c = p.commands[id];
    std::vector<uint32_t> deps;
    for (auto d : c.dependencies) deps.insert(deps.end(), records[d].begin(), records[d].end());
    std::sort(deps.begin(), deps.end());
    while (deps.size() > isa2::kMaxDependencies) {
      std::vector<uint32_t> joined;
      for (size_t i = 0; i < deps.size(); i += isa2::kMaxDependencies) {
        std::vector<uint32_t> group(deps.begin() + i,
                                    deps.begin() + std::min(deps.size(), i + isa2::kMaxDependencies));
        if (!near(group)) return fail(id, "dependency more than 65535 records back");
        auto barrier = isa2::barrierRecord(next, group);
        if (!barrier) return R::failure(barrier.error);
        out.insert(out.end(), barrier.value.begin(), barrier.value.end());
        joined.push_back(next++);
      }
      deps = std::move(joined);
    }
    if (c.kind == Command::Kind::Core) {
      auto body = isa2::lowerCoreBlock(c.words);
      if (!body) return fail(id, body.error);
      if (body.value.size() >= isa2::kCoreWords) return fail(id, "core block exceeds 2^24 words");
      if (!near(deps)) return fail(id, "dependency more than 65535 records back");
      auto begin = isa2::coreBeginRecord(next, deps, uint32_t(body.value.size()), c.core, p.l1Bytes);
      if (!begin) return fail(id, begin.error);
      out.insert(out.end(), begin.value.begin(), begin.value.end());
      out.insert(out.end(), body.value.begin(), body.value.end());
      coreWords += body.value.size();
      records[id].push_back(next++);
      continue;
    }
    auto transfers = gdmaTransfers(c);
    if (!transfers) return fail(id, transfers.error);
    for (const auto &t : transfers.value) {
      if (!near(deps)) return fail(id, "dependency more than 65535 records back");
      auto record = isa2::gdmaRecord(c.kind == Command::Kind::Load, next, deps, t.lp6, t.l2, uint32_t(t.rowBytes),
                                     uint32_t(t.rows), uint32_t(t.lp6Stride), uint32_t(t.l2Stride));
      if (!record) return fail(id, record.error);
      out.insert(out.end(), record.value.begin(), record.value.end());
      records[id].push_back(next++);
    }
  }
  out.push_back(0x2f);
  if (out.size() > UINT32_MAX || coreWords > UINT32_MAX) return R::failure("program exceeds binary field size");
  out[0] = kProgramMagic; out[1] = kProgramVersion; out[2] = uint32_t(out.size());
  out[3] = next; out[4] = uint32_t(coreWords);
  auto verified = verifyProgramWords(out);
  if (!verified) return R::failure(verified.error);
  return R::success({std::move(out), next, coreWords});
}
Result<std::vector<uint8_t>> encodeProgram(const Program &program) {
  auto encoded = encodeProgramWords(program);
  if (!encoded) return Result<std::vector<uint8_t>>::failure(encoded.error);
  std::vector<uint8_t> bytes;
  bytes.reserve(encoded.value.words.size() * sizeof(uint32_t));
  for (auto w : encoded.value.words) for (unsigned b = 0; b < 4; ++b) bytes.push_back(uint8_t(w >> (8*b)));
  return Result<std::vector<uint8_t>>::success(std::move(bytes));
}
Result<std::string> systemManifest(const Program &p) {
  auto encoded = encodeProgramWords(p);
  if (!encoded) return Result<std::string>::failure(encoded.error);
  return Result<std::string>::success(systemManifestForEncoded(p, encoded.value));
}
namespace {
std::string manifest(const Program &p, uint64_t words, uint32_t commands, uint64_t coreWords) {
  std::string s = std::string("{\n  \"schema\": \"") + kProgramSchema + "\",\n  \"program\": \"program.bin\",\n  \"scheduling_policy\": \"compiler_static\",\n  \"logical_to_physical\": [";
  for (unsigned i = 0; i < p.cores; ++i) s += (i ? ", " : "") + std::to_string(i);
  return s + "],\n  \"program_word_count\": " + std::to_string(words) +
    ",\n  \"command_count\": " + std::to_string(commands) +
    ",\n  \"core_instruction_count\": " + std::to_string(coreWords) +
    ",\n  \"l2_bytes_required\": " + std::to_string(p.l2Bytes) +
    ",\n  \"l2_regions\": [{\"name\": \"l2\", \"offset\": 0, \"size\": " +
    std::to_string(p.l2Bytes) + ", \"alignment\": 64}]\n}\n";
}
} // namespace
std::string systemManifestForEncoded(const Program &p, const std::vector<uint8_t> &encoded) {
  // Successfully encoded program: header words 2..4 are words/records/core words.
  auto header = [&](size_t i) {
    uint32_t w = 0;
    for (unsigned b = 0; b < 4; ++b) w |= uint32_t(encoded[4 * i + b]) << (8 * b);
    return w;
  };
  return manifest(p, header(2), header(3), header(4));
}
std::string systemManifestForEncoded(const Program &p, const EncodedProgram &encoded) {
  return manifest(p, encoded.words.size(), encoded.commands, encoded.coreWords);
}
} // namespace npu
