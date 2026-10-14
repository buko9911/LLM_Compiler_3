#include "npu/Decode/Target/PlenaProgram.h"
#include "npu/Target/IsaV2.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdlib>
#include <fstream>

using namespace mlir::hybrid::plena;
namespace isa2 = ::npu::isa2;

namespace {
std::vector<uint32_t> loadU32Independent(uint32_t reg, uint32_t value) {
  uint32_t upper = value >> 12, lower = value & 0xFFF;
  std::vector<uint32_t> w{OP_LUI | reg << 6 | upper << 10};
  if (lower) w.push_back(OP_ADDI | reg << 6 | reg << 10 | lower << 14);
  return w;
}

bool fits(uint64_t value, unsigned bits) { return value < (uint64_t(1) << bits); }

/// Every dependency must be an earlier command within a 16-bit distance; at most 255.
bool depsFit(size_t ordinal, const std::vector<int> &deps) {
  if (deps.size() > 255) return false;
  for (int d : deps)
    if (d < 0 || size_t(d) >= ordinal || ordinal - size_t(d) >= (size_t(1) << 16)) return false;
  return true;
}
} // namespace

void ProgramBuilder::track(uint32_t word) {
  uint32_t opcode = word & 0x3F, rd = word >> 6 & 0xF, rs1 = word >> 10 & 0xF;
  if (opcode == OP_ADDI) gp[rd] = gp[rs1] + (word >> 14);
  else if (opcode == OP_LUI) gp[rd] = ((word >> 10) & 0xFFFFF) << 12;
  else if (opcode == OP_C_SET) control[int(word >> 22 & 0xF)] = gp[rd];
}

void ProgramBuilder::ensureSegment() {
  if (segDeps) return;
  segDeps = frontier;
  if (!replay) return;
  uint32_t saved15 = gp[15];
  for (uint32_t r = 0; r < 16; ++r)
    for (uint32_t w : loadU32Independent(r, gp[r])) cur.push_back(w);
  for (auto &[funct, value] : control) {
    for (uint32_t w : loadU32Independent(15, value)) cur.push_back(w);
    cur.push_back(OP_C_SET | 15u << 6 | uint32_t(funct) << 22);
  }
  for (uint32_t w : loadU32Independent(15, saved15)) cur.push_back(w);
  replay = false;
}

void ProgramBuilder::append(uint32_t word) {
  ensureSegment();
  cur.push_back(word);
  track(word);
}

void ProgramBuilder::appendRecord(const std::array<uint32_t, 4> &words) {
  ensureSegment();
  cur.insert(cur.end(), words.begin(), words.end());
}

int ProgramBuilder::push(Command c) {
  // Every record of a phase follows the phase entry, directly or through earlier records.
  if (c.deps.empty()) c.deps = base;
  // Large streaming rings can need more events than one ISA record can encode.
  // Join them without dropping hazards or imposing a software tile-count limit.
  while (c.deps.size() > 255) {
    std::vector<int> joined;
    for (size_t i = 0; i < c.deps.size(); i += 255) {
      Command barrier;
      barrier.kind = Kind::Barrier;
      barrier.deps.assign(c.deps.begin() + i, c.deps.begin() + std::min(c.deps.size(), i + 255));
      joined.push_back(push(std::move(barrier)));
    }
    c.deps = std::move(joined);
  }
  c.phase = phase;
  cmds.push_back(std::move(c));
  return int(cmds.size()) - 1;
}

int ProgramBuilder::flushBlock(bool updateFrontier) {
  if (cur.empty()) return -1;
  Command c;
  c.kind = Kind::CoreBlock;
  c.deps = segDeps ? *segDeps : std::vector<int>{};
  c.logicalCore = logicalCore;
  c.isaStart = int64_t(isa.size());
  c.isaCount = int64_t(cur.size());
  isa.insert(isa.end(), cur.begin(), cur.end());
  cur.clear();
  int id = push(std::move(c));
  segDeps.reset();
  replay = true;
  if (updateFrontier) frontier = {id};
  return id;
}

void ProgramBuilder::beginPhase(const std::string &name, std::vector<int> entry) {
  flushBlock();
  gp.fill(0);
  control.clear();
  replay = false;
  segDeps.reset();
  logicalCore = 0;
  phaseNames.push_back(name);
  phase = int(phaseNames.size()) - 1;
  phaseStart = cmds.size();
  base = entry;
  frontier = std::move(entry);
}

std::vector<int> ProgramBuilder::endPhase() {
  flushBlock();
  size_t end = cmds.size();
  if (end == phaseStart) return base;
  std::vector<bool> used(end - phaseStart, false);
  for (size_t id = phaseStart; id < end; ++id)
    for (int d : cmds[id].deps)
      if (d >= int(phaseStart)) used[size_t(d) - phaseStart] = true;
  std::vector<int> sinks;
  for (size_t id = phaseStart; id < end; ++id)
    if (!used[id - phaseStart]) sinks.push_back(int(id));
  // A record joins at most 255 events: join larger sets in chunks, then join the chunks.
  while (sinks.size() > 1 || cmds[size_t(sinks[0])].kind != Kind::Barrier) {
    std::vector<int> joined;
    for (size_t i = 0; i < sinks.size(); i += 255) {
      Command b;
      b.kind = Kind::Barrier;
      b.deps.assign(sinks.begin() + int64_t(i), sinks.begin() + int64_t(std::min(sinks.size(), i + 255)));
      joined.push_back(push(std::move(b)));
    }
    sinks = std::move(joined);
  }
  frontier = sinks;
  base = sinks;
  return sinks;
}

int ProgramBuilder::pimRecord(const std::string &name, std::vector<int> deps, const std::vector<::npu::pim::Primitive> &commands) {
  assert(!commands.empty());
  flushBlock();
  phaseNames.push_back(name);
  phase = int(phaseNames.size()) - 1;
  phaseStart = cmds.size();
  base = deps;
  Command c;
  c.kind = Kind::Pim;
  c.deps = std::move(deps);
  c.pimFirst = int64_t(pimPrimitives.size());
  c.pimCount = int64_t(commands.size());
  pimPrimitives.insert(pimPrimitives.end(), commands.begin(), commands.end());
  int id = push(std::move(c));
  frontier = {id};
  base = {id};
  return id;
}

int ProgramBuilder::appendGdma(Kind kind, std::vector<int> deps, uint64_t lp6, uint64_t l2, uint32_t bytes,
                               uint32_t rows, uint32_t lp6Stride, uint32_t l2Stride) {
  assert(bytes > 0 && rows > 0);
  Command c;
  c.kind = kind;
  c.deps = std::move(deps);
  c.lp6 = lp6;
  c.l2 = l2;
  c.rowBytes = bytes;
  c.rows = rows;
  c.lp6Stride = lp6Stride ? lp6Stride : bytes;
  c.l2Stride = l2Stride ? l2Stride : bytes;
  reserveL2(l2, uint64_t(rows - 1) * c.l2Stride + bytes);
  return push(std::move(c));
}

int ProgramBuilder::gdmaLoad(uint64_t lp6, uint64_t l2, uint32_t bytes, uint32_t rows, uint32_t lp6Stride,
                             uint32_t l2Stride, bool overlap) {
  if (overlap && !cur.empty()) {
    std::vector<int> deps = segDeps ? *segDeps : frontier;
    int cb = flushBlock(false);
    int g = appendGdma(Kind::GdmaLoad, deps, lp6, l2, bytes, rows, lp6Stride, l2Stride);
    frontier.clear();
    if (cb >= 0) frontier.push_back(cb);
    frontier.push_back(g);
    return g;
  }
  flushBlock();
  int g = appendGdma(Kind::GdmaLoad, frontier, lp6, l2, bytes, rows, lp6Stride, l2Stride);
  frontier = {g};
  return g;
}

int ProgramBuilder::gdmaStore(uint64_t lp6, uint64_t l2, uint32_t bytes, uint32_t rows, uint32_t lp6Stride,
                              uint32_t l2Stride, bool allowFollowingOverlap) {
  int cb = flushBlock();
  int g = appendGdma(Kind::GdmaStore, frontier, lp6, l2, bytes, rows, lp6Stride, l2Stride);
  frontier = (allowFollowingOverlap && cb >= 0) ? std::vector<int>{cb} : std::vector<int>{g};
  return g;
}

bool ProgramBuilder::finish(const std::string &dir, int logicalCores) {
  flushBlock();
  // ISA ver 2.0: every core block is lowered from the builder stream; a command's completion
  // event is its ordinal.
  std::vector<uint32_t> out(isa2::kHeaderWords, 0);
  int64_t coreCount = 0;
  auto bad = [](const std::string &what, size_t id) {
    llvm::errs() << "program.bin: command " << id << ": " << what << "\n";
    return false;
  };
  for (size_t id = 0; id < cmds.size(); ++id) {
    const Command &c = cmds[id];
    if (!depsFit(id, c.deps)) return bad("dependencies do not fit the ISA ver 2.0 record", id);
    const std::vector<uint32_t> deps(c.deps.begin(), c.deps.end());
    if (c.kind == Kind::Barrier) {
      auto record = isa2::barrierRecord(uint32_t(id), deps);
      if (!record) return bad(record.error, id);
      out.insert(out.end(), record.value.begin(), record.value.end());
      continue;
    }
    if (c.kind == Kind::Pim) {
      if (!fits(uint64_t(c.pimCount), 32)) return bad("PIM record has too many primitives", id);
      auto record = isa2::pimRecord(uint32_t(id), deps, pimPrimitives.data() + c.pimFirst, uint32_t(c.pimCount));
      if (!record) return bad(record.error, id);
      for (uint32_t w : record.value)
        out.push_back(w);
      continue;
    }
    if (c.kind == Kind::CoreBlock) {
      std::vector<uint32_t> in(isa.begin() + c.isaStart, isa.begin() + c.isaStart + c.isaCount), body;
      if (std::getenv("HYBRID_DUMP_BUILDER_STREAM")) {
        std::ofstream dump(dir + "/builder_stream.bin", std::ios::binary | std::ios::app);
        uint32_t n = uint32_t(in.size());
        dump.write(reinterpret_cast<const char *>(&n), 4);
        dump.write(reinterpret_cast<const char *>(in.data()), std::streamsize(in.size() * 4));
      }
      auto lowered = isa2::lowerCoreBlock(in);
      if (!lowered) return bad(lowered.error, id);
      body = std::move(lowered.value);
      if (!fits(body.size(), 24) || c.logicalCore < 0 || c.logicalCore >= int(isa2::kCoreAny))
        return bad("core block does not fit CORE_BEGIN", id);
      uint64_t l1 = 4 * 1024 * 1024;
      auto record = isa2::coreBeginRecord(uint32_t(id), deps, uint32_t(body.size()), uint32_t(c.logicalCore), l1);
      if (!record) return bad(record.error, id);
      out.insert(out.end(), record.value.begin(), record.value.end());
      out.insert(out.end(), body.begin(), body.end());
      coreCount += int64_t(body.size());
      continue;
    }
    if (!fits(c.rowBytes, 20) || !fits(c.lp6, 36) || !fits(c.l2, 24) || c.rows == 0 || !fits(c.rows - 1, 16) ||
        !fits(c.lp6Stride, 24) || !fits(c.l2Stride, 24))
      return bad("GDMA fields do not fit the ISA ver 2.0 record", id);
    auto record = isa2::gdmaRecord(c.kind == Kind::GdmaLoad, uint32_t(id), deps, c.lp6, c.l2, c.rowBytes, c.rows,
                                   c.lp6Stride, c.l2Stride);
    if (!record) return bad(record.error, id);
    out.insert(out.end(), record.value.begin(), record.value.end());
  }
  out.push_back(isa2::OP_PROGRAM_END);
  out[0] = isa2::kMagic;
  out[1] = isa2::kVersion;
  out[2] = uint32_t(out.size());
  out[3] = uint32_t(cmds.size());
  out[4] = uint32_t(coreCount);
  {
    std::ofstream f(dir + "/program.bin", std::ios::binary);
    if (!f) return false;
    f.write(reinterpret_cast<const char *>(out.data()), std::streamsize(out.size() * 4));
  }
  llvm::json::Array symbols, l2regions, placement;
  bool hasPim = false;
  for (size_t id = 0; id < cmds.size(); ++id) {
    const Command &c = cmds[id];
    // Symbol = <phase>.<kind>_<id>; a PIM record is named after its phase.
    std::string name = phaseNames[size_t(c.phase)];
    if (c.kind == Kind::Pim) hasPim = true;
    else
      name += std::string(c.kind == Kind::CoreBlock ? ".core_block_" : (c.kind == Kind::Barrier ? ".barrier_" : ".gdma_")) +
              std::to_string(id);
    symbols.push_back(llvm::json::Object{{"id", int64_t(id)}, {"name", name}});
  }
  for (int c = 0; c < logicalCores; ++c) placement.push_back(c);
  if (l2HighWater)
    l2regions.push_back(llvm::json::Object{{"name", "compiler_workspace"}, {"offset", 0}, {"size", int64_t(l2HighWater)},
                                           {"alignment", 64}});
  llvm::json::Object sys{{"schema", "plena.v2.unified_program.isa_v2.0"}, {"program", "program.bin"},
                         {"program_word_count", int64_t(out.size())}, {"command_count", int64_t(cmds.size())},
                         {"core_instruction_count", coreCount}, {"command_symbols", std::move(symbols)},
                         {"scheduling_policy", "compiler_static"}, {"logical_to_physical", std::move(placement)},
                         {"l2_bytes_required", int64_t(l2HighWater)}, {"l2_regions", std::move(l2regions)}};
  if (hasPim) {
    sys["pim_primitive_abi"] = int64_t(::npu::pim::kPrimitiveVersion);
    auto kernel = llvm::json::parse(pimKernelJson);
    if (!kernel) {
      llvm::consumeError(kernel.takeError());
      llvm::errs() << "program.bin: PIM records need a PIM kernel configuration\n";
      return false;
    }
    sys["pim_kernel"] = std::move(*kernel);
  }
  std::error_code ec;
  llvm::raw_fd_ostream js(dir + "/system.json", ec);
  if (ec) return false;
  js << llvm::formatv("{0:2}", llvm::json::Value(std::move(sys))) << "\n";
  return true;
}
