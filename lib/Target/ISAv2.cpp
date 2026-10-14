// PLENA ISA ver 2.0: encoders, the builder-stream lowering, system records.
// The lowering follows NPU_Simulator tools/isa_v2.py (lower_core_block)
// decision for decision; tests compare the two word for word.
#include "npu/Target/ISAv2.h"
#include <array>
#include <cstdio>
#include <map>
#include <optional>
#include <string>

namespace npu::isa2 {
namespace {
// ── ISA ver 2.0 core opcodes ────────────────────────────────────────────────
constexpr uint32_t C_LI = 0x20, C_SET = 0x21, C_ADDI = 0x22, C_LOOP_BEGIN = 0x23,
                   C_LOOP_END = 0x24, C_LUI = 0x25, V_OP = 0x30, V_CVT = 0x31,
                   V_LD = 0x32, V_ST = 0x33, S_OP = 0x34, S_LD = 0x35, S_ST = 0x36,
                   M_LD = 0x37, LDMA_R = 0x38, LDMA_I = 0x39, M_LD_ACC = 0x3A,
                   M_MMA = 0x3B, M_WB = 0x3C, C_SYNC = 0x3F;
constexpr unsigned F16 = 0, F32 = 1, I8 = 2, I32 = 3;
constexpr unsigned SET_ROWS = 0, SET_SRC_STRIDE = 1, SET_DST_STRIDE = 2, SET_VL = 3, SET_BYTES = 4;
constexpr unsigned SYNC_LDMA = 0, SYNC_MATRIX = 1, SYNC_VPU = 2, SYNC_ALL = 3, SYNC_EVENT = 4;
constexpr unsigned MAX_LOCAL_EVENT = 15;
constexpr uint32_t M_WB_STRIDE_EXTENDED = 1023;
constexpr uint64_t IMM22 = uint64_t(1) << 22;
constexpr uint64_t VECTOR_WINDOW = uint64_t(16) << 16, SCALAR_WINDOW = uint64_t(4) << 18;

using Words = std::vector<uint32_t>;
using Reg = std::optional<uint32_t>;

// Builder-stream (ISA ver 1.0) function fields -> ver 2.0 operation codes.
int setFromFunct(unsigned funct) {
  switch (funct) {
  case 0: return SET_ROWS; case 1: return SET_SRC_STRIDE; case 2: return SET_DST_STRIDE;
  case 3: return SET_VL; case 7: return SET_BYTES; default: return -1;
  }
}
int f16Binary(unsigned f) { switch (f) { case 1: return 0; case 2: return 1; case 3: return 2; case 4: return 4; case 5: return 5; default: return -1; } }
int f16Unary(unsigned f) { switch (f) { case 1: return 6; case 2: return 8; case 3: return 9; case 4: return 10; case 5: return 11; case 6: return 7; default: return -1; } }
// Functions 9.. of the same groups are the FP32 forms of the first ones; ver 2.0
// carries the type in the dtype field of one V_OP.
int f32Binary(unsigned f) { return f >= 9 && f <= 13 ? f16Binary(f - 8) : -1; }
int f32Unary(unsigned f) { return f == 9 || f == 10 ? f16Unary(f - 7) : -1; }
bool isStridedLdma(unsigned f) { return f == 4 || f == 5 || f == 6 || f == 7 || f == 9 || f == 12 || f == 13; }
bool isTaggedLdma(unsigned f) { return f >= 10 && f <= 13; }
bool isImmediateLdma(unsigned f) { return f <= 3 || f == 8 || f == 10 || f == 11; }

class Encoder {
public:
  std::string error;
  bool fits(uint64_t value, unsigned bits, const char *name) {
    if (value < (uint64_t(1) << bits)) return true;
    if (error.empty()) error = std::string(name) + "=" + std::to_string(value) + " does not fit " + std::to_string(bits) + " bits";
    return false;
  }
  bool fail(const std::string &message) { if (error.empty()) error = message; return false; }

  uint32_t cLi(unsigned rd, uint64_t v) { fits(rd, 4, "rd"); fits(v, 22, "C_LI immediate"); return C_LI | rd << 6 | uint32_t(v) << 10; }
  uint32_t cLui(unsigned rd, uint64_t upper) { fits(rd, 4, "rd"); fits(upper, 20, "C_LUI immediate"); return C_LUI | rd << 6 | uint32_t(upper) << 10; }
  uint32_t cAddi(unsigned rd, unsigned rs1, uint64_t v) { fits(v, 18, "C_ADDI immediate"); return C_ADDI | rd << 6 | rs1 << 10 | uint32_t(v) << 14; }
  void loadConstant(Words &out, unsigned rd, uint64_t value) {
    value &= 0xffffffffu;
    if (value < IMM22) { out.push_back(cLi(rd, value)); return; }
    out.push_back(cLui(rd, value >> 12));
    if (value & 0xfff) out.push_back(cAddi(rd, rd, value & 0xfff));
  }
  uint32_t cSet(unsigned sel, uint64_t v) { fits(v, 22, "C_SET immediate"); return C_SET | sel << 6 | uint32_t(v) << 10; }
  uint32_t cSetGp(unsigned sel, unsigned gp) { return C_SET | sel << 6 | 1u << 9 | gp << 10; }
  uint32_t cLoopBegin(uint64_t count) {
    if (!count) fail("loop count must be positive");
    fits(count, 25, "loop count");
    return C_LOOP_BEGIN | uint32_t(count) << 7;
  }
  uint32_t cLoopBeginGp(unsigned gp) { return C_LOOP_BEGIN | 1u << 6 | gp << 7; }
  uint32_t cSync(unsigned kind, unsigned event = 0) { return C_SYNC | kind << 6 | event << 10; }
  uint32_t vOp(unsigned op, unsigned dtype, unsigned rd, unsigned rs1 = 0, unsigned rs2 = 0, unsigned rs3 = 0) {
    fits(op, 6, "V_OP op");
    return V_OP | op << 6 | dtype << 12 | rd << 14 | rs1 << 18 | rs2 << 22 | rs3 << 26;
  }
  uint32_t vCvt(unsigned mode, unsigned src, unsigned dst, unsigned rd, unsigned rs1, unsigned rs2, unsigned rs3,
                unsigned rnd, unsigned sat) {
    return V_CVT | mode << 6 | src << 10 | dst << 12 | rd << 14 | rs1 << 18 | rs2 << 22 | rs3 << 26 |
           rnd << 30 | sat << 31;
  }
  uint32_t vMem(bool store, unsigned dtype, unsigned vreg, unsigned base, uint64_t offset) {
    if (offset % 16) fail("V_LD/V_ST offset must be a multiple of 16 bytes");
    fits(offset / 16, 16, "V_LD/V_ST offset");
    return (store ? V_ST : V_LD) | dtype << 6 | vreg << 8 | base << 12 | uint32_t(offset / 16) << 16;
  }
  uint32_t sMem(bool store, unsigned sreg, unsigned base, uint64_t offset) {
    if (offset % 4) fail("S_LD/S_ST offset must be a multiple of 4 bytes");
    fits(offset / 4, 18, "S_LD/S_ST offset");
    return (store ? S_ST : S_LD) | sreg << 6 | base << 10 | uint32_t(offset / 4) << 14;
  }
  uint32_t sOp(unsigned op, unsigned rd, unsigned rs1, unsigned rs2) {
    return S_OP | op << 6 | rd << 10 | rs1 << 14 | rs2 << 18;
  }
  void mLd(Words &out, unsigned kind, unsigned gp, uint64_t mn, uint64_t k, std::optional<uint64_t> stride) {
    if (mn < 1 || mn > 32 || k < 1 || k > (1u << 14)) {
      fail("M_LD shape MN=" + std::to_string(mn) + " K=" + std::to_string(k) + " out of range");
      return;
    }
    uint32_t word = M_LD | kind << 6 | gp << 8 | uint32_t(mn - 1) << 12 | uint32_t(k - 1) << 17;
    if (!stride) { out.push_back(word); return; }
    fits(*stride, 32, "stride");
    out.push_back(word | 1u << 31); out.push_back(uint32_t(*stride));
  }
  void mWb(Words &out, unsigned dtype, unsigned gp, uint64_t m, uint64_t n, uint64_t stride) {
    if (m < 1 || m > 32 || n < 1 || n > 32) { fail("M_WB shape out of range"); return; }
    uint32_t word = M_WB | dtype << 6 | gp << 8 | uint32_t(m - 1) << 12 | uint32_t(n - 1) << 17;
    if (stride % 64 == 0 && stride / 64 < M_WB_STRIDE_EXTENDED) { out.push_back(word | uint32_t(stride / 64) << 22); return; }
    fits(stride, 32, "M_WB stride");
    out.push_back(word | M_WB_STRIDE_EXTENDED << 22); out.push_back(uint32_t(stride));
  }
  uint32_t mMma(bool fp16, bool accumulate, unsigned count) {
    fits(count, 10, "M_MMA count");
    return M_MMA | unsigned(fp16) << 6 | unsigned(accumulate) << 7 | count << 8;
  }
  uint32_t ldmaR(unsigned op, unsigned l1, unsigned l2, unsigned eventGp) {
    return LDMA_R | op << 6 | l1 << 10 | l2 << 14 | eventGp << 18;
  }
  void ldmaI(Words &out, unsigned op, uint64_t l1, uint64_t l2, uint64_t bytes, unsigned event) {
    fits(l1 / 32, 18, "LDMA.I L1"); fits(l2 / 32, 18, "LDMA.I L2"); fits(bytes / 32, 14, "LDMA.I bytes");
    out.push_back(LDMA_I | op << 6 | event << 10 | uint32_t(l1 / 32) << 14);
    out.push_back(uint32_t(l2 / 32) | uint32_t(bytes / 32) << 18);
  }
};

// ── the lowering of one builder-stream core block ───────────────────────────
class Lowering : public Encoder {
public:
  Words out;
  // Builder-stream GP values; nullopt = only the physical register holds it.
  std::array<Reg, 16> value{};
  // Physical register contents known to the lowering (nullopt = unknown).
  std::array<Reg, 16> physical{};
  std::array<Reg, 5> control{}, physicalControl{};
  std::map<uint32_t, unsigned> events;
  int loopDepth = 0;

  Lowering() {
    value.fill(0u); physical.fill(0u);
    control = {Reg(1u), Reg(0u), Reg(0u), Reg(0u), Reg(0u)};  // rows, src, dst, VL, bytes
    physicalControl = control;
  }

  // A register that may be overwritten: not holding a loop value and not
  // already chosen for another operand of the same instruction (`keep`).
  int scratch(unsigned home, std::initializer_list<int> keep = {}) {
    auto kept = [&](int r) { for (int k : keep) if (k == r) return true; return false; };
    if (value[home] && !kept(int(home))) return int(home);
    for (int r = 0; r < 16; ++r)
      if (value[r] && !kept(r)) return r;
    fail("no GP register is free to hold a constant");
    return 0;
  }
  // A register that holds `v`; loads it into `home` if needed.
  int materialize(uint32_t v, unsigned home, std::initializer_list<int> keep = {}) {
    if (physical[home] == v) return int(home);
    for (int r = 0; r < 16; ++r)
      if (physical[r] == v) return r;
    int target = scratch(home, keep);
    loadConstant(out, unsigned(target), v);
    physical[target] = v;
    return target;
  }
  int use(unsigned reg, std::initializer_list<int> keep = {}) {
    return value[reg] ? materialize(*value[reg], reg, keep) : int(reg);
  }
  // (base register, byte offset) for a scaled-offset memory access.
  std::pair<int, uint64_t> base(unsigned reg, uint64_t unit, uint64_t window) {
    if (!value[reg]) return {int(reg), 0};
    const uint64_t address = *value[reg], limit = window / unit;
    auto usable = [&](int candidate) {
      auto b = physical[candidate];
      return b && *b <= address && (address - *b) % unit == 0 && (address - *b) / unit < limit;
    };
    if (usable(int(reg))) return {int(reg), address - *physical[reg]};
    for (int r = 0; r < 16; ++r)
      if (usable(r)) return {r, address - *physical[r]};
    uint64_t b = address % unit ? address : address - address % window;
    int target = scratch(reg);
    loadConstant(out, unsigned(target), b);
    physical[target] = uint32_t(b);
    return {target, address - b};
  }
  unsigned event(unsigned reg) {
    if (!value[reg]) { fail("a local LDMA event must be a constant in the builder stream"); return 0; }
    auto found = events.find(*value[reg]);
    if (found != events.end()) return found->second;
    if (events.size() == MAX_LOCAL_EVENT) { fail("a core block uses more than 15 local LDMA events"); return 0; }
    unsigned id = unsigned(events.size()) + 1;
    events[*value[reg]] = id;
    return id;
  }
  void need(unsigned sel) {
    Reg wanted = control[sel];
    if (!wanted) {
      if (physicalControl[sel]) fail("control value set from a loop register was overwritten");
      return;
    }
    if (physicalControl[sel] != wanted) {
      if (*wanted < IMM22) out.push_back(cSet(sel, *wanted));
      else out.push_back(cSetGp(sel, unsigned(materialize(*wanted, 15))));
      physicalControl[sel] = wanted;
    }
  }
  void syncForLoop() {
    for (unsigned r = 0; r < 16; ++r) {
      if (value[r] && physical[r] != value[r]) loadConstant(out, r, *value[r]);
      value[r].reset(); physical[r].reset();
    }
    for (unsigned sel = 0; sel < 5; ++sel) {
      if (control[sel]) need(sel);
      control[sel].reset(); physicalControl[sel].reset();
    }
  }

  void instruction(const uint32_t *record) {
    const uint32_t word = record[0];
    const unsigned opcode = word & 0x3f;
    const unsigned rd = word >> 6 & 15, rs1 = word >> 10 & 15, rs2 = word >> 14 & 15, rs3 = word >> 18 & 15;
    const unsigned funct = word >> 22 & 15;
    const bool inLoop = loopDepth > 0;
    switch (opcode) {
    case 0x25: {  // C_LUI
      uint32_t upper = word >> 10 & 0xfffff;
      if (inLoop) { out.push_back(cLui(rd, upper)); value[rd].reset(); physical[rd].reset(); }
      else value[rd] = upper << 12;
      return;
    }
    case 0x22: {  // C_ADDI
      uint32_t immediate = word >> 14;
      if (inLoop || !value[rs1]) { out.push_back(cAddi(rd, rs1, immediate)); value[rd].reset(); physical[rd].reset(); }
      else value[rd] = uint32_t(uint64_t(*value[rs1]) + immediate);
      return;
    }
    case 0x39: {  // C_SET gp
      int sel = setFromFunct(funct);
      if (sel < 0) { fail("invalid builder-stream C_SET"); return; }
      if (!value[rd]) { out.push_back(cSetGp(unsigned(sel), rd)); control[sel].reset(); physicalControl[sel].reset(); }
      else control[sel] = value[rd];
      return;
    }
    case 0x23: {  // C_LOOP_BEGIN gp
      Reg count = value[rd];
      syncForLoop();
      out.push_back(count ? cLoopBegin(*count) : cLoopBeginGp(rd));
      ++loopDepth;
      return;
    }
    case 0x24: out.push_back(C_LOOP_END); --loopDepth; return;
    case 0x3f: {  // C_WAIT_* / C_FENCE_ALL
      static const int kinds[] = {-1, SYNC_LDMA, SYNC_MATRIX, SYNC_VPU, SYNC_ALL};
      if (funct < 1 || funct > 4 || (word >> 6 & 0xffff)) { fail("invalid builder-stream wait"); return; }
      out.push_back(cSync(unsigned(kinds[funct])));
      return;
    }
    case 0x2e: out.push_back(cSync(SYNC_EVENT, event(rd))); return;
    case 0x38: ldma(funct, rd, rs1, rs2); return;
    case 0x37: matrixLoad(record); return;
    case 0x3a: {  // M_LOAD_ACC
      uint32_t m = record[1], n = record[2], stride = record[3];
      if (m < 1 || m > 32 || n < 1 || n > 32) { fail("M_LD_ACC shape out of range"); return; }
      unsigned gp = unsigned(use(rs1));
      uint32_t w = M_LD_ACC | unsigned(funct == 2) << 6 | gp << 7 | (m - 1) << 11 | (n - 1) << 16;
      if (m == 1 || stride == n * 4) out.push_back(w);
      else { out.push_back(w | 1u << 21); out.push_back(stride); }
      return;
    }
    case 0x3c: {  // M_WRITEOUT
      uint32_t m = record[1], n = record[2], stride = record[3];
      unsigned dtype, element;
      if (funct == 1) { dtype = 0; element = 2; }
      else if (funct == 4) { dtype = 1; element = 4; }
      else if (funct == 2) { dtype = 2; element = 4; }
      else { fail("invalid builder-stream M_WRITEOUT"); return; }
      unsigned gp = unsigned(use(rd));
      mWb(out, dtype, gp, m, n, (m == 1 || stride == n * element) ? 0 : stride);
      return;
    }
    case 0x35: memory(funct, rd, rs1); return;
    case 0x30: {
      int op = f16Binary(funct); unsigned type = F16;
      if (op < 0) { op = f32Binary(funct); type = F32; }
      if (op < 0) break;
      need(SET_VL); out.push_back(vOp(unsigned(op), type, rd, rs1, rs2)); return;
    }
    case 0x31: {
      int op = f16Unary(funct); unsigned type = F16;
      if (op < 0) { op = f32Unary(funct); type = F32; }
      if (op < 0) break;
      need(SET_VL); out.push_back(vOp(unsigned(op), type, rd, rs1)); return;
    }
    case 0x32: need(SET_VL); out.push_back(vOp(27 + funct, F16, rd, rs1, rs2)); return;
    case 0x33: {
      unsigned op, dtype;
      switch (funct) {
      case 1: op = 33; dtype = F16; break; case 2: op = 34; dtype = F16; break;
      case 3: op = 35; dtype = F16; break; case 4: op = 33; dtype = F32; break;
      case 5: op = 34; dtype = F32; break; default: fail("invalid builder-stream reduction"); return;
      }
      need(SET_VL);
      out.push_back(vOp(op, dtype, rd, rs1, op == 35 ? rs2 : 0));
      return;
    }
    case 0x34: out.push_back(sOp(funct - 1, rd, rs1, rs2)); return;
    case 0x3d: {
      need(SET_VL);
      switch (funct) {
      case 1: out.push_back(vCvt(2, F16, I8, rd, rs1, rs2, 0, 1, 1)); return;
      case 2: out.push_back(vCvt(6, I8, F16, rd, rs1, rs2, 0, 1, 0)); return;
      case 3: out.push_back(vCvt(7, I32, F16, rd, rs1, rs2, rs3, 1, 0)); return;
      case 4: out.push_back(vCvt(0, F16, F32, rd, rs1, 0, 0, 1, 0)); return;
      case 5: out.push_back(vCvt(0, F32, F16, rd, rs1, 0, 0, 1, 0)); return;
      default: break;
      }
      break;
    }
    default: break;
    }
    char hex[16];
    snprintf(hex, sizeof hex, "%#010x", word);
    fail(std::string("unsupported builder-stream instruction ") + hex);
  }

  void matrixLoad(const uint32_t *record) {
    const uint32_t word = record[0];
    const unsigned funct = word >> 22 & 15;
    const uint32_t rows = record[1], columns = record[2], stride = record[3];
    int kind;
    switch (funct) { case 1: kind = 0; break; case 3: kind = 1; break; case 5: kind = 2; break; case 7: kind = 3; break;
                     default: fail("invalid builder-stream M_LOAD"); return; }
    const uint32_t element = (funct == 1 || funct == 5) ? 1 : 2;
    const bool weight = funct == 1 || funct == 3;
    const uint64_t k = weight ? rows : columns, mn = weight ? columns : rows;
    unsigned gp = unsigned(use(word >> 10 & 15));
    const bool contiguous = rows == 1 || uint64_t(stride) == uint64_t(columns) * element;
    mLd(out, unsigned(kind), gp, mn, k, contiguous ? std::nullopt : std::optional<uint64_t>(stride));
  }

  void memory(unsigned funct, unsigned rd, unsigned rs1) {
    const bool store = funct % 2 == 0;
    const unsigned address = store ? rd : rs1, data = store ? rs1 : rd;
    if (funct >= 9) {
      auto [b, offset] = base(address, 4, SCALAR_WINDOW);
      out.push_back(sMem(store, data, unsigned(b), offset));
      return;
    }
    need(SET_VL);
    auto [b, offset] = base(address, 16, VECTOR_WINDOW);
    out.push_back(vMem(store, (funct - 1) / 2, data, unsigned(b), offset));
  }

  void ldma(unsigned funct, unsigned rd, unsigned rs1, unsigned rs2) {
    Reg l1 = value[rd], l2 = value[rs1], size = control[SET_BYTES];
    unsigned ev = isTaggedLdma(funct) ? event(rs2) : 0;
    if (isImmediateLdma(funct) && l1 && l2 && size &&
        *l1 % 32 == 0 && *l1 < (1u << 23) && *l2 % 32 == 0 && *l2 < (1u << 23) &&
        *size % 32 == 0 && *size > 0 && *size < (1u << 19)) {
      ldmaI(out, funct, *l1, *l2, *size, ev);
      return;
    }
    need(SET_BYTES);
    if (isStridedLdma(funct)) { need(SET_ROWS); need(SET_SRC_STRIDE); need(SET_DST_STRIDE); }
    int first = use(rd);
    int second = use(rs1, {first});
    int eventRegister = ev ? materialize(ev, rs2, {first, second}) : 0;
    out.push_back(ldmaR(funct, unsigned(first), unsigned(second), unsigned(eventRegister)));
  }
};

// Builder-stream records that span four words (header + three payload words).
bool isRecordOpcode(unsigned opcode) { return opcode == 0x36 || opcode == 0x37 || opcode == 0x3a || opcode == 0x3c; }

Words dependencyWords(Encoder &e, uint32_t ordinal, const std::vector<uint32_t> &dependencies) {
  Words words;
  if (dependencies.size() > kMaxDependencies) { e.fail("more than 255 dependencies; join them with a BARRIER"); return words; }
  for (size_t i = 0; i < dependencies.size(); i += 2) {
    uint32_t word = 0;
    for (size_t half = 0; half < 2 && i + half < dependencies.size(); ++half) {
      uint32_t dep = dependencies[i + half];
      if (dep >= ordinal || ordinal - dep > kMaxDistance) {
        e.fail("record " + std::to_string(ordinal) + " depends on " + std::to_string(dep) + ": distance out of range");
        return words;
      }
      word |= (ordinal - dep) << (16 * half);
    }
    words.push_back(word);
  }
  return words;
}
} // namespace

Result<std::vector<uint32_t>> lowerCoreBlock(const std::vector<uint32_t> &words) {
  using R = Result<std::vector<uint32_t>>;
  Lowering lowering;
  size_t index = 0;
  while (index < words.size() && lowering.error.empty()) {
    const uint32_t word = words[index];
    const unsigned opcode = word & 0x3f;
    if (opcode == 0x3b) {
      // A run of one-word M_MMA slices of one rectangle -> one M_MMA.
      const unsigned mode = word >> 22 & 15;
      if (word >> 27 || (mode != 1 && mode != 3)) return R::failure("invalid builder-stream M_MMA");
      bool accumulate = word >> 26 & 1;
      unsigned count = 0;
      while (index < words.size() && (words[index] & 0x3f) == 0x3b && (words[index] >> 22 & 15) == mode) {
        if (count && !(words[index] >> 26 & 1)) break;
        ++count; ++index;
      }
      while (count) {
        unsigned chunk = count < 1023 ? count : 1023;
        lowering.out.push_back(lowering.mMma(mode == 3, accumulate, chunk));
        accumulate = true;
        count -= chunk;
      }
      continue;
    }
    const size_t length = isRecordOpcode(opcode) ? 4 : 1;
    if (index + length > words.size()) return R::failure("truncated four-word builder-stream record");
    lowering.instruction(words.data() + index);
    index += length;
  }
  if (!lowering.error.empty()) return R::failure(lowering.error);
  if (lowering.loopDepth) return R::failure("unterminated hardware loop");
  return R::success(std::move(lowering.out));
}

unsigned instructionWords(const uint32_t *w, size_t available) {
  if (!available) return 0;
  const uint32_t word = w[0];
  unsigned length = 1;
  switch (word & 0x3f) {
  case C_LI: case C_SET: case C_ADDI: case C_LOOP_BEGIN: case C_LOOP_END: case C_LUI:
  case V_OP: case V_CVT: case V_LD: case V_ST: case S_OP: case S_LD: case S_ST:
  case LDMA_R: case M_MMA: case C_SYNC: break;
  case LDMA_I: length = 2; break;
  case M_LD: length = word >> 31 ? 2 : 1; break;
  case M_LD_ACC: length = word >> 21 & 1 ? 2 : 1; break;
  case M_WB: length = (word >> 22) == M_WB_STRIDE_EXTENDED ? 2 : 1; break;
  default: return 0;
  }
  return length <= available ? length : 0;
}

Result<bool> verifyCoreWords(const std::vector<uint32_t> &words) {
  using R = Result<bool>;
  if (words.empty()) return R::failure("empty core block");
  size_t last = 0;
  for (size_t i = 0; i < words.size();) {
    unsigned length = instructionWords(words.data() + i, words.size() - i);
    if (!length) return R::failure("invalid ISA ver 2.0 core instruction at word " + std::to_string(i));
    last = i;
    i += length;
  }
  if (words[last] != (C_SYNC | SYNC_ALL << 6))
    return R::failure("core block must end with C_SYNC ALL to expose completed accesses");
  return R::success(true);
}

Result<std::vector<uint32_t>> gdmaRecord(bool load, uint32_t ordinal, const std::vector<uint32_t> &dependencies,
                                         uint64_t lp6, uint64_t l2, uint64_t rowBytes, uint64_t rows,
                                         uint64_t lp6Stride, uint64_t l2Stride) {
  using R = Result<std::vector<uint32_t>>;
  Encoder e;
  if (!rowBytes || !rows) return R::failure("GDMA moves nothing");
  const bool twoD = rows != 1 || lp6Stride != rowBytes || l2Stride != rowBytes;
  e.fits(dependencies.size(), 8, "GDMA dependency count");
  e.fits(rowBytes, 20, "GDMA row bytes");
  e.fits(lp6, 36, "GDMA LP6 address");
  e.fits(l2, 24, "GDMA L2 address");
  // [5:0] opcode, [6] 2-D, [14:7] dependencies, [34:15] row bytes,
  // [70:35] LP6 byte address, [94:71] L2 byte address.
  unsigned __int128 record = (load ? kGdmaLoad : kGdmaStore);
  record |= (unsigned __int128)twoD << 6;
  record |= (unsigned __int128)dependencies.size() << 7;
  record |= (unsigned __int128)rowBytes << 15;
  record |= (unsigned __int128)lp6 << 35;
  record |= (unsigned __int128)l2 << 71;
  std::vector<uint32_t> words{uint32_t(record), uint32_t(record >> 32), uint32_t(record >> 64)};
  if (twoD) {
    e.fits(rows - 1, 16, "GDMA rows");
    e.fits(lp6Stride, 24, "GDMA LP6 stride");
    e.fits(l2Stride, 24, "GDMA L2 stride");
    uint64_t extension = (rows - 1) | lp6Stride << 16 | l2Stride << 40;
    words.push_back(uint32_t(extension)); words.push_back(uint32_t(extension >> 32));
  }
  auto deps = dependencyWords(e, ordinal, dependencies);
  if (!e.error.empty()) return R::failure(e.error);
  words.insert(words.end(), deps.begin(), deps.end());
  return R::success(std::move(words));
}

Result<std::vector<uint32_t>> coreBeginRecord(uint32_t ordinal, const std::vector<uint32_t> &dependencies,
                                              uint64_t coreWords, unsigned logicalCore, uint64_t l1Bytes) {
  using R = Result<std::vector<uint32_t>>;
  Encoder e;
  if (logicalCore >= kCoreAny) return R::failure("logical core must be below 31 (31 is dynamic placement)");
  const uint64_t l1Units = (l1Bytes + 7) / 8;
  e.fits(l1Units, 20, "L1 size");
  e.fits(coreWords, 24, "core word count");
  e.fits(dependencies.size(), 8, "CORE_BEGIN dependencies");
  // W0 [6] extension (none), [11:7] logical core, [31:12] L1 bytes / 8;
  // W1 [23:0] core word count, [31:24] dependency count.
  std::vector<uint32_t> words{kCoreBegin | logicalCore << 7 | uint32_t(l1Units) << 12,
                              uint32_t(coreWords) | uint32_t(dependencies.size()) << 24};
  auto deps = dependencyWords(e, ordinal, dependencies);
  if (!e.error.empty()) return R::failure(e.error);
  words.insert(words.end(), deps.begin(), deps.end());
  return R::success(std::move(words));
}

Result<std::vector<uint32_t>> barrierRecord(uint32_t ordinal, const std::vector<uint32_t> &dependencies) {
  using R = Result<std::vector<uint32_t>>;
  Encoder e;
  e.fits(dependencies.size(), 8, "BARRIER dependencies");
  std::vector<uint32_t> words{kBarrier | uint32_t(dependencies.size()) << 6};
  auto deps = dependencyWords(e, ordinal, dependencies);
  if (!e.error.empty()) return R::failure(e.error);
  words.insert(words.end(), deps.begin(), deps.end());
  return R::success(std::move(words));
}
template <typename Primitive>
static Result<std::vector<uint32_t>> pimRecordImpl(uint32_t ordinal, const std::vector<uint32_t> &deps,
                                       const Primitive *commands, uint32_t count) {
  using R = Result<std::vector<uint32_t>>;
  if (!count || !commands) return R::failure("PIM record needs executable primitives");
  if (deps.size() > kMaxDependencies) return R::failure("too many PIM dependencies");
  for (uint32_t d : deps)
    if (d >= ordinal || ordinal - d > kMaxDistance) return R::failure("invalid PIM dependency distance");
  // Zero inline count selects a full-width count word. Large token batches
  // must not acquire a new phase-size limit when macros become primitives.
  const uint32_t inlineCount = count < kPimCommands ? count : 0;
  std::vector<uint32_t> words{OP_PIM_PRIMITIVES | uint32_t(deps.size()) << 6 | inlineCount << 14};
  if (!inlineCount) words.push_back(count);
  Encoder encoder;
  for (uint32_t d : dependencyWords(encoder, ordinal, deps)) words.push_back(d);
  if (!encoder.error.empty()) return R::failure(encoder.error);
  for (uint32_t i = 0; i < count; ++i) {
    auto encoded = pim::encode(commands[i]);
    if (!encoded) return R::failure("PIM primitive " + std::to_string(i) + ": " + encoded.error);
    words.insert(words.end(), encoded.value.begin(), encoded.value.end());
  }
  return R::success(std::move(words));
}
Result<std::vector<uint32_t>> pimRecord(uint32_t ordinal, const std::vector<uint32_t> &deps,
                                       const pim::Primitive *commands, uint32_t count) {
  return pimRecordImpl(ordinal, deps, commands, count);
}
Result<std::vector<uint32_t>> pimRecord(uint32_t ordinal, const std::vector<uint32_t> &deps,
                                       const pim::PaPrimitive *commands, uint32_t count) {
  return pimRecordImpl(ordinal, deps, commands, count);
}

} // namespace npu::isa2
