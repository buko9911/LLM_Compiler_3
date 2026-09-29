#include "npu/Target/IsaV2.h"
#include <algorithm>
#include <array>
#include <map>
#include <optional>
#include <string>

namespace npu::isa2 {
namespace {
// ver 2.0 core opcodes, bits [5:0].
constexpr uint32_t C_LI = 0x20, C_SET = 0x21, C_ADDI = 0x22, C_LOOP_BEGIN = 0x23, C_LOOP_END = 0x24,
                   C_LUI = 0x25, V_OP = 0x30, V_CVT = 0x31, V_LD = 0x32, V_ST = 0x33, S_OP = 0x34,
                   S_LD = 0x35, S_ST = 0x36, M_LD = 0x37, LDMA_R = 0x38, LDMA_I = 0x39, M_LD_ACC = 0x3a,
                   M_MMA = 0x3b, M_WB = 0x3c, C_SYNC = 0x3f;
// System records.
constexpr uint32_t CORE_BEGIN = 0x26, GDMA_LOAD = 0x2b, GDMA_STORE = 0x2c, BARRIER = 0x2d;
constexpr uint32_t F16 = 0, F32 = 1, I8 = 2, I32 = 3;
constexpr int SET_ROWS = 0, SET_SRC = 1, SET_DST = 2, SET_VL = 3, SET_BYTES = 4, kSets = 5;
constexpr uint32_t SYNC_EVENT = 4, kMaxEvent = 15, kImm22 = 1u << 22, kWbStrideExtended = 1023;
constexpr uint32_t kVectorWindow = 16u << 16, kScalarWindow = 4u << 18;

uint32_t cLi(uint32_t rd, uint32_t v) { return C_LI | rd << 6 | v << 10; }
uint32_t cLui(uint32_t rd, uint32_t upper) { return C_LUI | rd << 6 | upper << 10; }
uint32_t cAddi(uint32_t rd, uint32_t rs1, uint32_t v) { return C_ADDI | rd << 6 | rs1 << 10 | v << 14; }
uint32_t cSet(uint32_t sel, uint32_t v) { return C_SET | sel << 6 | v << 10; }
uint32_t cSetGp(uint32_t sel, uint32_t gp) { return C_SET | sel << 6 | 1u << 9 | gp << 10; }
uint32_t cSync(uint32_t kind, uint32_t event = 0) { return C_SYNC | kind << 6 | event << 10; }
uint32_t vOp(uint32_t op, uint32_t dt, uint32_t rd, uint32_t rs1, uint32_t rs2 = 0, uint32_t rs3 = 0) {
  return V_OP | op << 6 | dt << 12 | rd << 14 | rs1 << 18 | rs2 << 22 | rs3 << 26;
}
uint32_t vCvt(uint32_t mode, uint32_t src, uint32_t dst, uint32_t rd, uint32_t rs1, uint32_t rs2, uint32_t rs3,
              uint32_t rnd, uint32_t sat) {
  return V_CVT | mode << 6 | src << 10 | dst << 12 | rd << 14 | rs1 << 18 | rs2 << 22 | rs3 << 26 | rnd << 30 |
         sat << 31;
}
uint32_t sOp(uint32_t op, uint32_t rd, uint32_t rs1, uint32_t rs2) {
  return S_OP | op << 6 | rd << 10 | rs1 << 14 | rs2 << 18;
}
void loadConstant(std::vector<uint32_t> &out, uint32_t rd, uint32_t v) {
  if (v < kImm22) { out.push_back(cLi(rd, v)); return; }
  out.push_back(cLui(rd, v >> 12));
  if (v & 0xfff) out.push_back(cAddi(rd, rd, v & 0xfff));
}

struct Lowering {
  std::vector<uint32_t> out;
  std::string error;
  // Builder-stream GP values (unset: only the physical register holds it) and
  // what the lowering knows of the physical registers.
  std::array<std::optional<uint32_t>, 16> value, physical;
  std::array<std::optional<uint32_t>, kSets> control, physControl;
  std::map<uint32_t, uint32_t> events;
  int loopDepth = 0;

  Lowering() {
    value.fill(0u);
    physical.fill(0u);
    control = {1u, 0u, 0u, 0u, 0u};
    physControl = control;
  }
  bool ok() const { return error.empty(); }
  void fail(const std::string &message) { if (ok()) error = message; }

  // A register that may be overwritten: it does not hold a loop value and was
  // not already chosen for another operand of the same instruction.
  int scratch(int home, int keepA = -1, int keepB = -1) {
    auto free = [&](int r) { return value[size_t(r)] && r != keepA && r != keepB; };
    if (free(home)) return home;
    for (int r = 0; r < 16; ++r)
      if (free(r)) return r;
    fail("no GP register is free to hold a constant");
    return home;
  }
  int materialize(uint32_t v, int home, int keepA = -1, int keepB = -1) {
    if (physical[size_t(home)] == v) return home;
    for (int r = 0; r < 16; ++r)
      if (physical[size_t(r)] == v) return r;
    int target = scratch(home, keepA, keepB);
    loadConstant(out, uint32_t(target), v);
    physical[size_t(target)] = v;
    return target;
  }
  int use(int reg, int keep = -1) {
    return value[size_t(reg)] ? materialize(*value[size_t(reg)], reg, keep) : reg;
  }
  // (base register, byte offset) of a scaled-offset memory access.
  std::pair<int, uint32_t> base(int reg, uint32_t unit, uint32_t window) {
    if (!value[size_t(reg)]) return {reg, 0};
    uint32_t address = *value[size_t(reg)], limit = window / unit;
    auto fits = [&](int r) {
      const auto &b = physical[size_t(r)];
      return b && *b <= address && (address - *b) % unit == 0 && (address - *b) / unit < limit;
    };
    if (fits(reg)) return {reg, address - *physical[size_t(reg)]};
    for (int r = 0; r < 16; ++r)
      if (fits(r)) return {r, address - *physical[size_t(r)]};
    uint32_t b = address % unit ? address : address - address % window;
    int target = scratch(reg);
    loadConstant(out, uint32_t(target), b);
    physical[size_t(target)] = b;
    return {target, address - b};
  }
  uint32_t event(int reg) {
    if (!value[size_t(reg)]) { fail("a local LDMA event must be a constant in the builder stream"); return 1; }
    auto it = events.find(*value[size_t(reg)]);
    if (it != events.end()) return it->second;
    if (events.size() == kMaxEvent) { fail("a core block uses more than 15 local LDMA events"); return 1; }
    uint32_t id = uint32_t(events.size()) + 1;
    events[*value[size_t(reg)]] = id;
    return id;
  }

  void need(int sel, std::optional<uint32_t> wanted) {
    if (!wanted) {
      if (physControl[size_t(sel)]) fail("control value set from a loop register was overwritten");
      return;
    }
    if (physControl[size_t(sel)] == wanted) return;
    if (*wanted < kImm22) out.push_back(cSet(uint32_t(sel), *wanted));
    else out.push_back(cSetGp(uint32_t(sel), uint32_t(materialize(*wanted, 15))));
    physControl[size_t(sel)] = wanted;
  }
  void need(int sel) { need(sel, control[size_t(sel)]); }
  // A loop body may change any value per iteration: write every tracked value
  // before the loop and track nothing inside it.
  void syncForLoop() {
    for (size_t r = 0; r < 16; ++r) {
      if (value[r] && physical[r] != value[r]) loadConstant(out, uint32_t(r), *value[r]);
      value[r].reset();
      physical[r].reset();
    }
    for (int s = 0; s < kSets; ++s) {
      if (control[size_t(s)]) need(s);
      control[size_t(s)].reset();
      physControl[size_t(s)].reset();
    }
  }

  void instruction(const uint32_t *w) {
    const uint32_t word = w[0], opcode = word & 63, funct = word >> 22 & 15;
    const int rd = int(word >> 6 & 15), rs1 = int(word >> 10 & 15), rs2 = int(word >> 14 & 15),
              rs3 = int(word >> 18 & 15);
    const bool inLoop = loopDepth > 0;
    switch (opcode) {
    case 0x25: { // C_LUI_U32
      uint32_t upper = word >> 10 & 0xfffff;
      if (inLoop) {
        out.push_back(cLui(uint32_t(rd), upper));
        value[size_t(rd)].reset();
        physical[size_t(rd)].reset();
      } else value[size_t(rd)] = upper << 12;
      return;
    }
    case 0x22: { // C_ADDI_U32
      uint32_t imm = word >> 14;
      if (inLoop || !value[size_t(rs1)]) {
        out.push_back(cAddi(uint32_t(rd), uint32_t(rs1), imm));
        value[size_t(rd)].reset();
        physical[size_t(rd)].reset();
      } else value[size_t(rd)] = *value[size_t(rs1)] + imm;
      return;
    }
    case 0x39: { // C_SET_* gp
      static const int fromFunct[16] = {SET_ROWS, SET_SRC, SET_DST, SET_VL, -1, -1, -1, SET_BYTES,
                                        -1, -1, -1, -1, -1, -1, -1, -1};
      int sel = fromFunct[funct];
      if (sel < 0) return fail("invalid builder-stream C_SET");
      if (!value[size_t(rd)]) {
        out.push_back(cSetGp(uint32_t(sel), uint32_t(rd)));
        control[size_t(sel)].reset();
        physControl[size_t(sel)].reset();
      } else control[size_t(sel)] = value[size_t(rd)];
      return;
    }
    case 0x23: { // C_LOOP_BEGIN gp
      std::optional<uint32_t> count = value[size_t(rd)];
      if (count && (!*count || *count >= 1u << 25)) return fail("hardware loop count outside 1..2^25-1");
      syncForLoop();
      out.push_back(count ? C_LOOP_BEGIN | *count << 7 : C_LOOP_BEGIN | 1u << 6 | uint32_t(rd) << 7);
      ++loopDepth;
      return;
    }
    case 0x24:
      if (!loopDepth) return fail("C_LOOP_END without a matching C_LOOP_BEGIN");
      out.push_back(C_LOOP_END);
      --loopDepth;
      return;
    case 0x3f:
      if (funct < 1 || funct > 4 || (word >> 6 & 0xffff)) return fail("invalid builder-stream wait");
      out.push_back(cSync(funct - 1));
      return;
    case 0x2e:
      out.push_back(cSync(SYNC_EVENT, event(rd)));
      return;
    case 0x38: return ldma(funct, rd, rs1, rs2);
    case 0x37: return matrixLoad(w);
    case 0x3a: {
      uint32_t m = w[1], n = w[2], stride = w[3];
      if (m < 1 || m > 32 || n < 1 || n > 32) return fail("M_LD_ACC shape out of range");
      int gp = use(rs1);
      uint32_t head = M_LD_ACC | uint32_t(funct == 2) << 6 | uint32_t(gp) << 7 | (m - 1) << 11 | (n - 1) << 16;
      if (m == 1 || stride == n * 4) out.push_back(head);
      else out.insert(out.end(), {head | 1u << 21, stride});
      return;
    }
    case 0x3c: {
      uint32_t m = w[1], n = w[2], stride = w[3];
      if (funct != 1 && funct != 2 && funct != 4) return fail("invalid builder-stream M_WRITEOUT");
      if (m < 1 || m > 32 || n < 1 || n > 32) return fail("M_WB shape out of range");
      uint32_t dtype = funct == 1 ? 0 : funct == 4 ? 1 : 2, element = funct == 1 ? 2 : 4;
      int gp = use(rd);
      uint32_t head = M_WB | dtype << 6 | uint32_t(gp) << 8 | (m - 1) << 12 | (n - 1) << 17;
      if (m == 1 || stride == n * element) out.push_back(head);
      else if (stride % 64 == 0 && stride / 64 < kWbStrideExtended) out.push_back(head | (stride / 64) << 22);
      else out.insert(out.end(), {head | kWbStrideExtended << 22, stride});
      return;
    }
    case 0x35: return memory(funct, rd, rs1);
    case 0x30: {
      static const uint32_t op[6] = {0, 0, 1, 2, 4, 5};
      if (funct < 1 || funct > 5) return fail("invalid builder-stream vector binary op");
      need(SET_VL);
      out.push_back(vOp(op[funct], F16, uint32_t(rd), uint32_t(rs1), uint32_t(rs2)));
      return;
    }
    case 0x31: {
      static const uint32_t op[7] = {0, 6, 8, 9, 10, 11, 7};
      if (funct < 1 || funct > 6) return fail("invalid builder-stream vector unary op");
      need(SET_VL);
      out.push_back(vOp(op[funct], F16, uint32_t(rd), uint32_t(rs1)));
      return;
    }
    case 0x32:
      if (funct < 1 || funct > 5) return fail("invalid builder-stream vector-scalar op");
      need(SET_VL);
      out.push_back(vOp(27 + funct, F16, uint32_t(rd), uint32_t(rs1), uint32_t(rs2)));
      return;
    case 0x33: {
      static const uint32_t op[6] = {0, 33, 34, 35, 33, 34};
      static const uint32_t dt[6] = {0, F16, F16, F16, F32, F32};
      if (funct < 1 || funct > 5) return fail("invalid builder-stream reduction");
      need(SET_VL);
      out.push_back(vOp(op[funct], dt[funct], uint32_t(rd), uint32_t(rs1), funct == 3 ? uint32_t(rs2) : 0));
      return;
    }
    case 0x34:
      if (funct < 1 || funct > 8) return fail("invalid builder-stream scalar op");
      out.push_back(sOp(funct - 1, uint32_t(rd), uint32_t(rs1), uint32_t(rs2)));
      return;
    case 0x3d: {
      need(SET_VL);
      uint32_t d = uint32_t(rd), a = uint32_t(rs1), b = uint32_t(rs2), c = uint32_t(rs3);
      switch (funct) {
      case 1: out.push_back(vCvt(2, F16, I8, d, a, b, 0, 1, 1)); return;   // V_QUANT_F16_I8
      case 2: out.push_back(vCvt(6, I8, F16, d, a, b, 0, 1, 0)); return;   // V_DEQUANT_WEIGHT_I8_F16
      case 3: out.push_back(vCvt(7, I32, F16, d, a, b, c, 1, 0)); return;  // V_DEQUANT_I32_F16
      case 4: out.push_back(vCvt(0, F16, F32, d, a, 0, 0, 1, 0)); return;  // V_CAST_F16_F32
      case 5: out.push_back(vCvt(0, F32, F16, d, a, 0, 0, 1, 0)); return;  // V_CAST_F32_F16
      default: return fail("invalid builder-stream conversion");
      }
    }
    case 0x36: return typed(w);
    default: return fail("unsupported builder-stream instruction");
    }
  }

  void matrixLoad(const uint32_t *w) {
    const uint32_t word = w[0], funct = word >> 22 & 15, rows = w[1], columns = w[2], stride = w[3];
    if (funct != 1 && funct != 3 && funct != 5 && funct != 7) return fail("invalid builder-stream M_LOAD");
    const uint32_t kind = funct == 1 ? 0 : funct == 3 ? 1 : funct == 5 ? 2 : 3;
    const uint32_t element = (funct == 1 || funct == 5) ? 1 : 2;
    const bool weight = funct == 1 || funct == 3;
    const uint32_t k = weight ? rows : columns, mn = weight ? columns : rows;
    if (mn < 1 || mn > 32 || k < 1 || k > (1u << 14)) return fail("M_LD shape out of range");
    int gp = use(int(word >> 10 & 15));
    uint32_t head = M_LD | kind << 6 | uint32_t(gp) << 8 | (mn - 1) << 12 | (k - 1) << 17;
    if (rows == 1 || stride == columns * element) out.push_back(head);
    else out.insert(out.end(), {head | 1u << 31, stride});
  }

  void memory(uint32_t funct, int rd, int rs1) {
    if (funct < 1 || funct > 10) return fail("invalid builder-stream load/store");
    const bool store = funct % 2 == 0;
    const int addressReg = store ? rd : rs1, data = store ? rs1 : rd;
    if (funct >= 9) {
      auto [b, offset] = base(addressReg, 4, kScalarWindow);
      out.push_back((store ? S_ST : S_LD) | uint32_t(data) << 6 | uint32_t(b) << 10 | (offset / 4) << 14);
      return;
    }
    need(SET_VL);
    auto [b, offset] = base(addressReg, 16, kVectorWindow);
    out.push_back((store ? V_ST : V_LD) | ((funct - 1) / 2) << 6 | uint32_t(data) << 8 | uint32_t(b) << 12 |
                  (offset / 16) << 16);
  }

  void ldma(uint32_t funct, int rd, int rs1, int rs2) {
    if (funct > 13) return fail("invalid builder-stream LDMA");
    const bool tagged = funct >= 10;
    const bool strided = funct == 4 || funct == 5 || funct == 6 || funct == 7 || funct == 9 || funct == 12 ||
                         funct == 13;
    const bool immediate = funct <= 3 || funct == 8 || funct == 10 || funct == 11;
    const uint32_t ev = tagged ? event(rs2) : 0;
    const std::optional<uint32_t> l1 = value[size_t(rd)], l2 = value[size_t(rs1)], size = control[SET_BYTES];
    if (immediate && l1 && l2 && size && *l1 % 32 == 0 && *l1 < (1u << 23) && *l2 % 32 == 0 &&
        *l2 < (1u << 23) && *size % 32 == 0 && *size > 0 && *size < (1u << 19)) {
      out.insert(out.end(), {LDMA_I | funct << 6 | ev << 10 | (*l1 / 32) << 14, *l2 / 32 | (*size / 32) << 18});
      return;
    }
    need(SET_BYTES);
    if (strided) {
      need(SET_ROWS);
      need(SET_SRC);
      need(SET_DST);
    }
    int first = use(rd);
    int second = use(rs1, first);
    int eventReg = ev ? materialize(ev, rs2, first, second) : 0;
    out.push_back(LDMA_R | funct << 6 | uint32_t(first) << 10 | uint32_t(second) << 14 | uint32_t(eventReg) << 18);
  }

  // Typed vector record (0x36): [header, op, dtype, elements].
  void typed(const uint32_t *w) {
    const uint32_t word = w[0], op = w[1], dtype = w[2], elements = w[3];
    uint32_t rd = word >> 6 & 15, rs1 = word >> 10 & 15, rs2 = word >> 14 & 15, rs3 = word >> 18 & 15;
    if (dtype > 3) return fail("invalid typed vector dtype");
    need(SET_VL, elements);
    const bool integer = dtype == I8 || dtype == I32;
    if (op <= 26) {
      if (op == 19) rs2 = uint32_t(use(int(rs2)));   // SLICE: GP element offset
      out.push_back(vOp(op, dtype, rd, rs1, rs2, rs3));
    } else if (op >= 31 && op <= 36) {
      if (integer) {                                  // integer scalar operands are GPs
        if (op == 31) rs1 = uint32_t(use(int(rs1)));
        else rs2 = uint32_t(use(int(rs2)));
      }
      out.push_back(vOp(op - 4, dtype, rd, rs1, rs2, rs3));
    } else if (op >= 27 && op <= 30) {
      static const uint32_t target[4] = {F16, F32, I8, I32};
      const uint32_t to = target[op - 27];
      const bool floatOut = to == F16 || to == F32, floatIn = dtype == F16 || dtype == F32;
      out.push_back(vCvt(0, dtype, to, rd, rs1, 0, 0, floatOut ? 1 : 0, !floatOut && floatIn ? 1 : 0));
    } else if (op == 37) {
      out.push_back(vCvt(4, dtype, I8, rd, rs1, rs2, rs3, 1, 1));
    } else if (op == 38 || op == 39) {
      out.push_back(vCvt(8, dtype, op == 38 ? F32 : F16, rd, rs1, rs2, rs3, 1, 0));
    } else if (op == 40) {
      out.push_back(vCvt(1, dtype == F32 ? I32 : F32, dtype, rd, rs1, 0, 0, 0, 0));
    } else fail("unsupported typed vector operation");
  }
};

uint32_t field(uint64_t v, unsigned bits) { return uint32_t(v & ((uint64_t(1) << bits) - 1)); }

std::vector<uint32_t> dependencyWords(uint32_t ordinal, const std::vector<uint32_t> &deps) {
  std::vector<uint32_t> words;
  for (size_t i = 0; i < deps.size(); i += 2) {
    uint32_t low = ordinal - deps[i];
    uint32_t high = i + 1 < deps.size() ? ordinal - deps[i + 1] : 0;
    words.push_back(low | high << 16);
  }
  return words;
}
} // namespace

Result<std::vector<uint32_t>> lowerCoreBlock(const std::vector<uint32_t> &in) {
  using R = Result<std::vector<uint32_t>>;
  Lowering l;
  size_t i = 0;
  while (i < in.size() && l.ok()) {
    const uint32_t word = in[i], opcode = word & 63;
    if (opcode == 0x3b) {
      // A run of one-word M_MMA slices of one rectangle -> one M_MMA count.
      const uint32_t mode = word >> 22 & 15;
      if ((word >> 27) || (mode != 1 && mode != 3) || (word >> 6 & 0xffff))
        return R::failure("invalid builder-stream M_MMA");
      bool accumulate = word >> 26 & 1;
      uint32_t count = 0;
      while (i < in.size() && (in[i] & 63) == 0x3b && (in[i] >> 22 & 15) == mode) {
        if (count && !(in[i] >> 26 & 1)) break;
        ++count;
        ++i;
      }
      while (count) {
        const uint32_t chunk = std::min<uint32_t>(count, 1023);
        l.out.push_back(M_MMA | uint32_t(mode == 3) << 6 | uint32_t(accumulate) << 7 | chunk << 8);
        accumulate = true;
        count -= chunk;
      }
      continue;
    }
    const size_t length = (opcode == 0x36 || opcode == 0x37 || opcode == 0x3a || opcode == 0x3c) ? 4 : 1;
    if (i + length > in.size()) return R::failure("truncated four-word builder-stream record");
    l.instruction(&in[i]);
    i += length;
  }
  if (l.ok() && l.loopDepth) l.fail("unterminated hardware loop");
  if (!l.ok()) return R::failure(l.error);
  return R::success(std::move(l.out));
}

unsigned instructionWords(uint32_t word) {
  const uint32_t opcode = word & 63;
  if (opcode == LDMA_I) return 2;
  if (opcode == M_LD && word >> 31) return 2;
  if (opcode == M_LD_ACC && (word >> 21 & 1)) return 2;
  if (opcode == M_WB && word >> 22 == kWbStrideExtended) return 2;
  return 1;
}

Result<bool> verifyCoreBlock(const uint32_t *words, size_t count) {
  using R = Result<bool>;
  unsigned depth = 0;
  size_t loopStart[4] = {};
  for (size_t i = 0; i < count; i += instructionWords(words[i])) {
    const uint32_t opcode = words[i] & 63;
    const bool known = (opcode >= C_LI && opcode <= C_LUI) || (opcode >= V_OP && opcode <= M_WB) ||
                       opcode == C_SYNC;
    if (!known) return R::failure("invalid ISA ver 2.0 opcode at core word " + std::to_string(i));
    if (i + instructionWords(words[i]) > count) return R::failure("truncated ISA ver 2.0 instruction");
    if (opcode == C_LOOP_BEGIN) {
      if (depth == 4) return R::failure("hardware loop nesting exceeds four");
      loopStart[depth++] = i;
    } else if (opcode == C_LOOP_END) {
      if (!depth) return R::failure("C_LOOP_END without a matching C_LOOP_BEGIN");
      if (loopStart[--depth] + 1 == i) return R::failure("empty hardware loop body");
    }
  }
  if (depth) return R::failure("unterminated hardware loop");
  return R::success(true);
}

std::vector<uint32_t> gdmaRecord(bool load, uint32_t ordinal, const std::vector<uint32_t> &deps, uint64_t lp6,
                                 uint64_t l2, uint32_t rowBytes, uint32_t rows, uint32_t lp6Stride,
                                 uint32_t l2Stride) {
  // 96 bits: [5:0] opcode, [6] 2-D, [14:7] dependencies, [34:15] row bytes,
  // [70:35] LP6 byte address, [94:71] L2 byte address.
  const bool twoD = rows != 1 || lp6Stride != rowBytes || l2Stride != rowBytes;
  unsigned __int128 r = (load ? GDMA_LOAD : GDMA_STORE) | uint32_t(twoD) << 6 | field(deps.size(), 8) << 7;
  r |= (unsigned __int128)field(rowBytes, 20) << 15;
  r |= (unsigned __int128)(lp6 & (kGdmaLp6 - 1)) << 35;
  r |= (unsigned __int128)field(l2, 24) << 71;
  std::vector<uint32_t> words{uint32_t(r), uint32_t(r >> 32), uint32_t(r >> 64)};
  if (twoD) {
    // [15:0] rows - 1, [39:16] LP6 row stride, [63:40] L2 row stride.
    uint64_t e = field(rows - 1, 16) | uint64_t(field(lp6Stride, 24)) << 16 | uint64_t(field(l2Stride, 24)) << 40;
    words.push_back(uint32_t(e));
    words.push_back(uint32_t(e >> 32));
  }
  for (uint32_t d : dependencyWords(ordinal, deps)) words.push_back(d);
  return words;
}

std::vector<uint32_t> coreBeginRecord(uint32_t ordinal, const std::vector<uint32_t> &deps, uint32_t coreWords,
                                      uint32_t logicalCore, uint64_t l1Bytes) {
  // W0 [6] extension (0), [11:7] logical core, [31:12] L1 bytes / 8;
  // W1 [23:0] core words, [31:24] dependencies.
  std::vector<uint32_t> words{CORE_BEGIN | field(logicalCore, 5) << 7 | field((l1Bytes + 7) / 8, 20) << 12,
                              field(coreWords, 24) | field(deps.size(), 8) << 24};
  for (uint32_t d : dependencyWords(ordinal, deps)) words.push_back(d);
  return words;
}

std::vector<uint32_t> barrierRecord(uint32_t ordinal, const std::vector<uint32_t> &deps) {
  std::vector<uint32_t> words{BARRIER | field(deps.size(), 8) << 6};
  for (uint32_t d : dependencyWords(ordinal, deps)) words.push_back(d);
  return words;
}

Result<std::pair<uint32_t, uint32_t>> pimGemv(unsigned partition, uint32_t sweeps, uint32_t weightTile,
                                              uint32_t activationTile, uint32_t outputTile) {
  using R = Result<std::pair<uint32_t, uint32_t>>;
  if (partition > 3 || sweeps < 1 || sweeps > 256) return R::failure("PIM_GEMV partition or sweeps out of range");
  if (weightTile % 512 || activationTile % 32 || outputTile % 8)
    return R::failure("PIM_GEMV tiles must be multiples of 512 (weight), 32 (activation) and 8 (output)");
  if (weightTile / 512 >= 1u << 18 || activationTile / 32 >= 1u << 16 || outputTile / 8 >= 1u << 16)
    return R::failure("PIM_GEMV tile id does not fit its field");
  // W0 [3:0] type 0, [5:4] partition, [13:6] sweeps - 1, [31:14] weight tile / 512;
  // W1 [15:0] activation tile / 32, [31:16] output tile / 8.
  return R::success({partition << 4 | (sweeps - 1) << 6 | (weightTile / 512) << 14,
                     activationTile / 32 | (outputTile / 8) << 16});
}
} // namespace npu::isa2
