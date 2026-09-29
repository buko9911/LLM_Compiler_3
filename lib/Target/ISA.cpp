#include "npu/Target/ISA.h"
#include <algorithm>
#include <charconv>
#include <cctype>
#include <iomanip>
#include <sstream>
#include <array>
#include <unordered_map>

namespace npu {
namespace {
enum class Slot { Rd = 6, Rs1 = 10, Rs2 = 14, Rs3 = 18 };
struct Spec { const char *name; uint32_t opcode, funct; std::vector<Slot> slots; };
const std::vector<Spec> specs = {
#include "ISASpec.inc"
};
struct Decoding { const Spec *spec = nullptr; uint32_t mask = 0; };
const auto decoding = [] {
  std::array<Decoding,1024> table{};
  for (const auto &s : specs) {
    auto &entry = table[s.opcode | s.funct << 6];
    entry.spec = &s;
    for (auto slot : s.slots) entry.mask |= 15u << unsigned(slot);
  }
  return table;
}();
const auto encoding = [] {
  std::unordered_map<std::string_view,const Spec *> table;
  for (const auto &s : specs) table.emplace(s.name,&s);
  return table;
}();
const Spec *decode(uint32_t word) {
  const auto &entry = decoding[(word & 63) | ((word >> 22) & 15) << 6];
  if (!entry.spec || (word & ~entry.mask) != (entry.spec->opcode | entry.spec->funct << 22))
    return nullptr;
  return entry.spec;
}

// ── Matrix records (ISA ver 1.0) ──────────────────────────────────────────
constexpr uint32_t kMatrixLoad = 0x37, kMma = 0x3b, kWriteout = 0x3c;

struct MatrixName { const char *name; uint32_t opcode, funct; };
// funct: load 1/3 weight I8/F16, 5/7 activation I8/F16; MMA 1/3 I8/F16;
// writeout 1/2/4 F16/I32/F32. Matches src/op.rs decode_matrix_load/decode_record.
const MatrixName matrixNames[] = {
    {"M_LOAD_WEIGHT_I8", kMatrixLoad, 1}, {"M_LOAD_WEIGHT_F16", kMatrixLoad, 3},
    {"M_LOAD_ACT_I8", kMatrixLoad, 5},    {"M_LOAD_ACT_F16", kMatrixLoad, 7},
    {"M_MMA_I8I8I32", kMma, 1},           {"M_MMA_F16F16F32", kMma, 3},
    {"M_WRITEOUT_F16", kWriteout, 1},     {"M_WRITEOUT_I32", kWriteout, 2},
    {"M_WRITEOUT_F32", kWriteout, 4},
};
const MatrixName *matrixByName(std::string_view name) {
  for (const auto &m : matrixNames) if (name == m.name) return &m;
  return nullptr;
}
const MatrixName *matrixByWord(uint32_t word) {
  for (const auto &m : matrixNames)
    if ((word & 63) == m.opcode && ((word >> 22) & 15) == m.funct) return &m;
  return nullptr;
}
uint32_t field(uint32_t word, unsigned shift) { return (word >> shift) & 15; }

Result<bool> checkMatrixRecord(const uint32_t *w) {
  using R = Result<bool>;
  const auto *m = matrixByWord(w[0]);
  if (!m) return R::failure("invalid matrix record header");
  const uint32_t rd = field(w[0],6), rs1 = field(w[0],10), rs2 = field(w[0],14), rs3 = field(w[0],18);
  if (m->opcode == kMma) {
    if (w[0] >> 27 || rd || rs1 || rs2 || rs3) return R::failure("M_MMA has reserved fields set");
    return R::success(true);
  }
  if (w[0] >> 26) return R::failure("matrix record header uses reserved bits");
  const uint32_t rows = w[1], columns = w[2], stride = w[3];
  if (!rows || !columns) return R::failure("matrix extents must be positive");
  if (m->opcode == kMatrixLoad) {
    if (rd || rs2 || rs3) return R::failure("M_LOAD uses only rs1 as the address register");
    const uint32_t element = (m->funct == 1 || m->funct == 5) ? 1 : 2;
    const bool weight = m->funct <= 3;
    // W is [K,N] and A is [M,K]; the array bounds N and M, K is temporal.
    if ((weight ? columns : rows) > 32) return R::failure("matrix operand exceeds the 32-wide array");
    if (uint64_t(stride) < uint64_t(columns) * element || stride % element)
      return R::failure("invalid matrix operand byte stride");
    return R::success(true);
  }
  if (rs1 || rs2 || rs3) return R::failure("M_WRITEOUT uses only rd as the address register");
  const uint32_t element = m->funct == 1 ? 2 : 4;
  if (rows > 32 || columns > 32) return R::failure("writeout exceeds the 32x32 accumulator");
  if (uint64_t(stride) < uint64_t(columns) * element || stride % element)
    return R::failure("invalid matrix output byte stride");
  return R::success(true);
}

std::string_view trim(std::string_view v) {
  while (!v.empty() && std::isspace(static_cast<unsigned char>(v.front()))) v.remove_prefix(1);
  while (!v.empty() && std::isspace(static_cast<unsigned char>(v.back()))) v.remove_suffix(1);
  return v;
}
std::string_view uncomment(std::string_view v) {
  return trim(v.substr(0, std::min(v.find(';'), v.find("//"))));
}
Result<uint32_t> number(std::string_view v) {
  v = trim(v);
  if (v.empty()) return Result<uint32_t>::failure("expected unsigned 32-bit integer");
  int base = 10;
  if (v.size() > 2 && v[0] == '0' && (v[1] == 'x' || v[1] == 'X')) {
    base = 16; v.remove_prefix(2);
  }
  uint32_t value = 0;
  auto r = std::from_chars(v.data(), v.data() + v.size(), value, base);
  if (v.empty() || r.ec != std::errc() || r.ptr != v.data() + v.size())
    return Result<uint32_t>::failure("expected unsigned 32-bit integer");
  return Result<uint32_t>::success(value);
}
std::string text(const std::string &name, const std::vector<uint32_t> &operands) {
  std::string out = name;
  for (size_t i = 0; i < operands.size(); ++i)
    out += (i ? ", " : " ") + std::to_string(operands[i]);
  return out;
}
struct Parsed { std::string name; std::vector<std::string> operands; };
Parsed parse(std::string_view line) {
  line = uncomment(line);
  auto at = line.find_first_of(" \t\r\n");
  Parsed p{std::string(line.substr(0, at)), {}};
  std::transform(p.name.begin(), p.name.end(), p.name.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  auto rest = at == line.npos ? std::string_view{} : trim(line.substr(at));
  while (!rest.empty()) {
    auto comma = rest.find(',');
    p.operands.emplace_back(trim(rest.substr(0, comma)));
    if (comma == rest.npos) break;
    rest = trim(rest.substr(comma + 1));
    if (rest.empty()) { p.operands.emplace_back(); break; }
  }
  return p;
}
} // namespace

Result<std::vector<uint32_t>> encodeMatrixLoad(MatrixOperand operand, MatrixType type,
                                               unsigned addressRegister, uint32_t rows,
                                               uint32_t columns, uint32_t strideBytes) {
  using R = Result<std::vector<uint32_t>>;
  if (addressRegister > 15) return R::failure("register index outside 0..15");
  const uint32_t funct = (operand == MatrixOperand::Weight ? 1 : 5) + (type == MatrixType::F16 ? 2 : 0);
  std::vector<uint32_t> w{kMatrixLoad | addressRegister << 10 | funct << 22, rows, columns, strideBytes};
  auto ok = checkMatrixRecord(w.data());
  if (!ok) return R::failure(ok.error);
  return R::success(std::move(w));
}
uint32_t encodeMma(MatrixType type, bool accumulate) {
  return kMma | (type == MatrixType::F16 ? 3u : 1u) << 22 | uint32_t(accumulate) << 26;
}
Result<std::vector<uint32_t>> encodeMatrixWriteout(WriteoutType type, unsigned addressRegister,
                                                   uint32_t rows, uint32_t columns,
                                                   uint32_t strideBytes) {
  using R = Result<std::vector<uint32_t>>;
  if (addressRegister > 15) return R::failure("register index outside 0..15");
  const uint32_t funct = type == WriteoutType::F16 ? 1 : type == WriteoutType::I32 ? 2 : 4;
  std::vector<uint32_t> w{kWriteout | addressRegister << 6 | funct << 22, rows, columns, strideBytes};
  auto ok = checkMatrixRecord(w.data());
  if (!ok) return R::failure(ok.error);
  return R::success(std::move(w));
}
std::vector<uint32_t> mmaSequence(MatrixType type, uint32_t k, bool accumulateFirst) {
  std::vector<uint32_t> out;
  const uint32_t slices = std::max<uint32_t>(1, (k + kMmaTileK - 1) / kMmaTileK);
  for (uint32_t i = 0; i < slices; ++i) out.push_back(encodeMma(type, accumulateFirst || i > 0));
  return out;
}

unsigned recordWords(uint32_t header) {
  const uint32_t op = header & 63;
  // 0x36 typed vector and 0x3a M_LOAD_ACC are four-word too; this compiler
  // does not emit them, but a stream walker must still step over them.
  return op == kMatrixLoad || op == kWriteout || op == 0x36 || op == 0x3a ? 4 : 1;
}

Result<std::vector<uint32_t>> assembleInstruction(std::string_view line) {
  using R = Result<std::vector<uint32_t>>;
  auto p = parse(line);
  if (const auto *m = matrixByName(p.name)) {
    if (m->opcode == kMma) {
      if (p.operands.size() != 1) return R::failure(p.name + " takes one operand: INIT or ACC");
      std::string mode = p.operands[0];
      std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c) { return std::toupper(c); });
      if (mode != "INIT" && mode != "ACC") return R::failure(p.name + " takes one operand: INIT or ACC");
      return R::success({encodeMma(m->funct == 3 ? MatrixType::F16 : MatrixType::I8, mode == "ACC")});
    }
    if (p.operands.size() != 4) return R::failure(p.name + " requires register, rows, columns, stride_bytes");
    std::array<uint32_t,4> v{};
    for (size_t i = 0; i < 4; ++i) {
      auto n = number(p.operands[i]);
      if (!n) return R::failure(p.name + ": " + n.error);
      v[i] = n.value;
    }
    if (v[0] > 15) return R::failure(p.name + ": register index outside 0..15");
    const uint32_t header = m->opcode == kMatrixLoad ? (m->opcode | v[0] << 10 | m->funct << 22)
                                                     : (m->opcode | v[0] << 6 | m->funct << 22);
    std::vector<uint32_t> w{header, v[1], v[2], v[3]};
    auto ok = checkMatrixRecord(w.data());
    if (!ok) return R::failure(p.name + ": " + ok.error);
    return R::success(std::move(w));
  }
  std::vector<uint32_t> args;
  for (const auto &operand : p.operands) {
    auto n = number(operand);
    if (!n) return R::failure(p.name + ": " + n.error);
    args.push_back(n.value);
  }
  if (p.name == "C_ADDI_U32") {
    if (args.size() != 3 || args[0] > 15 || args[1] > 15 || args[2] >= (1u << 18))
      return R::failure("C_ADDI_U32 requires rd4, rs1_4, imm18");
    return R::success({0x22 | args[0] << 6 | args[1] << 10 | args[2] << 14});
  }
  if (p.name == "C_LUI_U32") {
    if (args.size() != 2 || args[0] > 15 || args[1] >= (1u << 20))
      return R::failure("C_LUI_U32 requires rd4, imm20");
    return R::success({0x25 | args[0] << 6 | args[1] << 10});
  }
  if (auto found = encoding.find(p.name); found != encoding.end()) {
    const auto &s = *found->second;
    if (args.size() != s.slots.size()) return R::failure(p.name + ": wrong operand count");
    uint32_t word = s.opcode | s.funct << 22;
    for (size_t i = 0; i < args.size(); ++i) {
      if (args[i] > 15) return R::failure(p.name + ": register index outside 0..15");
      word |= args[i] << unsigned(s.slots[i]);
    }
    return R::success({word});
  }
  return R::failure("unknown instruction: " + p.name);
}

Result<uint32_t> assembleLine(std::string_view line) {
  auto r = assembleInstruction(line);
  if (!r) return Result<uint32_t>::failure(r.error);
  if (r.value.size() != 1)
    return Result<uint32_t>::failure("multi-word matrix record; use assembleInstruction");
  return Result<uint32_t>::success(r.value.front());
}

Result<std::string> disassembleRecord(const uint32_t *words, size_t available) {
  using R = Result<std::string>;
  if (!available) return R::failure("missing instruction header");
  const uint32_t word = words[0];
  if (const auto *m = matrixByWord(word)) {
    const unsigned n = recordWords(word);
    if (available < n) return R::failure("truncated matrix record");
    auto ok = checkMatrixRecord(words);
    if (!ok) return R::failure(ok.error);
    if (m->opcode == kMma) return R::success(std::string(m->name) + (word >> 26 & 1 ? " ACC" : " INIT"));
    const uint32_t reg = m->opcode == kMatrixLoad ? field(word,10) : field(word,6);
    return R::success(text(m->name, {reg, words[1], words[2], words[3]}));
  }
  auto opcode = word & 63;
  if (opcode == 0x22) return R::success(text("C_ADDI_U32", {(word >> 6) & 15, (word >> 10) & 15, word >> 14}));
  if (opcode == 0x25 && word >> 30 == 0)
    return R::success(text("C_LUI_U32", {(word >> 6) & 15, (word >> 10) & 0xfffff}));
  if (const auto *s = decode(word)) {
    std::vector<uint32_t> args;
    for (auto slot : s->slots) args.push_back((word >> unsigned(slot)) & 15);
    return R::success(text(s->name,args));
  }
  return R::failure("invalid or reserved core instruction");
}

Result<std::string> disassemble(uint32_t word) {
  if (recordWords(word) != 1) return Result<std::string>::failure("multi-word matrix record");
  return disassembleRecord(&word, 1);
}

bool isValidInstruction(uint32_t word) {
  return recordWords(word) == 1 && disassemble(word);
}

Result<std::vector<std::vector<uint32_t>>> splitRecords(const std::vector<uint32_t> &words) {
  using R = Result<std::vector<std::vector<uint32_t>>>;
  std::vector<std::vector<uint32_t>> out;
  for (size_t i = 0; i < words.size();) {
    const unsigned n = recordWords(words[i]);
    auto ok = disassembleRecord(words.data() + i, words.size() - i);
    if (!ok) return R::failure("word " + std::to_string(i) + ": " + ok.error);
    out.emplace_back(words.begin() + i, words.begin() + i + n);
    i += n;
  }
  return R::success(std::move(out));
}

Result<std::vector<uint32_t>> assemble(std::string_view source) {
  using R = Result<std::vector<uint32_t>>;
  std::istringstream in{std::string(source)};
  std::vector<uint32_t> words;
  std::string line;
  unsigned lineNo = 0;
  while (std::getline(in, line)) {
    ++lineNo;
    if (uncomment(line).empty()) continue;
    auto r = assembleInstruction(line);
    if (!r) return R::failure("line " + std::to_string(lineNo) + ": " + r.error);
    words.insert(words.end(), r.value.begin(), r.value.end());
  }
  return R::success(std::move(words));
}
Result<std::vector<uint32_t>> parseMem(std::string_view source) {
  using R = Result<std::vector<uint32_t>>;
  std::istringstream in{std::string(source)};
  std::vector<uint32_t> words;
  std::string line;
  unsigned lineNo = 0;
  while (std::getline(in, line)) {
    ++lineNo;
    if (uncomment(line).empty()) continue;
    auto r = number(uncomment(line));
    if (!r) return R::failure("invalid core word on line " + std::to_string(lineNo));
    words.push_back(r.value);
  }
  auto records = splitRecords(words);
  if (!records) return R::failure(records.error);
  return R::success(std::move(words));
}
std::string formatMem(const std::vector<uint32_t> &words) {
  std::ostringstream out;
  for (auto w : words) out << "0x" << std::hex << std::setw(8) << std::setfill('0') << w << '\n';
  return out.str();
}
Result<std::vector<uint32_t>> materialize(unsigned reg, uint32_t value) {
  using R = Result<std::vector<uint32_t>>;
  if (reg > 15) return R::failure("register index outside 0..15");
  std::vector<uint32_t> words{0x25u | reg << 6 | (value >> 12) << 10};
  if (value & 0xfff) words.push_back(0x22u | reg << 6 | reg << 10 | (value & 0xfff) << 14);
  return R::success(std::move(words));
}
std::vector<InstructionInfo> instructionSet() {
  std::vector<InstructionInfo> out{{"C_ADDI_U32", 3}, {"C_LUI_U32", 2}};
  for (const auto &s : specs) out.push_back({s.name, unsigned(s.slots.size())});
  for (const auto &m : matrixNames) out.push_back({m.name, m.opcode == kMma ? 1u : 4u});
  return out;
}
} // namespace npu
