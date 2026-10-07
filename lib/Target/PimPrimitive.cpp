#include "npu/Target/PimPrimitive.h"
#include <limits>
#include <sstream>

namespace npu::pim {
namespace {
// Indices in the canonical operand tuple (header at index 0).
std::vector<unsigned> operands(Opcode op) {
  switch (op) {
  case Opcode::LoadActivation: return {2};
  case Opcode::LoadScale: return {3};
  case Opcode::ClearOutput: return {4};
  case Opcode::StoreOutput: return {4, 5};
  case Opcode::GemvTile: return {1, 2, 3, 4};
  case Opcode::GemvStream: return {1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14};
  }
  return {};
}
} // namespace

size_t instructionWords(uint32_t header) {
  if (header >> 24 == kPaPrimitiveVersion) {
    switch (Opcode(header & 255)) {
    case Opcode::LoadActivation: case Opcode::LoadScale:
    case Opcode::ClearOutput: case Opcode::StoreOutput: return 4;
    case Opcode::GemvTile: return 6;
    case Opcode::GemvStream: return 13;
    }
    return 0;
  }
  if (header >> 24 != kPrimitiveVersion) return 0;
  switch (Opcode(header & 255)) {
  case Opcode::LoadActivation: case Opcode::LoadScale: case Opcode::ClearOutput: return 2;
  case Opcode::StoreOutput: return 3;
  case Opcode::GemvTile: return 5;
  case Opcode::GemvStream: return 14;
  }
  return 0;
}
const char *opcodeName(Opcode op) {
  switch (op) {
  case Opcode::LoadActivation: return "LOAD_ACTIVATION_TILE";
  case Opcode::LoadScale: return "LOAD_SCALE_TILE";
  case Opcode::ClearOutput: return "CLEAR_OUTPUT";
  case Opcode::GemvTile: return "GEMV_TILE";
  case Opcode::GemvStream: return "GEMV_STREAM";
  case Opcode::StoreOutput: return "STORE_OUTPUT";
  }
  return "INVALID";
}

Result<bool> verify(const Primitive &p) {
  using R = Result<bool>;
  if (p.partition > 255 || p.bankSet > 1 || p.bank < -1 || p.bank > 31)
    return R::failure("PIM primitive partition or bank selection out of range");
  if (p.weight < -1 || p.activation < -1 || p.scale < -1 || p.accumulator < -1 || p.output < -1)
    return R::failure("PIM tile IDs must be nonnegative or -1 (unused)");
  bool valid = false;
  switch (p.opcode) {
  case Opcode::LoadActivation: valid = p.activation >= 0; break;
  case Opcode::LoadScale: valid = p.scale >= 0; break;
  case Opcode::ClearOutput: valid = p.accumulator >= 0; break;
  case Opcode::StoreOutput: valid = p.accumulator >= 0 && p.output >= 0; break;
  case Opcode::GemvTile: valid = p.weight >= 0 && p.activation >= 0 && p.accumulator >= 0; break;
  case Opcode::GemvStream: {
    auto fits = [](int32_t base, int32_t n, int32_t stride) {
      return base >= 0 && n > 0 && stride >= 0 &&
             int64_t(base) + int64_t(n - 1) * stride <= std::numeric_limits<int32_t>::max();
    };
    valid = fits(p.weight, p.inputs, p.weightInputStride) &&
            fits(p.weight, p.outputs, p.weightOutputStride) &&
            fits(p.activation, p.inputs, p.activationInputStride) &&
            fits(p.accumulator, p.outputs, p.accumulatorOutputStride) && p.scaleInputPeriod > 0 &&
            p.scaleInputStride >= 0 && p.scaleOutputStride >= 0;
    if (valid) valid = int64_t(p.weight) + int64_t(p.inputs - 1) * p.weightInputStride +
                          int64_t(p.outputs - 1) * p.weightOutputStride <= std::numeric_limits<int32_t>::max();
    if (valid && p.scale >= 0)
      valid = int64_t(p.scale) + int64_t((p.inputs - 1) / p.scaleInputPeriod) * p.scaleInputStride +
                  int64_t(p.outputs - 1) * p.scaleOutputStride <= std::numeric_limits<int32_t>::max();
    break;
  }
  default: return R::failure("unknown executable PIM primitive (STAGE is not executable)");
  }
  return valid ? R::success(true) : R::failure(std::string(opcodeName(p.opcode)) + ": missing or invalid operands");
}

Result<EncodedPrimitive> encode(const Primitive &p) {
  using R = Result<EncodedPrimitive>;
  if (auto ok = verify(p); !ok) return R::failure(ok.error);
  const std::array<uint32_t, 15> all{kPrimitiveVersion << 24 | uint32_t(p.bank + 1) << 18 | p.bankSet << 16 |
                         p.partition << 8 | uint32_t(p.opcode),
                     uint32_t(p.weight), uint32_t(p.activation), uint32_t(p.scale),
                     uint32_t(p.accumulator), uint32_t(p.output), uint32_t(p.inputs), uint32_t(p.outputs),
                     uint32_t(p.weightInputStride), uint32_t(p.weightOutputStride),
                     uint32_t(p.activationInputStride), uint32_t(p.scaleInputPeriod),
                     uint32_t(p.scaleInputStride), uint32_t(p.scaleOutputStride),
                     uint32_t(p.accumulatorOutputStride)};
  EncodedPrimitive words{all[0]};
  for (auto i : operands(p.opcode)) words.push_back(all[i]);
  return R::success(std::move(words));
}

Result<Primitive> decode(const uint32_t *words, size_t count) {
  using R = Result<Primitive>;
  if (!count || words[0] >> 24 != kPrimitiveVersion || !instructionWords(words[0]) || count != instructionWords(words[0]))
    return R::failure("invalid PIM primitive ABI version or instruction length");
  std::array<uint32_t, 15> w{words[0], UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX,
                            UINT32_MAX, 0, 0, 0, 0, 0, 1, 0, 0, 0};
  size_t at = 1;
  for (auto i : operands(Opcode(words[0] & 255))) w[i] = words[at++];
  Primitive p;
  p.opcode = Opcode(w[0] & 255); p.partition = w[0] >> 8 & 255;
  p.bankSet = w[0] >> 16 & 3; p.bank = int32_t(w[0] >> 18 & 63) - 1;
  p.weight = int32_t(w[1]); p.activation = int32_t(w[2]); p.scale = int32_t(w[3]);
  p.accumulator = int32_t(w[4]); p.output = int32_t(w[5]);
  p.inputs = int32_t(w[6]); p.outputs = int32_t(w[7]);
  p.weightInputStride = int32_t(w[8]); p.weightOutputStride = int32_t(w[9]);
  p.activationInputStride = int32_t(w[10]); p.scaleInputPeriod = int32_t(w[11]);
  p.scaleInputStride = int32_t(w[12]); p.scaleOutputStride = int32_t(w[13]);
  p.accumulatorOutputStride = int32_t(w[14]);
  if (auto ok = verify(p); !ok) return R::failure(ok.error);
  return R::success(p);
}

std::string print(const Primitive &p) {
  std::ostringstream out;
  out << opcodeName(p.opcode) << " partition=" << p.partition << " bank_set=" << p.bankSet << " bank=" << p.bank;
  if (p.weight >= 0) out << " weight_tile=" << p.weight;
  if (p.activation >= 0) out << " activation_tile=" << p.activation;
  if (p.scale >= 0) out << " scale_tile=" << p.scale;
  if (p.accumulator >= 0) out << " accumulator=" << p.accumulator;
  if (p.output >= 0) out << " output_tile=" << p.output;
  if (p.opcode == Opcode::GemvStream)
    out << " input_count=" << p.inputs << " output_count=" << p.outputs
        << " weight_input_stride=" << p.weightInputStride << " weight_output_stride=" << p.weightOutputStride
        << " activation_input_stride=" << p.activationInputStride << " scale_input_period=" << p.scaleInputPeriod
        << " scale_input_stride=" << p.scaleInputStride << " scale_output_stride=" << p.scaleOutputStride
        << " accumulator_output_stride=" << p.accumulatorOutputStride;
  return out.str();
}

std::string printProgram(const std::vector<Primitive> &program) {
  std::ostringstream out;
  out << "// PIM primitive ABI v1; tile addresses are fixed by pim_plan/system.pim_kernel.\n";
  for (size_t i = 0; i < program.size(); ++i) out << i << ": " << print(program[i]) << '\n';
  return out.str();
}

Result<bool> verify(const PaPrimitive &p) {
  using R = Result<bool>;
  if ((p.pa & 31) || p.accumulator > INT32_MAX)
    return R::failure("PA must be packet aligned; accumulator must fit int32");
  switch (p.opcode) {
  case Opcode::LoadActivation: case Opcode::LoadScale:
  case Opcode::ClearOutput: case Opcode::StoreOutput: return R::success(true);
  case Opcode::GemvTile: case Opcode::GemvStream:
    if (p.scalePosition != -1 || p.activationPosition >= 32)
      return R::failure("PA GEMV v2 requires scale-less W8A8 and a valid IREG position");
    if (p.opcode == Opcode::GemvTile) return R::success(true);
    if (!p.inputs || p.inputs > 32 - p.activationPosition || !p.outputs || p.outputs > 32 ||
        (p.weightInputByteStride & 31) || (p.weightOutputByteStride & 31) ||
        p.accumulatorOutputStride > INT32_MAX)
      return R::failure("invalid PA stream extent or byte stride");
    if (uint64_t(p.outputs - 1) * p.accumulatorOutputStride > uint64_t(INT32_MAX) - p.accumulator)
      return R::failure("PA stream accumulator overflow");
    {
      uint64_t remaining = UINT64_MAX - p.pa;
      if (p.inputs > 1 && p.weightInputByteStride > remaining / (p.inputs - 1))
        return R::failure("PA input stride overflow");
      remaining -= uint64_t(p.inputs - 1) * p.weightInputByteStride;
      if (p.outputs > 1 && p.weightOutputByteStride > remaining / (p.outputs - 1))
        return R::failure("PA output stride overflow");
    }
    return R::success(true);
  }
  return R::failure("invalid PA primitive opcode");
}

Result<EncodedPrimitive> encode(const PaPrimitive &p) {
  using R = Result<EncodedPrimitive>;
  if (auto ok = verify(p); !ok) return R::failure(ok.error);
  EncodedPrimitive w{kPaPrimitiveVersion << 24 | uint32_t(p.singleBank) << 18 | uint32_t(p.opcode),
                     uint32_t(p.pa), uint32_t(p.pa >> 32)};
  switch (p.opcode) {
  case Opcode::LoadActivation: case Opcode::LoadScale: w.push_back(p.registerIndex); break;
  case Opcode::ClearOutput: case Opcode::StoreOutput: w.push_back(p.accumulator); break;
  case Opcode::GemvTile: case Opcode::GemvStream:
    w.insert(w.end(), {p.activationPosition, uint32_t(p.scalePosition), p.accumulator});
    if (p.opcode == Opcode::GemvStream)
      w.insert(w.end(), {p.inputs, p.outputs, uint32_t(p.weightInputByteStride), uint32_t(p.weightInputByteStride >> 32),
                        uint32_t(p.weightOutputByteStride), uint32_t(p.weightOutputByteStride >> 32), p.accumulatorOutputStride});
  }
  return R::success(std::move(w));
}

Result<PaPrimitive> decodePa(const uint32_t *w, size_t count) {
  using R = Result<PaPrimitive>;
  if (!count || w[0] >> 24 != kPaPrimitiveVersion || !instructionWords(w[0]) || count != instructionWords(w[0]) ||
      (w[0] & 0x00ffff00) != (w[0] & (1u << 18)))
    return R::failure("invalid PA primitive ABI/header/length");
  PaPrimitive p; p.opcode = Opcode(w[0] & 255); p.singleBank = bool(w[0] & (1u << 18));
  p.pa = uint64_t(w[1]) | uint64_t(w[2]) << 32;
  switch (p.opcode) {
  case Opcode::LoadActivation: case Opcode::LoadScale: p.registerIndex = w[3]; break;
  case Opcode::ClearOutput: case Opcode::StoreOutput: p.accumulator = w[3]; break;
  case Opcode::GemvTile: case Opcode::GemvStream:
    p.activationPosition = w[3]; p.scalePosition = int32_t(w[4]); p.accumulator = w[5];
    if (p.opcode == Opcode::GemvStream) {
      p.inputs = w[6]; p.outputs = w[7];
      p.weightInputByteStride = uint64_t(w[8]) | uint64_t(w[9]) << 32;
      p.weightOutputByteStride = uint64_t(w[10]) | uint64_t(w[11]) << 32;
      p.accumulatorOutputStride = w[12];
    }
  }
  if (auto ok = verify(p); !ok) return R::failure(ok.error);
  return R::success(p);
}

std::string print(const PaPrimitive &p) {
  std::ostringstream s;
  s << opcodeName(p.opcode) << "_PA pa=0x" << std::hex << p.pa << std::dec
    << " single_bank=" << p.singleBank;
  if (p.opcode == Opcode::LoadActivation || p.opcode == Opcode::LoadScale) s << " register=" << p.registerIndex;
  else s << " accumulator=" << p.accumulator;
  if (p.opcode == Opcode::GemvTile || p.opcode == Opcode::GemvStream) s << " ireg_position=" << p.activationPosition;
  if (p.opcode == Opcode::GemvStream)
    s << " inputs=" << p.inputs << " outputs=" << p.outputs
      << " weight_input_byte_stride=" << p.weightInputByteStride << " weight_output_byte_stride=" << p.weightOutputByteStride;
  return s.str();
}
} // namespace npu::pim
