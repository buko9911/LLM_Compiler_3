#include "plena/Target/ISA.h"
#include <gtest/gtest.h>
#include <array>

namespace {
constexpr uint32_t r(unsigned op, unsigned d, unsigned a, unsigned b, unsigned c, unsigned f) {
  return op | d << 6 | a << 10 | b << 14 | c << 18 | f << 22;
}
TEST(ISA, SimulatorDecoderVectors) {
  // Transcribed from transactional_emulator/src/op.rs #[cfg(test)].
  const std::pair<const char *, uint32_t> vectors[] = {
    {"V_LOAD_F16 1, 2", r(0x35,1,2,0,0,1)},
    {"V_SUB_F16 1, 2, 3", r(0x30,1,2,3,0,2)},
    {"V_EXP_F16 1, 2", r(0x31,1,2,0,0,2)},
    {"V_DOT_F16_F32 1, 2, 3", r(0x33,1,2,3,0,3)},
    {"L1_COPY_STRIDED 1, 2", r(0x38,1,2,0,0,9)},
    {"L2_LOAD_RAW_ASYNC_EVENT 1, 2, 3", r(0x38,1,2,3,0,10)},
    {"C_WAIT_EVENT 3", r(0x2e,3,0,0,0,0)},
    {"L2_LOAD_RAW 1, 2", r(0x38,1,2,0,0,0)},
    {"C_FENCE_ALL", r(0x3f,0,0,0,0,4)},
    {"C_WAIT_LDMA", r(0x3f,0,0,0,0,1)},
  };
  for (const auto &v : vectors) {
    auto encoded = plena::assembleLine(v.first);
    ASSERT_TRUE(bool(encoded)) << encoded.error;
    EXPECT_EQ(encoded.value, v.second) << v.first;
    EXPECT_TRUE(plena::isValidInstruction(v.second));
    auto decoded = plena::disassemble(v.second);
    ASSERT_TRUE(bool(decoded)); EXPECT_EQ(decoded.value, v.first);
  }
}
TEST(ISA, RejectsReservedFields) {
  const uint32_t invalid[] = {1, 0x30u | 1u << 26, 0x25u | 1u << 30, 0x24u | 1u << 6,
    r(0x2b,1,2,3,0,0), r(0x30,1,2,3,1,1), r(0x31,1,2,1,0,1),
    r(0x3b,1,0,0,0,3), r(0x3b,0,1,0,0,3), r(0x37,1,2,0,0,3),
    r(0x3f,1,0,0,0,4), r(0x3f,0,0,0,0,6), r(0x3d,1,2,3,4,6), r(0x3e,1,2,3,4,1)};
  for (auto w : invalid) {
    EXPECT_FALSE(bool(plena::disassemble(w))) << w;
    EXPECT_FALSE(plena::isValidInstruction(w)) << w;
  }
  for (unsigned fn = 6; fn <= 15; ++fn) EXPECT_FALSE(bool(plena::disassemble(r(0x30,1,2,3,0,fn))));
}
TEST(ISA, EveryInstructionRoundTripsAtRegisterBoundary) {
  for (const auto &info : plena::instructionSet()) {
    if (info.name.rfind("M_", 0) == 0) continue; // matrix records: MatrixRecords* tests
    std::string line = info.name;
    for (unsigned i = 0; i < info.operands; ++i) line += (i ? ", " : " ") + std::string("15");
    auto encoded = plena::assembleLine(line); ASSERT_TRUE(bool(encoded)) << line;
    auto decoded = plena::disassemble(encoded.value); ASSERT_TRUE(bool(decoded)) << line;
    EXPECT_EQ(plena::assembleLine(decoded.value).value, encoded.value);
  }
}
// ISA ver 1.0 matrix records, transcribed from NPU_Simulator main
// tools/test_v2_assembler.py and src/op.rs decode_record.
TEST(ISA, MatrixRecordsMatchSimulator) {
  using W = std::vector<uint32_t>;
  const std::pair<const char *, W> records[] = {
    {"M_LOAD_WEIGHT_F16 2, 64, 4, 8", {r(0x37,0,2,0,0,3), 64, 4, 8}},
    {"M_LOAD_ACT_F16 4, 4, 64, 128", {r(0x37,0,4,0,0,7), 4, 64, 128}},
    {"M_LOAD_WEIGHT_I8 3, 64, 32, 32", {r(0x37,0,3,0,0,1), 64, 32, 32}},
    {"M_MMA_F16F16F32 INIT", {0x3bu | 3u << 22}},
    {"M_MMA_F16F16F32 ACC", {0x3bu | 3u << 22 | 1u << 26}},
    {"M_MMA_I8I8I32 INIT", {0x3bu | 1u << 22}},
    {"M_WRITEOUT_F16 6, 4, 4, 8", {r(0x3c,6,0,0,0,1), 4, 4, 8}},
    {"M_WRITEOUT_F32 6, 32, 32, 128", {r(0x3c,6,0,0,0,4), 32, 32, 128}},
  };
  for (const auto &[text, words] : records) {
    auto encoded = plena::assembleInstruction(text);
    ASSERT_TRUE(bool(encoded)) << text << ": " << encoded.error;
    EXPECT_EQ(encoded.value, words) << text;
    EXPECT_EQ(plena::recordWords(words[0]), words.size()) << text;
    auto decoded = plena::disassembleRecord(words.data(), words.size());
    ASSERT_TRUE(bool(decoded)) << text;
    EXPECT_EQ(decoded.value, text);
  }
  auto load = plena::encodeMatrixLoad(plena::MatrixOperand::Weight, plena::MatrixType::F16, 2, 64, 4, 8);
  ASSERT_TRUE(bool(load)); EXPECT_EQ(load.value, records[0].second);
  auto out = plena::encodeMatrixWriteout(plena::WriteoutType::F16, 6, 4, 4, 8);
  ASSERT_TRUE(bool(out)); EXPECT_EQ(out.value, records[6].second);
  // A multi-word record never passes as one word.
  EXPECT_FALSE(bool(plena::assembleLine("M_WRITEOUT_F16 6, 4, 4, 8")));
  EXPECT_FALSE(bool(plena::disassemble(r(0x37,0,2,0,0,3))));
}
TEST(ISA, MatrixRecordsRejectWhatTheSimulatorRejects) {
  for (const char *line : {
         "M_LOAD_WEIGHT_F16 2, 64, 4, 6",    // stride below N x 2 bytes
         "M_LOAD_WEIGHT_F16 2, 64, 4, 9",    // stride not a multiple of the element
         "M_LOAD_WEIGHT_F16 2, 64, 33, 66",  // N wider than the array
         "M_LOAD_ACT_F16 4, 33, 64, 128",    // M taller than the array
         "M_LOAD_ACT_F16 4, 0, 64, 128",     // zero extent
         "M_WRITEOUT_F16 6, 4, 33, 66",      // wider than the accumulator
         "M_WRITEOUT_F32 6, 4, 4, 8",        // FP32 needs 4 bytes per element
         "M_LOAD_ACT_F16 16, 4, 64, 128",    // register index
         "M_MMA_F16F16F32", "M_MMA_F16F16F32 1", "M_LOAD_ACT_F16 4, 4, 64"})
    EXPECT_FALSE(bool(plena::assembleInstruction(line))) << line;
  const uint32_t reservedMma[] = {0x3bu | 3u << 22 | 1u << 27, 0x3bu | 3u << 22 | 1u << 6, 0x3bu | 2u << 22};
  for (auto w : reservedMma) EXPECT_FALSE(bool(plena::disassembleRecord(&w, 1))) << w;
}
TEST(ISA, MmaSequenceCoversKInFixedSlices) {
  const uint32_t init = 0x3bu | 3u << 22, acc = init | 1u << 26;
  using W = std::vector<uint32_t>;
  EXPECT_EQ(plena::mmaSequence(plena::MatrixType::F16, 32, false), W({init}));
  EXPECT_EQ(plena::mmaSequence(plena::MatrixType::F16, 40, false), W({init, acc}));
  EXPECT_EQ(plena::mmaSequence(plena::MatrixType::F16, 96, false), W({init, acc, acc}));
  EXPECT_EQ(plena::mmaSequence(plena::MatrixType::F16, 64, true), W({acc, acc}));
}
TEST(ISA, MatrixPayloadWordsAreNotInstructions) {
  // Mirrors op.rs matrix_payload_words_are_not_instructions: the payload word
  // 0x24 must not be taken for C_LOOP_END.
  std::vector<uint32_t> words{r(0x37,0,1,0,0,7), 1, 0x24, 0x48, r(0x3f,0,0,0,0,4)};
  auto records = plena::splitRecords(words);
  ASSERT_TRUE(bool(records)) << records.error;
  EXPECT_EQ(records.value.size(), 2u);
  words.pop_back(); words.pop_back();
  EXPECT_FALSE(bool(plena::splitRecords(words))); // truncated record
}
TEST(ISA, StrictParsingAndImmediates) {
  for (const char *line : {"C_LUI_U32 16, 0", "C_LUI_U32 0, 1048576", "C_ADDI_U32 0, 0, 262144",
       "M_LOAD_ACT_F16 -1", "M_LOAD_ACT_F16 1junk", "M_LOAD_ACT_F16 1,", "C_LUI_U32 0,,1",
       "C_LUI_U32 0 1", "M_MMA_F16F16F32 0", "UNKNOWN", "C_LUI_U32 0, 4294967296"})
    EXPECT_FALSE(bool(plena::assembleLine(line))) << line;
  EXPECT_EQ(plena::assembleLine("C_ADDI_U32 15, 15, 262143").value, 0xffffffe2u);
  auto source = plena::assemble("; comment\nC_LUI_U32 0, 0xabc // comment\nC_FENCE_ALL\n");
  ASSERT_TRUE(bool(source)); EXPECT_EQ(source.value.size(), 2u);
  EXPECT_EQ(plena::parseMem(plena::formatMem(source.value)).value, source.value);
}
TEST(ISA, ConstantsDoNotDependOnGPZero) {
  for (uint32_t value : {0u, 1u, 4095u, 4096u, 0x80000000u, 0xffffffffu}) {
    for (unsigned dst : {0u, 7u, 15u}) {
      std::array<uint32_t,16> gp; gp.fill(0xdeadbeef);
      auto words = plena::materialize(dst, value); ASSERT_TRUE(bool(words));
      for (auto word : words.value) {
        unsigned rd = word >> 6 & 15, rs = word >> 10 & 15;
        if ((word & 63) == 0x25) gp[rd] = ((word >> 10) & 0xfffff) << 12;
        else gp[rd] = gp[rs] + (word >> 14);
      }
      EXPECT_EQ(gp[dst], value);
      if (dst) { EXPECT_EQ(gp[0], 0xdeadbeefu); }
    }
  }
}
} // namespace
