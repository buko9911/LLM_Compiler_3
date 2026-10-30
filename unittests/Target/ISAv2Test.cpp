// ISA ver 2.0: core block lowering, system records and PIM records. Expected words come from
// NPU_Simulator tools/isa_v2.py (lower_core_block and the record encoders) for the same inputs.
#include "Support/Assembler.h"
#include "Support/ProgramVerifier.h"
#include "npu/Target/ISAv2.h"
#include <gtest/gtest.h>

namespace {
std::vector<uint32_t> lower(const char *text) {
  auto words = npu::assemble(text);
  EXPECT_TRUE(bool(words)) << words.error;
  auto lowered = npu::isa2::lowerCoreBlock(words.value);
  EXPECT_TRUE(bool(lowered)) << lowered.error;
  return lowered.value;
}
} // namespace

TEST(ISAv2, MatrixBlockMatchesSimulatorLowering) {
  // Constants appear where they are read; two K slices become one M_MMA count=2.
  EXPECT_EQ(lower("C_LUI_U32 12, 1\nM_LOAD_WEIGHT_F16 12, 64, 32, 64\nC_LUI_U32 13, 0\n"
                  "M_LOAD_ACT_F16 13, 32, 64, 128\nM_MMA_F16F16F32 INIT\nM_MMA_F16F16F32 ACC\n"
                  "C_LUI_U32 12, 2\nM_WRITEOUT_F16 12, 32, 32, 64\nC_WAIT_MATRIX\nC_FENCE_ALL"),
            std::vector<uint32_t>({0x00400320u, 0x007ffc77u, 0x007ffdf7u, 0x0000027bu, 0x00800320u,
                                   0x003ffc3cu, 0x0000007fu, 0x000000ffu}));
}

TEST(ISAv2, LocalDmaBlockMatchesSimulatorLowering) {
  // A strided transfer keeps its control values (C_SET immediates) and uses
  // LDMA.R; an aligned contiguous one becomes LDMA.I with all immediates.
  EXPECT_EQ(lower("C_LUI_U32 15, 0\nC_ADDI_U32 15, 15, 64\nC_SET_DMA_BYTES 15\n"
                  "C_LUI_U32 15, 0\nC_ADDI_U32 15, 15, 32\nC_SET_DMA_ROWS 15\n"
                  "C_LUI_U32 15, 0\nC_ADDI_U32 15, 15, 512\nC_SET_DMA_SOURCE_STRIDE 15\n"
                  "C_LUI_U32 15, 0\nC_ADDI_U32 15, 15, 64\nC_SET_DMA_DESTINATION_STRIDE 15\n"
                  "C_LUI_U32 12, 1\nC_LUI_U32 13, 16\nL2_LOAD_STRIDED_ASYNC 12, 13\n"
                  "C_LUI_U32 12, 0\nC_LUI_U32 13, 17\nC_ADDI_U32 13, 13, 64\nL2_LOAD_RAW_ASYNC 12, 13\n"
                  "C_WAIT_LDMA\nC_FENCE_ALL"),
            std::vector<uint32_t>({0x00010121u, 0x00008021u, 0x00080061u, 0x000100a1u, 0x00400320u,
                                   0x04000360u, 0x000371b8u, 0x000000b9u, 0x00080882u, 0x0000003fu,
                                   0x000000ffu}));
}

TEST(ISAv2, HardwareLoopKeepsRegisterWritesInside) {
  // Values live on loop entry are materialised before C_LOOP_BEGIN; inside the
  // loop every GP write stays, and vector memory uses base + offset.
  EXPECT_EQ(lower("C_LUI_U32 15, 0\nC_ADDI_U32 15, 15, 32\nC_SET_VECTOR_ELEMENTS 15\n"
                  "C_LUI_U32 5, 0\nC_LUI_U32 6, 1\nC_LUI_U32 7, 0\nC_ADDI_U32 7, 7, 4\n"
                  "C_LOOP_BEGIN 7\nV_LOAD_F16 1, 5\nV_EXP_F16 2, 1\nV_STORE_F16 6, 2\n"
                  "C_ADDI_U32 5, 5, 64\nC_ADDI_U32 6, 6, 64\nC_LOOP_END\nC_WAIT_VPU\nC_FENCE_ALL"),
            std::vector<uint32_t>({0x004001a0u, 0x000011e0u, 0x000083e0u, 0x000080e1u, 0x00000223u,
                                   0x00005132u, 0x00048230u, 0x00006233u, 0x00101562u, 0x001019a2u,
                                   0x00000024u, 0x000000bfu, 0x000000ffu}));
}

TEST(ISAv2, VectorMemoryOffsetFromLoopRegister) {
  // A builder V_MEM may carry a byte offset (vectorMemoryOffsetBits). Inside a hardware loop the
  // address register is known only to the core: the offset goes to the V_LD/V_ST offset field.
  // Outside a loop the offset is folded into the known address like any other access.
  auto words = npu::assemble("C_LUI_U32 15, 0\nC_ADDI_U32 15, 15, 32\nC_SET_VECTOR_ELEMENTS 15\n"
                             "C_LUI_U32 11, 1\nC_LUI_U32 3, 0\nC_ADDI_U32 3, 3, 4\n"
                             "C_LOOP_BEGIN 3\nV_LOAD_F16 1, 11\nV_LOAD_F16 2, 11\nV_ADD_F16 3, 1, 2\n"
                             "V_STORE_F16 11, 3\nC_ADDI_U32 11, 11, 128\nC_LOOP_END\n"
                             "C_LUI_U32 4, 2\nV_LOAD_F16 5, 4\nC_WAIT_VPU\nC_FENCE_ALL");
  ASSERT_TRUE(bool(words)) << words.error;
  std::vector<size_t> memory;
  for (size_t i = 0; i < words.value.size(); ++i)
    if ((words.value[i] & 0x3f) == 0x35) memory.push_back(i);
  ASSERT_EQ(memory.size(), 4u);
  words.value[memory[1]] |= npu::isa2::vectorMemoryOffsetBits(64);
  words.value[memory[2]] |= npu::isa2::vectorMemoryOffsetBits(96);
  words.value[memory[3]] |= npu::isa2::vectorMemoryOffsetBits(48);
  EXPECT_EQ(npu::isa2::vectorMemoryOffset(npu::isa2::vectorMemoryOffsetBits(0x3fff0)), 0x3fff0u);
  auto lowered = npu::isa2::lowerCoreBlock(words.value);
  ASSERT_TRUE(bool(lowered)) << lowered.error;
  std::vector<std::pair<unsigned, unsigned>> accesses;   // (base register, offset / 16)
  for (uint32_t w : lowered.value)
    if ((w & 0x3f) == 0x32 || (w & 0x3f) == 0x33) accesses.push_back({w >> 12 & 15, w >> 16});
  ASSERT_EQ(accesses.size(), 4u);
  EXPECT_EQ(accesses[0], std::make_pair(11u, 0u));
  EXPECT_EQ(accesses[1], std::make_pair(11u, 4u));
  EXPECT_EQ(accesses[2], std::make_pair(11u, 6u));
  // After the loop the builder value of GP4 (0x2000) plus 48 bytes is folded: GP4 holds the
  // 1 MiB-aligned base 0 and the offset field holds 0x2030 / 16.
  EXPECT_EQ(accesses[3], std::make_pair(4u, 0x203u));
  EXPECT_TRUE(bool(npu::isa2::verifyCoreWords(lowered.value)));
}

TEST(ISAv2, TypedFp32GuardsAndCastsMatchSimulatorLowering) {
  // EXP.F32, CMP_EQ.F32 -> I8, SELECT.F32, F32->F16, a seven-lane
  // F16->F32 tail, and BCAST.F32. Golden words were independently obtained
  // with transactional_emulator/tools/isa_v2.py::lower_core_block.
  auto result = npu::isa2::lowerCoreBlock({
      0x76u, 8, 1, 16, 0x4b6u, 13, 1, 16, 0x804f6u, 16, 1, 16,
      0xd36u, 27, 1, 16, 0x1176u, 28, 0, 7, 0x15b6u, 18, 1, 7, 0x100003fu});
  ASSERT_TRUE(bool(result)) << result.error;
  EXPECT_EQ(result.value, std::vector<uint32_t>({
      0x40e1u, 0x5230u, 0x49370u, 0x804d430u, 0x400d0431u,
      0x1ce1u, 0x40115031u, 0x1594b0u, 0xffu}));
  EXPECT_TRUE(bool(npu::isa2::verifyCoreWords(result.value)));
}

TEST(ISAv2, TypedRecordsRejectInvalidTypesAndRegisterOverflow) {
  for (const auto &record : std::vector<std::vector<uint32_t>>{
           {0x36, 8, 4, 16},   // Unknown dtype.
           {0x36, 41, 1, 16},  // Unknown operation.
           {0x36, 8, 2, 16},   // EXP on I8.
           {0x36, 21, 1, 16},  // AND on FP32.
           {0x36, 8, 1, 0},    // Empty vector.
           {0x36, 8, 1, 17},   // FP32 source exceeds 512 bits.
           {0x36, 28, 0, 17},  // F16 -> FP32 result exceeds 512 bits.
           {0x400036, 8, 1, 16}}) { // Reserved builder header bits.
    auto result = npu::isa2::lowerCoreBlock(record);
    EXPECT_FALSE(bool(result));
    EXPECT_FALSE(result.error.empty());
  }
}

TEST(ISAv2, DiagnosticSourceMapPreservesEncodingAndMergedMmaOrigin) {
  std::vector<uint32_t> builder{0x3bu | (1u << 22),
                                0x3bu | (1u << 22) | (1u << 26),
                                0x3bu | (3u << 22)};
  std::vector<size_t> sources{99};
  auto plain = npu::isa2::lowerCoreBlock(builder);
  auto mapped = npu::isa2::lowerCoreBlock(builder, &sources);
  ASSERT_TRUE(bool(plain));
  ASSERT_TRUE(bool(mapped));
  EXPECT_EQ(mapped.value, plain.value);
  EXPECT_EQ(sources, std::vector<size_t>({0, 2}));
}

TEST(ISAv2, RecordsMatchSimulatorEncoders) {
  auto gdma = npu::isa2::gdmaRecord(true, 5, {1, 3}, 0x123456, 0x2000, 256, 4, 1024, 256);
  ASSERT_TRUE(bool(gdma)) << gdma.error;
  EXPECT_EQ(gdma.value, std::vector<uint32_t>({0x0080016b, 0x0091a2b0, 0x00100000, 0x04000003,
                                               0x00010000, 0x00020004}));
  auto flat = npu::isa2::gdmaRecord(false, 2, {}, 4096, 65536, 4096, 1, 4096, 4096);
  ASSERT_TRUE(bool(flat));
  EXPECT_EQ(flat.value, std::vector<uint32_t>({0x0800002c, 0x00008000, 0x00800000}));
  auto core = npu::isa2::coreBeginRecord(7, {2, 4, 6}, 21, 3, 10240);
  ASSERT_TRUE(bool(core));
  EXPECT_EQ(core.value, std::vector<uint32_t>({0x005001a6, 0x03000015, 0x00030005, 0x00000001}));
  auto barrier = npu::isa2::barrierRecord(9, {1, 2, 3});
  ASSERT_TRUE(bool(barrier));
  EXPECT_EQ(barrier.value, std::vector<uint32_t>({0x000000ed, 0x00070008, 0x00000006}));
}

TEST(ISAv2, RecordsRejectWhatTheSimulatorCannotDecode) {
  EXPECT_FALSE(bool(npu::isa2::gdmaRecord(true, 1, {}, 0, 0, 1u << 20, 1, 1u << 20, 1u << 20)));  // row bytes
  EXPECT_FALSE(bool(npu::isa2::gdmaRecord(true, 1, {}, 0, 1u << 24, 64, 1, 64, 64)));             // L2 address
  EXPECT_FALSE(bool(npu::isa2::coreBeginRecord(1, {}, 1, 31, 0)));                                 // core 31
  EXPECT_FALSE(bool(npu::isa2::barrierRecord(70000, {1})));                                        // distance
  EXPECT_FALSE(bool(npu::isa2::barrierRecord(3, {3})));                                            // not earlier
}

// ---- whole blocks and programs ------------------------------------------------------------

namespace {
std::vector<uint32_t> checkedWords(npu::Result<std::vector<uint32_t>> r) {
  EXPECT_TRUE(bool(r)) << r.error;
  return std::move(r.value);
}

// Builder stream covering GP constants, both LDMA forms, vector memory with a
// shared base, the M_MMA run, a strided writeout and a hardware loop.
constexpr const char *kBlock =
    "C_LUI_U32 1, 0x10\n C_LUI_U32 2, 0x20\n C_ADDI_U32 3, 0, 1024\n C_SET_DMA_BYTES 3\n"
    "L2_LOAD_RAW_ASYNC 1, 2\n"
    "C_ADDI_U32 4, 0, 4\n C_SET_DMA_ROWS 4\n C_ADDI_U32 5, 0, 256\n C_SET_DMA_SOURCE_STRIDE 5\n"
    "C_SET_DMA_DESTINATION_STRIDE 5\n C_ADDI_U32 6, 0, 64\n C_SET_DMA_BYTES 6\n"
    "L2_LOAD_STRIDED_ASYNC 1, 2\n C_WAIT_LDMA\n"
    "C_ADDI_U32 7, 0, 32\n C_SET_VECTOR_ELEMENTS 7\n C_ADDI_U32 9, 1, 4096\n"
    "V_LOAD_F16 1, 9\n V_ADD_F16 2, 1, 1\n V_STORE_F16 9, 2\n"
    "M_LOAD_WEIGHT_F16 1, 96, 32, 64\n M_LOAD_ACT_F16 2, 32, 96, 192\n"
    "M_MMA_F16F16F32 INIT\n M_MMA_F16F16F32 ACC\n M_MMA_F16F16F32 ACC\n"
    "M_WRITEOUT_F16 1, 32, 32, 128\n"
    "C_ADDI_U32 8, 0, 3\n C_LOOP_BEGIN 8\n V_ADD_F16 2, 2, 1\n C_LOOP_END\n C_FENCE_ALL";

TEST(ISAv2, LowersLikeTheSimulatorReference) {
  // IESLAB_NPU_Simulator transactional_emulator/tools/isa_v2.py lower_core_block
  // of the same 40 builder words (75725ad).
  const std::vector<uint32_t> expected = {
      0x020000b9, 0x00801000, 0x00010121, 0x00001021, 0x00040061, 0x000400a1, 0x04000060, 0x080000a0,
      0x000085b8, 0x0000003f, 0x000080e1, 0x11009132, 0x00448030, 0x11009233, 0x00bff177, 0x00bff2f7,
      0x0000037b, 0x00bff13c, 0x001000e0, 0x00001120, 0x00040160, 0x000101a0, 0x000081e0, 0x00000e20,
      0x04400260, 0x000001a3, 0x00488030, 0x00000024, 0x000000ff};
  auto builder = npu::assemble(kBlock);
  ASSERT_TRUE(bool(builder)) << builder.error;
  ASSERT_EQ(builder.value.size(), 40u);
  ASSERT_TRUE(bool(npu::verifyCore(builder.value)));
  auto lowered = npu::isa2::lowerCoreBlock(builder.value);
  ASSERT_TRUE(bool(lowered)) << lowered.error;
  EXPECT_EQ(lowered.value, expected);
  EXPECT_TRUE(bool(npu::isa2::verifyCoreBlock(lowered.value.data(), lowered.value.size())));
  // LDMA.I is two words; a one-word read of it leaves its second word behind.
  EXPECT_FALSE(bool(npu::isa2::verifyCoreBlock(lowered.value.data(), 1)));
}

TEST(ISAv2, RejectsWhatVer20CannotHold) {
  auto lower = [](const char *text) {
    auto words = npu::assemble(text);
    return words ? npu::isa2::lowerCoreBlock(words.value)
                 : npu::Result<std::vector<uint32_t>>::failure(words.error);
  };
  EXPECT_FALSE(bool(lower("C_ADDI_U32 1, 0, 2\n C_LOOP_BEGIN 1\n C_WAIT_VPU")));   // unterminated
  // A local event must be a constant: inside a loop every GP is a loop value.
  EXPECT_FALSE(bool(lower("C_ADDI_U32 1, 0, 2\n C_LOOP_BEGIN 1\n C_ADDI_U32 2, 0, 5\n C_WAIT_EVENT 2\n"
                          "C_LOOP_END\n C_FENCE_ALL")));
  std::string events;
  for (int e = 1; e <= 16; ++e)
    events += "C_ADDI_U32 2, 0, " + std::to_string(e) + "\n C_WAIT_EVENT 2\n";
  EXPECT_FALSE(bool(lower(events.c_str())));   // 16 local events
  EXPECT_FALSE(bool(lower("C_LOOP_BEGIN 0\n C_WAIT_VPU\n C_LOOP_END")));   // zero trip count
}

TEST(ISAv2, SystemRecordsMatchTheSimulatorDecoder) {
  // runner.rs unified_binary_decodes_gdma_events_and_core_begin.
  EXPECT_EQ(checkedWords(npu::isa2::gdmaRecord(true, 0, {}, 0x1000, 0x2000, 64, 1, 64, 64)),
            (std::vector<uint32_t>{0x2b | 64u << 15, 0x1000u << 3, 0x2000u << 7}));
  EXPECT_EQ(checkedWords(npu::isa2::coreBeginRecord(1, {0}, 1, 0, 64)),
            (std::vector<uint32_t>{0x26 | 8u << 12, 1 | 1u << 24, 1}));
  // 2-D: rows - 1, then both strides; dependencies 3, 7, 9 of record 10 at
  // distances 7, 3, 1, two per word.
  EXPECT_EQ(checkedWords(npu::isa2::gdmaRecord(false, 10, {3, 7, 9}, 0, 64, 32, 4, 256, 128)),
            (std::vector<uint32_t>{0x2c | 1u << 6 | 3u << 7 | 32u << 15, 0, 64u << 7, 3 | 256u << 16,
                                   128u << 8, 7 | 3u << 16, 1}));
  EXPECT_EQ(checkedWords(npu::isa2::barrierRecord(5, {1, 2})), (std::vector<uint32_t>{0x2d | 2u << 6, 4 | 3u << 16}));
}

TEST(ISAv2, PimPrimitiveRecordFields) {
  npu::pim::Primitive clear; clear.accumulator = 3; clear.partition = 13;
  auto record = npu::isa2::pimRecord(3, {1}, &clear, 1);
  ASSERT_TRUE(bool(record)) << record.error;
  EXPECT_EQ(record.value.size(), 4u);
  EXPECT_EQ(record.value[0], 0x2eu | 1u << 6 | 1u << 14);
  EXPECT_EQ(record.value[1], 2u);
  EXPECT_EQ(record.value[2], 1u << 24 | 13u << 8 | 5u);
  EXPECT_EQ(record.value[3], 3u);
}

TEST(ISAv2, PimPaPrimitivePreservesUpperAddressAndByteStrides) {
  npu::pim::PaPrimitive p;
  p.opcode = npu::pim::Opcode::GemvStream; p.pa = 0x208000020ull;
  p.inputs = 32; p.outputs = 8; p.weightInputByteStride = 256; p.weightOutputByteStride = 1u << 17;
  auto e = npu::pim::encode(p);
  ASSERT_TRUE(bool(e)) << e.error;
  EXPECT_EQ(e.value, (std::vector<uint32_t>{2u << 24 | 7, 0x08000020, 2, 0, UINT32_MAX, 0,
                                          32, 8, 256, 0, 1u << 17, 0, 1}));
  auto d = npu::pim::decodePa(e.value.data(), e.value.size());
  ASSERT_TRUE(bool(d));
  EXPECT_EQ(d.value.pa, p.pa);
  EXPECT_EQ(npu::pim::print(d.value), npu::pim::print(p));
  EXPECT_FALSE(bool(npu::pim::decode(e.value.data(), e.value.size())));
  auto bad = e.value; bad[0] |= 1u << 8;
  EXPECT_FALSE(bool(npu::pim::decodePa(bad.data(), bad.size())));
  p.pa = UINT64_MAX - 31;
  EXPECT_FALSE(bool(npu::pim::encode(p)));
  npu::pim::PaPrimitive clear; clear.pa = 0x208000000ull;
  auto record = npu::isa2::pimRecord(0, {}, &clear, 1);
  ASSERT_TRUE(bool(record));
  auto program = std::vector<uint32_t>{npu::isa2::kMagic, npu::isa2::kVersion,
      uint32_t(6 + record.value.size()), 1, 0};
  program.insert(program.end(), record.value.begin(), record.value.end());
  program.push_back(npu::isa2::OP_PROGRAM_END);
  EXPECT_TRUE(bool(npu::verifyProgramWords(program)));
}

TEST(ISAv2, PimPrimitivePreservesArbitraryTileAndStreamOperands) {
  // Not the retired macro's 512/32/8 aligned addresses or 32x16 geometry.
  npu::pim::Primitive p;
  p.opcode = npu::pim::Opcode::GemvStream; p.partition = 13; p.bankSet = 1;
  p.weight = 7; p.activation = 5; p.scale = 9; p.accumulator = 2;
  p.inputs = 3; p.outputs = 2;
  p.weightInputStride = 2; p.weightOutputStride = 11; p.activationInputStride = 1;
  p.scaleInputPeriod = 2; p.scaleInputStride = 3; p.scaleOutputStride = 4;
  p.accumulatorOutputStride = 2;
  auto e = npu::pim::encode(p);
  ASSERT_TRUE(bool(e)) << e.error;
  const npu::pim::EncodedPrimitive golden{1u << 24 | 1u << 16 | 13u << 8 | 7u,
      7, 5, 9, 2, 3, 2, 2, 11, 1, 2, 3, 4, 2};
  EXPECT_EQ(e.value, golden);
  auto d = npu::pim::decode(golden.data(), golden.size());
  ASSERT_TRUE(bool(d));
  EXPECT_EQ(npu::pim::print(d.value), npu::pim::print(p));
  p.inputs = 0;
  EXPECT_FALSE(bool(npu::pim::encode(p)));
  p.inputs = 3; p.weight = INT32_MAX;
  EXPECT_FALSE(bool(npu::pim::encode(p)));
  auto bad = golden; bad[0] = 1u << 24; // STAGE cannot be executed.
  EXPECT_FALSE(bool(npu::pim::decode(bad.data(), bad.size())));
  bad = golden; bad[0] |= 2u << 24;
  EXPECT_FALSE(bool(npu::pim::decode(bad.data(), bad.size())));
  EXPECT_FALSE(bool(npu::pim::decode(golden.data(), golden.size() - 1)));
}

TEST(ISAv2, LargePimPhaseUsesExtendedCountWithoutSplittingLiveState) {
  npu::pim::Primitive clear; clear.accumulator = 0;
  std::vector<npu::pim::Primitive> instructions(1u << 18, clear);
  auto record = npu::isa2::pimRecord(0, {}, instructions.data(), instructions.size());
  ASSERT_TRUE(bool(record)) << record.error;
  EXPECT_EQ(record.value[0], npu::isa2::OP_PIM_PRIMITIVES);
  EXPECT_EQ(record.value[1], instructions.size());
  auto program = std::vector<uint32_t>{npu::isa2::kMagic, npu::isa2::kVersion,
      uint32_t(6 + record.value.size()), 1, 0};
  program.insert(program.end(), record.value.begin(), record.value.end());
  program.push_back(npu::isa2::OP_PROGRAM_END);
  EXPECT_TRUE(bool(npu::verifyProgramWords(program)));
  program[6] = 1; // short counts must use the inline encoding
  EXPECT_FALSE(bool(npu::verifyProgramWords(program)));
}
} // namespace
