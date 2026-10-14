#include "npu/Target/ISA.h"
#include "npu/Target/IsaV2.h"
#include "npu/Target/Program.h"
#include <gtest/gtest.h>

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

TEST(IsaV2, LowersLikeTheSimulatorReference) {
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

TEST(IsaV2, RejectsWhatVer20CannotHold) {
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

TEST(IsaV2, SystemRecordsMatchTheSimulatorDecoder) {
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

TEST(IsaV2, PimPrimitiveRecordFields) {
  npu::pim::Primitive clear; clear.accumulator = 3; clear.partition = 13;
  auto record = npu::isa2::pimRecord(3, {1}, &clear, 1);
  ASSERT_TRUE(bool(record)) << record.error;
  EXPECT_EQ(record.value.size(), 4u);
  EXPECT_EQ(record.value[0], 0x2eu | 1u << 6 | 1u << 14);
  EXPECT_EQ(record.value[1], 2u);
  EXPECT_EQ(record.value[2], 1u << 24 | 13u << 8 | 5u);
  EXPECT_EQ(record.value[3], 3u);
}

TEST(IsaV2, PimPaPrimitivePreservesUpperAddressAndByteStrides) {
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
  npu::Segment segment; segment.name = "pa_pim"; segment.paPim = {clear};
  auto linked = npu::linkProgram({segment});
  ASSERT_TRUE(bool(linked)) << linked.error;
  EXPECT_TRUE(bool(npu::verifyProgramWords(linked.value.words)));
}

TEST(IsaV2, PimPrimitivePreservesArbitraryTileAndStreamOperands) {
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

TEST(IsaV2, LargePimPhaseUsesExtendedCountWithoutSplittingLiveState) {
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
  // Relinking an extended record must preserve its payload and dependency boundary.
  auto linked = npu::linkProgram({{"large", program, {}}});
  ASSERT_TRUE(bool(linked)) << linked.error;
  EXPECT_TRUE(bool(npu::verifyProgramWords(linked.value.words)));
  program[6] = 1; // short counts must use the inline encoding
  EXPECT_FALSE(bool(npu::verifyProgramWords(program)));
}
} // namespace
