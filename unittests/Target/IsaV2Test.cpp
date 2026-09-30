#include "npu/Target/ISA.h"
#include "npu/Target/IsaV2.h"
#include "npu/Target/Program.h"
#include <gtest/gtest.h>

namespace {
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
  EXPECT_EQ(npu::isa2::gdmaRecord(true, 0, {}, 0x1000, 0x2000, 64, 1, 64, 64),
            (std::vector<uint32_t>{0x2b | 64u << 15, 0x1000u << 3, 0x2000u << 7}));
  EXPECT_EQ(npu::isa2::coreBeginRecord(1, {0}, 1, 0, 64),
            (std::vector<uint32_t>{0x26 | 8u << 12, 1 | 1u << 24, 1}));
  // 2-D: rows - 1, then both strides; dependencies 3, 7, 9 of record 10 at
  // distances 7, 3, 1, two per word.
  EXPECT_EQ(npu::isa2::gdmaRecord(false, 10, {3, 7, 9}, 0, 64, 32, 4, 256, 128),
            (std::vector<uint32_t>{0x2c | 1u << 6 | 3u << 7 | 32u << 15, 0, 64u << 7, 3 | 256u << 16,
                                   128u << 8, 7 | 3u << 16, 1}));
  EXPECT_EQ(npu::isa2::barrierRecord(5, {1, 2}), (std::vector<uint32_t>{0x2d | 2u << 6, 4 | 3u << 16}));
}

TEST(IsaV2, PimRecordCarriesItsCommands) {
  // runner.rs pim_record_carries_its_commands_and_dependencies: W0 count and
  // dependencies, the dependency words, two words per PIM_GEMV.
  const uint32_t commands[] = {1, 2, 3, 4};
  EXPECT_EQ(npu::isa2::pimRecord(3, {1}, commands, 2),
            (std::vector<uint32_t>{0x27 | 1u << 6 | 2u << 14, 2, 1, 2, 3, 4}));
}

TEST(IsaV2, PimGemvFields) {
  auto command = npu::isa2::pimGemv(1, 10, 1024, 320, 16);
  ASSERT_TRUE(bool(command)) << command.error;
  EXPECT_EQ(command.value.first, 1u << 2 | 9u << 6 | 2u << 14);
  EXPECT_EQ(command.value.second, 10u | 2u << 16);
  auto last = npu::isa2::pimGemv(15, 1, 0, 0, 0);            // subchannel 7 of rank 1 on four channels
  ASSERT_TRUE(bool(last));
  EXPECT_EQ(last.value.first, 15u << 2);
  EXPECT_FALSE(bool(npu::isa2::pimGemv(0, 1, 100, 0, 0)));   // weight tile not a multiple of 512
  EXPECT_FALSE(bool(npu::isa2::pimGemv(16, 1, 0, 0, 0)));    // partition
  EXPECT_FALSE(bool(npu::isa2::pimGemv(0, 257, 0, 0, 0)));   // sweeps
}
} // namespace
