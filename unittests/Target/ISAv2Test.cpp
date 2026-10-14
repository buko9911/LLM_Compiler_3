// ISA ver 2.0. Expected words come from NPU_Simulator tools/isa_v2.py
// (lower_core_block and the record encoders) for the same inputs.
#include "npu/Target/ISA.h"
#include "npu/Target/ISAv2.h"
#include "npu/Target/Program.h"
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

TEST(ISAv2, ManyDependenciesAreJoinedByBarriers) {
  // 300 loads into disjoint L2 rows, then one core block reading all of them:
  // it may wait for at most 255 records, so a BARRIER joins the rest.
  npu::Program p{1, 64, 300 * 64, 300 * 64, {}};
  for (unsigned i = 0; i < 300; ++i) {
    npu::Command c; c.kind = npu::Command::Kind::Load; c.dramOffset = c.l2Offset = i * 64; c.bytes = 64;
    p.commands.push_back(c);
  }
  npu::Command reader; reader.words = npu::assemble("C_FENCE_ALL").value;
  reader.accesses.push_back({npu::MemorySpace::L2, 0, 300 * 64, false});
  p.commands.push_back(reader);
  auto encoded = npu::encodeProgramWords(p);
  ASSERT_TRUE(bool(encoded)) << encoded.error;
  EXPECT_EQ(encoded.value.commands, 300u + 2u + 1u);  // loads, two barriers, the core block
  EXPECT_TRUE(bool(npu::verifyProgramWords(encoded.value.words)));
}

TEST(ISAv2, LargeContiguousTransferIsSplitIntoRows) {
  npu::Program p{1, 64, 1u << 22, 1u << 22, {}};
  npu::Command c; c.kind = npu::Command::Kind::Load; c.bytes = 1u << 21;  // 2 MiB in one row
  p.commands.push_back(c);
  auto encoded = npu::encodeProgramWords(p);
  ASSERT_TRUE(bool(encoded)) << encoded.error;
  const auto &w = encoded.value.words;
  const uint32_t rowBytes = w[5] >> 15 | (w[6] & 7) << 17, rows = (w[8] & 0xffff) + 1;
  EXPECT_LT(rowBytes, 1u << 20);
  EXPECT_EQ(uint64_t(rowBytes) * rows, uint64_t(1) << 21);
}
