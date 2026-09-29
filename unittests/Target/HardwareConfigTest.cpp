#include "npu/Target/HardwareConfig.h"
#include <gtest/gtest.h>
namespace {
const char *settings = R"toml(
[TRANSACTIONAL.ARCHITECTURE]
array_rows = 32
array_columns = 32
l1_sram_size_bytes = 4_194_304
lp6_capacity_bytes = 17179869184
[TRANSACTIONAL.SYSTEM]
num_cores = 4
[TRANSACTIONAL.L2]
size_bytes = 8388608
[TRANSACTIONAL.DMA.GLOBAL]
lp6_link_bytes_per_cycle = 32
[TRANSACTIONAL.NOC]
bytes_per_cycle = 32
[TRANSACTIONAL.MATRIX_TIMING.FP16]
macs_per_pe_per_cycle = 1
pipeline_latency_cycles = 0
[TRANSACTIONAL.MATRIX_MICROARCHITECTURE]
accumulator_tiles = 1
)toml";
TEST(HardwareConfig, SharedSettingsAndFingerprint) {
  auto c = npu::parseHardwareConfig(settings); ASSERT_TRUE(bool(c)) << c.error;
  EXPECT_EQ(c.value.l1Bytes, 4194304u); EXPECT_EQ(c.value.cores, 4u);
  auto changed = npu::parseHardwareConfig(std::string(settings) + "# modified\n");
  ASSERT_TRUE(bool(changed)); EXPECT_NE(c.value.fingerprint, changed.value.fingerprint);
}
TEST(HardwareConfig, MissingDuplicateAndInvalidKnownKeysFail) {
  EXPECT_FALSE(bool(npu::parseHardwareConfig("[TRANSACTIONAL.SYSTEM]\nnum_cores=1\n")));
  EXPECT_FALSE(bool(npu::parseHardwareConfig(std::string(settings) + "accumulator_tiles=1\n")));
  for (const char *bad : {"0", "-1", "32junk", "32.0", "32_", "3__2"}) {
    std::string input(settings); auto at = input.find("array_rows = 32");
    input.replace(at, std::string("array_rows = 32").size(), std::string("array_rows = ") + bad);
    EXPECT_FALSE(bool(npu::parseHardwareConfig(input))) << bad;
  }
}
} // namespace
