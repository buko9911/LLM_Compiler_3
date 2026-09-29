#ifndef PLENA_TARGET_HARDWARECONFIG_H
#define PLENA_TARGET_HARDWARECONFIG_H
#include "plena/Target/Result.h"
#include <cstdint>
#include <string>
#include <string_view>
namespace plena {
struct HardwareConfig {
  uint64_t l1Bytes = 0, l2Bytes = 0, dramBytes = 0;
  uint64_t cores = 0, arrayRows = 0, arrayColumns = 0;
  uint64_t dramBytesPerCycle = 0, nocBytesPerCycle = 0;
  uint64_t matrixMACsPerCycle = 0, matrixLatency = 0;
  // Stable digest of the complete file; conservative invalidation on changes.
  std::string fingerprint;
};
// Deliberately restricted to the simulator's section/key integer configuration.
// Required known keys are strict (missing/duplicate/invalid is an error).
Result<HardwareConfig> parseHardwareConfig(std::string_view source);
Result<HardwareConfig> loadHardwareConfig(const std::string &path);
} // namespace plena
#endif
