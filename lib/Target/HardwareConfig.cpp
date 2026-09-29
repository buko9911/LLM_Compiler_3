#include "plena/Target/HardwareConfig.h"
#include <algorithm>
#include <charconv>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <map>
#include <sstream>

namespace plena {
namespace {
std::string trim(std::string s) {
  auto first = s.find_first_not_of(" \t\r\n");
  if (first == s.npos) return {};
  return s.substr(first, s.find_last_not_of(" \t\r\n") - first + 1);
}
}
Result<HardwareConfig> parseHardwareConfig(std::string_view source) {
  using R = Result<HardwareConfig>;
  HardwareConfig config;
  uint64_t accumulators = 0;
  std::map<std::string, uint64_t *> keys = {
    {"TRANSACTIONAL.ARCHITECTURE.array_rows", &config.arrayRows},
    {"TRANSACTIONAL.ARCHITECTURE.array_columns", &config.arrayColumns},
    {"TRANSACTIONAL.ARCHITECTURE.l1_sram_size_bytes", &config.l1Bytes},
    {"TRANSACTIONAL.ARCHITECTURE.lp6_capacity_bytes", &config.dramBytes},
    {"TRANSACTIONAL.SYSTEM.num_cores", &config.cores},
    {"TRANSACTIONAL.L2.size_bytes", &config.l2Bytes},
    {"TRANSACTIONAL.DMA.GLOBAL.lp6_link_bytes_per_cycle", &config.dramBytesPerCycle},
    {"TRANSACTIONAL.NOC.bytes_per_cycle", &config.nocBytesPerCycle},
    {"TRANSACTIONAL.MATRIX_TIMING.FP16.macs_per_pe_per_cycle", &config.matrixMACsPerCycle},
    {"TRANSACTIONAL.MATRIX_TIMING.FP16.pipeline_latency_cycles", &config.matrixLatency},
    {"TRANSACTIONAL.MATRIX_MICROARCHITECTURE.accumulator_tiles", &accumulators},
  };
  std::map<std::string, bool> seen;
  std::istringstream stream{std::string(source)};
  std::string line, section;
  while (std::getline(stream, line)) {
    line = trim(line.substr(0, line.find('#')));
    if (line.empty()) continue;
    if (line.front() == '[') {
      if (line.back() != ']' || line.size() < 3) return R::failure("invalid settings section");
      section = line.substr(1, line.size()-2); continue;
    }
    auto equals = line.find('=');
    if (equals == line.npos) return R::failure("invalid settings assignment");
    auto key = section + "." + trim(line.substr(0, equals));
    auto found = keys.find(key);
    if (found == keys.end()) continue;
    if (seen[key]) return R::failure("duplicate setting: " + key);
    auto value = trim(line.substr(equals + 1));
    if (value.empty() || value.front() == '_' || value.back() == '_' || value.find("__") != value.npos)
      return R::failure("invalid integer setting: " + key);
    value.erase(std::remove(value.begin(), value.end(), '_'), value.end());
    uint64_t n = 0;
    auto parsed = std::from_chars(value.data(), value.data() + value.size(), n);
    if (parsed.ec != std::errc() || parsed.ptr != value.data() + value.size())
      return R::failure("invalid integer setting: " + key);
    if (!n && found->second != &config.matrixLatency) return R::failure("setting must be positive: " + key);
    *found->second = n; seen[key] = true;
  }
  for (const auto &key : keys) if (!seen[key.first]) return R::failure("missing setting: " + key.first);
  if (config.arrayRows != 32 || config.arrayColumns != 32 || accumulators != 1)
    return R::failure("unsupported hardware: requires 32x32 array and one accumulator");
  if (config.l1Bytes > UINT32_MAX || config.l2Bytes > UINT32_MAX || config.cores > 65536 ||
      config.dramBytes > INT64_MAX) return R::failure("hardware capacity exceeds encoding limits");
  uint64_t hash = 14695981039346656037ull;
  for (unsigned char ch : source) { hash ^= ch; hash *= 1099511628211ull; }
  std::ostringstream digest; digest << std::hex << std::setw(16) << std::setfill('0') << hash;
  config.fingerprint = digest.str();
  return R::success(std::move(config));
}
Result<HardwareConfig> loadHardwareConfig(const std::string &path) {
  std::ifstream in(path);
  if (!in) return Result<HardwareConfig>::failure("cannot open settings: " + path);
  std::string source{std::istreambuf_iterator<char>(in), {}};
  return parseHardwareConfig(source);
}
} // namespace plena
