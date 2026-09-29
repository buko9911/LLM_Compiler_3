#include "npu/Target/Quantize.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace npu {
float halfToFloat(uint16_t bits) {
  const uint32_t sign = uint32_t(bits & 0x8000) << 16;
  uint32_t exponent = (bits >> 10) & 0x1f, mantissa = bits & 0x3ff, out;
  if (exponent == 0x1f) {
    out = sign | 0x7f800000 | (mantissa << 13);
  } else if (exponent) {
    out = sign | ((exponent + 112) << 23) | (mantissa << 13);
  } else if (!mantissa) {
    out = sign;
  } else {
    // Subnormal: shift the leading one into the implicit bit.
    int shift = 0;
    while (!(mantissa & 0x400)) { mantissa <<= 1; ++shift; }
    out = sign | (uint32_t(113 - shift) << 23) | ((mantissa & 0x3ff) << 13);
  }
  float value;
  std::memcpy(&value, &out, sizeof value);
  return value;
}

void quantizeChannels(const uint16_t *weights, uint64_t outputs, uint64_t inputs, bool channelMajor,
                      int8_t *q, float *scales) {
  auto at = [&](uint64_t n, uint64_t k) {
    return halfToFloat(weights[channelMajor ? n * inputs + k : k * outputs + n]);
  };
  for (uint64_t n = 0; n < outputs; ++n) {
    float peak = 0.0f;
    for (uint64_t k = 0; k < inputs; ++k) peak = std::max(peak, std::fabs(at(n, k)));
    const float scale = peak / 127.0f;
    scales[n] = scale;
    for (uint64_t k = 0; k < inputs; ++k) {
      float value = scale == 0.0f ? 0.0f : std::nearbyint(at(n, k) / scale);
      q[n * inputs + k] = int8_t(std::clamp(value, -127.0f, 127.0f));
    }
  }
}
} // namespace npu
