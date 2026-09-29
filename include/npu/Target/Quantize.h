#ifndef NPU_TARGET_QUANTIZE_H
#define NPU_TARGET_QUANTIZE_H
#include <cstdint>

namespace npu {
// FP16 bits to FP32, exactly (every FP16 value is an FP32 value).
float halfToFloat(uint16_t bits);

// W8A8 weight quantization, per output channel, the contract quantizeGraph
// states (lib/Transforms/Quantize.cpp):
//
//   s[n]    = fl32(max_k |W[n,k]|) / 127        FP32 division
//   q[n,k]  = clamp(rne(W[n,k] / s[n]), -127, 127),  0 where s[n] = 0
//
// `weights` holds FP16 bits of W with N output channels and K inputs, either
// channel-major ([N][K], a PyTorch linear weight) or input-major ([K][N]).
// q is written channel-major, [N][K]; scales has N entries.
void quantizeChannels(const uint16_t *weights, uint64_t outputs, uint64_t inputs, bool channelMajor,
                      int8_t *q, float *scales);
} // namespace npu
#endif
