//===- SafeTensors.h - read-only mmap view of a (sharded) checkpoint -----*- C++ -*-===//
#ifndef HYBRID_SUPPORT_SAFETENSORS_H
#define HYBRID_SUPPORT_SAFETENSORS_H

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mlir::hybrid {

class SafeTensors {
public:
  struct Tensor {
    std::string dtype;               // "BF16", "F16", "F32"
    std::vector<int64_t> shape;
    const uint8_t *data = nullptr;
    int64_t numel() const {
      int64_t n = 1;
      for (int64_t d : shape) n *= d;
      return n;
    }
  };
  /// Opens <dir>/model.safetensors.index.json (or <dir>/model.safetensors).
  static std::unique_ptr<SafeTensors> open(const std::string &dir, std::string &err);
  ~SafeTensors();
  const Tensor *find(const std::string &name) const;
  /// Tensor converted to FP32 (row-major).
  bool toF32(const std::string &name, std::vector<float> &out, std::vector<int64_t> &shape, std::string &err) const;

private:
  struct Shard {
    void *map = nullptr;
    size_t size = 0;
  };
  std::vector<Shard> shards;
  std::map<std::string, Tensor> tensors;
  bool addShard(const std::string &path, std::string &err);
};

float bf16ToF32(uint16_t v);
float f16ToF32(uint16_t v);
uint16_t f32ToF16(float f);

} // namespace mlir::hybrid

#endif
