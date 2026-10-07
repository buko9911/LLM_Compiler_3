#include "npu/Decode/Support/SafeTensors.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include <cstring>
#include <fcntl.h>
#include <set>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace mlir::hybrid;

float mlir::hybrid::bf16ToF32(uint16_t v) {
  uint32_t x = uint32_t(v) << 16;
  float f;
  std::memcpy(&f, &x, 4);
  return f;
}

float mlir::hybrid::f16ToF32(uint16_t h) {
  uint32_t sign = uint32_t(h & 0x8000) << 16, exp = (h >> 10) & 0x1F, mant = h & 0x3FF, x;
  if (exp == 0) {
    if (mant == 0) x = sign;
    else {
      int e = -1;
      do {
        ++e;
        mant <<= 1;
      } while (!(mant & 0x400));
      x = sign | uint32_t(127 - 15 - e) << 23 | (mant & 0x3FF) << 13;
    }
  } else if (exp == 31) {
    x = sign | 0x7F800000 | mant << 13;
  } else {
    x = sign | (exp + 127 - 15) << 23 | mant << 13;
  }
  float f;
  std::memcpy(&f, &x, 4);
  return f;
}

uint16_t mlir::hybrid::f32ToF16(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000, mant = x & 0x7FFFFF;
  int32_t exp = int32_t((x >> 23) & 0xFF) - 127 + 15;
  if (((x >> 23) & 0xFF) == 0xFF) return uint16_t(sign | 0x7C00 | (mant ? 0x200 : 0));
  if (exp >= 31) return uint16_t(sign | 0x7C00);
  if (exp <= 0) {
    if (exp < -10) return uint16_t(sign);
    mant |= 0x800000;
    uint32_t shift = uint32_t(14 - exp);
    uint32_t half = mant >> shift, rem = mant & ((1u << shift) - 1), mid = 1u << (shift - 1);
    if (rem > mid || (rem == mid && (half & 1))) half++;
    return uint16_t(sign | half);
  }
  uint32_t half = uint32_t(exp) << 10 | (mant >> 13);
  uint32_t rem = mant & 0x1FFF;
  if (rem > 0x1000 || (rem == 0x1000 && (half & 1))) half++;
  return uint16_t(sign | half);
}

SafeTensors::~SafeTensors() {
  for (Shard &s : shards)
    if (s.map) munmap(s.map, s.size);
}

bool SafeTensors::addShard(const std::string &path, std::string &err) {
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    err = "cannot open " + path;
    return false;
  }
  struct stat st;
  fstat(fd, &st);
  void *m = mmap(nullptr, size_t(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (m == MAP_FAILED) {
    err = "cannot mmap " + path;
    return false;
  }
  shards.push_back({m, size_t(st.st_size)});
  const uint8_t *base = static_cast<const uint8_t *>(m);
  uint64_t headerLen;
  std::memcpy(&headerLen, base, 8);
  auto header = llvm::json::parse(llvm::StringRef(reinterpret_cast<const char *>(base + 8), headerLen));
  if (!header) {
    err = "bad safetensors header in " + path;
    llvm::consumeError(header.takeError());
    return false;
  }
  const uint8_t *data = base + 8 + headerLen;
  for (auto &[key, value] : *header->getAsObject()) {
    if (key == "__metadata__") continue;
    const llvm::json::Object *o = value.getAsObject();
    Tensor t;
    t.dtype = o->getString("dtype")->str();
    for (const llvm::json::Value &d : *o->getArray("shape")) t.shape.push_back(*d.getAsInteger());
    const llvm::json::Array *offs = o->getArray("data_offsets");
    t.data = data + *(*offs)[0].getAsInteger();
    tensors[key.str()] = t;
  }
  return true;
}

std::unique_ptr<SafeTensors> SafeTensors::open(const std::string &dir, std::string &err) {
  auto st = std::unique_ptr<SafeTensors>(new SafeTensors());
  auto index = llvm::MemoryBuffer::getFile(dir + "/model.safetensors.index.json");
  if (!index) {
    if (!st->addShard(dir + "/model.safetensors", err)) return nullptr;
    return st;
  }
  auto json = llvm::json::parse((*index)->getBuffer());
  if (!json) {
    err = "bad index json";
    llvm::consumeError(json.takeError());
    return nullptr;
  }
  std::set<std::string> files;
  for (auto &[k, v] : *json->getAsObject()->getObject("weight_map")) files.insert(v.getAsString()->str());
  for (const std::string &f : files)
    if (!st->addShard(dir + "/" + f, err)) return nullptr;
  return st;
}

const SafeTensors::Tensor *SafeTensors::find(const std::string &name) const {
  auto it = tensors.find(name);
  return it == tensors.end() ? nullptr : &it->second;
}

bool SafeTensors::toF32(const std::string &name, std::vector<float> &out, std::vector<int64_t> &shape,
                        std::string &err) const {
  const Tensor *t = find(name);
  if (!t) {
    err = "checkpoint has no tensor " + name;
    return false;
  }
  shape = t->shape;
  int64_t n = t->numel();
  out.resize(size_t(n));
  if (t->dtype == "BF16" || t->dtype == "F16") {
    bool bf = t->dtype == "BF16";
    for (int64_t i = 0; i < n; ++i) {
      uint16_t v;
      std::memcpy(&v, t->data + 2 * i, 2);
      out[size_t(i)] = bf ? bf16ToF32(v) : f16ToF32(v);
    }
  } else if (t->dtype == "F32") {
    std::memcpy(out.data(), t->data, size_t(n) * 4);
  } else {
    err = "unsupported dtype " + t->dtype + " for " + name;
    return false;
  }
  return true;
}
