//===- DriverImage.cpp - decode LP6 image writer ---------------------------===//
//
// lp6.bin: FP16 dense weights, per-row weight scales, PIM weight packets and NPU GEMV weight
// blocks (INT8, per-row symmetric), NPU constant pools.
//===----------------------------------------------------------------------===//

#include "DriverImage.h"
#include "npu/Decode/Support/SafeTensors.h"
#include "npu/Decode/Transforms/Passes.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <mutex>
#include <sys/mman.h>
#include <unistd.h>

using namespace mlir;
using namespace mlir::hybrid;
namespace json = llvm::json;

namespace {

/// Per-row symmetric INT8: s = max|W[n,:]| / 127 (FP32),
/// q = clip(rint(W / s), -127, 127).
bool quantizeRows(const SafeTensors &st, const std::string &name, std::vector<int8_t> &q, std::vector<float> &s,
                  int64_t &n, int64_t &k, std::string &err) {
  const SafeTensors::Tensor *t = st.find(name);
  if (!t || t->shape.size() != 2) {
    err = "no 2-D tensor " + name;
    return false;
  }
  n = t->shape[0];
  k = t->shape[1];
  bool bf = t->dtype == "BF16";
  if (!bf && t->dtype != "F16") {
    err = "unsupported dtype " + t->dtype;
    return false;
  }
  q.resize(size_t(n * k));
  s.resize(size_t(n));
  std::vector<float> row(static_cast<size_t>(k));
  for (int64_t r = 0; r < n; ++r) {
    const uint16_t *src = reinterpret_cast<const uint16_t *>(t->data) + r * k;
    float mx = 0;
    for (int64_t c = 0; c < k; ++c) {
      uint16_t v;
      std::memcpy(&v, src + c, 2);
      row[size_t(c)] = bf ? bf16ToF32(v) : f16ToF32(v);
      mx = std::max(mx, std::fabs(row[size_t(c)]));
    }
    float sc = mx / 127.0f;
    if (!(sc > 0)) sc = 1e-12f;
    s[size_t(r)] = sc;
    for (int64_t c = 0; c < k; ++c) {
      float v = std::nearbyint(row[size_t(c)] / sc);
      q[size_t(r * k + c)] = int8_t(std::min(127.0f, std::max(-127.0f, v)));
    }
  }
  return true;
}

class Image {
public:
  bool open(const std::string &path, uint64_t bytes, std::string &err) {
    size = (bytes + 63) / 64 * 64;
    fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0 || ftruncate(fd, off_t(size)) != 0) {
      err = "cannot create " + path;
      return false;
    }
    void *m = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (m == MAP_FAILED) {
      err = "cannot mmap " + path;
      return false;
    }
    base = static_cast<uint8_t *>(m);
    return true;
  }
  void write(uint64_t addr, const void *data, size_t n) {
    if (addr + n > size) {
      llvm::errs() << "image write out of range at " << addr << "\n";
      std::abort();
    }
    std::memcpy(base + addr, data, n);
  }
  ~Image() {
    if (base) munmap(base, size);
    if (fd >= 0) ::close(fd);
  }
  uint64_t size = 0;

private:
  int fd = -1;
  uint8_t *base = nullptr;
};

} // namespace

bool driver::writeImage(const std::string &model, const std::string &outDir, uint64_t imageBytes, ModuleOp mod,
                        func::FuncOp step, const pim::Program &prog, const TargetSpec &spec, const LP6Extra &extra,
                        size_t &quantized, std::string &err) {
  auto st = SafeTensors::open(model, err);
  if (!st) {
    llvm::errs() << err << "\n";
    return false;
  }
  Image img;
  if (!img.open(outDir + "/lp6.bin", imageBytes, err)) {
    llvm::errs() << err << "\n";
    return false;
  }
  for (BlockArgument a : step.getArguments()) {
    auto kind = step.getArgAttrOfType<StringAttr>(a.getArgNumber(), kKind);
    if (!kind || kind.getValue() != "weight") continue;
    std::string name = step.getArgAttrOfType<StringAttr>(a.getArgNumber(), kName).str();
    std::vector<float> w;
    std::vector<int64_t> shape;
    if (!st->toF32(name, w, shape, err)) {
      llvm::errs() << err << "\n";
      return false;
    }
    auto ty = cast<ShapedType>(a.getType());
    if (int64_t(w.size()) != ty.getNumElements() || !ty.getElementType().isF16()) {
      llvm::errs() << "dense weight " << name << " does not match its argument\n";
      return false;
    }
    std::vector<uint16_t> h(w.size());
    for (size_t i = 0; i < w.size(); ++i) h[i] = f32ToF16(w[i]);
    img.write(uint64_t(step.getArgAttrOfType<DictionaryAttr>(a.getArgNumber(), kLP6).getAs<IntegerAttr>("addr").getInt()),
              h.data(), h.size() * 2);
  }
  for (const LP6Blob &b : extra.blobs) img.write(b.addr, b.bytes.data(), b.bytes.size());
  // Quantized weights: PIM packets, NPU GEMV blocks, per-row scales (hyb.weights), all from one
  // per-row INT8 quantization of the checkpoint tensor.
  struct NpuWeight {
    std::string layout;
    uint64_t addr, bytes;
    std::vector<int64_t> shape;
  };
  std::map<std::string, std::vector<NpuWeight>> npuWeights;
  if (auto wa = mod->getAttrOfType<StringAttr>("hyb.weights"))
    if (auto wj = json::parse(wa.getValue()))
      for (const json::Value &v : *wj->getAsArray()) {
        const json::Object *o = v.getAsObject();
        std::vector<int64_t> shape;
        for (const auto &d : *o->getArray("shape")) shape.push_back(*d.getAsInteger());
        npuWeights[o->getString("name")->str()].push_back(
            {o->getString("layout")->str(), uint64_t(*o->getInteger("addr")),
             uint64_t(*o->getInteger("bytes")),std::move(shape)});
      }
  std::vector<std::string> names;
  for (const pim::Projection &p : prog.projections)
    if (!p.common && std::find(names.begin(), names.end(), p.name) == names.end()) names.push_back(p.name);
  for (auto &[name, list] : npuWeights)
    if (std::find(names.begin(), names.end(), name) == names.end()) names.push_back(name);
  std::atomic<bool> ok{true};
  std::mutex errMutex;
  llvm::parallelFor(0, names.size(), [&](size_t i) {
    std::vector<int8_t> q;
    std::vector<float> s;
    int64_t n, k;
    std::string e;
    auto fail = [&](const std::string &msg) {
      std::lock_guard<std::mutex> g(errMutex);
      llvm::errs() << msg << "\n";
      ok = false;
    };
    if (!quantizeRows(*st, names[i], q, s, n, k, e)) return fail(e);
    if (auto it = npuWeights.find(names[i]); it != npuWeights.end())
      for (const NpuWeight &nw : it->second) {
        if (nw.layout == "row_scale") {
          if (nw.bytes != uint64_t(n)*4 || nw.shape != std::vector<int64_t>{n})
            return fail("row_scale shape mismatch: " + names[i]);
          img.write(nw.addr, s.data(), s.size() * 4);
        } else if (nw.layout == "npu_nk32") {
          if (n % 32) return fail("npu_nk32 needs N % 32 == 0: " + names[i]);
          if (nw.bytes != uint64_t(n)*k || nw.shape != std::vector<int64_t>{n/32,k,32})
            return fail("npu_nk32 shape mismatch: " + names[i]);
          // Research option --common-layout: the same rows at the placement's addresses (NPU view of the common layout).
          const pim::CommonWeight *common = nullptr;
          for (const pim::CommonWeight &c : extra.commonWeights)
            if (nw.addr >= c.logical && nw.addr < c.logical + c.bytes) common = &c;
          if (common) {
            const int64_t firstTile = int64_t(nw.addr - common->logical) / (k * 32);
            std::vector<int8_t> line(32);
            for (int64_t j = 0; j < n / 32; ++j)
              for (int64_t kk = 0; kk < k; ++kk) {
                for (int c = 0; c < 32; ++c) line[size_t(c)] = q[size_t((j * 32 + c) * k + kk)];
                img.write(common->address(firstTile + j, kk), line.data(), 32);
              }
            continue;
          }
          // Block j = rows 32j..32j+31 as the [K, 32] MMA weight tile: T[k][c] = q[32j + c][k].
          std::vector<int8_t> blk(size_t(k * 32));
          for (int64_t j = 0; j < n / 32; ++j) {
            for (int64_t kk = 0; kk < k; ++kk)
              for (int c = 0; c < 32; ++c) blk[size_t(kk * 32 + c)] = q[size_t((j * 32 + c) * k + kk)];
            img.write(nw.addr + uint64_t(j * k * 32), blk.data(), blk.size());
          }
        } else {
          return fail("unknown weight layout " + nw.layout);
        }
      }
    for (const pim::Projection &p : prog.projections) {
      if (p.common || p.name != names[i]) continue;
      if (p.n != n || p.k != k) return fail("projection " + p.name + " shape mismatch");
      prog.forEachWeightPacket(p, q.data(), [&](uint64_t addr, const std::array<uint8_t, 32> &pkt) {
        img.write(addr, pkt.data(), 32);
      });
    }
  });
  if (!ok) return false;
  // Research option --common-layout: a group of GEMVs is one projection with the rows of its tensors in order.
  std::vector<size_t> commons;
  for (size_t i = 0; i < prog.projections.size(); ++i)
    if (prog.projections[i].common) commons.push_back(i);
  llvm::parallelFor(0, commons.size(), [&](size_t i) {
    const pim::Projection &p = prog.projections[commons[i]];
    std::vector<int8_t> all(size_t(p.n * p.k));
    int64_t at = 0;
    for (const auto &seg : p.segments) {
      std::vector<int8_t> q;
      std::vector<float> s;
      int64_t n, k;
      std::string e;
      if (!quantizeRows(*st, seg.first, q, s, n, k, e) || n != seg.second || k != p.k) {
        std::lock_guard<std::mutex> g(errMutex);
        llvm::errs() << "common projection " << seg.first << ": " << e << "\n";
        ok = false;
        return;
      }
      std::memcpy(all.data() + at * p.k, q.data(), q.size());
      at += n;
    }
    prog.forEachWeightPacket(p, all.data(), [&](uint64_t addr, const std::array<uint8_t, 32> &pkt) { img.write(addr, pkt.data(), 32); });
  });
  if (!ok) return false;
  quantized = names.size() + commons.size();
  return true;
}
