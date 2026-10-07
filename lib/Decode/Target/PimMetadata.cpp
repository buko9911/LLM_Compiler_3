#include "npu/Decode/PimMetadata.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <limits>
#include <set>

namespace npu::decode {
using namespace ::npu::pim;
llvm::json::Object pimRuntimeStorageLayout(const Program &program) {
  const auto base = program.baseRows();
  const int64_t outputRows = program.nextOutput / kPacketsPerRow +
                            (program.nextOutput % kPacketsPerRow != 0);
  // Preserve Program's existing activation reservation by logical tile count.
  // Runtime packet packing must not shrink this region or shift later regions.
  return llvm::json::Object{
      {"version", 1}, {"packets_per_row", kPacketsPerRow},
      {"weight_base_row", base.weight},
      {"weight_row_count", base.activation - base.weight},
      {"activation_base_row", base.activation},
      {"activation_row_count", base.output - base.activation},
      {"scale_base_row", base.output}, {"scale_row_count", 0},
      {"output_base_row", base.output}, {"output_row_count", outputRows}};
}

llvm::json::Value toJSON(const Program &program) {

  llvm::json::Array a, p, ph;
  for (const Activation &x : program.activations)
    a.push_back(llvm::json::Object{{"k", x.k}, {"base", x.base}, {"parts", llvm::json::Array(x.parts)}});
  for (const Projection &x : program.projections)
    p.push_back(llvm::json::Object{{"weight", x.name}, {"n", x.n}, {"k", x.k}, {"n_pad", x.nPad},
                                   {"stream_weight_base", llvm::json::Array(x.streamWeightBase)}, {"output_base", x.outputBase},
                                   {"acts", llvm::json::Array(x.activations)}, {"stream_part", llvm::json::Array(x.streamPart)},
                                   {"compute_first", x.computeFirst}, {"compute_count", x.computeCount}});
  for (auto [f, c] : program.phases) ph.push_back(llvm::json::Array{f, c});
  return llvm::json::Object{{"channels", program.channels}, {"partitions", program.partitions}, {"kernel_row", program.kernelRow},
                            {"weight_layout", program.weightLayout()}, {"bankset_pingpong_lead", program.banksetPingPongLead},
                            {"next_w", llvm::json::Array(program.nextWeight)},
                            {"next_a", program.nextActivation}, {"next_o", program.nextOutput}, {"acts", std::move(a)},
                            {"projections", std::move(p)}, {"phases", std::move(ph)}};
}

Program fromJSON(const llvm::json::Value &v) {
  Program pr;
  const llvm::json::Object *o = v.getAsObject();
  pr.channels = int(o->getInteger("channels").value_or(2));
  pr.partitions = int(*o->getInteger("partitions"));
  pr.kernelRow = *o->getInteger("kernel_row");
  pr.banksetPingPongLead = int(o->getInteger("bankset_pingpong_lead").value_or(0));
  if (pr.banksetPingPongLead < 0 || pr.banksetPingPongLead > 63 || pr.banksetPingPongLead == 1 ||
      o->getString("weight_layout").value_or("even-v1") != pr.weightLayout())
    llvm::report_fatal_error("inconsistent PIM weight layout / ping-pong lead");
  pr.nextWeight.clear();
  for (const llvm::json::Value &c : *o->getArray("next_w")) pr.nextWeight.push_back(*c.getAsInteger());
  pr.nextActivation = *o->getInteger("next_a");
  pr.nextOutput = *o->getInteger("next_o");
  for (const llvm::json::Value &x : *o->getArray("acts")) {
    const llvm::json::Object *xo = x.getAsObject();
    Activation a;
    a.k = *xo->getInteger("k");
    a.base = *xo->getInteger("base");
    for (const llvm::json::Value &c : *xo->getArray("parts")) a.parts.push_back(int(*c.getAsInteger()));
    pr.activations.push_back(a);
  }
  for (const llvm::json::Value &x : *o->getArray("projections")) {
    const llvm::json::Object *xo = x.getAsObject();
    Projection p;
    p.name = xo->getString("weight")->str();
    p.n = *xo->getInteger("n");
    p.k = *xo->getInteger("k");
    p.nPad = *xo->getInteger("n_pad");
    for (const llvm::json::Value &c : *xo->getArray("stream_weight_base")) p.streamWeightBase.push_back(*c.getAsInteger());
    p.outputBase = *xo->getInteger("output_base");
    for (const llvm::json::Value &c : *xo->getArray("acts")) p.activations.push_back(*c.getAsInteger());
    for (const llvm::json::Value &c : *xo->getArray("stream_part")) p.streamPart.push_back(int(*c.getAsInteger()));
    p.computeFirst = *xo->getInteger("compute_first");
    p.computeCount = *xo->getInteger("compute_count");
    pr.projections.push_back(p);
  }
  for (const llvm::json::Value &x : *o->getArray("phases")) {
    const llvm::json::Array *pa = x.getAsArray();
    pr.phases.push_back({*(*pa)[0].getAsInteger(), *(*pa)[1].getAsInteger()});
  }
  return pr;
}

} // namespace npu::decode

namespace npu::decode {
namespace {
constexpr const char *storageSchema = "lp6-pim-weight-storage-v1";
constexpr const char *storageMapping = "rarobacobgch-32B-rank8GiB-v1";
constexpr const char *storageQuant = "row-symmetric-i8-fp32-maxabs-div127-nearbyint-clip127-zero1e-12-v1";
bool digestValid(const std::string &s) {
  return s.size() == 64 && std::all_of(s.begin(), s.end(), [](char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
  });
}
} // namespace

void PimStorageDigest::integer(uint64_t value) {
  uint8_t b[8];
  for (int i = 0; i < 8; ++i) b[i] = uint8_t(value >> (8 * i));
  hash.update(llvm::ArrayRef<uint8_t>(b, 8));
}
void PimStorageDigest::text(llvm::StringRef value) {
  integer(value.size());
  hash.update(value);
}
std::string PimStorageDigest::finish() {
  constexpr char digits[] = "0123456789abcdef";
  std::string result;
  for (uint8_t b : hash.final()) { result += digits[b >> 4]; result += digits[b & 15]; }
  return result;
}

std::string pimWeightLayoutDigest(const PimWeightStorage &s) {
  PimStorageDigest h;
  h.text(storageSchema); h.text(storageMapping); h.text(storageQuant);
  for (int64_t v : {int64_t(s.channels), int64_t(s.partitions), int64_t(pim::kRankBytes),
                    int64_t(32), int64_t(pim::kPUs), int64_t(pim::kLanes), s.kernelRow,
                    int64_t(s.banksetPingPongLead), s.capacityBytes}) h.integer(uint64_t(v));
  h.integer(s.weights.size());
  for (const auto &w : s.weights) {
    h.text(w.name); h.integer(w.n); h.integer(w.k); h.integer(w.nPad);
    h.integer(w.scaleAddress); h.integer(w.scaleBytes); h.integer(w.streamPart.size());
    for (int p : w.streamPart) h.integer(p);
    for (int64_t b : w.streamWeightBase) h.integer(b);
  }
  return h.finish();
}
std::string pimWeightManifestDigest(const PimWeightStorage &s) {
  PimStorageDigest h;
  h.text("lp6-pim-weight-manifest-digest-v1"); h.text(pimWeightLayoutDigest(s));
  for (const auto &w : s.weights) {
    h.text(w.name); h.text(w.sourceDtype); h.text(w.sourceSHA256);
    h.text(w.packetSHA256); h.text(w.scaleSHA256);
  }
  return h.finish();
}

Result<bool> validatePimWeightStorage(const PimWeightStorage &s, bool content) {
  auto fail = [](const std::string &m) { return Result<bool>::failure("PIM weight storage: " + m); };
  if ((s.channels != 2 && s.channels != 8) || s.partitions < 1 || s.partitions > s.channels * pim::kRanks)
    return fail("unsupported channel/partition geometry");
  if (s.capacityBytes <= 0 || s.capacityBytes > pim::kRankBytes * pim::kRanks)
    return fail("invalid LP6 capacity");
  if (s.banksetPingPongLead < 0 || s.banksetPingPongLead == 1 || s.banksetPingPongLead > 63)
    return fail("invalid bankset layout/lead");
  pim::Program geometry;
  geometry.channels = s.channels;
  const int64_t rowBytes = geometry.rowBytes(), rankRows = pim::kRankBytes / rowBytes;
  if (s.kernelRow <= 0 || s.kernelRow >= rankRows) return fail("kernel row outside rank");
  if (s.weights.empty()) return fail("empty weight inventory");
  const uint64_t weightBegin = uint64_t(s.kernelRow * rowBytes);
  std::vector<std::vector<std::pair<int64_t, int64_t>>> intervals(size_t(s.partitions));
  std::vector<std::pair<uint64_t, uint64_t>> scales;
  std::string previous;
  for (const auto &w : s.weights) {
    if (w.name.empty() || (!previous.empty() && w.name <= previous))
      return fail("weight names must be unique and sorted");
    previous = w.name;
    if (w.n <= 0 || w.k <= 0 || w.k % pim::kKPerSweep ||
        w.n > s.capacityBytes || w.k > s.capacityBytes ||
        w.nPad != (w.n + pim::kRowsPerStream - 1) / pim::kRowsPerStream * pim::kRowsPerStream ||
        w.nPad > s.capacityBytes / w.k)
      return fail(w.name + ": invalid N/K/Npad or K is not a multiple of 256");
    const int64_t streams = w.nPad / pim::kRowsPerStream;
    const int64_t tiles = (w.k / pim::kKPerSweep) * pim::kAccs * pim::kInTiles;
    if (w.streamPart.size() != size_t(streams) || w.streamWeightBase.size() != size_t(streams) ||
        !std::is_sorted(w.streamPart.begin(), w.streamPart.end()))
      return fail(w.name + ": invalid stream arrays/order");
    for (size_t i = 0; i < w.streamPart.size(); ++i) {
      int part = w.streamPart[i];
      int64_t base = w.streamWeightBase[i];
      if (part < 0 || part >= s.partitions || base < 0 || tiles > INT32_MAX || base > INT32_MAX - tiles)
        return fail(w.name + ": invalid stream partition/tile interval");
      int64_t end = base + tiles, endRow = s.kernelRow + (end + pim::kPacketsPerRow - 1) / pim::kPacketsPerRow;
      if (endRow > rankRows || uint64_t(part / s.channels) * pim::kRankBytes + uint64_t(endRow * rowBytes) >
                                  uint64_t(s.capacityBytes))
        return fail(w.name + ": weight rows exceed rank/capacity");
      intervals[size_t(part)].push_back({base, end});
    }
    if (w.scaleBytes != uint64_t(w.n) * 4 || w.scaleAddress < 0x10000 || w.scaleAddress % 4 ||
        w.scaleAddress > weightBegin || w.scaleBytes > weightBegin - w.scaleAddress)
      return fail(w.name + ": scale range overlaps kernel/zero page or has invalid size/alignment");
    scales.push_back({w.scaleAddress, w.scaleAddress + w.scaleBytes});
    if (content && ((w.sourceDtype != "F16" && w.sourceDtype != "BF16") ||
                    !digestValid(w.sourceSHA256) || !digestValid(w.packetSHA256) || !digestValid(w.scaleSHA256)))
      return fail(w.name + ": missing/invalid source or content digest");
  }
  for (auto &v : intervals) {
    std::sort(v.begin(), v.end());
    for (size_t i = 1; i < v.size(); ++i)
      if (v[i].first < v[i-1].second) return fail("overlapping weight tile intervals");
  }
  std::sort(scales.begin(), scales.end());
  for (size_t i = 1; i < scales.size(); ++i)
    if (scales[i].first < scales[i-1].second) return fail("overlapping scale ranges");
  return Result<bool>::success(true);
}

llvm::json::Value toJSON(const PimWeightStorage &s) {
  llvm::json::Array weights;
  for (const auto &w : s.weights)
    weights.push_back(llvm::json::Object{
      {"name", w.name}, {"n", w.n}, {"k", w.k}, {"n_pad", w.nPad},
      {"stream_part", llvm::json::Array(w.streamPart)}, {"stream_weight_base", llvm::json::Array(w.streamWeightBase)},
      {"scale_address", int64_t(w.scaleAddress)}, {"scale_bytes", int64_t(w.scaleBytes)},
      {"source_dtype", w.sourceDtype}, {"source_sha256", w.sourceSHA256},
      {"packet_sha256", w.packetSHA256}, {"scale_sha256", w.scaleSHA256}});
  return llvm::json::Object{
    {"schema", storageSchema}, {"mapping", storageMapping}, {"quantization", storageQuant},
    {"channels", s.channels}, {"partitions", s.partitions}, {"rank_bytes", pim::kRankBytes},
    {"packet_bytes", 32}, {"pus", pim::kPUs}, {"lanes", pim::kLanes},
    {"kernel_row", s.kernelRow}, {"bankset_pingpong_lead", s.banksetPingPongLead},
    {"capacity_bytes", s.capacityBytes}, {"weights", std::move(weights)},
    {"layout_sha256", pimWeightLayoutDigest(s)}, {"manifest_sha256", pimWeightManifestDigest(s)}};
}

Result<PimWeightStorage> parsePimWeightStorage(const llvm::json::Value &v) {
  auto fail = [](const std::string &m) { return Result<PimWeightStorage>::failure("PIM weight storage: " + m); };
  const auto *o = v.getAsObject();
  if (!o) return fail("manifest must be an object");
  if (o->getString("schema") != storageSchema || o->getString("mapping") != storageMapping ||
      o->getString("quantization") != storageQuant || o->getInteger("rank_bytes") != pim::kRankBytes ||
      o->getInteger("packet_bytes") != 32 || o->getInteger("pus") != pim::kPUs || o->getInteger("lanes") != pim::kLanes)
    return fail("missing or unsupported schema/mapping/quantization/hardware contract");
  auto ch = o->getInteger("channels"), parts = o->getInteger("partitions"), lead = o->getInteger("bankset_pingpong_lead");
  auto row = o->getInteger("kernel_row"), cap = o->getInteger("capacity_bytes");
  auto layoutHash = o->getString("layout_sha256"), manifestHash = o->getString("manifest_sha256");
  const auto *weights = o->getArray("weights");
  if (!ch || !parts || !lead || !row || !cap || !weights || !layoutHash || !manifestHash ||
      *ch < 1 || *ch > 8 || *parts < 1 || *parts > 16 || *lead < 0 || *lead > 63)
    return fail("missing/invalid manifest field");
  PimWeightStorage s;
  s.channels = int(*ch); s.partitions = int(*parts); s.banksetPingPongLead = int(*lead);
  s.kernelRow = *row; s.capacityBytes = *cap;
  for (const auto &value : *weights) {
    const auto *x = value.getAsObject();
    if (!x) return fail("weight entry must be an object");
    auto name = x->getString("name"), dtype = x->getString("source_dtype");
    auto source = x->getString("source_sha256"), packet = x->getString("packet_sha256"), scale = x->getString("scale_sha256");
    auto n = x->getInteger("n"), k = x->getInteger("k"), pad = x->getInteger("n_pad");
    auto addr = x->getInteger("scale_address"), bytes = x->getInteger("scale_bytes");
    const auto *p = x->getArray("stream_part"), *b = x->getArray("stream_weight_base");
    if (!name || !dtype || !source || !packet || !scale || !n || !k || !pad || !addr || !bytes || !p || !b ||
        *addr < 0 || *bytes < 0) return fail("missing/invalid weight entry field");
    PimWeightStorageEntry e;
    e.name = name->str(); e.sourceDtype = dtype->str(); e.sourceSHA256 = source->str();
    e.packetSHA256 = packet->str(); e.scaleSHA256 = scale->str();
    e.n = *n; e.k = *k; e.nPad = *pad; e.scaleAddress = uint64_t(*addr); e.scaleBytes = uint64_t(*bytes);
    for (const auto &part : *p) {
      auto pi = part.getAsInteger();
      if (!pi || *pi < 0 || *pi > 15) return fail("invalid stream_part element");
      e.streamPart.push_back(int(*pi));
    }
    for (const auto &base : *b) {
      auto bi = base.getAsInteger();
      if (!bi) return fail("invalid stream_weight_base element");
      e.streamWeightBase.push_back(*bi);
    }
    s.weights.push_back(std::move(e));
  }
  auto checked = validatePimWeightStorage(s, true);
  if (!checked) return Result<PimWeightStorage>::failure(checked.error);
  if (*layoutHash != pimWeightLayoutDigest(s)) return fail("layout SHA256 mismatch");
  if (*manifestHash != pimWeightManifestDigest(s)) return fail("manifest SHA256 mismatch");
  return Result<PimWeightStorage>::success(std::move(s));
}

Result<bool> bindPimWeightStorage(pim::Program &p, const PimWeightStorage &s) {
  auto fail = [](const std::string &m) { return Result<bool>::failure("PIM weight storage: " + m); };
  if (p.channels != s.channels || p.partitions != s.partitions || p.banksetPingPongLead != s.banksetPingPongLead)
    return fail("target hardware/layout differs from imported storage");
  if (p.projections.size() != s.weights.size()) return fail("projection inventory differs from imported storage");
  std::map<std::string, const PimWeightStorageEntry *> byName;
  for (const auto &w : s.weights) byName[w.name] = &w;
  std::set<std::string> seen;
  for (const auto &q : p.projections) {
    auto it = byName.find(q.name);
    if (!seen.insert(q.name).second) return fail("repeated PIM weight names are not supported");
    if (it == byName.end() || q.n != it->second->n || q.k != it->second->k || q.nPad != it->second->nPad)
      return fail(q.name + ": projection name/shape differs from imported storage");
  }
  p.kernelRow = s.kernelRow;
  p.nextWeight.assign(size_t(p.partitions), 0);
  for (auto &a : p.activations) a.parts.clear();
  for (auto &q : p.projections) {
    const auto &w = *byName.at(q.name);
    q.streamPart = w.streamPart; q.streamWeightBase = w.streamWeightBase;
    for (size_t i = 0; i < q.streamPart.size(); ++i) {
      const int part = q.streamPart[i];
      p.nextWeight[size_t(part)] = std::max(p.nextWeight[size_t(part)], q.streamWeightBase[i] + q.streamTiles());
      for (int64_t ai : q.activations) {
        if (ai < 0 || ai >= int64_t(p.activations.size())) return fail("invalid new invocation activation");
        auto &parts = p.activations[size_t(ai)].parts;
        if (std::find(parts.begin(), parts.end(), part) == parts.end()) parts.push_back(part);
      }
    }
  }
  for (auto &a : p.activations) std::sort(a.parts.begin(), a.parts.end());
  return Result<bool>::success(true);
}
} // namespace npu::decode
