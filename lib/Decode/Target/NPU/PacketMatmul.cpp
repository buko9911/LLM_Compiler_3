//===- PacketMatmul.cpp - NPU GEMV over PIM weight packets (--pim-packet-npu=g4) -===//
//
// The NPU reads the INT8 weights of a projection directly from an imported PIM weight image
// (packet layout) instead of a dense NPU copy. Every physical weight address comes from
// WeightPacketLocation; the schedule is checked against the fixed hardware.
//===----------------------------------------------------------------------===//

#include "npu/Decode/Target/PacketMatmul.h"
#include "npu/Decode/Dialect/Hyb/HybDialect.h"
#include "npu/Target/ISAv2.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <set>

using namespace mlir;
using namespace mlir::hybrid;
using namespace mlir::hybrid::plena;

namespace {
constexpr uint32_t kCores = 4, kG = 4, kSeg = 256, kMaxBlock = 2048;
constexpr uint32_t kX = 0x4000, kA = 0x10000, kC = 0x18000, kE = 0x20000;
constexpr uint32_t kB = 0x40000, kY = 0x200000, kRawL2 = 0x40000, kYL2 = 0x500000;
constexpr uint64_t kNoAddress = std::numeric_limits<uint64_t>::max();

bool overlap(uint64_t a, uint64_t bytes, uint64_t b, uint64_t otherBytes) {
  return a < b + otherBytes && b < a + bytes;
}

struct Dense {
  uint64_t address = 0, bytes = 0;
  uint32_t rows = 0, columns = 0, elementBytes = 0;
};

struct Range {
  std::string name;
  uint64_t offset, bytes;
};

struct Stripe {
  std::array<uint32_t, 128> output{};
  std::vector<uint64_t> segments;
  uint32_t core = 0;
};

struct Plan {
  Dense x, y;
  uint32_t m = 0, n = 0, k = 0, block = 0, slot = 0;
  uint64_t l1Peak = 0, l2Peak = 0, eventsUpperBound = 0, packetCount = 0;
  int64_t projection = -1;
  std::vector<Range> l1, l2;
  std::vector<Stripe> stripes;
  std::array<std::vector<size_t>, kCores> byCore;
  std::string json;
};

LogicalResult fail(std::string &error, const std::string &message) {
  error = "pim_packet_g4: " + message;
  return failure();
}

LogicalResult denseArgument(const NpuPhaseInput &in, Value value, unsigned bits,
                            Dense &view, std::string &error) {
  func::FuncOp phase = in.phase;
  auto argument = dyn_cast<BlockArgument>(value);
  auto type = dyn_cast<MemRefType>(value.getType());
  if (!argument || argument.getOwner() != &phase.getBody().front() ||
      argument.getArgNumber() >= in.argLP6.size())
    return fail(error, "input and init must be direct callee block arguments");
  if (!type || type.getRank() != 2 || !type.hasStaticShape() ||
      !type.getElementType().isInteger(bits) || type.getDimSize(0) <= 0 || type.getDimSize(1) <= 0)
    return fail(error, "expected static positive dense i8[M,K] input and i32[M,N] init");
  if (Attribute space = type.getMemorySpace()) {
    auto number = dyn_cast<IntegerAttr>(space);
    if (!number || number.getInt() != 0) return fail(error, "input/output must reside in LP6 memory space");
  }
  SmallVector<int64_t> strides;
  int64_t offset = 0;
  if (failed(type.getStridesAndOffset(strides, offset)) || offset != 0 ||
      strides.size() != 2 || strides[0] != type.getDimSize(1) || strides[1] != 1)
    return fail(error, "non-dense argument view is unsupported");
  auto placement = in.argLP6[argument.getArgNumber()];
  auto layout = placement ? placement.getAs<StringAttr>("layout") : StringAttr{};
  auto address = placement ? placement.getAs<IntegerAttr>("addr") : IntegerAttr{};
  auto bytes = placement ? placement.getAs<IntegerAttr>("bytes") : IntegerAttr{};
  if (!layout || layout.getValue() != "dense" || !address || !bytes ||
      address.getInt() < 0 || bytes.getInt() <= 0)
    return fail(error, "argument lacks a valid dense LP6 addr/bytes placement");
  uint64_t rows = uint64_t(type.getDimSize(0)), columns = uint64_t(type.getDimSize(1));
  if (rows > 8 || columns >= ::npu::isa2::kGdmaStride ||
      columns > std::numeric_limits<uint64_t>::max() / rows / (bits / 8))
    return fail(error, "argument dimensions exceed the bounded dense-view contract");
  uint64_t required = rows * columns * (bits / 8), base = uint64_t(address.getInt());
  uint64_t limit = std::min<uint64_t>(uint64_t(in.spec->lp6.capacityBytes), ::npu::isa2::kGdmaLp6);
  if (uint64_t(bytes.getInt()) < required || base % 32 || base >= limit || required > limit - base)
    return fail(error, "dense LP6 view is too small, unaligned, or outside target/ISA capacity");
  view = {base, required, uint32_t(rows), uint32_t(columns), bits / 8};
  return success();
}

LogicalResult checkRanges(const std::vector<Range> &ranges, uint64_t capacity,
                          uint64_t &peak, std::string &error) {
  peak = 0;
  for (size_t i = 0; i < ranges.size(); ++i) {
    const auto &r = ranges[i];
    if (!r.bytes || r.offset > capacity || r.bytes > capacity - r.offset)
      return fail(error, "scratch range exceeds target capacity: " + r.name);
    peak = std::max(peak, r.offset + r.bytes);
    for (size_t j = i + 1; j < ranges.size(); ++j)
      if (overlap(r.offset, r.bytes, ranges[j].offset, ranges[j].bytes))
        return fail(error, "scratch ranges alias: " + r.name + "/" + ranges[j].name);
  }
  return success();
}

LogicalResult makePlan(const NpuPhaseInput &in, Plan &plan, std::string &error) {
  if (!in.phase || !in.program || !in.pim || !in.spec)
    return fail(error, "missing phase, builder, PIM storage program, or target");
  func::FuncOp phase = in.phase;
  auto kernel = in.phase->getAttrOfType<StringAttr>("hyb.npu_kernel");
  if (!kernel || kernel.getValue() != "pim_packet_g4")
    return fail(error, "phase is not explicitly selected for pim_packet_g4");
  if (!in.weightPrefetch) return fail(error, "the fixed packet schedule requires weight prefetch");
  if (phase.getBody().getBlocks().size() != 1)
    return fail(error, "only a single-block qmatmul phase is supported");
  hyb::QMatmulOp operation;
  for (Operation &op : phase.getBody().front()) {
    if (auto q = dyn_cast<hyb::QMatmulOp>(op)) {
      if (operation) return fail(error, "only one qmatmul per packet phase is supported");
      operation = q;
    } else if (!isa<func::ReturnOp>(op)) {
      return fail(error, "unsupported IR operation in packet phase: " + op.getName().getStringRef().str());
    }
  }
  if (!operation || operation->getNumResults())
    return fail(error, "expected exactly one bufferized qmatmul");
  auto index = operation->getAttrOfType<IntegerAttr>("hyb.packet_projection");
  if (!index || index.getInt() < 0 || uint64_t(index.getInt()) >= in.pim->projections.size())
    return fail(error, "missing or invalid hyb.packet_projection storage index");
  plan.projection = index.getInt();
  const auto &projection = in.pim->projections[size_t(plan.projection)];
  if (failed(denseArgument(in, operation.getInput(), 8, plan.x, error)) ||
      failed(denseArgument(in, operation.getInit(), 32, plan.y, error))) return failure();
  plan.m = plan.x.rows; plan.k = plan.x.columns; plan.n = plan.y.columns;
  if (plan.m != plan.y.rows || plan.m < 1 || plan.m > 8 || plan.k % kSeg || plan.n % 4096 ||
      projection.k != plan.k || projection.n != plan.n || projection.nPad != projection.n ||
      projection.name != operation.getWeight())
    return fail(error, "unsupported shape/storage identity; require M1..8, K%256=0, N%4096=0 and matching weight");
  if (in.spec->npu.cores < int(kCores) || in.spec->npu.saRows != 32 || in.spec->npu.saCols != 32 ||
      in.spec->pim.channels != 8 || in.pim->channels != 8 || in.pim->banksetPingPongLead != 0)
    return fail(error, "requires four NPU cores, 32x32 SA, eight subchannels, and even-v1 weight storage");
  if (in.spec->lp6.capacityBytes <= 0 || in.spec->npu.l1Bytes <= 0 || in.spec->npu.l2Bytes <= 0 ||
      in.spec->npu.completionEventSlots <= 0)
    return fail(error, "invalid target capacities");
  if (uint64_t(plan.n) * 4 >= ::npu::isa2::kGdmaRowBytes)
    return fail(error, "output row exceeds the GDMA record width");
  plan.block = std::min(kMaxBlock, plan.k); plan.slot = 128 * plan.block;
  plan.l1 = {{"X_segment", kX, uint64_t(plan.m) * kSeg}, {"A_operand", kA, 32 * 4 * kSeg},
      {"partial_C", kC, uint64_t(128) * plan.m * 4}, {"E_expansion", kE, uint64_t(4) * plan.m * kSeg},
      {"B_all_K", kB, uint64_t(16) * plan.m * plan.k}, {"canonical_Y", kY, plan.y.bytes}};
  plan.l2 = {{"X_segments", kX, plan.x.bytes}, {"raw_weight_slots", kRawL2, uint64_t(8) * plan.slot},
             {"canonical_Y", kYL2, plan.y.bytes}};
  if (failed(checkRanges(plan.l1, uint64_t(in.spec->npu.l1Bytes), plan.l1Peak, error)) ||
      failed(checkRanges(plan.l2, uint64_t(in.spec->npu.l2Bytes), plan.l2Peak, error))) return failure();
  // Each GDMA and core block consumes a completion-event slot. Bound the full
  // schedule before enumerating packets; extra joins cover dependency fan-in.
  uint64_t segments = plan.k / kSeg, stripes = plan.n / 128;
  uint64_t blocks = (plan.k + plan.block - 1) / plan.block;
  plan.eventsUpperBound = segments + 4 + stripes * (segments + blocks) + 5 +
                         4 * ((segments + 254) / 255) + 2;
  if (uint64_t(in.program->numCommands()) + plan.eventsUpperBound >
      std::min<uint64_t>(65535, uint64_t(in.spec->npu.completionEventSlots)))
    return fail(error, "packet plan exceeds the bounded completion-event budget");
  if (projection.streamPart.size() != size_t(projection.streams()) ||
      projection.streamWeightBase.size() != projection.streamPart.size())
    return fail(error, "incomplete storage projection stream metadata");
  std::map<int64_t, std::set<int>> sameBase;
  for (size_t i = 0; i < projection.streamPart.size(); ++i) {
    int part = projection.streamPart[i]; int64_t base = projection.streamWeightBase[i];
    if (part < 0 || part >= 8 || base < 0 || !sameBase[base].insert(part).second)
      return fail(error, "requires unique rank0 streams in each same-base SC group");
  }
  for (const auto &entry : sameBase)
    if (entry.second.size() != 8)
      return fail(error, "requires a complete same-rank/base group of eight subchannels");

  std::vector<::npu::pim::WeightPacketLocation> first;
  first.reserve(plan.n / 4);
  in.pim->forEachWeightPacketLocation(projection, [&](const auto &location) {
    if (!location.inputBegin) first.push_back(location);
  });
  if (first.size() != plan.n / 4) return fail(error, "wrong first-K packet coverage");
  std::sort(first.begin(), first.end(), [](const auto &a, const auto &b) { return a.address < b.address; });
  std::vector<int64_t> packetOfOutput(plan.n, -1);
  std::vector<bool> outputSeen(plan.n, false);
  for (size_t q = 0; q < first.size(); ++q) {
    for (int64_t n : first[q].outputChannels) {
      if (n < 0 || n >= plan.n || outputSeen[size_t(n)])
        return fail(error, "duplicate, padded, or out-of-range logical output");
      outputSeen[size_t(n)] = true;
    }
    packetOfOutput[size_t(first[q].outputChannels[0])] = int64_t(q);
  }
  plan.stripes.resize(size_t(stripes));
  for (size_t s = 0; s < plan.stripes.size(); ++s) {
    auto &stripe = plan.stripes[s]; stripe.segments.assign(size_t(segments), kNoAddress);
    uint64_t base = first[s * 32].address;
    if (base % 1024) return fail(error, "physical q32 stripe is not 1KiB aligned");
    for (size_t q = 0; q < 32; ++q) {
      const auto &location = first[s * 32 + q];
      if (location.address != base + q * 32)
        return fail(error, "physical q32 packet stripe is not contiguous");
      for (size_t lane = 0; lane < 4; ++lane) {
        uint32_t n = uint32_t(location.outputChannels[lane]);
        uint32_t owner = (n % 64) / 16;
        if (!q && !lane) stripe.core = owner;
        if (owner != stripe.core) return fail(error, "packet stripe violates disjoint output ownership");
        stripe.output[q * 4 + lane] = n;
      }
    }
    plan.byCore[stripe.core].push_back(s);
  }
  for (const auto &owned : plan.byCore)
    if (owned.size() * kCores != plan.stripes.size())
      return fail(error, "unbalanced or incomplete four-core output partition");

  // A bit per logical packet detects holes/duplicates without storing every
  // PA or copying the weight tensor into the compiler. Every actual PA is
  // checked against its segment's 32KiB contiguous-transfer description.
  plan.packetCount = uint64_t(plan.n) * plan.k / 32;
  std::vector<uint64_t> seen(size_t((plan.packetCount + 63) / 64), 0);
  uint64_t count = 0;
  in.pim->forEachWeightPacketLocation(projection, [&](const auto &location) {
    if (!error.empty()) return;
    if (location.inputBegin < 0 || location.inputBegin >= plan.k || location.inputBegin % 8 ||
        location.outputChannels[0] < 0 || location.outputChannels[0] >= plan.n) {
      (void)fail(error, "iterator returned an invalid logical packet"); return;
    }
    int64_t ordinal = packetOfOutput[size_t(location.outputChannels[0])];
    if (ordinal < 0 || first[size_t(ordinal)].outputChannels != location.outputChannels) {
      (void)fail(error, "output lanes change between K segments"); return;
    }
    if (location.address >= uint64_t(::npu::pim::kRankBytes) ||
        location.address + 32 > uint64_t(in.spec->lp6.capacityBytes) || location.address % 32 ||
        overlap(location.address, 32, plan.x.address, plan.x.bytes) ||
        overlap(location.address, 32, plan.y.address, plan.y.bytes)) {
      (void)fail(error, "weight PA is outside rank0 or aliases dense input/output"); return;
    }
    size_t stripeIndex = size_t(ordinal) / 32, q = size_t(ordinal) % 32;
    size_t segment = size_t(location.inputBegin) / kSeg, j = size_t(location.inputBegin % kSeg) / 8;
    uint64_t delta = j * 1024 + q * 32;
    if (location.address < delta) { (void)fail(error, "invalid packet segment base"); return; }
    uint64_t segmentBase = location.address - delta;
    auto &saved = plan.stripes[stripeIndex].segments[segment];
    if (saved == kNoAddress) saved = segmentBase;
    else if (saved != segmentBase) { (void)fail(error, "actual K256 packet PAs are not contiguous"); return; }
    uint64_t key = (uint64_t(stripeIndex) * segments + segment) * 1024 + j * 32 + q;
    uint64_t mask = uint64_t(1) << (key % 64);
    if (seen[size_t(key / 64)] & mask) { (void)fail(error, "duplicate logical K/output packet"); return; }
    seen[size_t(key / 64)] |= mask; ++count;
  });
  if (!error.empty()) return failure();
  if (count != plan.packetCount) return fail(error, "incomplete K/output packet coverage");
  for (const auto &stripe : plan.stripes)
    for (uint64_t address : stripe.segments)
      if (address == kNoAddress || address % 1024 || address + 32768 > uint64_t(::npu::pim::kRankBytes))
        return fail(error, "invalid whole-segment GDMA address");

  llvm::json::Array l1, l2;
  for (const auto &r : plan.l1)
    l1.push_back(llvm::json::Object{{"name", r.name}, {"offset", int64_t(r.offset)}, {"bytes", int64_t(r.bytes)}});
  for (const auto &r : plan.l2)
    l2.push_back(llvm::json::Object{{"name", r.name}, {"offset", int64_t(r.offset)}, {"bytes", int64_t(r.bytes)}});
  llvm::json::Object metadata{{"kernel", "pim_packet_g4"}, {"group_lanes", 4}, {"M", plan.m},
      {"N", plan.n}, {"K", plan.k}, {"Kseg", kSeg}, {"Kblock", plan.block}, {"cores", kCores},
      {"raw_slots_per_core", 2}, {"raw_slot_bytes", plan.slot}, {"l1_peak_bytes", int64_t(plan.l1Peak)},
      {"l2_peak_bytes", int64_t(plan.l2Peak)}, {"l1_ranges", std::move(l1)}, {"l2_ranges", std::move(l2)},
      {"input_lp6", int64_t(plan.x.address)}, {"output_lp6", int64_t(plan.y.address)},
      {"input_dtype", "i8"}, {"output_dtype", "i32"}, {"partial_dtype", "i32"},
      {"output_lp6_store_included", true}, {"packet_location_api", "Program::forEachWeightPacketLocation"},
      {"all_packet_contiguity_checked", true}, {"packet_count", int64_t(plan.packetCount)},
      {"projection_index", plan.projection}, {"same_base_8SC_groups", int64_t(sameBase.size())},
      {"B_generations", 4}, {"B_bytes_per_core", int64_t(16) * plan.m * plan.k},
      {"output_bytes_written_exactly_once", true}, {"exclusive_phase_scratch", true},
      {"raw_slot_reuse_waits_for_core_completion", true}, {"global_optimality_claimed", false}};
  llvm::raw_string_ostream stream(plan.json);
  stream << llvm::formatv("{0}", llvm::json::Value(std::move(metadata)));
  return success();
}

class Emitter {
public:
  Emitter(ProgramBuilder &builder, const Plan &plan) : w(builder), p(plan) {}

  void run() {
    const std::vector<int> entry = w.frontier;
    std::vector<int> inputs;
    for (uint32_t segment = 0; segment < p.k / kSeg; ++segment)
      inputs.push_back(w.appendGdma(ProgramBuilder::Kind::GdmaLoad, entry,
          p.x.address + segment * kSeg, kX + segment * p.m * kSeg,
          kSeg, p.m, p.k, kSeg));
    std::array<int, kCores> prepared;
    for (uint32_t core = 0; core < kCores; ++core) {
      begin(core, inputs);
      for (uint32_t segment = 0; segment < p.k / kSeg; ++segment) expand(segment);
      wait(4);
      prepared[core] = w.flushBlock(false);
    }
    std::vector<int> allPrepared(prepared.begin(), prepared.end());
    const uint32_t blocks = (p.k + p.block - 1) / p.block;
    const size_t jobs = p.byCore[0].size() * blocks;
    std::array<std::vector<int>, kCores> done;
    for (size_t local = 0; local < jobs; ++local) {
      for (uint32_t core = 0; core < kCores; ++core) {
        const auto &stripe = p.stripes[p.byCore[core][local / blocks]];
        uint32_t k0 = uint32_t(local % blocks) * p.block;
        uint32_t length = std::min(p.block, p.k - k0);
        uint32_t slot = kRawL2 + (core * 2 + uint32_t(local % 2)) * p.slot;
        std::vector<int> stageDeps = local < 2 ? allPrepared : std::vector<int>{done[core][local - 2]};
        std::vector<int> deps;
        for (uint32_t kk = 0; kk < length; kk += kSeg)
          deps.push_back(w.appendGdma(ProgramBuilder::Kind::GdmaLoad, stageDeps,
              stripe.segments[(k0 + kk) / kSeg], slot + kk * 128, 32768, 1, 32768, 32768));
        deps.push_back(local ? done[core].back() : prepared[core]);
        begin(core, deps);
        if (k0) {
          load(7, kC);
          w.appendRecord(matrixLoadAccumulator(7, 32, kG * p.m, kG * p.m * 4, true));
        }
        for (uint32_t kk = 0; kk < length; kk += kSeg) {
          config(7, 32); config(0, 32); config(1, 1024); config(2, 32);
          for (uint32_t q = 0; q < 32; ++q) {
            load(1, kA + q * kG * kSeg); load(2, slot + kk * 128 + q * 32);
            emit(rform(OP_LDMA, 1, 2, 0, 0, 6));
          }
          wait(1);
          load(5, kB + ((k0 + kk) / kSeg) * 16 * p.m * kSeg);
          w.appendRecord(matrixLoad(5, kG * kSeg, kG * p.m, kG * p.m, 1));
          load(6, kA);
          w.appendRecord(matrixLoad(6, 32, kG * kSeg, kG * kSeg, 5));
          for (uint32_t word : matrixMma(kG * kSeg, 1, k0 || kk)) emit(word);
        }
        load(7, kC);
        w.appendRecord(matrixWriteout(7, 32, kG * p.m, kG * p.m * 4, 2));
        if (k0 + length == p.k) {
          for (uint32_t q = 0; q < 32; ++q)
            for (uint32_t lane = 0; lane < 4; ++lane)
              copy(kC + (q * 4 * p.m + lane) * 4, kY + stripe.output[q * 4 + lane] * 4,
                   4, p.m, 16, p.n * 4);
        }
        wait(4);
        done[core].push_back(w.flushBlock(false));
      }
    }
    std::vector<int> stores;
    for (uint32_t core = 0; core < kCores; ++core) {
      begin(core, {done[core].back()});
      config(7, 64); config(0, p.n / 64 * p.m); config(1, 256); config(2, 256);
      load(1, kY + core * 64); load(2, kYL2 + core * 64);
      emit(rform(OP_LDMA, 1, 2, 0, 0, 5));
      wait(4);
      stores.push_back(w.flushBlock(false));
    }
    int output = w.appendGdma(ProgramBuilder::Kind::GdmaStore, stores, p.y.address, kYL2,
                             p.n * 4, p.m, p.n * 4, p.n * 4);
    w.frontier = {output};
    w.reserveL2(0, p.l2Peak);
  }

private:
  ProgramBuilder &w;
  const Plan &p;
  void emit(uint32_t word) { w.append(word); }
  void load(uint32_t reg, uint32_t value) {
    emit(OP_LUI | reg << 6 | (value >> 12) << 10);
    if (value & 0xfff) emit(OP_ADDI | reg << 6 | reg << 10 | (value & 0xfff) << 14);
  }
  void config(uint32_t field, uint32_t value) {
    load(15, value); emit(rform(OP_C_SET, 15, 0, 0, 0, field));
  }
  void wait(uint32_t builderKind) { emit(rform(OP_C_WAIT, 0, 0, 0, 0, builderKind)); }
  void begin(uint32_t core, const std::vector<int> &deps) {
    // Every previous block has already been flushed. Changing the frontier
    // before appending its first word gives this block exactly its own DAG.
    w.logicalCore = int(core); w.frontier = deps;
  }
  void copy(uint32_t source, uint32_t destination, uint32_t bytes, uint32_t rows,
            uint32_t sourceStride, uint32_t destinationStride) {
    config(7, bytes); config(0, rows); config(1, sourceStride); config(2, destinationStride);
    load(1, destination); load(2, source); emit(rform(OP_LDMA, 1, 2, 0, 0, 9));
  }
  void zero(uint32_t address, uint32_t bytes) {
    load(0, 0);
    w.appendRecord(typedVector(T_SPLAT, DT_I8, 15, 0, 0, 0, 64));
    config(3, 64);
    for (uint32_t offset = 0; offset < bytes; offset += 64) {
      load(1, address + offset); emit(rform(OP_V_MEM, 1, 15, 0, 0, 6));
    }
    wait(3);
  }
  void expand(uint32_t segment) {
    const uint32_t cols = 4 * p.m, b = kB + segment * 16 * p.m * kSeg;
    config(7, p.m * kSeg); load(1, kX); load(2, kX + segment * p.m * kSeg);
    emit(rform(OP_LDMA, 1, 2, 0, 0, 0));
    zero(b, 16 * p.m * kSeg); zero(kE, kSeg * cols);
    for (uint32_t row = 0; row < p.m; ++row)
      copy(kX + row * kSeg, kE + 4 * row, 1, kSeg, 1, cols);
    for (uint32_t lane = 0; lane < 4; ++lane)
      copy(kE, b + 8 * lane * cols + lane, 8 * cols - lane, kSeg / 8, 8 * cols, 32 * cols);
  }
};
} // namespace

LogicalResult mlir::hybrid::emitPimPacketNpuPhase(const NpuPhaseInput &in, NpuPhaseStats &stats,
                                                std::string &error) {
  error.clear();
  Plan plan;
  if (failed(makePlan(in, plan, error))) return failure();
  auto &builder = *in.program;
  const int64_t words = builder.coreWords();
  const int commands = builder.numCommands();
  Emitter(builder, plan).run();
  in.phase->setAttr("hyb.packet_plan", StringAttr::get(in.phase->getContext(), plan.json));
  stats.coreWords = builder.coreWords() - words;
  stats.commands = builder.numCommands() - commands;
  stats.l1Peak = int64_t(plan.l1Peak); stats.cores = kCores;
  stats.l2StagingPerCore = int64_t(2) * plan.slot;
  stats.weightStreams.push_back({int64_t(plan.slot), int64_t(plan.stripes.size() / 4) *
      ((plan.k + plan.block - 1) / plan.block), 2, 1, true, true,
      "pim_packet_g4: checked fixed-PA Kseg256, bounded Kblock, two raw L2 slots/core; native B and INT32 spill/reload"});
  return success();
}
