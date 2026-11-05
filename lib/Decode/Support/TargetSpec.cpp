#include "npu/Decode/Support/TargetSpec.h"
#include "npu/Decode/Support/PimCost.h"
#include "npu/Target/PimLayout.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include <algorithm>
#include <cmath>

using namespace mlir;
using namespace mlir::hybrid;

FailureOr<TargetSpec> TargetSpec::load(llvm::StringRef path, std::string &error) {
  TargetSpec s;
  if (path.empty()) return s;
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    error = ("cannot read target spec " + path).str();
    return failure();
  }
  auto parsed = llvm::json::parse((*buffer)->getBuffer());
  if (!parsed) {
    error = llvm::toString(parsed.takeError());
    return failure();
  }
  llvm::json::Object *root = parsed->getAsObject();
  if (!root) {
    error = "target spec must be a JSON object";
    return failure();
  }
  auto num = [](llvm::json::Object *o, llvm::StringRef k, auto &dst) {
    if (!o) return;
    if (auto v = o->getNumber(k)) dst = static_cast<std::remove_reference_t<decltype(dst)>>(*v);
  };
  llvm::json::Object *n = root->getObject("npu"), *p = root->getObject("pim"), *l = root->getObject("lp6");
  // The target describes the hardware, in full: there are no hidden defaults. Every optimization is
  // applied by the compiler itself when the shapes and these numbers make it pay, so a key that is not
  // a target parameter (for example a removed on/off switch) is an error, and so is a missing one.
  static constexpr llvm::StringLiteral kNpuKeys[] = {
      "clock_mhz", "cores", "sa_rows", "sa_cols", "l1_bytes", "l2_bytes", "gemv_fixed_us",
      "l1_read_bytes_per_cycle", "l1_write_bytes_per_cycle", "mma_k_per_cycle", "mma_pipeline_cycles",
      "vector_lanes", "vector_exp_per_cycle", "vector_rcp_per_cycle", "vector_sqrt_per_cycle", "gdma_read_gbps", "ldma_gbps_per_core", "command_fetch_words_per_cycle",
      "core_split_us", "core_split_no_stream_us", "l1_reserved_bytes", "l2_staging_bytes_per_core", "weight_stream_max_slots",
      "gemv_tile_bytes", "completion_event_slots", "staging_misc_bytes", "gdma_lookahead", "input_tile_columns"};
  static constexpr llvm::StringLiteral kPimKeys[] = {
      "clock_mhz", "pus", "lanes", "accumulators", "k_per_sweep", "channels", "ranks", "partitions",
      "concurrent_ranks", "nlld_ck", "bankset_pingpong_lead", "ldmac_ck", "sweep_ck", "stream_ck", "two_rank_factor",
      "readback_gbps", "act_write_gbps", "readback_ns_per_output", "npu_run_split_us", "pim_layer_exposed_us", "record_fetch_us", "max_records", "stream_accs",
      "max_split_k"};
  static constexpr llvm::StringLiteral kLp6Keys[] = {"capacity_bytes"};
  auto unknownKey = [&](llvm::json::Object *o, llvm::StringRef name, llvm::ArrayRef<llvm::StringLiteral> keys) {
    if (o)
      for (auto &kv : *o) {
        llvm::StringRef key = kv.first;
        if (!key.starts_with("_") && !llvm::is_contained(keys, key)) {
          error = (name + "." + key + " is not a target parameter (optimizations have no switches)").str();
          return true;
        }
      }
    return false;
  };
  auto missingKey = [&](llvm::json::Object *o, llvm::StringRef name, llvm::ArrayRef<llvm::StringLiteral> keys) {
    for (llvm::StringRef key : keys)
      if (!o || !o->get(key)) {
        error = ("target spec is missing " + name + "." + key).str();
        return true;
      }
    return false;
  };
  if (unknownKey(n, "npu", kNpuKeys) || unknownKey(p, "pim", kPimKeys) || unknownKey(l, "lp6", kLp6Keys) ||
      missingKey(n, "npu", kNpuKeys) || missingKey(p, "pim", kPimKeys) || missingKey(l, "lp6", kLp6Keys))
    return failure();
  num(n, "clock_mhz", s.npu.clockMHz); num(n, "cores", s.npu.cores); num(n, "sa_rows", s.npu.saRows);
  num(n, "sa_cols", s.npu.saCols); num(n, "l1_bytes", s.npu.l1Bytes); num(n, "l2_bytes", s.npu.l2Bytes);
  num(n, "gemv_fixed_us", s.npu.gemvFixedUs);
  num(n, "l1_read_bytes_per_cycle", s.npu.l1ReadBytesPerCycle);
  num(n, "l1_write_bytes_per_cycle", s.npu.l1WriteBytesPerCycle);
  num(n, "mma_k_per_cycle", s.npu.mmaKPerCycle); num(n, "mma_pipeline_cycles", s.npu.mmaPipelineCycles);
  num(n, "vector_lanes", s.npu.vectorLanes); num(n, "vector_exp_per_cycle", s.npu.vectorExpPerCycle);
  num(n, "vector_rcp_per_cycle", s.npu.vectorRcpPerCycle); num(n, "vector_sqrt_per_cycle", s.npu.vectorSqrtPerCycle);
  num(n, "gdma_read_gbps", s.npu.gdmaReadGBps); num(n, "core_split_us", s.npu.coreSplitUs); num(n, "core_split_no_stream_us", s.npu.coreSplitNoStreamUs);
  num(n, "ldma_gbps_per_core", s.npu.ldmaGBpsPerCore);
  num(n, "command_fetch_words_per_cycle", s.npu.commandFetchWordsPerCycle);
  if (s.npu.l1ReadBytesPerCycle <= 0 || s.npu.l1WriteBytesPerCycle <= 0 || s.npu.mmaKPerCycle <= 0 ||
      s.npu.mmaPipelineCycles < 0 || s.npu.vectorLanes <= 0 || s.npu.vectorExpPerCycle <= 0 ||
      s.npu.vectorRcpPerCycle <= 0 || s.npu.vectorSqrtPerCycle <= 0 || !(s.npu.gdmaReadGBps > 0) || !(s.npu.ldmaGBpsPerCore > 0) || s.npu.commandFetchWordsPerCycle < 1 ||
      !(s.npu.coreSplitUs >= 0) || !(s.npu.coreSplitNoStreamUs >= 0)) {
    error = "npu timing values must be positive (core_split costs non-negative)";
    return failure();
  }
  num(p, "clock_mhz", s.pim.clockMHz); num(p, "pus", s.pim.pus); num(p, "lanes", s.pim.lanes);
  num(p, "accumulators", s.pim.accumulators); num(p, "k_per_sweep", s.pim.kPerSweep);
  num(p, "channels", s.pim.channels); num(p, "ranks", s.pim.ranks); num(p, "partitions", s.pim.partitions);
  num(p, "nlld_ck", s.pim.nlldCK);
  num(p, "ldmac_ck", s.pim.ldmacCK); num(p, "sweep_ck", s.pim.sweepCK); num(p, "stream_ck", s.pim.streamCK);
  num(p, "readback_gbps", s.pim.readbackGBps); num(p, "act_write_gbps", s.pim.actWriteGBps);
  num(p, "readback_ns_per_output", s.pim.readbackNsPerOutput);
  num(p, "npu_run_split_us", s.pim.npuRunSplitUs);
  num(p, "pim_layer_exposed_us", s.pim.layerExposedUs);
  num(p, "two_rank_factor", s.pim.twoRankFactor);
  num(p, "record_fetch_us", s.pim.recordFetchUs); num(p, "max_records", s.pim.maxRecords);
  num(p, "max_split_k", s.pim.maxSplitK); num(p, "bankset_pingpong_lead", s.pim.banksetPingPongLead);
  if (p) {
    if (auto v = p->getBoolean("concurrent_ranks")) s.pim.concurrentRanks = *v;
    if (auto *accs = p->getArray("stream_accs")) {
      s.pim.streamAccs.clear();
      for (const llvm::json::Value &v : *accs)
        if (auto a = v.getAsInteger()) s.pim.streamAccs.push_back(int(*a));
    }
  }
  if (s.pim.ldmacCK < s.pim.nlldCK || s.pim.sweepCK < 0 || s.pim.streamCK < 0 || !(s.pim.readbackGBps > 0) ||
      !(s.pim.actWriteGBps > 0) || s.pim.readbackNsPerOutput < 0 || s.pim.npuRunSplitUs < 0 ||
      s.pim.layerExposedUs < 0 || !(s.pim.twoRankFactor >= 1) || !(s.pim.recordFetchUs > 0) ||
      s.pim.maxRecords < 1 || s.pim.maxRecords > 64 ||
      s.pim.maxSplitK < 1 || s.pim.streamAccs.empty() || s.pim.banksetPingPongLead < 0 ||
      s.pim.banksetPingPongLead > 63) {
    error = "pim stream costs: ldmac_ck >= nlld_ck, the others nonnegative (rates positive), max_split_k >= 1, "
            "stream_accs nonempty, bankset_pingpong_lead 0..63";
    return failure();
  }
  // The PIM program layout (npu/Target/PimLayout.h) is written for the geometry of the LPDDR6-PIM draft.
  if (s.pim.pus != npu::pim::kPUs || s.pim.lanes != npu::pim::kLanes || s.pim.accumulators != npu::pim::kAccs ||
      s.pim.kPerSweep != npu::pim::kKPerSweep || s.pim.ranks != npu::pim::kRanks || s.pim.partitions < 1 ||
      s.pim.partitions > s.pim.channels * s.pim.ranks) {
    error = "pim geometry: this compiler supports the LPDDR6-PIM draft (pus 8, lanes 4, accumulators 16, "
            "k_per_sweep 256, ranks 2) and partitions 1 .. channels x ranks";
    return failure();
  }
  for (int a : s.pim.streamAccs)
    if (a < 2 || a > 16 || a % 2) {
      error = "pim.stream_accs entries must be even accumulator counts 2..16";
      return failure();
    }
  num(l, "capacity_bytes", s.lp6.capacityBytes);
  num(n, "l1_reserved_bytes", s.npu.l1ReservedBytes); num(n, "l2_staging_bytes_per_core", s.npu.l2StagingPerCore);
  num(n, "weight_stream_max_slots", s.npu.weightStreamMaxSlots);
  num(n, "gemv_tile_bytes", s.npu.gemvTileBytes); num(n, "completion_event_slots", s.npu.completionEventSlots);
  num(n, "staging_misc_bytes", s.npu.stagingMiscBytes);
  if (n && n->get("gdma_lookahead")) {
    auto value = n->getInteger("gdma_lookahead");
    if (!value || *value < 0 || *value > 1024) {
      error = "npu.gdma_lookahead must be an integer in [0, 1024]";
      return failure();
    }
    s.npu.gdmaLookahead = *value;
  }
  auto positiveIntegerOption = [&](llvm::StringRef key, int64_t &dst) {
    if (!n || !n->get(key)) return true;
    auto value = n->getInteger(key);
    if (!value || *value <= 0 || *value > INT32_MAX) {
      error = ("npu." + key + " must be a positive 32-bit integer").str();
      return false;
    }
    dst = *value;
    return true;
  };
  if (!positiveIntegerOption("input_tile_columns", s.npu.inputTileColumns)) return failure();
  if (s.npu.stagingMiscBytes < 0 || s.npu.stagingMiscBytes % 64) {
    error = "npu.staging_misc_bytes must be a nonnegative multiple of 64";
    return failure();
  }
  if (llvm::json::Object *m = root->getObject("mapping")) {
    if (auto p = m->getString("policy")) s.mapping.policy = p->str();
    if (llvm::json::Object *o = m->getObject("overrides"))
      for (auto &[k, v] : *o)
        if (auto t = v.getAsString()) s.mapping.overrides.push_back({k.str(), t->str()});
  }
  if (s.mapping.policy != "cost" && s.mapping.policy != "npu" && s.mapping.policy != "pim") {
    error = "mapping.policy must be cost, npu or pim";
    return failure();
  }
  if (s.npu.cores <= 0 || s.npu.l2Bytes < 64 || s.npu.l2StagingPerCore < 0 ||
      s.npu.l2StagingPerCore % 64 || s.npu.l2StagingPerCore > s.npu.l2Bytes / s.npu.cores ||
      s.npu.weightStreamMaxSlots < 0) {
    error = "invalid NPU staging policy: positive cores/L2, staging=0 (auto) or a fitting 64-byte-aligned window, "
            "and a nonnegative ring cap required";
    return failure();
  }
  return s;
}

GemvCost mlir::hybrid::estimateGemvGroup(const TargetSpec &spec, int64_t tokens, llvm::ArrayRef<GemvShape> group) {
  GemvCost c{};
  if (group.empty()) return c;
  // PIM (W8A8): the fastest stream layout of the group (PimCost.h: accumulators per stream and
  // split-K per projection, streams balanced over one phase's partitions, every token repeating
  // every stream). Boundary work done by the NPU: the INT8 activation written as whole columns to
  // every PU of the ranks that read it (K x PUs x subchannels x ranks bytes per token), the INT32 outputs gathered and the
  // split-K partial sums added.
  std::vector<int64_t> rows;
  for (const GemvShape &g : group) rows.push_back(g.n);
  pimcost::Estimate e;
  pimcost::choose(spec, tokens, group.front().k, rows, pimcost::phasePartitions(spec), true, e);
  c.pimUs = e.pimUs;
  double outputs = 0;
  for (const GemvShape &g : group) outputs += double(g.n);
  c.pimBoundaryUs = e.scatterUs + e.gatherUs + e.sumUs + double(tokens) * outputs * spec.pim.readbackNsPerOutput / 1e3;
  // NPU: weights and row scales streamed from LP6 once for all tokens at the sustained GDMA read rate,
  // or the systolic array time when that is longer.
  double bytes = 0, macs = 0;
  for (const GemvShape &g : group) {
    bytes += double(g.n) * g.k + g.n * 4.0;
    macs += double(g.n) * g.k * tokens;
  }
  double streamUs = bytes / (spec.npu.gdmaReadGBps * 1e3);
  double mmaUs = macs / (double(spec.npu.cores) * spec.npu.saRows * spec.npu.saCols * 2 * spec.npu.clockMHz);
  c.npuUs = std::max(streamUs, mmaUs) + spec.npu.gemvFixedUs * group.size();
  return c;
}
