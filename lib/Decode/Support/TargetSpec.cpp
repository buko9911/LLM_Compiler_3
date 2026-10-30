#include "npu/Decode/Support/TargetSpec.h"
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
  num(n, "clock_mhz", s.npu.clockMHz); num(n, "cores", s.npu.cores); num(n, "sa_rows", s.npu.saRows);
  num(n, "sa_cols", s.npu.saCols); num(n, "l1_bytes", s.npu.l1Bytes); num(n, "l2_bytes", s.npu.l2Bytes);
  num(n, "lp6_read_gbps", s.npu.lp6ReadGBps); num(n, "gemv_fixed_us", s.npu.gemvFixedUs);
  num(p, "clock_mhz", s.pim.clockMHz); num(p, "pus", s.pim.pus); num(p, "lanes", s.pim.lanes);
  num(p, "channels", s.pim.channels); num(p, "ranks", s.pim.ranks); num(p, "partitions", s.pim.partitions);
  num(p, "nlld_ck", s.pim.nlldCK); num(p, "efficiency", s.pim.efficiency); num(p, "phase_fixed_us", s.pim.phaseFixedUs);
  num(p, "act_write_us_per_kib", s.pim.actWriteUsPerKiB); num(p, "act_write_fixed_us", s.pim.actWriteFixedUs);
  num(p, "readback_us_per_kib", s.pim.readbackUsPerKiB); num(p, "readback_fixed_us", s.pim.readbackFixedUs);
  if (p)
    if (auto v = p->getBoolean("concurrent_ranks")) s.pim.concurrentRanks = *v;
  num(l, "capacity_bytes", s.lp6.capacityBytes); num(l, "row_bytes", s.lp6.rowBytes);
  num(n, "l1_reserved_bytes", s.npu.l1ReservedBytes); num(n, "l2_staging_bytes_per_core", s.npu.l2StagingPerCore);
  num(n, "weight_stream_max_slots", s.npu.weightStreamMaxSlots);
  num(n, "weight_prefetch_batch_tiles", s.npu.weightPrefetchBatchTiles);
  if (n)
    if (auto v = n->getBoolean("weight_prefetch_cross_batch")) s.npu.weightPrefetchCrossBatch = *v;
  num(n, "gemv_tile_bytes", s.npu.gemvTileBytes); num(n, "completion_event_slots", s.npu.completionEventSlots); num(n, "distribute_min_elements", s.npu.distributeMinElements);
  num(n, "dequant_min_elements", s.npu.dequantMinElements);
  num(n, "staging_misc_bytes", s.npu.stagingMiscBytes);
  auto booleanOption = [&](llvm::StringRef key, bool &dst) {
    if (!n || !n->get(key)) return true;
    auto value = n->getBoolean(key);
    if (!value) {
      error = ("npu." + key + " must be a boolean").str();
      return false;
    }
    dst = *value;
    return true;
  };
  if (!booleanOption("activation_residency", s.npu.activationResidency) ||
      !booleanOption("attention_fusion", s.npu.attentionFusion) ||
      !booleanOption("gemv_epilogue_fusion", s.npu.gemvEpilogueFusion) ||
      !booleanOption("projection_fusion", s.npu.projectionFusion) ||
      !booleanOption("weight_stream_early_release", s.npu.weightStreamEarlyRelease) ||
      !booleanOption("matrix_init_elision", s.npu.matrixInitElision) ||
      !booleanOption("async_output_stores", s.npu.asyncOutputStores) ||
      !booleanOption("async_input_loads", s.npu.asyncInputLoads) ||
      !booleanOption("staging_early_release", s.npu.stagingEarlyRelease) ||
      !booleanOption("memory_dependency_schedule", s.npu.memoryDependencySchedule) ||
      !booleanOption("command_schedule", s.npu.commandSchedule) ||
      !booleanOption("gdma_lookahead_streams_only", s.npu.gdmaLookaheadStreamsOnly) ||
      !booleanOption("cross_phase_prefetch", s.npu.crossPhasePrefetch) ||
      !booleanOption("staging_misc_fit_rings", s.npu.stagingMiscFitRings) ||
      !booleanOption("elementwise_distribution", s.npu.elementwiseDistribution) ||
      !booleanOption("row_distribution", s.npu.rowDistribution) ||
      !booleanOption("vector_scalar_reuse", s.npu.vectorScalarReuse) ||
      !booleanOption("vector_chunk_pipeline", s.npu.vectorChunkPipeline) ||
      !booleanOption("quant_fused_max", s.npu.quantFusedMax) ||
      !booleanOption("vector_loops", s.npu.vectorLoops) ||
      !booleanOption("input_tile_pipeline", s.npu.inputTilePipeline) ||
      !booleanOption("gemv_epilogue_pipeline", s.npu.gemvEpiloguePipeline) ||
      !booleanOption("matmul_rhs_packing", s.npu.matmulRhsPacking) ||
      !booleanOption("matmul_transpose_output", s.npu.matmulTransposeOutput) ||
      !booleanOption("async_output_store_vector_overlap", s.npu.asyncOutputStoreVectorOverlap) ||
      !booleanOption("weight_stream_rotate_slots", s.npu.weightStreamRotateSlots) ||
      !booleanOption("flash_attention", s.npu.flashAttention) ||
      !booleanOption("flash_decoding", s.npu.flashDecoding) ||
      !booleanOption("flash_kv_slicing", s.npu.flashKvSlicing))
    return failure();
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
  if (!positiveIntegerOption("input_tile_columns", s.npu.inputTileColumns) ||
      !positiveIntegerOption("attention_query_tile", s.npu.attentionQueryTile) ||
      !positiveIntegerOption("attention_key_tile", s.npu.attentionKeyTile) ||
      !positiveIntegerOption("attention_splits", s.npu.attentionSplits))
    return failure();
  if (s.npu.flashDecoding && (!s.npu.flashAttention || s.npu.attentionSplits < 2 ||
                            s.npu.attentionSplits > s.npu.cores)) {
    error = "npu.flash_decoding requires flash_attention and attention_splits in [2, cores]";
    return failure();
  }
  if (!s.npu.flashDecoding && s.npu.attentionSplits != 1) {
    error = "npu.attention_splits must be 1 when flash_decoding is disabled";
    return failure();
  }
  if (s.npu.flashKvSlicing && !s.npu.flashAttention) {
    error = "npu.flash_kv_slicing requires flash_attention";
    return failure();
  }
  if (s.npu.stagingMiscBytes < 0 || s.npu.stagingMiscBytes % 64) {
    error = "npu.staging_misc_bytes must be a nonnegative multiple of 64";
    return failure();
  }
  if (s.npu.dequantMinElements < 0) {
    error = "npu.dequant_min_elements must be nonnegative (0 inherits distribute_min_elements)";
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
      s.npu.weightStreamMaxSlots < 0 || s.npu.weightPrefetchBatchTiles < 0) {
    error = "invalid NPU staging policy: positive cores/L2, staging=0 (auto) or a fitting 64-byte-aligned window, "
            "nonnegative ring cap and batch tiles (0 selects auto) required";
    return failure();
  }
  return s;
}

GemvCost mlir::hybrid::estimateGemvGroup(const TargetSpec &spec, int64_t tokens, llvm::ArrayRef<GemvShape> group) {
  GemvCost c{};
  // PIM (W8A8): a stream is 512 output rows (8 PUs x 16 accumulators x 4 lanes); a sweep of a
  // stream is 32 LdMACs x 16 outputs over 256 inputs. Streams of the group are balanced over
  // the partitions; every token repeats every stream.
  const int64_t rowsPerStream = spec.pim.pus * 16 * spec.pim.lanes;
  std::vector<int64_t> load(spec.pim.parallelPartitions(), 0);
  int64_t actBytes = 0, outBytes = 0;
  for (const GemvShape &g : group) {
    int64_t streams = (g.n + rowsPerStream - 1) / rowsPerStream, sweeps = g.k / 256;
    for (int64_t s = 0; s < streams; ++s) *std::min_element(load.begin(), load.end()) += sweeps;
    actBytes = std::max(actBytes, g.k);
    outBytes += g.n * 4;
  }
  double ldmacs = double(*std::max_element(load.begin(), load.end())) * 16 * 32 * tokens;
  c.pimUs = ldmacs * spec.pim.nlldCK / spec.pim.clockMHz / spec.pim.efficiency + spec.pim.phaseFixedUs;
  // Boundary work done by the NPU: write the INT8 activation into the PIM layout of every
  // partition, read the INT32 outputs back.
  c.pimBoundaryUs = tokens * (spec.pim.actWriteFixedUs + spec.pim.actWriteUsPerKiB * actBytes / 1024.0 +
                              spec.pim.readbackFixedUs + spec.pim.readbackUsPerKiB * outBytes / 1024.0);
  // NPU: weights streamed from LP6 once for all tokens (memory bound at M <= 32).
  double bytes = 0, macs = 0;
  for (const GemvShape &g : group) {
    bytes += double(g.n) * g.k + g.n * 4.0;
    macs += double(g.n) * g.k * tokens;
  }
  double streamUs = bytes / (spec.npu.lp6ReadGBps * 1e3);
  double mmaUs = macs / (double(spec.npu.cores) * spec.npu.saRows * spec.npu.saCols * 2 * spec.npu.clockMHz);
  c.npuUs = std::max(streamUs, mmaUs) + spec.npu.gemvFixedUs * group.size();
  return c;
}
