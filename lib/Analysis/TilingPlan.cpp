#include "plena/Analysis/TilingPlan.h"
#include <algorithm>
#include <set>
#include <sstream>

namespace plena {
namespace {
uint64_t ceilDiv(uint64_t n, uint64_t d) { return n/d + (n%d != 0); }
uint64_t arrayDivisor(uint64_t extent) {
  uint64_t tile = std::min<uint64_t>(32,extent);
  while (extent % tile) --tile;
  return tile;
}
std::vector<uint64_t> candidates(uint64_t extent, uint64_t base) {
  std::set<uint64_t> out{std::min(base,extent),extent,arrayDivisor(extent)};
  for (uint64_t n = base; n < extent; n *= 2) out.insert(n);
  return {out.begin(),out.end()};
}
BufferRequirement buffer(const std::string &name, MemorySpace space, uint64_t bytes) {
  BufferRequirement r; r.name = name; r.space = space; r.bytes = bytes; r.end = 1; return r;
}
}
Result<std::vector<TilingPlan>> enumerateTilingPlans(const TilingRequest &request) {
  using R = Result<std::vector<TilingPlan>>;
  const auto &s = request.shape; const auto &h = request.hardware;
  // Keeps all byte/traffic/cycle arithmetic below uint64_t; reject rather than wrap.
  constexpr uint64_t maxElements = uint64_t(1) << 40;
  if (!s.m || !s.n || !s.k || s.m > maxElements || s.n > maxElements / s.m ||
      s.k > maxElements / (s.m*s.n)) return R::failure("invalid or oversized matmul shape");
  if (h.arrayRows != 32 || h.arrayColumns != 32 || !h.cores || !h.dramBytesPerCycle ||
      !h.nocBytesPerCycle || !h.matrixMACsPerCycle || h.matrixLatency > (1ull<<20))
    return R::failure("invalid hardware cost model");
  if (request.graphResidentBytes >= h.l2Bytes) return R::failure("graph residency leaves no L2 staging capacity");
  MemoryLimits limits{h.l1Bytes,h.l2Bytes,h.dramBytes,1,h.dramBytesPerCycle,0};
  std::vector<TilingPlan> plans;
  auto ms = candidates(s.m,32), ns = candidates(s.n,32), ks = candidates(s.k,32);
  for (auto m : ms) for (auto n : ns) {
    uint64_t lane = n;
    if (h.cores > 1) {
      auto want = std::max<uint64_t>(32,n/h.cores);
      want -= want % 32;
      for (; want > 32 && n % want; want -= 32) {}
      if (n % want == 0) lane = want;
    }
    for (auto lm : candidates(m,32)) for (auto ln : candidates(lane,32)) for (auto k : ks) {
      // A single hardware accumulator cannot hold several partial output tiles.
      // Larger L1 panels therefore retain full K; chunked K uses one array tile.
      if (m % lm || lane % ln || (k != s.k && (lm > 32 || ln > 32))) continue;
      // With a single row tile, activation hoisting already reuses A across the
      // entire core lane. A wider panel only delays the first MMA in that case.
      if (lm <= 32 && ln > 32) continue;
      const uint64_t innerPanels = ceilDiv(s.n,n) > 1 ? ceilDiv(s.n,n) : ceilDiv(s.m,m);
      const uint64_t l1Stages = std::min<uint64_t>(innerPanels,std::max(1u,request.stages));
      if (2*l1Stages*(lm*k+k*ln+std::min(lm,32ul)*std::min(ln,32ul)) > h.l1Bytes) continue;
      TilingPlan p;
      p.schedule = k == s.k ? Schedule::ResidentPanels : Schedule::Hybrid;
      p.numerical = request.numerical;
      p.l2 = {m,n,s.k}; p.l1 = {lm,ln,k};
      p.array = {arrayDivisor(lm),arrayDivisor(ln),k};
      p.coreSplit = std::min<uint64_t>(h.cores,n/lane);
      if (request.graphResidentBytes)
        p.memory.push_back(buffer("graph.resident",MemorySpace::L2,request.graphResidentBytes));
      // L2 stages and the L1 allocations inside the unrolled panel loop are
      // simultaneously live. Reserve both; the accumulator is dedicated state.
      p.stages = std::max(1u,request.stages);
      const uint64_t stageBytes = uint64_t(p.stages) * 2;
      if (m*s.k > UINT64_MAX/stageBytes || s.k*n > UINT64_MAX/stageBytes ||
          m*n > UINT64_MAX/stageBytes) continue;
      p.memory.push_back(buffer("l2.activation",MemorySpace::L2,stageBytes*m*s.k));
      p.memory.push_back(buffer("l2.weight",MemorySpace::L2,stageBytes*s.k*n));
      p.memory.push_back(buffer("l2.output",MemorySpace::L2,stageBytes*m*n));
      p.memory.push_back(buffer("l1.activation",MemorySpace::L1,2*l1Stages*p.l1.m*k));
      p.memory.push_back(buffer("l1.weight",MemorySpace::L1,2*l1Stages*k*p.l1.n));
      p.memory.push_back(buffer("l1.output",MemorySpace::L1,2*l1Stages*p.array.m*p.array.n));
      auto memory = planMemory(p.memory,limits);
      if (!memory) continue;
      p.placement = std::move(memory.value);
      // Each L2 MxN panel is fetched once, and reused over its array-sized tiles.
      p.dramTrafficBytes = 2*(s.m*s.k*ceilDiv(s.n,n) + s.k*s.n*ceilDiv(s.m,m) + s.m*s.n);
      const uint64_t activationN = lm <= 32 && k == s.k ? lane : ln;
      p.localTrafficBytes = 2*(s.m*s.k*ceilDiv(s.n,activationN) +
                              s.k*s.n*ceilDiv(s.m,lm) + s.m*s.n);
      auto tiles = ceilDiv(s.m,p.array.m)*ceilDiv(s.n,p.array.n);
      auto kChunks = ceilDiv(s.k,k);
      // Array work includes startup for each K chunk.
      auto compute = tiles*(ceilDiv(s.k,h.matrixMACsPerCycle) + kChunks*(61+h.matrixLatency));
      auto writeout = ceilDiv(s.m*s.n,32) + tiles;
      // Core work is parallel, but all cores share DRAM and NoC bandwidth.
      // L2 stages permit overlap only between global transfer and core work;
      // retain the serial local-transfer cost for the resident-panel path.
      auto coreCycles = ceilDiv(compute + writeout,p.coreSplit) +
                        ceilDiv(p.localTrafficBytes,h.nocBytesPerCycle);
      auto globalCycles = ceilDiv(p.dramTrafficBytes,h.dramBytesPerCycle);
      auto panels = ceilDiv(s.m,m)*ceilDiv(s.n,n);
      p.estimatedCycles = p.stages > 1 && panels > 1
          ? std::max(coreCycles,globalCycles) + ceilDiv(std::min(coreCycles,globalCycles),panels)
          : coreCycles + globalCycles;
      plans.push_back(std::move(p));
    }
  }
  if (request.searchStages && request.stages > 1) {
    auto single = request; single.stages = 1; single.searchStages = false;
    auto alternatives = enumerateTilingPlans(single);
    if (alternatives)
      for (auto &p : alternatives.value) plans.push_back(std::move(p));
  }
  std::stable_sort(plans.begin(),plans.end(),[](const auto &a,const auto &b) {
    if (a.estimatedCycles != b.estimatedCycles) return a.estimatedCycles < b.estimatedCycles;
    return a.placement.l2HighWater < b.placement.l2HighWater;
  });
  if (plans.empty() && request.stages > 1) {
    auto single = request; single.stages = 1;
    return enumerateTilingPlans(single);
  }
  if (plans.empty()) return R::failure("no full-K L2 plan fits; cross-GDMA partial sums require a supported numerical policy and lowering");
  return R::success(std::move(plans));
}
Result<TilingPlan> chooseTilingPlan(const TilingRequest &request) {
  auto plans = enumerateTilingPlans(request);
  if (!plans) return Result<TilingPlan>::failure(plans.error);
  return Result<TilingPlan>::success(std::move(plans.value.front()));
}
std::string describe(const TilingPlan &p) {
  std::ostringstream out;
  out << (p.schedule == Schedule::ResidentPanels ? "ResidentPanels" : "Hybrid")
      << " L2=" << p.l2.m << 'x' << p.l2.n << 'x' << p.l2.k
      << " L1=" << p.l1.m << 'x' << p.l1.n << 'x' << p.l1.k
      << " cores=" << p.coreSplit << " stages=" << p.stages
      << " local_bytes=" << p.localTrafficBytes
      << " cycles=" << p.estimatedCycles << " DRAM_bytes=" << p.dramTrafficBytes
      << " L2_high_water=" << p.placement.l2HighWater;
  return out.str();
}
} // namespace plena
