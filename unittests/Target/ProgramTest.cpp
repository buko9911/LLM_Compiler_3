#include "plena/Target/ISA.h"
#include "plena/Target/Program.h"
#include <gtest/gtest.h>
#include <random>
#include <set>

namespace {
plena::Command core(std::vector<plena::Access> accesses = {}) {
  plena::Command c; c.words = plena::assemble("C_FENCE_ALL").value; c.accesses = std::move(accesses); return c;
}
plena::Command load(unsigned offset) {
  plena::Command c; c.kind = plena::Command::Kind::Load; c.dramOffset = offset; c.l2Offset = offset; c.bytes = 64; return c;
}
// A hardware loop body runs at least once and every iteration continues where
// the previous one stopped, so verification checks the body from its entry state
// and again from the state that pass produces.
plena::Result<bool> verifyAssembly(const std::string &text) {
  auto words = plena::assemble(text);
  if (!words) return plena::Result<bool>::failure(words.error);
  return plena::verifyCore(words.value);
}
// ISA ver 1.0: W [K,N] and A [M,K] stay in the operand buffer; each fixed
// 32x32x32 M_MMA consumes one K slice, INIT first, ACC after.
constexpr const char *kLoadPair32 =
    "M_LOAD_WEIGHT_F16 1, 32, 32, 64\n M_LOAD_ACT_F16 2, 32, 32, 64\n";
TEST(CoreLoop, AccumulatesAcrossIterations) {
  // K streaming: each iteration reloads both operands and adds into the same
  // accumulator. The state is live on entry to every iteration.
  EXPECT_TRUE(bool(verifyAssembly(std::string(kLoadPair32) +
      "M_MMA_F16F16F32 INIT\n C_LOOP_BEGIN 7\n" + kLoadPair32 + "M_MMA_F16F16F32 ACC\n C_LOOP_END\n"
      "M_WRITEOUT_F16 3, 32, 32, 64\n C_FENCE_ALL")));
}
TEST(CoreLoop, RejectsInitOnALiveAccumulatorInTheSecondIteration) {
  auto result = verifyAssembly(std::string("C_LOOP_BEGIN 7\n") + kLoadPair32 +
      "M_MMA_F16F16F32 INIT\n C_LOOP_END\n M_WRITEOUT_F16 3, 32, 32, 64\n C_FENCE_ALL");
  EXPECT_FALSE(bool(result));
}
TEST(CoreLoop, RejectsWriteoutWithoutAccumulatorOnTheSecondIteration) {
  // The first pass drains the accumulator the block set up; the second finds
  // nothing to drain. Unrolling would have caught this, so the loop form must too.
  auto result = verifyAssembly(std::string(kLoadPair32) +
      "M_MMA_F16F16F32 INIT\n C_LOOP_BEGIN 7\n M_WRITEOUT_F16 3, 32, 32, 64\n C_LOOP_END\n C_FENCE_ALL");
  EXPECT_FALSE(bool(result));
  EXPECT_NE(result.error.find("writeout without live accumulator"), std::string::npos) << result.error;
}
TEST(CoreLoop, RejectsUnbalancedAndMalformedLoops) {
  EXPECT_FALSE(bool(verifyAssembly("C_LOOP_BEGIN 7\n C_WAIT_VPU\n C_FENCE_ALL")));
  EXPECT_FALSE(bool(verifyAssembly("C_LOOP_END\n C_FENCE_ALL")));
  EXPECT_FALSE(bool(verifyAssembly("C_LOOP_BEGIN 7\n C_LOOP_END\n C_FENCE_ALL")));
  // The hardware stack holds four levels.
  EXPECT_FALSE(bool(verifyAssembly(
      "C_LOOP_BEGIN 1\n C_LOOP_BEGIN 2\n C_LOOP_BEGIN 3\n C_LOOP_BEGIN 4\n C_LOOP_BEGIN 5\n"
      "C_WAIT_VPU\n C_LOOP_END\n C_LOOP_END\n C_LOOP_END\n C_LOOP_END\n C_LOOP_END\n C_FENCE_ALL")));
  EXPECT_TRUE(bool(verifyAssembly(
      "C_LOOP_BEGIN 1\n C_LOOP_BEGIN 2\n C_LOOP_BEGIN 3\n C_LOOP_BEGIN 4\n"
      "C_WAIT_VPU\n C_LOOP_END\n C_LOOP_END\n C_LOOP_END\n C_LOOP_END\n C_FENCE_ALL")));
}
TEST(CoreLoop, StillRejectsOperandLoadCrossingTheBoundary) {
  auto result = verifyAssembly("M_LOAD_WEIGHT_F16 1, 32, 32, 64\n C_FENCE_ALL");
  EXPECT_FALSE(bool(result));
  EXPECT_NE(result.error.find("operand load crosses CORE boundary"), std::string::npos) << result.error;
  // Reloading a slot the MMAs have not drained is what the simulator rejects.
  result = verifyAssembly("C_LOOP_BEGIN 7\n M_LOAD_WEIGHT_F16 1, 32, 32, 64\n C_LOOP_END\n C_FENCE_ALL");
  EXPECT_FALSE(bool(result));
  EXPECT_NE(result.error.find("already holds an unconsumed weight"), std::string::npos) << result.error;
}
TEST(Program, HazardsWithoutGlobalSerialization) {
  using M = plena::MemorySpace;
  plena::Program p{1, 256, 256, 1024, {load(0), core({{M::L2,0,64,false}}), load(64), load(0)}};
  auto s = plena::schedule(p); ASSERT_TRUE(bool(s)) << s.error;
  EXPECT_EQ(s.value.commands[1].dependencies, std::vector<uint32_t>({0}));
  EXPECT_TRUE(s.value.commands[2].dependencies.empty());
  EXPECT_EQ(s.value.commands[3].dependencies, std::vector<uint32_t>({0,1}));
  auto bytes = plena::encodeProgram(s.value); ASSERT_TRUE(bool(bytes));
  EXPECT_EQ(std::string(bytes.value.begin(), bytes.value.begin()+4), "PLNA");
}
TEST(Program, PartialOverlapAndCoreOrder) {
  using M = plena::MemorySpace;
  plena::Program p{2, 256, 256, 1024, {core({{M::L2,0,64,true}}), core({{M::L2,32,64,false}}), core()}};
  p.commands[1].core = 1;
  auto s = plena::schedule(p); ASSERT_TRUE(bool(s));
  EXPECT_EQ(s.value.commands[1].dependencies, std::vector<uint32_t>({0}));
  EXPECT_EQ(s.value.commands[2].dependencies, std::vector<uint32_t>({0}));
  p.commands[0].accesses[0].offset = 255;
  EXPECT_FALSE(bool(plena::schedule(p)));
}
TEST(Program, StridedColumnsStayIndependentButRealHazardsRemain) {
  using M = plena::MemorySpace;
  plena::Program p{2, 256, 512, 1024, {
      core({{M::L2,0,32,true,4,128}}), core({{M::L2,32,32,true,4,128}}),
      core({{M::L2,256,32,false}})}};
  p.commands[1].core = 1;
  auto s = plena::schedule(p); ASSERT_TRUE(bool(s)) << s.error;
  EXPECT_TRUE(s.value.commands[1].dependencies.empty());
  EXPECT_EQ(s.value.commands[2].dependencies, std::vector<uint32_t>({0}));
  p.commands[1].accesses[0].offset = 16;
  s = plena::schedule(p); ASSERT_TRUE(bool(s));
  EXPECT_EQ(s.value.commands[1].dependencies, std::vector<uint32_t>({0}));
  p.commands[1].accesses[0].rows = UINT64_MAX;
  EXPECT_FALSE(bool(plena::schedule(p)));
}
TEST(Program, DMAHolesDoNotCreateDependencies) {
  auto dma = load(0); dma.bytes = 32; dma.rows = 4; dma.dramStride = dma.l2Stride = 128;
  plena::Program p{1, 256, 512, 1024, {dma, core({{plena::MemorySpace::L2,32,32,true,4,128}})}};
  auto s = plena::schedule(p); ASSERT_TRUE(bool(s)) << s.error;
  EXPECT_TRUE(s.value.commands[1].dependencies.empty());
  p.commands[1].accesses[0].offset = 128;
  p.commands[1].accesses[0].rows = 1;
  s = plena::schedule(p); ASSERT_TRUE(bool(s));
  EXPECT_EQ(s.value.commands[1].dependencies, std::vector<uint32_t>({0}));
}
TEST(Program, AccumulatorAndOperandLifetimes) {
  auto check = [](const std::string &s) {
    auto words = plena::assemble(s);
    return words ? plena::verifyCore(words.value) : plena::Result<bool>::failure(words.error);
  };
  const std::string pair64 = "M_LOAD_WEIGHT_F16 1, 64, 32, 64\nM_LOAD_ACT_F16 2, 32, 64, 128\n";
  const std::string out = "M_WRITEOUT_F16 3, 32, 32, 64\nC_FENCE_ALL";
  EXPECT_FALSE(bool(check("M_MMA_F16F16F32 INIT\nC_FENCE_ALL")));                 // empty buffer
  EXPECT_TRUE(bool(check(pair64 + "M_MMA_F16F16F32 INIT\nM_MMA_F16F16F32 ACC\n" + out)));
  EXPECT_FALSE(bool(check(pair64 + "M_MMA_F16F16F32 INIT\n" + out)));              // K slice left
  EXPECT_FALSE(bool(check(pair64 + "M_MMA_F16F16F32 INIT\nM_MMA_F16F16F32 INIT\n" + out)));
  EXPECT_FALSE(bool(check(pair64 + "M_MMA_F16F16F32 ACC\nM_MMA_F16F16F32 ACC\n" + out)));
  EXPECT_FALSE(bool(check(pair64 + "M_MMA_F16F16F32 INIT\nM_MMA_F16F16F32 ACC\n"
                          "M_MMA_F16F16F32 ACC\n" + out)));                         // buffer drained
  EXPECT_FALSE(bool(check(pair64 + "M_MMA_F16F16F32 INIT\nM_MMA_F16F16F32 ACC\nC_FENCE_ALL")));
  EXPECT_FALSE(bool(check("M_LOAD_WEIGHT_F16 1, 64, 32, 64\nM_LOAD_ACT_F16 2, 32, 32, 64\n"
                          "M_MMA_F16F16F32 INIT\n" + out)));                        // K differs
  EXPECT_FALSE(bool(check(out)));
}
TEST(Program, RejectsCorruptedProgramAndTruncation) {
  plena::Program p{1,64,64,64,{core()}};
  auto encoded = plena::encodeProgram(p); ASSERT_TRUE(bool(encoded));
  std::vector<uint32_t> words;
  for (size_t i = 0; i < encoded.value.size(); i += 4) {
    uint32_t w = 0;
    for (unsigned b = 0; b < 4; ++b) w |= uint32_t(encoded.value[i+b]) << (8*b);
    words.push_back(w);
  }
  ASSERT_TRUE(bool(plena::verifyProgramWords(words)));
  auto corrupt = words; corrupt[23] = 0x1; // core instruction after the L1 span
  EXPECT_FALSE(bool(plena::verifyProgramWords(corrupt)));
  for (size_t size = 0; size < words.size(); ++size) {
    auto shortWords = std::vector<uint32_t>(words.begin(),words.begin()+size);
    if (size > 2) shortWords[2] = uint32_t(size);
    EXPECT_FALSE(bool(plena::verifyProgramWords(shortWords))) << size;
  }
}
TEST(Program, RectangularHazardsMatchBytewiseOracle) {
  using M = plena::MemorySpace;
  std::mt19937 rng(109);
  plena::Program p{4,256,256,256,{}};
  std::vector<std::set<unsigned>> ancestors;
  std::vector<std::set<unsigned>> readers(256);
  std::vector<int> writers(256,-1), previous(4,-1);
  for (unsigned id = 0; id < 160; ++id) {
    auto c = core(); c.core = rng()%4;
    for (unsigned j = 0; j < 5; ++j) {
      uint64_t width = 1+rng()%8, stride = width+rng()%8, rows = 1+rng()%5;
      uint64_t offset = rng()%(257-((rows-1)*stride+width));
      c.accesses.push_back({M::L2,offset,width,bool(rng()%2),rows,stride});
    }
    std::set<unsigned> deps;
    if (previous[c.core] >= 0) deps.insert(previous[c.core]);
    previous[c.core] = id;
    // Union accesses for this command before updating state. This independent
    // oracle enumerates bytes, never intervals or rectangular merges.
    bool reads[256]{}, writes[256]{};
    for (const auto &a : c.accesses)
      for (unsigned row = 0; row < a.rows; ++row)
        for (unsigned col = 0; col < a.bytes; ++col)
          (a.write ? writes : reads)[a.offset+row*a.stride+col] = true;
    for (unsigned b = 0; b < 256; ++b) {
      if ((reads[b] || writes[b]) && writers[b] >= 0) deps.insert(writers[b]);
      if (writes[b]) {
        deps.insert(readers[b].begin(),readers[b].end());
        readers[b].clear(); writers[b] = id;
      } else if (reads[b]) readers[b].insert(id);
    }
    auto reachable = deps;
    for (auto d : deps) reachable.insert(ancestors[d].begin(),ancestors[d].end());
    ancestors.push_back(std::move(reachable)); p.commands.push_back(std::move(c));
  }
  auto result = plena::schedule(p); ASSERT_TRUE(bool(result)) << result.error;
  std::vector<std::set<unsigned>> actual;
  for (unsigned id = 0; id < p.commands.size(); ++id) {
    std::set<unsigned> reachable;
    for (auto d : result.value.commands[id].dependencies) {
      reachable.insert(d); reachable.insert(actual[d].begin(),actual[d].end());
    }
    EXPECT_EQ(reachable,ancestors[id]) << "command " << id;
    actual.push_back(std::move(reachable));
  }
}
} // namespace
