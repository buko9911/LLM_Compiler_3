#include "npu/Analysis/MemoryPlan.h"
#include <gtest/gtest.h>
namespace {
using namespace npu;
BufferRequirement buffer(const char *name, uint64_t begin, uint64_t end) {
  BufferRequirement r; r.name = name; r.bytes = 64; r.begin = begin; r.end = end; return r;
}
TEST(MemoryPlan, CompletionExtendsLifetime) {
  MemoryLimits limits{64,64,1024,1,32,2};
  auto a = buffer("async",0,10), b = buffer("next",5,12);
  EXPECT_FALSE(bool(planMemory({a,b}, limits)));
  b.begin = 10;
  auto p = planMemory({a,b}, limits); ASSERT_TRUE(bool(p));
  EXPECT_EQ(p.value.buffers[0].address, p.value.buffers[1].address);
  p.value.buffers[1].requirement.begin = 5;
  EXPECT_FALSE(bool(verifyMemoryPlan(p.value,limits)));
}
TEST(MemoryPlan, ChoosesLegalFallbackAtPlacement) {
  MemoryLimits limits{64,64,1024,1,32,2};
  auto a = buffer("graph",0,10), b = buffer("staging",1,5);
  b.spillable = true;
  auto p = planMemory({a,b}, limits); ASSERT_TRUE(bool(p));
  EXPECT_EQ(p.value.buffers[1].action, BufferPlacement::Action::Spill);
  b.pure = b.inputsAvailable = b.preservesNumerics = true; b.rematerializeCycles = 1;
  p = planMemory({a,b}, limits); ASSERT_TRUE(bool(p));
  EXPECT_EQ(p.value.buffers[1].action, BufferPlacement::Action::Rematerialize);
  b.preservesNumerics = false;
  p = planMemory({a,b}, limits); ASSERT_TRUE(bool(p));
  EXPECT_EQ(p.value.buffers[1].action, BufferPlacement::Action::Spill);
}
TEST(MemoryPlan, PrivateCoreStorageAndDedicatedAccumulator) {
  MemoryLimits limits{64,64,1024,2,32,2};
  auto a = buffer("core0",0,10), b = buffer("core1",0,10);
  a.space = b.space = MemorySpace::L1; b.core = 1;
  auto p = planMemory({a,b}, limits); ASSERT_TRUE(bool(p));
  EXPECT_EQ(p.value.buffers[0].address, p.value.buffers[1].address);
  b.space = MemorySpace::Accumulator;
  EXPECT_FALSE(bool(planMemory({a,b},limits)));
}
} // namespace
