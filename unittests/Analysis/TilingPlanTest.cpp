#include "plena/Analysis/TilingPlan.h"
#include <gtest/gtest.h>
namespace {
plena::TilingRequest request() {
  plena::TilingRequest r;
  r.shape = {1024,1536,576};
  r.hardware = {4194304,8388608,17179869184,1,32,32,32,32,1,0,"test"};
  return r;
}
TEST(TilingPlan, SearchRanksOnlyFeasiblePlans) {
  auto r = request(); auto plans = plena::enumerateTilingPlans(r);
  ASSERT_TRUE(bool(plans)) << plans.error;
  EXPECT_GT(plans.value.size(), 100u);
  uint64_t last = 0;
  for (const auto &p : plans.value) {
    EXPECT_GE(p.estimatedCycles,last); last = p.estimatedCycles;
    EXPECT_EQ(p.l2.k,576u);
    EXPECT_LE(p.placement.l1HighWater[0],r.hardware.l1Bytes);
    EXPECT_LE(p.placement.l2HighWater,r.hardware.l2Bytes);
  }
}
TEST(TilingPlan, L1PressureChoosesKChunksWithoutCrossingCoreBoundary) {
  auto r = request(); r.hardware.l1Bytes = 8192;
  auto p = plena::chooseTilingPlan(r); ASSERT_TRUE(bool(p)) << p.error;
  EXPECT_EQ(p.value.schedule,plena::Schedule::Hybrid);
  EXPECT_LT(p.value.l1.k,r.shape.k); EXPECT_EQ(p.value.l2.k,r.shape.k);
}
TEST(TilingPlan, GraphBudgetAndTailShapes) {
  auto r = request(); r.shape = {1,33,65};
  auto p = plena::chooseTilingPlan(r); ASSERT_TRUE(bool(p)) << p.error;
  EXPECT_EQ(p.value.array.m,1u);
  r.graphResidentBytes = r.hardware.l2Bytes;
  EXPECT_FALSE(bool(plena::chooseTilingPlan(r)));
  r.graphResidentBytes = 0; r.hardware.l2Bytes = 64;
  EXPECT_FALSE(bool(plena::chooseTilingPlan(r)));
}
TEST(TilingPlan, DoubleStagingFallsBackWhenOnlyOneFits) {
  auto r = request(); r.shape = {32,32,32}; r.stages = 2;
  r.hardware.l2Bytes = 8192;
  auto p = plena::chooseTilingPlan(r); ASSERT_TRUE(bool(p)) << p.error;
  EXPECT_EQ(p.value.stages,1u);
  r.hardware.l2Bytes = 16384;
  p = plena::chooseTilingPlan(r); ASSERT_TRUE(bool(p)) << p.error;
  EXPECT_EQ(p.value.stages,2u);
  EXPECT_EQ(p.value.placement.l2HighWater,12288u);
}
TEST(TilingPlan, StageByteCountDoesNotWrapAt32Bits) {
  auto r = request(); r.shape = {32,32,32}; r.stages = 1u << 31;
  r.hardware.l2Bytes = uint64_t(1) << 50;
  auto p = plena::chooseTilingPlan(r); ASSERT_TRUE(bool(p)) << p.error;
  EXPECT_EQ(p.value.stages,r.stages);
  EXPECT_EQ(p.value.placement.l2HighWater,uint64_t(r.stages)*6144);
}
TEST(TilingPlan, HierarchicalPanelsReuseWeightsAcrossRows) {
  auto r = request(); r.shape = {512,4096,4096}; r.hardware.cores = 4;
  r.stages = 2; r.searchStages = true;
  auto plans = plena::enumerateTilingPlans(r);
  ASSERT_TRUE(bool(plans)) << plans.error;
  const auto &best = plans.value.front();
  EXPECT_GT(best.l1.m,32u);
  EXPECT_EQ(best.l1.k,r.shape.k);
  EXPECT_LE(best.array.m,32u); EXPECT_LE(best.array.n,32u);
  EXPECT_GT(best.coreSplit,1u);
  bool single = false, doubleStage = false, chunked = false;
  for (const auto &p : plans.value) {
    single |= p.stages == 1; doubleStage |= p.stages == 2;
    if (p.l1.k < r.shape.k) {
      chunked = true;
      EXPECT_LE(p.l1.m,32u); EXPECT_LE(p.l1.n,32u);
    }
  }
  EXPECT_TRUE(single); EXPECT_TRUE(doubleStage); EXPECT_TRUE(chunked);
  EXPECT_LT(best.localTrafficBytes,2*(r.shape.m*r.shape.k*((r.shape.n+31)/32)+
                                     r.shape.k*r.shape.n*((r.shape.m+31)/32)+r.shape.m*r.shape.n));
}
} // namespace
