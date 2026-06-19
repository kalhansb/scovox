/// @file
/// @brief Step-7 gate: per-voxel and per-frame consensus merge for v2.

#include <gtest/gtest.h>
#include "scovox/consensus_merge_v2.hpp"

namespace {

scovox::SemBetaVoxel mkSem(float occ, float fre, float unk,
                           uint16_t c0, float n0,
                           uint16_t c1 = 0xFFFF, float n1 = 0.f) {
  scovox::SemBetaVoxel v = scovox::defaultSemBetaVoxel();
  v.a_occ = occ; v.a_free = fre; v.a_unk = unk;
  v.sem_cls[0] = c0; v.sem_cnt[0] = n0;
  v.sem_cls[1] = c1; v.sem_cnt[1] = n1;
  return v;
}

}  // namespace

TEST(MergeTsdf, WeightedAverageDistance) {
  // Two robots see the surface with different distances and weights.
  // Fused distance is the weight-weighted average; weights sum.
  scovox::TsdfVoxel a{+0.10f, 2.0f};
  scovox::TsdfVoxel b{-0.10f, 4.0f};
  auto f = scovox::mergeTsdf(a, b);
  EXPECT_NEAR(f.weight, 6.0f, 1e-5f);
  // d_fused = (0.10*2 + (-0.10)*4) / 6 = (0.20 - 0.40)/6 = -0.0333
  EXPECT_NEAR(f.distance, -0.0333f, 1e-3f);
}

TEST(MergeTsdf, SymmetricInArguments) {
  scovox::TsdfVoxel a{+0.05f, 1.0f};
  scovox::TsdfVoxel b{-0.10f, 2.0f};
  auto fab = scovox::mergeTsdf(a, b);
  auto fba = scovox::mergeTsdf(b, a);
  EXPECT_FLOAT_EQ(fab.distance, fba.distance);
  EXPECT_FLOAT_EQ(fab.weight,   fba.weight);
}

TEST(MergeTsdf, UnobservedSideIsIdentity) {
  scovox::TsdfVoxel a{+0.05f, 3.0f};
  scovox::TsdfVoxel zero{0.0f, 0.0f};
  auto f = scovox::mergeTsdf(a, zero);
  EXPECT_NEAR(f.distance, +0.05f, 1e-5f);
  EXPECT_NEAR(f.weight,    3.0f,  1e-5f);
}

TEST(MergeSemBeta, BetaConsensusRemovesSharedPrior) {
  // Two robots both at Beta(1+2, 1+1) for example. The shared Beta(1,1)
  // prior is removed by the consensus rule.
  auto a = mkSem(/*occ=*/3.0f, /*fre=*/2.0f, /*unk=*/0.5f, 7, 2.0f);
  auto b = mkSem(/*occ=*/4.0f, /*fre=*/1.0f, /*unk=*/0.0f, 7, 3.0f);
  auto f = scovox::mergeSemBeta(a, b);
  EXPECT_NEAR(f.a_occ,  3.0f + 4.0f - 1.0f, 1e-5f);  // 6.0
  EXPECT_NEAR(f.a_free, 2.0f + 1.0f - 1.0f, 1e-5f);  // 2.0
  EXPECT_NEAR(f.a_unk,  0.5f + 0.0f,        1e-5f);  // a_unk just sums
  // Same class -> sparse_add merges into slot 0: cnt = 2+3 = 5
  EXPECT_EQ(f.sem_cls[0], uint16_t(7));
  EXPECT_NEAR(f.sem_cnt[0], 5.0f, 1e-5f);
}

TEST(MergeSemBeta, DifferentClassesPopulateBothSlots) {
  auto a = mkSem(2.0f, 1.0f, 0.0f, /*c0=*/7,  /*n0=*/3.0f);
  auto b = mkSem(2.0f, 1.0f, 0.0f, /*c0=*/11, /*n0=*/2.0f);
  auto f = scovox::mergeSemBeta(a, b);
  // slot 0: class 7 with count 3 (carried from a); slot 1: class 11 with count 2 (added from b).
  // Order may differ depending on sparse_add ordering, but the multiset must contain {7:3, 11:2}.
  std::vector<std::pair<uint16_t, float>> got;
  for (int i = 0; i < scovox::K_TOP; ++i) {
    if (f.sem_cnt[i] > 0.f) got.emplace_back(f.sem_cls[i], f.sem_cnt[i]);
  }
  ASSERT_EQ(got.size(), 2u);
  bool found_7 = false, found_11 = false;
  for (auto& p : got) {
    if (p.first == 7  && std::abs(p.second - 3.0f) < 1e-5f) found_7  = true;
    if (p.first == 11 && std::abs(p.second - 2.0f) < 1e-5f) found_11 = true;
  }
  EXPECT_TRUE(found_7);
  EXPECT_TRUE(found_11);
}

TEST(MergeFrames, UnionOfCoordsWithPerCoordMerge) {
  scovox::BinarySerializerV2::Frame a;
  a.resolution = 0.05f;
  a.tsdf_deltas.push_back({{1, 0, 0}, {+0.05f, 2.0f}});
  a.tsdf_deltas.push_back({{2, 0, 0}, {-0.05f, 1.0f}});
  a.sembeta_deltas.push_back({{1, 0, 0}, mkSem(2.f, 1.f, 0.f, 7, 1.5f)});

  scovox::BinarySerializerV2::Frame b;
  b.resolution = 0.05f;
  b.tsdf_deltas.push_back({{1, 0, 0}, {-0.05f, 4.0f}});  // overlap
  b.tsdf_deltas.push_back({{3, 0, 0}, {+0.10f, 1.0f}});  // unique to b
  b.sembeta_deltas.push_back({{1, 0, 0}, mkSem(3.f, 1.f, 0.f, 7, 2.5f)});

  auto f = scovox::mergeFrames(a, b);

  // 3 unique TSDF coords (1,0,0)+(2,0,0)+(3,0,0)
  EXPECT_EQ(f.tsdf_deltas.size(), 3u);
  // 1 SemBeta coord (only (1,0,0) in both)
  EXPECT_EQ(f.sembeta_deltas.size(), 1u);

  // Find the merged TSDF voxel at (1,0,0) and verify weighted average.
  for (const auto& d : f.tsdf_deltas) {
    if (d.coord.x == 1 && d.coord.y == 0 && d.coord.z == 0) {
      EXPECT_NEAR(d.data.weight, 6.0f, 1e-5f);
      // d = (0.05*2 + (-0.05)*4)/6 = -0.0167
      EXPECT_NEAR(d.data.distance, -0.0167f, 1e-3f);
    }
  }
}
