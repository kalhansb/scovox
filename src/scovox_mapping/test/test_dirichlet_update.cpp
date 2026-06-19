/// @file test_dirichlet_update.cpp
/// Task 1.7: Dirichlet semantic update correctness (>=12 tests).

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <Eigen/Core>
#include "scovox/scovoxmap.hpp"

using namespace scovox;

// Helper: create a map with gate wide-open so semantics always fuse.
// Uses high w_occ and low semantic_occ_gate to ensure the gate passes.
static Map makeSemanticMap() {
  Params p;
  p.resolution = 0.5;
  p.w_occ = 20.0f;        // heavy hit weight so gate opens quickly
  p.semantic_occ_gate = 0.0f;  // never block semantics by gate
  p.kappa0 = 2.0f;
  return Map(p);
}

// Helper: build up occupancy so the gate is open, then return the voxel position.
static Eigen::Vector3f prepareOccupied(Map& map, int n_hits = 5) {
  Eigen::Vector3f origin(0, 0, 0), hit(2, 0, 0);
  for (int i = 0; i < n_hits; ++i)
    map.integrateRay(origin, hit);
  return hit;
}

// =====================================================================
// 1. Single-class repeated observation -> dominant count grows
// =====================================================================

TEST(DirichletUpdate, SingleClassRepeatedGrows) {
  auto map = makeSemanticMap();
  Eigen::Vector3f hit = prepareOccupied(map);

  // Observe class 3 at 100% confidence, 10 times
  std::vector<float> probs(10, 0.0f);
  probs[3] = 1.0f;
  Eigen::Vector3f origin(0, 0, 0);
  for (int i = 0; i < 10; ++i)
    map.integrateRay(origin, hit, false, &probs);

  Voxel v = defaultVoxel();
  ASSERT_TRUE(map.getVoxel(hit, v));

  // Class 3 should dominate
  float cls3_cnt = 0.f;
  for (int si = 0; si < K_TOP; ++si) {
    if (v.sem_cls[si] == 3 && v.sem_cnt[si] > 0.f) cls3_cnt = v.sem_cnt[si];
  }
  EXPECT_GT(cls3_cnt, 5.0f);
}

TEST(DirichletUpdate, SingleClassCountIncreases) {
  auto map = makeSemanticMap();
  Eigen::Vector3f hit = prepareOccupied(map);
  Eigen::Vector3f origin(0, 0, 0);

  std::vector<float> probs(10, 0.0f);
  probs[2] = 0.9f;

  map.integrateRay(origin, hit, false, &probs);
  Voxel v1 = defaultVoxel();
  ASSERT_TRUE(map.getVoxel(hit, v1));
  float cnt1 = 0.f;
  for (int si = 0; si < K_TOP; ++si) if (v1.sem_cls[si] == 2) cnt1 = v1.sem_cnt[si];

  map.integrateRay(origin, hit, false, &probs);
  Voxel v2 = defaultVoxel();
  ASSERT_TRUE(map.getVoxel(hit, v2));
  float cnt2 = 0.f;
  for (int si = 0; si < K_TOP; ++si) if (v2.sem_cls[si] == 2) cnt2 = v2.sem_cnt[si];

  EXPECT_GT(cnt2, cnt1);
}

// =====================================================================
// 2. Mixed observation -> counts proportional to p_obs
// =====================================================================

TEST(DirichletUpdate, MixedObsProportionalCounts) {
  auto map = makeSemanticMap();
  Eigen::Vector3f hit = prepareOccupied(map);
  Eigen::Vector3f origin(0, 0, 0);

  // 70% class 1, 30% class 2 — both above min_confidence
  std::vector<float> probs(10, 0.0f);
  probs[1] = 0.7f;
  probs[2] = 0.3f;

  for (int i = 0; i < 20; ++i)
    map.integrateRay(origin, hit, false, &probs);

  Voxel v = defaultVoxel();
  ASSERT_TRUE(map.getVoxel(hit, v));

  float cnt1 = 0.f, cnt2 = 0.f;
  for (int si = 0; si < K_TOP; ++si) {
    if (v.sem_cls[si] == 1) cnt1 = v.sem_cnt[si];
    if (v.sem_cls[si] == 2) cnt2 = v.sem_cnt[si];
  }

  EXPECT_GT(cnt1, 0.f);
  EXPECT_GT(cnt2, 0.f);
  // Ratio should approximate 0.7/0.3 ~ 2.33
  if (cnt2 > 0.f) {
    float ratio = cnt1 / cnt2;
    EXPECT_NEAR(ratio, 0.7f / 0.3f, 0.5f);
  }
}

TEST(DirichletUpdate, TwoClassesBothStored) {
  auto map = makeSemanticMap();
  Eigen::Vector3f hit = prepareOccupied(map);
  Eigen::Vector3f origin(0, 0, 0);

  std::vector<float> probs(10, 0.0f);
  probs[4] = 0.6f;
  probs[7] = 0.4f;

  for (int i = 0; i < 10; ++i)
    map.integrateRay(origin, hit, false, &probs);

  Voxel v = defaultVoxel();
  ASSERT_TRUE(map.getVoxel(hit, v));

  bool has4 = false, has7 = false;
  for (int si = 0; si < K_TOP; ++si) {
    if (v.sem_cls[si] == 4 && v.sem_cnt[si] > 0.f) has4 = true;
    if (v.sem_cls[si] == 7 && v.sem_cnt[si] > 0.f) has7 = true;
  }
  EXPECT_TRUE(has4);
  EXPECT_TRUE(has7);
}

// =====================================================================
// 3. Gate allows update when p_occ exceeds threshold
// =====================================================================

TEST(DirichletUpdate, GateAllowsHighEvidence) {
  Params p;
  p.resolution = 0.5;
  p.w_occ = 20.0f;
  p.semantic_occ_gate = 0.0f;  // very permissive
  p.kappa0 = 2.0f;
  Map map(p);

  Eigen::Vector3f origin(0, 0, 0), hit(2, 0, 0);
  std::vector<float> probs(10, 0.0f);
  probs[5] = 0.9f;

  for (int i = 0; i < 5; ++i)
    map.integrateRay(origin, hit, false, &probs);

  Voxel v = defaultVoxel();
  ASSERT_TRUE(map.getVoxel(hit, v));

  float total_sem = 0.f;
  for (int si = 0; si < K_TOP; ++si) total_sem += v.sem_cnt[si];
  EXPECT_GT(total_sem, 0.0f);
}

// =====================================================================
// 4. a_unk increases proportionally to (1 - sum(above-threshold p_obs))
// =====================================================================

TEST(DirichletUpdate, UnknownGrowsWithUncertainty) {
  auto map = makeSemanticMap();
  Eigen::Vector3f hit = prepareOccupied(map);
  Eigen::Vector3f origin(0, 0, 0);

  // sum of named p = 0.9 -> unknown gets weight proportional to (1 - 0.9) = 0.1
  std::vector<float> probs(10, 0.0f);
  probs[1] = 0.6f;
  probs[2] = 0.3f;

  for (int i = 0; i < 10; ++i)
    map.integrateRay(origin, hit, false, &probs);

  Voxel v = defaultVoxel();
  ASSERT_TRUE(map.getVoxel(hit, v));
  EXPECT_GT(v.a_unk, 0.0f);
}

TEST(DirichletUpdate, CertainObsMinimalUnknown) {
  auto map = makeSemanticMap();
  Eigen::Vector3f hit = prepareOccupied(map);
  Eigen::Vector3f origin(0, 0, 0);

  // max_p = 1.0 -> unknown gets weight proportional to 0.0
  std::vector<float> probs(10, 0.0f);
  probs[3] = 1.0f;

  for (int i = 0; i < 10; ++i)
    map.integrateRay(origin, hit, false, &probs);

  Voxel v = defaultVoxel();
  ASSERT_TRUE(map.getVoxel(hit, v));
  // a_unk should be zero or negligible when observation is 100% certain
  EXPECT_LT(v.a_unk, 0.01f);
}

TEST(DirichletUpdate, UnknownProportionalToComplement) {
  // Compare two maps: one with max_p=0.8, another with max_p=0.5
  auto map1 = makeSemanticMap();
  auto map2 = makeSemanticMap();
  Eigen::Vector3f hit1 = prepareOccupied(map1);
  Eigen::Vector3f hit2 = prepareOccupied(map2);
  Eigen::Vector3f origin(0, 0, 0);

  std::vector<float> probs1(10, 0.0f), probs2(10, 0.0f);
  probs1[1] = 0.8f;  // unknown proportional to 0.2
  probs2[1] = 0.5f;  // unknown proportional to 0.5

  for (int i = 0; i < 10; ++i) {
    map1.integrateRay(origin, hit1, false, &probs1);
    map2.integrateRay(origin, hit2, false, &probs2);
  }

  Voxel v1 = defaultVoxel(), v2 = defaultVoxel();
  ASSERT_TRUE(map1.getVoxel(hit1, v1));
  ASSERT_TRUE(map2.getVoxel(hit2, v2));

  // map2 (more uncertain) should have more a_unk
  EXPECT_GT(v2.a_unk, v1.a_unk);
}

// =====================================================================
// 5. Additional edge cases
// =====================================================================

TEST(DirichletUpdate, NullProbsSkipsSemUpdate) {
  auto map = makeSemanticMap();
  Eigen::Vector3f origin(0, 0, 0), hit(2, 0, 0);

  // Integrate with nullptr class_probs
  for (int i = 0; i < 10; ++i)
    map.integrateRay(origin, hit, false, nullptr);

  Voxel v = defaultVoxel();
  ASSERT_TRUE(map.getVoxel(hit, v));
  float total_sem = 0.f;
  for (int si = 0; si < K_TOP; ++si) total_sem += v.sem_cnt[si];
  EXPECT_FLOAT_EQ(total_sem, 0.0f);
  EXPECT_FLOAT_EQ(v.a_unk, 0.0f);
}

TEST(DirichletUpdate, A0DeltaTracksTotal) {
  auto map = makeSemanticMap();
  Eigen::Vector3f hit = prepareOccupied(map);
  Eigen::Vector3f origin(0, 0, 0);

  std::vector<float> probs(10, 0.0f);
  probs[1] = 0.6f;
  probs[2] = 0.3f;

  for (int i = 0; i < 10; ++i)
    map.integrateRay(origin, hit, false, &probs);

  Voxel v = defaultVoxel();
  ASSERT_TRUE(map.getVoxel(hit, v));

  // a0() should equal sum of all named class counts + a_unk
  float sum_named = 0.f;
  for (int si = 0; si < K_TOP; ++si) sum_named += v.sem_cnt[si];
  EXPECT_NEAR(v.a0(), sum_named + v.a_unk, 0.01f);
}

TEST(DirichletUpdate, QualityScalesEvidence) {
  auto map1 = makeSemanticMap();
  auto map2 = makeSemanticMap();
  Eigen::Vector3f hit1 = prepareOccupied(map1);
  Eigen::Vector3f hit2 = prepareOccupied(map2);
  Eigen::Vector3f origin(0, 0, 0);

  std::vector<float> probs(10, 0.0f);
  probs[1] = 0.9f;

  // quality=1.0 vs quality=0.5
  map1.integrateRay(origin, hit1, false, &probs, /*quality=*/1.0f);
  map2.integrateRay(origin, hit2, false, &probs, /*quality=*/0.5f);

  Voxel v1 = defaultVoxel(), v2 = defaultVoxel();
  ASSERT_TRUE(map1.getVoxel(hit1, v1));
  ASSERT_TRUE(map2.getVoxel(hit2, v2));

  float cnt1 = 0.f, cnt2 = 0.f;
  for (int si = 0; si < K_TOP; ++si) if (v1.sem_cls[si] == 1) cnt1 = v1.sem_cnt[si];
  for (int si = 0; si < K_TOP; ++si) if (v2.sem_cls[si] == 1) cnt2 = v2.sem_cnt[si];

  EXPECT_GT(cnt1, cnt2) << "Higher quality should produce more evidence";
}

// =====================================================================
// 6. Semantic mode variants (ablation baselines)
// =====================================================================

static Map makeNaiveMap() {
  Params p;
  p.resolution = 0.5;
  p.w_occ = 20.0f;
  p.semantic_occ_gate = 0.0f;
  p.semantic_mode = SemanticMode::NAIVE;
  return Map(p);
}

static Map makeMajorityMap() {
  Params p;
  p.resolution = 0.5;
  p.w_occ = 20.0f;
  p.semantic_occ_gate = 0.0f;
  p.semantic_mode = SemanticMode::MAJORITY_VOTE;
  return Map(p);
}

TEST(SemanticMode, NaiveOverwritesPreviousClass) {
  auto map = makeNaiveMap();
  Eigen::Vector3f hit = prepareOccupied(map);
  Eigen::Vector3f origin(0, 0, 0);

  // First observe class 3
  std::vector<float> probs1(10, 0.0f);
  probs1[3] = 1.0f;
  map.integrateRay(origin, hit, false, &probs1);

  // Then observe class 7
  std::vector<float> probs2(10, 0.0f);
  probs2[7] = 1.0f;
  map.integrateRay(origin, hit, false, &probs2);

  Voxel v = defaultVoxel();
  ASSERT_TRUE(map.getVoxel(hit, v));

  // Only class 7 should remain (last-observation-wins)
  EXPECT_EQ(v.sem_cls[0], 7);
  EXPECT_FLOAT_EQ(v.sem_cnt[0], 1.0f);
  // No other class should have evidence
  for (int i = 1; i < K_TOP; ++i)
    EXPECT_FLOAT_EQ(v.sem_cnt[i], 0.0f);
  EXPECT_FLOAT_EQ(v.a_unk, 0.0f);
}

TEST(SemanticMode, NaivePicksDominantClass) {
  auto map = makeNaiveMap();
  Eigen::Vector3f hit = prepareOccupied(map);
  Eigen::Vector3f origin(0, 0, 0);

  // Observe mixed: class 2 dominant
  std::vector<float> probs(10, 0.0f);
  probs[2] = 0.7f;
  probs[5] = 0.3f;
  map.integrateRay(origin, hit, false, &probs);

  Voxel v = defaultVoxel();
  ASSERT_TRUE(map.getVoxel(hit, v));
  EXPECT_EQ(v.sem_cls[0], 2);
}

TEST(SemanticMode, MajorityVoteAccumulates) {
  auto map = makeMajorityMap();
  Eigen::Vector3f hit = prepareOccupied(map);
  Eigen::Vector3f origin(0, 0, 0);

  std::vector<float> probs(10, 0.0f);
  probs[4] = 1.0f;

  // 10 observations -> count should be 10
  for (int i = 0; i < 10; ++i)
    map.integrateRay(origin, hit, false, &probs);

  Voxel v = defaultVoxel();
  ASSERT_TRUE(map.getVoxel(hit, v));

  float cnt4 = 0.f;
  for (int si = 0; si < K_TOP; ++si)
    if (v.sem_cls[si] == 4) cnt4 = v.sem_cnt[si];

  EXPECT_NEAR(cnt4, 10.0f, 0.1f);
}

TEST(SemanticMode, MajorityVoteIgnoresQuality) {
  auto map1 = makeMajorityMap();
  auto map2 = makeMajorityMap();
  Eigen::Vector3f hit1 = prepareOccupied(map1);
  Eigen::Vector3f hit2 = prepareOccupied(map2);
  Eigen::Vector3f origin(0, 0, 0);

  std::vector<float> probs(10, 0.0f);
  probs[1] = 0.9f;

  // quality=1.0 vs quality=0.1 — majority vote ignores both
  map1.integrateRay(origin, hit1, false, &probs, 1.0f);
  map2.integrateRay(origin, hit2, false, &probs, 0.1f);

  Voxel v1 = defaultVoxel(), v2 = defaultVoxel();
  ASSERT_TRUE(map1.getVoxel(hit1, v1));
  ASSERT_TRUE(map2.getVoxel(hit2, v2));

  float cnt1 = 0.f, cnt2 = 0.f;
  for (int si = 0; si < K_TOP; ++si) if (v1.sem_cls[si] == 1) cnt1 = v1.sem_cnt[si];
  for (int si = 0; si < K_TOP; ++si) if (v2.sem_cls[si] == 1) cnt2 = v2.sem_cnt[si];

  // Both should add exactly 1
  EXPECT_FLOAT_EQ(cnt1, cnt2);
}

TEST(SemanticMode, MajorityVoteDominantClassWins) {
  auto map = makeMajorityMap();
  Eigen::Vector3f hit = prepareOccupied(map);
  Eigen::Vector3f origin(0, 0, 0);

  // 7 observations of class 2, 3 of class 5
  std::vector<float> probs2(10, 0.0f), probs5(10, 0.0f);
  probs2[2] = 1.0f;
  probs5[5] = 1.0f;

  for (int i = 0; i < 7; ++i) map.integrateRay(origin, hit, false, &probs2);
  for (int i = 0; i < 3; ++i) map.integrateRay(origin, hit, false, &probs5);

  Voxel v = defaultVoxel();
  ASSERT_TRUE(map.getVoxel(hit, v));

  // Find dominant class
  float best_cnt = 0.f; uint16_t best_cls = 0;
  for (int si = 0; si < K_TOP; ++si) {
    if (v.sem_cnt[si] > best_cnt) { best_cnt = v.sem_cnt[si]; best_cls = v.sem_cls[si]; }
  }
  EXPECT_EQ(best_cls, 2);
  EXPECT_NEAR(best_cnt, 7.0f, 0.1f);
}
