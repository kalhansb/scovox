/// @file test_semantic_audit.cpp
/// Regression tests for the semantic-model audit fixes:
///   Bug 1: mass conservation when sparse_add takes the DROP/EVICT path.

#include <gtest/gtest.h>
#include <Eigen/Core>
#include "scovox/scovoxmap.hpp"
#include "scovox/node_utils.hpp"

using namespace scovox;

namespace {

Map makeWideOpenMap() {
  Params p;
  p.resolution = 0.5;
  // Symmetric high evidence weights so path voxels become confidently free
  // (and hit voxel confidently occupied) after a small number of warmup
  // rays. Under the joint ray-casting likelihood this drives reach_prob → 1
  // at the hit voxel — needed for single-observation mass invariants.
  p.w_occ = 20.0f;
  p.w_free = 20.0f;
  p.semantic_occ_gate = 0.0f;
  p.kappa0 = 2.0f;
  return Map(p);
}

Eigen::Vector3f openGate(Map& map, int hits = 50) {
  Eigen::Vector3f origin(0, 0, 0), hit(2, 0, 0);
  for (int i = 0; i < hits; ++i) map.integrateRay(origin, hit);
  return hit;
}

float totalSem(const Voxel& v) {
  float s = v.a_unk;
  for (int i = 0; i < K_TOP; ++i) s += v.sem_cnt[i];
  return s;
}

}  // namespace

// =====================================================================
// Bug 1: mass conservation in DROP path
// =====================================================================

TEST(SemanticAudit, MassConservedWhenMoreClassesThanKtop) {
  // K_TOP = 2. Three classes above min_confidence forces sparse_add to take
  // the DROP path on the third (its inc <= existing min slot count).
  // Pre-fix: total mass added = w * Σp_i (over-credits by ~20%).
  // Post-fix: total mass added = w (exactly).
  auto map = makeWideOpenMap();
  Eigen::Vector3f hit = openGate(map);
  Eigen::Vector3f origin(0, 0, 0);

  std::vector<float> probs(10, 0.0f);
  probs[0] = 0.5f;
  probs[1] = 0.3f;
  probs[2] = 0.2f;  // sum = 1.0, normalized

  Voxel before = defaultVoxel();
  ASSERT_TRUE(map.getVoxel(hit, before));
  const float total_before = totalSem(before);

  map.integrateRay(origin, hit, false, &probs);

  Voxel after = defaultVoxel();
  ASSERT_TRUE(map.getVoxel(hit, after));
  const float total_after = totalSem(after);

  // gate = 1 (hard threshold), kappa0 = 2 → w = 2.
  const float delta = total_after - total_before;
  EXPECT_NEAR(delta, 2.0f, 0.05f)
      << "Single observation must add exactly w pseudocounts";
}

TEST(SemanticAudit, MassConservedAcrossManyObservations) {
  // 100 observations of a 3-class distribution. Cumulative mass must equal
  // 100 * w to within FP. Pre-fix this drifted as observations evicted slots.
  auto map = makeWideOpenMap();
  Eigen::Vector3f hit = openGate(map);
  Eigen::Vector3f origin(0, 0, 0);

  std::vector<float> probs(10, 0.0f);
  probs[3] = 0.5f;
  probs[4] = 0.3f;
  probs[5] = 0.2f;

  Voxel before = defaultVoxel();
  ASSERT_TRUE(map.getVoxel(hit, before));
  const float total_before = totalSem(before);

  for (int i = 0; i < 100; ++i)
    map.integrateRay(origin, hit, false, &probs);

  Voxel after = defaultVoxel();
  ASSERT_TRUE(map.getVoxel(hit, after));
  const float total_after = totalSem(after);

  // Expected: 100 * w ≈ 200. Tolerance accounts for first-obs gate ramp.
  const float delta = total_after - total_before;
  EXPECT_NEAR(delta, 200.0f, 2.0f);
}

TEST(SemanticAudit, MassConservedOnEvictionWithFilledSlots) {
  // Pre-fill slots, then send a third class — exercises sparse_add eviction.
  auto map = makeWideOpenMap();
  Eigen::Vector3f hit = openGate(map);
  Eigen::Vector3f origin(0, 0, 0);

  // Fill K_TOP=2 slots with classes 5 and 6.
  std::vector<float> obs1(10, 0.0f);
  obs1[5] = 0.6f;
  obs1[6] = 0.4f;
  map.integrateRay(origin, hit, false, &obs1);

  Voxel mid = defaultVoxel();
  ASSERT_TRUE(map.getVoxel(hit, mid));
  const float mid_total = totalSem(mid);

  // New class 7 with strong probability — should evict the smaller slot.
  std::vector<float> obs2(10, 0.0f);
  obs2[7] = 0.9f;
  map.integrateRay(origin, hit, false, &obs2);

  Voxel end = defaultVoxel();
  ASSERT_TRUE(map.getVoxel(hit, end));
  const float end_total = totalSem(end);

  EXPECT_NEAR(end_total - mid_total, 2.0f, 0.05f);
}

TEST(SemanticAudit, MassConservedOnDropTinyClass) {
  // Pre-fill slots, then send a third class with TINY probability that
  // can't displace the existing min slot — DROP path.
  auto map = makeWideOpenMap();
  Eigen::Vector3f hit = openGate(map);
  Eigen::Vector3f origin(0, 0, 0);

  std::vector<float> obs1(10, 0.0f);
  obs1[5] = 0.6f;
  obs1[6] = 0.3f;
  map.integrateRay(origin, hit, false, &obs1);
  map.integrateRay(origin, hit, false, &obs1);  // build up the slots
  map.integrateRay(origin, hit, false, &obs1);

  Voxel mid = defaultVoxel();
  ASSERT_TRUE(map.getVoxel(hit, mid));
  const float mid_total = totalSem(mid);

  // Class 7 at the bare minimum — too small to evict.
  std::vector<float> obs2(10, 0.0f);
  obs2[7] = 0.15f;
  map.integrateRay(origin, hit, false, &obs2);

  Voxel end = defaultVoxel();
  ASSERT_TRUE(map.getVoxel(hit, end));
  const float end_total = totalSem(end);

  EXPECT_NEAR(end_total - mid_total, 2.0f, 0.05f);
}

// =====================================================================
// argmaxClassConfidence (node_utils.hpp) — Hutter-framework K_v+1
// Dirichlet posterior probability of the argmax tracked class.
// =====================================================================

TEST(ArgmaxClassConfidence, ZeroEvidenceReturnsZeroConfidence) {
  // Regression guard against the "{0, 1.0} on empty voxel" hallucination
  // surfaced in the 2026-05-03 review. With n_active == 0 the helper must
  // *not* compute (0+1)/(0+0+0+1) and return maximal confidence for
  // class 0 — that would be silent garbage downstream of any threshold.
  Voxel v = defaultVoxel();  // a_occ=1, a_free=1, all sem_cnt=0, a_unk=0
  auto [cls, p] = argmaxClassConfidence(v);
  EXPECT_EQ(cls, 0);
  EXPECT_FLOAT_EQ(p, 0.f);
}

TEST(ArgmaxClassConfidence, BetaEvidenceButNoSemanticsStaysZero) {
  // A voxel with strong Beta evidence (e.g. carved free, or hit by depth
  // without a semantic label) but zero Dirichlet observations must still
  // report confidence 0, not the Dirichlet-prior-only value.
  Voxel v = defaultVoxel();
  v.a_occ = 50.f;
  v.a_free = 5.f;
  auto [cls, p] = argmaxClassConfidence(v);
  EXPECT_EQ(cls, 0);
  EXPECT_FLOAT_EQ(p, 0.f);
}

TEST(ArgmaxClassConfidence, SingleClassDominantWithPrior) {
  // One tracked slot with cnt=10, no eviction, no a_unk: confidence is
  // bounded away from both 0 and 1 — the Dirichlet prior + Hutter
  // residual prevent the degenerate 1.0 the pre-2026-05-03 raw bv/sm
  // rule would produce here.
  Voxel v = defaultVoxel();
  v.sem_cls[0] = 7;
  v.sem_cnt[0] = 10.f;
  auto [cls, p] = argmaxClassConfidence(v);
  EXPECT_EQ(cls, 7);
  EXPECT_GT(p, 0.f);
  EXPECT_LT(p, 1.f);
}

TEST(ArgmaxClassConfidence, EvictedMassShrinksConfidence) {
  // Two cells with the same argmax-class evidence. The one with non-zero
  // a_unk (representing K_TOP-evicted untracked-class mass) must report
  // *lower* confidence than the one with a_unk == 0 — that's the whole
  // reason a_unk has to participate in the denominator.
  Voxel a = defaultVoxel();
  a.sem_cls[0] = 3;
  a.sem_cnt[0] = 8.f;

  Voxel b = a;
  b.a_unk = 5.f;  // evicted untracked-class mass

  auto [cls_a, p_a] = argmaxClassConfidence(a);
  auto [cls_b, p_b] = argmaxClassConfidence(b);
  EXPECT_EQ(cls_a, 3);
  EXPECT_EQ(cls_b, 3);
  EXPECT_LT(p_b, p_a);
}

TEST(ArgmaxClassConfidence, ConfidenceInValidRange) {
  // Stress over a non-trivial configuration: probability must always
  // lie in [0, 1].
  Voxel v = defaultVoxel();
  for (int i = 0; i < K_TOP; ++i) {
    v.sem_cls[i] = static_cast<uint16_t>(i + 1);
    v.sem_cnt[i] = static_cast<float>(K_TOP - i);
  }
  v.a_unk = 3.f;
  auto [cls, p] = argmaxClassConfidence(v);
  EXPECT_GE(p, 0.f);
  EXPECT_LE(p, 1.f);
  (void)cls;
}

