/// @file test_consensus.cpp
/// Task 1.9 + 2.3: Beta-principled consensus fusion (>=10 tests).

#include <gtest/gtest.h>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "scovox/scovoxmap.hpp"

using namespace scovox;

static Params makeParams() {
  Params p;
  p.resolution = 0.5;
  p.consensus_kl_threshold = 5.0f;
  p.consensus_tau_occ_gate = 0.6f;
  return p;
}

static Voxel makeOcc(float a_occ, float a_free) {
  Voxel v{}; v.a_occ = a_occ; v.a_free = a_free; return v;
}

static Voxel makeOccSem(float a_occ, float a_free,
                        uint16_t cls1, float cnt1,
                        uint16_t cls2 = 0, float cnt2 = 0.f) {
  Voxel v{}; v.a_occ = a_occ; v.a_free = a_free;
  v.sem_cls[0] = cls1; v.sem_cnt[0] = cnt1; v.sem_cls[1] = cls2; v.sem_cnt[1] = cnt2;
  return v;
}

// =====================================================================
// Beta combination: Beta(a1+a2-1, b1+b2-1)
// =====================================================================

TEST(Consensus, BetaCombinationFormula) {
  // Beta(10,5) + Beta(8,3) -> Beta(10+8-1, 5+3-1) = Beta(17, 7)
  Map map(makeParams());
  Voxel dst = makeOcc(10, 5);
  Voxel src = makeOcc(8, 3);
  map.consensusMerge(dst, src);
  EXPECT_FLOAT_EQ(dst.a_occ, 17.f);
  EXPECT_FLOAT_EQ(dst.a_free, 7.f);
}

TEST(Consensus, IdenticalVoxelsPreserveProbability) {
  Map map(makeParams());
  Voxel dst = makeOcc(20, 5);
  float p_before = dst.p_occ();
  Voxel src = makeOcc(20, 5);
  map.consensusMerge(dst, src);
  // Same ratio -> same probability
  EXPECT_NEAR(dst.p_occ(), p_before, 0.02f);
}

TEST(Consensus, IdenticalVoxelsIncreaseEvidence) {
  Map map(makeParams());
  Voxel dst = makeOcc(20, 5);
  float s_before = dst.a_occ + dst.a_free;
  Voxel src = makeOcc(20, 5);
  map.consensusMerge(dst, src);
  // Combined evidence should be greater
  EXPECT_GT(dst.a_occ + dst.a_free, s_before);
}

TEST(Consensus, PriorSourceAddsNoEvidence) {
  // Merging with a prior Beta(1,1) should not change dst
  // Beta(10,5) + Beta(1,1) -> Beta(10+1-1, 5+1-1) = Beta(10, 5)
  Map map(makeParams());
  Voxel dst = makeOcc(10, 5);
  Voxel src = makeOcc(1, 1);
  map.consensusMerge(dst, src);
  EXPECT_FLOAT_EQ(dst.a_occ, 10.f);
  EXPECT_FLOAT_EQ(dst.a_free, 5.f);
}

TEST(Consensus, PriorMergePreservesOneFloor) {
  // Two prior-only voxels merged: Beta(1,1) ⊕ Beta(1,1) = Beta(1+1−1, 1+1−1)
  // = Beta(1, 1). The merge math itself preserves the lower bound on
  // valid inputs (no clamp needed; the previous max(1, ·) floor was
  // removed 2026-05-03).
  Map map(makeParams());
  Voxel dst = makeOcc(1, 1);
  Voxel src = makeOcc(1, 1);
  map.consensusMerge(dst, src);
  EXPECT_FLOAT_EQ(dst.a_occ, 1.f);
  EXPECT_FLOAT_EQ(dst.a_free, 1.f);
}

TEST(Consensus, HighEvidenceDominatesLow) {
  Map map(makeParams());
  Voxel dst = makeOcc(100, 10);  // p ~ 0.91, strong
  Voxel src = makeOcc(3, 2);     // p = 0.60, weak

  float p_before = dst.p_occ();
  map.consensusMerge(dst, src);
  // Result dominated by dst's high evidence
  EXPECT_GT(dst.p_occ(), 0.85f);
  EXPECT_NEAR(dst.p_occ(), p_before, 0.06f);
}

// =====================================================================
// Beta-KL utility (the function itself is preserved in scovox_core for
// callers that want an explicit disagreement metric; consensusMerge no
// longer consults it — return value used to be a discarded `conflict`
// bool, removed 2026-05-03)
// =====================================================================

TEST(Consensus, BetaKLSymmetryProperty) {
  // KL(a||b) != KL(b||a) in general, but both should be non-negative
  Voxel a = makeOcc(20, 5);
  Voxel b = makeOcc(5, 20);
  float kl_ab = betaKL(a, b);
  float kl_ba = betaKL(b, a);
  EXPECT_GT(kl_ab, 0.f);
  EXPECT_GT(kl_ba, 0.f);
}

TEST(Consensus, BetaKLZeroForIdentical) {
  Voxel a = makeOcc(10, 5);
  EXPECT_NEAR(betaKL(a, a), 0.f, 1e-5f);
}

// =====================================================================
// Semantic additive Dirichlet merge
// =====================================================================

TEST(Consensus, SemanticAdditiveMerge) {
  Params p = makeParams();
  p.consensus_tau_occ_gate = 0.0f;  // always merge semantics
  Map map(p);

  Voxel dst = makeOccSem(20, 5, 5, 10.0f, 3, 2.0f);
  Voxel src = makeOccSem(20, 5, 5, 8.0f, 3, 1.0f);
  map.consensusMerge(dst, src);

  // Counts should be additive
  float cls5 = 0.f, cls3 = 0.f;
  for (int si = 0; si < K_TOP; ++si) {
    if (dst.sem_cls[si] == 5) cls5 = dst.sem_cnt[si];
    if (dst.sem_cls[si] == 3) cls3 = dst.sem_cnt[si];
  }
  EXPECT_NEAR(cls5, 18.0f, 0.01f);  // 10 + 8
  EXPECT_NEAR(cls3, 3.0f, 0.01f);   // 2 + 1
}

// 2026-05-03: gate removed from consensusMerge — Dirichlet evidence is
// additive under conditional independence given the latent (occ, class).
// This test now asserts the *new* invariant: semantics merge regardless of
// post-merge occupancy. Even when the merged Beta says "free", the source's
// semantic counts must still be folded into dst.
TEST(Consensus, SemanticMergedRegardlessOfOccupancy) {
  Params p = makeParams();
  p.consensus_tau_occ_gate = 0.9f;  // would have blocked the merge pre-2026-05-03
  Map map(p);

  // Both free after merge: Beta(3+3-1, 20+20-1) = Beta(5, 39), p~0.11
  Voxel dst = makeOccSem(3, 20, 1, 10.0f);
  Voxel src = makeOccSem(3, 20, 2, 8.0f);
  map.consensusMerge(dst, src);

  EXPECT_LT(dst.p_occ(), 0.5f);
  // src's class 2 with cnt 8 must now be present in dst — the gate no longer
  // blocks it. Either as a new slot, or (if dst was full) routed into a_unk.
  float cls2_count = 0.f;
  for (int si = 0; si < K_TOP; ++si) {
    if (dst.sem_cls[si] == 2) cls2_count = dst.sem_cnt[si];
  }
  EXPECT_GT(cls2_count + dst.a_unk, 0.f);
  // Class 1's count is preserved (additive identity, src had no class-1 mass).
  float cls1_count = 0.f;
  for (int si = 0; si < K_TOP; ++si) {
    if (dst.sem_cls[si] == 1) cls1_count = dst.sem_cnt[si];
  }
  EXPECT_FLOAT_EQ(cls1_count, 10.0f);
}
