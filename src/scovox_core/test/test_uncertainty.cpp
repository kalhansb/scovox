/// @file test_uncertainty.cpp
/// Task 1.8: Uncertainty function correctness (>=15 tests).

#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include "scovox/uncertainty.hpp"

using namespace scovox;

// Helper to build a voxel with given Beta params
static Voxel makeBeta(float a, float b) {
  Voxel v = defaultVoxel();
  v.a_occ = a;
  v.a_free = b;
  return v;
}

// Helper to build a voxel with semantic evidence
static Voxel makeSemantic(float a_occ, float a_free,
                          uint16_t cls1, float cnt1,
                          uint16_t cls2 = 0, float cnt2 = 0.f,
                          float a_unk = 0.f) {
  Voxel v = defaultVoxel();
  v.a_occ = a_occ;
  v.a_free = a_free;
  v.sem_cls[0] = cls1; v.sem_cnt[0] = cnt1;
  v.sem_cls[1] = cls2; v.sem_cnt[1] = cnt2;
  v.a_unk = a_unk;
  return v;
}

// =====================================================================
// Beta variance
// =====================================================================

TEST(Uncertainty, VarianceBeta11ExactlyOneTwelfth) {
  auto v = makeBeta(1, 1);
  EXPECT_NEAR(variance(v), 1.0f / 12.0f, 1e-7f);
}

TEST(Uncertainty, VarianceBeta100_100MuchSmaller) {
  auto v_prior = makeBeta(1, 1);
  auto v_obs = makeBeta(100, 100);
  EXPECT_LT(variance(v_obs), variance(v_prior) * 0.02f);
}

TEST(Uncertainty, VarianceSymmetric) {
  EXPECT_FLOAT_EQ(variance(makeBeta(3, 7)), variance(makeBeta(7, 3)));
}

TEST(Uncertainty, VarianceDecreasesMonotonically) {
  float prev = variance(makeBeta(1, 1));
  for (float n : {5.f, 10.f, 50.f, 100.f}) {
    float cur = variance(makeBeta(n, n));
    EXPECT_LT(cur, prev) << "Variance should decrease at n=" << n;
    prev = cur;
  }
}

TEST(Uncertainty, VarianceZeroForDegenerate) {
  Voxel v = defaultVoxel();
  v.a_occ = 0.f;
  v.a_free = 0.f;
  EXPECT_FLOAT_EQ(variance(v), 0.0f);
}

// =====================================================================
// Beta entropy
// =====================================================================

TEST(Uncertainty, EntropyBeta11IsZero) {
  auto v = makeBeta(1, 1);
  EXPECT_NEAR(entropy(v), 0.0f, 1e-5f);
}

TEST(Uncertainty, EntropyDecreasesWithPeaking) {
  float h_uniform = entropy(makeBeta(1, 1));
  float h_peaked = entropy(makeBeta(20, 2));
  EXPECT_GT(h_uniform, h_peaked);
}

TEST(Uncertainty, EntropySymmetric) {
  EXPECT_NEAR(entropy(makeBeta(5, 10)), entropy(makeBeta(10, 5)), 1e-5f);
}

// scovox::entropy() is the Beta DIFFERENTIAL entropy, which is unbounded
// BELOW and diverges to large negatives on near-point-mass voxels — the
// documented "entropy trap". This is exactly the state every occupied
// split-substrate voxel reaches: a_occ = C*alpha_0 + accumulated evidence
// while a_free stays pinned at the prior alpha_0 (=0.01, C=14 -> occ prior
// 0.14). Pin the divergence so nobody re-uses entropy() as if it were a
// bounded Shannon stat: at Beta(0.14+50, 0.01) the closed form is ~-99
// (not in [0, ln2]). The map's "mean Shannon entropy" stat deliberately uses
// the bounded Bernoulli form below precisely because this poisons the mean.
TEST(Uncertainty, EntropyBetaNearPointMassDivergesNegative) {
  const float alpha0 = 0.01f;     // kDefaultDirichletPrior
  const float C = 14.f;           // default num_classes
  // Near-point-mass occupied voxel (huge a_occ, tiny a_free) — exercises the
  // Beta differential-entropy divergence regardless of the prior. The C*alpha0
  // term is just a convenient large offset, NOT the shipped occupancy prior
  // (now symmetric Beta(1,1); see docs/occupancy_prior.md).
  float h = entropy(makeBeta(C * alpha0 + 50.f, alpha0));
  EXPECT_TRUE(std::isfinite(h)) << "differential entropy must stay finite, got " << h;
  EXPECT_LT(h, -10.f) << "Beta differential entropy must diverge negative on a "
                         "near-point-mass voxel, got " << h;
}

// The bounded occupancy-uncertainty stat that production code (e.g. the
// occupancy map-stats aggregator, and the H_y term inside
// expectedInformationGain) uses INSTEAD of entropy() on the same near-
// point-mass voxel: Bernoulli Shannon entropy H(p_occ) with p_occ =
// a_occ/(a_occ+a_free). It is bounded in [0, ln2] regardless of how
// concentrated the Beta is — the property entropy() above lacks.
TEST(Uncertainty, BernoulliShannonEntropyBoundedOnNearPointMass) {
  const float alpha0 = 0.01f;
  const float C = 14.f;
  auto v = makeBeta(C * alpha0 + 50.f, alpha0);
  const float s = v.a_occ + v.a_free;
  const float p = v.a_occ / s;
  // Mirror the production H_y guard in uncertainty.cpp.
  float h_bern = 0.f;
  if (p > 1e-7f && p < 1.f - 1e-7f)
    h_bern = -p * std::log(p) - (1.f - p) * std::log(1.f - p);
  EXPECT_GE(h_bern, 0.f);
  EXPECT_LE(h_bern, static_cast<float>(M_LN2) + 1e-6f)
      << "Bernoulli Shannon entropy must be bounded by ln2, got " << h_bern;
  // And it is a small positive number here (near-certain occupancy), NOT the
  // ~-99 the differential entropy reports for the identical voxel.
  EXPECT_LT(h_bern, 0.1f);
}

// =====================================================================
// Expected Information Gain
// =====================================================================

TEST(Uncertainty, EIGPriorGreaterThanObserved) {
  float eig_prior = expectedInformationGain(makeBeta(1, 1));
  float eig_obs = expectedInformationGain(makeBeta(100, 100));
  EXPECT_GT(eig_prior, eig_obs);
}

TEST(Uncertainty, EIGVeryCertainApproxZero) {
  float eig = expectedInformationGain(makeBeta(200, 2));
  EXPECT_LT(eig, 0.01f);
}

TEST(Uncertainty, EIGAlwaysNonNegative) {
  for (float a : {1.f, 2.f, 5.f, 10.f, 50.f, 200.f}) {
    for (float b : {1.f, 2.f, 5.f, 10.f, 50.f, 200.f}) {
      EXPECT_GE(expectedInformationGain(makeBeta(a, b)), -1e-6f)
        << "Negative EIG at Beta(" << a << "," << b << ")";
    }
  }
}

TEST(Uncertainty, EIGMaximalAtUniformPrior) {
  float eig_uniform = expectedInformationGain(makeBeta(1, 1));
  for (auto [a, b] : std::initializer_list<std::pair<float,float>>{
       {2,2}, {5,5}, {10,1}, {1,10}, {50,50}}) {
    EXPECT_GE(eig_uniform, expectedInformationGain(makeBeta(a, b)) - 1e-6f)
      << "Beta(" << a << "," << b << ") has higher EIG than uniform";
  }
}

TEST(Uncertainty, EIGSymmetric) {
  EXPECT_NEAR(expectedInformationGain(makeBeta(10, 3)),
              expectedInformationGain(makeBeta(3, 10)), 1e-6f);
}

TEST(Uncertainty, EIGDecreasesMonotonically) {
  float prev = expectedInformationGain(makeBeta(1, 1));
  for (float n : {2.f, 5.f, 10.f, 50.f, 100.f}) {
    float cur = expectedInformationGain(makeBeta(n, n));
    EXPECT_LT(cur, prev) << "EIG should decrease at n=" << n;
    prev = cur;
  }
}

// =====================================================================
// Dirichlet semantic entropy
// =====================================================================

TEST(Uncertainty, SemanticEntropyDecreasesWithRepeatedObs) {
  auto v1 = makeSemantic(10, 2, 1, 5.0f, 2, 5.0f, 1.0f);
  float h1 = semanticEntropy(v1);
  auto v2 = makeSemantic(10, 2, 1, 50.0f, 2, 1.0f, 1.0f);
  float h2 = semanticEntropy(v2);
  EXPECT_GT(h1, h2);
}

TEST(Uncertainty, SemanticEntropyZeroForSingleClass) {
  auto v = makeSemantic(10, 2, 1, 100.0f, 0, 0.f, 0.001f);
  float h = semanticEntropy(v);
  EXPECT_LT(h, 0.5f);
}

// =====================================================================
// Dirichlet semantic variance
// =====================================================================

TEST(Uncertainty, SemanticVarianceNonNegative) {
  auto v = makeSemantic(10, 2, 3, 10.0f, 5, 5.0f, 2.0f);
  EXPECT_GE(semanticVariance(v, 3), 0.f);
  EXPECT_GE(semanticVariance(v, 5), 0.f);
}

TEST(Uncertainty, SemanticVarianceZeroForAbsentClass) {
  auto v = makeSemantic(10, 2, 1, 10.0f, 2, 5.0f);
  EXPECT_FLOAT_EQ(semanticVariance(v, 99), 0.0f);
}

TEST(Uncertainty, SemanticVarianceDecreasesWithEvidence) {
  auto v_low = makeSemantic(10, 2, 1, 2.0f, 2, 2.0f, 1.0f);
  auto v_high = makeSemantic(10, 2, 1, 50.0f, 2, 50.0f, 10.0f);
  EXPECT_GT(semanticVariance(v_low, 1), semanticVariance(v_high, 1));
}
