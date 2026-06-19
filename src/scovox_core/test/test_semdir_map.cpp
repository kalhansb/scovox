/// @file
/// @brief Step-7.5 gate tests for SemDirMap (unified-Dirichlet voxel map).
/// Pinned invariants: symmetric Dirichlet prior at first observation, miss
/// does not flip p_occ upward, wall-blocking, evidence saturation, mass
/// conservation (the strict-equality upgrade of the legacy E5.1 invariant),
/// and the dirichlet_min_p_occ gate.

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cmath>
#include <vector>

#include "scovox/semdir_map.hpp"

namespace {

constexpr float kRes   = 0.05f;
constexpr float kAlpha = scovox::kDefaultDirichletPrior;  // 0.01

scovox::SemDirMap makeMap() {
  scovox::SemDirMap::Params p;
  p.resolution               = kRes;
  p.dirichlet_min_p_occ      = 0.5f;
  p.kappa0                   = 1.0f;
  p.evidence_saturation      = 0.0f;
  p.num_classes              = 14;
  p.alpha_0                  = kAlpha;
  return scovox::SemDirMap(p);
}

/// Sum of all Dirichlet pseudo-counts in a voxel — used for mass
/// conservation tests.
float totalMass(const scovox::SemDirVoxel& v) {
  return v.s_total();
}

}  // namespace

// ===========================================================================
// PARITY / INVARIANT TESTS — block merge if any fails.
// ===========================================================================

TEST(SemDirMapInvariants, FirstObservationStartsFromDirichletPrior) {
  auto m = makeMap();
  // One hit ray with quality=1.0 and no semantics. Two-stream calibration
  // (mirrors SemBeta — semdir_map.cpp:applyHitUpdate):
  //   Stream A: alpha_other += w_occ · q (no class commit)
  //   p_occ_post = (s_occ_pre + w_occ·q) / (s_total_pre + w_occ·q)
  //   Stream B (gated): class_share = kappa0 · p_occ_post · q; routes via
  //     softmax to slots+OTHER. With sem_probs=nullptr, all of Stream B's
  //     class_share lands in alpha_other (dirichletUpdate nullptr branch).
  // At default makeMap (w_occ=kappa0=1, q=1), prior s_occ=14α s_total=15α:
  //   Stream A bump: +1.0 → alpha_other = 12α + 1.0
  //   p_occ_post   = (14α+1)/(15α+1) = 1.14/1.15
  //   Stream B above gate → class_share = 1.14/1.15 ≈ 0.9913 added to OTHER
  m.integrateHit(Eigen::Vector3f(0, 0, 0),
                 Eigen::Vector3f(1.0f, 0, 0),
                 /*sem_probs=*/nullptr,
                 /*quality=*/1.0f);
  auto v = m.getVoxel(Eigen::Vector3f(1.0f, 0, 0));
  ASSERT_TRUE(v.has_value());

  // FREE prior untouched by hit (carve stops one voxel short of endpoint).
  EXPECT_NEAR(v->alpha_free, kAlpha, 1e-5f);
  // Compute the expected Stream A + Stream B contributions analytically.
  const float w_occ_share = 1.0f * 1.0f;   // w_occ · quality
  const float s_occ_post  = 14.f * kAlpha + w_occ_share;
  const float s_tot_post  = 15.f * kAlpha + w_occ_share;
  const float p_occ_post  = s_occ_post / s_tot_post;
  const float class_share = 1.0f * p_occ_post * 1.0f;   // kappa0 · p_occ_post · q
  EXPECT_NEAR(v->alpha_other,
              12.f * kAlpha + w_occ_share + class_share, 1e-5f);
  // Empty slots stay at prior α_0 (no semantics → sparse_add never fires).
  for (int i = 0; i < scovox::K_TOP; ++i) {
    EXPECT_EQ(v->cls[i], uint16_t(0xFFFF));
    EXPECT_NEAR(v->cnt[i], kAlpha, 1e-5f);
  }
  EXPECT_GT(v->p_occ(), 0.99f);
}

TEST(SemDirMapInvariants, MissDoesNotFlipOccupancyUpward) {
  auto m = makeMap();
  m.integrateMiss(Eigen::Vector3f(0, 0, 0),
                  Eigen::Vector3f(1.0f, 0, 0),
                  /*quality=*/1.0f);

  auto v_mid = m.getVoxel(Eigen::Vector3f(0.5f, 0, 0));
  ASSERT_TRUE(v_mid.has_value());
  EXPECT_LT(v_mid->p_occ(), 0.5f)
      << "miss must drive p_occ down, not up";
  // Class slots and OTHER unchanged by a miss.
  for (int i = 0; i < scovox::K_TOP; ++i) {
    EXPECT_EQ(v_mid->cls[i], uint16_t(0xFFFF));
    EXPECT_NEAR(v_mid->cnt[i], kAlpha, 1e-5f);
  }
  EXPECT_NEAR(v_mid->alpha_other, 12.f * kAlpha, 1e-5f);
}

TEST(SemDirMapInvariants, MassConservationStrictEquality) {
  // Upgraded E5.1 invariant: under unified Dirichlet, ΔΣα == Σ Δ inputs
  // is a strict equality (not a one-sided ≥ 0 slack like the SemBeta era).
  auto m = makeMap();

  // Track total mass before and after a sequence of mixed updates.
  // Initial state: voxel doesn't exist → mass contribution is 0.
  float input_mass = 0.f;
  const float w_occ  = 1.0f;
  const float w_free = 0.5f;  // default

  // Hit ray: contributes `w_occ * quality` to endpoint voxel + ~N carved
  // voxels each gaining `w_free * quality`. We can't easily enumerate the
  // carved voxel count, so instead pick a single-voxel ray and check
  // exactly.
  std::vector<float> probs(14, 0.f);
  probs[3] = 1.0f;  // class 3, one-hot

  m.integrateHit(Eigen::Vector3f(1.0f, 0, 0),
                 Eigen::Vector3f(1.0f, 0, 0),  // same voxel → no carve
                 &probs, /*quality=*/1.0f);
  // Two-stream calibration (mirrors SemBeta):
  //   Stream A: alpha_other += w_occ · quality
  //   Stream B: class_share = kappa0 · p_occ_post · quality
  // p_occ_post = (14α + w_occ·q) / (15α + w_occ·q) = 1.14/1.15.
  // At kappa0=1, q=1 → class_share = 1.14/1.15. Total input mass per hit
  // = w_occ·q (Stream A) + class_share (Stream B routed via sparse_add).
  const float s_occ_post  = 14.f * kAlpha + w_occ * 1.0f;
  const float s_tot_post  = 15.f * kAlpha + w_occ * 1.0f;
  const float p_occ_post  = s_occ_post / s_tot_post;
  input_mass += w_occ * 1.0f;                       // Stream A
  input_mass += 1.0f * p_occ_post * 1.0f;           // Stream B (kappa0=1, q=1)
  (void)w_free;  // unused (same-voxel ray, no carve)

  // The hit was a same-voxel "ray" so no carving happened (carveRay
  // early-returns when k0 == k_end). Endpoint mass should equal
  // prior_total + input_mass exactly.
  auto v = m.getVoxel(Eigen::Vector3f(1.0f, 0, 0));
  ASSERT_TRUE(v.has_value());
  const float prior_total = kAlpha        // FREE prior
                          + 12.f * kAlpha // OTHER prior (C − K = 12)
                          + 2.f * kAlpha; // K_TOP cnt-slot placeholders × α_0
  // Total prior should be (C+1)·α_0 = 15·α_0.
  EXPECT_NEAR(prior_total, 15.f * kAlpha, 1e-6f);
  EXPECT_NEAR(totalMass(*v), prior_total + input_mass, 1e-5f)
      << "mass conservation violated: ΔΣα ≠ Σ Δ inputs";
}

TEST(SemDirMapInvariants, DirichletUpdateRoutesByClassProb) {
  // Two-stream: Stream A bumps alpha_other by w_occ·q, then p_occ_post is
  // computed, then (above gate) Stream B routes kappa0·p_occ_post·q via
  // softmax. With one-hot probs[class 5]=1.0, all of Stream B lands in
  // slot 0. p_occ_post = (14α+w_occ·q)/(15α+w_occ·q) > 0.5 → gate fires.
  auto m = makeMap();
  std::vector<float> probs(14, 0.f);
  probs[5] = 1.0f;  // class 5 one-hot
  m.integrateHit(Eigen::Vector3f(0, 0, 0),
                 Eigen::Vector3f(1.0f, 0, 0),
                 &probs, /*quality=*/1.0f);
  auto v = m.getVoxel(Eigen::Vector3f(1.0f, 0, 0));
  ASSERT_TRUE(v.has_value());
  const float w_occ_share = 1.0f * 1.0f;   // makeMap: w_occ=1, q=1
  const float p_occ_post = (14.f * kAlpha + w_occ_share)
                         / (15.f * kAlpha + w_occ_share);
  const float class_share = 1.0f * p_occ_post * 1.0f;  // kappa0·p_occ_post·q
  EXPECT_EQ(v->cls[0], uint16_t(5));
  EXPECT_NEAR(v->cnt[0], kAlpha + class_share, 1e-5f);
  EXPECT_EQ(v->cls[1], uint16_t(0xFFFF));
  EXPECT_NEAR(v->cnt[1], kAlpha, 1e-5f);
  // Stream A also landed: alpha_other = (C−K)·α_0 + w_occ·q.
  EXPECT_NEAR(v->alpha_other, 12.f * kAlpha + w_occ_share, 1e-5f);
}

TEST(SemDirMapInvariants, BelowGateRoutesHitMassToOther) {
  // Force a low-p_occ regime by pre-populating with strong free evidence,
  // then deliver a hit. The hit's occupied mass should land entirely in
  // OTHER (below the dirichlet_min_p_occ gate), preserving the p_occ
  // marginal without committing to a class.
  scovox::SemDirMap::Params p;
  p.resolution = kRes;
  p.dirichlet_min_p_occ = 0.5f;  // hard gate
  p.num_classes = 14;
  p.alpha_0 = kAlpha;
  scovox::SemDirMap m(p);

  // Pre-populate: heavy free evidence.
  {
    auto acc = m.grid().createAccessor();
    auto coord = m.grid().posToCoord(1.0f, 0.f, 0.f);
    auto pre = scovox::defaultSemDirVoxel(14, kAlpha);
    pre.alpha_free = 10.0f;  // strong free → p_occ very small
    acc.setValue(coord, pre);
  }
  std::vector<float> probs(14, 0.f); probs[3] = 1.0f;
  // applyHitUpdate directly (skip the carve to keep things deterministic).
  auto coord = m.grid().posToCoord(1.0f, 0.f, 0.f);
  m.applyHitUpdate(coord, &probs, /*quality=*/1.0f);

  auto v = m.getVoxel(Eigen::Vector3f(1.0f, 0, 0));
  ASSERT_TRUE(v.has_value());
  // Below the gate — class slots should be unchanged.
  EXPECT_EQ(v->cls[0], uint16_t(0xFFFF));
  EXPECT_NEAR(v->cnt[0], kAlpha, 1e-5f);
  // alpha_other = (C−K)·α_0 prior + 1.0 hit mass.
  EXPECT_NEAR(v->alpha_other, 12.f * kAlpha + 1.0f, 1e-5f);
}

TEST(SemDirMapInvariants, WallBlockingStopsCarving) {
  auto m = makeMap();
  Eigen::Vector3f wall_pos(0.5f, 0, 0);
  Eigen::Vector3f beyond(1.0f, 0, 0);
  {
    auto acc = m.grid().createAccessor();
    auto coord = m.grid().posToCoord(wall_pos.x(), wall_pos.y(), wall_pos.z());
    auto pre = scovox::defaultSemDirVoxel(14, kAlpha);
    pre.alpha_other = 100.f;  // very high → p_occ ≈ 1
    pre.alpha_free  = 1.f;
    acc.setValue(coord, pre);
  }
  m.integrateHit(Eigen::Vector3f(0, 0, 0), beyond, nullptr, 1.0f);

  // Voxels BEYOND the wall (closer to the hit) should not have been carved.
  auto v_past = m.getVoxel(Eigen::Vector3f(0.75f, 0, 0));
  if (v_past.has_value()) {
    EXPECT_NEAR(v_past->alpha_free, kAlpha, 1e-5f);
  }
  // Wall voxel still has its high alpha_other.
  auto v_wall = m.getVoxel(wall_pos);
  ASSERT_TRUE(v_wall.has_value());
  EXPECT_NEAR(v_wall->alpha_other, 100.f, 1e-3f);
}

TEST(SemDirMapInvariants, EvidenceSaturationCapsTotalMass) {
  scovox::SemDirMap::Params p;
  p.resolution          = kRes;
  p.evidence_saturation = 5.0f;
  p.w_occ               = 1.0f;
  p.dirichlet_min_p_occ = 0.5f;
  p.num_classes         = 14;
  p.alpha_0             = kAlpha;
  scovox::SemDirMap m(p);

  for (int i = 0; i < 100; ++i) {
    m.integrateHit(Eigen::Vector3f(0, 0, 0),
                   Eigen::Vector3f(1.0f, 0, 0),
                   nullptr, 1.0f);
  }
  auto v = m.getVoxel(Eigen::Vector3f(1.0f, 0, 0));
  ASSERT_TRUE(v.has_value());
  EXPECT_LE(v->s_total(), 5.0f + 1e-3f) << "saturation cap on s_total()";
}

TEST(SemDirMapInvariants, DrainTouchedClearsBuffer) {
  auto m = makeMap();
  m.integrateHit(Eigen::Vector3f(0, 0, 0),
                 Eigen::Vector3f(1.0f, 0, 0),
                 nullptr, 1.0f);
  EXPECT_GT(m.touchedCount(), 0u);
  auto t1 = m.drainTouched();
  EXPECT_GT(t1.size(), 0u);
  EXPECT_EQ(m.touchedCount(), 0u);
}

// ===========================================================================
// Layout invariants — header-only.
// ===========================================================================

TEST(SemDirVoxelLayout, DefaultIsAtSymmetricDirichletPrior) {
  // Q5 invariant generalised: every freshly-allocated SemDirVoxel must
  // read back at the symmetric Dirichlet prior — α_0 on FREE, (C−K)·α_0
  // on OTHER, α_0 per cnt[] slot with 0xFFFF on cls[]. Failure means
  // first-hit integration would silently increment from 0 instead of
  // from prior (same class of bug as the K=1 uninit-memory regression).
  auto v = scovox::defaultSemDirVoxel(14, kAlpha);
  EXPECT_NEAR(v.alpha_free, kAlpha, 1e-7f);
  EXPECT_NEAR(v.alpha_other, 12.f * kAlpha, 1e-7f);
  for (int i = 0; i < scovox::K_TOP; ++i) {
    EXPECT_EQ(v.cls[i], uint16_t(0xFFFF));
    EXPECT_NEAR(v.cnt[i], kAlpha, 1e-7f);
  }
  // p_occ at prior = S_occ / S = (12α + 2α) / (12α + 2α + α) = 14/15.
  const float s_occ = 12.f * kAlpha + 2.f * kAlpha;
  const float s_tot = s_occ + kAlpha;
  EXPECT_NEAR(v.p_occ(), s_occ / s_tot, 1e-5f);

  // Total prior sum must equal (C + 1) · α_0 — the true generative
  // Dir(α_0, …, α_0) over `C + 1` categories.
  const float total_prior = v.alpha_free + v.alpha_other;
  float slot_prior = 0.f;
  for (int i = 0; i < scovox::K_TOP; ++i) slot_prior += v.cnt[i];
  EXPECT_NEAR(total_prior + slot_prior, 15.f * kAlpha, 1e-6f)
      << "total prior must be (C+1)·α_0";
}

TEST(SemDirVoxelLayout, ZeroInitialisedHasNoPrior) {
  // Bonxai zero-initialises new leaf blocks. Without defaultSemDirVoxel
  // the first hit would increment from α=0 — silent posterior corruption.
  // Pin: zero state really is zero (so defaultSemDirVoxel is necessary).
  scovox::SemDirVoxel v{};
  EXPECT_EQ(v.alpha_free,  0.0f);
  EXPECT_EQ(v.alpha_other, 0.0f);
  for (int i = 0; i < scovox::K_TOP; ++i) {
    EXPECT_EQ(v.cls[i], uint16_t(0));
    EXPECT_EQ(v.cnt[i], 0.0f);
  }
}
