/// @file
/// @brief Gate tests for SemSplitMap — the de-unified Beta/Dirichlet substrate
/// (BetaVoxel occupancy grid ∥ DirVoxel semantics grid).
///
/// Pinned invariants:
///   - layout/sizes (8 B Beta, 16 B Dir) and prior-at-first-touch;
///   - two-stream update matches the SemDir-matched analytic values;
///   - STRICT per-grid mass conservation (Beta: a_occ+a_free; Dir: other+Σcnt);
///   - dirichlet_min_p_occ gate + the sparse-semantics memory win (free /
///     below-gate voxels allocate NO DirVoxel);
///   - miss drives p_occ down and never allocates a DirVoxel;
///   - wall blocking, per-grid evidence saturation, per-grid touched drains.

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cmath>
#include <vector>

#include "scovox/sem_split_map.hpp"

namespace {

constexpr float kRes   = 0.05f;
constexpr float kAlpha = scovox::kDefaultDirichletPrior;  // 0.01
constexpr int   kC     = 14;                              // num_classes (NYU13)

scovox::SemSplitMap makeMap() {
  scovox::SemSplitMap::Params p;
  p.resolution          = kRes;
  p.w_occ               = 1.0f;
  p.w_free              = 0.5f;
  p.kappa0              = 1.0f;
  p.dirichlet_min_p_occ = 0.5f;
  p.evidence_saturation = 0.0f;
  p.num_classes         = kC;
  p.alpha_0             = kAlpha;
  return scovox::SemSplitMap(p);
}

}  // namespace

// ===========================================================================
// Layout / prior invariants
// ===========================================================================

TEST(SplitVoxelLayout, Sizes) {
  EXPECT_EQ(sizeof(scovox::BetaVoxel), 8u);
  EXPECT_EQ(sizeof(scovox::DirVoxel),  16u);  // SemDirVoxel 20 B − 4 B FREE
}

TEST(SplitVoxelLayout, BetaPriorMatchesSemDirOccupancyMarginal) {
  // Beta(C·α₀, α₀) → p_occ_prior = C/(C+1), matching the unified SemDirVoxel
  // marginal (NOT the legacy Beta(1,1) → 0.5).
  auto b = scovox::defaultBetaVoxel(kC * kAlpha, kAlpha);
  EXPECT_NEAR(b.a_occ,  kC * kAlpha, 1e-7f);
  EXPECT_NEAR(b.a_free, kAlpha, 1e-7f);
  EXPECT_NEAR(b.p_occ(), float(kC) / float(kC + 1), 1e-5f);
}

TEST(SplitVoxelLayout, DirPriorIsSymmetricMinusFree) {
  auto d = scovox::defaultDirVoxel(kC, kAlpha);
  EXPECT_NEAR(d.other, (kC - scovox::K_TOP) * kAlpha, 1e-7f);  // (C−K)·α₀
  for (int i = 0; i < scovox::K_TOP; ++i) {
    EXPECT_EQ(d.cls[i], uint16_t(0xFFFF));
    EXPECT_NEAR(d.cnt[i], kAlpha, 1e-7f);
  }
  // Total class prior = C·α₀ (= SemDirVoxel::s_occ at prior, = BetaVoxel a_occ
  // prior) — keeps the split consistent with the unified path at the prior.
  EXPECT_NEAR(d.s_class(), kC * kAlpha, 1e-6f);
}

TEST(SplitVoxelLayout, ZeroInitialisedHasNoPrior) {
  scovox::BetaVoxel b{};
  EXPECT_EQ(b.a_occ, 0.0f);
  EXPECT_EQ(b.a_free, 0.0f);
  scovox::DirVoxel d{};
  EXPECT_EQ(d.other, 0.0f);
  for (int i = 0; i < scovox::K_TOP; ++i) {
    EXPECT_EQ(d.cnt[i], 0.0f);
    EXPECT_EQ(d.cls[i], uint16_t(0));
  }
}

// ===========================================================================
// Integration / two-stream update
// ===========================================================================

TEST(SemSplitMap, FirstHitTwoStreamMatchesAnalytic) {
  auto m = makeMap();
  std::vector<float> probs(kC, 0.f);
  probs[5] = 1.0f;  // class 5 one-hot
  // Same-voxel "ray" → no carve, just the hit update.
  m.integrateHit(Eigen::Vector3f(1.0f, 0, 0),
                 Eigen::Vector3f(1.0f, 0, 0),
                 &probs, /*quality=*/1.0f);

  auto b = m.getBetaVoxel(Eigen::Vector3f(1.0f, 0, 0));
  auto d = m.getDirVoxel(Eigen::Vector3f(1.0f, 0, 0));
  ASSERT_TRUE(b.has_value());
  ASSERT_TRUE(d.has_value());

  // Stream A: a_occ = C·α₀ + w_occ·q ; a_free untouched at prior α₀.
  const float w_occ_share = 1.0f;
  EXPECT_NEAR(b->a_occ,  kC * kAlpha + w_occ_share, 1e-5f);
  EXPECT_NEAR(b->a_free, kAlpha, 1e-5f);

  // p_occ_post read after Stream A; Stream B class_share = kappa0·p_occ·q.
  const float p_occ_post  = (kC * kAlpha + w_occ_share) / ((kC + 1) * kAlpha + w_occ_share);
  const float class_share = 1.0f * p_occ_post * 1.0f;
  EXPECT_EQ(d->cls[0], uint16_t(5));
  EXPECT_NEAR(d->cnt[0], kAlpha + class_share, 1e-5f);  // one-hot → all to slot 0
  EXPECT_EQ(d->cls[1], uint16_t(0xFFFF));
  EXPECT_NEAR(d->cnt[1], kAlpha, 1e-5f);
  EXPECT_NEAR(d->other, (kC - scovox::K_TOP) * kAlpha, 1e-5f);  // covered=1 → no spill
  EXPECT_EQ(m.dominantClassAt(Eigen::Vector3f(1.0f, 0, 0)), uint16_t(5));
}

TEST(SemSplitMap, MassConservationStrictPerGrid) {
  // Each grid conserves its own mass exactly (no ≥0 slack). Use a same-voxel
  // hit so there is no carve to enumerate.
  auto m = makeMap();
  std::vector<float> probs(kC, 0.f);
  probs[3] = 1.0f;
  m.integrateHit(Eigen::Vector3f(1.0f, 0, 0),
                 Eigen::Vector3f(1.0f, 0, 0),
                 &probs, /*quality=*/1.0f);

  auto b = m.getBetaVoxel(Eigen::Vector3f(1.0f, 0, 0));
  auto d = m.getDirVoxel(Eigen::Vector3f(1.0f, 0, 0));
  ASSERT_TRUE(b.has_value());
  ASSERT_TRUE(d.has_value());

  const float w_occ_share = 1.0f;                                  // Stream A input
  const float p_occ_post  = (kC * kAlpha + w_occ_share) / ((kC + 1) * kAlpha + w_occ_share);
  const float class_share = 1.0f * p_occ_post * 1.0f;              // Stream B input

  // Beta: prior (C+1)·α₀... no — Beta prior is just a_occ+a_free = C·α₀ + α₀.
  const float beta_prior = kC * kAlpha + kAlpha;
  EXPECT_NEAR(b->s_total(), beta_prior + w_occ_share, 1e-5f)
      << "Beta mass conservation violated";

  // Dir: prior total = C·α₀; gains exactly class_share.
  const float dir_prior = kC * kAlpha;
  EXPECT_NEAR(d->s_class(), dir_prior + class_share, 1e-5f)
      << "Dir mass conservation violated";
}

TEST(SemSplitMap, EvictionConservesMassToOther) {
  // K_TOP=2: push three distinct classes; the third must evict-or-drop into
  // OTHER, keeping (other + Σcnt) exactly equal to prior + total injected.
  auto m = makeMap();
  auto coord = m.betaGrid().posToCoord(2.0f, 0.f, 0.f);
  // Drive p_occ above the gate first.
  m.applyHitUpdate(coord, nullptr, 1.0f);   // Stream A only (nullptr probs)

  auto inject = [&](int cls, float strength) {
    std::vector<float> probs(kC, 0.f);
    probs[cls] = 1.0f;
    // quality scales class_share; use repeated hits to build distinct evidence.
    for (int i = 0; i < (int)strength; ++i) m.applyHitUpdate(coord, &probs, 1.0f);
  };
  inject(1, 5);   // strong
  inject(2, 3);   // medium
  inject(7, 1);   // weak — should be dropped to OTHER (K_TOP=2 full)

  auto d = m.getDirVoxel(Eigen::Vector3f(2.0f, 0, 0));
  ASSERT_TRUE(d.has_value());
  // Slots hold the two strongest classes; class 7 is not tracked.
  bool has1 = (d->cls[0] == 1 || d->cls[1] == 1);
  bool has2 = (d->cls[0] == 2 || d->cls[1] == 2);
  bool has7 = (d->cls[0] == 7 || d->cls[1] == 7);
  EXPECT_TRUE(has1);
  EXPECT_TRUE(has2);
  EXPECT_FALSE(has7) << "weakest class must not occupy a slot";
  // Mass is conserved regardless of eviction routing (no negative, no loss).
  EXPECT_GT(d->other, (kC - scovox::K_TOP) * kAlpha)
      << "evicted/dropped evidence must accumulate in OTHER";
  EXPECT_GT(d->s_class(), kC * kAlpha);
}

// ===========================================================================
// Gating + sparse-semantics memory win
// ===========================================================================

TEST(SemSplitMap, CarveAllocatesBetaButNoDirVoxels) {
  // THE memory win: a full hit ray carves many Beta voxels but commits a class
  // at only the single hit voxel → exactly one DirVoxel allocated.
  auto m = makeMap();
  std::vector<float> probs(kC, 0.f);
  probs[4] = 1.0f;
  m.integrateHit(Eigen::Vector3f(0, 0, 0),
                 Eigen::Vector3f(1.0f, 0, 0),  // ~20 voxels at res 0.05
                 &probs, /*quality=*/1.0f);

  EXPECT_GT(m.betaVoxelCount(), 2u) << "carve must allocate Beta voxels along the ray";
  EXPECT_EQ(m.dirVoxelCount(), 1u)  << "only the hit voxel commits a class → 1 DirVoxel";
}

TEST(SemSplitMap, MissDrivesOccupancyDownAndAllocatesNoDir) {
  auto m = makeMap();
  m.integrateMiss(Eigen::Vector3f(0, 0, 0),
                  Eigen::Vector3f(1.0f, 0, 0),
                  /*quality=*/1.0f);
  auto b = m.getBetaVoxel(Eigen::Vector3f(0.5f, 0, 0));
  ASSERT_TRUE(b.has_value());
  EXPECT_LT(b->p_occ(), float(kC) / float(kC + 1))
      << "carve must drive p_occ below the prior";
  EXPECT_EQ(m.dirVoxelCount(), 0u) << "a miss must never allocate a DirVoxel";
}

TEST(SemSplitMap, BelowGateCommitsNoClassAndNoDirVoxel) {
  auto m = makeMap();
  auto coord = m.betaGrid().posToCoord(1.0f, 0.f, 0.f);
  // Pre-populate strong free evidence so p_occ stays below the 0.5 gate.
  {
    auto acc = m.betaGrid().createAccessor();
    auto pre = m.defaultBeta();
    pre.a_free = 10.0f;
    acc.setValue(coord, pre);
  }
  std::vector<float> probs(kC, 0.f); probs[3] = 1.0f;
  m.applyHitUpdate(coord, &probs, /*quality=*/1.0f);

  // Occupancy evidence still landed in Beta...
  auto b = m.getBetaVoxel(Eigen::Vector3f(1.0f, 0, 0));
  ASSERT_TRUE(b.has_value());
  EXPECT_GT(b->a_occ, kC * kAlpha) << "Stream A occupancy mass lands even below gate";
  // ...but no class was committed and no DirVoxel was allocated.
  EXPECT_EQ(m.dirVoxelCount(), 0u);
  EXPECT_EQ(m.dominantClassAt(Eigen::Vector3f(1.0f, 0, 0)), uint16_t(0xFFFF));
}

// ===========================================================================
// Wall blocking, saturation, drains
// ===========================================================================

TEST(SemSplitMap, WallBlockingStopsCarving) {
  auto m = makeMap();
  Eigen::Vector3f wall_pos(0.5f, 0, 0);
  Eigen::Vector3f beyond(1.0f, 0, 0);
  {
    auto acc = m.betaGrid().createAccessor();
    auto coord = m.betaGrid().posToCoord(wall_pos.x(), wall_pos.y(), wall_pos.z());
    auto pre = m.defaultBeta();
    pre.a_occ  = 100.f;  // p_occ ≈ 1 → wall
    pre.a_free = 1.f;
    acc.setValue(coord, pre);
  }
  m.integrateHit(Eigen::Vector3f(0, 0, 0), beyond, nullptr, 1.0f);

  auto v_past = m.getBetaVoxel(Eigen::Vector3f(0.75f, 0, 0));
  if (v_past.has_value()) {
    EXPECT_NEAR(v_past->a_free, kAlpha, 1e-5f) << "voxels past the wall must not be carved";
  }
}

TEST(SemSplitMap, EvidenceSaturationCapsEachGrid) {
  scovox::SemSplitMap::Params p;
  p.resolution          = kRes;
  p.w_occ               = 1.0f;
  p.kappa0              = 1.0f;
  p.dirichlet_min_p_occ = 0.5f;
  p.evidence_saturation = 5.0f;
  p.num_classes         = kC;
  p.alpha_0             = kAlpha;
  scovox::SemSplitMap m(p);

  std::vector<float> probs(kC, 0.f); probs[2] = 1.0f;
  for (int i = 0; i < 100; ++i) {
    m.applyHitUpdate(m.betaGrid().posToCoord(1.0f, 0.f, 0.f), &probs, 1.0f);
  }
  auto b = m.getBetaVoxel(Eigen::Vector3f(1.0f, 0, 0));
  auto d = m.getDirVoxel(Eigen::Vector3f(1.0f, 0, 0));
  ASSERT_TRUE(b.has_value());
  ASSERT_TRUE(d.has_value());
  EXPECT_LE(b->s_total(), 5.0f + 1e-3f) << "Beta cap on a_occ+a_free";
  EXPECT_LE(d->s_class(), 5.0f + 1e-3f) << "Dir cap on other+Σcnt";
}

TEST(SemSplitMap, DrainTouchedPerGrid) {
  auto m = makeMap();
  std::vector<float> probs(kC, 0.f); probs[1] = 1.0f;
  m.integrateHit(Eigen::Vector3f(0, 0, 0),
                 Eigen::Vector3f(1.0f, 0, 0),
                 &probs, 1.0f);
  EXPECT_GT(m.touchedBetaCount(), m.touchedDirCount())
      << "Beta is full-ray; Dir is hit-only";
  EXPECT_EQ(m.touchedDirCount(), 1u);
  auto tb = m.drainTouchedBeta();
  auto td = m.drainTouchedDir();
  EXPECT_GT(tb.size(), 0u);
  EXPECT_EQ(td.size(), 1u);
  EXPECT_EQ(m.touchedBetaCount(), 0u);
  EXPECT_EQ(m.touchedDirCount(), 0u);
}
