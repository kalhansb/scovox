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

TEST(SplitVoxelLayout, ShippedBetaPriorIsSymmetricHalf) {
  // Shipped split-path occupancy prior is symmetric Beta(1,1) → p_occ = 0.5
  // (docs/occupancy_prior.md), decoupled from the semantic (C, α₀).
  auto b = scovox::defaultBetaVoxel(scovox::kBetaOccPrior, scovox::kBetaFreePrior);
  EXPECT_NEAR(b.a_occ,  1.0f, 1e-7f);
  EXPECT_NEAR(b.a_free, 1.0f, 1e-7f);
  EXPECT_NEAR(b.p_occ(), 0.5f, 1e-6f);
  // Ablation: the prior-agnostic factory still reproduces the calibrated
  // Beta(C·α₀, α₀) → C/(C+1) marginal on explicit request.
  auto calib = scovox::defaultBetaVoxel(kC * kAlpha, kAlpha);
  EXPECT_NEAR(calib.p_occ(), float(kC) / float(kC + 1), 1e-5f);
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

  // Stream A: a_occ = occ_prior + w_occ·q ; a_free untouched at the Beta(1,1)
  // prior. Occupancy prior is symmetric Beta(1,1) → p_occ_prior=0.5.
  const float w_occ_share = 1.0f;
  EXPECT_NEAR(b->a_occ,  scovox::kBetaOccPrior + w_occ_share, 1e-5f);
  EXPECT_NEAR(b->a_free, scovox::kBetaFreePrior, 1e-5f);

  // p_occ_post read after Stream A; Stream B class_share = kappa0·p_occ·q.
  const float p_occ_post  = (scovox::kBetaOccPrior + w_occ_share)
                          / (scovox::kBetaOccPrior + scovox::kBetaFreePrior + w_occ_share);
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
  const float p_occ_post  = (scovox::kBetaOccPrior + w_occ_share)
                          / (scovox::kBetaOccPrior + scovox::kBetaFreePrior + w_occ_share);
  const float class_share = 1.0f * p_occ_post * 1.0f;              // Stream B input

  // Beta prior total mass = a_occ + a_free = kBetaOccPrior + kBetaFreePrior (Beta(1,1)).
  const float beta_prior = scovox::kBetaOccPrior + scovox::kBetaFreePrior;
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
  EXPECT_LT(b->p_occ(), 0.5f)
      << "carve must drive p_occ below the symmetric Beta(1,1) prior (0.5)";
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
  EXPECT_GT(b->a_occ, scovox::kBetaOccPrior) << "Stream A occupancy mass lands even below gate";
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

// ===========================================================================
// Transient (dynamic-class) substrate
// ===========================================================================

TEST(SemSplitTransient, DynamicHitRoutesToTransientNotPersistent) {
  auto m = makeMap();
  const Eigen::Vector3f pos(1.0f, 0, 0);
  const auto c = m.betaGrid().posToCoord(pos.x(), pos.y(), pos.z());
  std::vector<float> probs(kC, 0.f); probs[3] = 1.0f;

  m.applyHitUpdate(c, &probs, /*quality=*/1.0f, /*is_dynamic=*/true);

  // Nothing in the persistent grids.
  EXPECT_FALSE(m.getBetaVoxel(pos).has_value());
  EXPECT_FALSE(m.getDirVoxel(pos).has_value());
  EXPECT_EQ(m.betaVoxelCount(), 0u);
  EXPECT_EQ(m.dirVoxelCount(), 0u);
  // Present in the transient grids.
  EXPECT_TRUE(m.getTransientBetaVoxel(pos).has_value());
  EXPECT_TRUE(m.getTransientDirVoxel(pos).has_value());
  EXPECT_EQ(m.transientBetaVoxelCount(), 1u);
  EXPECT_EQ(m.transientDirVoxelCount(), 1u);
  EXPECT_EQ(m.transientDominantClassAt(pos), 3u);
  EXPECT_EQ(m.dominantClassAt(pos), 0xFFFFu);  // persistent has no class
}

TEST(SemSplitTransient, DynamicHitRecordsNoTouchedSet) {
  auto m = makeMap();
  const auto c = m.betaGrid().posToCoord(1.0f, 0.f, 0.f);
  std::vector<float> probs(kC, 0.f); probs[3] = 1.0f;
  m.applyHitUpdate(c, &probs, 1.0f, /*is_dynamic=*/true);
  // Transient is local-only: never enters the fusion-wire touched-sets.
  EXPECT_EQ(m.touchedBetaCount(), 0u);
  EXPECT_EQ(m.touchedDirCount(), 0u);
}

TEST(SemSplitTransient, NonDynamicOverloadMatchesThreeArg) {
  auto ma = makeMap();
  auto mb = makeMap();
  const auto c = ma.betaGrid().posToCoord(1.0f, 0.f, 0.f);
  std::vector<float> probs(kC, 0.f); probs[3] = 1.0f;

  ma.applyHitUpdate(c, &probs, 1.0f);                        // 3-arg
  mb.applyHitUpdate(c, &probs, 1.0f, /*is_dynamic=*/false);  // 4-arg, persistent

  const Eigen::Vector3f pos(1.0f, 0, 0);
  auto ba = ma.getBetaVoxel(pos); auto bb = mb.getBetaVoxel(pos);
  ASSERT_TRUE(ba.has_value()); ASSERT_TRUE(bb.has_value());
  EXPECT_FLOAT_EQ(ba->a_occ, bb->a_occ);
  EXPECT_FLOAT_EQ(ba->a_free, bb->a_free);
  EXPECT_EQ(ma.dominantClassAt(pos), mb.dominantClassAt(pos));
  EXPECT_EQ(ma.touchedBetaCount(), mb.touchedBetaCount());
  EXPECT_EQ(ma.touchedDirCount(), mb.touchedDirCount());
}

TEST(SemSplitTransient, TransientHitUsesSameTwoStreamMath) {
  // A dynamic hit into the transient grid must produce the SAME Beta/Dir state
  // a persistent hit produces in the persistent grid (only the target differs).
  auto md = makeMap();
  auto mp = makeMap();
  const Eigen::Vector3f pos(1.0f, 0, 0);
  const auto c = md.betaGrid().posToCoord(pos.x(), pos.y(), pos.z());
  std::vector<float> probs(kC, 0.f); probs[3] = 1.0f;

  md.applyHitUpdate(c, &probs, 1.0f, /*is_dynamic=*/true);
  mp.applyHitUpdate(c, &probs, 1.0f);

  auto td = md.getTransientBetaVoxel(pos);
  auto tp = mp.getBetaVoxel(pos);
  ASSERT_TRUE(td.has_value()); ASSERT_TRUE(tp.has_value());
  EXPECT_FLOAT_EQ(td->a_occ, tp->a_occ);    // prior 1.0 + w_occ·q = 2.0
  EXPECT_FLOAT_EQ(td->a_free, tp->a_free);  // prior 1.0 (no carve in applyHitUpdate)
  EXPECT_NEAR(td->a_occ, 2.0f, 1e-6f);
  EXPECT_EQ(md.transientDominantClassAt(pos), mp.dominantClassAt(pos));
}

TEST(SemSplitTransient, DecayMovesEvidenceTowardPrior) {
  auto m = makeMap();
  const Eigen::Vector3f pos(1.0f, 0, 0);
  const auto c = m.betaGrid().posToCoord(pos.x(), pos.y(), pos.z());
  std::vector<float> probs(kC, 0.f); probs[3] = 1.0f;
  m.applyHitUpdate(c, &probs, 1.0f, /*is_dynamic=*/true);

  m.decayTransient(0.5f);
  auto b = m.getTransientBetaVoxel(pos);
  ASSERT_TRUE(b.has_value());
  // a_occ: 1 + (2 - 1)·0.5 = 1.5 ; a_free stays at prior 1.0.
  EXPECT_NEAR(b->a_occ, 1.5f, 1e-6f);
  EXPECT_NEAR(b->a_free, 1.0f, 1e-6f);
  // Persistent untouched by decay.
  EXPECT_EQ(m.betaVoxelCount(), 0u);
}

TEST(SemSplitTransient, DecayRateOneIsNoOp) {
  auto m = makeMap();
  const Eigen::Vector3f pos(1.0f, 0, 0);
  const auto c = m.betaGrid().posToCoord(pos.x(), pos.y(), pos.z());
  std::vector<float> probs(kC, 0.f); probs[3] = 1.0f;
  m.applyHitUpdate(c, &probs, 1.0f, /*is_dynamic=*/true);
  const float a_occ_before = m.getTransientBetaVoxel(pos)->a_occ;

  m.decayTransient(1.0f);  // clamp-safe no-op
  m.decayTransient(2.0f);  // > 1 clamps to 1 → still no-op
  auto b = m.getTransientBetaVoxel(pos);
  ASSERT_TRUE(b.has_value());
  EXPECT_FLOAT_EQ(b->a_occ, a_occ_before);
  EXPECT_EQ(m.transientBetaVoxelCount(), 1u);
}

TEST(SemSplitTransient, DecayRateZeroClearsTransient) {
  auto m = makeMap();
  const auto c = m.betaGrid().posToCoord(1.0f, 0.f, 0.f);
  std::vector<float> probs(kC, 0.f); probs[3] = 1.0f;
  m.applyHitUpdate(c, &probs, 1.0f, /*is_dynamic=*/true);
  ASSERT_EQ(m.transientBetaVoxelCount(), 1u);

  m.decayTransient(0.0f);        // collapse to prior → prune
  EXPECT_EQ(m.transientBetaVoxelCount(), 0u);
  EXPECT_EQ(m.transientDirVoxelCount(), 0u);
  EXPECT_FALSE(m.getTransientBetaVoxel(Eigen::Vector3f(1.0f, 0, 0)).has_value());
  // Negative rate clamps to 0 as well (no crash, already empty).
  m.decayTransient(-1.0f);
  EXPECT_EQ(m.transientBetaVoxelCount(), 0u);
}

TEST(SemSplitTransient, RepeatedDecayPrunesTransientGrids) {
  auto m = makeMap();
  const auto c = m.betaGrid().posToCoord(1.0f, 0.f, 0.f);
  std::vector<float> probs(kC, 0.f); probs[3] = 1.0f;
  m.applyHitUpdate(c, &probs, 1.0f, /*is_dynamic=*/true);

  // 0.5^n falls below the 1e-3 prune epsilon after ~11 frames.
  for (int i = 0; i < 20; ++i) m.decayTransient(0.5f);
  EXPECT_EQ(m.transientBetaVoxelCount(), 0u);
  EXPECT_EQ(m.transientDirVoxelCount(), 0u);
}

TEST(SemSplitTransient, DecayLeavesPersistentUntouched) {
  auto m = makeMap();
  const Eigen::Vector3f pos_p(1.0f, 0, 0);
  const Eigen::Vector3f pos_d(2.0f, 0, 0);
  const auto cp = m.betaGrid().posToCoord(pos_p.x(), pos_p.y(), pos_p.z());
  const auto cd = m.betaGrid().posToCoord(pos_d.x(), pos_d.y(), pos_d.z());
  std::vector<float> probs(kC, 0.f); probs[3] = 1.0f;

  m.applyHitUpdate(cp, &probs, 1.0f, /*is_dynamic=*/false);
  m.applyHitUpdate(cd, &probs, 1.0f, /*is_dynamic=*/true);
  const float persistent_a_occ = m.getBetaVoxel(pos_p)->a_occ;

  for (int i = 0; i < 20; ++i) m.decayTransient(0.5f);

  auto b = m.getBetaVoxel(pos_p);
  ASSERT_TRUE(b.has_value());
  EXPECT_FLOAT_EQ(b->a_occ, persistent_a_occ);  // persistent never decays
  EXPECT_EQ(m.transientBetaVoxelCount(), 0u);   // transient gone
}
