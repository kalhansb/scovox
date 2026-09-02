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
  // 8 B at float storage, 4 B under SCOVOX_BETA_U16; written from the counter
  // width so either build is checked rather than skipped.
  EXPECT_EQ(sizeof(scovox::BetaVoxel), 2u * sizeof(scovox::BetaCount));
  EXPECT_EQ(sizeof(scovox::BetaCount), SCOVOX_BETA_U16 ? 2u : 4u);
  // DirVoxel is 4 B fixed + 6 B per slot, plus 2 B per slot for each optional
  // track. SCOVOX_TRACK_QMAX is on by default (evict_by_confidence reads it),
  // so the shipped K_TOP=2 voxel is 20 B; the expression is written out so a
  // build that turns a track off is still checked rather than skipped.
  constexpr std::size_t slot = 6u + (SCOVOX_TRACK_QMAX ? 2u : 0u)
                                  + (SCOVOX_TRACK_NHIT ? 2u : 0u);
  EXPECT_EQ(sizeof(scovox::DirVoxel), ((4u + slot * scovox::K_TOP + 3u) / 4u) * 4u);
}

TEST(SplitVoxelLayout, ShippedBetaPriorIsSymmetricHalf) {
  // Shipped split-path occupancy prior is symmetric Beta(1,1) → p_occ = 0.5
  // (docs/occupancy_prior.md), decoupled from the semantic (C, α₀).
  auto b = scovox::defaultBetaVoxel(scovox::kBetaOccPrior, scovox::kBetaFreePrior);
  EXPECT_NEAR(b.a_occ,  1.0f, 1e-7f);
  EXPECT_NEAR(b.a_free, 1.0f, 1e-7f);
  EXPECT_NEAR(b.p_occ(), 0.5f, 1e-6f);
  // Ablation: the prior-agnostic factory still reproduces the calibrated
  // Beta(C·α₀, α₀) → C/(C+1) marginal on explicit request. Its parameters are
  // α₀-scale (0.14 / 0.01), and no fixed-point lattice can hold both those and
  // the thousands of units a long run accumulates — that spread needs a
  // dynamic range of ~2e5 against uint16's 65535 — so under SCOVOX_BETA_U16
  // the ablation is unavailable and the shipped Beta(1,1) prior above, which
  // sits exactly on the lattice, is the only one offered.
  if constexpr (SCOVOX_BETA_U16 == 0) {
    auto calib = scovox::defaultBetaVoxel(kC * kAlpha, kAlpha);
    EXPECT_NEAR(calib.p_occ(), float(kC) / float(kC + 1), 1e-5f);
  }
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

// ---------------------------------------------------------------------------
// Beta counter storage (BetaCount / SCOVOX_BETA_U16)
// ---------------------------------------------------------------------------

// Runs in BOTH builds. Every occupancy weight this repo ships sits on the 1/8
// lattice, so a long accumulation must land on the exact analytic total under
// fixed-point storage just as it does under float -- this is the invariant
// SCOVOX_BETA_U16's scale is chosen to hold, so it is asserted unconditionally
// rather than behind the flag.
TEST(BetaCountStorage, ShippedWeightsAccumulateExactly) {
  for (const float w : {1.0f, 1.5f, 6.0f, 0.5f, 0.125f}) {
    scovox::BetaVoxel b = scovox::defaultBetaVoxel();
    for (int i = 0; i < 200; ++i) b.a_occ += w;
    EXPECT_FLOAT_EQ(b.a_occ, 1.0f + 200.0f * w) << "w = " << w;
  }
}

TEST(BetaCountStorage, PriorAndPOccRoundTrip) {
  scovox::BetaVoxel b = scovox::defaultBetaVoxel(scovox::kBetaOccPrior,
                                                 scovox::kBetaFreePrior);
  EXPECT_FLOAT_EQ(b.a_occ, 1.0f);
  EXPECT_FLOAT_EQ(b.a_free, 1.0f);
  EXPECT_FLOAT_EQ(b.p_occ(), 0.5f);
  b.a_occ += 3.0f;                       // Beta(4, 1)
  EXPECT_FLOAT_EQ(b.s_total(), 5.0f);
  EXPECT_FLOAT_EQ(b.p_occ(), 0.8f);
  b.a_occ  *= 0.5f;                      // the saturation rescale's operation
  b.a_free *= 0.5f;
  EXPECT_FLOAT_EQ(b.p_occ(), 0.8f);      // p_occ is preserved
  EXPECT_FLOAT_EQ(b.s_total(), 2.5f);
}

#if SCOVOX_BETA_U16
TEST(BetaCountStorage, StoresClampInsteadOfWrapping) {
  scovox::BetaVoxel b{};
  b.a_occ = 4.0f * scovox::BetaCount::kMax;
  EXPECT_FLOAT_EQ(b.a_occ, scovox::BetaCount::kMax);
  b.a_occ += 1000.0f;                    // already at the ceiling
  EXPECT_FLOAT_EQ(b.a_occ, scovox::BetaCount::kMax);
  b.a_free = -5.0f;                      // Beta parameters are non-negative
  EXPECT_FLOAT_EQ(b.a_free, 0.0f);
}

TEST(BetaCountStorage, IncrementsBelowHalfACountVanish) {
  // Documented cost of fixed point: a ray weaker than half the stored
  // resolution rounds back to where it started.
  const float res = scovox::BetaCount::kInv;
  scovox::BetaVoxel b = scovox::defaultBetaVoxel();
  for (int i = 0; i < 50; ++i) b.a_occ += 0.4f * res;
  EXPECT_FLOAT_EQ(b.a_occ, 1.0f);
  for (int i = 0; i < 4; ++i) b.a_occ += 0.6f * res;   // rounds up each time
  EXPECT_FLOAT_EQ(b.a_occ, 1.0f + 4.0f * res);
}

TEST(BetaCountStorage, SaturationGuardKeepsCountersOffTheCeiling) {
  // applyBetaSaturation's u16 guard is unconditional -- evidence_saturation is
  // 0 here -- so a voxel observed far past the storage ceiling keeps taking
  // evidence instead of pinning at BetaCount::kMax and going deaf. Halving is
  // p_occ-preserving; that property is pinned by PriorAndPOccRoundTrip.
  scovox::SemSplitMap::Params p;
  p.resolution          = kRes;
  p.w_occ               = 500.0f;   // ~16 hits to cross the high-water mark
  p.w_free              = 0.5f;
  p.kappa0              = 1.0f;
  p.dirichlet_min_p_occ = 0.5f;
  p.evidence_saturation = 0.0f;
  p.num_classes         = kC;
  p.alpha_0             = kAlpha;
  scovox::SemSplitMap m(p);

  const Eigen::Vector3f pt(1.0f, 0, 0);
  std::vector<float> probs(kC, 0.f);
  probs[5] = 1.0f;
  for (int i = 0; i < 60; ++i) {          // 30,001 units deposited, ceiling 8191.9
    m.integrateHit(pt, pt, &probs, /*quality=*/1.0f);
    auto b = m.getBetaVoxel(pt);
    ASSERT_TRUE(b.has_value());
    EXPECT_LE(b->a_occ, 0.9f * scovox::BetaCount::kMax) << "hit " << i;
    EXPECT_GT(b->a_occ, 0.0f) << "hit " << i;
  }
}

#endif  // SCOVOX_BETA_U16

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

// Plant a confidently-occupied "wall" voxel at `pos` in map `m`.
static void plantWall(scovox::SemSplitMap& m, const Eigen::Vector3f& pos) {
  auto acc   = m.betaGrid().createAccessor();
  auto coord = m.betaGrid().posToCoord(pos.x(), pos.y(), pos.z());
  auto pre   = m.defaultBeta();
  pre.a_occ  = 100.f;  // p_occ ≈ 1 → wall
  pre.a_free = 1.f;
  acc.setValue(coord, pre);
}

TEST(SemSplitMap, WallGuardOptInStopsCarving) {
  // The wall guard is OFF by default (trust the recent scan). A positive
  // carve_skip_occ_threshold re-enables the legacy occupancy-blocked carve, but
  // only on the IMMEDIATE (unbatched) path — for offline tools / ablations.
  scovox::SemSplitMap::Params p;
  p.resolution = kRes; p.w_occ = 1.0f; p.w_free = 0.5f; p.kappa0 = 1.0f;
  p.dirichlet_min_p_occ = 0.5f; p.num_classes = kC; p.alpha_0 = kAlpha;
  p.carve_skip_occ_threshold = 0.7f;  // guard ON
  scovox::SemSplitMap m(p);

  plantWall(m, Eigen::Vector3f(0.5f, 0, 0));
  m.integrateHit(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(1.0f, 0, 0), nullptr, 1.0f);

  EXPECT_FALSE(m.getBetaVoxel(Eigen::Vector3f(0.75f, 0, 0)).has_value())
      << "guard on: carve stops at the wall — voxels past it are never touched";
  auto v_before = m.getBetaVoxel(Eigen::Vector3f(0.25f, 0, 0));
  ASSERT_TRUE(v_before.has_value());
  EXPECT_GT(v_before->a_free, scovox::kBetaFreePrior)
      << "voxels before the wall are still carved";
}

TEST(SemSplitMap, TrustRecentScanCarvesThroughStaleObstacleByDefault) {
  // Default (guard off): a beam that reached its return proves the whole segment
  // is free NOW, so a stale/moved obstacle in its path receives free evidence
  // (clears over time) and voxels past it are carved. Immediate path.
  auto m = makeMap();  // guard off (default)
  plantWall(m, Eigen::Vector3f(0.5f, 0, 0));
  m.integrateHit(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(1.0f, 0, 0), nullptr, 1.0f);

  auto obst = m.getBetaVoxel(Eigen::Vector3f(0.5f, 0, 0));
  ASSERT_TRUE(obst.has_value());
  EXPECT_GT(obst->a_free, 1.0f) << "stale obstacle gets free evidence (no guard)";
  auto v_past = m.getBetaVoxel(Eigen::Vector3f(0.75f, 0, 0));
  ASSERT_TRUE(v_past.has_value()) << "voxels past the stale obstacle are carved";
  EXPECT_GT(v_past->a_free, scovox::kBetaFreePrior);
}

// ===========================================================================
// Batched per-scan carve (beginCarveFrame / flushCarveFrame)
// ===========================================================================

TEST(SemSplitMapBatched, WritesOncePerVoxelAcrossRays) {
  // Full-ray carving crossed by many rays writes each shared voxel ONCE per
  // scan (max evidence), not once per ray — the fast full-ray carve.
  auto m = makeMap();  // w_free = 0.5
  m.beginCarveFrame();
  for (int i = 0; i < 5; ++i)
    m.integrateHit(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(1.0f, 0, 0), nullptr, 1.0f);
  const std::size_t written = m.flushCarveFrame();
  EXPECT_GT(written, 0u);

  auto b = m.getBetaVoxel(Eigen::Vector3f(0.5f, 0, 0));  // interior, crossed by all 5
  ASSERT_TRUE(b.has_value());
  // One w_free*quality (=0.5) applied, not 5× (which would give prior + 2.5).
  EXPECT_NEAR(b->a_free, scovox::kBetaFreePrior + 0.5f, 1e-5f);
}

TEST(SemSplitMapBatched, OccupiedWinsSkipsSameScanHitVoxel) {
  // A voxel that is a surface return in this scan must not be carved free, even
  // if another ray grazes through it (occupied-wins).
  auto m = makeMap();
  const Eigen::Vector3f X(0.5f, 0, 0);
  m.beginCarveFrame();
  m.integrateHit(Eigen::Vector3f(0, 0, 0), X, nullptr, 1.0f);                      // hit at X
  m.integrateHit(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(1.0f, 0, 0), nullptr, 1.0f);  // grazes X
  m.flushCarveFrame();

  auto b = m.getBetaVoxel(X);
  ASSERT_TRUE(b.has_value());
  EXPECT_NEAR(b->a_free, scovox::kBetaFreePrior, 1e-5f)
      << "same-scan surface voxel is not carved free (occupied-wins)";
  EXPECT_GT(b->a_occ, scovox::kBetaOccPrior) << "X still accrued its hit occupancy";
}

TEST(SemSplitMapBatched, ClearsStaleObstacleNoGuard) {
  // Batched path is guard-free: a stale obstacle a beam passes through receives
  // free evidence and clears over scans.
  auto m = makeMap();
  plantWall(m, Eigen::Vector3f(0.5f, 0, 0));
  m.beginCarveFrame();
  m.integrateHit(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(1.0f, 0, 0), nullptr, 1.0f);
  m.flushCarveFrame();

  auto b = m.getBetaVoxel(Eigen::Vector3f(0.5f, 0, 0));
  ASSERT_TRUE(b.has_value());
  EXPECT_GT(b->a_free, 1.0f) << "stale obstacle receives free evidence (guard-free)";
}

// ===========================================================================
// Batched surface hits (SemSplitParams::batch_hits)
// ===========================================================================

TEST(SemSplitMapBatched, HitsCountOncePerFrame) {
  // A depth image lands many rays in one surface voxel. Batched, the frame is
  // ONE observation; un-batched, each ray is its own.
  const Eigen::Vector3f X(0.5f, 0, 0);
  const int kRays = 7;

  auto run = [&](bool batch) {
    scovox::SemSplitMap::Params p;
    p.resolution = kRes; p.w_occ = 1.0f; p.w_free = 0.5f; p.kappa0 = 1.0f;
    p.dirichlet_min_p_occ = 0.5f; p.num_classes = kC; p.alpha_0 = kAlpha;
    p.batch_hits = batch;
    scovox::SemSplitMap m(p);
    m.beginCarveFrame();
    for (int i = 0; i < kRays; ++i)
      m.integrateHit(Eigen::Vector3f(0, 0, 0), X, nullptr, 1.0f);
    m.flushCarveFrame();
    auto b = m.getBetaVoxel(X);
    EXPECT_TRUE(b.has_value());
    return b->a_occ;
  };

  EXPECT_FLOAT_EQ(run(true),  scovox::kBetaOccPrior + 1.0f)
      << "batched: one deposit for the whole frame";
  EXPECT_FLOAT_EQ(run(false), scovox::kBetaOccPrior + kRays * 1.0f)
      << "un-batched: a_occ counts rays, not observations";
}

TEST(SemSplitMapBatched, HitClassEvidenceAlsoCountsOncePerFrame) {
  // Stream B is staged with Stream A, so `cnt` is bounded by the frame count
  // too — the property the uint16 counter sizing rests on.
  const Eigen::Vector3f X(0.5f, 0, 0);
  std::vector<float> probs(kC, 0.f);
  probs[3] = 1.0f;

  auto run = [&](bool batch) {
    scovox::SemSplitMap::Params p;
    p.resolution = kRes; p.w_occ = 1.0f; p.w_free = 0.5f; p.kappa0 = 1.0f;
    p.dirichlet_min_p_occ = 0.5f; p.num_classes = kC; p.alpha_0 = kAlpha;
    p.batch_hits = batch;
    scovox::SemSplitMap m(p);
    m.beginCarveFrame();
    for (int i = 0; i < 5; ++i)
      m.integrateHit(Eigen::Vector3f(0, 0, 0), X, &probs, 1.0f);
    m.flushCarveFrame();
    auto d = m.getDirVoxel(X);
    EXPECT_TRUE(d.has_value());
    return d->cnt[0];
  };

  EXPECT_LT(run(true), run(false)) << "batched class evidence is the smaller";
  // One deposit is class_share = kappa0 * p_occ_post * quality on top of the
  // per-dim prior; five deposits must land strictly more than one.
  EXPECT_GT(run(true), kAlpha);
}

TEST(SemSplitMapBatched, StrongestRayOfTheFrameWins) {
  // The staged hit mirrors the carve stage's per-voxel MAX rule, so a weaker
  // ray arriving later must not displace a stronger one.
  const Eigen::Vector3f X(0.5f, 0, 0);
  auto run = [&](float q_first, float q_second) {
    auto m = makeMap();
    m.beginCarveFrame();
    m.integrateHit(Eigen::Vector3f(0, 0, 0), X, nullptr, q_first);
    m.integrateHit(Eigen::Vector3f(0, 0, 0), X, nullptr, q_second);
    m.flushCarveFrame();
    return m.getBetaVoxel(X)->a_occ;
  };
  const float strong = scovox::kBetaOccPrior + 1.0f;   // w_occ 1.0 x q 1.0
  EXPECT_FLOAT_EQ(run(1.0f, 0.25f), strong) << "weaker second ray is dropped";
  EXPECT_FLOAT_EQ(run(0.25f, 1.0f), strong) << "stronger second ray supersedes";
}

TEST(SemSplitMapBatched, StagedHitStillWinsAgainstTheCarve) {
  // Occupied-wins survives the deferral: the hit is written at flush BEFORE
  // the staged carves, and the carve walk still skips the voxel.
  auto m = makeMap();
  const Eigen::Vector3f X(0.5f, 0, 0);
  m.beginCarveFrame();
  m.integrateHit(Eigen::Vector3f(0, 0, 0), X, nullptr, 1.0f);
  // A second beam passing straight through the same voxel to a farther return.
  m.integrateHit(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(1.0f, 0, 0), nullptr, 1.0f);
  m.flushCarveFrame();

  auto b = m.getBetaVoxel(X);
  ASSERT_TRUE(b.has_value());
  EXPECT_FLOAT_EQ(b->a_free, scovox::kBetaFreePrior) << "no free evidence at a hit voxel";
  EXPECT_GT(b->a_occ, scovox::kBetaOccPrior);
}

TEST(SemSplitMapBatched, RaySpreadKeepsTheImmediateWrite) {
  // raySpreadDeposit reads the endpoint's Beta occupancy the instant Stream A
  // lands, so a spread-carrying hit must not be deferred.
  scovox::SemSplitMap::Params p;
  p.resolution = kRes; p.w_occ = 1.0f; p.w_free = 0.5f; p.kappa0 = 1.0f;
  p.dirichlet_min_p_occ = 0.5f; p.num_classes = kC; p.alpha_0 = kAlpha;
  p.ray_spread = 1;
  scovox::SemSplitMap m(p);

  const Eigen::Vector3f X(0.5f, 0, 0);
  m.beginCarveFrame();
  m.integrateHit(Eigen::Vector3f(0, 0, 0), X, nullptr, 1.0f);
  auto mid = m.getBetaVoxel(X);
  ASSERT_TRUE(mid.has_value()) << "spread-carrying hit writes during the ray loop";
  EXPECT_FLOAT_EQ(mid->a_occ, scovox::kBetaOccPrior + 1.0f);
  m.flushCarveFrame();
  EXPECT_FLOAT_EQ(m.getBetaVoxel(X)->a_occ, scovox::kBetaOccPrior + 1.0f)
      << "flush must not deposit a second time";
}

TEST(SemSplitMapBatched, ImmediateWhenNoFrameIsOpen) {
  // Batching is a per-scan mechanism; a direct caller with no frame open keeps
  // the write-in-place contract whatever the flag says.
  auto m = makeMap();
  const Eigen::Vector3f X(0.5f, 0, 0);
  m.integrateHit(Eigen::Vector3f(0, 0, 0), X, nullptr, 1.0f);
  auto b = m.getBetaVoxel(X);
  ASSERT_TRUE(b.has_value());
  EXPECT_FLOAT_EQ(b->a_occ, scovox::kBetaOccPrior + 1.0f);
}

TEST(SemSplitMapBatched, CanDisableFreeSpaceCarve) {
  // Turning off batched free-space carve suppresses staging entirely while
  // leaving hit updates intact.
  scovox::SemSplitMap::Params p;
  p.resolution = kRes; p.w_occ = 1.0f; p.w_free = 0.5f; p.kappa0 = 1.0f;
  p.dirichlet_min_p_occ = 0.5f; p.num_classes = kC; p.alpha_0 = kAlpha;
  p.batch_free_carve = false;
  scovox::SemSplitMap m(p);

  const Eigen::Vector3f X(1.0f, 0, 0);
  m.beginCarveFrame();
  m.integrateHit(Eigen::Vector3f(0, 0, 0), X, nullptr, 1.0f);
  EXPECT_EQ(m.flushCarveFrame(), 0u) << "batch-free carve disabled: nothing stages";

  EXPECT_FALSE(m.getBetaVoxel(Eigen::Vector3f(0.5f, 0, 0)).has_value())
      << "mid-ray free-space voxel stays untouched";
  auto hit = m.getBetaVoxel(X);
  ASSERT_TRUE(hit.has_value()) << "hit voxel still updates";
  EXPECT_GT(hit->a_occ, scovox::kBetaOccPrior);
}

TEST(SemSplitMapBatched, DynamicHitDoesNotSuppressPersistentCarve) {
  // Occupied-wins is PERSISTENT-only: a dynamic endpoint routes occupancy to the
  // transient grid, so it must not block another ray's persistent free carve of
  // that voxel (the is_dynamic contract — persistent grid stays free there).
  auto m = makeMap();
  const Eigen::Vector3f X(0.5f, 0, 0);
  std::vector<float> probs(kC, 0.f); probs[3] = 1.0f;

  m.beginCarveFrame();
  m.integrateHit(Eigen::Vector3f(0, 0, 0), X, &probs, 1.0f, /*is_dynamic=*/true);       // dynamic hit at X
  m.integrateHit(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(1.0f, 0, 0), nullptr, 1.0f); // persistent carve through X
  m.flushCarveFrame();

  // Persistent free carve at X is NOT suppressed by the same-scan dynamic hit.
  auto b = m.getBetaVoxel(X);
  ASSERT_TRUE(b.has_value())
      << "persistent free carve must land at a dynamic-endpoint voxel";
  EXPECT_GT(b->a_free, scovox::kBetaFreePrior);
  // The dynamic occupancy lives only in the transient grid.
  auto t = m.getTransientBetaVoxel(X);
  ASSERT_TRUE(t.has_value());
  EXPECT_GT(t->a_occ, scovox::kBetaOccPrior);
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

// ===========================================================================
// E6 precondition — `inc_mode` conserves mass on all three paths
//
// NEW_EXPERIMENTS_PLAN.md E6 sweeps `inc_mode` as a candidate lever.  The
// header claims all three modes conserve the strict invariant
// `Δ(other + Σcnt) == class_share` (sem_split_map.hpp:194-195) and differ only
// in how that mass is split.  Before the sweep prices the modes against each
// other, that claim is checked rather than trusted: a mode that quietly LOSES
// mass would show up in the sweep as a semantic effect, and E6 would promote a
// leak.
//
// SOFT was already pinned (SemSplitMap.MassConservationPerGrid, one-hot,
// single hit).  HARD and THRESH were not covered at all, and neither was any
// mode on a softmax that is not one-hot -- which is the only input on which
// the three modes take different branches.
// ===========================================================================

namespace {

scovox::SemSplitMap makeMapMode(int inc_mode, float inc_thresh) {
  scovox::SemSplitMap::Params p;
  p.resolution          = kRes;
  p.w_occ               = 1.0f;
  p.w_free              = 0.5f;
  p.kappa0              = 1.0f;
  p.dirichlet_min_p_occ = 0.5f;
  p.evidence_saturation = 0.0f;
  p.num_classes         = kC;
  p.alpha_0             = kAlpha;
  p.inc_mode            = inc_mode;
  p.inc_thresh          = inc_thresh;
  return scovox::SemSplitMap(p);
}

// Neither one-hot nor uniform, and straddling inc_thresh = 0.10 so the three
// modes provably diverge: SOFT deposits four classes, THRESH three (class 9 at
// 0.08 falls below), HARD one (the argmax).  Four distinct classes against
// K_TOP = 2 also forces the evict/drop routing, which is where mass is most
// at risk of going missing.
std::vector<float> mixedSoftmax() {
  std::vector<float> p(kC, 0.f);
  p[1] = 0.50f;
  p[2] = 0.30f;
  p[7] = 0.12f;
  p[9] = 0.08f;
  return p;
}

// Deposit `n` hits and return the TOTAL `class_share` injected.  class_share
// is `kappa0 · p_occ_post · quality` and p_occ_post climbs with every hit, so
// it must be accumulated per hit rather than multiplied out.
float injectAccumulatingShare(scovox::SemSplitMap& m,
                              const Eigen::Vector3f& pos,
                              const std::vector<float>* probs,
                              float quality, int n) {
  const auto coord = m.betaGrid().posToCoord(pos.x(), pos.y(), pos.z());
  float total = 0.f;
  for (int i = 0; i < n; ++i) {
    m.applyHitUpdate(coord, probs, quality);
    auto b = m.getBetaVoxel(pos);
    EXPECT_TRUE(b.has_value());
    // Below the gate Stream B deposits nothing, so counting a share there
    // would make the test wrong rather than the code.
    EXPECT_GE(b->p_occ(), 0.5f) << "hit " << i << " fell below the deposit gate";
    total += 1.0f * b->p_occ() * quality;
  }
  return total;
}

}  // namespace

TEST(IncMode, AllThreeModesConserveDirMassUnderEviction) {
  const Eigen::Vector3f pos(2.0f, 0.f, 0.f);
  const auto probs = mixedSoftmax();

  for (const auto& mode : {std::make_pair(0, "SOFT"),
                           std::make_pair(1, "HARD"),
                           std::make_pair(2, "THRESH")}) {
    auto  m     = makeMapMode(mode.first, 0.10f);
    const float share = injectAccumulatingShare(m, pos, &probs, 1.0f, 6);

    auto d = m.getDirVoxel(pos);
    ASSERT_TRUE(d.has_value()) << mode.second;
    // The whole invariant, exactly: prior + everything injected, nothing lost
    // to a `continue` and nothing conjured by a rescale.
    EXPECT_NEAR(d->s_class(), kC * kAlpha + share, 1e-4f)
        << "inc_mode " << mode.second << " does not conserve Dir mass";
    // And no slot may go negative on the way -- a clamped-at-zero eviction is
    // still conserving only if the displaced mass reached OTHER.
    for (int i = 0; i < scovox::K_TOP; ++i)
      EXPECT_GE(d->cnt[i], 0.f) << mode.second << " slot " << i;
    EXPECT_GE(d->other, 0.f) << mode.second;
  }
}

TEST(IncMode, ThreshRoutesEverySubThresholdClassToOther) {
  // inc_thresh above every class probability: the loop deposits nothing at
  // all, and the residual line must send the ENTIRE class_share to OTHER.
  // This is the branch where a `continue` most plausibly drops mass, because
  // `deposited` stays 0 and the correction is a single subtraction.
  const Eigen::Vector3f pos(3.0f, 0.f, 0.f);
  const auto probs = mixedSoftmax();          // max is 0.50
  auto  m     = makeMapMode(2, 0.90f);
  const float share = injectAccumulatingShare(m, pos, &probs, 1.0f, 4);

  auto d = m.getDirVoxel(pos);
  ASSERT_TRUE(d.has_value());
  EXPECT_NEAR(d->s_class(), kC * kAlpha + share, 1e-4f)
      << "THRESH lost mass when every class was below threshold";
  EXPECT_NEAR(d->other, (kC - scovox::K_TOP) * kAlpha + share, 1e-4f)
      << "sub-threshold mass must land in OTHER, not vanish";
  for (int i = 0; i < scovox::K_TOP; ++i)
    EXPECT_EQ(d->cls[i], uint16_t(0xFFFF)) << "no slot may be filled";
}

TEST(IncMode, TheThreeModesAreNotTheSameProgram) {
  // Without this, the conservation test above could be passing on three
  // identical execution paths -- the same failure `build_set.sh`'s md5
  // uniqueness check exists to catch, in runtime clothing.  A conservation
  // proof over one path restated three times is not a proof over three paths.
  const Eigen::Vector3f pos(4.0f, 0.f, 0.f);
  const auto probs = mixedSoftmax();

  auto slots = [&](int mode) {
    auto m = makeMapMode(mode, 0.10f);
    injectAccumulatingShare(m, pos, &probs, 1.0f, 3);
    auto d = m.getDirVoxel(pos);
    EXPECT_TRUE(d.has_value());
    // OTHER separates the modes even when the surviving slot labels agree.
    return std::make_pair(std::make_pair(d->cls[0], d->cls[1]), d->other);
  };

  const auto soft   = slots(0);
  const auto hard   = slots(1);

  // HARD gives all mass to the argmax, so exactly one slot is ever filled.
  EXPECT_EQ(hard.first.second, uint16_t(0xFFFF))
      << "HARD must collapse to a single class";
  EXPECT_NE(soft.first.second, uint16_t(0xFFFF))
      << "SOFT must populate both slots from this softmax";
  EXPECT_NE(hard.second, soft.second)
      << "HARD and SOFT must route different amounts to OTHER";
}

// Only class 1 clears inc_thresh = 0.10, so under THRESH the second slot is
// STARVED -- which is the regime in which THRESH is not merely SOFT under
// another name.  Contrast with the test below.
namespace {
std::vector<float> starvingSoftmax() {
  std::vector<float> p(kC, 0.f);
  p[1] = 0.85f;
  p[9] = 0.08f;
  p[7] = 0.07f;
  return p;
}
}  // namespace

TEST(IncMode, ThreshDivergesFromSoftOnlyWhenItStarvesASlot) {
  const Eigen::Vector3f pos(5.0f, 0.f, 0.f);
  const auto probs = starvingSoftmax();

  auto run = [&](int mode) {
    auto m = makeMapMode(mode, 0.10f);
    injectAccumulatingShare(m, pos, &probs, 1.0f, 3);
    auto d = m.getDirVoxel(pos);
    EXPECT_TRUE(d.has_value());
    return *d;
  };

  const auto soft   = run(0);
  const auto thresh = run(2);

  // SOFT lets class 9 (0.08) take the free second slot; THRESH never deposits
  // it, so the slot stays at its empty sentinel and the mass lands in OTHER.
  EXPECT_NE(soft.cls[1], uint16_t(0xFFFF))
      << "SOFT should fill the second slot with the sub-threshold class";
  EXPECT_EQ(thresh.cls[1], uint16_t(0xFFFF))
      << "THRESH must leave the second slot empty";
  EXPECT_GT(thresh.other, soft.other)
      << "the starved class's mass must show up in OTHER";
}

TEST(IncMode, ThreshEqualsSoftWhenTheSubThresholdClassWouldBeDroppedAnyway) {
  // NOT a redundant restatement of the test above -- it pins the OPPOSITE
  // regime, and it is the one E6 should expect to be in.
  //
  // On mixedSoftmax() four classes compete for K_TOP = 2 slots.  Classes 7
  // (0.12) and 9 (0.08) lose to 1 and 2 under SOFT and are dropped to OTHER by
  // the admit comparator; under THRESH class 9 is refused entry and its mass
  // is routed to OTHER by the residual line.  Different code paths, same
  // destination, same amount -- so the two modes leave the voxel in an
  // IDENTICAL state.
  //
  // The consequence for E6 is a prediction, made here rather than discovered
  // in the sweep: `inc_thresh` can only matter when it starves a slot that
  // would otherwise have been filled.  At K_TOP = 2 the slots are contested
  // almost everywhere (E0 measured 8.46% of argmax arrivals finding no slot at
  // all), so a THRESH level that does not reach the SECOND-strongest class's
  // probability is predicted inert -- and an "inert" verdict for it is then a
  // statement about the level, not about the lever.
  const Eigen::Vector3f pos(6.0f, 0.f, 0.f);
  const auto probs = mixedSoftmax();

  auto run = [&](int mode) {
    auto m = makeMapMode(mode, 0.10f);
    injectAccumulatingShare(m, pos, &probs, 1.0f, 3);
    auto d = m.getDirVoxel(pos);
    EXPECT_TRUE(d.has_value());
    return *d;
  };

  const auto soft   = run(0);
  const auto thresh = run(2);

  EXPECT_EQ(thresh.cls[0], soft.cls[0]);
  EXPECT_EQ(thresh.cls[1], soft.cls[1]);
  EXPECT_NEAR(thresh.other,  soft.other,  1e-6f);
  EXPECT_NEAR(thresh.cnt[0], soft.cnt[0], 1e-6f);
  EXPECT_NEAR(thresh.cnt[1], soft.cnt[1], 1e-6f);
}
