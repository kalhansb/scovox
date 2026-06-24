/// @file
/// @brief v4 consensus merge: independent BetaVoxel (occupancy) + DirVoxel
/// (semantics) merges, each conjugate and mass-conserving.

#include <gtest/gtest.h>

#include <algorithm>
#include <bonxai/bonxai.hpp>
#include <stdexcept>
#include <vector>

#include "scovox/consensus_merge_v4.hpp"

namespace {

constexpr float    kAlpha = scovox::kDefaultDirichletPrior;  // 0.01
constexpr uint16_t kC     = 14;

scovox::BetaVoxel betaPrior() {
  // Shipped occupancy prior: symmetric Beta(1,1) → p_occ=0.5 (docs/occupancy_prior.md).
  return scovox::defaultBetaVoxel(scovox::kBetaOccPrior, scovox::kBetaFreePrior);
}
scovox::DirVoxel dirPrior() {
  return scovox::defaultDirVoxel(kC, kAlpha);
}

}  // namespace

// ===========================================================================
// BetaVoxel merge (conjugate Beta)
// ===========================================================================

TEST(ConsensusMergeV4, BetaMergeWithPriorPreservesEvidence) {
  scovox::BetaVoxel v{2.0f, 3.0f};      // both dims strictly above the prior (1,1)
  auto f = scovox::mergeBeta(v, betaPrior(), kC, kAlpha);
  EXPECT_FLOAT_EQ(f.a_occ,  v.a_occ);   // duplicated prior cancels
  EXPECT_FLOAT_EQ(f.a_free, v.a_free);
}

TEST(ConsensusMergeV4, BetaMergeOfTwoPriorsIsPrior) {
  auto f = scovox::mergeBeta(betaPrior(), betaPrior(), kC, kAlpha);
  EXPECT_FLOAT_EQ(f.a_occ,  scovox::kBetaOccPrior);   // two priors fold back to the prior
  EXPECT_FLOAT_EQ(f.a_free, scovox::kBetaFreePrior);
  EXPECT_NEAR(f.p_occ(), 0.5f, 1e-6f);                // symmetric prior → p_occ stays 0.5
}

TEST(ConsensusMergeV4, BetaMergeIsSymmetricAndAdditive) {
  scovox::BetaVoxel a{3.0f, 1.0f};
  scovox::BetaVoxel b{1.0f, 4.0f};
  auto ab = scovox::mergeBeta(a, b, kC, kAlpha);
  auto ba = scovox::mergeBeta(b, a, kC, kAlpha);
  EXPECT_FLOAT_EQ(ab.a_occ,  ba.a_occ);
  EXPECT_FLOAT_EQ(ab.a_free, ba.a_free);
  // a_occ = 3 + 1 − occ_prior ; a_free = 1 + 4 − free_prior (symmetric Beta(1,1)).
  EXPECT_FLOAT_EQ(ab.a_occ,  4.0f - scovox::kBetaOccPrior);
  EXPECT_FLOAT_EQ(ab.a_free, 5.0f - scovox::kBetaFreePrior);
}

TEST(ConsensusMergeV4, BetaMassConservation) {
  // Δ(a_occ + a_free) = a.total + b.total − (occ_prior + free_prior) (one prior removed).
  scovox::BetaVoxel a{2.5f, 1.5f};
  scovox::BetaVoxel b{0.5f, 3.0f};
  auto f = scovox::mergeBeta(a, b, kC, kAlpha);
  const float beta_prior_total = scovox::kBetaOccPrior + scovox::kBetaFreePrior;
  EXPECT_FLOAT_EQ(f.s_total(), a.s_total() + b.s_total() - beta_prior_total);
}

// ===========================================================================
// DirVoxel merge (slot reconciliation, no FREE dimension)
// ===========================================================================

TEST(ConsensusMergeV4, DirMergeWithPriorPreservesEvidence) {
  auto v = dirPrior();
  v.other = 0.30f;
  v.cls[0] = 7; v.cnt[0] = 1.5f;
  v.cls[1] = 3; v.cnt[1] = 0.8f;
  auto f = scovox::mergeDir(v, dirPrior(), kC, kAlpha);
  EXPECT_FLOAT_EQ(f.other, v.other);
  EXPECT_EQ(f.cls[0], v.cls[0]); EXPECT_FLOAT_EQ(f.cnt[0], v.cnt[0]);
  EXPECT_EQ(f.cls[1], v.cls[1]); EXPECT_FLOAT_EQ(f.cnt[1], v.cnt[1]);
}

TEST(ConsensusMergeV4, DirCoincidingClassSumsObservedEvidence) {
  auto a = dirPrior(); a.cls[0] = 5; a.cnt[0] = kAlpha + 1.0f;
  auto b = dirPrior(); b.cls[0] = 5; b.cnt[0] = kAlpha + 2.0f;
  auto f = scovox::mergeDir(a, b, kC, kAlpha);
  EXPECT_EQ(f.cls[0], uint16_t(5));
  EXPECT_NEAR(f.cnt[0], kAlpha + 3.0f, 1e-5f);  // one duplicated prior removed
}

TEST(ConsensusMergeV4, DirEvictionRoutesToOther) {
  auto a = dirPrior(); a.cls[0] = 1; a.cnt[0] = kAlpha + 5.0f; a.cls[1] = 2; a.cnt[1] = kAlpha + 4.0f;
  auto b = dirPrior(); b.cls[0] = 3; b.cnt[0] = kAlpha + 3.0f; b.cls[1] = 4; b.cnt[1] = kAlpha + 2.0f;
  auto f = scovox::mergeDir(a, b, kC, kAlpha);
  std::vector<uint16_t> kept;
  for (int i = 0; i < scovox::K_TOP; ++i) if (f.cls[i] != 0xFFFF) kept.push_back(f.cls[i]);
  std::sort(kept.begin(), kept.end());
  ASSERT_EQ(kept.size(), 2u);
  EXPECT_EQ(kept[0], uint16_t(1));
  EXPECT_EQ(kept[1], uint16_t(2));
  // OTHER = merged prior (C−K)·α₀ + evicted evidence (3 + 2).
  EXPECT_NEAR(f.other, (kC - scovox::K_TOP) * kAlpha + 3.0f + 2.0f, 1e-5f);
}

TEST(ConsensusMergeV4, DirMassConservation) {
  // Δ(other + Σcnt) = a.s_class + b.s_class − C·α₀ (one prior removed), holds
  // through eviction (evicted mass routes to OTHER, never lost).
  auto a = dirPrior(); a.cls[0] = 1; a.cnt[0] = kAlpha + 5.0f; a.cls[1] = 2; a.cnt[1] = kAlpha + 4.0f;
  auto b = dirPrior(); b.cls[0] = 3; b.cnt[0] = kAlpha + 3.0f; b.cls[1] = 4; b.cnt[1] = kAlpha + 2.0f;
  auto f = scovox::mergeDir(a, b, kC, kAlpha);
  const float dir_prior_total = kC * kAlpha;
  EXPECT_NEAR(f.s_class(), a.s_class() + b.s_class() - dir_prior_total, 1e-5f);
}

// ===========================================================================
// num_classes ≤ K_TOP edge (residual_dims ≤ 0). The OTHER prior is
// (num_classes − K_TOP)·α₀, which is zero at num_classes==K_TOP and would go
// NEGATIVE at num_classes<K_TOP. defaultDirVoxel / mergeDir both clamp it at 0
// so the prior subtraction in mergeDir can never become prior INFLATION. These
// pin that clamp on the reachable scovox_core path (the receiver-side
// projectBetaDirToVoxel / isPriorDir helpers live in the mapping node and are
// covered by the explicit mergeFramesV4 reject below + node tests).
// ===========================================================================

TEST(ConsensusMergeV4, DefaultDirVoxelClampsOtherPriorAtKTop) {
  // num_classes == K_TOP: residual_dims == 0 → OTHER prior is exactly 0, not α₀.
  auto v2 = scovox::defaultDirVoxel(scovox::K_TOP, kAlpha);
  EXPECT_FLOAT_EQ(v2.other, 0.f);
  for (int i = 0; i < scovox::K_TOP; ++i) {
    EXPECT_FLOAT_EQ(v2.cnt[i], kAlpha);          // per-slot prior still α₀
    EXPECT_EQ(v2.cls[i], uint16_t(0xFFFF));      // empty sentinels
  }
  // class prior total collapses to K_TOP·α₀ (the slots), OTHER contributes none.
  EXPECT_FLOAT_EQ(v2.s_class(), scovox::K_TOP * kAlpha);
}

TEST(ConsensusMergeV4, DefaultDirVoxelClampsNegativeResidualToZero) {
  // num_classes == 1 < K_TOP: residual_dims == −1 → without the clamp OTHER
  // would be −α₀; defaultDirVoxel must floor it at 0.
  auto v1 = scovox::defaultDirVoxel(/*num_classes=*/1, kAlpha);
  EXPECT_FLOAT_EQ(v1.other, 0.f);
  for (int i = 0; i < scovox::K_TOP; ++i) EXPECT_FLOAT_EQ(v1.cnt[i], kAlpha);
}

TEST(ConsensusMergeV4, DirMergeAtKTopDoesNotInflateOther) {
  // At num_classes==K_TOP the merged OTHER prior is 0, so merging two priors
  // must leave OTHER at 0 (no phantom (C−K)·α₀ reintroduced) — the regression
  // the clamp guards: an unclamped negative other_prior would ADD mass here.
  const uint16_t kc = scovox::K_TOP;
  auto a = scovox::defaultDirVoxel(kc, kAlpha);
  auto b = scovox::defaultDirVoxel(kc, kAlpha);
  auto f = scovox::mergeDir(a, b, kc, kAlpha);
  EXPECT_FLOAT_EQ(f.other, 0.f);
  // Mass conservation: Δ(other+Σcnt) == a.s_class + b.s_class − (K_TOP·α₀).
  EXPECT_NEAR(f.s_class(),
              a.s_class() + b.s_class() - kc * kAlpha, 1e-6f);
}

TEST(ConsensusMergeV4, DirMergeAtKTopEvictionRoutesRawEvidenceToOther) {
  // With C==K_TOP the OTHER prior is 0, so an evicted class dumps ONLY its raw
  // evidence (cnt − α₀) into OTHER — there is no (C−K)·α₀ prior to add on top.
  const uint16_t kc = scovox::K_TOP;  // 2 == K_TOP: all input slots full
  auto a = scovox::defaultDirVoxel(kc, kAlpha);
  a.cls[0] = 1; a.cnt[0] = kAlpha + 5.0f;
  a.cls[1] = 2; a.cnt[1] = kAlpha + 4.0f;
  auto b = scovox::defaultDirVoxel(kc, kAlpha);
  b.cls[0] = 3; b.cnt[0] = kAlpha + 3.0f;
  b.cls[1] = 4; b.cnt[1] = kAlpha + 2.0f;
  auto f = scovox::mergeDir(a, b, kc, kAlpha);
  // Top-2 by evidence kept: classes 1 (5) and 2 (4); 3 and 4 evicted.
  std::vector<uint16_t> kept;
  for (int i = 0; i < scovox::K_TOP; ++i) if (f.cls[i] != 0xFFFF) kept.push_back(f.cls[i]);
  std::sort(kept.begin(), kept.end());
  ASSERT_EQ(kept.size(), 2u);
  EXPECT_EQ(kept[0], uint16_t(1));
  EXPECT_EQ(kept[1], uint16_t(2));
  // OTHER == 0 prior + evicted raw evidence (3 + 2), NOT (C−K)·α₀ + 5.
  EXPECT_NEAR(f.other, 3.0f + 2.0f, 1e-5f);
}

TEST(ConsensusMergeV4, FrameMergeRejectsNumClassesBelowKTop) {
  // num_classes < K_TOP cannot yield a valid OTHER prior; mergeFramesV4 must
  // reject the configuration upstream rather than fold with inflated unknowns.
  scovox::BinarySerializerV4::Frame A;
  A.num_classes = 1; A.alpha_0 = kAlpha;
  scovox::BinarySerializerV4::Frame B;
  B.num_classes = 1; B.alpha_0 = kAlpha;
  EXPECT_THROW(scovox::mergeFramesV4(A, B), std::runtime_error);
}

TEST(ConsensusMergeV4, FrameMergeAcceptsNumClassesEqualKTop) {
  // num_classes == K_TOP is the boundary: residual_dims == 0 is valid (OTHER
  // prior 0), so the merge must SUCCEED — the reject guard is strictly `< K_TOP`.
  const uint16_t kc = scovox::K_TOP;
  scovox::BinarySerializerV4::Frame A;
  A.num_classes = kc; A.alpha_0 = kAlpha; A.resolution = 0.05f;
  { auto va = scovox::defaultDirVoxel(kc, kAlpha); va.cls[0] = 0; va.cnt[0] = kAlpha + 1.0f;
    A.dir_deltas.push_back({Bonxai::CoordT{1, 2, 3}, va}); }
  scovox::BinarySerializerV4::Frame B;
  B.num_classes = kc; B.alpha_0 = kAlpha; B.resolution = 0.05f;
  { auto vb = scovox::defaultDirVoxel(kc, kAlpha); vb.cls[0] = 0; vb.cnt[0] = kAlpha + 2.0f;
    B.dir_deltas.push_back({Bonxai::CoordT{1, 2, 3}, vb}); }
  auto F = scovox::mergeFramesV4(A, B);
  ASSERT_EQ(F.dir_deltas.size(), 1u);
  EXPECT_EQ(F.dir_deltas[0].data.cls[0], uint16_t(0));
  EXPECT_NEAR(F.dir_deltas[0].data.cnt[0], kAlpha + 3.0f, 1e-5f);
  // OTHER never inflated above its (clamped-to-0) prior.
  EXPECT_FLOAT_EQ(F.dir_deltas[0].data.other, 0.f);
}

// ===========================================================================
// Frame merge — Beta + Dir merged independently; coords unioned
// ===========================================================================

TEST(ConsensusMergeV4, FrameMergeUnionsCoordsIndependently) {
  scovox::BinarySerializerV4::Frame A;
  A.num_classes = kC; A.alpha_0 = kAlpha; A.resolution = 0.05f;
  A.beta_deltas.push_back({Bonxai::CoordT{1, 2, 3}, scovox::BetaVoxel{2.0f, 1.0f}});
  { auto va = dirPrior(); va.cls[0] = 5; va.cnt[0] = kAlpha + 1.0f;
    A.dir_deltas.push_back({Bonxai::CoordT{1, 2, 3}, va}); }

  scovox::BinarySerializerV4::Frame B;
  B.num_classes = kC; B.alpha_0 = kAlpha; B.resolution = 0.05f;
  B.beta_deltas.push_back({Bonxai::CoordT{1, 2, 3}, scovox::BetaVoxel{3.0f, 2.0f}});  // same coord
  { auto vb = dirPrior(); vb.cls[0] = 5; vb.cnt[0] = kAlpha + 2.0f;
    B.dir_deltas.push_back({Bonxai::CoordT{1, 2, 3}, vb}); }
  { auto vb2 = dirPrior(); vb2.cls[0] = 9; vb2.cnt[0] = kAlpha + 0.5f;
    B.dir_deltas.push_back({Bonxai::CoordT{4, 5, 6}, vb2}); }  // unique coord

  auto F = scovox::mergeFramesV4(A, B);

  // Beta: one shared coord → merged.
  ASSERT_EQ(F.beta_deltas.size(), 1u);
  EXPECT_FLOAT_EQ(F.beta_deltas[0].data.a_occ,  2.0f + 3.0f - scovox::kBetaOccPrior);
  EXPECT_FLOAT_EQ(F.beta_deltas[0].data.a_free, 1.0f + 2.0f - scovox::kBetaFreePrior);

  // Dir: two coords, the shared one merged.
  ASSERT_EQ(F.dir_deltas.size(), 2u);
  bool seen_shared = false;
  for (const auto& d : F.dir_deltas) {
    if (d.coord.x == 1 && d.coord.y == 2 && d.coord.z == 3) {
      seen_shared = true;
      EXPECT_EQ(d.data.cls[0], uint16_t(5));
      EXPECT_NEAR(d.data.cnt[0], kAlpha + 3.0f, 1e-5f);
    }
  }
  EXPECT_TRUE(seen_shared);
}

TEST(ConsensusMergeV4, FrameMergeRejectsClassCountMismatch) {
  scovox::BinarySerializerV4::Frame A; A.num_classes = 14; A.alpha_0 = kAlpha;
  scovox::BinarySerializerV4::Frame B; B.num_classes = 19; B.alpha_0 = kAlpha;
  EXPECT_THROW(scovox::mergeFramesV4(A, B), std::runtime_error);
}

TEST(ConsensusMergeV4, FrameMergeRejectsAlphaMismatch) {
  scovox::BinarySerializerV4::Frame A; A.num_classes = 14; A.alpha_0 = 0.01f;
  scovox::BinarySerializerV4::Frame B; B.num_classes = 14; B.alpha_0 = 0.05f;
  EXPECT_THROW(scovox::mergeFramesV4(A, B), std::runtime_error);
}

// ===========================================================================
// End-to-end: serialize → deserialize → merge → serialize → deserialize.
// Exercises the wire codec and the consensus merge together (the full
// multi-robot pipeline), and pins that per-grid mass conservation survives it.
// ===========================================================================

namespace {
const scovox::BinarySerializerV4::BetaDelta* findBeta(
    const scovox::BinarySerializerV4::Frame& f, int x, int y, int z) {
  for (const auto& d : f.beta_deltas)
    if (d.coord.x == x && d.coord.y == y && d.coord.z == z) return &d;
  return nullptr;
}
const scovox::BinarySerializerV4::DirDelta* findDir(
    const scovox::BinarySerializerV4::Frame& f, int x, int y, int z) {
  for (const auto& d : f.dir_deltas)
    if (d.coord.x == x && d.coord.y == y && d.coord.z == z) return &d;
  return nullptr;
}
}  // namespace

TEST(ConsensusMergeV4, EndToEndWireMergeRoundTrip) {
  // Two robots observe the same voxel (1,2,3) plus one unique voxel each.
  scovox::BinarySerializerV4::Frame A;
  A.num_classes = kC; A.alpha_0 = kAlpha; A.resolution = 0.05f;
  A.beta_deltas.push_back({Bonxai::CoordT{1, 2, 3}, scovox::BetaVoxel{2.0f, 1.0f}});
  { auto va = dirPrior(); va.cls[0] = 5; va.cnt[0] = kAlpha + 1.0f;
    A.dir_deltas.push_back({Bonxai::CoordT{1, 2, 3}, va}); }
  A.beta_deltas.push_back({Bonxai::CoordT{9, 9, 9}, scovox::BetaVoxel{3.0f, 0.5f}});

  scovox::BinarySerializerV4::Frame B;
  B.num_classes = kC; B.alpha_0 = kAlpha; B.resolution = 0.05f;
  B.beta_deltas.push_back({Bonxai::CoordT{1, 2, 3}, scovox::BetaVoxel{3.0f, 2.0f}});
  { auto vb = dirPrior(); vb.cls[0] = 5; vb.cnt[0] = kAlpha + 2.0f;
    B.dir_deltas.push_back({Bonxai::CoordT{1, 2, 3}, vb}); }
  B.beta_deltas.push_back({Bonxai::CoordT{7, 7, 7}, scovox::BetaVoxel{1.5f, 1.5f}});

  // Wire round-trip each (default: no TSDF), then merge, then round-trip again.
  auto A2 = scovox::BinarySerializerV4::deserialize(scovox::BinarySerializerV4::serialize(A));
  auto B2 = scovox::BinarySerializerV4::deserialize(scovox::BinarySerializerV4::serialize(B));
  auto F  = scovox::mergeFramesV4(A2, B2);
  auto F2 = scovox::BinarySerializerV4::deserialize(scovox::BinarySerializerV4::serialize(F));

  // Union of coords: {(1,2,3),(9,9,9),(7,7,7)} for Beta; {(1,2,3)} for Dir.
  EXPECT_EQ(F2.beta_deltas.size(), 3u);
  EXPECT_EQ(F2.dir_deltas.size(),  1u);

  // Shared Beta voxel merged with one prior removed.
  const auto* bshared = findBeta(F2, 1, 2, 3);
  ASSERT_NE(bshared, nullptr);
  EXPECT_FLOAT_EQ(bshared->data.a_occ,  2.0f + 3.0f - scovox::kBetaOccPrior);
  EXPECT_FLOAT_EQ(bshared->data.a_free, 1.0f + 2.0f - scovox::kBetaFreePrior);

  // Unique Beta voxels pass through unchanged.
  ASSERT_NE(findBeta(F2, 9, 9, 9), nullptr);
  ASSERT_NE(findBeta(F2, 7, 7, 7), nullptr);
  EXPECT_FLOAT_EQ(findBeta(F2, 9, 9, 9)->data.a_occ, 3.0f);
  EXPECT_FLOAT_EQ(findBeta(F2, 7, 7, 7)->data.a_free, 1.5f);

  // Shared Dir voxel: class 5 evidence summed, mass conserved.
  const auto* dshared = findDir(F2, 1, 2, 3);
  ASSERT_NE(dshared, nullptr);
  EXPECT_EQ(dshared->data.cls[0], uint16_t(5));
  EXPECT_NEAR(dshared->data.cnt[0], kAlpha + 3.0f, 1e-5f);
  EXPECT_NEAR(dshared->data.s_class(), (kC * kAlpha) + (kC * kAlpha) - kC * kAlpha + 3.0f, 1e-4f);
}
