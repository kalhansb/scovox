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
  return scovox::defaultBetaVoxel(kC * kAlpha, kAlpha);  // a_occ=C·α₀, a_free=α₀
}
scovox::DirVoxel dirPrior() {
  return scovox::defaultDirVoxel(kC, kAlpha);
}

}  // namespace

// ===========================================================================
// BetaVoxel merge (conjugate Beta)
// ===========================================================================

TEST(ConsensusMergeV4, BetaMergeWithPriorPreservesEvidence) {
  scovox::BetaVoxel v{2.0f, 1.0f};
  auto f = scovox::mergeBeta(v, betaPrior(), kC, kAlpha);
  EXPECT_FLOAT_EQ(f.a_occ,  v.a_occ);   // duplicated prior cancels
  EXPECT_FLOAT_EQ(f.a_free, v.a_free);
}

TEST(ConsensusMergeV4, BetaMergeOfTwoPriorsIsPrior) {
  auto f = scovox::mergeBeta(betaPrior(), betaPrior(), kC, kAlpha);
  EXPECT_FLOAT_EQ(f.a_occ,  kC * kAlpha);
  EXPECT_FLOAT_EQ(f.a_free, kAlpha);
}

TEST(ConsensusMergeV4, BetaMergeIsSymmetricAndAdditive) {
  scovox::BetaVoxel a{3.0f, 1.0f};
  scovox::BetaVoxel b{1.0f, 4.0f};
  auto ab = scovox::mergeBeta(a, b, kC, kAlpha);
  auto ba = scovox::mergeBeta(b, a, kC, kAlpha);
  EXPECT_FLOAT_EQ(ab.a_occ,  ba.a_occ);
  EXPECT_FLOAT_EQ(ab.a_free, ba.a_free);
  // a_occ = 3 + 1 − C·α₀ ; a_free = 1 + 4 − α₀.
  EXPECT_FLOAT_EQ(ab.a_occ,  4.0f - kC * kAlpha);
  EXPECT_FLOAT_EQ(ab.a_free, 5.0f - kAlpha);
}

TEST(ConsensusMergeV4, BetaMassConservation) {
  // Δ(a_occ + a_free) = a.total + b.total − (C+1)·α₀ (one prior removed).
  scovox::BetaVoxel a{2.5f, 1.5f};
  scovox::BetaVoxel b{0.5f, 3.0f};
  auto f = scovox::mergeBeta(a, b, kC, kAlpha);
  const float beta_prior_total = (kC + 1) * kAlpha;
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
  EXPECT_FLOAT_EQ(F.beta_deltas[0].data.a_occ,  2.0f + 3.0f - kC * kAlpha);
  EXPECT_FLOAT_EQ(F.beta_deltas[0].data.a_free, 1.0f + 2.0f - kAlpha);

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
  EXPECT_FLOAT_EQ(bshared->data.a_occ,  2.0f + 3.0f - kC * kAlpha);
  EXPECT_FLOAT_EQ(bshared->data.a_free, 1.0f + 2.0f - kAlpha);

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
