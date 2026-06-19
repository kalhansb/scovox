/// @file
/// @brief Step-7.5 gate: v3 consensus merge (TSDF + unified-Dirichlet SemDir).

#include <gtest/gtest.h>

#include <bonxai/bonxai.hpp>
#include <stdexcept>

#include "scovox/consensus_merge_v3.hpp"

namespace {

constexpr float kAlpha = scovox::kDefaultDirichletPrior;
constexpr uint16_t kC = 14;

scovox::SemDirVoxel prior() {
  return scovox::defaultSemDirVoxel(kC, kAlpha);
}

}  // namespace

// ---------------------------------------------------------------------------
// TSDF merge (unchanged from v2 algorithm, duplicated header for v3 isolation)
// ---------------------------------------------------------------------------

TEST(ConsensusMergeV3, TsdfWeightedAverage) {
  scovox::TsdfVoxel a{0.10f, 2.0f};
  scovox::TsdfVoxel b{-0.04f, 3.0f};
  auto f = scovox::mergeTsdfV3(a, b);
  // d_fused = (0.10·2 + (-0.04)·3) / (2+3) = (0.20 - 0.12) / 5 = 0.016
  EXPECT_NEAR(f.distance, 0.016f, 1e-6f);
  EXPECT_NEAR(f.weight,   5.0f,   1e-6f);
}

TEST(ConsensusMergeV3, TsdfZeroWeightDegenerates) {
  scovox::TsdfVoxel z{0.f, 0.f};
  auto f = scovox::mergeTsdfV3(z, z);
  EXPECT_NEAR(f.distance, 0.0f, 1e-6f);
  EXPECT_NEAR(f.weight,   0.0f, 1e-6f);
}

// ---------------------------------------------------------------------------
// SemDir merge — Dirichlet consensus across all K_TOP + 2 dimensions
// ---------------------------------------------------------------------------

TEST(ConsensusMergeV3, MergeWithPriorIsIdentityOnEvidence) {
  // Merging a voxel with the prior must preserve all observed evidence —
  // the duplicated symmetric Dirichlet prior cancels exactly across both
  // sides. (Mathematical analogue of v2's "Beta consensus a_A + a_B − 1".)
  auto p = prior();
  scovox::SemDirVoxel v = prior();
  v.alpha_free  = 0.5f;
  v.alpha_other = 0.30f;
  v.cls[0] = 7;  v.cnt[0] = 1.5f;
  v.cls[1] = 3;  v.cnt[1] = 0.8f;
  auto f = scovox::mergeSemDir(v, p, kC, kAlpha);
  EXPECT_FLOAT_EQ(f.alpha_free,  v.alpha_free);
  EXPECT_FLOAT_EQ(f.alpha_other, v.alpha_other);
  EXPECT_EQ(f.cls[0], v.cls[0]);
  EXPECT_FLOAT_EQ(f.cnt[0], v.cnt[0]);
  EXPECT_EQ(f.cls[1], v.cls[1]);
  EXPECT_FLOAT_EQ(f.cnt[1], v.cnt[1]);
}

TEST(ConsensusMergeV3, MergeIsSymmetric) {
  scovox::SemDirVoxel a = prior();
  a.alpha_free = 0.2f;  a.alpha_other = 0.20f;
  a.cls[0] = 5; a.cnt[0] = 1.0f;
  a.cls[1] = 8; a.cnt[1] = 0.5f;

  scovox::SemDirVoxel b = prior();
  b.alpha_free = 0.4f;  b.alpha_other = 0.15f;
  b.cls[0] = 5; b.cnt[0] = 0.8f;   // matches a's slot 0
  b.cls[1] = 2; b.cnt[1] = 0.3f;   // unique to b

  auto f_ab = scovox::mergeSemDir(a, b, kC, kAlpha);
  auto f_ba = scovox::mergeSemDir(b, a, kC, kAlpha);

  EXPECT_FLOAT_EQ(f_ab.alpha_free,  f_ba.alpha_free);
  EXPECT_FLOAT_EQ(f_ab.alpha_other, f_ba.alpha_other);
  // Top-K class IDs may be permuted between symmetric merges if counts
  // tie — collect into multisets keyed by class.
  std::vector<std::pair<uint16_t, float>> ab, ba;
  for (int i = 0; i < scovox::K_TOP; ++i) {
    if (f_ab.cls[i] != 0xFFFF) ab.emplace_back(f_ab.cls[i], f_ab.cnt[i]);
    if (f_ba.cls[i] != 0xFFFF) ba.emplace_back(f_ba.cls[i], f_ba.cnt[i]);
  }
  std::sort(ab.begin(), ab.end());
  std::sort(ba.begin(), ba.end());
  ASSERT_EQ(ab.size(), ba.size());
  for (std::size_t i = 0; i < ab.size(); ++i) {
    EXPECT_EQ(ab[i].first,         ba[i].first);
    EXPECT_FLOAT_EQ(ab[i].second,  ba[i].second);
  }
}

TEST(ConsensusMergeV3, CoincidingClassSumsObservedEvidence) {
  // Both sides track class 5 with observed_evidence e_A = 1.0 and e_B = 2.0
  // (slot cnt = α_0 + evidence). Merge should give slot 0 cnt = α_0 + 3.0
  // — the duplicated prior is removed once.
  scovox::SemDirVoxel a = prior();
  a.cls[0] = 5; a.cnt[0] = kAlpha + 1.0f;
  scovox::SemDirVoxel b = prior();
  b.cls[0] = 5; b.cnt[0] = kAlpha + 2.0f;
  auto f = scovox::mergeSemDir(a, b, kC, kAlpha);
  EXPECT_EQ(f.cls[0], uint16_t(5));
  EXPECT_NEAR(f.cnt[0], kAlpha + 3.0f, 1e-5f);
}

TEST(ConsensusMergeV3, EvictionOnMerge) {
  // 4 distinct classes across the two top-K=2 vectors → 2 must evict.
  // Evicted classes' observed evidence (cnt − α_0) lands in alpha_other.
  scovox::SemDirVoxel a = prior();
  a.cls[0] = 1; a.cnt[0] = kAlpha + 5.0f;
  a.cls[1] = 2; a.cnt[1] = kAlpha + 4.0f;
  scovox::SemDirVoxel b = prior();
  b.cls[0] = 3; b.cnt[0] = kAlpha + 3.0f;
  b.cls[1] = 4; b.cnt[1] = kAlpha + 2.0f;
  auto f = scovox::mergeSemDir(a, b, kC, kAlpha);
  // Top-K should be {1, 2} (highest counts).
  std::vector<uint16_t> kept;
  for (int i = 0; i < scovox::K_TOP; ++i)
    if (f.cls[i] != 0xFFFF) kept.push_back(f.cls[i]);
  std::sort(kept.begin(), kept.end());
  ASSERT_EQ(kept.size(), 2u);
  EXPECT_EQ(kept[0], uint16_t(1));
  EXPECT_EQ(kept[1], uint16_t(2));
  // OTHER: a + b OTHER priors merged (12α) + evicted classes' evidence (3 + 2).
  const float other_prior_one = 12.f * kAlpha;
  const float expected_other = other_prior_one + 3.0f + 2.0f;
  // Both inputs were at OTHER's prior with no extra evidence.
  EXPECT_NEAR(f.alpha_other, expected_other, 1e-5f);
}

TEST(ConsensusMergeV3, FrameMergeUnionsCoords) {
  scovox::BinarySerializerV3::Frame A;
  A.num_classes = kC; A.alpha_0 = kAlpha; A.resolution = 0.05f;
  scovox::SemDirVoxel va = prior();
  va.cls[0] = 5; va.cnt[0] = kAlpha + 1.0f;
  A.semdir_deltas.push_back({Bonxai::CoordT{1, 2, 3}, va});

  scovox::BinarySerializerV3::Frame B;
  B.num_classes = kC; B.alpha_0 = kAlpha; B.resolution = 0.05f;
  scovox::SemDirVoxel vb = prior();
  vb.cls[0] = 5; vb.cnt[0] = kAlpha + 2.0f;
  B.semdir_deltas.push_back({Bonxai::CoordT{1, 2, 3}, vb});  // same coord
  scovox::SemDirVoxel vb2 = prior();
  vb2.cls[0] = 9; vb2.cnt[0] = kAlpha + 0.5f;
  B.semdir_deltas.push_back({Bonxai::CoordT{4, 5, 6}, vb2}); // unique

  auto F = scovox::mergeFramesV3(A, B);
  EXPECT_EQ(F.semdir_deltas.size(), 2u);
  // Find the (1,2,3) entry.
  bool seen_shared = false;
  for (const auto& d : F.semdir_deltas) {
    if (d.coord.x == 1 && d.coord.y == 2 && d.coord.z == 3) {
      seen_shared = true;
      EXPECT_EQ(d.data.cls[0], uint16_t(5));
      EXPECT_NEAR(d.data.cnt[0], kAlpha + 3.0f, 1e-5f);
    }
  }
  EXPECT_TRUE(seen_shared);
}

TEST(ConsensusMergeV3, FrameMergeRejectsClassCountMismatch) {
  scovox::BinarySerializerV3::Frame A; A.num_classes = 14; A.alpha_0 = kAlpha;
  scovox::BinarySerializerV3::Frame B; B.num_classes = 19; B.alpha_0 = kAlpha;
  EXPECT_THROW(scovox::mergeFramesV3(A, B), std::runtime_error);
}

TEST(ConsensusMergeV3, FrameMergeRejectsAlphaMismatch) {
  scovox::BinarySerializerV3::Frame A; A.num_classes = 14; A.alpha_0 = 0.01f;
  scovox::BinarySerializerV3::Frame B; B.num_classes = 14; B.alpha_0 = 0.05f;
  EXPECT_THROW(scovox::mergeFramesV3(A, B), std::runtime_error);
}
