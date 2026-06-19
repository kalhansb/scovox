/// @file
/// @brief Step-3 gate tests for SemBetaMap. Two block-the-merge invariants
/// (Q5: Beta(1,1) prior at first observation; Q5: miss-carves-no-TSDF)
/// plus a handful of behavioural pinning tests.

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <vector>

#include "scovox/sembeta_map.hpp"

namespace {

constexpr float kRes = 0.05f;

scovox::SemBetaMap makeMap() {
  scovox::SemBetaMap::Params p;
  p.resolution               = kRes;
  p.dirichlet_min_p_occ      = 0.5f;
  p.kappa0                   = 1.0f;
  p.evidence_saturation      = 0.0f;
  return scovox::SemBetaMap(p);
}

}  // namespace

// ===========================================================================
// PARITY / INVARIANT TESTS — block merge if any fails.
// ===========================================================================

TEST(SemBetaMapInvariants, FirstObservationStartsFromBeta11Prior) {
  auto m = makeMap();
  // One hit ray with quality=1.0 and no semantics. The endpoint voxel
  // should read back as Beta(1+w_occ, 1) = Beta(2, 1) under default params
  // — proving the prior was applied via defaultSemBetaVoxel(), not skipped
  // by Bonxai's pool zero-init.
  m.integrateHit(Eigen::Vector3f(0, 0, 0),
                 Eigen::Vector3f(1.0f, 0, 0),
                 /*sem_probs=*/nullptr,
                 /*quality=*/1.0f);
  auto v = m.getVoxel(Eigen::Vector3f(1.0f, 0, 0));
  ASSERT_TRUE(v.has_value());
  EXPECT_NEAR(v->a_occ,  1.0f + 1.0f * 1.0f, 1e-5f);  // prior 1 + w_occ * quality
  EXPECT_NEAR(v->a_free, 1.0f, 1e-5f);                // prior, no carve at hit
  EXPECT_NEAR(v->a_unk,  0.0f, 1e-5f);
}

TEST(SemBetaMapInvariants, MissDoesNotUpdateOccupancy) {
  auto m = makeMap();
  m.integrateMiss(Eigen::Vector3f(0, 0, 0),
                  Eigen::Vector3f(1.0f, 0, 0),
                  /*quality=*/1.0f);

  auto v_mid = m.getVoxel(Eigen::Vector3f(0.5f, 0, 0));
  ASSERT_TRUE(v_mid.has_value());
  EXPECT_LT(v_mid->a_occ, v_mid->a_free)  // free wins on a miss
      << "miss must drive p_occ down, not up";

  auto v_end = m.getVoxel(Eigen::Vector3f(1.0f, 0, 0));
  ASSERT_TRUE(v_end.has_value());
  EXPECT_NEAR(v_end->a_occ, 1.0f, 1e-5f) << "miss must not increment a_occ";
}

// ===========================================================================
// Behavioural tests
// ===========================================================================

TEST(SemBetaMap, DirichletUpdateGatedByPOcc) {
  // First observation alone is at the Beta(1,1) prior + one hit, p_occ=2/3,
  // which is above the default gate of 0.5. Semantic update should fire.
  auto m = makeMap();
  std::vector<float> probs{0.f, 1.f, 0.f, 0.f};  // class 1, one-hot
  m.integrateHit(Eigen::Vector3f(0, 0, 0),
                 Eigen::Vector3f(1.0f, 0, 0),
                 &probs, /*quality=*/1.0f);
  auto v = m.getVoxel(Eigen::Vector3f(1.0f, 0, 0));
  ASSERT_TRUE(v.has_value());
  // Slot 0 should now hold class 1 with non-zero count (kappa0 * p_occ * quality).
  EXPECT_EQ(v->sem_cls[0], uint16_t(1));
  EXPECT_GT(v->sem_cnt[0], 0.0f);
}

TEST(SemBetaMap, CarveDecaysOccupancyAlongRay) {
  auto m = makeMap();
  m.integrateHit(Eigen::Vector3f(0, 0, 0),
                 Eigen::Vector3f(1.0f, 0, 0),
                 nullptr, 1.0f);

  // A voxel mid-ray was carved free.
  auto v_mid = m.getVoxel(Eigen::Vector3f(0.5f, 0, 0));
  ASSERT_TRUE(v_mid.has_value());
  EXPECT_GT(v_mid->a_free, v_mid->a_occ);
}

TEST(SemBetaMap, WallBlockingStopsCarving) {
  auto m = makeMap();
  // Pre-populate a confidently-occupied wall voxel mid-ray.
  Eigen::Vector3f wall_pos(0.5f, 0, 0);
  Eigen::Vector3f beyond(1.0f, 0, 0);
  {
    auto acc = m.grid().createAccessor();
    auto coord = m.grid().posToCoord(wall_pos.x(), wall_pos.y(), wall_pos.z());
    auto pre = scovox::defaultSemBetaVoxel();
    pre.a_occ  = 100.f;
    pre.a_free = 1.f;
    acc.setValue(coord, pre);
  }
  // Now carve through it.
  m.integrateHit(Eigen::Vector3f(0, 0, 0), beyond, nullptr, 1.0f);

  // Voxels BEYOND the wall (closer to the hit) should not have been carved.
  auto v_past = m.getVoxel(Eigen::Vector3f(0.75f, 0, 0));
  // It either doesn't exist OR was never carved (still at prior).
  if (v_past.has_value()) {
    EXPECT_NEAR(v_past->a_free, 1.0f, 1e-5f);
  }

  // Wall voxel itself should still have its high a_occ (we didn't carve it).
  auto v_wall = m.getVoxel(wall_pos);
  ASSERT_TRUE(v_wall.has_value());
  EXPECT_NEAR(v_wall->a_occ, 100.f, 1e-3f);
}

TEST(SemBetaMap, EvidenceSaturationCapsAlphas) {
  scovox::SemBetaMap::Params p;
  p.resolution          = kRes;
  p.evidence_saturation = 5.0f;
  p.w_occ               = 1.0f;
  p.dirichlet_min_p_occ = 0.5f;
  scovox::SemBetaMap m(p);

  // Many hits at the same voxel — without saturation a_occ would grow
  // unboundedly. With cap=5, a_occ should not exceed 5.
  for (int i = 0; i < 100; ++i) {
    m.integrateHit(Eigen::Vector3f(0, 0, 0),
                   Eigen::Vector3f(1.0f, 0, 0),
                   nullptr, 1.0f);
  }
  auto v = m.getVoxel(Eigen::Vector3f(1.0f, 0, 0));
  ASSERT_TRUE(v.has_value());
  EXPECT_LE(v->a_occ, 5.0f + 1e-3f);
}

TEST(SemBetaMap, DrainTouchedClearsBuffer) {
  auto m = makeMap();
  m.integrateHit(Eigen::Vector3f(0, 0, 0),
                 Eigen::Vector3f(1.0f, 0, 0),
                 nullptr, 1.0f);
  EXPECT_GT(m.touchedCount(), 0u);
  auto t1 = m.drainTouched();
  EXPECT_GT(t1.size(), 0u);
  EXPECT_EQ(m.touchedCount(), 0u);
}
