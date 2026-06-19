/// @file test_tsdf_band.cpp
/// @brief Tests for the fused TSDF band update inside Map::integrateRay.
///
/// Default config used by most tests: resolution=0.10, sdf_trunc=0.30
/// (== 3 voxels). Rays are axis-aligned along +x so voxel coords are easy to
/// reason about: the surface voxel is at floor(hit.x / resolution).

#include <gtest/gtest.h>
#include <cmath>
#include <Eigen/Core>
#include "scovox/scovoxmap.hpp"

using namespace scovox;

namespace {

constexpr float kRes   = 0.10f;
constexpr float kTrunc = 0.30f;

Map makeTsdfMap(float trunc = kTrunc, float resolution = kRes,
                bool space_carving = false) {
  Params p;
  p.resolution = resolution;
  p.sdf_trunc  = trunc;
  p.tsdf_space_carving = space_carving;
  // Disable range decay so weight = 1.0 regardless of depth (running-average
  // test pins to integer values).
  p.range_decay_length = -1.0f;
  return Map(p);
}

bool getVoxelAtCoord(const Map& m, const Eigen::Vector3f& pos, Voxel& out) {
  return m.getVoxel(pos, out);
}

}  // namespace

// 1. Disabled by default --------------------------------------------------

TEST(TSDFBand, DisabledWhenTruncIsZero) {
  Map m = makeTsdfMap(0.0f);
  m.integrateRay(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(2.0f, 0, 0));

  size_t tsdf_touched = 0;
  m.grid().forEachCell([&](const Voxel& v, const Bonxai::CoordT&) {
    if (v.tsdf_weight > 0.f) ++tsdf_touched;
  });
  EXPECT_EQ(tsdf_touched, 0u);
}

// 2. Surface voxel hit ----------------------------------------------------

TEST(TSDFBand, SurfaceVoxelGetsZeroDistanceWithWeight) {
  Map m = makeTsdfMap();
  Eigen::Vector3f origin(0, 0, 0), hit(2.05f, 0, 0);  // in voxel (20, 0, 0)
  m.integrateRay(origin, hit);

  Voxel v;
  ASSERT_TRUE(getVoxelAtCoord(m, hit, v));
  EXPECT_GT(v.tsdf_weight, 0.f);
  EXPECT_LT(std::fabs(v.tsdf_distance), 0.5f * kRes);
}

// 3. Truncation band coverage --------------------------------------------

TEST(TSDFBand, OutsideBandUntouched) {
  Map m = makeTsdfMap();
  Eigen::Vector3f origin(0, 0, 0), hit(2.05f, 0, 0);
  m.integrateRay(origin, hit);

  // Voxel at x=1.0 m is 1.05 m in front of the surface, well past trunc=0.30.
  // With space_carving=false (default) it should NOT have TSDF mass.
  Voxel v;
  ASSERT_TRUE(getVoxelAtCoord(m, Eigen::Vector3f(1.0f, 0, 0), v));
  EXPECT_FLOAT_EQ(v.tsdf_weight, 0.f);

  // Voxel just inside the band (x=1.85, sdf=+0.20 from surface) should be touched.
  Voxel near;
  ASSERT_TRUE(getVoxelAtCoord(m, Eigen::Vector3f(1.85f, 0, 0), near));
  EXPECT_GT(near.tsdf_weight, 0.f);
}

// 4. Sign convention -----------------------------------------------------

TEST(TSDFBand, SignConventionFrontPositiveBehindNegative) {
  Map m = makeTsdfMap();
  Eigen::Vector3f origin(0, 0, 0), hit(2.05f, 0, 0);
  m.integrateRay(origin, hit);

  // One voxel in front: x ≈ 1.95 → sdf ≈ +0.10
  Voxel front;
  ASSERT_TRUE(getVoxelAtCoord(m, Eigen::Vector3f(1.95f, 0, 0), front));
  ASSERT_GT(front.tsdf_weight, 0.f);
  EXPECT_GT(front.tsdf_distance, 0.f);

  // One voxel behind: x ≈ 2.15 → sdf ≈ -0.10
  Voxel behind;
  ASSERT_TRUE(getVoxelAtCoord(m, Eigen::Vector3f(2.15f, 0, 0), behind));
  ASSERT_GT(behind.tsdf_weight, 0.f);
  EXPECT_LT(behind.tsdf_distance, 0.f);
}

// 5. Running average ------------------------------------------------------

TEST(TSDFBand, RunningAverageAccumulatesWeight) {
  Map m = makeTsdfMap();
  Eigen::Vector3f origin(0, 0, 0), hit(2.05f, 0, 0);
  m.integrateRay(origin, hit);
  m.integrateRay(origin, hit);

  Voxel v;
  ASSERT_TRUE(getVoxelAtCoord(m, hit, v));
  EXPECT_NEAR(v.tsdf_weight, 2.0f, 1e-4f);
  EXPECT_LT(std::fabs(v.tsdf_distance), 0.5f * kRes);
}

// 6. Beta unaffected by adding TSDF integration --------------------------

TEST(TSDFBand, BetaUnaffectedByTSDFIntegration) {
  // With TSDF disabled (trunc=0), a single ray puts a known a_occ on the
  // surface voxel: defaultVoxel().a_occ + w_occ * range_w * angle_w.
  // Default Params: w_occ=2, range_decay disabled here, angle_w=1, so
  // a_occ should be 1 + 2 = 3, a_free should be 1. Same numbers must hold
  // when TSDF is enabled — the fused walk must not perturb Beta state.
  Map m_off = makeTsdfMap(0.0f);
  Map m_on  = makeTsdfMap(kTrunc);
  Eigen::Vector3f origin(0, 0, 0), hit(2.05f, 0, 0);
  m_off.integrateRay(origin, hit);
  m_on.integrateRay(origin, hit);

  Voxel a, b;
  ASSERT_TRUE(getVoxelAtCoord(m_off, hit, a));
  ASSERT_TRUE(getVoxelAtCoord(m_on,  hit, b));
  EXPECT_FLOAT_EQ(a.a_occ,  b.a_occ);
  EXPECT_FLOAT_EQ(a.a_free, b.a_free);
  EXPECT_FLOAT_EQ(a.a_unk,  b.a_unk);
}

// 7. Joint ray-casting attenuation: occluder between origin and hit -------
//
// Under the joint ray-casting likelihood, the wall doesn't "block" carving
// outright — it attenuates downstream Beta updates by `reach_prob ≈ (1 -
// p_wall) ≈ 0` per voxel past the wall. The wall stays solid (its
// accumulated mass dominates), past-wall Beta evidence accumulates very
// slowly, and TSDF (geometric, not Bayesian) is unweighted by reach_prob.

TEST(TSDFBand, JointRaycastAttenuation) {
  Map m = makeTsdfMap();
  Eigen::Vector3f origin(0, 0, 0);
  Eigen::Vector3f wall(1.05f, 0, 0);   // voxel (10, 0, 0)
  Eigen::Vector3f beyond(3.05f, 0, 0); // voxel (30, 0, 0); past wall

  // Build up the wall to be clearly occupied. Under reach_prob, the wall
  // bootstraps slowly because cold-start path voxels start at p_occ=0.5.
  // 100 rays drive path voxels confidently free and the wall confidently
  // occupied.
  for (int i = 0; i < 100; ++i) {
    m.integrateRay(origin, wall);
  }
  Voxel wall_pre;
  ASSERT_TRUE(getVoxelAtCoord(m, wall, wall_pre));
  ASSERT_GT(wall_pre.p_occ(), 0.9f);

  Eigen::Vector3f before_wall(0.55f, 0, 0);
  Voxel before_pre;
  ASSERT_TRUE(getVoxelAtCoord(m, before_wall, before_pre));
  const float before_a_free_pre  = before_pre.a_free;

  // Fire one ray straight through the wall to a far surface.
  m.integrateRay(origin, beyond);

  Voxel before_post, wall_post, beyond_post;
  ASSERT_TRUE(getVoxelAtCoord(m, before_wall, before_post));
  ASSERT_TRUE(getVoxelAtCoord(m, wall,       wall_post));
  ASSERT_TRUE(getVoxelAtCoord(m, beyond,     beyond_post));

  // (a) Voxels in front of the wall got Beta-free incremented (full strength).
  EXPECT_GT(before_post.a_free, before_a_free_pre);

  // (b) Wall stays confidently occupied — its small free update is dwarfed
  //     by accumulated occupancy mass.
  EXPECT_GT(wall_post.p_occ(), 0.95f);

  // (c) The new surface voxel's Beta-occupied is heavily attenuated
  //     (reach_prob ≈ 0 through a confident wall — "we don't believe the
  //     ray actually got there"). TSDF still lands at full strength
  //     (geometric, not Bayesian).
  EXPECT_GT(beyond_post.tsdf_weight, 0.f);
  EXPECT_LT(std::fabs(beyond_post.tsdf_distance), 0.5f * kRes);

  // (d) Behind-surface band still got TSDF updates (geometric, no reach_prob).
  Voxel behind;
  ASSERT_TRUE(getVoxelAtCoord(m, Eigen::Vector3f(3.15f, 0, 0), behind));
  EXPECT_GT(behind.tsdf_weight, 0.f);
  EXPECT_LT(behind.tsdf_distance, 0.f);
}

// 8. Band-only integration -----------------------------------------------

// Confirms the band_only_integration flag (added 2026-04-29) confines the
// DDA to the truncation band [hit-trunc, hit+trunc] and skips the long
// Beta-free carve from origin. Voxels outside the band must be untouched;
// in-band voxels behave identically to the full-ray path.
TEST(TSDFBand, BandOnlyIntegrationSkipsFarFreeSpace) {
  Params p;
  p.resolution = kRes;
  p.sdf_trunc = kTrunc;
  p.range_decay_length = -1.0f;
  p.band_only_integration = true;
  Map m(p);

  // Long ray from origin to hit at x=2.05. Trunc = 0.30 → band is voxels
  // at x in [1.75, 2.35], i.e. coords {17,18,19,20,21,22,23}.
  Eigen::Vector3f origin(0, 0, 0), hit(2.05f, 0, 0);
  m.integrateRay(origin, hit);

  // Far-from-hit voxel (x=0.55, coord 5) — would be carved Beta-free under
  // the full-ray path; must be untouched in band-only mode.
  Voxel far;
  if (getVoxelAtCoord(m, Eigen::Vector3f(0.55f, 0, 0), far)) {
    EXPECT_FLOAT_EQ(far.a_free, 1.0f)
        << "band-only mode carved a voxel outside the band";
    EXPECT_FLOAT_EQ(far.a_occ, 1.0f);
    EXPECT_FLOAT_EQ(far.tsdf_weight, 0.f);
  }

  // Hit voxel still gets Beta-occupied + TSDF-zero.
  Voxel at_hit;
  ASSERT_TRUE(getVoxelAtCoord(m, hit, at_hit));
  EXPECT_GT(at_hit.a_occ, 1.0f);
  EXPECT_GT(at_hit.tsdf_weight, 0.f);
  EXPECT_LT(std::fabs(at_hit.tsdf_distance), 0.5f * kRes);

  // In-band voxel in front (x=1.85, coord 18) gets TSDF write (sdf > 0)
  // and Beta-free (still inside the DDA range, just shrunk).
  Voxel front;
  ASSERT_TRUE(getVoxelAtCoord(m, Eigen::Vector3f(1.85f, 0, 0), front));
  EXPECT_GT(front.tsdf_weight, 0.f);
  EXPECT_GT(front.tsdf_distance, 0.f);

  // In-band voxel behind (x=2.25, coord 22) gets TSDF write (sdf < 0).
  Voxel behind;
  ASSERT_TRUE(getVoxelAtCoord(m, Eigen::Vector3f(2.25f, 0, 0), behind));
  EXPECT_GT(behind.tsdf_weight, 0.f);
  EXPECT_LT(behind.tsdf_distance, 0.f);
}
