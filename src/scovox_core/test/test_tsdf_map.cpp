/// @file
/// @brief Step-2 gate of the split-grid refactor.
///
/// Three SLIM-VDB parity tests block the merge:
///   1. Euclidean SDF, not projected
///   2. Voxel CENTRE sample point, not lower corner
///   3. Per-voxel weighting via WeightFn (constant default)
///
/// Plus seven behavioural tests that pin Curless–Levoy fundamentals.

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cmath>

#include "scovox/tsdf_map.hpp"

namespace {

constexpr float kRes  = 0.05f;
constexpr float kTrunc = 0.15f;

scovox::TsdfMap makeMap(bool space_carving = false) {
  scovox::TsdfMap::Params p;
  p.resolution    = kRes;
  p.sdf_trunc     = kTrunc;
  p.space_carving = space_carving;
  return scovox::TsdfMap(p);
}

}  // namespace

// ===========================================================================
// PARITY TESTS — block merge if any fails. See plan §4.12.
// ===========================================================================

TEST(TsdfMapParity, EuclideanSDFnotProjected) {
  auto m = makeMap();

  // +X ray from (0,0,0) to (0.10, 0, 0). DDA walks voxels (k_x, 0, 0) but
  // each voxel's CENTRE is at y=z=0.025 (half-voxel offset from the ray).
  // So even a purely-axial ray gives off-axis voxel centres, and the two
  // SDF formulas diverge. Probe voxel (1, 0, 0): centre at (0.075, 0.025, 0.025).
  const Eigen::Vector3f origin(0.f, 0.f, 0.f);
  const Eigen::Vector3f endpoint(0.10f, 0.f, 0.f);
  m.integrateRay(origin, endpoint);

  const Eigen::Vector3f probe(0.075f, 0.025f, 0.025f);
  auto v = m.getVoxel(probe);
  ASSERT_TRUE(v.has_value());

  // Expected (Euclidean):
  //   vc - origin   = (0.075, 0.025, 0.025)
  //   endpoint - vc = (0.025, -0.025, -0.025)
  //   dist          = ‖endpoint - vc‖ = 0.025·√3 ≈ 0.0433
  //   proj          = 0.075·0.025 + 0.025·(-0.025) + 0.025·(-0.025)
  //                 = 0.001875 - 0.000625 - 0.000625 = +0.000625 → sign +1
  //   sdf           = +0.0433 (in front of surface, inside the band)
  EXPECT_NEAR(v->distance, +0.0433f, 1e-3f);

  // Distinguishability: the projected formula would have given
  //   sdf = depth - (vc - origin)·û = 0.10 - 0.075 = +0.025
  // Make sure we are NOT seeing that.
  EXPECT_LT(std::fabs(v->distance - 0.0433f), std::fabs(v->distance - 0.025f));
}

TEST(TsdfMapParity, VoxelCentreNotLowerCorner) {
  auto m = makeMap();

  // Endpoint exactly at the centre of voxel (1,0,0): vc = (0.075, 0.025, 0.025).
  // Ray along +X from origin.
  const Eigen::Vector3f origin(0.f, 0.f, 0.f);
  const Eigen::Vector3f endpoint(0.075f, 0.f, 0.f);
  m.integrateRay(origin, endpoint);

  // Probe the (1,0,0) voxel via a position inside it.
  const Eigen::Vector3f probe(0.075f, 0.025f, 0.025f);
  auto v = m.getVoxel(probe);
  ASSERT_TRUE(v.has_value());

  // Expected SDF for vc = (0.075, 0.025, 0.025), endpoint = (0.075, 0, 0):
  //   endpoint - vc = (0, -0.025, -0.025)
  //   dist          = 0.0354
  //   (vc-origin)·(endpoint-vc) = 0.075*0 + 0.025*(-0.025) + 0.025*(-0.025) = -0.00125
  //   sign          = -1
  //   sdf           = -0.0354
  EXPECT_NEAR(v->distance, -0.0354f, 1e-3f);

  // If the implementation used voxel LOWER CORNER (= 0.05, 0, 0) instead:
  //   endpoint - corner = (0.025, 0, 0), dist = 0.025
  //   (corner-origin)·(endpoint-corner) = 0.05*0.025 = +0.00125  → sign +1
  //   sdf = +0.025
  // Make sure we are NOT seeing that.
  EXPECT_LT(std::fabs(v->distance + 0.0354f), std::fabs(v->distance - 0.025f));
}

TEST(TsdfMapParity, PerVoxelWeightingViaWeightFn) {
  auto m = makeMap();

  // Gaussian weight centred at sdf=0 with stddev 0.05 — a band-edge voxel
  // at |sdf| ≈ trunc=0.15 should get weight ≈ exp(-9/2) ≈ 0.011, while a
  // near-surface voxel should get weight ≈ 1.0.
  auto wf = scovox::TsdfMap::exponential(0.05f);

  // A long ray so multiple voxels along it get touched at varying |sdf|.
  m.integrateRay(Eigen::Vector3f(0, 0, 0),
                 Eigen::Vector3f(0.50f, 0, 0),
                 wf);

  // Voxel near surface: position around endpoint.
  auto v_surface = m.getVoxel(Eigen::Vector3f(0.50f, 0, 0));
  ASSERT_TRUE(v_surface.has_value());
  // Voxel near band edge: trunc behind surface, |sdf| ≈ trunc.
  auto v_edge = m.getVoxel(Eigen::Vector3f(0.50f - kTrunc + 0.01f, 0, 0));
  ASSERT_TRUE(v_edge.has_value());

  // Surface voxel should have higher weight than edge voxel —
  // this is precisely the property a per-ray weight scheme would lack.
  EXPECT_GT(v_surface->weight, v_edge->weight)
      << "surface w=" << v_surface->weight
      << "  edge w="  << v_edge->weight;
  // Sanity: the edge weight is small, the surface weight is near 1.0.
  EXPECT_GT(v_surface->weight, 0.5f);
  EXPECT_LT(v_edge->weight,    0.3f);
}

// ===========================================================================
// BEHAVIOURAL TESTS — fundamentals of Curless–Levoy + DDA correctness.
// ===========================================================================

TEST(TsdfMap, CurlessLevoyConvergence) {
  // Integrate the same ray N times with constant weight; the surface voxel's
  // distance should converge to the (clamped) SDF — a fixed point of the
  // weighted running average.
  auto m = makeMap();
  const Eigen::Vector3f origin(0, 0, 0), endpoint(0.50f, 0, 0);

  for (int i = 0; i < 100; ++i) {
    m.integrateRay(origin, endpoint);
  }

  auto v = m.getVoxel(endpoint);
  ASSERT_TRUE(v.has_value());
  // Surface voxel: |sdf| << trunc, so distance ≈ true sdf (small in
  // magnitude). Weight accumulates linearly.
  EXPECT_NEAR(v->weight, 100.0f, 1e-3f);
  EXPECT_LT(std::fabs(v->distance), kTrunc);
}

TEST(TsdfMap, SignConventionFrontPositiveBehindNegative) {
  auto m = makeMap();
  m.integrateRay(Eigen::Vector3f(0, 0, 0),
                 Eigen::Vector3f(0.50f, 0, 0));

  // In front of surface (closer to origin): positive sdf.
  auto v_front = m.getVoxel(Eigen::Vector3f(0.42f, 0, 0));
  ASSERT_TRUE(v_front.has_value());
  EXPECT_GT(v_front->distance, 0.f);

  // Behind surface (further from origin, but still inside band): negative.
  auto v_back = m.getVoxel(Eigen::Vector3f(0.58f, 0, 0));
  ASSERT_TRUE(v_back.has_value());
  EXPECT_LT(v_back->distance, 0.f);
}

TEST(TsdfMap, OutsideBandUntouchedWithoutSpaceCarving) {
  auto m = makeMap(/*space_carving=*/false);
  m.integrateRay(Eigen::Vector3f(0, 0, 0),
                 Eigen::Vector3f(2.05f, 0, 0));

  // Voxel well in front of the band: not touched.
  auto v_far_front = m.getVoxel(Eigen::Vector3f(1.0f, 0, 0));
  EXPECT_FALSE(v_far_front.has_value())
      << "without space_carving, voxels >1m before the surface band must NOT exist";

  // Voxel inside the band: touched.
  auto v_band = m.getVoxel(Eigen::Vector3f(2.0f, 0, 0));
  EXPECT_TRUE(v_band.has_value());
}

TEST(TsdfMap, SpaceCarvingTouchesFreeSpace) {
  auto m = makeMap(/*space_carving=*/true);
  m.integrateRay(Eigen::Vector3f(0, 0, 0),
                 Eigen::Vector3f(2.05f, 0, 0));

  // With space carving, voxels well in front of the surface ARE walked.
  auto v_far_front = m.getVoxel(Eigen::Vector3f(1.0f, 0, 0));
  ASSERT_TRUE(v_far_front.has_value())
      << "with space_carving=true, voxels in [origin, hit-trunc] must be visited";
  // ...and clamped at +trunc (max free-space distance).
  EXPECT_NEAR(v_far_front->distance, kTrunc, 1e-3f);
}

TEST(TsdfMap, EmptyIntegrationNoOp) {
  auto m = makeMap();
  EXPECT_EQ(m.voxelCount(), 0u);
  EXPECT_EQ(m.touchedCount(), 0u);
}

TEST(TsdfMap, DegenerateRayDoesNotCrash) {
  auto m = makeMap();
  // origin == endpoint: depth ≈ 0. Must early-return cleanly.
  m.integrateRay(Eigen::Vector3f(1, 1, 1),
                 Eigen::Vector3f(1, 1, 1));
  EXPECT_EQ(m.voxelCount(), 0u);
}

TEST(TsdfMap, DrainTouchedDeduplicatesAndClears) {
  auto m = makeMap();
  m.integrateRay(Eigen::Vector3f(0, 0, 0),
                 Eigen::Vector3f(0.50f, 0, 0));
  const auto count_before_drain = m.touchedCount();
  EXPECT_GT(count_before_drain, 0u);

  auto t1 = m.drainTouched();
  EXPECT_EQ(m.touchedCount(), 0u);
  // Re-integrate the same ray; touched count should rise again.
  m.integrateRay(Eigen::Vector3f(0, 0, 0),
                 Eigen::Vector3f(0.50f, 0, 0));
  EXPECT_GT(m.touchedCount(), 0u);
  auto t2 = m.drainTouched();
  // Both drains should yield the same de-duplicated coord set (deterministic ray).
  EXPECT_EQ(t1.size(), t2.size());
}

TEST(TsdfMap, MemoryAccountingMatchesStruct) {
  auto m = makeMap();
  m.integrateRay(Eigen::Vector3f(0, 0, 0),
                 Eigen::Vector3f(0.50f, 0, 0));
  EXPECT_GT(m.voxelCount(),     0u);
  EXPECT_GT(m.gridMemoryBytes(), m.voxelCount() * sizeof(scovox::TsdfVoxel) - 1);
  // gridMemoryBytes >= voxelCount * sizeof(TsdfVoxel) (overhead is leaf
  // metadata + mask bits + accessor cache, which dwarfs raw cell bytes
  // at low fill ratios).
}
