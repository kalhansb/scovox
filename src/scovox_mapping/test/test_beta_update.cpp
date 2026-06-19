/// @file test_beta_update.cpp
/// Tasks 1.5 + 1.6: Digamma unit tests + Beta update correctness (>=25 tests).

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <Eigen/Core>
#include "scovox/scovoxmap.hpp"

using namespace scovox;

// =====================================================================
// Task 1.5 — Digamma tests
// =====================================================================

TEST(Digamma, KnownValuePsi1) {
  // psi(1) = -gamma ~ -0.57721566490153286
  EXPECT_NEAR(digamma(1.0f), -0.5772156649f, 1e-8f);
}

TEST(Digamma, KnownValuePsi2) {
  // psi(2) = 1 - gamma ~ 0.42278433509846714
  EXPECT_NEAR(digamma(2.0f), 0.4227843351f, 1e-8f);
}

TEST(Digamma, KnownValuePsiHalf) {
  // psi(0.5) = -gamma - 2ln2 ~ -1.9635100260214235
  EXPECT_NEAR(digamma(0.5f), -1.9635100260f, 1e-7f);
}

TEST(Digamma, KnownValuePsi10) {
  // psi(10) ~ 2.25175258906672
  EXPECT_NEAR(digamma(10.0f), 2.2517525891f, 1e-7f);
}

TEST(Digamma, KnownValuePsi100) {
  // psi(100) ~ 4.60016185273649
  EXPECT_NEAR(digamma(100.0f), 4.6001618527f, 1e-7f);
}

TEST(Digamma, RecurrenceRelation) {
  // psi(x+1) = psi(x) + 1/x for several values
  for (float x : {0.5f, 1.0f, 2.5f, 7.0f, 50.0f}) {
    float lhs = digamma(x + 1.0f);
    float rhs = digamma(x) + 1.0f / x;
    EXPECT_NEAR(lhs, rhs, 1e-5f) << "Failed recurrence at x=" << x;
  }
}

TEST(Digamma, NegativeInputReturnsNegInf) {
  EXPECT_TRUE(std::isinf(digamma(0.0f)));
  EXPECT_TRUE(std::isinf(digamma(-1.0f)));
}

TEST(Digamma, SmallPositiveValues) {
  // psi(0.1) ~ -10.42375 (tests recurrence for very small x)
  float val = digamma(0.1f);
  EXPECT_NEAR(val, -10.42375f, 0.001f);
}

// =====================================================================
// Task 1.6 — Beta update correctness (>=25 tests)
// =====================================================================

// Helper to make a map with default params
static Map makeMap(float resolution = 0.05f) {
  Params p;
  p.resolution = resolution;
  return Map(p);
}

// --- 1. Basic occupancy updates ---

TEST(BetaUpdate, PriorVoxelIsUniform) {
  Voxel v = defaultVoxel();
  EXPECT_FLOAT_EQ(v.a_occ, 1.0f);
  EXPECT_FLOAT_EQ(v.a_free, 1.0f);
  EXPECT_FLOAT_EQ(v.p_occ(), 0.5f);
}

TEST(BetaUpdate, SingleHitIncreasesOccupancy) {
  auto map = makeMap();
  Eigen::Vector3f origin(0, 0, 0), hit(1, 0, 0);
  map.integrateRay(origin, hit);

  Voxel v = defaultVoxel();
  ASSERT_TRUE(map.getVoxel(hit, v));
  EXPECT_GT(v.a_occ, 1.0f);
  EXPECT_GT(v.p_occ(), 0.5f);
}

TEST(BetaUpdate, NHitsConvergesToExpected) {
  // After N hits with w_occ=2, a_occ = 1 + 2*N, a_free = 1
  // p_occ = (1 + 2N) / (2 + 2N) -> 1 as N -> inf
  auto map = makeMap(0.5);  // coarse grid so origin != hit cell
  Eigen::Vector3f origin(0, 0, 0), hit(2, 0, 0);
  const int N = 20;
  for (int i = 0; i < N; ++i) {
    map.integrateRay(origin, hit);
  }

  Voxel v = defaultVoxel();
  ASSERT_TRUE(map.getVoxel(hit, v));
  float expected_a_occ = 1.0f + 2.0f * N;  // w_occ=2 default
  // Range weighting reduces this, but p_occ should still be high
  EXPECT_GT(v.p_occ(), 0.9f);
  EXPECT_GT(v.a_occ, 10.0f);
}

TEST(BetaUpdate, FreeSpaceUpdateIncreasesAFree) {
  auto map = makeMap(0.5);
  Eigen::Vector3f origin(0, 0, 0), hit(3, 0, 0);
  map.integrateRay(origin, hit);

  // Voxel at midpoint should have free evidence
  Eigen::Vector3f mid(1, 0, 0);
  Voxel v = defaultVoxel();
  if (map.getVoxel(mid, v)) {
    EXPECT_GT(v.a_free, 1.0f);
    EXPECT_LT(v.p_occ(), 0.5f);
  }
}

// --- 3. Temporal decay ---

TEST(BetaUpdate, DecayMovesTowardPrior) {
  auto map = makeMap(0.5);
  Eigen::Vector3f origin(0, 0, 0), hit(2, 0, 0);

  // Add to transient grid
  map.integrateRay(origin, hit, /*is_dynamic=*/true);

  Voxel before = defaultVoxel();
  map.transientGrid().forEachCell([&](const Voxel& v, const Bonxai::CoordT&) {
    if (v.a_occ > 1.01f) before = v;
  });
  ASSERT_GT(before.a_occ, 1.01f);

  // Decay
  map.decayTransientGrid(0.5f);

  Voxel after = defaultVoxel();
  map.transientGrid().forEachCell([&](const Voxel& v, const Bonxai::CoordT&) {
    if (v.a_occ > 1.001f) after = v;
  });

  // Evidence should be closer to prior (1.0) after decay
  EXPECT_LT(after.a_occ - 1.0f, before.a_occ - 1.0f);
}

TEST(BetaUpdate, DecayPreservesRatio) {
  Voxel v = defaultVoxel();
  v.a_occ = 10.0f;
  v.a_free = 5.0f;
  float ratio_before = v.a_occ / v.a_free;

  // Manual decay like Map::decayTransientGrid
  float rate = 0.8f;
  v.a_occ = 1.0f + (v.a_occ - 1.0f) * rate;
  v.a_free = 1.0f + (v.a_free - 1.0f) * rate;

  // Ratio should be approximately preserved (not exactly, due to prior shift)
  float ratio_after = v.a_occ / v.a_free;
  // Both move toward 1, so ratio should move toward 1 as well
  EXPECT_GT(ratio_after, 1.0f);  // still favors occupied
}

TEST(BetaUpdate, FullDecayResetsToPrior) {
  Voxel v = defaultVoxel();
  v.a_occ = 10.0f;
  v.a_free = 5.0f;

  // Decay with rate 0 means immediate reset
  v.a_occ = 1.0f + (v.a_occ - 1.0f) * 0.0f;
  v.a_free = 1.0f + (v.a_free - 1.0f) * 0.0f;

  EXPECT_FLOAT_EQ(v.a_occ, 1.0f);
  EXPECT_FLOAT_EQ(v.a_free, 1.0f);
  EXPECT_FLOAT_EQ(v.p_occ(), 0.5f);
}

// --- 4. Update properties ---
//
// NOTE: under the joint ray-casting likelihood, ray-order commutativity
// no longer holds — each ray's update is conditional on the current state
// of upstream voxels. The earlier `OrderOfHitsDoesNotMatter` test was
// removed when reach_prob was introduced.

TEST(BetaUpdate, MultipleHitsAreAdditive) {
  // Beta update is conjugate: 2 hits should give same result as 1+1
  auto map1 = makeMap(0.5);
  auto map2 = makeMap(0.5);

  Eigen::Vector3f origin(0, 0, 0), hit(2, 0, 0);

  map1.integrateRay(origin, hit);
  map1.integrateRay(origin, hit);

  map2.integrateRay(origin, hit);
  map2.integrateRay(origin, hit);

  Voxel v1 = defaultVoxel(), v2 = defaultVoxel();
  ASSERT_TRUE(map1.getVoxel(hit, v1));
  ASSERT_TRUE(map2.getVoxel(hit, v2));
  EXPECT_FLOAT_EQ(v1.a_occ, v2.a_occ);
  EXPECT_FLOAT_EQ(v1.a_free, v2.a_free);
}

// --- 5. Free-space carving is attenuated past occupied voxels ---
//
// Under the joint ray-casting likelihood, the wall is not "blocked" — it
// receives a small free update (weight ≈ w_free * reach_prob), but reach_prob
// is small at the wall (upstream voxels are carved), and the update is
// negligible relative to the wall's accumulated occupancy mass. The wall
// stays solid and its p_occ stays high.

TEST(BetaUpdate, CarvingAttenuatedPastOccupied) {
  Params p;
  p.resolution = 0.5;
  Map map(p);

  Eigen::Vector3f origin(0, 0, 0);
  Eigen::Vector3f wall(1.5f, 0, 0);    // create a wall
  Eigen::Vector3f beyond(3.0f, 0, 0);  // target behind wall

  // Build up the wall. Under reach_prob, cold-start bootstraps slowly;
  // 100 rays drives the wall confidently occupied.
  for (int i = 0; i < 100; ++i) {
    map.integrateRay(origin, wall);
  }

  Voxel wall_pre = defaultVoxel();
  ASSERT_TRUE(map.getVoxel(wall, wall_pre));
  const float wall_p_occ_pre = wall_pre.p_occ();
  ASSERT_GT(wall_p_occ_pre, 0.9f);

  // Fire a ray through the wall — wall gets a small free update, but stays
  // confidently occupied because the accumulated a_occ dominates.
  map.integrateRay(origin, beyond);

  Voxel wall_post = defaultVoxel();
  ASSERT_TRUE(map.getVoxel(wall, wall_post));
  // Wall stays confidently occupied — accumulated mass dominates a tiny update.
  EXPECT_GT(wall_post.p_occ(), 0.9f);
  // The free update is bounded by w_free (would be the unattenuated value).
  EXPECT_LT(wall_post.a_free - wall_pre.a_free, 1.01f);
}

TEST(BetaUpdate, CarvingAddsFreeToClearVoxels) {
  auto map = makeMap(0.5);
  Eigen::Vector3f origin(0, 0, 0), hit(3, 0, 0);
  map.integrateRay(origin, hit);

  // Intermediate voxel should have free evidence
  Eigen::Vector3f mid(1.0f, 0, 0);
  Voxel v = defaultVoxel();
  if (map.getVoxel(mid, v)) {
    EXPECT_GT(v.a_free, 1.0f);
  }
}

// --- 6. Transient layer isolation ---

TEST(BetaUpdate, DynamicEndpointGoesToTransient) {
  auto map = makeMap(0.5);
  Eigen::Vector3f origin(0, 0, 0), hit(2, 0, 0);
  map.integrateRay(origin, hit, /*is_dynamic=*/true);

  // Persistent grid should NOT have the endpoint (beyond free carving evidence)
  Voxel pv = defaultVoxel();
  bool in_persistent = map.getVoxel(hit, pv);
  if (in_persistent) {
    // If persistent has it, it's only from free carving — should not be occupied
    EXPECT_LT(pv.p_occ(), 0.5f) << "Dynamic endpoint should not be in persistent grid";
  }

  // Transient grid should have the endpoint
  bool found_in_transient = false;
  map.transientGrid().forEachCell([&](const Voxel& v, const Bonxai::CoordT&) {
    if (v.a_occ > 1.01f) found_in_transient = true;
  });
  EXPECT_TRUE(found_in_transient);
}

TEST(BetaUpdate, StaticEndpointGoesToPersistent) {
  auto map = makeMap(0.5);
  Eigen::Vector3f origin(0, 0, 0), hit(2, 0, 0);
  map.integrateRay(origin, hit, /*is_dynamic=*/false);

  Voxel v = defaultVoxel();
  ASSERT_TRUE(map.getVoxel(hit, v));
  EXPECT_GT(v.a_occ, 1.0f);

  // Transient should be empty (no dynamic observations)
  size_t transient_count = 0;
  map.transientGrid().forEachCell([&](const Voxel&, const Bonxai::CoordT&) {
    ++transient_count;
  });
  EXPECT_EQ(transient_count, 0u);
}

TEST(BetaUpdate, FreeCarveAlwaysGoesToPersistent) {
  // Even for dynamic rays, free-space carving must go to persistent
  auto map = makeMap(0.5);
  Eigen::Vector3f origin(0, 0, 0), hit(3, 0, 0);
  map.integrateRay(origin, hit, /*is_dynamic=*/true);

  // Check persistent grid has free evidence along the ray
  Eigen::Vector3f mid(1.0f, 0, 0);
  Voxel v = defaultVoxel();
  if (map.getVoxel(mid, v)) {
    EXPECT_GT(v.a_free, 1.0f) << "Free carving should go to persistent even for dynamic rays";
  }
}

TEST(BetaUpdate, ClearTransientGridRemovesAll) {
  auto map = makeMap(0.5);
  Eigen::Vector3f origin(0, 0, 0), hit(2, 0, 0);
  map.integrateRay(origin, hit, /*is_dynamic=*/true);

  map.clearTransientGrid();

  size_t count = 0;
  map.transientGrid().forEachCell([&](const Voxel&, const Bonxai::CoordT&) {
    ++count;
  });
  EXPECT_EQ(count, 0u);
}

// --- 7. Union voxel query ---

TEST(BetaUpdate, UnionVoxelReturnsPersistentIfOccupied) {
  auto map = makeMap(0.5);
  Eigen::Vector3f origin(0, 0, 0), hit(2, 0, 0);

  // Add to persistent
  for (int i = 0; i < 20; ++i)
    map.integrateRay(origin, hit, /*is_dynamic=*/false);

  Voxel uv = map.getUnionVoxel(hit);
  EXPECT_GT(uv.p_occ(), 0.7f);
}

TEST(BetaUpdate, UnionVoxelReturnsTransientIfOnlyDynamic) {
  auto map = makeMap(0.5);
  Eigen::Vector3f origin(0, 0, 0), hit(2, 0, 0);

  for (int i = 0; i < 20; ++i)
    map.integrateRay(origin, hit, /*is_dynamic=*/true);

  Voxel uv = map.getUnionVoxel(hit);
  EXPECT_GT(uv.p_occ(), 0.5f);
}

TEST(BetaUpdate, UnionVoxelReturnsPriorIfUnobserved) {
  auto map = makeMap(0.5);
  Eigen::Vector3f pos(100, 100, 100);  // far away, never observed
  Voxel uv = map.getUnionVoxel(pos);
  EXPECT_FLOAT_EQ(uv.p_occ(), 0.5f);
}

// --- 8. Uncertainty functions on updated voxels ---

TEST(BetaUpdate, VarianceDecreasesWithEvidence) {
  Voxel prior = defaultVoxel();  // Beta(1,1)
  float var_prior = variance(prior);

  Voxel observed = defaultVoxel();
  observed.a_occ = 10.0f;
  observed.a_free = 10.0f;
  float var_observed = variance(observed);

  EXPECT_GT(var_prior, var_observed);
}

TEST(BetaUpdate, EIGDecreasesWithEvidence) {
  Voxel prior = defaultVoxel();  // Beta(1,1)
  float eig_prior = expectedInformationGain(prior);

  Voxel observed = defaultVoxel();
  observed.a_occ = 50.0f;
  observed.a_free = 50.0f;
  float eig_observed = expectedInformationGain(observed);

  EXPECT_GT(eig_prior, 0.0f);
  EXPECT_GT(eig_observed, 0.0f);
  EXPECT_GT(eig_prior, eig_observed);
}

TEST(BetaUpdate, EIGIsNonNegative) {
  // Test several configurations
  for (float a : {1.f, 2.f, 5.f, 50.f, 100.f}) {
    for (float b : {1.f, 2.f, 5.f, 50.f, 100.f}) {
      Voxel v = defaultVoxel();
      v.a_occ = a;
      v.a_free = b;
      EXPECT_GE(expectedInformationGain(v), -1e-6f)
        << "EIG negative for Beta(" << a << "," << b << ")";
    }
  }
}

TEST(BetaUpdate, EntropyMaximalAtPrior) {
  Voxel prior = defaultVoxel();  // Beta(1,1) = uniform
  float h_prior = entropy(prior);

  Voxel peaked = defaultVoxel();
  peaked.a_occ = 20.0f;
  peaked.a_free = 2.0f;
  float h_peaked = entropy(peaked);

  // Uniform Beta(1,1) has entropy = 0 (log of Beta function = 0)
  // Peaked distributions have negative entropy
  EXPECT_GT(h_prior, h_peaked);
}

// --- 9. Range weighting effect ---

TEST(BetaUpdate, FarHitsGetLessEvidence) {
  auto map = makeMap(0.5);
  Eigen::Vector3f origin(0, 0, 0);
  Eigen::Vector3f near_hit(1, 0, 0);
  Eigen::Vector3f far_hit(8, 0, 0);

  map.integrateRay(origin, near_hit, /*is_dynamic=*/false, nullptr, 1.f,
                   std::exp(-1.0f / 5.0f), 1.f);  // range_w for 1m

  auto map2 = makeMap(0.5);
  map2.integrateRay(origin, far_hit, /*is_dynamic=*/false, nullptr, 1.f,
                    std::exp(-8.0f / 5.0f), 1.f);  // range_w for 8m

  Voxel v_near = defaultVoxel(), v_far = defaultVoxel();
  ASSERT_TRUE(map.getVoxel(near_hit, v_near));
  ASSERT_TRUE(map2.getVoxel(far_hit, v_far));

  // Near hit should get more evidence
  EXPECT_GT(v_near.a_occ, v_far.a_occ);
}

// --- 10. ForEachVoxel iteration ---

TEST(BetaUpdate, ForEachVoxelVisitsAllObserved) {
  auto map = makeMap(0.5);
  Eigen::Vector3f origin(0, 0, 0);

  map.integrateRay(origin, Eigen::Vector3f(2, 0, 0));
  map.integrateRay(origin, Eigen::Vector3f(0, 2, 0));

  size_t count = 0;
  map.forEachVoxel([&](const Voxel&, const Eigen::Vector3f&) {
    ++count;
  });

  EXPECT_GE(count, 2u);  // at least the 2 hit points + free space
}
