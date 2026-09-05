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
  // After N hits with w_occ=2, a_occ = prior + 2*N, a_free = prior
  // p_occ = (1 + 2N) / (2 + 2N) -> 1 as N -> inf
  auto map = makeMap(0.5);  // coarse grid so origin != hit cell
  Eigen::Vector3f origin(0, 0, 0), hit(2, 0, 0);
  const int N = 20;
  for (int i = 0; i < N; ++i) {
    map.integrateRay(origin, hit);
  }

  Voxel v = defaultVoxel();
  ASSERT_TRUE(map.getVoxel(hit, v));
  // Exact, not a bound: the deposit is w_occ * range_w * angle_w, this overload
  // defaults both weights to 1, and nothing on the static path recomputes a
  // range decay (range_decay_length is consulted only in carve_free).
  // evidence_saturation (1000) is far above the 41 reached here.
  const float expected_a_occ = 1.0f + 2.0f * N;  // prior 1 + w_occ=2 per hit
  EXPECT_FLOAT_EQ(v.a_occ, expected_a_occ);
  // The free deposit is guarded by `!at_hit`, so the endpoint keeps the
  // untouched prior however the walk reaches it; that is what makes p_occ
  // predictable rather than merely large.  (The DDA also stops short of the
  // endpoint here, but only because sdf_trunc is 0 and the band is empty --
  // the guard is the load-bearing half.)
  EXPECT_FLOAT_EQ(v.a_free, 1.0f);
  EXPECT_FLOAT_EQ(v.p_occ(), expected_a_occ / (expected_a_occ + 1.0f));
}

TEST(BetaUpdate, FreeSpaceUpdateIncreasesAFree) {
  auto map = makeMap(0.5);
  Eigen::Vector3f origin(0, 0, 0), hit(3, 0, 0);
  map.integrateRay(origin, hit);

  // ASSERT, not `if`: a total failure of free carving allocates nothing along
  // the ray, and under a conditional body that is a silent pass -- the one
  // regression this test exists to catch.
  Eigen::Vector3f mid(1, 0, 0);
  Voxel v = defaultVoxel();
  ASSERT_TRUE(map.getVoxel(mid, v)) << "free carving allocated no voxel at the midpoint";
  // Exact: one traversal deposits w_free * range_w = 1.0 on the 1.0 prior.
  EXPECT_FLOAT_EQ(v.a_free, 2.0f);
  EXPECT_FLOAT_EQ(v.a_occ, 1.0f);
  EXPECT_LT(v.p_occ(), 0.5f);
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

// Reads the one transient voxel a single dynamic ray leaves behind.  Both
// decay tests need it, and going through the grid is the point: the earlier
// versions re-implemented `1 + (x-1)*rate` in the test body and then asserted
// properties of the arithmetic they had just written, so a change to the
// production rule could not break them.
static Voxel soleTransientVoxel(Map& map) {
  Voxel found = defaultVoxel();
  size_t n = 0;
  map.transientGrid().forEachCell([&](const Voxel& v, const Bonxai::CoordT&) {
    if (v.a_occ > 1.0f) { found = v; ++n; }
  });
  EXPECT_EQ(n, 1u) << "expected exactly one occupied transient voxel, got " << n;
  return found;
}

TEST(BetaUpdate, DecayPreservesRatio) {
  auto map = makeMap(0.5);
  Eigen::Vector3f origin(0, 0, 0), hit(2, 0, 0);
  map.integrateRay(origin, hit, /*is_dynamic=*/true);

  const Voxel before = soleTransientVoxel(map);
  ASSERT_FLOAT_EQ(before.a_occ, 3.0f);   // prior 1 + w_occ 2
  ASSERT_FLOAT_EQ(before.a_free, 1.0f);  // free carving goes to the persistent grid
  const float ratio_before = before.a_occ / before.a_free;

  map.decayTransientGrid(0.8f);
  const Voxel after = soleTransientVoxel(map);
  const float ratio_after = after.a_occ / after.a_free;

  // Both buckets move toward the prior, so the ratio moves toward 1: above 1
  // still, but strictly below where it started.  The upper bound is the half
  // that distinguishes a decay from a rescale, so both are asserted.  With
  // b at the prior the algebra is 1 + (a-1)r over 1, exact in binary32 here.
  EXPECT_FLOAT_EQ(after.a_occ, 1.0f + 2.0f * 0.8f);
  EXPECT_FLOAT_EQ(after.a_free, 1.0f);
  EXPECT_GT(ratio_after, 1.0f);           // still favors occupied
  EXPECT_LT(ratio_after, ratio_before);   // but moved toward the prior
}

TEST(BetaUpdate, FullDecayResetsToPrior) {
  auto map = makeMap(0.5);
  Eigen::Vector3f origin(0, 0, 0), hit(2, 0, 0);
  map.integrateRay(origin, hit, /*is_dynamic=*/true);
  ASSERT_FLOAT_EQ(soleTransientVoxel(map).a_occ, 3.0f);

  map.decayTransientGrid(0.0f);  // rate 0 means immediate reset

  // soleTransientVoxel's `a_occ > 1.0f` filter finds nothing now, which is the
  // assertion: read the cell back directly instead.
  size_t n = 0;
  Voxel v = defaultVoxel();
  map.transientGrid().forEachCell([&](const Voxel& tv, const Bonxai::CoordT&) {
    v = tv; ++n;
  });
  ASSERT_EQ(n, 1u);
  EXPECT_FLOAT_EQ(v.a_occ, 1.0f);
  EXPECT_FLOAT_EQ(v.a_free, 1.0f);
  EXPECT_FLOAT_EQ(v.p_occ(), 0.5f);
}

// The semantic half of decayTransientGrid -- `sem_cnt[i] *= rate` and
// `a_unk *= rate` -- had no test at all: it is multiplicative where the Beta
// half is affine, so the two cannot cover each other.
TEST(BetaUpdate, DecayScalesSemanticCountsMultiplicatively) {
  auto map = makeMap(0.5);
  Eigen::Vector3f origin(0, 0, 0), hit(2, 0, 0);
  // Three classes against K_TOP slots: with K_TOP == 2 the smallest is routed
  // into a_unk by sparse_add, so a_unk carries real mass and its assertion
  // below is not 0 == 0 * rate.
  std::vector<float> probs(10, 0.0f);
  probs[3] = 0.5f;
  probs[5] = 0.3f;
  probs[7] = 0.2f;
  map.integrateRay(origin, hit, /*is_dynamic=*/true, &probs);

  const Voxel before = soleTransientVoxel(map);
  float cnt_before = 0.f;
  for (int i = 0; i < K_TOP; ++i) cnt_before += before.sem_cnt[i];
  ASSERT_GT(cnt_before, 0.f) << "no semantic mass deposited; the rest is vacuous";
  ASSERT_GT(before.a_unk, 0.f) << "no residual mass; the a_unk assertion would be vacuous";

  const float rate = 0.8f;
  map.decayTransientGrid(rate);
  const Voxel after = soleTransientVoxel(map);
  float cnt_after = 0.f;
  for (int i = 0; i < K_TOP; ++i) cnt_after += after.sem_cnt[i];

  // Multiplicative, so unlike the Beta buckets there is no prior floor: the
  // counts scale straight through zero.
  EXPECT_FLOAT_EQ(cnt_after, cnt_before * rate);
  EXPECT_FLOAT_EQ(after.a_unk, before.a_unk * rate);
}

// --- 4. Update properties ---
//
// NOTE: under the joint ray-casting likelihood, ray-order commutativity
// no longer holds — each ray's update is conditional on the current state
// of upstream voxels. The earlier `OrderOfHitsDoesNotMatter` test was
// removed when reach_prob was introduced.

TEST(BetaUpdate, MultipleHitsAreAdditive) {
  // Beta update is conjugate: N hits add N*w_occ of above-prior mass.  Both
  // maps used to get two rays each, so the only thing asserted was that
  // integrateRay is deterministic -- true of any update rule, additive or not.
  // The one-hit arm is what makes the comparison about additivity.
  auto map1 = makeMap(0.5);
  auto map2 = makeMap(0.5);

  Eigen::Vector3f origin(0, 0, 0), hit(2, 0, 0);

  map1.integrateRay(origin, hit);          // one hit

  map2.integrateRay(origin, hit);          // two hits
  map2.integrateRay(origin, hit);

  Voxel v1 = defaultVoxel(), v2 = defaultVoxel();
  ASSERT_TRUE(map1.getVoxel(hit, v1));
  ASSERT_TRUE(map2.getVoxel(hit, v2));
  // Exact: w_occ = 2 on the 1.0 prior, deposited once and twice.
  EXPECT_FLOAT_EQ(v1.a_occ, 3.0f);
  EXPECT_FLOAT_EQ(v2.a_occ, 5.0f);
  EXPECT_FLOAT_EQ(v2.a_occ - 1.0f, 2.0f * (v1.a_occ - 1.0f));
  // The endpoint takes no free mass on either arm, so the prior is untouched.
  EXPECT_FLOAT_EQ(v1.a_free, 1.0f);
  EXPECT_FLOAT_EQ(v2.a_free, 1.0f);
}

// --- 5. Carving a beam through an occupied voxel leaves it solid ---
//
// The wall guard is OFF by default (we trust the most recent scan). A beam that
// passes through the wall to a farther return therefore deposits ONE free
// increment (≤ w_free) on the wall voxel — but the wall stays confidently
// occupied because its accumulated a_occ (100 rays) dwarfs a single free update.
// This is the intended dynamic-clearing behaviour: repeated pass-through
// eventually clears a stale obstacle, one scan does not.

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

  // Fire one ray through the wall — with the guard off the wall gets a single
  // free increment, but stays confidently occupied (accumulated a_occ dominates).
  map.integrateRay(origin, beyond);

  Voxel wall_post = defaultVoxel();
  ASSERT_TRUE(map.getVoxel(wall, wall_post));
  // Wall stays confidently occupied — 100 rays of a_occ dominate one free update.
  EXPECT_GT(wall_post.p_occ(), 0.9f);
  // Exactly one free traversal this scan → bounded by w_free.
  EXPECT_LT(wall_post.a_free - wall_pre.a_free, 1.01f);
}

TEST(BetaUpdate, CarvingAddsFreeToClearVoxels) {
  auto map = makeMap(0.5);
  Eigen::Vector3f origin(0, 0, 0), hit(3, 0, 0);
  map.integrateRay(origin, hit);

  // ASSERT, not `if`: see FreeSpaceUpdateIncreasesAFree.
  Eigen::Vector3f mid(1.0f, 0, 0);
  Voxel v = defaultVoxel();
  ASSERT_TRUE(map.getVoxel(mid, v)) << "carving allocated no intermediate voxel";
  EXPECT_FLOAT_EQ(v.a_free, 2.0f);
}

// --- 6. Transient layer isolation ---

TEST(BetaUpdate, DynamicEndpointGoesToTransient) {
  auto map = makeMap(0.5);
  Eigen::Vector3f origin(0, 0, 0), hit(2, 0, 0);
  map.integrateRay(origin, hit, /*is_dynamic=*/true);

  // Persistent grid should NOT have the endpoint occupied.  The lookup may or
  // may not find a cell there (free carving skips the endpoint, so nothing
  // allocates it on this path), and either answer is consistent with the
  // contract -- what is not is an occupied one.
  Voxel pv = defaultVoxel();
  if (map.getVoxel(hit, pv)) {
    EXPECT_LT(pv.p_occ(), 0.5f) << "Dynamic endpoint should not be in persistent grid";
  }

  // Transient grid should have the endpoint -- AT THE ENDPOINT.  The coord was
  // discarded before, so an occupied transient voxel anywhere satisfied a test
  // whose name asserts a location.
  const auto k_hit = map.transientGrid().posToCoord(hit.x(), hit.y(), hit.z());
  bool found_at_endpoint = false;
  map.transientGrid().forEachCell([&](const Voxel& v, const Bonxai::CoordT& c) {
    if (c == k_hit) {
      found_at_endpoint = true;
      EXPECT_FLOAT_EQ(v.a_occ, 3.0f);   // prior 1 + w_occ 2, one dynamic hit
    }
  });
  EXPECT_TRUE(found_at_endpoint);
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

  // ASSERT, not `if`.  The regression this test names -- free carving routed
  // to the transient grid -- makes the persistent lookup fail, which under a
  // conditional body skipped the only assertion and passed.
  Eigen::Vector3f mid(1.0f, 0, 0);
  Voxel v = defaultVoxel();
  ASSERT_TRUE(map.getVoxel(mid, v))
      << "Free carving should go to persistent even for dynamic rays";
  // Exact, and NOT the 2.0 the two static-ray carve tests above see.  The
  // dynamic branch of integrateRay calls the two-argument `carve_free`, which
  // derives its own range weight exp(-|hit - origin| / range_decay_length)
  // (scovoxmap.cpp:69-74); the static branch instead carves with the range_w
  // its caller supplies, which defaults to 1.  Same geometry, different free
  // mass -- assert the value the dynamic path actually owes.
  const float range_w = std::exp(-(hit - origin).norm() / map.params().range_decay_length);
  EXPECT_FLOAT_EQ(v.a_free, 1.0f + map.params().w_free * range_w);
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

  map.integrateRay(origin, near_hit, /*is_dynamic=*/false, nullptr,
                   std::exp(-1.0f / 5.0f), 1.f);  // range_w for 1m

  auto map2 = makeMap(0.5);
  map2.integrateRay(origin, far_hit, /*is_dynamic=*/false, nullptr,
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

  // The name says ALL, so count them: at resolution 0.5 each ray allocates the
  // five coords 0..4 along its axis (the walk stops short of k_far, and the
  // explicit follow-up visit covers it), and the two rays share the origin
  // cell.  5 + 5 - 1 = 9.  `>= 2` tolerated forEachVoxel skipping seven.
  EXPECT_EQ(count, 9u);
}
