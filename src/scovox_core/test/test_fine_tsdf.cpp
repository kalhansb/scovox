/// @file
/// @brief Fine-resolution TSDF band gate
/// (docs/design/fine_tsdf_band_dbh_2026_07_30.md): refinement-region
/// registry, gated two-lattice integration, per-scan anchor
/// re-registration (drift absorption), the raw-return fine-only path
/// (refineHit), and the rev-7 wire fine stream + merge. The DBH circle
/// fit (dbh_fit.hpp — a POST-PROCESSING utility, not called by the
/// mapping runtime) doubles as the accuracy metric here.

#include <gtest/gtest.h>
#include <Eigen/Core>
#include <cmath>
#include <random>
#include <vector>

#include "scovox/binary_serializer.hpp"
#include "scovox/consensus_merge.hpp"
#include "scovox/dbh_fit.hpp"
#include "scovox/refinement_regions.hpp"
#include "scovox/scovox_map_split.hpp"

namespace {

constexpr float kPi = 3.14159265358979f;

scovox::ScovoxMapSplit::Params fineParams(bool anchor = true) {
  scovox::ScovoxMapSplit::Params p;
  p.resolution         = 0.10;
  p.tsdf.sdf_trunc     = 0.30f;
  p.fine_ratio_log2    = 2;      // res_fine = 0.025
  p.fine_anchor_enable = anchor;
  return p;
}

// The synthetic trunk every integration test orbits: axis (2.0, 0.0),
// radius 0.15 m, DBH slab z in [0.9, 1.7].
scovox::RefinementCylinder trunkRegion() {
  scovox::RefinementCylinder c;
  c.id     = 7;
  c.cx     = 2.0f;
  c.cy     = 0.0f;
  c.radius = 0.15f;
  c.z_lo   = 0.9f;
  c.z_hi   = 1.7f;
  return c;
}

/// Integrate `n_scans` simulated scans of the trunk from a sensor orbiting
/// at `orbit_r`, with a per-scan rigid pose-drift vector growing linearly
/// to `drift_end` (applied to origin AND endpoints — exactly what odometry
/// drift does to a scan). Each scan covers the trunk arc facing the sensor.
void orbitTrunk(scovox::ScovoxMapSplit& m, float drift_end_x, float drift_end_y,
                int n_scans = 48, float orbit_r = 3.0f, float noise_sigma = 0.005f) {
  const auto cyl = trunkRegion();
  std::mt19937 rng(42);
  std::normal_distribution<float> noise(0.f, noise_sigma);
  for (int s = 0; s < n_scans; ++s) {
    const float t = static_cast<float>(s) / static_cast<float>(n_scans - 1);
    const Eigen::Vector3f drift(t * drift_end_x, t * drift_end_y, 0.f);
    const float theta = 2.f * kPi * t;  // sensor bearing about the trunk
    const Eigen::Vector3f origin =
        Eigen::Vector3f(cyl.cx + orbit_r * std::cos(theta),
                        cyl.cy + orbit_r * std::sin(theta), 1.3f) + drift;
    m.beginCarveFrame();
    // Surface points on the arc facing the sensor (±60° about the
    // sensor-facing azimuth), across the slab.
    for (int a = -12; a <= 12; ++a) {
      const float phi = theta + kPi + static_cast<float>(a) * (kPi / 36.f);
      for (int zi = 0; zi < 8; ++zi) {
        const float z = 0.95f + 0.1f * static_cast<float>(zi);
        const float r_meas = cyl.radius + noise(rng);
        const Eigen::Vector3f hit =
            Eigen::Vector3f(cyl.cx + r_meas * std::cos(phi),
                            cyl.cy + r_meas * std::sin(phi), z) + drift;
        m.integrateHit(origin, hit, nullptr, 1.0f);
      }
    }
    m.flushCarveFrame();
  }
}

/// The post-processing step, exactly as an offline consumer would run it:
/// fitDbhCircle on the fine lattice against region 7's registered model.
scovox::DbhFitResult fitTrunk(scovox::ScovoxMapSplit& m) {
  const auto& reg = m.refinementRegions();
  const int idx = reg.indexOf(7);
  EXPECT_GE(idx, 0);
  return scovox::fitDbhCircle(m.fineTsdf().grid(), reg.cylinder(idx),
                              reg.gateRadius(idx),
                              m.fineTsdf().params().sdf_trunc);
}

}  // namespace

// ---------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------

TEST(RefinementRegions, LookupGateAndLifecycle) {
  scovox::RefinementRegions reg;
  auto cyl = trunkRegion();
  const int idx = reg.add(cyl, /*margin=*/0.15f);  // gate radius 0.30
  ASSERT_GE(idx, 0);
  EXPECT_EQ(reg.size(), 1u);

  // Inside gate + slab.
  EXPECT_EQ(reg.lookup(2.10f, 0.10f, 1.3f), idx);
  // Outside gate radius.
  EXPECT_EQ(reg.lookup(2.45f, 0.0f, 1.3f), -1);
  // Inside radius, outside z band.
  EXPECT_EQ(reg.lookup(2.0f, 0.0f, 0.5f), -1);
  EXPECT_EQ(reg.lookup(2.0f, 0.0f, 2.0f), -1);
  // Far away (different hash cell).
  EXPECT_EQ(reg.lookup(50.f, 50.f, 1.3f), -1);

  // Replace same id: bigger gate, same slot.
  cyl.radius = 0.30f;
  EXPECT_EQ(reg.add(cyl, 0.15f), idx);
  EXPECT_EQ(reg.size(), 1u);
  EXPECT_EQ(reg.lookup(2.40f, 0.0f, 1.3f), idx);

  // Remove: lookups go dark; unknown id is reported.
  EXPECT_TRUE(reg.remove(cyl.id));
  EXPECT_FALSE(reg.remove(cyl.id));
  EXPECT_EQ(reg.lookup(2.0f, 0.0f, 1.3f), -1);
  EXPECT_TRUE(reg.empty());
}

TEST(RefinementRegions, SpansMultipleHashCells) {
  // A region straddling a 4 m cell boundary must be found from both sides.
  scovox::RefinementRegions reg(4.0f);
  scovox::RefinementCylinder c;
  c.id = 1; c.cx = 4.0f; c.cy = 0.0f; c.radius = 0.2f; c.z_lo = 0.f; c.z_hi = 2.f;
  const int idx = reg.add(c, 0.15f);
  EXPECT_EQ(reg.lookup(3.90f, 0.0f, 1.0f), idx);
  EXPECT_EQ(reg.lookup(4.10f, 0.0f, 1.0f), idx);
}

// ---------------------------------------------------------------------
// Gated two-lattice integration
// ---------------------------------------------------------------------

TEST(FineTsdf, DisabledByDefaultAndLatticeGeometry) {
  scovox::ScovoxMapSplit::Params p;
  p.resolution = 0.10;
  scovox::ScovoxMapSplit off(p);
  EXPECT_FALSE(off.fineEnabled());
  EXPECT_EQ(off.addRefinementRegion(trunkRegion()), -1);

  scovox::ScovoxMapSplit on(fineParams());
  EXPECT_TRUE(on.fineEnabled());
  EXPECT_EQ(on.fineRatioLog2(), 2);
  EXPECT_NEAR(on.fineResolution(), 0.025, 1e-9);
  EXPECT_NEAR(on.fineTsdf().params().sdf_trunc, 3 * 0.025f, 1e-6f);
}

TEST(FineTsdf, FineVoxelsOnlyInsideRegisteredRegions) {
  scovox::ScovoxMapSplit m(fineParams());
  m.addRefinementRegion(trunkRegion());

  // A hit far from any region: coarse grids fill, fine stays empty.
  m.beginCarveFrame();
  m.integrateHit(Eigen::Vector3f(0, 5, 1), Eigen::Vector3f(3, 5, 1), nullptr, 1.f);
  m.flushCarveFrame();
  EXPECT_GT(m.tsdfVoxelCount(), 0u);
  EXPECT_EQ(m.fineVoxelCount(), 0u);

  // A hit on the registered trunk: fine band fills too.
  orbitTrunk(m, 0.f, 0.f, /*n_scans=*/4);
  EXPECT_GT(m.fineVoxelCount(), 0u);
  EXPECT_GT(m.fineLastFrameRays(), 0u);

  // Every fine voxel lies inside the gate cylinder + truncation shell.
  const auto& cyl = trunkRegion();
  const float max_r = 0.15f + 0.15f + 3 * 0.025f + 0.05f;
  m.fineTsdf().forEachVoxel([&](const scovox::TsdfVoxel&,
                                const Eigen::Vector3f& pos) {
    const float dx = pos.x() - cyl.cx, dy = pos.y() - cyl.cy;
    EXPECT_LE(std::sqrt(dx * dx + dy * dy), max_r);
  });
}

TEST(FineTsdf, ImmediateModeWithoutCarveFrame) {
  // Direct integrateHit callers (no begin/flush bracket) still refine —
  // uncorrected, mirroring SemSplitMap's immediate carve semantics.
  scovox::ScovoxMapSplit m(fineParams());
  m.addRefinementRegion(trunkRegion());
  m.integrateHit(Eigen::Vector3f(0, 0, 1.3f),
                 Eigen::Vector3f(2.0f - 0.15f, 0, 1.3f), nullptr, 1.f);
  EXPECT_GT(m.fineVoxelCount(), 0u);
}

TEST(FineTsdf, DynamicAndGeometryOffRaysNeverRefine) {
  scovox::ScovoxMapSplit m(fineParams());
  m.addRefinementRegion(trunkRegion());
  const Eigen::Vector3f o(0, 0, 1.3f), h(2.0f - 0.15f, 0, 1.3f);

  m.beginCarveFrame();
  m.integrateHit(o, h, nullptr, 1.f, /*is_dynamic=*/true);
  scovox::HitWeights prof{};
  prof.geometry_off = true;
  m.integrateHit(o, h, nullptr, 1.f, /*is_dynamic=*/false, &prof);
  m.flushCarveFrame();
  EXPECT_EQ(m.fineVoxelCount(), 0u);
}

// ---------------------------------------------------------------------
// Anchor fit
// ---------------------------------------------------------------------

TEST(AnchorFit, RecoversKnownShift) {
  // Points on a circle whose centre is offset by `o` from the model: the
  // fit must return Δ ≈ −o (damping shrinks it by ~λ/(λ+eig) ≈ 9 %).
  const Eigen::Vector2f c(2.0f, 0.0f);
  const float r = 0.15f;
  const Eigen::Vector2f o(0.05f, -0.03f);
  std::vector<Eigen::Vector2f> pts;
  for (int a = -12; a <= 12; ++a) {  // 120° arc — one scan's view
    const float phi = kPi + static_cast<float>(a) * (kPi / 36.f);
    pts.emplace_back(c + o + r * Eigen::Vector2f(std::cos(phi), std::sin(phi)));
  }
  const auto d = scovox::fitCylinderAnchorShift(pts, c, r);
  ASSERT_TRUE(d.has_value());
  EXPECT_NEAR(d->x(), -o.x(), 0.015f);
  EXPECT_NEAR(d->y(), -o.y(), 0.015f);
}

TEST(AnchorFit, RejectsThinAndWildFits) {
  const Eigen::Vector2f c(2.0f, 0.0f);
  const float r = 0.15f;
  std::vector<Eigen::Vector2f> few = {{2.15f, 0.f}, {2.14f, 0.04f}, {2.12f, 0.07f}};
  EXPECT_FALSE(scovox::fitCylinderAnchorShift(few, c, r).has_value());

  // A cluster 0.6 m off the model wants a shift beyond max_shift → reject.
  std::vector<Eigen::Vector2f> wild;
  for (int a = -12; a <= 12; ++a) {
    const float phi = kPi + static_cast<float>(a) * (kPi / 36.f);
    wild.emplace_back(c + Eigen::Vector2f(0.6f, 0.f) +
                      r * Eigen::Vector2f(std::cos(phi), std::sin(phi)));
  }
  EXPECT_FALSE(scovox::fitCylinderAnchorShift(wild, c, r).has_value());
}

// ---------------------------------------------------------------------
// End-to-end: drift smears the un-anchored fine map; the anchor absorbs it
// ---------------------------------------------------------------------

TEST(FineTsdf, AnchorAbsorbsOdometryDrift) {
  const float kDriftX = 0.08f;  // 8 cm accumulated over the orbit

  scovox::ScovoxMapSplit anchored(fineParams(/*anchor=*/true));
  anchored.addRefinementRegion(trunkRegion());
  orbitTrunk(anchored, kDriftX, 0.f);
  const auto fit_a = fitTrunk(anchored);

  scovox::ScovoxMapSplit raw(fineParams(/*anchor=*/false));
  raw.addRefinementRegion(trunkRegion());
  orbitTrunk(raw, kDriftX, 0.f);
  const auto fit_r = fitTrunk(raw);

  ASSERT_TRUE(fit_a.valid);
  EXPECT_GT(fit_a.n_voxels, 100u);
  EXPECT_GE(fit_a.arc_coverage, 0.75f);
  // Anchored radius lands inside the DBH budget.
  EXPECT_NEAR(fit_a.radius, 0.15f, 0.015f);
  // A fixed-direction drift displaces each azimuth's arc by the drift at
  // its observation time, so the un-anchored surface is a distorted,
  // TRANSLATED circle: its fitted radius can stay near-true while the
  // residual and the recovered centre absorb the damage. Those are the
  // discriminating metrics.
  if (fit_r.valid) {
    EXPECT_LT(fit_a.rms, fit_r.rms);
    const float ca = std::hypot(fit_a.cx - 2.0f, fit_a.cy - 0.0f);
    const float cr = std::hypot(fit_r.cx - 2.0f, fit_r.cy - 0.0f);
    EXPECT_LE(ca, cr + 1e-3f);
  }
}

TEST(FineTsdf, CleanOrbitDbhWithinBudget) {
  scovox::ScovoxMapSplit m(fineParams());
  m.addRefinementRegion(trunkRegion());
  orbitTrunk(m, 0.f, 0.f);
  const auto fit = fitTrunk(m);
  ASSERT_TRUE(fit.valid);
  EXPECT_NEAR(fit.radius, 0.15f, 0.01f);
  EXPECT_NEAR(fit.cx, 2.0f, 0.01f);
  EXPECT_NEAR(fit.cy, 0.0f, 0.01f);
  EXPECT_GE(fit.arc_coverage, 0.9f);
  // The clean-orbit RMS floor on this synthetic geometry is ~0.0255: the
  // ±60° scan arcs put oblique rays into the band whose projective bias
  // survives tent-weighting at the band's inner edge. It was ~0.022 under the
  // approximate Bresenham traversal, which skipped a share of those very
  // voxels; the exact DDA visits all of them, so more biased band-edge samples
  // enter the fit. The fitted geometry did not degrade with it — radius error
  // fell 4.09 mm -> 0.95 mm, cy improved, cx moved within tolerance and arc
  // coverage stayed at 1.0. Only the residual scatter grew.
  // This bound is a regression tripwire just above that floor, not the
  // engineering requirement: the model-refresh gate is 0.03, and the drift
  // test asserts the anchored map beats the smeared one on this same metric.
  EXPECT_LT(fit.rms, 0.027f);
}

TEST(FineTsdf, ExternalModelRefreshViaReAdd) {
  // The refresh loop lives OUTSIDE the mapper: a downstream estimator fits
  // on the fine map and re-publishes the region with the fitted model
  // (same id → same slot, in-place cylinder update). This contracts the
  // coarse-detector radius bias out of the anchor fit — see the design
  // doc's pose section.
  scovox::ScovoxMapSplit m(fineParams());
  const int idx0 = m.addRefinementRegion(trunkRegion());
  orbitTrunk(m, 0.f, 0.f, /*n_scans=*/24);
  const auto fit = fitTrunk(m);
  ASSERT_TRUE(fit.valid);

  auto refreshed = trunkRegion();
  refreshed.cx = fit.cx;
  refreshed.cy = fit.cy;
  refreshed.radius = fit.radius;
  EXPECT_EQ(m.addRefinementRegion(refreshed), idx0);  // slot-stable
  const auto& reg = m.refinementRegions();
  const int idx = reg.indexOf(7);
  ASSERT_EQ(idx, idx0);
  EXPECT_FLOAT_EQ(reg.cylinder(idx).radius, fit.radius);
  EXPECT_FLOAT_EQ(reg.cylinder(idx).cx, fit.cx);
}

TEST(FineTsdf, RefineHitIsFineBandOnly) {
  // The raw-return path: full-density sensor points routed around the
  // node's scan downsample. Fine lattice fills; the coarse substrates see
  // NOTHING (the downsampled medoid carries the coarse update, so raw
  // returns here must not double-count).
  scovox::ScovoxMapSplit m(fineParams());
  m.addRefinementRegion(trunkRegion());
  const Eigen::Vector3f o(0, 0, 1.3f);

  m.refineHit(o, Eigen::Vector3f(2.0f - 0.15f, 0, 1.3f));  // in-region
  EXPECT_GT(m.fineVoxelCount(), 0u);
  EXPECT_EQ(m.tsdfVoxelCount(), 0u);

  const auto n = m.fineVoxelCount();
  m.refineHit(o, Eigen::Vector3f(2.0f, 0, 3.0f));  // above the slab
  EXPECT_EQ(m.fineVoxelCount(), n);

  // Inside a carve frame it stages with everything else (anchor-eligible).
  m.beginCarveFrame();
  m.refineHit(o, Eigen::Vector3f(2.0f - 0.15f, 0.02f, 1.3f));
  m.flushCarveFrame();
  EXPECT_GE(m.fineLastFrameRays(), 1u);
  EXPECT_EQ(m.tsdfVoxelCount(), 0u);
}

// ---------------------------------------------------------------------
// Wire (rev 7) + merge
// ---------------------------------------------------------------------

TEST(FineWire, RoundTripAndValidation) {
  scovox::BinarySerializer::Frame f;
  f.resolution      = 0.10f;
  f.fine_ratio_log2 = 2;
  f.fine_tsdf_deltas.push_back({{40, -3, 52}, {0.02f, 6.f}});
  f.fine_tsdf_deltas.push_back({{41, -3, 52}, {-0.01f, 4.f}});
  f.beta_deltas.push_back({{4, 0, 13}, {3.f, 1.f}});

  const auto blob = scovox::BinarySerializer::serialize(f);
  const auto g = scovox::BinarySerializer::deserialize(blob);
  EXPECT_EQ(g.fine_ratio_log2, 2);
  ASSERT_EQ(g.fine_tsdf_deltas.size(), 2u);
  EXPECT_EQ(g.fine_tsdf_deltas[0].coord.x, 40);
  EXPECT_FLOAT_EQ(g.fine_tsdf_deltas[0].data.distance, 0.02f);
  EXPECT_FLOAT_EQ(g.fine_tsdf_deltas[1].data.weight, 4.f);
  ASSERT_EQ(g.beta_deltas.size(), 1u);

  // Fine records without a fine lattice tag are malformed on both ends.
  scovox::BinarySerializer::Frame bad = f;
  bad.fine_ratio_log2 = 0;
  EXPECT_THROW(scovox::BinarySerializer::serialize(bad), std::runtime_error);
}

TEST(FineWire, MergeCurlessLevoyAndLatticeCheck) {
  scovox::BinarySerializer::Frame a, b;
  a.resolution = b.resolution = 0.10f;
  a.fine_ratio_log2 = b.fine_ratio_log2 = 2;
  a.fine_tsdf_deltas.push_back({{10, 0, 0}, {0.02f, 2.f}});
  b.fine_tsdf_deltas.push_back({{10, 0, 0}, {-0.01f, 4.f}});
  b.fine_tsdf_deltas.push_back({{11, 0, 0}, {0.03f, 1.f}});

  const auto fused = scovox::mergeFrames(a, b);
  EXPECT_EQ(fused.fine_ratio_log2, 2);
  ASSERT_EQ(fused.fine_tsdf_deltas.size(), 2u);
  for (const auto& d : fused.fine_tsdf_deltas) {
    if (d.coord.x == 10) {
      EXPECT_FLOAT_EQ(d.data.weight, 6.f);
      EXPECT_NEAR(d.data.distance, (0.02f * 2.f - 0.01f * 4.f) / 6.f, 1e-6f);
    } else {
      EXPECT_EQ(d.coord.x, 11);
      EXPECT_FLOAT_EQ(d.data.weight, 1.f);
    }
  }

  // One side with no fine grid is compatible; two different lattices are not.
  scovox::BinarySerializer::Frame none;
  none.resolution = 0.10f;
  EXPECT_NO_THROW(scovox::mergeFrames(a, none));
  EXPECT_EQ(scovox::mergeFrames(a, none).fine_ratio_log2, 2);
  scovox::BinarySerializer::Frame k3 = b;
  k3.fine_ratio_log2 = 3;
  EXPECT_THROW(scovox::mergeFrames(a, k3), std::runtime_error);
}
