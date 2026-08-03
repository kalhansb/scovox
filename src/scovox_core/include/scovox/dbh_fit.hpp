#pragma once

/// @file dbh_fit.hpp
/// @brief POST-PROCESSING utility: robust circle fit for trunk DBH on a
/// fine TSDF lattice (docs/design/fine_tsdf_band_dbh_2026_07_30.md,
/// "DBH measurement").
///
/// Nothing in the mapping runtime calls this. The mapping node's job ends
/// at producing the fine lattice (locally, and on the wire via the rev-7
/// fine stream); DBH extraction runs afterwards — an offline tool or a
/// separate analysis node — on that map. The header lives in scovox_core
/// because any consumer already links this library to deserialize the map,
/// and because the fine-band tests use the fit as their accuracy metric.
///
/// The TSDF *is* a distance field, so no meshing or zero-crossing
/// extraction is needed: a fine voxel at horizontal distance `d` from the
/// trunk axis should satisfy `tsdf ≈ d − r`. The fit solves robust least
/// squares for `(cx, cy, r)` over every usable fine voxel in the
/// breast-height slab — sub-voxel by construction, since fusion has
/// already averaged per-scan range noise and the fit averages around the
/// circumference.
///
/// Voxel gating: `weight >= min_weight` (enough observations) and
/// `|tsdf| < band_frac · trunc`. Two reasons band_frac defaults well below
/// 1: a voxel clamped at ±trunc carries a *bound* ("at least this far"),
/// not a distance; and band-TAIL voxels are systematically inflated by the
/// projective TSDF bias on a convex trunk (an oblique ray's along-ray
/// distance to its endpoint exceeds the true radial distance, and interior
/// chord voxels sit far closer to the curved surface than to their own
/// endpoint). Near-surface voxels carry little of either bias — and they
/// are exactly where a drift-smeared double wall shows up, so the reported
/// RMS stays a clean pose-quality flag.

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <bonxai/bonxai.hpp>
#include <cmath>
#include <cstdint>
#include <vector>

#include "scovox/refinement_regions.hpp"
#include "scovox/tsdf_voxel.hpp"

namespace scovox {

struct DbhFitParams {
  float min_weight    = 0.5f;
  float band_frac     = 0.5f;   ///< use voxels with |tsdf| < band_frac·trunc
                                ///< (near-surface half — see header comment)
  int   max_iters     = 25;
  float huber_delta   = 0.02f;  ///< m
  int   min_voxels    = 20;
  int   arc_bins      = 24;     ///< azimuth bins for arc coverage (matches
                                ///< the campaign metric's 24-bin convention)
  float inlier_thresh = 0.03f;  ///< m; residual bound for arc-bin fill
  float max_radius    = 2.0f;   ///< m; sanity ceiling on the fitted radius
};

struct DbhFitResult {
  bool     valid        = false;
  float    cx           = 0.f;
  float    cy           = 0.f;
  float    radius       = 0.f;  ///< DBH = 2·radius
  float    rms          = 0.f;  ///< Huber-weighted residual RMS (m)
  float    arc_coverage = 0.f;  ///< filled azimuth bins / arc_bins, inliers
  uint32_t n_voxels     = 0;    ///< voxels that entered the fit
};

/// Fit `(cx, cy, r)` on the fine TSDF grid inside `cyl`'s breast slab.
/// `gate_radius` bounds the horizontal gather (model radius + margin, the
/// same gate integration used, plus the truncation shell); `trunc` is the
/// fine lattice's `sdf_trunc`. Initialised from the cylinder model.
inline DbhFitResult fitDbhCircle(const Bonxai::VoxelGrid<TsdfVoxel>& fine_grid,
                                 const RefinementCylinder& cyl,
                                 float gate_radius, float trunc,
                                 const DbhFitParams& p = {}) {
  DbhFitResult out;

  // Gather usable voxels. The fine grid only exists inside refinement
  // regions, so a full walk filtered per region is proportional to the
  // refined surface, not the map.
  struct Sample { float x, y, tsdf; };
  std::vector<Sample> samples;
  const float h = 0.5f * static_cast<float>(fine_grid.voxelSize());
  const float gather_r  = gate_radius + trunc;
  const float gather_r2 = gather_r * gather_r;
  const float band      = p.band_frac * trunc;
  fine_grid.forEachCell([&](const TsdfVoxel& v, const Bonxai::CoordT& c) {
    if (v.weight < p.min_weight) return;
    if (std::fabs(v.distance) >= band) return;
    const auto pos = fine_grid.coordToPos(c);
    const float x = static_cast<float>(pos.x) + h;
    const float y = static_cast<float>(pos.y) + h;
    const float z = static_cast<float>(pos.z) + h;
    if (z < cyl.z_lo || z > cyl.z_hi) return;
    const float dx = x - cyl.cx, dy = y - cyl.cy;
    if (dx * dx + dy * dy > gather_r2) return;
    samples.push_back({x, y, v.distance});
  });

  out.n_voxels = static_cast<uint32_t>(samples.size());
  if (static_cast<int>(samples.size()) < p.min_voxels) return out;

  // Robust IRLS Gauss–Newton on (cx, cy, r).
  // Residual e_i = (d_i − r) − tsdf_i, d_i = ‖p_i − c‖;
  // Jacobian ∂e/∂(cx, cy, r) = (−u_x, −u_y, −1) with u the unit radial.
  float cx = cyl.cx, cy = cyl.cy;
  float r  = (cyl.radius > 0.01f) ? cyl.radius : 0.1f;
  float rms = 0.f;
  for (int iter = 0; iter < p.max_iters; ++iter) {
    Eigen::Matrix3f H = Eigen::Matrix3f::Zero();
    Eigen::Vector3f g = Eigen::Vector3f::Zero();
    float wsum = 0.f, wsq = 0.f;
    for (const auto& s : samples) {
      const float qx = s.x - cx, qy = s.y - cy;
      const float d = std::sqrt(qx * qx + qy * qy);
      if (d < 1e-4f) continue;
      const float ux = qx / d, uy = qy / d;
      const float e = (d - r) - s.tsdf;
      const float ae = std::fabs(e);
      // Tent prior on surface proximity: the projective/curvature bias
      // grows ~linearly with |tsdf| (see header), so trust decays the same
      // way. Zero-crossing voxels — where a drift double wall shows up —
      // keep full weight, so the reported RMS stays a pose-quality flag.
      const float wt = 1.f - std::fabs(s.tsdf) / band;
      const float wh = (ae <= p.huber_delta) ? 1.f : p.huber_delta / ae;
      const float w = wt * wh;
      const Eigen::Vector3f J(-ux, -uy, -1.f);
      H += w * (J * J.transpose());
      g += w * e * J;
      wsum += w;
      wsq  += w * e * e;
    }
    if (wsum <= 0.f) return out;
    rms = std::sqrt(wsq / wsum);
    // Small absolute damping keeps a thin-arc system solvable; the radius
    // row is fully observed (∂e/∂r = −1 everywhere) so this only steadies
    // the poorly-constrained centre directions.
    H += 1e-3f * wsum * Eigen::Matrix3f::Identity();
    const Eigen::Vector3f step = H.ldlt().solve(-g);
    if (!step.allFinite()) return out;
    cx += step.x(); cy += step.y(); r += step.z();
    if (r <= 0.005f || r > p.max_radius) return out;
    if (step.norm() < 1e-5f) break;
  }
  if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(r) ||
      !std::isfinite(rms))
    return out;

  // Arc coverage over inlier azimuths about the FITTED axis.
  std::vector<uint8_t> bins(static_cast<std::size_t>(p.arc_bins), 0);
  for (const auto& s : samples) {
    const float qx = s.x - cx, qy = s.y - cy;
    const float d = std::sqrt(qx * qx + qy * qy);
    if (d < 1e-4f) continue;
    if (std::fabs((d - r) - s.tsdf) > p.inlier_thresh) continue;
    const float a = std::atan2(qy, qx);  // [-pi, pi]
    int b = static_cast<int>((a + static_cast<float>(M_PI)) /
                             (2.f * static_cast<float>(M_PI)) * p.arc_bins);
    b = std::min(std::max(b, 0), p.arc_bins - 1);
    bins[static_cast<std::size_t>(b)] = 1;
  }
  int filled = 0;
  for (const auto b : bins) filled += b;

  out.valid        = true;
  out.cx           = cx;
  out.cy           = cy;
  out.radius       = r;
  out.rms          = rms;
  out.arc_coverage = static_cast<float>(filled) / static_cast<float>(p.arc_bins);
  return out;
}

}  // namespace scovox
