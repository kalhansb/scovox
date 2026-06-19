/// @file
/// @brief SLIM-VDB-equivalent TSDF integration. Verbatim port of
/// `slimvdb::VDBVolume::Integrate` (CLOSED variant), using Bonxai's
/// `VoxelGrid<TsdfVoxel>` and `scovox::RayIterator` instead of OpenVDB's
/// `FloatGrid` and `openvdb::math::DDA`. The Bresenham-vs-OpenVDB DDA
/// difference and the `k_hit` revisit guard are documented as acknowledged
/// parity gaps in §1.1 of the design plan.

#include "scovox/tsdf_map.hpp"

#include <algorithm>
#include <cmath>

#include "scovox/ray_iterator.hpp"

namespace scovox {

namespace {

/// Strictly clamp params to valid range. sdf_trunc <= 0 is not a supported
/// configuration for this class — see plan §4.3.
TsdfMap::Params sanitise(TsdfMap::Params p) {
  if (p.sdf_trunc <= 0.f)   p.sdf_trunc   = 0.15f;
  if (p.resolution <= 0.0)  p.resolution  = 0.05;
  return p;
}

}  // namespace

TsdfMap::TsdfMap(const Params& p)
    : params_(sanitise(p))
    , grid_(params_.resolution, params_.inner_bits, params_.leaf_bits)
    , acc_(grid_.createAccessor())
    , touched_() {}

// ---------------------------------------------------------------------------
// Weight function factories
// ---------------------------------------------------------------------------

TsdfMap::WeightFn TsdfMap::constant(float w) {
  return [w](float /*sdf*/) { return w; };
}

TsdfMap::WeightFn TsdfMap::linear(float w_max, float trunc) {
  return [w_max, trunc](float sdf) {
    if (trunc <= 0.f) return w_max;
    const float t = std::min(1.f, std::fabs(sdf) / trunc);
    return std::max(0.f, w_max * (1.f - t));
  };
}

TsdfMap::WeightFn TsdfMap::exponential(float sigma) {
  return [sigma](float sdf) {
    if (sigma <= 0.f) return 1.f;
    return std::exp(-(sdf * sdf) / (2.f * sigma * sigma));
  };
}

TsdfMap::WeightFn TsdfMap::rangeDecay(float L_metres, float ray_depth_metres) {
  // Ablation hatch only — bakes a per-ray scalar into a per-voxel function.
  // Production SCovox row never calls this; the per-ray weighting moved to
  // SemBetaMap where it conceptually belongs.
  if (L_metres <= 0.f) return constant(1.0f);
  const float w = std::exp(-ray_depth_metres / L_metres);
  return constant(w);
}

// ---------------------------------------------------------------------------
// Integration
// ---------------------------------------------------------------------------

void TsdfMap::integrateRay(const Eigen::Vector3f& origin,
                           const Eigen::Vector3f& endpoint,
                           const WeightFn&        weight_fn) {
  integrateRayImpl(origin, endpoint, weight_fn);
}

void TsdfMap::integrateRayImpl(const Eigen::Vector3f& origin,
                               const Eigen::Vector3f& endpoint,
                               const WeightFn&        weight_fn) {
  const Eigen::Vector3f d = endpoint - origin;
  const float depth = d.norm();
  if (depth < 1e-4f) return;
  const Eigen::Vector3f u = d / depth;
  const float trunc = params_.sdf_trunc;
  const float h     = 0.5f * static_cast<float>(params_.resolution);

  // SLIM-VDB DDA range:
  //   space_carving=false (default): [endpoint - trunc·û, endpoint + trunc·û]
  //   space_carving=true:            [origin,             endpoint + trunc·û]
  const Eigen::Vector3f start_pos = params_.space_carving
      ? origin
      : Eigen::Vector3f(endpoint - trunc * u);
  const Eigen::Vector3f end_pos = endpoint + trunc * u;

  const CoordT k0    = grid_.posToCoord(start_pos.x(), start_pos.y(), start_pos.z());
  const CoordT k_far = grid_.posToCoord(end_pos.x(),   end_pos.y(),   end_pos.z());
  const CoordT k_hit = grid_.posToCoord(endpoint.x(),  endpoint.y(),  endpoint.z());

  if (k0 == k_far) {
    // Degenerate ray inside one voxel — visit it exactly once.
    visit(k0, origin, endpoint, h, trunc, weight_fn);
    return;
  }

  // Bresenham DDA can skip k_hit on oblique rays where the picked-axis path
  // doesn't intersect the true endpoint voxel. Track and revisit. See the
  // matching guard in scovox::Map::fused_integrate_ray_static (kept for the
  // SCovox row's geometry parity with itself across the refactor).
  bool k_hit_visited = false;

  RayIterator(k0, k_far, [&](const CoordT& c) -> bool {
    if (c == k_hit) k_hit_visited = true;
    visit(c, origin, endpoint, h, trunc, weight_fn);
    return true;
  });

  // RayIterator stops one step short of `key_end`; visit explicitly.
  if (k_far != k0) visit(k_far, origin, endpoint, h, trunc, weight_fn);
  // And revisit k_hit if Bresenham's picked path missed it.
  if (!k_hit_visited && k_hit != k_far && k_hit != k0) {
    visit(k_hit, origin, endpoint, h, trunc, weight_fn);
  }
}

void TsdfMap::visit(const CoordT&          c,
                    const Eigen::Vector3f& origin,
                    const Eigen::Vector3f& endpoint,
                    float                  h,
                    float                  trunc,
                    const WeightFn&        weight_fn) {
  // 1. Voxel CENTRE in world space (NOT lower corner) — matches
  // SLIM-VDB's `GetVoxelCenter = indexToWorld(c) + voxel_size/2`.
  const auto p = grid_.coordToPos(c);
  const Eigen::Vector3f vc(static_cast<float>(p.x) + h,
                           static_cast<float>(p.y) + h,
                           static_cast<float>(p.z) + h);

  // 2. Signed Euclidean SDF — slimvdb::ComputeSDF.
  const Eigen::Vector3f v_voxel_origin = vc - origin;
  const Eigen::Vector3f v_point_voxel  = endpoint - vc;
  const float dist = v_point_voxel.norm();
  const float proj = v_voxel_origin.dot(v_point_voxel);
  if (std::fabs(proj) < 1e-12f) {
    // Voxel exactly on the surface — sign undefined. Skip.
    return;
  }
  const float sign = (proj > 0.f) ? 1.f : -1.f;
  const float sdf  = sign * dist;

  // 3-4. Band gate + clamp + weight + Curless–Levoy. Shared with the
  // fused walker via applyBandUpdate so both paths converge on identical
  // per-voxel TSDF state for the same (c, sdf).
  (void)trunc;  // applyBandUpdate reads trunc from params_ — same value.
  applyBandUpdate(c, sdf, weight_fn);
}

void TsdfMap::applyBandUpdate(const CoordT&    c,
                              float            sdf,
                              const WeightFn&  weight_fn) {
  const float trunc = params_.sdf_trunc;

  // SLIM-VDB band gate: drop voxels too far behind the surface.
  // (When space_carving=true, in-front voxels with sdf > +trunc are also
  // visited; they receive the clamped d_clamped = +trunc value below.)
  if (sdf <= -trunc) return;

  // Truncation clamp + per-voxel weight + Curless–Levoy update.
  const float d_clamped = std::min(trunc, std::max(-trunc, sdf));
  const float w         = weight_fn(sdf);
  if (w <= 0.f) return;

  TsdfVoxel* v = acc_.value(c, /*create_if_missing=*/true);
  // Bonxai's value(c, true) zero-inits a new cell — exactly the
  // {distance:0, weight:0} unobserved state, so the running average
  // below is well-defined even on first touch (w_old = 0 ⇒ d_new = d_clamped).
  const float w_new = v->weight + w;
  if (w_new > 0.f) {
    v->distance = (v->distance * v->weight + d_clamped * w) / w_new;
    v->weight   = w_new;
    touched_.push_back(c);
  }
}

// ---------------------------------------------------------------------------
// Touched-set drain (Q7)
// ---------------------------------------------------------------------------

std::vector<TsdfMap::CoordT> TsdfMap::drainTouched() {
  std::sort(touched_.begin(), touched_.end(),
            [](const CoordT& a, const CoordT& b) {
              if (a.x != b.x) return a.x < b.x;
              if (a.y != b.y) return a.y < b.y;
              return a.z < b.z;
            });
  touched_.erase(
      std::unique(touched_.begin(), touched_.end(),
                  [](const CoordT& a, const CoordT& b) {
                    return a.x == b.x && a.y == b.y && a.z == b.z;
                  }),
      touched_.end());
  std::vector<CoordT> out = std::move(touched_);
  touched_.clear();
  return out;
}

// ---------------------------------------------------------------------------
// Voxel queries
// ---------------------------------------------------------------------------

std::optional<TsdfVoxel> TsdfMap::getVoxel(const Eigen::Vector3f& pos) const {
  auto const_acc = grid_.createConstAccessor();
  const CoordT c = grid_.posToCoord(pos.x(), pos.y(), pos.z());
  const TsdfVoxel* v = const_acc.value(c);
  if (!v) return std::nullopt;
  return *v;
}

std::size_t TsdfMap::voxelCount() const {
  return grid_.activeCellsCount();
}

}  // namespace scovox
