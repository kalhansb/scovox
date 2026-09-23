// Moved comments: doc/scovox_mapping_code_notes.md
#include "scovox/scovoxmap.hpp"
#include <algorithm>
#include <cmath>

namespace scovox {

Map::Map(const Params& p)
: params_(p),
  grid_(params_.resolution, params_.inner_bits, params_.leaf_bits),
  transient_grid_(params_.resolution, params_.inner_bits, params_.leaf_bits),
  acc_(grid_.createAccessor()),
  transient_acc_(transient_grid_.createAccessor()),
  const_acc_(grid_.createConstAccessor()),
  const_transient_acc_(transient_grid_.createConstAccessor())
{}

void Map::beta_update_occupied(Voxel* v, float range_w, float angle_w) const {
  v->a_occ += params_.w_occ * range_w * angle_w;
  apply_evidence_saturation(v);
}

void Map::beta_update_free(Voxel* v, float range_w) const {
  v->a_free += params_.w_free * range_w;
  apply_evidence_saturation(v);
}

// Optional pre-cleanup behaviour: cap (a_occ, a_free, sem_cnt) at
// `evidence_saturation`. Disabled when the param is 0 (default). Applied
// after each Beta/Dirichlet update so accumulated mass never runs away.
void Map::apply_evidence_saturation(Voxel* v) const {
  const float cap = static_cast<float>(params_.evidence_saturation);
  if (cap <= 0.f) return;
  // Scale a_occ and a_free by ONE shared factor so the larger lands at cap,
  // preserving p_occ. Do not scale each bucket separately: it double-scales and
  // the 1.0 floor then distorts p_occ. (notes: saturation-shared-beta-factor)
  const float max_beta = std::max(v->a_occ, v->a_free);
  if (max_beta > cap) {
    const float s = cap / max_beta;
    v->a_occ *= s;
    v->a_free *= s;
    // Beta(1,1) prior floor: a last-resort guard against an α drifting toward
    // 0 (e.g. a near-point-mass voxel whose minority bucket scales below the
    // prior). This is a safety net only; with the single shared factor above
    // the ratio is preserved in the common case and the floor rarely binds.
    if (v->a_occ < 1.0f) v->a_occ = 1.0f;
    if (v->a_free < 1.0f) v->a_free = 1.0f;
  }
  // Dirichlet: cap proportionally to preserve class ratios.
  float max_sem = v->a_unk;
  for (int i = 0; i < K_TOP; ++i) {
    if (v->sem_cnt[i] > max_sem) max_sem = v->sem_cnt[i];
  }
  if (max_sem > cap) {
    const float s = cap / max_sem;
    for (int i = 0; i < K_TOP; ++i) v->sem_cnt[i] *= s;
    v->a_unk *= s;
  }
}

void Map::carve_free(const Eigen::Vector3f& origin, const Eigen::Vector3f& hit) {
  float total_range = (hit - origin).norm();
  float range_w = 1.0f;
  if (params_.range_decay_length > 0 && total_range > 0.01f)
    range_w = std::exp(-total_range / params_.range_decay_length);
  carve_free(origin, hit, range_w);
}

void Map::carve_free(const Eigen::Vector3f& origin, const Eigen::Vector3f& hit,
                     float range_w_override) {
  CoordT key_origin = posToCoord(origin);
  CoordT key_end    = posToCoord(hit);
  if (key_origin == key_end) return;

  // Per-voxel independence assumption (OctoMap-style). Stop carving at the
  // first confidently-occupied voxel; this is the cheap "don't carve through
  // walls" guard that replaced the more principled but cold-start-expensive
  // reach_prob attenuation.
  const float skip = params_.carve_skip_occ_threshold;

  RayIterator(key_origin, key_end, [&](const CoordT& c) {
    if (c == key_end) return false;

    Voxel* v = acc_.value(c);
    if (skip > 0.f && v && v->p_occ() > skip) return false;  // guard opt-in: hit a wall, stop carving

    if (!v) {
      Voxel nv = defaultVoxel();
      nv.a_free += params_.w_free * range_w_override;
      apply_evidence_saturation(&nv);
      acc_.setValue(c, nv);
    } else {
      v->a_free += params_.w_free * range_w_override;
      apply_evidence_saturation(v);
    }
    return true;
  });
}

void Map::update_endpoint(const CoordT& c,
                          const std::vector<float>* class_probs,
                          float quality, float range_w, float angle_w)
{
  Voxel* v = acc_.value(c);
  if (!v) {
    Voxel nv = defaultVoxel();
    beta_update_occupied(&nv, range_w, angle_w);
    acc_.setValue(c, nv);
    v = acc_.value(c);
  } else {
    beta_update_occupied(v, range_w, angle_w);
  }

  if (!class_probs) {
    return;
  }

  apply_semantics(v, class_probs, quality);
}

void Map::update_endpoint_on(Grid::Accessor& target_acc, const CoordT& c,
                              const std::vector<float>* class_probs,
                              float quality, float range_w, float angle_w)
{
  Voxel* v = target_acc.value(c);
  if (!v) {
    Voxel nv = defaultVoxel();
    beta_update_occupied(&nv, range_w, angle_w);
    target_acc.setValue(c, nv);
    v = target_acc.value(c);
  } else {
    beta_update_occupied(v, range_w, angle_w);
  }

  if (!class_probs) {
    return;
  }

  apply_semantics(v, class_probs, quality);
}

void Map::apply_semantics(Voxel* v, const std::vector<float>* class_probs,
                          float quality) const {
  // DIRICHLET weights the update by p_occ. NAIVE and MAJORITY_VOTE are ablation
  // baselines with a hard p_occ > 0.5 cutoff, since their accumulators take no
  // continuous weight. (notes: semantics-mode-occupancy-gate)
  const float p_occ = v->p_occ();

  switch (params_.semantic_mode) {
    case SemanticMode::NAIVE:
      if (p_occ > 0.5f) naive_update_semantics(v, class_probs);
      break;
    case SemanticMode::MAJORITY_VOTE:
      if (p_occ > 0.5f) majority_vote_semantics(v, class_probs);
      break;
    case SemanticMode::DIRICHLET:
    default:
      if (p_occ >= params_.dirichlet_min_p_occ) {
        dirichlet_update_semantics(v, class_probs, quality, p_occ,
                                   params_.kappa0);
      }
      break;
  }
  apply_evidence_saturation(v);
}

void Map::integrateRay(const Eigen::Vector3f& origin,
                       const Eigen::Vector3f& hit,
                       bool is_dynamic,
                       const std::vector<float>* class_probs,
                       float quality, float range_w, float angle_w)
{
  if (is_dynamic) {
    // Transient layer: legacy two-pass path. TSDF is intentionally not
    // populated for dynamic observations — the surface "moves" frame to
    // frame so a running average would smear the SDF estimate.
    carve_free(origin, hit);
    auto c = posToCoord(hit);
    update_endpoint_on(transient_acc_, c, class_probs, quality, range_w, angle_w);
    return;
  }
  fused_integrate_ray_static(origin, hit, /*updated_coords=*/nullptr,
                             class_probs, quality, range_w, angle_w);
}

void Map::integrateRay(const Eigen::Vector3f& origin,
                       const Eigen::Vector3f& hit,
                       std::vector<CoordT>& updated_coords,
                       bool is_dynamic,
                       const std::vector<float>* class_probs,
                       float quality, float range_w, float angle_w)
{
  if (is_dynamic) {
    carve_free(origin, hit, updated_coords);
    auto c = posToCoord(hit);
    update_endpoint_on(transient_acc_, c, class_probs, quality, range_w, angle_w);
    updated_coords.push_back(c);
    return;
  }
  fused_integrate_ray_static(origin, hit, &updated_coords,
                             class_probs, quality, range_w, angle_w);
}

void Map::integrateEndpointOnly(const Eigen::Vector3f& hit,
                                bool is_dynamic,
                                const std::vector<float>* class_probs,
                                float quality, float range_w, float angle_w)
{
  auto c = posToCoord(hit);
  if (is_dynamic) {
    update_endpoint_on(transient_acc_, c, class_probs, quality, range_w, angle_w);
    return;
  }
  update_endpoint(c, class_probs, quality, range_w, angle_w);

  // Endpoint-only mode (carve_band == 0) skips the DDA on purpose. We still
  // record a TSDF surface mass on the hit voxel — `sdf == 0` for the hit
  // cell, with weight `range_w * angle_w` as in the fused walk.
  if (params_.sdf_trunc > 0.f) {
    Voxel* v = acc_.value(c);
    if (v) {
      const float w_ray = range_w * angle_w;
      const float new_w = v->tsdf_weight + w_ray;
      if (new_w > 0.f) {
        v->tsdf_distance = (v->tsdf_distance * v->tsdf_weight) / new_w;
        v->tsdf_weight   = new_w;
      }
    }
  }
}

void Map::fused_integrate_ray_static(const Eigen::Vector3f& origin,
                                     const Eigen::Vector3f& hit,
                                     std::vector<CoordT>* updated_coords,
                                     const std::vector<float>* class_probs,
                                     float quality, float range_w, float angle_w)
{
  const Eigen::Vector3f d = hit - origin;
  const float depth = d.norm();
  if (depth < 1e-4f) return;
  const Eigen::Vector3f u = d / depth;
  const float trunc = params_.sdf_trunc;          // 0 → TSDF disabled
  const float w_ray = range_w * angle_w;

  // The free-update weight is the caller's range_w (from the full sensor-to-hit
  // distance). Do not recompute it from depth here: with carve_band > 0, origin
  // is truncated and depth is about carve_band.
  // (notes: fused-carve-weight-caller-range)
  const float carve_w = range_w;

  const CoordT k_hit = posToCoord(hit);
  const CoordT k_far = (trunc > 0.f)
      ? posToCoord(Eigen::Vector3f(hit + trunc * u))
      : k_hit;
  // With band_only_integration, the DDA starts at hit - trunc instead of the
  // origin, skipping the long free carve. Falls back to the full ray when trunc
  // is 0 (TSDF disabled). (notes: fused-band-only-dda-start)
  const CoordT k0 = (params_.band_only_integration && trunc > 0.f)
      ? posToCoord(Eigen::Vector3f(hit - trunc * u))
      : posToCoord(origin);

  if (k0 == k_far) return;  // degenerate ray inside one voxel

  // Per-voxel independence: no joint ray-cast attenuation. Through-wall carving
  // is gated by carve_skip_occ_threshold instead.
  // (notes: fused-per-voxel-independence)
  bool past_wall = false;
  const float skip = params_.carve_skip_occ_threshold;

  // The DDA can skip k_hit on an oblique ray, so track whether it was visited
  // and visit it after the loop; the endpoint occupied, semantics and surface
  // TSDF updates must always land. (notes: fused-dda-skipped-hit-voxel)
  bool k_hit_visited = false;

  auto step = [&](const CoordT& c) -> bool {
    auto vp = grid_.coordToPos(c);
    const Eigen::Vector3f vc((float)vp.x, (float)vp.y, (float)vp.z);
    const float t   = (vc - origin).dot(u);
    const float sdf = depth - t;                  // + in front, − behind

    const bool at_hit   = (c == k_hit);
    if (at_hit) {
      if (k_hit_visited) return true;  // dedupe: don't double-fire at_hit
      k_hit_visited = true;
    }
    const bool past_hit = (sdf < 0.f);
    const bool in_band  = (trunc > 0.f) && (std::fabs(sdf) <= trunc
                          || (params_.tsdf_space_carving && sdf > trunc));

    Voxel  scratch;
    Voxel* v = acc_.value(c);
    const bool created = (v == nullptr);
    if (created) { scratch = defaultVoxel(); v = &scratch; }
    bool modified = false;

    // Wall detection: existing voxel confidently occupied → stop carving
    // through it AND past it. Endpoint and TSDF still update normally.
    if (skip > 0.f && !at_hit && !past_hit && !past_wall && !created &&
        v->p_occ() > skip) {
      past_wall = true;
    }

    // 1. Beta free for non-endpoint, in-front voxels (only before any wall).
    if (!at_hit && !past_hit && !past_wall) {
      v->a_free += params_.w_free * carve_w;
      apply_evidence_saturation(v);
      modified = true;
    }

    // 2. Beta occupied + semantics at the hit voxel.
    if (at_hit) {
      v->a_occ += params_.w_occ * range_w * angle_w;
      apply_evidence_saturation(v);
      if (class_probs) apply_semantics(v, class_probs, quality);
      modified = true;
    }

    // 3. TSDF fusion in band — unweighted (geometric, not Bayesian).
    if (trunc > 0.f && in_band) {
      const float tsdf_clamped = std::min(trunc, std::max(-trunc, sdf));
      const float new_w = v->tsdf_weight + w_ray;
      if (new_w > 0.f) {
        v->tsdf_distance =
            (v->tsdf_distance * v->tsdf_weight + tsdf_clamped * w_ray) / new_w;
        v->tsdf_weight = new_w;
        modified = true;
      }
    }

    if (modified) {
      if (created) acc_.setValue(c, *v);
      if (updated_coords) updated_coords->push_back(c);
    }
    return true;
  };

  // RayIterator stops one step short of `k_far` (its for-bound is
  // `i < maxc - 1`); follow up with one explicit visit so the last
  // band voxel — and `k_hit` itself when trunc==0 — is always reached.
  RayIterator(k0, k_far, step);
  step(k_far);
  // Bresenham may have skipped k_hit; the lambda dedupes via k_hit_visited.
  if (!k_hit_visited) step(k_hit);
}

void Map::carve_free(const Eigen::Vector3f& origin, const Eigen::Vector3f& hit,
                     std::vector<CoordT>& traversed_coords) {
  float total_range = (hit - origin).norm();
  float range_w = 1.0f;
  if (params_.range_decay_length > 0 && total_range > 0.01f)
    range_w = std::exp(-total_range / params_.range_decay_length);
  carve_free(origin, hit, traversed_coords, range_w);
}

void Map::carve_free(const Eigen::Vector3f& origin, const Eigen::Vector3f& hit,
                     std::vector<CoordT>& traversed_coords, float range_w_override) {
  CoordT key_origin = posToCoord(origin);
  CoordT key_end    = posToCoord(hit);
  if (key_origin == key_end) return;

  // Per-voxel independence (see the no-traversal carve_free overload above).
  const float skip = params_.carve_skip_occ_threshold;

  RayIterator(key_origin, key_end, [&](const CoordT& c) {
    if (c == key_end) return false;

    Voxel* v = acc_.value(c);
    if (skip > 0.f && v && v->p_occ() > skip) return false;  // guard opt-in: wall hit, stop

    if (!v) {
      Voxel nv = defaultVoxel();
      nv.a_free += params_.w_free * range_w_override;
      apply_evidence_saturation(&nv);
      acc_.setValue(c, nv);
    } else {
      v->a_free += params_.w_free * range_w_override;
      apply_evidence_saturation(v);
    }
    traversed_coords.push_back(c);
    return true;
  });
}

bool Map::getVoxel(const Eigen::Vector3f& pos, Voxel& out) const {
  auto c = grid_.posToCoord(Eigen::Vector3d(pos.x(), pos.y(), pos.z()));
  const Voxel* v = const_acc_.value(c);
  if (!v) return false;
  out = *v; return true;
}

Voxel Map::getUnionVoxel(const Eigen::Vector3f& pos) const {
  // Picker between persistent and transient grids at query time.
  // "Confidently occupied" = p_occ > 0.5 (leaning occupied).
  auto c = grid_.posToCoord(Eigen::Vector3d(pos.x(), pos.y(), pos.z()));

  const Voxel* pv = const_acc_.value(c);
  if (pv && pv->p_occ() > 0.5f) return *pv;

  const Voxel* tv = const_transient_acc_.value(c);
  if (tv && tv->p_occ() > 0.5f) return *tv;

  if (pv) return *pv;
  if (tv) return *tv;
  return defaultVoxel();
}

void Map::decayTransientGrid(float decay_rate) {
  std::vector<CoordT> coords;
  transient_grid_.forEachCell([&](const Voxel&, const Bonxai::CoordT& c) {
    coords.push_back(c);
  });

  auto acc = transient_grid_.createAccessor();
  for (const auto& c : coords) {
    Voxel* v = acc.value(c);
    if (!v) continue;

    v->a_occ = 1.0f + (v->a_occ - 1.0f) * decay_rate;
    v->a_free = 1.0f + (v->a_free - 1.0f) * decay_rate;

    for (int i = 0; i < K_TOP; ++i) {
      v->sem_cnt[i] *= decay_rate;
    }
    v->a_unk *= decay_rate;
  }
}

void Map::clearTransientGrid() {
  transient_grid_.clear(Bonxai::CLEAR_MEMORY);
}

void Map::consensusMerge(Voxel& dst, const Voxel& src) const {
  // Beta-conjugate fusion with the shared Beta(1,1) prior subtracted once.
  // Valid only while each ScovoxMapBinary comes from a robot's local map, never
  // the merged map. No floor: every Voxel keeps a_occ, a_free >= 1.
  // (notes: consensus-beta-merge-rule)
  dst.a_occ  = dst.a_occ  + src.a_occ  - 1.f;
  dst.a_free = dst.a_free + src.a_free - 1.f;

  // Always add src's Dirichlet evidence, whatever the merged occupancy. Hiding
  // colour on free voxels is a display concern: consumers filter on p_occ at
  // query time (e.g. dscovox_node's pointcloud colour gate).
  // (notes: consensus-dirichlet-merge-ungated)
  for (int i = 0; i < K_TOP; ++i) {
    if (src.sem_cnt[i] > 0.f) {
      sparse_add(dst.sem_cnt, dst.sem_cls, src.sem_cls[i], src.sem_cnt[i], &dst.a_unk);
    }
  }
  dst.a_unk += src.a_unk;

  // No conflict check here; betaKL() in scovox/uncertainty.hpp remains for
  // callers that want an explicit disagreement metric.
  // (notes: consensus-no-kl-conflict)
}

} // namespace scovox
