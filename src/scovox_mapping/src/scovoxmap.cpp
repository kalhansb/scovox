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
  // Beta proportional saturation: scale (a_occ, a_free) by a SINGLE factor so
  // the larger bucket lands at `cap`, preserving the ratio (and therefore
  // p_occ). Must be one shared factor: chaining two independent
  // scale-and-floor blocks (one per bucket) would double-scale when both
  // buckets exceed cap (neither bucket lands at cap, ratio drifts) and lets
  // the per-block 1.0 floor distort p_occ on lopsided voxels. We compute the
  // factor from max(a_occ, a_free) and apply it to both at once.
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
    if (v && v->p_occ() > skip) return false;  // hit a wall, stop carving

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
  // Bayesian-soft for DIRICHLET: weight by p_occ directly (no hard gate).
  // NAIVE and MAJORITY_VOTE are ablation baselines and use a hard `p_occ > 0.5`
  // cutoff (their accumulators don't take a continuous weight) so that the
  // only ablation variable across the three modes remains the per-observation
  // accumulation rule, not whether updates fire at all.
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

  // Beta free-update weight matches the legacy carve_free behavior exactly:
  // it uses the node-supplied range_w (computed from the FULL sensor→hit
  // distance), not a fresh exp(-segment_depth/decay) over the carve segment.
  // For the partial-ray (carve_band > 0) case `origin` is a truncated
  // origin and depth ≈ carve_band, so an in-function recomputation would
  // diverge from the unified model by orders of magnitude. Use the caller's value.
  const float carve_w = range_w;

  const CoordT k_hit = posToCoord(hit);
  const CoordT k_far = (trunc > 0.f)
      ? posToCoord(Eigen::Vector3f(hit + trunc * u))
      : k_hit;
  // Band-only mode (benchmarking / TSDF-only): start the DDA at the near
  // edge of the truncation band rather than the sensor origin. Skips the
  // long Beta-free carve from origin to (hit - trunc) and matches SLIM-VDB's
  // [depth-trunc, depth+trunc] DDA. Falls back to full-ray when trunc=0
  // (TSDF disabled) since there is no band to confine the walk to.
  const CoordT k0 = (params_.band_only_integration && trunc > 0.f)
      ? posToCoord(Eigen::Vector3f(hit - trunc * u))
      : posToCoord(origin);

  if (k0 == k_far) return;  // degenerate ray inside one voxel

  // Per-voxel independence assumption (OctoMap / log_odds-node style). Joint
  // ray-cast `reach_prob` was tried (commit 513c969) and reverted: cost
  // ~6 mIoU points on Replica m2f from cold-start damping (every voxel
  // starts at p_occ=0.5, so reach_prob ≈ 0.5^N along uninitialised rays).
  // KITTI was bit-flat either way. Through-wall carving is gated by
  // `carve_skip_occ_threshold` instead — cheap, well-tested, sufficient.
  bool past_wall = false;
  const float skip = params_.carve_skip_occ_threshold;

  // Bresenham DDA can skip k_hit when it's a corner-crossing voxel on the
  // line (it picks one voxel per dominant-axis step, so an oblique ray's
  // true hit voxel may not lie on the picked path even though it's on the
  // line). Track whether k_hit was visited and explicitly visit it after
  // the loop if not — guarantees the endpoint Beta-occupied + semantics +
  // surface TSDF mass always lands.
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
    if (!at_hit && !past_hit && !past_wall && !created &&
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
    if (v && v->p_occ() > skip) return false;  // wall hit, stop

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
  // Beta-conjugate posterior under conditional independence given θ:
  //   Beta(α₁, β₁) ⊕ Beta(α₂, β₂) = Beta(α₁ + α₂ − 1, β₁ + β₂ − 1)
  // with the shared Beta(1, 1) prior subtracted once.
  //
  // Conditional independence is upheld by the dscovox publish topology:
  // each ScovoxMapBinary is sent from a robot's *local* scovox map (its
  // own sensor observations), never from the merged dscovox map, so a
  // robot's own evidence cannot be echoed back into its source grid via
  // the fused output. The one residual correlation source — shared
  // classifier error if multiple robots run the same segmenter on similar
  // RGB — is a measurement-model concern on the *semantic* path; the
  // Bayesian fix is a per-source evidence discount on the Dirichlet
  // counts, not a change of merge rule, and is unaddressed here.
  //
  // No `max(1, ·)` floor: every Voxel in this codebase is constructed with
  // a_occ, a_free ≥ 1 (defaultVoxel() and the integration paths preserve
  // this), so α₁ + α₂ − 1 ≥ 1 algebraically. The previous floor was a
  // non-Bayesian guard against malformed inputs that the type system never
  // produces; removed 2026-05-03.
  dst.a_occ  = dst.a_occ  + src.a_occ  - 1.f;
  dst.a_free = dst.a_free + src.a_free - 1.f;

  // Dirichlet semantic merge: always combine src's evidence regardless of
  // post-merge occupancy. Under conditional independence given the latent
  // (occ, class) the additive Dirichlet has no occupancy condition. The
  // "don't put semantic colour on free voxels" intent of the legacy gate is
  // a *display* concern — downstream consumers filter on p_occ at
  // query/visualization time (e.g. dscovox_node's pointcloud publish gates
  // colour by `cf >= sem_gate_`).
  for (int i = 0; i < K_TOP; ++i) {
    if (src.sem_cnt[i] > 0.f) {
      sparse_add(dst.sem_cnt, dst.sem_cls, src.sem_cls[i], src.sem_cnt[i], &dst.a_unk);
    }
  }
  dst.a_unk += src.a_unk;

  // No betaKL conflict computation. The previous code computed and returned
  // a `conflict` bool but every caller discarded it; the threshold knob
  // (`consensus_kl_threshold`) was log-only and never gated fusion.
  // betaKL() itself remains in scovox/uncertainty.hpp for callers that want
  // an explicit disagreement metric outside the merge path.
}

} // namespace scovox
