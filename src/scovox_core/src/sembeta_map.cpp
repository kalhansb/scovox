/// @file
/// @brief SCovox-contribution Beta + sparse-Dirichlet integration on the
/// SemBeta voxel grid. Direct port of `scovox::Map`'s SCOVOX-mode body
/// from `scovoxmap.cpp`, restricted to the SemBeta fields.

#include "scovox/sembeta_map.hpp"

#include <algorithm>
#include <cmath>

#include "scovox/ray_iterator.hpp"

namespace scovox {

namespace {

// SemBeta-only port of the Voxel-typed semantic update functions in
// `semantics.hpp`. Behaviour identical to the Voxel-typed versions —
// duplicated here only because converting `semantics.hpp` to templates
// would touch the legacy fused-Voxel call sites mid-refactor. After the
// fused Voxel struct retires (Step 5), `semantics.hpp` should be
// templatised and these duplicates deleted.

void dirichletUpdate(SemBetaVoxel*               v,
                     const std::vector<float>*   class_probs,
                     float                       quality,
                     float                       p_occ,
                     float                       kappa0) {
  if (!class_probs) return;
  const auto& obs = *class_probs;
  if (obs.empty()) return;

  const float w = kappa0 * p_occ * std::clamp(quality, 0.0f, 1.0f);
  if (w <= 0.f) return;

  float sum_p = 0.f;
  for (size_t i = 0; i < obs.size(); ++i) if (obs[i] > 0.f) sum_p += obs[i];
  const float norm = (sum_p > 1.0f) ? (1.0f / sum_p) : 1.0f;

  for (size_t i = 0; i < obs.size(); ++i) {
    if (obs[i] <= 0.f) continue;
    const float inc = w * obs[i] * norm;
    if (inc <= 0.f) continue;
    sparse_add(v->sem_cnt, v->sem_cls,
               static_cast<uint16_t>(i), inc, &v->a_unk);
  }

  const float covered = sum_p * norm;
  const float uinc    = w * (1.0f - covered);
  if (uinc > 0.f) v->a_unk += uinc;
}

void naiveUpdate(SemBetaVoxel* v, const std::vector<float>* class_probs) {
  if (!class_probs) return;
  const auto& obs = *class_probs;
  if (obs.empty()) return;
  auto it = std::max_element(obs.begin(), obs.end());
  if (*it <= 0.f) return;
  const uint16_t label = static_cast<uint16_t>(std::distance(obs.begin(), it));
  for (int i = 0; i < K_TOP; ++i) { v->sem_cls[i] = 0xFFFF; v->sem_cnt[i] = 0.f; }
  v->a_unk = 0.f;
  v->sem_cls[0] = label;
  v->sem_cnt[0] = 1.f;
}

void majorityVoteUpdate(SemBetaVoxel* v, const std::vector<float>* class_probs) {
  if (!class_probs) return;
  const auto& obs = *class_probs;
  if (obs.empty()) return;
  auto it = std::max_element(obs.begin(), obs.end());
  if (*it <= 0.f) return;
  const uint16_t label = static_cast<uint16_t>(std::distance(obs.begin(), it));
  sparse_add(v->sem_cnt, v->sem_cls, label, 1.0f, &v->a_unk);
}

SemBetaMap::Params sanitise(SemBetaMap::Params p) {
  if (p.resolution <= 0.0)            p.resolution         = 0.05;
  if (p.range_decay_length < 0.f)     p.range_decay_length = 0.f;  // 0 = disabled
  return p;
}

}  // namespace

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

SemBetaMap::SemBetaMap(const Params& p)
    : params_(sanitise(p))
    , grid_(params_.resolution, params_.inner_bits, params_.leaf_bits)
    , acc_(grid_.createAccessor())
    , touched_() {}

// ---------------------------------------------------------------------------
// Allocation helper (Q5: enforce Beta(1,1) prior at first touch)
// ---------------------------------------------------------------------------

SemBetaVoxel* SemBetaMap::getOrAllocate(const CoordT& c) {
  // Try without creating first. If absent, build a defaultSemBetaVoxel()
  // via setValue, then re-fetch.
  SemBetaVoxel* v = acc_.value(c, /*create_if_missing=*/false);
  if (v) return v;

  const SemBetaVoxel def = defaultSemBetaVoxel();
  acc_.setValue(c, def);
  v = acc_.value(c, /*create_if_missing=*/false);
  return v;
}

// ---------------------------------------------------------------------------
// Public integration entry points
// ---------------------------------------------------------------------------

void SemBetaMap::integrateHit(const Eigen::Vector3f&    origin,
                              const Eigen::Vector3f&    endpoint,
                              const std::vector<float>* sem_probs,
                              float                     quality) {
  carveRay(origin, endpoint, quality, /*inclusive_endpoint=*/false);
  const CoordT k_hit = grid_.posToCoord(endpoint.x(), endpoint.y(), endpoint.z());
  updateHit(k_hit, sem_probs, quality);
}

void SemBetaMap::integrateMiss(const Eigen::Vector3f& origin,
                               const Eigen::Vector3f& endpoint,
                               float                  quality) {
  carveRay(origin, endpoint, quality, /*inclusive_endpoint=*/true);
}

// ---------------------------------------------------------------------------
// Carve loop — per-voxel a_free update along the ray
// ---------------------------------------------------------------------------

void SemBetaMap::carveRay(const Eigen::Vector3f& origin,
                          const Eigen::Vector3f& endpoint,
                          float                  quality,
                          bool                   inclusive_endpoint) {
  const CoordT k0    = grid_.posToCoord(origin.x(),   origin.y(),   origin.z());
  const CoordT k_end = grid_.posToCoord(endpoint.x(), endpoint.y(), endpoint.z());
  if (k0 == k_end) return;

  const float w_inc = params_.w_free * quality;
  if (w_inc <= 0.f && !inclusive_endpoint) return;

  RayIterator(k0, k_end, [&](const CoordT& c) -> bool {
    if (c == k_end) return false;  // hit voxel handled separately for hits
    return applyCarveUpdate(c, quality);
  });

  if (inclusive_endpoint) {
    // Misses: also a_free the endpoint voxel — applyCarveUpdate's wall
    // guard short-circuits this when k_end is occupied.
    (void)applyCarveUpdate(k_end, quality);
  }
}

// ---------------------------------------------------------------------------
// Per-voxel public API (fused walker — Step 12.10)
// ---------------------------------------------------------------------------

bool SemBetaMap::applyCarveUpdate(const CoordT& c, float quality) {
  const float w_inc = params_.w_free * quality;
  if (w_inc <= 0.f) return true;  // no-op; not a wall, caller continues

  const float skip = params_.carve_skip_occ_threshold;

  SemBetaVoxel* v = acc_.value(c, /*create_if_missing=*/false);
  if (v && v->p_occ() > skip) return false;  // wall — caller stops carving

  if (!v) {
    SemBetaVoxel nv = defaultSemBetaVoxel();
    nv.a_free += w_inc;
    applyEvidenceSaturation(&nv);
    acc_.setValue(c, nv);
  } else {
    v->a_free += w_inc;
    applyEvidenceSaturation(v);
  }
  touched_.push_back(c);
  return true;
}

void SemBetaMap::applyHitUpdate(const CoordT&             c,
                                const std::vector<float>* sem_probs,
                                float                     quality) {
  // Identical body to the private updateHit — exposed so the fused walker
  // doesn't need friend access.
  updateHit(c, sem_probs, quality);
}

// ---------------------------------------------------------------------------
// Hit voxel update — Beta a_occ + sparse-Dirichlet semantics
// ---------------------------------------------------------------------------

void SemBetaMap::updateHit(const CoordT&             c,
                           const std::vector<float>* sem_probs,
                           float                     quality) {
  SemBetaVoxel* v = getOrAllocate(c);

  // Beta a_occ update. The angle_w factor that the legacy code applied is
  // folded into `quality` by callers (per Q4) — here we just trust the
  // scalar.
  v->a_occ += params_.w_occ * quality;
  applyEvidenceSaturation(v);

  // Semantic update gated by p_occ.
  const float p_occ = v->p_occ();

  switch (params_.semantic_mode) {
    case SemanticMode::NAIVE:
      if (p_occ > 0.5f) naiveUpdate(v, sem_probs);
      break;
    case SemanticMode::MAJORITY_VOTE:
      if (p_occ > 0.5f) majorityVoteUpdate(v, sem_probs);
      break;
    case SemanticMode::DIRICHLET:
    default:
      if (p_occ >= params_.dirichlet_min_p_occ) {
        dirichletUpdate(v, sem_probs, quality, p_occ, params_.kappa0);
      }
      break;
  }
  applyEvidenceSaturation(v);

  touched_.push_back(c);
}

// ---------------------------------------------------------------------------
// Beta + Dirichlet evidence cap (port of Map::apply_evidence_saturation)
// ---------------------------------------------------------------------------

void SemBetaMap::applyEvidenceSaturation(SemBetaVoxel* v) const {
  const float cap = params_.evidence_saturation;
  if (cap <= 0.f) return;

  if (v->a_occ > cap) {
    const float s = cap / v->a_occ;
    v->a_occ = cap;
    v->a_free *= s;
    if (v->a_free < 1.0f) v->a_free = 1.0f;
  }
  if (v->a_free > cap) {
    const float s = cap / v->a_free;
    v->a_free = cap;
    v->a_occ *= s;
    if (v->a_occ < 1.0f) v->a_occ = 1.0f;
  }

  float max_sem = v->a_unk;
  for (int i = 0; i < K_TOP; ++i) if (v->sem_cnt[i] > max_sem) max_sem = v->sem_cnt[i];
  if (max_sem > cap) {
    const float s = cap / max_sem;
    for (int i = 0; i < K_TOP; ++i) v->sem_cnt[i] *= s;
    v->a_unk *= s;
  }
}

// ---------------------------------------------------------------------------
// Touched-set drain (Q7)
// ---------------------------------------------------------------------------

std::vector<SemBetaMap::CoordT> SemBetaMap::drainTouched() {
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

std::optional<SemBetaVoxel> SemBetaMap::getVoxel(const Eigen::Vector3f& pos) const {
  auto const_acc = grid_.createConstAccessor();
  const CoordT c = grid_.posToCoord(pos.x(), pos.y(), pos.z());
  const SemBetaVoxel* v = const_acc.value(c);
  if (!v) return std::nullopt;
  return *v;
}

std::size_t SemBetaMap::voxelCount() const {
  return grid_.activeCellsCount();
}

}  // namespace scovox
