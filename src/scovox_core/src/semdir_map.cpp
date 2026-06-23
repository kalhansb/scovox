/// @file
/// @brief Unified-Dirichlet integration on the SemDir voxel grid. Replaces
/// `SemBetaMap`'s Beta + sparse Dirichlet two-stage path with a single
/// Dirichlet update.
///
/// Design and mass-flow rationale: docs/design/unified_dirichlet_design_2026_05_13.md

#include "scovox/semdir_map.hpp"

#include <algorithm>
#include <cmath>

#include "scovox/ray_iterator.hpp"
#include "scovox/voxel.hpp"   // g_sparse_*_count counters (shared with legacy path)

namespace scovox {

namespace {

/// Heavy-hitter sparse_add for the unified Dirichlet, parametrised by the
/// per-dim prior `α_0`. Routes evidence into one of the top-K_TOP class
/// slots (Space-Saving / Metwally 2005) or the OTHER bucket — never lost.
///
/// Slots are detected as empty by `cls[i] == 0xFFFF`. Empty slots carry the
/// constant prior `α_0` in `cnt[i]`; filled slots carry `α_0 + evidence`.
/// On eviction, the slot's *accumulated evidence* (`cnt[i] − α_0`) flows
/// into OTHER; the prior `α_0` stays in the slot as the placeholder for
/// the next class to fill it. This preserves the strict mass invariant
///   ΔΣα == Σ Δ inputs
/// — i.e. mass conservation is exact under the unified scheme, not the
/// `≥ 0` slack of the legacy `sparse_add` in voxel.hpp.
inline void sparse_add_unified(float*    cnt,
                               uint16_t* cls,
                               uint16_t  c,
                               float     inc,
                               float*    alpha_other,
                               float     alpha_0) {
  // (1) Match — incoming class already tracked in a slot.
  for (int i = 0; i < K_TOP; ++i) {
    if (cls[i] != 0xFFFF && cls[i] == c) {
      cnt[i] += inc;
      g_sparse_match_count.fetch_add(1, std::memory_order_relaxed);
      return;
    }
  }
  // (2) Empty slot available — fill it. The slot's α_0 prior stays; we
  // just add the observed mass on top.
  for (int i = 0; i < K_TOP; ++i) {
    if (cls[i] == 0xFFFF) {
      cls[i]  = c;
      cnt[i]  = alpha_0 + inc;
      g_sparse_empty_count.fetch_add(1, std::memory_order_relaxed);
      return;
    }
  }
  // (3) All slots filled — evict-or-drop using observed-evidence
  // (cnt − α_0) as the comparison key. Posterior-predictive Dirichlet
  // optimality argument is identical to the legacy `sparse_add`
  // (voxel.hpp:99-128) — see there for the derivation.
  int min_i = 0;
  for (int i = 1; i < K_TOP; ++i) if (cnt[i] < cnt[min_i]) min_i = i;
  const float evicted_evidence = cnt[min_i] - alpha_0;
  if (inc > evicted_evidence) {
    // Evict: incoming class displaces slot min_i. The displaced slot's
    // accumulated evidence flows to OTHER; the α_0 prior stays as the
    // slot's placeholder for the incoming class to build on.
    *alpha_other += evicted_evidence;
    cls[min_i] = c;
    cnt[min_i] = alpha_0 + inc;
    g_sparse_evict_count.fetch_add(1, std::memory_order_relaxed);
  } else {
    // Drop: incoming evidence smaller than every tracked class.
    // Mass goes to OTHER (not lost — strict mass conservation).
    *alpha_other += inc;
    g_sparse_drop_count.fetch_add(1, std::memory_order_relaxed);
  }
}

/// Unified-Dirichlet hit update for SemanticMode::DIRICHLET.
///
/// Mass flow per hit — **mirrors SemBeta's two-stream calibration exactly**:
///   Stream A (occupancy, no class commit):
///     alpha_other += w_occ · quality      — done in applyHitUpdate
///                                            BEFORE this function is called
///                                            (parallels SemBeta `a_occ +=
///                                            w_occ·q` before the Dirichlet
///                                            sub-update).
///   Stream B (Dirichlet, class commit):
///     class_share = kappa0 · p_occ_post · quality
///       — where p_occ_post is read AFTER Stream A's a_other bump. This
///         matches SemBeta which reads p_occ between the Beta and Dirichlet
///         sub-updates (sembeta_map.cpp:202-206,218).
///     for (c, p_c) ∈ class_probs:
///       inc = class_share · p_c · norm
///       sparse_add_unified(slots, c, inc, OTHER, α_0)
///     uncovered_softmax = class_share · (1 − covered)
///     OTHER += uncovered_softmax
///
/// Below the `dirichlet_min_p_occ` gate the caller skips this function
/// entirely — Stream A's `w_occ · q` is the only mass that lands (just like
/// SemBeta which skipped its `dirichletUpdate` below the gate).
///
/// History (2026-05-14): pre-fix code lumped Stream A + Stream B into one
/// softmax with p_occ_pre, which caused a ~0.025 KITTI regression because
/// (a) it committed Stream A's mass to specific classes (SemBeta did not),
/// and (b) p_occ_pre ≈ 14/15 on the very first hit instead of p_occ_post
/// ≈ s_occ_post/s_total_post. Pre-2026-05-14 had an even worse bug:
/// `total_mass · (1 - kappa0)` went negative at kappa0=2.0.
void dirichletUpdate(SemDirVoxel*              v,
                     const std::vector<float>* class_probs,
                     float                     class_share,
                     float                     alpha_0) {
  if (!class_probs || class_probs->empty()) {
    // No softmax → all class_share goes to OTHER (we know mass landed
    // here but we have no class signal to distribute it across slots).
    v->alpha_other += class_share;
    return;
  }
  const auto& obs = *class_probs;

  float sum_p = 0.f;
  for (size_t i = 0; i < obs.size(); ++i) if (obs[i] > 0.f) sum_p += obs[i];
  if (sum_p <= 0.f) {
    v->alpha_other += class_share;
    return;
  }
  const float norm = (sum_p > 1.0f) ? (1.0f / sum_p) : 1.0f;

  for (size_t i = 0; i < obs.size(); ++i) {
    if (obs[i] <= 0.f) continue;
    const float inc = class_share * obs[i] * norm;
    if (inc <= 0.f) continue;
    sparse_add_unified(v->cnt, v->cls,
                       static_cast<uint16_t>(i), inc,
                       &v->alpha_other, alpha_0);
  }
  const float covered = sum_p * norm;
  v->alpha_other += class_share * (1.0f - covered);
}

/// NAIVE mode: overwrite slot 0 with the argmax label at `α_0 + 1.0`,
/// dump the previous occupied mass to OTHER. `w_occ` / `quality` /
/// `kappa0` are intentionally ignored here — NAIVE is a debug baseline.
void naiveUpdate(SemDirVoxel* v, const std::vector<float>* class_probs, float alpha_0) {
  if (!class_probs || class_probs->empty()) return;
  const auto& obs = *class_probs;
  auto it = std::max_element(obs.begin(), obs.end());
  if (*it <= 0.f) return;
  const uint16_t label = static_cast<uint16_t>(std::distance(obs.begin(), it));

  // Conserve mass: dump every slot's accumulated evidence into OTHER, then
  // reset slots. The slot prior (α_0) stays; only the observed-evidence
  // portion (cnt[i] − α_0) flows to OTHER.
  for (int i = 0; i < K_TOP; ++i) {
    if (v->cls[i] != 0xFFFF) {
      v->alpha_other += v->cnt[i] - alpha_0;
      v->cls[i] = 0xFFFF;
      v->cnt[i] = alpha_0;
    }
  }
  v->cls[0] = label;
  v->cnt[0] = alpha_0 + 1.0f;
}

/// MAJORITY_VOTE mode: a single +1.0 sparse_add to the argmax class.
void majorityVoteUpdate(SemDirVoxel*              v,
                        const std::vector<float>* class_probs,
                        float                     alpha_0) {
  if (!class_probs || class_probs->empty()) return;
  const auto& obs = *class_probs;
  auto it = std::max_element(obs.begin(), obs.end());
  if (*it <= 0.f) return;
  const uint16_t label = static_cast<uint16_t>(std::distance(obs.begin(), it));
  sparse_add_unified(v->cnt, v->cls, label, 1.0f, &v->alpha_other, alpha_0);
}

SemDirMap::Params sanitise(SemDirMap::Params p) {
  if (p.resolution <= 0.0)            p.resolution         = 0.05;
  if (p.range_decay_length < 0.f)     p.range_decay_length = 0.f;
  if (p.alpha_0 <= 0.f)               p.alpha_0            = kDefaultDirichletPrior;
  if (p.num_classes < (K_TOP + 1))    p.num_classes        = K_TOP + 1;  // need at least one class outside top-K for OTHER to be meaningful
  return p;
}

}  // namespace

// ===========================================================================
// Construction
// ===========================================================================

SemDirMap::SemDirMap(const Params& p)
    : params_(sanitise(p))
    , grid_(params_.resolution, params_.inner_bits, params_.leaf_bits)
    , acc_(grid_.createAccessor())
    , touched_()
    , other_prior_(static_cast<float>(static_cast<int>(params_.num_classes) - K_TOP)
                   * params_.alpha_0) {}

// ===========================================================================
// Allocation
// ===========================================================================

SemDirVoxel* SemDirMap::getOrAllocate(const CoordT& c) {
  SemDirVoxel* v = acc_.value(c, /*create_if_missing=*/false);
  if (v) return v;
  const SemDirVoxel def = defaultSemDirVoxel(params_.num_classes, params_.alpha_0);
  acc_.setValue(c, def);
  v = acc_.value(c, /*create_if_missing=*/false);
  return v;
}

// ===========================================================================
// Public integration entry points
// ===========================================================================

void SemDirMap::integrateHit(const Eigen::Vector3f&    origin,
                             const Eigen::Vector3f&    endpoint,
                             const std::vector<float>* sem_probs,
                             float                     quality) {
  carveRay(origin, endpoint, quality, /*inclusive_endpoint=*/false);
  const CoordT k_hit = grid_.posToCoord(endpoint.x(), endpoint.y(), endpoint.z());
  applyHitUpdate(k_hit, sem_probs, quality);
}

void SemDirMap::integrateMiss(const Eigen::Vector3f& origin,
                              const Eigen::Vector3f& endpoint,
                              float                  quality) {
  carveRay(origin, endpoint, quality, /*inclusive_endpoint=*/true);
}

// ===========================================================================
// Carve loop
// ===========================================================================

void SemDirMap::carveRay(const Eigen::Vector3f& origin,
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
    (void)applyCarveUpdate(k_end, quality);
  }
}

// ===========================================================================
// Per-voxel API (exposed for the ScovoxMapSplit fused walker)
// ===========================================================================

bool SemDirMap::applyCarveUpdate(const CoordT& c, float quality) {
  const float w_inc = params_.w_free * quality;
  if (w_inc <= 0.f) return true;  // no-op; not a wall, caller continues

  const float skip = params_.carve_skip_occ_threshold;

  SemDirVoxel* v = acc_.value(c, /*create_if_missing=*/false);
  // Wall guard: stop carving through a voxel only when it is CONFIDENTLY
  // occupied — i.e. it carries real occupied evidence beyond the prior AND its
  // posterior p_occ exceeds the skip threshold. The unified/split occupancy
  // prior is p_occ = C/(C+1) ≈ 0.933 (not 0.5), so an allocated-but-unobserved
  // voxel (e.g. hit with quality≈0, or allocated then never carved) already
  // sits above the default carve_skip_occ_threshold=0.7. Gating on p_occ alone
  // would treat such prior-only voxels as walls and truncate the carve ray.
  // The prior occupied mass is C·α₀ (= num_classes·alpha_0); require s_occ
  // above it before the wall short-circuits.
  if (v) {
    const float prior_s_occ =
        static_cast<float>(params_.num_classes) * params_.alpha_0;
    const float occ_evidence = v->s_occ() - prior_s_occ;
    constexpr float kWallEvidenceEps = 1e-4f;
    if (occ_evidence > kWallEvidenceEps && v->p_occ() > skip)
      return false;  // wall — caller stops carving
  }

  if (!v) {
    SemDirVoxel nv = defaultSemDirVoxel(params_.num_classes, params_.alpha_0);
    nv.alpha_free += w_inc;
    applyEvidenceSaturation(&nv);
    acc_.setValue(c, nv);
  } else {
    v->alpha_free += w_inc;
    applyEvidenceSaturation(v);
  }
  touched_.push_back(c);
  return true;
}

void SemDirMap::applyHitUpdate(const CoordT&             c,
                               const std::vector<float>* sem_probs,
                               float                     quality) {
  SemDirVoxel* v = getOrAllocate(c);

  // Stream A — occupancy mass, no class commitment. Mirrors SemBeta's
  // `v->a_occ += w_occ·quality` (sembeta_map.cpp:202). In SemDir there is
  // no separate a_occ; alpha_other is the cleanest home for "this voxel
  // saw a hit but we don't want to commit the mass to a class yet" since
  // it's already the lumped-OTHER bucket. Always applied (every mode,
  // every gate state) — matches SemBeta which ran the Beta update before
  // any mode branching or gating.
  const float w_occ_share = params_.w_occ * quality;
  if (w_occ_share > 0.f) v->alpha_other += w_occ_share;

  // Saturate now so p_occ_post reflects the cap. SemBeta saturates here
  // too (sembeta_map.cpp:203). In SemDir the cap is on s_total() and the
  // scale-down is uniform, so this is typically a no-op until evidence
  // accumulates — matters mainly for long-lived voxels.
  applyEvidenceSaturation(v);

  // p_occ_post — read AFTER Stream A lands. Matches SemBeta line 206.
  // On a fresh voxel with prior (C+1)·α_0, p_occ_post = (s_occ_pre +
  // w_occ·q) / (s_total_pre + w_occ·q); on prior-only NYU13 (C=14, α_0=
  // 0.01) at w_occ=2.0, q=1.0 this is (14·0.01+2)/(15·0.01+2) ≈ 0.928
  // vs pre = 14/15 ≈ 0.933 — small but compounding gap over 100 frames.
  const float p_occ_post = v->p_occ();

  switch (params_.semantic_mode) {
    case SemanticMode::NAIVE:
      if (p_occ_post > 0.5f) {
        naiveUpdate(v, sem_probs, params_.alpha_0);
      }
      break;

    case SemanticMode::MAJORITY_VOTE:
      if (p_occ_post > 0.5f) {
        majorityVoteUpdate(v, sem_probs, params_.alpha_0);
      }
      break;

    case SemanticMode::DIRICHLET:
    default:
      if (p_occ_post >= params_.dirichlet_min_p_occ) {
        // Stream B — Dirichlet class commit, gated. class_share =
        // kappa0 · p_occ_post · quality matches SemBeta's
        // dirichletUpdate(w = kappa0·p_occ·q) exactly.
        const float class_share = params_.kappa0 * p_occ_post * quality;
        if (class_share > 0.f) {
          dirichletUpdate(v, sem_probs, class_share, params_.alpha_0);
        }
      }
      // Below gate: nothing more — Stream A already landed.
      break;
  }
  applyEvidenceSaturation(v);
  touched_.push_back(c);
}

// ===========================================================================
// Evidence saturation — single cap on s_total()
// ===========================================================================

void SemDirMap::applyEvidenceSaturation(SemDirVoxel* v) const {
  const float cap = params_.evidence_saturation;
  if (cap <= 0.f) return;

  const float s = v->s_total();
  if (s <= cap) return;

  // Uniform scale-down: preserves the distribution shape (and so all
  // marginals — p_occ, per-class probs, OTHER share — are invariant under
  // saturation). Distinct from the SemBeta path which rescaled Beta and
  // Dirichlet sub-vectors independently.
  const float k = cap / s;
  v->alpha_free  *= k;
  v->alpha_other *= k;
  for (int i = 0; i < K_TOP; ++i) {
    v->cnt[i] *= k;
  }
}

// ===========================================================================
// Touched-set drain
// ===========================================================================

std::vector<SemDirMap::CoordT> SemDirMap::drainTouched() {
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

// ===========================================================================
// Queries
// ===========================================================================

std::optional<SemDirVoxel> SemDirMap::getVoxel(const Eigen::Vector3f& pos) const {
  auto const_acc = grid_.createConstAccessor();
  const CoordT c = grid_.posToCoord(pos.x(), pos.y(), pos.z());
  const SemDirVoxel* v = const_acc.value(c);
  if (!v) return std::nullopt;
  return *v;
}

std::size_t SemDirMap::voxelCount() const {
  return grid_.activeCellsCount();
}

}  // namespace scovox
