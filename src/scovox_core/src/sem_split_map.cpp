/// @file
/// @brief Split Beta/Dirichlet integration — two parallel Bonxai grids.
/// De-unifies `SemDirMap` into a `BetaVoxel` occupancy grid + a `DirVoxel`
/// occupied-class grid, using the SemBeta two-stream update with SemDir-matched
/// priors and strict per-grid mass conservation.

#include "scovox/sem_split_map.hpp"

#include <algorithm>
#include <cmath>

#include "scovox/ray_iterator.hpp"

namespace scovox {

namespace {

/// Sparse-Dirichlet hit update on a `DirVoxel` (DIRICHLET mode). Mirrors
/// `dirichletUpdate` in semdir_map.cpp / sembeta_map.cpp: distribute
/// `class_share` over the observed softmax, routing uncovered + evicted mass
/// to OTHER. Total mass added to the voxel is exactly `class_share`.
void dirichletUpdate(DirVoxel*                 d,
                     const std::vector<float>* class_probs,
                     float                     class_share,
                     float                     alpha_0) {
  if (!class_probs || class_probs->empty()) {
    d->other += class_share;  // mass landed but no class signal to distribute
    return;
  }
  const auto& obs = *class_probs;

  float sum_p = 0.f;
  for (size_t i = 0; i < obs.size(); ++i) if (obs[i] > 0.f) sum_p += obs[i];
  if (sum_p <= 0.f) {
    d->other += class_share;
    return;
  }
  const float norm = (sum_p > 1.0f) ? (1.0f / sum_p) : 1.0f;

  for (size_t i = 0; i < obs.size(); ++i) {
    if (obs[i] <= 0.f) continue;
    const float inc = class_share * obs[i] * norm;
    if (inc <= 0.f) continue;
    sparse_add_class(d->cnt, d->cls, static_cast<uint16_t>(i), inc, &d->other, alpha_0);
  }
  const float covered = sum_p * norm;
  d->other += class_share * (1.0f - covered);
}

/// NAIVE mode: overwrite slot 0 with the argmax label at `α₀ + 1`, dumping
/// previous slot evidence to OTHER. Conserves mass.
void naiveUpdate(DirVoxel* d, const std::vector<float>* class_probs, float alpha_0) {
  if (!class_probs || class_probs->empty()) return;
  const auto& obs = *class_probs;
  auto it = std::max_element(obs.begin(), obs.end());
  if (*it <= 0.f) return;
  const uint16_t label = static_cast<uint16_t>(std::distance(obs.begin(), it));

  for (int i = 0; i < K_TOP; ++i) {
    if (d->cls[i] != 0xFFFF) {
      d->other += d->cnt[i] - alpha_0;  // conserve evidence, keep prior
      d->cls[i] = 0xFFFF;
      d->cnt[i] = alpha_0;
    }
  }
  d->cls[0] = label;
  d->cnt[0] = alpha_0 + 1.0f;
}

/// MAJORITY_VOTE mode: single +1 sparse-add to the argmax class.
void majorityVoteUpdate(DirVoxel* d, const std::vector<float>* class_probs, float alpha_0) {
  if (!class_probs || class_probs->empty()) return;
  const auto& obs = *class_probs;
  auto it = std::max_element(obs.begin(), obs.end());
  if (*it <= 0.f) return;
  const uint16_t label = static_cast<uint16_t>(std::distance(obs.begin(), it));
  sparse_add_class(d->cnt, d->cls, label, 1.0f, &d->other, alpha_0);
}

SemSplitMap::Params sanitise(SemSplitMap::Params p) {
  if (p.resolution <= 0.0)         p.resolution         = 0.05;
  if (p.range_decay_length < 0.f)  p.range_decay_length = 0.f;
  if (p.alpha_0 <= 0.f)            p.alpha_0            = kDefaultDirichletPrior;
  if (p.num_classes < (K_TOP + 1)) p.num_classes        = K_TOP + 1;
  return p;
}

}  // namespace

// ===========================================================================
// Construction
// ===========================================================================

SemSplitMap::SemSplitMap(const Params& p)
    : params_(sanitise(p))
    , beta_grid_(params_.resolution, params_.inner_bits, params_.leaf_bits)
    , dir_grid_(params_.resolution, params_.inner_bits, params_.leaf_bits)
    , beta_acc_(beta_grid_.createAccessor())
    , dir_acc_(dir_grid_.createAccessor())
    , touched_beta_()
    , touched_dir_()
    , beta_occ_prior_(static_cast<float>(params_.num_classes) * params_.alpha_0)
    , beta_free_prior_(params_.alpha_0) {}

// ===========================================================================
// Allocation (enforce prior at first touch — Bonxai zero-inits leaf blocks)
// ===========================================================================

BetaVoxel* SemSplitMap::getOrAllocateBeta(const CoordT& c) {
  BetaVoxel* v = beta_acc_.value(c, /*create_if_missing=*/false);
  if (v) return v;
  beta_acc_.setValue(c, defaultBetaVoxel(beta_occ_prior_, beta_free_prior_));
  return beta_acc_.value(c, /*create_if_missing=*/false);
}

DirVoxel* SemSplitMap::getOrAllocateDir(const CoordT& c) {
  DirVoxel* v = dir_acc_.value(c, /*create_if_missing=*/false);
  if (v) return v;
  dir_acc_.setValue(c, defaultDirVoxel(params_.num_classes, params_.alpha_0));
  return dir_acc_.value(c, /*create_if_missing=*/false);
}

// ===========================================================================
// Public integration entry points
// ===========================================================================

void SemSplitMap::integrateHit(const Eigen::Vector3f&    origin,
                               const Eigen::Vector3f&    endpoint,
                               const std::vector<float>* sem_probs,
                               float                     quality) {
  carveRay(origin, endpoint, quality, /*inclusive_endpoint=*/false);
  const CoordT k_hit = beta_grid_.posToCoord(endpoint.x(), endpoint.y(), endpoint.z());
  applyHitUpdate(k_hit, sem_probs, quality);
}

void SemSplitMap::integrateMiss(const Eigen::Vector3f& origin,
                                const Eigen::Vector3f& endpoint,
                                float                  quality) {
  carveRay(origin, endpoint, quality, /*inclusive_endpoint=*/true);
}

// ===========================================================================
// Carve loop — Beta a_free along the ray (occupancy grid only)
// ===========================================================================

void SemSplitMap::carveRay(const Eigen::Vector3f& origin,
                           const Eigen::Vector3f& endpoint,
                           float                  quality,
                           bool                   inclusive_endpoint) {
  const CoordT k0    = beta_grid_.posToCoord(origin.x(),   origin.y(),   origin.z());
  const CoordT k_end = beta_grid_.posToCoord(endpoint.x(), endpoint.y(), endpoint.z());
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
// Per-voxel API (for the ScovoxMapSplit fused walker)
// ===========================================================================

bool SemSplitMap::applyCarveUpdate(const CoordT& c, float quality) {
  const float w_inc = params_.w_free * quality;
  if (w_inc <= 0.f) return true;  // no-op; not a wall

  const float skip = params_.carve_skip_occ_threshold;

  BetaVoxel* v = beta_acc_.value(c, /*create_if_missing=*/false);
  if (v && v->p_occ() > skip) return false;  // wall — stop carving

  if (!v) {
    BetaVoxel nv = defaultBetaVoxel(beta_occ_prior_, beta_free_prior_);
    nv.a_free += w_inc;
    applyBetaSaturation(&nv);
    beta_acc_.setValue(c, nv);
  } else {
    v->a_free += w_inc;
    applyBetaSaturation(v);
  }
  touched_beta_.push_back(c);
  return true;
}

void SemSplitMap::applyHitUpdate(const CoordT&             c,
                                 const std::vector<float>* sem_probs,
                                 float                     quality) {
  // ---- Stream A: occupancy (Beta grid), always. ----
  BetaVoxel* b = getOrAllocateBeta(c);
  const float w_occ_share = params_.w_occ * quality;
  if (w_occ_share > 0.f) b->a_occ += w_occ_share;
  applyBetaSaturation(b);
  touched_beta_.push_back(c);

  // p_occ_post — read AFTER Stream A lands (matches SemDir/SemBeta timing).
  const float p_occ_post = b->p_occ();

  // ---- Stream B: class commit (Dir grid), gated. ----
  // The Dir voxel is allocated lazily, only when a class is actually
  // committed — free / below-gate voxels never allocate a 16 B DirVoxel.
  switch (params_.semantic_mode) {
    case SemanticMode::NAIVE:
      if (p_occ_post > 0.5f) {
        DirVoxel* d = getOrAllocateDir(c);
        naiveUpdate(d, sem_probs, params_.alpha_0);
        applyDirSaturation(d);
        touched_dir_.push_back(c);
      }
      break;

    case SemanticMode::MAJORITY_VOTE:
      if (p_occ_post > 0.5f) {
        DirVoxel* d = getOrAllocateDir(c);
        majorityVoteUpdate(d, sem_probs, params_.alpha_0);
        applyDirSaturation(d);
        touched_dir_.push_back(c);
      }
      break;

    case SemanticMode::DIRICHLET:
    default:
      if (p_occ_post >= params_.dirichlet_min_p_occ) {
        const float class_share = params_.kappa0 * p_occ_post * quality;
        if (class_share > 0.f) {
          DirVoxel* d = getOrAllocateDir(c);
          dirichletUpdate(d, sem_probs, class_share, params_.alpha_0);
          applyDirSaturation(d);
          touched_dir_.push_back(c);
        }
      }
      break;
  }
}

// ===========================================================================
// Evidence saturation — uniform per-grid scale-down (preserves marginals)
// ===========================================================================

void SemSplitMap::applyBetaSaturation(BetaVoxel* b) const {
  const float cap = params_.evidence_saturation;
  if (cap <= 0.f) return;
  const float s = b->s_total();
  if (s <= cap) return;
  const float k = cap / s;            // preserves p_occ
  b->a_occ  *= k;
  b->a_free *= k;
}

void SemSplitMap::applyDirSaturation(DirVoxel* d) const {
  const float cap = params_.evidence_saturation;
  if (cap <= 0.f) return;
  const float s = d->s_class();
  if (s <= cap) return;
  const float k = cap / s;            // preserves per-class probabilities
  d->other *= k;
  for (int i = 0; i < K_TOP; ++i) d->cnt[i] *= k;
}

// ===========================================================================
// Touched-set drains
// ===========================================================================

namespace {
std::vector<Bonxai::CoordT> sortUnique(std::vector<Bonxai::CoordT>&& in) {
  std::sort(in.begin(), in.end(), [](const Bonxai::CoordT& a, const Bonxai::CoordT& b) {
    if (a.x != b.x) return a.x < b.x;
    if (a.y != b.y) return a.y < b.y;
    return a.z < b.z;
  });
  in.erase(std::unique(in.begin(), in.end(),
                       [](const Bonxai::CoordT& a, const Bonxai::CoordT& b) {
                         return a.x == b.x && a.y == b.y && a.z == b.z;
                       }),
           in.end());
  return std::move(in);
}
}  // namespace

std::vector<SemSplitMap::CoordT> SemSplitMap::drainTouchedBeta() {
  std::vector<CoordT> out = sortUnique(std::move(touched_beta_));
  touched_beta_.clear();
  return out;
}

std::vector<SemSplitMap::CoordT> SemSplitMap::drainTouchedDir() {
  std::vector<CoordT> out = sortUnique(std::move(touched_dir_));
  touched_dir_.clear();
  return out;
}

// ===========================================================================
// Queries
// ===========================================================================

std::optional<BetaVoxel> SemSplitMap::getBetaVoxel(const Eigen::Vector3f& pos) const {
  auto acc = beta_grid_.createConstAccessor();
  const CoordT c = beta_grid_.posToCoord(pos.x(), pos.y(), pos.z());
  const BetaVoxel* v = acc.value(c);
  if (!v) return std::nullopt;
  return *v;
}

std::optional<DirVoxel> SemSplitMap::getDirVoxel(const Eigen::Vector3f& pos) const {
  auto acc = dir_grid_.createConstAccessor();
  const CoordT c = dir_grid_.posToCoord(pos.x(), pos.y(), pos.z());
  const DirVoxel* v = acc.value(c);
  if (!v) return std::nullopt;
  return *v;
}

uint16_t SemSplitMap::dominantClassAt(const Eigen::Vector3f& pos) const {
  auto acc = dir_grid_.createConstAccessor();
  const CoordT c = dir_grid_.posToCoord(pos.x(), pos.y(), pos.z());
  const DirVoxel* v = acc.value(c);
  if (!v) return 0xFFFF;
  return dominantClass(*v, params_.alpha_0);
}

std::size_t SemSplitMap::betaVoxelCount() const { return beta_grid_.activeCellsCount(); }
std::size_t SemSplitMap::dirVoxelCount()  const { return dir_grid_.activeCellsCount(); }

}  // namespace scovox
