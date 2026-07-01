/// @file
/// @brief Split Beta/Dirichlet integration — two parallel Bonxai grids.
/// De-unifies `SemDirMap` into a `BetaVoxel` occupancy grid + a `DirVoxel`
/// occupied-class grid, using the SemBeta two-stream update with SemDir-matched
/// priors and strict per-grid mass conservation.

#include "scovox/sem_split_map.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

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
    , transient_beta_grid_(params_.resolution, params_.inner_bits, params_.leaf_bits)
    , transient_dir_grid_(params_.resolution, params_.inner_bits, params_.leaf_bits)
    , beta_acc_(beta_grid_.createAccessor())
    , dir_acc_(dir_grid_.createAccessor())
    , transient_beta_acc_(transient_beta_grid_.createAccessor())
    , transient_dir_acc_(transient_dir_grid_.createAccessor())
    , touched_beta_()
    , touched_dir_()
    // Shipped occupancy prior is symmetric Beta(1,1) → p_occ=0.5, decoupled from
    // the semantic (num_classes, α₀). See docs/occupancy_prior.md.
    , beta_occ_prior_(kBetaOccPrior)
    , beta_free_prior_(kBetaFreePrior) {}

// ===========================================================================
// Allocation (enforce prior at first touch — Bonxai zero-inits leaf blocks)
// ===========================================================================

BetaVoxel* SemSplitMap::getOrAllocateBetaOn(BetaGrid::Accessor& acc, const CoordT& c) {
  BetaVoxel* v = acc.value(c, /*create_if_missing=*/false);
  if (v) return v;
  acc.setValue(c, defaultBetaVoxel(beta_occ_prior_, beta_free_prior_));
  return acc.value(c, /*create_if_missing=*/false);
}

DirVoxel* SemSplitMap::getOrAllocateDirOn(DirGrid::Accessor& acc, const CoordT& c) {
  DirVoxel* v = acc.value(c, /*create_if_missing=*/false);
  if (v) return v;
  acc.setValue(c, defaultDirVoxel(params_.num_classes, params_.alpha_0));
  return acc.value(c, /*create_if_missing=*/false);
}

BetaVoxel* SemSplitMap::getOrAllocateBeta(const CoordT& c) {
  return getOrAllocateBetaOn(beta_acc_, c);
}

DirVoxel* SemSplitMap::getOrAllocateDir(const CoordT& c) {
  return getOrAllocateDirOn(dir_acc_, c);
}

// ===========================================================================
// Public integration entry points
// ===========================================================================

void SemSplitMap::integrateHit(const Eigen::Vector3f&    origin,
                               const Eigen::Vector3f&    endpoint,
                               const std::vector<float>* sem_probs,
                               float                     quality,
                               const HitWeights*         prof) {
  carveRay(origin, endpoint, quality, /*inclusive_endpoint=*/false, prof);
  const CoordT k_hit = beta_grid_.posToCoord(endpoint.x(), endpoint.y(), endpoint.z());
  applyHitUpdate(k_hit, sem_probs, quality, prof);
}

void SemSplitMap::integrateHit(const Eigen::Vector3f&    origin,
                               const Eigen::Vector3f&    endpoint,
                               const std::vector<float>* sem_probs,
                               float                     quality,
                               bool                      is_dynamic,
                               const HitWeights*         prof) {
  // Free-space carve is always persistent; only the endpoint hit is routed.
  carveRay(origin, endpoint, quality, /*inclusive_endpoint=*/false, prof);
  const CoordT k_hit = beta_grid_.posToCoord(endpoint.x(), endpoint.y(), endpoint.z());
  applyHitUpdate(k_hit, sem_probs, quality, is_dynamic, prof);
}

void SemSplitMap::integrateMiss(const Eigen::Vector3f& origin,
                                const Eigen::Vector3f& endpoint,
                                float                  quality,
                                const HitWeights*      prof) {
  carveRay(origin, endpoint, quality, /*inclusive_endpoint=*/true, prof);
}

// ===========================================================================
// Carve loop — Beta a_free along the ray (occupancy grid only)
// ===========================================================================

void SemSplitMap::carveRay(const Eigen::Vector3f& origin,
                           const Eigen::Vector3f& endpoint,
                           float                  quality,
                           bool                   inclusive_endpoint,
                           const HitWeights*      prof) {
  const CoordT k0    = beta_grid_.posToCoord(origin.x(),   origin.y(),   origin.z());
  const CoordT k_end = beta_grid_.posToCoord(endpoint.x(), endpoint.y(), endpoint.z());
  if (k0 == k_end) return;

  // Per-source w_free (prof) overrides the global one for this ray. A semantics-
  // only source (prof->w_free == 0, e.g. RGB-D "pure LiDAR authority") short-
  // circuits the whole carve here, so it never deposits a_free onto the shared
  // Beta grid — matching applyCarveUpdate's own w_inc<=0 no-op per voxel.
  const float w_free = prof ? prof->w_free : params_.w_free;
  const float w_inc  = w_free * quality;
  if (w_inc <= 0.f && !inclusive_endpoint) return;

  RayIterator(k0, k_end, [&](const CoordT& c) -> bool {
    if (c == k_end) return false;  // hit voxel handled separately for hits
    return applyCarveUpdate(c, quality, prof);
  });

  if (inclusive_endpoint) {
    (void)applyCarveUpdate(k_end, quality, prof);
  }
}

// ===========================================================================
// Per-voxel API (for the ScovoxMapSplit fused walker)
// ===========================================================================

bool SemSplitMap::applyCarveUpdate(const CoordT& c, float quality,
                                   const HitWeights* prof) {
  const float w_free = prof ? prof->w_free : params_.w_free;
  const float w_inc  = w_free * quality;
  if (w_inc <= 0.f) return true;  // no-op; not a wall

  // Batched path (a carve frame is open — the live pipeline): stage the
  // strongest free vote for this voxel and defer the write to flushCarveFrame.
  // No grid read, no wall guard: a scan trusts its own beam — every voxel it
  // traversed to reach a return is free NOW (see class docs). One write per
  // voxel per scan, block-ordered at flush.
  if (carve_frame_open_) {
    auto it = carve_stage_.find(c);
    if (it == carve_stage_.end()) carve_stage_.emplace(c, w_inc);
    else if (w_inc > it->second)  it->second = w_inc;
    return true;
  }

  // Immediate path (no frame open): write in place. The wall guard is OFF by
  // default (carve_skip_occ_threshold <= 0); a positive threshold restores the
  // legacy occupancy-blocked carve for direct callers (offline tools / ablations).
  const float skip = params_.carve_skip_occ_threshold;
  BetaVoxel* v = beta_acc_.value(c, /*create_if_missing=*/false);
  if (skip > 0.f && v && v->p_occ() > skip) return false;  // wall — stop carving

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

// ===========================================================================
// Batched per-scan carve — begin / flush
// ===========================================================================

void SemSplitMap::beginCarveFrame() {
  carve_stage_.clear();   // retains bucket capacity → no per-scan realloc churn
  carve_hits_.clear();
  carve_frame_open_ = true;
}

std::size_t SemSplitMap::flushCarveFrame() {
  if (!carve_frame_open_) return 0;

  // Gather staged voxels, dropping any a same-scan hit claimed as surface
  // (occupied-wins). Then sort by leaf block so all writes into one 8^3 block
  // are consecutive and reuse the cached accessor — recovering the locality the
  // scan-order per-ray scatter throws away.
  std::vector<std::pair<CoordT, float>> items;
  items.reserve(carve_stage_.size());
  for (const auto& kv : carve_stage_) {
    if (carve_hits_.count(kv.first)) continue;  // occupied-wins
    items.push_back(kv);
  }
  const int lb = static_cast<int>(params_.leaf_bits);
  std::sort(items.begin(), items.end(),
            [lb](const std::pair<CoordT, float>& a,
                 const std::pair<CoordT, float>& b) {
              const std::array<int32_t, 3> ba{a.first.x >> lb, a.first.y >> lb, a.first.z >> lb};
              const std::array<int32_t, 3> bb{b.first.x >> lb, b.first.y >> lb, b.first.z >> lb};
              return ba < bb;
            });

  for (const auto& item : items) {
    const CoordT& c   = item.first;
    const float   inc = item.second;
    BetaVoxel* v = beta_acc_.value(c, /*create_if_missing=*/false);
    if (!v) {
      BetaVoxel nv = defaultBetaVoxel(beta_occ_prior_, beta_free_prior_);
      nv.a_free += inc;
      applyBetaSaturation(&nv);
      beta_acc_.setValue(c, nv);
    } else {
      v->a_free += inc;
      applyBetaSaturation(v);
    }
    touched_beta_.push_back(c);
  }

  const std::size_t n = items.size();
  carve_frame_open_ = false;
  return n;
}

void SemSplitMap::applyHitUpdate(const CoordT&             c,
                                 const std::vector<float>* sem_probs,
                                 float                     quality,
                                 const HitWeights*         prof) {
  applyHitUpdateOn(c, sem_probs, quality, beta_acc_, dir_acc_,
                   &touched_beta_, &touched_dir_, prof);
}

void SemSplitMap::applyHitUpdate(const CoordT&             c,
                                 const std::vector<float>* sem_probs,
                                 float                     quality,
                                 bool                      is_dynamic,
                                 const HitWeights*         prof) {
  if (is_dynamic) {
    // Route to the transient substrate. No touched-set: transient voxels are
    // local-only and never drained to the fusion wire.
    applyHitUpdateOn(c, sem_probs, quality, transient_beta_acc_,
                     transient_dir_acc_, nullptr, nullptr, prof);
  } else {
    applyHitUpdateOn(c, sem_probs, quality, beta_acc_, dir_acc_,
                     &touched_beta_, &touched_dir_, prof);
  }
}

void SemSplitMap::applyHitUpdateOn(const CoordT&             c,
                                   const std::vector<float>* sem_probs,
                                   float                     quality,
                                   BetaGrid::Accessor&       bacc,
                                   DirGrid::Accessor&        dacc,
                                   std::vector<CoordT>*      touched_beta,
                                   std::vector<CoordT>*      touched_dir,
                                   const HitWeights*         prof) {
  // Occupied-wins: a PERSISTENT surface return in this scan must not be carved
  // free even if another ray grazes through it. Gate on `touched_beta` — it is
  // non-null only on the persistent path (the transient/dynamic path passes
  // nullptr). A dynamic endpoint routes its occupancy to the transient grid and,
  // by the is_dynamic contract, the persistent grid stays free there, so it must
  // NOT suppress another ray's legitimate persistent free carve of that voxel.
  if (carve_frame_open_ && touched_beta) carve_hits_.insert(c);

  // RGB-D→LiDAR BKI spread: a semantics-only source with a kernel radius spreads
  // its class onto nearby LiDAR-occupied voxels instead of committing at the lone
  // endpoint voxel `c` (which the LiDAR downsample almost never leaves occupied).
  // Deposits only into the Dir grid `dacc`; reads LiDAR occupancy from the
  // PERSISTENT Beta grid inside the helper. A no-label point (sem_probs null)
  // contributes nothing — a semantics-only source must not touch geometry.
  if (prof && prof->kernel_radius > 0.f) {
    if (sem_probs && !sem_probs->empty())
      applyHitUpdateKernel(c, sem_probs, quality, dacc, touched_dir, prof);
    return;
  }

  // Per-source overrides (fusion). Null prof => the map's global params_ — the
  // single-sensor path, byte-identical. A semantics-only source passes w_occ=0
  // (RGB-D "pure LiDAR authority"), so Stream A is skipped by the existing
  // `w_occ_share > 0` guard and this voxel's occupancy stays 100% LiDAR-built;
  // Stream B then gates on that LiDAR occupancy. kappa0/min_p_occ are per-source
  // too; evidence_saturation / alpha_0 remain global (per-grid caps / priors).
  const float w_occ     = prof ? prof->w_occ              : params_.w_occ;
  const float kappa0    = prof ? prof->kappa0             : params_.kappa0;
  const float min_p_occ = prof ? prof->dirichlet_min_p_occ : params_.dirichlet_min_p_occ;

  // ---- Stream A: occupancy (Beta grid), always. ----
  BetaVoxel* b = getOrAllocateBetaOn(bacc, c);
  const float w_occ_share = w_occ * quality;
  if (w_occ_share > 0.f) b->a_occ += w_occ_share;
  applyBetaSaturation(b);
  if (touched_beta) touched_beta->push_back(c);

  // p_occ_post — read AFTER Stream A lands (matches SemDir/SemBeta timing).
  const float p_occ_post = b->p_occ();

  // ---- Stream B: class commit (Dir grid), gated. ----
  // The Dir voxel is allocated lazily, only when a class is actually
  // committed — free / below-gate voxels never allocate a 16 B DirVoxel.
  switch (params_.semantic_mode) {
    case SemanticMode::NAIVE:
      if (p_occ_post > 0.5f) {
        DirVoxel* d = getOrAllocateDirOn(dacc, c);
        naiveUpdate(d, sem_probs, params_.alpha_0);
        applyDirSaturation(d);
        if (touched_dir) touched_dir->push_back(c);
      }
      break;

    case SemanticMode::MAJORITY_VOTE:
      if (p_occ_post > 0.5f) {
        DirVoxel* d = getOrAllocateDirOn(dacc, c);
        majorityVoteUpdate(d, sem_probs, params_.alpha_0);
        applyDirSaturation(d);
        if (touched_dir) touched_dir->push_back(c);
      }
      break;

    case SemanticMode::DIRICHLET:
    default:
      if (p_occ_post >= min_p_occ) {
        const float class_share = kappa0 * p_occ_post * quality;
        if (class_share > 0.f) {
          DirVoxel* d = getOrAllocateDirOn(dacc, c);
          dirichletUpdate(d, sem_probs, class_share, params_.alpha_0);
          applyDirSaturation(d);
          if (touched_dir) touched_dir->push_back(c);
        }
      }
      break;
  }
}

// ===========================================================================
// BKI (S-BKI) semantic spread — RGB-D→LiDAR fusion
// ===========================================================================
//
// For a semantics-only source (RGB-D: w_occ=0, geometry_off) with
// `prof->kernel_radius = l > 0`, the class is not committed at the single
// endpoint voxel `c` but spread to its neighborhood, following the Semantic
// Bayesian Kernel Inference update (Gan et al., RA-L 2020, Eq. 9):
//
//     α*ᵏ  +=  k(d) · (κ₀ · p_occ · q)      for every voxel within radius l
//
// with the Melkumyan–Ramos compactly-supported sparse kernel (Eq. 10, σ₀=1),
// which is exactly zero at d ≥ l so the neighborhood is finite:
//
//     k(d) = ⅓(2+cos(2π d/l))(1 − d/l) + (1/2π)·sin(2π d/l),   d < l.
//
// Pure LiDAR authority is preserved: a neighbor receives a label ONLY if it is
// occupied in the PERSISTENT Beta grid (`p_occ ≥ dirichlet_min_p_occ`), so
// RGB-D can never paint a voxel LiDAR hasn't confirmed as surface. `p_occ` also
// weights the deposit, so weakly-occupied voxels get proportionally less label.
void SemSplitMap::applyHitUpdateKernel(const CoordT&             c,
                                       const std::vector<float>* sem_probs,
                                       float                     quality,
                                       DirGrid::Accessor&        dacc,
                                       std::vector<CoordT>*      touched_dir,
                                       const HitWeights*         prof) {
  const float l         = prof->kernel_radius;                 // > 0 (caller-gated)
  const float kappa0    = prof->kappa0;
  const float min_p_occ = prof->dirichlet_min_p_occ;
  const float res       = static_cast<float>(params_.resolution);
  if (res <= 0.f) return;
  const int   R         = std::max(1, static_cast<int>(std::floor(l / res)));
  const float inv_l     = 1.0f / l;
  constexpr float kTwoPi = 6.283185307179586f;

  for (int dz = -R; dz <= R; ++dz)
    for (int dy = -R; dy <= R; ++dy)
      for (int dx = -R; dx <= R; ++dx) {
        const float d = res * std::sqrt(static_cast<float>(dx*dx + dy*dy + dz*dz));
        if (d >= l) continue;                                  // compact support

        const CoordT n{c.x + dx, c.y + dy, c.z + dz};
        // LiDAR authority: read persistent occupancy; skip voxels LiDAR never
        // built (nullptr) or that are not confidently occupied. Never allocates.
        const BetaVoxel* b = beta_acc_.value(n, /*create_if_missing=*/false);
        if (!b) continue;
        const float p_occ = b->p_occ();
        if (p_occ < min_p_occ) continue;

        const float t  = d * inv_l;                            // in [0,1)
        const float wk = (1.0f/3.0f) * (2.0f + std::cos(kTwoPi * t)) * (1.0f - t)
                       + (1.0f/kTwoPi) * std::sin(kTwoPi * t);
        const float class_share = kappa0 * p_occ * quality * wk;
        if (class_share <= 0.f) continue;

        DirVoxel* dv = getOrAllocateDirOn(dacc, n);
        dirichletUpdate(dv, sem_probs, class_share, params_.alpha_0);
        applyDirSaturation(dv);
        if (touched_dir) touched_dir->push_back(n);
      }
}

// ===========================================================================
// Transient substrate — per-frame decay + queries
// ===========================================================================

void SemSplitMap::decayTransient(float rate) {
  if (rate < 0.f) rate = 0.f;
  if (rate > 1.f) rate = 1.f;
  // Voxels whose evidence has decayed to within this much of the prior carry no
  // meaningful signal; prune them so the transient grids stay bounded.
  constexpr float kPruneEps = 1e-3f;

  // ---- Beta: decay a_occ / a_free toward their priors, prune at-prior cells.
  {
    const float ap = beta_occ_prior_;
    const float fp = beta_free_prior_;
    std::vector<CoordT> prune;
    transient_beta_grid_.forEachCell([&](BetaVoxel& v, const CoordT& c) {
      v.a_occ  = ap + (v.a_occ  - ap) * rate;
      v.a_free = fp + (v.a_free - fp) * rate;
      if (std::fabs(v.a_occ - ap) < kPruneEps && std::fabs(v.a_free - fp) < kPruneEps)
        prune.push_back(c);
    });
    auto acc = transient_beta_grid_.createAccessor();
    for (const auto& c : prune) acc.setCellOff(c);
  }

  // ---- Dir: decay cnt toward α₀ and other toward its (C−K_TOP)·α₀ prior; free
  //      a slot that decays to prior, prune the voxel when nothing is left.
  {
    const float a0 = params_.alpha_0;
    const int   residual = static_cast<int>(params_.num_classes) - K_TOP;
    const float other_prior = (residual > 0) ? (residual * a0) : 0.f;
    std::vector<CoordT> prune;
    transient_dir_grid_.forEachCell([&](DirVoxel& v, const CoordT& c) {
      v.other = other_prior + (v.other - other_prior) * rate;
      bool any = std::fabs(v.other - other_prior) >= kPruneEps;
      for (int i = 0; i < K_TOP; ++i) {
        if (v.cls[i] == 0xFFFF) continue;
        v.cnt[i] = a0 + (v.cnt[i] - a0) * rate;
        if (std::fabs(v.cnt[i] - a0) < kPruneEps) {
          v.cls[i] = 0xFFFF;   // slot faded to prior → release it
          v.cnt[i] = a0;
        } else {
          any = true;
        }
      }
      if (!any) prune.push_back(c);
    });
    auto acc = transient_dir_grid_.createAccessor();
    for (const auto& c : prune) acc.setCellOff(c);
  }
}

std::optional<BetaVoxel> SemSplitMap::getTransientBetaVoxel(const Eigen::Vector3f& pos) const {
  auto acc = transient_beta_grid_.createConstAccessor();
  const CoordT c = transient_beta_grid_.posToCoord(pos.x(), pos.y(), pos.z());
  const BetaVoxel* v = acc.value(c);
  if (!v) return std::nullopt;
  return *v;
}

std::optional<DirVoxel> SemSplitMap::getTransientDirVoxel(const Eigen::Vector3f& pos) const {
  auto acc = transient_dir_grid_.createConstAccessor();
  const CoordT c = transient_dir_grid_.posToCoord(pos.x(), pos.y(), pos.z());
  const DirVoxel* v = acc.value(c);
  if (!v) return std::nullopt;
  return *v;
}

uint16_t SemSplitMap::transientDominantClassAt(const Eigen::Vector3f& pos) const {
  auto acc = transient_dir_grid_.createConstAccessor();
  const CoordT c = transient_dir_grid_.posToCoord(pos.x(), pos.y(), pos.z());
  const DirVoxel* v = acc.value(c);
  if (!v) return 0xFFFF;
  return dominantClass(*v, params_.alpha_0, params_.num_classes);
}

std::size_t SemSplitMap::transientBetaVoxelCount() const {
  return transient_beta_grid_.activeCellsCount();
}
std::size_t SemSplitMap::transientDirVoxelCount() const {
  return transient_dir_grid_.activeCellsCount();
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
  const float k       = cap / s;      // preserves per-class probabilities
  const float alpha_0 = params_.alpha_0;
  d->other *= k;
  for (int i = 0; i < K_TOP; ++i) {
    d->cnt[i] *= k;
    // A FILLED slot must never scale below its α₀ prior: a slot conceptually
    // holds α₀ + observed evidence, and eroding α₀ makes sparse_add_class read a
    // negative evicted_evidence (cnt − α₀ < 0) and subtract mass from OTHER. Floor
    // FILLED slots only — flooring empty slots (cnt ≈ k·α₀) would re-inflate
    // s_class back above the saturation cap.
    if (d->cls[i] != 0xFFFF && d->cnt[i] < alpha_0) d->cnt[i] = alpha_0;
  }
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
  return dominantClass(*v, params_.alpha_0, params_.num_classes);
}

std::size_t SemSplitMap::betaVoxelCount() const { return beta_grid_.activeCellsCount(); }
std::size_t SemSplitMap::dirVoxelCount()  const { return dir_grid_.activeCellsCount(); }

}  // namespace scovox
