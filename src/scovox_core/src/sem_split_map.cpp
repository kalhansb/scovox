/// @file
/// @brief Split Beta/Dirichlet integration — two parallel Bonxai grids.
/// De-unifies `SemDirMap` into a `BetaVoxel` occupancy grid + a `DirVoxel`
/// occupied-class grid, using the SemBeta two-stream update with SemDir-matched
/// priors and strict per-grid mass conservation.

#include "scovox/sem_split_map.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "scovox/ray_iterator.hpp"

namespace scovox {

#if SCOVOX_DEPOSIT_TRACE
DepositTraceFn g_deposit_trace = nullptr;
#endif

namespace {

/// Single entry point to `sparse_add_class` from this file. Folds the two
/// build-define branches (`SCOVOX_TRACK_QMAX`, `SCOVOX_TRACK_NHIT`) into one
/// place so the deposit paths below read as plain calls, and so a new tracked
/// field never has to be threaded through them again. Behaviour is unchanged:
/// with both defines off this is exactly the six-argument shipped call.
inline void sparse_add_class_traced(DirVoxel* d, uint16_t c, float inc,
                                    float alpha_0, float p,
                                    bool evict_by_confidence,
                                    uint8_t* outcome) {
#if SCOVOX_TRACK_QMAX
  uint16_t* qmax = d->qmax;
#else
  uint16_t* qmax = nullptr;
  (void)evict_by_confidence;
  (void)p;
#endif
#if SCOVOX_TRACK_NHIT
  uint16_t* nhit = d->nhit;
#else
  uint16_t* nhit = nullptr;
#endif
#if SCOVOX_TRACK_QMAX
  sparse_add_class(d->cnt, d->cls, c, inc, &d->other, alpha_0, p, qmax,
                   evict_by_confidence, nhit, outcome);
#else
  sparse_add_class(d->cnt, d->cls, c, inc, &d->other, alpha_0, -1.0f, nullptr,
                   true, nhit, outcome);
#endif
}

/// Sparse-Dirichlet hit update on a `DirVoxel` (DIRICHLET mode). Mirrors
/// `dirichletUpdate` in semdir_map.cpp / sembeta_map.cpp: distribute
/// `class_share` over the observed softmax, routing uncovered + evicted mass
/// to OTHER. Total mass added to the voxel is exactly `class_share`.
void dirichletUpdate(DirVoxel*                 d,
                     const std::vector<float>* class_probs,
                     float                     class_share,
                     float                     alpha_0,
                     bool                      evict_by_confidence = false,
                     int                       inc_mode  = 0,
                     float                     inc_thresh = 0.0f,
                     int32_t                   tx = 0,
                     int32_t                   ty = 0,
                     int32_t                   tz = 0) {
#if SCOVOX_DEPOSIT_TRACE
  // Snapshot before each deposit so the record can report the slot state the
  // comparator actually saw. Which slot a deposit touched is recoverable from
  // the branch code plus this snapshot, so `sparse_add_class` does not need to
  // report it: match/fill/evict all leave `c` sitting in the slot they used,
  // and a drop is by definition weighed against the weakest slot.
  auto trace_one = [&](uint16_t c, float p, float inc) {
    if (!g_deposit_trace) {
      sparse_add_class_traced(d, c, inc, alpha_0, p, evict_by_confidence, nullptr);
      return;
    }
    float    pre_cnt [K_TOP];
    uint16_t pre_cls [K_TOP];
    uint16_t pre_nhit[K_TOP];
    int      weakest = 0;
    for (int i = 0; i < K_TOP; ++i) {
      pre_cnt[i]  = d->cnt[i];
      pre_cls[i]  = d->cls[i];
#if SCOVOX_TRACK_NHIT
      pre_nhit[i] = d->nhit[i];
#else
      pre_nhit[i] = 0;
#endif
      if (d->cnt[i] < d->cnt[weakest]) weakest = i;
    }
    uint8_t outcome = 255;
    sparse_add_class_traced(d, c, inc, alpha_0, p, evict_by_confidence, &outcome);
    uint8_t slot = 0xFF;
    if (outcome == 1 || outcome == 2 || outcome == 3) {
      for (int i = 0; i < K_TOP; ++i) if (d->cls[i] == c) { slot = static_cast<uint8_t>(i); break; }
    } else if (outcome == 4) {
      slot = static_cast<uint8_t>(weakest);
    }
    const float    cb = (slot == 0xFF) ? 0.f : pre_cnt[slot];
    const uint16_t nb = (slot == 0xFF) ? uint16_t{0}
                      : (pre_cls[slot] == 0xFFFF ? uint16_t{0} : pre_nhit[slot]);
    g_deposit_trace(tx, ty, tz, c, p, inc, outcome, slot, cb, nb);
  };
#else
  (void)tx; (void)ty; (void)tz;
  auto trace_one = [&](uint16_t c, float p, float inc) {
    sparse_add_class_traced(d, c, inc, alpha_0, p, evict_by_confidence, nullptr);
  };
#endif
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

  // HARD: collapse the observation onto its argmax before depositing anything.
  // The same total mass moves, but it is not split — one class takes all of
  // `class_share`, and the confidence the comparator sees is still that class's
  // own (normalized) probability, not the share.
  if (inc_mode == 1) {
    size_t bi = 0;
    float  bp = -1.f;
    for (size_t i = 0; i < obs.size(); ++i) if (obs[i] > bp) { bp = obs[i]; bi = i; }
    if (bp <= 0.f) { d->other += class_share; return; }
    const float p_i = bp * norm;
    trace_one(static_cast<uint16_t>(bi), p_i, class_share);
    return;
  }

  float deposited = 0.f;
  for (size_t i = 0; i < obs.size(); ++i) {
    if (obs[i] <= 0.f) continue;
    // The class's own (normalized) probability: what fraction of this
    // observation backs class i, independent of how much mass the voxel is
    // being given. That is the confidence the eviction comparator wants —
    // `inc` mixes it with class_share and so tracks the geometry, not the label.
    const float p_i = obs[i] * norm;
    // THRESH: a class this unsure contributes no evidence at all. Its mass is
    // not deleted — the residual line below routes it to OTHER, so the strict
    // invariant still holds.
    if (inc_mode == 2 && p_i < inc_thresh) continue;
    const float inc = class_share * p_i;
    if (inc <= 0.f) continue;
    deposited += inc;
    // qmax is recorded unconditionally (see sparse_add_class_traced): it costs
    // nothing on a build that has the field, it never reaches the comparator
    // unless `evict_by_confidence` says so, and without it an evidence-eviction
    // dump carries no confidence trail for an offline readout rule to read.
    trace_one(static_cast<uint16_t>(i), p_i, inc);
  }
  if (inc_mode == 2) {
    d->other += class_share - deposited;
  } else {
    // Kept in its original form so the SOFT path stays bit-identical to the
    // pre-patch build rather than merely algebraically equal.
    const float covered = sum_p * norm;
    d->other += class_share * (1.0f - covered);
  }
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
  // Dir-grid block geometry: independent of the Beta grid's, but never coarser
  // (a sparser grid never wants BIGGER blocks) and never below Bonxai's
  // minimum of 1. 0 is accepted as "inherit leaf_bits".
  if (p.dir_leaf_bits == 0)          p.dir_leaf_bits      = p.leaf_bits;
  if (p.dir_leaf_bits > p.leaf_bits) p.dir_leaf_bits      = p.leaf_bits;
  // Band and ball are two answers to the same question; running both would
  // double-deposit every endpoint and make neither arm interpretable. The ball
  // is the older knob, so it wins and the band is dropped — a config asking for
  // both gets the documented `semantic_spread_radius` behaviour, not a hybrid.
  if (p.semantic_band_length < 0.f)  p.semantic_band_length = 0.f;
  if (p.semantic_spread_radius > 0.f) p.semantic_band_length = 0.f;
  // The ray spread is a third answer to the same deposit-shape question; the
  // ball and the band each already exclude the other, and the spread yields to
  // both for the same reason — two spreads at once would double-deposit and
  // make neither arm interpretable.
  if (p.ray_spread < 0 || p.ray_spread > 4) p.ray_spread = 0;
  if (p.semantic_spread_radius > 0.f || p.semantic_band_length > 0.f)
    p.ray_spread = 0;
  return p;
}

}  // namespace

// ===========================================================================
// Construction
// ===========================================================================

SemSplitMap::SemSplitMap(const Params& p)
    : params_(sanitise(p))
    , beta_grid_(params_.resolution, params_.inner_bits, params_.leaf_bits)
    // Dir grids get their OWN (smaller) leaf blocks — see Params::dir_leaf_bits.
    // Coord identity with the Beta grid is unaffected: posToCoord/coordToPos
    // depend only on `resolution`.
    , dir_grid_(params_.resolution, params_.inner_bits, params_.dir_leaf_bits)
    , transient_beta_grid_(params_.resolution, params_.inner_bits, params_.leaf_bits)
    , transient_dir_grid_(params_.resolution, params_.inner_bits, params_.dir_leaf_bits)
    , fallback_dir_grid_(params_.resolution, params_.inner_bits, params_.dir_leaf_bits)
    , beta_acc_(beta_grid_.createAccessor())
    , dir_acc_(dir_grid_.createAccessor())
    , transient_beta_acc_(transient_beta_grid_.createAccessor())
    , transient_dir_acc_(transient_dir_grid_.createAccessor())
    , fallback_dir_acc_(fallback_dir_grid_.createAccessor())
    , touched_beta_()
    , touched_dir_()
    // Beta-grid block geometry: the flush walk must reproduce the accessor's
    // own leaf order, so the stage is keyed by the SAME (sanitised) leaf_bits.
    , carve_stage_(params_.leaf_bits)
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
  if (params_.ray_spread != 0)
    raySpreadDeposit(origin, endpoint, k_hit, sem_probs, quality, prof);
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
  // A dynamic endpoint routes to the transient substrate; smearing a moving
  // object's class into persistent neighbours would defeat that routing, so
  // the spread applies to persistent hits only.
  if (params_.ray_spread != 0 && !is_dynamic)
    raySpreadDeposit(origin, endpoint, k_hit, sem_probs, quality, prof);
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
    if (!params_.batch_free_carve) return true;
    carve_stage_.add(c, w_inc);  // per-voxel max, keyed by leaf block
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
  carve_stage_.beginFrame();  // retains all capacity → no per-scan realloc churn
  carve_hits_.clear();
  carve_frame_open_ = true;
}

std::size_t SemSplitMap::flushCarveFrame() {
  if (!carve_frame_open_) return 0;

  // Walk staged voxels leaf-block-ordered — CarveStage stages by block, so no
  // per-voxel sort is needed; the block sequence is the same ascending
  // (x>>lb, y>>lb, z>>lb) order the retired sort produced, keeping all writes
  // into one block consecutive on the cached accessor AND first-touching Beta
  // root-map blocks in the identical order (serialized-bytes invariant).
  // Same-scan hit voxels are dropped inline (occupied-wins); a block whose
  // staged voxels are ALL hits is never written, so it allocates nothing —
  // exactly as when the retired path filtered before sorting.
  std::size_t n = 0;
  carve_stage_.forEachStagedBlockOrdered([&](const CoordT& c, float inc) {
    if (carve_hits_.count(c)) return;  // occupied-wins
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
    ++n;
  });

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
      applyHitUpdateKernel(c, sem_probs, quality, dacc, touched_dir,
                           prof->kernel_radius, prof->kappa0,
                           prof->dirichlet_min_p_occ);
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
      // Single-sensor semantic spread. Stream A has already committed this
      // scan's occupancy at `c` above, so the endpoint is in the neighbourhood
      // the kernel sees and carries its freshly-updated p_occ — the spread adds
      // neighbours, it does not redirect the deposit away from the hit.
      if (params_.semantic_spread_radius > 0.f && sem_probs && !sem_probs->empty()) {
        applyHitUpdateKernel(c, sem_probs, quality, dacc, touched_dir,
                             params_.semantic_spread_radius, kappa0, min_p_occ);
        break;
      }
      if (p_occ_post >= min_p_occ) {
        const float class_share = kappa0 * p_occ_post * quality;
        if (class_share > 0.f) {
          DirVoxel* d = getOrAllocateDirOn(dacc, c);
          dirichletUpdate(d, sem_probs, class_share, params_.alpha_0,
                          params_.evict_by_confidence,
                          params_.inc_mode, params_.inc_thresh, c.x, c.y, c.z);
          applyDirSaturation(d);
          if (touched_dir) touched_dir->push_back(c);
        }
      }
      break;
  }
}

// ===========================================================================
// §16 three-voxel ray spread — Stream B only, at the surface's ray neighbours
// ===========================================================================
//
// Geometry is build_stream_ray3.py's exactly: normalise the world-space
// viewing ray, take the exact DDA step from the endpoint past the next voxel
// boundary along +d (behind) and −d (front), nudged by 1e-6 m, in double
// precision. Both neighbours are distinct from the surface voxel and from
// each other by construction. The offline builder dropped a neighbour that
// left its GT-bounded grid; the native map is unbounded, so that margin case
// does not arise.
//
// Evidence: each neighbour takes the SAME `class_share` the surface deposit
// used, through the same `dirichletUpdate` machinery (slots, eviction,
// inc_mode, saturation, trace hook — a spread deposit is a deposit). The
// surface's own `dirichlet_min_p_occ` admission is checked once, at the
// surface; the target's occupancy is read never and written never (see the
// Params doc for why a target-side gate would measure the gate, not the
// spread, and why Stream A stays endpoint-only).

void SemSplitMap::raySpreadDeposit(const Eigen::Vector3f&    origin,
                                   const Eigen::Vector3f&    endpoint,
                                   const CoordT&             k_hit,
                                   const std::vector<float>* sem_probs,
                                   float                     quality,
                                   const HitWeights*         prof) {
  if (params_.semantic_mode != SemanticMode::DIRICHLET) return;
  if (!sem_probs || sem_probs->empty()) return;
  // The kernel path keeps its own spread semantics (see applyHitUpdateOn).
  if (prof && prof->kernel_radius > 0.f) return;

  // Mirror the surface deposit's admission and share. Stream A has already
  // landed at k_hit, so this p_occ is the one the endpoint commit itself saw.
  const float kappa0    = prof ? prof->kappa0              : params_.kappa0;
  const float min_p_occ = prof ? prof->dirichlet_min_p_occ : params_.dirichlet_min_p_occ;
  const BetaVoxel* b = beta_acc_.value(k_hit, /*create_if_missing=*/false);
  if (!b) return;
  const float p_occ_post = b->p_occ();
  if (p_occ_post < min_p_occ) return;
  const float class_share = kappa0 * p_occ_post * quality;
  if (class_share <= 0.f) return;

  const Eigen::Vector3d w = endpoint.cast<double>();
  Eigen::Vector3d d = (endpoint - origin).cast<double>();
  const double norm = d.norm();
  if (norm <= 0.0) return;
  d /= norm;

  const double  res   = params_.resolution;
  const int32_t cc[3] = {k_hit.x, k_hit.y, k_hit.z};

  // First voxel strictly past the next boundary from `w` along `sign * d`.
  // Voxel c spans [c·res, (c+1)·res) (Bonxai posToCoord is floor(p/res)).
  auto step_coord = [&](double sign, CoordT* out) -> bool {
    const Eigen::Vector3d dir = sign * d;
    double tmin = std::numeric_limits<double>::infinity();
    for (int a = 0; a < 3; ++a) {
      if (dir[a] == 0.0) continue;
      const double bound = (dir[a] > 0.0 ? double(cc[a] + 1) : double(cc[a])) * res;
      const double t = (bound - w[a]) / dir[a];
      if (t > 0.0 && t < tmin) tmin = t;
    }
    if (!std::isfinite(tmin)) return false;
    constexpr double kStepEps = 1e-6;  // metres past the boundary, as offline
    const Eigen::Vector3d p = w + (tmin + kStepEps) * dir;
    *out = dir_grid_.posToCoord(p.x(), p.y(), p.z());
    return true;
  };

  const int mode = params_.ray_spread;  // 1 behind, 2 front, 3 all3, 4 fallback
  CoordT tc;
  if (mode != 2 && step_coord(+1.0, &tc)) {   // behind: modes 1, 3, 4
    DirVoxel* dv = getOrAllocateDirOn(dir_acc_, tc);
    dirichletUpdate(dv, sem_probs, class_share, params_.alpha_0,
                    params_.evict_by_confidence, params_.inc_mode,
                    params_.inc_thresh, tc.x, tc.y, tc.z);
    applyDirSaturation(dv);
    touched_dir_.push_back(tc);
  }
  if (mode != 1 && step_coord(-1.0, &tc)) {   // front: modes 2, 3, 4
    // Mode 4 records the front claim in the fallback grid, which never feeds
    // the fusion wire — no touched-set entry, like the transient substrate.
    const bool to_fallback = (mode == 4);
    DirVoxel* dv = getOrAllocateDirOn(to_fallback ? fallback_dir_acc_ : dir_acc_, tc);
    dirichletUpdate(dv, sem_probs, class_share, params_.alpha_0,
                    params_.evict_by_confidence, params_.inc_mode,
                    params_.inc_thresh, tc.x, tc.y, tc.z);
    applyDirSaturation(dv);
    if (!to_fallback) touched_dir_.push_back(tc);
  }
}

// ===========================================================================
// SLIM-VDB-style flat semantic band — Stream B only, on the walked ray
// ===========================================================================
//
// SLIM-VDB's Integrate (VDBVolume.cpp) walks [depth−sdf_trunc, depth+sdf_trunc]
// per point because that is what the TSDF update needs, and folds the semantic
// write into the same DDA iteration:
//
//     if (sdf > -sdf_trunc_) { ...tsdf/weight...; alpha[label] += 1; }
//
// This is that write. The caller (ScovoxMapSplit::integrateHitFused) owns the
// |sdf| ≤ band gate and the endpoint exclusion; by the time we are here the
// voxel has already been chosen. Keep this function branch-light: it runs once
// per band voxel per point, which on KITTI is ~5 extra calls per return.
void SemSplitMap::applyBandSemantic(const CoordT&             c,
                                    const std::vector<float>* sem_probs,
                                    float                     quality,
                                    const HitWeights*         prof) {
  // No class signal ⇒ nothing to pool. Unlike the endpoint path we must NOT
  // fall through to `other += class_share` here: a bare geometric return
  // carries no opinion about its neighbours' classes, and dumping prior mass
  // into every band voxel would dilute exactly the evidence this is meant to
  // concentrate.
  if (!sem_probs || sem_probs->empty()) return;

  // Only DIRICHLET has a meaningful notion of accumulating fractional evidence.
  // NAIVE overwrites slot 0 and MAJORITY_VOTE takes a hard argmax, so banding
  // them would let the LAST band voxel visited win the whole neighbourhood —
  // an order-dependent result, not a pooling one.
  if (params_.semantic_mode != SemanticMode::DIRICHLET) return;

  const float kappa0    = prof ? prof->kappa0              : params_.kappa0;
  const float min_p_occ = prof ? prof->dirichlet_min_p_occ : params_.dirichlet_min_p_occ;

  float class_share;
  if (params_.semantic_band_require_occ) {
    // LiDAR authority, read-only: no Beta voxel here means no beam has ever
    // stopped near this cell, so it is free space in front of the surface (or
    // the unobserved interior behind it) and takes no label.
    // `create_if_missing=false` is load-bearing — allocating would grow the
    // Beta grid along every ray.
    const BetaVoxel* b = beta_acc_.value(c, /*create_if_missing=*/false);
    if (!b) return;
    const float p_occ = b->p_occ();
    if (p_occ < min_p_occ) return;
    class_share = kappa0 * p_occ * quality;
  } else {
    // Faithful SLIM-VDB mirror: no occupancy model, no gate, flat weight. This
    // is `alpha[label] += 1` with kappa0 as the unit. Skipping the Beta read is
    // not just a shortcut — an ungated band voxel may have no Beta entry at all,
    // so there is no p_occ to weight by, and inventing one (say 1.0) would
    // quietly re-introduce a different rule again.
    class_share = kappa0 * quality;
  }
  if (class_share <= 0.f) return;

  DirVoxel* d = getOrAllocateDirOn(dir_acc_, c);
  dirichletUpdate(d, sem_probs, class_share, params_.alpha_0,
                  params_.evict_by_confidence,
                  params_.inc_mode, params_.inc_thresh, c.x, c.y, c.z);
  applyDirSaturation(d);
  touched_dir_.push_back(c);
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
// Lazily build (then reuse) the in-support offset/weight table for radius l.
// The expressions and the dz/dy/dx nesting are copied verbatim from the old
// per-hit loop so every wk bit-matches what that loop computed, and the
// per-hit iteration below visits neighbors in the identical order.
const std::vector<SemSplitMap::SpreadOffset>& SemSplitMap::spreadTable(float l) {
  for (const auto& t : spread_tables_)
    if (t.l == l) return t.offsets;

  const float res       = static_cast<float>(params_.resolution);
  const int   R         = std::max(1, static_cast<int>(std::floor(l / res)));
  const float inv_l     = 1.0f / l;
  constexpr float kTwoPi = 6.283185307179586f;

  SpreadTable tab;
  tab.l = l;
  for (int dz = -R; dz <= R; ++dz)
    for (int dy = -R; dy <= R; ++dy)
      for (int dx = -R; dx <= R; ++dx) {
        const float d = res * std::sqrt(static_cast<float>(dx*dx + dy*dy + dz*dz));
        if (d >= l) continue;                                  // compact support
        const float t  = d * inv_l;                            // in [0,1)
        const float wk = (1.0f/3.0f) * (2.0f + std::cos(kTwoPi * t)) * (1.0f - t)
                       + (1.0f/kTwoPi) * std::sin(kTwoPi * t);
        tab.offsets.push_back({static_cast<int16_t>(dx), static_cast<int16_t>(dy),
                               static_cast<int16_t>(dz), wk});
      }
  spread_tables_.push_back(std::move(tab));
  return spread_tables_.back().offsets;
}

void SemSplitMap::applyHitUpdateKernel(const CoordT&             c,
                                       const std::vector<float>* sem_probs,
                                       float                     quality,
                                       DirGrid::Accessor&        dacc,
                                       std::vector<CoordT>*      touched_dir,
                                       float                     l,
                                       float                     kappa0,
                                       float                     min_p_occ) {
  if (l <= 0.f) return;                                        // caller-gated
  // Same float-cast guard as before the table split: spreadTable divides by
  // the float-cast resolution, so gate on that exact value.
  if (static_cast<float>(params_.resolution) <= 0.f) return;

  // Audit item 10: zero transcendentals per hit — iterate the precomputed
  // in-support offsets and accumulate.
  for (const auto& o : spreadTable(l)) {
    const CoordT n{c.x + o.dx, c.y + o.dy, c.z + o.dz};
    // LiDAR authority: read persistent occupancy; skip voxels LiDAR never
    // built (nullptr) or that are not confidently occupied. Never allocates.
    const BetaVoxel* b = beta_acc_.value(n, /*create_if_missing=*/false);
    if (!b) continue;
    const float p_occ = b->p_occ();
    if (p_occ < min_p_occ) continue;

    const float class_share = kappa0 * p_occ * quality * o.wk;
    if (class_share <= 0.f) continue;

    DirVoxel* dv = getOrAllocateDirOn(dacc, n);
    dirichletUpdate(dv, sem_probs, class_share, params_.alpha_0,
                        params_.evict_by_confidence,
                        params_.inc_mode, params_.inc_thresh, n.x, n.y, n.z);
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
  // Negative means "share the global cap", which is what this read used to be.
  const float cap = params_.class_evidence_saturation >= 0.f
                      ? params_.class_evidence_saturation
                      : params_.evidence_saturation;
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
void sortUnique(std::vector<Bonxai::CoordT>& in) {
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
}
}  // namespace

// Swap the accumulator into the per-stream member scratch instead of moving
// it out: the accumulator inherits the previous drain's capacity (no realloc
// ramp each frame) and the returned buffer stays alive until the same
// stream's next drain.

const std::vector<SemSplitMap::CoordT>& SemSplitMap::drainTouchedBeta() {
  scratch_beta_.clear();
  std::swap(touched_beta_, scratch_beta_);
  sortUnique(scratch_beta_);
  return scratch_beta_;
}

const std::vector<SemSplitMap::CoordT>& SemSplitMap::drainTouchedDir() {
  scratch_dir_.clear();
  std::swap(touched_dir_, scratch_dir_);
  sortUnique(scratch_dir_);
  return scratch_dir_;
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
