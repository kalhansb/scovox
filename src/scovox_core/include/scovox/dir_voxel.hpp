#pragma once

/// @file dir_voxel.hpp
/// @brief 16-byte sparse-Dirichlet semantics voxel for the split Beta/Dirichlet
/// refactor.
///
/// Holds *only* the semantic state — a Dirichlet over `{top-K classes, OTHER}`
/// — with no occupancy. Occupancy lives in the parallel
/// `Bonxai::VoxelGrid<BetaVoxel>` (see `beta_voxel.hpp`).
///
/// Relationship to the unified `SemDirVoxel`: this is `SemDirVoxel` with the
/// `FREE` dimension removed (`FREE` is now the Beta grid's `a_free`). What
/// remains is the *occupied-class* Dirichlet: the top-K class slots plus the
/// `OTHER` bucket that lumps the `C − K_TOP` collapsed class dimensions and
/// any evicted slot mass. De-unifying this way means class evidence no longer
/// feeds back into the occupancy marginal — occupancy and semantics are
/// conditionally independent given the observation (the SemBeta two-stream
/// model), which is the intended meaning of "split Beta and Dirichlet".
///
/// Layout (K_TOP = 2):
///   offset 0:   other  (float, 4 B)  — lumped OTHER / evicted mass
///   offset 4:   cnt[0] (float, 4 B)  — α for top-K slot 0
///   offset 8:   cnt[1] (float, 4 B)  — α for top-K slot 1
///   offset 12:  cls[0] (uint16, 2 B) — class id at slot 0 (0xFFFF = empty)
///   offset 14:  cls[1] (uint16, 2 B) — class id at slot 1
///   total: 16 B at K_TOP=2.
///
/// Mass conservation: `sparse_add_class` preserves the strict invariant
///   Δ(other + Σ cnt) == Σ Δ inputs
/// — every increment lands somewhere (matched slot / empty slot / evicted-to-
/// OTHER / dropped-to-OTHER), never lost. This is the same eviction-to-OTHER
/// discipline as `sparse_add_unified` in semdir_map.cpp, ported to the
/// occupancy-free Dirichlet; it is *not* the legacy `voxel.hpp::sparse_add`
/// with its `≥ 0` slack.
///
/// One bounded exception, from evidence saturation
/// (`SemSplitMap::applyDirSaturation`): the rescale multiplies an EMPTY slot's
/// α₀ placeholder down to k·α₀ (only FILLED slots are floored back to α₀ —
/// flooring empty ones would re-inflate `s_class` past the cap). The next
/// empty-slot fill overwrites that placeholder with `α₀ + inc`, so the fill's
/// Δ exceeds `inc` by α₀·(1−k) ≤ α₀. This merely restores the eroded prior,
/// is bounded by α₀ per fill, and needs no downstream guard.
/// Moved comments: doc/scovox_core_code_notes.md

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "scovox/voxel.hpp"  // K_TOP + g_sparse_*_count counters

/// Build-time per-slot confidence track for the eviction comparator; it changes
/// sizeof(DirVoxel) (16 B to 20 B at K_TOP=2). qmax is never serialized, so a
/// =1 sender stays wire-compatible with a =0 receiver.
/// (notes: dirvoxel-track-qmax)
#ifndef SCOVOX_TRACK_QMAX
#define SCOVOX_TRACK_QMAX 0
#endif

namespace scovox {

/// Per-voxel occupied-class Dirichlet state. `cls[i] == 0xFFFF` marks an empty
/// slot; `cnt[i]` then holds the per-slot prior `α₀` (set by
/// `defaultDirVoxel()` — never zero, to keep closed-form variance valid).
struct DirVoxel {
  /// Lumped pseudo-counts for the `C − K_TOP` classes outside the top-K slots
  /// plus any evicted slot evidence. Conserved by `sparse_add_class`.
  float    other;

  /// Top-K class slots, by accumulated `cnt[i]`. `cls[i] == 0xFFFF` is the
  /// empty-slot sentinel; an empty slot's `cnt[i]` holds the prior `α₀`.
  float    cnt[K_TOP];
  uint16_t cls[K_TOP];

#if SCOVOX_TRACK_QMAX
  /// Running maximum of the *deposit probability* each slot has ever seen, as
  /// u16 fixed point (`round(q · 65535)`). Read only by the confidence
  /// eviction comparator in `sparse_add_class`; zero-init means "no deposit
  /// yet", which is only ever the state of an EMPTY slot.
  uint16_t qmax[K_TOP];
#endif

  /// Total class (occupied-semantic) evidence: `other + Σ cnt`. The analogue
  /// of `SemDirVoxel::s_occ()` (which additionally folds in occupancy mass);
  /// here it is purely semantic.
  inline float s_class() const noexcept {
    float s = other;
    for (int i = 0; i < K_TOP; ++i) s += cnt[i];
    return s;
  }
};

// Layout invariant: sizeof(DirVoxel) = 4 + 6·K_TOP bytes rounded up to 4-byte
// alignment (16 B at K_TOP=2). (notes: dirvoxel-layout-size)
/// 6 B per slot (4 cnt + 2 cls), or 8 B when the qmax confidence track is
/// compiled in.
constexpr std::size_t kDirSlotBytes = SCOVOX_TRACK_QMAX ? 8u : 6u;
constexpr std::size_t kDirExpectedSize =
    ((4u + kDirSlotBytes * static_cast<std::size_t>(K_TOP) + 3u) / 4u) * 4u;
static_assert(sizeof(DirVoxel) == kDirExpectedSize,
    "DirVoxel size mismatch — layout is 4 B fixed + 6 B per K_TOP slot "
    "(8 B with SCOVOX_TRACK_QMAX) rounded up to 4-byte alignment.");
static_assert(SCOVOX_TRACK_QMAX || K_TOP != 2 || sizeof(DirVoxel) == 16,
    "Production K_TOP=2 invariant: DirVoxel must be exactly 16 B "
    "(SemDirVoxel 20 B minus the 4 B FREE dimension moved to BetaVoxel).");
static_assert(!SCOVOX_TRACK_QMAX || K_TOP != 2 || sizeof(DirVoxel) == 20,
    "K_TOP=2 with SCOVOX_TRACK_QMAX: 16 B + 2 B per slot confidence = 20 B.");
static_assert(std::is_trivial_v<DirVoxel>,
    "DirVoxel must be trivial for Bonxai's pool allocator (zero-init).");
static_assert(std::is_standard_layout_v<DirVoxel>,
    "DirVoxel must have standard layout for byte-for-byte wire emit.");
static_assert(offsetof(DirVoxel, cnt) == offsetof(DirVoxel, other) + sizeof(float),
    "DirVoxel layout: cnt[] must immediately follow other.");
static_assert(offsetof(DirVoxel, cls) == offsetof(DirVoxel, cnt) + K_TOP * sizeof(float),
    "DirVoxel layout: cls[] must immediately follow cnt[K_TOP] with no padding.");

/// Default symmetric prior: each empty slot holds cnt[i] = α₀ (cls 0xFFFF) and
/// other = (C − K_TOP)·α₀ for the collapsed out-of-K dimensions.
/// (notes: dirvoxel-default-prior)
inline DirVoxel defaultDirVoxel(uint16_t num_classes = 14,
                                float    alpha_0     = kDefaultDirichletPrior) noexcept {
  DirVoxel v{};                     // zero-init
  const int residual_dims = static_cast<int>(num_classes) - K_TOP;
  v.other = (residual_dims > 0) ? (residual_dims * alpha_0) : 0.f;
  for (int i = 0; i < K_TOP; ++i) {
    v.cnt[i] = alpha_0;             // per-dim prior on each top-K slot
    v.cls[i] = 0xFFFF;             // empty-slot sentinels
  }
  return v;
}

/// Space-Saving sparse-add of inc into a top-K slot or OTHER, never lost:
/// Δ(other + Σcnt) == inc. q (the deposit's class probability) only feeds the
/// confidence eviction comparator; ignored unless qmax != nullptr.
/// (notes: dirvoxel-sparse-add-class)
inline void sparse_add_class(float*    cnt,
                             uint16_t* cls,
                             uint16_t  c,
                             float     inc,
                             float*    other,
                             float     alpha_0,
                             float     q    = -1.0f,
                             uint16_t* qmax = nullptr) {
  // Fixed-point form of the incoming confidence, computed once.
  const bool     track = (qmax != nullptr) && (q >= 0.0f);
  const uint16_t q_fx  = !track            ? uint16_t{0}
                       : (q >= 1.0f)       ? uint16_t{65535}
                                           : static_cast<uint16_t>(q * 65535.0f + 0.5f);
  // (0) Sentinel guard: 0xFFFF marks an empty slot, so a real class id 0xFFFF
  // must never enter a slot (it would read as empty and break the mass
  // invariant); route it to OTHER. (notes: dirvoxel-sentinel-class)
  if (c == 0xFFFF) {
    *other += inc;
    g_sparse_drop_count.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  // (1) Match — incoming class already tracked in a slot.
  for (int i = 0; i < K_TOP; ++i) {
    if (cls[i] != 0xFFFF && cls[i] == c) {
      cnt[i] += inc;
      if (track && q_fx > qmax[i]) qmax[i] = q_fx;
      g_sparse_match_count.fetch_add(1, std::memory_order_relaxed);
      return;
    }
  }
  // (2) Empty slot available — fill it. The slot's α₀ prior stays; add on top.
  for (int i = 0; i < K_TOP; ++i) {
    if (cls[i] == 0xFFFF) {
      cls[i] = c;
      cnt[i] = alpha_0 + inc;
      if (track) qmax[i] = q_fx;
      g_sparse_empty_count.fetch_add(1, std::memory_order_relaxed);
      return;
    }
  }
  // (3) All slots filled — evict-or-drop using observed-evidence (cnt − α₀) as
  // the comparison key (posterior-predictive Space-Saving; see voxel.hpp).
  int min_i = 0;
  for (int i = 1; i < K_TOP; ++i) if (cnt[i] < cnt[min_i]) min_i = i;
  // Clamp at 0: a saturation rescale can erode a filled slot below α₀, and
  // negative evicted evidence would subtract mass from OTHER. Normally a no-op,
  // as saturation floors filled slots at α₀. (notes: dirvoxel-evict-clamp)
  const float raw_evicted = cnt[min_i] - alpha_0;
  const float evicted_evidence = raw_evicted > 0.f ? raw_evicted : 0.f;
  // Both comparators conserve Δ(other + Σcnt) == inc and differ only in which
  // deposit wins. Default: inc must beat the slot's evidence (frequent wins);
  // with qmax tracking, higher confidence wins.
  // (notes: dirvoxel-eviction-comparator)
  const bool evict_now = track ? (q_fx > qmax[min_i])
                               : (inc > evicted_evidence);
  if (evict_now) {
    // Evict: incoming class displaces slot min_i. Displaced accumulated
    // evidence flows to OTHER; the α₀ placeholder stays for the new class.
    *other += evicted_evidence;
    cls[min_i] = c;
    cnt[min_i] = alpha_0 + inc;
    if (track) qmax[min_i] = q_fx;
    g_sparse_evict_count.fetch_add(1, std::memory_order_relaxed);
  } else {
    // Drop: incoming evidence smaller than every tracked class. Mass to OTHER.
    *other += inc;
    g_sparse_drop_count.fetch_add(1, std::memory_order_relaxed);
  }
}

/// Argmax of the top-K slots by observed evidence (cnt − α₀). Returns 0xFFFF if
/// no slot is filled or OTHER's observed evidence (other minus its (C −
/// K_TOP)·α₀ prior) exceeds every slot's. (notes: dirvoxel-dominant-class)
inline uint16_t dominantClass(const DirVoxel& v,
                              float    alpha_0     = kDefaultDirichletPrior,
                              uint16_t num_classes = 14) noexcept {
  uint16_t cls = 0xFFFF;
  float best_evidence = 0.f;
  for (int i = 0; i < K_TOP; ++i) {
    if (v.cls[i] == 0xFFFF) continue;
    const float evidence = v.cnt[i] - alpha_0;
    if (evidence > best_evidence) {
      best_evidence = evidence;
      cls           = v.cls[i];
    }
  }
  // OTHER's observed evidence = bucket minus its (C − K_TOP)·α₀ prior, clamped
  // at 0 (defaultDirVoxel clamps the residual when num_classes ≤ K_TOP). Only
  // genuine out-of-K evidence — not the prior — may veto the tracked argmax.
  const int   residual_dims = static_cast<int>(num_classes) - K_TOP;
  const float other_prior   = (residual_dims > 0) ? (residual_dims * alpha_0) : 0.f;
  const float other_evidence = v.other - other_prior;
  if (other_evidence > best_evidence) return 0xFFFF;
  return cls;
}

}  // namespace scovox
