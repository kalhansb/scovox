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

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "scovox/voxel.hpp"  // K_TOP + g_sparse_*_count counters

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

  /// Total class (occupied-semantic) evidence: `other + Σ cnt`. The analogue
  /// of `SemDirVoxel::s_occ()` (which additionally folds in occupancy mass);
  /// here it is purely semantic.
  inline float s_class() const noexcept {
    float s = other;
    for (int i = 0; i < K_TOP; ++i) s += cnt[i];
    return s;
  }
};

// Layout invariants. K_TOP=2 (production / paper default):
//   sizeof == 4 (other) + 4·K_TOP (cnt) + 2·K_TOP (cls) = 4 + 12 = 16.
// General K_TOP: 4 + 6·K_TOP, rounded up to 4-byte alignment for the trailing
// uint16_t pair.
constexpr std::size_t kDirExpectedSize =
    ((4u + 6u * static_cast<std::size_t>(K_TOP) + 3u) / 4u) * 4u;
static_assert(sizeof(DirVoxel) == kDirExpectedSize,
    "DirVoxel size mismatch — layout is 4 B fixed + 6 B per K_TOP slot "
    "rounded up to 4-byte alignment.");
static_assert(K_TOP != 2 || sizeof(DirVoxel) == 16,
    "Production K_TOP=2 invariant: DirVoxel must be exactly 16 B "
    "(SemDirVoxel 20 B minus the 4 B FREE dimension moved to BetaVoxel).");
static_assert(std::is_trivial_v<DirVoxel>,
    "DirVoxel must be trivial for Bonxai's pool allocator (zero-init).");
static_assert(std::is_standard_layout_v<DirVoxel>,
    "DirVoxel must have standard layout for byte-for-byte wire emit.");
static_assert(offsetof(DirVoxel, cnt) == offsetof(DirVoxel, other) + sizeof(float),
    "DirVoxel layout: cnt[] must immediately follow other.");
static_assert(offsetof(DirVoxel, cls) == offsetof(DirVoxel, cnt) + K_TOP * sizeof(float),
    "DirVoxel layout: cls[] must immediately follow cnt[K_TOP] with no padding.");

/// Default symmetric prior. The top-K slots and OTHER carry the same per-dim
/// prior `α₀` that the unified `SemDirVoxel` uses, minus the FREE dimension:
///   - each `cnt[i] = α₀`  (K_TOP slot placeholders)
///   - `other = (C − K_TOP) · α₀`  (the collapsed out-of-K dimensions)
/// Total class prior `= C · α₀`, equal to `SemDirVoxel::s_occ()` at prior and
/// to `BetaVoxel`'s `a_occ` prior (`C·α₀`) — keeping the split consistent with
/// the live path at the prior.
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

/// Heavy-hitter sparse-add into the occupied-class Dirichlet, parametrised by
/// the per-dim prior `α₀`. Routes `inc` into one of the top-K_TOP class slots
/// (Space-Saving / Metwally 2005) or the OTHER bucket — never lost. Preserves
/// the strict mass invariant `Δ(other + Σcnt) == inc`.
///
/// Direct port of `sparse_add_unified` (semdir_map.cpp), with `alpha_other`
/// renamed `other` and no FREE interaction (FREE is in the Beta grid).
inline void sparse_add_class(float*    cnt,
                             uint16_t* cls,
                             uint16_t  c,
                             float     inc,
                             float*    other,
                             float     alpha_0) {
  // (1) Match — incoming class already tracked in a slot.
  for (int i = 0; i < K_TOP; ++i) {
    if (cls[i] != 0xFFFF && cls[i] == c) {
      cnt[i] += inc;
      g_sparse_match_count.fetch_add(1, std::memory_order_relaxed);
      return;
    }
  }
  // (2) Empty slot available — fill it. The slot's α₀ prior stays; add on top.
  for (int i = 0; i < K_TOP; ++i) {
    if (cls[i] == 0xFFFF) {
      cls[i] = c;
      cnt[i] = alpha_0 + inc;
      g_sparse_empty_count.fetch_add(1, std::memory_order_relaxed);
      return;
    }
  }
  // (3) All slots filled — evict-or-drop using observed-evidence (cnt − α₀) as
  // the comparison key (posterior-predictive Space-Saving; see voxel.hpp).
  int min_i = 0;
  for (int i = 1; i < K_TOP; ++i) if (cnt[i] < cnt[min_i]) min_i = i;
  const float evicted_evidence = cnt[min_i] - alpha_0;
  if (inc > evicted_evidence) {
    // Evict: incoming class displaces slot min_i. Displaced accumulated
    // evidence flows to OTHER; the α₀ placeholder stays for the new class.
    *other += evicted_evidence;
    cls[min_i] = c;
    cnt[min_i] = alpha_0 + inc;
    g_sparse_evict_count.fetch_add(1, std::memory_order_relaxed);
  } else {
    // Drop: incoming evidence smaller than every tracked class. Mass to OTHER.
    *other += inc;
    g_sparse_drop_count.fetch_add(1, std::memory_order_relaxed);
  }
}

/// Argmax of the top-K class slots by observed evidence (`cnt − α₀`). Returns
/// 0xFFFF if no slot is filled, or if `other` exceeds every slot's evidence
/// (the bulk of the class mass is on out-of-K classes, so committing to a
/// tracked class would be misleading). Mirrors the `SemDirVoxel` overload in
/// mesh_labelling.hpp, restricted to the occupied-class Dirichlet.
inline uint16_t dominantClass(const DirVoxel& v,
                              float alpha_0 = kDefaultDirichletPrior) noexcept {
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
  if (v.other > best_evidence) return 0xFFFF;
  return cls;
}

}  // namespace scovox
