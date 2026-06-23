#pragma once

/// @file semdir_voxel.hpp
/// @brief Unified Dirichlet voxel — collapses Beta(occupancy) + sparse
/// Dirichlet(semantics) into a single Dirichlet over `{class_1, …, class_C,
/// FREE}`, truncated to top-K_TOP class slots + an OTHER bucket that absorbs
/// evicted mass.
///
/// Design: see `docs/design/unified_dirichlet_design_2026_05_13.md`.
///
/// The truncation is a Dirichlet marginalisation: the true generative model
/// is `Dir(α_1, …, α_C, α_FREE)`, and `Dir(α_{c_1}, …, α_{c_K}, α_OTHER,
/// α_FREE)` with `α_OTHER = Σ_{c ∉ top-K} α_c` is the proper marginal. Every
/// quantity of interest (`p_occ`, argmax-of-top-K, occupancy variance) is
/// preserved exactly; only the per-class breakdown *outside* top-K is lost.
///
/// Layout (K_TOP = 2):
///   offset 0:   alpha_free  (float, 4 B)  — α[FREE]
///   offset 4:   alpha_other (float, 4 B)  — α[OTHER], lumped evicted mass
///   offset 8:   cnt[0]      (float, 4 B)  — α for top-K slot 0
///   offset 12:  cnt[1]      (float, 4 B)  — α for top-K slot 1
///   offset 16:  cls[0]      (uint16, 2 B) — class id at slot 0 (0xFFFF=empty)
///   offset 18:  cls[1]      (uint16, 2 B) — class id at slot 1
///   total: 20 B at K_TOP=2 — smaller than SemBetaVoxel (24 B) because
///   `OTHER` replaces the separate `a_unk` slot AND the Beta `(a_occ,
///   a_free)` pair collapses into a single Dirichlet vector with `FREE` as
///   one dimension.

#include <cstdint>
#include <cstddef>
#include <type_traits>

#include "scovox/voxel.hpp"  // K_TOP (still canonical here during transition)

namespace scovox {

/// Per-voxel unified Dirichlet state. `cls[i] = 0xFFFF` marks an empty slot;
/// `cnt[i]` then holds the per-slot Dirichlet prior `α_0` (set by
/// `defaultSemDirVoxel()` — never zero, to keep variance closed forms valid).
struct SemDirVoxel {
  /// Dirichlet pseudo-count for the FREE category. Bumped by carve / no-return
  /// rays via `SemDirMap::integrateMiss` or the carve loop of `integrateHit`.
  float    alpha_free;

  /// Lumped pseudo-counts for the `C − K_TOP − 1` classes outside the top-K
  /// slots. Conserved by `sparse_add_other` for mass-conservation tests (the
  /// upgraded E5.1 invariant becomes a strict equality under unified
  /// Dirichlet, not the `≥ 0` slack of the SemBeta era).
  float    alpha_other;

  /// Top-K class slots, by accumulated `cnt[i]`. Maintained by `update_hit`.
  /// `cls[i] == 0xFFFF` is the "empty slot" sentinel.
  float    cnt[K_TOP];
  uint16_t cls[K_TOP];

  /// Total occupied evidence: top-K class counts + OTHER bucket.
  /// (Excludes FREE — see `p_occ` for the marginal.)
  inline float s_occ() const noexcept {
    float s = alpha_other;
    for (int i = 0; i < K_TOP; ++i) s += cnt[i];
    return s;
  }

  /// Total evidence: s_occ + FREE.
  inline float s_total() const noexcept { return s_occ() + alpha_free; }

  /// Posterior occupancy: marginal of the Dirichlet onto `{occupied, free}`.
  /// `Dir(s_occ, alpha_free)` is itself a Beta when the two-way marginal is
  /// taken, so `p_occ = s_occ / s_total` matches the closed-form Beta mean.
  /// Returns 0.5 when no evidence has been accumulated (only reachable if
  /// `defaultSemDirVoxel()` was bypassed, which `Bonxai::VoxelGrid::
  /// Accessor::value(c, defaultSemDirVoxel())` prevents).
  inline float p_occ() const noexcept {
    const float s = s_total();
    return (s > 0.f) ? (s_occ() / s) : 0.5f;
  }
};

// Layout invariants. K_TOP=2 (production / paper / wire format default):
//   sizeof == 8 + 4·K_TOP + 2·K_TOP = 8 + 12 = 20.
// General K_TOP: 8 (alpha_free + alpha_other) + 4·K_TOP (cnt) + 2·K_TOP
// (cls), rounded up to 4-byte alignment for the trailing uint16_t pair.
constexpr std::size_t kSemDirExpectedSize =
    ((8u + 6u * static_cast<std::size_t>(K_TOP) + 3u) / 4u) * 4u;
static_assert(sizeof(SemDirVoxel) == kSemDirExpectedSize,
    "SemDirVoxel size mismatch — layout is 8 B fixed + 6 B per K_TOP slot "
    "rounded up to 4-byte alignment.");
static_assert(K_TOP != 2 || sizeof(SemDirVoxel) == 20,
    "Production K_TOP=2 invariant: SemDirVoxel must be exactly 20 B (wire "
    "format v3 byte-for-byte emit path depends on this).");
static_assert(std::is_trivial_v<SemDirVoxel>,
    "SemDirVoxel must be trivial for Bonxai's pool allocator (zero-init).");
static_assert(std::is_standard_layout_v<SemDirVoxel>,
    "SemDirVoxel must have standard layout for the v3 wire format's "
    "byte-for-byte reinterpret_cast emit path.");
static_assert(offsetof(SemDirVoxel, alpha_other)
    == offsetof(SemDirVoxel, alpha_free) + sizeof(float),
    "Layout: alpha_other must immediately follow alpha_free.");
static_assert(offsetof(SemDirVoxel, cnt)
    == offsetof(SemDirVoxel, alpha_other) + sizeof(float),
    "Layout: cnt[] must immediately follow alpha_other.");
static_assert(offsetof(SemDirVoxel, cls)
    == offsetof(SemDirVoxel, cnt) + K_TOP * sizeof(float),
    "Layout: cls[] must immediately follow cnt[K_TOP] with no padding.");

// `kDefaultDirichletPrior` (the symmetric Dirichlet prior α₀) is defined in
// voxel.hpp — shared across the SemBeta / SemDir / split substrates so they
// agree on the ship default. Included transitively above.

/// Build a freshly-primed voxel at the symmetric Dirichlet prior.
/// **Required at every allocation** for the same reason as
/// `defaultSemBetaVoxel()`: Bonxai's pool allocator zero-inits new leaf
/// blocks, so without this factory the first integration call would
/// increment from `α = 0` instead of from the prior, silently
/// mis-weighting every posterior forever. See §"Open questions" of the
/// design doc for `α_0` default rationale.
///
/// `num_classes` (`C`) is the dataset's total class count. The OTHER prior
/// is `(C − K_TOP) · α_0` because OTHER summarises that many collapsed
/// Dirichlet dimensions (C class dims, K_TOP of which are represented as
/// fungible placeholders in the slots, leaving C − K_TOP in OTHER). For
/// NYU13 (C=14) at K_TOP=2 the OTHER prior is `12 · α_0`. The cnt[] slots
/// and FREE each carry the per-dim prior `α_0`.
///
/// Total system prior: K_TOP·α_0 (slot placeholders) + (C − K_TOP)·α_0
/// (OTHER) + α_0 (FREE) = (C + 1) · α_0 — matches the true generative
/// `Dir(α_0, …, α_0)` over `C+1` categories.
inline SemDirVoxel defaultSemDirVoxel(uint16_t num_classes = 14,
                                      float    alpha_0    = kDefaultDirichletPrior) noexcept {
  SemDirVoxel v{};                  // zero-init the struct
  v.alpha_free = alpha_0;           // one Dirichlet dim
  // OTHER summarises (C − K_TOP) collapsed dims, each with prior α_0.
  // Clamped to ≥ 0 in case the caller mis-passes num_classes < K_TOP
  // (degenerate but recoverable).
  const int residual_dims = static_cast<int>(num_classes) - K_TOP;
  v.alpha_other = (residual_dims > 0) ? (residual_dims * alpha_0) : 0.f;
  for (int i = 0; i < K_TOP; ++i) {
    v.cnt[i] = alpha_0;             // per-dim prior on each top-K slot
    v.cls[i] = 0xFFFF;              // empty-slot sentinels
  }
  return v;
}

}  // namespace scovox
