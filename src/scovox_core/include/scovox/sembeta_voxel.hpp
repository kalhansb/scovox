#pragma once

/// @file
/// @brief Bayesian occupancy + sparse Dirichlet voxel for the split-grid
/// SCovox refactor.
///
/// Holds the SCovox-specific state — Beta(α_occ, α_free) occupancy posterior
/// and K-top sparse Dirichlet semantic counts — without the TSDF fields.
/// Lives in a `Bonxai::VoxelGrid<SemBetaVoxel>` parallel to the TSDF grid.
///
/// Why split: the TSDF grid is band-only (touched on ~2·trunc/voxel
/// voxels per ray) and has a clean SLIM-VDB-equivalent integration path.
/// The SemBeta grid is full-ray (touched on every carved voxel) and holds
/// SCovox's contribution. Different access patterns and different lifecycle
/// rules; cleaner to keep them in separate grids.
///
/// Cross-grid invariant (per Q5 of the 2026-05-08 grill-me design review):
/// SemBeta is the **primary** grid for non-mesh queries. Consumers walking
/// SemBeta and querying TSDF must handle TSDF-absence with sentinels
/// (NaN distance, zero weight). Only `labelMesh` walks TSDF and queries
/// SemBeta; absence in that direction returns class id `0xFFFF`.
/// Moved comments: doc/scovox_core_code_notes.md

#include <cstdint>
#include <type_traits>

#include "scovox/voxel.hpp"  // for K_TOP and sparse_add() — both still
                             // canonical in voxel.hpp during the transition;
                             // they migrate here after the legacy fused
                             // `Voxel` struct is retired.

namespace scovox {

/// 24-byte voxel at K_TOP=2: a_occ, a_free, a_unk, then sem_cnt[K_TOP], then
/// sem_cls[K_TOP], no internal padding. The static_asserts below pin this
/// layout. (notes: sembeta-voxel-layout)
struct SemBetaVoxel {
  /// Beta posterior parameters (α). a_occ + a_free = total observation count
  /// after the Beta(1,1) prior is consumed. p_occ = a_occ / (a_occ + a_free).
  float    a_occ;
  float    a_free;

  /// Unknown / residual Dirichlet mass. Holds (i) class evidence that didn't
  /// fit in any of the K_TOP slots and (ii) evicted slot mass during
  /// sparse-add eviction (Space-Saving heavy-hitter accounting). Conserved
  /// by `sparse_add` for mass-conservation tests (E5.1).
  float    a_unk;

  /// Top-K sparse Dirichlet slots, by evidence weight. `sem_cls[i] = 0xFFFF`
  /// is the "empty slot" sentinel (no class tracked at slot i yet).
  /// `sem_cnt[i]` accumulates Dirichlet pseudo-counts for class `sem_cls[i]`.
  /// Maintained by `sparse_add()` (defined in voxel.hpp during the transition).
  float    sem_cnt[K_TOP];
  uint16_t sem_cls[K_TOP];

  /// Posterior occupancy probability under Beta(a_occ, a_free).
  /// Returns 0.5 when no evidence has been accumulated (defaults at the
  /// Beta(0,0) state — only reachable if `defaultSemBetaVoxel()` was bypassed,
  /// which `Bonxai::VoxelGrid::Accessor::value(c, /*default=*/...)` prevents).
  inline float p_occ() const noexcept {
    const float s = a_occ + a_free;
    return (s > 0.f) ? (a_occ / s) : 0.5f;
  }

  /// Total Dirichlet concentration: sum of all semantic counts + unknown.
  /// Used by entropy computations in `uncertainty.hpp`.
  inline float a0() const noexcept {
    float s = a_unk;
    for (int i = 0; i < K_TOP; ++i) s += sem_cnt[i];
    return s;
  }
};

// Size is 12 B plus 6 B per K_TOP slot, rounded up to 4-byte alignment; at
// K_TOP=2 (the production default) it must be exactly 24 B.
// (notes: sembeta-size-invariant)
constexpr std::size_t kSemBetaExpectedSize =
    ((12u + 6u * static_cast<std::size_t>(K_TOP) + 3u) / 4u) * 4u;
static_assert(sizeof(SemBetaVoxel) == kSemBetaExpectedSize,
    "SemBetaVoxel size mismatch — layout is 12 B fixed + 6 B per K_TOP slot "
    "rounded up to 4-byte alignment.");
static_assert(K_TOP != 2 || sizeof(SemBetaVoxel) == 24,
    "Production K_TOP=2 invariant: SemBetaVoxel must be exactly 24 B "
    "(the paper's 24 B/voxel memory claim depends on this).");
static_assert(std::is_trivial_v<SemBetaVoxel>,
    "SemBetaVoxel must be trivial for Bonxai's pool allocator (zero-init).");
static_assert(std::is_standard_layout_v<SemBetaVoxel>,
    "SemBetaVoxel must have standard layout for the offsetof() invariants "
    "below to be well-defined.");
static_assert(offsetof(SemBetaVoxel, a_free)
    == offsetof(SemBetaVoxel, a_occ) + sizeof(float),
    "Beta layout: a_free must immediately follow a_occ.");
static_assert(offsetof(SemBetaVoxel, a_unk)
    == offsetof(SemBetaVoxel, a_free) + sizeof(float),
    "Beta layout: a_unk must immediately follow a_free.");
static_assert(offsetof(SemBetaVoxel, sem_cnt)
    == offsetof(SemBetaVoxel, a_unk) + sizeof(float),
    "Layout: sem_cnt must immediately follow a_unk.");
static_assert(offsetof(SemBetaVoxel, sem_cls)
    == offsetof(SemBetaVoxel, sem_cnt) + K_TOP * sizeof(float),
    "Layout: sem_cls must immediately follow sem_cnt[K_TOP] with no padding.");

/// Beta(1,1) prior and empty-slot sentinels. Required at every allocation:
/// Bonxai zero-inits new leaf blocks, so without it the posterior starts from
/// 0, not the prior. Write it in before any other field.
/// (notes: sembeta-default-factory)
inline SemBetaVoxel defaultSemBetaVoxel() noexcept {
  SemBetaVoxel v{};                     // zero-init the struct
  v.a_occ      = 1.0f;                  // Beta(1,1) prior (uninformative)
  v.a_free     = 1.0f;
  // a_unk = 0, sem_cnt[*] = 0 — already zero-init.
  // Loop over all K_TOP slots; hard-coded slot indices write past the array
  // when K_TOP != 2. (notes: sembeta-default-slot-loop)
  for (int i = 0; i < K_TOP; ++i) {
    v.sem_cls[i] = 0xFFFF;             // empty-slot sentinels
  }
  return v;
}

}  // namespace scovox
