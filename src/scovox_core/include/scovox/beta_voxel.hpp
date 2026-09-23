#pragma once

/// @file beta_voxel.hpp
/// @brief 8-byte Beta occupancy voxel for the split Beta/Dirichlet refactor.
///
/// Holds *only* the occupancy posterior — `Beta(a_occ, a_free)` — with no
/// semantic state. The semantic Dirichlet lives in a parallel
/// `Bonxai::VoxelGrid<DirVoxel>` (see `dir_voxel.hpp`), and the TSDF geometry
/// in a third `Bonxai::VoxelGrid<TsdfVoxel>`. The three grids are coordinated
/// by `SemSplitMap` + `ScovoxMapSplit`.
///
/// Why a dedicated occupancy grid (de-unifying `SemDirVoxel`):
///   - Occupancy is **full-ray / dense** — `a_free` is bumped on every carved
///     voxel along a ray. Semantics is **hit-only / sparse** — class counts
///     are committed only at (near) the surface, gated by `p_occ`.
///   - In the unified `SemDirVoxel` (20 B) and the combined `SemBetaVoxel`
///     (24 B), every free-space voxel still carries the ~12-16 B of semantic
///     slots it never uses. Splitting keeps free-space leaf-blocks at 8 B and
///     allocates the 16 B `DirVoxel` only where a class is actually observed.
///   - The two attributes are co-touched only at the single hit voxel per ray
///     (where the Dirichlet update is gated on this grid's `p_occ`), so the
///     extra accessor lookup is paid once per ray, not per carved voxel.
///
/// Prior choice: the **shipped** split-path occupancy prior is symmetric
/// **Beta(1,1)** (`kBetaOccPrior`/`kBetaFreePrior` below), so an unobserved
/// voxel has `p_occ = 0.5`. This was chosen over the calibrated
/// `Beta(C·α₀, α₀)` prior (`p_occ = C/(C+1) ≈ 0.933`, which matched the unified
/// `SemDirVoxel` occupancy marginal) and over the Jeffreys prior
/// `Beta(0.5,0.5)` — for single-ray-noise robustness against the carve
/// wall-guard; see docs/occupancy_prior.md for the full derivation and the
/// Jeffreys runner-up. The factory is prior-agnostic, so the calibrated prior
/// `defaultBetaVoxel(C·α₀, α₀)` remains available as an ablation.
/// Moved comments: doc/scovox_core_code_notes.md

#include <cstddef>
#include <type_traits>

namespace scovox {

/// 8-byte Beta occupancy voxel. `a_occ + a_free` is the total observation
/// count after the prior is consumed; `p_occ = a_occ / (a_occ + a_free)`.
struct BetaVoxel {
  /// Beta posterior parameters (α). Bumped by `a_occ += w_occ·quality` at a
  /// hit (Stream A) and `a_free += w_free·quality` along the carve ray.
  float a_occ;
  float a_free;

  /// Posterior occupancy probability under `Beta(a_occ, a_free)`.
  /// Returns 0.5 when no evidence has been accumulated (only reachable if the
  /// default-voxel factory was bypassed, which the allocation path prevents).
  inline float p_occ() const noexcept {
    const float s = a_occ + a_free;
    return (s > 0.f) ? (a_occ / s) : 0.5f;
  }

  /// Total occupancy evidence (concentration) `a_occ + a_free`. The analogue
  /// of `SemDirVoxel::s_total()` restricted to the occupancy marginal; used by
  /// the evidence-saturation cap.
  inline float s_total() const noexcept { return a_occ + a_free; }
};

static_assert(sizeof(BetaVoxel) == 8,
    "BetaVoxel must be exactly 8 bytes — the free-space memory win of the "
    "Beta/Dirichlet split depends on this (matches TsdfVoxel's 8 B).");
static_assert(std::is_trivial_v<BetaVoxel>,
    "BetaVoxel must be trivial for Bonxai's pool allocator (zero-init).");
static_assert(std::is_standard_layout_v<BetaVoxel>,
    "BetaVoxel must have standard layout for byte-for-byte wire emit.");
static_assert(offsetof(BetaVoxel, a_free) == offsetof(BetaVoxel, a_occ) + sizeof(float),
    "BetaVoxel layout: a_free must immediately follow a_occ.");

/// Shipped occupancy prior Beta(1,1), p_occ = 0.5. Single source of truth for
/// SemSplitMap, mergeBeta, isPriorBeta, the sender's emit gate and SSMI
/// baseline; it is not sent on the wire, so both ends must share it.
/// (notes: beta-prior-single-source)
constexpr float kBetaOccPrior  = 1.0f;
constexpr float kBetaFreePrior = 1.0f;

/// Beta prior factory, required at every allocation: Bonxai zero-initialises
/// new leaf blocks, and starting from Beta(0,0) mis-weights the posterior
/// forever. The 1.0/1.0 defaults are the shipped Beta(1,1) prior.
/// (notes: beta-prior-factory-first-touch)
inline BetaVoxel defaultBetaVoxel(float occ_prior = 1.0f,
                                  float free_prior = 1.0f) noexcept {
  BetaVoxel v{};            // zero-init
  v.a_occ  = occ_prior;
  v.a_free = free_prior;
  return v;
}

}  // namespace scovox
