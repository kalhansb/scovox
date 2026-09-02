#pragma once

/// @file beta_voxel.hpp
/// @brief Beta occupancy voxel for the split Beta/Dirichlet refactor.
/// 8 bytes at float storage; 4 under `SCOVOX_BETA_U16` (see below).
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

#include <cstddef>
#include <cstdint>
#include <type_traits>

/// Storage width of one Beta parameter. 0 = `float` (8 B voxel, the shipped
/// default). 1 = fixed-point `uint16` (4 B voxel), which halves the occupancy
/// grid — 72.6 MB on a 19 M-voxel SceneNN map, where 98.2% of voxels carry a
/// Beta cell and no Dirichlet cell.
///
/// `batch_hits` is a hard prerequisite: un-batched, `a_occ` counts depth
/// *pixels*, and the largest value measured on this suite is 756,508 — 23x
/// past the widest `uint16` range any usable scale can reach. Batched, a voxel
/// takes at most one deposit per scan, so the same map's ceiling is
/// `prior + w_occ x frames` ~ 1.5 x 1300 ~ 1951.
#ifndef SCOVOX_BETA_U16
#define SCOVOX_BETA_U16 0
#endif

/// Fixed-point counts per unit of Beta evidence, i.e. the reciprocal of the
/// stored resolution. 8 gives a resolution of 0.125 and a ceiling of 8191.9,
/// which represents every weight this repo ships (1.0, 1.5, 6.0 and the 1/8
/// lattice around them) exactly, and leaves ~4x headroom over the batched
/// worst case above. Raising it trades ceiling for resolution one-for-one:
/// 16 -> 0.0625 / 4095.9, 2 -> 0.5 / 32767.5.
#ifndef SCOVOX_BETA_U16_SCALE
#define SCOVOX_BETA_U16_SCALE 8
#endif

namespace scovox {

#if SCOVOX_BETA_U16

/// One Beta parameter held as `uint16` counts of `1/SCOVOX_BETA_U16_SCALE`.
///
/// Substitutable for `float` at every read site: the implicit `operator float`
/// makes `a_occ + a_free`, `a_occ / s`, `EXPECT_NEAR(v.a_occ, 3.0f, ...)` and
/// every other read resolve to the same float arithmetic as before. Only the
/// three mutating forms the map uses — `=`, `+=`, `*=` — round-trip through
/// storage, and only two behaviours differ from `float`:
///
///   - **Increments below half a count vanish.** `a_occ += x` with
///     `x < 0.5/SCALE` (0.0625 at the default) rounds back to the value it
///     started from, so a stream of such rays accumulates nothing. Every
///     shipped weight is >= 1.0, so this needs `quality < 0.0625` to bite.
///   - **Stores clamp instead of wrapping.** `SemSplitMap::applyBetaSaturation`
///     rescales both parameters — preserving `p_occ` — before a counter can
///     reach the ceiling, so the clamp here is the backstop, not the mechanism.
///
/// One prior is out of reach: the calibrated `Beta(C·α₀, α₀)` ablation is
/// α₀-scale (0.14 / 0.01), and holding those alongside the thousands of units
/// a full run accumulates needs a dynamic range near 2e5 — past `uint16`'s
/// 65,535 at any scale. The shipped symmetric `Beta(1,1)` prior sits exactly on
/// the lattice; run the calibrated ablation under float storage.
///
/// Contractions toward a target also stop moving once the residual falls under
/// the resolution, so a decay that must *reach* its target has to say so —
/// see the snap in `SemSplitMap::decayTransient`.
class BetaCountU16 {
 public:
  static constexpr float kScale = static_cast<float>(SCOVOX_BETA_U16_SCALE);
  static constexpr float kInv   = 1.0f / kScale;
  /// Largest representable value. Reads as a plain float evidence count.
  static constexpr float kMax   = 65535.0f / kScale;

  BetaCountU16() = default;   ///< trivial (no member initialiser) -- Bonxai zero-inits
  constexpr BetaCountU16(float x) noexcept : v_(pack(x)) {}

  constexpr operator float() const noexcept {
    return static_cast<float>(v_) * kInv;
  }

  constexpr BetaCountU16& operator=(float x) noexcept {
    v_ = pack(x);
    return *this;
  }
  constexpr BetaCountU16& operator+=(float x) noexcept {
    return *this = (static_cast<float>(*this) + x);
  }
  constexpr BetaCountU16& operator*=(float x) noexcept {
    return *this = (static_cast<float>(*this) * x);
  }

 private:
  /// Round-to-nearest into the fixed-point lattice. NaN and negatives floor to
  /// 0 (`!(s > 0)` catches both); overflow clamps rather than wrapping.
  static constexpr uint16_t pack(float x) noexcept {
    const float s = x * kScale;
    if (!(s > 0.0f)) return 0u;
    if (s >= 65535.0f) return 65535u;
    return static_cast<uint16_t>(s + 0.5f);
  }

  uint16_t v_;
};

static_assert(sizeof(BetaCountU16) == 2, "BetaCountU16 must be exactly 2 bytes.");
static_assert(std::is_trivial_v<BetaCountU16>,
    "BetaCountU16 must stay trivial -- Bonxai's pool allocator zero-inits blocks.");
static_assert(std::is_standard_layout_v<BetaCountU16>,
    "BetaCountU16 must have standard layout for byte-for-byte wire emit.");

using BetaCount = BetaCountU16;

#else

using BetaCount = float;

#endif  // SCOVOX_BETA_U16

/// Beta occupancy voxel. `a_occ + a_free` is the total observation
/// count after the prior is consumed; `p_occ = a_occ / (a_occ + a_free)`.
struct BetaVoxel {
  /// Beta posterior parameters (α). Bumped by `a_occ += w_occ·quality` at a
  /// hit (Stream A) and `a_free += w_free·quality` along the carve ray.
  BetaCount a_occ;
  BetaCount a_free;

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

static_assert(sizeof(BetaVoxel) == 2 * sizeof(BetaCount),
    "BetaVoxel must be exactly its two counters wide — the free-space memory "
    "win of the Beta/Dirichlet split depends on it carrying nothing else. "
    "8 B at float storage (matching TsdfVoxel), 4 B under SCOVOX_BETA_U16.");
static_assert(std::is_trivial_v<BetaVoxel>,
    "BetaVoxel must be trivial for Bonxai's pool allocator (zero-init).");
static_assert(std::is_standard_layout_v<BetaVoxel>,
    "BetaVoxel must have standard layout for byte-for-byte wire emit.");
static_assert(offsetof(BetaVoxel, a_free) == offsetof(BetaVoxel, a_occ) + sizeof(BetaCount),
    "BetaVoxel layout: a_free must immediately follow a_occ.");

/// Shipped split-substrate occupancy prior: symmetric **Beta(1,1)** (uniform /
/// Bayes–Laplace) → prior `p_occ = 0.5`. SINGLE SOURCE OF TRUTH for the split
/// occupancy prior: allocation (`SemSplitMap`), the consensus merge's
/// prior-subtraction (`mergeBeta`), the receiver's at-prior detection
/// (`isPriorBeta`), the sender's emit gate, and the SSMI unobserved baseline
/// all reference these constants, so sender and receiver stay consistent — the
/// prior is a compile-time constant, NOT carried on the wire. Decoupled from
/// the semantic `(num_classes, α₀)` because occupancy and semantics are
/// independent priors. See docs/occupancy_prior.md (incl. the Jeffreys
/// `Beta(0.5,0.5)` runner-up and the conditions to switch).
constexpr float kBetaOccPrior  = 1.0f;
constexpr float kBetaFreePrior = 1.0f;

/// Beta prior factory. **Required at every allocation**: Bonxai's pool
/// allocator zero-initialises new leaf blocks, leaving `a_occ = a_free = 0`.
/// Without this, the first integration would increment from `Beta(0,0)`
/// instead of from the prior, silently mis-weighting the posterior forever
/// (the same first-touch invariant as `defaultSemBetaVoxel` /
/// `defaultSemDirVoxel`).
///
/// The factory is prior-agnostic. The 1.0/1.0 default IS the shipped symmetric
/// Beta(1,1) occupancy prior (`p_occ = 0.5`), which `SemSplitMap` passes
/// explicitly via `kBetaOccPrior` / `kBetaFreePrior`. Pass `occ_prior = C·α₀`,
/// `free_prior = α₀` to reproduce the old calibrated unified-Dirichlet marginal
/// (`p_occ = C/(C+1)`) as an ablation. See docs/occupancy_prior.md.
inline BetaVoxel defaultBetaVoxel(float occ_prior = 1.0f,
                                  float free_prior = 1.0f) noexcept {
  BetaVoxel v{};            // zero-init
  v.a_occ  = occ_prior;
  v.a_free = free_prior;
  return v;
}

}  // namespace scovox
