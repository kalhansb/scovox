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
/// Prior choice: the **shipped** split-path occupancy prior is the symmetric
/// Jeffreys prior **Beta(0.5,0.5)** (`kBetaOccPrior`/`kBetaFreePrior` below),
/// so an unobserved voxel has `p_occ = 0.5`. It is preferred over the
/// calibrated `Beta(C·α₀, α₀)` prior (`p_occ = C/(C+1) ≈ 0.933`, which matched
/// the unified `SemDirVoxel` occupancy marginal) because that one is asymmetric
/// and biases an unobserved voxel toward occupied, and over `Beta(1,1)` because
/// Jeffreys is reparameterisation-invariant while the `p_occ > 0.5` admission
/// gate is identical under both (see the cancellation argument at the constants
/// below). The factory is prior-agnostic, so the calibrated prior
/// `defaultBetaVoxel(C·α₀, α₀)` remains available as an ablation.

#include <cstddef>
#include <cmath>
#include <limits>
#include <cstdint>
#include <type_traits>

/// Storage width of one Beta parameter. 0 = `float` (8 B voxel).
/// 1 = fixed-point `uint16` (4 B voxel), which halves the occupancy grid.
/// **This is the shipped default as of 2026-09-04.**
///
/// WHY IT IS THE DEFAULT RATHER THAN AN OPT-IN. The occupancy grid is where
/// this map's memory is. A `BetaVoxel` is allocated for every voxel a ray
/// touches; a `DirVoxel` only for a voxel that took a semantic deposit, which
/// is a small minority of them. A byte off `BetaVoxel` is therefore worth many
/// times the same byte off `DirVoxel`, and layout work that starts at the
/// semantic cell is optimising the smaller grid.
///
/// It is a free saving, not a trade. Every weight this repo ships is a whole
/// eighth, so uint16-of-eighths accumulates the *same* number float does — not
/// a near number (see `kBetaLatticeStep` below, and `sanitise()`, which snaps
/// the weights so a future config cannot silently break it). The A/B against
/// the float build is a byte-compare of the dumps, not a metric comparison:
/// anything short of byte identity means a weight left the lattice, or
/// `applyBetaSaturation` fired and halved a voxel under uint16 that float kept.
///
/// Independent of the semantic deposit model — occupancy takes no class
/// evidence — so the result carries to any `hit_share` setting.
///
/// `batch_hits` is a hard prerequisite: un-batched, `a_occ` counts depth
/// *pixels*, and the largest value measured on this suite is 756,508 — 23x
/// past the widest `uint16` range any usable scale can reach. Batched, a voxel
/// takes at most one deposit per scan, so the same map's ceiling is
/// `prior + w_occ x frames` ~ 1.5 x 1300 ~ 1951.
#ifndef SCOVOX_BETA_U16
#define SCOVOX_BETA_U16 1
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
///     started from, so a stream of such rays accumulates nothing. Since the
///     per-observation increment is now exactly `w_occ` / `w_free` with no
///     confidence factor, and both are >= 1.0 in every shipped config, no
///     increment can land in that dead band.
///   - **Stores clamp instead of wrapping.** `SemSplitMap::applyBetaSaturation`
///     rescales both parameters — preserving `p_occ` — before a counter can
///     reach the ceiling, so the clamp here is the backstop, not the mechanism.
///
/// One prior is out of reach: the calibrated `Beta(C·α₀, α₀)` ablation is
/// α₀-scale (0.14 / 0.01), and holding those alongside the thousands of units
/// a full run accumulates needs a dynamic range near 2e5 — past `uint16`'s
/// 65,535 at any scale. The shipped symmetric `Beta(0.5,0.5)` prior sits exactly on
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

// ---------------------------------------------------------------------------
// The count identity, and the lattice it needs
// ---------------------------------------------------------------------------
//
// Every admitted observation contributes a FIXED increment — `a_occ += w_occ`
// at a hit, `a_free += w_free` along the carve — with no per-observation
// confidence factor. So the accumulated parameters are determined entirely by
// two integer counts:
//
//     a_occ  = kBetaOccPrior  + w_occ  * n_hit
//     a_free = kBetaFreePrior + w_free * n_miss
//
// That identity is what lets fixed-point storage be EXACT rather than merely
// close: under `SCOVOX_BETA_U16` a parameter is an integer number of
// `1/SCOVOX_BETA_U16_SCALE` units, so if the prior and both weights are
// themselves whole multiples of that unit, every reachable value lands on the
// lattice and no accumulation rounds. At the shipped scale of 8 the candidate
// weights are exact — `w_occ = 1.5 = 12/8`, `w_free = 1.0 = 8/8`, prior
// `1.0 = 8/8` — and the storage is integer counts denominated in eighths.
//
// Float storage needs the same discipline for a different reason: 1.3 has no
// exact binary representation, so `a_occ += 1.3f` drifts off the identity by
// accumulated rounding. Whole eighths are dyadic and exact in both.
//
// The identity breaks if a weight is NOT on the lattice: each increment then
// rounds, the errors accumulate in one direction, and `n_hit` is no longer
// recoverable from `a_occ`. `kBetaLatticeStep` and the two helpers below make
// that condition checkable instead of assumed. Two sanitisers snap the two
// places a weight can enter: `SemSplitMap::sanitise` for `Params`, and
// `sanitise(HitWeights&)` for the per-source fusion profiles, which do NOT go
// through the first. Any third entry point has to snap for itself -- the
// per-ray read sites take the weight raw, deliberately, so the hot path pays
// no rounding.

/// Spacing of the weight lattice. This is deliberately the SAME under both
/// storage modes, and that is the point: if `w_occ` / `w_free` were quantised
/// only under fixed point, turning `SCOVOX_BETA_U16` on would silently move
/// the weights and any A/B across the flag would be comparing two models
/// rather than two storage layouts. Quantising identically in both makes the
/// flag a pure storage choice.
///
/// Float storage is not exempt from needing a lattice, only from needing THIS
/// one: binary floating point cannot represent 1.3 either, so accumulating it
/// drifts just as surely as fixed point rounds it. A whole multiple of 1/8 is
/// a dyadic rational, exact in float and exact on the fixed-point lattice at
/// the shipped scale, so one rule covers both.
/// Derived from `SCOVOX_BETA_U16_SCALE` in BOTH storage modes, not just under
/// fixed point. A float build compiled with a non-default scale must snap to
/// the same lattice the matching u16 build would use, or the two stop being
/// storage variants of one model at exactly the moment someone tunes the
/// scale -- which is the failure this constant exists to prevent.
constexpr float kBetaLatticeStep = 1.0f / static_cast<float>(SCOVOX_BETA_U16_SCALE);
#if SCOVOX_BETA_U16
static_assert(kBetaLatticeStep == BetaCount::kInv,
              "lattice step must equal the fixed-point storage resolution");
#endif

/// Nearest exactly-representable weight to `x`.
inline float beta_lattice_snap(float x) noexcept {
  return std::nearbyint(x / kBetaLatticeStep) * kBetaLatticeStep;
}

/// True when `x` is on the lattice, so repeated `+= x` never rounds.
inline bool beta_lattice_exact(float x) noexcept {
  return beta_lattice_snap(x) == x;
}

/// How many increments of `w` a parameter admits before it stops being exact —
/// the storage ceiling under fixed point, and effectively unbounded under
/// float. Reported rather than enforced: `applyBetaSaturation` halves both
/// parameters well short of the ceiling, which preserves `p_occ` but does end
/// the count identity for that voxel.
/// Returns `infinity` when the parameter has no accumulation ceiling (float
/// storage) and `0` when `w <= 0` admits no increments at all -- the two cases
/// a single `0` sentinel used to conflate.
inline double beta_max_increments([[maybe_unused]] float w,
                                  [[maybe_unused]] float prior) noexcept {
  if (w <= 0.0f) return 0.0;
#if SCOVOX_BETA_U16
  return static_cast<double>(BetaCount::kMax - prior) / static_cast<double>(w);
#else
  (void)prior;
  return std::numeric_limits<double>::infinity();  // float has no ceiling
#endif
}

/// Beta occupancy voxel. `a_occ + a_free` is the total observation
/// count after the prior is consumed; `p_occ = a_occ / (a_occ + a_free)`.
struct BetaVoxel {
  /// Beta posterior parameters (α). Bumped by `a_occ += w_occ` at a
  /// hit (Stream A) and `a_free += w_free` along the carve ray.
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

/// Shipped split-substrate occupancy prior: symmetric **Beta(0.5,0.5)**
/// (Jeffreys) → prior `p_occ = 0.5`. SINGLE SOURCE OF TRUTH for the split
/// occupancy prior: allocation (`SemSplitMap`), the consensus merge's
/// prior-subtraction (`mergeBeta`), the receiver's at-prior detection
/// (`isPriorBeta`), the sender's emit gate, and the SSMI unobserved baseline
/// all reference these constants, so sender and receiver stay consistent — the
/// prior is a compile-time constant, NOT carried on the wire. Decoupled from
/// the semantic `(num_classes, α₀)` because occupancy and semantics are
/// independent priors.
///
/// Jeffreys is the reference prior for a Bernoulli rate: it is the unique
/// symmetric Beta invariant under reparameterisation of that rate, so it does
/// not privilege any occupancy scale. Both eighths-exact (`0.5 = 4/8`), so the
/// integer-lattice storage identity below is unchanged.
///
/// WHY THE SWITCH IS SAFE AT THE ADMISSION GATE, AND WHERE IT IS NOT.
/// Occupancy admission tests `p_occ > 0.5`, and for ANY symmetric `Beta(a,a)`
///     `p_occ > 0.5  ⇺  a + W_occ > a + W_free  ⇺  W_occ > W_free`,
/// so `a` cancels exactly and the admitted set is prior-invariant. That is
/// algebra, not a tolerance. It does NOT extend to a threshold other than 0.5:
/// `carve_skip_occ_threshold` (the wall guard, off by default and absent from
/// the batched live path) compares `p_occ` against an arbitrary value, and a
/// smaller prior reaches an extreme `p_occ` after fewer looks — one hit gives
/// `0.80` here against `0.71` under `Beta(1,1)` at `w_occ = 1.5`. A caller that
/// turns the wall guard on is choosing a regime where the prior is load-bearing
/// and should re-check its threshold.
constexpr float kBetaOccPrior  = 0.5f;
constexpr float kBetaFreePrior = 0.5f;

/// Beta prior factory. **Required at every allocation**: Bonxai's pool
/// allocator zero-initialises new leaf blocks, leaving `a_occ = a_free = 0`.
/// Without this, the first integration would increment from `Beta(0,0)`
/// instead of from the prior, silently mis-weighting the posterior forever
/// (the same first-touch invariant as `defaultSemBetaVoxel` /
/// `defaultSemDirVoxel`).
///
/// The factory is prior-agnostic. The 0.5/0.5 default IS the shipped symmetric
/// Jeffreys occupancy prior (`p_occ = 0.5`), which `SemSplitMap` passes
/// explicitly via `kBetaOccPrior` / `kBetaFreePrior`. Pass `occ_prior = C·α₀`,
/// `free_prior = α₀` to reproduce the old calibrated unified-Dirichlet marginal
/// (`p_occ = C/(C+1)`) as an ablation, or `1.0/1.0` for Bayes–Laplace.
inline BetaVoxel defaultBetaVoxel(float occ_prior = 0.5f,
                                  float free_prior = 0.5f) noexcept {
  BetaVoxel v{};            // zero-init
  v.a_occ  = occ_prior;
  v.a_free = free_prior;
  return v;
}

}  // namespace scovox
