#pragma once

/// @file semdir_map.hpp
/// @brief Unified-Dirichlet SCovox-contribution map (Step 7.5 of the
/// split-grid refactor). Replaces `SemBetaMap` with a single Dirichlet over
/// `{top-K classes, OTHER, FREE}`.
///
/// Lives in a `Bonxai::VoxelGrid<SemDirVoxel>` parallel to the TSDF grid.
/// API surface mirrors `SemBetaMap`:
///   - `integrateHit(origin, endpoint, sem_probs, quality)`
///   - `integrateMiss(origin, endpoint, quality)`
///   - per-voxel `applyCarveUpdate` / `applyHitUpdate` for the fused walker
///   - `drainTouched()` for wire-format publishing
///   - `getVoxel(pos)` / `forEachVoxel(fn)` queries
///
/// What changes vs. SemBetaMap:
///   - One Dirichlet update per hit, not Beta-then-Dirichlet. The occupied-
///     evidence mass `w_occ · quality` is distributed across top-K class
///     slots (via Space-Saving) plus OTHER (residual softmax mass).
///   - Carve / miss bumps `alpha_free`, not Beta `a_free`. Same scalar.
///   - Wall-blocking gate is on `p_occ()` (Dirichlet marginal), same form.
///   - Evidence saturation caps `s_total()`, not per-dim alphas separately.
///   - `kappa0` is **deprecated** (the dual-scale Beta-vs-Dirichlet knob
///     no longer exists). Kept in `Params` for launch-file backward compat;
///     interpreted as a unit-less scaling on the per-class share.

#include <Eigen/Core>
#include <bonxai/bonxai.hpp>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "scovox/semantics.hpp"  // SemanticMode enum (DIRICHLET / NAIVE / MAJORITY_VOTE)
#include "scovox/semdir_voxel.hpp"

namespace scovox {

class SemDirMap {
 public:
  using Grid   = Bonxai::VoxelGrid<SemDirVoxel>;
  using CoordT = Bonxai::CoordT;

  /// All update knobs. Defaults match SemBetaMap so a launch file porting from
  /// `SemBetaMap::Params` to `SemDirMap::Params` produces equivalent runs;
  /// the unified-Dirichlet semantic differences are isolated to `kappa0`
  /// (now a no-op multiplier — see below) and the new `num_classes` /
  /// `alpha_0` priors.
  struct Params {
    double  resolution                 = 0.05;   ///< matches TsdfMap resolution
    uint8_t inner_bits                 = 2;
    uint8_t leaf_bits                  = 3;

    float   w_occ                      = 1.0f;   ///< per-hit total occupied-evidence mass
    float   w_free                     = 0.5f;   ///< per-carve FREE-evidence mass
    /// Multiplier on the per-class share of `w_occ · quality`. Default 1.0
    /// = all occupied mass distributed by softmax. `<1.0` reroutes part of
    /// the hit mass straight to OTHER (i.e. "I know it's occupied, but I
    /// distrust the class label proportionally"). **Deprecated under the
    /// unified-Dirichlet scheme** — kept only so SemBetaMap launch files
    /// migrate without edits. Set to 1.0 unless ablating.
    float   kappa0                     = 1.0f;
    float   dirichlet_min_p_occ        = 0.5f;   ///< gate per-class update
    float   evidence_saturation        = 0.0f;   ///< 0 disables; cap on s_total()
    float   carve_skip_occ_threshold   = 0.95f;  ///< wall-blocking guard
    float   range_decay_length         = 50.0f;  ///< exp(-r/L); 0 disables (unused here,
                                                  ///< quality is computed by caller)

    /// Dataset class count `C`. Determines the OTHER prior
    /// `(C − K_TOP − 1) · alpha_0`. Plumbed through the v3 wire format header
    /// so per-robot mappers and consensus_node agree. Default 14 (NYU13);
    /// SemanticKITTI uses 19.
    uint16_t num_classes               = 14;

    /// Symmetric Dirichlet prior `α_0` applied per underlying dim. See
    /// `kDefaultDirichletPrior` in semdir_voxel.hpp for rationale.
    float    alpha_0                   = kDefaultDirichletPrior;

    SemanticMode semantic_mode         = SemanticMode::DIRICHLET;
  };

  explicit SemDirMap(const Params& p);
  ~SemDirMap() = default;

  SemDirMap(const SemDirMap&)            = delete;
  SemDirMap& operator=(const SemDirMap&) = delete;
  SemDirMap(SemDirMap&&)                 = default;
  SemDirMap& operator=(SemDirMap&&)      = default;

  // ----------------------------------------------------------------------
  // Integration
  // ----------------------------------------------------------------------

  /// Real return. Carve free along [origin, endpoint) (per-voxel FREE bump),
  /// then update the endpoint voxel: distribute `w_occ · quality` of
  /// occupied mass across top-K class slots + OTHER.
  /// `sem_probs` may be nullptr; in that case all hit mass goes to OTHER
  /// (occupied evidence accumulated, class unidentified).
  void integrateHit(const Eigen::Vector3f&         origin,
                    const Eigen::Vector3f&         endpoint,
                    const std::vector<float>*      sem_probs,
                    float                          quality);

  /// No-return. Carve free along [origin, endpoint] (inclusive of endpoint).
  /// No hit update.
  void integrateMiss(const Eigen::Vector3f& origin,
                     const Eigen::Vector3f& endpoint,
                     float                  quality);

  // ----------------------------------------------------------------------
  // Per-voxel API (used by the ScovoxMapSplit fused walker)
  // ----------------------------------------------------------------------

  /// Per-voxel carve: FREE += `w_free · quality`, wall-blocked when
  /// `p_occ > carve_skip_occ_threshold`. Returns `false` if blocked.
  bool applyCarveUpdate(const CoordT& c, float quality);

  /// Per-voxel hit: distribute `w_occ · quality` of occupied mass across
  /// top-K + OTHER per softmax probabilities. Gated by `p_occ` per
  /// `semantic_mode` (DIRICHLET below the gate routes everything to OTHER).
  void applyHitUpdate(const CoordT&             c,
                      const std::vector<float>* sem_probs,
                      float                     quality);

  // ----------------------------------------------------------------------
  // Touched-set drain (Q7)
  // ----------------------------------------------------------------------

  std::vector<CoordT> drainTouched();
  void                clearTouched() noexcept { touched_.clear(); }
  std::size_t         touchedCount() const noexcept { return touched_.size(); }

  // ----------------------------------------------------------------------
  // Voxel queries
  // ----------------------------------------------------------------------

  std::optional<SemDirVoxel> getVoxel(const Eigen::Vector3f& pos) const;

  template <typename Fn>
  void forEachVoxel(Fn&& fn) const {
    const float h = 0.5f * static_cast<float>(params_.resolution);
    grid_.forEachCell([&](const SemDirVoxel& v, const CoordT& c) {
      const auto p = grid_.coordToPos(c);
      fn(v, Eigen::Vector3f(static_cast<float>(p.x) + h,
                            static_cast<float>(p.y) + h,
                            static_cast<float>(p.z) + h));
    });
  }

  // ----------------------------------------------------------------------
  // Memory accounting
  // ----------------------------------------------------------------------

  std::size_t voxelCount()      const;
  std::size_t gridMemoryBytes() const { return grid_.memUsage(); }

  // ----------------------------------------------------------------------
  // Accessors
  // ----------------------------------------------------------------------

  const Params& params() const noexcept { return params_; }
  const Grid&   grid()   const noexcept { return grid_; }
  Grid&         grid()         noexcept { return grid_; }

  /// Default-constructed voxel at this map's prior (`num_classes`, `α_0`).
  /// Exposed for callers that need to pre-populate test fixtures.
  SemDirVoxel defaultVoxel() const noexcept {
    return defaultSemDirVoxel(params_.num_classes, params_.alpha_0);
  }

 private:
  Params              params_;
  Grid                grid_;
  Grid::Accessor      acc_;
  std::vector<CoordT> touched_;
  /// Pre-computed OTHER prior `(C − K_TOP − 1) · α_0`. Hot-path constant.
  float               other_prior_;

  /// Carve free along the ray. `inclusive_endpoint` controls whether the
  /// endpoint voxel is also bumped (for misses) or skipped (for hits, where
  /// the endpoint gets the occupied update instead).
  void carveRay(const Eigen::Vector3f& origin,
                const Eigen::Vector3f& endpoint,
                float                  quality,
                bool                   inclusive_endpoint);

  /// Resolve `acc_.value(c, true)` and ensure the result is at the symmetric
  /// Dirichlet prior. Returns the in-grid pointer.
  SemDirVoxel* getOrAllocate(const CoordT& c);

  /// Cap on `s_total()` (mass-conserving scale-down of the whole vector).
  void applyEvidenceSaturation(SemDirVoxel* v) const;
};

}  // namespace scovox
