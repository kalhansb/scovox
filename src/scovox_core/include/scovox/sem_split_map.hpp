#pragma once

/// @file sem_split_map.hpp
/// @brief Split Beta/Dirichlet SCovox-contribution map — the de-unified
/// alternative to `SemDirMap`.
///
/// Owns **two** Bonxai grids instead of `SemDirMap`'s one:
///   - `Bonxai::VoxelGrid<BetaVoxel>` (8 B) — occupancy, full-ray (carved on
///     every traversed voxel).
///   - `Bonxai::VoxelGrid<DirVoxel>`  (16 B) — occupied-class semantics,
///     hit-only and gated (allocated only where a class is committed).
///
/// API mirrors `SemDirMap` (`integrateHit/Miss`, `applyCarveUpdate/HitUpdate`,
/// `drainTouched*`, `getVoxel`, `forEach*`, memory accounting) so the
/// `ScovoxMapSplit` composer can select it behind a runtime flag.
///
/// Update model — the SemBeta two-stream calibration, with SemDir-matched
/// priors (see `beta_voxel.hpp` / `dir_voxel.hpp`):
///   carve:  Beta `a_free += w_free·quality`  (full ray, wall-guarded)
///   hit:    Stream A  Beta `a_occ += w_occ·quality`   (always)
///           read `p_occ_post` from the Beta grid
///           Stream B  (gated on `p_occ_post`) commit `kappa0·p_occ_post·
///                     quality` of class mass into the Dir grid via the
///                     mass-conserving `sparse_add_class`.
///
/// Difference vs. unified `SemDirVoxel`: class evidence (Stream B) does **not**
/// feed back into the occupancy marginal — occupancy is driven solely by
/// occupancy evidence. This is the intended decoupling of the split; it costs
/// the unified single-vector formalism but keeps strict *per-grid* mass
/// conservation (Beta: `a_occ + a_free`; Dir: `other + Σcnt`).

#include <Eigen/Core>
#include <bonxai/bonxai.hpp>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "scovox/beta_voxel.hpp"
#include "scovox/dir_voxel.hpp"
#include "scovox/semantics.hpp"  // SemanticMode

namespace scovox {

class SemSplitMap {
 public:
  using BetaGrid = Bonxai::VoxelGrid<BetaVoxel>;
  using DirGrid  = Bonxai::VoxelGrid<DirVoxel>;
  using CoordT   = Bonxai::CoordT;

  /// Knobs mirror `SemDirMap::Params` 1:1 so a launch file porting between the
  /// two substrates needs no edits. The occupancy prior is derived from
  /// `num_classes` / `alpha_0` to match the unified path's `p_occ` marginal.
  struct Params {
    double  resolution                 = 0.05;
    uint8_t inner_bits                 = 2;
    uint8_t leaf_bits                  = 3;

    float   w_occ                      = 1.0f;   ///< Beta a_occ increment per hit (Stream A)
    float   w_free                     = 0.5f;   ///< Beta a_free increment per carve
    float   kappa0                     = 1.0f;   ///< class-share multiplier (Stream B)
    float   dirichlet_min_p_occ        = 0.5f;   ///< gate per-class update on Beta p_occ
    float   evidence_saturation        = 0.0f;   ///< 0 disables; per-grid cap (see below)
    float   carve_skip_occ_threshold   = 0.95f;  ///< wall-blocking guard
    float   range_decay_length         = 50.0f;  ///< exp(-r/L); 0 disables (caller-applied)

    /// Dataset class count `C`. Sets the OTHER prior `(C − K_TOP)·α₀` and the
    /// Beta occupancy prior `a_occ = C·α₀` so `p_occ_prior = C/(C+1)` matches
    /// the unified path.
    uint16_t num_classes               = 14;
    /// Symmetric per-dim Dirichlet prior `α₀`.
    float    alpha_0                   = kDefaultDirichletPrior;

    SemanticMode semantic_mode         = SemanticMode::DIRICHLET;
  };

  explicit SemSplitMap(const Params& p);
  ~SemSplitMap() = default;

  SemSplitMap(const SemSplitMap&)            = delete;
  SemSplitMap& operator=(const SemSplitMap&) = delete;
  SemSplitMap(SemSplitMap&&)                 = default;
  SemSplitMap& operator=(SemSplitMap&&)      = default;

  // ----------------------------------------------------------------------
  // Integration
  // ----------------------------------------------------------------------

  /// Real return: carve free (Beta) along [origin, endpoint), then Stream A
  /// occupancy + gated Stream B class commit at the endpoint voxel.
  void integrateHit(const Eigen::Vector3f&    origin,
                    const Eigen::Vector3f&    endpoint,
                    const std::vector<float>* sem_probs,
                    float                     quality);

  /// No-return: carve free (Beta) along [origin, endpoint] inclusive. No hit.
  void integrateMiss(const Eigen::Vector3f& origin,
                     const Eigen::Vector3f& endpoint,
                     float                  quality);

  // ----------------------------------------------------------------------
  // Per-voxel API (used by the ScovoxMapSplit fused walker)
  // ----------------------------------------------------------------------

  /// Per-voxel carve: Beta `a_free += w_free·quality`, wall-blocked when
  /// `p_occ > carve_skip_occ_threshold`. Touches the Beta grid only. Returns
  /// `false` if blocked (caller stops carving the rest of the ray).
  bool applyCarveUpdate(const CoordT& c, float quality);

  /// Per-voxel hit: Stream A (Beta) always, then gated Stream B (Dir). Touches
  /// the Beta grid always; allocates + touches the Dir grid only when a class
  /// is actually committed (the sparse-semantics memory win).
  void applyHitUpdate(const CoordT&             c,
                      const std::vector<float>* sem_probs,
                      float                     quality);

  // ----------------------------------------------------------------------
  // Touched-set drains (per grid — Beta is full-ray, Dir is hit-sparse)
  // ----------------------------------------------------------------------

  std::vector<CoordT> drainTouchedBeta();
  std::vector<CoordT> drainTouchedDir();
  void                clearTouched() noexcept { touched_beta_.clear(); touched_dir_.clear(); }
  std::size_t         touchedBetaCount() const noexcept { return touched_beta_.size(); }
  std::size_t         touchedDirCount()  const noexcept { return touched_dir_.size(); }

  // ----------------------------------------------------------------------
  // Voxel queries
  // ----------------------------------------------------------------------

  std::optional<BetaVoxel> getBetaVoxel(const Eigen::Vector3f& pos) const;
  std::optional<DirVoxel>  getDirVoxel(const Eigen::Vector3f& pos) const;

  /// Argmax class at `pos`, or 0xFFFF if no Dir voxel / no dominant class.
  uint16_t dominantClassAt(const Eigen::Vector3f& pos) const;

  template <typename Fn>
  void forEachBeta(Fn&& fn) const {
    const float h = 0.5f * static_cast<float>(params_.resolution);
    beta_grid_.forEachCell([&](const BetaVoxel& v, const CoordT& c) {
      const auto p = beta_grid_.coordToPos(c);
      fn(v, Eigen::Vector3f(static_cast<float>(p.x) + h,
                            static_cast<float>(p.y) + h,
                            static_cast<float>(p.z) + h));
    });
  }

  template <typename Fn>
  void forEachDir(Fn&& fn) const {
    const float h = 0.5f * static_cast<float>(params_.resolution);
    dir_grid_.forEachCell([&](const DirVoxel& v, const CoordT& c) {
      const auto p = dir_grid_.coordToPos(c);
      fn(v, Eigen::Vector3f(static_cast<float>(p.x) + h,
                            static_cast<float>(p.y) + h,
                            static_cast<float>(p.z) + h));
    });
  }

  // ----------------------------------------------------------------------
  // Memory accounting
  // ----------------------------------------------------------------------

  std::size_t betaVoxelCount()    const;
  std::size_t dirVoxelCount()     const;
  std::size_t betaGridMemoryBytes() const { return beta_grid_.memUsage(); }
  std::size_t dirGridMemoryBytes()  const { return dir_grid_.memUsage(); }

  // ----------------------------------------------------------------------
  // Accessors
  // ----------------------------------------------------------------------

  const Params&   params()   const noexcept { return params_; }
  const BetaGrid& betaGrid() const noexcept { return beta_grid_; }
  BetaGrid&       betaGrid()       noexcept { return beta_grid_; }
  const DirGrid&  dirGrid()  const noexcept { return dir_grid_; }
  DirGrid&        dirGrid()        noexcept { return dir_grid_; }

  BetaVoxel defaultBeta() const noexcept {
    return defaultBetaVoxel(beta_occ_prior_, beta_free_prior_);
  }
  DirVoxel defaultDir() const noexcept {
    return defaultDirVoxel(params_.num_classes, params_.alpha_0);
  }

 private:
  Params               params_;
  BetaGrid             beta_grid_;
  DirGrid              dir_grid_;
  BetaGrid::Accessor   beta_acc_;
  DirGrid::Accessor    dir_acc_;
  std::vector<CoordT>  touched_beta_;
  std::vector<CoordT>  touched_dir_;

  /// Beta priors derived from (num_classes, alpha_0) to match SemDir's p_occ
  /// marginal. Hot-path constants.
  float                beta_occ_prior_;   ///< C · α₀
  float                beta_free_prior_;  ///< α₀

  void carveRay(const Eigen::Vector3f& origin,
                const Eigen::Vector3f& endpoint,
                float                  quality,
                bool                   inclusive_endpoint);

  BetaVoxel* getOrAllocateBeta(const CoordT& c);
  DirVoxel*  getOrAllocateDir(const CoordT& c);

  void applyBetaSaturation(BetaVoxel* b) const;
  void applyDirSaturation(DirVoxel* d) const;
};

}  // namespace scovox
