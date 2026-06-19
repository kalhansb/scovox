#pragma once

/// @file
/// @brief SCovox-contribution half of the split-grid refactor.
///
/// Owns a `Bonxai::VoxelGrid<SemBetaVoxel>` and runs the SCovox-specific
/// integration logic — Beta-occupied at the hit, Beta-free along the carve
/// ray, sparse-Dirichlet semantic update at the hit (gated by `p_occ`),
/// evidence saturation, wall-blocking guard.
///
/// What it does NOT do:
///   - TSDF distance / weight — lives in `TsdfMap`.
///   - Mesh extraction — `TsdfMap::extractGeometry()` + `labelMesh()`.

#include <Eigen/Core>
#include <bonxai/bonxai.hpp>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "scovox/sembeta_voxel.hpp"
#include "scovox/semantics.hpp"  // SemanticMode enum

namespace scovox {

class SemBetaMap {
 public:
  using Grid   = Bonxai::VoxelGrid<SemBetaVoxel>;
  using CoordT = Bonxai::CoordT;

  /// All Beta + Dirichlet integration knobs. Set once at construction; do not
  /// mutate at runtime (we do not currently re-emit a touched marker on
  /// param-change). Defaults match the SCovox production launch defaults.
  struct Params {
    double  resolution                 = 0.05;   ///< matches TsdfMap resolution
    uint8_t inner_bits                 = 2;
    uint8_t leaf_bits                  = 3;

    float   w_occ                      = 1.0f;   ///< Beta a_occ increment per hit
    float   w_free                     = 0.5f;   ///< Beta a_free increment per carve
    float   kappa0                     = 1.0f;   ///< Dirichlet concentration scale
    float   dirichlet_min_p_occ        = 0.5f;   ///< gate Dirichlet by p_occ
    float   evidence_saturation        = 0.0f;   ///< 0 disables; cap on a_*/sem_cnt
    float   carve_skip_occ_threshold   = 0.95f;  ///< wall-blocking guard
    float   range_decay_length         = 50.0f;  ///< exp(-r/L); 0 disables

    SemanticMode semantic_mode         = SemanticMode::DIRICHLET;
  };

  explicit SemBetaMap(const Params& p);
  ~SemBetaMap() = default;

  SemBetaMap(const SemBetaMap&)            = delete;
  SemBetaMap& operator=(const SemBetaMap&) = delete;
  SemBetaMap(SemBetaMap&&)                 = default;
  SemBetaMap& operator=(SemBetaMap&&)      = default;

  // ----------------------------------------------------------------------
  // Integration (Q4: separate hit and miss entry points)
  // ----------------------------------------------------------------------

  /// Real return: the ray ends in a measured surface point.
  ///   - Carve free along [origin, endpoint), Beta a_free update per voxel.
  ///   - At endpoint voxel: Beta a_occ update + sparse-Dirichlet semantic
  ///     update (gated by `p_occ >= dirichlet_min_p_occ` in DIRICHLET mode,
  ///     by `p_occ > 0.5` in NAIVE / MAJORITY_VOTE).
  /// `quality` is the per-ray confidence scalar (e.g. range_w · angle_w).
  /// `sem_probs` may be nullptr; semantic update is skipped if so.
  void integrateHit(const Eigen::Vector3f&         origin,
                    const Eigen::Vector3f&         endpoint,
                    const std::vector<float>*      sem_probs,
                    float                          quality);

  /// No-return: the ray exited at max-range without hitting a surface.
  ///   - Carve free along [origin, endpoint], Beta a_free update per voxel.
  ///   - **No** a_occ update (no surface to anchor to).
  ///   - **No** semantic update.
  /// `quality` is typically `range_w` evaluated at max range.
  void integrateMiss(const Eigen::Vector3f& origin,
                     const Eigen::Vector3f& endpoint,
                     float                  quality);

  // ----------------------------------------------------------------------
  // Fused-walker per-voxel API (2026-05-09)
  // ----------------------------------------------------------------------
  //
  // `ScovoxMapSplit::integrateHitFused` walks the union DDA (TSDF band
  // ⊇ SemBeta carve) once and feeds per-voxel updates into both grids
  // without recomputing SDF or rerunning Bresenham. These two methods
  // expose the per-voxel tail of `carveRay` and `updateHit` for that
  // path. Behaviour is byte-identical to the existing private helpers;
  // only the surface is widened.

  /// Per-voxel carve update at `c` — Beta a_free + wall-blocking guard.
  /// Returns false iff `c` is a wall (`p_occ > carve_skip_occ_threshold`),
  /// in which case the caller must stop emitting carve updates for the
  /// remainder of this ray. Allocates the voxel via `defaultSemBetaVoxel`
  /// if absent. No-op (returns true) when `w_free * quality <= 0`.
  bool applyCarveUpdate(const CoordT& c, float quality);

  /// Per-voxel hit update at `c` — Beta a_occ + sparse-Dirichlet semantic
  /// (gated by `p_occ` per `semantic_mode`). Allocates if absent.
  void applyHitUpdate(const CoordT&             c,
                      const std::vector<float>* sem_probs,
                      float                     quality);

  // ----------------------------------------------------------------------
  // Touched-set drain (Q7)
  // ----------------------------------------------------------------------

  std::vector<CoordT> drainTouched();
  /// O(n) clear of the touched buffer without sort+unique — use on the
  /// no-publisher path. See TsdfMap::clearTouched for rationale.
  void                clearTouched() noexcept { touched_.clear(); }
  std::size_t         touchedCount() const noexcept { return touched_.size(); }

  // ----------------------------------------------------------------------
  // Voxel queries
  // ----------------------------------------------------------------------

  std::optional<SemBetaVoxel> getVoxel(const Eigen::Vector3f& pos) const;

  template <typename Fn>
  void forEachVoxel(Fn&& fn) const {
    const float h = 0.5f * static_cast<float>(params_.resolution);
    grid_.forEachCell([&](const SemBetaVoxel& v, const CoordT& c) {
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

 private:
  Params              params_;
  Grid                grid_;
  Grid::Accessor      acc_;
  std::vector<CoordT> touched_;

  /// Carve free along [origin, endpoint). Stops at `carve_skip_occ_threshold`-
  /// confident occupied voxels (wall-blocking). `inclusive_endpoint` controls
  /// whether the endpoint voxel is also a_free'd (for misses) or skipped (for
  /// hits, where the endpoint gets a_occ instead).
  void carveRay(const Eigen::Vector3f& origin,
                const Eigen::Vector3f& endpoint,
                float                  quality,
                bool                   inclusive_endpoint);

  /// Update the hit voxel: Beta a_occ + (optionally) sparse-Dirichlet
  /// semantic update. Allocates the voxel via the defaultSemBetaVoxel()
  /// factory if it didn't exist (Q5 invariant).
  void updateHit(const CoordT&             c,
                 const std::vector<float>* sem_probs,
                 float                     quality);

  /// Resolve `acc_.value(c, true)` and ensure the result is in the
  /// Beta(1,1) prior state (Q5). Returns the in-grid pointer.
  SemBetaVoxel* getOrAllocate(const CoordT& c);

  /// Beta evidence-saturation cap (port from scovoxmap.cpp::apply_evidence_saturation).
  void applyEvidenceSaturation(SemBetaVoxel* v) const;
};

}  // namespace scovox
