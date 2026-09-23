#pragma once

/// @file
/// @brief SLIM-VDB-equivalent TSDF substrate for the split-grid SCovox
/// refactor.
///
/// Owns a `Bonxai::VoxelGrid<TsdfVoxel>` and integrates depth/lidar rays
/// into it using the verbatim VDBFusion math:
///   1. Voxel **centre** sample point (not lower corner).
///   2. Signed **Euclidean** distance to the measured endpoint.
///   3. **Per-voxel** weighting via a caller-supplied `WeightFn`; default
///      `constant(1.0)` matches SLIM-VDB's pipeline scripts byte-for-byte.
///
/// What it does NOT do:
///   - Beta occupancy (a_occ / a_free / wall blocking) — lives in `SemBetaMap`.
///   - Sparse Dirichlet (`sem_cnt` / `sem_cls` / `sparse_add`) — also
///     `SemBetaMap`.
///   - Mesh extraction with labels — that's `extractGeometry()` (geometry
///     only) plus `labelMesh(geom, tsdf, sembeta)` (free function in
///     `mesh_labelling.hpp`).
///
/// Cross-grid contract: `TsdfMap` deliberately does not include
/// `sembeta_voxel.hpp`. The SLIM-VDB-equivalent row constructs and uses a
/// `TsdfMap` *alone*, with no SemBeta dependency.
/// Moved comments: doc/scovox_core_code_notes.md

#include <Eigen/Core>
#include <bonxai/bonxai.hpp>
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

#include "scovox/tsdf_voxel.hpp"

namespace scovox {

class TsdfMap {
 public:
  using Grid    = Bonxai::VoxelGrid<TsdfVoxel>;
  using CoordT  = Bonxai::CoordT;

  /// Per-voxel weight function. Receives the SDF at the voxel and returns
  /// a non-negative weight. `constant(1.0)` is SLIM-VDB's default; other
  /// factories are exposed for ablations.
  using WeightFn = std::function<float(float /*sdf*/)>;

  struct Params {
    /// Voxel size in metres. Same as the SemBetaMap resolution.
    double  resolution    = 0.05;
    /// Bonxai grid hierarchy parameters (8^inner_bits inner mask, 8^leaf_bits
    /// leaf mask). Default 2/3 → 4×4×4 inner blocks of 8×8×8 leaves.
    uint8_t inner_bits    = 2;
    uint8_t leaf_bits     = 3;
    /// Signed-distance truncation (m): behind the surface |sdf| > sdf_trunc is
    /// skipped, in front it depends on space_carving. Must be > 0; values <= 0
    /// are reset at construction, as TSDF-off is unsupported here.
    /// (notes: tsdf-sdf-trunc)
    float   sdf_trunc     = 0.15f;
    /// SLIM-VDB-style space carving. False (default): walk only
    /// `[hit - trunc·û, hit + trunc·û]`, matching SLIM-VDB's
    /// `space_carving=false` Replica/KITTI default. True: walk
    /// `[origin, hit + trunc·û]`, matching SLIM-VDB's `space_carving=true`.
    bool    space_carving = false;
  };

  explicit TsdfMap(const Params& p);
  ~TsdfMap() = default;

  TsdfMap(const TsdfMap&)            = delete;
  TsdfMap& operator=(const TsdfMap&) = delete;
  TsdfMap(TsdfMap&&)                 = default;
  TsdfMap& operator=(TsdfMap&&)      = default;

  // ----------------------------------------------------------------------
  // Integration
  // ----------------------------------------------------------------------

  /// Curless-Levoy update along one ray's truncation band (world positions,
  /// metres): sdf from voxel centre to endpoint, clamped to ±sdf_trunc; sdf <=
  /// -sdf_trunc or zero weight is skipped. Touched coords go to drainTouched().
  /// (notes: tsdf-integrate-ray)
  void integrateRay(const Eigen::Vector3f& origin,
                    const Eigen::Vector3f& endpoint,
                    const WeightFn&        weight_fn = constant(1.0f));

  // ----------------------------------------------------------------------
  // Touched-set drain (Q7) — for the dual-stream wire format on dSCovox
  // ----------------------------------------------------------------------

  /// Returns the set of unique voxel coords touched since the last drain,
  /// then clears the internal buffer. O(n log n) in the touched count
  /// (sort + unique).
  std::vector<CoordT> drainTouched();

  /// O(n) clear of the touched buffer, no sort+unique. Use it when the drained
  /// coords would be discarded anyway (no publisher).
  /// (notes: tsdf-clear-touched)
  void clearTouched() noexcept { touched_.clear(); }

  /// Touched-set size without draining. Diagnostics / rate-limiting.
  std::size_t touchedCount() const noexcept { return touched_.size(); }

  // ----------------------------------------------------------------------
  // Voxel queries
  // ----------------------------------------------------------------------

  /// Lookup by world-space position. Returns nullopt if no voxel exists at
  /// that coord (i.e. unobserved).
  std::optional<TsdfVoxel> getVoxel(const Eigen::Vector3f& pos) const;

  /// Walk every observed voxel. `fn` is called as
  /// `fn(const TsdfVoxel&, Eigen::Vector3f voxel_centre_world)`.
  /// The position passed in is the voxel **centre** (NOT lower corner) —
  /// consistent with SLIM-VDB's `xform.indexToWorld(c) + voxel_size/2`.
  template <typename Fn>
  void forEachVoxel(Fn&& fn) const {
    const float h = 0.5f * static_cast<float>(params_.resolution);
    grid_.forEachCell([&](const TsdfVoxel& v, const CoordT& c) {
      const auto p = grid_.coordToPos(c);
      fn(v, Eigen::Vector3f(static_cast<float>(p.x) + h,
                            static_cast<float>(p.y) + h,
                            static_cast<float>(p.z) + h));
    });
  }

  // ----------------------------------------------------------------------
  // Memory accounting (Q2 / feedback_slimvdb_memory_measurement.md)
  // ----------------------------------------------------------------------

  std::size_t voxelCount()      const;
  std::size_t gridMemoryBytes() const { return grid_.memUsage(); }

  // ----------------------------------------------------------------------
  // Weighting function factories
  // ----------------------------------------------------------------------

  /// SLIM-VDB / VDBFusion default. Used by both
  /// `slimvdb_pipelines/replica_pipeline.cpp` and
  /// `slimvdb_pipelines/kitti_pipeline.cpp`.
  static WeightFn constant(float w = 1.0f);

  /// Tent function: weight 1.0 at the surface, 0.0 at the band edges.
  static WeightFn linear(float w_max, float trunc);

  /// Gaussian centred on the surface with stddev `sigma`.
  static WeightFn exponential(float sigma);

  /// SCovox-historical: per-ray-end range decay applied as if it were a
  /// per-voxel function. Provided as an ablation hatch only — not used by
  /// the production SCovox row.
  static WeightFn rangeDecay(float L_metres, float ray_depth_metres);

  // ----------------------------------------------------------------------
  // Accessors
  // ----------------------------------------------------------------------

  const Params& params() const noexcept { return params_; }
  const Grid&   grid()   const noexcept { return grid_; }
  Grid&         grid()         noexcept { return grid_; }

  // ----------------------------------------------------------------------
  // Fused-walker per-voxel API (2026-05-09)
  // ----------------------------------------------------------------------
  // applyBandUpdate is the per-voxel tail of visit() after the SDF math,
  // exposed so ScovoxMapSplit::integrateHitFused can feed precomputed SDFs
  // without recomputing them here. (notes: tsdf-fused-walker-api)

  /// Band update at c for a precomputed sdf (positive before the endpoint).
  /// Drops sdf <= -trunc and weight_fn(sdf) <= 0; otherwise clamps to ±trunc,
  /// runs Curless-Levoy and pushes c to the touched buffer.
  /// (notes: tsdf-apply-band-update)
  void applyBandUpdate(const CoordT& c, float sdf, const WeightFn& weight_fn);

 private:
  Params              params_;
  Grid                grid_;
  Grid::Accessor      acc_;
  std::vector<CoordT> touched_;

  /// The actual integration body. Translates origin/endpoint into Bonxai
  /// coords, runs the DDA (matching SLIM-VDB's range setup modulo the
  /// `RayIterator` parity gaps in §1.1 of the design doc), and applies the
  /// Curless–Levoy update at every visited voxel inside the truncation band.
  void integrateRayImpl(const Eigen::Vector3f& origin,
                        const Eigen::Vector3f& endpoint,
                        const WeightFn&        weight_fn);

  /// Per-voxel update body — broken out so the DDA loop and the explicit
  /// `k_far` / `k_hit` revisits share one source of truth for the math.
  void visit(const CoordT&          c,
             const Eigen::Vector3f& origin,
             const Eigen::Vector3f& endpoint,
             float                  half_voxel,
             float                  trunc,
             const WeightFn&        weight_fn);
};

}  // namespace scovox
