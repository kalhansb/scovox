#pragma once
/// @file map_interface.hpp
/// @brief SCovox map parameters and core type re-exports.
///
/// Zero ROS dependencies.  Part of scovox_core.
///
/// The concrete Map class lives in scovox/scovoxmap.hpp (scovox_mapping package).
/// The LogOddsMap class lives in log_odds_map.hpp (log_odds_mapping package).
/// Moved comments: doc/scovox_core_code_notes.md

#include <vector>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <bonxai/bonxai.hpp>

// Re-export core types so downstream code needs only one include
#include "scovox/voxel.hpp"
#include "scovox/uncertainty.hpp"
#include "scovox/semantics.hpp"
#include "scovox/ray_iterator.hpp"

namespace scovox {

// =====================================================================
// Parameters
// =====================================================================

struct Params {
  double resolution = 0.05;  ///< Voxel edge length (meters)

  // -- Bonxai grid structure --
  uint8_t inner_bits = 2;  ///< Inner grid dimension bits (2 → 4×4×4 inner cells)
  uint8_t leaf_bits  = 3;  ///< Leaf block dimension bits (3 → 8×8×8 = 512 voxels/block)
                           ///< Use leaf_bits=1 for sparse outdoor LiDAR to reduce waste
  /// Leaf block bits for the semantic (Dir) grid only; see
  /// SemSplitMap::Params::dir_leaf_bits. 2 gives 4×4×4 = 64 voxels per block.
  /// Clamped to <= leaf_bits, never coarser than the occupancy grid.
  /// (notes: params-dir-leaf-bits)
  uint8_t dir_leaf_bits = 2;

  // -- Occupancy update weights --
  // Beta-Bernoulli equivalent of OctoMap log-odds: w_occ = (2·prob_hit − 1)/(1
  // − prob_hit), w_free = 1/prob_miss − 2. The ratio w_occ/w_free matters more
  // than scale for steady-state p_occ. (notes: params-occupancy-weights)
  float w_free = 1.0f;   ///< Evidence added per free-space traversal
  float w_occ  = 2.0f;   ///< Evidence added per hit observation

  // Wall protection during free-space carving; <= 0 (the default) disables it,
  // so a beam clears stale obstacles it crosses. The batched (live) carve path
  // ignores it; only the immediate path uses a positive value.
  // (notes: params-carve-wall-guard)
  float carve_skip_occ_threshold = 0.0f;

  // Batched free-space carve toggle. When false, the live batch path still
  // accepts hit updates but stages no free-space evidence; immediate carve
  // calls are unchanged.
  bool batch_free_carve = true;

  // -- Production knobs (load-bearing on Replica m2f mIoU) --
  // (notes: params-production-knobs)
  uint16_t evidence_saturation = 1000;       ///< Cap on (a_occ, a_free, sem_cnt). 0 = disabled.
  float    dirichlet_min_p_occ = 0.5f;       ///< Skip Dirichlet update when p_occ below this. 0 = disabled.

  // -- Semantics --
  SemanticMode semantic_mode = SemanticMode::DIRICHLET;
  float kappa0 = 2.0f;
  /// Number of explicitly tracked semantic classes per voxel (Space-Saving
  /// heavy hitters): a class with more than n/k of a voxel's n observations is
  /// always tracked, with count error at most n/k. (notes: params-top-k)
  int   top_k  = K_TOP;
  float semantic_occ_gate = 0.5f;  ///< Publish/visualization threshold (post-mapping)

  // -- Consensus fusion (used only by Map::consensusMerge in scovox_mapping) --
  // Both fields are no-ops (consensusMerge is a pure Beta–Dirichlet conjugate
  // update); kept so launch files that set them stay compatible.
  // (notes: params-consensus-deprecated)
  float consensus_kl_threshold  = 5.0f;
  float consensus_tau_occ_gate  = 0.6f;

  // -- Range/angle weighting --
  float range_decay_length = 5.0f;
  float min_range = 0.3f;
  float max_range = 10.0f;
  float grazing_angle_threshold = 0.3f;

  // -- Free-carve guard --

  // -- TSDF integration --
  // Truncation distance in metres; 0 disables TSDF integration. The node sets
  // it from sdf_trunc_voxels * resolution so it scales with resolution.
  // (notes: params-sdf-trunc)
  float sdf_trunc          = 0.0f;
  // If true, also write TSDF mass for voxels in [origin, depth-trunc].
  // Off mirrors the VDBFusion default — voxels far in front of any surface
  // are not part of the TSDF.
  bool  tsdf_space_carving = false;

  // -- Band-only integration mode (benchmarking / TSDF-only deployments) --
  // When true, fused_integrate_ray_static walks only the truncation band around
  // the hit, so no free space is carved outside it (planner and frontier
  // detection see none there). Default off; for benchmarking.
  // (notes: params-band-only-integration)
  bool  band_only_integration = false;
};

} // namespace scovox
