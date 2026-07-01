#pragma once
/// @file map_interface.hpp
/// @brief SCovox map parameters and core type re-exports.
///
/// Zero ROS dependencies.  Part of scovox_core.
///
/// The concrete Map class lives in scovox/scovoxmap.hpp (scovox_mapping package).
/// The LogOddsMap class lives in log_odds_map.hpp (log_odds_mapping package).

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

  // -- Occupancy update weights --
  // Sensor-physics derivation (Beta-Bernoulli equivalent of OctoMap log-odds).
  // From Beta(1,1) prior + inverse sensor model (prob_hit, prob_miss):
  //   w_occ  = (2·prob_hit  − 1) / (1 − prob_hit)
  //   w_free = 1/prob_miss − 2
  //
  // Calibration table (per sensor):
  //   RGB-D conservative (OctoMap defaults: prob_hit=0.7, prob_miss=0.4):
  //     w_occ  = (1.4−1)/0.3 = 1.33
  //     w_free = 1/0.4 − 2   = 0.5
  //     Why: noisy stereo / structured-light depth at range — many spurious
  //     hits at object edges, frequent missed returns on dark/specular
  //     surfaces. Low confidence per observation; rely on accumulation.
  //
  //   RGB-D moderate (current defaults: prob_hit≈0.75, prob_miss≈0.33):
  //     w_occ=2.0, w_free=1.0
  //     Why: typical depth camera (RealSense, Azure Kinect) at indoor range
  //     with clean optics — slightly more confident than OctoMap's worst-case.
  //
  //   LiDAR (prob_hit≈0.9, prob_miss≈0.15):
  //     w_occ  = (1.8−1)/0.1 = 8.0
  //     w_free = 1/0.15 − 2  ≈ 4.67
  //     Why: time-of-flight LiDAR has very low FP rate (returns are physical
  //     reflections) and low FN rate in clear air — each ray is high-evidence,
  //     so weights are large and the map converges fast.
  //
  // Refit from data only if calibration logs show drift. Ratio w_occ/w_free
  // matters more than absolute scale for steady-state p_occ.
  float w_free = 1.0f;   ///< Evidence added per free-space traversal
  float w_occ  = 2.0f;   ///< Evidence added per hit observation

  // Wall protection during free-space carving — DISABLED by default (<= 0).
  // We trust the most recent LiDAR scan: a beam that reached its return proves
  // every voxel it traversed is free NOW, so a previously-occupied (moved/stale)
  // obstacle in its path should clear, not block the carve. The batched carve
  // path (the live pipeline) ignores this guard entirely; a positive value only
  // re-enables it for the immediate (unbatched) path — offline tools/ablations.
  // (Formerly 0.7; the joint ray-cast reach_prob variant was reverted for a
  // ~6 mIoU cold-start tax on indoor RGB-D — see docs/exp_ablations.md.)
  float carve_skip_occ_threshold = 0.0f;

  // -- Production knobs (load-bearing on Replica m2f mIoU) --
  // Together these restore OLD-pipeline mIoU within noise (verified by
  // res_step_bayesian: 0.3488 vs OLD baseline_newgate_05 0.3481). The four
  // pre-cleanup knobs (smc, smooth-gate sigmoid) were proved redundant —
  // these two carry all the load. See docs/exp_ablations.md.
  uint16_t evidence_saturation = 1000;       ///< Cap on (a_occ, a_free, sem_cnt). 0 = disabled.
  float    dirichlet_min_p_occ = 0.5f;       ///< Skip Dirichlet update when p_occ below this. 0 = disabled.

  // -- Semantics --
  SemanticMode semantic_mode = SemanticMode::DIRICHLET;
  float kappa0 = 2.0f;
  /// Number of explicitly tracked semantic classes per voxel.
  ///
  /// The sparse representation is equivalent to the Space-Saving /
  /// Misra-Gries streaming heavy-hitter algorithm (Metwally et al.
  /// ICDT 2005; Misra & Gries 1982).
  ///
  /// Per Cormode (2016, Encyclopedia of Algorithms): any class with
  /// true frequency > n/k is guaranteed to be tracked, and the
  /// additive count error on any tracked class is at most n/k,
  /// where n = total observations at that voxel and k = top_k.
  ///
  /// Calibration: at 10 fps with ~5 s voxel residence (n ~ 50),
  /// K=2 tracks everything above ~25% observation share; K=4 tracks
  /// above ~12%. Indoor scenes rarely have >2 genuine classes per
  /// voxel (interior + boundary), so K=2 is sufficient for most
  /// deployments; K=4 provides margin for noisy classifiers.
  int   top_k  = K_TOP;
  float semantic_occ_gate = 0.5f;  ///< Publish/visualization threshold (post-mapping)

  // -- Consensus fusion (used only by Map::consensusMerge in scovox_mapping) --
  // Deprecated 2026-05-03: both fields are no-ops. consensusMerge is now a
  // pure Beta–Dirichlet conjugate update (additive under conditional
  // independence). The KL-conflict gate was never read by any caller and
  // the post-merge occupancy gate on semantics was non-Bayesian. Fields
  // retained so launch files that set them remain compatible.
  float consensus_kl_threshold  = 5.0f;
  float consensus_tau_occ_gate  = 0.6f;

  // -- Range/angle weighting --
  float range_decay_length = 5.0f;
  float min_range = 0.3f;
  float max_range = 10.0f;
  float grazing_angle_threshold = 0.3f;

  // -- Free-carve guard --

  // -- TSDF integration --
  // Truncation distance in metres. 0 disables TSDF integration. The node
  // sets this from `sdf_trunc_voxels * resolution` so it scales with
  // resolution; a fixed metre default would silently break across launch
  // files at different resolutions (5 cm vs 10 cm).
  float sdf_trunc          = 0.0f;
  // If true, also write TSDF mass for voxels in [origin, depth-trunc].
  // Off mirrors the VDBFusion default — voxels far in front of any surface
  // are not part of the TSDF.
  bool  tsdf_space_carving = false;

  // -- Band-only integration mode (benchmarking / TSDF-only deployments) --
  // When true, fused_integrate_ray_static walks only the truncation band
  //   [posToCoord(hit - sdf_trunc * u), posToCoord(hit + sdf_trunc * u)]
  // instead of the full ray from origin. This matches SLIM-VDB / VDBFusion's
  // band-only DDA and skips per-voxel Beta-free carving along the long part
  // of the ray. The Beta occupancy posterior degrades to "carved only inside
  // the band"; downstream consumers that rely on free-space outside the band
  // (planner, frontier detection) will see no carved free voxels there.
  // Default off — the full-ray walk is the SCovox feature that distinguishes
  // it from a TSDF-only system; this flag is for benchmarking the integration
  // cost of that feature against SLIM-VDB on matched workloads.
  bool  band_only_integration = false;
};

} // namespace scovox
