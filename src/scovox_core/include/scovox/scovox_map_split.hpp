#pragma once

/// @file scovox_map_split.hpp
/// @brief Composer for the split-grid SCovox refactor (Step 7.5: unified
/// Dirichlet).
///
/// Owns one `TsdfMap` (band-only, SLIM-VDB-equivalent) and one `SemDirMap`
/// (full-ray, unified Dirichlet over `{top-K classes, OTHER, FREE}`).
/// Per-frame integration dispatches to both. Mesh / pointcloud extraction
/// comes from TsdfMap geometry + labelMesh / labelPointCloud against SemDir.
///
/// Naming note: the legacy `scovox::Map` (in scovox_mapping/scovoxmap.hpp)
/// is kept untouched for the existing pipeline. The pre-unified `SemBetaMap`
/// is kept as orphan code (still builds, no production caller) so the
/// just-landed 87 scovox_core tests remain green. This composer migrates
/// to `SemDirMap` per the 2026-05-13 design doc; once the ROS layer
/// (Step 8) and experiments validate the new path, SemBetaMap retires.

#include <Eigen/Core>
#include <algorithm>
#include <bonxai/bonxai.hpp>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "scovox/marching_cubes.hpp"
#include "scovox/mesh_labelling.hpp"
#include "scovox/ray_iterator.hpp"
#include "scovox/semdir_map.hpp"
#include "scovox/semdir_voxel.hpp"
#include "scovox/tsdf_map.hpp"
#include "scovox/tsdf_voxel.hpp"

namespace scovox {

class ScovoxMapSplit {
 public:
  using CoordT = Bonxai::CoordT;

  struct Params {
    /// Shared resolution / leaf hierarchy for both grids — coord identity
    /// across grids is required by labelMesh and the per-voxel cross-grid
    /// queries in scovox_node's NPZ publisher.
    double  resolution = 0.05;
    uint8_t inner_bits = 2;
    uint8_t leaf_bits  = 3;

    TsdfMap::Params   tsdf;
    SemDirMap::Params semdir;

    /// `mesh_min_weight_publish` and friends previously lived on Map::Params;
    /// they're consumed here as per-call args to `extractMesh`/`extractPointCloud`.

    /// Fused ray walker (Step 12.10, 2026-05-09). When true, `integrateHit`
    /// runs a single Bresenham DDA over the union range
    /// `[Hp - sdf_trunc·û, Hp + sdf_trunc·û]` and feeds per-voxel updates
    /// into both grids. When false, falls back to the legacy split path
    /// (TsdfMap DDA + separate SemDirMap DDA + endpoint hit). Default
    /// true; flip to false for A/B parity testing.
    bool fused_walker = true;
  };

  explicit ScovoxMapSplit(const Params& p)
      : tsdf_(   [&]{ auto t = p.tsdf;
                      t.resolution = p.resolution;
                      t.inner_bits = p.inner_bits;
                      t.leaf_bits  = p.leaf_bits;
                      return t; }())
      , semdir_( [&]{ auto s = p.semdir;
                      s.resolution = p.resolution;
                      s.inner_bits = p.inner_bits;
                      s.leaf_bits  = p.leaf_bits;
                      return s; }())
      , resolution_(p.resolution)
      , fused_walker_(p.fused_walker) {}

  // -------------------------------------------------------------------
  // Per-frame integration
  // -------------------------------------------------------------------

  /// Real return: TSDF band update + SemDir carve + unified-Dirichlet
  /// hit. Dispatches to either the fused single-DDA walker (default;
  /// Step 12.10) or the legacy split path (two DDAs, kept for A/B parity).
  void integrateHit(const Eigen::Vector3f&         origin,
                    const Eigen::Vector3f&         endpoint,
                    const std::vector<float>*      sem_probs,
                    float                          quality) {
    if (fused_walker_) {
      integrateHitFused(origin, endpoint, sem_probs, quality);
    } else {
      integrateHitSplit(origin, endpoint, sem_probs, quality);
    }
  }

  /// Fused walker — one Bresenham DDA, per-voxel SDF computed once and
  /// dispatched into both `TsdfMap::applyBandUpdate` and (in the carve
  /// zone) `SemDirMap::applyCarveUpdate`/`applyHitUpdate`.
  void integrateHitFused(const Eigen::Vector3f&    origin,
                         const Eigen::Vector3f&    endpoint,
                         const std::vector<float>* sem_probs,
                         float                     quality) {
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();

    const Eigen::Vector3f d = endpoint - origin;
    const float depth = d.norm();
    if (depth < 1e-4f) {
      const auto t1 = clk::now();
      semdir_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
      return;
    }
    const Eigen::Vector3f u = d / depth;
    const float carve_band = depth;

    const auto& tparams = tsdf_.params();
    const float trunc = tparams.sdf_trunc;
    const float h     = 0.5f * static_cast<float>(tparams.resolution);

    const float walk_back = tparams.space_carving
        ? depth
        : std::max(depth, trunc);
    const Eigen::Vector3f start_pos = endpoint - walk_back * u;
    const Eigen::Vector3f end_pos   = endpoint + trunc * u;

    auto& grid = tsdf_.grid();
    const auto k0    = grid.posToCoord(start_pos.x(), start_pos.y(), start_pos.z());
    const auto k_far = grid.posToCoord(end_pos.x(),   end_pos.y(),   end_pos.z());
    const auto k_hit = grid.posToCoord(endpoint.x(),  endpoint.y(),  endpoint.z());

    const auto tsdf_weight_fn = TsdfMap::constant(1.0f);  // SLIM-VDB default
    bool carve_blocked = false;
    bool k_hit_visited = false;

    auto visit_one = [&](const CoordT& c) {
      if (c == k_hit) k_hit_visited = true;

      const auto p = grid.coordToPos(c);
      const Eigen::Vector3f vc(static_cast<float>(p.x) + h,
                               static_cast<float>(p.y) + h,
                               static_cast<float>(p.z) + h);
      const Eigen::Vector3f v_voxel_origin = vc - origin;
      const Eigen::Vector3f v_point_voxel  = endpoint - vc;
      const float dist = v_point_voxel.norm();
      const float proj = v_voxel_origin.dot(v_point_voxel);
      if (std::fabs(proj) < 1e-12f) return;
      const float sign = (proj > 0.f) ? 1.f : -1.f;
      const float sdf  = sign * dist;

      // (1) TSDF band update — gate + clamp + Curless–Levoy.
      if (sdf <= trunc + h) {
        tsdf_.applyBandUpdate(c, sdf, tsdf_weight_fn);
      }

      // (2) SemDir carve (interior of carve band, not the hit voxel)
      // (3) Hit (endpoint voxel) — unified Dirichlet update.
      if (c == k_hit) {
        semdir_.applyHitUpdate(c, sem_probs, quality);
      } else if (!carve_blocked && sdf > 0.f && sdf <= carve_band) {
        if (!semdir_.applyCarveUpdate(c, quality)) {
          carve_blocked = true;
        }
      }
    };

    if (k0 == k_far) {
      visit_one(k0);
    } else {
      RayIterator(k0, k_far, [&](const CoordT& c) -> bool {
        visit_one(c);
        return true;
      });
      visit_one(k_far);
      if (!k_hit_visited && k_hit != k_far && k_hit != k0) {
        visit_one(k_hit);
      }
    }

    const auto t1 = clk::now();
    tsdf_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
  }

  /// Legacy split walker (two DDAs). Kept for A/B parity testing.
  void integrateHitSplit(const Eigen::Vector3f&    origin,
                         const Eigen::Vector3f&    endpoint,
                         const std::vector<float>* sem_probs,
                         float                     quality) {
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    tsdf_.integrateRay(origin, endpoint);
    const auto t1 = clk::now();
    semdir_.integrateHit(origin, endpoint, sem_probs, quality);
    const auto t2 = clk::now();
    tsdf_ns_    += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    semdir_ns_  += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
  }

  /// No-return: SemDir carve only; no TSDF update.
  void integrateMiss(const Eigen::Vector3f& origin,
                     const Eigen::Vector3f& endpoint,
                     float                  quality) {
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    semdir_.integrateMiss(origin, endpoint, quality);
    const auto t1 = clk::now();
    semdir_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
  }

  // -------------------------------------------------------------------
  // Per-call timing accumulators
  // -------------------------------------------------------------------

  std::int64_t tsdfTimeUs()   const noexcept { return tsdf_ns_   / 1000; }
  std::int64_t semdirTimeUs() const noexcept { return semdir_ns_ / 1000; }
  void         resetTiming()        noexcept { tsdf_ns_ = 0; semdir_ns_ = 0; }

  // -------------------------------------------------------------------
  // Surface extraction
  // -------------------------------------------------------------------

  /// Triangle mesh from TSDF zero-crossing with per-triangle labels from SemDir.
  TriangleMesh extractMesh(float min_weight) const {
    auto geom = scovox::extractMesh(tsdf_.grid(), min_weight, resolution_);
    geom.tri_labels = scovox::labelMesh(geom, tsdf_.grid(), semdir_.grid());
    return geom;
  }

  /// Voxel-centre point cloud with per-point labels from SemDir.
  std::pair<std::vector<Eigen::Vector3f>, std::vector<uint16_t>>
  extractPointCloud(float min_weight) const {
    auto positions = scovox::extractPointCloud(
        tsdf_.grid(), min_weight, resolution_);
    auto labels = scovox::labelPointCloud(positions, semdir_.grid());
    return {positions, labels};
  }

  // -------------------------------------------------------------------
  // Wire-format support (Q7: drain-on-publish)
  // -------------------------------------------------------------------

  std::vector<CoordT> drainTouchedTsdf()   { return tsdf_.drainTouched();   }
  std::vector<CoordT> drainTouchedSemDir() { return semdir_.drainTouched(); }

  void clearTouchedTsdf()   noexcept { tsdf_.clearTouched();   }
  void clearTouchedSemDir() noexcept { semdir_.clearTouched(); }

  // -------------------------------------------------------------------
  // Memory / diagnostics
  // -------------------------------------------------------------------

  std::size_t tsdfVoxelCount()   const { return tsdf_.voxelCount();        }
  std::size_t semdirVoxelCount() const { return semdir_.voxelCount();      }
  std::size_t tsdfGridBytes()    const { return tsdf_.gridMemoryBytes();   }
  std::size_t semdirGridBytes()  const { return semdir_.gridMemoryBytes(); }

  // -------------------------------------------------------------------
  // Direct grid access (consensus_node, NPZ publisher)
  // -------------------------------------------------------------------

  TsdfMap&         tsdf()         { return tsdf_;   }
  const TsdfMap&   tsdf()   const { return tsdf_;   }
  SemDirMap&       semdir()       { return semdir_; }
  const SemDirMap& semdir() const { return semdir_; }

  double resolution() const { return resolution_; }

 private:
  TsdfMap     tsdf_;
  SemDirMap   semdir_;
  double      resolution_;
  bool        fused_walker_;

  std::int64_t tsdf_ns_   = 0;
  std::int64_t semdir_ns_ = 0;
};

}  // namespace scovox
