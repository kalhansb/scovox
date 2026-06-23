#pragma once

/// @file scovox_map_split.hpp
/// @brief Composer for the split-grid SCovox refactor (Step 7.5: unified
/// Dirichlet), extended with a selectable semantic substrate.
///
/// Owns one `TsdfMap` (band-only, SLIM-VDB-equivalent) and ONE of two semantic
/// substrates, chosen by `Params::substrate`:
///   - `SemanticSubstrate::SEMDIR` (default): a `SemDirMap` — full-ray, unified
///     Dirichlet over `{top-K classes, OTHER, FREE}`. Wire format v3.
///   - `SemanticSubstrate::SPLIT`: a `SemSplitMap` — a `BetaVoxel` occupancy
///     grid ∥ a `DirVoxel` semantics grid (de-unified). Wire format v4.
///
/// Per-frame integration dispatches to TSDF + the active semantic substrate.
/// Mesh / pointcloud extraction comes from TsdfMap geometry + labelMesh /
/// labelPointCloud against the active semantic grid.
///
/// Naming note: the legacy `scovox::Map` (in scovox_mapping/scovoxmap.hpp) is
/// kept untouched for the existing pipeline. The pre-unified `SemBetaMap` is
/// kept as orphan code. The SEMDIR substrate stays byte-for-byte the prior
/// behaviour; SPLIT is the additive de-unification path (validate via the
/// flag, then promote).

#include <Eigen/Core>
#include <algorithm>
#include <bonxai/bonxai.hpp>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "scovox/marching_cubes.hpp"
#include "scovox/mesh_labelling.hpp"
#include "scovox/ray_iterator.hpp"
#include "scovox/sem_split_map.hpp"
#include "scovox/semdir_map.hpp"
#include "scovox/semdir_voxel.hpp"
#include "scovox/tsdf_map.hpp"
#include "scovox/tsdf_voxel.hpp"

namespace scovox {

/// Which semantic substrate the composer integrates into.
enum class SemanticSubstrate { SEMDIR, SPLIT };

class ScovoxMapSplit {
 public:
  using CoordT = Bonxai::CoordT;

  struct Params {
    /// Shared resolution / leaf hierarchy for all grids — coord identity
    /// across grids is required by labelMesh and the per-voxel cross-grid
    /// queries in scovox_node's publisher.
    double  resolution = 0.05;
    uint8_t inner_bits = 2;
    uint8_t leaf_bits  = 3;

    TsdfMap::Params   tsdf;
    SemDirMap::Params semdir;

    /// Semantic substrate selector. Default SEMDIR keeps the prior behaviour;
    /// SPLIT builds a `SemSplitMap` instead, deriving its knobs from `semdir`
    /// (the two Params are field-for-field equivalent).
    SemanticSubstrate substrate = SemanticSubstrate::SEMDIR;

    /// Fused ray walker. When true, `integrateHit` runs a single Bresenham
    /// DDA over the union range and feeds per-voxel updates into both the
    /// TSDF and the active semantic substrate. When false, falls back to the
    /// legacy split path (two DDAs). Default true.
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
      , fused_walker_(p.fused_walker)
      , use_split_(p.substrate == SemanticSubstrate::SPLIT) {
    // SPLIT: build the SemSplitMap, deriving its params from `semdir` (shared
    // resolution/hierarchy + the identical Bayesian knobs). semdir_ remains
    // constructed but unused (an empty grid + params, a few hundred bytes) so
    // the SEMDIR accessors stay valid and the diff vs. the prior code is small.
    if (use_split_) {
      semsplit_.emplace(toSplitParams(p.semdir, p.resolution,
                                      p.inner_bits, p.leaf_bits));
    }
  }

  // -------------------------------------------------------------------
  // Per-frame integration
  // -------------------------------------------------------------------

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
  /// dispatched into both `TsdfMap::applyBandUpdate` and (in the carve zone)
  /// the active semantic substrate's `applyCarveUpdate`/`applyHitUpdate`.
  void integrateHitFused(const Eigen::Vector3f&    origin,
                         const Eigen::Vector3f&    endpoint,
                         const std::vector<float>* sem_probs,
                         float                     quality) {
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();

    const Eigen::Vector3f d = endpoint - origin;
    const float depth = d.norm();
    if (depth < 1e-4f) {
      // Degenerate ray (origin≈endpoint): no TSDF or semantic work happens.
      // Attribute the (near-zero) bracket to tsdf_ns_ — same accumulator the
      // main return below uses — so the fused walker reports ALL of its time in
      // one bucket. (Routing this to semdir_ns_ would be doubly wrong: it does
      // no semantic work, and it splits the fused path's time across two
      // accumulators whose per-substrate split is meaningless on this path.)
      const auto t1 = clk::now();
      tsdf_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
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

      // (3) Hit (endpoint voxel) — semantic/occupancy update. Run this BEFORE
      // the proj≈0 early-return: when the endpoint lands exactly on a voxel
      // centre, v_point_voxel≈0 so proj≈0, and returning here would skip the
      // hit update entirely — leaving the surface voxel at prior and diverging
      // the fused walker from the non-fused SemDirMap::integrateHit, which
      // applies the hit unconditionally. The TSDF band update below may still
      // skip on proj≈0 (its sign is ill-defined there), but semHit must not.
      if (c == k_hit) {
        semHit(c, sem_probs, quality);
      }

      if (std::fabs(proj) < 1e-12f) return;
      const float sign = (proj > 0.f) ? 1.f : -1.f;
      const float sdf  = sign * dist;

      // (1) TSDF band update — gate + clamp + Curless–Levoy. The fused walker
      // always walks back to the origin (walk_back = max(depth, trunc)), so the
      // upper gate here must MATCH the non-fused TsdfMap::integrateRay band,
      // which depends on space_carving:
      //   - space_carving=false (Replica/KITTI default): the non-fused path
      //     walks only [hit−trunc, hit+trunc], so we keep the `sdf <= trunc + h`
      //     band gate; dropping it would write the whole front ray that the
      //     non-fused path never touches (and break the band invariant).
      //   - space_carving=true: the non-fused path walks [origin, hit+trunc] and
      //     applyBandUpdate clamps every in-front voxel (incl. sdf > trunc) to
      //     +trunc, so we drop the upper gate to integrate the full carve front.
      // applyBandUpdate owns the lower gate (`sdf <= -trunc` → drop) for both.
      if (tparams.space_carving || sdf <= trunc + h) {
        tsdf_.applyBandUpdate(c, sdf, tsdf_weight_fn);
      }

      // (2) semantic carve (interior of carve band, not the hit voxel).
      if (c != k_hit && !carve_blocked && sdf > 0.f && sdf <= carve_band) {
        if (!semCarve(c, quality)) {
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
    // Fused walker: TSDF band updates and semantic hit/carve are interleaved in
    // ONE per-voxel loop, so wall-clock cannot be cleanly attributed per
    // substrate without bracketing every applyBandUpdate vs semHit/semCarve with
    // a clock read — two steady_clock::now() calls per voxel would dominate and
    // distort the very cost being measured in this hot Bresenham loop. We
    // therefore report the COMBINED TSDF+semantic time under tsdf_ns_ and leave
    // semdir_ns_ untouched on the fused path (it reads 0). For a true
    // per-substrate split, run the non-fused integrateHitSplit walker, which
    // times the two DDAs separately. See tsdfTimeUs()/semdirTimeUs() docs.
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
    if (use_split_) semsplit_->integrateHit(origin, endpoint, sem_probs, quality);
    else            semdir_.integrateHit(origin, endpoint, sem_probs, quality);
    const auto t2 = clk::now();
    tsdf_ns_    += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    semdir_ns_  += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
  }

  /// No-return: semantic carve only; no TSDF update.
  void integrateMiss(const Eigen::Vector3f& origin,
                     const Eigen::Vector3f& endpoint,
                     float                  quality) {
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    if (use_split_) semsplit_->integrateMiss(origin, endpoint, quality);
    else            semdir_.integrateMiss(origin, endpoint, quality);
    const auto t1 = clk::now();
    semdir_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
  }

  // -------------------------------------------------------------------
  // Per-call timing accumulators
  // -------------------------------------------------------------------

  /// Accumulated TSDF time. NOTE: on the fused walker (fused_walker=true, the
  /// default) this is the COMBINED TSDF+semantic integration time — the fused
  /// loop interleaves both substrates and is not separable without per-voxel
  /// clock overhead. Only the non-fused integrateHitSplit / integrateMiss paths
  /// attribute TSDF and semantic time to separate accumulators.
  std::int64_t tsdfTimeUs()   const noexcept { return tsdf_ns_   / 1000; }
  /// Accumulated semantic-substrate time. On the fused walker this is 0 by
  /// design (the combined cost is reported under tsdfTimeUs()); it is non-zero
  /// only via integrateHitSplit (non-fused) and integrateMiss.
  std::int64_t semdirTimeUs() const noexcept { return semdir_ns_ / 1000; }
  void         resetTiming()        noexcept { tsdf_ns_ = 0; semdir_ns_ = 0; }

  // -------------------------------------------------------------------
  // Surface extraction
  // -------------------------------------------------------------------

  /// Triangle mesh from TSDF zero-crossing with per-triangle semantic labels.
  TriangleMesh extractMesh(float min_weight) const {
    auto geom = scovox::extractMesh(tsdf_.grid(), min_weight, resolution_);
    if (use_split_) {
      geom.tri_labels = scovox::labelMesh(geom, tsdf_.grid(),
                                          semsplit_->dirGrid(), semsplit_->params().alpha_0);
    } else {
      geom.tri_labels = scovox::labelMesh(geom, tsdf_.grid(), semdir_.grid(),
                                          semdir_.params().alpha_0);
    }
    return geom;
  }

  /// Voxel-centre point cloud with per-point semantic labels.
  std::pair<std::vector<Eigen::Vector3f>, std::vector<uint16_t>>
  extractPointCloud(float min_weight) const {
    auto positions = scovox::extractPointCloud(tsdf_.grid(), min_weight, resolution_);
    std::vector<uint16_t> labels;
    if (use_split_) {
      labels = scovox::labelPointCloud(positions, semsplit_->dirGrid(),
                                       semsplit_->params().alpha_0);
    } else {
      labels = scovox::labelPointCloud(positions, semdir_.grid(),
                                       semdir_.params().alpha_0);
    }
    return {positions, labels};
  }

  // -------------------------------------------------------------------
  // Wire-format support (drain-on-publish)
  // -------------------------------------------------------------------

  std::vector<CoordT> drainTouchedTsdf()   { return tsdf_.drainTouched(); }

  /// SEMDIR substrate: the single unified-Dirichlet touched set (v3 publish).
  /// Intended for `!usesSplit()`; harmless (empty) under SPLIT since `semdir_`
  /// is never integrated into there.
  std::vector<CoordT> drainTouchedSemDir() { return semdir_.drainTouched(); }

  /// SPLIT substrate: per-grid touched sets (v4 publish). Beta is full-ray;
  /// Dir is hit-sparse. Return empty under SEMDIR (guards the empty optional).
  std::vector<CoordT> drainTouchedBeta() {
    return use_split_ ? semsplit_->drainTouchedBeta() : std::vector<CoordT>{};
  }
  std::vector<CoordT> drainTouchedDir() {
    return use_split_ ? semsplit_->drainTouchedDir() : std::vector<CoordT>{};
  }

  void clearTouchedTsdf()   noexcept { tsdf_.clearTouched(); }
  void clearTouchedSemDir() noexcept {
    if (use_split_) semsplit_->clearTouched();
    else            semdir_.clearTouched();
  }

  // -------------------------------------------------------------------
  // Memory / diagnostics
  // -------------------------------------------------------------------

  std::size_t tsdfVoxelCount()   const { return tsdf_.voxelCount();      }
  std::size_t tsdfGridBytes()    const { return tsdf_.gridMemoryBytes(); }

  /// In SEMDIR mode these report the SemDir grid. In SPLIT mode the voxel
  /// count reports the Dir (semantics) grid and bytes report Beta + Dir
  /// combined, so existing memlog call sites stay meaningful.
  std::size_t semdirVoxelCount() const {
    return use_split_ ? semsplit_->dirVoxelCount() : semdir_.voxelCount();
  }
  std::size_t semdirGridBytes()  const {
    return use_split_
        ? (semsplit_->betaGridMemoryBytes() + semsplit_->dirGridMemoryBytes())
        : semdir_.gridMemoryBytes();
  }

  // SPLIT-only per-grid accounting (for v4 memlog / parity reporting).
  // Return 0 under SEMDIR so callers never deref the empty optional.
  std::size_t betaVoxelCount() const { return use_split_ ? semsplit_->betaVoxelCount() : 0; }
  std::size_t dirVoxelCount()  const { return use_split_ ? semsplit_->dirVoxelCount()  : 0; }
  std::size_t betaGridBytes()  const { return use_split_ ? semsplit_->betaGridMemoryBytes() : 0; }
  std::size_t dirGridBytes()   const { return use_split_ ? semsplit_->dirGridMemoryBytes()  : 0; }

  // -------------------------------------------------------------------
  // Direct grid / substrate access
  // -------------------------------------------------------------------

  TsdfMap&         tsdf()         { return tsdf_;   }
  const TsdfMap&   tsdf()   const { return tsdf_;   }

  /// Valid only when `!usesSplit()`.
  SemDirMap&       semdir()       { return semdir_; }
  const SemDirMap& semdir() const { return semdir_; }

  /// Valid only when `usesSplit()` — asserts the precondition (returning a
  /// reference, there is no safe empty fallback).
  SemSplitMap& semsplit() {
    assert(use_split_ && "ScovoxMapSplit::semsplit() requires substrate == SPLIT");
    return *semsplit_;
  }
  const SemSplitMap& semsplit() const {
    assert(use_split_ && "ScovoxMapSplit::semsplit() requires substrate == SPLIT");
    return *semsplit_;
  }

  bool   usesSplit()  const noexcept { return use_split_; }
  double resolution() const { return resolution_; }

 private:
  static SemSplitMap::Params toSplitParams(const SemDirMap::Params& s,
                                           double res, uint8_t ib, uint8_t lb) {
    SemSplitMap::Params o;
    o.resolution               = res;
    o.inner_bits               = ib;
    o.leaf_bits                = lb;
    o.w_occ                    = s.w_occ;
    o.w_free                   = s.w_free;
    o.kappa0                   = s.kappa0;
    o.dirichlet_min_p_occ      = s.dirichlet_min_p_occ;
    o.evidence_saturation      = s.evidence_saturation;
    o.carve_skip_occ_threshold = s.carve_skip_occ_threshold;
    o.range_decay_length       = s.range_decay_length;
    o.num_classes              = s.num_classes;
    o.alpha_0                  = s.alpha_0;
    o.semantic_mode            = s.semantic_mode;
    return o;
  }

  /// Per-voxel semantic carve dispatch (active substrate).
  bool semCarve(const CoordT& c, float quality) {
    return use_split_ ? semsplit_->applyCarveUpdate(c, quality)
                      : semdir_.applyCarveUpdate(c, quality);
  }
  /// Per-voxel semantic hit dispatch (active substrate).
  void semHit(const CoordT& c, const std::vector<float>* sem_probs, float quality) {
    if (use_split_) semsplit_->applyHitUpdate(c, sem_probs, quality);
    else            semdir_.applyHitUpdate(c, sem_probs, quality);
  }

  TsdfMap                    tsdf_;
  SemDirMap                  semdir_;       ///< SEMDIR substrate (always built)
  std::optional<SemSplitMap> semsplit_;     ///< SPLIT substrate (built iff use_split_)
  double                     resolution_;
  bool                       fused_walker_;
  bool                       use_split_;

  std::int64_t tsdf_ns_   = 0;
  std::int64_t semdir_ns_ = 0;
};

}  // namespace scovox
