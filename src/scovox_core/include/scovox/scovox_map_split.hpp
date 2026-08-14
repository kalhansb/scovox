#pragma once

/// @file scovox_map_split.hpp
/// @brief Composer for the split-grid SCovox substrate (split).
///
/// Owns one `TsdfMap` (band-only, SLIM-VDB-equivalent) and one `SemSplitMap`
/// semantic substrate — a `BetaVoxel` occupancy grid ∥ a `DirVoxel` semantics
/// grid (de-unified). This is the wire path; it is the only substrate.
///
/// Per-frame integration dispatches to TSDF + the SemSplitMap substrate.
/// Mesh / pointcloud extraction comes from TsdfMap geometry + labelMesh /
/// labelPointCloud against the Dir (semantics) grid.

#include <Eigen/Core>
#include <algorithm>
#include <bonxai/bonxai.hpp>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "scovox/marching_cubes.hpp"
#include "scovox/mesh_labelling.hpp"
#include "scovox/ray_iterator.hpp"
#include "scovox/refinement_regions.hpp"
#include "scovox/sem_split_map.hpp"
#include "scovox/tsdf_map.hpp"
#include "scovox/tsdf_voxel.hpp"

namespace scovox {

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
    /// Semantic (Dir) grid leaf bits — the ONE piece of geometry that is not
    /// shared. Coord identity across grids is a function of `resolution` only,
    /// so the labelMesh / cross-grid queries above are unaffected. See
    /// SemSplitMap::Params::dir_leaf_bits. 0 = inherit leaf_bits.
    uint8_t dir_leaf_bits = 2;

    TsdfMap::Params     tsdf;
    SemSplitMap::Params semsplit;

    /// Fused ray walker. When true, `integrateHit` runs a single Bresenham
    /// DDA over the union range and feeds per-voxel updates into both the
    /// TSDF and the semantic substrate. When false, falls back to the
    /// two-DDA split path. Default true.
    bool fused_walker = true;

    /// When false, the fused walker skips every TSDF band write, so the TsdfMap
    /// grid stays empty. For callers that disabled TSDF (sdf_trunc passed as 0)
    /// but whose TsdfMap::Params still sanitises sdf_trunc back to a positive
    /// default — the band integration is then pure dead work if the TsdfMap grid
    /// is never read. Default true → TSDF integrates exactly as before. The
    /// occupancy (Beta) and semantic (Dir) substrates are unaffected either way.
    bool tsdf_enabled = true;

    // ---- Fine TSDF band (localized two-lattice refinement) ----
    // docs/design/fine_tsdf_band_dbh_2026_07_30.md. 0 = off (default —
    // byte-identical behaviour, no fine grid allocated). k > 0 adds a second
    // sparse TSDF-only lattice at res_fine = resolution / 2^k, written only
    // inside registered refinement cylinders (addRefinementRegion), in a
    // ±fine_sdf_trunc_voxels fine-voxel band around gated hits. Independent
    // of tsdf_enabled: the fine band can run with the coarse TSDF off
    // (e.g. the occupancy-only LiDAR config).
    uint8_t fine_ratio_log2       = 0;
    int     fine_sdf_trunc_voxels = 3;      ///< fine trunc = this · res_fine
    float   fine_region_margin    = 0.15f;  ///< gate radius = model r + margin
    /// Per-scan anchor re-registration (drift absorption). When true, each
    /// scan's staged in-region hits are rigidly shifted by the 2-DoF fit
    /// against the region's canonical cylinder before fusing — see
    /// fitCylinderAnchorShift. When false, hits fuse at their odometry pose.
    bool            fine_anchor_enable = true;
    AnchorFitParams fine_anchor;
  };

  explicit ScovoxMapSplit(const Params& p)
      : tsdf_(    [&]{ auto t = p.tsdf;
                       t.resolution = p.resolution;
                       t.inner_bits = p.inner_bits;
                       t.leaf_bits  = p.leaf_bits;
                       return t; }())
      , semsplit_([&]{ auto s = p.semsplit;
                       s.resolution    = p.resolution;
                       s.inner_bits    = p.inner_bits;
                       s.leaf_bits     = p.leaf_bits;
                       s.dir_leaf_bits = p.dir_leaf_bits;
                       return s; }())
      , resolution_(p.resolution)
      , fused_walker_(p.fused_walker)
      , tsdf_enabled_(p.tsdf_enabled)
      , sem_band_(semsplit_.params().semantic_band_length)
      , fine_ratio_log2_(p.fine_ratio_log2)
      , fine_region_margin_(p.fine_region_margin)
      , fine_anchor_enable_(p.fine_anchor_enable)
      , fine_anchor_params_(p.fine_anchor) {
    // The band lives in the fused walker only (see integrateHitSplit). Asking
    // for both is a request that cannot be honoured, and honouring it silently
    // as "endpoint only" would hand back a null result that looks like a
    // measurement. Refuse loudly and zero the knob so `sem_band_` and the
    // params() the node prints agree with what actually runs.
    if (sem_band_ > 0.f && !fused_walker_) {
      std::fprintf(stderr,
          "[scovox] FATAL CONFIG: semantic_band_length=%.3f requires "
          "fused_walker=true; the split walker does not implement the band. "
          "Refusing to run endpoint-only under a band label.\n",
          static_cast<double>(sem_band_));
      std::abort();
    }
    if (fine_ratio_log2_ > 0) {
      // Second, independent TSDF lattice at res_fine. Band-only by
      // construction (space_carving=false → integrateRay walks exactly
      // [hit − trunc·û, hit + trunc·û]); trunc derives from the FINE voxel
      // count so it works with the coarse TSDF disabled (sdf_trunc = 0).
      TsdfMap::Params fp;
      fp.resolution    = p.resolution / static_cast<double>(1u << fine_ratio_log2_);
      fp.inner_bits    = p.inner_bits;
      fp.leaf_bits     = p.leaf_bits;
      fp.sdf_trunc     = static_cast<float>(
          std::max(1, p.fine_sdf_trunc_voxels) * fp.resolution);
      fp.space_carving = false;
      fine_tsdf_ = std::make_unique<TsdfMap>(fp);
    }
  }

  // -------------------------------------------------------------------
  // Per-frame integration
  // -------------------------------------------------------------------

  /// `is_dynamic` (default false) routes the endpoint's occupancy + semantics to
  /// the transient substrate and suppresses the persistent TSDF surface for this
  /// ray, so moving objects leave no permanent geometry; the free-space carve
  /// stays persistent. See SemSplitMap::applyHitUpdate / decayTransient.
  void integrateHit(const Eigen::Vector3f&         origin,
                    const Eigen::Vector3f&         endpoint,
                    const std::vector<float>*      sem_probs,
                    float                          quality,
                    bool                           is_dynamic = false,
                    const HitWeights*              prof = nullptr) {
    // Fine TSDF band routing — walker-independent, so the fused and split
    // paths stay in parity. Dynamic rays leave no fine surface (same rule
    // as the coarse TSDF); geometry-off sources (RGB-D overlay) never
    // touch geometry. One null-check when the fine band is off.
    if (fine_tsdf_ && !is_dynamic && !(prof && prof->geometry_off)) {
      stageFineHit(origin, endpoint);
    }
    if (fused_walker_) {
      integrateHitFused(origin, endpoint, sem_probs, quality, is_dynamic, prof);
    } else {
      integrateHitSplit(origin, endpoint, sem_probs, quality, is_dynamic, prof);
    }
  }

  /// Fused walker — one Bresenham DDA, per-voxel SDF computed once and
  /// dispatched into both `TsdfMap::applyBandUpdate` and (in the carve zone)
  /// the semantic substrate's `applyCarveUpdate`/`applyHitUpdate`.
  void integrateHitFused(const Eigen::Vector3f&    origin,
                         const Eigen::Vector3f&    endpoint,
                         const std::vector<float>* sem_probs,
                         float                     quality,
                         bool                      is_dynamic = false,
                         const HitWeights*         prof = nullptr) {
    // A geometry-off source (RGB-D semantics overlay) writes NO TSDF band this
    // ray, so its noisy depth never enters the Curless–Levoy surface — geometry
    // stays 100% LiDAR. Null prof / geometry_off=false keeps TSDF exactly as
    // before. Combined with the existing is_dynamic suppression below.
    const bool geometry_off = (prof && prof->geometry_off);
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();

    const Eigen::Vector3f d = endpoint - origin;
    const float depth = d.norm();
    if (depth < 1e-4f) {
      // Degenerate ray (origin≈endpoint): no TSDF or semantic work happens.
      // Attribute the (near-zero) bracket to tsdf_ns_ — same accumulator the
      // main return below uses — so the fused walker reports ALL of its time in
      // one bucket. (Routing this to sem_ns_ would be doubly wrong: it does
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

    // SLIM-VDB-style flat semantic band (Params::semantic_band_length). Decided
    // once per ray, not per voxel, so the branch inside the DDA is a bool test.
    // Excluded cases, all of them deliberate:
    //   is_dynamic     — a moving object must not paint its class onto the
    //                    static surfaces its beam passes through or stops on;
    //                    its endpoint already routes to the transient grids.
    //   geometry_off /
    //   kernel_radius  — an RGB-D overlay source owns the BKI ball path
    //                    (applyHitUpdateKernel). Running the band as well would
    //                    deposit that source's class twice per hit.
    //   no sem_probs   — a bare geometric return has no opinion to pool.
    const bool band_active = sem_band_ > 0.f && !is_dynamic && !geometry_off
                          && !(prof && prof->kernel_radius > 0.f)
                          && sem_probs && !sem_probs->empty();

    const float walk_back = tparams.space_carving
        ? depth
        : std::max(depth, trunc);
    // The band is symmetric about the surface, but the walk behind it normally
    // stops at `trunc`. Extend the far end when the band reaches deeper, else
    // the behind-surface half is silently clipped and the knob stops meaning
    // what it says. At band ≤ trunc — the mirror configuration, both 0.30 m on
    // KITTI — this is the old expression exactly and costs no extra steps.
    // Voxels gained beyond trunc have sdf ≤ −trunc, which applyBandUpdate drops
    // and the semCarve gate (sdf > 0) never sees, so TSDF/occupancy are unmoved.
    const float back_reach = band_active ? std::max(trunc, sem_band_) : trunc;
    const Eigen::Vector3f start_pos = endpoint - walk_back * u;
    const Eigen::Vector3f end_pos   = endpoint + back_reach * u;

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
      // the fused walker from the non-fused SemSplitMap::integrateHit, which
      // applies the hit unconditionally. The TSDF band update below may still
      // skip on proj≈0 (its sign is ill-defined there), but semHit must not.
      if (c == k_hit) {
        semHit(c, sem_probs, quality, is_dynamic, prof);
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
      // Dynamic rays write NO persistent TSDF: a moving object must not leave a
      // permanent surface. Its occupancy/semantics live in the transient grids
      // (semHit above, is_dynamic=true); the free-space carve below stays
      // persistent (the air the object passed through is genuinely free).
      if (tsdf_enabled_ && !is_dynamic && !geometry_off && (tparams.space_carving || sdf <= trunc + h)) {
        tsdf_.applyBandUpdate(c, sdf, tsdf_weight_fn);
      }

      // (1b) SLIM-VDB-style flat semantic band. This is the exact window
      // SLIM-VDB's Integrate writes `alpha[label] += 1` over — `sdf > -trunc`
      // on a ray it truncates at depth ± trunc — evaluated on voxels this DDA
      // is already standing on, which is why it costs no traversal.
      //
      // The endpoint is excluded: semHit above already deposited there, and
      // banding it too would give the surface voxel double weight relative to
      // its neighbours, inverting the smoothing this is meant to apply.
      //
      // Ordered BEFORE the carve so the occupancy `applyBandSemantic` reads is
      // this voxel's pre-carve state on the immediate path. On the live batched
      // path the carve is staged until flushCarveFrame, so p_occ cannot move
      // mid-scan and the two orders coincide — but they must not diverge
      // between paths, so the order is pinned here rather than left to luck.
      if (band_active && c != k_hit && sdf > -sem_band_ && sdf <= sem_band_) {
        semBand(c, sem_probs, quality, prof);
      }

      // (2) semantic carve (interior of carve band, not the hit voxel).
      if (c != k_hit && !carve_blocked && sdf > 0.f && sdf <= carve_band) {
        if (!semCarve(c, quality, prof)) {
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
    // sem_ns_ untouched on the fused path (it reads 0). For a true per-substrate
    // split, run the non-fused integrateHitSplit walker, which times the two
    // DDAs separately. See tsdfTimeUs()/semdirTimeUs() docs.
    tsdf_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
  }

  /// Non-fused split walker (two DDAs). Kept for A/B parity testing.
  ///
  /// NOTE: this path does NOT implement `semantic_band_length`. The band is
  /// defined as "deposit on the voxels the walker is already standing on", and
  /// the whole claim being tested is that this costs no extra traversal — which
  /// is only true of the fused walker's single DDA. Reimplementing it here
  /// would mean a third DDA and would measure something else. Callers get a
  /// hard warning at construction rather than silent endpoint-only numbers;
  /// see the `band + !fused_walker` check in ScovoxMapSplit's constructor.
  void integrateHitSplit(const Eigen::Vector3f&    origin,
                         const Eigen::Vector3f&    endpoint,
                         const std::vector<float>* sem_probs,
                         float                     quality,
                         bool                      is_dynamic = false,
                         const HitWeights*         prof = nullptr) {
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    // Dynamic rays write no persistent TSDF (no ghost surface); the carve inside
    // semsplit_.integrateHit stays persistent, only the endpoint routes. A
    // geometry-off source (RGB-D overlay) also writes no TSDF — geometry stays
    // LiDAR-only (parity with the fused walker's line-185 gate).
    if (!is_dynamic && !(prof && prof->geometry_off)) tsdf_.integrateRay(origin, endpoint);
    const auto t1 = clk::now();
    semsplit_.integrateHit(origin, endpoint, sem_probs, quality, is_dynamic, prof);
    const auto t2 = clk::now();
    tsdf_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    sem_ns_  += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
  }

  /// No-return: semantic carve only; no TSDF update.
  void integrateMiss(const Eigen::Vector3f& origin,
                     const Eigen::Vector3f& endpoint,
                     float                  quality,
                     const HitWeights*      prof = nullptr) {
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    semsplit_.integrateMiss(origin, endpoint, quality, prof);
    const auto t1 = clk::now();
    sem_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
  }

  /// Per-frame decay of the transient (dynamic-class) substrate. Call once per
  /// integrated frame; see SemSplitMap::decayTransient.
  void decayTransient(float rate) { semsplit_.decayTransient(rate); }

  // -------------------------------------------------------------------
  // Per-scan carve batching (universal free-space path)
  // -------------------------------------------------------------------

  /// Open a carve frame on the semantic substrate: every carve (fused walker or
  /// non-fused integrateHit/Miss) is staged read-free until flushCarveFrame().
  /// Wrap a whole scan's rays in beginCarveFrame()/flushCarveFrame(). See
  /// SemSplitMap for the rationale (full-ray free-space, one write per voxel).
  /// Also opens the fine-band scan frame: gated hits are staged per region
  /// so flushCarveFrame can anchor-correct the whole scan before fusing.
  void beginCarveFrame() {
    semsplit_.beginCarveFrame();
    beginFineFrame();
  }

  /// Write all staged carves for the scan (one Beta update per unique voxel,
  /// block-ordered, occupied-wins). Timed into the same tsdf_ns_ bucket as the
  /// fused walk, so tsdfTimeUs() reflects total carve cost (walk staging +
  /// flush). Returns the number of voxels written (carve only — fine-band ray
  /// count is reported via fineLastFrameRays()).
  std::size_t flushCarveFrame() {
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    const std::size_t n = semsplit_.flushCarveFrame();
    flushFineFrame();
    const auto t1 = clk::now();
    tsdf_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    return n;
  }

  // -------------------------------------------------------------------
  // Fine TSDF band — localized two-lattice refinement
  // (docs/design/fine_tsdf_band_dbh_2026_07_30.md)
  // -------------------------------------------------------------------

  bool fineEnabled() const noexcept { return fine_tsdf_ != nullptr; }
  uint8_t fineRatioLog2() const noexcept { return fine_ratio_log2_; }
  double fineResolution() const {
    return fine_tsdf_ ? fine_tsdf_->params().resolution : 0.0;
  }

  /// Register (or replace, keyed on id) a refinement cylinder. Hits inside
  /// `radius + fine_region_margin` (and the z band) start routing to the
  /// fine lattice immediately. No-op returning -1 when the fine band is off.
  int addRefinementRegion(const RefinementCylinder& cyl) {
    if (!fine_tsdf_) return -1;
    const int idx = fine_regions_.add(cyl, fine_region_margin_);
    if (static_cast<std::size_t>(idx) >= fine_staged_.size())
      fine_staged_.resize(fine_regions_.slotCount());
    return idx;
  }

  /// Unregister a region. Already-fused fine voxels stay in the lattice
  /// (the region gate is an integration-time policy, not storage).
  bool removeRefinementRegion(uint32_t id) {
    return fine_tsdf_ ? fine_regions_.remove(id) : false;
  }

  const RefinementRegions& refinementRegions() const { return fine_regions_; }

  /// Feed one raw sensor return to the fine band ONLY — no coarse-map write.
  /// This is the full-density path: nodes typically voxel-grid-downsample
  /// each scan before `integrateHit`, which caps what the fine lattice can
  /// see at one return per downsample cell. Routing every raw (deskewed)
  /// return here instead gives refinement regions the sensor's native point
  /// density while the coarse map keeps its downsampled diet. Same gate,
  /// staging, and per-scan anchor treatment as dispatcher-staged hits;
  /// out-of-region endpoints are a no-op after one O(1) hash lookup.
  void refineHit(const Eigen::Vector3f& origin, const Eigen::Vector3f& endpoint) {
    if (fine_tsdf_) stageFineHit(origin, endpoint);
  }

  TsdfMap&       fineTsdf()       { return *fine_tsdf_; }
  const TsdfMap& fineTsdf() const { return *fine_tsdf_; }

  std::vector<CoordT> drainTouchedFine() {
    return fine_tsdf_ ? fine_tsdf_->drainTouched() : std::vector<CoordT>{};
  }
  void clearTouchedFine() noexcept {
    if (fine_tsdf_) fine_tsdf_->clearTouched();
  }
  std::size_t fineVoxelCount() const {
    return fine_tsdf_ ? fine_tsdf_->voxelCount() : 0;
  }
  std::size_t fineGridBytes() const {
    return fine_tsdf_ ? fine_tsdf_->gridMemoryBytes() : 0;
  }

  /// Fine-band rays fused in the last flushed scan frame.
  std::size_t fineLastFrameRays() const noexcept { return fine_last_rays_; }
  /// Anchor shift magnitude applied in the last flushed frame (max over
  /// regions; 0 when no region had enough points or the anchor is off).
  float fineLastFrameShift() const noexcept { return fine_last_shift_; }
  /// Cumulative scans-with-region where the anchor fit fell back to
  /// odometry (too few points / rejected shift).
  std::size_t fineAnchorFallbacks() const noexcept { return fine_anchor_fallbacks_; }

  // -------------------------------------------------------------------
  // Per-call timing accumulators
  // -------------------------------------------------------------------

  /// Accumulated TSDF time. NOTE: on the fused walker (fused_walker=true, the
  /// default) this is the COMBINED TSDF+semantic integration time — the fused
  /// loop interleaves both substrates and is not separable without per-voxel
  /// clock overhead. Only the non-fused integrateHitSplit / integrateMiss paths
  /// attribute TSDF and semantic time to separate accumulators.
  std::int64_t tsdfTimeUs()   const noexcept { return tsdf_ns_ / 1000; }
  /// Accumulated semantic-substrate time. On the fused walker this is 0 by
  /// design (the combined cost is reported under tsdfTimeUs()); it is non-zero
  /// only via integrateHitSplit (non-fused) and integrateMiss.
  std::int64_t semdirTimeUs() const noexcept { return sem_ns_ / 1000; }
  void         resetTiming()        noexcept { tsdf_ns_ = 0; sem_ns_ = 0; }

  // -------------------------------------------------------------------
  // Surface extraction
  // -------------------------------------------------------------------

  /// Triangle mesh from TSDF zero-crossing with per-triangle semantic labels.
  TriangleMesh extractMesh(float min_weight) const {
    auto geom = scovox::extractMesh(tsdf_.grid(), min_weight, resolution_);
    geom.tri_labels = scovox::labelMesh(geom, tsdf_.grid(),
                                        semsplit_.dirGrid(), semsplit_.params().alpha_0);
    return geom;
  }

  /// Voxel-centre point cloud with per-point semantic labels.
  std::pair<std::vector<Eigen::Vector3f>, std::vector<uint16_t>>
  extractPointCloud(float min_weight) const {
    auto positions = scovox::extractPointCloud(tsdf_.grid(), min_weight, resolution_);
    auto labels = scovox::labelPointCloud(positions, semsplit_.dirGrid(),
                                          semsplit_.params().alpha_0);
    return {positions, labels};
  }

  // -------------------------------------------------------------------
  // Wire-format support (drain-on-publish)
  // -------------------------------------------------------------------

  std::vector<CoordT> drainTouchedTsdf() { return tsdf_.drainTouched(); }

  /// SPLIT substrate: per-grid touched sets (split publish). Beta is full-ray;
  /// Dir is hit-sparse.
  std::vector<CoordT> drainTouchedBeta() { return semsplit_.drainTouchedBeta(); }
  std::vector<CoordT> drainTouchedDir()  { return semsplit_.drainTouchedDir(); }

  void clearTouchedTsdf()   noexcept { tsdf_.clearTouched(); }
  void clearTouchedSemDir() noexcept { semsplit_.clearTouched(); }

  // -------------------------------------------------------------------
  // Memory / diagnostics
  // -------------------------------------------------------------------

  std::size_t tsdfVoxelCount()   const { return tsdf_.voxelCount();      }
  std::size_t tsdfGridBytes()    const { return tsdf_.gridMemoryBytes(); }

  /// Voxel count reports the Dir (semantics) grid; bytes report Beta + Dir
  /// combined, so existing memlog call sites stay meaningful.
  std::size_t semdirVoxelCount() const { return semsplit_.dirVoxelCount(); }
  std::size_t semdirGridBytes()  const {
    return semsplit_.betaGridMemoryBytes() + semsplit_.dirGridMemoryBytes();
  }

  // SPLIT per-grid accounting (for memlog / parity reporting).
  std::size_t betaVoxelCount() const { return semsplit_.betaVoxelCount(); }
  std::size_t dirVoxelCount()  const { return semsplit_.dirVoxelCount();  }
  std::size_t betaGridBytes()  const { return semsplit_.betaGridMemoryBytes(); }
  std::size_t dirGridBytes()   const { return semsplit_.dirGridMemoryBytes();  }

  // -------------------------------------------------------------------
  // Direct grid / substrate access
  // -------------------------------------------------------------------

  TsdfMap&         tsdf()         { return tsdf_;   }
  const TsdfMap&   tsdf()   const { return tsdf_;   }

  SemSplitMap&       semsplit()       { return semsplit_; }
  const SemSplitMap& semsplit() const { return semsplit_; }

  double resolution() const { return resolution_; }

 private:
  // -------------------------------------------------------------------
  // Fine-band internals
  // -------------------------------------------------------------------

  struct FineStagedHit {
    Eigen::Vector3f origin;
    Eigen::Vector3f endpoint;
  };

  void beginFineFrame() {
    if (!fine_tsdf_) return;
    fine_frame_open_ = true;
    if (fine_staged_.size() < fine_regions_.slotCount())
      fine_staged_.resize(fine_regions_.slotCount());
    for (auto& v : fine_staged_) v.clear();
  }

  /// Gate + stage one hit for the fine band. Inside an open scan frame the
  /// hit is buffered per region for the anchor-corrected flush; outside a
  /// frame (direct integrateHit callers, e.g. unit tests without the carve
  /// bracket) it fuses immediately, uncorrected — mirroring SemSplitMap's
  /// immediate-vs-batched carve semantics.
  void stageFineHit(const Eigen::Vector3f& origin,
                    const Eigen::Vector3f& endpoint) {
    const int idx =
        fine_regions_.lookup(endpoint.x(), endpoint.y(), endpoint.z());
    if (idx < 0) return;
    if (fine_frame_open_ && static_cast<std::size_t>(idx) < fine_staged_.size()) {
      fine_staged_[static_cast<std::size_t>(idx)].push_back({origin, endpoint});
    } else {
      fine_tsdf_->integrateRay(origin, endpoint);
    }
  }

  /// Per-region anchor fit + band fusion of all staged hits. The whole
  /// scan's in-region rays are translated by the fitted Δ (a rigid shift of
  /// the sensor pose in the horizontal plane) so the fusion input aligns
  /// with the region's canonical cylinder — drift is absorbed at the door,
  /// before the irreversible Curless–Levoy average. Fit failure → Δ = 0.
  std::size_t flushFineFrame() {
    fine_frame_open_ = false;
    if (!fine_tsdf_) return 0;
    fine_last_rays_  = 0;
    fine_last_shift_ = 0.f;
    for (std::size_t slot = 0; slot < fine_staged_.size(); ++slot) {
      auto& hits = fine_staged_[slot];
      if (hits.empty()) continue;
      if (!fine_regions_.slotActive(static_cast<int>(slot))) {
        hits.clear();  // region removed mid-frame
        continue;
      }
      Eigen::Vector3f shift = Eigen::Vector3f::Zero();
      if (fine_anchor_enable_) {
        const auto& cyl = fine_regions_.cylinder(static_cast<int>(slot));
        std::vector<Eigen::Vector2f> xy;
        xy.reserve(hits.size());
        for (const auto& hh : hits)
          xy.emplace_back(hh.endpoint.x(), hh.endpoint.y());
        // Align the scan TO the model: points sit at p + Δ, so the fitted
        // Δ is applied directly to the staged rays.
        const auto d = fitCylinderAnchorShift(
            xy, Eigen::Vector2f(cyl.cx, cyl.cy), cyl.radius,
            fine_anchor_params_);
        if (d) {
          shift.x() = d->x();
          shift.y() = d->y();
          fine_last_shift_ = std::max(fine_last_shift_, d->norm());
        } else {
          ++fine_anchor_fallbacks_;
        }
      }
      for (const auto& hh : hits)
        fine_tsdf_->integrateRay(hh.origin + shift, hh.endpoint + shift);
      fine_last_rays_ += hits.size();
      hits.clear();
    }
    return fine_last_rays_;
  }

  /// Per-voxel semantic carve dispatch (SPLIT substrate). `prof` carries the
  /// per-source w_free (null => global params_).
  bool semCarve(const CoordT& c, float quality, const HitWeights* prof = nullptr) {
    return semsplit_.applyCarveUpdate(c, quality, prof);
  }
  /// Per-voxel semantic hit dispatch (SPLIT substrate). `is_dynamic` routes the
  /// endpoint to the transient grids (see SemSplitMap::applyHitUpdate). `prof`
  /// carries the per-source w_occ/kappa0/min_p_occ (null => global params_).
  void semHit(const CoordT& c, const std::vector<float>* sem_probs, float quality,
              bool is_dynamic, const HitWeights* prof = nullptr) {
    semsplit_.applyHitUpdate(c, sem_probs, quality, is_dynamic, prof);
  }
  void semBand(const CoordT& c, const std::vector<float>* sem_probs, float quality,
               const HitWeights* prof = nullptr) {
    semsplit_.applyBandSemantic(c, sem_probs, quality, prof);
  }

  TsdfMap     tsdf_;       ///< TSDF surface (band-only)
  SemSplitMap semsplit_;   ///< Split Beta/Dir semantic substrate
  double      resolution_;
  bool        fused_walker_;
  bool        tsdf_enabled_;
  /// Cached from semsplit_.params() AFTER sanitise(), so a config that sets
  /// both band and spread reads 0 here and the walker's band branch stays cold.
  /// Declared after semsplit_ so the ctor's member-init order is valid.
  float       sem_band_;

  // Fine TSDF band state (null / empty when fine_ratio_log2 == 0).
  std::unique_ptr<TsdfMap>    fine_tsdf_;
  RefinementRegions           fine_regions_;
  std::vector<std::vector<FineStagedHit>> fine_staged_;  ///< per region slot
  uint8_t         fine_ratio_log2_;
  float           fine_region_margin_;
  bool            fine_anchor_enable_;
  AnchorFitParams fine_anchor_params_;
  bool            fine_frame_open_       = false;
  std::size_t     fine_last_rays_        = 0;
  float           fine_last_shift_       = 0.f;
  std::size_t     fine_anchor_fallbacks_ = 0;

  std::int64_t tsdf_ns_ = 0;
  std::int64_t sem_ns_  = 0;
};

}  // namespace scovox
