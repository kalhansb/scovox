#pragma once

/// @file mesh_labelling.hpp
/// @brief Per-triangle / per-point semantic labelling for split-grid SCovox.
///
/// Companion to the TsdfVoxel-typed extractors in `marching_cubes.hpp`.
/// After mesh extraction returns geometry only, these free functions look
/// up the SemBeta grid by world position and add labels:
///
///   labelMesh(geom, tsdf_grid, sembeta_grid)         → per-triangle labels
///   labelPointCloud(positions, sembeta_grid)         → per-point labels
///
/// Cross-grid absence policy (Q5 of the design plan): a missing SemBeta
/// voxel returns the sentinel class id `0xFFFF` ("unknown"). This can
/// happen legitimately (Dirichlet gated by `dirichlet_min_p_occ`, or
/// SLIM-VDB-only mode with no SemBeta grid) and consumers must filter
/// for it.

#include <cstdint>
#include <vector>

#include <Eigen/Core>
#include <bonxai/bonxai.hpp>

#include "scovox/sembeta_voxel.hpp"
#include "scovox/semdir_voxel.hpp"
#include "scovox/tsdf_voxel.hpp"
#include "scovox/marching_cubes.hpp"  // TriangleMesh, mc_tables::corners

namespace scovox {

namespace detail {

/// Argmax of SemBeta sparse-Dirichlet slots. Returns 0xFFFF if no slot
/// has positive count (= voxel allocated but Dirichlet never updated).
inline uint16_t dominantClass(const SemBetaVoxel& v) {
  uint16_t cls = 0xFFFF;
  float best = 0.f;
  for (int i = 0; i < K_TOP; ++i) {
    if (v.sem_cnt[i] > best) {
      best = v.sem_cnt[i];
      cls  = v.sem_cls[i];
    }
  }
  return cls;
}

/// Argmax of SemDir top-K class slots. Returns 0xFFFF if no slot is
/// filled (cls == 0xFFFF everywhere) OR if every slot's observed-evidence
/// (cnt − α_0) is dominated by `alpha_other` — i.e. the bulk of occupied
/// mass is on classes outside the top-K, so committing to any of the
/// tracked classes would be misleading. The `alpha_0` argument is the
/// per-dim prior used to subtract the slot's placeholder mass; pass
/// `kDefaultDirichletPrior` (0.01) at call sites where the map's α_0
/// isn't conveniently in scope — the small additive constant doesn't
/// change argmax outcomes meaningfully at typical evidence levels.
inline uint16_t dominantClass(const SemDirVoxel& v,
                              float alpha_0 = kDefaultDirichletPrior) {
  uint16_t cls = 0xFFFF;
  float best_evidence = 0.f;
  for (int i = 0; i < K_TOP; ++i) {
    if (v.cls[i] == 0xFFFF) continue;
    const float evidence = v.cnt[i] - alpha_0;
    if (evidence > best_evidence) {
      best_evidence = evidence;
      cls           = v.cls[i];
    }
  }
  // OTHER dominates the top-K: refuse to commit to a tracked class.
  // This is the new knob the unified Dirichlet exposes that SemBeta
  // could not — "I have lots of occupied evidence but it's spread across
  // classes that aren't in my top-K" is now a queryable state.
  if (v.alpha_other > best_evidence) return 0xFFFF;
  return cls;
}

}  // namespace detail

/// Per-triangle labels. The anchor voxel is the cube's positive-side
/// (in-front-of-surface) corner, identified via TSDF sign in `tsdf_grid`.
/// Returns a vector aligned with `geom.triangles` (one label per triangle).
///
/// The implementation walks each triangle, recovers the anchor voxel coord
/// by mapping the triangle's centroid back to the TSDF grid, then queries
/// SemBeta. This is approximate (a triangle's centroid doesn't always land
/// exactly in the anchor voxel) but matches what `Map::extractMesh` already
/// does for the legacy fused Voxel grid.
inline std::vector<uint16_t> labelMesh(
    const TriangleMesh&                      geom,
    const Bonxai::VoxelGrid<TsdfVoxel>&      tsdf_grid,
    const Bonxai::VoxelGrid<SemBetaVoxel>&   sembeta_grid)
{
  std::vector<uint16_t> labels;
  labels.reserve(geom.triangles.size());

  auto sembeta_acc = sembeta_grid.createConstAccessor();
  auto tsdf_acc    = tsdf_grid.createConstAccessor();
  (void)tsdf_acc;  // reserved for future TSDF-sign disambiguation

  for (const auto& tri : geom.triangles) {
    if (tri[0] >= (int)geom.vertices.size() ||
        tri[1] >= (int)geom.vertices.size() ||
        tri[2] >= (int)geom.vertices.size()) {
      labels.push_back(0xFFFF);
      continue;
    }
    const Eigen::Vector3f centroid =
        (geom.vertices[tri[0]] + geom.vertices[tri[1]] + geom.vertices[tri[2]]) / 3.0f;
    const auto coord = sembeta_grid.posToCoord(
        centroid.x(), centroid.y(), centroid.z());
    const SemBetaVoxel* v = sembeta_acc.value(coord);
    labels.push_back(v ? detail::dominantClass(*v) : uint16_t(0xFFFF));
  }
  return labels;
}

/// Per-point labels. For each world-space position, query the SemBeta
/// grid and emit the argmax class. Sentinel `0xFFFF` on miss.
inline std::vector<uint16_t> labelPointCloud(
    const std::vector<Eigen::Vector3f>&     positions,
    const Bonxai::VoxelGrid<SemBetaVoxel>&  sembeta_grid)
{
  std::vector<uint16_t> labels;
  labels.reserve(positions.size());
  auto acc = sembeta_grid.createConstAccessor();
  for (const auto& p : positions) {
    const auto coord = sembeta_grid.posToCoord(p.x(), p.y(), p.z());
    const SemBetaVoxel* v = acc.value(coord);
    labels.push_back(v ? detail::dominantClass(*v) : uint16_t(0xFFFF));
  }
  return labels;
}

// ---------------------------------------------------------------------------
// SemDirVoxel overloads (unified Dirichlet, Step 7.5)
// ---------------------------------------------------------------------------

/// Per-triangle labels against a SemDir grid. Same anchor-via-centroid
/// strategy as the SemBeta overload; argmax now refuses to commit when
/// `alpha_other` exceeds every top-K slot's observed evidence (the new
/// "out-of-K dominant" sentinel).
inline std::vector<uint16_t> labelMesh(
    const TriangleMesh&                     geom,
    const Bonxai::VoxelGrid<TsdfVoxel>&     tsdf_grid,
    const Bonxai::VoxelGrid<SemDirVoxel>&   semdir_grid,
    float                                   alpha_0 = kDefaultDirichletPrior)
{
  std::vector<uint16_t> labels;
  labels.reserve(geom.triangles.size());

  auto semdir_acc = semdir_grid.createConstAccessor();
  auto tsdf_acc   = tsdf_grid.createConstAccessor();
  (void)tsdf_acc;  // reserved for future TSDF-sign disambiguation

  for (const auto& tri : geom.triangles) {
    if (tri[0] >= (int)geom.vertices.size() ||
        tri[1] >= (int)geom.vertices.size() ||
        tri[2] >= (int)geom.vertices.size()) {
      labels.push_back(0xFFFF);
      continue;
    }
    const Eigen::Vector3f centroid =
        (geom.vertices[tri[0]] + geom.vertices[tri[1]] + geom.vertices[tri[2]]) / 3.0f;
    const auto coord = semdir_grid.posToCoord(
        centroid.x(), centroid.y(), centroid.z());
    const SemDirVoxel* v = semdir_acc.value(coord);
    labels.push_back(v ? detail::dominantClass(*v, alpha_0) : uint16_t(0xFFFF));
  }
  return labels;
}

/// Per-point labels against a SemDir grid.
inline std::vector<uint16_t> labelPointCloud(
    const std::vector<Eigen::Vector3f>&     positions,
    const Bonxai::VoxelGrid<SemDirVoxel>&   semdir_grid,
    float                                   alpha_0 = kDefaultDirichletPrior)
{
  std::vector<uint16_t> labels;
  labels.reserve(positions.size());
  auto acc = semdir_grid.createConstAccessor();
  for (const auto& p : positions) {
    const auto coord = semdir_grid.posToCoord(p.x(), p.y(), p.z());
    const SemDirVoxel* v = acc.value(coord);
    labels.push_back(v ? detail::dominantClass(*v, alpha_0) : uint16_t(0xFFFF));
  }
  return labels;
}

}  // namespace scovox
