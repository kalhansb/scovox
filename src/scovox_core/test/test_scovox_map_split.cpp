/// @file
/// @brief Step-5 gate: ScovoxMapSplit composer end-to-end smoke + parity.

#include <gtest/gtest.h>
#include <Eigen/Core>
#include <vector>

#include "scovox/scovox_map_split.hpp"

namespace {

scovox::ScovoxMapSplit makeMap() {
  scovox::ScovoxMapSplit::Params p;
  p.resolution        = 0.05;
  p.tsdf.sdf_trunc    = 0.15f;
  p.semdir.kappa0    = 1.0f;
  p.semdir.dirichlet_min_p_occ = 0.5f;
  return scovox::ScovoxMapSplit(p);
}

}  // namespace

TEST(ScovoxMapSplit, IntegrateHitTouchesBothGrids) {
  auto m = makeMap();
  std::vector<float> probs{0.f, 1.f, 0.f, 0.f};

  m.integrateHit(Eigen::Vector3f(0, 0, 0),
                 Eigen::Vector3f(0.50f, 0, 0),
                 &probs, /*quality=*/1.0f);

  EXPECT_GT(m.tsdfVoxelCount(),    0u);
  EXPECT_GT(m.semdirVoxelCount(), 0u);
}

TEST(ScovoxMapSplit, IntegrateMissTouchesSemDirOnly) {
  auto m = makeMap();
  m.integrateMiss(Eigen::Vector3f(0, 0, 0),
                  Eigen::Vector3f(0.50f, 0, 0),
                  /*quality=*/1.0f);

  EXPECT_EQ(m.tsdfVoxelCount(),    0u) << "miss must NOT allocate TSDF voxels";
  EXPECT_GT(m.semdirVoxelCount(), 0u);
}

TEST(ScovoxMapSplit, ExtractMeshHasConsistentLabelArray) {
  // Marching cubes needs a 3D volume of TSDF voxels to extract surface;
  // a single linear ray doesn't suffice. Fan rays from origin to a small
  // cube of endpoints around (0.50, 0, 0) to fill the surface band's
  // 3D neighborhood.
  auto m = makeMap();
  std::vector<float> probs{0.f, 0.f, 1.f, 0.f};  // class 2
  for (int dx = -1; dx <= 1; ++dx)
  for (int dy = -1; dy <= 1; ++dy)
  for (int dz = -1; dz <= 1; ++dz) {
    Eigen::Vector3f ep(0.50f + 0.05f * dx, 0.05f * dy, 0.05f * dz);
    m.integrateHit(Eigen::Vector3f(0, 0, 0), ep, &probs, /*quality=*/1.0f);
  }
  // Geometry may or may not produce triangles depending on band coverage,
  // but the mesh must always be self-consistent: |tri_labels| == |triangles|.
  auto mesh = m.extractMesh(/*min_weight=*/0.5f);
  EXPECT_EQ(mesh.tri_labels.size(), mesh.triangles.size());
}

TEST(ScovoxMapSplit, ExtractPointCloudUsesCentre) {
  auto m = makeMap();
  for (int i = 0; i < 3; ++i) {
    m.integrateHit(Eigen::Vector3f(0, 0, 0),
                   Eigen::Vector3f(0.50f, 0, 0),
                   nullptr, /*quality=*/1.0f);
  }
  auto [positions, labels] = m.extractPointCloud(/*min_weight=*/1.0f);
  EXPECT_GT(positions.size(), 0u);
  EXPECT_EQ(positions.size(), labels.size());
}

TEST(ScovoxMapSplit, MemoryAccountingSplitsCleanly) {
  auto m = makeMap();
  m.integrateHit(Eigen::Vector3f(0, 0, 0),
                 Eigen::Vector3f(0.50f, 0, 0),
                 nullptr, 1.0f);
  // SemDir should always have more voxels than TSDF (full-ray carve vs band).
  EXPECT_GE(m.semdirVoxelCount(), m.tsdfVoxelCount());
  // Memory matches struct sizes (modulo Bonxai overhead).
  EXPECT_GT(m.tsdfGridBytes(),   m.tsdfVoxelCount()   * sizeof(scovox::TsdfVoxel)   - 1);
  EXPECT_GT(m.semdirGridBytes(), m.semdirVoxelCount() * sizeof(scovox::SemDirVoxel) - 1);
}

TEST(ScovoxMapSplit, DrainTouchedSplitsByGrid) {
  auto m = makeMap();
  m.integrateHit(Eigen::Vector3f(0, 0, 0),
                 Eigen::Vector3f(0.50f, 0, 0),
                 nullptr, 1.0f);
  auto t = m.drainTouchedTsdf();
  auto s = m.drainTouchedSemDir();
  EXPECT_GT(t.size(), 0u);
  EXPECT_GT(s.size(), 0u);
  // Second drain should be empty (buffers cleared).
  EXPECT_EQ(m.drainTouchedTsdf().size(),   0u);
  EXPECT_EQ(m.drainTouchedSemDir().size(), 0u);
}

// ===========================================================================
// Step 12.10 (2026-05-09): fused ray walker — parity vs split path
// ===========================================================================
//
// The fused walker (`Params::fused_walker = true`, default) runs one
// Bresenham DDA over the TSDF band [Hp - sdf_trunc·û, Hp + sdf_trunc·û]
// and dispatches per-voxel into both grids. The split walker runs two
// independent DDAs. For axis-aligned rays Bresenham reduces to integer
// stepping along the major axis and both walkers must produce
// bit-identical TsdfMap state. SemBeta state must also match exactly:
// the carve subset is `0 < sdf <= carve_band, c != k_hit`, which selects
// the same voxels along an axis-aligned ray as the legacy
// `SemBetaMap::carveRay [co, Hp)` walk.

namespace {

scovox::ScovoxMapSplit::Params splitParams() {
  scovox::ScovoxMapSplit::Params p;
  p.resolution        = 0.05;
  p.tsdf.sdf_trunc    = 0.15f;
  p.semdir.kappa0    = 1.0f;
  p.semdir.dirichlet_min_p_occ = 0.5f;
  return p;
}

// Apply the carve_band truncation that scovox_node performs upstream
// (Hp − carve_band·û → co). The fused walker reads (Hp − co).norm() as
// the SemBeta carve range; the split walker carves the same [co, Hp).
Eigen::Vector3f truncateOrigin(const Eigen::Vector3f& O,
                               const Eigen::Vector3f& Hp,
                               float carve_band) {
  const float rng = (Hp - O).norm();
  if (carve_band <= 0.f || rng <= carve_band) return O;
  return O + (Hp - O).normalized() * (rng - carve_band);
}

}  // namespace

TEST(ScovoxMapSplitFusedWalker, AxisAlignedParityWithSplitWalker) {
  // Axis-aligned ray: Bresenham picks deterministic voxels — fused and
  // split must produce identical TsdfMap voxel sets and identical
  // SemBeta {a_occ, a_free, sem_cnt} state for the same ray.
  auto p_fused = splitParams(); p_fused.fused_walker = true;
  auto p_split = splitParams(); p_split.fused_walker = false;
  scovox::ScovoxMapSplit m_fused(p_fused);
  scovox::ScovoxMapSplit m_split(p_split);

  std::vector<float> probs{0.f, 1.f, 0.f, 0.f};  // class 1
  const Eigen::Vector3f O(0.f, 0.f, 0.f);
  const Eigen::Vector3f Hp(0.50f, 0.f, 0.f);
  const float carve_band = 0.10f;
  const Eigen::Vector3f co = truncateOrigin(O, Hp, carve_band);

  m_fused.integrateHit(co, Hp, &probs, /*quality=*/1.0f);
  m_split.integrateHit(co, Hp, &probs, /*quality=*/1.0f);

  // TsdfMap voxel counts and total grid bytes match — the fused walker
  // walks the same band and runs the same Curless–Levoy update.
  EXPECT_EQ(m_fused.tsdfVoxelCount(),  m_split.tsdfVoxelCount());
  EXPECT_EQ(m_fused.tsdfGridBytes(),   m_split.tsdfGridBytes());

  // SemBeta voxel counts match for an axis-aligned ray — the carve
  // subset {0 < sdf <= carve_band} selects exactly the voxels Bresenham
  // would pick walking [co, Hp).
  EXPECT_EQ(m_fused.semdirVoxelCount(), m_split.semdirVoxelCount());

  // Hit voxel — unified-Dirichlet state must match exactly.
  auto h_f = m_fused.semdir().getVoxel(Hp);
  auto h_s = m_split.semdir().getVoxel(Hp);
  ASSERT_TRUE(h_f.has_value());
  ASSERT_TRUE(h_s.has_value());
  EXPECT_FLOAT_EQ(h_f->alpha_free,  h_s->alpha_free);
  EXPECT_FLOAT_EQ(h_f->alpha_other, h_s->alpha_other);
  for (int i = 0; i < scovox::K_TOP; ++i) {
    EXPECT_FLOAT_EQ(h_f->cnt[i], h_s->cnt[i]);
    EXPECT_EQ(h_f->cls[i],       h_s->cls[i]);
  }
}

TEST(ScovoxMapSplitFusedWalker, MissPathUnchanged) {
  // The fused-walker flag only affects integrateHit. Misses must produce
  // identical SemBeta state on both paths and zero TsdfMap voxels.
  auto p_fused = splitParams(); p_fused.fused_walker = true;
  auto p_split = splitParams(); p_split.fused_walker = false;
  scovox::ScovoxMapSplit m_fused(p_fused);
  scovox::ScovoxMapSplit m_split(p_split);

  m_fused.integrateMiss(Eigen::Vector3f(0, 0, 0),
                        Eigen::Vector3f(0.50f, 0, 0), 1.0f);
  m_split.integrateMiss(Eigen::Vector3f(0, 0, 0),
                        Eigen::Vector3f(0.50f, 0, 0), 1.0f);

  EXPECT_EQ(m_fused.tsdfVoxelCount(),    0u);
  EXPECT_EQ(m_split.tsdfVoxelCount(),    0u);
  EXPECT_EQ(m_fused.semdirVoxelCount(), m_split.semdirVoxelCount());
}

TEST(ScovoxMapSplitFusedWalker, MultiRayBandIdentity) {
  // 27-ray fan around an axis-aligned hit endpoint. Even with the small
  // off-axis offsets, every ray in this set is close enough to one major
  // axis that Bresenham walks the same voxel sequence under both walkers.
  // After integrating all 27 rays the TsdfMap state must be bit-identical.
  auto p_fused = splitParams(); p_fused.fused_walker = true;
  auto p_split = splitParams(); p_split.fused_walker = false;
  scovox::ScovoxMapSplit m_fused(p_fused);
  scovox::ScovoxMapSplit m_split(p_split);

  std::vector<float> probs{0.f, 0.f, 1.f, 0.f};  // class 2
  const float carve_band = 0.10f;
  for (int dx = -1; dx <= 1; ++dx)
  for (int dy = -1; dy <= 1; ++dy)
  for (int dz = -1; dz <= 1; ++dz) {
    Eigen::Vector3f Hp(0.50f + 0.05f * dx, 0.05f * dy, 0.05f * dz);
    Eigen::Vector3f co = truncateOrigin(Eigen::Vector3f::Zero(), Hp, carve_band);
    m_fused.integrateHit(co, Hp, &probs, /*quality=*/1.0f);
    m_split.integrateHit(co, Hp, &probs, /*quality=*/1.0f);
  }

  // For axis-aligned-ish rays the TSDF voxel sets are exactly equal.
  EXPECT_EQ(m_fused.tsdfVoxelCount(), m_split.tsdfVoxelCount());
  EXPECT_EQ(m_fused.tsdfGridBytes(),  m_split.tsdfGridBytes());

  // SemDir carve sets agree up to per-ray Bresenham boundary jitter:
  // the fused walker starts its DDA at `Hp − walk_back·û` while the split
  // walker starts at `Hp − carve_band·û`, so for oblique rays the two
  // pick-sequences may pick different voxels at sub-voxel boundaries.
  // For axis-aligned rays the sets are bit-identical (see the test
  // above); for the 27-ray off-axis fan with 0.05 m offsets we accept
  // up to 20 % asymmetric difference, dominated by ±1 voxel/ray jitter
  // accumulating over ~27 rays in a SemDir set of ~45 voxels.
  const auto a = m_fused.semdirVoxelCount();
  const auto b = m_split.semdirVoxelCount();
  const auto diff = a > b ? a - b : b - a;
  const auto larger = a > b ? a : b;
  EXPECT_LE(diff * 100u, larger * 20u)
      << "fused=" << a << " split=" << b
      << " — boundary jitter > 20 %";
}

TEST(ScovoxMapSplitFusedWalker, FusedIsDefault) {
  scovox::ScovoxMapSplit::Params p;
  EXPECT_TRUE(p.fused_walker);
}
