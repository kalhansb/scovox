/// @file
/// @brief Step-4 gate tests: TSDF-typed extractors + mesh_labelling free fns.

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <bonxai/bonxai.hpp>

#include "scovox/marching_cubes.hpp"
#include "scovox/mesh_labelling.hpp"
#include "scovox/sembeta_voxel.hpp"
#include "scovox/tsdf_voxel.hpp"

namespace {

constexpr double kRes = 0.05;

scovox::TsdfVoxel mkTsdf(float distance, float weight) {
  return {distance, weight};
}

scovox::SemBetaVoxel mkSem(uint16_t cls, float cnt) {
  scovox::SemBetaVoxel v = scovox::defaultSemBetaVoxel();
  v.sem_cls[0] = cls;
  v.sem_cnt[0] = cnt;
  return v;
}

}  // namespace

TEST(TsdfMarchingCubes, ExtractZeroCrossingFindsSurface) {
  // 2 voxels along +X, one positive, one negative → one zero-crossing.
  Bonxai::VoxelGrid<scovox::TsdfVoxel> grid(kRes, 2, 3);
  auto acc = grid.createAccessor();
  acc.setValue({0, 0, 0}, mkTsdf(+0.05f, 1.0f));
  acc.setValue({1, 0, 0}, mkTsdf(-0.05f, 1.0f));

  auto pts = scovox::extractZeroCrossing(grid, /*min_weight=*/0.5f, kRes);
  ASSERT_EQ(pts.size(), 1u);
  // Sentinel labels — labelling is decoupled.
  EXPECT_EQ(pts[0].semantic_class, uint16_t(0xFFFF));
}

TEST(TsdfMarchingCubes, ExtractPointCloudUsesCentre) {
  Bonxai::VoxelGrid<scovox::TsdfVoxel> grid(kRes, 2, 3);
  auto acc = grid.createAccessor();
  acc.setValue({0, 0, 0}, mkTsdf(0.0f, 2.0f));

  auto positions = scovox::extractPointCloud(grid, /*min_weight=*/1.0f, kRes);
  ASSERT_EQ(positions.size(), 1u);
  // Voxel (0,0,0): lower corner = (0,0,0); centre = (h, h, h) with h=0.025.
  EXPECT_NEAR(positions[0].x(), 0.025f, 1e-5f);
  EXPECT_NEAR(positions[0].y(), 0.025f, 1e-5f);
  EXPECT_NEAR(positions[0].z(), 0.025f, 1e-5f);
}

TEST(TsdfMarchingCubes, MinWeightFilters) {
  Bonxai::VoxelGrid<scovox::TsdfVoxel> grid(kRes, 2, 3);
  auto acc = grid.createAccessor();
  acc.setValue({0, 0, 0}, mkTsdf(0.0f, 0.3f));   // below threshold
  acc.setValue({1, 0, 0}, mkTsdf(0.0f, 5.0f));   // above

  auto positions = scovox::extractPointCloud(grid, /*min_weight=*/1.0f, kRes);
  EXPECT_EQ(positions.size(), 1u);
}

TEST(LabelMesh, AnchorLookupReturnsArgmaxClass) {
  // Place the surface at ~(0.025, 0.025, 0.025) and a SemBeta voxel
  // whose argmax is class 7 at the same anchor coord.
  Bonxai::VoxelGrid<scovox::TsdfVoxel> tsdf(kRes, 2, 3);
  Bonxai::VoxelGrid<scovox::SemBetaVoxel> sb(kRes, 2, 3);
  {
    auto a_t = tsdf.createAccessor();
    auto a_s = sb.createAccessor();
    a_t.setValue({0, 0, 0}, mkTsdf(+0.02f, 2.0f));
    a_t.setValue({1, 0, 0}, mkTsdf(-0.02f, 2.0f));
    a_t.setValue({0, 1, 0}, mkTsdf(+0.02f, 2.0f));
    a_t.setValue({1, 1, 0}, mkTsdf(-0.02f, 2.0f));
    a_t.setValue({0, 0, 1}, mkTsdf(+0.02f, 2.0f));
    a_t.setValue({1, 0, 1}, mkTsdf(-0.02f, 2.0f));
    a_t.setValue({0, 1, 1}, mkTsdf(+0.02f, 2.0f));
    a_t.setValue({1, 1, 1}, mkTsdf(-0.02f, 2.0f));

    // Populate SemBeta on every cube corner so the centroid lookup always
    // lands in a voxel with class 7.
    for (int dx = 0; dx <= 1; ++dx)
    for (int dy = 0; dy <= 1; ++dy)
    for (int dz = 0; dz <= 1; ++dz) {
      a_s.setValue({dx, dy, dz}, mkSem(/*cls=*/7, /*cnt=*/3.0f));
    }
  }

  auto geom = scovox::extractMesh(tsdf, /*min_weight=*/1.0f, kRes);
  ASSERT_GT(geom.triangles.size(), 0u);
  auto labels = scovox::labelMesh(geom, tsdf, sb);
  ASSERT_EQ(labels.size(), geom.triangles.size());
  // Every triangle's centroid should land in a populated SemBeta voxel
  // since we filled all 8 corner coords.
  for (auto l : labels) {
    EXPECT_EQ(l, uint16_t(7)) << "triangle label should be class 7";
  }
}

TEST(LabelPointCloud, MissReturnsSentinel) {
  Bonxai::VoxelGrid<scovox::SemBetaVoxel> sb(kRes, 2, 3);
  // Empty SemBeta grid.
  std::vector<Eigen::Vector3f> positions{
      Eigen::Vector3f(1.0f, 1.0f, 1.0f),
      Eigen::Vector3f(2.0f, 2.0f, 2.0f)};
  auto labels = scovox::labelPointCloud(positions, sb);
  ASSERT_EQ(labels.size(), 2u);
  EXPECT_EQ(labels[0], uint16_t(0xFFFF));
  EXPECT_EQ(labels[1], uint16_t(0xFFFF));
}
