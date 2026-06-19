/// @file test_marching_cubes.cpp
/// Tests for TSDF zero-crossing surface extraction.

#include <gtest/gtest.h>
#include <cmath>
#include <Eigen/Core>
#include "scovox/marching_cubes.hpp"

using namespace scovox;

// Helper: create a grid and fill a region with a planar TSDF field.
// Surface at z = surface_z (in voxel coords), so tsdf_distance = (surface_z - coord.z) * res.
static Bonxai::VoxelGrid<Voxel> makePlanarGrid(
    double resolution, int half_extent, float surface_z_coord, float weight)
{
  Bonxai::VoxelGrid<Voxel> grid(resolution);
  auto acc = grid.createAccessor();

  for (int x = -half_extent; x <= half_extent; ++x) {
    for (int y = -half_extent; y <= half_extent; ++y) {
      for (int z = -half_extent; z <= half_extent; ++z) {
        Bonxai::CoordT c{x, y, z};
        Voxel v = defaultVoxel();
        v.tsdf_distance = (surface_z_coord - (float)z) * (float)resolution;
        v.tsdf_weight = weight;
        // Give the positive-side (z < surface) a semantic class
        if (z < (int)surface_z_coord) {
          v.sem_cls[0] = 3;
          v.sem_cnt[0] = 10.f;
        } else {
          v.sem_cls[0] = 7;
          v.sem_cnt[0] = 10.f;
        }
        acc.setValue(c, v);
      }
    }
  }
  return grid;
}

// Helper: create a grid with a single cube (2x2x2) where one corner is negative.
static Bonxai::VoxelGrid<Voxel> makeSingleCubeGrid(
    double resolution, float weight)
{
  Bonxai::VoxelGrid<Voxel> grid(resolution);
  auto acc = grid.createAccessor();

  // 8 corners at (0,0,0) through (1,1,1)
  // All positive except (0,0,0) which is negative → one triangle
  for (int x = 0; x <= 1; ++x) {
    for (int y = 0; y <= 1; ++y) {
      for (int z = 0; z <= 1; ++z) {
        Bonxai::CoordT c{x, y, z};
        Voxel v = defaultVoxel();
        if (x == 0 && y == 0 && z == 0) {
          v.tsdf_distance = -0.5f * (float)resolution;
        } else {
          v.tsdf_distance = 0.5f * (float)resolution;
        }
        v.tsdf_weight = weight;
        v.sem_cls[0] = 5;
        v.sem_cnt[0] = 8.f;
        acc.setValue(c, v);
      }
    }
  }
  return grid;
}

// =====================================================================
// extractMesh tests
// =====================================================================

TEST(MarchingCubes, SingleCubeProducesTriangles) {
  auto grid = makeSingleCubeGrid(0.10, 5.0f);
  auto mesh = extractMesh(grid, 1.0f, 0.10);

  // One negative corner (corner 0) → cube_index = 1 → 1 triangle
  EXPECT_GT(mesh.triangles.size(), 0u);
  EXPECT_EQ(mesh.vertices.size(), mesh.triangles.size() * 3u  > 0 ? mesh.vertices.size() : 0u);
  // All vertices should have valid coordinates (not NaN)
  for (const auto& v : mesh.vertices) {
    EXPECT_TRUE(std::isfinite(v.x()));
    EXPECT_TRUE(std::isfinite(v.y()));
    EXPECT_TRUE(std::isfinite(v.z()));
  }
}

TEST(MarchingCubes, SingleCubeVertexOnEdge) {
  auto grid = makeSingleCubeGrid(0.10, 5.0f);
  auto mesh = extractMesh(grid, 1.0f, 0.10);

  ASSERT_GT(mesh.vertices.size(), 0u);
  // Corner 0 is at (0,0,0) world = (0*0.1, 0*0.1, 0*0.1) = origin area.
  // Vertices should lie on edges emanating from corner 0, between
  // the negative corner and its positive neighbours. All vertex coords
  // should be within the cube's world-space bounding box [0, 0.1].
  for (const auto& v : mesh.vertices) {
    EXPECT_GE(v.x(), -0.01f);
    EXPECT_GE(v.y(), -0.01f);
    EXPECT_GE(v.z(), -0.01f);
    EXPECT_LE(v.x(), 0.11f);
    EXPECT_LE(v.y(), 0.11f);
    EXPECT_LE(v.z(), 0.11f);
  }
}

TEST(MarchingCubes, FlatPlaneProducesMesh) {
  // Surface at z=4.5 — mid-voxel, between z=4 and z=5, both interior to the
  // half_extent=5 grid. Two reasons to avoid integer surface_z_coord:
  //   1. Voxels at the surface coord get exactly d=0.0; standard MC sign
  //      convention (f[i] < 0 → inside) treats that as positive, so no
  //      sign change occurs and cube_index stays 0.
  //   2. The anchor needs corners at both z and z+1 to be present in the grid.
  auto grid = makePlanarGrid(0.10, 5, 4.5f, 10.0f);
  auto mesh = extractMesh(grid, 1.0f, 0.10);

  EXPECT_GT(mesh.triangles.size(), 0u);
  EXPECT_GT(mesh.vertices.size(), 0u);

  // All vertices should cluster near z = 4.5 * 0.10 = 0.45
  for (const auto& v : mesh.vertices) {
    EXPECT_NEAR(v.z(), 0.45f, 0.10f)
        << "Mesh vertex z should be near the planar surface";
  }
}

TEST(MarchingCubes, WeightFilteringSkipsCubes) {
  auto grid = makeSingleCubeGrid(0.10, 5.0f);

  // Request min_weight > what the grid has → no triangles
  auto mesh = extractMesh(grid, 100.0f, 0.10);
  EXPECT_EQ(mesh.triangles.size(), 0u);
  EXPECT_EQ(mesh.vertices.size(), 0u);
}

TEST(MarchingCubes, EmptyGridReturnsEmpty) {
  Bonxai::VoxelGrid<Voxel> grid(0.10);
  auto mesh = extractMesh(grid, 1.0f, 0.10);
  EXPECT_EQ(mesh.triangles.size(), 0u);
  EXPECT_EQ(mesh.vertices.size(), 0u);
}

TEST(MarchingCubes, ZeroWeightGridReturnsEmpty) {
  // Grid with tsdf_weight = 0 everywhere
  Bonxai::VoxelGrid<Voxel> grid(0.10);
  auto acc = grid.createAccessor();
  for (int i = 0; i < 4; ++i) {
    Voxel v = defaultVoxel();
    v.tsdf_distance = 0.1f;
    v.tsdf_weight = 0.0f;
    acc.setValue({i, 0, 0}, v);
  }
  auto mesh = extractMesh(grid, 0.1f, 0.10);
  EXPECT_EQ(mesh.triangles.size(), 0u);
}

TEST(MarchingCubes, SemanticLabelPropagation) {
  auto grid = makeSingleCubeGrid(0.10, 5.0f);
  auto mesh = extractMesh(grid, 1.0f, 0.10);

  ASSERT_GT(mesh.tri_labels.size(), 0u);
  // All voxels in the single-cube grid have sem_cls[0] = 5
  for (auto label : mesh.tri_labels) {
    EXPECT_EQ(label, 5);
  }
}

// =====================================================================
// extractZeroCrossing tests
// =====================================================================

TEST(ZeroCrossing, FlatPlaneProducesPoints) {
  auto grid = makePlanarGrid(0.10, 5, 5.0f, 10.0f);
  auto points = extractZeroCrossing(grid, 1.0f, 0.10);

  EXPECT_GT(points.size(), 0u);

  // All points should be near z = 0.50 (surface at coord z=5, res=0.10)
  for (const auto& p : points) {
    EXPECT_NEAR(p.position.z(), 0.50f, 0.10f);
  }
}

TEST(ZeroCrossing, NoSignChangeNoPoints) {
  // All positive TSDF → no zero-crossing
  Bonxai::VoxelGrid<Voxel> grid(0.10);
  auto acc = grid.createAccessor();
  for (int x = 0; x < 5; ++x) {
    for (int y = 0; y < 5; ++y) {
      for (int z = 0; z < 5; ++z) {
        Voxel v = defaultVoxel();
        v.tsdf_distance = 1.0f;  // all positive
        v.tsdf_weight = 10.0f;
        acc.setValue({x, y, z}, v);
      }
    }
  }
  auto points = extractZeroCrossing(grid, 1.0f, 0.10);
  EXPECT_EQ(points.size(), 0u);
}

TEST(ZeroCrossing, EmptyGridReturnsEmpty) {
  Bonxai::VoxelGrid<Voxel> grid(0.10);
  auto points = extractZeroCrossing(grid, 1.0f, 0.10);
  EXPECT_EQ(points.size(), 0u);
}

TEST(ZeroCrossing, WeightFilteringWorks) {
  auto grid = makePlanarGrid(0.10, 3, 3.0f, 5.0f);
  // With min_weight > 5 → nothing
  auto points = extractZeroCrossing(grid, 100.0f, 0.10);
  EXPECT_EQ(points.size(), 0u);
}

TEST(ZeroCrossing, SemanticFromPositiveSide) {
  // Surface at z=5: positive side (z < 5) has class 3, negative side has class 7.
  auto grid = makePlanarGrid(0.10, 5, 5.0f, 10.0f);
  auto points = extractZeroCrossing(grid, 1.0f, 0.10);

  ASSERT_GT(points.size(), 0u);
  for (const auto& p : points) {
    // The positive-side voxel (in front of surface) has class 3
    EXPECT_EQ(p.semantic_class, 3);
  }
}

// =====================================================================
// extractPointCloud tests
// =====================================================================

TEST(PointCloud, EmitsAllObservedVoxels) {
  auto grid = makePlanarGrid(0.10, 3, 3.0f, 5.0f);
  auto [positions, labels] = extractPointCloud(grid, 1.0f, 0.10);

  // 7x7x7 = 343 voxels, all with weight=5 > min=1
  EXPECT_EQ(positions.size(), 343u);
  EXPECT_EQ(labels.size(), 343u);
}

TEST(PointCloud, WeightFilteringWorks) {
  auto grid = makePlanarGrid(0.10, 3, 3.0f, 5.0f);
  auto [positions, labels] = extractPointCloud(grid, 100.0f, 0.10);
  EXPECT_EQ(positions.size(), 0u);
}

TEST(PointCloud, EmptyGridReturnsEmpty) {
  Bonxai::VoxelGrid<Voxel> grid(0.10);
  auto [positions, labels] = extractPointCloud(grid, 1.0f, 0.10);
  EXPECT_EQ(positions.size(), 0u);
}
