#pragma once
/// @file marching_cubes.hpp
/// @brief TSDF zero-crossing surface extraction — header-only, zero ROS deps.
///
/// Provides two extractors operating on Bonxai::VoxelGrid<scovox::Voxel>:
///
///   extractMesh()          — full marching cubes → triangle mesh + per-tri labels
///   extractZeroCrossing()  — lightweight 3-axis neighbour sign-change → point cloud
///
/// The marching cubes lookup tables originate from Paul Bourke's public-domain
/// specification (same tables used by Open3D and SLIM-VDB/VDBFusion).
///
/// Part of scovox_core.  No ROS, no OpenVDB, no GPU dependencies.

#include <vector>
#include <array>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <unordered_map>
#include <Eigen/Core>
#include <bonxai/bonxai.hpp>
#include "scovox/voxel.hpp"
#include "scovox/tsdf_voxel.hpp"  // split-grid refactor: TSDF-only voxel

namespace scovox {

// =====================================================================
// Output types
// =====================================================================

struct TriangleMesh {
  std::vector<Eigen::Vector3f> vertices;
  std::vector<Eigen::Vector3i> triangles;
  std::vector<uint16_t> tri_labels;   ///< dominant semantic class per triangle
};

struct SurfacePoint {
  Eigen::Vector3f position;
  float distance;                     ///< interpolated SDF at the crossing
  uint16_t semantic_class;            ///< dominant class from the occupied side
};

// =====================================================================
// Marching cubes lookup tables  (public domain — Paul Bourke)
// =====================================================================

namespace mc_tables {

// Corner offsets for the 8 vertices of a marching-cubes cube.
// Indexed as: corner i → (dx, dy, dz) offset from the anchor voxel coord.
inline const Bonxai::CoordT corners[8] = {
    {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0},
    {0,0,1}, {1,0,1}, {1,1,1}, {0,1,1},
};

// edge_to_vert[e] = {corner_a, corner_b} for edge e
inline const int edge_to_vert[12][2] = {
    {0,1},{1,2},{3,2},{0,3},{4,5},{5,6},{7,6},{4,7},{0,4},{1,5},{2,6},{3,7},
};

// Edge shift: first 3 = anchor offset from cube origin, 4th = axis (0=x,1=y,2=z)
// Used as a unique edge key for vertex deduplication.
struct EdgeShift { int32_t dx, dy, dz; uint8_t axis; };
inline const EdgeShift edge_shift[12] = {
    {0,0,0,0}, {1,0,0,1}, {0,1,0,0}, {0,0,0,1},
    {0,0,1,0}, {1,0,1,1}, {0,1,1,0}, {0,0,1,1},
    {0,0,0,2}, {1,0,0,2}, {1,1,0,2}, {0,1,0,2},
};

// 256-entry edge table: which edges are intersected for each cube configuration.
inline const int edge_table[256] = {
    0x0,   0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c, 0x80c, 0x905, 0xa0f, 0xb06, 0xc0a,
    0xd03, 0xe09, 0xf00, 0x190, 0x99,  0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c, 0x99c, 0x895,
    0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90, 0x230, 0x339, 0x33,  0x13a, 0x636, 0x73f, 0x435,
    0x53c, 0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30, 0x3a0, 0x2a9, 0x1a3, 0xaa,
    0x7a6, 0x6af, 0x5a5, 0x4ac, 0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0, 0x460,
    0x569, 0x663, 0x76a, 0x66,  0x16f, 0x265, 0x36c, 0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963,
    0xa69, 0xb60, 0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff,  0x3f5, 0x2fc, 0xdfc, 0xcf5, 0xfff,
    0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0, 0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55,  0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950, 0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6,
    0x2cf, 0x1c5, 0xcc,  0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0, 0x8c0, 0x9c9,
    0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc, 0xcc,  0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9,
    0x7c0, 0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c, 0x15c, 0x55,  0x35f, 0x256,
    0x55a, 0x453, 0x759, 0x650, 0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc, 0x2fc,
    0x3f5, 0xff,  0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0, 0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f,
    0xd65, 0xc6c, 0x36c, 0x265, 0x16f, 0x66,  0x76a, 0x663, 0x569, 0x460, 0xca0, 0xda9, 0xea3,
    0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac, 0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa,  0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c, 0x53c, 0x435, 0x73f, 0x636, 0x13a,
    0x33,  0x339, 0x230, 0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c, 0x69c, 0x795,
    0x49f, 0x596, 0x29a, 0x393, 0x99,  0x190, 0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905,
    0x80c, 0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0,
};

// 256 × 16 triangle table: up to 5 triangles per cube config, −1 terminated.
inline const int tri_table[256][16] = {
    {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,8,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,1,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,8,3,9,8,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,8,3,1,2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {9,2,10,0,2,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {2,8,3,2,10,8,10,9,8,-1,-1,-1,-1,-1,-1,-1},
    {3,11,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,11,2,8,11,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,9,0,2,3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,11,2,1,9,11,9,8,11,-1,-1,-1,-1,-1,-1,-1},
    {3,10,1,11,10,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,10,1,0,8,10,8,11,10,-1,-1,-1,-1,-1,-1,-1},
    {3,9,0,3,11,9,11,10,9,-1,-1,-1,-1,-1,-1,-1},
    {9,8,10,10,8,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,7,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,3,0,7,3,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,1,9,8,4,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,1,9,4,7,1,7,3,1,-1,-1,-1,-1,-1,-1,-1},
    {1,2,10,8,4,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {3,4,7,3,0,4,1,2,10,-1,-1,-1,-1,-1,-1,-1},
    {9,2,10,9,0,2,8,4,7,-1,-1,-1,-1,-1,-1,-1},
    {2,10,9,2,9,7,2,7,3,7,9,4,-1,-1,-1,-1},
    {8,4,7,3,11,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {11,4,7,11,2,4,2,0,4,-1,-1,-1,-1,-1,-1,-1},
    {9,0,1,8,4,7,2,3,11,-1,-1,-1,-1,-1,-1,-1},
    {4,7,11,9,4,11,9,11,2,9,2,1,-1,-1,-1,-1},
    {3,10,1,3,11,10,7,8,4,-1,-1,-1,-1,-1,-1,-1},
    {1,11,10,1,4,11,1,0,4,7,11,4,-1,-1,-1,-1},
    {4,7,8,9,0,11,9,11,10,11,0,3,-1,-1,-1,-1},
    {4,7,11,4,11,9,9,11,10,-1,-1,-1,-1,-1,-1,-1},
    {9,5,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {9,5,4,0,8,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,5,4,1,5,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {8,5,4,8,3,5,3,1,5,-1,-1,-1,-1,-1,-1,-1},
    {1,2,10,9,5,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {3,0,8,1,2,10,4,9,5,-1,-1,-1,-1,-1,-1,-1},
    {5,2,10,5,4,2,4,0,2,-1,-1,-1,-1,-1,-1,-1},
    {2,10,5,3,2,5,3,5,4,3,4,8,-1,-1,-1,-1},
    {9,5,4,2,3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,11,2,0,8,11,4,9,5,-1,-1,-1,-1,-1,-1,-1},
    {0,5,4,0,1,5,2,3,11,-1,-1,-1,-1,-1,-1,-1},
    {2,1,5,2,5,8,2,8,11,4,8,5,-1,-1,-1,-1},
    {10,3,11,10,1,3,9,5,4,-1,-1,-1,-1,-1,-1,-1},
    {4,9,5,0,8,1,8,10,1,8,11,10,-1,-1,-1,-1},
    {5,4,0,5,0,11,5,11,10,11,0,3,-1,-1,-1,-1},
    {5,4,8,5,8,10,10,8,11,-1,-1,-1,-1,-1,-1,-1},
    {9,7,8,5,7,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {9,3,0,9,5,3,5,7,3,-1,-1,-1,-1,-1,-1,-1},
    {0,7,8,0,1,7,1,5,7,-1,-1,-1,-1,-1,-1,-1},
    {1,5,3,3,5,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {9,7,8,9,5,7,10,1,2,-1,-1,-1,-1,-1,-1,-1},
    {10,1,2,9,5,0,5,3,0,5,7,3,-1,-1,-1,-1},
    {8,0,2,8,2,5,8,5,7,10,5,2,-1,-1,-1,-1},
    {2,10,5,2,5,3,3,5,7,-1,-1,-1,-1,-1,-1,-1},
    {7,9,5,7,8,9,3,11,2,-1,-1,-1,-1,-1,-1,-1},
    {9,5,7,9,7,2,9,2,0,2,7,11,-1,-1,-1,-1},
    {2,3,11,0,1,8,1,7,8,1,5,7,-1,-1,-1,-1},
    {11,2,1,11,1,7,7,1,5,-1,-1,-1,-1,-1,-1,-1},
    {9,5,8,8,5,7,10,1,3,10,3,11,-1,-1,-1,-1},
    {5,7,0,5,0,9,7,11,0,1,0,10,11,10,0,-1},
    {11,10,0,11,0,3,10,5,0,8,0,7,5,7,0,-1},
    {11,10,5,7,11,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {10,6,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,8,3,5,10,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {9,0,1,5,10,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,8,3,1,9,8,5,10,6,-1,-1,-1,-1,-1,-1,-1},
    {1,6,5,2,6,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,6,5,1,2,6,3,0,8,-1,-1,-1,-1,-1,-1,-1},
    {9,6,5,9,0,6,0,2,6,-1,-1,-1,-1,-1,-1,-1},
    {5,9,8,5,8,2,5,2,6,3,2,8,-1,-1,-1,-1},
    {2,3,11,10,6,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {11,0,8,11,2,0,10,6,5,-1,-1,-1,-1,-1,-1,-1},
    {0,1,9,2,3,11,5,10,6,-1,-1,-1,-1,-1,-1,-1},
    {5,10,6,1,9,2,9,11,2,9,8,11,-1,-1,-1,-1},
    {6,3,11,6,5,3,5,1,3,-1,-1,-1,-1,-1,-1,-1},
    {0,8,11,0,11,5,0,5,1,5,11,6,-1,-1,-1,-1},
    {3,11,6,0,3,6,0,6,5,0,5,9,-1,-1,-1,-1},
    {6,5,9,6,9,11,11,9,8,-1,-1,-1,-1,-1,-1,-1},
    {5,10,6,4,7,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,3,0,4,7,3,6,5,10,-1,-1,-1,-1,-1,-1,-1},
    {1,9,0,5,10,6,8,4,7,-1,-1,-1,-1,-1,-1,-1},
    {10,6,5,1,9,7,1,7,3,7,9,4,-1,-1,-1,-1},
    {6,1,2,6,5,1,4,7,8,-1,-1,-1,-1,-1,-1,-1},
    {1,2,5,5,2,6,3,0,4,3,4,7,-1,-1,-1,-1},
    {8,4,7,9,0,5,0,6,5,0,2,6,-1,-1,-1,-1},
    {7,3,9,7,9,4,3,2,9,5,9,6,2,6,9,-1},
    {3,11,2,7,8,4,10,6,5,-1,-1,-1,-1,-1,-1,-1},
    {5,10,6,4,7,2,4,2,0,2,7,11,-1,-1,-1,-1},
    {0,1,9,4,7,8,2,3,11,5,10,6,-1,-1,-1,-1},
    {9,2,1,9,11,2,9,4,11,7,11,4,5,10,6,-1},
    {8,4,7,3,11,5,3,5,1,5,11,6,-1,-1,-1,-1},
    {5,1,11,5,11,6,1,0,11,7,11,4,0,4,11,-1},
    {0,5,9,0,6,5,0,3,6,11,6,3,8,4,7,-1},
    {6,5,9,6,9,11,4,7,9,7,11,9,-1,-1,-1,-1},
    {10,4,9,6,4,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,10,6,4,9,10,0,8,3,-1,-1,-1,-1,-1,-1,-1},
    {10,0,1,10,6,0,6,4,0,-1,-1,-1,-1,-1,-1,-1},
    {8,3,1,8,1,6,8,6,4,6,1,10,-1,-1,-1,-1},
    {1,4,9,1,2,4,2,6,4,-1,-1,-1,-1,-1,-1,-1},
    {3,0,8,1,2,9,2,4,9,2,6,4,-1,-1,-1,-1},
    {0,2,4,4,2,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {8,3,2,8,2,4,4,2,6,-1,-1,-1,-1,-1,-1,-1},
    {10,4,9,10,6,4,11,2,3,-1,-1,-1,-1,-1,-1,-1},
    {0,8,2,2,8,11,4,9,10,4,10,6,-1,-1,-1,-1},
    {3,11,2,0,1,6,0,6,4,6,1,10,-1,-1,-1,-1},
    {6,4,1,6,1,10,4,8,1,2,1,11,8,11,1,-1},
    {9,6,4,9,3,6,9,1,3,11,6,3,-1,-1,-1,-1},
    {8,11,1,8,1,0,11,6,1,9,1,4,6,4,1,-1},
    {3,11,6,3,6,0,0,6,4,-1,-1,-1,-1,-1,-1,-1},
    {6,4,8,11,6,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {7,10,6,7,8,10,8,9,10,-1,-1,-1,-1,-1,-1,-1},
    {0,7,3,0,10,7,0,9,10,6,7,10,-1,-1,-1,-1},
    {10,6,7,1,10,7,1,7,8,1,8,0,-1,-1,-1,-1},
    {10,6,7,10,7,1,1,7,3,-1,-1,-1,-1,-1,-1,-1},
    {1,2,6,1,6,8,1,8,9,8,6,7,-1,-1,-1,-1},
    {2,6,9,2,9,1,6,7,9,0,9,3,7,3,9,-1},
    {7,8,0,7,0,6,6,0,2,-1,-1,-1,-1,-1,-1,-1},
    {7,3,2,6,7,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {2,3,11,10,6,8,10,8,9,8,6,7,-1,-1,-1,-1},
    {2,0,7,2,7,11,0,9,7,6,7,10,9,10,7,-1},
    {1,8,0,1,7,8,1,10,7,6,7,10,2,3,11,-1},
    {11,2,1,11,1,7,10,6,1,6,7,1,-1,-1,-1,-1},
    {8,9,6,8,6,7,9,1,6,11,6,3,1,3,6,-1},
    {0,9,1,11,6,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {7,8,0,7,0,6,3,11,0,11,6,0,-1,-1,-1,-1},
    {7,11,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {7,6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {3,0,8,11,7,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,1,9,11,7,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {8,1,9,8,3,1,11,7,6,-1,-1,-1,-1,-1,-1,-1},
    {10,1,2,6,11,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,2,10,3,0,8,6,11,7,-1,-1,-1,-1,-1,-1,-1},
    {2,9,0,2,10,9,6,11,7,-1,-1,-1,-1,-1,-1,-1},
    {6,11,7,2,10,3,10,8,3,10,9,8,-1,-1,-1,-1},
    {7,2,3,6,2,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {7,0,8,7,6,0,6,2,0,-1,-1,-1,-1,-1,-1,-1},
    {2,7,6,2,3,7,0,1,9,-1,-1,-1,-1,-1,-1,-1},
    {1,6,2,1,8,6,1,9,8,8,7,6,-1,-1,-1,-1},
    {10,7,6,10,1,7,1,3,7,-1,-1,-1,-1,-1,-1,-1},
    {10,7,6,1,7,10,1,8,7,1,0,8,-1,-1,-1,-1},
    {0,3,7,0,7,10,0,10,9,6,10,7,-1,-1,-1,-1},
    {7,6,10,7,10,8,8,10,9,-1,-1,-1,-1,-1,-1,-1},
    {6,8,4,11,8,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {3,6,11,3,0,6,0,4,6,-1,-1,-1,-1,-1,-1,-1},
    {8,6,11,8,4,6,9,0,1,-1,-1,-1,-1,-1,-1,-1},
    {9,4,6,9,6,3,9,3,1,11,3,6,-1,-1,-1,-1},
    {6,8,4,6,11,8,2,10,1,-1,-1,-1,-1,-1,-1,-1},
    {1,2,10,3,0,11,0,6,11,0,4,6,-1,-1,-1,-1},
    {4,11,8,4,6,11,0,2,9,2,10,9,-1,-1,-1,-1},
    {10,9,3,10,3,2,9,4,3,11,3,6,4,6,3,-1},
    {8,2,3,8,4,2,4,6,2,-1,-1,-1,-1,-1,-1,-1},
    {0,4,2,4,6,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,9,0,2,3,4,2,4,6,4,3,8,-1,-1,-1,-1},
    {1,9,4,1,4,2,2,4,6,-1,-1,-1,-1,-1,-1,-1},
    {8,1,3,8,6,1,8,4,6,6,10,1,-1,-1,-1,-1},
    {10,1,0,10,0,6,6,0,4,-1,-1,-1,-1,-1,-1,-1},
    {4,6,3,4,3,8,6,10,3,0,3,9,10,9,3,-1},
    {10,9,4,6,10,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,9,5,7,6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,8,3,4,9,5,11,7,6,-1,-1,-1,-1,-1,-1,-1},
    {5,0,1,5,4,0,7,6,11,-1,-1,-1,-1,-1,-1,-1},
    {11,7,6,8,3,4,3,5,4,3,1,5,-1,-1,-1,-1},
    {9,5,4,10,1,2,7,6,11,-1,-1,-1,-1,-1,-1,-1},
    {6,11,7,1,2,10,0,8,3,4,9,5,-1,-1,-1,-1},
    {7,6,11,5,4,10,4,2,10,4,0,2,-1,-1,-1,-1},
    {3,4,8,3,5,4,3,2,5,10,5,2,11,7,6,-1},
    {7,2,3,7,6,2,5,4,9,-1,-1,-1,-1,-1,-1,-1},
    {9,5,4,0,8,6,0,6,2,6,8,7,-1,-1,-1,-1},
    {3,6,2,3,7,6,1,5,0,5,4,0,-1,-1,-1,-1},
    {6,2,8,6,8,7,2,1,8,4,8,5,1,5,8,-1},
    {9,5,4,10,1,6,1,7,6,1,3,7,-1,-1,-1,-1},
    {1,6,10,1,7,6,1,0,7,8,7,0,9,5,4,-1},
    {4,0,10,4,10,5,0,3,10,6,10,7,3,7,10,-1},
    {7,6,10,7,10,8,5,4,10,4,8,10,-1,-1,-1,-1},
    {6,9,5,6,11,9,11,8,9,-1,-1,-1,-1,-1,-1,-1},
    {3,6,11,0,6,3,0,5,6,0,9,5,-1,-1,-1,-1},
    {0,11,8,0,5,11,0,1,5,5,6,11,-1,-1,-1,-1},
    {6,11,3,6,3,5,5,3,1,-1,-1,-1,-1,-1,-1,-1},
    {1,2,10,9,5,11,9,11,8,11,5,6,-1,-1,-1,-1},
    {0,11,3,0,6,11,0,9,6,5,6,9,1,2,10,-1},
    {11,8,5,11,5,6,8,0,5,10,5,2,0,2,5,-1},
    {6,11,3,6,3,5,2,10,3,10,5,3,-1,-1,-1,-1},
    {5,8,9,5,2,8,5,6,2,3,8,2,-1,-1,-1,-1},
    {9,5,6,9,6,0,0,6,2,-1,-1,-1,-1,-1,-1,-1},
    {1,5,8,1,8,0,5,6,8,3,8,2,6,2,8,-1},
    {1,5,6,2,1,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,3,6,1,6,10,3,8,6,5,6,9,8,9,6,-1},
    {10,1,0,10,0,6,9,5,0,5,6,0,-1,-1,-1,-1},
    {0,3,8,5,6,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {10,5,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {11,5,10,7,5,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {11,5,10,11,7,5,8,3,0,-1,-1,-1,-1,-1,-1,-1},
    {5,11,7,5,10,11,1,9,0,-1,-1,-1,-1,-1,-1,-1},
    {10,7,5,10,11,7,9,8,1,8,3,1,-1,-1,-1,-1},
    {11,1,2,11,7,1,7,5,1,-1,-1,-1,-1,-1,-1,-1},
    {0,8,3,1,2,7,1,7,5,7,2,11,-1,-1,-1,-1},
    {9,7,5,9,2,7,9,0,2,2,11,7,-1,-1,-1,-1},
    {7,5,2,7,2,11,5,9,2,3,2,8,9,8,2,-1},
    {2,5,10,2,3,5,3,7,5,-1,-1,-1,-1,-1,-1,-1},
    {8,2,0,8,5,2,8,7,5,10,2,5,-1,-1,-1,-1},
    {9,0,1,5,10,3,5,3,7,3,10,2,-1,-1,-1,-1},
    {9,8,2,9,2,1,8,7,2,10,2,5,7,5,2,-1},
    {1,3,5,3,7,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,8,7,0,7,1,1,7,5,-1,-1,-1,-1,-1,-1,-1},
    {9,0,3,9,3,5,5,3,7,-1,-1,-1,-1,-1,-1,-1},
    {9,8,7,5,9,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {5,8,4,5,10,8,10,11,8,-1,-1,-1,-1,-1,-1,-1},
    {5,0,4,5,11,0,5,10,11,11,3,0,-1,-1,-1,-1},
    {0,1,9,8,4,10,8,10,11,10,4,5,-1,-1,-1,-1},
    {10,11,4,10,4,5,11,3,4,9,4,1,3,1,4,-1},
    {2,5,1,2,8,5,2,11,8,4,5,8,-1,-1,-1,-1},
    {0,4,11,0,11,3,4,5,11,2,11,1,5,1,11,-1},
    {0,2,5,0,5,9,2,11,5,4,5,8,11,8,5,-1},
    {9,4,5,2,11,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {2,5,10,3,5,2,3,4,5,3,8,4,-1,-1,-1,-1},
    {5,10,2,5,2,4,4,2,0,-1,-1,-1,-1,-1,-1,-1},
    {3,10,2,3,5,10,3,8,5,4,5,8,0,1,9,-1},
    {5,10,2,5,2,4,1,9,2,9,4,2,-1,-1,-1,-1},
    {8,4,5,8,5,3,3,5,1,-1,-1,-1,-1,-1,-1,-1},
    {0,4,5,1,0,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {8,4,5,8,5,3,9,0,5,0,3,5,-1,-1,-1,-1},
    {9,4,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,11,7,4,9,11,9,10,11,-1,-1,-1,-1,-1,-1,-1},
    {0,8,3,4,9,7,9,11,7,9,10,11,-1,-1,-1,-1},
    {1,10,11,1,11,4,1,4,0,7,4,11,-1,-1,-1,-1},
    {3,1,4,3,4,8,1,10,4,7,4,11,10,11,4,-1},
    {4,11,7,9,11,4,9,2,11,9,1,2,-1,-1,-1,-1},
    {9,7,4,9,11,7,9,1,11,2,11,1,0,8,3,-1},
    {11,7,4,11,4,2,2,4,0,-1,-1,-1,-1,-1,-1,-1},
    {11,7,4,11,4,2,8,3,4,3,2,4,-1,-1,-1,-1},
    {2,9,10,2,7,9,2,3,7,7,4,9,-1,-1,-1,-1},
    {9,10,7,9,7,4,10,2,7,8,7,0,2,0,7,-1},
    {3,7,10,3,10,2,7,4,10,1,10,0,4,0,10,-1},
    {1,10,2,8,7,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,9,1,4,1,7,7,1,3,-1,-1,-1,-1,-1,-1,-1},
    {4,9,1,4,1,7,0,8,1,8,7,1,-1,-1,-1,-1},
    {4,0,3,7,4,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,8,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {9,10,8,10,11,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {3,0,9,3,9,11,11,9,10,-1,-1,-1,-1,-1,-1,-1},
    {0,1,10,0,10,8,8,10,11,-1,-1,-1,-1,-1,-1,-1},
    {3,1,10,11,3,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,2,11,1,11,9,9,11,8,-1,-1,-1,-1,-1,-1,-1},
    {3,0,9,3,9,11,1,2,9,2,11,9,-1,-1,-1,-1},
    {0,2,11,8,0,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {3,2,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {2,3,8,2,8,10,10,8,9,-1,-1,-1,-1,-1,-1,-1},
    {9,10,2,0,9,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {2,3,8,2,8,10,0,1,8,1,10,8,-1,-1,-1,-1},
    {1,10,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,3,8,9,1,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,9,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,3,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
};

}  // namespace mc_tables

// =====================================================================
// Internal helpers
// =====================================================================

namespace detail {

/// Dominant semantic class from the sparse K_TOP slots.
inline uint16_t dominantClass(const Voxel& v) {
  uint16_t cls = 0;
  float best = 0.f;
  for (int i = 0; i < K_TOP; ++i) {
    if (v.sem_cnt[i] > best) {
      best = v.sem_cnt[i];
      cls  = v.sem_cls[i];
    }
  }
  return cls;
}

/// Edge key for vertex deduplication: (anchor_x, anchor_y, anchor_z, axis).
struct EdgeKey {
  int32_t x, y, z;
  uint8_t axis;
  bool operator==(const EdgeKey& o) const {
    return x == o.x && y == o.y && z == o.z && axis == o.axis;
  }
};

struct EdgeKeyHash {
  size_t operator()(const EdgeKey& k) const {
    auto h = std::hash<int64_t>{}(
        (int64_t(k.x) * 73856093) ^
        (int64_t(k.y) * 19349669) ^
        (int64_t(k.z) * 83492791));
    h ^= std::hash<uint8_t>{}(k.axis) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
};

}  // namespace detail

// =====================================================================
// extractMesh — marching cubes on TSDF zero-crossing
// =====================================================================

/// Extract a triangle mesh from the TSDF zero-crossing via marching cubes.
///
/// Iterates every active voxel, treats it as the (0,0,0) corner of a cube
/// whose 8 corners are the voxel and its 7 positive-offset neighbours.
/// Cubes with any corner below `min_weight` are skipped.
///
/// Vertex positions are sub-voxel interpolated along edges where a sign
/// change occurs, matching the standard marching cubes algorithm used by
/// Open3D and SLIM-VDB.
///
/// @param grid       Bonxai VoxelGrid containing Voxel with tsdf_distance/tsdf_weight.
/// @param min_weight Minimum tsdf_weight for a corner to count as valid.
/// @param resolution Voxel edge length in metres.
/// @return           Triangle mesh with per-triangle semantic labels.
inline TriangleMesh extractMesh(
    const Bonxai::VoxelGrid<Voxel>& grid,
    float min_weight,
    double resolution)
{
  using namespace mc_tables;
  (void)resolution;
  TriangleMesh mesh;
  auto acc = grid.createConstAccessor();

  // Edge → global vertex index (deduplication across adjacent cubes)
  std::unordered_map<detail::EdgeKey, int, detail::EdgeKeyHash> edge_to_vidx;
  int edge_local[12];

  grid.forEachCell([&](const Voxel& /*anchor_v*/, const Bonxai::CoordT& coord) {
    // Gather 8 corner TSDF values
    float f[8];
    const Voxel* cv[8];
    bool valid = true;
    for (int i = 0; i < 8; ++i) {
      Bonxai::CoordT cc = {
        coord.x + corners[i].x,
        coord.y + corners[i].y,
        coord.z + corners[i].z};
      cv[i] = acc.value(cc);
      if (!cv[i] || cv[i]->tsdf_weight < min_weight) {
        valid = false;
        break;
      }
      f[i] = cv[i]->tsdf_distance;
    }
    if (!valid) return;

    // Compute cube index from sign of each corner
    int cube_index = 0;
    for (int i = 0; i < 8; ++i) {
      if (f[i] < 0.f) cube_index |= (1 << i);
    }
    if (cube_index == 0 || cube_index == 255) return;

    const int32_t x = coord.x;
    const int32_t y = coord.y;
    const int32_t z = coord.z;

    // Process each intersected edge
    for (int e = 0; e < 12; ++e) {
      if (!(edge_table[cube_index] & (1 << e))) continue;

      const auto& es = edge_shift[e];
      detail::EdgeKey ek{x + es.dx, y + es.dy, z + es.dz, es.axis};

      auto it = edge_to_vidx.find(ek);
      if (it != edge_to_vidx.end()) {
        edge_local[e] = it->second;
      } else {
        // Interpolate vertex along the edge
        const int v0 = edge_to_vert[e][0];
        const int v1 = edge_to_vert[e][1];
        const float f0 = std::fabs(f[v0]);
        const float f1 = std::fabs(f[v1]);
        const float t = (f0 + f1 > 1e-10f) ? f0 / (f0 + f1) : 0.5f;

        // World positions of the two corner voxels
        auto p0 = grid.coordToPos({
          coord.x + corners[v0].x,
          coord.y + corners[v0].y,
          coord.z + corners[v0].z});
        auto p1 = grid.coordToPos({
          coord.x + corners[v1].x,
          coord.y + corners[v1].y,
          coord.z + corners[v1].z});

        Eigen::Vector3f vertex(
          (float)p0.x + t * (float)(p1.x - p0.x),
          (float)p0.y + t * (float)(p1.y - p0.y),
          (float)p0.z + t * (float)(p1.z - p0.z));

        int vidx = (int)mesh.vertices.size();
        mesh.vertices.push_back(vertex);
        edge_to_vidx[ek] = vidx;
        edge_local[e] = vidx;
      }
    }

    // Emit triangles
    // Semantic label: argmax from the anchor voxel's sparse slots.
    // Matches SLIM-VDB which reads semantics from the anchor coord.
    const Voxel* anchor = acc.value(coord);
    uint16_t label = anchor ? detail::dominantClass(*anchor) : 0;

    for (int i = 0; tri_table[cube_index][i] != -1; i += 3) {
      mesh.triangles.emplace_back(
        edge_local[tri_table[cube_index][i]],
        edge_local[tri_table[cube_index][i + 2]],
        edge_local[tri_table[cube_index][i + 1]]);
      mesh.tri_labels.push_back(label);
    }
  });

  return mesh;
}

// =====================================================================
// extractZeroCrossing — lightweight sign-change point cloud
// =====================================================================

/// Extract zero-crossing surface points by checking 3 positive-axis
/// neighbours per active voxel. Cheaper than full marching cubes —
/// produces sub-voxel interpolated points wherever the TSDF sign flips.
///
/// Semantic label is taken from the positive-side voxel (the one "in
/// front" of the surface, where the sensor observed the hit).
///
/// @param grid       Bonxai VoxelGrid containing Voxel with tsdf_distance/tsdf_weight.
/// @param min_weight Minimum tsdf_weight for both neighbours to be valid.
/// @param resolution Voxel edge length in metres.
/// @return           Vector of sub-voxel surface points with semantic labels.
inline std::vector<SurfacePoint> extractZeroCrossing(
    const Bonxai::VoxelGrid<Voxel>& grid,
    float min_weight,
    double resolution)
{
  (void)resolution;  // implicitly encoded in grid.coordToPos()
  std::vector<SurfacePoint> points;
  auto acc = grid.createConstAccessor();

  static const Bonxai::CoordT axes[3] = {{1,0,0}, {0,1,0}, {0,0,1}};

  grid.forEachCell([&](const Voxel& v, const Bonxai::CoordT& c) {
    if (v.tsdf_weight < min_weight) return;

    for (int a = 0; a < 3; ++a) {
      Bonxai::CoordT nc = {c.x + axes[a].x, c.y + axes[a].y, c.z + axes[a].z};
      const Voxel* nv = acc.value(nc);
      if (!nv || nv->tsdf_weight < min_weight) continue;

      // Sign change → zero-crossing between c and nc
      if ((v.tsdf_distance > 0.f) != (nv->tsdf_distance > 0.f)) {
        const float f0 = std::fabs(v.tsdf_distance);
        const float f1 = std::fabs(nv->tsdf_distance);
        const float t = (f0 + f1 > 1e-10f) ? f0 / (f0 + f1) : 0.5f;

        auto p0 = grid.coordToPos(c);
        auto p1 = grid.coordToPos(nc);
        Eigen::Vector3f pos(
          (float)p0.x + t * (float)(p1.x - p0.x),
          (float)p0.y + t * (float)(p1.y - p0.y),
          (float)p0.z + t * (float)(p1.z - p0.z));

        // Semantic label from the occupied side (positive SDF = sensor side)
        const Voxel& occ_side = (v.tsdf_distance > 0.f) ? v : *nv;
        uint16_t label = detail::dominantClass(occ_side);

        points.push_back({pos, 0.f, label});
      }
    }
  });

  return points;
}

// =====================================================================
// extractPointCloud — SLIM-VDB style: emit voxel centers with labels
// =====================================================================

/// Extract a point cloud of voxel centres where tsdf_weight exceeds the
/// threshold. This matches SLIM-VDB's `ExtractPointCloud` — every
/// sufficiently-observed voxel emits one point at its centre with the
/// argmax semantic label.
///
/// @param grid       Bonxai VoxelGrid containing Voxel with tsdf_distance/tsdf_weight.
/// @param min_weight Minimum tsdf_weight for emission.
/// @param resolution Voxel edge length in metres (used for centre offset).
/// @return           Pair of (positions, labels).
inline std::pair<std::vector<Eigen::Vector3f>, std::vector<uint16_t>>
extractPointCloud(
    const Bonxai::VoxelGrid<Voxel>& grid,
    float min_weight,
    double resolution)
{
  (void)resolution;
  std::vector<Eigen::Vector3f> positions;
  std::vector<uint16_t> labels;

  grid.forEachCell([&](const Voxel& v, const Bonxai::CoordT& c) {
    if (v.tsdf_weight < min_weight) return;

    auto p = grid.coordToPos(c);
    positions.emplace_back((float)p.x, (float)p.y, (float)p.z);
    labels.push_back(detail::dominantClass(v));
  });

  return {positions, labels};
}

// =====================================================================
// Split-grid refactor — TsdfVoxel-typed overloads (geometry only, no labels)
// =====================================================================
//
// These mirror the three legacy extractors above but operate on
// `Bonxai::VoxelGrid<TsdfVoxel>`, which has no semantic fields. Labels
// come via free functions in `mesh_labelling.hpp` (Step 4) after
// extraction, looking up the SemBeta grid by coord.
//
// Half-voxel centre offset: SLIM-VDB / VDBFusion convention is to
// interpolate between voxel CENTRES, not lower corners. The Voxel-typed
// extractors above use lower corners (Bonxai's `coordToPos`); these new
// overloads add `+0.5*resolution` per axis to match SLIM-VDB exactly.

inline TriangleMesh extractMesh(
    const Bonxai::VoxelGrid<TsdfVoxel>& grid,
    float min_weight,
    double resolution)
{
  using namespace mc_tables;
  TriangleMesh mesh;
  auto acc = grid.createConstAccessor();
  std::unordered_map<detail::EdgeKey, int, detail::EdgeKeyHash> edge_to_vidx;
  int edge_local[12];
  const float h = 0.5f * static_cast<float>(resolution);

  grid.forEachCell([&](const TsdfVoxel& /*anchor_v*/, const Bonxai::CoordT& coord) {
    float f[8];
    bool valid = true;
    for (int i = 0; i < 8; ++i) {
      Bonxai::CoordT cc = {
        coord.x + corners[i].x,
        coord.y + corners[i].y,
        coord.z + corners[i].z};
      const TsdfVoxel* cv = acc.value(cc);
      if (!cv || cv->weight < min_weight) {
        valid = false;
        break;
      }
      f[i] = cv->distance;
    }
    if (!valid) return;

    int cube_index = 0;
    for (int i = 0; i < 8; ++i) if (f[i] < 0.f) cube_index |= (1 << i);
    if (cube_index == 0 || cube_index == 255) return;

    const int32_t x = coord.x;
    const int32_t y = coord.y;
    const int32_t z = coord.z;

    for (int e = 0; e < 12; ++e) {
      if (!(edge_table[cube_index] & (1 << e))) continue;
      const auto& es = edge_shift[e];
      detail::EdgeKey ek{x + es.dx, y + es.dy, z + es.dz, es.axis};

      auto it = edge_to_vidx.find(ek);
      if (it != edge_to_vidx.end()) {
        edge_local[e] = it->second;
      } else {
        const int v0 = edge_to_vert[e][0];
        const int v1 = edge_to_vert[e][1];
        const float f0 = std::fabs(f[v0]);
        const float f1 = std::fabs(f[v1]);
        const float t = (f0 + f1 > 1e-10f) ? f0 / (f0 + f1) : 0.5f;

        auto p0 = grid.coordToPos({
          coord.x + corners[v0].x,
          coord.y + corners[v0].y,
          coord.z + corners[v0].z});
        auto p1 = grid.coordToPos({
          coord.x + corners[v1].x,
          coord.y + corners[v1].y,
          coord.z + corners[v1].z});

        Eigen::Vector3f vertex(
          (float)p0.x + h + t * (float)(p1.x - p0.x),
          (float)p0.y + h + t * (float)(p1.y - p0.y),
          (float)p0.z + h + t * (float)(p1.z - p0.z));

        int vidx = (int)mesh.vertices.size();
        mesh.vertices.push_back(vertex);
        edge_to_vidx[ek] = vidx;
        edge_local[e] = vidx;
      }
    }

    for (int i = 0; tri_table[cube_index][i] != -1; i += 3) {
      mesh.triangles.emplace_back(
        edge_local[tri_table[cube_index][i]],
        edge_local[tri_table[cube_index][i + 2]],
        edge_local[tri_table[cube_index][i + 1]]);
      mesh.tri_labels.push_back(0xFFFF);  // sentinel — labelled separately
    }
  });

  return mesh;
}

inline std::vector<SurfacePoint> extractZeroCrossing(
    const Bonxai::VoxelGrid<TsdfVoxel>& grid,
    float min_weight,
    double resolution)
{
  std::vector<SurfacePoint> points;
  auto acc = grid.createConstAccessor();
  static const Bonxai::CoordT axes[3] = {{1,0,0}, {0,1,0}, {0,0,1}};
  const float h = 0.5f * static_cast<float>(resolution);

  grid.forEachCell([&](const TsdfVoxel& v, const Bonxai::CoordT& c) {
    if (v.weight < min_weight) return;

    for (int a = 0; a < 3; ++a) {
      Bonxai::CoordT nc = {c.x + axes[a].x, c.y + axes[a].y, c.z + axes[a].z};
      const TsdfVoxel* nv = acc.value(nc);
      if (!nv || nv->weight < min_weight) continue;

      if ((v.distance > 0.f) != (nv->distance > 0.f)) {
        const float f0 = std::fabs(v.distance);
        const float f1 = std::fabs(nv->distance);
        const float t = (f0 + f1 > 1e-10f) ? f0 / (f0 + f1) : 0.5f;

        auto p0 = grid.coordToPos(c);
        auto p1 = grid.coordToPos(nc);
        Eigen::Vector3f pos(
          (float)p0.x + h + t * (float)(p1.x - p0.x),
          (float)p0.y + h + t * (float)(p1.y - p0.y),
          (float)p0.z + h + t * (float)(p1.z - p0.z));

        points.push_back({pos, 0.f, 0xFFFF});
      }
    }
  });

  return points;
}

inline std::vector<Eigen::Vector3f> extractPointCloud(
    const Bonxai::VoxelGrid<TsdfVoxel>& grid,
    float min_weight,
    double resolution)
{
  std::vector<Eigen::Vector3f> positions;
  const float h = 0.5f * static_cast<float>(resolution);
  grid.forEachCell([&](const TsdfVoxel& v, const Bonxai::CoordT& c) {
    if (v.weight < min_weight) return;
    auto p = grid.coordToPos(c);
    positions.emplace_back(
        (float)p.x + h, (float)p.y + h, (float)p.z + h);
  });
  return positions;
}

}  // namespace scovox
