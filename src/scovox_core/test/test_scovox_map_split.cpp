/// @file
/// @brief Step-5 gate: ScovoxMapSplit composer end-to-end smoke + parity.

#include <gtest/gtest.h>
#include <Eigen/Core>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <utility>
#include <vector>

#include "scovox/beta_voxel.hpp"
#include "scovox/dir_voxel.hpp"
#include "scovox/scovox_map_split.hpp"
#include "scovox/tsdf_voxel.hpp"

namespace {

scovox::ScovoxMapSplit makeMap() {
  scovox::ScovoxMapSplit::Params p;
  p.resolution        = 0.05;
  p.tsdf.sdf_trunc    = 0.15f;
  p.semsplit.kappa0    = 1.0f;
  p.semsplit.dirichlet_min_p_occ = 0.5f;
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

TEST(ScovoxMapSplit, IntegrateMissCarvesOccupancyOnly) {
  auto m = makeMap();
  m.integrateMiss(Eigen::Vector3f(0, 0, 0),
                  Eigen::Vector3f(0.50f, 0, 0),
                  /*quality=*/1.0f);

  EXPECT_EQ(m.tsdfVoxelCount(),  0u) << "miss must NOT allocate TSDF voxels";
  // SPLIT substrate: a no-return ray carves the full-ray Beta (occupancy)
  // grid but commits no class, so the hit-sparse Dir (semantics) grid stays
  // empty. (Under the old unified SemDir grid both lived in one voxel.)
  EXPECT_GT(m.betaVoxelCount(), 0u) << "miss must carve occupancy (Beta grid)";
  EXPECT_EQ(m.dirVoxelCount(),  0u) << "miss commits no semantics (Dir grid empty)";
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
  // The full-ray occupancy (Beta) grid should always have at least as many
  // voxels as the band-only TSDF grid (full-ray carve vs band).
  EXPECT_GE(m.betaVoxelCount(), m.tsdfVoxelCount());
  // Memory matches struct sizes (modulo Bonxai overhead).
  EXPECT_GT(m.tsdfGridBytes(),   m.tsdfVoxelCount()   * sizeof(scovox::TsdfVoxel)   - 1);
  // Split grid is Beta∥Dir: semdirGridBytes() == Beta bytes + Dir bytes, so it
  // must exceed the raw Dir-cell footprint (and the raw Beta footprint too).
  EXPECT_GT(m.semdirGridBytes(), m.dirVoxelCount()  * sizeof(scovox::DirVoxel)  - 1);
  EXPECT_GT(m.semdirGridBytes(), m.betaVoxelCount() * sizeof(scovox::BetaVoxel) - 1);
}

TEST(ScovoxMapSplit, DrainTouchedSplitsByGrid) {
  auto m = makeMap();
  m.integrateHit(Eigen::Vector3f(0, 0, 0),
                 Eigen::Vector3f(0.50f, 0, 0),
                 nullptr, 1.0f);
  auto t = m.drainTouchedTsdf();
  auto sb = m.drainTouchedBeta();
  auto sd = m.drainTouchedDir();
  EXPECT_GT(t.size(),  0u);
  EXPECT_GT(sb.size(), 0u);
  EXPECT_GT(sd.size(), 0u);
  // Second drain should be empty (buffers cleared).
  EXPECT_EQ(m.drainTouchedTsdf().size(), 0u);
  EXPECT_EQ(m.drainTouchedBeta().size(), 0u);
  EXPECT_EQ(m.drainTouchedDir().size(),  0u);
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
  p.semsplit.kappa0    = 1.0f;
  p.semsplit.dirichlet_min_p_occ = 0.5f;
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

  // Hit voxel — split Beta occupancy state must match exactly.
  auto b_f = m_fused.semsplit().getBetaVoxel(Hp);
  auto b_s = m_split.semsplit().getBetaVoxel(Hp);
  ASSERT_TRUE(b_f.has_value());
  ASSERT_TRUE(b_s.has_value());
  EXPECT_FLOAT_EQ(b_f->a_occ,  b_s->a_occ);
  EXPECT_FLOAT_EQ(b_f->a_free, b_s->a_free);
  EXPECT_FLOAT_EQ(b_f->p_occ(), b_s->p_occ());

  // Hit voxel — split Dir class evidence must match exactly.
  auto d_f = m_fused.semsplit().getDirVoxel(Hp);
  auto d_s = m_split.semsplit().getDirVoxel(Hp);
  ASSERT_TRUE(d_f.has_value());
  ASSERT_TRUE(d_s.has_value());
  EXPECT_FLOAT_EQ(d_f->other, d_s->other);
  for (int i = 0; i < scovox::K_TOP; ++i) {
    EXPECT_FLOAT_EQ(d_f->cnt[i], d_s->cnt[i]);
    EXPECT_EQ(d_f->cls[i],       d_s->cls[i]);
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

// ===========================================================================
// SPLIT substrate (SemSplitMap: BetaVoxel ∥ DirVoxel) behind the flag
// ===========================================================================

namespace {

scovox::ScovoxMapSplit makeSplitMap(bool fused = true) {
  scovox::ScovoxMapSplit::Params p;
  p.resolution                 = 0.05;
  p.tsdf.sdf_trunc             = 0.15f;
  p.semsplit.kappa0              = 1.0f;
  p.semsplit.dirichlet_min_p_occ = 0.5f;
  p.semsplit.num_classes         = 14;
  p.semsplit.alpha_0             = scovox::kDefaultDirichletPrior;
  p.fused_walker               = fused;
  return scovox::ScovoxMapSplit(p);
}

}  // namespace

TEST(ScovoxMapSplitSubstrate, IntegrateHitTouchesAllThreeGrids) {
  auto m = makeSplitMap();
  std::vector<float> probs{0.f, 1.f, 0.f, 0.f};  // class 1
  m.integrateHit(Eigen::Vector3f(0, 0, 0),
                 Eigen::Vector3f(0.50f, 0, 0),
                 &probs, /*quality=*/1.0f);

  EXPECT_GT(m.tsdfVoxelCount(), 0u);
  EXPECT_GT(m.betaVoxelCount(), 0u);
  EXPECT_GT(m.dirVoxelCount(),  0u);
  // Beta is full-ray; Dir is hit-only → far fewer Dir voxels.
  EXPECT_GT(m.betaVoxelCount(), m.dirVoxelCount());
}

TEST(ScovoxMapSplitSubstrate, MissAllocatesBetaButNoDir) {
  auto m = makeSplitMap();
  m.integrateMiss(Eigen::Vector3f(0, 0, 0),
                  Eigen::Vector3f(0.50f, 0, 0),
                  /*quality=*/1.0f);
  EXPECT_EQ(m.tsdfVoxelCount(), 0u) << "miss must NOT allocate TSDF voxels";
  EXPECT_GT(m.betaVoxelCount(), 0u);
  EXPECT_EQ(m.dirVoxelCount(),  0u) << "miss must NOT allocate Dir voxels";
}

TEST(ScovoxMapSplitSubstrate, MeshAndPointCloudLabelArraysConsistent) {
  auto m = makeSplitMap();
  std::vector<float> probs{0.f, 0.f, 1.f, 0.f};  // class 2
  for (int dx = -1; dx <= 1; ++dx)
  for (int dy = -1; dy <= 1; ++dy)
  for (int dz = -1; dz <= 1; ++dz) {
    Eigen::Vector3f ep(0.50f + 0.05f * dx, 0.05f * dy, 0.05f * dz);
    m.integrateHit(Eigen::Vector3f(0, 0, 0), ep, &probs, /*quality=*/1.0f);
  }
  auto mesh = m.extractMesh(/*min_weight=*/0.5f);
  EXPECT_EQ(mesh.tri_labels.size(), mesh.triangles.size());

  auto [positions, labels] = m.extractPointCloud(/*min_weight=*/0.5f);
  EXPECT_EQ(positions.size(), labels.size());
}

TEST(ScovoxMapSplitSubstrate, DrainTouchedPerGrid) {
  auto m = makeSplitMap();
  std::vector<float> probs{0.f, 1.f, 0.f, 0.f};
  m.integrateHit(Eigen::Vector3f(0, 0, 0),
                 Eigen::Vector3f(0.50f, 0, 0),
                 &probs, 1.0f);
  auto tb = m.drainTouchedBeta();
  auto td = m.drainTouchedDir();
  EXPECT_GT(tb.size(), td.size()) << "Beta full-ray vs Dir hit-only";
  EXPECT_EQ(td.size(), 1u);
  // Buffers cleared.
  EXPECT_EQ(m.drainTouchedBeta().size(), 0u);
  EXPECT_EQ(m.drainTouchedDir().size(),  0u);
}

// ===========================================================================
// Dynamic-class routing (transient substrate) — both walkers
// ===========================================================================

// A dynamic hit must: (a) write NO persistent TSDF geometry (no ghost surface),
// (b) commit NO persistent semantics, (c) still carve persistent free space in
// front, (d) deposit occupancy + semantics into the transient grids.
static void checkDynamicHitRouting(bool fused) {
  auto m = makeSplitMap(fused);
  std::vector<float> probs(14, 0.f); probs[3] = 1.0f;  // class 3
  const Eigen::Vector3f Hp(0.50f, 0, 0);

  m.integrateHit(Eigen::Vector3f(0, 0, 0), Hp, &probs, /*quality=*/1.0f,
                 /*is_dynamic=*/true);

  EXPECT_EQ(m.tsdfVoxelCount(), 0u) << "dynamic hit must leave NO persistent TSDF";
  EXPECT_EQ(m.dirVoxelCount(),  0u) << "dynamic hit commits no persistent class";
  EXPECT_GT(m.betaVoxelCount(), 0u) << "free-space carve stays persistent";
  // Transient grids carry the endpoint.
  EXPECT_GT(m.semsplit().transientBetaVoxelCount(), 0u);
  EXPECT_EQ(m.semsplit().transientDirVoxelCount(),  1u);
  EXPECT_EQ(m.semsplit().transientDominantClassAt(Hp), 3u);
  EXPECT_EQ(m.semsplit().dominantClassAt(Hp), 0xFFFFu);  // persistent: none
}

TEST(ScovoxMapSplitDynamic, FusedHitRoutesToTransient)  { checkDynamicHitRouting(true);  }
TEST(ScovoxMapSplitDynamic, SplitHitRoutesToTransient)  { checkDynamicHitRouting(false); }

TEST(ScovoxMapSplitDynamic, NonDynamicDefaultUnchanged) {
  // is_dynamic defaults false → identical to the legacy 4-arg path: all three
  // persistent grids populate, transient stays empty.
  auto m = makeSplitMap();
  std::vector<float> probs(14, 0.f); probs[3] = 1.0f;
  m.integrateHit(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0.50f, 0, 0),
                 &probs, /*quality=*/1.0f);  // no is_dynamic arg

  EXPECT_GT(m.tsdfVoxelCount(), 0u);
  EXPECT_GT(m.dirVoxelCount(),  0u);
  EXPECT_EQ(m.semsplit().transientBetaVoxelCount(), 0u);
  EXPECT_EQ(m.semsplit().transientDirVoxelCount(),  0u);
}

TEST(ScovoxMapSplitDynamic, DecayTransientPassthrough) {
  auto m = makeSplitMap();
  std::vector<float> probs(14, 0.f); probs[3] = 1.0f;
  m.integrateHit(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0.50f, 0, 0),
                 &probs, 1.0f, /*is_dynamic=*/true);
  ASSERT_GT(m.semsplit().transientBetaVoxelCount(), 0u);

  m.decayTransient(0.0f);  // collapse to prior → prune via the composer
  EXPECT_EQ(m.semsplit().transientBetaVoxelCount(), 0u);
  EXPECT_EQ(m.semsplit().transientDirVoxelCount(),  0u);
}

// ===========================================================================
// Far-voxel skip — bit-identity vs the full walk (SCOVOX_DISABLE_FAR_SKIP)
// ===========================================================================
//
// When a carve frame is open and batch_free_carve=false, applyCarveUpdate is a
// guaranteed no-op, so integrateHitFused's far-voxel skip may early-return any
// voxel whose Chebyshev coord distance to the hit exceeds the band threshold.
// The claim is BIT identity, not approximate parity: a full-walk map (skip
// disabled via the per-instance env latch) and a skipping map fed identical
// rays must agree byte-for-byte on the TSDF/Beta/Dir grids and produce the
// same touched-lists, every scan.

namespace {

/// RAII guard for the per-instance construction-time env latch.
struct ScopedEnv {
  explicit ScopedEnv(const char* key, const char* value) : key_(key) {
    ::setenv(key_, value, /*overwrite=*/1);
  }
  ~ScopedEnv() { ::unsetenv(key_); }
  const char* key_;
};

void appendBytes(std::vector<uint8_t>& out, const void* p, std::size_t n) {
  const auto* b = static_cast<const uint8_t*>(p);
  out.insert(out.end(), b, b + n);
}

// Field-wise raw-byte serialisers (bit-level compare without touching struct
// padding; float == would pass -0.0 vs 0.0 and fail NaN vs NaN).
void serialiseCell(std::vector<uint8_t>& out, const scovox::TsdfVoxel& v) {
  appendBytes(out, &v.distance, sizeof v.distance);
  appendBytes(out, &v.weight,   sizeof v.weight);
}
void serialiseCell(std::vector<uint8_t>& out, const scovox::BetaVoxel& v) {
  appendBytes(out, &v.a_occ,  sizeof v.a_occ);
  appendBytes(out, &v.a_free, sizeof v.a_free);
}
void serialiseCell(std::vector<uint8_t>& out, const scovox::DirVoxel& v) {
  appendBytes(out, &v.other, sizeof v.other);
  for (int i = 0; i < scovox::K_TOP; ++i) {
    appendBytes(out, &v.cnt[i], sizeof v.cnt[i]);
    appendBytes(out, &v.cls[i], sizeof v.cls[i]);
  }
}

struct CellDump {
  Bonxai::CoordT       c;
  std::vector<uint8_t> bytes;
  bool operator==(const CellDump& o) const { return c == o.c && bytes == o.bytes; }
};

/// Every active cell of a Bonxai grid, coord-sorted, values as raw bytes.
template <typename GridT>
std::vector<CellDump> dumpGrid(const GridT& g) {
  std::vector<CellDump> out;
  g.forEachCell([&](const auto& v, const Bonxai::CoordT& c) {
    CellDump d;
    d.c = c;
    serialiseCell(d.bytes, v);
    out.push_back(std::move(d));
  });
  std::sort(out.begin(), out.end(), [](const CellDump& a, const CellDump& b) {
    if (a.c.x != b.c.x) return a.c.x < b.c.x;
    if (a.c.y != b.c.y) return a.c.y < b.c.y;
    return a.c.z < b.c.z;
  });
  return out;
}

void expectCoordListsEqual(const std::vector<Bonxai::CoordT>& a,
                           const std::vector<Bonxai::CoordT>& b,
                           const char* which, int scan) {
  ASSERT_EQ(a.size(), b.size()) << which << " touched-list size, scan " << scan;
  for (std::size_t i = 0; i < a.size(); ++i) {
    EXPECT_TRUE(a[i] == b[i])
        << which << " touched[" << i << "] differs, scan " << scan;
  }
}

scovox::ScovoxMapSplit::Params farSkipParams(float band) {
  scovox::ScovoxMapSplit::Params p;
  p.resolution                    = 0.05;
  p.tsdf.sdf_trunc                = 0.15f;
  p.semsplit.kappa0               = 1.0f;
  p.semsplit.dirichlet_min_p_occ  = 0.5f;
  p.semsplit.num_classes          = 14;
  // Arms the skip's precondition: with a carve frame open, this flag makes
  // every applyCarveUpdate a guaranteed no-op (it does NOT fall back to the
  // immediate write path — the documented batch_free_carve trap).
  p.semsplit.batch_free_carve     = false;
  p.semsplit.semantic_band_length = band;
  return p;
}

// Fires one scan of rays into `fire`. Shared by the identity and the inertness
// tests so both exercise exactly the same geometry. `s` is the caller's LCG
// state (Numerical Recipes constants): identical ray sets on every run and
// platform, no <random> distribution drift.
//
// Every coordinate is drawn into a NAMED LOCAL before the Eigen constructor.
// Function-argument evaluation order is unspecified in C++, so three uf() calls
// passed directly would make the ray set compiler-defined — both arms would
// still see the same rays, but a mutation verdict recorded on one compiler
// would not describe another, and these rays carry exactly that evidence.
template <typename FireFn>
void fireScanRays(FireFn&& fire, int scan, uint32_t& s) {
  auto lcg = [&]() { s = s * 1664525u + 1013904223u; return s; };
  auto uf  = [&](float lo, float hi) {
    return lo + (hi - lo) * static_cast<float>(lcg() >> 8) / 16777216.0f;
  };

  // Random fan: long rays (up to 3 m at trunc 0.15) so the overwhelming
  // majority of visited voxels are far in front of the band — the voxels the
  // skip removes.
  for (int r = 0; r < 60; ++r) {
    const float ox = uf(-0.1f, 0.1f);
    const float oy = uf(-0.1f, 0.1f);
    const float oz = uf(-0.1f, 0.1f);
    const Eigen::Vector3f O(ox, oy, oz);
    const float dx = uf(-1.f, 1.f);
    const float dy = uf(-1.f, 1.f);
    const float dz = uf(-1.f, 1.f);
    Eigen::Vector3f dir(dx, dy, dz);
    if (dir.norm() < 1e-3f) dir = Eigen::Vector3f::UnitX();
    dir.normalize();
    const Eigen::Vector3f Hp = O + dir * uf(0.4f, 3.0f);
    std::vector<float> probs(14, 0.f);
    probs[lcg() % 14] = 1.f;
    const float q = uf(0.5f, 1.f);
    fire(O, Hp, &probs, q, /*dyn=*/(r % 7) == 3);
  }

  // Exact gate-boundary geometry: axis-aligned rays whose endpoints step in
  // h/2 = 0.0125 m increments across voxel centres and boundaries, so
  // band-edge voxels land exactly on the trunc/sem_band gate comparisons the
  // +1 guard ring protects (endpoint-on-centre also exercises the proj≈0
  // early return at the hit voxel).
  if (scan == 0) {
    std::vector<float> probs(14, 0.f);
    probs[2] = 1.f;
    for (int i = 0; i < 12; ++i) {
      const float x = 0.500f + 0.0125f * static_cast<float>(i);
      fire(Eigen::Vector3f(-1.5f, 0.025f, 0.025f),
           Eigen::Vector3f(x, 0.025f, 0.025f), &probs, 1.0f, false);
    }
    // Degenerate ray: the early-return path, no state on either side.
    fire(Eigen::Vector3f(0.2f, 0.2f, 0.2f),
         Eigen::Vector3f(0.2f, 0.2f, 0.2f), &probs, 1.0f, false);
  }
}

void runFarSkipIdentity(float band) {
  const auto p = farSkipParams(band);
  std::unique_ptr<scovox::ScovoxMapSplit> full;  // skip disabled (full walk)
  {
    ScopedEnv disable("SCOVOX_DISABLE_FAR_SKIP", "1");
    full = std::make_unique<scovox::ScovoxMapSplit>(p);
  }
  auto skip = std::make_unique<scovox::ScovoxMapSplit>(p);

  // Latch canary: the env kill-switch must have SPLIT the two instances. A
  // broken latch (env-name typo, lost member init) would otherwise turn the
  // comparison below into skip-vs-skip and stay silently green.
  ASSERT_TRUE(full->farSkipDisabled());
  ASSERT_FALSE(skip->farSkipDisabled());

  uint32_t s = 0xC0FFEEu;
  auto fire = [&](const Eigen::Vector3f& O, const Eigen::Vector3f& Hp,
                  const std::vector<float>* probs, float q, bool dyn) {
    full->integrateHit(O, Hp, probs, q, dyn);
    skip->integrateHit(O, Hp, probs, q, dyn);
  };

  for (int scan = 0; scan < 5; ++scan) {
    full->beginCarveFrame();
    skip->beginCarveFrame();

    fireScanRays(fire, scan, s);

    EXPECT_EQ(full->flushCarveFrame(), skip->flushCarveFrame())
        << "flush write count, scan " << scan;

    expectCoordListsEqual(full->drainTouchedTsdf(), skip->drainTouchedTsdf(),
                          "tsdf", scan);
    expectCoordListsEqual(full->drainTouchedBeta(), skip->drainTouchedBeta(),
                          "beta", scan);
    expectCoordListsEqual(full->drainTouchedDir(), skip->drainTouchedDir(),
                          "dir", scan);

    EXPECT_TRUE(dumpGrid(full->tsdf().grid()) == dumpGrid(skip->tsdf().grid()))
        << "TSDF grid diverged, scan " << scan << ", band " << band;
    EXPECT_TRUE(dumpGrid(full->semsplit().betaGrid()) ==
                dumpGrid(skip->semsplit().betaGrid()))
        << "Beta grid diverged, scan " << scan << ", band " << band;
    EXPECT_TRUE(dumpGrid(full->semsplit().dirGrid()) ==
                dumpGrid(skip->semsplit().dirGrid()))
        << "Dir grid diverged, scan " << scan << ", band " << band;
  }

  // Guard against a vacuous pass: the ray set must actually have written all
  // three grids.
  EXPECT_GT(full->tsdfVoxelCount(), 0u);
  EXPECT_GT(full->betaVoxelCount(), 0u);
  EXPECT_GT(full->dirVoxelCount(),  0u);
}

// The skip must be INERT unless ALL THREE of its preconditions hold. Each
// config below satisfies two and breaks the third, and each is chosen so that
// arming the skip anyway would be OBSERVABLE in a grid:
//
//   production    batch_free_carve=true — the SHIPPED configuration. Carve
//                 really stages, so skipped far voxels never reach the flush.
//                 This is the dangerous mutant: arming the skip here silently
//                 discards the bulk of the free-space carve.
//   spacecarving  space_carving=true — the TSDF upper gate is dropped for the
//                 carve front (see gate (1) in integrateHitFused), so far
//                 voxels DO take a band write and losing them shows in tsdf.
//   noframe       no carve frame open — applyCarveUpdate takes the immediate
//                 write path instead of returning early, so carve is real.
//
// These are TAUTOLOGICAL on correct code — far_skip is false in both arms, so
// both run the identical path — and that is the point: their entire value is
// that deleting any one conjunct from `far_skip` fails one of them, where
// FarSkipBitIdenticalToFullWalk on its own would still pass green.
void runFarSkipInert(const char* label,
                     bool space_carving,
                     bool batch_free_carve,
                     bool open_frame) {
  auto p = farSkipParams(/*band=*/0.0f);
  p.tsdf.space_carving        = space_carving;
  p.semsplit.batch_free_carve = batch_free_carve;

  std::unique_ptr<scovox::ScovoxMapSplit> full;  // skip disabled (full walk)
  {
    ScopedEnv disable("SCOVOX_DISABLE_FAR_SKIP", "1");
    full = std::make_unique<scovox::ScovoxMapSplit>(p);
  }
  auto skip = std::make_unique<scovox::ScovoxMapSplit>(p);

  const std::string l_tsdf = std::string(label) + " tsdf";
  const std::string l_beta = std::string(label) + " beta";
  const std::string l_dir  = std::string(label) + " dir";

  uint32_t s = 0xC0FFEEu;
  auto fire = [&](const Eigen::Vector3f& O, const Eigen::Vector3f& Hp,
                  const std::vector<float>* probs, float q, bool dyn) {
    full->integrateHit(O, Hp, probs, q, dyn);
    skip->integrateHit(O, Hp, probs, q, dyn);
  };

  for (int scan = 0; scan < 3; ++scan) {
    if (open_frame) {
      full->beginCarveFrame();
      skip->beginCarveFrame();
    }

    fireScanRays(fire, scan, s);

    if (open_frame) {
      EXPECT_EQ(full->flushCarveFrame(), skip->flushCarveFrame())
          << label << ": flush write count, scan " << scan;
    }

    expectCoordListsEqual(full->drainTouchedTsdf(), skip->drainTouchedTsdf(),
                          l_tsdf.c_str(), scan);
    expectCoordListsEqual(full->drainTouchedBeta(), skip->drainTouchedBeta(),
                          l_beta.c_str(), scan);
    expectCoordListsEqual(full->drainTouchedDir(), skip->drainTouchedDir(),
                          l_dir.c_str(), scan);

    EXPECT_TRUE(dumpGrid(full->tsdf().grid()) == dumpGrid(skip->tsdf().grid()))
        << label << ": TSDF grid diverged, scan " << scan;
    EXPECT_TRUE(dumpGrid(full->semsplit().betaGrid()) ==
                dumpGrid(skip->semsplit().betaGrid()))
        << label << ": Beta grid diverged, scan " << scan;
    EXPECT_TRUE(dumpGrid(full->semsplit().dirGrid()) ==
                dumpGrid(skip->semsplit().dirGrid()))
        << label << ": Dir grid diverged, scan " << scan;
  }

  // Non-vacuity: the carve must actually have written Beta voxels in this
  // config, else "inert" would be indistinguishable from "nothing happened"
  // and the mutant would have nothing to break.
  EXPECT_GT(full->betaVoxelCount(), 0u) << label << ": no carve happened";
  EXPECT_GT(full->tsdfVoxelCount(), 0u) << label << ": no TSDF written";
}

// ===========================================================================
// Far-voxel fast carve — bit-identity vs the full walk (SCOVOX_DISABLE_FAR_CARVE)
// ===========================================================================
//
// The batch_free_carve=true counterpart of the skip above: on the SHIPPED
// batched config, a voxel beyond far_thr can only take the carve branch — the
// TSDF and semantic-band gates provably fail out there — so integrateHitFused
// carves it directly with integer-only tests and skips the float body. The
// claim is the same BIT identity: a full-walk map (fast carve disabled via the
// per-instance env latch) and a fast-carving map fed identical rays must agree
// byte-for-byte on all three grids, the touched-lists, and the flush count.
//
// Like the far-skip identity, this cannot prove IN-PROCESS that the fast path
// actually armed (identity is exactly the absence of a difference); execution
// proof is the batched perf/digest pass (band_perf), mirroring how the skip's
// arming was proven on real KITTI.

void runFarCarveIdentity(float band) {
  auto p = farSkipParams(band);
  // The fast carve arms on the SHIPPED batched configuration — the exact flag
  // value that keeps the far-voxel SKIP disarmed (mutually exclusive by
  // construction, so exactly one fast path is under test here).
  p.semsplit.batch_free_carve = true;

  std::unique_ptr<scovox::ScovoxMapSplit> full;  // fast carve disabled
  {
    ScopedEnv disable("SCOVOX_DISABLE_FAR_CARVE", "1");
    full = std::make_unique<scovox::ScovoxMapSplit>(p);
  }
  auto fast = std::make_unique<scovox::ScovoxMapSplit>(p);

  // Latch canary: the env kill-switch must have SPLIT the two instances. A
  // broken latch (env-name typo, lost member init) would otherwise turn the
  // comparison below into fast-vs-fast and stay silently green.
  ASSERT_TRUE(full->farCarveDisabled());
  ASSERT_FALSE(fast->farCarveDisabled());

  uint32_t s = 0xC0FFEEu;
  auto fire = [&](const Eigen::Vector3f& O, const Eigen::Vector3f& Hp,
                  const std::vector<float>* probs, float q, bool dyn) {
    full->integrateHit(O, Hp, probs, q, dyn);
    fast->integrateHit(O, Hp, probs, q, dyn);
  };

  for (int scan = 0; scan < 5; ++scan) {
    full->beginCarveFrame();
    fast->beginCarveFrame();

    fireScanRays(fire, scan, s);

    // Short rays straddling depth == trunc (0.15): below it the walk starts
    // BEHIND the origin (walk_back = trunc > depth) and the fast carve must
    // stay disarmed — behind-origin voxels carve nothing in the exact body.
    // Brackets the `depth >= trunc` arming boundary from both sides.
    {
      auto lcg = [&]() { s = s * 1664525u + 1013904223u; return s; };
      auto uf  = [&](float lo, float hi) {
        return lo + (hi - lo) * static_cast<float>(lcg() >> 8) / 16777216.0f;
      };
      std::vector<float> probs(14, 0.f);
      probs[3] = 1.f;
      for (int r = 0; r < 10; ++r) {
        const float ox = uf(-0.2f, 0.2f);
        const float oy = uf(-0.2f, 0.2f);
        const float oz = uf(-0.2f, 0.2f);
        const Eigen::Vector3f O(ox, oy, oz);
        const float dx = uf(-1.f, 1.f);
        const float dy = uf(-1.f, 1.f);
        const float dz = uf(-1.f, 1.f);
        Eigen::Vector3f dir(dx, dy, dz);
        if (dir.norm() < 1e-3f) dir = Eigen::Vector3f::UnitZ();
        dir.normalize();
        const float len = uf(0.05f, 0.25f);
        const Eigen::Vector3f Hp = O + dir * len;
        fire(O, Hp, &probs, 0.9f, /*dyn=*/false);
      }
    }

    EXPECT_EQ(full->flushCarveFrame(), fast->flushCarveFrame())
        << "flush write count, scan " << scan << ", band " << band;

    expectCoordListsEqual(full->drainTouchedTsdf(), fast->drainTouchedTsdf(),
                          "tsdf", scan);
    expectCoordListsEqual(full->drainTouchedBeta(), fast->drainTouchedBeta(),
                          "beta", scan);
    expectCoordListsEqual(full->drainTouchedDir(), fast->drainTouchedDir(),
                          "dir", scan);

    EXPECT_TRUE(dumpGrid(full->tsdf().grid()) == dumpGrid(fast->tsdf().grid()))
        << "TSDF grid diverged, scan " << scan << ", band " << band;
    EXPECT_TRUE(dumpGrid(full->semsplit().betaGrid()) ==
                dumpGrid(fast->semsplit().betaGrid()))
        << "Beta grid diverged, scan " << scan << ", band " << band;
    EXPECT_TRUE(dumpGrid(full->semsplit().dirGrid()) ==
                dumpGrid(fast->semsplit().dirGrid()))
        << "Dir grid diverged, scan " << scan << ", band " << band;
  }

  // Non-vacuity: on this config the carve genuinely stages and flushes, and
  // the ray fan writes all three grids.
  EXPECT_GT(full->tsdfVoxelCount(), 0u);
  EXPECT_GT(full->betaVoxelCount(), 0u);
  EXPECT_GT(full->dirVoxelCount(),  0u);
}

// The fast carve must be INERT when space_carving=true — the ONE arming
// conjunct whose deletion is observable: space carving drops the TSDF upper
// gate, far front voxels DO take a band write, and a fast path that returns
// right after carving loses those writes from the TSDF grid.
//
// The remaining conjuncts are conservative scoping, not correctness
// load-bearing, so no inert config can kill their deletion mutants — recorded
// here so nobody hunts for the missing tests: with batch_free_carve=false the
// far SKIP fires first and staged/immediate carving is a no-op either way;
// with no frame open the immediate-path carve is exactly what the exact body
// would do beyond far_thr (latch semantics kept verbatim); with depth < trunc
// no walk voxel can exceed far_thr at all (walk span < far_thr·res each way).
void runFarCarveInertWhenSpaceCarving() {
  auto p = farSkipParams(/*band=*/0.0f);
  p.semsplit.batch_free_carve = true;
  p.tsdf.space_carving        = true;

  std::unique_ptr<scovox::ScovoxMapSplit> full;  // fast carve disabled
  {
    ScopedEnv disable("SCOVOX_DISABLE_FAR_CARVE", "1");
    full = std::make_unique<scovox::ScovoxMapSplit>(p);
  }
  auto fast = std::make_unique<scovox::ScovoxMapSplit>(p);

  uint32_t s = 0xC0FFEEu;
  auto fire = [&](const Eigen::Vector3f& O, const Eigen::Vector3f& Hp,
                  const std::vector<float>* probs, float q, bool dyn) {
    full->integrateHit(O, Hp, probs, q, dyn);
    fast->integrateHit(O, Hp, probs, q, dyn);
  };

  for (int scan = 0; scan < 3; ++scan) {
    full->beginCarveFrame();
    fast->beginCarveFrame();

    fireScanRays(fire, scan, s);

    EXPECT_EQ(full->flushCarveFrame(), fast->flushCarveFrame())
        << "spacecarving: flush write count, scan " << scan;

    expectCoordListsEqual(full->drainTouchedTsdf(), fast->drainTouchedTsdf(),
                          "spacecarving tsdf", scan);
    expectCoordListsEqual(full->drainTouchedBeta(), fast->drainTouchedBeta(),
                          "spacecarving beta", scan);
    expectCoordListsEqual(full->drainTouchedDir(), fast->drainTouchedDir(),
                          "spacecarving dir", scan);

    EXPECT_TRUE(dumpGrid(full->tsdf().grid()) == dumpGrid(fast->tsdf().grid()))
        << "spacecarving: TSDF grid diverged, scan " << scan;
    EXPECT_TRUE(dumpGrid(full->semsplit().betaGrid()) ==
                dumpGrid(fast->semsplit().betaGrid()))
        << "spacecarving: Beta grid diverged, scan " << scan;
    EXPECT_TRUE(dumpGrid(full->semsplit().dirGrid()) ==
                dumpGrid(fast->semsplit().dirGrid()))
        << "spacecarving: Dir grid diverged, scan " << scan;
  }

  EXPECT_GT(full->tsdfVoxelCount(), 0u) << "spacecarving: no TSDF written";
  EXPECT_GT(full->betaVoxelCount(), 0u) << "spacecarving: no carve happened";
}

}  // namespace

TEST(ScovoxMapSplitFarSkip, FarSkipBitIdenticalToFullWalk) {
  runFarSkipIdentity(/*band=*/0.0f);   // endpoint-only semantics (shipped)
  runFarSkipIdentity(/*band=*/0.30f);  // band > trunc: exercises the max() arm
}

TEST(ScovoxMapSplitFarSkip, InertUnlessEveryPreconditionHolds) {
  //                 label           space_carving  batch_free_carve  frame
  runFarSkipInert("production",      false,         true,             true);
  runFarSkipInert("spacecarving",    true,          false,            true);
  runFarSkipInert("noframe",         false,         false,            false);
}

TEST(ScovoxMapSplitFarCarve, FarCarveBitIdenticalToFullWalk) {
  runFarCarveIdentity(/*band=*/0.0f);   // endpoint-only semantics (shipped)
  runFarCarveIdentity(/*band=*/0.30f);  // band > trunc: exercises the max() arm
}

TEST(ScovoxMapSplitFarCarve, InertWhenSpaceCarving) {
  runFarCarveInertWhenSpaceCarving();
}

// ===========================================================================
// Inlined voxel centre — bit-identity against the coordToPos reference
// ===========================================================================
//
// `integrateHitFused` computes the voxel centre inline as
// `(float)((double)c.x * res) + h` instead of calling `grid.coordToPos(c)`.
// The split walker (`TsdfMap::visit`) still calls coordToPos, so it is the
// REFERENCE implementation of that expression and the two must agree bitwise.
//
// This needs to compare TSDF *values*, not the voxel counts and grid bytes
// AxisAlignedParityWithSplitWalker settles for: the narrowing order is worth a
// single ulp of the centre, which moves `sdf` and therefore the stored
// `distance` without necessarily allocating or freeing a single voxel. Counts
// cannot see it; bytes can.
//
// The affected coords follow NO simple pattern — they are wherever rounding
// c·res to float and then adding h lands on a different float from rounding
// the whole (c·res + h) once. At res=0.05 that is 244 of the 801 coords in
// [-400, 400]; in 0..80 it is exactly
//   {3,4,6,13,14,18,19,21,26,31,36,43,44,48,49,53,54,58,59,63,64,68,69,73,74,78,79}
// which is why the ray below has to be aimed at specific coords rather than
// anywhere convenient.
//
// Axis-aligned rays only — off-axis, the fused walker's longer walk-back gives
// Bresenham a different start and the two may legitimately pick different
// voxels at sub-voxel boundaries (see MultiRayBandIdentity's 20% tolerance).

TEST(ScovoxMapSplitFusedWalker, AxisAlignedTsdfValuesBitIdenticalToSplitWalker) {
  auto p_fused = splitParams(); p_fused.fused_walker = true;
  auto p_split = splitParams(); p_split.fused_walker = false;
  scovox::ScovoxMapSplit m_fused(p_fused);
  scovox::ScovoxMapSplit m_split(p_split);

  std::vector<float> probs{0.f, 1.f, 0.f, 0.f};
  // EXACTLY the ray AxisAlignedParityWithSplitWalker establishes voxel-set
  // parity on — including its carve_band origin truncation, which is what makes
  // the parity hold. With the truncated origin the ray is SHORTER than sdf_trunc,
  // so the fused walker's `walk_back = max(depth, trunc)` collapses to `trunc`
  // and both walkers span exactly [hit − trunc, hit + trunc]. Fired from the raw
  // origin instead, the fused walker walks the whole 0.5 m ray and admits
  // `sdf ≤ trunc + h` while the split walker starts at `hit − trunc`, and the two
  // write different voxel SETS (measured: 6 vs 7 on this ray, 24 vs 27 over a
  // 24-ray multi-axis fan) — a walk-range difference that would swamp the ulp
  // this test is here to see.
  // TWO further constraints, both load-bearing — get either wrong and the test
  // passes vacuously (verified by mutating the walker back to the all-double
  // form, which this ray catches and the obvious ones do not):
  //
  //  1. The band must COVER a coord where the two narrowing orders actually
  //     disagree. At res=0.05 those are x ∈ {3,4,6,13,14,18,19,21,26,...} —
  //     NOT every coord. Endpoint 0.325 walks [0.175, 0.475] = coords 3..9,
  //     covering 3, 4 and 6; endpoint 0.50 walks coords 7..12 and covers NONE.
  //  2. The ray must run through voxel CENTRES in y and z. Off-centre, every
  //     voxel picks up 0.025 m offsets in y and z whose squares dominate the
  //     ulp in x, and the difference vanishes in the norm.
  //
  // HOW the mutant actually loses voxels (6 written → 4), measured rather than
  // assumed: it is the proj≈0 degeneracy guard, NOT the `sdf ≤ trunc + h` band
  // gate. Under the all-double order the centres land on
  //   coord 4 → 0.22499999403953552 == the truncated origin co.x, bit for bit
  //   coord 6 → 0.32499998807907104 == the endpoint Hp.x, bit for bit
  // (and y/z already sit exactly on h), so v_voxel_origin and v_point_voxel
  // become exactly the zero vector, proj is exactly 0, and both voxels take the
  // `fabs(proj) < 1e-12` early return before any band update. The current order
  // puts each centre one ulp off, both survive, and the band writes 6.
  const Eigen::Vector3f O(0.000f, 0.025f, 0.025f);
  const Eigen::Vector3f Hp(0.325f, 0.025f, 0.025f);
  const Eigen::Vector3f co = truncateOrigin(O, Hp, /*carve_band=*/0.10f);
  m_fused.integrateHit(co, Hp, &probs, /*quality=*/1.0f);
  m_split.integrateHit(co, Hp, &probs, /*quality=*/1.0f);

  const auto df = dumpGrid(m_fused.tsdf().grid());
  const auto ds = dumpGrid(m_split.tsdf().grid());
  ASSERT_GT(df.size(), 0u) << "vacuous: no TSDF voxels written";
  EXPECT_TRUE(df == ds)
      << "fused TSDF state diverged from the coordToPos reference "
      << "(fused=" << df.size() << " split=" << ds.size() << " voxels)";
}
