/// @file
/// @brief Differential gate for CarveStage (carve_stage.hpp) — the
/// leaf-block-keyed batched-carve accumulator that replaced the
/// `unordered_map<CoordT,float>` + per-scan `std::sort` staging in
/// `SemSplitMap` (efficiency_audit_2026_08_26.md, item 1).
///
/// Pinned invariants (what flushCarveFrame relies on):
///   - per-voxel value = MAX staged weight (order-independent);
///   - staged-voxel SET identical to the reference (⇒ identical flush count);
///   - block visit order = the RETIRED implementation's sort order:
///     ascending lexicographic (x>>lb, y>>lb, z>>lb), each block visited
///     exactly once (⇒ identical Beta root-map first-touch order ⇒ identical
///     serialized bytes). Within-block order is NOT pinned (structurally
///     invisible — dense leaf arrays + drainTouchedBeta's sortUnique), so the
///     comparison here is block-granular.
///   - beginFrame isolates frames completely (capacity reuse leaks nothing);
///   - negative coordinates key/reconstruct exactly (floor semantics);
///   - leaf_bits=1 (8-cell block, one partial mask word) works;
///   - index growth past the initial capacity preserves every entry.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

#include "scovox/carve_stage.hpp"

namespace {

using Bonxai::CoordT;
using Entry = std::pair<CoordT, float>;

// ---------------------------------------------------------------------------
// Reference: the retired SemSplitMap staging, verbatim semantics —
// unordered_map max-accumulate, then the old flush comparator.
// ---------------------------------------------------------------------------

struct Reference {
  std::unordered_map<CoordT, float> stage;

  void add(const CoordT& c, float w) {
    auto it = stage.find(c);
    if (it == stage.end()) stage.emplace(c, w);
    else if (w > it->second) it->second = w;
  }

  std::vector<Entry> ordered(int lb) const {
    std::vector<Entry> items(stage.begin(), stage.end());
    std::sort(items.begin(), items.end(),
              [lb](const Entry& a, const Entry& b) {
                const std::array<int32_t, 3> ba{a.first.x >> lb, a.first.y >> lb, a.first.z >> lb};
                const std::array<int32_t, 3> bb{b.first.x >> lb, b.first.y >> lb, b.first.z >> lb};
                return ba < bb;
              });
    return items;
  }
};

std::vector<Entry> collect(scovox::CarveStage& cs) {
  std::vector<Entry> out;
  cs.forEachStagedBlockOrdered([&](const CoordT& c, float w) {
    out.emplace_back(c, w);
  });
  return out;
}

std::array<int32_t, 3> blockKey(const CoordT& c, int lb) {
  return {c.x >> lb, c.y >> lb, c.z >> lb};
}

/// Block-granular equivalence: identical (coord → weight) set AND identical
/// block visit sequence, each block exactly once. Within-block order is
/// deliberately NOT compared (unpinned in both implementations).
void expectBlockEquivalent(std::vector<Entry> got, std::vector<Entry> ref, int lb) {
  // (b) block visit sequences as (key, run length) pairs. Comparing run
  // LENGTHS too (not just collapsed keys) means both a non-consecutive block
  // revisit AND a same-key split across two ADJACENT runs (e.g. a duplicate
  // slot for one block) surface as a mismatch.
  auto blockSeq = [lb](const std::vector<Entry>& v) {
    std::vector<std::pair<std::array<int32_t, 3>, std::size_t>> seq;
    for (const Entry& e : v) {
      const auto k = blockKey(e.first, lb);
      if (seq.empty() || seq.back().first != k) seq.emplace_back(k, 0u);
      ++seq.back().second;
    }
    return seq;
  };
  EXPECT_EQ(blockSeq(got), blockSeq(ref));

  // (a) exact (coord, weight) multiset — coords are unique, weights are the
  // max of identical float inputs on both sides, so equality is bit-exact.
  auto byCoord = [](const Entry& a, const Entry& b) {
    if (a.first.x != b.first.x) return a.first.x < b.first.x;
    if (a.first.y != b.first.y) return a.first.y < b.first.y;
    return a.first.z < b.first.z;
  };
  std::sort(got.begin(), got.end(), byCoord);
  std::sort(ref.begin(), ref.end(), byCoord);
  ASSERT_EQ(got.size(), ref.size());
  for (std::size_t i = 0; i < got.size(); ++i) {
    EXPECT_EQ(got[i].first, ref[i].first) << "coord mismatch at " << i;
    EXPECT_EQ(got[i].second, ref[i].second)
        << "weight mismatch at (" << got[i].first.x << "," << got[i].first.y
        << "," << got[i].first.z << ")";
  }
}

// ---------------------------------------------------------------------------
// Deterministic input generator (no rand()/time seeds). Emits short axis
// walks from pseudo-random starts — ray-coherent runs that cross block
// boundaries, exercising the last-leaf cache — with revisits for the
// max-accumulate path, spanning negative and positive coordinates.
// ---------------------------------------------------------------------------

struct Lcg {
  uint64_t s;
  explicit Lcg(uint64_t seed) : s(seed) {}
  uint64_t next() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return s >> 16;
  }
  int32_t coord(int32_t half_range) {
    return int32_t(next() % uint64_t(2 * half_range)) - half_range;
  }
  float weight() { return 0.05f + float(next() % 997u) * 0.01f; }
};

template <typename AddFn>
void generate(uint64_t seed, int n_walks, int32_t half_range, AddFn&& add) {
  Lcg rng(seed);
  for (int wlk = 0; wlk < n_walks; ++wlk) {
    CoordT    c{rng.coord(half_range), rng.coord(half_range), rng.coord(half_range)};
    const int axis = int(rng.next() % 3u);
    const int step = (rng.next() & 1u) ? 1 : -1;
    const int len  = 4 + int(rng.next() % 12u);
    for (int i = 0; i < len; ++i) {
      add(c, rng.weight());
      c[std::size_t(axis)] += step;
    }
  }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(CarveStage, EmptyFrameVisitsNothing) {
  scovox::CarveStage cs(3);
  EXPECT_TRUE(cs.empty());
  EXPECT_EQ(cs.size(), 0u);
  EXPECT_EQ(cs.blockCount(), 0u);
  EXPECT_TRUE(collect(cs).empty());

  cs.beginFrame();  // beginFrame on an already-empty stage is a no-op
  EXPECT_TRUE(collect(cs).empty());
}

TEST(CarveStage, MaxAccumulatePerVoxel) {
  scovox::CarveStage cs(3);
  const CoordT c{5, -3, 12};
  cs.add(c, 1.0f);
  cs.add(c, 3.0f);
  cs.add(c, 2.0f);  // lower than the running max — must not regress it
  EXPECT_EQ(cs.size(), 1u);
  EXPECT_EQ(cs.blockCount(), 1u);
  const auto got = collect(cs);
  ASSERT_EQ(got.size(), 1u);
  EXPECT_EQ(got[0].first, c);
  EXPECT_EQ(got[0].second, 3.0f);
}

TEST(CarveStage, NegativeCoordsKeyAndReconstructExactly) {
  // Floor-division block keys around the origin at lb=3: −1 and −8 share
  // block −1; −9 is block −2; 0 and 7 share block 0. Every coord must round-
  // trip bit-exactly through key + in-block offset.
  scovox::CarveStage cs(3);
  Reference          ref;
  const CoordT probes[] = {
      {-1, -1, -1}, {-8, -8, -8}, {-9, -1, -1}, {0, 0, 0},
      {7, 7, 7},    {-1, 7, -8},  {8, -9, 0},   {-64, 63, -1},
  };
  float w = 0.5f;
  for (const CoordT& c : probes) {
    cs.add(c, w);
    ref.add(c, w);
    w += 0.25f;
  }
  expectBlockEquivalent(collect(cs), ref.ordered(3), 3);
}

TEST(CarveStage, DifferentialVsRetiredStaging_Lb3) {
  const int lb = 3;
  scovox::CarveStage cs(lb);
  Reference          ref;
  generate(/*seed=*/2026'08'26ULL, /*n_walks=*/4000, /*half_range=*/400,
           [&](const CoordT& c, float w) { cs.add(c, w); ref.add(c, w); });
  EXPECT_EQ(cs.size(), ref.stage.size());
  expectBlockEquivalent(collect(cs), ref.ordered(lb), lb);
}

TEST(CarveStage, DifferentialVsRetiredStaging_Lb1) {
  // lb=1: 8-cell blocks, one PARTIAL 64-bit mask word — exercises the
  // ceil(cells/64) word count and the small-block reconstruction.
  const int lb = 1;
  scovox::CarveStage cs(lb);
  Reference          ref;
  generate(/*seed=*/0xC0FFEEULL, /*n_walks=*/1500, /*half_range=*/60,
           [&](const CoordT& c, float w) { cs.add(c, w); ref.add(c, w); });
  EXPECT_EQ(cs.size(), ref.stage.size());
  expectBlockEquivalent(collect(cs), ref.ordered(lb), lb);
}

TEST(CarveStage, IndexGrowthPreservesEverything) {
  // Force multiple doublings of the block index (initial capacity 1024, grown
  // when the block count would exceed half of it): spread walks over a range
  // that yields well over 512 distinct leaf blocks.
  const int lb = 3;
  scovox::CarveStage cs(lb);
  Reference          ref;
  generate(/*seed=*/42ULL, /*n_walks=*/6000, /*half_range=*/4000,
           [&](const CoordT& c, float w) { cs.add(c, w); ref.add(c, w); });
  ASSERT_GT(cs.blockCount(), 1024u);  // the growth path definitely ran
  EXPECT_EQ(cs.size(), ref.stage.size());
  expectBlockEquivalent(collect(cs), ref.ordered(lb), lb);
}

TEST(CarveStage, FrameReuseLeaksNothing) {
  const int lb = 3;
  scovox::CarveStage cs(lb);

  // Frame A: large population (forces slot-pool and index capacity).
  generate(/*seed=*/7ULL, /*n_walks=*/3000, /*half_range=*/300,
           [&](const CoordT& c, float w) { cs.add(c, w); });
  const std::size_t frame_a_size = cs.size();
  ASSERT_GT(frame_a_size, 0u);

  // Frame B: smaller, PARTIALLY overlapping coordinate range — reused slots
  // must not resurrect frame-A voxels or weights.
  cs.beginFrame();
  EXPECT_TRUE(cs.empty());
  Reference ref;
  generate(/*seed=*/7ULL, /*n_walks=*/500, /*half_range=*/350,
           [&](const CoordT& c, float w) { cs.add(c, w); ref.add(c, w); });
  EXPECT_LT(cs.size(), frame_a_size);
  EXPECT_EQ(cs.size(), ref.stage.size());
  expectBlockEquivalent(collect(cs), ref.ordered(lb), lb);

  // And a third frame, back to back, still clean.
  cs.beginFrame();
  Reference ref2;
  generate(/*seed=*/99ULL, /*n_walks=*/800, /*half_range=*/200,
           [&](const CoordT& c, float w) { cs.add(c, w); ref2.add(c, w); });
  EXPECT_EQ(cs.size(), ref2.stage.size());
  expectBlockEquivalent(collect(cs), ref2.ordered(lb), lb);
}

}  // namespace
