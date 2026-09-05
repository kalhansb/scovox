// Unit tests for scovox::emitSnapshotOrTouched and scovox::gateAndRefresh —
// the two idioms extracted from SCovoxNode::publishBinaryMap, which repeated
// the traversal four times (TSDF, fine TSDF, Beta, Dir) and the change gate
// twice (Beta, Dir). Both now have exactly one definition, and these tests are
// what that definition is worth: the publish path itself is not compiled by any
// test, so before the extraction the gate's load-bearing rules lived only in a
// comment that the second copy had to restate.
//
// The rule these pin hardest is the setValue one. `*value(c, true) = v` looks
// equivalent and is not: the miss lookup immediately above caches a null leaf
// pointer for the inner key, and value(c, true) skips the leaf refresh on a
// same-key hit, so it hands back nullptr even with create_if_missing. A voxel's
// FIRST emit on a delta tick takes exactly that path, so the bad form
// dereferences a null pointer there — it does not degrade, it crashes the
// publisher, and it does so only once a real subscriber is attached.

#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include <bonxai/bonxai.hpp>

#include "scovox/beta_voxel.hpp"
#include "scovox/node_utils.hpp"

namespace {

using Bonxai::CoordT;
using scovox::BetaVoxel;

constexpr double kRes = 0.10;
constexpr int kInnerBits = 2;
constexpr int kLeafBits = 3;

BetaVoxel beta(float a_occ, float a_free) {
  BetaVoxel v;
  v.a_occ = a_occ;
  v.a_free = a_free;
  return v;
}

bool sameCoord(const CoordT& a, const CoordT& b) {
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

// Stand-in for the map's touched-coord bookkeeping. drain() returns by
// reference and empties the set, exactly as SemSplitMap::drainTouchedBeta does;
// the counters let a test assert WHICH arm ran, not just what it emitted.
struct TouchedSet {
  std::vector<CoordT> coords;
  int drains = 0;
  int clears = 0;
  const std::vector<CoordT>& drain() {
    ++drains;
    scratch_ = coords;
    coords.clear();
    return scratch_;
  }
  void clear() {
    ++clears;
    coords.clear();
  }

 private:
  std::vector<CoordT> scratch_;
};

struct Fixture {
  Bonxai::VoxelGrid<BetaVoxel> live{kRes, kInnerBits, kLeafBits};
  TouchedSet touched;
  std::vector<std::pair<CoordT, BetaVoxel>> emitted;

  void put(const CoordT& c, const BetaVoxel& v) {
    auto acc = live.createAccessor();
    acc.setValue(c, v);
    touched.coords.push_back(c);
  }

  void run(bool snapshot) {
    scovox::emitSnapshotOrTouched(
        snapshot, live,
        [&]() -> const std::vector<CoordT>& { return touched.drain(); },
        [&] { touched.clear(); },
        [&](const BetaVoxel& v, const CoordT& c) { emitted.emplace_back(c, v); });
  }
};

// ---------------------------------------------------------------- traversal

TEST(EmitSnapshotOrTouched, SnapshotEmitsEveryCellAndClearsTouched) {
  Fixture f;
  f.put({0, 0, 0}, beta(3.f, 1.f));
  f.put({1, 0, 0}, beta(4.f, 1.f));
  f.put({40, 7, 3}, beta(5.f, 1.f));  // different root block
  f.touched.coords.clear();           // a snapshot must not depend on this
  f.touched.coords.push_back({0, 0, 0});

  f.run(true);

  EXPECT_EQ(f.emitted.size(), 3u);
  EXPECT_EQ(f.touched.clears, 1);
  EXPECT_EQ(f.touched.drains, 0);
  // Leaving the touched set populated would re-emit these on the next delta.
  EXPECT_TRUE(f.touched.coords.empty());
}

TEST(EmitSnapshotOrTouched, DeltaEmitsOnlyTouchedCoords) {
  Fixture f;
  f.put({0, 0, 0}, beta(3.f, 1.f));
  f.put({1, 0, 0}, beta(4.f, 1.f));
  f.touched.coords.clear();
  f.touched.coords.push_back({1, 0, 0});

  f.run(false);

  ASSERT_EQ(f.emitted.size(), 1u);
  EXPECT_TRUE(sameCoord(f.emitted[0].first, CoordT{1, 0, 0}));
  EXPECT_FLOAT_EQ(f.emitted[0].second.a_occ, 4.f);
  EXPECT_EQ(f.touched.drains, 1);
  EXPECT_EQ(f.touched.clears, 0);  // drain already emptied it
  EXPECT_TRUE(f.touched.coords.empty());
}

TEST(EmitSnapshotOrTouched, DeltaSkipsTouchedCoordWithNoLiveVoxel) {
  Fixture f;
  f.put({0, 0, 0}, beta(3.f, 1.f));
  f.touched.coords.push_back({9, 9, 9});  // touched, never written

  f.run(false);

  ASSERT_EQ(f.emitted.size(), 1u);
  EXPECT_TRUE(sameCoord(f.emitted[0].first, CoordT{0, 0, 0}));
}

TEST(EmitSnapshotOrTouched, EmptyGridSnapshotStillClears) {
  Fixture f;
  f.touched.coords.push_back({2, 2, 2});
  f.run(true);
  EXPECT_TRUE(f.emitted.empty());
  EXPECT_EQ(f.touched.clears, 1);
  EXPECT_TRUE(f.touched.coords.empty());
}

// --------------------------------------------------------------- change gate

using GateGrid = Bonxai::VoxelGrid<BetaVoxel>;
using StampGrid = Bonxai::VoxelGrid<double>;

// Deliberately ASYMMETRIC, mirroring the node's real predicates: both
// betaChangedSinceEmit and dirChangedSinceEmit read the LAST-EMITTED voxel as
// the baseline and ask whether the live one has grown past a threshold above
// it. A symmetric stand-in ("did it move at all") would pass just as happily
// with the two arguments swapped — and since both call sites now forward
// through a generic `const auto&` wrapper, a swap would compile silently and
// invert the gate. GateDropsWhenEvidenceFalls is what makes the order a test.
auto grewByOne = [](const BetaVoxel& last, const BetaVoxel& now) {
  return now.a_occ > last.a_occ + 1.f;
};

struct GateFixture {
  GateGrid gate{kRes, kInnerBits, kLeafBits};
  StampGrid stamps{kRes, kInnerBits, kLeafBits};
  std::optional<GateGrid::Accessor> gacc;
  std::optional<StampGrid::Accessor> tacc;

  void arm(bool with_stamps) {
    gacc.emplace(gate.createAccessor());
    if (with_stamps) tacc.emplace(stamps.createAccessor());
  }
  bool call(const CoordT& c, const BetaVoxel& v, bool snapshot, double t_now) {
    return scovox::gateAndRefresh(gacc, tacc, c, v, snapshot, t_now, grewByOne);
  }
};

TEST(GateAndRefresh, NoGateGridEmitsUnconditionally) {
  GateFixture f;  // gacc left disengaged: gating is off
  EXPECT_TRUE(f.call({0, 0, 0}, beta(3.f, 1.f), false, 10.0));
  EXPECT_TRUE(f.call({0, 0, 0}, beta(3.f, 1.f), false, 10.0));
  auto acc = f.gate.createAccessor();
  EXPECT_EQ(acc.value({0, 0, 0}, false), nullptr);
}

// The regression that motivated extracting this: the FIRST emit of a voxel is
// a gate miss, and the refresh has to create the entry through setValue.
TEST(GateAndRefresh, FirstEmitCreatesTheGateEntry) {
  GateFixture f;
  f.arm(true);
  EXPECT_TRUE(f.call({5, 6, 7}, beta(3.f, 1.f), /*snapshot=*/false, 10.0));

  auto acc = f.gate.createAccessor();
  const auto* g = acc.value({5, 6, 7}, false);
  ASSERT_NE(g, nullptr);
  EXPECT_FLOAT_EQ(g->a_occ, 3.f);
  auto sacc = f.stamps.createAccessor();
  const auto* t = sacc.value({5, 6, 7}, false);
  ASSERT_NE(t, nullptr);
  EXPECT_DOUBLE_EQ(*t, 10.0);
}

TEST(GateAndRefresh, DeltaDropsUnchangedVoxelAndLeavesStampAlone) {
  GateFixture f;
  f.arm(true);
  ASSERT_TRUE(f.call({1, 1, 1}, beta(3.f, 1.f), false, 10.0));

  EXPECT_FALSE(f.call({1, 1, 1}, beta(3.f, 1.f), false, 25.0));

  auto sacc = f.stamps.createAccessor();
  const auto* t = sacc.value({1, 1, 1}, false);
  ASSERT_NE(t, nullptr);
  // A dropped voxel was not emitted, so its last-emit time must not advance —
  // otherwise the heartbeat would count it as freshly re-pinned.
  EXPECT_DOUBLE_EQ(*t, 10.0);
}

TEST(GateAndRefresh, DeltaEmitsChangedVoxelAndAdvancesGateAndStamp) {
  GateFixture f;
  f.arm(true);
  ASSERT_TRUE(f.call({1, 1, 1}, beta(3.f, 1.f), false, 10.0));

  EXPECT_TRUE(f.call({1, 1, 1}, beta(9.f, 1.f), false, 25.0));

  auto acc = f.gate.createAccessor();
  EXPECT_FLOAT_EQ(acc.value({1, 1, 1}, false)->a_occ, 9.f);
  auto sacc = f.stamps.createAccessor();
  EXPECT_DOUBLE_EQ(*sacc.value({1, 1, 1}, false), 25.0);
}

TEST(GateAndRefresh, SnapshotBypassesTheChangeCheckButStillRefreshes) {
  GateFixture f;
  f.arm(true);
  ASSERT_TRUE(f.call({2, 0, 0}, beta(3.f, 1.f), false, 10.0));

  // Identical value: a delta tick would drop it, a snapshot must not.
  EXPECT_TRUE(f.call({2, 0, 0}, beta(3.f, 1.f), /*snapshot=*/true, 40.0));

  auto sacc = f.stamps.createAccessor();
  EXPECT_DOUBLE_EQ(*sacc.value({2, 0, 0}, false), 40.0);
}

TEST(GateAndRefresh, StampsOptionalWhenHeartbeatDisarmed) {
  GateFixture f;
  f.arm(false);  // gate armed, stamp twin not allocated
  EXPECT_TRUE(f.call({3, 3, 3}, beta(3.f, 1.f), false, 10.0));
  EXPECT_FALSE(f.call({3, 3, 3}, beta(3.f, 1.f), false, 10.0));
  auto acc = f.gate.createAccessor();
  EXPECT_NE(acc.value({3, 3, 3}, false), nullptr);
}

// F1: pins the predicate's argument order. Evidence FALLING must not re-emit.
// Swap the two arguments inside gateAndRefresh and this is the test that goes
// red — 9 > 3 + 1 would read as growth.
TEST(GateAndRefresh, GateDropsWhenEvidenceFalls) {
  GateFixture f;
  f.arm(true);
  ASSERT_TRUE(f.call({4, 4, 4}, beta(9.f, 1.f), false, 10.0));

  EXPECT_FALSE(f.call({4, 4, 4}, beta(3.f, 1.f), false, 25.0));

  auto acc = f.gate.createAccessor();
  // Dropped means the gate still holds the last value actually emitted.
  EXPECT_FLOAT_EQ(acc.value({4, 4, 4}, false)->a_occ, 9.f);
}

// F3: the dominant path on a real snapshot tick is a coord with no gate entry
// yet (fresh subscriber, first emit of every voxel). The other snapshot test
// pre-seeds the entry with a delta call, so it never covers creation.
TEST(GateAndRefresh, SnapshotCreatesAMissingGateEntry) {
  GateFixture f;
  f.arm(true);
  EXPECT_TRUE(f.call({8, 1, 2}, beta(3.f, 1.f), /*snapshot=*/true, 55.0));

  auto acc = f.gate.createAccessor();
  const auto* g = acc.value({8, 1, 2}, false);
  ASSERT_NE(g, nullptr);
  EXPECT_FLOAT_EQ(g->a_occ, 3.f);
  EXPECT_DOUBLE_EQ(*f.stamps.createAccessor().value({8, 1, 2}, false), 55.0);
}

// ------------------------------------------------------------- the two, wired
// Neither helper is used alone in the node: the Beta and Dir sections pass an
// emit lambda whose first act is to consult the gate and return early. That
// composition is the thing publishBinaryMap actually contains, and nothing
// above exercises it — an emit that returns early still has to leave the
// traversal's own bookkeeping (drain/clear) correct.
TEST(PublishPath, GatedEmitDropsUnchangedVoxelsButStillDrainsTheTouchedSet) {
  Fixture f;
  GateFixture g;
  g.arm(true);
  f.put({0, 0, 0}, beta(3.f, 1.f));
  f.put({1, 0, 0}, beta(3.f, 1.f));

  auto gated_run = [&](bool snapshot, double t_now) {
    scovox::emitSnapshotOrTouched(
        snapshot, f.live,
        [&]() -> const std::vector<CoordT>& { return f.touched.drain(); },
        [&] { f.touched.clear(); },
        [&](const BetaVoxel& v, const CoordT& c) {
          if (!g.call(c, v, snapshot, t_now)) return;
          f.emitted.emplace_back(c, v);
        });
  };

  gated_run(false, 10.0);          // first sight of both: gate misses, both emit
  ASSERT_EQ(f.emitted.size(), 2u);

  f.emitted.clear();
  f.touched.coords.push_back({0, 0, 0});
  f.touched.coords.push_back({1, 0, 0});
  gated_run(false, 20.0);          // unchanged: gate drops both

  EXPECT_TRUE(f.emitted.empty());
  EXPECT_EQ(f.touched.drains, 2);
  // The drop happens INSIDE emit, so the traversal must still have consumed the
  // touched set. Leaving it populated would re-offer these coords forever.
  EXPECT_TRUE(f.touched.coords.empty());
}

}  // namespace
