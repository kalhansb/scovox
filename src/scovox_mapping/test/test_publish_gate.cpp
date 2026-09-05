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
#include <utility>
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

// Stand-in for the map's touched-coord bookkeeping. drain() SWAPS the set into
// a member scratch and returns that by reference — the same shape as
// SemSplitMap::drainTouchedBeta and TsdfMap::drainTouched, and the reason the
// helper may hold the returned reference across its whole loop. Assigning
// instead of swapping would also satisfy the helper's static_assert while
// copying a frame of coords per tick, which is the cost that assert exists to
// prevent, so the stand-in must not model it that way.
// The counters let a test assert WHICH arm ran, not just what it emitted.
struct TouchedSet {
  std::vector<CoordT> coords;
  int drains = 0;
  int clears = 0;
  const std::vector<CoordT>& drain() {
    ++drains;
    scratch_.clear();
    std::swap(coords, scratch_);
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

// Snapshot traversal yields no defined order, so a test that cares WHICH voxel
// carried WHICH value looks the coord up rather than indexing.
const BetaVoxel* emittedAt(const Fixture& f, const CoordT& c) {
  for (const auto& [coord, v] : f.emitted)
    if (sameCoord(coord, c)) return &v;
  return nullptr;
}

// ---------------------------------------------------------------- traversal

TEST(EmitSnapshotOrTouched, SnapshotEmitsEveryCellAndClearsTouched) {
  Fixture f;
  f.put({0, 0, 0}, beta(3.f, 1.f));
  f.put({1, 0, 0}, beta(4.f, 1.f));
  f.put({40, 7, 3}, beta(5.f, 1.f));  // different root block
  f.touched.coords.clear();           // a snapshot must not depend on this
  f.touched.coords.push_back({0, 0, 0});

  f.run(true);

  ASSERT_EQ(f.emitted.size(), 3u);
  // Cardinality alone would also pass an arm that emitted one cell's value
  // three times, or paired every value with the wrong coord — so pin the
  // pairing. a_occ is unique per coord above precisely so this can.
  const BetaVoxel* a = emittedAt(f, {0, 0, 0});
  const BetaVoxel* b = emittedAt(f, {1, 0, 0});
  const BetaVoxel* c = emittedAt(f, {40, 7, 3});
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  ASSERT_NE(c, nullptr);
  EXPECT_FLOAT_EQ(a->a_occ, 3.f);
  EXPECT_FLOAT_EQ(b->a_occ, 4.f);
  EXPECT_FLOAT_EQ(c->a_occ, 5.f);
  EXPECT_EQ(f.touched.clears, 1);
  EXPECT_EQ(f.touched.drains, 0);
  // Leaving the touched set populated would re-emit these on the next delta.
  EXPECT_TRUE(f.touched.coords.empty());
}

// Bonxai's forEachCell takes its visitor BY VALUE (twice — the outer overload
// forwards a copy into the const one), which is why the snapshot arm passes the
// emit callable through std::ref. Every emit callable in the node captures by
// reference and so cannot notice, which is exactly why the guard needs a
// functor that does: without std::ref this functor's count stays 0 on the
// snapshot arm while the delta arm reports 2, and the two arms silently
// disagree about which object they called.
struct CountingEmit {
  int seen = 0;  // by value inside the functor, on purpose
  void operator()(const BetaVoxel&, const CoordT&) { ++seen; }
};

TEST(EmitSnapshotOrTouched, BothArmsCallTheCallerSFunctorNotACopy) {
  Fixture f;
  f.put({0, 0, 0}, beta(3.f, 1.f));
  f.put({1, 0, 0}, beta(4.f, 1.f));
  auto drain = [&]() -> const std::vector<CoordT>& { return f.touched.drain(); };
  auto clear = [&] { f.touched.clear(); };

  CountingEmit delta_emit;
  scovox::emitSnapshotOrTouched(false, f.live, drain, clear, delta_emit);
  EXPECT_EQ(delta_emit.seen, 2);

  CountingEmit snapshot_emit;
  scovox::emitSnapshotOrTouched(true, f.live, drain, clear, snapshot_emit);
  EXPECT_EQ(snapshot_emit.seen, 2);
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
// with the two arguments swapped. GateDropsWhenEvidenceFalls is what makes the
// order a test — inside gateAndRefresh. It does not reach the node's two
// call-site wrappers, which forward through a generic `const auto&` pair and so
// would also swap silently; those live in a translation unit no test target
// links, and are covered by review only.
auto grewByOne = [](const BetaVoxel& last, const BetaVoxel& now) {
  return now.a_occ > last.a_occ + 1.f;
};

struct GateFixture {
  GateGrid gate{kRes, kInnerBits, kLeafBits};
  StampGrid stamps{kRes, kInnerBits, kLeafBits};
  std::optional<GateGrid::Accessor> gacc;
  std::optional<StampGrid::Accessor> stamp_acc;

  void arm(bool with_stamps) {
    gacc.emplace(gate.createAccessor());
    if (with_stamps) stamp_acc.emplace(stamps.createAccessor());
  }
  bool call(const CoordT& c, const BetaVoxel& v, bool snapshot, double t_now) {
    return scovox::gateAndRefresh(gacc, stamp_acc, c, v, snapshot, t_now,
                                  grewByOne);
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

// The miss lookup must NOT create. With create_if_missing the first sight of a
// voxel returns a value-initialised BetaVoxel instead of nullptr, and the
// predicate then runs against a_occ = 0 rather than being skipped. Chosen so
// the two answers differ: 0.5 is not "grown by one" over that phantom
// baseline, so a creating lookup would DROP a voxel's first ever emit.
TEST(GateAndRefresh, MissLookupDoesNotCreateAPhantomBaseline) {
  GateFixture f;
  f.arm(true);
  EXPECT_TRUE(f.call({7, 7, 7}, beta(0.5f, 1.f), /*snapshot=*/false, 10.0));

  auto acc = f.gate.createAccessor();
  const auto* g = acc.value({7, 7, 7}, false);
  ASSERT_NE(g, nullptr);
  EXPECT_FLOAT_EQ(g->a_occ, 0.5f);
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

  // The arm a fresh subscriber actually takes: every voxel walked by
  // forEachCell, every gate entry refreshed through the snapshot bypass, and
  // clear() rather than drain() consuming the touched set. It combines the two
  // corners the delta arm never reaches, so it gets its own assertions on the
  // coord/value pairing rather than a count.
  f.emitted.clear();
  f.touched.coords.push_back({0, 0, 0});
  gated_run(true, 30.0);

  ASSERT_EQ(f.emitted.size(), 2u);
  const BetaVoxel* a = emittedAt(f, {0, 0, 0});
  const BetaVoxel* b = emittedAt(f, {1, 0, 0});
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  EXPECT_FLOAT_EQ(a->a_occ, 3.f);
  EXPECT_FLOAT_EQ(b->a_occ, 3.f);
  EXPECT_EQ(f.touched.drains, 2);  // the snapshot arm does not drain
  EXPECT_EQ(f.touched.clears, 1);
  EXPECT_TRUE(f.touched.coords.empty());
  // Bypassed the comparison, but still refreshed: the stamps must have moved.
  auto sacc = g.stamps.createAccessor();
  ASSERT_NE(sacc.value({0, 0, 0}, false), nullptr);
  EXPECT_DOUBLE_EQ(*sacc.value({0, 0, 0}, false), 30.0);
  EXPECT_DOUBLE_EQ(*sacc.value({1, 0, 0}, false), 30.0);
}

}  // namespace
