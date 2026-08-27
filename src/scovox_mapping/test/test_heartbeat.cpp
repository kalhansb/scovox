// Unit tests for scovox::heartbeatReemit — the heartbeat walk extracted from
// SCovoxNode when the gate shadow grids were split into raw-voxel value grids
// plus heartbeat-only VoxelGrid<double> stamp twins (audit item 7). The tests
// pin the helper's exact semantics — a stamp is touched ONLY on a successful
// emit, so vetoed / not-yet-live voxels are re-checked every tick — and prove
// the split walk emits the same (coord, value) sequence as the pre-split
// combined-struct walk it replaced, which is what keeps heartbeat wire output
// byte-identical across the split.

#include <cstdint>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <bonxai/bonxai.hpp>

#include "scovox/beta_voxel.hpp"
#include "scovox/dir_voxel.hpp"
#include "scovox/node_utils.hpp"

namespace {

using Bonxai::CoordT;
using scovox::BetaVoxel;
using scovox::DirVoxel;

constexpr double kRes = 0.10;
constexpr int kInnerBits = 2;
constexpr int kLeafBits = 3;

struct Emitted {
  CoordT c;
  BetaVoxel v;
};

bool sameCoord(const CoordT& a, const CoordT& b) {
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

// A fixture holding the split trio: live grid, gate value grid, stamp twin —
// all with identical geometry, populated in lockstep insertion order (the
// invariant the emit paths in scovox_node maintain).
struct SplitGrids {
  Bonxai::VoxelGrid<BetaVoxel> live{kRes, kInnerBits, kLeafBits};
  Bonxai::VoxelGrid<BetaVoxel> gate{kRes, kInnerBits, kLeafBits};
  Bonxai::VoxelGrid<double> stamps{kRes, kInnerBits, kLeafBits};

  // Insert a voxel present in all three grids (the normal lockstep state).
  void put(const CoordT& c, const BetaVoxel& live_v, const BetaVoxel& gate_v,
           double t_emit) {
    auto la = live.createAccessor();
    auto ga = gate.createAccessor();
    auto ta = stamps.createAccessor();
    la.setValue(c, live_v);
    ga.setValue(c, gate_v);
    ta.setValue(c, t_emit);
  }

  double stampAt(const CoordT& c) {
    auto ta = stamps.createAccessor();
    const double* t = ta.value(c, false);
    EXPECT_NE(t, nullptr);
    return t ? *t : -1.0;
  }

  BetaVoxel gateAt(const CoordT& c) {
    auto ga = gate.createAccessor();
    const BetaVoxel* g = ga.value(c, false);
    EXPECT_NE(g, nullptr);
    return g ? *g : BetaVoxel{-1.f, -1.f};
  }

  std::vector<Emitted> run(double t_now, double period,
                           bool accept_all = true) {
    std::vector<Emitted> out;
    auto la = live.createAccessor();
    scovox::heartbeatReemit(
        gate, stamps, la, t_now, period,
        [&](const BetaVoxel&) { return accept_all; },
        [&](const BetaVoxel& v, const CoordT& c) { out.push_back({c, v}); });
    return out;
  }
};

TEST(HeartbeatReemit, StaleVoxelReemittedGateRefreshedStampUpdated) {
  SplitGrids g;
  const CoordT c{1, 2, 3};
  // Live has moved on since the last emit; gate still holds the old copy.
  g.put(c, BetaVoxel{5.f, 1.f}, BetaVoxel{2.f, 1.f}, /*t_emit=*/10.0);

  auto out = g.run(/*t_now=*/20.0, /*period=*/5.0);

  ASSERT_EQ(out.size(), 1u);
  EXPECT_TRUE(sameCoord(out[0].c, c));
  EXPECT_FLOAT_EQ(out[0].v.a_occ, 5.f);   // emits the LIVE value
  EXPECT_FLOAT_EQ(out[0].v.a_free, 1.f);
  EXPECT_FLOAT_EQ(g.gateAt(c).a_occ, 5.f);  // gate copy refreshed to live
  EXPECT_DOUBLE_EQ(g.stampAt(c), 20.0);     // stamp advanced to t_now
}

TEST(HeartbeatReemit, FreshVoxelSkippedEntirely) {
  SplitGrids g;
  const CoordT c{0, 0, 0};
  g.put(c, BetaVoxel{5.f, 1.f}, BetaVoxel{2.f, 1.f}, /*t_emit=*/18.0);

  auto out = g.run(/*t_now=*/20.0, /*period=*/5.0);

  EXPECT_TRUE(out.empty());
  EXPECT_FLOAT_EQ(g.gateAt(c).a_occ, 2.f);  // gate NOT refreshed
  EXPECT_DOUBLE_EQ(g.stampAt(c), 18.0);     // stamp untouched
}

TEST(HeartbeatReemit, ExactlyPeriodOldReemits) {
  // Boundary: the guard is `t_now - t_emit < period`, so age == period fires.
  SplitGrids g;
  const CoordT c{0, 0, 0};
  g.put(c, BetaVoxel{3.f, 1.f}, BetaVoxel{3.f, 1.f}, /*t_emit=*/15.0);

  auto out = g.run(/*t_now=*/20.0, /*period=*/5.0);

  ASSERT_EQ(out.size(), 1u);
  EXPECT_DOUBLE_EQ(g.stampAt(c), 20.0);
}

TEST(HeartbeatReemit, AcceptVetoLeavesStampSoVoxelRetriesNextTick) {
  SplitGrids g;
  const CoordT c{4, 5, 6};
  g.put(c, BetaVoxel{5.f, 1.f}, BetaVoxel{2.f, 1.f}, /*t_emit=*/10.0);

  // Vetoed: no emit, no gate refresh, and — the load-bearing part — the stamp
  // stays at its old value, so the voxel is re-examined on the next tick.
  auto out = g.run(/*t_now=*/20.0, /*period=*/5.0, /*accept_all=*/false);
  EXPECT_TRUE(out.empty());
  EXPECT_FLOAT_EQ(g.gateAt(c).a_occ, 2.f);
  EXPECT_DOUBLE_EQ(g.stampAt(c), 10.0);

  // Next tick the veto lifts and the voxel goes out.
  out = g.run(/*t_now=*/21.0, /*period=*/5.0, /*accept_all=*/true);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_DOUBLE_EQ(g.stampAt(c), 21.0);
}

TEST(HeartbeatReemit, MissingLiveVoxelLeavesStampSoVoxelRetriesNextTick) {
  SplitGrids g;
  const CoordT c{7, 8, 9};
  // Gate + stamp exist but the live grid has no cell (e.g. the live voxel was
  // dropped by a rebuild). Stamp must stay put so the coord is retried.
  {
    auto ga = g.gate.createAccessor();
    auto ta = g.stamps.createAccessor();
    ga.setValue(c, BetaVoxel{2.f, 1.f});
    ta.setValue(c, 10.0);
  }

  auto out = g.run(/*t_now=*/20.0, /*period=*/5.0);
  EXPECT_TRUE(out.empty());
  EXPECT_DOUBLE_EQ(g.stampAt(c), 10.0);

  // Live voxel appears; next tick emits it.
  {
    auto la = g.live.createAccessor();
    la.setValue(c, BetaVoxel{6.f, 2.f});
  }
  out = g.run(/*t_now=*/21.0, /*period=*/5.0);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_FLOAT_EQ(out[0].v.a_occ, 6.f);
  EXPECT_DOUBLE_EQ(g.stampAt(c), 21.0);
}

TEST(HeartbeatReemit, GateCellAbsentStillEmitsAndStamps) {
  // Defensive tolerance in the helper: if the gate value grid somehow lacks
  // the cell (lockstep normally prevents this), the emit still happens and the
  // stamp still advances — the walk never allocates into the gate grid.
  SplitGrids g;
  const CoordT c{2, 2, 2};
  {
    auto la = g.live.createAccessor();
    auto ta = g.stamps.createAccessor();
    la.setValue(c, BetaVoxel{4.f, 1.f});
    ta.setValue(c, 10.0);
  }

  auto out = g.run(/*t_now=*/20.0, /*period=*/5.0);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_DOUBLE_EQ(g.stampAt(c), 20.0);
  auto ga = g.gate.createAccessor();
  EXPECT_EQ(ga.value(c, false), nullptr);  // still not allocated
}

TEST(HeartbeatReemit, DirBinarizeStyleVetoPerVoxel) {
  // Dir-flavoured run: the accept lambda mirrors the node's binarize veto
  // (reject when the dominant slot is the empty sentinel). Vetoed voxels keep
  // their stamps; accepted ones emit.
  Bonxai::VoxelGrid<DirVoxel> live{kRes, kInnerBits, kLeafBits};
  Bonxai::VoxelGrid<DirVoxel> gate{kRes, kInnerBits, kLeafBits};
  Bonxai::VoxelGrid<double> stamps{kRes, kInnerBits, kLeafBits};

  DirVoxel labelled{};
  labelled.other = 0.1f;
  labelled.cnt[0] = 5.f;
  labelled.cls[0] = 3;
  for (int i = 1; i < scovox::K_TOP; ++i) labelled.cls[i] = 0xFFFF;

  DirVoxel unlabelled{};
  unlabelled.other = 0.1f;
  for (int i = 0; i < scovox::K_TOP; ++i) unlabelled.cls[i] = 0xFFFF;

  const CoordT ca{0, 0, 0};  // labelled → emits
  const CoordT cb{1, 0, 0};  // sentinel-dominant → vetoed
  {
    auto la = live.createAccessor();
    auto ga = gate.createAccessor();
    auto ta = stamps.createAccessor();
    la.setValue(ca, labelled);
    ga.setValue(ca, labelled);
    ta.setValue(ca, 10.0);
    la.setValue(cb, unlabelled);
    ga.setValue(cb, unlabelled);
    ta.setValue(cb, 10.0);
  }

  std::vector<CoordT> emitted;
  auto la = live.createAccessor();
  scovox::heartbeatReemit(
      gate, stamps, la, /*t_now=*/20.0, /*period=*/5.0,
      [](const DirVoxel& v) { return v.cls[0] != 0xFFFF; },
      [&](const DirVoxel&, const CoordT& c) { emitted.push_back(c); });

  ASSERT_EQ(emitted.size(), 1u);
  EXPECT_TRUE(sameCoord(emitted[0], ca));
  auto ta = stamps.createAccessor();
  EXPECT_DOUBLE_EQ(*ta.value(ca, false), 20.0);
  EXPECT_DOUBLE_EQ(*ta.value(cb, false), 10.0);  // vetoed → retried next tick
}

// The equivalence test: the pre-split node kept combined structs
// `{ value; double t_emit; }` in ONE grid and walked that; the split keeps the
// stamp in a twin grid and walks the twin. Because the twin is written in
// lockstep (same coords, same insertion order, same geometry), its active set
// and Bonxai iteration order match the combined grid's — so the emitted
// (coord, value) sequence, gate contents, and stamps must be identical.
struct GateBetaRef {
  BetaVoxel v;
  double t_emit;
};

TEST(HeartbeatReemit, SplitWalkMatchesPreSplitCombinedWalk) {
  // Coords spread across several leaves and root blocks (root span here is
  // 2^(2+3) = 32 voxels per axis), plus negatives — enough to exercise the
  // iteration-order claim, not just single-cell behaviour.
  const std::vector<CoordT> coords = {
      {0, 0, 0},   {1, 2, 3},    {7, 7, 7},     {8, 0, 0},
      {31, 31, 31}, {32, 0, 0},  {40, 12, -3},  {-1, -1, -1},
      {-40, 5, 90}, {100, 100, 100}, {3, -60, 17}, {64, 64, 64},
  };

  // Per-coord scenario, cycling: 0 = stale+accepted, 1 = fresh, 2 = vetoed,
  // 3 = live voxel missing.
  const double t_now = 100.0, period = 5.0;
  auto scenario = [&](size_t i) { return static_cast<int>(i % 4); };
  auto liveVoxel = [](size_t i) {
    return BetaVoxel{static_cast<float>(i) + 1.5f, 2.f};
  };
  auto gateVoxel = [](size_t i) {
    return BetaVoxel{static_cast<float>(i) + 0.5f, 1.f};
  };
  auto stampOf = [&](size_t i) {
    return scenario(i) == 1 ? (t_now - 1.0) : (t_now - 50.0);
  };
  auto accept = [&](const BetaVoxel& v) {
    // Veto keyed off the value so both walks apply the identical predicate:
    // scenario-2 voxels get a marker a_free.
    return v.a_free != 99.f;
  };

  // --- Reference: combined-struct grid, original walk. ---
  Bonxai::VoxelGrid<BetaVoxel> ref_live{kRes, kInnerBits, kLeafBits};
  Bonxai::VoxelGrid<GateBetaRef> ref_gate{kRes, kInnerBits, kLeafBits};
  {
    auto la = ref_live.createAccessor();
    auto ga = ref_gate.createAccessor();
    for (size_t i = 0; i < coords.size(); ++i) {
      BetaVoxel lv = liveVoxel(i);
      if (scenario(i) == 2) lv.a_free = 99.f;
      if (scenario(i) != 3) la.setValue(coords[i], lv);
      ga.setValue(coords[i], GateBetaRef{gateVoxel(i), stampOf(i)});
    }
  }
  std::vector<Emitted> ref_out;
  {
    auto la = ref_live.createAccessor();
    auto ga = ref_gate.createAccessor();
    ref_gate.forEachCell([&](GateBetaRef& g, const CoordT& c) {
      if (t_now - g.t_emit < period) return;
      const BetaVoxel* v = la.value(c, false);
      if (!v) return;
      if (!accept(*v)) return;
      g.v = *v;
      g.t_emit = t_now;
      ref_out.push_back({c, *v});
    });
    (void)ga;
  }

  // --- Split trio, same insertion order, walked via heartbeatReemit. ---
  SplitGrids sp;
  {
    auto la = sp.live.createAccessor();
    auto ga = sp.gate.createAccessor();
    auto ta = sp.stamps.createAccessor();
    for (size_t i = 0; i < coords.size(); ++i) {
      BetaVoxel lv = liveVoxel(i);
      if (scenario(i) == 2) lv.a_free = 99.f;
      if (scenario(i) != 3) la.setValue(coords[i], lv);
      ga.setValue(coords[i], gateVoxel(i));
      ta.setValue(coords[i], stampOf(i));
    }
  }
  std::vector<Emitted> sp_out;
  {
    auto la = sp.live.createAccessor();
    scovox::heartbeatReemit(
        sp.gate, sp.stamps, la, t_now, period, accept,
        [&](const BetaVoxel& v, const CoordT& c) { sp_out.push_back({c, v}); });
  }

  // Same emissions in the same ORDER (wire order is the invariant), with the
  // same values.
  ASSERT_EQ(sp_out.size(), ref_out.size());
  ASSERT_GT(ref_out.size(), 0u);  // the scenario mix must actually emit
  for (size_t i = 0; i < ref_out.size(); ++i) {
    EXPECT_TRUE(sameCoord(sp_out[i].c, ref_out[i].c)) << "emission " << i;
    EXPECT_FLOAT_EQ(sp_out[i].v.a_occ, ref_out[i].v.a_occ) << "emission " << i;
    EXPECT_FLOAT_EQ(sp_out[i].v.a_free, ref_out[i].v.a_free) << "emission " << i;
  }

  // Same post-state per coord: gate copy and stamp agree with the reference.
  {
    auto ga = sp.gate.createAccessor();
    auto ta = sp.stamps.createAccessor();
    auto rga = ref_gate.createAccessor();
    for (size_t i = 0; i < coords.size(); ++i) {
      const GateBetaRef* r = rga.value(coords[i], false);
      const BetaVoxel* g = ga.value(coords[i], false);
      const double* t = ta.value(coords[i], false);
      ASSERT_NE(r, nullptr);
      ASSERT_NE(g, nullptr);
      ASSERT_NE(t, nullptr);
      EXPECT_FLOAT_EQ(g->a_occ, r->v.a_occ) << "coord " << i;
      EXPECT_FLOAT_EQ(g->a_free, r->v.a_free) << "coord " << i;
      EXPECT_DOUBLE_EQ(*t, r->t_emit) << "coord " << i;
    }
  }
}

}  // namespace
