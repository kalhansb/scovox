// Tests for ShareShaperCore (share_shaper.hpp) — the pure core of the
// per-robot scovox_bin egress pacer. The properties under test are the ones
// the field depends on:
//
//  1. Losslessness: shaping is invisible to the merger. Applying the shaped
//     chunks with the merger's snapshot-replace rule reproduces EXACTLY the
//     final state of applying the raw frames. This is the property that makes
//     the pacing rate a pure staleness/bandwidth trade.
//  2. Coalescing: a voxel that changes N times while queued goes on the wire
//     once, with the latest value — the mechanism that bounds the backlog by
//     map size instead of time.
//  3. Pacing: no chunk exceeds the uncompressed budget, and cumulative wire
//     bytes respect the token bucket (burst + rate·t + one-chunk overdraft).
//  4. Resync: markAllDirty reproduces the full store — twice, because a
//     SECOND reconnecting peer needs the same full state again.
//  5. The guard rails: prior pinning (rejected the way the MERGER rejects it),
//     refusing to pin an unusable header, epoch reset on a resolution change,
//     TSDF scoping, and zero traffic for a byte-identical redundant snapshot.
//  6. Fairness: a Beta backlog never starves Dir, within a chunk or across
//     chunks (semantics are a first-class planner input).
//  7. Degenerate configs degrade instead of wedging or silently starving.
//  8. A resync fired mid-backlog — the realistic reconnect — stays consistent.
//  9. A compressor failure re-queues rather than losing updates.
// 10. Bucket accounting across an idle link: credit is earned while nobody
//     listens, and capped at one chunk on rejoin so it isn't a burst.
// 11. clearStore (the node's epoch reset on a moved sender origin) drops the
//     resync mirror without disturbing the pin.

#include <gtest/gtest.h>

#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <tuple>
#include <vector>

#include "scovox/share_shaper.hpp"

namespace {

using scovox::BetaVoxel;
using scovox::BinarySerializer;
using scovox::DirVoxel;
using scovox::ScovoxBinarySerializer;
using scovox::ShareShaperCore;
using Frame = BinarySerializer::Frame;

// Deterministic LCG so the "random" maps are reproducible across runs.
struct Lcg {
  uint32_t s;
  explicit Lcg(uint32_t seed) : s(seed) {}
  uint32_t next() { return s = s * 1664525u + 1013904223u; }
  float uf() { return static_cast<float>(next() % 10000) / 100.0f + 0.01f; }
};

BetaVoxel beta(float a_occ, float a_free) { return BetaVoxel{a_occ, a_free}; }

DirVoxel dir(Lcg& rng) {
  DirVoxel v{};
  v.other = rng.uf();
  for (int i = 0; i < scovox::K_TOP; ++i) {
    v.cnt[i] = rng.uf();
    v.cls[i] = static_cast<uint16_t>(rng.next() % 14);
  }
  return v;
}

Frame emptyFrame() {
  Frame f;
  f.resolution = 0.1f;
  f.num_classes = 14;
  f.alpha_0 = 0.01f;
  return f;
}

ShareShaperCore::Config bigBucket() {
  // Effectively unpaced: for tests about state, not pacing.
  ShareShaperCore::Config c;
  c.target_rate_mbps = 1e6;
  c.burst_bytes = 1u << 30;
  c.max_chunk_bytes = 64 * 1024;
  return c;
}

// Decode one shaped chunk back into a Frame (what a merger receives).
Frame decode(const ShareShaperCore::Chunk& c) {
  const std::string blob = ScovoxBinarySerializer::decompressLZ4(c.data);
  EXPECT_FALSE(blob.empty());
  return BinarySerializer::deserialize(blob);
}

// Drain until dry, asserting termination.
std::vector<ShareShaperCore::Chunk> drainAll(ShareShaperCore& core) {
  std::vector<ShareShaperCore::Chunk> out;
  for (int i = 0; i < 10000 && core.dirtyCount() > 0; ++i) {
    auto chunks = core.drain(1.0);
    for (auto& c : chunks) out.push_back(std::move(c));
  }
  EXPECT_EQ(core.dirtyCount(), 0u);
  return out;
}

using Key = std::tuple<int32_t, int32_t, int32_t>;
Key key(const Bonxai::CoordT& c) { return {c.x, c.y, c.z}; }

// The merger's ingest rule, reduced to its essence: latest value per coord
// (snapshot-replace). Applying frames IN ORDER through this model is the
// reference the shaped stream must match.
struct Model {
  std::map<Key, BetaVoxel> beta;
  std::map<Key, DirVoxel> dir;
  void apply(const Frame& f) {
    for (const auto& d : f.beta_deltas) beta[key(d.coord)] = d.data;
    for (const auto& d : f.dir_deltas) dir[key(d.coord)] = d.data;
  }
};

bool same(const BetaVoxel& a, const BetaVoxel& b) {
  return std::memcmp(&a, &b, sizeof a) == 0;
}
bool same(const DirVoxel& a, const DirVoxel& b) {
  return std::memcmp(&a, &b, sizeof a) == 0;
}

void expectModelsEqual(const Model& raw, const Model& shaped) {
  ASSERT_EQ(raw.beta.size(), shaped.beta.size());
  ASSERT_EQ(raw.dir.size(), shaped.dir.size());
  for (const auto& [k, v] : raw.beta) {
    auto it = shaped.beta.find(k);
    ASSERT_NE(it, shaped.beta.end());
    EXPECT_TRUE(same(v, it->second));
  }
  for (const auto& [k, v] : raw.dir) {
    auto it = shaped.dir.find(k);
    ASSERT_NE(it, shaped.dir.end());
    EXPECT_TRUE(same(v, it->second));
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. Losslessness under interleaved ingest/drain.
// ---------------------------------------------------------------------------
TEST(ShareShaper, ShapedStreamReproducesRawFinalState) {
  ShareShaperCore core(bigBucket());
  Lcg rng(42);
  Model raw, shaped;

  // 30 frames, each touching a mix of fresh and previously-seen coords (so
  // coalescing and re-dirtying both happen), with drains interleaved at an
  // irregular cadence (so some updates ship immediately and others coalesce).
  for (int fi = 0; fi < 30; ++fi) {
    Frame f = emptyFrame();
    for (int i = 0; i < 200; ++i) {
      const int32_t x = static_cast<int32_t>(rng.next() % 40);
      const int32_t y = static_cast<int32_t>(rng.next() % 40);
      const int32_t z = static_cast<int32_t>(rng.next() % 8);
      f.beta_deltas.push_back({{x, y, z}, beta(rng.uf(), rng.uf())});
      if (i % 3 == 0) f.dir_deltas.push_back({{x, y, z}, dir(rng)});
    }
    raw.apply(f);
    ASSERT_TRUE(core.ingest(f).accepted);
    if (fi % 4 == 1) {
      for (const auto& c : core.drain(1.0)) shaped.apply(decode(c));
    }
  }
  for (const auto& c : drainAll(core)) shaped.apply(decode(c));

  expectModelsEqual(raw, shaped);
}

// ---------------------------------------------------------------------------
// 2. Coalescing: N queued updates to one coord → one wire record, latest value.
// ---------------------------------------------------------------------------
TEST(ShareShaper, QueuedUpdatesToOneCoordEmitOnceWithLatestValue) {
  ShareShaperCore core(bigBucket());
  const Bonxai::CoordT c{5, 6, 7};
  for (float a : {1.0f, 2.0f, 3.0f}) {
    Frame f = emptyFrame();
    f.beta_deltas.push_back({c, beta(a, 9.0f)});
    ASSERT_TRUE(core.ingest(f).accepted);
  }
  // One dirty entry despite three updates — the FIFO holds a coord at most once.
  EXPECT_EQ(core.dirtyCount(), 1u);

  auto chunks = drainAll(core);
  ASSERT_EQ(chunks.size(), 1u);
  Frame out = decode(chunks[0]);
  ASSERT_EQ(out.beta_deltas.size(), 1u);
  EXPECT_FLOAT_EQ(out.beta_deltas[0].data.a_occ, 3.0f);  // latest, not first
}

// ---------------------------------------------------------------------------
// 3. Pacing: chunk-size bound and token-bucket byte-rate bound.
// ---------------------------------------------------------------------------
TEST(ShareShaper, DrainRespectsChunkSizeAndTokenBucket) {
  ShareShaperCore::Config cfg;
  cfg.target_rate_mbps = 1.0;        // 125000 B/s
  cfg.burst_bytes = 16 * 1024;
  cfg.max_chunk_bytes = 32 * 1024;
  ShareShaperCore core(cfg);

  // One big "snapshot" frame: 20k beta voxels ≈ 400 KB uncompressed of
  // LCG-noise values (which LZ4 barely compresses), far beyond the bucket.
  Lcg rng(7);
  Frame f = emptyFrame();
  for (int i = 0; i < 20000; ++i) {
    f.beta_deltas.push_back(
        {{i % 100, (i / 100) % 100, i / 10000}, beta(rng.uf(), rng.uf())});
  }
  ASSERT_TRUE(core.ingest(f).accepted);

  const double dt = 0.1;
  const double rate_bps = cfg.target_rate_mbps * 1e6 / 8.0;
  // Generous per-chunk wire-size cap: LZ4 can expand incompressible input a
  // little past the uncompressed size.
  const std::size_t chunk_wire_cap = cfg.max_chunk_bytes + 4096;

  std::size_t total_wire = 0;
  std::size_t drains_with_output = 0;
  int ticks = 0;
  while (core.dirtyCount() > 0 && ticks < 10000) {
    ++ticks;
    auto chunks = core.drain(dt);
    if (!chunks.empty()) ++drains_with_output;
    for (const auto& c : chunks) {
      EXPECT_LE(c.uncompressed_bytes, cfg.max_chunk_bytes);
      EXPECT_LE(c.data.size(), chunk_wire_cap);
      total_wire += c.data.size();
    }
    // Token-bucket invariant after every tick: cumulative wire bytes never
    // exceed initial burst + refill so far + one chunk of overdraft.
    const double budget =
        cfg.burst_bytes + rate_bps * (ticks * dt) + chunk_wire_cap;
    EXPECT_LE(static_cast<double>(total_wire), budget)
        << "bucket exceeded at tick " << ticks;
  }
  EXPECT_EQ(core.dirtyCount(), 0u);
  // The point of the node: a burst does NOT go out in one or two ticks. With
  // ~400 KB queued against a 16 KB bucket at 12.5 KB/tick refill this needs
  // dozens of paced emissions.
  EXPECT_GT(drains_with_output, 10u);
}

// ---------------------------------------------------------------------------
// 4. Resync reproduces the full store — twice in a row.
// ---------------------------------------------------------------------------
TEST(ShareShaper, ResyncReproducesFullStoreForEachNewPeer) {
  ShareShaperCore core(bigBucket());
  Lcg rng(3);
  Model raw;
  for (int fi = 0; fi < 5; ++fi) {
    Frame f = emptyFrame();
    for (int i = 0; i < 100; ++i) {
      const int32_t x = static_cast<int32_t>(rng.next() % 30);
      const int32_t y = static_cast<int32_t>(rng.next() % 30);
      f.beta_deltas.push_back({{x, y, 0}, beta(rng.uf(), rng.uf())});
      if (i % 2 == 0) f.dir_deltas.push_back({{x, y, 0}, dir(rng)});
    }
    raw.apply(f);
    ASSERT_TRUE(core.ingest(f).accepted);
  }
  drainAll(core);  // steady state: everything delivered, nothing dirty

  for (int peer = 0; peer < 2; ++peer) {
    core.markAllDirty();
    EXPECT_EQ(core.dirtyCount(), core.storeBetaCount() + core.storeDirCount());
    Model resynced;
    for (const auto& c : drainAll(core)) resynced.apply(decode(c));
    expectModelsEqual(raw, resynced);  // full state, correct values, both times
  }

  // Resync of an EMPTY store is a no-op (peer joins before the first frame).
  ShareShaperCore empty(bigBucket());
  empty.markAllDirty();
  EXPECT_EQ(empty.dirtyCount(), 0u);
  EXPECT_TRUE(empty.drain(1.0).empty());
}

// ---------------------------------------------------------------------------
// 5a. A prior mismatch is rejected whole — the same way the MERGER rejects it
// (dscovox_node compares num_classes/alpha_0 against its pinned fleet prior),
// so a misconfigured fleet fails identically with or without the shaper.
// ---------------------------------------------------------------------------
TEST(ShareShaper, MismatchedPriorIsRejectedWhole) {
  ShareShaperCore core(bigBucket());
  Frame f = emptyFrame();
  f.beta_deltas.push_back({{1, 1, 1}, beta(5.0f, 1.0f)});
  ASSERT_TRUE(core.ingest(f).accepted);  // pins 0.1 / 14 / 0.01

  Frame bad_alpha = emptyFrame();
  bad_alpha.alpha_0 = 0.5f;  // would mis-subtract in the merger's refold
  bad_alpha.beta_deltas.push_back({{2, 2, 2}, beta(1.0f, 5.0f)});
  auto r1 = core.ingest(bad_alpha);
  EXPECT_FALSE(r1.accepted);
  EXPECT_NE(r1.reject_reason, nullptr);

  Frame bad_classes = emptyFrame();
  bad_classes.num_classes = 20;
  bad_classes.beta_deltas.push_back({{3, 3, 3}, beta(1.0f, 5.0f)});
  EXPECT_FALSE(core.ingest(bad_classes).accepted);

  // The rejected frames' deltas must not have leaked into the store.
  EXPECT_EQ(core.storeBetaCount(), 1u);
  EXPECT_EQ(core.dirtyCount(), 1u);
}

// ---------------------------------------------------------------------------
// 5a-bis. A header the shaper cannot legally re-emit is rejected before it can
// be PINNED (the pin is permanent for the epoch and is written into every
// outgoing frame).
// ---------------------------------------------------------------------------
TEST(ShareShaper, UnusableHeaderIsRejectedBeforePinning) {
  for (auto mutate : std::vector<std::function<void(Frame&)>>{
           [](Frame& f) { f.resolution = 0.0f; },
           [](Frame& f) { f.resolution = -0.1f; },
           [](Frame& f) { f.resolution = std::numeric_limits<float>::quiet_NaN(); },
           [](Frame& f) { f.num_classes = 0; },
           [](Frame& f) { f.alpha_0 = 0.0f; },
           [](Frame& f) { f.alpha_0 = std::numeric_limits<float>::infinity(); }}) {
    ShareShaperCore core(bigBucket());
    Frame f = emptyFrame();
    mutate(f);
    f.beta_deltas.push_back({{1, 1, 1}, beta(5.0f, 1.0f)});
    auto r = core.ingest(f);
    EXPECT_FALSE(r.accepted);
    EXPECT_NE(r.reject_reason, nullptr);
    EXPECT_FALSE(core.pinned());
    EXPECT_EQ(core.storeBetaCount(), 0u);
  }
}

// ---------------------------------------------------------------------------
// 5a-ter. A resolution change starts a NEW EPOCH instead of being rejected.
// The merger has no resolution guard of its own, so a shaper that rejected
// here would freeze every peer's view of this robot (its own loopback map would
// keep working, hiding the fault) — and the old coords cannot be re-expressed
// in the new grid, so the store must go.
// ---------------------------------------------------------------------------
TEST(ShareShaper, ResolutionChangeStartsNewEpoch) {
  ShareShaperCore core(bigBucket());
  Frame old_epoch = emptyFrame();
  for (int i = 0; i < 50; ++i)
    old_epoch.beta_deltas.push_back({{i, 0, 0}, beta(5.0f, 1.0f)});
  ASSERT_TRUE(core.ingest(old_epoch).accepted);
  drainAll(core);
  ASSERT_EQ(core.storeBetaCount(), 50u);

  Frame new_epoch = emptyFrame();
  new_epoch.resolution = 0.2f;
  new_epoch.beta_deltas.push_back({{7, 7, 7}, beta(2.0f, 3.0f)});
  auto r = core.ingest(new_epoch);
  EXPECT_TRUE(r.accepted);
  EXPECT_TRUE(r.epoch_reset);
  EXPECT_NE(r.epoch_reason, nullptr);
  // Old epoch dropped whole; the new frame is all that remains.
  EXPECT_EQ(core.storeBetaCount(), 1u);
  EXPECT_FLOAT_EQ(core.resolution(), 0.2f);

  // A resync now replays ONLY new-epoch coords, at the new resolution.
  core.markAllDirty();
  auto chunks = drainAll(core);
  ASSERT_EQ(chunks.size(), 1u);
  Frame out = decode(chunks[0]);
  EXPECT_FLOAT_EQ(out.resolution, 0.2f);
  ASSERT_EQ(out.beta_deltas.size(), 1u);
  EXPECT_EQ(out.beta_deltas[0].coord.x, 7);
}

// ---------------------------------------------------------------------------
// 5b. TSDF is out of scope: dropped on ingest, never re-emitted.
// ---------------------------------------------------------------------------
TEST(ShareShaper, TsdfDeltasAreDroppedNotForwarded) {
  ShareShaperCore core(bigBucket());
  Frame f = emptyFrame();
  f.tsdf_deltas.push_back({{1, 2, 3}, scovox::TsdfVoxel{0.5f, 2.0f}});
  f.beta_deltas.push_back({{1, 2, 3}, beta(4.0f, 1.0f)});
  auto r = core.ingest(f);
  ASSERT_TRUE(r.accepted);
  EXPECT_EQ(r.tsdf_dropped, 1u);

  auto chunks = drainAll(core);
  ASSERT_EQ(chunks.size(), 1u);
  Frame out = decode(chunks[0]);
  EXPECT_TRUE(out.tsdf_deltas.empty());
  EXPECT_EQ(out.beta_deltas.size(), 1u);
}

// ---------------------------------------------------------------------------
// 5c. A byte-identical redundant snapshot produces ZERO shaped traffic.
// This is what absorbs the sender's spurious full dumps (e.g. a debug
// subscriber attaching to the raw topic bumps the count and triggers a
// snapshot every peer has already seen).
// ---------------------------------------------------------------------------
TEST(ShareShaper, RedundantSnapshotIsAbsorbedSilently) {
  ShareShaperCore core(bigBucket());
  Lcg rng(11);
  Frame f = emptyFrame();
  for (int i = 0; i < 500; ++i) {
    f.beta_deltas.push_back({{i, -i, 0}, beta(rng.uf(), rng.uf())});
    if (i % 4 == 0) f.dir_deltas.push_back({{i, -i, 0}, dir(rng)});
  }
  ASSERT_TRUE(core.ingest(f).accepted);
  drainAll(core);

  auto r = core.ingest(f);  // the same frame again, byte for byte
  ASSERT_TRUE(r.accepted);
  EXPECT_EQ(r.beta_changed, 0u);
  EXPECT_EQ(r.dir_changed, 0u);
  EXPECT_EQ(core.dirtyCount(), 0u);
  EXPECT_TRUE(core.drain(1.0).empty());
}

// ---------------------------------------------------------------------------
// 6. Alternation: a huge Beta backlog does not starve Dir. Semantics are a
// first-class planner input (sender comment in scovox_node), so a resync must
// deliver Dir records alongside the free-space Beta bulk, not after it.
// ---------------------------------------------------------------------------
TEST(ShareShaper, DirRecordsAreNotStarvedBehindBetaBacklog) {
  ShareShaperCore::Config cfg = bigBucket();
  cfg.max_chunk_bytes = 2048;  // small chunks so the first one can't fit all
  ShareShaperCore core(cfg);
  Lcg rng(23);
  Frame f = emptyFrame();
  for (int i = 0; i < 5000; ++i)
    f.beta_deltas.push_back({{i, 0, 0}, beta(rng.uf(), rng.uf())});
  for (int i = 0; i < 20; ++i) f.dir_deltas.push_back({{i, 1, 0}, dir(rng)});
  ASSERT_TRUE(core.ingest(f).accepted);

  auto first = core.drain(1.0);
  ASSERT_FALSE(first.empty());
  // Strict alternation → the very first chunk carries Dir records even though
  // Beta outnumbers them 250:1.
  EXPECT_GT(first.front().n_dir, 0u);
  EXPECT_GT(first.front().n_beta, 0u);
}

// ---------------------------------------------------------------------------
// 6b. Alternation also carries ACROSS chunks. At the minimum budget a chunk
// holds a single record, so a turn that reset per chunk would always pick Beta
// and starve Dir for as long as new Beta kept arriving.
// ---------------------------------------------------------------------------
TEST(ShareShaper, AlternationCarriesAcrossChunks) {
  ShareShaperCore::Config cfg = bigBucket();
  cfg.max_chunk_bytes = ShareShaperCore::kMinChunkBytes;  // one record per chunk
  ShareShaperCore core(cfg);
  Lcg rng(31);
  Frame f = emptyFrame();
  for (int i = 0; i < 100; ++i)
    f.beta_deltas.push_back({{i, 0, 0}, beta(rng.uf(), rng.uf())});
  for (int i = 0; i < 100; ++i) f.dir_deltas.push_back({{i, 1, 0}, dir(rng)});
  ASSERT_TRUE(core.ingest(f).accepted);

  auto chunks = core.drain(1.0);
  ASSERT_GE(chunks.size(), 4u);
  std::size_t beta_in_first_four = 0, dir_in_first_four = 0;
  for (std::size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(chunks[i].n_beta + chunks[i].n_dir, 1u);  // one record per chunk
    beta_in_first_four += chunks[i].n_beta;
    dir_in_first_four += chunks[i].n_dir;
  }
  EXPECT_EQ(beta_in_first_four, 2u);
  EXPECT_EQ(dir_in_first_four, 2u);
}

// ---------------------------------------------------------------------------
// 7. A chunk budget too small for a record must not wedge the pacer. Below
// kMinChunkBytes buildChunk could pop NOTHING (a permanent stall that drain
// would misreport as an LZ4 failure), and a budget between the two record sizes
// would starve the larger stream forever while the smaller kept flowing —
// silent, because the output looks healthy. Config is clamped instead.
// ---------------------------------------------------------------------------
TEST(ShareShaper, DegenerateChunkBudgetIsClampedNotWedged) {
  for (std::size_t chunk : {std::size_t{0}, std::size_t{1}, std::size_t{40},
                            std::size_t{50}, ShareShaperCore::kMinChunkBytes}) {
    ShareShaperCore::Config cfg = bigBucket();
    cfg.max_chunk_bytes = chunk;
    ShareShaperCore core(cfg);
    Lcg rng(chunk == 0 ? 1u : static_cast<uint32_t>(chunk));
    Model raw;
    Frame f = emptyFrame();
    for (int i = 0; i < 40; ++i) {
      f.beta_deltas.push_back({{i, 0, 0}, beta(rng.uf(), rng.uf())});
      f.dir_deltas.push_back({{i, 1, 0}, dir(rng)});
    }
    raw.apply(f);
    ASSERT_TRUE(core.ingest(f).accepted);

    Model shaped;
    for (const auto& c : drainAll(core)) shaped.apply(decode(c));
    // Both streams drain to completion — no stall, no starved stream.
    EXPECT_EQ(core.dirtyCount(), 0u) << "chunk budget " << chunk;
    expectModelsEqual(raw, shaped);
  }
}

// ---------------------------------------------------------------------------
// 8. A resync fired while a backlog is still queued (the realistic case: a peer
// reconnects mid-drain) must not double-queue or lose anything.
// ---------------------------------------------------------------------------
TEST(ShareShaper, ResyncMidBacklogStaysConsistent) {
  ShareShaperCore::Config cfg;
  cfg.target_rate_mbps = 1.0;
  cfg.burst_bytes = 8 * 1024;    // small bucket → the first drain is partial
  cfg.max_chunk_bytes = 2048;    // many small chunks
  ShareShaperCore core(cfg);
  Lcg rng(99);
  Model raw;
  Frame f = emptyFrame();
  for (int i = 0; i < 1000; ++i) {
    f.beta_deltas.push_back({{i, 0, 0}, beta(rng.uf(), rng.uf())});
    if (i % 5 == 0) f.dir_deltas.push_back({{i, 1, 0}, dir(rng)});
  }
  raw.apply(f);
  ASSERT_TRUE(core.ingest(f).accepted);

  // Partially drain, so the FIFOs hold some entries and the store holds both
  // dirty and already-delivered ones.
  Model shaped;
  for (const auto& c : core.drain(0.0001)) shaped.apply(decode(c));
  ASSERT_GT(core.dirtyCount(), 0u);
  ASSERT_LT(core.dirtyCount(), core.storeBetaCount() + core.storeDirCount());

  core.markAllDirty();
  // Everything queued exactly once — the invariant a double-push would break.
  EXPECT_EQ(core.dirtyCount(), core.storeBetaCount() + core.storeDirCount());
  for (const auto& c : drainAll(core)) shaped.apply(decode(c));
  expectModelsEqual(raw, shaped);
}

// ---------------------------------------------------------------------------
// 9. A compressor failure re-queues the popped entries: the shaper's promise is
// that an update is never lost silently (the raw sender drops the frame).
// ---------------------------------------------------------------------------
TEST(ShareShaper, CompressFailureRequeuesEverything) {
  bool fail = true;
  ShareShaperCore::Config cfg = bigBucket();
  cfg.compress = [&fail](const std::string& blob) {
    return fail ? std::vector<uint8_t>{}
                : ScovoxBinarySerializer::compressLZ4(blob);
  };
  ShareShaperCore core(cfg);
  Lcg rng(5);
  Model raw;
  Frame f = emptyFrame();
  for (int i = 0; i < 100; ++i) {
    f.beta_deltas.push_back({{i, 2, 3}, beta(rng.uf(), rng.uf())});
    if (i % 3 == 0) f.dir_deltas.push_back({{i, 2, 3}, dir(rng)});
  }
  raw.apply(f);
  ASSERT_TRUE(core.ingest(f).accepted);
  const std::size_t queued = core.dirtyCount();

  EXPECT_TRUE(core.drain(1.0).empty());       // nothing goes out
  EXPECT_TRUE(core.lastCompressFailed());
  EXPECT_EQ(core.dirtyCount(), queued);       // ...and nothing is lost

  fail = false;
  Model shaped;
  for (const auto& c : drainAll(core)) shaped.apply(decode(c));
  EXPECT_FALSE(core.lastCompressFailed());
  expectModelsEqual(raw, shaped);             // full state still delivered
}

// ---------------------------------------------------------------------------
// 10. Bucket accounting across an idle period: refill() earns credit without
// emitting (what the node does while no peer is subscribed), and the match
// handler caps that credit to one chunk so the rejoin isn't itself a burst.
// ---------------------------------------------------------------------------
TEST(ShareShaper, IdleRefillEarnsCreditAndResyncCapsIt) {
  ShareShaperCore::Config cfg;
  cfg.target_rate_mbps = 1.0;
  cfg.burst_bytes = 256 * 1024;
  cfg.max_chunk_bytes = 32 * 1024;

  // Same run twice — the only difference is whether the credit earned while
  // nobody was listening is capped before the first post-rejoin drain.
  auto rejoinWireBytes = [&cfg](bool cap) {
    ShareShaperCore core(cfg);
    Lcg rng(17);
    Frame f = emptyFrame();
    // Far more than the bucket can hold, so a backlog is guaranteed to survive
    // the first drain however well LZ4 does on this data.
    for (int i = 0; i < 200000; ++i) {
      f.beta_deltas.push_back(
          {{i % 100, (i / 100) % 100, i / 10000}, beta(rng.uf(), rng.uf())});
    }
    EXPECT_TRUE(core.ingest(f).accepted);
    EXPECT_GT(core.drain(0.0).size(), 0u);   // spend the bring-up bucket
    EXPECT_GT(core.dirtyCount(), 0u) << "fixture must leave a backlog";
    // 60 s of idle link: credit accrues (capped at burst_bytes), nothing ships.
    for (int i = 0; i < 600; ++i) core.refill(0.1);
    if (cap) core.capTokensForResync();
    std::size_t wire = 0;
    for (const auto& c : core.drain(0.0)) wire += c.data.size();
    return wire;
  };

  const std::size_t capped = rejoinWireBytes(true);
  const std::size_t uncapped = rejoinWireBytes(false);

  // Idling really did earn credit — an uncapped rejoin discharges most of the
  // 256 KB bucket in the first tick, which is the burst this node exists to
  // remove, at the moment the link is weakest.
  EXPECT_GT(uncapped, 4 * capped);
  // Capped: one chunk of credit, plus the documented one-chunk overdraft.
  EXPECT_GT(capped, 0u);
  EXPECT_LE(capped, 2 * cfg.max_chunk_bytes + 4096);
}

// ---------------------------------------------------------------------------
// 11. clearStore drops the resync mirror but keeps the pin: the node calls it
// when map_from_source jumps (sender restarted with a new origin), because
// replaying old coords under the new pose would misplace a whole mission.
// ---------------------------------------------------------------------------
TEST(ShareShaper, ClearStoreDropsMirrorAndKeepsPin) {
  ShareShaperCore core(bigBucket());
  Frame f = emptyFrame();
  for (int i = 0; i < 20; ++i) f.beta_deltas.push_back({{i, 0, 0}, beta(1.0f, 2.0f)});
  ASSERT_TRUE(core.ingest(f).accepted);

  core.clearStore();
  EXPECT_EQ(core.storeBetaCount(), 0u);
  EXPECT_EQ(core.storeDirCount(), 0u);
  EXPECT_EQ(core.dirtyCount(), 0u);
  EXPECT_TRUE(core.pinned());
  EXPECT_FLOAT_EQ(core.resolution(), 0.1f);
  core.markAllDirty();
  EXPECT_TRUE(core.drain(1.0).empty());  // nothing left to resend

  // Post-clear ingest works normally and is all that a resync now carries.
  Frame g = emptyFrame();
  g.beta_deltas.push_back({{99, 0, 0}, beta(3.0f, 4.0f)});
  ASSERT_TRUE(core.ingest(g).accepted);
  auto chunks = drainAll(core);
  ASSERT_EQ(chunks.size(), 1u);
  EXPECT_EQ(decode(chunks[0]).beta_deltas.size(), 1u);
}
