#pragma once

/// @file share_shaper.hpp
/// @brief Pure core of the per-robot scovox_bin egress shaper (share_shaper_node).
///
/// Sits between a robot's own scovox_node and the radio: ingests that robot's
/// ScovoxMapBinary delta frames, holds a latest-value-per-coord store, and
/// drains it as small, token-bucket-paced chunks. Peers' dscovox mergers
/// subscribe to the shaped stream instead of the raw one, so the link sees a
/// steady byte rate instead of bursts — in particular the reconnect snapshot
/// (a full-grid dump in ONE message at the sender) becomes a paced trickle.
///
/// Why this is lossless: the dscovox ingest is snapshot-replace
/// (dscovox_node.cpp `*v = d.data`), so only the LATEST value per SOURCE coord
/// ever matters to the fused result. That makes coalescing safe by
/// construction: if a voxel changes three times while queued here, emitting
/// only the newest value is a no-op for the merged map. Consequently the pacing
/// rate is purely a staleness/bandwidth trade, never a correctness one — set it
/// below the steady-state input rate and the peer's map just converges later.
///
/// One caveat on that argument, stated precisely: the merger replaces per MAP
/// coord, which it derives from the source coord through map_from_source and a
/// floor() (dscovox_node.cpp toMapPos + posToCoord). Under a pose that is not
/// grid-aligned, two neighbouring source voxels can alias onto ONE map voxel,
/// and then which of them lands last is order-dependent. The raw path is
/// already arbitrary there (the sender's touched-set is unordered); the shaper
/// only widens that window from one 2 Hz tick to the backlog age. The effect is
/// confined to a seam voxel holding the older of two adjacent measurements
/// until either is next touched, and the deployment discipline (identity
/// map->integration_frame statics, one fleet resolution) makes it vacuous.
///
/// Memory: the store is a latest-value MIRROR of the sender's shared grid,
/// which never prunes (rolling mode windows the planning-map PUBLICATION, not
/// the shared Beta/Dir grids), so it grows monotonically with mapped volume at
/// roughly 48 B/voxel of hash node plus bucket array — order 350-450 MB RSS at
/// the ~6M voxels seen on a 1 h forest mission. Fine beside the mapper on
/// 16 GB-class compute; budget for it explicitly on 8 GB Jetson-class hardware
/// or multi-hour missions (shaper_diag prints the live store counts).
///
/// Epochs: (resolution, num_classes, alpha_0) are pinned from the first frame.
/// A later RESOLUTION change means the sender was restarted with a different
/// config and its coords no longer mean what the store holds, so the store is
/// cleared and re-pinned — a new epoch — rather than the frames being rejected.
/// Rejecting would be worse than useless here: the merger has no resolution
/// guard of its own (only num_classes/alpha_0), so a rejecting shaper would
/// silently freeze every peer's view of this robot while its own loopback map
/// kept working. A PRIOR change is still rejected whole, because the merger
/// rejects mismatched priors too — failing the same way as the raw path is the
/// point.
///
/// What the core is NOT:
///  - It never transforms coordinates. Deltas stay in the sender's source-grid
///    coord space verbatim; the carried map_from_source pose does the map-frame
///    placement at the merger, exactly as for the unshaped stream.
///  - It is not a merger. One shaper serves ONE upstream stream (its own
///    robot's LOCAL map). Feeding it a fused/dscovox output would re-introduce
///    the evidence-echo the local→fused topology exists to prevent (see the
///    dscovox_node.cpp header banner).
///
/// Threading: none. The node drives ingest/drain from a single-threaded
/// executor; the core is plain single-threaded state.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include <bonxai/bonxai.hpp>

#include "scovox/beta_voxel.hpp"
#include "scovox/binary_serializer.hpp"
#include "scovox/dir_voxel.hpp"
#include "scovox/lz4_codec.hpp"

namespace scovox {

class ShareShaperCore {
 public:
  struct Config {
    /// Token-bucket refill rate for the shaped stream, in megabits/s of
    /// COMPRESSED (on-the-wire) bytes. The knob is staleness-vs-radio: below
    /// the steady-state input rate the backlog coalesces (bounded by map
    /// size, never by time) and the peer map lags; it never corrupts.
    double target_rate_mbps = 5.0;
    /// Bucket depth in bytes. Worst-case instantaneous burst is
    /// burst_bytes + one compressed chunk (the bucket allows an overdraft of
    /// one chunk — see drain()).
    std::size_t burst_bytes = 256 * 1024;
    /// UNCOMPRESSED payload budget per chunk, in bytes. Keeps individual DDS
    /// samples small enough to avoid the IP-fragmentation storms that a
    /// multi-MB snapshot sample causes on WiFi. Raised to kMinChunkBytes if
    /// set below it (a budget too small for the largest record could pop
    /// nothing at all, or starve the larger stream forever).
    std::size_t max_chunk_bytes = 128 * 1024;
    /// Test seam only: replaces the LZ4 compressor. Empty (the default) means
    /// the production ScovoxBinarySerializer::compressLZ4. It exists so the
    /// compress-failure re-queue path — the reason no update is ever lost
    /// silently — is reachable from a unit test.
    std::function<std::vector<uint8_t>(const std::string&)> compress;
  };

  /// One shaped wire message: an LZ4-compressed BinarySerializer frame plus
  /// bookkeeping for the node's diagnostics.
  struct Chunk {
    std::vector<uint8_t> data;      ///< compressed blob for ScovoxMapBinary.data
    std::size_t n_beta = 0;
    std::size_t n_dir = 0;
    std::size_t uncompressed_bytes = 0;
  };

  struct IngestResult {
    bool accepted = false;
    /// Human-readable reason when !accepted (prior mismatch, unusable prior).
    const char* reject_reason = nullptr;
    /// True when this frame started a NEW epoch: the store was cleared and the
    /// pin re-taken because the sender's resolution changed under us. The node
    /// logs this loudly — peers keep their old-epoch voxels (the merger never
    /// evicts), so their fused map needs a merger restart to be clean again.
    bool epoch_reset = false;
    const char* epoch_reason = nullptr;
    std::size_t beta_changed = 0;   ///< beta deltas that differed from the store
    std::size_t dir_changed = 0;    ///< dir deltas that differed from the store
    std::size_t tsdf_dropped = 0;   ///< tsdf deltas discarded (out of scope)
  };

  explicit ShareShaperCore(const Config& cfg) : cfg_(cfg) {
    // Guard rails, so a mis-set config degrades instead of wedging: with a
    // chunk budget below one record buildChunk() can pop NOTHING (a permanent
    // stall that drain() would misreport as an LZ4 failure), and with a budget
    // between the two record sizes the larger stream starves forever while the
    // smaller keeps flowing — silent, because output looks healthy.
    cfg_.max_chunk_bytes = std::max(cfg_.max_chunk_bytes, kMinChunkBytes);
    // A non-positive rate never refills the bucket: nothing would ever ship,
    // with no error anywhere but out=0.00 in the diag line.
    if (!(cfg_.target_rate_mbps > 0.0))
      cfg_.target_rate_mbps = Config{}.target_rate_mbps;
    // Start with a full bucket so the first frames after bring-up go out
    // immediately instead of waiting one refill period.
    tokens_ = static_cast<double>(cfg_.burst_bytes);
  }

  /// Ingest one decoded upstream frame. The FIRST accepted frame pins
  /// (resolution, num_classes, alpha_0). Afterwards:
  ///  - a PRIOR mismatch (num_classes/alpha_0) is rejected whole, with the same
  ///    1e-6 tolerance the merger uses — mixing priors mis-subtracts in the
  ///    refold, and the merger would reject the frame anyway;
  ///  - a RESOLUTION change starts a new epoch (store cleared, pin re-taken),
  ///    because the merger has no resolution guard and a rejecting shaper would
  ///    silently freeze peers. See the Epochs note in the file header.
  /// Values identical to the store are absorbed with zero output — this is what
  /// makes a redundant sender snapshot (e.g. a debug subscriber attaching
  /// upstream) produce no shaped traffic at all.
  IngestResult ingest(const BinarySerializer::Frame& frame) {
    IngestResult r;
    // Refuse to PIN anything unusable: the pin is permanent for the epoch and
    // is written into every outgoing frame's header. deserialize() already
    // rejects these upstream, so this is defence in depth for direct callers.
    if (!(frame.resolution > 0.0f) || !std::isfinite(frame.resolution) ||
        frame.num_classes == 0 || !(frame.alpha_0 > 0.0f) ||
        !std::isfinite(frame.alpha_0)) {
      r.reject_reason = "unusable resolution/prior in frame header";
      return r;
    }
    if (pinned_ && (frame.num_classes != num_classes_ ||
                    std::fabs(frame.alpha_0 - alpha_0_) > 1e-6f)) {
      r.reject_reason = "num_classes/alpha_0 mismatch vs pinned";
      return r;
    }
    if (pinned_ && frame.resolution != resolution_) {
      // New epoch: the old coords are in a different metric grid, so nothing
      // in the store can be re-expressed. Drop it and start over.
      clearStore();
      r.epoch_reset = true;
      r.epoch_reason = "sender resolution changed — store dropped, re-pinned";
    }
    if (!pinned_ || r.epoch_reset) {
      pinned_ = true;
      resolution_ = frame.resolution;
      num_classes_ = frame.num_classes;
      alpha_0_ = frame.alpha_0;
    }
    r.accepted = true;
    r.tsdf_dropped = frame.tsdf_deltas.size();

    for (const auto& d : frame.beta_deltas)
      r.beta_changed += upsert(beta_, beta_fifo_, d.coord, d.data) ? 1 : 0;
    for (const auto& d : frame.dir_deltas)
      r.dir_changed += upsert(dir_, dir_fifo_, d.coord, d.data) ? 1 : 0;
    return r;
  }

  /// Mark every stored coord dirty: a full paced resync. Called by the node
  /// when a NEW downstream subscriber matches (a peer merger reconnecting
  /// after an outage, or joining fresh). The whole store then trickles out
  /// under the same token bucket as ordinary deltas — this REPLACES the
  /// sender's one-message snapshot dump for the radio path. Note the shaped
  /// topic is shared by all peers, so a resync for one reconnecting peer is
  /// re-sent to the others too; with a 2-robot fleet that is moot, and
  /// per-peer topics are the escape hatch if it ever isn't.
  void markAllDirty() {
    remarkAll(beta_, beta_fifo_);
    remarkAll(dir_, dir_fifo_);
  }

  /// Drop everything held for resync, keeping the pin. Called on an epoch
  /// change: a resolution change (from ingest) or a map_from_source
  /// discontinuity (from the node), either of which means the stored coords no
  /// longer describe where the sender thinks that evidence is. Replaying them
  /// under the new pose/grid would place a whole mission's terrain wrongly in a
  /// peer that restarts later, which is worse than not resending it at all.
  void clearStore() {
    beta_.clear();
    dir_.clear();
    beta_fifo_.clear();
    dir_fifo_.clear();
  }

  /// Advance the token bucket by dt seconds and emit as many chunks as it
  /// allows. Chunks alternate beta/dir records (semantics are a first-class
  /// planner input — a resync must not starve Dir behind the free-space Beta
  /// bulk). The bucket spends ACTUAL compressed bytes and may overdraft by up
  /// to one chunk (compressed size is only known after packing); the debt
  /// just delays the next emission, keeping the logic single-pass.
  std::vector<Chunk> drain(double dt_sec) {
    refill(dt_sec);
    std::vector<Chunk> out;
    if (!pinned_) return out;
    while (tokens_ > 0.0 && (!beta_fifo_.empty() || !dir_fifo_.empty())) {
      Chunk c = buildChunk();
      // With the budget clamped to kMinChunkBytes a chunk always carries at
      // least one record, so an empty blob can ONLY mean the compressor
      // failed — and buildChunk has already re-marked those entries dirty.
      if (c.data.empty()) break;
      tokens_ -= static_cast<double>(c.data.size());
      out.push_back(std::move(c));
    }
    return out;
  }

  /// Advance the bucket WITHOUT emitting. The node calls this instead of
  /// drain() while no peer is subscribed, so an idle link still earns credit
  /// (plain token-bucket semantics): otherwise a link that dropped mid-backlog
  /// — tokens near zero — would rejoin with an empty bucket and trickle the
  /// resync out at one tick's worth per tick.
  void refill(double dt_sec) {
    tokens_ = std::min(static_cast<double>(cfg_.burst_bytes),
                       tokens_ + rateBytesPerSec() * dt_sec);
  }

  /// Clamp the bucket to one chunk of credit. Called when a peer matches:
  /// bring-up deliberately starts with a FULL bucket (the first deltas go out
  /// without waiting), but a re-join must not open with a burst_bytes dump at
  /// the exact moment the link is weakest — that burst is what this node exists
  /// to remove.
  void capTokensForResync() {
    tokens_ = std::min(tokens_, static_cast<double>(cfg_.max_chunk_bytes));
  }

  bool pinned() const { return pinned_; }
  float resolution() const { return resolution_; }
  uint16_t numClasses() const { return num_classes_; }
  float alpha0() const { return alpha_0_; }
  std::size_t storeBetaCount() const { return beta_.size(); }
  std::size_t storeDirCount() const { return dir_.size(); }
  std::size_t dirtyCount() const { return beta_fifo_.size() + dir_fifo_.size(); }
  /// True iff the last buildChunk hit an LZ4 error (diagnostic only).
  bool lastCompressFailed() const { return compress_failed_; }

  /// Wire record sizes (bytes, uncompressed), taken from the serializer itself
  /// rather than re-derived: buildChunk() budgets a frame BEFORE serializing
  /// it, and a hand-copied layout that drifted from serialize() would silently
  /// overrun max_chunk_bytes (re-introducing the fragmentation this node
  /// exists to prevent) or under-fill every chunk.
  static constexpr std::size_t kPerBeta = BinarySerializer::kPerBeta;
  static constexpr std::size_t kPerDir = BinarySerializer::kPerDir;
  /// Fixed frame cost (header + the three stream counts), reserved out of the
  /// per-chunk budget so the SERIALIZED blob honours max_chunk_bytes exactly.
  static constexpr std::size_t kFrameOverhead = BinarySerializer::kFrameOverhead;
  /// Smallest chunk budget that can carry either record type. Config is raised
  /// to this — below it, emission stalls or one stream starves (see the
  /// constructor).
  static constexpr std::size_t kMinChunkBytes =
      kFrameOverhead + (kPerBeta > kPerDir ? kPerBeta : kPerDir);

 private:
  // Same CoordT hash/equality as dscovox_node's touched-set (Boost-style mix).
  struct CoordTHash {
    std::size_t operator()(const Bonxai::CoordT& c) const noexcept {
      auto h = std::hash<int32_t>{}(c.x);
      h ^= std::hash<int32_t>{}(c.y) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
      h ^= std::hash<int32_t>{}(c.z) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
      return h;
    }
  };
  struct CoordTEqual {
    bool operator()(const Bonxai::CoordT& a, const Bonxai::CoordT& b) const noexcept {
      return a.x == b.x && a.y == b.y && a.z == b.z;
    }
  };

  /// Latest-value store entry. INVARIANT: `dirty` is true iff the coord is in
  /// the corresponding FIFO exactly once — upsert only pushes on a false→true
  /// transition, so a coord re-dirtied while queued is NOT re-pushed (its
  /// store value is simply newer when the FIFO reaches it: coalescing).
  template <typename V>
  struct Entry {
    V value{};
    bool dirty = false;
  };
  template <typename V>
  using Store = std::unordered_map<Bonxai::CoordT, Entry<V>, CoordTHash, CoordTEqual>;

  /// Returns true when the store value changed (new coord or different
  /// bytes). BetaVoxel/DirVoxel are trivial standard-layout PODs with no
  /// padding (static_asserts in their headers), so memcmp is an exact
  /// value compare.
  template <typename V>
  static bool upsert(Store<V>& store, std::deque<Bonxai::CoordT>& fifo,
                     const Bonxai::CoordT& c, const V& v) {
    auto [it, inserted] = store.try_emplace(c);
    if (!inserted && std::memcmp(&it->second.value, &v, sizeof(V)) == 0)
      return false;
    it->second.value = v;
    if (!it->second.dirty) {
      it->second.dirty = true;
      fifo.push_back(c);
    }
    return true;
  }

  template <typename V>
  static void remarkAll(Store<V>& store, std::deque<Bonxai::CoordT>& fifo) {
    // Rebuild the FIFO from scratch so the coord-queued-once invariant holds
    // regardless of how many entries were already dirty.
    fifo.clear();
    for (auto& [c, e] : store) {
      e.dirty = true;
      fifo.push_back(c);
    }
  }

  double rateBytesPerSec() const { return cfg_.target_rate_mbps * 1e6 / 8.0; }

  /// Pop dirty coords into one frame up to max_chunk_bytes (uncompressed),
  /// serialize + LZ4. On the (essentially unreachable) LZ4 failure the popped
  /// entries are re-marked dirty so no update is silently lost — unlike the
  /// raw sender path, which drops the frame outright in that case.
  Chunk buildChunk() {
    compress_failed_ = false;
    BinarySerializer::Frame f;
    f.resolution = resolution_;
    f.num_classes = num_classes_;
    f.alpha_0 = alpha_0_;

    // The constructor guarantees max_chunk_bytes >= kMinChunkBytes, so the
    // budget always fits at least one record of either type.
    std::size_t budget = cfg_.max_chunk_bytes - kFrameOverhead;
    // Strict alternation while both queues are non-empty, then drain the rest
    // of whichever remains. Terminates via the break: every taken record
    // shrinks the budget, and the queues are finite. The turn is a MEMBER so
    // alternation carries across chunks too: with a budget that fits only one
    // record per chunk, restarting at beta every time would starve Dir for as
    // long as new Beta keeps arriving.
    while (true) {
      const bool beta_avail = !beta_fifo_.empty() && budget >= kPerBeta;
      const bool dir_avail = !dir_fifo_.empty() && budget >= kPerDir;
      if (!beta_avail && !dir_avail) break;
      if ((take_beta_ && beta_avail) || !dir_avail) {
        const Bonxai::CoordT c = beta_fifo_.front();
        beta_fifo_.pop_front();
        auto& e = beta_.at(c);
        e.dirty = false;
        f.beta_deltas.push_back({c, e.value});
        budget -= kPerBeta;
      } else {
        const Bonxai::CoordT c = dir_fifo_.front();
        dir_fifo_.pop_front();
        auto& e = dir_.at(c);
        e.dirty = false;
        f.dir_deltas.push_back({c, e.value});
        budget -= kPerDir;
      }
      take_beta_ = !take_beta_;
    }

    Chunk out;
    out.n_beta = f.beta_deltas.size();
    out.n_dir = f.dir_deltas.size();
    if (f.beta_deltas.empty() && f.dir_deltas.empty()) return out;

    // Options default = share_tsdf=false → tsdf_count=0 on the wire, matching
    // the production sender (TSDF never crosses to dscovox).
    const std::string blob = BinarySerializer::serialize(f);
    out.uncompressed_bytes = blob.size();
    out.data = cfg_.compress ? cfg_.compress(blob)
                             : ScovoxBinarySerializer::compressLZ4(blob);
    compress_failed_ = out.data.empty();
    if (compress_failed_) {
      // Undo the dirty-clear: push the popped coords back (tail order is
      // fine — the merger is order-independent across coords).
      for (const auto& d : f.beta_deltas) {
        auto& e = beta_.at(d.coord);
        if (!e.dirty) { e.dirty = true; beta_fifo_.push_back(d.coord); }
      }
      for (const auto& d : f.dir_deltas) {
        auto& e = dir_.at(d.coord);
        if (!e.dirty) { e.dirty = true; dir_fifo_.push_back(d.coord); }
      }
      out.n_beta = out.n_dir = 0;
    }
    return out;
  }

  Config cfg_;
  double tokens_ = 0.0;
  bool pinned_ = false;
  bool compress_failed_ = false;
  bool take_beta_ = true;   ///< whose turn it is (see buildChunk's alternation)
  float resolution_ = 0.0f;
  uint16_t num_classes_ = 0;
  float alpha_0_ = 0.0f;
  Store<BetaVoxel> beta_;
  Store<DirVoxel> dir_;
  std::deque<Bonxai::CoordT> beta_fifo_;
  std::deque<Bonxai::CoordT> dir_fifo_;
};

}  // namespace scovox
