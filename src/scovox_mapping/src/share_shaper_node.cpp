// share_shaper_node.cpp — per-robot scovox_bin egress pacer.
//
// Runs ON the sending robot, between its scovox_node and the radio:
//
//   scovox_node ── scovox_bin ──┬── dscovox_node   (self: RAW loopback)
//                               └── share_shaper ── scovox_bin_shaped ──(radio)── peers' dscovox
//
// The robot's OWN merger stays on the raw topic (zero added latency for the
// planner's fused map, and a resync for a reconnecting peer can never starve
// the local map behind the token bucket). Only PEERS subscribe to the shaped
// topic. Shaping must happen before the radio to help — a receive-side buffer
// would still let the burst hit the WiFi.
//
// What this fixes at the sender (see scovox_node publishBinaryMap):
//  - The reconnect snapshot is a full-grid dump in ONE message, landing on the
//    link at the moment it is weakest. Here it becomes a paced trickle: the
//    shaper keeps a latest-value-per-coord store and, on a new downstream
//    match, re-marks it all dirty and drains it under the same token bucket
//    as ordinary deltas (ShareShaperCore::markAllDirty).
//  - The sender's snapshot trigger is a subscriber COUNT comparison, which
//    can miss a reconnect when counts coincide. With the shaper in place the
//    raw topic's subscriber set is {local merger, local shaper} — static
//    after bring-up — and reconnects are detected HERE via publisher matched
//    events (exact +1/-1 transitions, not count polling).
//
// Transparency contract: header.frame_id (the merger's source key) and
// map_from_source (the carried pose) pass through verbatim; coords are never
// transformed; priors are pinned from the first frame and enforced. Every
// envelope field dscovox reads is reproduced, so a merger cannot tell a shaped
// stream from a raw one apart from timing. (header.stamp is the shaper's own
// drain time rather than the sender's — nothing in dscovox reads it.)
//
// !! LOCAL STREAM ONLY !! One shaper per robot, fed by that robot's OWN
// scovox_node. Feeding it a fused/dscovox output would re-introduce the
// evidence-echo the one-way local→fused topology exists to prevent (see the
// dscovox_node.cpp header banner).

#include <rclcpp/rclcpp.hpp>
#include <scovox_msgs/msg/scovox_map_binary.hpp>

#include "scovox/share_shaper.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>

class ShareShaperNode : public rclcpp::Node {
 public:
  ShareShaperNode() : rclcpp::Node("share_shaper") {
    // Relative default resolves under the robot namespace:
    // /<robot>/scovox_node/scovox_bin — the same per-robot raw stream the
    // local merger reads.
    const auto in_topic =
        declare_parameter<std::string>("input_topic", "scovox_node/scovox_bin");
    const auto out_topic =
        declare_parameter<std::string>("output_topic", "~/scovox_bin_shaped");

    scovox::ShareShaperCore::Config cfg;
    cfg.target_rate_mbps = declare_parameter<double>("target_rate_mbps", 5.0);
    if (!(cfg.target_rate_mbps > 0.0)) {
      // Would never refill the bucket: the shaper would ingest happily and
      // publish nothing at all, visible only as out=0.00 in the diag line.
      RCLCPP_ERROR(get_logger(),
                   "target_rate_mbps=%.3f is not positive — falling back to "
                   "5.0 Mbps (a zero rate would silently publish nothing)",
                   cfg.target_rate_mbps);
      cfg.target_rate_mbps = 5.0;
    }
    cfg.burst_bytes = static_cast<std::size_t>(
        std::max<int64_t>(1, declare_parameter<int64_t>("burst_kb", 256)) * 1024);
    cfg.max_chunk_bytes = static_cast<std::size_t>(
        std::max<int64_t>(1, declare_parameter<int64_t>("max_chunk_kb", 128)) * 1024);
    tick_hz_ = declare_parameter<double>("tick_hz", 10.0);
    if (!(tick_hz_ > 0.0)) tick_hz_ = 10.0;
    // Each tick can spend at most one bucketful, so burst_bytes * tick_hz is a
    // hard ceiling on sustained throughput regardless of target_rate_mbps.
    const double ceiling_mbps =
        static_cast<double>(cfg.burst_bytes) * tick_hz_ * 8.0 / 1e6;
    if (cfg.target_rate_mbps > ceiling_mbps) {
      RCLCPP_WARN(get_logger(),
                  "target_rate_mbps=%.2f exceeds the burst*tick ceiling of "
                  "%.2f Mbps — the stream will cap there; raise burst_kb or "
                  "tick_hz to actually reach the requested rate",
                  cfg.target_rate_mbps, ceiling_mbps);
    }
    core_ = std::make_unique<scovox::ShareShaperCore>(cfg);

    // The INPUT hop mirrors the sender's QoS (reliable, KeepLast(50)) — it is
    // a loopback stream, so the sender's own depth is the relevant bound.
    const auto in_qos = rclcpp::QoS(rclcpp::KeepLast(50)).reliable();
    // The OUTPUT hop is the radio, and its history depth is the retransmit
    // window: RELIABLE + KEEP_LAST recycles the oldest sample once the writer
    // cache wraps, ACKed or not. Depth must therefore outlast the time it takes
    // to NOTICE a dead peer (DDS liveliness lease, ~10-20 s), or an outage in
    // between — longer than the window, shorter than the lease — drops chunks
    // with no unmatch event and so no resync, leaving permanent holes in the
    // peer's map for any voxel the robot has since driven away from. At 5 Mbps
    // and ~64 KB compressed chunks the stream is ~10 chunks/s, so the default
    // 300 buys ~30 s of history for ~20-40 MB of writer cache.
    const auto out_qos =
        rclcpp::QoS(rclcpp::KeepLast(static_cast<std::size_t>(std::max<int64_t>(
                        1, declare_parameter<int64_t>("out_history_depth", 300)))))
            .reliable();

    // Matched events give exact transitions: a peer merger joining or
    // RE-joining after an outage → full paced resync. The unmatch side is
    // logged too — it is the direct "peer stopped listening" signal the raw
    // path never had.
    rclcpp::PublisherOptions pub_opts;
    pub_opts.event_callbacks.matched_callback =
        [this](const rclcpp::MatchedInfo& info) {
          const std::size_t cur = static_cast<std::size_t>(info.current_count);
          // current_count_change is a delta since the last time this event was
          // TAKEN, so an unmatch+rematch that lands inside one executor service
          // nets to zero — and the rejoined reader (new GUID, empty history)
          // would never get its resync. Drive the decision off our own previous
          // count and treat a no-net-change callback with peers present as a
          // swap. Resync is idempotent, so over-firing is the safe direction.
          const bool joined = cur > prev_matched_ || (cur > 0 && cur == prev_matched_);
          const bool left = cur < prev_matched_;
          prev_matched_ = cur;

          if (joined) {
            // A re-join must not discharge an idle-earned bucket as one burst.
            core_->capTokensForResync();
            const std::size_t store =
                core_->storeBetaCount() + core_->storeDirCount();
            // Skip the re-mark only when it would be a literal no-op (the whole
            // store is already queued) — that absorbs sub-tick discovery flaps
            // for free. Anything else re-marks: a genuinely new peer needs the
            // parts already drained, so a flapping link restarting the resync is
            // the conservative choice, not a bug. Watch resyncs= in the diag.
            if (store == 0) {
              RCLCPP_INFO(get_logger(),
                          "downstream subscriber matched (now %zu) — nothing "
                          "stored yet; deltas will flow as they arrive",
                          cur);
            } else if (core_->dirtyCount() >= store) {
              RCLCPP_INFO(get_logger(),
                          "downstream subscriber matched (now %zu) — resync "
                          "already fully queued (%zu voxels), not restarting it",
                          cur, store);
            } else {
              ++resyncs_;
              core_->markAllDirty();
              RCLCPP_INFO(get_logger(),
                          "downstream subscriber matched (now %zu) — starting "
                          "paced full resync of %zu voxels",
                          cur, store);
            }
          } else if (left) {
            RCLCPP_WARN(get_logger(),
                        "downstream subscriber unmatched (now %zu) — a peer "
                        "merger stopped listening (link loss or shutdown)",
                        cur);
          }
        };

    pub_ = create_publisher<scovox_msgs::msg::ScovoxMapBinary>(out_topic, out_qos,
                                                               pub_opts);
    sub_ = create_subscription<scovox_msgs::msg::ScovoxMapBinary>(
        in_topic, in_qos,
        std::bind(&ShareShaperNode::onBinary, this, std::placeholders::_1));

    timer_ = rclcpp::create_timer(
        this, get_clock(),
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(1.0 / tick_hz_)),
        std::bind(&ShareShaperNode::onTick, this));

    RCLCPP_INFO(get_logger(),
                "share_shaper started: '%s' -> '%s' at %.2f Mbps "
                "(burst %zu KB, chunk %zu KB, tick %.1f Hz, out history %zu)",
                in_topic.c_str(), out_topic.c_str(), cfg.target_rate_mbps,
                cfg.burst_bytes / 1024, cfg.max_chunk_bytes / 1024, tick_hz_,
                out_qos.depth());
  }

 private:
  void onBinary(const scovox_msgs::msg::ScovoxMapBinary::SharedPtr msg) {
    // Same envelope gates as the merger (dscovox onBinaryMap): the shaper
    // decodes on this host, so a version or endianness mismatch is just as
    // fatal here as there.
    if (msg->version != 5) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                           "expected envelope version 5, got %d (dropping)",
                           msg->version);
      return;
    }
    constexpr bool kHostLittleEndian =
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        true;
#else
        false;
#endif
    if (msg->little_endian != kHostLittleEndian) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                           "wire endianness mismatch from '%s' — dropping",
                           msg->header.frame_id.c_str());
      return;
    }
    // One shaper serves ONE upstream stream. A second frame_id means someone
    // remapped two robots' streams into this node, which would corrupt the
    // merger's source keying AND associate one robot's coords with the other's
    // carried pose — reject loud.
    if (src_frame_id_.empty()) {
      src_frame_id_ = msg->header.frame_id;
    } else if (msg->header.frame_id != src_frame_id_) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000,
                            "frame_id '%s' != pinned '%s' — one shaper per "
                            "robot; dropping",
                            msg->header.frame_id.c_str(), src_frame_id_.c_str());
      return;
    }
    const std::string buf = scovox::ScovoxBinarySerializer::decompressLZ4(msg->data);
    if (buf.empty()) {
      RCLCPP_ERROR(get_logger(), "LZ4 fail from '%s'", src_frame_id_.c_str());
      return;
    }
    scovox::BinarySerializer::Frame frame;
    try {
      frame = scovox::BinarySerializer::deserialize(buf);
    } catch (const std::exception& e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                           "deserialize failed: %s", e.what());
      return;
    }

    // A map_from_source discontinuity means the sender's source->map placement
    // moved: it was restarted and re-established its origin elsewhere, or c-slam
    // is live (which this topology forbids). Either way the stored coords no
    // longer describe where that evidence belongs, and replaying them under the
    // new pose on some later peer's resync would misplace a whole mission's
    // terrain. Under the intended static-TF regime the pose is bit-identical
    // frame after frame, so the tolerance below only absorbs float noise.
    if (have_pose_ && !sameTransform(pose_, msg->map_from_source)) {
      RCLCPP_WARN(get_logger(),
                  "map_from_source changed (sender restarted with a new origin, "
                  "or c-slam is publishing) — dropping the %zu-voxel resync "
                  "store; peers keep their old-epoch voxels until their merger "
                  "restarts",
                  core_->storeBetaCount() + core_->storeDirCount());
      core_->clearStore();
    }

    const auto r = core_->ingest(frame);
    if (!r.accepted) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                           "frame rejected: %s — dropping", r.reject_reason);
      return;
    }
    // Only a frame we actually accepted may re-stamp the pose the outgoing
    // chunks carry; otherwise a rejected frame from a reconfigured sender would
    // relabel the whole existing store with its new pose.
    pose_ = msg->map_from_source;
    have_pose_ = true;
    if (r.epoch_reset) {
      RCLCPP_WARN(get_logger(),
                  "new epoch from '%s': %s — the merger has no resolution guard, "
                  "so peers keep their old-epoch voxels until their merger "
                  "restarts",
                  src_frame_id_.c_str(), r.epoch_reason);
    }
    if (r.tsdf_dropped > 0 && !warned_tsdf_) {
      warned_tsdf_ = true;
      RCLCPP_WARN(get_logger(),
                  "upstream frames carry TSDF deltas (%zu in this frame) — the "
                  "shaper forwards Beta+Dir only. Nothing is lost relative to "
                  "the raw path: dscovox never reads the TSDF stream either "
                  "(and share_tsdf is off in every production config)",
                  r.tsdf_dropped);
    }
    ++frames_in_;
  }

  /// Exact-under-discipline pose compare: the tolerances only exist so float
  /// noise in a re-published static TF can't be mistaken for a moved origin.
  static bool sameTransform(const geometry_msgs::msg::Transform& a,
                            const geometry_msgs::msg::Transform& b) {
    constexpr double kTransTol = 1e-3;  // 1 mm
    constexpr double kRotTol = 1e-4;    // quaternion component
    return std::abs(a.translation.x - b.translation.x) < kTransTol &&
           std::abs(a.translation.y - b.translation.y) < kTransTol &&
           std::abs(a.translation.z - b.translation.z) < kTransTol &&
           std::abs(a.rotation.x - b.rotation.x) < kRotTol &&
           std::abs(a.rotation.y - b.rotation.y) < kRotTol &&
           std::abs(a.rotation.z - b.rotation.z) < kRotTol &&
           std::abs(a.rotation.w - b.rotation.w) < kRotTol;
  }

  void onTick() {
    const rclcpp::Time now = get_clock()->now();
    // Refill from measured elapsed time, not the nominal period (robust to
    // timer jitter and sim-time stalls). Clamp both ends: a backwards clock
    // jump refills nothing, and a FORWARDS jump refills at most 10 nominal
    // periods. Without the upper clamp a stepped clock (NTP/chrony correction
    // in the field, or two fighting /clock publishers in replay) makes every
    // tick see a huge dt, refills the bucket to cap each time, and the "paced"
    // stream runs at tick_hz × burst_bytes — 5× the target in testing. Lost
    // refill only delays delivery; it never corrupts.
    double dt = 1.0 / tick_hz_;
    if (last_tick_.nanoseconds() > 0) {
      dt = std::clamp((now - last_tick_).seconds(), 0.0, 10.0 / tick_hz_);
    }
    last_tick_ = now;

    // No subscriber → refill the bucket but emit nothing. Ingest keeps running
    // regardless, so the store and its dirty FIFOs stay current; the resync
    // callback above re-queues everything the moment a peer matches, and caps
    // the credit earned here to one chunk so the rejoin isn't itself a burst.
    if (pub_->get_subscription_count() == 0) {
      core_->refill(dt);
    } else {
      constexpr bool kHostLittleEndian =
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
          true;
#else
          false;
#endif
      for (auto& chunk : core_->drain(dt)) {
        scovox_msgs::msg::ScovoxMapBinary bin;
        bin.header.stamp = now;
        bin.header.frame_id = src_frame_id_;
        bin.version = 5;
        bin.little_endian = kHostLittleEndian;
        bin.map_from_source = pose_;
        bytes_out_ += chunk.data.size();
        bin.data = std::move(chunk.data);
        pub_->publish(std::move(bin));
        ++chunks_out_;
      }
      if (core_->lastCompressFailed()) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                             "LZ4 compression failed; entries re-queued");
      }
    }

    // Diag on a manual 5 s window so the printed Mbps matches the window it
    // was measured over (mirrors dscovox_diag's cadence).
    if (last_diag_.nanoseconds() == 0) last_diag_ = now;
    const double win = (now - last_diag_).seconds();
    if (win < 0.0) {
      // Clock went backwards (bag replay looping). Rebase instead of waiting
      // for the old timestamp to be overtaken, which would silence the diag for
      // as long as the rewind was.
      last_diag_ = now;
      diag_bytes_ = bytes_out_;
    } else if (win >= 5.0) {
      const double mbps =
          static_cast<double>(bytes_out_ - diag_bytes_) * 8.0 / (win * 1e6);
      RCLCPP_INFO(get_logger(),
                  "shaper_diag: in_frames=%zu out_chunks=%zu out=%.2f Mbps "
                  "backlog=%zu store=%zu/%zu (beta/dir) resyncs=%zu",
                  frames_in_, chunks_out_, mbps, core_->dirtyCount(),
                  core_->storeBetaCount(), core_->storeDirCount(), resyncs_);
      last_diag_ = now;
      diag_bytes_ = bytes_out_;
    }
  }

  std::unique_ptr<scovox::ShareShaperCore> core_;
  double tick_hz_{10.0};
  std::string src_frame_id_;
  geometry_msgs::msg::Transform pose_;
  bool have_pose_{false};
  bool warned_tsdf_{false};
  /// Our own view of the matched-subscriber count — matched_callback reports a
  /// delta since the last take, which coalesces an unmatch+rematch to zero.
  std::size_t prev_matched_{0};
  std::size_t frames_in_{0}, chunks_out_{0}, bytes_out_{0}, diag_bytes_{0};
  std::size_t resyncs_{0};
  rclcpp::Time last_tick_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_diag_{0, 0, RCL_ROS_TIME};
  rclcpp::Subscription<scovox_msgs::msg::ScovoxMapBinary>::SharedPtr sub_;
  rclcpp::Publisher<scovox_msgs::msg::ScovoxMapBinary>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  // SingleThreadedExecutor: the subscription callback, the pacer timer, and
  // the publisher matched-event handler all run on one thread, so the core
  // needs no locking.
  rclcpp::spin(std::make_shared<ShareShaperNode>());
  rclcpp::shutdown();
}
