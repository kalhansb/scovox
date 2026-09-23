// dscovox_node.cpp — multi-robot SCovox map merger.
//
// Inputs are each robot's LOCAL grid deltas, never a fused grid. Do not wire a
// dscovox ~/scovox output back in as an input: the evidence echo breaks the
// additive consensus. (notes: dscovox-local-to-fused-topology)
//
// K_TOP semantic-slot truncation lives in scovox (voxel.hpp / sparse_add)
// and is already applied by the time a binary reaches this node — the wire
// format carries at most K_TOP slots per voxel. dscovox cannot widen that;
// any K_TOP-related ablation belongs at the scovox layer (B1).
//
// One source grid per robot, keyed by header.frame_id, stored in map-frame
// coords: each delta voxel is transformed once by the pose carried in
// map_from_source. No TF listener. (notes: dscovox-source-grids-map-frame)
//
// !! REQUIRES c-slam DISABLED !!
// The first carried source->map pose is cached and never refreshed: correct
// only while TFs are static. Under loop closures voxels stay at stale map
// coords (ghosts); SourceGrid would need source-frame storage.
// (notes: dscovox-requires-cslam-disabled)
//
// Each binary refolds only the touched map-frame cells: reset fused[c] to the
// prior, then fold every source's current value (a_fused = a_1 + a_2 - 1).
// Equals a full rebuild; cannot double-count.
// (notes: dscovox-reset-then-refold)
//
// No submaps. No pose graph. No loop closures. No periodic rebuild.
// Moved comments: doc/dscovox_node_notes.md

#include <rclcpp/rclcpp.hpp>
#include <scovox_msgs/msg/scovox_map_binary.hpp>
#include <scovox_msgs/msg/scovox_map.hpp>
#include <scovox_msgs/msg/scovox_fusion_counters.hpp>
#include <scovox/binary_serializer.hpp>
#include <scovox/consensus_merge.hpp>
#include <scovox/lz4_codec.hpp>
#include <scovox/sembeta_voxel.hpp>
#include <scovox_msgs/srv/get_region.hpp>
#include <scovox_msgs/srv/get_occupancy_grid.hpp>
#include "scovox/scovoxmap.hpp"
#include "scovox/node_utils.hpp"
#include "scovox/dscovox_consensus.hpp"  // isPrior*/projectBetaDir*/refold* (shared w/ tests)
#include <bonxai/bonxai.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <geometry_msgs/msg/transform.hpp>
#include <scovox/uncertainty.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <shared_mutex>
#include <mutex>
#include <atomic>  // last_pc_pub_ns_ rate-limiter (shared-lock concurrent access)
#include <chrono>  // plan_glob_last_ steady_clock rate-limiter

namespace {

struct SourceGrid {
  std::string source_frame;                // header.frame_id of the binary
  // This source's latest (occupancy, semantics) snapshot per voxel, in
  // MAP-FRAME coords: beta_grid and dir_grid. No TSDF grid: TSDF does not cross
  // the wire (share_tsdf=false). (notes: source-grid-split-grids)
  std::unique_ptr<Bonxai::VoxelGrid<scovox::BetaVoxel>> beta_grid;
  std::unique_ptr<Bonxai::VoxelGrid<scovox::DirVoxel>>  dir_grid;
  // Source->map transform from the first update's carried map_from_source;
  // never refreshed, so valid only while TFs are static (c-slam disabled).
  // (notes: source-grid-cached-pose)
  Eigen::Isometry3d T_map_source{Eigen::Isometry3d::Identity()};
  bool pose_cached{false};
  // Cumulative per-source totals since node start, published as
  // ScovoxFusionCounters. Written under the unique_lock at ingest; read under a
  // shared lock by the fusion-counters timer.
  // (notes: source-grid-fusion-counters)
  uint64_t deltas_received{0};   // voxel deltas ingested from this source
  uint64_t cells_touched{0};     // fused cells written while integrating them
};

// Sequence tracking per source, kept apart from SourceGrid because it records
// frames the grid never sees: a TSDF-only chunk decodes to no occupancy or
// semantic deltas and returns before ingest, yet it is part of the sender's
// stream and the exchange test needs to know it arrived.
struct SeqTrack {
  uint64_t newest{0};   // highest ScovoxMapBinary.seq received (0 = none)
  uint64_t gaps{0};     // sequence numbers skipped, cumulative
};

inline Eigen::Isometry3d tfToIsometry(const geometry_msgs::msg::Transform& tf) {
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.translation() = Eigen::Vector3d(tf.translation.x,
                                    tf.translation.y,
                                    tf.translation.z);
  T.linear() = Eigen::Quaterniond(tf.rotation.w,
                                  tf.rotation.x,
                                  tf.rotation.y,
                                  tf.rotation.z).toRotationMatrix();
  return T;
}

// The receiver consensus helpers (isPriorBeta, isPriorDir, projectBetaDir*,
// refoldBeta, refoldDir) live in scovox/dscovox_consensus.hpp, shared with the
// unit tests; unqualified calls resolve by ADL.
// (notes: dscovox-consensus-helpers-shared)
} // namespace

class DSCovoxNode : public rclcpp::Node {
public:
  DSCovoxNode()
  : rclcpp::Node("dscovox_node")
  {
    input_topics_ = declare_parameter<std::vector<std::string>>(
        "input_topics", std::vector<std::string>{});
    if (input_topics_.empty()) {
      auto t1 = declare_parameter<std::string>("input_topic_1", "/robot1/scovox_node/scovox_bin");
      auto t2 = declare_parameter<std::string>("input_topic_2", "/robot2/scovox_node/scovox_bin");
      if (!t1.empty()) input_topics_.push_back(t1);
      if (!t2.empty()) input_topics_.push_back(t2);
    }
    {
      std::vector<std::string> uniq;
      for (auto& t : input_topics_)
        if (!t.empty() && std::find(uniq.begin(), uniq.end(), t) == uniq.end())
          uniq.push_back(t);
      input_topics_ = std::move(uniq);
    }
    if (input_topics_.empty()) throw std::runtime_error("No DSCovox input topics configured");

    map_frame_ = declare_parameter<std::string>("map_frame", "map");
    min_occ_ = declare_parameter<double>("occupancy_vis_threshold", 0.7);
    sem_gate_ = declare_parameter<double>("semantic_occ_gate", 0.5);

    // K_TOP is compile-time-locked into the wire Dir record size; a sender
    // built with a different K_TOP will fail deserialization. Announce ours
    // up front so build-skew is diagnosable before frames flow.
    RCLCPP_INFO(get_logger(),
      "dscovox wire (split Beta/Dir): receiver compiled with "
      "K_TOP=%d — every connected sender must match.",
      static_cast<int>(scovox::K_TOP));
    {
      int tk = declare_parameter<int>("semantic_top_k", scovox::K_TOP);
      top_k_ = (tk < 1) ? 1 : (tk > (int)scovox::K_TOP ? (int)scovox::K_TOP : tk);
      if (tk != top_k_) {
        RCLCPP_WARN(get_logger(),
          "semantic_top_k=%d clamped to %d (compile-time K_TOP cap). "
          "To raise this, recompile scovox_core with a larger K_TOP.",
          tk, top_k_);
      }
    }
    pub_hz_ = declare_parameter<double>("publish_rate_hz", 1.0);
    // Minimum interval between visualization-pointcloud publishes when
    // triggered by binary callbacks. Default 0.1s = 10 Hz cap. Lower this
    // for snappier RViz updates at the cost of CPU; raise it to throttle.
    pc_min_interval_s_ = declare_parameter<double>("pointcloud_min_interval_s", 0.1);
    last_pc_pub_ns_.store(get_clock()->now().nanoseconds(), std::memory_order_relaxed);

    // Ingest-side z-band clip on map-frame voxel centres; min >= max disables.
    // KEEP IN SYNC with scovox_node share_roi_z_min/max; must be a superset of
    // explo_planner roi_min_z/roi_max_z. (notes: dscovox-share-roi-z-band)
    share_z_min_ = declare_parameter<double>("share_roi_z_min", 0.0);
    share_z_max_ = declare_parameter<double>("share_roi_z_max", 0.0);

    // --- WORLD-FIXED planning map for the exploration planner ---------------
    // 2D projection over a FIXED envelope for the exploration planner and
    // simple_nav_3d's global planner, which read out-of-bounds as occupied. Not
    // ~/planning_map (scovox_node's rolling crop). Off by default.
    // (notes: dscovox-global-planning-map)
    pub_plan_glob_ = declare_parameter<bool>("publish_global_planning_map", false);
    plan_glob_res_ = declare_parameter<double>("global_planning_map_resolution", 0.40);
    plan_glob_sz_  = declare_parameter<double>("global_planning_map_size_m", 200.0);
    plan_glob_ox_  = declare_parameter<double>("global_planning_map_origin_x", -100.0);
    plan_glob_oy_  = declare_parameter<double>("global_planning_map_origin_y", -100.0);
    plan_glob_infl_ = declare_parameter<double>("global_planning_map_inflation_m", 1.5);
    // z band, map frame. Matches scovox_node's planning_map_min_z/max_z: the
    // slab a UGV body actually sweeps, so canopy above it is not projected
    // down as an obstacle.
    plan_glob_zmin_ = declare_parameter<double>("global_planning_map_min_z", 0.05);
    plan_glob_zmax_ = declare_parameter<double>("global_planning_map_max_z", 1.0);
    // Publish period, seconds. This runs on the publish timer and is O(active
    // voxels) for the projection plus O(occupied * (infl/res)^2) for the
    // inflation, over a message of O(size^2/res^2) bytes. A planner re-reads
    // the latched map about once per planning step, so ~1 Hz is generous.
    plan_glob_period_ = declare_parameter<double>("global_planning_map_period_sec", 1.0);

    initSemanticColors();

    // Explicit reliable + depth 1, mirroring scovox_node's pc_pub_: the fused
    // cloud is tens of MB and best-effort drops fragmented samples. Depth 1
    // caps publisher-side buffering. (notes: dscovox-pointcloud-qos)
    pc_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      declare_parameter<std::string>("pointcloud_topic", "~/pointcloud"),
      rclcpp::QoS(rclcpp::KeepLast(1)).reliable());

    // Full ScovoxMap snapshot of the fused Beta/Dir grids (topic form of
    // GetRegion), published by the publish timer when changed and subscribed.
    // Latched (KeepLast(1), reliable, transient_local): subscribers MUST match.
    // (notes: dscovox-fused-map-topic)
    scovox_map_pub_ = create_publisher<scovox_msgs::msg::ScovoxMap>(
      declare_parameter<std::string>("scovox_topic", "~/scovox"),
      rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local());

    // Latched QoS must match the exploration planner's subscriber exactly, or
    // the planner waits in INIT with no error. The topic is declared even when
    // disabled so ros2 param get reports it.
    // (notes: dscovox-global-planning-map-qos)
    {
      auto plg_t = declare_parameter<std::string>("global_planning_map_topic",
                                                 std::string("~/global_planning_map"));
      if (pub_plan_glob_) {
        pl_glob_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>(
          plg_t, rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local());
        RCLCPP_INFO(get_logger(),
          "dscovox global_planning_map: %.1f m envelope @ %.2f m/cell "
          "(origin %.1f, %.1f), z in [%.2f, %.2f], inflation %.2f m, period %.2f s",
          plan_glob_sz_, plan_glob_res_, plan_glob_ox_, plan_glob_oy_,
          plan_glob_zmin_, plan_glob_zmax_, plan_glob_infl_, plan_glob_period_);
      }
    }

    // Per-source integration counters, reliable and latched. Own timer,
    // independent of publish_rate_hz: the publish timer's outputs are gated on
    // change and subscribers and would fall silent when a peer does.
    // (notes: dscovox-fusion-counters-timer)
    fusion_counters_pub_ =
      create_publisher<scovox_msgs::msg::ScovoxFusionCounters>(
        declare_parameter<std::string>("fusion_counters_topic",
                                       std::string("~/fusion_counters")),
        rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local());
    const double fc_hz =
      declare_parameter<double>("fusion_counters_hz", 2.0);
    if (fc_hz > 0.0) {
      fusion_counters_timer_ = rclcpp::create_timer(
        this, get_clock(),
        std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::duration<double>(1.0 / fc_hz)),
        [this] { publishFusionCounters(); });
    }

    // Reliable KeepLast for binary deltas: a dropped delta is permanent voxel
    // loss. An overflowing reliable reader drops silently, so under a comms
    // emulator set scovox_bin_qos_depth >= its rx_qos_depth.
    // (notes: dscovox-bin-sub-qos-depth)
    const int bin_depth = std::max(
        1, static_cast<int>(declare_parameter<int>("scovox_bin_qos_depth", 50)));
    auto bin_qos =
        rclcpp::QoS(rclcpp::KeepLast(static_cast<size_t>(bin_depth))).reliable();
    for (auto& t : input_topics_) {
      subs_.push_back(create_subscription<scovox_msgs::msg::ScovoxMapBinary>(
        t, bin_qos,
        std::bind(&DSCovoxNode::onBinaryMap, this, std::placeholders::_1)));
    }

    get_region_srv_ = create_service<scovox_msgs::srv::GetRegion>(
      "~/get_region",
      std::bind(&DSCovoxNode::onGetRegion, this, std::placeholders::_1, std::placeholders::_2));
    get_occ_srv_ = create_service<scovox_msgs::srv::GetOccupancyGrid>(
      "~/get_occupancy_grid",
      std::bind(&DSCovoxNode::onGetOccupancyGrid, this, std::placeholders::_1, std::placeholders::_2));

    if (pub_hz_ > 0.0) {
      // Fallback publish when no binaries arrive. One shared_lock spans the
      // tick so all publishers see the same fused state; publish* helpers must
      // NOT lock mu_ (non-recursive std::shared_mutex).
      // (notes: dscovox-publish-timer-lock)
      publish_timer_ = rclcpp::create_timer(
        this, get_clock(),
        std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::duration<double>(1.0 / pub_hz_)),
        [this] {
          std::shared_lock<std::shared_mutex> lk(mu_);
          maybePublishPointCloud();
          // Fused map for the planner; uses this tick's shared_lock (must not
          // re-lock). Gated on pub_hz_ > 0: disabling the publish timer also
          // disables this topic. (notes: dscovox-fused-map-gated-on-pub-hz)
          publishFusedMap();
          // World-fixed 2D projection for the exploration planner. Shares this
          // tick's shared_lock (must not re-lock the non-recursive mutex) and
          // rate-limits itself against plan_glob_period_, so publish_rate_hz
          // and the planning-map rate stay independent.
          publishGlobalPlanningMap();
          // Map size is keyed on occupancy (the fused Beta grid).
          size_t fc = split_fused_beta_ ? split_fused_beta_->activeCellsCount() : 0;
          size_t ts = 0;
          for (auto& [k, sg] : sources_)
            if (sg.beta_grid) ts += sg.beta_grid->activeCellsCount();
          RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000,
            "dscovox_diag: sources=%zu src_voxels=%zu fused_voxels=%zu",
            sources_.size(), ts, fc);
        });
    }

    RCLCPP_INFO(get_logger(),
      "DSCovoxNode started: %zu inputs, frame='%s'",
      input_topics_.size(), map_frame_.c_str());
  }

private:
  // Hash/equality for Bonxai::CoordT so we can dedupe touched coords in a set.
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

  // ==================================================================
  // The only receive path; every binary is a split Beta/Dir envelope. Beta and
  // Dir grids ingest and refold independently; priors are pinned from the first
  // frame. Query services project them to scovox::Voxel.
  // (notes: dscovox-receive-path)
  // ==================================================================
  void onBinaryMap(const scovox_msgs::msg::ScovoxMapBinary::SharedPtr msg) {
    if (msg->version != 5) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "wire receiver expects envelope version 5, got %d (dropping)",
        msg->version);
      return;
    }
    // Frames are host-byte-order memcpy with no byte-swap decode, so a sender
    // of the other endianness is rejected rather than mis-decoded.
    // (notes: dscovox-endianness-reject)
    constexpr bool kHostLittleEndian =
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        true;
#else
        false;
#endif
    if (msg->little_endian != kHostLittleEndian) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "wire endianness mismatch from '%s' (frame little_endian=%d, host=%d) — "
        "cross-endian decode is unsupported; dropping",
        msg->header.frame_id.c_str(), (int)msg->little_endian, (int)kHostLittleEndian);
      return;
    }
    const std::string sf = msg->header.frame_id;
    std::string buf = scovox::ScovoxBinarySerializer::decompressLZ4(msg->data);
    if (buf.empty()) {
      RCLCPP_ERROR(get_logger(), "LZ4 fail '%s'", sf.c_str());
      return;
    }

    scovox::BinarySerializer::Frame frame;
    try {
      frame = scovox::BinarySerializer::deserialize(buf);
    } catch (const std::exception& e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "deserialize failed for '%s': %s", sf.c_str(), e.what());
      return;
    }
    noteSeq(sf, msg->seq);
    if (frame.beta_deltas.empty() && frame.dir_deltas.empty()) return;

    // The producer carries its source->map pose in the message; the merger
    // never touches TF. First pose wins: later frames' poses are ignored
    // (static TF, c-slam off). (notes: dscovox-carried-pose-first-wins)
    const Eigen::Isometry3d Tmo = tfToIsometry(msg->map_from_source);

    float src_res = frame.resolution > 0.f ? frame.resolution : 0.f;

    // Captured inside the lock, logged after release (keeps the info log out of
    // the critical section). Report the integration of this source's update into
    // the fused map: how much arrived, how much it touched, and the fleet size.
    bool   new_source      = false;   // first frame ever seen from this source
    size_t n_beta_deltas   = frame.beta_deltas.size();
    size_t n_dir_deltas    = frame.dir_deltas.size();
    size_t n_touched_beta  = 0;       // fused occupancy cells this frame changed
    size_t n_touched_dir   = 0;       // fused semantic  cells this frame changed
    size_t n_sources       = 0;       // robots currently contributing to the fusion

    {
      std::unique_lock<std::shared_mutex> lk(mu_);
      // Reject num_classes==0 outright (invalid Dirichlet dimension): pinning it
      // would re-pin/re-log every frame and bypass the mismatch guard, letting a
      // second num_classes==0 source with a different alpha_0 fuse unchecked.
      if (frame.num_classes == 0) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
          "frame from '%s' has num_classes=0 (invalid) — dropping", sf.c_str());
        return;
      }
      // Pin the symmetric-Dirichlet prior on first valid frame, then assert
      // match (cross-prior fusion mis-subtracts in mergeBeta/mergeDir).
      // prior_pinned_ (not fused_num_classes_==0) is the "not yet pinned"
      // sentinel — see the prior_pinned_ member comment.
      if (!prior_pinned_) {
        prior_pinned_ = true;
        fused_num_classes_ = frame.num_classes;
        fused_alpha_0_     = frame.alpha_0;
        RCLCPP_INFO(get_logger(),
          "receive: pinned num_classes=%u alpha_0=%.4f (from first frame, src='%s')",
          (unsigned)fused_num_classes_, fused_alpha_0_, sf.c_str());
      } else {
        if (frame.num_classes != fused_num_classes_ ||
            std::abs(frame.alpha_0 - fused_alpha_0_) > 1e-6f) {
          RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
            "prior mismatch from '%s' (got C=%u α=%.4f, pinned C=%u α=%.4f) — dropping frame",
            sf.c_str(),
            (unsigned)frame.num_classes, frame.alpha_0,
            (unsigned)fused_num_classes_, fused_alpha_0_);
          return;
        }
      }

      if (res_ <= 0.f) res_ = src_res > 0.f ? src_res : 0.1f;
      if (src_res <= 0.f) src_res = res_;

      if (!split_fused_beta_) {
        scovox::Params P; P.resolution = res_;
        split_fused_beta_ = std::make_unique<Bonxai::VoxelGrid<scovox::BetaVoxel>>(
            P.resolution, P.inner_bits, P.leaf_bits);
        split_fused_dir_  = std::make_unique<Bonxai::VoxelGrid<scovox::DirVoxel>>(
            P.resolution, P.inner_bits, P.leaf_bits);
      }

      auto it = sources_.find(sf);
      if (it == sources_.end()) {
        SourceGrid sg;
        sg.source_frame = sf;
        sg.T_map_source = Tmo;
        sg.pose_cached = true;
        scovox::Params P; P.resolution = res_;
        sg.beta_grid = std::make_unique<Bonxai::VoxelGrid<scovox::BetaVoxel>>(
            P.resolution, P.inner_bits, P.leaf_bits);
        sg.dir_grid  = std::make_unique<Bonxai::VoxelGrid<scovox::DirVoxel>>(
            P.resolution, P.inner_bits, P.leaf_bits);
        it = sources_.emplace(sf, std::move(sg)).first;
        new_source = true;
      } else if (!it->second.pose_cached) {
        it->second.T_map_source = Tmo;
        it->second.pose_cached = true;
      }
      auto& src = it->second;
      if (!src.beta_grid) {
        scovox::Params P; P.resolution = res_;
        src.beta_grid = std::make_unique<Bonxai::VoxelGrid<scovox::BetaVoxel>>(
            P.resolution, P.inner_bits, P.leaf_bits);
        src.dir_grid  = std::make_unique<Bonxai::VoxelGrid<scovox::DirVoxel>>(
            P.resolution, P.inner_bits, P.leaf_bits);
      }

      // Step 1 — ingest both deltas into this source's MAP-FRAME grids.
      // Centre-sample posToCoord (floor() picks whichever map voxel contains
      // the bulk of the source voxel's volume even when the pose is unaligned).
      std::unordered_set<Bonxai::CoordT, CoordTHash, CoordTEqual> touched_beta, touched_dir;
      touched_beta.reserve(frame.beta_deltas.size());
      touched_dir.reserve(frame.dir_deltas.size());
      const Eigen::Isometry3d Te = src.T_map_source;
      const double half_src_res = 0.5 * double(src_res);
      auto toMapPos = [&](const Bonxai::CoordT& sc) {
        Eigen::Vector3d sp(
          double(sc.x) * double(src_res) + half_src_res,
          double(sc.y) * double(src_res) + half_src_res,
          double(sc.z) * double(src_res) + half_src_res);
        return Eigen::Vector3d(Te * sp);
      };
      // Shared-ROI z-band clip (see share_roi_z_min/max in the constructor).
      // Applied to Beta AND Dir identically, in the MAP frame (post-transform).
      const bool zband = share_z_max_ > share_z_min_;

      {
        auto ba = src.beta_grid->createAccessor();
        for (auto& d : frame.beta_deltas) {
          const Eigen::Vector3d mp = toMapPos(d.coord);
          if (zband && (mp.z() < share_z_min_ || mp.z() > share_z_max_)) continue;
          auto mc = src.beta_grid->posToCoord(mp.x(), mp.y(), mp.z());
          auto* v = ba.value(mc, true);
          if (!v) continue;
          *v = d.data;     // snapshot-replace
          touched_beta.insert(mc);
        }
      }
      {
        auto da = src.dir_grid->createAccessor();
        for (auto& d : frame.dir_deltas) {
          const Eigen::Vector3d mp = toMapPos(d.coord);
          if (zband && (mp.z() < share_z_min_ || mp.z() > share_z_max_)) continue;
          auto mc = src.dir_grid->posToCoord(mp.x(), mp.y(), mp.z());
          auto* v = da.value(mc, true);
          if (!v) continue;
          *v = d.data;     // snapshot-replace
          touched_dir.insert(mc);
        }
      }

      // Step 2 — refold each touched cell from all sources, per grid.
      {
        auto fa = split_fused_beta_->createAccessor();
        std::vector<Bonxai::VoxelGrid<scovox::BetaVoxel>::Accessor> source_accs;
        source_accs.reserve(sources_.size());
        for (auto& [k, sg] : sources_)
          if (sg.beta_grid) source_accs.emplace_back(sg.beta_grid->createAccessor());
        for (const auto& mc : touched_beta) refoldCellBeta(mc, fa, source_accs);
      }
      {
        auto fa = split_fused_dir_->createAccessor();
        // Fold Dir sources in sorted source-id order: mergeDir truncates to
        // top-K and OTHER mass cannot climb back, so fused labels depend on
        // fold order. Beta merge is commutative.
        // (notes: dscovox-dir-fold-order)
        std::vector<const std::string*> keys;
        keys.reserve(sources_.size());
        for (auto& [k, sg] : sources_)
          if (sg.dir_grid) keys.push_back(&k);
        std::sort(keys.begin(), keys.end(),
                  [](const std::string* a, const std::string* b) { return *a < *b; });
        std::vector<Bonxai::VoxelGrid<scovox::DirVoxel>::Accessor> source_accs;
        source_accs.reserve(keys.size());
        for (const std::string* k : keys)
          source_accs.emplace_back(sources_.at(*k).dir_grid->createAccessor());
        for (const auto& mc : touched_dir) refoldCellDir(mc, fa, source_accs);
      }

      // Snapshot the counts for the post-lock info log below.
      n_touched_beta = touched_beta.size();
      n_touched_dir  = touched_dir.size();
      n_sources      = sources_.size();

      // Counted after the reject paths, so rejected frames are not delivery.
      // deltas_received counts deltas as presented, before the z-band clip;
      // cells_touched is post-clip.
      // (notes: dscovox-counters-presented-not-stored)
      src.deltas_received += n_beta_deltas + n_dir_deltas;
      src.cells_touched   += n_touched_beta + n_touched_dir;
    }  // unique_lock released

    // Announce the integration of this source's map update into the fused map.
    // A brand-new robot joining the fusion is a distinct, notable event, so it
    // gets its own line before the per-update detail.
    if (new_source) {
      RCLCPP_INFO(get_logger(),
        "dscovox: new robot source '%s' joined the fused map (now %zu source%s)",
        sf.c_str(), n_sources, n_sources == 1 ? "" : "s");
    }
    RCLCPP_INFO(get_logger(),
      "dscovox: integrated update from '%s' — occupancy %zu deltas / %zu fused cells, "
      "semantics %zu deltas / %zu fused cells (%zu source%s fused)",
      sf.c_str(), n_beta_deltas, n_touched_beta, n_dir_deltas, n_touched_dir,
      n_sources, n_sources == 1 ? "" : "s");

    // The fused grid changed; mark it so the publish timer re-publishes the
    // fused-map topic on its next tick (and only then). Reaching here implies
    // non-empty deltas were fused (empty frames returned early above).
    fused_dirty_.store(true, std::memory_order_relaxed);

    // Step 3 — visualisation publish.
    {
      std::shared_lock<std::shared_mutex> rlk(mu_);
      maybePublishPointCloud();
    }
  }

  // Per-cell refold scratch: source-voxel pointer lists reused across cells so
  // the hot incremental refold stays allocation-free (the ingest path holds the
  // unique_lock, so these are touched single-threaded).
  std::vector<const scovox::BetaVoxel*> refold_beta_src_;
  std::vector<const scovox::DirVoxel*>  refold_dir_src_;

  // BetaVoxel-typed refold. Reset fused[mc] to the symmetric Beta(1,1) occupancy
  // prior, then fold every source's value via mergeBeta (conjugate Beta consensus).
  // The reset-then-refold core lives in scovox::refoldBeta (shared with tests).
  void refoldCellBeta(
      const Bonxai::CoordT& mc,
      Bonxai::VoxelGrid<scovox::BetaVoxel>::Accessor& fa,
      std::vector<Bonxai::VoxelGrid<scovox::BetaVoxel>::Accessor>& source_accs)
  {
    auto* fv = fa.value(mc, true);
    if (!fv) return;
    refold_beta_src_.clear();
    for (auto& sa : source_accs) refold_beta_src_.push_back(sa.value(mc, false));
    *fv = scovox::refoldBeta(refold_beta_src_, fused_num_classes_, fused_alpha_0_);
  }

  // DirVoxel-typed refold. Reset fused[mc] to the symmetric Dirichlet prior,
  // then fold every source's value via mergeDir (slot-reconciling consensus).
  // The reset-then-refold core lives in scovox::refoldDir (shared with tests).
  void refoldCellDir(
      const Bonxai::CoordT& mc,
      Bonxai::VoxelGrid<scovox::DirVoxel>::Accessor& fa,
      std::vector<Bonxai::VoxelGrid<scovox::DirVoxel>::Accessor>& source_accs)
  {
    auto* fv = fa.value(mc, true);
    if (!fv) return;
    refold_dir_src_.clear();
    for (auto& sa : source_accs) refold_dir_src_.push_back(sa.value(mc, false));
    *fv = scovox::refoldDir(refold_dir_src_, fused_num_classes_, fused_alpha_0_);
  }

  // Rate-limited pointcloud publish, called from the binary callback tail and
  // the timer fallback; free with no subscriber. Caller must hold mu_ (shared).
  // (notes: dscovox-maybe-publish-pointcloud)
  void maybePublishPointCloud() {
    // Runs under a shared lock, concurrently from the timer and a callback:
    // claim the window with one atomic compare_exchange so exactly one caller
    // publishes per interval. (notes: dscovox-pointcloud-cas-rate-limit)
    const int64_t now_ns = get_clock()->now().nanoseconds();
    const int64_t min_dt_ns =
        static_cast<int64_t>(pc_min_interval_s_ * 1e9);
    int64_t last_ns = last_pc_pub_ns_.load(std::memory_order_relaxed);
    if (now_ns - last_ns < min_dt_ns) return;
    // Only the thread that wins the CAS publishes; a loser (last_ns advanced
    // under us) bails to avoid a double publish in the same window.
    if (!last_pc_pub_ns_.compare_exchange_strong(
            last_ns, now_ns, std::memory_order_relaxed)) {
      return;
    }
    publishPointCloud();
  }

  // Split-substrate visualisation publisher. Walks the fused Beta grid
  // (occupancy) and joins the fused Dir grid (semantics) at each coord,
  // projecting to a transient SemBetaVoxel for the 11-field PointCloud2 schema.
  void publishPointCloud() {
    if (!pc_pub_ || !split_fused_beta_ ||
        pc_pub_->get_subscription_count() == 0) return;
    auto& g = *split_fused_beta_;
    auto dacc = split_fused_dir_->createConstAccessor();
    const float ot = (float)min_occ_;
    size_t cnt = 0;
    g.forEachCell([&](const scovox::BetaVoxel& v, const Bonxai::CoordT&) {
      // Skip prior-only cells via isPriorBeta, not a p_occ threshold, so the
      // gate holds for any prior. Mirrors the RPC walkers' gate.
      // (notes: dscovox-pointcloud-prior-gate)
      if (isPriorBeta(v, fused_num_classes_, fused_alpha_0_)) return;
      if (v.p_occ() >= ot) ++cnt;
    });
    if (!cnt) return;

    sensor_msgs::msg::PointCloud2 cl;
    cl.header.frame_id = map_frame_;
    cl.header.stamp = get_clock()->now();
    cl.height = 1;
    cl.is_dense = true;
    cl.is_bigendian = false;
    sensor_msgs::PointCloud2Modifier md(cl);
    md.setPointCloud2Fields(11,
      "x", 1, sensor_msgs::msg::PointField::FLOAT32,
      "y", 1, sensor_msgs::msg::PointField::FLOAT32,
      "z", 1, sensor_msgs::msg::PointField::FLOAT32,
      "rgb", 1, sensor_msgs::msg::PointField::FLOAT32,
      "occupancy_prob", 1, sensor_msgs::msg::PointField::FLOAT32,
      "semantic_class", 1, sensor_msgs::msg::PointField::UINT8,
      "semantic_confidence", 1, sensor_msgs::msg::PointField::FLOAT32,
      "posterior_variance", 1, sensor_msgs::msg::PointField::FLOAT32,
      "eig", 1, sensor_msgs::msg::PointField::FLOAT32,
      "a_occ", 1, sensor_msgs::msg::PointField::FLOAT32,
      "a_free", 1, sensor_msgs::msg::PointField::FLOAT32);
    md.resize(cnt);
    sensor_msgs::PointCloud2Iterator<float>
      ix(cl, "x"), iy(cl, "y"), iz(cl, "z"), ir(cl, "rgb"),
      ip(cl, "occupancy_prob"), ic(cl, "semantic_confidence"),
      iv(cl, "posterior_variance"), ie(cl, "eig"),
      iao(cl, "a_occ"), iaf(cl, "a_free");
    sensor_msgs::PointCloud2Iterator<uint8_t> ik(cl, "semantic_class");
    g.forEachCell([&](const scovox::BetaVoxel& vb, const Bonxai::CoordT& co) {
      // Same prior gate as the counting pass above — keep the two passes in
      // lock-step so the emitted point count matches md.resize(cnt).
      if (isPriorBeta(vb, fused_num_classes_, fused_alpha_0_)) return;
      float pr = vb.p_occ();
      if (pr < ot) return;
      const scovox::DirVoxel* dv = dacc.value(co);
      const scovox::SemBetaVoxel v =
          projectBetaDirToSemBetaForViz(vb, dv, fused_alpha_0_, fused_num_classes_);
      auto p = g.coordToPos(co);
      *ix = p.x; *iy = p.y; *iz = p.z; *ip = pr;
      const auto [best_cls, cf] = scovox::argmaxClassConfidence(v);
      // semantic_class is UINT8 (RViz and pointcloud_to_npz.py read one byte);
      // class ids >= 256 emit 0 (unknown) rather than alias. Mirror this limit
      // in scovox_node.cpp's pointcloud publishers.
      // (notes: dscovox-semantic-class-uint8)
      const uint8_t bc = (best_cls < 256) ? static_cast<uint8_t>(best_cls) : 0;
      *ik = bc; *ic = cf;
      float r = 1, gg = 1, b = 1;
      if (v.a0() > 0 && cf >= sem_gate_ && bc < sem_col_.size()) {
        r = sem_col_[bc][0]; gg = sem_col_[bc][1]; b = sem_col_[bc][2];
      }
      uint32_t rp = ((uint32_t)(r * 255) << 16) | ((uint32_t)(gg * 255) << 8) | (uint32_t)(b * 255);
      // Bit-pack RGB into the float field via memcpy (not a reinterpret_cast,
      // which is a strict-aliasing UB the compiler warns on); folds to a move.
      float rgb_f; std::memcpy(&rgb_f, &rp, sizeof(rgb_f)); *ir = rgb_f;
      *iv = scovox::variance(v);
      *ie = scovox::expectedInformationGain(v);
      *iao = v.a_occ; *iaf = v.a_free;
      ++ix; ++iy; ++iz; ++ir; ++ip; ++ik; ++ic; ++iv; ++ie; ++iao; ++iaf;
    });
    pc_pub_->publish(cl);
  }

  // GetRegion core. Walks the Bonxai grid whose cell type projects to a
  // scovox::Voxel via `project(cell, coord) -> scovox::Voxel` (wire: join the Dir
  // grid at `coord` + prior-subtract). Bbox in coord space, top-K selection,
  // and the mass-conserving a_unk fold produce the byte-exact response.
  template <typename CellT, typename IsPriorFn, typename ProjectFn>
  void regionOnGrid(const scovox_msgs::srv::GetRegion::Request::SharedPtr rq,
                    scovox_msgs::srv::GetRegion::Response::SharedPtr rs,
                    Bonxai::VoxelGrid<CellT>& g,
                    IsPriorFn isPriorCell, ProjectFn project)
  {
    auto& m = rs->map;
    auto mn = g.posToCoord(rq->min_corner.x, rq->min_corner.y, rq->min_corner.z);
    auto mx = g.posToCoord(rq->max_corner.x, rq->max_corner.y, rq->max_corner.z);
    g.forEachCell([&](const CellT& cell, const Bonxai::CoordT& c) {
      if (isPriorCell(cell)) return;
      if (c.x < mn.x || c.x > mx.x || c.y < mn.y || c.y > mx.y || c.z < mn.z || c.z > mx.z) return;
      const scovox::Voxel v = project(cell, c);
      auto p = g.coordToPos(c);
      scovox_msgs::msg::ScovoxVoxel dv;
      dv.position.x = p.x; dv.position.y = p.y; dv.position.z = p.z;
      dv.a_occ = std::max(0.f, v.a_occ);
      dv.a_free = std::max(0.f, v.a_free);
      dv.a_unk = std::max(0.f, v.a_unk);
      // Pick the top top_k_ strongest classes (sparse_add doesn't sort) and
      // fold dropped mass into a_unk so total semantic evidence is preserved.
      const auto top = scovox::selectTopKSemantics(v, top_k_);
      for (size_t i = 0; i < top.kept_count; ++i) {
        scovox_msgs::msg::ScovoxSemanticEvidence se;
        se.class_id = top.kept[i].first;
        se.evidence_count = top.kept[i].second;
        dv.semantic_evidence.push_back(se);
      }
      dv.a_unk += top.dropped_mass;
      m.voxels.push_back(std::move(dv));
    });
  }

  // Lock-free GetRegion core shared by the service and the fused-map publisher.
  // Caller MUST hold at least a shared lock on mu_; never lock it here
  // (non-recursive). (notes: dscovox-fill-region-lock-free)
  void fillRegion(const scovox_msgs::srv::GetRegion::Request::SharedPtr& rq,
                  const scovox_msgs::srv::GetRegion::Response::SharedPtr& rs)
  {
    auto& m = rs->map;
    m.header.stamp = get_clock()->now();
    m.header.frame_id = map_frame_;
    m.resolution = res_;
    m.occupancy_threshold = (float)min_occ_;
    m.semantic_threshold = (float)sem_gate_;
    m.max_semantic_classes = (uint8_t)top_k_;
    if (!split_fused_beta_) return;
    const uint16_t C = fused_num_classes_;
    const float a0 = fused_alpha_0_;
    auto dacc = split_fused_dir_->createConstAccessor();
    regionOnGrid<scovox::BetaVoxel>(
        rq, rs, *split_fused_beta_,
        [this](const scovox::BetaVoxel& b) { return isPriorBeta(b, fused_num_classes_, fused_alpha_0_); },
        [dacc, C, a0](const scovox::BetaVoxel& b, const Bonxai::CoordT& c) mutable {
          return projectBetaDirToVoxel(b, dacc.value(c), C, a0);
        });
  }

  void onGetRegion(const scovox_msgs::srv::GetRegion::Request::SharedPtr rq,
                   scovox_msgs::srv::GetRegion::Response::SharedPtr rs)
  {
    // split_fused_* is kept current incrementally — read lock.
    std::shared_lock<std::shared_mutex> lk(mu_);
    fillRegion(rq, rs);
  }

  // Publish the whole fused map as ScovoxMap; the caller holds the shared lock,
  // so do NOT lock mu_. Subscribers are checked before fused_dirty_ is
  // consumed, so a change waits until someone listens.
  // (notes: dscovox-publish-fused-map)
  void publishFusedMap()
  {
    if (!scovox_map_pub_ || scovox_map_pub_->get_subscription_count() == 0)
      return;
    if (!fused_dirty_.exchange(false)) return;
    // Full-coverage bbox of +-1e8 voxels, scaled by res_ so the coord cast
    // stays inside int32 at any resolution. Consumers apply their own ROI clip.
    // (notes: dscovox-full-coverage-bbox)
    const double big = 1e8 * std::max(static_cast<double>(res_), 1e-3);
    auto rq = std::make_shared<scovox_msgs::srv::GetRegion::Request>();
    rq->min_corner.x = rq->min_corner.y = rq->min_corner.z = -big;
    rq->max_corner.x = rq->max_corner.y = rq->max_corner.z =  big;
    auto rs = std::make_shared<scovox_msgs::srv::GetRegion::Response>();
    fillRegion(rq, rs);
    scovox_map_pub_->publish(rs->map);
  }

  // Every decoded frame advances its sender's sequence, whether or not it
  // carries anything dscovox fuses. seq 0 is an unstamped sender: not tracked.
  // A value below the stored one is a sender restart and re-baselines.
  void noteSeq(const std::string& sf, uint64_t seq)
  {
    if (seq == 0) return;
    std::lock_guard<std::mutex> lk(seq_mu_);
    SeqTrack& t = seq_by_source_[sf];
    if (seq > t.newest + 1) t.gaps += seq - t.newest - 1;
    t.newest = seq;
  }

  // Own timer and own shared lock, no dirty or subscriber gate, so a quiet peer
  // is distinguishable from a quiet dscovox. Sorted by source frame; entries
  // are never removed. (notes: dscovox-fusion-counters-no-gates)
  void publishFusionCounters()
  {
    if (!fusion_counters_pub_) return;
    scovox_msgs::msg::ScovoxFusionCounters msg;
    msg.header.stamp    = now();
    msg.header.frame_id = map_frame_;
    {
      // A source heard only through frames that never reach ingest has a
      // sequence entry and no grid; it is listed with zero counters.
      std::shared_lock<std::shared_mutex> lk(mu_);
      std::lock_guard<std::mutex> slk(seq_mu_);
      msg.source_frame.reserve(sources_.size());
      for (const auto& [k, sg] : sources_) msg.source_frame.push_back(k);
      for (const auto& [k, t] : seq_by_source_)
        if (!sources_.count(k)) msg.source_frame.push_back(k);
      std::sort(msg.source_frame.begin(), msg.source_frame.end());
      const size_t n = msg.source_frame.size();
      msg.deltas_received.reserve(n);
      msg.cells_touched.reserve(n);
      msg.newest_seq.reserve(n);
      msg.seq_gaps.reserve(n);
      for (const auto& k : msg.source_frame) {
        const auto sit = sources_.find(k);
        msg.deltas_received.push_back(sit == sources_.end() ? 0 : sit->second.deltas_received);
        msg.cells_touched.push_back(sit == sources_.end() ? 0 : sit->second.cells_touched);
        const auto qit = seq_by_source_.find(k);
        msg.newest_seq.push_back(qit == seq_by_source_.end() ? 0 : qit->second.newest);
        msg.seq_gaps.push_back(qit == seq_by_source_.end() ? 0 : qit->second.gaps);
      }
    }
    fusion_counters_pub_->publish(msg);
  }

  // World-fixed 2D projection of the fused grid; caller holds mu_ (shared). -1
  // unknown (unobserved or prior-only, never 0 or 100), 0 free (p_occ below
  // threshold), 100 occupied (wins ties).
  // (notes: dscovox-global-map-three-state)
  void publishGlobalPlanningMap() {
    if (!pl_glob_pub_ || !split_fused_beta_) return;
    // Nothing subscribed ⇒ nothing to pay for. transient_local still replays
    // the last sample to a late joiner, and the first subscriber pulls a fresh
    // one on the next tick.
    if (pl_glob_pub_->get_subscription_count() == 0) return;
    const auto now = std::chrono::steady_clock::now();
    if (plan_glob_period_ > 0.0 &&
        plan_glob_last_.time_since_epoch().count() != 0 &&
        std::chrono::duration<double>(now - plan_glob_last_).count() <
            plan_glob_period_)
      return;
    plan_glob_last_ = now;

    const double res = (plan_glob_res_ > 0.0) ? plan_glob_res_
                                              : std::max((double)res_, 1e-3);
    const int w = std::max(1, (int)std::round(plan_glob_sz_ / res));
    const int h = w;
    nav_msgs::msg::OccupancyGrid g;
    g.header.stamp = get_clock()->now();
    // map_frame_, not an integration frame: the planner indexes this grid with
    // raw world XY and applies NO transform. scovox_node's equivalent stamps
    // <robot>/odom and is only correct while map->odom happens to be identity;
    // the merger already works in the map frame, so that hazard is absent here.
    g.header.frame_id = map_frame_;
    g.info.resolution = (float)res;
    g.info.width = (uint32_t)w;
    g.info.height = (uint32_t)h;
    g.info.origin.position.x = plan_glob_ox_;
    g.info.origin.position.y = plan_glob_oy_;
    g.info.origin.orientation.w = 1.0;
    g.data.assign((size_t)w * (size_t)h, -1);

    auto& bg = *split_fused_beta_;
    const float ot = (float)min_occ_;
    bg.forEachCell([&](const scovox::BetaVoxel& v, const Bonxai::CoordT& c) {
      if (isPriorBeta(v, fused_num_classes_, fused_alpha_0_)) return;
      const auto p = bg.coordToPos(c);
      if (p.z < plan_glob_zmin_ || p.z > plan_glob_zmax_) return;
      const int gx = (int)std::floor((p.x - plan_glob_ox_) / res);
      const int gy = (int)std::floor((p.y - plan_glob_oy_) / res);
      if (gx < 0 || gy < 0 || gx >= w || gy >= h) return;
      const size_t i = (size_t)gy * (size_t)w + (size_t)gx;
      if (v.p_occ() >= ot) g.data[i] = 100;
      else if (g.data[i] != 100) g.data[i] = 0;
    });

    // Dilate occupied cells by the body radius: the planner does a single-cell
    // free check and relies on the map already carrying the clearance.
    const int ic = (int)std::ceil(plan_glob_infl_ / res);
    if (ic > 0) {
      const int ic2 = ic * ic;
      auto inf = g.data;
      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
          if (g.data[(size_t)y * (size_t)w + (size_t)x] != 100) continue;
          for (int dy = -ic; dy <= ic; ++dy) {
            for (int dx = -ic; dx <= ic; ++dx) {
              if (dx * dx + dy * dy > ic2) continue;
              const int nx = x + dx, ny = y + dy;
              if (nx >= 0 && nx < w && ny >= 0 && ny < h)
                inf[(size_t)ny * (size_t)w + (size_t)nx] = 100;
            }
          }
        }
      }
      g.data = std::move(inf);
    }
    pl_glob_pub_->publish(g);
  }

  // Substrate-agnostic GetOccupancyGrid core. 2D max-projection of p_occ over
  // [z_min, z_max]. Occupancy-only ⇒ no projection to scovox::Voxel needed:
  // both scovox::Voxel and BetaVoxel expose p_occ(); only `isPriorCell` and the
  // grid type vary. Output bytes are identical across substrates.
  template <typename CellT, typename IsPriorFn>
  void occupancyGridOnGrid(const scovox_msgs::srv::GetOccupancyGrid::Request::SharedPtr rq,
                           scovox_msgs::srv::GetOccupancyGrid::Response::SharedPtr rs,
                           Bonxai::VoxelGrid<CellT>& g, IsPriorFn isPriorCell)
  {
    double r2 = (rq->resolution_2d > 0.0) ? rq->resolution_2d : (double)res_;
    float ot = (float)min_occ_;
    double xn = std::numeric_limits<double>::max(), xx = -xn, yn = xn, yx = -yn;
    std::unordered_map<int64_t, float> cells;
    g.forEachCell([&](const CellT& v, const Bonxai::CoordT& c) {
      if (isPriorCell(v)) return;
      auto p = g.coordToPos(c);
      if (p.z < rq->z_min || p.z > rq->z_max) return;
      float pr = v.p_occ();
      xn = std::min(xn, (double)p.x); xx = std::max(xx, (double)p.x);
      yn = std::min(yn, (double)p.y); yx = std::max(yx, (double)p.y);
      int64_t k = ((int64_t)(int32_t)std::floor(p.x / r2) << 32) |
                  ((int64_t)(uint32_t)(int32_t)std::floor(p.y / r2));
      auto& mp = cells[k];
      mp = std::max(mp, pr);
    });
    if (cells.empty()) { rs->grid = nav_msgs::msg::OccupancyGrid(); return; }
    int32_t ox = (int32_t)std::floor(xn / r2), oy = (int32_t)std::floor(yn / r2);
    uint32_t w = (uint32_t)((int32_t)std::floor(xx / r2) - ox + 1);
    uint32_t h = (uint32_t)((int32_t)std::floor(yx / r2) - oy + 1);
    auto& og = rs->grid;
    og.header.stamp = get_clock()->now();
    og.header.frame_id = map_frame_;
    og.info.resolution = (float)r2;
    og.info.width = w;
    og.info.height = h;
    og.info.origin.position.x = ox * r2;
    og.info.origin.position.y = oy * r2;
    og.info.origin.position.z = rq->z_min;
    og.info.origin.orientation.w = 1.0;
    og.data.assign(w * h, -1);
    for (auto& [k, mp] : cells) {
      uint32_t c = (uint32_t)((int32_t)(k >> 32) - ox);
      uint32_t r = (uint32_t)((int32_t)(k & 0xFFFFFFFF) - oy);
      if (c < w && r < h) og.data[r * w + c] = (mp >= ot) ? (int8_t)std::min(100.f, mp * 100.f) : 0;
    }
  }

  void onGetOccupancyGrid(const scovox_msgs::srv::GetOccupancyGrid::Request::SharedPtr rq,
                          scovox_msgs::srv::GetOccupancyGrid::Response::SharedPtr rs)
  {
    // split_fused_beta_ is kept current incrementally — read lock.
    std::shared_lock<std::shared_mutex> lk(mu_);
    if (!split_fused_beta_) { rs->grid = nav_msgs::msg::OccupancyGrid(); return; }
    occupancyGridOnGrid<scovox::BetaVoxel>(
        rq, rs, *split_fused_beta_,
        [this](const scovox::BetaVoxel& b) { return isPriorBeta(b, fused_num_classes_, fused_alpha_0_); });
  }

  void initSemanticColors() { sem_col_ = scovox::generateSemanticColors(256); }

  // Members
  std::vector<std::string> input_topics_;
  std::string map_frame_;
  double min_occ_, sem_gate_;
  double pub_hz_;
  // Shared-ROI z-band ingest clip (map frame; min >= max = off). KEEP IN SYNC
  // with scovox_node share_roi_z_min/max + explo_planner roi_min_z/roi_max_z.
  double share_z_min_{0.0}, share_z_max_{0.0};
  double pc_min_interval_s_;
  // World-fixed planning map for the exploration planner (see the constructor
  // block and publishGlobalPlanningMap). plan_glob_last_ is touched only from
  // the publish timer.
  bool   pub_plan_glob_{false};
  double plan_glob_res_{0.40}, plan_glob_sz_{200.0}, plan_glob_ox_{-100.0},
         plan_glob_oy_{-100.0}, plan_glob_infl_{1.5},
         plan_glob_zmin_{0.05}, plan_glob_zmax_{1.0}, plan_glob_period_{1.0};
  std::chrono::steady_clock::time_point plan_glob_last_{};
  // Pointcloud rate-limit timestamp in ns. Atomic because
  // maybePublishPointCloud runs under only a shared lock from two paths;
  // compare_exchange lets exactly one caller publish per window.
  // (notes: dscovox-last-pc-pub-atomic)
  std::atomic<int64_t> last_pc_pub_ns_{0};
  // Set when onBinaryMap fuses new data; consumed (exchanged false) by
  // publishFusedMap so the fused-map topic is only rebuilt+serialized when it
  // actually changed, not every publish tick. Mirrors scovox_node's sm_dirty_.
  std::atomic<bool> fused_dirty_{false};
  int top_k_;
  float res_{0.f};
  std::vector<std::array<float, 3>> sem_col_;
  // One source grid per robot, keyed by header.frame_id of incoming binaries.
  std::unordered_map<std::string, SourceGrid> sources_;
  // Per-source sequence tracking; its own lock, taken after mu_ when both are
  // held (publishFusionCounters), alone in noteSeq.
  std::unordered_map<std::string, SeqTrack> seq_by_source_;
  std::mutex seq_mu_;
  // Split Beta/Dirichlet fused grids. Allocated lazily on the first wire
  // frame; null otherwise. Occupancy ∥ semantics, merged independently
  // (consensus_merge.hpp). Share the pinned (fused_num_classes_, fused_alpha_0_).
  std::unique_ptr<Bonxai::VoxelGrid<scovox::BetaVoxel>> split_fused_beta_;
  std::unique_ptr<Bonxai::VoxelGrid<scovox::DirVoxel>>  split_fused_dir_;
  uint16_t fused_num_classes_{0};
  float    fused_alpha_0_{scovox::kDefaultDirichletPrior};
  // Set when the prior is pinned from the first valid frame. Do not use
  // fused_num_classes_==0 as the unpinned sentinel; num_classes==0 frames are
  // rejected explicitly. (notes: dscovox-prior-pinned-flag)
  bool     prior_pinned_{false};
  mutable std::shared_mutex mu_;
  std::vector<rclcpp::Subscription<scovox_msgs::msg::ScovoxMapBinary>::SharedPtr> subs_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pc_pub_;
  rclcpp::Publisher<scovox_msgs::msg::ScovoxMap>::SharedPtr scovox_map_pub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr pl_glob_pub_;
  rclcpp::Publisher<scovox_msgs::msg::ScovoxFusionCounters>::SharedPtr
      fusion_counters_pub_;
  rclcpp::Service<scovox_msgs::srv::GetRegion>::SharedPtr get_region_srv_;
  rclcpp::Service<scovox_msgs::srv::GetOccupancyGrid>::SharedPtr get_occ_srv_;
  rclcpp::TimerBase::SharedPtr publish_timer_;
  rclcpp::TimerBase::SharedPtr fusion_counters_timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DSCovoxNode>());
  rclcpp::shutdown();
}
