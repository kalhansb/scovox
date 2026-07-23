// dscovox_node.cpp — multi-robot SCovox map merger.
//
// Each connected scovox_node ships ScovoxMapBinary deltas of its persistent
// LOCAL scovox grid (i.e. that robot's own sensor observations only — NEVER
// the fused dscovox grid). This one-way local→fused topology is what keeps
// the additive Beta–Dirichlet consensus Bayesian: there is no comms-level
// echo where one robot's evidence is shipped back into another robot's
// source grid via the merged map, so conditional independence of the two
// sources given the latent voxel state holds at the protocol level. The
// dscovox `~/scovox` publish is for downstream consumers (planner,
// visualisation) only — wiring it back as an input to another dscovox would
// re-introduce evidence-echo and break the additive rule.
//
// K_TOP semantic-slot truncation lives in scovox (voxel.hpp / sparse_add)
// and is already applied by the time a binary reaches this node — the wire
// format carries at most K_TOP slots per voxel. dscovox cannot widen that;
// any K_TOP-related ablation belongs at the scovox layer (B1).
//
// The merger keeps one source grid per robot keyed by the binary's
// header.frame_id (e.g. "atlas/odom"). Source grids are stored directly in
// MAP-FRAME coordinates: at receive time we transform each delta voxel once
// using the source->map pose the producer captured and CARRIED in the message
// (ScovoxMapBinary.map_from_source), cached the first time we see that source.
// This node holds NO TF listener — the pose rides with the data.
//
// !! REQUIRES c-slam DISABLED !!
// The cached source->map pose is never refreshed (first carried pose wins).
// This is correct only while TFs are static. Re-enabling c-slam (loop closures
// / pose-graph optimization) is a CORRECTNESS BUG: the first pose jump leaves
// every voxel in the source grid at its old map-frame coord, producing ghost
// voxels at pre-loop-closure positions and missing voxels at the new positions.
// Before turning c-slam back on, refactor SourceGrid to store evidence in
// source-frame coords + reproject on each carried-pose change, with a
// per-source "pose changed → reproject" handler. See ablation entry C5 in
// docs/issues/ablations_punch_list.md for the design.
//
// On every binary we incrementally update the fused grid by, for each touched
// map-frame coord, resetting fused[c] to the prior and re-folding the current
// state of every source's grid at c. The fold uses the same Beta-conjugate
// consensus as before: a_fused = a_1 + a_2 - 1. Cells the binary did not
// touch are not visited — their existing fused value is still correct because
// no source's contribution at those cells changed.
//
// The reset-then-refold pattern is what keeps this bit-for-bit equivalent to
// a from-scratch rebuild while making the work proportional to the delta size
// instead of the total map size. Critically, it cannot double-count: a
// source's previous contribution at a cell is wiped before that source's
// current contribution is folded back in.
//
// No submaps. No pose graph. No loop closures. No periodic rebuild.

#include <rclcpp/rclcpp.hpp>
#include <scovox_msgs/msg/scovox_map_binary.hpp>
#include <scovox_msgs/msg/scovox_map.hpp>
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
#include <atomic>  // last_pc_pub_ns_ rate-limiter (shared-lock concurrent access)

namespace {

struct SourceGrid {
  std::string source_frame;                // header.frame_id of the binary
  // This source's contribution to the world, stored in MAP-FRAME coords.
  // Each entry is the latest snapshot of (occupancy, semantics) received for
  // that map-frame voxel from this robot.
  //
  // Split Beta/Dirichlet (wire format) receiver populates these two grids:
  // occupancy (BetaVoxel) ∥ semantics (DirVoxel). No TsdfMap on the receiver
  // because share_tsdf=false is the wire default — TSDF state never crosses the
  // wire to dscovox in the production path.
  std::unique_ptr<Bonxai::VoxelGrid<scovox::BetaVoxel>> beta_grid;
  std::unique_ptr<Bonxai::VoxelGrid<scovox::DirVoxel>>  dir_grid;
  // Cached static source->map transform. Taken from the first update's carried
  // map_from_source pose and never refreshed — this assumes TFs are static
  // (c-slam disabled). Under c-slam, loop closures change this transform and the
  // cache becomes a correctness bug. See the file-header banner and C5 in
  // ablations_punch_list.md.
  Eigen::Isometry3d T_map_source{Eigen::Isometry3d::Identity()};
  bool pose_cached{false};
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

// The pure receiver-side consensus helpers — kPriorSlop, isPriorBeta/isPriorDir,
// projectBetaDirToSemBetaForViz, projectBetaDirToVoxel, and the refoldBeta/
// refoldDir cores — now live in scovox/dscovox_consensus.hpp (namespace scovox)
// so the unit tests can exercise the SAME code this node runs (findings
// #18/#19/#20). The unqualified call sites below resolve to them via ADL (every
// call passes a scovox-typed voxel argument).
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

    // Shared-ROI z-band — receive-side defensive mirror of the sender's wire
    // filter. Records whose MAP-frame voxel centre falls outside
    // [share_roi_z_min, share_roi_z_max] are dropped at ingest, so the fused
    // map honours the band even if one sender was launched without it.
    // KEEP IN SYNC with scovox_node share_roi_z_min/share_roi_z_max (sender
    // wire filter, applied in its integration frame) and with explo_planner
    // exploration_params.yaml roi_min_z/roi_max_z: the shared band must be a
    // SUPERSET of the planner band. min >= max (default 0/0) disables.
    share_z_min_ = declare_parameter<double>("share_roi_z_min", 0.0);
    share_z_max_ = declare_parameter<double>("share_roi_z_max", 0.0);

    initSemanticColors();

    // Explicit reliable + depth 1, mirroring scovox_node's pc_pub_: the fused
    // cloud is tens of MB (one point per fused voxel) and SystemDefaultsQoS
    // resolved to BEST_EFFORT on this build — every fragmented sample dropped
    // and RViz never received a single cloud. Depth 1 caps publisher-side
    // buffering (see scovox_node's OOM note).
    pc_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      declare_parameter<std::string>("pointcloud_topic", "~/pointcloud"),
      rclcpp::QoS(rclcpp::KeepLast(1)).reliable());

    // Fused-map topic for downstream consumers (planner, visualisation). Topic
    // form of the on-demand GetRegion service: a full snapshot of the fused
    // Beta/Dir grids projected to a ScovoxMap, published from the publish timer
    // below when the map has changed and someone is subscribed (see
    // publishFusedMap). Latched QoS — KeepLast(1) + reliable + transient_local —
    // so the last published snapshot is retained and replayed to a (re)connecting
    // subscriber; the very first subscriber receives it on the next publish tick
    // after connecting. Only one full snapshot is ever retained (the voxel dump
    // can be large). Any subscriber MUST match this QoS or it receives nothing.
    scovox_map_pub_ = create_publisher<scovox_msgs::msg::ScovoxMap>(
      declare_parameter<std::string>("scovox_topic", "~/scovox"),
      rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local());

    // Reliable + deeper queue for binary submap deltas. The old
    // SystemDefaultsQoS resolved to BEST_EFFORT on this build, which
    // silently dropped large ScovoxMapBinary payloads under any
    // backpressure. Combined with scovox_node's fire-and-forget
    // dirty_.clear() after publish, drops became permanent voxel loss.
    // KeepLast(50) absorbs publish bursts when this node is busy
    // rebuilding the fused grid; reliable forces redelivery of any
    // packet the transport drops.
    auto bin_qos = rclcpp::QoS(rclcpp::KeepLast(50)).reliable();
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
      // The fused grid is kept current incrementally inside onBinaryMap.
      // The visualization pointcloud is normally driven from there too;
      // this timer is the fallback (when no binaries arrive).
      //
      // One outer shared_lock spans every publisher in the tick so they all
      // see the same fused state — without it, an onBinaryMap callback can
      // mutate the fused grids between any two of them. publish* helpers
      // must NOT take mu_ themselves — std::shared_mutex is non-recursive
      // so re-locking here would be UB.
      publish_timer_ = rclcpp::create_timer(
        this, get_clock(),
        std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::duration<double>(1.0 / pub_hz_)),
        [this] {
          std::shared_lock<std::shared_mutex> lk(mu_);
          maybePublishPointCloud();
          // Topic form of GetRegion: publish the whole fused map for the
          // planner. Shares this tick's shared_lock (publishFusedMap must not
          // re-lock the non-recursive shared_mutex). NOTE: gated on pub_hz_ > 0
          // like every other publisher here — disabling the publish timer also
          // disables the fused-map topic the planner depends on.
          publishFusedMap();
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
  // ScovoxMapBinary receive path — the node's only receive path. Every binary
  // is a split Beta/Dir envelope. Operates on the de-unified BetaVoxel
  // (occupancy) ∥ DirVoxel (semantics) grids + consensus_merge.hpp. The two
  // grids ingest + refold INDEPENDENTLY: a touched coord may be in either or
  // both. Priors are pinned from the first frame's header.
  //
  // NOTE: the RPC query services (GetRegion / GetOccupancyGrid) project the
  // split Beta(+Dir) grids into a transient scovox::Voxel via a
  // substrate-agnostic templated core. The SEMANTIC query math uses the
  // raw-evidence convention; the OCCUPANCY math uses the symmetric Beta(1,1)
  // prior (p_occ=0.5) — see projectBetaDirToVoxel / docs/occupancy_prior.md.
  // Occupancy-only services (GetOccupancyGrid) read just the Beta grid;
  // GetRegion joins the Dir grid for per-class evidence.
  // ==================================================================
  void onBinaryMap(const scovox_msgs::msg::ScovoxMapBinary::SharedPtr msg) {
    if (msg->version != 5) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "wire receiver expects envelope version 5, got %d (dropping)",
        msg->version);
      return;
    }
    // The frame body is serialized in host byte order (BinarySerializer uses raw
    // field memcpy, not an endian-canonical encoding). The publisher stamps
    // msg->little_endian from its host; until a byte-swapping decode path exists,
    // a sender of the opposite endianness can only be mis-decoded. Reject it loud
    // rather than silently corrupting the fused map. (No-op on a homogeneous
    // little-endian fleet, which is every supported target today.)
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
    if (frame.beta_deltas.empty() && frame.dir_deltas.empty()) return;

    // The producer captured its source->map pose at publish time and carried it
    // in the message, so the merger never touches TF. First-pose-wins: the first
    // frame from a source pins T_map_source (below); later frames' carried poses
    // are ignored. The static-TF / c-slam-off assumption is unchanged — only its
    // mechanism moved from a TF lookup to the message's map_from_source.
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
        // Fold the Dir sources in a deterministic (sorted-by-source-id) order.
        // mergeDir truncates to top-K and a class dumped to OTHER cannot climb
        // back, so the fused slots — and hence dominantClass / mesh labels —
        // depend on fold order. Iterating sources_ (an unordered_map) directly
        // would let them flip across runs and rehashes; sort the keys first so
        // the refold is reproducible. (Beta merge is additive/commutative and
        // needs no ordering.)
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

  // Rate-limited visualization publish. Called from the binary callback's
  // tail (so the user-visible map updates as soon as ingest produces fresh
  // data) and from the timer as a fallback (so the map keeps refreshing in
  // RViz even if no binaries arrive). publishPointCloud already returns
  // early if no subscriber, so this is free when nobody is watching.
  //
  // Caller must hold mu_ (shared). The lock is hoisted to the call sites so
  // every publisher in a single timer tick sees the same fused state.
  void maybePublishPointCloud() {
    // Race-free rate limit: this runs under only a SHARED lock and can execute
    // concurrently from the timer and a binary callback. Claim the window with a
    // single atomic compare_exchange so exactly one caller proceeds per interval
    // (a plain read-then-write of a non-atomic timestamp would be a data race
    // and could let both publish in the same window). See last_pc_pub_ns_.
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
      // Skip prior-only cells via isPriorBeta, independent of the prior's p_occ.
      // (Gating on isPriorBeta rather than a p_occ threshold keeps this correct
      // for any prior: the old calibrated prior p_occ ≈ 0.933 exceeded the 0.7
      // threshold and would publish as phantom occupied; the prior is now
      // Beta(1,1)/0.5. See docs/occupancy_prior.md.) Mirror the RPC walkers' gate.
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
      // The semantic_class PointField is UINT8 (wire/schema locked — RViz and
      // pointcloud_to_npz.py read it as one byte), but argmaxClassConfidence
      // returns a uint16_t class id. A naive static_cast<uint8_t> of an id >=256
      // (e.g. a 360-class taxonomy) would silently alias to id%256 and collide
      // with an unrelated class for BOTH the label and the palette colour. Emit
      // 0 (unknown) instead so the >255 case is unambiguous rather than wrong.
      // Mirror this same UINT8 limit in scovox_node.cpp's pointcloud publishers.
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

  // Lock-free GetRegion core, shared by the service handler and the fused-map
  // topic publisher. Caller MUST already hold (at least) a shared lock on mu_:
  // std::shared_mutex is non-recursive, so this function must never lock it
  // itself. Fills rs->map header + the voxels inside rq's bbox (regionOnGrid
  // does the clip) by projecting the fused split Beta/Dir grids.
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

  // Publish the ENTIRE fused map as a ScovoxMap topic. Called from the publish
  // timer, which already holds a shared lock on mu_ — this must NOT lock mu_
  // (non-recursive shared_mutex). Skipped when no one is subscribed (so a large
  // voxel dump isn't built for nothing) and when the map hasn't changed since
  // the last publish. The subscription check comes first so dirtiness is
  // preserved while nobody listens, then delivered on the first tick after a
  // subscriber connects. The latched (transient_local) QoS retains that last
  // published snapshot and replays it to any later (re)connecting subscriber.
  void publishFusedMap()
  {
    if (!scovox_map_pub_ || scovox_map_pub_->get_subscription_count() == 0)
      return;
    if (!fused_dirty_.exchange(false)) return;
    // Full-coverage bbox: regionOnGrid clips in coord space via posToCoord
    // (floor(corner/res_)). Size the corner relative to res_ so it lands at
    // ±1e8 voxels — far outside any real map yet comfortably inside int32 at
    // ANY resolution (a fixed metric constant would overflow the cast at very
    // fine resolutions). Consumers re-apply their own ROI clip on ingest.
    const double big = 1e8 * std::max(static_cast<double>(res_), 1e-3);
    auto rq = std::make_shared<scovox_msgs::srv::GetRegion::Request>();
    rq->min_corner.x = rq->min_corner.y = rq->min_corner.z = -big;
    rq->max_corner.x = rq->max_corner.y = rq->max_corner.z =  big;
    auto rs = std::make_shared<scovox_msgs::srv::GetRegion::Response>();
    fillRegion(rq, rs);
    scovox_map_pub_->publish(rs->map);
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
  // Rate-limiter timestamp for the visualisation pointcloud, stored as raw
  // nanoseconds in a std::atomic. maybePublishPointCloud() runs under only a
  // SHARED lock (from both the binary-callback tail and the publish timer), so
  // two readers can execute concurrently; a plain rclcpp::Time would be torn /
  // race-written here. The atomic + compare_exchange below makes the
  // read-decide-update a single race-free claim so exactly one caller publishes
  // per window even under a multi-threaded executor.
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
  // Split Beta/Dirichlet fused grids. Allocated lazily on the first wire
  // frame; null otherwise. Occupancy ∥ semantics, merged independently
  // (consensus_merge.hpp). Share the pinned (fused_num_classes_, fused_alpha_0_).
  std::unique_ptr<Bonxai::VoxelGrid<scovox::BetaVoxel>> split_fused_beta_;
  std::unique_ptr<Bonxai::VoxelGrid<scovox::DirVoxel>>  split_fused_dir_;
  uint16_t fused_num_classes_{0};
  float    fused_alpha_0_{scovox::kDefaultDirichletPrior};
  // Dedicated "prior has been pinned" flag for the wire receive path. We must
  // NOT overload fused_num_classes_==0 as the "not yet pinned" sentinel: a
  // sender that ships num_classes==0 (misconfig/upstream bug) would store 0 and
  // re-pin/re-log every frame, and the cross-prior mismatch guard would never
  // run (so a second num_classes==0 source with a different alpha_0 would fuse
  // without rejection). num_classes==0 frames are rejected explicitly in the path.
  bool     prior_pinned_{false};
  mutable std::shared_mutex mu_;
  std::vector<rclcpp::Subscription<scovox_msgs::msg::ScovoxMapBinary>::SharedPtr> subs_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pc_pub_;
  rclcpp::Publisher<scovox_msgs::msg::ScovoxMap>::SharedPtr scovox_map_pub_;
  rclcpp::Service<scovox_msgs::srv::GetRegion>::SharedPtr get_region_srv_;
  rclcpp::Service<scovox_msgs::srv::GetOccupancyGrid>::SharedPtr get_occ_srv_;
  rclcpp::TimerBase::SharedPtr publish_timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DSCovoxNode>());
  rclcpp::shutdown();
}
