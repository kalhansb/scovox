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
// using the source->map TF, which is cached the first time we see that source.
//
// !! REQUIRES c-slam DISABLED !!
// The cached source->map TF is never refreshed. This is correct only while
// TFs are static. Re-enabling c-slam (loop closures / pose-graph
// optimization) is a CORRECTNESS BUG: the first TF jump leaves every voxel
// in the source grid at its old map-frame coord, producing ghost voxels at
// pre-loop-closure positions and missing voxels at the new positions. Before
// turning c-slam back on, refactor SourceGrid to store evidence in
// source-frame coords + project on demand at the current TF, with a
// per-source "TF changed → reproject" handler. See ablation entry C5 in
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
#include <scovox/binary_serializer_v2.hpp>
#include <scovox/consensus_merge_v2.hpp>
#include <scovox/binary_serializer_v3.hpp>
#include <scovox/consensus_merge_v3.hpp>
#include <scovox/semdir_voxel.hpp>
#include <scovox/sembeta_voxel.hpp>
#include <scovox_msgs/srv/get_region.hpp>
#include <scovox_msgs/srv/get_occupancy_grid.hpp>
#include <scovox_msgs/srv/score_candidates.hpp>
#include <scovox_msgs/msg/map_stats.hpp>
#include "scovox/scovoxmap.hpp"
#include "scovox/node_utils.hpp"
#include <bonxai/bonxai.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2/time.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <scovox/uncertainty.hpp>
#include <scovox/ray_iterator.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <limits>
#include <shared_mutex>

namespace {
using Ser = scovox::ScovoxBinarySerializer;

struct SourceGrid {
  std::string source_frame;                // header.frame_id of the binary
  // This source's contribution to the world, stored in MAP-FRAME coords.
  // Each entry is the latest snapshot of (a_occ, a_free, a_unk, semantics)
  // received for that map-frame voxel from this robot.
  //
  // Step 7 (D9) — exactly ONE of these two grids is allocated per source,
  // determined by the node-level `use_split` flag:
  //   - use_split=false (legacy): `grid` holds a fused 32-byte Voxel grid
  //     populated from v1 wire format.
  //   - use_split=true  (split):  `sem_grid` holds a 24-byte SemBetaVoxel
  //     grid populated from v2 wire format. No TsdfMap on the receiver
  //     because share_tsdf=false is the v2 default — TSDF state never
  //     crosses the wire to dscovox in the production path.
  std::unique_ptr<scovox::Map> grid;
  std::unique_ptr<Bonxai::VoxelGrid<scovox::SemBetaVoxel>> sem_grid;
  // Step 8 v3 (Step 7.5 substrate) — wire_format=v3 receiver populates this
  // SemDirVoxel grid in place of sem_grid. Exactly one of {grid, sem_grid,
  // semdir_grid} is allocated per source; the receiver's wire_format_v3_
  // flag picks which.
  std::unique_ptr<Bonxai::VoxelGrid<scovox::SemDirVoxel>> semdir_grid;
  // Cached static source->map transform. Looked up once on first sight and
  // never refreshed — this assumes TFs are static (c-slam disabled). Under
  // c-slam, loop closures change this TF and the cache becomes a correctness
  // bug. See the file-header banner and C5 in ablations_punch_list.md.
  Eigen::Isometry3d T_map_source{Eigen::Isometry3d::Identity()};
  bool tf_cached{false};
};

inline Eigen::Isometry3d tfToIsometry(const geometry_msgs::msg::TransformStamped& tf) {
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.translation() = Eigen::Vector3d(tf.transform.translation.x,
                                    tf.transform.translation.y,
                                    tf.transform.translation.z);
  T.linear() = Eigen::Quaterniond(tf.transform.rotation.w,
                                  tf.transform.rotation.x,
                                  tf.transform.rotation.y,
                                  tf.transform.rotation.z).toRotationMatrix();
  return T;
}

static constexpr float PRIOR_THRESH = 1.01f;
inline bool isPrior(const scovox::Voxel& v) {
  return v.a_occ <= PRIOR_THRESH && v.a_free <= PRIOR_THRESH;
}
inline bool isPrior(const scovox::SemBetaVoxel& v) {
  return v.a_occ <= PRIOR_THRESH && v.a_free <= PRIOR_THRESH;
}
/// SemDir-typed "is at prior" check. Mirrors the v2/v1 PRIOR_THRESH 1.01
/// slop — a voxel is "at prior" iff (a) alpha_free is within 1 epsilon of
/// the per-dim α_0, (b) alpha_other is within 1 epsilon of (C−K)·α_0, and
/// (c) no slot has filled. Caller must pass num_classes and alpha_0 from
/// the frame header so the test matches the sender's prior exactly.
inline bool isPriorSemDir(const scovox::SemDirVoxel& v,
                          uint16_t num_classes, float alpha_0) {
  const float other_prior =
      static_cast<float>(static_cast<int>(num_classes) - scovox::K_TOP) * alpha_0;
  // The "slop" floor matches the 1.01 absolute threshold for SemBeta: one
  // wire-quantisation worth of inflation past the analytic prior. For
  // ship α_0 = 0.01 this is ~1% slack on every test.
  const float slop = std::max(0.01f, 0.01f * alpha_0);
  if (v.alpha_free  > alpha_0     + slop) return false;
  if (v.alpha_other > other_prior + slop) return false;
  for (int i = 0; i < scovox::K_TOP; ++i)
    if (v.cls[i] != 0xFFFF) return false;
  return true;
}

/// Cheap on-the-fly SemDir → SemBeta projection for the v3 visualisation
/// path. Lossless on p_occ and per-class evidence; same projection rule
/// scovox_node uses to bridge SemDir into the legacy SemBeta wire format.
/// Avoids needing SemDir overloads of variance / EIG / argmaxClassConfidence
/// — those land alongside the wider scovox::uncertainty cleanup in a
/// follow-up.
inline scovox::SemBetaVoxel projectSemDirToSemBetaForViz(
    const scovox::SemDirVoxel& v) {
  scovox::SemBetaVoxel out{};
  out.a_occ  = v.s_occ();
  out.a_free = v.alpha_free;
  out.a_unk  = v.alpha_other;
  for (int i = 0; i < scovox::K_TOP; ++i) {
    out.sem_cnt[i] = v.cnt[i];
    out.sem_cls[i] = v.cls[i];
  }
  return out;
}
} // namespace

class DSCovoxNode : public rclcpp::Node {
public:
  DSCovoxNode()
  : rclcpp::Node("dscovox_node"),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_)
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

    // Step 7 (D8/D9) — split-grid v2 toggle. When true:
    //   - `split_fused_sem_` (a Bonxai::VoxelGrid<SemBetaVoxel>) is the
    //     consensus-merged grid; `fused_` (legacy 32-byte Voxel) stays null.
    //   - SourceGrid holds `sem_grid` instead of `grid`.
    //   - onBinaryMap accepts ONLY v2 envelopes (msg->version == 2);
    //     v1 frames are warned-and-dropped.
    //   - publishPointCloud walks the SemBeta grid and emits the same
    //     16-field PointCloud2 schema as the legacy path (D4).
    // When false (default): legacy v1 receive path is preserved verbatim.
    use_split_ = declare_parameter<bool>("use_split", false);
    // Receiver-side hint only — does not affect payload parsing because
    // share_tsdf=false (the v2 default) just elides the TSDF section, and
    // the v2 deserializer handles tsdf_count=0 naturally. Declared here
    // for symmetry with scovox_node and Step-9 launch-arg consistency.
    share_tsdf_ = declare_parameter<bool>("share_tsdf", false);
    // Step 8 — wire format selector. v2 is the production default; v3 is
    // the SemDir-native receive path (consensus_merge_v3.hpp). The flag
    // also gates which fused grid is allocated (split_fused_sem_ vs
    // split_fused_semdir_) so callers must NOT swap mid-run. v3 receiver
    // requires use_split=true on this node and on every connected sender;
    // dscovox does not bridge v2 senders into the SemDir fused grid.
    {
      const std::string fmt = declare_parameter<std::string>("wire_format", "v2");
      if (fmt == "v3")      wire_format_v3_ = true;
      else if (fmt == "v2") wire_format_v3_ = false;
      else {
        RCLCPP_WARN(get_logger(),
          "dscovox wire_format='%s' unrecognised (want v2|v3); defaulting v2.",
          fmt.c_str());
        wire_format_v3_ = false;
      }
    }
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
    kl_thresh_ = declare_parameter<double>("consensus_kl_threshold", 5.0);
    tau_gate_ = declare_parameter<double>("tau_occ_gate", 0.6);
    pub_hz_ = declare_parameter<double>("publish_rate_hz", 1.0);
    // Minimum interval between visualization-pointcloud publishes when
    // triggered by binary callbacks. Default 0.1s = 10 Hz cap. Lower this
    // for snappier RViz updates at the cost of CPU; raise it to throttle.
    pc_min_interval_s_ = declare_parameter<double>("pointcloud_min_interval_s", 0.1);
    last_pc_pub_time_ = get_clock()->now();
    pub_plan_ = declare_parameter<bool>("publish_planning_map", false);
    plan_res_ = declare_parameter<double>("planning_map_resolution", 0.20);
    plan_sz_ = declare_parameter<double>("planning_map_size_m", 80.0);
    plan_ox_ = declare_parameter<double>("planning_map_origin_x", -40.0);
    plan_oy_ = declare_parameter<double>("planning_map_origin_y", -40.0);
    plan_zmin_ = declare_parameter<double>("planning_map_min_z", -1.0);
    plan_zmax_ = declare_parameter<double>("planning_map_max_z", 2.0);
    plan_inf_ = declare_parameter<double>("planning_map_inflation_m", 0.0);

    initSemanticColors();

    pc_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      declare_parameter<std::string>("pointcloud_topic", "~/pointcloud"),
      rclcpp::SystemDefaultsQoS());
    if (pub_plan_) {
      og_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>(
        declare_parameter<std::string>("planning_map_topic", "~/planning_map"),
        rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local());
    }
    sm_pub_ = create_publisher<scovox_msgs::msg::ScovoxMap>(
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
    score_candidates_srv_ = create_service<scovox_msgs::srv::ScoreCandidates>(
      "~/score_candidates",
      std::bind(&DSCovoxNode::onScoreCandidates, this, std::placeholders::_1, std::placeholders::_2));

    // Frontier centroids publisher — lightweight topic (~2 KB) so the
    // exploration planner can build frontier candidates without fetching
    // the full voxel grid.
    frontier_min_z_ = declare_parameter<double>("frontier_min_z", -0.5);
    frontier_max_z_ = declare_parameter<double>("frontier_max_z", 2.0);
    frontier_cluster_radius_ = declare_parameter<double>("frontier_cluster_radius", 5.0);
    frontier_pub_ = create_publisher<geometry_msgs::msg::PoseArray>(
      declare_parameter<std::string>("frontier_centroids_topic", "~/frontier_centroids"),
      rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local());

    if (pub_hz_ > 0.0) {
      // The fused grid is kept current incrementally inside onBinaryMap.
      // The visualization pointcloud is normally driven from there too;
      // this timer is the fallback (when no binaries arrive) plus the
      // primary driver for the heavier ScovoxMap and PlanningMap topics
      // that don't need visualization-rate refresh.
      //
      // One outer shared_lock spans every publisher in the tick so they all
      // see the same fused state — without it, an onBinaryMap callback can
      // mutate `fused_` between any two of them and the pointcloud, ScovoxMap
      // and planning_map can disagree within a single tick. publish* helpers
      // must NOT take map_mtx_ themselves — std::shared_mutex is non-recursive
      // so re-locking here would be UB.
      publish_timer_ = rclcpp::create_timer(
        this, get_clock(),
        std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::duration<double>(1.0 / pub_hz_)),
        [this] {
          std::shared_lock<std::shared_mutex> lk(mu_);
          maybePublishPointCloud();
          publishScovoxMap();
          if (pub_plan_ && og_pub_) publishPlanningMap();
          publishFrontierCentroids();
          size_t fc = 0, ts = 0;
          if (use_split_) {
            fc = split_fused_sem_ ? split_fused_sem_->activeCellsCount() : 0;
            for (auto& [k, sg] : sources_)
              if (sg.sem_grid) ts += sg.sem_grid->activeCellsCount();
          } else {
            fc = fused_ ? fused_->grid().activeCellsCount() : 0;
            for (auto& [k, sg] : sources_)
              if (sg.grid) ts += sg.grid->grid().activeCellsCount();
          }
          RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000,
            "dscovox_diag: mode=%s sources=%zu src_voxels=%zu fused_voxels=%zu",
            use_split_ ? "split" : "legacy",
            sources_.size(), ts, fc);
        });
    }

    RCLCPP_INFO(get_logger(),
      "DSCovoxNode started: %zu inputs, frame='%s'",
      input_topics_.size(), map_frame_.c_str());
  }

private:
  scovox::Params fusedP() const {
    scovox::Params P;
    P.resolution = res_;
    P.consensus_kl_threshold = (float)kl_thresh_;
    P.consensus_tau_occ_gate = (float)tau_gate_;
    return P;
  }

  bool lookupTF(const std::string& frame, Eigen::Isometry3d& T) {
    try {
      T = tfToIsometry(tf_buffer_.lookupTransform(
        map_frame_, frame, rclcpp::Time(0), rclcpp::Duration::from_seconds(0.1)));
      return true;
    } catch (...) { return false; }
  }

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

  // ScovoxMapBinary handler. Decode the delta, write each voxel into this
  // source's map-frame grid (overwriting that source's previous contribution),
  // and incrementally re-fold the affected fused cells from all sources.
  //
  // The reset-to-prior + refold-all-sources pattern in refoldCell is what
  // keeps this equivalent to a from-scratch rebuild and prevents
  // double-counting when the same source publishes the same voxel twice.
  void onBinaryMap(const scovox_msgs::msg::ScovoxMapBinary::SharedPtr msg) {
    if (use_split_) {
      if (wire_format_v3_) onBinaryMapV3(msg);
      else                 onBinaryMapV2(msg);
      return;
    }
    if (msg->version != 1) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "Bad version %d", msg->version);
      return;
    }
    const std::string sf = msg->header.frame_id;
    std::string buf = Ser::decompressLZ4(msg->data);
    if (buf.empty()) {
      RCLCPP_ERROR(get_logger(), "LZ4 fail '%s'", sf.c_str());
      return;
    }
    auto upd = Ser::deserializeIncremental(buf, top_k_);
    if (upd.voxels.empty()) return;

    // Resolve / cache the source->map TF before taking the lock. Under the
    // c-slam-disabled assumption this is static, so we only ever look it up
    // once per source. If c-slam is re-enabled this path must be reworked
    // (see C5 in ablations_punch_list.md).
    Eigen::Isometry3d Tmo;
    {
      std::shared_lock<std::shared_mutex> rlk(mu_);
      auto it = sources_.find(sf);
      if (it != sources_.end() && it->second.tf_cached) {
        Tmo = it->second.T_map_source;
      } else {
        rlk.unlock();
        if (!lookupTF(sf, Tmo)) {
          RCLCPP_WARN(get_logger(), "No TF for '%s'", sf.c_str());
          return;
        }
      }
    }

    // Source resolution for transforming source-frame coords -> world position.
    // Prefer the binary's stated resolution; fall back to res_ once it's set.
    float src_res = upd.resolution > 0.f ? upd.resolution : Ser::parseResolution(buf);

    {
      std::unique_lock<std::shared_mutex> lk(mu_);
      if (res_ <= 0.f) {
        res_ = src_res > 0.f ? src_res : 0.1f;
      }
      if (src_res <= 0.f) src_res = res_;
      if (!fused_) fused_ = std::make_unique<scovox::Map>(fusedP());

      auto it = sources_.find(sf);
      if (it == sources_.end()) {
        SourceGrid sg;
        sg.source_frame = sf;
        sg.T_map_source = Tmo;
        sg.tf_cached = true;
        scovox::Params P;
        P.resolution = res_;
        sg.grid = std::make_unique<scovox::Map>(P);
        it = sources_.emplace(sf, std::move(sg)).first;
      } else if (!it->second.tf_cached) {
        it->second.T_map_source = Tmo;
        it->second.tf_cached = true;
      }
      auto& src = it->second;

      // Step 1: ingest the delta into this source's MAP-FRAME grid. Each delta
      // voxel is given in source-frame coords; transform once to a world
      // position and then to the corresponding map-frame voxel coord.
      std::unordered_set<Bonxai::CoordT, CoordTHash, CoordTEqual> touched;
      touched.reserve(upd.voxels.size());
      auto sa = src.grid->grid().createAccessor();
      const Eigen::Isometry3d Te = src.T_map_source;
      for (auto& w : upd.voxels) {
        // Source-frame world position of the *center* of this voxel. We must
        // sample the center, not the lower corner, because posToCoord floors:
        // for any source->map TF that is not voxel-grid-aligned (e.g. a half-
        // voxel translation), the lower corner deterministically lands on a
        // voxel boundary and floor() biases every assignment to the same
        // neighbour, when in reality the source voxel overlaps two map voxels
        // 50/50. Sampling the center instead means floor() picks whichever
        // map voxel actually contains most of the source voxel's volume.
        const double half_src_res = 0.5 * double(src_res);
        Eigen::Vector3d sp(
          double(w.x) * double(src_res) + half_src_res,
          double(w.y) * double(src_res) + half_src_res,
          double(w.z) * double(src_res) + half_src_res);
        Eigen::Vector3d mp = Te * sp;
        auto mc = src.grid->grid().posToCoord(mp.x(), mp.y(), mp.z());
        auto* v = sa.value(mc, true);
        if (!v) continue;
        v->a_occ = w.a_occ;
        v->a_free = w.a_free;
        v->a_unk = w.a_unk;
        // Reset the K_TOP semantic slots so the new state replaces (not merges
        // with) any leftover slot from a previous binary.
        for (int j = 0; j < scovox::K_TOP; ++j) { v->sem_cls[j] = 0; v->sem_cnt[j] = 0.f; }
        for (size_t j = 0; j < w.top.size() && j < (size_t)scovox::K_TOP; ++j) {
          v->sem_cls[j] = (uint16_t)w.top[j].first;
          v->sem_cnt[j] = w.top[j].second;
        }
        touched.insert(mc);
      }

      // Step 2: refold each touched cell from all sources' current state.
      // Build per-source accessors once outside the per-cell loop so each
      // cell lookup keeps Bonxai's leaf cache warm. Previously refoldCell
      // built a fresh accessor per source per cell, throwing away the
      // cache and turning every value() into a cold root→inner→leaf walk.
      auto fa = fused_->grid().createAccessor();
      std::vector<scovox::Map::Grid::Accessor> source_accs;
      source_accs.reserve(sources_.size());
      for (auto& [k, sg] : sources_) {
        source_accs.emplace_back(sg.grid->grid().createAccessor());
      }
      for (const auto& mc : touched) {
        refoldCell(mc, fa, source_accs);
      }
    }  // unique_lock released here

    // Step 3: drive the visualization pointcloud as soon as fused is fresh,
    // rate-limited so we don't burn CPU on every binary. maybePublishPointCloud
    // expects the caller to hold the shared lock (the unique_lock above is
    // released first because std::shared_mutex is non-recursive).
    {
      std::shared_lock<std::shared_mutex> rlk(mu_);
      maybePublishPointCloud();
    }
  }

  // Reset fused[mc] to prior, then fold every source's current value at mc.
  // This is the per-cell equivalent of what rebuildFused() used to do for
  // every cell at once. Equivalent to a from-scratch rebuild for this cell
  // because:
  //   - We start from prior, so any stale contribution is wiped.
  //   - We iterate every source, so we never miss a source's contribution.
  //   - The first non-prior source is copied directly (matching the old
  //     "first writer copies, rest merge" pattern in rebuildFused).
  //   - consensusMerge with the rest is order-equivalent for the alpha terms
  //     because every voxel has a_* >= 1, so the max(1, ...) floor never bites.
  //
  // `source_accs` must be in the same order as iteration over sources_ at
  // construction time. The caller builds these once outside the per-cell
  // loop so each value() call benefits from Bonxai's per-accessor leaf cache.
  void refoldCell(const Bonxai::CoordT& mc,
                  scovox::Map::Grid::Accessor& fa,
                  std::vector<scovox::Map::Grid::Accessor>& source_accs) {
    auto* fv = fa.value(mc, true);
    if (!fv) return;
    *fv = scovox::defaultVoxel();
    for (auto& sa : source_accs) {
      auto* sv = sa.value(mc, false);
      if (!sv || isPrior(*sv)) continue;
      if (isPrior(*fv)) *fv = *sv;
      else fused_->consensusMerge(*fv, *sv);
    }
  }

  // ==================================================================
  // Step 7 (D8) — split-grid v2 receive path. Mirrors the legacy v1
  // pipeline below but operates on Bonxai::VoxelGrid<SemBetaVoxel>:
  //   1. Validate envelope (msg->version==2).
  //   2. Cache source->map TF (shared with v1 path).
  //   3. Decompress LZ4, deserialise the v2 dual-stream frame.
  //      Per D2/D9 the production setting is share_tsdf=false on the
  //      sender, so f.tsdf_deltas is normally empty here — we ignore
  //      whatever TSDF deltas might be present (consensus dscovox does
  //      NOT own a TsdfMap; each robot keeps its own local TSDF for
  //      mesh extraction). f.sembeta_deltas is the only consumed payload.
  //   4. Ingest each SemBeta delta into the source's MAP-FRAME
  //      Bonxai::VoxelGrid<SemBetaVoxel> at the TF-transformed voxel
  //      centre (centre-sample, not lower-corner — same anti-aliasing
  //      argument as the v1 path).
  //   5. Refold every touched map-frame coord by reset-to-prior +
  //      mergeSemBeta(consensus_merge_v2.hpp) over every source's
  //      contribution. Equivalent to a from-scratch rebuild for those
  //      cells, no double-counting.
  //   6. Drive the visualisation publisher.
  // ==================================================================
  void onBinaryMapV2(const scovox_msgs::msg::ScovoxMapBinary::SharedPtr msg) {
    if (msg->version != 2) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "use_split=true expects v2 envelope, got version %d (dropping)",
        msg->version);
      return;
    }
    const std::string sf = msg->header.frame_id;
    std::string buf = scovox::ScovoxBinarySerializer::decompressLZ4(msg->data);
    if (buf.empty()) {
      RCLCPP_ERROR(get_logger(), "LZ4 fail '%s'", sf.c_str());
      return;
    }

    scovox::BinarySerializerV2::Frame frame;
    try {
      frame = scovox::BinarySerializerV2::deserialize(buf);
    } catch (const std::exception& e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "v2 deserialize failed for '%s': %s", sf.c_str(), e.what());
      return;
    }
    if (frame.sembeta_deltas.empty()) return;

    // TF cache (mirrors v1 path).
    Eigen::Isometry3d Tmo;
    {
      std::shared_lock<std::shared_mutex> rlk(mu_);
      auto it = sources_.find(sf);
      if (it != sources_.end() && it->second.tf_cached) {
        Tmo = it->second.T_map_source;
      } else {
        rlk.unlock();
        if (!lookupTF(sf, Tmo)) {
          RCLCPP_WARN(get_logger(), "No TF for '%s'", sf.c_str());
          return;
        }
      }
    }

    float src_res = frame.resolution > 0.f ? frame.resolution : 0.f;

    {
      std::unique_lock<std::shared_mutex> lk(mu_);
      if (res_ <= 0.f) {
        res_ = src_res > 0.f ? src_res : 0.1f;
      }
      if (src_res <= 0.f) src_res = res_;

      if (!split_fused_sem_) {
        scovox::Params P;
        P.resolution = res_;
        split_fused_sem_ = std::make_unique<Bonxai::VoxelGrid<scovox::SemBetaVoxel>>(
            P.resolution, P.inner_bits, P.leaf_bits);
      }

      auto it = sources_.find(sf);
      if (it == sources_.end()) {
        SourceGrid sg;
        sg.source_frame = sf;
        sg.T_map_source = Tmo;
        sg.tf_cached = true;
        scovox::Params P;
        P.resolution = res_;
        sg.sem_grid = std::make_unique<Bonxai::VoxelGrid<scovox::SemBetaVoxel>>(
            P.resolution, P.inner_bits, P.leaf_bits);
        it = sources_.emplace(sf, std::move(sg)).first;
      } else if (!it->second.tf_cached) {
        it->second.T_map_source = Tmo;
        it->second.tf_cached = true;
      }
      auto& src = it->second;
      if (!src.sem_grid) {
        // Source first seen on the legacy path; create the SemBeta grid
        // on demand. (Mode-switch midstream is not supported but this
        // makes it fail-soft rather than null-deref.)
        scovox::Params P; P.resolution = res_;
        src.sem_grid = std::make_unique<Bonxai::VoxelGrid<scovox::SemBetaVoxel>>(
            P.resolution, P.inner_bits, P.leaf_bits);
      }

      // Step 1 — ingest the delta into this source's MAP-FRAME grid.
      // Centre-sample (not lower-corner) for the same anti-aliasing
      // reason the v1 path uses (see refoldCell preamble in the legacy
      // path). posToCoord floors; sampling at the centre means the
      // floored cell deterministically contains the bulk of the source
      // voxel's volume even when the TF is not voxel-grid-aligned.
      std::unordered_set<Bonxai::CoordT, CoordTHash, CoordTEqual> touched;
      touched.reserve(frame.sembeta_deltas.size());
      auto sa = src.sem_grid->createAccessor();
      const Eigen::Isometry3d Te = src.T_map_source;
      const double half_src_res = 0.5 * double(src_res);
      for (auto& d : frame.sembeta_deltas) {
        Eigen::Vector3d sp(
          double(d.coord.x) * double(src_res) + half_src_res,
          double(d.coord.y) * double(src_res) + half_src_res,
          double(d.coord.z) * double(src_res) + half_src_res);
        Eigen::Vector3d mp = Te * sp;
        auto mc = src.sem_grid->posToCoord(mp.x(), mp.y(), mp.z());
        auto* v = sa.value(mc, true);
        if (!v) continue;
        // Snapshot replace — same semantic as v1: each binary carries the
        // current state of the touched voxels, not an additive delta. The
        // refold step below puts the merge back together across sources.
        *v = d.data;
        touched.insert(mc);
      }

      // Step 2 — refold each touched cell from all sources' current
      // SemBeta state.
      auto fa = split_fused_sem_->createAccessor();
      std::vector<Bonxai::VoxelGrid<scovox::SemBetaVoxel>::Accessor> source_accs;
      source_accs.reserve(sources_.size());
      for (auto& [k, sg] : sources_) {
        if (sg.sem_grid) source_accs.emplace_back(sg.sem_grid->createAccessor());
      }
      for (const auto& mc : touched) {
        refoldCellSemBeta(mc, fa, source_accs);
      }
    }  // unique_lock released

    // Step 3 — visualisation publisher driven from the callback tail.
    {
      std::shared_lock<std::shared_mutex> rlk(mu_);
      maybePublishPointCloud();
    }
  }

  // ==================================================================
  // Step 8 — split-grid v3 receive path. Same skeleton as onBinaryMapV2
  // but on SemDirVoxel + consensus_merge_v3.hpp. Header carries
  // num_classes + alpha_0 so the receiver knows the symmetric Dirichlet
  // prior used by the sender; the first received frame fixes these for
  // the lifetime of the node. A subsequent frame with mismatched
  // (num_classes, alpha_0) is dropped with a throttled warning — fusing
  // across mismatched priors would silently bias the consensus voxels.
  // ==================================================================
  void onBinaryMapV3(const scovox_msgs::msg::ScovoxMapBinary::SharedPtr msg) {
    if (msg->version != 3) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "wire_format=v3 expects v3 envelope, got version %d (dropping)",
        msg->version);
      return;
    }
    const std::string sf = msg->header.frame_id;
    std::string buf = scovox::ScovoxBinarySerializer::decompressLZ4(msg->data);
    if (buf.empty()) {
      RCLCPP_ERROR(get_logger(), "LZ4 fail '%s'", sf.c_str());
      return;
    }

    scovox::BinarySerializerV3::Frame frame;
    try {
      frame = scovox::BinarySerializerV3::deserialize(buf);
    } catch (const std::exception& e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "v3 deserialize failed for '%s': %s", sf.c_str(), e.what());
      return;
    }
    if (frame.semdir_deltas.empty()) return;

    // TF cache (same pattern as v1/v2).
    Eigen::Isometry3d Tmo;
    {
      std::shared_lock<std::shared_mutex> rlk(mu_);
      auto it = sources_.find(sf);
      if (it != sources_.end() && it->second.tf_cached) {
        Tmo = it->second.T_map_source;
      } else {
        rlk.unlock();
        if (!lookupTF(sf, Tmo)) {
          RCLCPP_WARN(get_logger(), "No TF for '%s'", sf.c_str());
          return;
        }
      }
    }

    float src_res = frame.resolution > 0.f ? frame.resolution : 0.f;

    {
      std::unique_lock<std::shared_mutex> lk(mu_);
      // Pin the symmetric-Dirichlet prior on first valid frame, then
      // assert match on every subsequent frame. Cross-prior fusion is
      // a correctness bug (the OTHER subtraction in mergeSemDir uses
      // the receiver's stored α_0; mismatched senders would over- or
      // under-subtract).
      if (fused_num_classes_ == 0) {
        fused_num_classes_ = frame.num_classes;
        fused_alpha_0_     = frame.alpha_0;
        RCLCPP_INFO(get_logger(),
          "v3 receive: pinned num_classes=%u alpha_0=%.4f (from first frame, src='%s')",
          (unsigned)fused_num_classes_, fused_alpha_0_, sf.c_str());
      } else {
        if (frame.num_classes != fused_num_classes_ ||
            std::abs(frame.alpha_0 - fused_alpha_0_) > 1e-6f) {
          RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
            "v3 prior mismatch from '%s' (got C=%u α=%.4f, pinned C=%u α=%.4f) — dropping frame",
            sf.c_str(),
            (unsigned)frame.num_classes, frame.alpha_0,
            (unsigned)fused_num_classes_, fused_alpha_0_);
          return;
        }
      }

      if (res_ <= 0.f) res_ = src_res > 0.f ? src_res : 0.1f;
      if (src_res <= 0.f) src_res = res_;

      if (!split_fused_semdir_) {
        scovox::Params P; P.resolution = res_;
        split_fused_semdir_ = std::make_unique<Bonxai::VoxelGrid<scovox::SemDirVoxel>>(
            P.resolution, P.inner_bits, P.leaf_bits);
      }

      auto it = sources_.find(sf);
      if (it == sources_.end()) {
        SourceGrid sg;
        sg.source_frame = sf;
        sg.T_map_source = Tmo;
        sg.tf_cached = true;
        scovox::Params P; P.resolution = res_;
        sg.semdir_grid = std::make_unique<Bonxai::VoxelGrid<scovox::SemDirVoxel>>(
            P.resolution, P.inner_bits, P.leaf_bits);
        it = sources_.emplace(sf, std::move(sg)).first;
      } else if (!it->second.tf_cached) {
        it->second.T_map_source = Tmo;
        it->second.tf_cached = true;
      }
      auto& src = it->second;
      if (!src.semdir_grid) {
        scovox::Params P; P.resolution = res_;
        src.semdir_grid = std::make_unique<Bonxai::VoxelGrid<scovox::SemDirVoxel>>(
            P.resolution, P.inner_bits, P.leaf_bits);
      }

      // Step 1 — ingest the delta into this source's MAP-FRAME grid.
      // Centre-sample posToCoord (same anti-aliasing argument as v2).
      std::unordered_set<Bonxai::CoordT, CoordTHash, CoordTEqual> touched;
      touched.reserve(frame.semdir_deltas.size());
      auto sa = src.semdir_grid->createAccessor();
      const Eigen::Isometry3d Te = src.T_map_source;
      const double half_src_res = 0.5 * double(src_res);
      for (auto& d : frame.semdir_deltas) {
        Eigen::Vector3d sp(
          double(d.coord.x) * double(src_res) + half_src_res,
          double(d.coord.y) * double(src_res) + half_src_res,
          double(d.coord.z) * double(src_res) + half_src_res);
        Eigen::Vector3d mp = Te * sp;
        auto mc = src.semdir_grid->posToCoord(mp.x(), mp.y(), mp.z());
        auto* v = sa.value(mc, true);
        if (!v) continue;
        *v = d.data;     // snapshot-replace, same contract as v2
        touched.insert(mc);
      }

      // Step 2 — refold each touched cell from all sources' SemDir state.
      auto fa = split_fused_semdir_->createAccessor();
      std::vector<Bonxai::VoxelGrid<scovox::SemDirVoxel>::Accessor> source_accs;
      source_accs.reserve(sources_.size());
      for (auto& [k, sg] : sources_) {
        if (sg.semdir_grid) source_accs.emplace_back(sg.semdir_grid->createAccessor());
      }
      for (const auto& mc : touched) {
        refoldCellSemDir(mc, fa, source_accs);
      }
    }  // unique_lock released

    // Step 3 — visualisation publish.
    {
      std::shared_lock<std::shared_mutex> rlk(mu_);
      maybePublishPointCloud();
    }
  }

  // SemDir-typed refold. Reset fused[mc] to defaultSemDirVoxel (using the
  // pinned receiver prior), then fold every source's current value at mc
  // via mergeSemDir. First non-prior source copied directly; rest merged.
  void refoldCellSemDir(
      const Bonxai::CoordT& mc,
      Bonxai::VoxelGrid<scovox::SemDirVoxel>::Accessor& fa,
      std::vector<Bonxai::VoxelGrid<scovox::SemDirVoxel>::Accessor>& source_accs)
  {
    auto* fv = fa.value(mc, true);
    if (!fv) return;
    *fv = scovox::defaultSemDirVoxel(fused_num_classes_, fused_alpha_0_);
    bool seeded = false;
    for (auto& sa : source_accs) {
      auto* sv = sa.value(mc, false);
      if (!sv || isPriorSemDir(*sv, fused_num_classes_, fused_alpha_0_)) continue;
      if (!seeded) { *fv = *sv; seeded = true; }
      else         { *fv = scovox::mergeSemDir(*fv, *sv,
                                               fused_num_classes_, fused_alpha_0_); }
    }
  }

  // SemBeta-typed analogue of refoldCell. Reset fused[mc] to the
  // SemBeta default (Beta(1,1) prior, empty Dirichlet slots), then
  // fold every source's current value at mc via mergeSemBeta. The
  // first non-prior source is copied directly; the rest are merged
  // (Beta consensus + sparse_add replay per consensus_merge_v2.hpp).
  void refoldCellSemBeta(
      const Bonxai::CoordT& mc,
      Bonxai::VoxelGrid<scovox::SemBetaVoxel>::Accessor& fa,
      std::vector<Bonxai::VoxelGrid<scovox::SemBetaVoxel>::Accessor>& source_accs)
  {
    auto* fv = fa.value(mc, true);
    if (!fv) return;
    *fv = scovox::defaultSemBetaVoxel();
    bool seeded = false;
    for (auto& sa : source_accs) {
      auto* sv = sa.value(mc, false);
      if (!sv || isPrior(*sv)) continue;
      if (!seeded) { *fv = *sv; seeded = true; }
      else         { *fv = scovox::mergeSemBeta(*fv, *sv); }
    }
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
    auto now = get_clock()->now();
    if ((now - last_pc_pub_time_).seconds() < pc_min_interval_s_) return;
    last_pc_pub_time_ = now;
    if (use_split_) {
      if (wire_format_v3_) publishPointCloudV3();
      else                 publishPointCloudV2();
    } else {
      publishPointCloud();
    }
  }

  // Caller must hold mu_ (shared). Inner lock removed so that the timer body
  // and onBinaryMap can hold one outer shared_lock spanning every publisher
  // — std::shared_mutex is non-recursive so re-locking here would be UB.
  void publishPointCloud() {
    if (!pc_pub_ || !fused_ || pc_pub_->get_subscription_count() == 0) return;
    auto& g = fused_->grid();
    float ot = (float)min_occ_;
    size_t cnt = 0;
    g.forEachCell([&](const scovox::Voxel& v, const Bonxai::CoordT&) { if (v.p_occ() >= ot) ++cnt; });
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
    g.forEachCell([&](const scovox::Voxel& v, const Bonxai::CoordT& co) {
      float pr = v.p_occ();
      if (pr < ot) return;
      auto p = g.coordToPos(co);
      *ix = p.x; *iy = p.y; *iz = p.z; *ip = pr;
      // Hutter-framework (K+1)-Dirichlet posterior probability of the
      // argmax tracked class — consistent with semanticEntropy /
      // semanticVariance. See node_utils.hpp::argmaxClassConfidence.
      const auto [best_cls, cf] = scovox::argmaxClassConfidence(v);
      const uint8_t bc = static_cast<uint8_t>(best_cls);
      *ik = bc; *ic = cf;
      float r = 1, gg = 1, b = 1;
      if (v.a0() > 0 && cf >= sem_gate_ && bc < sem_col_.size()) {
        r = sem_col_[bc][0]; gg = sem_col_[bc][1]; b = sem_col_[bc][2];
      }
      uint32_t rp = ((uint32_t)(r * 255) << 16) | ((uint32_t)(gg * 255) << 8) | (uint32_t)(b * 255);
      *ir = *reinterpret_cast<float*>(&rp);
      *iv = scovox::variance(v);
      *ie = scovox::expectedInformationGain(v);
      *iao = v.a_occ; *iaf = v.a_free;
      ++ix; ++iy; ++iz; ++ir; ++ip; ++ik; ++ic; ++iv; ++ie; ++iao; ++iaf;
    });
    pc_pub_->publish(cl);
  }

  // Step 7 (D4) — split-mode pointcloud publisher. Walks split_fused_sem_
  // (Bonxai::VoxelGrid<SemBetaVoxel>) and emits the same 11-field schema
  // as publishPointCloud above so RViz visualisations and downstream
  // consumers (pointcloud_to_npz.py, eval scripts) need no schema-aware
  // routing. The D6 helper overloads (variance / expectedInformationGain /
  // argmaxClassConfidence) keep this body byte-identical to the legacy
  // version up to the voxel type.
  //
  // Caller must hold mu_ (shared). Inner lock removed for the same
  // shared_mutex non-recursivity reason as publishPointCloud.
  void publishPointCloudV2() {
    if (!pc_pub_ || !split_fused_sem_ || pc_pub_->get_subscription_count() == 0) return;
    auto& g = *split_fused_sem_;
    float ot = (float)min_occ_;
    size_t cnt = 0;
    g.forEachCell([&](const scovox::SemBetaVoxel& v, const Bonxai::CoordT&) {
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
    g.forEachCell([&](const scovox::SemBetaVoxel& v, const Bonxai::CoordT& co) {
      float pr = v.p_occ();
      if (pr < ot) return;
      auto p = g.coordToPos(co);
      *ix = p.x; *iy = p.y; *iz = p.z; *ip = pr;
      const auto [best_cls, cf] = scovox::argmaxClassConfidence(v);
      const uint8_t bc = static_cast<uint8_t>(best_cls);
      *ik = bc; *ic = cf;
      float r = 1, gg = 1, b = 1;
      if (v.a0() > 0 && cf >= sem_gate_ && bc < sem_col_.size()) {
        r = sem_col_[bc][0]; gg = sem_col_[bc][1]; b = sem_col_[bc][2];
      }
      uint32_t rp = ((uint32_t)(r * 255) << 16) | ((uint32_t)(gg * 255) << 8) | (uint32_t)(b * 255);
      *ir = *reinterpret_cast<float*>(&rp);
      *iv = scovox::variance(v);
      *ie = scovox::expectedInformationGain(v);
      *iao = v.a_occ; *iaf = v.a_free;
      ++ix; ++iy; ++iz; ++ir; ++ip; ++ik; ++ic; ++iv; ++ie; ++iao; ++iaf;
    });
    pc_pub_->publish(cl);
  }

  // Step 8 — v3 visualisation publisher. Walks split_fused_semdir_
  // (Bonxai::VoxelGrid<SemDirVoxel>) and emits the same 11-field schema
  // as publishPointCloudV2 by projecting each SemDirVoxel back to a
  // transient SemBetaVoxel-equivalent (projectSemDirToSemBetaForViz).
  // The projection is lossless on p_occ and on per-class evidence —
  // RViz colours and downstream NPZ scorers stay byte-identical to the
  // v2 path.
  void publishPointCloudV3() {
    if (!pc_pub_ || !split_fused_semdir_ ||
        pc_pub_->get_subscription_count() == 0) return;
    auto& g = *split_fused_semdir_;
    float ot = (float)min_occ_;
    size_t cnt = 0;
    g.forEachCell([&](const scovox::SemDirVoxel& v, const Bonxai::CoordT&) {
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
    g.forEachCell([&](const scovox::SemDirVoxel& v_sd, const Bonxai::CoordT& co) {
      float pr = v_sd.p_occ();
      if (pr < ot) return;
      const scovox::SemBetaVoxel v = projectSemDirToSemBetaForViz(v_sd);
      auto p = g.coordToPos(co);
      *ix = p.x; *iy = p.y; *iz = p.z; *ip = pr;
      const auto [best_cls, cf] = scovox::argmaxClassConfidence(v);
      const uint8_t bc = static_cast<uint8_t>(best_cls);
      *ik = bc; *ic = cf;
      float r = 1, gg = 1, b = 1;
      if (v.a0() > 0 && cf >= sem_gate_ && bc < sem_col_.size()) {
        r = sem_col_[bc][0]; gg = sem_col_[bc][1]; b = sem_col_[bc][2];
      }
      uint32_t rp = ((uint32_t)(r * 255) << 16) | ((uint32_t)(gg * 255) << 8) | (uint32_t)(b * 255);
      *ir = *reinterpret_cast<float*>(&rp);
      *iv = scovox::variance(v);
      *ie = scovox::expectedInformationGain(v);
      *iao = v.a_occ; *iaf = v.a_free;
      ++ix; ++iy; ++iz; ++ir; ++ip; ++ik; ++ic; ++iv; ++ie; ++iao; ++iaf;
    });
    pc_pub_->publish(cl);
  }

  // Publish the fused grid as a full ScovoxMap so downstream consumers
  // (e.g. exploration planner) can read merged voxel posteriors directly.
  // Frame is map_frame_; voxel positions are in map-frame coordinates.
  //
  // Caller must hold mu_ (shared). Inner lock removed so the timer body holds
  // a single outer shared_lock spanning every publisher.
  void publishScovoxMap() {
    if (!sm_pub_ || !fused_) return;
    if (sm_pub_->get_subscription_count() == 0) return;
    auto& g = fused_->grid();
    scovox_msgs::msg::ScovoxMap m;
    m.header.stamp = get_clock()->now();
    m.header.frame_id = map_frame_;
    m.resolution = res_;
    m.occupancy_threshold = (float)min_occ_;
    m.semantic_threshold = (float)sem_gate_;
    m.max_semantic_classes = (uint8_t)top_k_;
    m.voxels.reserve(g.activeCellsCount());
    g.forEachCell([&](const scovox::Voxel& v, const Bonxai::CoordT& c) {
      if (isPrior(v)) return;
      auto p = g.coordToPos(c);
      scovox_msgs::msg::ScovoxVoxel vv;
      vv.position.x = p.x; vv.position.y = p.y; vv.position.z = p.z;
      vv.a_occ = v.a_occ; vv.a_free = v.a_free; vv.a_unk = v.a_unk;
      // Pick the top top_k_ strongest classes (sparse_add doesn't sort) and
      // fold dropped mass into a_unk so total semantic evidence is preserved.
      const auto top = scovox::selectTopKSemantics(v, top_k_);
      for (size_t i = 0; i < top.kept_count; ++i) {
        scovox_msgs::msg::ScovoxSemanticEvidence se;
        se.class_id = top.kept[i].first;
        se.evidence_count = top.kept[i].second;
        vv.semantic_evidence.push_back(se);
      }
      vv.a_unk += top.dropped_mass;
      m.voxels.push_back(std::move(vv));
    });
    sm_pub_->publish(m);
  }

  // Caller must hold mu_ (shared). Inner lock removed so the timer body holds
  // a single outer shared_lock spanning every publisher.
  void publishPlanningMap() {
    if (!og_pub_ || !fused_) return;
    if (og_pub_->get_subscription_count() == 0) return;
    int W = std::max(1, (int)std::round(plan_sz_ / plan_res_));
    nav_msgs::msg::OccupancyGrid gm;
    gm.header.stamp = get_clock()->now();
    gm.header.frame_id = map_frame_;
    gm.info.resolution = (float)plan_res_;
    gm.info.width = W;
    gm.info.height = W;
    gm.info.origin.position.x = plan_ox_;
    gm.info.origin.position.y = plan_oy_;
    gm.info.origin.orientation.w = 1.0;
    gm.data.assign(W * W, -1);
    float ot = (float)min_occ_;
    fused_->grid().forEachCell([&](const scovox::Voxel& v, const Bonxai::CoordT& c) {
      auto p = fused_->grid().coordToPos(c);
      if (p.z < plan_zmin_ || p.z > plan_zmax_) return;
      int gx = (int)std::floor(((double)p.x - plan_ox_) / plan_res_);
      int gy = (int)std::floor(((double)p.y - plan_oy_) / plan_res_);
      if (gx < 0 || gy < 0 || gx >= W || gy >= W) return;
      int i = gy * W + gx;
      if (v.p_occ() >= ot) gm.data[i] = 100;
      else if (gm.data[i] != 100) gm.data[i] = 0;
    });
    int ic = (int)std::ceil(plan_inf_ / plan_res_);
    int ic2 = ic * ic;
    if (ic > 0) {
      auto inf = gm.data;
      for (int y = 0; y < W; ++y) for (int x = 0; x < W; ++x) {
        if (gm.data[y * W + x] != 100) continue;
        for (int dy = -ic; dy <= ic; ++dy) for (int dx = -ic; dx <= ic; ++dx) {
          if (dx * dx + dy * dy > ic2) continue;
          int nx = x + dx, ny = y + dy;
          if (nx >= 0 && nx < W && ny >= 0 && ny < W) inf[ny * W + nx] = 100;
        }
      }
      gm.data = std::move(inf);
    }
    og_pub_->publish(gm);
  }

  void onGetRegion(const scovox_msgs::srv::GetRegion::Request::SharedPtr rq,
                   scovox_msgs::srv::GetRegion::Response::SharedPtr rs)
  {
    // fused_ is kept current incrementally — just take the read lock.
    std::shared_lock<std::shared_mutex> lk(mu_);
    auto& m = rs->map;
    m.header.stamp = get_clock()->now();
    m.header.frame_id = map_frame_;
    m.resolution = res_;
    m.occupancy_threshold = (float)min_occ_;
    m.semantic_threshold = (float)sem_gate_;
    m.max_semantic_classes = (uint8_t)top_k_;
    if (!fused_) return;
    auto& g = fused_->grid();
    auto mn = g.posToCoord(rq->min_corner.x, rq->min_corner.y, rq->min_corner.z);
    auto mx = g.posToCoord(rq->max_corner.x, rq->max_corner.y, rq->max_corner.z);
    g.forEachCell([&](const scovox::Voxel& v, const Bonxai::CoordT& c) {
      if (isPrior(v)) return;
      if (c.x < mn.x || c.x > mx.x || c.y < mn.y || c.y > mx.y || c.z < mn.z || c.z > mx.z) return;
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

  // ================================================================
  // Frontier centroids — lightweight publish
  // ================================================================

  // Caller must hold mu_ (shared).
  void publishFrontierCentroids() {
    if (!frontier_pub_ || !fused_) return;
    if (frontier_pub_->get_subscription_count() == 0) return;
    auto& g = fused_->grid();
    float fmin = (float)frontier_min_z_, fmax = (float)frontier_max_z_;

    // Phase 1: find frontier voxels (free cells with at least one unknown neighbor)
    std::vector<Eigen::Vector3f> frontier_cells;
    auto acc = g.createConstAccessor();
    static const Bonxai::CoordT offsets[6] = {
        {1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
    g.forEachCell([&](const scovox::Voxel& v, const Bonxai::CoordT& c) {
      if (isPrior(v)) return;
      if (v.p_occ() >= 0.5f) return;  // only free voxels
      auto p = g.coordToPos(c);
      if (p.z < fmin || p.z > fmax) return;
      for (const auto& off : offsets) {
        Bonxai::CoordT nb{c.x + off.x, c.y + off.y, c.z + off.z};
        if (!acc.value(nb)) {
          frontier_cells.emplace_back((float)p.x, (float)p.y, (float)p.z);
          return;
        }
      }
    });

    // Phase 2: cluster by coarse grid binning
    float inv_bin = 1.0f / (float)frontier_cluster_radius_;
    struct BinKey {
      int bx, by, bz;
      bool operator==(const BinKey& o) const {
        return bx == o.bx && by == o.by && bz == o.bz;
      }
    };
    struct BinHash {
      size_t operator()(const BinKey& k) const {
        return std::hash<int64_t>()(
            (int64_t(k.bx) * 73856093) ^
            (int64_t(k.by) * 19349663) ^
            (int64_t(k.bz) * 83492791));
      }
    };
    struct BinData { Eigen::Vector3f sum = Eigen::Vector3f::Zero(); int count = 0; };

    std::unordered_map<BinKey, BinData, BinHash> bins;
    for (const auto& p : frontier_cells) {
      BinKey key{(int)std::floor(p.x() * inv_bin),
                 (int)std::floor(p.y() * inv_bin),
                 (int)std::floor(p.z() * inv_bin)};
      auto& bd = bins[key];
      bd.sum += p;
      bd.count++;
    }

    geometry_msgs::msg::PoseArray msg;
    msg.header.stamp = get_clock()->now();
    msg.header.frame_id = map_frame_;
    msg.poses.reserve(bins.size());
    for (const auto& [key, bd] : bins) {
      Eigen::Vector3f centroid = bd.sum / (float)bd.count;
      geometry_msgs::msg::Pose pose;
      pose.position.x = centroid.x();
      pose.position.y = centroid.y();
      pose.position.z = centroid.z();
      pose.orientation.w = 1.0;
      msg.poses.push_back(pose);
    }
    frontier_pub_->publish(msg);
  }

  // ================================================================
  // ScoreCandidates service — FOV raycasting on fused grid
  // ================================================================

  void onScoreCandidates(
      const scovox_msgs::srv::ScoreCandidates::Request::SharedPtr rq,
      scovox_msgs::srv::ScoreCandidates::Response::SharedPtr rs)
  {
    std::shared_lock<std::shared_mutex> lk(mu_);
    const size_t n = rq->candidates.size();
    rs->scores.resize(n, 0.0f);

    if (!fused_) return;
    auto& g = fused_->grid();

    // Select scoring function
    enum ScoringMode { SM_EIG, SM_ENTROPY, SM_FRONTIER, SM_RANDOM, SM_SSMI };
    ScoringMode mode = SM_EIG;
    if (rq->scoring_mode == "entropy")       mode = SM_ENTROPY;
    else if (rq->scoring_mode == "frontier") mode = SM_FRONTIER;
    else if (rq->scoring_mode == "random")   mode = SM_RANDOM;
    else if (rq->scoring_mode == "ssmi")     mode = SM_SSMI;

    // Score candidates via FOV raycasting (skipped when n == 0; map
    // stats below still run so the random planner gets diagnostic data).
    if (n > 0) {

    const float hfov = rq->fov_hfov;
    const float vfov = rq->fov_vfov;
    const int h_rays = rq->fov_h_rays > 0 ? rq->fov_h_rays : 16;
    const int v_rays = rq->fov_v_rays > 0 ? rq->fov_v_rays : 12;
    const float max_range = rq->fov_max_range > 0.f ? rq->fov_max_range : 10.0f;
    const float occ_stop = rq->fov_occ_stop > 0.f ? rq->fov_occ_stop : 0.7f;

    std::vector<Eigen::Vector3f> ray_dirs;
    ray_dirs.reserve(h_rays * v_rays);
    const float h_step = hfov / (float)h_rays;
    const float v_step = vfov / (float)v_rays;
    const float h_start = -hfov * 0.5f + h_step * 0.5f;
    const float v_start = -vfov * 0.5f + v_step * 0.5f;
    for (int vi = 0; vi < v_rays; ++vi) {
      float pitch = v_start + (float)vi * v_step;
      float cp = std::cos(pitch), sp = std::sin(pitch);
      for (int hi = 0; hi < h_rays; ++hi) {
        float yaw = h_start + (float)hi * h_step;
        ray_dirs.emplace_back(cp * std::cos(yaw), cp * std::sin(yaw), sp);
      }
    }

    for (size_t ci = 0; ci < n; ++ci) {
      const auto& cand = rq->candidates[ci];
      Eigen::Vector3f pos(cand.position.x, cand.position.y, cand.position.z);
      float cy = std::cos(cand.yaw), sy = std::sin(cand.yaw);
      float total_score = 0.0f;

      auto acc = g.createConstAccessor();

      if (mode == SM_SSMI) {
        // SSMI-style MI lower bound with ray marginalisation.
        // Per-ray: weight each voxel's KL contribution by the probability
        // the ray reaches it, matching Asgharivaskasi & Atanasov (TRO 2023).
        for (const auto& dir : ray_dirs) {
          Eigen::Vector3f world_dir(
              cy * dir.x() - sy * dir.y(),
              sy * dir.x() + cy * dir.y(),
              dir.z());
          Eigen::Vector3f ray_end = pos + world_dir * max_range;
          auto c_origin = g.posToCoord(pos.x(), pos.y(), pos.z());
          auto c_end    = g.posToCoord(ray_end.x(), ray_end.y(), ray_end.z());

          float reach = 1.0f;
          float free_kl_acc = 0.0f;

          scovox::RayIterator(c_origin, c_end,
              [&](const Bonxai::CoordT& c) -> bool {
                const scovox::Voxel* ptr = acc.value(c);
                scovox::Voxel v;
                if (ptr && !isPrior(*ptr)) {
                  v = *ptr;
                } else {
                  v = scovox::defaultVoxel();
                }
                float p = v.p_occ();
                float kl_occ  = scovox::ssmiOccKL(v);
                float kl_free = scovox::ssmiFreeKL(v);

                total_score += reach * p * (kl_occ + free_kl_acc);
                free_kl_acc += kl_free;
                reach *= (1.0f - p);

                if (ptr && !isPrior(*ptr) && p >= occ_stop) return false;
                return true;
              });
          // "No hit" event: all voxels observed as free.
          total_score += reach * free_kl_acc;
        }
      } else {
        // Original per-voxel scoring (EIG / entropy / frontier / random).
        for (const auto& dir : ray_dirs) {
          Eigen::Vector3f world_dir(
              cy * dir.x() - sy * dir.y(),
              sy * dir.x() + cy * dir.y(),
              dir.z());
          Eigen::Vector3f ray_end = pos + world_dir * max_range;

          auto c_origin = g.posToCoord(pos.x(), pos.y(), pos.z());
          auto c_end    = g.posToCoord(ray_end.x(), ray_end.y(), ray_end.z());

          scovox::RayIterator(c_origin, c_end,
              [&](const Bonxai::CoordT& c) -> bool {
                const scovox::Voxel* ptr = acc.value(c);
                float s = 0.0f;
                if (ptr && !isPrior(*ptr)) {
                  switch (mode) {
                    case SM_EIG:      s = scovox::expectedInformationGain(*ptr); break;
                    case SM_ENTROPY: {
                      float p = ptr->p_occ();
                      s = (p > 1e-7f && p < 1.f - 1e-7f)
                          ? -p * std::log(p) - (1.f - p) * std::log(1.f - p)
                          : 0.f;
                      break;
                    }
                    case SM_FRONTIER: s = 0.0f; break;
                    case SM_RANDOM:   s = 0.0f; break;
                    case SM_SSMI:     break;  // handled above
                  }
                  total_score += s;
                  if (ptr->p_occ() >= occ_stop) return false;
                } else {
                  switch (mode) {
                    case SM_EIG: {
                      scovox::Voxel prior;
                      prior.a_occ = 1.0f; prior.a_free = 1.0f;
                      s = scovox::expectedInformationGain(prior);
                      break;
                    }
                    case SM_ENTROPY: {
                      s = -0.5f * std::log(0.5f) - 0.5f * std::log(0.5f);
                      break;
                    }
                    case SM_FRONTIER: s = 1.0f; break;
                    case SM_RANDOM:   s = 0.0f; break;
                    case SM_SSMI:     break;
                  }
                  total_score += s;
                }
                return true;
              });
        }
      }
      rs->scores[ci] = std::isfinite(total_score) ? total_score : 0.0f;
    }

    } // if (n > 0)

    // Compute map stats within the ROI bounding box
    auto mn = g.posToCoord(rq->min_corner.x, rq->min_corner.y, rq->min_corner.z);
    auto mx = g.posToCoord(rq->max_corner.x, rq->max_corner.y, rq->max_corner.z);
    uint32_t obs_count = 0;
    double sum_eig = 0.0, sum_ent = 0.0;
    uint32_t frontier_count = 0;
    auto stats_acc = g.createConstAccessor();
    static const Bonxai::CoordT nb_offsets[6] = {
        {1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
    g.forEachCell([&](const scovox::Voxel& v, const Bonxai::CoordT& c) {
      if (isPrior(v)) return;
      if (c.x < mn.x || c.x > mx.x || c.y < mn.y || c.y > mx.y ||
          c.z < mn.z || c.z > mx.z) return;
      obs_count++;
      sum_eig += scovox::expectedInformationGain(v);
      sum_ent += scovox::entropy(v);
      // Frontier check: free voxel with unknown neighbor
      if (v.p_occ() < 0.5f) {
        for (const auto& off : nb_offsets) {
          Bonxai::CoordT nb{c.x + off.x, c.y + off.y, c.z + off.z};
          if (!stats_acc.value(nb)) { frontier_count++; break; }
        }
      }
    });
    rs->stats.observed_voxels = obs_count;
    rs->stats.mean_eig = obs_count > 0 ? (float)(sum_eig / obs_count) : 0.0f;
    rs->stats.mean_entropy = obs_count > 0 ? (float)(sum_ent / obs_count) : 0.0f;
    rs->stats.frontier_voxels = frontier_count;
  }

  void onGetOccupancyGrid(const scovox_msgs::srv::GetOccupancyGrid::Request::SharedPtr rq,
                          scovox_msgs::srv::GetOccupancyGrid::Response::SharedPtr rs)
  {
    // fused_ is kept current incrementally — just take the read lock.
    std::shared_lock<std::shared_mutex> lk(mu_);
    if (!fused_) { rs->grid = nav_msgs::msg::OccupancyGrid(); return; }
    double r2 = (rq->resolution_2d > 0.0) ? rq->resolution_2d : (double)res_;
    float ot = (float)min_occ_;
    double xn = std::numeric_limits<double>::max(), xx = -xn, yn = xn, yx = -yn;
    std::unordered_map<int64_t, float> cells;
    fused_->grid().forEachCell([&](const scovox::Voxel& v, const Bonxai::CoordT& c) {
      if (isPrior(v)) return;
      auto p = fused_->grid().coordToPos(c);
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

  void initSemanticColors() { sem_col_ = scovox::generateSemanticColors(256); }

  // Members
  std::vector<std::string> input_topics_;
  std::string map_frame_;
  double min_occ_, sem_gate_;
  double kl_thresh_, tau_gate_, pub_hz_;
  double pc_min_interval_s_;
  rclcpp::Time last_pc_pub_time_;
  int top_k_;
  bool pub_plan_;
  double plan_res_, plan_sz_, plan_ox_, plan_oy_, plan_zmin_, plan_zmax_, plan_inf_;
  float res_{0.f};
  std::vector<std::array<float, 3>> sem_col_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  // One source grid per robot, keyed by header.frame_id of incoming binaries.
  std::unordered_map<std::string, SourceGrid> sources_;
  // Legacy v1 fused grid (use_split_=false). Null in split mode.
  std::unique_ptr<scovox::Map> fused_;
  // Step 7 (D9) split-grid v2 fused grid. Null in legacy mode. Holds the
  // consensus-merged SemBeta state; no fused TsdfMap on dscovox because
  // share_tsdf=false (the v2 default) means TSDF state never crosses the
  // wire — each robot keeps its own local TsdfMap for mesh extraction.
  std::unique_ptr<Bonxai::VoxelGrid<scovox::SemBetaVoxel>> split_fused_sem_;
  // Step 8 — v3 SemDir fused grid + pinned priors. Allocated lazily on the
  // first v3 frame; null while wire_format=v2 (or before any frame arrives).
  // fused_num_classes_ is 0 until pinned, which is the also the sentinel
  // "not yet seen" check in onBinaryMapV3.
  std::unique_ptr<Bonxai::VoxelGrid<scovox::SemDirVoxel>> split_fused_semdir_;
  uint16_t fused_num_classes_{0};
  float    fused_alpha_0_{scovox::kDefaultDirichletPrior};
  bool use_split_{false};
  bool share_tsdf_{false};
  bool wire_format_v3_{false};
  mutable std::shared_mutex mu_;
  std::vector<rclcpp::Subscription<scovox_msgs::msg::ScovoxMapBinary>::SharedPtr> subs_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pc_pub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr og_pub_;
  rclcpp::Publisher<scovox_msgs::msg::ScovoxMap>::SharedPtr sm_pub_;
  rclcpp::Service<scovox_msgs::srv::GetRegion>::SharedPtr get_region_srv_;
  rclcpp::Service<scovox_msgs::srv::GetOccupancyGrid>::SharedPtr get_occ_srv_;
  rclcpp::Service<scovox_msgs::srv::ScoreCandidates>::SharedPtr score_candidates_srv_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr frontier_pub_;
  double frontier_min_z_, frontier_max_z_, frontier_cluster_radius_;
  rclcpp::TimerBase::SharedPtr publish_timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DSCovoxNode>());
  rclcpp::shutdown();
}
