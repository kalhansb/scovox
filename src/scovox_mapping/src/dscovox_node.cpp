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
#include <scovox/binary_serializer_v4.hpp>
#include <scovox/consensus_merge_v4.hpp>
#include <scovox/lz4_codec.hpp>
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
#include <atomic>  // last_pc_pub_ns_ rate-limiter (shared-lock concurrent access)

namespace {

struct SourceGrid {
  std::string source_frame;                // header.frame_id of the binary
  // This source's contribution to the world, stored in MAP-FRAME coords.
  // Each entry is the latest snapshot of (occupancy, semantics) received for
  // that map-frame voxel from this robot.
  //
  // Split Beta/Dirichlet (v4 wire format) receiver populates these two grids:
  // occupancy (BetaVoxel) ∥ semantics (DirVoxel). No TsdfMap on the receiver
  // because share_tsdf=false is the v4 default — TSDF state never crosses the
  // wire to dscovox in the production path.
  std::unique_ptr<Bonxai::VoxelGrid<scovox::BetaVoxel>> beta_grid;
  std::unique_ptr<Bonxai::VoxelGrid<scovox::DirVoxel>>  dir_grid;
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

// Shared at-prior epsilon for the v4 receiver isPrior* tests. This MUST
// match the sender's at-prior emit gate (scovox_node.cpp uses + 1e-4f on the
// alpha_free / alpha_other / beta priors). If the receiver slop is looser than
// the sender gate (the old max(0.01, 0.01·α_0)), a barely-observed voxel that
// the sender deliberately put on the wire — e.g. a_free = α_0 + 0.005, which
// clears the sender's 1e-4 gate — is classified "at prior" here and silently
// dropped from the fused map. Keeping the two epsilons identical guarantees
// every emitted voxel survives the refold.
static constexpr float kPriorSlop = 1e-4f;

/// BetaVoxel "is at prior" check for the split (v4) consensus refold. A voxel
/// is at prior iff a_occ ≈ kBetaOccPrior and a_free ≈ kBetaFreePrior (the
/// symmetric Beta(1,1) occupancy prior; see docs/occupancy_prior.md). `slop`
/// matches isPriorDir's one-quantum tolerance.
inline bool isPriorBeta(const scovox::BetaVoxel& v,
                        uint16_t num_classes, float alpha_0) {
  (void)num_classes; (void)alpha_0;  // occupancy prior is the symmetric constant
  // Shipped occupancy prior is symmetric Beta(1,1) (kBetaOccPrior=kBetaFreePrior
  // =1, p_occ=0.5) — decoupled from (num_classes, α₀); see docs/occupancy_prior.md.
  // slop = kPriorSlop (1e-4) matches the v4 sender's at-prior emit gate so a
  // barely-observed Beta voxel the sender put on the wire is not dropped on refold.
  const float slop = kPriorSlop;
  return v.a_occ <= scovox::kBetaOccPrior + slop &&
         v.a_free <= scovox::kBetaFreePrior + slop;
}

/// DirVoxel "is at prior" check: OTHER ≈ (C−K)·α_0 and no slot filled.
inline bool isPriorDir(const scovox::DirVoxel& v,
                       uint16_t num_classes, float alpha_0) {
  // Clamp residual_dims at 0 to match defaultDirVoxel: for num_classes <= K_TOP
  // the OTHER prior is 0, not (C-K)*alpha_0 < 0. A negative other_prior would
  // make `v.other > other_prior + slop` true for genuine prior voxels and so
  // misclassify them as observed.
  const int residual_dims = static_cast<int>(num_classes) - scovox::K_TOP;
  const float other_prior =
      (residual_dims > 0) ? (static_cast<float>(residual_dims) * alpha_0) : 0.f;
  // Match the v4 sender's at-prior emit gate (kPriorSlop = 1e-4). See kPriorSlop.
  const float slop = kPriorSlop;
  if (v.other > other_prior + slop) return false;
  for (int i = 0; i < scovox::K_TOP; ++i)
    if (v.cls[i] != 0xFFFF) return false;
  return true;
}

/// Project split Beta(occupancy) + Dir(semantics) → SemBetaVoxel for the v4
/// visualisation path (reuses the shared argmax/variance/EIG helpers). The
/// Dir pointer may be null (occupancy-only voxel). Per-class evidence has the
/// α_0 prior subtracted so empty slots read 0, matching the sender's viz.
inline scovox::SemBetaVoxel projectBetaDirToSemBetaForViz(
    const scovox::BetaVoxel& b, const scovox::DirVoxel* d, float alpha_0,
    uint16_t num_classes) {
  scovox::SemBetaVoxel out{};
  out.a_occ  = b.a_occ;
  out.a_free = b.a_free;
  if (d) {
    // RAW-evidence convention: subtract the OTHER bucket's (C-K)*alpha_0 prior
    // (clamped at 0 for C<=K_TOP, matching defaultDirVoxel) just as the RPC
    // projectBetaDirToVoxel does. argmaxClassConfidence / effectiveResidual /
    // semanticVariance all assume a_unk holds raw evicted mass with no prior;
    // leaving the prior in (out.a_unk = d->other) inflated the confidence
    // denominator by (C-K)*alpha_0 and made the published semantic_confidence
    // disagree with the GetRegion RPC for the identical voxel.
    const int residual_dims = static_cast<int>(num_classes) - scovox::K_TOP;
    const float other_prior =
        (residual_dims > 0) ? (static_cast<float>(residual_dims) * alpha_0) : 0.f;
    out.a_unk = std::max(0.f, d->other - other_prior);
    for (int i = 0; i < scovox::K_TOP; ++i) {
      out.sem_cnt[i] = std::max(0.f, d->cnt[i] - alpha_0);
      out.sem_cls[i] = d->cls[i];
    }
  } else {
    out.a_unk = 0.f;
    for (int i = 0; i < scovox::K_TOP; ++i) { out.sem_cnt[i] = 0.f; out.sem_cls[i] = 0xFFFF; }
  }
  return out;
}

/// Project split Beta(occupancy) + Dir(semantics) → the legacy fused
/// scovox::Voxel for the v4 RPC query services (GetRegion / ScoreCandidates /
/// GetOccupancyGrid). The Dir pointer may be null (occupancy-only voxel, or a
/// caller that only needs occupancy — EIG/entropy/SSMI are occupancy-only).
///
/// scovox::Voxel stores RAW semantic evidence (the Dirichlet prior is applied
/// at query time, not in storage), whereas DirVoxel stores prior-inflated
/// counts. So the projection subtracts the per-class α_0 from each tracked
/// slot and the OTHER bucket's (C−K)·α_0 prior, yielding the same raw-evidence
/// convention selectTopKSemantics / argmaxClassConfidence expect — the SEMANTIC
/// fields are byte-identical to what the v1 fused path produced. Empty Dir slots
/// (cnt == α_0) collapse to sem_cnt == 0 and are skipped by every consumer's
/// `sem_cnt > 0` test.
///
/// OCCUPANCY now uses the SAME prior as v1: a_occ / a_free are copied verbatim
/// from the BetaVoxel, which ships the symmetric Beta(1,1) prior
/// (a_occ = a_free = 1.0 → prior p_occ = 0.5) — identical to the v1 fused Voxel
/// (defaultVoxel). So there is no longer a prior-induced p_occ / variance / EIG /
/// SSMI gap vs v1 at the prior. (Historically the split path used a calibrated
/// Beta(C·α_0, α_0) prior, p_occ = C/(C+1) ≈ 0.933; that was switched to
/// Beta(1,1) — see docs/occupancy_prior.md.)
inline scovox::Voxel projectBetaDirToVoxel(
    const scovox::BetaVoxel& b, const scovox::DirVoxel* d,
    uint16_t num_classes, float alpha_0) {
  scovox::Voxel out{};            // zero-init: a_unk / sem_cnt / sem_cls / tsdf = 0
  out.a_occ  = b.a_occ;
  out.a_free = b.a_free;
  if (d) {
    // Clamp residual_dims at 0 to mirror defaultDirVoxel: when num_classes <=
    // K_TOP there are no residual classes, so the OTHER prior is 0 (defaultDir
    // stored other=0). An unclamped (C-K)*alpha_0 < 0 would make the subtraction
    // d->other - other_prior = d->other + |prior| ADD a phantom alpha_0 of
    // unknown mass, skewing every projected voxel's entropy/EIG/argmax.
    const int residual_dims = static_cast<int>(num_classes) - scovox::K_TOP;
    const float other_prior =
        (residual_dims > 0) ? (static_cast<float>(residual_dims) * alpha_0) : 0.f;
    out.a_unk = std::max(0.f, d->other - other_prior);
    for (int i = 0; i < scovox::K_TOP; ++i) {
      out.sem_cnt[i] = std::max(0.f, d->cnt[i] - alpha_0);
      out.sem_cls[i] = d->cls[i];
    }
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

    // K_TOP is compile-time-locked into the v4 Dir record size; a sender
    // built with a different K_TOP will fail deserialization. Announce ours
    // up front so build-skew is diagnosable before frames flow.
    RCLCPP_INFO(get_logger(),
      "dscovox wire_format=v4 (split Beta/Dir): receiver compiled with "
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

    initSemanticColors();

    pc_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      declare_parameter<std::string>("pointcloud_topic", "~/pointcloud"),
      rclcpp::SystemDefaultsQoS());

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
          // v4 map size is keyed on occupancy (the fused Beta grid).
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

  // ScovoxMapBinary handler. The node is v4-only: every binary is a v4
  // (split Beta/Dir) envelope, so dispatch straight to the v4 receive path.
  void onBinaryMap(const scovox_msgs::msg::ScovoxMapBinary::SharedPtr msg) {
    onBinaryMapV4(msg);
  }

  // ==================================================================
  // Split-grid v4 receive path — the node's only receive path. Operates on
  // the de-unified BetaVoxel (occupancy) ∥ DirVoxel (semantics) grids +
  // consensus_merge_v4.hpp. The two grids ingest + refold INDEPENDENTLY: a
  // touched coord may be in either or both. Priors are pinned from the first
  // frame's header.
  //
  // NOTE: the RPC query services (GetRegion / GetOccupancyGrid /
  // ScoreCandidates) project the split Beta(+Dir) grids into a transient
  // scovox::Voxel via a substrate-agnostic templated core. The SEMANTIC query
  // math uses the raw-evidence convention; the OCCUPANCY math uses the
  // symmetric Beta(1,1) prior (p_occ=0.5) — see projectBetaDirToVoxel /
  // docs/occupancy_prior.md. Occupancy-only services (ScoreCandidates /
  // GetOccupancyGrid) read just the Beta grid; GetRegion joins the Dir grid
  // for per-class evidence.
  // ==================================================================
  void onBinaryMapV4(const scovox_msgs::msg::ScovoxMapBinary::SharedPtr msg) {
    if (msg->version != 4) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "wire_format=v4 expects v4 envelope, got version %d (dropping)",
        msg->version);
      return;
    }
    const std::string sf = msg->header.frame_id;
    std::string buf = scovox::ScovoxBinarySerializer::decompressLZ4(msg->data);
    if (buf.empty()) {
      RCLCPP_ERROR(get_logger(), "LZ4 fail '%s'", sf.c_str());
      return;
    }

    scovox::BinarySerializerV4::Frame frame;
    try {
      frame = scovox::BinarySerializerV4::deserialize(buf);
    } catch (const std::exception& e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "v4 deserialize failed for '%s': %s", sf.c_str(), e.what());
      return;
    }
    if (frame.beta_deltas.empty() && frame.dir_deltas.empty()) return;

    // Resolve / cache the source->map TF (static; cached on first sight).
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
      // Reject num_classes==0 outright (invalid Dirichlet dimension): pinning it
      // would re-pin/re-log every frame and bypass the mismatch guard, letting a
      // second num_classes==0 source with a different alpha_0 fuse unchecked.
      if (frame.num_classes == 0) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
          "v4 frame from '%s' has num_classes=0 (invalid) — dropping", sf.c_str());
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
          "v4 receive: pinned num_classes=%u alpha_0=%.4f (from first frame, src='%s')",
          (unsigned)fused_num_classes_, fused_alpha_0_, sf.c_str());
      } else {
        if (frame.num_classes != fused_num_classes_ ||
            std::abs(frame.alpha_0 - fused_alpha_0_) > 1e-6f) {
          RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
            "v4 prior mismatch from '%s' (got C=%u α=%.4f, pinned C=%u α=%.4f) — dropping frame",
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
        sg.tf_cached = true;
        scovox::Params P; P.resolution = res_;
        sg.beta_grid = std::make_unique<Bonxai::VoxelGrid<scovox::BetaVoxel>>(
            P.resolution, P.inner_bits, P.leaf_bits);
        sg.dir_grid  = std::make_unique<Bonxai::VoxelGrid<scovox::DirVoxel>>(
            P.resolution, P.inner_bits, P.leaf_bits);
        it = sources_.emplace(sf, std::move(sg)).first;
      } else if (!it->second.tf_cached) {
        it->second.T_map_source = Tmo;
        it->second.tf_cached = true;
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
      // the bulk of the source voxel's volume even when the TF is unaligned).
      std::unordered_set<Bonxai::CoordT, CoordTHash, CoordTEqual> touched_beta, touched_dir;
      touched_beta.reserve(frame.beta_deltas.size());
      touched_dir.reserve(frame.dir_deltas.size());
      const Eigen::Isometry3d Te = src.T_map_source;
      const double half_src_res = 0.5 * double(src_res);
      auto toMapCoord = [&](const Bonxai::CoordT& sc, auto& grid_ptr) {
        Eigen::Vector3d sp(
          double(sc.x) * double(src_res) + half_src_res,
          double(sc.y) * double(src_res) + half_src_res,
          double(sc.z) * double(src_res) + half_src_res);
        Eigen::Vector3d mp = Te * sp;
        return grid_ptr->posToCoord(mp.x(), mp.y(), mp.z());
      };

      {
        auto ba = src.beta_grid->createAccessor();
        for (auto& d : frame.beta_deltas) {
          auto mc = toMapCoord(d.coord, src.beta_grid);
          auto* v = ba.value(mc, true);
          if (!v) continue;
          *v = d.data;     // snapshot-replace
          touched_beta.insert(mc);
        }
      }
      {
        auto da = src.dir_grid->createAccessor();
        for (auto& d : frame.dir_deltas) {
          auto mc = toMapCoord(d.coord, src.dir_grid);
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
    }  // unique_lock released

    // Step 3 — visualisation publish.
    {
      std::shared_lock<std::shared_mutex> rlk(mu_);
      maybePublishPointCloud();
    }
  }

  // BetaVoxel-typed refold. Reset fused[mc] to the symmetric Beta(1,1) occupancy
  // prior, then fold every source's value via mergeBeta (conjugate Beta consensus).
  void refoldCellBeta(
      const Bonxai::CoordT& mc,
      Bonxai::VoxelGrid<scovox::BetaVoxel>::Accessor& fa,
      std::vector<Bonxai::VoxelGrid<scovox::BetaVoxel>::Accessor>& source_accs)
  {
    auto* fv = fa.value(mc, true);
    if (!fv) return;
    // Reset to the shipped symmetric Beta(1,1) occupancy prior (p_occ=0.5);
    // must match mergeBeta + isPriorBeta. See docs/occupancy_prior.md.
    *fv = scovox::defaultBetaVoxel(scovox::kBetaOccPrior, scovox::kBetaFreePrior);
    bool seeded = false;
    for (auto& sa : source_accs) {
      auto* sv = sa.value(mc, false);
      if (!sv || isPriorBeta(*sv, fused_num_classes_, fused_alpha_0_)) continue;
      if (!seeded) { *fv = *sv; seeded = true; }
      else         { *fv = scovox::mergeBeta(*fv, *sv,
                                             fused_num_classes_, fused_alpha_0_); }
    }
  }

  // DirVoxel-typed refold. Reset fused[mc] to the symmetric Dirichlet prior,
  // then fold every source's value via mergeDir (slot-reconciling consensus).
  void refoldCellDir(
      const Bonxai::CoordT& mc,
      Bonxai::VoxelGrid<scovox::DirVoxel>::Accessor& fa,
      std::vector<Bonxai::VoxelGrid<scovox::DirVoxel>::Accessor>& source_accs)
  {
    auto* fv = fa.value(mc, true);
    if (!fv) return;
    *fv = scovox::defaultDirVoxel(fused_num_classes_, fused_alpha_0_);
    bool seeded = false;
    for (auto& sa : source_accs) {
      auto* sv = sa.value(mc, false);
      if (!sv || isPriorDir(*sv, fused_num_classes_, fused_alpha_0_)) continue;
      if (!seeded) { *fv = *sv; seeded = true; }
      else         { *fv = scovox::mergeDir(*fv, *sv,
                                            fused_num_classes_, fused_alpha_0_); }
    }
  }

  // Rate-limited visualization publish. Called from the binary callback's
  // tail (so the user-visible map updates as soon as ingest produces fresh
  // data) and from the timer as a fallback (so the map keeps refreshing in
  // RViz even if no binaries arrive). publishPointCloudV4 already returns
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
    publishPointCloudV4();
  }

  // v4 split-substrate visualisation publisher. Walks the fused Beta grid
  // (occupancy) and joins the fused Dir grid (semantics) at each coord,
  // projecting to a transient SemBetaVoxel for the 11-field PointCloud2 schema.
  void publishPointCloudV4() {
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
      *ir = *reinterpret_cast<float*>(&rp);
      *iv = scovox::variance(v);
      *ie = scovox::expectedInformationGain(v);
      *iao = v.a_occ; *iaf = v.a_free;
      ++ix; ++iy; ++iz; ++ir; ++ip; ++ik; ++ic; ++iv; ++ie; ++iao; ++iaf;
    });
    pc_pub_->publish(cl);
  }

  // GetRegion core. Walks the Bonxai grid whose cell type projects to a
  // scovox::Voxel via `project(cell, coord) -> scovox::Voxel` (v4: join the Dir
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

  void onGetRegion(const scovox_msgs::srv::GetRegion::Request::SharedPtr rq,
                   scovox_msgs::srv::GetRegion::Response::SharedPtr rs)
  {
    // split_fused_* (v4) is kept current incrementally — read lock.
    std::shared_lock<std::shared_mutex> lk(mu_);
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

  // ================================================================
  // ScoreCandidates service — FOV raycasting on the fused Beta grid
  // ================================================================

  // Substrate-agnostic ScoreCandidates core (FOV raycast scoring + ROI map
  // stats). Operates on any Bonxai grid whose cell type projects to a
  // scovox::Voxel via `project(cell) -> scovox::Voxel`. All scoring signals
  // (EIG / entropy / SSMI KL / frontier) are occupancy-only, so the projection
  // may leave semantics empty; v4 passes the Beta grid with a Dir-free
  // projection. One body ⇒ the raycast math can never drift between substrates.
  //
  // `unobserved_prior` is the scovox::Voxel used for cells a ray traverses that
  // are NOT allocated/observed. It MUST carry the SAME occupancy prior the
  // substrate uses for its observed cells, otherwise the SSMI reach *= (1-p)
  // accumulation and KL terms mix two occupancy priors and skew candidate
  // ranking. Both v1 and v4 now ship the symmetric Beta(1,1) prior (p_occ=0.5)
  // — v1 via defaultVoxel(), v4 via defaultBetaVoxel(kBetaOccPrior,
  // kBetaFreePrior); see docs/occupancy_prior.md.
  template <typename CellT, typename ProjectFn, typename IsPriorFn>
  void scoreCandidatesOnGrid(
      const scovox_msgs::srv::ScoreCandidates::Request::SharedPtr rq,
      scovox_msgs::srv::ScoreCandidates::Response::SharedPtr rs,
      Bonxai::VoxelGrid<CellT>& g, ProjectFn project, IsPriorFn isPriorCell,
      const scovox::Voxel& unobserved_prior)
  {
    const size_t n = rq->candidates.size();

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
                const CellT* ptr = acc.value(c);
                const bool obs = ptr && !isPriorCell(*ptr);
                // Use the substrate's own unobserved prior so the reach/KL
                // accumulation stays on one consistent occupancy prior across
                // observed and unobserved cells. Both substrates ship the
                // symmetric Beta(1,1) prior (p_occ=0.5); see docs/occupancy_prior.md.
                scovox::Voxel v = obs ? project(*ptr) : unobserved_prior;
                float p = v.p_occ();
                float kl_occ  = scovox::ssmiOccKL(v);
                float kl_free = scovox::ssmiFreeKL(v);

                // Guard each KL term so one non-finite voxel (e.g. digamma
                // underflow on a near-point-mass cell) drops only that term
                // instead of poisoning the whole candidate's accumulated score
                // (which the post-hoc isfinite(total_score) below would then
                // zero entirely). Mirrors the map-stats per-term isfinite skip.
                const float term = reach * p * (kl_occ + free_kl_acc);
                if (std::isfinite(term)) total_score += term;
                if (std::isfinite(kl_free)) free_kl_acc += kl_free;
                reach *= (1.0f - p);

                if (obs && p >= occ_stop) return false;
                return true;
              });
          // "No hit" event: all voxels observed as free.
          const float no_hit = reach * free_kl_acc;
          if (std::isfinite(no_hit)) total_score += no_hit;
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
                const CellT* ptr = acc.value(c);
                float s = 0.0f;
                if (ptr && !isPriorCell(*ptr)) {
                  const scovox::Voxel v = project(*ptr);
                  switch (mode) {
                    case SM_EIG:      s = scovox::expectedInformationGain(v); break;
                    case SM_ENTROPY: {
                      float p = v.p_occ();
                      s = (p > 1e-7f && p < 1.f - 1e-7f)
                          ? -p * std::log(p) - (1.f - p) * std::log(1.f - p)
                          : 0.f;
                      break;
                    }
                    case SM_FRONTIER: s = 0.0f; break;
                    case SM_RANDOM:   s = 0.0f; break;
                    case SM_SSMI:     break;  // handled above
                  }
                  // Per-term isfinite guard: a single non-finite EIG (e.g. a
                  // projected Beta voxel whose digamma args underflow) must drop
                  // only this voxel, not collapse the candidate's whole score to
                  // 0 via the post-hoc isfinite(total_score) check below.
                  if (std::isfinite(s)) total_score += s;
                  if (v.p_occ() >= occ_stop) return false;
                } else {
                  // Unobserved cell: score against the substrate's occupancy
                  // prior (`unobserved_prior`), NOT a hardcoded Beta(1,1). This
                  // matches the SSMI branch above and a freshly-allocated
                  // observed cell, so EIG/entropy stay on ONE consistent
                  // occupancy prior even if kBetaOccPrior is retuned (e.g. to the
                  // Jeffreys Beta(0.5,0.5) runner-up). With today's Beta(1,1) the
                  // result is identical to the previous hardcoded p=0.5.
                  switch (mode) {
                    case SM_EIG:
                      s = scovox::expectedInformationGain(unobserved_prior);
                      break;
                    case SM_ENTROPY: {
                      const float p = unobserved_prior.p_occ();
                      s = (p > 1e-7f && p < 1.f - 1e-7f)
                          ? -p * std::log(p) - (1.f - p) * std::log(1.f - p)
                          : 0.f;
                      break;
                    }
                    case SM_FRONTIER: s = 1.0f; break;
                    case SM_RANDOM:   s = 0.0f; break;
                    case SM_SSMI:     break;
                  }
                  // Same per-term isfinite guard as the observed branch.
                  if (std::isfinite(s)) total_score += s;
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
    // finite_eig_count tracks only the voxels actually summed into sum_eig.
    // EIG terms that are non-finite are skipped (below) but obs_count still
    // increments for them; dividing sum_eig by obs_count would bias mean_eig
    // toward zero. Divide by finite_eig_count so the mean reflects the summed
    // voxels. (sum_ent is always finite — the entropy term clamps to 0 outside
    // the open (0,1) range — so it keeps dividing by obs_count.)
    uint32_t finite_eig_count = 0;
    double sum_eig = 0.0, sum_ent = 0.0;
    uint32_t frontier_count = 0;
    auto stats_acc = g.createConstAccessor();
    static const Bonxai::CoordT nb_offsets[6] = {
        {1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
    g.forEachCell([&](const CellT& cell, const Bonxai::CoordT& c) {
      if (isPriorCell(cell)) return;
      if (c.x < mn.x || c.x > mx.x || c.y < mn.y || c.y > mx.y ||
          c.z < mn.z || c.z > mx.z) return;
      const scovox::Voxel v = project(cell);
      obs_count++;
      const float eig = scovox::expectedInformationGain(v);
      if (std::isfinite(eig)) { sum_eig += eig; ++finite_eig_count; }
      // mean_entropy is documented (MapStats.msg) as "Mean Shannon entropy":
      // the BOUNDED Bernoulli occupancy entropy H(p_occ) ∈ [0, ln2], identical
      // to the SM_ENTROPY scoring mode above. scovox::entropy() is instead the
      // Beta *differential* entropy (unbounded below); on the split substrate's
      // near-point-mass occupied voxels (a_free pinned at α_0) it diverges to
      // absurd negatives (≈ −1e28, formerly −inf), poisoning the mean. Compute
      // the documented Shannon entropy directly so the stat stays finite and
      // meaningful: MapStats.mean_entropy reports bounded Bernoulli entropy
      // ∈ [0, ln2], matching the MapStats.msg contract.
      const float p = v.p_occ();
      sum_ent += (p > 1e-7f && p < 1.f - 1e-7f)
          ? -p * std::log(p) - (1.f - p) * std::log(1.f - p) : 0.f;
      // Frontier check: free voxel with unknown neighbor
      if (v.p_occ() < 0.5f) {
        for (const auto& off : nb_offsets) {
          Bonxai::CoordT nb{c.x + off.x, c.y + off.y, c.z + off.z};
          if (!stats_acc.value(nb)) { frontier_count++; break; }
        }
      }
    });
    rs->stats.observed_voxels = obs_count;
    rs->stats.mean_eig = finite_eig_count > 0 ? (float)(sum_eig / finite_eig_count) : 0.0f;
    rs->stats.mean_entropy = obs_count > 0 ? (float)(sum_ent / obs_count) : 0.0f;
    rs->stats.frontier_voxels = frontier_count;
  }

  void onScoreCandidates(
      const scovox_msgs::srv::ScoreCandidates::Request::SharedPtr rq,
      scovox_msgs::srv::ScoreCandidates::Response::SharedPtr rs)
  {
    std::shared_lock<std::shared_mutex> lk(mu_);
    const size_t n = rq->candidates.size();
    rs->scores.resize(n, 0.0f);

    if (!split_fused_beta_) return;
    const uint16_t C = fused_num_classes_;
    const float a0 = fused_alpha_0_;
    // Scoring is occupancy-only ⇒ a Dir-free projection (semantics stay empty).
    // The unobserved-cell prior is the SHIPPED split Beta prior — symmetric
    // Beta(1,1), p_occ=0.5 — so SSMI reach/KL on unallocated v4 cells uses the
    // SAME prior as freshly-allocated observed cells (the key invariant: the
    // unobserved baseline must equal the allocation prior). See
    // docs/occupancy_prior.md.
    const scovox::Voxel v4_unobserved = projectBetaDirToVoxel(
        scovox::defaultBetaVoxel(scovox::kBetaOccPrior, scovox::kBetaFreePrior),
        nullptr, C, a0);
    scoreCandidatesOnGrid<scovox::BetaVoxel>(
        rq, rs, *split_fused_beta_,
        [C, a0](const scovox::BetaVoxel& b) { return projectBetaDirToVoxel(b, nullptr, C, a0); },
        [this](const scovox::BetaVoxel& b) { return isPriorBeta(b, fused_num_classes_, fused_alpha_0_); },
        v4_unobserved);
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
    // split_fused_beta_ (v4) is kept current incrementally — read lock.
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
  double pc_min_interval_s_;
  // Rate-limiter timestamp for the visualisation pointcloud, stored as raw
  // nanoseconds in a std::atomic. maybePublishPointCloud() runs under only a
  // SHARED lock (from both the binary-callback tail and the publish timer), so
  // two readers can execute concurrently; a plain rclcpp::Time would be torn /
  // race-written here. The atomic + compare_exchange below makes the
  // read-decide-update a single race-free claim so exactly one caller publishes
  // per window even under a multi-threaded executor.
  std::atomic<int64_t> last_pc_pub_ns_{0};
  int top_k_;
  float res_{0.f};
  std::vector<std::array<float, 3>> sem_col_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  // One source grid per robot, keyed by header.frame_id of incoming binaries.
  std::unordered_map<std::string, SourceGrid> sources_;
  // Split Beta/Dirichlet (v4) fused grids. Allocated lazily on the first v4
  // frame; null otherwise. Occupancy ∥ semantics, merged independently
  // (consensus_merge_v4.hpp). Share the pinned (fused_num_classes_, fused_alpha_0_).
  std::unique_ptr<Bonxai::VoxelGrid<scovox::BetaVoxel>> split_fused_beta_;
  std::unique_ptr<Bonxai::VoxelGrid<scovox::DirVoxel>>  split_fused_dir_;
  uint16_t fused_num_classes_{0};
  float    fused_alpha_0_{scovox::kDefaultDirichletPrior};
  // Dedicated "prior has been pinned" flag for the v4 receive path. We must
  // NOT overload fused_num_classes_==0 as the "not yet pinned" sentinel: a
  // sender that ships num_classes==0 (misconfig/upstream bug) would store 0 and
  // re-pin/re-log every frame, and the cross-prior mismatch guard would never
  // run (so a second num_classes==0 source with a different alpha_0 would fuse
  // without rejection). num_classes==0 frames are rejected explicitly in the path.
  bool     prior_pinned_{false};
  mutable std::shared_mutex mu_;
  std::vector<rclcpp::Subscription<scovox_msgs::msg::ScovoxMapBinary>::SharedPtr> subs_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pc_pub_;
  rclcpp::Service<scovox_msgs::srv::GetRegion>::SharedPtr get_region_srv_;
  rclcpp::Service<scovox_msgs::srv::GetOccupancyGrid>::SharedPtr get_occ_srv_;
  rclcpp::Service<scovox_msgs::srv::ScoreCandidates>::SharedPtr score_candidates_srv_;
  rclcpp::TimerBase::SharedPtr publish_timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DSCovoxNode>());
  rclcpp::shutdown();
}
