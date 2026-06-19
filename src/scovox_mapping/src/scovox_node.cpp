#include <chrono>
#include <cstdio>
#include <fstream>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2/time.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <scovox/uncertainty.hpp>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include "scovox/scovoxmap.hpp"
#include "scovox/scovox_map_split.hpp"
#include "scovox/node_utils.hpp"
#include "scovox_msgs/msg/scovox_map.hpp"
#include "scovox_msgs/msg/scovox_voxel.hpp"
#include "scovox_msgs/msg/scovox_semantic_evidence.hpp"
#include "scovox_msgs/msg/scovox_map_binary.hpp"
#include "scovox/binary_serializer.hpp"
#include "scovox/binary_serializer_v2.hpp"
#include "scovox/binary_serializer_v3.hpp"
#include "scovox/marching_cubes.hpp"
#include "scovox/mesh_labelling.hpp"
#include "scovox_msgs/srv/extract_mesh.hpp"

namespace enc = sensor_msgs::image_encodings;

namespace {
/// SemDir → SemBeta projection for the v2-shaped wire / publishPointCloudV2
/// fields. The composer's substrate is now SemDirVoxel (unified Dirichlet,
/// Step 7.5) but the v2 BinarySerializerV2 wire format + the per-voxel
/// PointCloud2 field schema (a_occ, a_free, a_unk, sem_cnt0/cls0/cnt1/cls1)
/// still expect the SemBeta layout. The projection is lossless on p_occ
/// (`s_occ()` collapses into a_occ, alpha_free → a_free, alpha_other → a_unk)
/// and on per-class accumulated evidence (cnt[i] − α_0 → sem_cnt[i]).
/// Empty slots (cls[i] == 0xFFFF) project to SemBeta's "empty slot" with
/// sem_cnt = 0.
/// **DEPRECATED — retire after v3 wire format is the default**.  Step 8
/// landed the v3 sender (publishBinaryMapV3) and the dscovox v3 receiver
/// (onBinaryMapV3 in dscovox_node.cpp).  v3 carries SemDirVoxel as-is at
/// 32 B/voxel vs the v2 SemBeta block's 37 B/voxel, and the receiver
/// rebuilds the symmetric Dirichlet prior from the wire header rather
/// than the sender's launch params.
///
/// Currently still alive because:
///   1. wire_format launch arg defaults to "v2" for backward compat
///      (existing dscovox receivers, NPZ analysis tooling, …)
///   2. publishPointCloudV2 / publishTSDFPointCloudV2 / NPZ-dump paths
///      still call this on a hot path to bridge SemDir → SemBeta-shaped
///      visualisation fields (a_occ, a_free, a_unk, sem_cnt0/cls0/…).
/// Removal plan: (a) flip wire_format default to v3 once the 8-scene
/// E2.1 fusion batch confirms parity with the v2 numbers; (b) drop
/// publishBinaryMapV2 + projectSemDirToSemBeta; (c) replace the
/// publishPointCloudV2 SemBeta-field walker with a SemDir-native
/// publisher that calls SemDirVoxel-typed variance/EIG helpers (the
/// same uncertainty.hpp cleanup the dscovox v3 publisher already needs).
inline scovox::SemBetaVoxel projectSemDirToSemBeta(const scovox::SemDirVoxel& v) {
  scovox::SemBetaVoxel p{};
  p.a_occ  = v.s_occ();
  p.a_free = v.alpha_free;
  p.a_unk  = v.alpha_other;
  for (int i = 0; i < scovox::K_TOP; ++i) {
    if (v.cls[i] == 0xFFFF) {
      p.sem_cnt[i] = 0.f;
      p.sem_cls[i] = 0xFFFF;
    } else {
      // Subtract the per-slot α_0 prior so the SemBeta projection sees
      // observed evidence only (matching legacy sem_cnt semantics where
      // empty slots had cnt=0 and filled slots had cnt=Σ_observations).
      p.sem_cnt[i] = std::max(0.f, v.cnt[i] - scovox::kDefaultDirichletPrior);
      p.sem_cls[i] = v.cls[i];
    }
  }
  return p;
}
}  // namespace

class SCovoxNode : public rclcpp::Node {
public:
  SCovoxNode() : Node("scovox_node"), tf_buffer_(this->get_clock(), tf2::Duration(std::chrono::seconds(600))), tf_listener_(tf_buffer_) {
    auto P = declareMapParams();
    // Always allocate the legacy `map_` so its params() are available to
    // node-level sensor filters (range_decay_length, min_range, max_range,
    // grazing_angle_threshold, …) regardless of mode. In split mode no rays
    // are integrated into it; the empty grid + params block costs only a
    // few hundred bytes of overhead.
    map_ = std::make_unique<scovox::Map>(P);
    declareNodeParams();   // sets use_split_ / share_tsdf_ before substrate creation
    if (use_split_ && P.sdf_trunc == 0.f) {
      RCLCPP_WARN(get_logger(),
        "TSDF off (enable_tsdf=false or sdf_trunc_voxels=0) has no clean effect in split "
        "mode (use_split=true): TsdfMap clamps sdf_trunc back up so it keeps integrating a "
        "surface, yet ~/tsdf_pointcloud is suppressed (the publisher is gated on the legacy "
        "map's sdf_trunc=0). Use the legacy fused path (use_split=false) for TSDF-free mapping.");
    }
    if (use_split_) {
      // Step 8 (D7) — split-grid v2 substrate. ScovoxMapSplit::Params shares
      // (resolution, inner_bits, leaf_bits) across both grids; per-substrate
      // params (sdf_trunc, w_occ/w_free/kappa0, evidence_saturation, …) flow
      // through from launch params via TsdfMap::Params and SemBetaMap::Params.
      scovox::ScovoxMapSplit::Params SP;
      SP.resolution = P.resolution;
      SP.inner_bits = P.inner_bits;
      SP.leaf_bits  = P.leaf_bits;
      // TsdfMap: SLIM-VDB-equivalent defaults; only sdf_trunc + space_carving
      // carry over from P. weighting_function defaults to constant(1).
      SP.tsdf.sdf_trunc     = P.sdf_trunc;
      SP.tsdf.space_carving = P.tsdf_space_carving;
      // SemDirMap (Step 7.5: unified Dirichlet, supersedes SemBetaMap as the
      // composer's semantic substrate): every Bayesian / sparse-Dirichlet knob
      // from legacy P maps 1:1. semantic_occ_gate / min_range / max_range /
      // grazing_angle_threshold are node-level sensor filters consumed
      // BEFORE integrateHit so they are not mirrored. evidence_saturation
      // widens uint16→float.
      SP.semdir.w_free                  = P.w_free;
      SP.semdir.w_occ                   = P.w_occ;
      SP.semdir.kappa0                  = P.kappa0;
      SP.semdir.carve_skip_occ_threshold = P.carve_skip_occ_threshold;
      SP.semdir.evidence_saturation     = static_cast<float>(P.evidence_saturation);
      SP.semdir.dirichlet_min_p_occ     = P.dirichlet_min_p_occ;
      SP.semdir.range_decay_length      = static_cast<float>(P.range_decay_length);
      SP.semdir.semantic_mode           = P.semantic_mode;
      // num_classes / alpha_0 — dataset-dependent priors that govern the
      // OTHER bucket's prior mass `(C − K_TOP)·α_0` and the FREE prior
      // `α_0`. Wrong num_classes shifts p_occ_prior and the eviction
      // capacity (KITTI=20 vs NYU13=14 gives p_occ_prior=0.952 vs 0.933);
      // post-SemDir KITTI mIoU regression analysis 2026-05-14 traced
      // ~0.005 of a 0.025 gap to C=14 leaking into seq08. Defaults match
      // SemDirMap::Params defaults (NYU13 / 0.01) so legacy launches still
      // boot. The two new launch params (declared in declareNodeParams)
      // are pure SemDir knobs — ignored on the legacy fused-Voxel path.
      SP.semdir.num_classes             = num_classes_;
      SP.semdir.alpha_0                 = alpha_0_;
      // Step 12.10 (2026-05-09) — fused single-DDA ray walker. Default
      // true; set false via `fused_walker:=false` launch arg for A/B
      // parity testing against the legacy two-DDA split path.
      SP.fused_walker = fused_walker_;
      split_map_ = std::make_unique<scovox::ScovoxMapSplit>(SP);
    }
    loadSemanticColorMap();  initializeSemanticColors();
    setupSubscribers();
    setupPublishers();
    if (pub_tsdf_ && map_->params().sdf_trunc > 0.f && !use_split_) {
      // ExtractMesh service walks the legacy fused grid; not yet ported to
      // ScovoxMapSplit. RViz mesh consumers in split mode can still get
      // the surface point cloud via ~/tsdf_pointcloud (publishTSDFPointCloudV2
      // does the cross-grid label join). Triangle-mesh extraction in split
      // mode is a follow-on (composer.extractMesh exists; just needs the
      // service plumbing once a paper figure or planner consumer demands it).
      extract_mesh_srv_ = create_service<scovox_msgs::srv::ExtractMesh>(
          "~/extract_mesh",
          std::bind(&SCovoxNode::onExtractMesh, this,
                    std::placeholders::_1, std::placeholders::_2));
    }
    double sm_rate = this->declare_parameter<double>("scovox_publish_rate", 1.0);
    sm_timer_ = rclcpp::create_timer(this, get_clock(), std::chrono::duration<double>(1.0/sm_rate),
      [this]{
        // Hold one shared_lock for the whole timer tick so the ScovoxMap and
        // PointCloud always represent the same map state, and so neither
        // helper races against scheduleMemUsage's detached reader thread.
        // SingleThreadedExecutor already serializes us against onImages, but
        // wrapping here makes the contract explicit and survives a future
        // switch to MultiThreadedExecutor. publishScovoxMap and
        // publishPointCloud must NOT take map_mtx_ themselves —
        // std::shared_mutex is non-recursive, so re-locking here would be UB.
        std::shared_lock<std::shared_mutex> lock(map_mtx_);
        publishScovoxMap();
        if (pub_pc_) publishPointCloud();
        if (tsdf_pub_) publishTSDFPointCloud();
      });
    RCLCPP_INFO(get_logger(), "SCovox ready res=%.3f mode=%s frame=%s use_split=%d share_tsdf=%d fused_walker=%d",
      P.resolution, mode_.c_str(), int_frame_.c_str(), (int)use_split_, (int)share_tsdf_, (int)fused_walker_);
    RCLCPP_INFO(get_logger(), "grid leaf_bits=%u inner_bits=%u block=%ux%ux%u voxels_per_leaf=%u",
      map_->grid().leafBits(), map_->grid().innetBits(),
      1u << map_->grid().leafBits(), 1u << map_->grid().leafBits(), 1u << map_->grid().leafBits(),
      1u << (3u * map_->grid().leafBits()));
    RCLCPP_INFO(get_logger(), "sizeof(Mask)=%zu sizeof(LeafGrid)=%zu sizeof(Voxel)=%zu trivial=%d",
      sizeof(Bonxai::Mask), sizeof(scovox::Map::Grid::LeafGrid), sizeof(scovox::Voxel),
      (int)std::is_trivial_v<scovox::Voxel>);
    RCLCPP_INFO(get_logger(), "TSDF: sdf_trunc=%.3f m space_carving=%d band_only=%d (sdf_trunc=0 means off; set via enable_tsdf / sdf_trunc_voxels)",
      P.sdf_trunc, (int)P.tsdf_space_carving, (int)P.band_only_integration);
    if (use_split_) {
      RCLCPP_INFO(get_logger(), "split substrate: sizeof(TsdfVoxel)=%zu sizeof(SemDirVoxel)=%zu",
        sizeof(scovox::TsdfVoxel), sizeof(scovox::SemDirVoxel));
    }
  }
private:
  scovox::Params declareMapParams() {
    scovox::Params P;
    auto dp = [&](auto n, auto d){ return this->declare_parameter<decltype(d)>(n, d); };
    P.resolution = dp("resolution", 0.10);
    P.inner_bits = (uint8_t)std::clamp((int)dp("inner_bits", 2), 1, 4);
    P.leaf_bits  = (uint8_t)std::clamp((int)dp("leaf_bits", 3), 1, 4);
    P.w_free = dp("w_free", 1.0);  P.w_occ = dp("w_occ", 2.0);
    P.kappa0 = dp("kappa0", 2.0);
    {
      const int requested_top_k = (int)dp("semantic_top_k", (int)scovox::K_TOP);
      P.top_k = std::clamp(requested_top_k, 1, (int)scovox::K_TOP);
      if (requested_top_k != P.top_k) {
        RCLCPP_WARN(get_logger(),
          "semantic_top_k=%d clamped to %d (compile-time K_TOP cap). "
          "To raise this, recompile scovox_core with a larger K_TOP.",
          requested_top_k, P.top_k);
      }
    }
    // TSDF: truncation distance is set in voxel units so it scales with
    // resolution across launch files. The whole legacy fused path treats
    // sdf_trunc==0 as "TSDF off" — no band walk in fused_integrate_ray_static,
    // no ~/tsdf_pointcloud publisher, no ~/extract_mesh service. `enable_tsdf`
    // (default true) is the explicit off-switch that forces it there; split
    // mode (use_split=true) can't honor it (TsdfMap re-clamps sdf_trunc<=0 in
    // tsdf_map.cpp), so the constructor warns. `carve_band` is independent.
    {
      const bool enable_tsdf = dp("enable_tsdf", true);
      const int sdf_trunc_voxels = (int)dp("sdf_trunc_voxels", 3);
      P.sdf_trunc = (enable_tsdf && sdf_trunc_voxels > 0)
        ? (float)(sdf_trunc_voxels * P.resolution) : 0.f;
    }
    P.tsdf_space_carving = dp("tsdf_space_carving", false);
    P.band_only_integration = dp("band_only_integration", false);
    P.semantic_occ_gate = dp("semantic_occ_gate", 0.5);
    P.carve_skip_occ_threshold = dp("carve_skip_occ_threshold", 0.7);
    P.evidence_saturation = static_cast<uint16_t>(dp("evidence_saturation", 1000));
    P.dirichlet_min_p_occ = dp("dirichlet_min_p_occ", 0.5);
    sem_vis_thresh_ = dp("semantic_vis_threshold", -1.0);
    P.range_decay_length = dp("range_decay_length", -1.0);
    P.min_range = dp("min_range", 0.3);  P.max_range = dp("max_range", 10.0);
    P.grazing_angle_threshold = dp("grazing_angle_threshold", -1.0);
    { std::string sm = dp("semantic_mode", std::string("dirichlet"));
      if (sm == "naive") P.semantic_mode = scovox::SemanticMode::NAIVE;
      else if (sm == "majority_vote") P.semantic_mode = scovox::SemanticMode::MAJORITY_VOTE;
      else P.semantic_mode = scovox::SemanticMode::DIRICHLET; }
    max_sem_ = dp("max_semantic_classes", 10);
    transient_decay_rate_ = dp("transient_decay_rate", 0.8);
    for (auto c : dp("dynamic_classes", std::vector<int64_t>{}))
      if (c >= 0 && c < max_sem_) dyn_cls_.insert((uint8_t)c);
    return P;
  }
  void declareNodeParams() {
    auto dp = [&](auto n, auto d){ return this->declare_parameter<decltype(d)>(n, d); };
    base_frame_ = dp("base_frame", std::string("base_link"));
    map_frame_ = dp("map_frame", std::string("map"));
    int_frame_ = dp("integration_frame", std::string("odom"));
    depth_topic_ = dp("depth_topic", std::string("/atlas/rgbd_camera/depth_image"));
    di_topic_ = dp("depth_info_topic", std::string("/atlas/rgbd_camera/camera_info"));
    seg_topic_ = dp("seg_topic", std::string("/atlas/segmentation/colored"));
    input_pc_topic_ = dp("input_pointcloud_topic", std::string(""));
    stride_ = std::max(1, (int)dp("stride", 1));
    min_d_ = dp("min_depth", 0.1);  max_d_ = dp("max_depth", 10.0);
    trace_nr_ = dp("trace_no_return_rays", false);
    carve_band_ = dp("carve_band", -1.0);
    // mode = "persistent": single-robot, no binary publish to dscovox.
    // mode = "rolling":    publishes ScovoxMapBinary updates for the merger
    //                      and the planning_map is a rolling crop around the
    //                      robot (Phase 2). The underlying voxel grid is
    //                      fully persistent in both modes — no pruning.
    mode_ = dp("mode", std::string("rolling"));
    if (mode_ != "rolling" && mode_ != "persistent") mode_ = "rolling";
    robot_id_ = dp("robot_id", std::string(""));
    pub_pc_ = dp("publish_pointcloud", true);
    min_occ_ = dp("occupancy_vis_threshold", 0.7);
    pub_plan_ = dp("publish_planning_map", true);
    plan_res_ = dp("planning_map_resolution", 0.20);
    plan_sz_ = dp("planning_map_size_m", 80.0);
    plan_ox_ = dp("planning_map_origin_x", -40.0);
    plan_oy_ = dp("planning_map_origin_y", -40.0);
    plan_zmin_ = dp("planning_map_min_z", -1.0);
    plan_zmax_ = dp("planning_map_max_z", 2.0);
    plan_infl_ = dp("planning_map_inflation_m", 0.0);
    // Side length of the robot-centered planning_map crop window in
    // mode=rolling. Ignored in mode=persistent (which uses the fixed
    // (plan_ox_, plan_oy_, plan_sz_) envelope).
    plan_window_size_m_ = dp("planning_map_window_size_m", 20.0);
    pub_tsdf_ = dp("publish_tsdf_pointcloud", true);
    min_tsdf_w_ = dp("min_tsdf_weight_publish", 0.5);
    dataset_mode_ = dp("dataset_mode", false);
    // Diagnostic memory/RSS logging (scheduleMemUsage). Default off so the
    // per-grid memUsageDetailed walk + detached reader thread don't run on
    // the production mapping hot path; set true to profile memory.
    mem_log_ = dp("log_mem_usage", false);
    // Step 8 (D7) — split-grid v2 toggle. When true, this node integrates
    // into a `ScovoxMapSplit` (TsdfMap + SemBetaMap) instead of the legacy
    // fused `scovox::Map`, walks SemBeta on ~/pointcloud, walks TsdfMap +
    // labelPointCloud on ~/tsdf_pointcloud, and emits BinarySerializerV2
    // frames (msg->version=2) on ~/scovox_bin. Default false for backward
    // compat — legacy Replica/KITTI experiments reproduce verbatim.
    use_split_ = dp("use_split", false);
    // Sender-side wire toggle. When use_split_=true:
    //   share_tsdf=false (default): emit SemBeta-only frames (37 B/voxel,
    //     dscovox-fusion-only path; each robot keeps its local TSDF).
    //   share_tsdf=true: emit dual-stream TSDF + SemBeta (57 B/voxel,
    //     opt-in for fused-geometry consensus).
    // Ignored when use_split_=false (legacy v1 wire format has no
    // share_tsdf concept). Defaults match BinarySerializerV2::Options.
    share_tsdf_ = dp("share_tsdf", false);
    // Step 12.10 (2026-05-09) — fused single-DDA ray walker. Default true.
    // Set false to fall back to the legacy two-DDA split path
    // (TsdfMap::integrateRay + SemBetaMap::integrateHit) for A/B parity
    // testing. Ignored when use_split_=false (legacy fused-Voxel path).
    fused_walker_ = dp("fused_walker", true);
    // SemDir priors (use_split=true only). num_classes is the dataset's
    // total class count — sets the OTHER bucket's prior mass to
    // (num_classes − K_TOP) · alpha_0 so the implicit Dirichlet still
    // marginalises onto the true (C+1)-category distribution. Defaults
    // match SemDirMap::Params (NYU13 / 0.01). KITTI launches override
    // num_classes:=20; Replica/SceneNet stay at 14.
    num_classes_ = std::max<int>(scovox::K_TOP + 1,
                                 dp("num_classes", (int)scovox::SemDirMap::Params{}.num_classes));
    alpha_0_     = (float)dp("dirichlet_prior",
                             (double)scovox::SemDirMap::Params{}.alpha_0);
    if (alpha_0_ <= 0.f) alpha_0_ = scovox::kDefaultDirichletPrior;
    // Wire-format selector (use_split=true only). v2 keeps the SemBeta-
    // projected wire so the existing dscovox_node v2 receiver still works
    // — production safe default during the SemDir transition. v3 emits
    // SemDirVoxel-native frames (20 B/voxel vs v2's 37 B/voxel projected)
    // and carries num_classes/alpha_0 in the header so the receiver doesn't
    // need to share launch params. Set v3 once the dscovox receiver lands.
    {
      const std::string fmt = dp("wire_format", std::string("v2"));
      if (fmt == "v3") wire_format_v3_ = true;
      else if (fmt == "v2") wire_format_v3_ = false;
      else {
        RCLCPP_WARN(get_logger(),
          "wire_format='%s' unrecognised (want v2|v3); defaulting v2.", fmt.c_str());
        wire_format_v3_ = false;
      }
    }
    // Audit hook (split-grid only): when non-empty, every periodic memlog
    // tick overwrites this path with a flat binary snapshot of TsdfMap:
    //   uint64_t n; then n × {float x, float y, float z, float distance,
    //   float weight} = 20 B/voxel, voxel-centre coords in scovox_node's
    //   world frame. Lets a parity-test harness compare the TsdfMap voxel
    //   set against SLIM-VDB's voxels.bin after a Tr_inv frame conversion
    //   (see tools/tsdf_parity_test.py). Empty default → no-op.
    tsdf_dump_path_ = dp("tsdf_dump_path", std::string{});
    pointcloud_mode_ = !input_pc_topic_.empty();
    // TF stability gate: skip sensor frames until the TF pose has been
    // stable (no jumps > threshold) for at least this many seconds.
    // Prevents ghost voxels at origin when TF is briefly wrong at startup.
    startup_tf_stable_sec_ = dp("startup_tf_stable_sec", 2.0);
    startup_tf_jump_thresh_ = dp("startup_tf_jump_threshold", 0.5);
    // Soft-probability ablation: directory of <frame>.topk flat-binary blobs
    // produced by topk_npz_to_bin.py. When non-empty, scovox_node uses the
    // frame index (low 16 bits of header.stamp.nanosec, set by the replay
    // node) to look up per-point/per-pixel top-K class distributions and
    // feeds them into the Dirichlet update instead of the one-hot built
    // from the hard label. Must contain only zero-padded names like
    // "000000.topk". Empty string = legacy hard-label path.
    topk_probs_dir_ = dp("topk_probs_dir", std::string(""));
    topk_topk_max_  = (int)dp("topk_probs_max_k", 5);
    // E5.2: per-frame eviction stats CSV. Empty = disabled. When set,
    // appends one line per frame: frame,match,empty,evict,drop (deltas).
    evict_csv_path_ = dp("eviction_stats_csv", std::string(""));
    if (!evict_csv_path_.empty()) {
      evict_csv_.open(evict_csv_path_, std::ios::out | std::ios::trunc);
      if (evict_csv_.is_open()) {
        evict_csv_ << "frame,match,empty,evict,drop\n";
        evict_csv_.flush();
        RCLCPP_INFO(get_logger(), "eviction_stats_csv → %s", evict_csv_path_.c_str());
      } else {
        RCLCPP_WARN(get_logger(), "eviction_stats_csv: failed to open %s", evict_csv_path_.c_str());
      }
    }
  }

  void logEvictionDelta(uint64_t frame_id) {
    if (!evict_csv_.is_open()) return;
    const uint64_t m = scovox::g_sparse_match_count.load(std::memory_order_relaxed);
    const uint64_t e = scovox::g_sparse_empty_count.load(std::memory_order_relaxed);
    const uint64_t v = scovox::g_sparse_evict_count.load(std::memory_order_relaxed);
    const uint64_t d = scovox::g_sparse_drop_count.load(std::memory_order_relaxed);
    evict_csv_ << frame_id << ',' << (m - last_match_) << ',' << (e - last_empty_)
               << ',' << (v - last_evict_) << ',' << (d - last_drop_) << '\n';
    evict_csv_.flush();
    last_match_ = m; last_empty_ = e; last_evict_ = v; last_drop_ = d;
  }
  void setupSubscribers() {
    if (pointcloud_mode_) {
      auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
      input_pc_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(input_pc_topic_, qos,
        std::bind(&SCovoxNode::onPointCloud, this, std::placeholders::_1));
      RCLCPP_INFO(get_logger(), "PointCloud2 input mode: topic=%s", input_pc_topic_.c_str());
    } else {
      di_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(di_topic_, rclcpp::QoS(10),
        [this](sensor_msgs::msg::CameraInfo::SharedPtr m){ di_ = *m; have_di_.store(true, std::memory_order_release); });
      if (dataset_mode_) {
        auto qos = rclcpp::QoS(rclcpp::KeepLast(1000)).reliable();
        ds_depth_sub_ = create_subscription<sensor_msgs::msg::Image>(depth_topic_, qos,
          [this](sensor_msgs::msg::Image::ConstSharedPtr m){ onDatasetDepth(m); });
        ds_seg_sub_ = create_subscription<sensor_msgs::msg::Image>(seg_topic_, qos,
          [this](sensor_msgs::msg::Image::ConstSharedPtr m){ onDatasetSeg(m); });
        RCLCPP_INFO(get_logger(), "Dataset mode: exact timestamp matching enabled");
      } else {
        d_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(this, depth_topic_, rmw_qos_profile_sensor_data);
        s_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(this, seg_topic_, rmw_qos_profile_sensor_data);
        using SP = message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image>;
        sync_ = std::make_shared<message_filters::Synchronizer<SP>>(SP(2), *d_sub_, *s_sub_);
        sync_->registerCallback(std::bind(&SCovoxNode::onImages, this, std::placeholders::_1, std::placeholders::_2));
      }
    }
  }
  void setupPublishers() {
    auto dp = [&](auto n, auto d){ return this->declare_parameter<decltype(d)>(n, d); };
    auto pc_t = dp("pointcloud_topic", std::string("~/pointcloud"));
    auto sm_t = dp("scovox_topic", std::string("~/scovox"));
    auto pl_t = dp("planning_map_topic", std::string("~/planning_map"));
    auto tsdf_t = dp("tsdf_pointcloud_topic", std::string("~/tsdf_pointcloud"));
    sm_pub_ = create_publisher<scovox_msgs::msg::ScovoxMap>(sm_t, 10);
    if (mode_ == "rolling") {
      // Explicit reliable + deeper queue for binary deltas. dscovox_node
      // mirrors this. The int-overload (just `, 10`) was nominally
      // RELIABLE but combined with the subscriber's SystemDefaultsQoS
      // (which resolved to BEST_EFFORT here) the connection downgraded
      // and silently dropped large submap payloads. Pinning both ends
      // explicitly removes the ambiguity.
      auto bin_qos = rclcpp::QoS(rclcpp::KeepLast(50)).reliable();
      bin_pub_ = create_publisher<scovox_msgs::msg::ScovoxMapBinary>(
        sm_t + std::string("_bin"), bin_qos);
    }
    // Queue depth 1 (was 10): on KITTI 10 cm a single PointCloud2 can be
    // 2 GB. With reliable QoS + a slow subscriber the publisher would
    // otherwise buffer up to ~20 GB and OOM the host during NPZ capture.
    pc_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
        pc_t, rclcpp::QoS(rclcpp::KeepLast(1)).reliable());
    if (pub_plan_) pl_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>(pl_t, rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local());
    if (pub_tsdf_ && map_->params().sdf_trunc > 0.f) {
      tsdf_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
          tsdf_t, rclcpp::QoS(rclcpp::KeepLast(1)).reliable());
    }
  }
  static Eigen::Isometry3f toE(const geometry_msgs::msg::TransformStamped& t) {
    Eigen::Isometry3f T = Eigen::Isometry3f::Identity();
    T.translation() << t.transform.translation.x, t.transform.translation.y, t.transform.translation.z;
    T.linear() = Eigen::Quaternionf(t.transform.rotation.w, t.transform.rotation.x,
                                     t.transform.rotation.y, t.transform.rotation.z).toRotationMatrix();
    return T;
  }
  // --- Dataset mode: match depth & seg by frame index (low 16 bits of nanosec) ---
  static uint16_t frameId(const sensor_msgs::msg::Image::ConstSharedPtr& m) {
    return (uint16_t)(m->header.stamp.nanosec & 0xFFFF);
  }

  void onDatasetDepth(const sensor_msgs::msg::Image::ConstSharedPtr& m) {
    uint16_t fid = frameId(m);
    auto it = ds_seg_cache_.find(fid);
    if (it != ds_seg_cache_.end()) {
      RCLCPP_INFO(get_logger(), "Matched frame %u (depth arrived second)", fid);
      onImages(m, it->second);
      ds_seg_cache_.erase(it);
    } else {
      ds_depth_cache_[fid] = m;
      if (ds_depth_cache_.size() > 200) { RCLCPP_WARN(get_logger(), "Depth cache overflow (%zu), clearing stale entries", ds_depth_cache_.size()); ds_depth_cache_.clear(); }
    }
  }
  void onDatasetSeg(const sensor_msgs::msg::Image::ConstSharedPtr& m) {
    uint16_t fid = frameId(m);
    auto it = ds_depth_cache_.find(fid);
    if (it != ds_depth_cache_.end()) {
      RCLCPP_INFO(get_logger(), "Matched frame %u (seg arrived second)", fid);
      onImages(it->second, m);
      ds_depth_cache_.erase(it);
    } else {
      ds_seg_cache_[fid] = m;
      if (ds_seg_cache_.size() > 200) { RCLCPP_WARN(get_logger(), "Seg cache overflow (%zu), clearing stale entries", ds_seg_cache_.size()); ds_seg_cache_.clear(); }
    }
  }

  static size_t getVmRSSKB() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
      if (line.rfind("VmRSS:", 0) == 0) {
        size_t kb = 0;
        sscanf(line.c_str(), "VmRSS: %zu", &kb);
        return kb;
      }
    }
    return 0;
  }

  void onImages(const sensor_msgs::msg::Image::ConstSharedPtr& depth, const sensor_msgs::msg::Image::ConstSharedPtr& seg) {
    auto t_start = std::chrono::high_resolution_clock::now();
    if (use_split_ && split_map_) split_map_->resetTiming();
    ++frame_recv_;
    uint16_t replay_idx = (uint16_t)(depth->header.stamp.nanosec & 0xFFFF);
    if (!have_di_.load(std::memory_order_acquire)) { RCLCPP_WARN(get_logger(), "recv=%zu replay=%u: waiting for CameraInfo", frame_recv_, replay_idx); return; }
    std::unique_lock<std::shared_mutex> lock(map_mtx_);
    if (depth->width != seg->width || depth->height != seg->height) { RCLCPP_WARN(get_logger(), "recv=%zu replay=%u: size mismatch", frame_recv_, replay_idx); return; }
    if (std::abs(rclcpp::Time(seg->header.stamp).seconds() - rclcpp::Time(depth->header.stamp).seconds()) > 0.05) { RCLCPP_WARN(get_logger(), "recv=%zu replay=%u: timestamp mismatch dt=%.3f", frame_recv_, replay_idx, std::abs(rclcpp::Time(seg->header.stamp).seconds() - rclcpp::Time(depth->header.stamp).seconds())); return; }
    const int W = (int)depth->width, H = (int)depth->height;
    const double fx = di_.k[0], fy = di_.k[4], cx = di_.k[2], cy = di_.k[5];
    const bool d16 = (depth->encoding == enc::TYPE_16UC1 || depth->encoding == enc::MONO16);
    const bool d32 = (depth->encoding == enc::TYPE_32FC1);
    if (!d16 && !d32) { RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "Bad depth enc"); return; }
    const bool srgb8 = (seg->encoding==enc::RGB8||seg->encoding==enc::BGR8||seg->encoding==enc::RGBA8||seg->encoding==enc::BGRA8);
    if (!srgb8) { RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "Bad seg enc"); return; }
    const float dsc = d16 ? 0.001f : 1.0f;
    const int st = stride_;
    const bool rgba = (seg->encoding==enc::RGBA8||seg->encoding==enc::BGRA8);
    const int sch = rgba ? 4 : 3;
    const bool srgb = (seg->encoding==enc::RGB8||seg->encoding==enc::RGBA8);
    if (!dyn_cls_.empty()) map_->decayTransientGrid((float)transient_decay_rate_);
    Eigen::Isometry3f T_oo;
    bool tf_exact = false;
    try { T_oo = toE(tf_buffer_.lookupTransform(int_frame_, depth->header.frame_id, depth->header.stamp, rclcpp::Duration::from_seconds(0.2)));
      tf_exact = true;
    } catch (...) { try { T_oo = toE(tf_buffer_.lookupTransform(int_frame_, depth->header.frame_id, rclcpp::Time(0), rclcpp::Duration::from_seconds(0.2)));
      RCLCPP_WARN(get_logger(), "recv=%zu replay=%u: TF FALLBACK to Time(0) — pose may be wrong!", frame_recv_, replay_idx);
    } catch (const std::exception& e) { RCLCPP_WARN(get_logger(), "recv=%zu replay=%u: TF FAILED: %s", frame_recv_, replay_idx, e.what()); return; } }
    if (tf_exact) { RCLCPP_DEBUG(get_logger(), "recv=%zu replay=%u: TF exact match", frame_recv_, replay_idx); }
    static const Eigen::Matrix3f kR = (Eigen::Matrix3f() << 0,0,1, -1,0,0, 0,-1,0).finished();
    T_oo.linear() = T_oo.linear() * kR;
    Eigen::Vector3f O;
    try { auto t = tf_buffer_.lookupTransform(int_frame_, base_frame_, depth->header.stamp, rclcpp::Duration::from_seconds(0.2));
      O << t.transform.translation.x, t.transform.translation.y, t.transform.translation.z;
    } catch (...) { try { auto t = tf_buffer_.lookupTransform(int_frame_, base_frame_, rclcpp::Time(0), rclcpp::Duration::from_seconds(0.2));
      O << t.transform.translation.x, t.transform.translation.y, t.transform.translation.z;
      RCLCPP_WARN(get_logger(), "recv=%zu replay=%u: observer TF FALLBACK to Time(0)", frame_recv_, replay_idx);
    } catch (const std::exception& e) { RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "TF(o): %s", e.what()); return; } }
    // --- TF stability gate: skip frames until pose is stable ----
    if (!tf_stable_) {
      if (tf_prev_valid_) {
        float jump = (O - tf_prev_pos_).norm();
        if (jump > startup_tf_jump_thresh_) {
          tf_stable_since_ = this->now();
          RCLCPP_WARN(get_logger(), "TF jump %.2f m during startup, resetting stability timer", jump);
        } else if ((this->now() - tf_stable_since_).seconds() >= startup_tf_stable_sec_) {
          tf_stable_ = true;
          RCLCPP_INFO(get_logger(), "TF stable for %.1f s — starting integration", startup_tf_stable_sec_);
        }
      } else {
        tf_stable_since_ = this->now();
      }
      tf_prev_pos_ = O;
      tf_prev_valid_ = true;
      if (!tf_stable_) {
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
            "Waiting for TF stabilization (%.1f / %.1f s)...",
            (this->now() - tf_stable_since_).seconds(), startup_tf_stable_sec_);
        return;
      }
    }
    auto t_tf = std::chrono::high_resolution_clock::now();
    const auto& P = map_->params();
    std::vector<scovox::Map::CoordT> rc;
    std::vector<Eigen::Vector3f> nr_eps;
    auto rdZ = [&](int pu, int pv) -> float {
      if (pu<0||pu>=W||pv<0||pv>=H) return 0.f;
      if (d16) { float d = float(reinterpret_cast<const uint16_t*>(depth->data.data()+pv*depth->step)[pu])*dsc; return (std::isfinite(d)&&d>0.f)?d:0.f; }
      else { float d = reinterpret_cast<const float*>(depth->data.data()+pv*depth->step)[pu]; return (std::isfinite(d)&&d>0.f)?d:0.f; }
    };
    // Soft-prob path: load per-frame top-K image once. We use the
    // depth-image stamp's low 16 bits as the frame index (same convention
    // the replay node sets via _stamp_from_index).
    bool use_topk = false;
    if (!topk_probs_dir_.empty()) {
      uint16_t fid = (uint16_t)(depth->header.stamp.nanosec & 0xFFFF);
      use_topk = loadTopkForFrame(fid, /*image_mode=*/true);
    }

    std::vector<float> cp(max_sem_, 0.f);
    for (int v=0; v<H; v+=st) for (int u=0; u<W; u+=st) {
      float z = d16 ? float(reinterpret_cast<const uint16_t*>(depth->data.data()+v*depth->step)[u])*dsc
                     : reinterpret_cast<const float*>(depth->data.data()+v*depth->step)[u];
      if (!std::isfinite(z) || z > (float)max_d_) {
        if (trace_nr_) { float zf=(float)max_d_; nr_eps.push_back(T_oo*Eigen::Vector3f(float((u-cx)*zf/fx),float((v-cy)*zf/fy),zf)); }
        continue;
      }
      if (z < (float)min_d_) continue;
      Eigen::Vector3f Hp = T_oo * Eigen::Vector3f(float((u-cx)*z/fx), float((v-cy)*z/fy), z);
      bool vs = false;
      if (use_topk) {
        vs = fillCpFromTopkImage(u, v, cp);
      } else {
        const uint8_t* px = seg->data.data() + v*seg->step + sch*u;
        uint8_t r = srgb?px[0]:px[2], g = px[1], b = srgb?px[2]:px[0];
        uint16_t lbl = lookupLabel((uint32_t(r)<<16)|(uint32_t(g)<<8)|uint32_t(b));
        if (lbl > 0 && lbl < max_sem_) { std::fill(cp.begin(), cp.end(), 0.f); cp[lbl] = 1.f; vs = true; }
      }
      float rng = (Hp-O).norm(), rw = 1.f;
      if (P.range_decay_length > 0) { if (rng<P.min_range||rng>P.max_range) continue; rw = std::exp(-rng/float(P.range_decay_length)); }
      float aw = 1.f;
      if (P.grazing_angle_threshold > 0 && rng > 0.01f) {
        Eigen::Vector3f rd = (Hp-O).normalized();
        float zl=rdZ(u-st,v), zr=rdZ(u+st,v), zu=rdZ(u,v-st), zd=rdZ(u,v+st);
        if (zl>0&&zr>0&&zu>0&&zd>0) {
          Eigen::Vector3f pl(float((u-st-cx)*zl/fx),float((v-cy)*zl/fy),zl), pr(float((u+st-cx)*zr/fx),float((v-cy)*zr/fy),zr);
          Eigen::Vector3f pu2(float((u-cx)*zu/fx),float((v-st-cy)*zu/fy),zu), pd(float((u-cx)*zd/fx),float((v+st-cy)*zd/fy),zd);
          Eigen::Vector3f no = (pr-pl).cross(pd-pu2); float nl = no.norm();
          if (nl>1e-6f) { float ca = std::abs(rd.dot(T_oo.linear()*(no/nl))); if (ca<P.grazing_angle_threshold) aw = ca/float(P.grazing_angle_threshold); }
        }
      }
      float q = rw*aw;
      bool dyn = false;
      if (vs && !dyn_cls_.empty()) for (size_t i=0;i<cp.size();i++) if (cp[i] > 0.f && dyn_cls_.count((uint8_t)i)) { dyn=true; break; }
      integrateHit(O, Hp, rng, dyn, vs ? &cp : nullptr, q, rw, aw, rc);
    }
    carveNoReturnRays(O, nr_eps, rc);
    auto t_integrate = std::chrono::high_resolution_clock::now();
    size_t bin_bytes_ = 0;
    if (bin_pub_) { auto [bv,bm] = publishBinaryMap(); bin_bytes_ = bv; (void)bm; }
    else if (use_split_ && split_map_) {
      // No bin_pub_ in persistent mode → publishBinaryMapV2 is never
      // called → TsdfMap/SemBetaMap touched buffers grow unbounded
      // (every integrated ray appends coords). Clear via the O(n)
      // path: drainTouched* sorts+uniques (~1 s/frame at Replica res
      // 0.05), but the result is unused here, so a plain clear is
      // ~µs. The bin_pub_ branch above still uses drainTouched* for
      // the wire-format dedup it needs.
      split_map_->clearTouchedTsdf();
      split_map_->clearTouchedSemDir();
    }
    if (pub_plan_ && pl_pub_) publishPlanningMap();
    // Pointcloud is published on the 1Hz timer — not per-frame, to avoid blocking integration
    auto t_end = std::chrono::high_resolution_clock::now();
    float frame_ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    size_t mem_kb = getVmRSSKB();
    float tf_ms = std::chrono::duration<float, std::milli>(t_tf - t_start).count();
    float integrate_ms = std::chrono::duration<float, std::milli>(t_integrate - t_tf).count();
    float publish_ms = std::chrono::duration<float, std::milli>(t_end - t_integrate).count();
    if (use_split_ && split_map_) {
      const float t_ms = static_cast<float>(split_map_->tsdfTimeUs())    / 1000.0f;
      const float s_ms = static_cast<float>(split_map_->semdirTimeUs()) / 1000.0f;
      RCLCPP_INFO(get_logger(),
                  "recv=%zu replay=%u frame_ms=%.1f tf_ms=%.1f integrate_ms=%.1f "
                  "publish_ms=%.1f rss_mb=%.1f tsdf_ms=%.1f sembeta_ms=%.1f bin_bytes=%zu",
                  frame_recv_, replay_idx, frame_ms, tf_ms, integrate_ms,
                  publish_ms, mem_kb / 1024.0, t_ms, s_ms, bin_bytes_);
    } else {
      RCLCPP_INFO(get_logger(), "recv=%zu replay=%u frame_ms=%.1f tf_ms=%.1f integrate_ms=%.1f publish_ms=%.1f rss_mb=%.1f",
                  frame_recv_, replay_idx, frame_ms, tf_ms, integrate_ms, publish_ms, mem_kb / 1024.0);
    }
    if (frame_recv_ % 10 == 1) scheduleMemUsage();
    logEvictionDelta(frame_recv_);
    logTopkSummary(/*throttle_ms=*/10000);
  }

  // ── PointCloud2 input path (LiDAR) ──────────────────────────────────────
  void onPointCloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud) {
    auto t_start = std::chrono::high_resolution_clock::now();
    if (use_split_ && split_map_) split_map_->resetTiming();
    ++frame_recv_;
    uint16_t replay_idx = (uint16_t)(cloud->header.stamp.nanosec & 0xFFFF);
    std::unique_lock<std::shared_mutex> lock(map_mtx_);

    // Find field offsets
    int off_x=-1, off_y=-1, off_z=-1, off_lbl=-1;
    uint8_t lbl_type = 0;
    for (auto& f : cloud->fields) {
      if (f.name=="x") off_x=f.offset;
      else if (f.name=="y") off_y=f.offset;
      else if (f.name=="z") off_z=f.offset;
      else if (f.name=="semantic_label") { off_lbl=f.offset; lbl_type=f.datatype; }
    }
    if (off_x<0||off_y<0||off_z<0) { RCLCPP_WARN(get_logger(), "PointCloud2 missing xyz fields"); return; }

    // TF: sensor frame -> integration frame (NO kR rotation — LiDAR is already ROS convention)
    Eigen::Isometry3f T_oi;
    try { T_oi = toE(tf_buffer_.lookupTransform(int_frame_, cloud->header.frame_id, cloud->header.stamp, rclcpp::Duration::from_seconds(0.2)));
    } catch (...) { try { T_oi = toE(tf_buffer_.lookupTransform(int_frame_, cloud->header.frame_id, rclcpp::Time(0), rclcpp::Duration::from_seconds(0.2)));
    } catch (const std::exception& e) { RCLCPP_WARN(get_logger(), "recv=%zu: TF FAILED: %s", frame_recv_, e.what()); return; } }

    // Sensor origin in integration frame
    Eigen::Vector3f O;
    try { auto t = tf_buffer_.lookupTransform(int_frame_, base_frame_, cloud->header.stamp, rclcpp::Duration::from_seconds(0.2));
      O << t.transform.translation.x, t.transform.translation.y, t.transform.translation.z;
    } catch (...) { try { auto t = tf_buffer_.lookupTransform(int_frame_, base_frame_, rclcpp::Time(0), rclcpp::Duration::from_seconds(0.2));
      O << t.transform.translation.x, t.transform.translation.y, t.transform.translation.z;
    } catch (...) { return; } }

    // --- TF stability gate (same as onImages path) ---
    if (!tf_stable_) {
      if (tf_prev_valid_) {
        float jump = (O - tf_prev_pos_).norm();
        if (jump > startup_tf_jump_thresh_) {
          tf_stable_since_ = this->now();
        } else if ((this->now() - tf_stable_since_).seconds() >= startup_tf_stable_sec_) {
          tf_stable_ = true;
          RCLCPP_INFO(get_logger(), "TF stable for %.1f s — starting integration", startup_tf_stable_sec_);
        }
      } else {
        tf_stable_since_ = this->now();
      }
      tf_prev_pos_ = O;
      tf_prev_valid_ = true;
      if (!tf_stable_) {
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
            "Waiting for TF stabilization (%.1f / %.1f s)...",
            (this->now() - tf_stable_since_).seconds(), startup_tf_stable_sec_);
        return;
      }
    }

    auto t_tf = std::chrono::high_resolution_clock::now();
    const auto& P = map_->params();
    std::vector<scovox::Map::CoordT> rc;

    const uint8_t* data = cloud->data.data();
    const int step = (int)cloud->point_step;
    const int N = (int)(cloud->width * cloud->height);

    // Soft-prob: load the per-frame top-K table once. The replay node sets
    // soft_prob_passthrough so the cloud is in raw .bin order, matching
    // the .topk row order; we apply the same range filter below.
    bool use_topk = false;
    if (!topk_probs_dir_.empty()) {
      uint16_t fid = (uint16_t)(cloud->header.stamp.nanosec & 0xFFFF);
      use_topk = loadTopkForFrame(fid, /*image_mode=*/false);
    }

    std::vector<float> cp(max_sem_, 0.f);
    for (int i = 0; i < N; ++i) {
      const uint8_t* p = data + i * step;
      float x = *reinterpret_cast<const float*>(p + off_x);
      float y = *reinterpret_cast<const float*>(p + off_y);
      float z = *reinterpret_cast<const float*>(p + off_z);
      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) continue;

      Eigen::Vector3f Hp = T_oi * Eigen::Vector3f(x, y, z);
      float rng = (Hp - O).norm();
      if (rng < P.min_range || rng > P.max_range) continue;
      float rw = (P.range_decay_length > 0) ? std::exp(-rng / float(P.range_decay_length)) : 1.f;
      float q = rw;

      bool vs = false;
      if (use_topk) {
        vs = fillCpFromTopkPoint((size_t)i, cp);
      } else {
        // Semantic label from point cloud field
        uint16_t lbl = 0;
        if (off_lbl >= 0) {
          if (lbl_type == sensor_msgs::msg::PointField::UINT16)
            lbl = *reinterpret_cast<const uint16_t*>(p + off_lbl);
          else if (lbl_type == sensor_msgs::msg::PointField::UINT8)
            lbl = *reinterpret_cast<const uint8_t*>(p + off_lbl);
          else if (lbl_type == sensor_msgs::msg::PointField::UINT32)
            lbl = (uint16_t)std::min(*reinterpret_cast<const uint32_t*>(p + off_lbl), (uint32_t)0xFFFF);
        }
        if (lbl > 0 && lbl < max_sem_) { std::fill(cp.begin(), cp.end(), 0.f); cp[lbl] = 1.f; vs = true; }
      }

      bool dyn = false;
      if (vs && !dyn_cls_.empty()) for (size_t j=0;j<cp.size();j++) if (cp[j] > 0.f && dyn_cls_.count((uint8_t)j)) { dyn=true; break; }
      integrateHit(O, Hp, rng, dyn, vs ? &cp : nullptr, q, rw, 1.f, rc);
    }

    auto t_integrate = std::chrono::high_resolution_clock::now();
    size_t bin_bytes_ = 0;
    if (bin_pub_) { auto [bv,bm] = publishBinaryMap(); bin_bytes_ = bv; (void)bm; }
    else if (use_split_ && split_map_) {
      // No bin_pub_ in persistent mode → publishBinaryMapV2 is never
      // called → TsdfMap/SemBetaMap touched buffers grow unbounded
      // (every integrated ray appends coords). Clear via the O(n)
      // path: drainTouched* sorts+uniques (~1 s/frame at Replica res
      // 0.05), but the result is unused here, so a plain clear is
      // ~µs. The bin_pub_ branch above still uses drainTouched* for
      // the wire-format dedup it needs.
      split_map_->clearTouchedTsdf();
      split_map_->clearTouchedSemDir();
    }
    if (pub_plan_ && pl_pub_) publishPlanningMap();
    auto t_end = std::chrono::high_resolution_clock::now();
    float frame_ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    size_t mem_kb = getVmRSSKB();
    float tf_ms = std::chrono::duration<float, std::milli>(t_tf - t_start).count();
    float integrate_ms = std::chrono::duration<float, std::milli>(t_integrate - t_tf).count();
    float publish_ms = std::chrono::duration<float, std::milli>(t_end - t_integrate).count();
    if (use_split_ && split_map_) {
      const float t_ms = static_cast<float>(split_map_->tsdfTimeUs())    / 1000.0f;
      const float s_ms = static_cast<float>(split_map_->semdirTimeUs()) / 1000.0f;
      RCLCPP_INFO(get_logger(),
                  "recv=%zu replay=%u frame_ms=%.1f tf_ms=%.1f integrate_ms=%.1f "
                  "publish_ms=%.1f rss_mb=%.1f tsdf_ms=%.1f sembeta_ms=%.1f bin_bytes=%zu",
                  frame_recv_, replay_idx, frame_ms, tf_ms, integrate_ms,
                  publish_ms, mem_kb / 1024.0, t_ms, s_ms, bin_bytes_);
    } else {
      RCLCPP_INFO(get_logger(), "recv=%zu replay=%u frame_ms=%.1f tf_ms=%.1f integrate_ms=%.1f publish_ms=%.1f rss_mb=%.1f",
                  frame_recv_, replay_idx, frame_ms, tf_ms, integrate_ms, publish_ms, mem_kb / 1024.0);
    }
    if (frame_recv_ % 10 == 1) scheduleMemUsage();
    logEvictionDelta(frame_recv_);
    logTopkSummary(/*throttle_ms=*/10000);
  }

  void scheduleMemUsage() {
    if (!mem_log_) return;   // diagnostic-only; gated by log_mem_usage param
    std::thread([this]() {
      std::shared_lock<std::shared_mutex> lock(map_mtx_);
      auto d = map_->grid().memUsageDetailed();
      auto dt = map_->transientGrid().memUsageDetailed();
      size_t total = d.total_bytes + dt.total_bytes;
      RCLCPP_INFO(get_logger(), "[memUsage] voxels=%zu tvox=%zu bonxai_mb=%.1f rss_mb=%.1f bpv=%zu",
                  d.active_voxels, dt.active_voxels, total / (1024.0 * 1024.0),
                  getVmRSSKB() / 1024.0, d.active_voxels > 0 ? total / d.active_voxels : 0);
      RCLCPP_INFO(get_logger(),
                  "[memDetail] inner=%zu leaf=%zu vox/leaf=%.1f fill=%.1f%% "
                  "root_mb=%.2f inner_mb=%.2f leafmeta_mb=%.2f pool_mb=%.2f "
                  "pool_cap=%zu pool_used=%zu pool_waste_mb=%.2f",
                  d.inner_count, d.leaf_count, d.avg_active_per_leaf(), d.leaf_fill_ratio() * 100.0f,
                  d.root_map_bytes / (1024.0 * 1024.0),
                  d.inner_grid_bytes / (1024.0 * 1024.0),
                  d.leaf_meta_bytes / (1024.0 * 1024.0),
                  d.pool_alloc_bytes / (1024.0 * 1024.0),
                  d.pool_capacity, d.pool_used,
                  d.pool_waste_bytes() / (1024.0 * 1024.0));
      // Step 8 (D11) — split-grid per-substrate memory log. Consumed by
      // eval_e13_byte_parity.py to compare TsdfMap bytes against
      // SLIM-VDB's vdb_tsdf_mb_final (acceptance gate 15%, paper headline
      // reports the actually-measured ratio).
      if (split_map_) {
        const double tsdf_mb    = static_cast<double>(split_map_->tsdfGridBytes())    / (1024.0 * 1024.0);
        const double sembeta_mb = static_cast<double>(split_map_->semdirGridBytes()) / (1024.0 * 1024.0);
        RCLCPP_INFO(get_logger(),
                    "[memSplit] tsdf_voxels=%zu tsdf_grid_mb=%.3f "
                    "semdir_voxels=%zu semdir_grid_mb=%.3f",
                    split_map_->tsdfVoxelCount(), tsdf_mb,
                    split_map_->semdirVoxelCount(), sembeta_mb);
        if (!tsdf_dump_path_.empty()) {
          // Overwrite snapshot of TsdfMap on every memlog tick — last
          // write before kill -9 is what the parity tool consumes.
          const std::string tmp = tsdf_dump_path_ + ".tmp";
          std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
          if (f) {
            const uint64_t n =
                static_cast<uint64_t>(split_map_->tsdf().voxelCount());
            f.write(reinterpret_cast<const char*>(&n), sizeof(n));
            split_map_->tsdf().forEachVoxel(
                [&](const scovox::TsdfVoxel& v, const Eigen::Vector3f& p) {
                  const float buf[5] = {p.x(), p.y(), p.z(),
                                        v.distance, v.weight};
                  f.write(reinterpret_cast<const char*>(buf), sizeof(buf));
                });
            f.close();
            std::rename(tmp.c_str(), tsdf_dump_path_.c_str());
            RCLCPP_INFO(get_logger(),
                        "[tsdf_dump] wrote %lu voxels (%.1f MB) to %s",
                        static_cast<unsigned long>(n),
                        n * 20.0 / (1024.0 * 1024.0),
                        tsdf_dump_path_.c_str());
          }
        }
      }
    }).detach();
  }

  // Compute the voxel coord for an endpoint position. Used to add the hit
  // voxel to dirty_ for the integrateEndpointOnly branches (integrateRay
  // already pushes the endpoint into rc itself, so this isn't needed there).
  inline Bonxai::CoordT endpointCoord(const Eigen::Vector3f& Hp) const {
    return map_->grid().posToCoord(Eigen::Vector3d(Hp.x(), Hp.y(), Hp.z()));
  }

  void integrateHit(const Eigen::Vector3f& O, const Eigen::Vector3f& Hp, float rng,
                    bool dyn, const std::vector<float>* cp, float q, float rw, float aw,
                    std::vector<scovox::Map::CoordT>& rc) {
    rc.clear();
    if (use_split_) {
      // Split-grid path (Step 8). TsdfMap walks the SDF band, SemBetaMap
      // walks the carve band leading up to the hit. dyn / rw / aw
      // collapse: q already bakes in rw*aw; dyn (transient dynamic-class
      // decay) is a legacy-only feature retired by D12.
      //
      // carve_band parity with legacy: when `carve_band_ > 0` (Replica /
      // KITTI launch default = 0.1), walk SemBeta along only the last
      // `carve_band` metres before the surface, matching the production
      // mIoU baselines. This was originally elided ("full ray always" per
      // D5) but cost ~5× FPS at Replica res 0.05 (~50 voxels/ray vs
      // ~10 voxels/ray) without an mIoU benefit. carve_band <= 0 still
      // falls back to full-ray for users who explicitly opt in.
      (void)rng; (void)dyn; (void)rw; (void)aw;  // unused in split mode
      Eigen::Vector3f co = O;
      if (carve_band_ > 0) {
        const float d = rng - static_cast<float>(carve_band_);
        if (d > 0) co = O + (Hp - O).normalized() * d;
      }
      split_map_->integrateHit(co, Hp, cp, q);
      sm_dirty_.store(true, std::memory_order_relaxed);
      return;
    }
    if (carve_band_ == 0) {
      map_->integrateEndpointOnly(Hp, dyn, cp, q, rw, aw);
      if (bin_pub_) dirty_.insert(endpointCoord(Hp));
    } else if (carve_band_ > 0) {
      // Partial-ray mode: walk a band of carve_band metres in front of the
      // surface (Beta-free) plus sdf_trunc metres behind (TSDF only). Both
      // are handled by the fused walk by passing a truncated origin —
      // earlier code split this into carve_free + integrateEndpointOnly,
      // which left the TSDF band unpopulated (single-cell hit only).
      float d = rng - (float)carve_band_;
      Eigen::Vector3f co = (d > 0) ? O + (Hp - O).normalized() * d : O;
      map_->integrateRay(co, Hp, rc, dyn, cp, q, rw, aw);
      if (bin_pub_) for (auto& cc : rc) dirty_.insert(cc);
    } else {
      map_->integrateRay(O, Hp, rc, dyn, cp, q, rw, aw);
      if (bin_pub_) for (auto& cc : rc) dirty_.insert(cc);
    }
    sm_dirty_.store(true, std::memory_order_relaxed);
  }
  void carveNoReturnRays(const Eigen::Vector3f& O, const std::vector<Eigen::Vector3f>& nr_eps,
                         std::vector<scovox::Map::CoordT>& rc) {
    if (use_split_) {
      // SemBeta-only carve along no-return rays (no TSDF surface to anchor).
      // q=1.0f matches the legacy carve_free which doesn't apply rw/aw.
      for (auto& hf : nr_eps) split_map_->integrateMiss(O, hf, 1.0f);
      sm_dirty_.store(true, std::memory_order_relaxed);
      return;
    }
    if (carve_band_ == 0) return;
    for (auto& hf : nr_eps) {
      rc.clear();
      map_->carve_free(O, hf, rc);
      if (bin_pub_) for (auto& cc : rc) dirty_.insert(cc);
    }
  }
  void loadSemanticColorMap() {
    const std::vector<int64_t> dk={0x4382,0x21C1,0x8605,0xCA88,0xA8C7,0x6544,0}, dc={1,2,4,6,5,3,0};
    auto ck = declare_parameter<std::vector<int64_t>>("semantic_color_map_keys", dk);
    auto ci = declare_parameter<std::vector<int64_t>>("semantic_color_map_classes", dc);
    color_map_.clear();
    for (size_t i=0, n=std::min(ck.size(),ci.size()); i<n; ++i)
      if (ck[i]>=0 && ck[i]<=0xFFFFFF && ci[i]>=0 && ci[i]<=0xFFFF)
        color_map_[(uint32_t)ck[i]] = (uint16_t)ci[i];
  }
  uint16_t lookupLabel(uint32_t k) const { auto it=color_map_.find(k); return it!=color_map_.end()?it->second:0; }

  // Load the .topk flat-binary blob for `frame_id` from topk_probs_dir_.
  // Pointcloud layout: [u32 N][u8 C][N*C u8 probs(×255)] — slot j == SCovox class id.
  // Image     layout: [u16 H][u16 W][u8 C][H*W*C u8 probs(×255)] — slot j == SCovox class id.
  // Slot 0 is unknown/unlabeled by convention. Returns true on success and
  // populates the cache; false on missing/corrupt file (caller falls back
  // to the legacy hard-label one-hot path).
  bool loadTopkForFrame(uint16_t frame_id, bool image_mode) {
    if (topk_cache_valid_ && topk_cache_frame_ == frame_id &&
        topk_cache_is_image_ == image_mode) {
      // Cache hit — the per-frame counter already incremented on the
      // miss-and-load. Don't double-count.
      return true;
    }
    char path[1024];
    std::snprintf(path, sizeof(path), "%s/%06u.topk",
                  topk_probs_dir_.c_str(), (unsigned)frame_id);
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                           "topk: cannot open %s — falling back to one-hot", path);
      topk_cache_valid_ = false;
      ++topk_load_failure_;
      return false;
    }
    size_t total;
    uint8_t C;
    if (image_mode) {
      uint16_t H, W;
      f.read(reinterpret_cast<char*>(&H), 2);
      f.read(reinterpret_cast<char*>(&W), 2);
      f.read(reinterpret_cast<char*>(&C), 1);
      total = (size_t)H * W * C;
      topk_cache_h_ = H; topk_cache_w_ = W; topk_cache_c_ = C;
      topk_cache_n_ = 0;
    } else {
      uint32_t N;
      f.read(reinterpret_cast<char*>(&N), 4);
      f.read(reinterpret_cast<char*>(&C), 1);
      total = (size_t)N * C;
      topk_cache_n_ = N; topk_cache_c_ = C;
      topk_cache_h_ = 0; topk_cache_w_ = 0;
    }
    topk_cache_probs_.resize(total);
    f.read(reinterpret_cast<char*>(topk_cache_probs_.data()), total);
    if (!f.good() && !f.eof()) {
      RCLCPP_WARN(get_logger(), "topk: short read for %s", path);
      topk_cache_valid_ = false;
      ++topk_load_failure_;
      return false;
    }
    topk_cache_frame_ = frame_id;
    topk_cache_is_image_ = image_mode;
    topk_cache_valid_ = true;
    ++topk_load_success_;
    if (!topk_first_load_logged_) {
      // One-shot INFO so an operator can confirm soft-prob mode is live
      // without grepping for the per-frame trace. After this, only the
      // shutdown summary (in ~SCovoxNode) reports counters.
      RCLCPP_INFO(get_logger(),
                  "topk: first frame loaded from %s (image_mode=%d, C=%u%s)",
                  topk_probs_dir_.c_str(), (int)image_mode, (unsigned)topk_cache_c_,
                  image_mode ? "" : ", point-mode");
      topk_first_load_logged_ = true;
    }
    return true;
  }
  static inline float dequantize_prob(uint8_t q) { return float(q) * (1.0f / 255.0f); }

  // Emit a running tally of soft-prob loader outcomes. Guards the
  // silent-fallback footgun: when `topk_probs_dir_` is set but loads
  // intermittently fail, the per-frame WARN is throttled at 5 s and easy
  // to miss in long batch runs; this INFO is throttled at the caller's
  // rate (default 10 s in the per-frame paths) and shows running totals
  // so the operator (or a smoke-gate assert) can verify soft-prob
  // dispatched on every expected frame. No-op when topk is disabled.
  void logTopkSummary(int throttle_ms) {
    if (topk_probs_dir_.empty()) return;
    const uint64_t total = topk_load_success_ + topk_load_failure_;
    if (total == 0) return;
    const double fail_pct = 100.0 * static_cast<double>(topk_load_failure_)
                          / static_cast<double>(total);
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), throttle_ms,
        "topk loader: loaded=%lu fallback_to_one_hot=%lu (%.1f%% miss) dir=%s",
        (unsigned long)topk_load_success_,
        (unsigned long)topk_load_failure_,
        fail_pct,
        topk_probs_dir_.c_str());
  }

  // Fill `cp` (size max_sem_) with the top-K probabilities for the given
  // pointcloud row index. Returns true if any class with id in [1, max_sem_)
  // got non-zero probability — false means the row is degenerate (all zero
  // or all out-of-range), caller should treat as no semantic observation.
  bool fillCpFromTopkPoint(size_t row, std::vector<float>& cp) const {
    if (!topk_cache_valid_ || topk_cache_is_image_) return false;
    if (row >= topk_cache_n_) return false;
    std::fill(cp.begin(), cp.end(), 0.f);
    const size_t base = row * topk_cache_c_;
    const int C = std::min<int>(topk_cache_c_, max_sem_);
    bool any = false;
    // Slot index *is* the SCovox class id. Skip slot 0 (=unknown) so it
    // contributes to a_unk (escape mass) rather than getting mapped to a
    // real class.
    for (int j = 1; j < C; ++j) {
      const float p = dequantize_prob(topk_cache_probs_[base + j]);
      if (p > 0.f) {
        cp[j] = p;
        any = true;
      }
    }
    return any;
  }

  // Replica image variant: fill `cp` from per-pixel dense distribution at (u, v).
  bool fillCpFromTopkImage(int u, int v, std::vector<float>& cp) const {
    if (!topk_cache_valid_ || !topk_cache_is_image_) return false;
    if (u < 0 || v < 0 || u >= (int)topk_cache_w_ || v >= (int)topk_cache_h_) return false;
    std::fill(cp.begin(), cp.end(), 0.f);
    const size_t base = ((size_t)v * topk_cache_w_ + u) * topk_cache_c_;
    const int C = std::min<int>(topk_cache_c_, max_sem_);
    bool any = false;
    for (int j = 1; j < C; ++j) {
      const float p = dequantize_prob(topk_cache_probs_[base + j]);
      if (p > 0.f) {
        cp[j] = p;
        any = true;
      }
    }
    return any;
  }
  // Hash/equality helpers so Bonxai::CoordT can live in an unordered_set.
  // Used by dirty_ to track which voxels have been touched since the last
  // ScovoxMapBinary publish.
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
  void initializeSemanticColors() {
    auto base = scovox::generateSemanticColors(static_cast<size_t>(max_sem_));
    sem_col_.clear();
    sem_col_.reserve(base.size());
    for (auto& rgb : base) {
      std_msgs::msg::ColorRGBA c; c.r = rgb[0]; c.g = rgb[1]; c.b = rgb[2]; c.a = 0.8f;
      sem_col_.push_back(c);
    }
  }
  // Caller must hold map_mtx_ (shared). Lock removed from this function so
  // the timer body can hold one outer shared_lock spanning both publishers.
  std::pair<size_t,double> publishScovoxMap() {
    if (sm_pub_->get_subscription_count() == 0) return {0, 0.0};
    if (!sm_dirty_.exchange(false)) return {0, 0.0};
    scovox_msgs::msg::ScovoxMap m; m.header.stamp=get_clock()->now(); m.header.frame_id=int_frame_;
    const auto& P=map_->params(); m.resolution=P.resolution; m.occupancy_threshold=min_occ_;
    m.semantic_threshold=P.semantic_occ_gate; m.max_semantic_classes=max_sem_;
    m.voxels.reserve(map_->grid().activeCellsCount());
    map_->grid().forEachCell([&](const scovox::Voxel& v, const Bonxai::CoordT& c) {
      scovox_msgs::msg::ScovoxVoxel vv;
      vv.position.x=c.x*P.resolution; vv.position.y=float(c.y*P.resolution); vv.position.z=float(c.z*P.resolution);
      vv.a_occ=v.a_occ; vv.a_free=v.a_free; vv.a_unk=v.a_unk;
      // Pick the top P.top_k strongest classes (sparse_add doesn't sort) and
      // fold dropped mass into a_unk so total semantic evidence is preserved.
      const auto top = scovox::selectTopKSemantics(v, P.top_k);
      for (size_t i = 0; i < top.kept_count; ++i) {
        scovox_msgs::msg::ScovoxSemanticEvidence e;
        e.class_id = top.kept[i].first;
        e.evidence_count = top.kept[i].second;
        vv.semantic_evidence.push_back(e);
      }
      vv.a_unk += top.dropped_mass;
      m.voxels.push_back(vv);
    });
    sm_pub_->publish(m); return {m.voxels.size(), 0.0};
  }
  // Publish only the voxels that have been touched since the last call. The
  // dscovox merger keys per-source grids by header.frame_id and overwrites
  // the matching voxels per binary; voxels not in the binary are kept, so
  // the merger's view stays exactly in sync with this robot's persistent
  // grid as long as it sees every dirty voxel at least once.
  //
  // To handle a fresh dscovox connecting after this node has already started,
  // we detect subscriber-count transitions from 0 to >0 and re-mark every
  // non-prior cell as dirty so the next publish carries a full snapshot.
  std::pair<size_t,double> publishBinaryMap() {
    if (!bin_pub_) return {0, 0};
    if (use_split_) {
      return wire_format_v3_ ? publishBinaryMapV3() : publishBinaryMapV2();
    }

    const size_t cur_sub = bin_pub_->get_subscription_count();
    if (cur_sub > prev_sub_count_) {
      // New subscriber appeared — flag everything non-prior so they get a
      // complete snapshot, not just deltas observed since startup.
      map_->grid().forEachCell([&](const scovox::Voxel& v, const Bonxai::CoordT& c) {
        if (v.a_occ > 1.01f || v.a_free > 1.01f) dirty_.insert(c);
      });
    }
    prev_sub_count_ = cur_sub;

    if (cur_sub == 0) return {0, 0};   // no point shipping; keep dirty_
    if (dirty_.empty()) return {0, 0}; // nothing changed since last publish

    auto acc = map_->grid().createAccessor();
    std::vector<scovox::ScovoxBinarySerializer::CoordVoxelPair> vps;
    vps.reserve(dirty_.size());
    for (const auto& coord : dirty_) {
      auto* cell = acc.value(coord, false);
      if (!cell) continue;
      // Drop voxels that are still at the prior — they carry no information.
      if (cell->a_occ <= 1.01f && cell->a_free <= 1.01f) continue;
      scovox::ScovoxBinarySerializer::CoordVoxelPair cv;
      cv.x = coord.x; cv.y = coord.y; cv.z = coord.z;
      cv.a_occ = cell->a_occ; cv.a_free = cell->a_free; cv.a_unk = cell->a_unk;
      for (int si = 0; si < scovox::K_TOP; ++si)
        if (cell->sem_cnt[si] > 0)
          cv.semantics.emplace_back((uint16_t)cell->sem_cls[si], cell->sem_cnt[si]);
      vps.push_back(std::move(cv));
    }

    auto data = scovox::ScovoxBinarySerializer::serializeIncremental(
      float(map_->params().resolution), vps, map_->params().top_k);
    auto comp = scovox::ScovoxBinarySerializer::compressLZ4(data);
    scovox_msgs::msg::ScovoxMapBinary bin;
    bin.header.stamp = get_clock()->now();
    bin.header.frame_id = int_frame_;
    bin.version = 1;
#if __BYTE_ORDER__==__ORDER_LITTLE_ENDIAN__
    bin.little_endian = true;
#else
    bin.little_endian = false;
#endif
    if (!comp.empty()) bin.data = std::move(comp);
    else bin.data.assign(reinterpret_cast<const uint8_t*>(data.data()),
                         reinterpret_cast<const uint8_t*>(data.data()) + data.size());
    double mb = double(bin.data.size()) / (1024.0 * 1024.0);
    bin_pub_->publish(std::move(bin));
    dirty_.clear();
    return {vps.size(), mb};
  }

  // Step 8 (D2/D8) — split-grid v2 binary publish path. Drains touched
  // TSDF + SemBeta coords from the split substrate, reads each voxel's
  // current state, builds a BinarySerializerV2::Frame, optionally elides
  // the TSDF section per share_tsdf_, LZ4-compresses, and publishes with
  // msg->version=2 so dscovox_node's onBinaryMapV2 router accepts it.
  //
  // share_tsdf_=false (the v2 default): tsdf_count=0, no TSDF deltas
  // serialized — receiver gets SemBeta-only frame, ~35% wire savings vs
  // dual-stream. share_tsdf_=true: full dual stream for fused-geometry
  // consensus on the receiver side.
  std::pair<size_t,double> publishBinaryMapV2() {
    if (!bin_pub_) return {0, 0};

    const size_t cur_sub = bin_pub_->get_subscription_count();
    bool snapshot = false;
    if (cur_sub > prev_sub_count_) {
      // New subscriber → flag every non-prior voxel for emission so they
      // get a full snapshot instead of just post-subscribe deltas. Same
      // contract as the legacy v1 path.
      snapshot = true;
    }
    prev_sub_count_ = cur_sub;
    if (cur_sub == 0) {
      // Drop the touched buffers when nobody is listening — otherwise
      // they'd grow unbounded. Equivalent to legacy `dirty_.clear()`
      // skip-on-no-subs behaviour.
      (void)split_map_->drainTouchedTsdf();
      (void)split_map_->drainTouchedSemDir();
      return {0, 0};
    }

    scovox::BinarySerializerV2::Frame frame;
    frame.resolution = static_cast<float>(split_map_->resolution());

    // ----- TSDF section (elided when share_tsdf_=false) -----
    if (share_tsdf_) {
      auto& tsdf_grid = split_map_->tsdf().grid();
      auto tacc = tsdf_grid.createAccessor();
      auto emit_tsdf = [&](const scovox::TsdfVoxel& v, const Bonxai::CoordT& c) {
        if (v.weight <= 0.f) return;  // unobserved
        scovox::BinarySerializerV2::TsdfDelta d;
        d.coord = c;
        d.data = v;
        frame.tsdf_deltas.push_back(d);
      };
      if (snapshot) {
        // Snapshot: walk every active TSDF voxel.
        tsdf_grid.forEachCell(emit_tsdf);
        (void)split_map_->drainTouchedTsdf();  // discard (snapshot supersedes)
      } else {
        for (const auto& c : split_map_->drainTouchedTsdf()) {
          if (auto* v = tacc.value(c, false)) emit_tsdf(*v, c);
        }
      }
    } else {
      // Drain the touched buffer even when not emitting so it doesn't
      // accumulate; receiver elision is the share_tsdf=false contract.
      (void)split_map_->drainTouchedTsdf();
    }

    // ----- SemDir section (always emitted) -----
    // Substrate is now SemDirVoxel (unified Dirichlet, Step 7.5). The v2 wire
    // format still expects SemBetaVoxel-shaped deltas; we project the SemDir
    // state into the legacy Beta layout here. The projection is lossless on
    // p_occ and per-class evidence:
    //   a_occ  ≈ s_occ()      (Σ top-K + OTHER)
    //   a_free ≈ alpha_free
    //   a_unk  ≈ alpha_other
    //   sem_cnt[i]/sem_cls[i] ← cnt[i]/cls[i]
    // TODO Step 8: route this through BinarySerializerV3 directly (skip the
    // SemDir → SemBeta projection — the v3 wire format carries SemDirVoxel
    // as-is at 32 B/voxel vs v2's 37 B SemBeta).
    {
      auto& sem_grid = split_map_->semdir().grid();
      auto sacc = sem_grid.createAccessor();
      auto emit_sem = [&](const scovox::SemDirVoxel& v, const Bonxai::CoordT& c) {
        // Drop voxels still at the symmetric Dirichlet prior with no
        // accumulated evidence — they carry no information.
        const float prior_s_occ  = v.s_occ() - v.alpha_other;  // K_TOP slot priors
        (void)prior_s_occ;
        const bool at_prior =
            (v.alpha_free <= scovox::kDefaultDirichletPrior + 1e-4f) &&
            (v.alpha_other <= 12.f * scovox::kDefaultDirichletPrior + 1e-4f);
        if (at_prior) {
          bool any_sem = false;
          for (int i = 0; i < scovox::K_TOP; ++i)
            if (v.cls[i] != 0xFFFF) { any_sem = true; break; }
          if (!any_sem) return;
        }
        scovox::BinarySerializerV2::SemBetaDelta d;
        d.coord = c;
        scovox::SemBetaVoxel proj{};
        proj.a_occ  = v.s_occ();
        proj.a_free = v.alpha_free;
        proj.a_unk  = v.alpha_other;
        for (int i = 0; i < scovox::K_TOP; ++i) {
          proj.sem_cnt[i] = v.cnt[i];
          proj.sem_cls[i] = v.cls[i];
        }
        d.data = proj;
        frame.sembeta_deltas.push_back(d);
      };
      if (snapshot) {
        sem_grid.forEachCell(emit_sem);
        (void)split_map_->drainTouchedSemDir();
      } else {
        for (const auto& c : split_map_->drainTouchedSemDir()) {
          if (auto* v = sacc.value(c, false)) emit_sem(*v, c);
        }
      }
    }

    if (frame.tsdf_deltas.empty() && frame.sembeta_deltas.empty()) {
      return {0, 0};  // nothing to ship; touched buffers already drained
    }

    scovox::BinarySerializerV2::Options opts;
    opts.share_tsdf = share_tsdf_;
    auto data = scovox::BinarySerializerV2::serialize(frame, opts);
    auto comp = scovox::ScovoxBinarySerializer::compressLZ4(data);

    scovox_msgs::msg::ScovoxMapBinary bin;
    bin.header.stamp = get_clock()->now();
    bin.header.frame_id = int_frame_;
    bin.version = 2;  // v2 envelope — dscovox_node onBinaryMapV2 routes on this
#if __BYTE_ORDER__==__ORDER_LITTLE_ENDIAN__
    bin.little_endian = true;
#else
    bin.little_endian = false;
#endif
    if (!comp.empty()) bin.data = std::move(comp);
    else bin.data.assign(reinterpret_cast<const uint8_t*>(data.data()),
                         reinterpret_cast<const uint8_t*>(data.data()) + data.size());
    const size_t emitted = frame.tsdf_deltas.size() + frame.sembeta_deltas.size();
    double mb = double(bin.data.size()) / (1024.0 * 1024.0);
    bin_pub_->publish(std::move(bin));
    return {emitted, mb};
  }

  // Step 8 — split-grid v3 binary publish path. Mirrors publishBinaryMapV2's
  // touched-drain / snapshot-on-resub / share_tsdf elision contract, but
  // serialises through BinarySerializerV3 directly from SemDirVoxel state
  // (no SemBeta projection). Header carries num_classes + alpha_0 so the
  // receiver can rebuild the symmetric Dirichlet prior without sharing
  // launch params.
  //
  // Wire-format choice is the launch-time `wire_format` param (v2|v3);
  // v3 stays opt-in until the dscovox consensus node lands an
  // onBinaryMapV3 handler (otherwise v3 frames just get dropped by the
  // receiver's "Bad version" guard).
  std::pair<size_t,double> publishBinaryMapV3() {
    if (!bin_pub_) return {0, 0};

    const size_t cur_sub = bin_pub_->get_subscription_count();
    bool snapshot = false;
    if (cur_sub > prev_sub_count_) snapshot = true;
    prev_sub_count_ = cur_sub;
    if (cur_sub == 0) {
      // Drain (drop) touched buffers so they don't grow unbounded while
      // nobody listens. Matches v2 behaviour.
      (void)split_map_->drainTouchedTsdf();
      (void)split_map_->drainTouchedSemDir();
      return {0, 0};
    }

    scovox::BinarySerializerV3::Frame frame;
    frame.resolution  = static_cast<float>(split_map_->resolution());
    frame.num_classes = static_cast<uint16_t>(num_classes_);
    frame.alpha_0     = alpha_0_;

    // ----- TSDF section (elided when share_tsdf_=false) -----
    if (share_tsdf_) {
      auto& tsdf_grid = split_map_->tsdf().grid();
      auto tacc = tsdf_grid.createAccessor();
      auto emit_tsdf = [&](const scovox::TsdfVoxel& v, const Bonxai::CoordT& c) {
        if (v.weight <= 0.f) return;
        scovox::BinarySerializerV3::TsdfDelta d;
        d.coord = c;
        d.data  = v;
        frame.tsdf_deltas.push_back(d);
      };
      if (snapshot) {
        tsdf_grid.forEachCell(emit_tsdf);
        (void)split_map_->drainTouchedTsdf();
      } else {
        for (const auto& c : split_map_->drainTouchedTsdf()) {
          if (auto* v = tacc.value(c, false)) emit_tsdf(*v, c);
        }
      }
    } else {
      (void)split_map_->drainTouchedTsdf();
    }

    // ----- SemDir section (always emitted, SemDirVoxel-native) -----
    {
      auto& sem_grid = split_map_->semdir().grid();
      auto sacc = sem_grid.createAccessor();
      // "At prior" check mirrors the v2 path: alpha_free still ≈ α_0, the
      // OTHER bucket still ≈ (C − K) · α_0, and no slot has filled. Such
      // voxels carry no posterior information and stay off the wire.
      const float other_prior = static_cast<float>(num_classes_ - scovox::K_TOP) * alpha_0_;
      auto emit_sem = [&](const scovox::SemDirVoxel& v, const Bonxai::CoordT& c) {
        const bool at_prior =
            (v.alpha_free  <= alpha_0_     + 1e-4f) &&
            (v.alpha_other <= other_prior  + 1e-4f);
        if (at_prior) {
          bool any_sem = false;
          for (int i = 0; i < scovox::K_TOP; ++i)
            if (v.cls[i] != 0xFFFF) { any_sem = true; break; }
          if (!any_sem) return;
        }
        scovox::BinarySerializerV3::SemDirDelta d;
        d.coord = c;
        d.data  = v;
        frame.semdir_deltas.push_back(d);
      };
      if (snapshot) {
        sem_grid.forEachCell(emit_sem);
        (void)split_map_->drainTouchedSemDir();
      } else {
        for (const auto& c : split_map_->drainTouchedSemDir()) {
          if (auto* v = sacc.value(c, false)) emit_sem(*v, c);
        }
      }
    }

    if (frame.tsdf_deltas.empty() && frame.semdir_deltas.empty()) {
      return {0, 0};
    }

    scovox::BinarySerializerV3::Options opts;
    opts.share_tsdf = share_tsdf_;
    auto data = scovox::BinarySerializerV3::serialize(frame, opts);
    auto comp = scovox::ScovoxBinarySerializer::compressLZ4(data);

    scovox_msgs::msg::ScovoxMapBinary bin;
    bin.header.stamp    = get_clock()->now();
    bin.header.frame_id = int_frame_;
    bin.version         = 3;   // v3 envelope — dscovox onBinaryMapV3 routes
#if __BYTE_ORDER__==__ORDER_LITTLE_ENDIAN__
    bin.little_endian = true;
#else
    bin.little_endian = false;
#endif
    if (!comp.empty()) bin.data = std::move(comp);
    else bin.data.assign(reinterpret_cast<const uint8_t*>(data.data()),
                         reinterpret_cast<const uint8_t*>(data.data()) + data.size());
    const size_t emitted = frame.tsdf_deltas.size() + frame.semdir_deltas.size();
    double mb = double(bin.data.size()) / (1024.0 * 1024.0);
    bin_pub_->publish(std::move(bin));
    return {emitted, mb};
  }

  // Publish planning_map as a 2D projection of the persistent voxel grid.
  //
  // mode=persistent: fixed (plan_ox_, plan_oy_, plan_sz_) envelope.
  // mode=rolling:    robot-centered crop of side plan_window_size_m_, snapped
  //                  to grid resolution to avoid sub-cell jitter as the robot
  //                  moves. The underlying grid is unchanged — only the
  //                  publication is windowed.
  void publishPlanningMap() {
    if (!pl_pub_ || pl_pub_->get_subscription_count() == 0) return;

    double ox = plan_ox_, oy = plan_oy_, sz = plan_sz_;
    if (mode_ == "rolling") {
      Eigen::Vector3f O;
      try {
        auto t = tf_buffer_.lookupTransform(
          int_frame_, base_frame_,
          rclcpp::Time(0), rclcpp::Duration::from_seconds(0.05));
        O << t.transform.translation.x, t.transform.translation.y, t.transform.translation.z;
      } catch (const std::exception& e) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
          "publishPlanningMap: TF lookup failed (%s); skipping rolling crop", e.what());
        return;
      }
      sz = plan_window_size_m_;
      const double r = plan_res_;
      ox = std::floor((O.x() - sz * 0.5) / r) * r;
      oy = std::floor((O.y() - sz * 0.5) / r) * r;
    }

    const int w = std::max(1, (int)std::round(sz / plan_res_));
    const int h = w;
    nav_msgs::msg::OccupancyGrid g;
    g.header.stamp = get_clock()->now();
    g.header.frame_id = int_frame_;
    g.info.resolution = float(plan_res_);
    g.info.width = w;
    g.info.height = h;
    g.info.origin.position.x = ox;
    g.info.origin.position.y = oy;
    g.info.origin.orientation.w = 1.0;
    g.data.assign(w * h, -1);
    map_->grid().forEachCell([&](const scovox::Voxel& v, const Bonxai::CoordT& c) {
      auto p = map_->grid().coordToPos(c);
      if (p.z < plan_zmin_ || p.z > plan_zmax_) return;
      int gx = int(std::floor((p.x - ox) / plan_res_));
      int gy = int(std::floor((p.y - oy) / plan_res_));
      if (gx < 0 || gy < 0 || gx >= w || gy >= h) return;
      int i = gy * w + gx;
      if (v.p_occ() >= float(min_occ_)) g.data[i] = 100;
      else if (g.data[i] != 100) g.data[i] = 0;
    });
    int ic=int(std::ceil(plan_infl_/plan_res_)); int ic2=ic*ic;
    if (ic>0) { auto inf=g.data; for (int y=0;y<h;++y) for (int x=0;x<w;++x) { if (g.data[y*w+x]!=100) continue;
      for (int dy=-ic;dy<=ic;++dy) for (int dx=-ic;dx<=ic;++dx) { if (dx*dx+dy*dy>ic2) continue;
        int nx=x+dx,ny=y+dy;
        if (nx>=0&&nx<w&&ny>=0&&ny<h) inf[ny*w+nx]=100; } } g.data=std::move(inf); }
    pl_pub_->publish(g);
  }
  // Caller must hold map_mtx_ (shared). The timer body locks once for both
  // publishScovoxMap and publishPointCloud so they see the same map state.
  void publishPointCloud() {
    if (use_split_) { publishPointCloudV2(); return; }
    if (pc_pub_->get_subscription_count()==0) return;
    sensor_msgs::msg::PointCloud2 cl; cl.header.frame_id=int_frame_; cl.header.stamp=get_clock()->now();
    cl.height=1; cl.is_dense=true; cl.is_bigendian=false;
    sensor_msgs::PointCloud2Modifier mod(cl);
    // Layout: 11 base fields (xyz + rgb + occupancy/semantic summary + Beta a_occ/a_free)
    // followed by E5.1 raw mass fields (a_unk, sem_cnt0, sem_cls0, sem_cnt1, sem_cls1).
    // Adding the raw counts is what lets verify_mass_conservation.py check
    // a_unk + Σ sem_cnt invariance externally.
    // 2026-05-10: K_TOP sweep (P6.1/P6.2) — emit slot 0 + slot 1 always.
    // For K=1 the second slot is dummy (cnt=0, cls=0xFFFF). For K>2 the
    // emission is the first 2 Misra-Gries slots in registration order
    // (not necessarily top-2 by count); semantic_class is still computed
    // via argmaxClassConfidence() which walks all K, so mIoU is unaffected.
    static_assert(scovox::K_TOP >= 1, "publishPointCloud requires at least 1 sparse slot");
    mod.setPointCloud2Fields(16, "x",1,sensor_msgs::msg::PointField::FLOAT32, "y",1,sensor_msgs::msg::PointField::FLOAT32,
      "z",1,sensor_msgs::msg::PointField::FLOAT32, "rgb",1,sensor_msgs::msg::PointField::FLOAT32,
      "occupancy_prob",1,sensor_msgs::msg::PointField::FLOAT32, "semantic_class",1,sensor_msgs::msg::PointField::UINT8,
      "semantic_confidence",1,sensor_msgs::msg::PointField::FLOAT32,
      "posterior_variance",1,sensor_msgs::msg::PointField::FLOAT32,
      "eig",1,sensor_msgs::msg::PointField::FLOAT32,
      "a_occ",1,sensor_msgs::msg::PointField::FLOAT32,
      "a_free",1,sensor_msgs::msg::PointField::FLOAT32,
      "a_unk",1,sensor_msgs::msg::PointField::FLOAT32,
      "sem_cnt0",1,sensor_msgs::msg::PointField::FLOAT32,
      "sem_cls0",1,sensor_msgs::msg::PointField::UINT16,
      "sem_cnt1",1,sensor_msgs::msg::PointField::FLOAT32,
      "sem_cls1",1,sensor_msgs::msg::PointField::UINT16);
    // Skip voxels still at the Beta prior (a_occ=a_free=1.0 → p_occ=0.5).
    // The fused TSDF walk creates behind-surface band voxels with TSDF-only
    // evidence (Beta untouched); they must NOT pollute the occupancy cloud
    // when min_occ_=0.5 (which uses `>=`). The TSDF zero-crossing publisher
    // already exposes those voxels separately.
    auto has_beta_evidence=[](const scovox::Voxel& v){
      return v.a_occ > 1.001f || v.a_free > 1.001f;
    };
    size_t cnt=0;
    auto cntF=[&](const scovox::Voxel& v, const Bonxai::CoordT&){
      if (v.p_occ()>=min_occ_ && has_beta_evidence(v)) ++cnt; };
    map_->grid().forEachCell(cntF); map_->transientGrid().forEachCell(cntF); mod.resize(cnt);
    sensor_msgs::PointCloud2Iterator<float> ix(cl,"x"),iy(cl,"y"),iz(cl,"z"),ir(cl,"rgb"),ip(cl,"occupancy_prob"),ic2(cl,"semantic_confidence"),
      iv(cl,"posterior_variance"),ie(cl,"eig"),iao(cl,"a_occ"),iaf(cl,"a_free"),
      iau(cl,"a_unk"),isn0(cl,"sem_cnt0"),isn1(cl,"sem_cnt1");
    sensor_msgs::PointCloud2Iterator<uint8_t> icl(cl,"semantic_class");
    sensor_msgs::PointCloud2Iterator<uint16_t> isc0(cl,"sem_cls0"),isc1(cl,"sem_cls1");
    const auto& P=map_->params();
    auto emit=[&](const scovox::Voxel& v, const Bonxai::CoordT& co, const scovox::Map::Grid& sg) {
      float pr=v.p_occ(); if (pr<min_occ_ || !has_beta_evidence(v)) return;
      auto ps=sg.coordToPos(co); uint8_t bc=0; float cf=0,r=1,g=1,b=1;
      if (v.a0()>0 && scovox::K_TOP>0) {
        // Hutter-framework (K+1)-Dirichlet posterior probability of the
        // argmax tracked class — consistent with semanticEntropy /
        // semanticVariance. See node_utils.hpp::argmaxClassConfidence.
        const auto [best_cls, p_best] = scovox::argmaxClassConfidence(v);
        bc = static_cast<uint8_t>(best_cls);
        cf = p_best;
        float vis_gate = sem_vis_thresh_ >= 0 ? (float)sem_vis_thresh_ : P.semantic_occ_gate;
        if (cf>=vis_gate && bc<sem_col_.size()) { auto& c=sem_col_[bc]; r=c.r; g=c.g; b=c.b; }
      }
      uint32_t rp=(uint32_t(r*255)<<16)|(uint32_t(g*255)<<8)|uint32_t(b*255);
      *ix=ps.x; *iy=ps.y; *iz=ps.z; *ir=*reinterpret_cast<float*>(&rp); *ip=pr; *icl=bc; *ic2=cf;
      *iv=scovox::variance(v); *ie=scovox::expectedInformationGain(v);
      *iao=v.a_occ; *iaf=v.a_free;
      *iau=v.a_unk;
      *isn0=v.sem_cnt[0]; *isc0=v.sem_cls[0];
      // K=1: slot 1 is dummy. K>=2: emit slot 1 (first 2 Misra-Gries slots).
      if constexpr (scovox::K_TOP >= 2) {
        *isn1=v.sem_cnt[1]; *isc1=v.sem_cls[1];
      } else {
        *isn1=0.0f; *isc1=static_cast<uint16_t>(0xFFFF);
      }
      ++ix;++iy;++iz;++ir;++ip;++icl;++ic2;++iv;++ie;++iao;++iaf;
      ++iau;++isn0;++isc0;++isn1;++isc1;
    };
    const auto& gr=map_->grid(); const auto& tg=map_->transientGrid();
    gr.forEachCell([&](const scovox::Voxel& v, const Bonxai::CoordT& c){ emit(v,c,gr); });
    tg.forEachCell([&](const scovox::Voxel& v, const Bonxai::CoordT& c){ emit(v,c,tg); });
    pc_pub_->publish(cl);
  }

  // Step 8 (D4) — split-mode pointcloud publisher. Walks the SemBeta
  // grid (the prediction-set grid post-refactor — has_beta_evidence
  // gates apply directly to a_occ/a_free) and emits the same 16-field
  // schema as publishPointCloud above so pointcloud_to_npz.py / RViz /
  // eval scripts need no version-aware routing. The D6 helper overloads
  // (variance / EIG / argmaxClassConfidence) keep the per-point body
  // byte-identical to the legacy version up to voxel type. No transient
  // grid in split mode (legacy-only feature, retired by D12 framing).
  void publishPointCloudV2() {
    if (!pc_pub_ || !split_map_ || pc_pub_->get_subscription_count() == 0) return;
    sensor_msgs::msg::PointCloud2 cl;
    cl.header.frame_id = int_frame_;
    cl.header.stamp = get_clock()->now();
    cl.height = 1; cl.is_dense = true; cl.is_bigendian = false;
    sensor_msgs::PointCloud2Modifier mod(cl);
    // 2026-05-10: same K-tolerance as publishPointCloud above (see comment).
    static_assert(scovox::K_TOP >= 1, "publishPointCloudV2 requires at least 1 sparse slot");
    mod.setPointCloud2Fields(16,
      "x",1,sensor_msgs::msg::PointField::FLOAT32, "y",1,sensor_msgs::msg::PointField::FLOAT32,
      "z",1,sensor_msgs::msg::PointField::FLOAT32, "rgb",1,sensor_msgs::msg::PointField::FLOAT32,
      "occupancy_prob",1,sensor_msgs::msg::PointField::FLOAT32, "semantic_class",1,sensor_msgs::msg::PointField::UINT8,
      "semantic_confidence",1,sensor_msgs::msg::PointField::FLOAT32,
      "posterior_variance",1,sensor_msgs::msg::PointField::FLOAT32,
      "eig",1,sensor_msgs::msg::PointField::FLOAT32,
      "a_occ",1,sensor_msgs::msg::PointField::FLOAT32,
      "a_free",1,sensor_msgs::msg::PointField::FLOAT32,
      "a_unk",1,sensor_msgs::msg::PointField::FLOAT32,
      "sem_cnt0",1,sensor_msgs::msg::PointField::FLOAT32,
      "sem_cls0",1,sensor_msgs::msg::PointField::UINT16,
      "sem_cnt1",1,sensor_msgs::msg::PointField::FLOAT32,
      "sem_cls1",1,sensor_msgs::msg::PointField::UINT16);
    // Project SemDir → SemBeta on the fly; "has evidence beyond prior" gate
    // becomes "s_occ > 14·α_0 + ε" / "alpha_free > α_0 + ε".
    auto has_evidence = [](const scovox::SemDirVoxel& v) {
      constexpr float prior_s_occ = 14.f * scovox::kDefaultDirichletPrior; // (C-K)·α + K·α at C=14, K=2
      return v.s_occ() > prior_s_occ + 1e-3f ||
             v.alpha_free > scovox::kDefaultDirichletPrior + 1e-3f;
    };
    const auto& g = split_map_->semdir().grid();
    size_t cnt = 0;
    g.forEachCell([&](const scovox::SemDirVoxel& v, const Bonxai::CoordT&) {
      if (v.p_occ() >= min_occ_ && has_evidence(v)) ++cnt;
    });
    if (!cnt) { pc_pub_->publish(cl); return; }
    mod.resize(cnt);
    sensor_msgs::PointCloud2Iterator<float> ix(cl,"x"), iy(cl,"y"), iz(cl,"z"), ir(cl,"rgb"),
      ip(cl,"occupancy_prob"), ic2(cl,"semantic_confidence"),
      iv(cl,"posterior_variance"), ie(cl,"eig"),
      iao(cl,"a_occ"), iaf(cl,"a_free"),
      iau(cl,"a_unk"), isn0(cl,"sem_cnt0"), isn1(cl,"sem_cnt1");
    sensor_msgs::PointCloud2Iterator<uint8_t>  icl(cl,"semantic_class");
    sensor_msgs::PointCloud2Iterator<uint16_t> isc0(cl,"sem_cls0"), isc1(cl,"sem_cls1");
    // Visualisation colour gate: prefer explicit override, else fall back
    // to the SemBeta-default 0.5 (legacy used P.semantic_occ_gate which
    // lives on the legacy Map::Params; in split mode it's a node-side
    // sensor filter consumed before integrate, see declareNodeParams).
    const float vis_gate = sem_vis_thresh_ >= 0 ? (float)sem_vis_thresh_ : 0.5f;
    g.forEachCell([&](const scovox::SemDirVoxel& v_sd, const Bonxai::CoordT& co) {
      float pr = v_sd.p_occ();
      if (pr < min_occ_ || !has_evidence(v_sd)) return;
      // Project once; reuse for argmax / variance / EIG / wire fields.
      const scovox::SemBetaVoxel v = projectSemDirToSemBeta(v_sd);
      auto ps = g.coordToPos(co);
      uint8_t bc = 0; float cf = 0, r = 1, gg = 1, b = 1;
      if (v.a0() > 0 && scovox::K_TOP > 0) {
        const auto [best_cls, p_best] = scovox::argmaxClassConfidence(v);
        bc = static_cast<uint8_t>(best_cls);
        cf = p_best;
        if (cf >= vis_gate && bc < sem_col_.size()) { auto& c = sem_col_[bc]; r = c.r; gg = c.g; b = c.b; }
      }
      uint32_t rp = (uint32_t(r * 255) << 16) | (uint32_t(gg * 255) << 8) | uint32_t(b * 255);
      *ix = ps.x; *iy = ps.y; *iz = ps.z; *ir = *reinterpret_cast<float*>(&rp);
      *ip = pr; *icl = bc; *ic2 = cf;
      *iv = scovox::variance(v); *ie = scovox::expectedInformationGain(v);
      *iao = v.a_occ; *iaf = v.a_free;
      *iau = v.a_unk;
      *isn0 = v.sem_cnt[0]; *isc0 = v.sem_cls[0];
      if constexpr (scovox::K_TOP >= 2) {
        *isn1 = v.sem_cnt[1]; *isc1 = v.sem_cls[1];
      } else {
        *isn1 = 0.0f; *isc1 = static_cast<uint16_t>(0xFFFF);
      }
      ++ix; ++iy; ++iz; ++ir; ++ip; ++icl; ++ic2; ++iv; ++ie; ++iao; ++iaf;
      ++iau; ++isn0; ++isc0; ++isn1; ++isc1;
    });
    pc_pub_->publish(cl);
  }

  // Publish a thin shell at the TSDF zero-crossing. Caller must hold
  // map_mtx_ (shared); no transient grid involved (TSDF is persistent-only).
  // Uses 3-axis sign-change with sub-voxel interpolation (extractZeroCrossing)
  // and emits the dominant semantic class per surface point.
  void publishTSDFPointCloud() {
    if (use_split_) { publishTSDFPointCloudV2(); return; }
    if (!tsdf_pub_ || tsdf_pub_->get_subscription_count() == 0) return;
    if (map_->params().sdf_trunc <= 0.f) return;

    auto points = scovox::extractZeroCrossing(
        map_->grid(), (float)min_tsdf_w_, map_->params().resolution);
    if (points.empty()) return;

    sensor_msgs::msg::PointCloud2 cl;
    cl.header.frame_id = int_frame_;
    cl.header.stamp = get_clock()->now();
    cl.height = 1;
    cl.is_dense = true;
    cl.is_bigendian = false;
    sensor_msgs::PointCloud2Modifier mod(cl);
    mod.setPointCloud2Fields(5,
        "x", 1, sensor_msgs::msg::PointField::FLOAT32,
        "y", 1, sensor_msgs::msg::PointField::FLOAT32,
        "z", 1, sensor_msgs::msg::PointField::FLOAT32,
        "distance", 1, sensor_msgs::msg::PointField::FLOAT32,
        "semantic_class", 1, sensor_msgs::msg::PointField::UINT16);
    mod.resize(points.size());

    sensor_msgs::PointCloud2Iterator<float> ix(cl,"x"), iy(cl,"y"), iz(cl,"z"), id(cl,"distance");
    sensor_msgs::PointCloud2Iterator<uint16_t> ic(cl,"semantic_class");
    for (const auto& p : points) {
      *ix = p.position.x(); *iy = p.position.y(); *iz = p.position.z();
      *id = p.distance; *ic = p.semantic_class;
      ++ix; ++iy; ++iz; ++id; ++ic;
    }
    tsdf_pub_->publish(cl);
  }

  // Step 8 (D5) — split-mode TSDF zero-crossing publisher. Walks
  // TsdfMap for the surface geometry then runs labelPointCloud against
  // SemBeta to attach the per-point semantic class. The cross-grid join
  // uses the 0xFFFF sentinel where SemBeta has no voxel at the surface
  // coord (D5 — same convention labelMesh / extractZeroCrossing already
  // produce). 5-field schema matches the legacy publisher byte-for-byte.
  void publishTSDFPointCloudV2() {
    if (!tsdf_pub_ || !split_map_ || tsdf_pub_->get_subscription_count() == 0) return;
    const auto& tsdf_grid = split_map_->tsdf().grid();
    const double res = split_map_->resolution();
    if (res <= 0.0) return;

    auto points = scovox::extractZeroCrossing(
        tsdf_grid, (float)min_tsdf_w_, res);
    if (points.empty()) return;

    // Cross-grid label join — labelPointCloud queries SemBeta at each
    // surface vertex's anchor voxel and writes the argmax class
    // (sentinel 0xFFFF if absent). Mirrors mesh_labelling.hpp::labelMesh
    // for triangle-mesh consumers.
    std::vector<Eigen::Vector3f> positions;
    positions.reserve(points.size());
    for (const auto& p : points) positions.push_back(p.position);
    auto labels = scovox::labelPointCloud(positions, split_map_->semdir().grid());

    sensor_msgs::msg::PointCloud2 cl;
    cl.header.frame_id = int_frame_;
    cl.header.stamp = get_clock()->now();
    cl.height = 1; cl.is_dense = true; cl.is_bigendian = false;
    sensor_msgs::PointCloud2Modifier mod(cl);
    mod.setPointCloud2Fields(5,
        "x", 1, sensor_msgs::msg::PointField::FLOAT32,
        "y", 1, sensor_msgs::msg::PointField::FLOAT32,
        "z", 1, sensor_msgs::msg::PointField::FLOAT32,
        "distance", 1, sensor_msgs::msg::PointField::FLOAT32,
        "semantic_class", 1, sensor_msgs::msg::PointField::UINT16);
    mod.resize(points.size());
    sensor_msgs::PointCloud2Iterator<float> ix(cl,"x"), iy(cl,"y"), iz(cl,"z"), id(cl,"distance");
    sensor_msgs::PointCloud2Iterator<uint16_t> ic(cl,"semantic_class");
    for (size_t i = 0; i < points.size(); ++i) {
      const auto& p = points[i];
      *ix = p.position.x(); *iy = p.position.y(); *iz = p.position.z();
      *id = p.distance; *ic = labels[i];
      ++ix; ++iy; ++iz; ++id; ++ic;
    }
    tsdf_pub_->publish(cl);
  }

  void onExtractMesh(
      const scovox_msgs::srv::ExtractMesh::Request::SharedPtr rq,
      scovox_msgs::srv::ExtractMesh::Response::SharedPtr rs)
  {
    std::shared_lock<std::shared_mutex> lk(map_mtx_);
    if (!map_ || map_->params().sdf_trunc <= 0.f) {
      rs->vertex_count = 0;
      rs->triangle_count = 0;
      return;
    }

    auto mesh = scovox::extractMesh(
        map_->grid(), rq->min_weight, map_->params().resolution);
    rs->vertex_count = mesh.vertices.size();
    rs->triangle_count = mesh.triangles.size();

    if (!rq->output_path.empty() && !mesh.vertices.empty()) {
      std::ofstream ply(rq->output_path);
      if (ply.is_open()) {
        ply << "ply\nformat ascii 1.0\n";
        ply << "element vertex " << mesh.vertices.size() << "\n";
        ply << "property float x\nproperty float y\nproperty float z\n";
        ply << "element face " << mesh.triangles.size() << "\n";
        ply << "property list uchar int vertex_indices\n";
        ply << "end_header\n";
        for (const auto& v : mesh.vertices)
          ply << v.x() << " " << v.y() << " " << v.z() << "\n";
        for (const auto& t : mesh.triangles)
          ply << "3 " << t.x() << " " << t.y() << " " << t.z() << "\n";
        ply.close();
        rs->ply_path = rq->output_path;
      }
    }
  }
  std::string mode_, base_frame_, map_frame_, int_frame_, depth_topic_, di_topic_, seg_topic_, input_pc_topic_, robot_id_;
  bool dataset_mode_{false}, pointcloud_mode_{false};
  // Diagnostic-only: periodic per-grid memory/RSS logging. Off by default —
  // when on, scheduleMemUsage spawns a detached reader thread every ~10
  // frames and walks the whole grid. Enable via `log_mem_usage:=true`.
  bool mem_log_{false};
  // TF stability gate
  double startup_tf_stable_sec_{2.0}, startup_tf_jump_thresh_{0.5};
  bool tf_stable_{false}, tf_prev_valid_{false};
  rclcpp::Time tf_stable_since_;
  Eigen::Vector3f tf_prev_pos_{Eigen::Vector3f::Zero()};
  size_t frame_recv_{0};
  // E5.2 — per-frame eviction stats CSV writer.
  std::string evict_csv_path_;
  std::ofstream evict_csv_;
  uint64_t last_match_{0}, last_empty_{0}, last_evict_{0}, last_drop_{0};
  // Dataset mode: cache unmatched depth/seg by frame index
  std::unordered_map<uint16_t, sensor_msgs::msg::Image::ConstSharedPtr> ds_depth_cache_, ds_seg_cache_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr ds_depth_sub_, ds_seg_sub_;
  int max_sem_, stride_{1}; double min_d_{0.1}, max_d_{10.0}, min_occ_, transient_decay_rate_{0.8}, sem_vis_thresh_{-1.0};
  // Soft-probability mode: directory of <frame>.topk flat-binary blobs, with
  // file names matching the low 16 bits of header.stamp.nanosec the replay
  // node sets (zero-padded to 6 digits). Empty = legacy hard-label path.
  std::string topk_probs_dir_;
  int topk_topk_max_{5};
  // Cache the most recently loaded frame so we don't read the file twice
  // (once for the hit lookup, once for the integration loop).
  uint16_t topk_cache_frame_{0xFFFF};
  bool     topk_cache_valid_{false};
  bool     topk_cache_is_image_{false};
  uint32_t topk_cache_n_{0};
  uint16_t topk_cache_h_{0}, topk_cache_w_{0};
  // Number of dense class slots per pixel/point. Slot j IS the SCovox
  // class id (slot 0 = unknown). Quantized to uint8 (×255) for storage.
  uint8_t  topk_cache_c_{0};
  std::vector<uint8_t> topk_cache_probs_;
  // Soft-prob loader telemetry — guards the silent-fallback footgun the
  // 5-s-throttled "topk: cannot open" warn alone cannot catch. Counted
  // per-frame in loadTopkForFrame; logged at shutdown so the smoke gate
  // can assert "topk dispatched on N frames" (where N matches the integrate
  // count) rather than guessing from mIoU alone.
  uint64_t topk_load_success_{0};
  uint64_t topk_load_failure_{0};
  bool     topk_first_load_logged_{false};
  bool trace_nr_{false}, pub_pc_, pub_plan_{false}, pub_tsdf_{true};
  double carve_band_{-1.0};
  double min_tsdf_w_{0.5};
  double plan_res_{0.2}, plan_sz_{80}, plan_ox_{-40}, plan_oy_{-40}, plan_zmin_{-1}, plan_zmax_{2}, plan_infl_{0};
  double plan_window_size_m_{20.0};
  std::atomic<bool> sm_dirty_{false};
  std::unordered_map<uint32_t,uint16_t> color_map_;
  std::unique_ptr<scovox::Map> map_;
  // Step 8 (D7) — split-grid v2 substrate. Allocated only when use_split_
  // is true; legacy `map_` stays null in that case to make accidental
  // legacy-path access null-deref loudly rather than silently produce an
  // empty parallel grid. Owns one TsdfMap + one SemBetaMap.
  std::unique_ptr<scovox::ScovoxMapSplit> split_map_;
  bool use_split_{false};
  bool share_tsdf_{false};
  bool fused_walker_{true};       // Step 12.10 — single-DDA hit-ray walker
  // Step 8 — SemDir dataset priors (use_split=true only). Defaults
  // match SemDirMap::Params; KITTI launches override via num_classes:=20.
  int   num_classes_{14};
  float alpha_0_{scovox::kDefaultDirichletPrior};
  // Step 8 — wire format selector. false = v2 (SemBeta-projected, current
  // production); true = v3 (SemDir-native, 20 B/voxel + header priors).
  bool  wire_format_v3_{false};
  std::string tsdf_dump_path_{};  // audit hook — see [tsdf_dump] memlog branch
  std::vector<std_msgs::msg::ColorRGBA> sem_col_;
  std::unordered_set<uint8_t> dyn_cls_;
  // Set of voxel coordinates touched by integration since the last binary
  // publish. Each entry corresponds to a voxel whose state may have changed
  // and needs to be re-shipped to dscovox. Cleared on successful publish.
  std::unordered_set<Bonxai::CoordT, CoordTHash, CoordTEqual> dirty_;
  // Last observed binary subscriber count. When this transitions from 0 to
  // >0 the next publish ships a full snapshot so a freshly-connected dscovox
  // sees this robot's complete current state, not just deltas since startup.
  size_t prev_sub_count_{0};
  sensor_msgs::msg::CameraInfo di_; std::atomic<bool> have_di_{false};
  mutable std::shared_mutex map_mtx_;  // protects map_, dirty_
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr di_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr input_pc_sub_;
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> d_sub_, s_sub_;
  using ISP = message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image,sensor_msgs::msg::Image>;
  std::shared_ptr<message_filters::Synchronizer<ISP>> sync_;
  rclcpp::TimerBase::SharedPtr sm_timer_;
  rclcpp::Publisher<scovox_msgs::msg::ScovoxMap>::SharedPtr sm_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pc_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr tsdf_pub_;
  rclcpp::Publisher<scovox_msgs::msg::ScovoxMapBinary>::SharedPtr bin_pub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr pl_pub_;
  rclcpp::Service<scovox_msgs::srv::ExtractMesh>::SharedPtr extract_mesh_srv_;
  tf2_ros::Buffer tf_buffer_; tf2_ros::TransformListener tf_listener_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SCovoxNode>());
  const uint64_t evict = scovox::g_sparse_evict_count.load(std::memory_order_relaxed);
  const uint64_t drop  = scovox::g_sparse_drop_count.load(std::memory_order_relaxed);
  const uint64_t total = evict + drop;
  std::fprintf(stderr,
      "[scovox_node] sparse_add K_TOP overflow: evict=%lu drop=%lu total_overflow=%lu "
      "(K_TOP=%d)\n",
      static_cast<unsigned long>(evict),
      static_cast<unsigned long>(drop),
      static_cast<unsigned long>(total),
      scovox::K_TOP);
  rclcpp::shutdown();
  return 0;
}
