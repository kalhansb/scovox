#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
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
#include <deque>
#include "scovox/map_interface.hpp"   // scovox::Params (launch-param plumbing)
#include "scovox/scovox_map_split.hpp"
#include "scovox/node_utils.hpp"
#include "scovox/topk_provider.hpp"
#include "scovox_msgs/msg/scovox_map.hpp"
#include "scovox_msgs/msg/scovox_voxel.hpp"
#include "scovox_msgs/msg/scovox_semantic_evidence.hpp"
#include "scovox_msgs/msg/scovox_map_binary.hpp"
#include "scovox/binary_serializer.hpp"
#include "scovox/lz4_codec.hpp"
#include "scovox/marching_cubes.hpp"
#include "scovox/mesh_labelling.hpp"
#include "scovox_msgs/srv/extract_mesh.hpp"

namespace enc = sensor_msgs::image_encodings;

class SCovoxNode : public rclcpp::Node {
public:
  SCovoxNode() : Node("scovox_node"), tf_buffer_(this->get_clock(), tf2::Duration(std::chrono::seconds(600))), tf_listener_(tf_buffer_) {
    auto P = declareMapParams();
    declareNodeParams();
    // Cache the launch param block. scovox::Params still carries the
    // node-level sensor filters (range_decay_length, min_range, max_range,
    // grazing_angle_threshold, semantic_occ_gate, resolution, top_k) that the
    // integration + publish paths read; with the legacy map_ object gone this
    // is their owner.
    map_params_ = P;
    // Per-source fusion weight profiles (consulted only when fuse_lidar_rgbd_).
    buildFusionProfiles(P);
    // Soft-probability (.topk) loader. Constructed here, after params, so it
    // captures the configured dir + class count; topk_->enabled() gates the
    // hot path in place of the old `!topk_probs_dir_.empty()` checks.
    topk_ = std::make_unique<scovox::TopkProvider>(
        get_logger(), get_clock(), topk_probs_dir_, max_sem_);
    // Raw launch sdf_trunc (0 == "TSDF off"). TsdfMap re-clamps a <=0 value
    // back up internally, so the ~/tsdf_pointcloud publisher, the
    // ~/extract_mesh service, and publishTSDFPointCloud gate on this cached
    // launch value rather than the (always-positive) TsdfMap::params().sdf_trunc.
    sdf_trunc_launch_ = P.sdf_trunc;
    // Split-grid substrate. ScovoxMapSplit::Params shares
    // (resolution, inner_bits, leaf_bits) across both grids; per-substrate
    // params (sdf_trunc, w_occ/w_free/kappa0, evidence_saturation, …) flow
    // through from launch params via TsdfMap::Params and SemSplitMap::Params.
    scovox::ScovoxMapSplit::Params SP;
    SP.resolution = P.resolution;
    SP.inner_bits = P.inner_bits;
    SP.leaf_bits  = P.leaf_bits;
    // TsdfMap: SLIM-VDB-equivalent defaults; only sdf_trunc + space_carving
    // carry over from P. weighting_function defaults to constant(1).
    SP.tsdf.sdf_trunc     = P.sdf_trunc;
    SP.tsdf.space_carving = P.tsdf_space_carving;
    // enable_tsdf:false sets sdf_trunc=0, but TsdfMap::sanitise re-clamps that
    // back up to 0.15 — silently keeping the fused walker's per-voxel TSDF band
    // writes (TsdfVoxel alloc + Curless-Levoy average + touched push) running
    // even though the occupancy/LiDAR config never reads the TsdfMap grid
    // (tsdf_pub_ null, no extract_mesh, share_tsdf_=false). Thread the real
    // intent through so those dead writes are skipped. Default true keeps every
    // TSDF-on config — and all scovox_core gtests, which build with
    // sdf_trunc=0.15 — byte-identical; only the (unread) TsdfMap grid changes.
    SP.tsdf_enabled = (sdf_trunc_launch_ > 0.f);
    // SemSplitMap (de-unified BetaVoxel ∥ DirVoxel substrate): every
    // Bayesian / sparse-Dirichlet knob from launch P maps 1:1.
    // semantic_occ_gate / min_range / max_range / grazing_angle_threshold are
    // node-level sensor filters consumed BEFORE integrateHit so they are not
    // mirrored. evidence_saturation widens uint16→float.
    SP.semsplit.w_free                  = P.w_free;
    SP.semsplit.w_occ                   = P.w_occ;
    SP.semsplit.kappa0                  = P.kappa0;
    SP.semsplit.carve_skip_occ_threshold = P.carve_skip_occ_threshold;
    SP.semsplit.evidence_saturation     = static_cast<float>(P.evidence_saturation);
    SP.semsplit.dirichlet_min_p_occ     = P.dirichlet_min_p_occ;
    SP.semsplit.range_decay_length      = static_cast<float>(P.range_decay_length);
    SP.semsplit.semantic_mode           = P.semantic_mode;
    // num_classes / alpha_0 — dataset-dependent priors that govern the
    // OTHER bucket's prior mass `(C − K_TOP)·α_0`. Wrong num_classes shifts
    // the semantic prior and the eviction capacity (KITTI=20 vs NYU13=14).
    // Defaults match SemSplitMap::Params defaults (NYU13 / 0.01). KITTI
    // launches override num_classes:=20.
    SP.semsplit.num_classes             = num_classes_;
    SP.semsplit.alpha_0                 = alpha_0_;
    // Step 12.10 (2026-05-09) — fused single-DDA ray walker. Default
    // true; set false via `fused_walker:=false` launch arg for A/B
    // parity testing against the two-DDA split path.
    SP.fused_walker = fused_walker_;
    split_map_ = std::make_unique<scovox::ScovoxMapSplit>(SP);
    loadSemanticColorMap();  initializeSemanticColors();
    setupSubscribers();
    setupPublishers();
    if (pub_tsdf_ && P.sdf_trunc > 0.f) {
      // Triangle-mesh extraction from the split substrate (TSDF geometry +
      // Dir-grid labels). RViz mesh consumers can also get the surface point
      // cloud via ~/tsdf_pointcloud (publishTSDFPointCloud does the cross-grid
      // label join).
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
    RCLCPP_INFO(get_logger(), "SCovox ready res=%.3f mode=%s frame=%s share_tsdf=%d fused_walker=%d",
      P.resolution, mode_.c_str(), int_frame_.c_str(), (int)share_tsdf_, (int)fused_walker_);
    {
      const auto& tg = split_map_->tsdf().grid();
      RCLCPP_INFO(get_logger(), "grid leaf_bits=%u inner_bits=%u block=%ux%ux%u voxels_per_leaf=%u",
        tg.leafBits(), tg.innetBits(),
        1u << tg.leafBits(), 1u << tg.leafBits(), 1u << tg.leafBits(),
        1u << (3u * tg.leafBits()));
    }
    RCLCPP_INFO(get_logger(), "TSDF: sdf_trunc=%.3f m space_carving=%d band_only=%d (sdf_trunc=0 means off; set via enable_tsdf / sdf_trunc_voxels)",
      P.sdf_trunc, (int)P.tsdf_space_carving, (int)P.band_only_integration);
    RCLCPP_INFO(get_logger(), "split substrate: sizeof(TsdfVoxel)=%zu sizeof(BetaVoxel)=%zu sizeof(DirVoxel)=%zu",
      sizeof(scovox::TsdfVoxel), sizeof(scovox::BetaVoxel), sizeof(scovox::DirVoxel));
  }
  ~SCovoxNode() override {
    // Join the diagnostic memlog worker (scheduleMemUsage) before any node
    // member is destroyed. The worker holds a shared_lock on map_mtx_ and
    // dereferences map_/split_map_/logger via the captured `this`; if it
    // were detached (the prior behaviour) it could still be walking the grid
    // when rclcpp::spin() returns and this object is torn down → UAF. Joining
    // here makes shutdown deterministic; the worker is read-only and bounded
    // by one grid walk, so the join is short.
    if (mem_log_thread_.joinable()) mem_log_thread_.join();
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
    P.carve_skip_occ_threshold = dp("carve_skip_occ_threshold", 0.0);  // <=0 = guard off (trust recent scan)
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
    // Dynamic classes: a hit whose argmax(cp) is one of these is routed to the
    // transient decaying grid instead of the persistent map (see node
    // integrateHit). Empty (default) = every hit is persistent and the
    // per-frame decay pass is skipped.
    for (auto c : dp("dynamic_classes", std::vector<int64_t>{}))
      if (c >= 0 && c < max_sem_) dyn_cls_.insert((uint16_t)c);
    return P;
  }
  void declareNodeParams() {
    auto dp = [&](auto n, auto d){ return this->declare_parameter<decltype(d)>(n, d); };
    base_frame_ = dp("base_frame", std::string("base_link"));
    map_frame_ = dp("map_frame", std::string("map"));
    int_frame_ = dp("integration_frame", std::string("odom"));
    // ── LiDAR + RGB-D fusion (opt-in) ────────────────────────────────────────
    // Master switch. Default false → today's either/or single-sensor path: every
    // integrateHit passes prof=nullptr, so the substrate reads the global map
    // weights and behaviour is byte-identical. true → the node subscribes BOTH
    // streams and hands each its own HitWeights profile (built in buildFusionProfiles),
    // so LiDAR and RGB-D write the ONE SemSplitMap with their own sensor models.
    fuse_lidar_rgbd_ = dp("fuse_lidar_rgbd", false);
    // Per-stream ray-origin frames (fusion only). LiDAR carve rays originate at
    // the LiDAR body/optical frame, RGB-D at the camera; both default to
    // base_frame_ so an unset fused config stays parity with the single path.
    lidar_base_frame_ = dp("lidar_base_frame", base_frame_);
    rgbd_base_frame_  = dp("rgbd_base_frame",  base_frame_);
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
    // Sender-side wire toggle for the TSDF stream:
    //   share_tsdf=false (default): emit Beta + Dir only (dscovox-fusion-only
    //     path; each robot keeps its local TSDF).
    //   share_tsdf=true: also emit the TSDF stream (opt-in for fused-geometry
    //     consensus). Maps to BinarySerializer::Options.share_tsdf.
    share_tsdf_ = dp("share_tsdf", false);
    // Step 12.10 (2026-05-09) — fused single-DDA ray walker. Default true.
    // Set false to fall back to the two-DDA split path for A/B parity testing.
    fused_walker_ = dp("fused_walker", true);
    // Semantic priors. num_classes is the dataset's total class count — sets
    // the OTHER bucket's prior mass to (num_classes − K_TOP) · alpha_0 so the
    // implicit Dirichlet still marginalises onto the true (C+1)-category
    // distribution. Defaults match SemSplitMap::Params (NYU13 / 0.01). KITTI
    // launches override num_classes:=20; Replica/SceneNet stay at 14.
    num_classes_ = std::max<int>(scovox::K_TOP + 1,
                                 dp("num_classes", (int)scovox::SemSplitMap::Params{}.num_classes));
    alpha_0_     = (float)dp("dirichlet_prior",
                             (double)scovox::SemSplitMap::Params{}.alpha_0);
    if (alpha_0_ <= 0.f) alpha_0_ = scovox::kDefaultDirichletPrior;
    // Audit hook (split-grid only): when non-empty, every periodic memlog
    // tick overwrites this path with a flat binary snapshot of TsdfMap:
    //   uint64_t n; then n × {float x, float y, float z, float distance,
    //   float weight} = 20 B/voxel, voxel-centre coords in scovox_node's
    //   world frame. Lets a parity-test harness compare the TsdfMap voxel
    //   set against SLIM-VDB's voxels.bin after a Tr_inv frame conversion
    //   (see tools/tsdf_parity_test.py). Empty default → no-op.
    tsdf_dump_path_ = dp("tsdf_dump_path", std::string{});
    pointcloud_mode_ = !input_pc_topic_.empty();
    // ── Intra-scan deskew (gyro-based rotation correction) ──────────────────
    // deskew_mode: "auto" (deskew iff the cloud has a per-point time field),
    // "on"/"gyro" (force; warn if the field is missing), or "off" (never — set
    // this for already-deskewed feeds like /glim_ros/points, which still carry a
    // `t` field and would otherwise be double-corrected). Phase 1 is rotation
    // only: each point is rotated from its capture time back to scan-start using
    // gyro integrated across the scan, then placed by the single scan-start pose.
    deskew_mode_ = dp("deskew_mode", std::string("auto"));
    if (deskew_mode_ == "gyro") deskew_mode_ = "on";
    if (deskew_mode_ != "off" && deskew_mode_ != "on" && deskew_mode_ != "auto") deskew_mode_ = "auto";
    imu_topic_ = dp("imu_topic", std::string("/imu/data"));
    imu_frame_ = dp("imu_frame", std::string(""));   // empty → take from first /imu msg
    deskew_window_sec_ = dp("deskew_window_sec", 0.2);
    imu_retention_sec_ = dp("imu_retention_sec", 1.0);
    deskew_min_angle_deg_ = dp("deskew_min_angle_deg", 0.0);  // skip scans rotating < this
    // Phase 2: also shift endpoints by the sensor's odom-frame velocity × the
    // per-point time offset (intra-scan translation). Velocity is differenced
    // from consecutive scan poses — no IMU accel, no latency. Off by default;
    // rotation captures the bulk of the smear.
    deskew_translation_ = dp("deskew_translation", false);
    // TF placement timing. The raw /ouster/points scan reaches scovox at the same
    // instant it reaches GLIM, so GLIM has not yet computed/broadcast the
    // odom<-os_lidar pose for that stamp. With a short timeout the exact-stamp
    // lookup fails and we fall back to Time(0) (the PREVIOUS scan's pose) →
    // mis-placed scan → accumulation smear. tf_lookup_timeout_sec lets scovox
    // WAIT for GLIM's TF (the TransformListener fills the buffer on its own
    // thread, so this blocks only the main loop, not TF intake). tf_require_exact
    // drops a scan rather than integrating it at a stale Time(0) pose.
    tf_lookup_timeout_sec_ = dp("tf_lookup_timeout_sec", 0.2);
    tf_require_exact_ = dp("tf_require_exact", false);
    // Uniform voxel-grid downsample, applied per-scan in the SENSOR frame BEFORE
    // integration (after deskew) — this is what GLIM does in preprocessing
    // (config_preprocess.json: voxel-grid @ downsample_resolution). The raw
    // full-res cloud over-samples the noisy surface so every scan fills the tails
    // of the per-column z-distribution → thick smear; collapsing points to one
    // centroid per voxel cuts that tail-sampling without throwing away coverage.
    // 0.0 = off (full per-point path, unchanged). Geometric only: when >0 the
    // per-point semantic/top-k labels are dropped (fine for the raw LiDAR path).
    downsample_voxel_size_ = dp("downsample_voxel_size", 0.0);
    {
      auto gb = dp("gyro_bias", std::vector<double>{0.0, 0.0, 0.0});
      if (gb.size() == 3) gyro_bias_ = Eigen::Vector3f((float)gb[0], (float)gb[1], (float)gb[2]);
    }
    // TF stability gate: skip sensor frames until the TF pose has been
    // stable (no jumps > threshold) for at least this many seconds.
    // Prevents ghost voxels at origin when TF is briefly wrong at startup.
    startup_tf_stable_sec_ = dp("startup_tf_stable_sec", 2.0);
    startup_tf_jump_thresh_ = dp("startup_tf_jump_threshold", 0.5);
    // Runtime divergence guard. Once the startup gate has declared the pose
    // stable, keep watching frame-to-frame pose jumps. A jump larger than
    // runtime_tf_jump_threshold means localization has diverged / teleported
    // (e.g. the NDT track lost lock): drop that frame AND re-arm the startup
    // stabilization so we stop integrating against the bad pose until it
    // settles again. Set runtime_tf_gate=false to keep the legacy
    // startup-only behaviour. The runtime threshold should sit above real
    // frame-to-frame motion (walking ~0.1-0.15 m at 10 Hz) so normal travel
    // never trips it.
    runtime_tf_gate_ = dp("runtime_tf_gate", true);
    runtime_tf_jump_thresh_ = dp("runtime_tf_jump_threshold", 1.0);
    // Localization reject gate. A frame-to-frame jump gate cannot see a pose
    // that is *frozen* — when an external localizer (e.g. NDT map-matcher)
    // loses lock it rejects scans, stops updating its pose, but keeps
    // re-broadcasting the stale transform on a timer. The TF therefore looks
    // fresh and jump-free while the robot keeps moving, so scovox would smear
    // every new scan onto the stuck pose. This gate subscribes to the
    // localizer's /alignment_status (diagnostic_msgs/DiagnosticArray) and skips
    // integration whenever it reports the pose is stale: accepted_gap_sec (time
    // since the last accepted update) exceeds reject_gate_max_accepted_gap_sec,
    // or consecutive_rejected_updates reaches reject_gate_min_consecutive.
    reject_gate_enable_ = dp("reject_gate_enable", false);
    alignment_status_topic_ = dp("alignment_status_topic", std::string("/alignment_status"));
    reject_gate_max_accepted_gap_sec_ = dp("reject_gate_max_accepted_gap_sec", 0.5);
    reject_gate_min_consecutive_ = (int)dp("reject_gate_min_consecutive", 0);
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

  // Build the per-source HitWeights profiles used when fuse_lidar_rgbd_ is on.
  // LiDAR defaults fall back to the global map weights (parity with the single-
  // sensor path when only the master switch is flipped); override lidar_w_occ:=8
  // etc. for the high-evidence ToF calibration (map_interface.hpp). RGB-D is
  // "pure LiDAR authority": w_occ=0 (Stream A skipped → occupancy stays LiDAR-
  // built, Stream B gates on it), w_free=0 (no carve onto the shared Beta grid),
  // geometry_off=true (no TSDF band). A/B "zero vs small w_occ" is a param flip.
  void buildFusionProfiles(const scovox::Params& P) {
    auto dp = [&](auto n, auto d){ return this->declare_parameter<decltype(d)>(n, d); };
    lidar_prof_.w_occ  = (float)dp("lidar_w_occ",  (double)P.w_occ);
    lidar_prof_.w_free = (float)dp("lidar_w_free", (double)P.w_free);
    lidar_prof_.kappa0 = (float)dp("lidar_kappa0", (double)P.kappa0);
    lidar_prof_.dirichlet_min_p_occ = (float)dp("lidar_dirichlet_min_p_occ", (double)P.dirichlet_min_p_occ);
    lidar_prof_.geometry_off = false;  // LiDAR owns TSDF geometry
    lidar_prof_.kernel_radius = 0.0f;  // LiDAR commits geometry+labels at its exact hit voxel — never spreads
    rgbd_prof_.w_occ  = (float)dp("rgbd_w_occ",  0.0);
    rgbd_prof_.w_free = (float)dp("rgbd_w_free", 0.0);
    rgbd_prof_.kappa0 = (float)dp("rgbd_kappa0", (double)P.kappa0);
    // Gate MUST be strictly above the Beta(1,1) prior (p_occ=0.5): with
    // rgbd_w_occ=0 an RGB-D hit on a voxel LiDAR never touched allocates a Beta
    // voxel at prior, and the DIRICHLET gate is `p_occ_post >= min_p_occ`, so a
    // 0.5 default would commit semantics on prior-only geometry — defeating pure
    // LiDAR authority. 0.55 rejects the prior AND LiDAR-carved-free voxels while a
    // single LiDAR hit (p_occ≈0.9, even a weak q≈0.05 hit ≈0.58) admits. Raise
    // toward 0.6 for stricter authority (worsens leading-edge temporal recall).
    rgbd_prof_.dirichlet_min_p_occ = (float)dp("rgbd_dirichlet_min_p_occ", 0.55);
    rgbd_prof_.geometry_off = dp("rgbd_geometry_off", true);
    // RGB-D→LiDAR BKI spread radius `l` (metres). 0 = classic exact-voxel gate
    // (RGB-D labels a voxel only if its own endpoint coincides with a LiDAR-
    // occupied voxel — starved by the LiDAR downsample). >0 spreads each RGB-D
    // label onto every LiDAR-occupied voxel within `l` via the S-BKI kernel.
    // Start ~0.4 (LiDAR length-scale from Gan et al.); ~1 voxel at resolution
    // 0.10. Cost scales as (2·l/res+1)³ persistent-Beta lookups per RGB-D point.
    rgbd_prof_.kernel_radius = (float)dp("rgbd_kernel_radius", 0.0);
    if (fuse_lidar_rgbd_) {
      RCLCPP_INFO(get_logger(),
        "FUSION on: LiDAR{w_occ=%.2f w_free=%.2f kappa0=%.2f min_p=%.2f} "
        "RGB-D{w_occ=%.2f w_free=%.2f kappa0=%.2f min_p=%.2f geom_off=%d kernel_l=%.2f}",
        lidar_prof_.w_occ, lidar_prof_.w_free, lidar_prof_.kappa0, lidar_prof_.dirichlet_min_p_occ,
        rgbd_prof_.w_occ, rgbd_prof_.w_free, rgbd_prof_.kappa0, rgbd_prof_.dirichlet_min_p_occ,
        (int)rgbd_prof_.geometry_off, rgbd_prof_.kernel_radius);
    }
  }
  // Per-callback profile selectors. Null when fusion is off → substrate uses its
  // global params_ (byte-identical single-sensor path).
  const scovox::HitWeights* lidarProf() const { return fuse_lidar_rgbd_ ? &lidar_prof_ : nullptr; }
  const scovox::HitWeights* rgbdProf()  const { return fuse_lidar_rgbd_ ? &rgbd_prof_  : nullptr; }

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
    // Localization reject gate: listen to the localizer's diagnostic stream.
    if (reject_gate_enable_) {
      status_sub_ = create_subscription<diagnostic_msgs::msg::DiagnosticArray>(
        alignment_status_topic_, rclcpp::QoS(10),
        std::bind(&SCovoxNode::onAlignmentStatus, this, std::placeholders::_1));
      RCLCPP_INFO(get_logger(),
          "Reject gate ON: gating integration on %s (accepted_gap>=%.2fs%s)",
          alignment_status_topic_.c_str(), reject_gate_max_accepted_gap_sec_,
          reject_gate_min_consecutive_ > 0 ? " or consec rejects" : "");
    }
    // Fusion: subscribe BOTH streams when fuse_lidar_rgbd. Otherwise classic
    // either/or — LiDAR iff input_pointcloud_topic is set, else RGB-D.
    const bool want_lidar = pointcloud_mode_;
    const bool want_rgbd  = fuse_lidar_rgbd_ || !pointcloud_mode_;
    if (fuse_lidar_rgbd_ && !pointcloud_mode_)
      RCLCPP_WARN(get_logger(), "fuse_lidar_rgbd=true but input_pointcloud_topic empty — no LiDAR stream; running RGB-D only");
    if (want_lidar) {
      auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
      input_pc_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(input_pc_topic_, qos,
        std::bind(&SCovoxNode::onPointCloud, this, std::placeholders::_1));
      RCLCPP_INFO(get_logger(), "PointCloud2 input mode: topic=%s", input_pc_topic_.c_str());
      if (deskew_mode_ != "off") {
        // Best-effort sensor QoS: connects to BOTH reliable and best-effort IMU
        // publishers (a reliable sub would refuse a best-effort publisher and we
        // would silently get no gyro → deskew permanently falls back).
        imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
          imu_topic_, rclcpp::SensorDataQoS(),
          std::bind(&SCovoxNode::onImu, this, std::placeholders::_1));
        RCLCPP_INFO(get_logger(), "deskew=%s: subscribing IMU %s (gyro-only)",
                    deskew_mode_.c_str(), imu_topic_.c_str());
      }
    }
    if (want_rgbd) {
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
    if (pub_tsdf_ && sdf_trunc_launch_ > 0.f) {
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

  // TF quality gate shared by the depth and LiDAR paths. Returns true when the
  // observer pose `O` (sensor origin in the integration frame) is trustworthy
  // enough to integrate this frame. Two stages:
  //   (1) Startup stabilization — wait until the pose has been jump-free
  //       (< startup_tf_jump_threshold) for startup_tf_stable_sec before EVER
  //       integrating. Guards against ghost voxels at the origin while TF is
  //       briefly wrong at startup.
  //   (2) Runtime divergence guard — once stable, a frame-to-frame jump larger
  //       than runtime_tf_jump_threshold means localization diverged: drop the
  //       frame and re-arm stage (1) so integration pauses until the pose
  //       settles again.
  // Records tf_prev_pos_ on every frame (gated or not) so the jump is always
  // measured against the immediately preceding pose (the legacy code froze
  // tf_prev_pos_ once stable).
  bool tfGatePass(const Eigen::Vector3f& O) {
    const rclcpp::Time now = this->now();
    // Jump vs the previous pose; record O now so all paths update the reference
    // exactly once.
    const bool have_prev = tf_prev_valid_;
    const float jump = have_prev ? (O - tf_prev_pos_).norm() : 0.0f;
    tf_prev_pos_ = O;
    tf_prev_valid_ = true;

    // (2) Runtime divergence guard — only meaningful once already stable.
    if (tf_stable_ && runtime_tf_gate_ && have_prev && jump > runtime_tf_jump_thresh_) {
      ++tf_rearm_count_;
      RCLCPP_WARN(get_logger(),
          "Runtime TF jump %.2f m > %.2f m — localization diverged; pausing "
          "integration until pose re-stabilizes (rearm #%zu)",
          jump, runtime_tf_jump_thresh_, tf_rearm_count_);
      tf_stable_ = false;
      tf_stable_since_ = now;
    }
    // (1) Stabilization wait. At first startup this uses the lenient
    // startup_tf_jump_threshold (anti-stall while the TF chain warms up); after
    // a runtime divergence re-arm it uses the strict runtime_tf_jump_threshold
    // so we only resume once the pose is genuinely settled again.
    if (!tf_stable_) {
      const double settle_thresh =
          did_initial_stabilize_ ? runtime_tf_jump_thresh_ : startup_tf_jump_thresh_;
      if (!have_prev || jump > settle_thresh) {
        tf_stable_since_ = now;
      } else if ((now - tf_stable_since_).seconds() >= startup_tf_stable_sec_) {
        tf_stable_ = true;
        did_initial_stabilize_ = true;
        RCLCPP_INFO(get_logger(), "TF stable for %.1f s — integration enabled", startup_tf_stable_sec_);
      }
      if (!tf_stable_) {
        ++frames_gated_;
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
            "Waiting for TF stabilization (%.1f / %.1f s, %zu frames gated)...",
            (now - tf_stable_since_).seconds(), startup_tf_stable_sec_, frames_gated_);
        return false;
      }
    }
    return true;
  }

  // Parse the localizer's /alignment_status DiagnosticArray and cache the
  // tracking-quality scalars used by locRejecting(). accepted_gap_sec may be
  // "nan" before the first accepted pose; std::stod yields NaN which the gate
  // treats as "not rejecting" (the startup TF gate covers that window).
  void onAlignmentStatus(const diagnostic_msgs::msg::DiagnosticArray::ConstSharedPtr& msg) {
    for (const auto& st : msg->status) {
      for (const auto& kv : st.values) {
        if (kv.key == "accepted_gap_sec") {
          try { loc_accepted_gap_sec_ = std::stod(kv.value); } catch (...) {}
        } else if (kv.key == "consecutive_rejected_updates") {
          try { loc_consec_rejects_ = std::stol(kv.value); } catch (...) {}
        }
      }
    }
    loc_status_received_ = true;
  }

  // True when the localizer reports its pose is currently stale/frozen, so this
  // frame must NOT be integrated. Returns false (allow) until the first status
  // arrives so we never stall waiting on a localizer that does not publish it.
  bool locRejecting() const {
    if (!reject_gate_enable_ || !loc_status_received_) return false;
    const double gap = loc_accepted_gap_sec_;
    if (std::isfinite(gap) && gap >= reject_gate_max_accepted_gap_sec_) return true;
    if (reject_gate_min_consecutive_ > 0 &&
        loc_consec_rejects_ >= reject_gate_min_consecutive_) return true;
    return false;
  }

  // Single frame-admission gate: true only when this frame should be integrated.
  // Combines the TF quality gate and the localization reject gate so callers
  // (onImages / onPointCloud) share one check and all gate bookkeeping lives
  // here — tfGatePass() tallies frames_gated_, the reject count is tallied here.
  bool admitFrame(const Eigen::Vector3f& O) {
    if (!tfGatePass(O)) return false;
    if (locRejecting()) { ++frames_gated_reject_; return false; }
    return true;
  }

  void onImages(const sensor_msgs::msg::Image::ConstSharedPtr& depth, const sensor_msgs::msg::Image::ConstSharedPtr& seg) {
    auto t_start = std::chrono::high_resolution_clock::now();
    split_map_->resetTiming();
    ++frame_recv_;
    uint16_t replay_idx = (uint16_t)(depth->header.stamp.nanosec & 0xFFFF);
    last_input_stamp_ = rclcpp::Time(depth->header.stamp, RCL_ROS_TIME);  // for TF-safe map republish stamp
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
    // RGB-D semantic frames MUST integrate at the EXACT capture-time pose. The old
    // code fell back to Time(0) (latest pose) on an exact-stamp miss, but seg adds
    // ~250 ms of inference latency, so the depth stamp is ~250 ms old and "latest"
    // is ahead by the robot's motion — those semantic points smear into the map
    // (~5% of frames in testing). Reject instead: drop the frame on any exact-stamp
    // miss. (LiDAR onPointCloud keeps its own fallback; this policy is RGB-D only.)
    Eigen::Isometry3f T_oo;
    try { T_oo = toE(tf_buffer_.lookupTransform(int_frame_, depth->header.frame_id, depth->header.stamp, rclcpp::Duration::from_seconds(0.2)));
    } catch (const std::exception& e) {
      RCLCPP_WARN(get_logger(), "recv=%zu replay=%u: TF FAILED (no exact-stamp pose): %s", frame_recv_, replay_idx, e.what());
      return;
    }
    RCLCPP_DEBUG(get_logger(), "recv=%zu replay=%u: TF exact match", frame_recv_, replay_idx);
    static const Eigen::Matrix3f kR = (Eigen::Matrix3f() << 0,0,1, -1,0,0, 0,-1,0).finished();
    T_oo.linear() = T_oo.linear() * kR;
    Eigen::Vector3f O;
    const std::string& obs_frame = fuse_lidar_rgbd_ ? rgbd_base_frame_ : base_frame_;
    try { auto t = tf_buffer_.lookupTransform(int_frame_, obs_frame, depth->header.stamp, rclcpp::Duration::from_seconds(0.2));
      O << t.transform.translation.x, t.transform.translation.y, t.transform.translation.z;
    } catch (const std::exception& e) {
      // Same exact-stamp-or-reject policy for the ray-origin (observer) pose.
      RCLCPP_WARN(get_logger(), "recv=%zu replay=%u: observer TF FAILED (no exact-stamp pose): %s", frame_recv_, replay_idx, e.what());
      return;
    }
    // --- Frame-admission gate: TF stability + runtime divergence + reject ---
    if (!admitFrame(O)) return;
    decayTransientFrame();
    auto t_tf = std::chrono::high_resolution_clock::now();
    const auto& P = map_params_;
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
    if (topk_->enabled()) {
      uint16_t fid = (uint16_t)(depth->header.stamp.nanosec & 0xFFFF);
      use_topk = topk_->loadFrame(fid, /*image_mode=*/true);
    }

    // Batched carve frame: stage every ray's free-space read-free, write once
    // at flush (see SemSplitMap). RGB-D carries w_free=0 so it stages nothing,
    // but keeping the invariant universal covers any carving image source.
    split_map_->beginCarveFrame();
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
        vs = topk_->fillImage(u, v, cp);
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
      integrateHit(O, Hp, rng, vs ? &cp : nullptr, q, rgbdProf());
    }
    carveNoReturnRays(O, nr_eps, rgbdProf());
    split_map_->flushCarveFrame();
    auto t_integrate = std::chrono::high_resolution_clock::now();
    size_t bin_bytes_ = 0;
    if (bin_pub_) { auto [bv,bm] = publishBinaryMap(); bin_bytes_ = bv; (void)bm; }
    else {
      // No bin_pub_ in persistent mode → publishBinaryMap is never
      // called → TsdfMap/SemSplitMap touched buffers grow unbounded
      // (every integrated ray appends coords). Clear via the O(n)
      // path: drainTouched* sorts+uniques, but the result is unused here,
      // so a plain clear is ~µs. The bin_pub_ branch above still uses
      // drainTouched* for the wire-format dedup it needs.
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
    {
      const float t_ms = static_cast<float>(split_map_->tsdfTimeUs())    / 1000.0f;
      const float s_ms = static_cast<float>(split_map_->semdirTimeUs()) / 1000.0f;
      RCLCPP_INFO(get_logger(),
                  "recv=%zu replay=%u frame_ms=%.1f tf_ms=%.1f integrate_ms=%.1f "
                  "publish_ms=%.1f rss_mb=%.1f tsdf_ms=%.1f sembeta_ms=%.1f bin_bytes=%zu",
                  frame_recv_, replay_idx, frame_ms, tf_ms, integrate_ms,
                  publish_ms, mem_kb / 1024.0, t_ms, s_ms, bin_bytes_);
    }
    if (frame_recv_ % 10 == 1) scheduleMemUsage();
    logEvictionDelta(frame_recv_);
    topk_->logSummary(/*throttle_ms=*/10000);
  }

  // ── Intra-scan deskew (gyro-based) ──────────────────────────────────────
  // Small-angle SO(3) exponential → unit quaternion (rotation vector φ = ω·dt).
  static Eigen::Quaternionf quatExp(const Eigen::Vector3f& phi) {
    const float th = phi.norm();
    if (th < 1.0e-9f) return Eigen::Quaternionf::Identity();
    const float h = 0.5f * th;
    const Eigen::Vector3f ax = phi / th;
    const float s = std::sin(h);
    return Eigen::Quaternionf(std::cos(h), ax.x() * s, ax.y() * s, ax.z() * s);
  }
  // Per-point time offset (seconds, relative to scan start) from the cloud's
  // time field. UINT32 is nanoseconds since scan start; FLOAT32 is already a
  // relative second offset; FLOAT64 may carry an absolute stamp (tolerated by
  // subtracting the scan-start time t0_sec when the value looks absolute).
  static float decodePointTimeOffset(const uint8_t* tp, uint8_t t_type, double t0_sec) {
    switch (t_type) {
      case sensor_msgs::msg::PointField::UINT32:
        return static_cast<float>(*reinterpret_cast<const uint32_t*>(tp) * 1.0e-9);
      case sensor_msgs::msg::PointField::FLOAT32:
        return *reinterpret_cast<const float*>(tp);
      case sensor_msgs::msg::PointField::FLOAT64: {
        const double td = *reinterpret_cast<const double*>(tp);
        return static_cast<float>(td > 1.0e6 ? td - t0_sec : td);  // tolerate absolute stamps
      }
      default: return 0.f;
    }
  }

  // Buffer the gyro stream. The single-threaded executor serializes this against
  // onPointCloud, so no lock is needed (same contract the map_mtx_ comments rely
  // on). Stores angular velocity only — translation deskew is phase 2.
  void onImu(const sensor_msgs::msg::Imu::ConstSharedPtr& m) {
    if (imu_frame_.empty() && !m->header.frame_id.empty()) imu_frame_ = m->header.frame_id;
    ImuSample s;
    s.t = rclcpp::Time(m->header.stamp, RCL_ROS_TIME).seconds();
    s.w = Eigen::Vector3f((float)m->angular_velocity.x, (float)m->angular_velocity.y,
                          (float)m->angular_velocity.z) - gyro_bias_;
    imu_buf_.push_back(s);
    const double cutoff = s.t - imu_retention_sec_;
    while (!imu_buf_.empty() && imu_buf_.front().t < cutoff) imu_buf_.pop_front();
  }

  // Resolve + cache R_lidar_imu (rotates gyro from the IMU frame into the
  // LiDAR/cloud frame). GLIM's static imu→sensor TF supplies it. Returns false
  // until both the IMU frame id (from the first /imu msg) and the static TF are
  // available; the caller then falls back to the single-pose path.
  bool ensureLidarImuExtrinsic(const std::string& lidar_frame) {
    if (extrinsic_valid_) return true;
    if (imu_frame_.empty()) return false;
    try {
      auto t = tf_buffer_.lookupTransform(lidar_frame, imu_frame_, rclcpp::Time(0),
                                          rclcpp::Duration::from_seconds(0.05));
      R_lidar_imu_ = toE(t).linear();
      extrinsic_valid_ = true;
      RCLCPP_INFO(get_logger(), "deskew: cached extrinsic R[%s<-%s]",
                  lidar_frame.c_str(), imu_frame_.c_str());
      return true;
    } catch (const std::exception& e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                           "deskew: extrinsic %s<-%s not in TF yet (%s)",
                           lidar_frame.c_str(), imu_frame_.c_str(), e.what());
      return false;
    }
  }

  // Build the per-scan cumulative-rotation table over [t0, t0+window] by
  // strapdown-integrating buffered gyro expressed in the sensor frame:
  //   ΔR(t0→τ+dτ) = ΔR(t0→τ) · Exp(ω_s·dτ),  ω_s = R_lidar_imu · ω_imu.
  // Knot k stores (τ_k−t0, ΔR(t0→τ_k)); the point loop slerps between knots and
  // applies ΔR to map each point back to the scan-start sensor frame. Returns
  // false (→ no deskew) when the buffer can't cover the scan.
  bool buildDeskewTable(double t0, double window) {
    deskew_table_.clear();
    if (imu_buf_.size() < 2 || !extrinsic_valid_) return false;
    if (imu_buf_.back().t <= t0) return false;   // whole buffer precedes the scan
    const double t_end = t0 + window;
    deskew_table_.push_back({0.f, Eigen::Quaternionf::Identity()});
    Eigen::Quaternionf cumq = Eigen::Quaternionf::Identity();
    double last_t = t0;
    size_t idx = 0;
    while (idx < imu_buf_.size() && imu_buf_[idx].t <= t0) ++idx;
    // ω held over [last_t, next sample]: prefer the sample bracketing t0 on the left.
    Eigen::Vector3f w_prev = (idx > 0) ? imu_buf_[idx - 1].w : imu_buf_[idx].w;
    for (; idx < imu_buf_.size(); ++idx) {
      const double ts = imu_buf_[idx].t;
      if (ts > t_end) break;
      const double dt = ts - last_t;
      if (dt <= 0.0) { w_prev = imu_buf_[idx].w; continue; }
      const Eigen::Vector3f w_mid = 0.5f * (w_prev + imu_buf_[idx].w);   // trapezoidal
      cumq = cumq * quatExp((R_lidar_imu_ * w_mid) * static_cast<float>(dt));
      cumq.normalize();
      deskew_table_.push_back({static_cast<float>(ts - t0), cumq});
      last_t = ts;
      w_prev = imu_buf_[idx].w;
    }
    // Zero-order-hold extension to the window end so late points still resolve.
    if (last_t < t_end) {
      const double dt = t_end - last_t;
      cumq = cumq * quatExp((R_lidar_imu_ * w_prev) * static_cast<float>(dt));
      cumq.normalize();
      deskew_table_.push_back({static_cast<float>(t_end - t0), cumq});
    }
    return deskew_table_.size() >= 2;
  }

  // ── PointCloud2 input path (LiDAR) ──────────────────────────────────────
  void onPointCloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud) {
    auto t_start = std::chrono::high_resolution_clock::now();
    split_map_->resetTiming();
    ++frame_recv_;
    uint16_t replay_idx = (uint16_t)(cloud->header.stamp.nanosec & 0xFFFF);
    last_input_stamp_ = rclcpp::Time(cloud->header.stamp, RCL_ROS_TIME);  // for TF-safe map republish stamp
    std::unique_lock<std::shared_mutex> lock(map_mtx_);

    // Find field offsets
    int off_x=-1, off_y=-1, off_z=-1, off_lbl=-1, off_t=-1;
    uint8_t lbl_type = 0, t_type = 0;
    for (auto& f : cloud->fields) {
      if (f.name=="x") off_x=f.offset;
      else if (f.name=="y") off_y=f.offset;
      else if (f.name=="z") off_z=f.offset;
      else if (f.name=="semantic_label") { off_lbl=f.offset; lbl_type=f.datatype; }
      else if (f.name=="t"||f.name=="time"||f.name=="time_stamp"||f.name=="timestamp") { off_t=f.offset; t_type=f.datatype; }
    }
    if (off_x<0||off_y<0||off_z<0) { RCLCPP_WARN(get_logger(), "PointCloud2 missing xyz fields"); return; }

    // TF: sensor frame -> integration frame (NO kR rotation — LiDAR is already ROS convention).
    // Wait up to tf_lookup_timeout_sec_ for the EXACT-stamp pose; only fall back
    // to Time(0) (the previous scan's pose) if tf_require_exact_ is false. A
    // Time(0) fallback mis-places the whole scan and is the prime suspect for the
    // accumulation smear, so it is counted + warned.
    const auto tf_to = rclcpp::Duration::from_seconds(tf_lookup_timeout_sec_);
    Eigen::Isometry3f T_oi;
    try { T_oi = toE(tf_buffer_.lookupTransform(int_frame_, cloud->header.frame_id, cloud->header.stamp, tf_to));
    } catch (...) {
      ++tf_fallback_count_;
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "recv=%zu: no exact-stamp TF for %s after %.2fs (%zu fallbacks) — %s",
        frame_recv_, cloud->header.frame_id.c_str(), tf_lookup_timeout_sec_,
        tf_fallback_count_, tf_require_exact_ ? "DROPPING scan" : "using Time(0) (stale pose)");
      if (tf_require_exact_) return;
      try { T_oi = toE(tf_buffer_.lookupTransform(int_frame_, cloud->header.frame_id, rclcpp::Time(0), rclcpp::Duration::from_seconds(0.05)));
      } catch (const std::exception& e) { RCLCPP_WARN(get_logger(), "recv=%zu: TF FAILED: %s", frame_recv_, e.what()); return; } }

    // Sensor origin in integration frame (same exact-stamp-then-fallback policy).
    Eigen::Vector3f O;
    const std::string& obs_frame = fuse_lidar_rgbd_ ? lidar_base_frame_ : base_frame_;
    try { auto t = tf_buffer_.lookupTransform(int_frame_, obs_frame, cloud->header.stamp, tf_to);
      O << t.transform.translation.x, t.transform.translation.y, t.transform.translation.z;
    } catch (...) {
      if (tf_require_exact_) return;
      try { auto t = tf_buffer_.lookupTransform(int_frame_, obs_frame, rclcpp::Time(0), rclcpp::Duration::from_seconds(0.05));
      O << t.transform.translation.x, t.transform.translation.y, t.transform.translation.z;
    } catch (...) { return; } }

    // --- Frame-admission gate (shared with onImages path) ---
    if (!admitFrame(O)) return;
    // In fused mode the RGB-D (onImages) callback owns transient decay — LiDAR
    // carries no dynamic-class semantics, and decaying here too would fade RGB-D's
    // person/vehicle evidence ~twice per LiDAR+RGB-D pair. Single-sensor LiDAR
    // (fusion off) keeps its own decay.
    if (!fuse_lidar_rgbd_) decayTransientFrame();

    auto t_tf = std::chrono::high_resolution_clock::now();
    const auto& P = map_params_;

    // Range gate in SQUARED distance: the per-point sqrt in (Hp-O).norm() is
    // pure waste in the occupancy-only LiDAR config, where `rng` feeds only the
    // gate (range_decay_length<=0 -> rw=1; carve_band<=0 -> integrateHit
    // discards rng). Compare squared norms instead and take the sqrt lazily only
    // when a downstream consumer actually needs the true range. The (>0?sq:raw)
    // guard keeps the degenerate negative-threshold case bit-identical: x->x^2
    // is monotonic only on [0,inf), and rng>=0 always, so a negative min/max
    // threshold must be compared as-is (it can only ever always-pass/always-fail).
    const bool need_rng = (P.range_decay_length > 0) || (carve_band_ > 0);
    const float min_r2 = (P.min_range > 0.f) ? P.min_range * P.min_range : P.min_range;
    const float max_r2 = (P.max_range > 0.f) ? P.max_range * P.max_range : P.max_range;

    const uint8_t* data = cloud->data.data();
    const int step = (int)cloud->point_step;
    const size_t N = (size_t)cloud->width * (size_t)cloud->height;

    // Validate the buffer geometry before any reinterpret_cast read. A
    // malformed or truncated PointCloud2 (point_step too small for the declared
    // field offsets, or data shorter than width*height*point_step) would
    // otherwise drive an out-of-bounds read in the per-point loop below — a
    // crash or silent garbage integration on adversarial / buggy input. Valid
    // clouds always satisfy these (data.size() == height*row_step >=
    // width*height*point_step, and every field offset+size <= point_step).
    if (step <= 0) { RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "PointCloud2 has point_step=%d; dropping", step); return; }
    int max_field_end = std::max({off_x, off_y, off_z}) + (int)sizeof(float);
    if (off_lbl >= 0) {
      const int lbl_sz = (lbl_type == sensor_msgs::msg::PointField::UINT32) ? 4
                       : (lbl_type == sensor_msgs::msg::PointField::UINT16) ? 2 : 1;
      max_field_end = std::max(max_field_end, off_lbl + lbl_sz);
    }
    if (off_t >= 0) {
      const int t_sz = (t_type == sensor_msgs::msg::PointField::FLOAT64) ? 8 : 4;
      max_field_end = std::max(max_field_end, off_t + t_sz);
    }
    if (max_field_end > step || cloud->data.size() < N * (size_t)step) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "PointCloud2 buffer too small (point_step=%d field_end=%d size=%zu need=%zu); dropping",
        step, max_field_end, cloud->data.size(), N * (size_t)step);
      return;
    }

    // Soft-prob: load the per-frame top-K table once. The replay node sets
    // soft_prob_passthrough so the cloud is in raw .bin order, matching
    // the .topk row order; we apply the same range filter below.
    bool use_topk = false;
    if (topk_->enabled()) {
      uint16_t fid = (uint16_t)(cloud->header.stamp.nanosec & 0xFFFF);
      use_topk = topk_->loadFrame(fid, /*image_mode=*/false);
    }

    // ── Intra-scan deskew decision (gyro-based rotation) ──────────────────
    // auto/on: deskew iff the cloud carries a per-point time field AND a gyro
    // table can be built for this scan. off: never (already-deskewed feeds).
    // Any missing prerequisite → fall back to the legacy single-pose path so we
    // never integrate a half-built correction.
    const double t0_sec = last_input_stamp_.seconds();
    bool do_deskew = false;
    if (deskew_mode_ != "off" && off_t >= 0) {
      if (ensureLidarImuExtrinsic(cloud->header.frame_id) &&
          buildDeskewTable(t0_sec, deskew_window_sec_)) {
        const float wlast = std::min(1.f, std::abs(deskew_table_.back().q.w()));
        const float total_deg = 2.f * std::acos(wlast) * 57.29578f;
        if (deskew_min_angle_deg_ <= 0.0 || total_deg >= deskew_min_angle_deg_) {
          do_deskew = true;
          ++deskew_frames_;
        }
      } else {
        ++deskew_fallback_;
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
          "deskew on but IMU/extrinsic not ready (imu_buf=%zu extrinsic=%d frame=%s); "
          "integrating raw scan", imu_buf_.size(), (int)extrinsic_valid_,
          cloud->header.frame_id.c_str());
      }
    } else if (deskew_mode_ == "on" && off_t < 0) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "deskew_mode=on but cloud '%s' has no per-point time field; integrating raw scan",
        cloud->header.frame_id.c_str());
    }

    // ── Phase 2: intra-scan translation deskew (optional) ─────────────────
    // Shift each point's endpoint by the sensor's odom-frame velocity × its time
    // offset, so a point captured at t_i is placed at the sensor position at t_i
    // (not at scan-start). Velocity is differenced from consecutive scan poses
    // (no IMU accel, no latency). Endpoints only; the carve origin O stays at
    // scan-start (a ~0.1 m shift over a scan, negligible for free-space carving).
    Eigen::Vector3f v_odom = Eigen::Vector3f::Zero();
    bool apply_trans = false;
    if (deskew_translation_ && do_deskew) {
      const Eigen::Vector3f cur_trans = T_oi.translation();
      if (prev_scan_valid_) {
        const double dt = t0_sec - prev_scan_t_;
        if (dt > 1.0e-3 && dt < 1.0) {
          v_odom = (cur_trans - prev_scan_trans_) / static_cast<float>(dt);
          apply_trans = true;
        }
      }
      prev_scan_trans_ = cur_trans; prev_scan_t_ = t0_sec; prev_scan_valid_ = true;
    }

    // Per-point rotation lookup: advance a cursor through the knot table and
    // slerp between bracketing knots. Ouster points are column-time-ordered (the
    // 64 beams of a column share one `t`), so the cursor advances monotonically
    // and the same-offset cache collapses ~one slerp per column, not per point.
    size_t dk_cursor = 0;
    float dk_last_off = -1.0e30f;
    Eigen::Quaternionf dk_last_q = Eigen::Quaternionf::Identity();
    auto deskewRot = [&](float off) -> Eigen::Quaternionf {
      if (std::abs(off - dk_last_off) < 1.0e-4f) return dk_last_q;
      if (off <= deskew_table_.front().dt) { dk_last_off = off; dk_last_q = deskew_table_.front().q; return dk_last_q; }
      if (off >= deskew_table_.back().dt)  { dk_last_off = off; dk_last_q = deskew_table_.back().q;  return dk_last_q; }
      if (off < deskew_table_[dk_cursor].dt) dk_cursor = 0;   // non-monotonic guard
      while (dk_cursor + 1 < deskew_table_.size() && deskew_table_[dk_cursor + 1].dt <= off) ++dk_cursor;
      const auto& a = deskew_table_[dk_cursor];
      const auto& b = deskew_table_[dk_cursor + 1];
      const float denom = b.dt - a.dt;
      const float fr = denom > 1.0e-9f ? (off - a.dt) / denom : 0.f;
      dk_last_off = off; dk_last_q = a.q.slerp(fr, b.q);
      return dk_last_q;
    };

    // Uniform voxel-grid downsample (sensor frame), then integrate one centroid
    // per voxel. GLIM does this in preprocessing; on the raw cloud it collapses
    // the dense over-sampling that fills the per-column z-tails (the smear).
    // Geometric only (no semantics/top-k) — used for the raw LiDAR path.
    size_t ds_in = 0, ds_out = 0;
    // Batched carve frame: full-ray free-space is staged read-free per ray and
    // written once per voxel at flush (block-ordered). This is the fast full-ray
    // carve — the dense near-origin ray fan no longer re-writes the same voxel
    // thousands of times, and no per-voxel occupancy read stops the walk.
    split_map_->beginCarveFrame();
    if (downsample_voxel_size_ > 0.0) {
      const float inv = 1.0f / static_cast<float>(downsample_voxel_size_);
      struct Acc { float sx, sy, sz, soff; uint32_t n; };
      // pack three voxel indices into one int64 (21 bits each, signed-wrap safe
      // for |index| < 2^20 ≈ ±10 km at 1 cm — far beyond any LiDAR range here).
      auto vkey = [](int ix, int iy, int iz) -> int64_t {
        return (int64_t(ix) & 0x1FFFFF) | ((int64_t(iy) & 0x1FFFFF) << 21) | ((int64_t(iz) & 0x1FFFFF) << 42);
      };
      std::unordered_map<int64_t, Acc> grid;
      grid.reserve(N / 2 + 16);
      for (size_t i = 0; i < N; ++i) {
        const uint8_t* p = data + i * step;
        float x = *reinterpret_cast<const float*>(p + off_x);
        float y = *reinterpret_cast<const float*>(p + off_y);
        float z = *reinterpret_cast<const float*>(p + off_z);
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) continue;
        Eigen::Vector3f praw(x, y, z);
        float off_i = 0.f;
        if (do_deskew) {
          off_i = decodePointTimeOffset(p + off_t, t_type, t0_sec);
          praw = deskewRot(off_i) * praw;
        }
        const int ix = (int)std::floor(praw.x() * inv);
        const int iy = (int)std::floor(praw.y() * inv);
        const int iz = (int)std::floor(praw.z() * inv);
        Acc& a = grid[vkey(ix, iy, iz)];
        a.sx += praw.x(); a.sy += praw.y(); a.sz += praw.z(); a.soff += off_i; ++a.n;
      }
      ds_in = N; ds_out = grid.size();
      for (const auto& kv : grid) {
        const Acc& a = kv.second;
        const float invn = 1.0f / float(a.n);
        Eigen::Vector3f praw(a.sx * invn, a.sy * invn, a.sz * invn);  // voxel centroid (sensor frame)
        Eigen::Vector3f Hp = T_oi * praw;
        if (apply_trans) Hp += v_odom * (a.soff * invn);
        const float r2 = (Hp - O).squaredNorm();
        if (r2 < min_r2 || r2 > max_r2) continue;
        const float rng = need_rng ? std::sqrt(r2) : 0.f;
        const float rw = (P.range_decay_length > 0) ? std::exp(-rng / float(P.range_decay_length)) : 1.f;
        integrateHit(O, Hp, rng, nullptr, rw, lidarProf());
      }
    } else {
    std::vector<float> cp(max_sem_, 0.f);
    for (size_t i = 0; i < N; ++i) {
      const uint8_t* p = data + i * step;
      float x = *reinterpret_cast<const float*>(p + off_x);
      float y = *reinterpret_cast<const float*>(p + off_y);
      float z = *reinterpret_cast<const float*>(p + off_z);
      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) continue;

      // Intra-scan deskew: rotate each point from its capture-time sensor frame
      // back to scan-start, then place it with the single scan-start pose. No-op
      // (byte-identical to the legacy path) when do_deskew is false.
      Eigen::Vector3f praw(x, y, z);
      float off_i = 0.f;
      if (do_deskew) {
        off_i = decodePointTimeOffset(p + off_t, t_type, t0_sec);
        praw = deskewRot(off_i) * praw;
      }
      Eigen::Vector3f Hp = T_oi * praw;
      if (apply_trans) Hp += v_odom * off_i;
      const float r2 = (Hp - O).squaredNorm();
      if (r2 < min_r2 || r2 > max_r2) continue;
      const float rng = need_rng ? std::sqrt(r2) : 0.f;
      float rw = (P.range_decay_length > 0) ? std::exp(-rng / float(P.range_decay_length)) : 1.f;
      float q = rw;

      bool vs = false;
      if (use_topk) {
        vs = topk_->fillPoint((size_t)i, cp);
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

      integrateHit(O, Hp, rng, vs ? &cp : nullptr, q, lidarProf());
    }
    }  // end else (per-point path)
    split_map_->flushCarveFrame();  // one Beta write per carved voxel, block-ordered

    auto t_integrate = std::chrono::high_resolution_clock::now();
    size_t bin_bytes_ = 0;
    if (bin_pub_) { auto [bv,bm] = publishBinaryMap(); bin_bytes_ = bv; (void)bm; }
    else {
      // No bin_pub_ in persistent mode → publishBinaryMap is never
      // called → TsdfMap/SemSplitMap touched buffers grow unbounded
      // (every integrated ray appends coords). Clear via the O(n)
      // path: drainTouched* sorts+uniques, but the result is unused here,
      // so a plain clear is ~µs. The bin_pub_ branch above still uses
      // drainTouched* for the wire-format dedup it needs.
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
    {
      const float t_ms = static_cast<float>(split_map_->tsdfTimeUs())    / 1000.0f;
      const float s_ms = static_cast<float>(split_map_->semdirTimeUs()) / 1000.0f;
      RCLCPP_INFO(get_logger(),
                  "recv=%zu replay=%u frame_ms=%.1f tf_ms=%.1f integrate_ms=%.1f "
                  "publish_ms=%.1f rss_mb=%.1f tsdf_ms=%.1f sembeta_ms=%.1f bin_bytes=%zu "
                  "gated=%zu rearm=%zu reject_gated=%zu deskew=%d knots=%zu tf_fb=%zu ds=%zu/%zu",
                  frame_recv_, replay_idx, frame_ms, tf_ms, integrate_ms,
                  publish_ms, mem_kb / 1024.0, t_ms, s_ms, bin_bytes_,
                  frames_gated_, tf_rearm_count_, frames_gated_reject_,
                  (int)do_deskew, do_deskew ? deskew_table_.size() : (size_t)0,
                  tf_fallback_count_, ds_out, ds_in);
    }
    if (frame_recv_ % 10 == 1) scheduleMemUsage();
    logEvictionDelta(frame_recv_);
    topk_->logSummary(/*throttle_ms=*/10000);
  }

  void scheduleMemUsage() {
    if (!mem_log_) return;   // diagnostic-only; gated by log_mem_usage param
    // Single-in-flight guard: if the previous walk is still running (a grid
    // walk can outlast the ~10-frame relaunch cadence on large maps), skip
    // this tick instead of spawning a second worker. Two concurrent workers
    // would both open ${tsdf_dump_path_}.tmp and race on std::rename,
    // producing a truncated/corrupt dump. CAS so only one launcher wins.
    bool expected = false;
    if (!mem_log_inflight_.compare_exchange_strong(expected, true,
                                                   std::memory_order_acq_rel))
      return;
    // Join the previous (already-finished) worker handle before reusing the
    // member; the inflight flag guarantees it has run to completion. The
    // worker is OWNED (not detached) so ~SCovoxNode can join an in-flight
    // walk and avoid a use-after-free of `this`/map_mtx_/split_map_ at shutdown.
    if (mem_log_thread_.joinable()) mem_log_thread_.join();
    mem_log_thread_ = std::thread([this]() {
      std::shared_lock<std::shared_mutex> lock(map_mtx_);
      // Split-grid per-substrate memory log. Consumed by
      // eval_e13_byte_parity.py to compare TsdfMap bytes against
      // SLIM-VDB's vdb_tsdf_mb_final (acceptance gate 15%, paper headline
      // reports the actually-measured ratio).
      const double tsdf_mb    = static_cast<double>(split_map_->tsdfGridBytes())    / (1024.0 * 1024.0);
      const double sembeta_mb = static_cast<double>(split_map_->semdirGridBytes()) / (1024.0 * 1024.0);
      RCLCPP_INFO(get_logger(), "[memUsage] rss_mb=%.1f", getVmRSSKB() / 1024.0);
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
      // Release the in-flight slot last so the next tick can relaunch. The
      // handle itself is joined by the next scheduleMemUsage() or the
      // destructor; we deliberately do NOT detach (see member comment).
      mem_log_inflight_.store(false, std::memory_order_release);
    });
  }

  void integrateHit(const Eigen::Vector3f& O, const Eigen::Vector3f& Hp, float rng,
                    const std::vector<float>* cp, float q,
                    const scovox::HitWeights* prof = nullptr) {
    // Split-grid path. TsdfMap walks the SDF band, SemSplitMap walks the carve
    // band leading up to the hit. `q` already bakes in the range/grazing
    // weights (rw*aw) at the call site.
    //
    // carve_band: when `carve_band_ > 0` (Replica / KITTI launch default =
    // 0.1), walk the semantic carve along only the last `carve_band` metres
    // before the surface, matching the production mIoU baselines. carve_band
    // <= 0 falls back to full-ray.
    Eigen::Vector3f co = O;
    if (carve_band_ > 0) {
      const float d = rng - static_cast<float>(carve_band_);
      if (d > 0) co = O + (Hp - O).normalized() * d;
    }
    // Dynamic routing: a hit whose winning class is configured dynamic writes
    // its surface evidence into the transient decaying grid instead of the
    // persistent map (the free-space carve up to the hit stays persistent).
    // No cp (geometry-only path) or no configured dynamic classes => persistent.
    bool is_dynamic = false;
    if (cp && !dyn_cls_.empty()) {
      int best = -1; float best_p = 0.f;
      for (size_t i = 0; i < cp->size(); ++i)
        if ((*cp)[i] > best_p) { best_p = (*cp)[i]; best = (int)i; }
      if (best >= 0 && dyn_cls_.count((uint16_t)best)) is_dynamic = true;
    }
    split_map_->integrateHit(co, Hp, cp, q, is_dynamic, prof);
    sm_dirty_.store(true, std::memory_order_relaxed);
  }
  // Decay the transient (dynamic-class) grid one step toward the prior. Called
  // once per admitted frame, before this frame's hits are integrated, so stale
  // dynamic evidence fades whether or not new dynamic observations arrive.
  // No-op when no dynamic classes are configured.
  void decayTransientFrame() {
    if (dyn_cls_.empty()) return;
    split_map_->decayTransient(static_cast<float>(transient_decay_rate_));
    sm_dirty_.store(true, std::memory_order_relaxed);
  }
  void carveNoReturnRays(const Eigen::Vector3f& O, const std::vector<Eigen::Vector3f>& nr_eps,
                         const scovox::HitWeights* prof = nullptr) {
    // Beta-only carve along no-return rays (no TSDF surface to anchor).
    // q=1.0f matches the legacy carve which doesn't apply rw/aw. `prof` carries
    // the per-source w_free — a semantics-only source (RGB-D, w_free=0) deposits
    // NO a_free here, so its many no-return (sky/far) rays can't erode LiDAR
    // occupancy on the shared Beta grid.
    for (auto& hf : nr_eps) split_map_->integrateMiss(O, hf, 1.0f, prof);
    sm_dirty_.store(true, std::memory_order_relaxed);
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
    auto& ss = split_map_->semsplit();
    scovox_msgs::msg::ScovoxMap m; m.header.stamp=get_clock()->now(); m.header.frame_id=int_frame_;
    const auto& P=map_params_;
    const double res = split_map_->resolution();
    m.resolution=res; m.occupancy_threshold=min_occ_;
    m.semantic_threshold=P.semantic_occ_gate; m.max_semantic_classes=max_sem_;
    // Walk the Beta (occupancy) grid; join the Dir (semantics) grid at the same
    // coord. a_occ/a_free come from Beta; a_unk = Dir OTHER; per-class evidence
    // = cnt[i] − α_0 so empty slots read 0 (mirrors the pointcloud project).
    const auto& bgrid = ss.betaGrid();
    auto dacc = ss.dirGrid().createConstAccessor();
    m.voxels.reserve(bgrid.activeCellsCount());
    bgrid.forEachCell([&](const scovox::BetaVoxel& b, const Bonxai::CoordT& c) {
      scovox_msgs::msg::ScovoxVoxel vv;
      vv.position.x=c.x*res; vv.position.y=float(c.y*res); vv.position.z=float(c.z*res);
      vv.a_occ=b.a_occ; vv.a_free=b.a_free;
      const scovox::DirVoxel* dv = dacc.value(c);
      if (dv) {
        vv.a_unk = dv->other;
        for (int i = 0; i < scovox::K_TOP; ++i) {
          if (dv->cls[i] == 0xFFFF) continue;
          scovox_msgs::msg::ScovoxSemanticEvidence e;
          e.class_id = dv->cls[i];
          e.evidence_count = std::max(0.f, dv->cnt[i] - alpha_0_);
          vv.semantic_evidence.push_back(e);
        }
      } else {
        vv.a_unk = 0.f;
      }
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
  // Split-substrate binary publish path. Drains touched TSDF + Beta + Dir
  // coords from the SemSplitMap substrate, reads each voxel's current state,
  // builds a BinarySerializer::Frame (three streams), optionally elides the
  // TSDF section per share_tsdf_, LZ4-compresses, and publishes with
  // msg->version=4. Beta (occupancy) and Dir (semantics) cross the wire as
  // SEPARATE streams — the receiver merges each with its own conjugate rule
  // (consensus_merge.hpp), losslessly.
  //
  // Snapshot-on-resub + at-prior elision are applied per grid. This is the
  // node's only wire path; the SPLIT substrate (semsplit()) is always valid.
  std::pair<size_t,double> publishBinaryMap() {
    if (!bin_pub_) return {0, 0};
    auto& ss = split_map_->semsplit();

    const size_t cur_sub = bin_pub_->get_subscription_count();
    bool snapshot = (cur_sub > prev_sub_count_);
    prev_sub_count_ = cur_sub;
    if (cur_sub == 0) {
      (void)split_map_->drainTouchedTsdf();
      (void)split_map_->drainTouchedBeta();
      (void)split_map_->drainTouchedDir();
      return {0, 0};
    }

    scovox::BinarySerializer::Frame frame;
    frame.resolution  = static_cast<float>(split_map_->resolution());
    frame.num_classes = static_cast<uint16_t>(num_classes_);
    frame.alpha_0     = alpha_0_;

    const float beta_occ_prior = scovox::kBetaOccPrior;  // symmetric Beta(1,1) — see docs/occupancy_prior.md
    const float dir_other_prior =
        static_cast<float>(num_classes_ - scovox::K_TOP) * alpha_0_;

    // ----- TSDF section (elided when share_tsdf_=false) -----
    if (share_tsdf_) {
      auto& tsdf_grid = split_map_->tsdf().grid();
      auto tacc = tsdf_grid.createAccessor();
      auto emit_tsdf = [&](const scovox::TsdfVoxel& v, const Bonxai::CoordT& c) {
        if (v.weight <= 0.f) return;
        frame.tsdf_deltas.push_back({c, v});
      };
      if (snapshot) {
        tsdf_grid.forEachCell(emit_tsdf);
        (void)split_map_->drainTouchedTsdf();
      } else {
        for (const auto& c : split_map_->drainTouchedTsdf())
          if (auto* v = tacc.value(c, false)) emit_tsdf(*v, c);
      }
    } else {
      (void)split_map_->drainTouchedTsdf();
    }

    // ----- Beta section (occupancy; full-ray, always emitted) -----
    {
      auto& bgrid = ss.betaGrid();
      auto bacc = bgrid.createAccessor();
      auto emit_beta = [&](const scovox::BetaVoxel& v, const Bonxai::CoordT& c) {
        // At prior → no posterior information; keep off the wire.
        const bool at_prior = (v.a_occ  <= beta_occ_prior        + 1e-4f) &&
                              (v.a_free <= scovox::kBetaFreePrior + 1e-4f);
        if (at_prior) return;
        frame.beta_deltas.push_back({c, v});
      };
      if (snapshot) {
        bgrid.forEachCell(emit_beta);
        (void)split_map_->drainTouchedBeta();
      } else {
        for (const auto& c : split_map_->drainTouchedBeta())
          if (auto* v = bacc.value(c, false)) emit_beta(*v, c);
      }
    }

    // ----- Dir section (semantics; hit-sparse) -----
    {
      auto& dgrid = ss.dirGrid();
      auto dacc = dgrid.createAccessor();
      auto emit_dir = [&](const scovox::DirVoxel& v, const Bonxai::CoordT& c) {
        bool any_sem = false;
        for (int i = 0; i < scovox::K_TOP; ++i)
          if (v.cls[i] != 0xFFFF) { any_sem = true; break; }
        const bool at_prior = !any_sem && (v.other <= dir_other_prior + 1e-4f);
        if (at_prior) return;
        frame.dir_deltas.push_back({c, v});
      };
      if (snapshot) {
        dgrid.forEachCell(emit_dir);
        (void)split_map_->drainTouchedDir();
      } else {
        for (const auto& c : split_map_->drainTouchedDir())
          if (auto* v = dacc.value(c, false)) emit_dir(*v, c);
      }
    }

    if (frame.tsdf_deltas.empty() && frame.beta_deltas.empty() &&
        frame.dir_deltas.empty()) {
      return {0, 0};
    }

    scovox::BinarySerializer::Options opts;
    opts.share_tsdf = share_tsdf_;
    auto data = scovox::BinarySerializer::serialize(frame, opts);
    auto comp = scovox::ScovoxBinarySerializer::compressLZ4(data);

    scovox_msgs::msg::ScovoxMapBinary bin;
    bin.header.stamp    = get_clock()->now();
    bin.header.frame_id = int_frame_;
    bin.version         = 4;   // envelope version — dscovox onBinaryMap routes on it
#if __BYTE_ORDER__==__ORDER_LITTLE_ENDIAN__
    bin.little_endian = true;
#else
    bin.little_endian = false;
#endif
    if (comp.empty()) {
      // data is non-empty by the guard above, so compressLZ4 only returns empty
      // on a genuine LZ4 error. The consumer always expects the LZ4 framing, so
      // shipping the raw blob would be undecodable — drop the frame instead.
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "publishBinaryMap: LZ4 compression failed; dropping frame");
      return {0, 0};
    }
    bin.data = std::move(comp);
    const size_t emitted = frame.tsdf_deltas.size() + frame.beta_deltas.size()
                         + frame.dir_deltas.size();
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
    const auto& bgrid = split_map_->semsplit().betaGrid();
    bgrid.forEachCell([&](const scovox::BetaVoxel& v, const Bonxai::CoordT& c) {
      auto p = bgrid.coordToPos(c);
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
  // Split-substrate pointcloud publisher. Occupancy comes from the Beta
  // grid; semantics from the Dir grid at the same coord. The two are projected
  // into a SemBetaVoxel so the shared viz helpers (argmaxClassConfidence /
  // variance / expectedInformationGain) and the 16-field schema stay stable
  // for pointcloud_to_npz.py / RViz / eval scripts.
  void publishPointCloud() {
    if (!pc_pub_ || !split_map_ || pc_pub_->get_subscription_count() == 0) return;
    auto& ss = split_map_->semsplit();
    sensor_msgs::msg::PointCloud2 cl;
    cl.header.frame_id = int_frame_;
    // Stamp at the last integrated scan time (TF-resolvable), not now() — see
    // last_input_stamp_ note. Falls back to now() before the first integration.
    cl.header.stamp = (last_input_stamp_.nanoseconds() > 0) ? last_input_stamp_ : get_clock()->now();
    cl.height = 1; cl.is_dense = true; cl.is_bigendian = false;
    sensor_msgs::PointCloud2Modifier mod(cl);
    static_assert(scovox::K_TOP >= 1, "publishPointCloud requires at least 1 sparse slot");
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

    const float beta_occ_prior = scovox::kBetaOccPrior;  // symmetric Beta(1,1) — see docs/occupancy_prior.md
    auto has_beta_evidence = [&](const scovox::BetaVoxel& b) {
      return b.a_occ > beta_occ_prior + 1e-3f || b.a_free > scovox::kBetaFreePrior + 1e-3f;
    };
    // Project Beta(occupancy) + Dir(semantics) → SemBetaVoxel for the shared
    // helpers. Occupancy from Beta; per-class evidence from Dir (subtract the
    // α_0 prior so empty slots read 0).
    const auto& bgrid = ss.betaGrid();
    auto dacc = ss.dirGrid().createConstAccessor();
    // Transient (dynamic-class) overlay. Dynamic hits land in a parallel
    // decaying grid; at publish we overlay them on the persistent cloud so they
    // appear while live and fade as they decay. A transient voxel with
    // occupancy evidence overrides its persistent counterpart at the same coord
    // (faithful to the legacy query-time picker). Empty when no dynamic classes
    // are configured, in which case the override check + extra walk are skipped.
    const auto& tbgrid = ss.transientBetaGrid();
    auto tbacc = ss.transientBetaGrid().createConstAccessor();
    auto tdacc = ss.transientDirGrid().createConstAccessor();
    const bool have_transient = ss.transientBetaVoxelCount() > 0;
    // Project Beta(occupancy) + Dir(semantics) → SemBetaVoxel using the supplied
    // Dir accessor (persistent or transient). Per-class evidence subtracts the
    // α_0 prior so empty slots read 0.
    auto project = [&](const scovox::BetaVoxel& b, const Bonxai::CoordT& co, auto& diracc) {
      scovox::SemBetaVoxel v{};
      v.a_occ = b.a_occ; v.a_free = b.a_free;
      const scovox::DirVoxel* dv = diracc.value(co);
      if (dv) {
        v.a_unk = dv->other;
        for (int i = 0; i < scovox::K_TOP; ++i) {
          v.sem_cnt[i] = std::max(0.f, dv->cnt[i] - alpha_0_);
          v.sem_cls[i] = dv->cls[i];
        }
      } else {
        v.a_unk = 0.f;
        for (int i = 0; i < scovox::K_TOP; ++i) { v.sem_cnt[i] = 0.f; v.sem_cls[i] = 0xFFFF; }
      }
      return v;
    };
    // A transient voxel at `co` passes the same publish gate as a persistent one.
    auto transient_overrides = [&](const Bonxai::CoordT& co) {
      const scovox::BetaVoxel* tb = tbacc.value(co);
      return tb && tb->p_occ() >= min_occ_ && has_beta_evidence(*tb);
    };

    // Single grid walk: collect the voxels that pass the publish gate, then size
    // and fill the message from the scratch list (was two full forEachCell walks
    // re-evaluating the same predicate). forEachCell order is deterministic, so
    // the emitted cloud is byte-identical. Persistent voxels overridden by a
    // transient voxel at the same coord are dropped; the transient grid walk
    // then re-emits them with their (decaying) dynamic evidence.
    pc_scratch_.clear();
    bgrid.forEachCell([&](const scovox::BetaVoxel& b, const Bonxai::CoordT& co) {
      if (b.p_occ() >= min_occ_ && has_beta_evidence(b) &&
          !(have_transient && transient_overrides(co)))
        pc_scratch_.emplace_back(co, b, false);
    });
    if (have_transient) {
      tbgrid.forEachCell([&](const scovox::BetaVoxel& b, const Bonxai::CoordT& co) {
        if (b.p_occ() >= min_occ_ && has_beta_evidence(b))
          pc_scratch_.emplace_back(co, b, true);
      });
    }
    if (pc_scratch_.empty()) { pc_pub_->publish(cl); return; }
    mod.resize(pc_scratch_.size());
    sensor_msgs::PointCloud2Iterator<float> ix(cl,"x"), iy(cl,"y"), iz(cl,"z"), ir(cl,"rgb"),
      ip(cl,"occupancy_prob"), ic2(cl,"semantic_confidence"),
      iv(cl,"posterior_variance"), ie(cl,"eig"),
      iao(cl,"a_occ"), iaf(cl,"a_free"),
      iau(cl,"a_unk"), isn0(cl,"sem_cnt0"), isn1(cl,"sem_cnt1");
    sensor_msgs::PointCloud2Iterator<uint8_t>  icl(cl,"semantic_class");
    sensor_msgs::PointCloud2Iterator<uint16_t> isc0(cl,"sem_cls0"), isc1(cl,"sem_cls1");
    const float vis_gate = sem_vis_thresh_ >= 0 ? (float)sem_vis_thresh_ : 0.5f;
    for (const auto& [co, b, is_transient] : pc_scratch_) {
      float pr = b.p_occ();
      const scovox::SemBetaVoxel v = project(b, co, is_transient ? tdacc : dacc);
      auto ps = bgrid.coordToPos(co);
      uint8_t bc = 0; float cf = 0, r = 1, gg = 1, bb = 1;
      if (v.a0() > 0 && scovox::K_TOP > 0) {
        const auto [best_cls, p_best] = scovox::argmaxClassConfidence(v);
        // argmaxClassConfidence returns a uint16_t id, but `semantic_class` is a
        // UINT8 PointField; a naive cast of an id >=256 would alias to id%256 and
        // collide with an unrelated class for both the label and the palette
        // colour. Emit 0 (unknown) instead. (Mirrors dscovox_node.cpp's guard.)
        bc = (best_cls < 256) ? static_cast<uint8_t>(best_cls) : 0;
        cf = p_best;
        if (cf >= vis_gate && bc < sem_col_.size()) { auto& c = sem_col_[bc]; r = c.r; gg = c.g; bb = c.b; }
      }
      uint32_t rp = (uint32_t(r * 255) << 16) | (uint32_t(gg * 255) << 8) | uint32_t(bb * 255);
      *ix = float(ps.x); *iy = float(ps.y); *iz = float(ps.z);
      // Pack the RGB bit pattern into the float field without a strict-aliasing
      // violation (a reinterpret_cast<float*> of a uint32_t* is UB and warns
      // under -Wstrict-aliasing). memcpy is the canonical bit-cast; the
      // optimizer folds it to a register move.
      float rgb_f; std::memcpy(&rgb_f, &rp, sizeof(rgb_f)); *ir = rgb_f;
      *ip = pr; *icl = bc; *ic2 = cf;
      *iv = scovox::variance(v); *ie = scovox::expectedInformationGain(v);
      *iao = v.a_occ; *iaf = v.a_free; *iau = v.a_unk;
      *isn0 = v.sem_cnt[0]; *isc0 = v.sem_cls[0];
      if constexpr (scovox::K_TOP >= 2) {
        *isn1 = v.sem_cnt[1]; *isc1 = v.sem_cls[1];
      } else {
        *isn1 = 0.0f; *isc1 = static_cast<uint16_t>(0xFFFF);
      }
      ++ix; ++iy; ++iz; ++ir; ++ip; ++icl; ++ic2; ++iv; ++ie; ++iao; ++iaf;
      ++iau; ++isn0; ++isc0; ++isn1; ++isc1;
    }
    pc_pub_->publish(cl);
  }

  // Publish a thin shell at the TSDF zero-crossing. Caller must hold
  // map_mtx_ (shared). Walks TsdfMap for the surface geometry then runs
  // labelPointCloud against the Dir (semantics) grid to attach the per-point
  // semantic class. The cross-grid join uses the 0xFFFF sentinel where the Dir
  // grid has no voxel at the surface coord (same convention labelMesh /
  // extractZeroCrossing already produce). 5-field schema.
  void publishTSDFPointCloud() {
    if (!tsdf_pub_ || !split_map_ || tsdf_pub_->get_subscription_count() == 0) return;
    const auto& tsdf_grid = split_map_->tsdf().grid();
    const double res = split_map_->resolution();
    if (res <= 0.0) return;

    auto points = scovox::extractZeroCrossing(
        tsdf_grid, (float)min_tsdf_w_, res);
    if (points.empty()) return;

    // Cross-grid label join — labelPointCloud queries the Dir grid at each
    // surface vertex's anchor voxel and writes the argmax class (sentinel
    // 0xFFFF if absent). Mirrors mesh_labelling.hpp::labelMesh for
    // triangle-mesh consumers.
    std::vector<Eigen::Vector3f> positions;
    positions.reserve(points.size());
    for (const auto& p : points) positions.push_back(p.position);
    auto labels =
        scovox::labelPointCloud(positions, split_map_->semsplit().dirGrid(), alpha_0_);

    sensor_msgs::msg::PointCloud2 cl;
    cl.header.frame_id = int_frame_;
    // Stamp at the last integrated scan time (TF-resolvable), not now().
    cl.header.stamp = (last_input_stamp_.nanoseconds() > 0) ? last_input_stamp_ : get_clock()->now();
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
    if (!split_map_ || sdf_trunc_launch_ <= 0.f) {
      rs->vertex_count = 0;
      rs->triangle_count = 0;
      return;
    }

    // Split-substrate mesh: TSDF zero-crossing geometry + Dir-grid labels.
    auto mesh = split_map_->extractMesh(rq->min_weight);
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
  // ── LiDAR + RGB-D fusion (opt-in via fuse_lidar_rgbd) ────────────────────────
  // When true the node subscribes BOTH streams into the one split_map_ and hands
  // each callback its own HitWeights profile (lidarProf()/rgbdProf()); when false
  // the profiles are unused and every integrateHit passes nullptr (global params_).
  bool fuse_lidar_rgbd_{false};
  scovox::HitWeights lidar_prof_{};   ///< LiDAR: occupancy+TSDF owner (high w_occ/w_free)
  scovox::HitWeights rgbd_prof_{};    ///< RGB-D: semantics-only (w_occ=0, geometry_off)
  std::string lidar_base_frame_, rgbd_base_frame_;  ///< per-stream ray-origin frames
  // Diagnostic-only: periodic per-grid memory/RSS logging. Off by default —
  // when on, scheduleMemUsage spawns a reader thread every ~10 frames and
  // walks the whole grid. Enable via `log_mem_usage:=true`.
  bool mem_log_{false};
  // The memlog worker is OWNED (not detached) so the destructor can join it:
  // a detached thread captures `this` and the grid/mutex/logger, and would
  // use-after-free if it were still walking the grid when rclcpp::spin()
  // returns and the node is torn down. mem_log_inflight_ enforces a single
  // worker at a time — if a grid walk outlasts 10 frames of integration we
  // skip launching a second one rather than letting two threads race on the
  // ${tsdf_dump_path_}.tmp file (interleaved writes + a racing std::rename
  // corrupt the dump). The previous worker is joined before a new one starts.
  std::thread mem_log_thread_;
  std::atomic<bool> mem_log_inflight_{false};
  // TF stability gate
  double startup_tf_stable_sec_{2.0}, startup_tf_jump_thresh_{0.5};
  bool runtime_tf_gate_{true};
  double runtime_tf_jump_thresh_{1.0};
  bool tf_stable_{false}, tf_prev_valid_{false}, did_initial_stabilize_{false};
  rclcpp::Time tf_stable_since_;
  Eigen::Vector3f tf_prev_pos_{Eigen::Vector3f::Zero()};
  size_t tf_rearm_count_{0}, frames_gated_{0};
  // Localization reject gate state (option 2): driven by /alignment_status.
  bool reject_gate_enable_{false};
  std::string alignment_status_topic_;
  double reject_gate_max_accepted_gap_sec_{0.5};
  int reject_gate_min_consecutive_{0};
  rclcpp::Subscription<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr status_sub_;
  // Plain members: the SingleThreadedExecutor serializes onAlignmentStatus
  // against the frame callbacks, so no synchronization is needed.
  bool loc_status_received_{false};
  double loc_accepted_gap_sec_{0.0};
  long loc_consec_rejects_{0};
  size_t frames_gated_reject_{0};
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
  // topk_probs_dir_ survives only as the param sink + construction input for
  // topk_; the per-frame cache + loader telemetry now live in TopkProvider.
  std::string topk_probs_dir_;
  int topk_topk_max_{5};
  std::unique_ptr<scovox::TopkProvider> topk_;
  bool trace_nr_{false}, pub_pc_, pub_plan_{false}, pub_tsdf_{true};
  double carve_band_{-1.0};
  double min_tsdf_w_{0.5};
  double plan_res_{0.2}, plan_sz_{80}, plan_ox_{-40}, plan_oy_{-40}, plan_zmin_{-1}, plan_zmax_{2}, plan_infl_{0};
  double plan_window_size_m_{20.0};
  std::atomic<bool> sm_dirty_{false};
  std::unordered_map<uint32_t,uint16_t> color_map_;
  // Launch param block (scovox::Params). Carries the node-level sensor filters
  // (range_decay_length, min_range, max_range, grazing_angle_threshold,
  // semantic_occ_gate, resolution, top_k) read by the integration + publish
  // paths. Owner now that the legacy fused map_ is gone.
  scovox::Params map_params_;
  // Raw launch sdf_trunc (0 == "TSDF off"). TsdfMap re-clamps a <=0 value back
  // up internally, so the tsdf publisher / extract_mesh service / TSDF cloud
  // gate on this rather than TsdfMap::params().sdf_trunc.
  float sdf_trunc_launch_{0.f};
  // Split-grid substrate. Owns one TsdfMap + one SemSplitMap
  // (BetaVoxel ∥ DirVoxel). Always allocated — the node has one path.
  std::unique_ptr<scovox::ScovoxMapSplit> split_map_;
  bool share_tsdf_{false};        // TSDF stream toggle (wire opts.share_tsdf)
  bool fused_walker_{true};       // Step 12.10 — single-DDA hit-ray walker
  // Semantic dataset priors. Defaults match SemSplitMap::Params; KITTI launches
  // override via num_classes:=20.
  int   num_classes_{14};
  float alpha_0_{scovox::kDefaultDirichletPrior};
  std::string tsdf_dump_path_{};  // audit hook — see [tsdf_dump] memlog branch
  std::vector<std_msgs::msg::ColorRGBA> sem_col_;
  // Class ids whose argmax routes a hit to the transient decaying grid rather
  // than the persistent map. Parsed from the `dynamic_classes` launch param;
  // empty disables the transient path (and its per-frame decay) entirely.
  std::unordered_set<uint16_t> dyn_cls_;
  // Last observed binary subscriber count. When this transitions from 0 to
  // >0 the next publish ships a full snapshot so a freshly-connected dscovox
  // sees this robot's complete current state, not just deltas since startup.
  size_t prev_sub_count_{0};
  // Timestamp of the most recent integrated input (scan / depth). The full-map
  // republishers (publishPointCloud + the TSDF cloud) stamp with THIS, not
  // now(): the localizer (e.g. GLIM) has a valid integration_frame<-...<-map TF
  // at each scan time, but its TF can lag wall/playback time. Stamping the
  // republished map at now() makes RViz fail "Could not transform <int_frame> ->
  // map" (lookup into the future). Stamping at the last scan time keeps the map
  // transformable into map/any frame at any playback rate.
  rclcpp::Time last_input_stamp_{0, 0, RCL_ROS_TIME};
  sensor_msgs::msg::CameraInfo di_; std::atomic<bool> have_di_{false};
  mutable std::shared_mutex map_mtx_;  // protects split_map_
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr di_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr input_pc_sub_;
  // ── Intra-scan deskew (gyro-based rotation correction) ──────────────────
  std::string deskew_mode_{"auto"}, imu_topic_, imu_frame_;
  double deskew_window_sec_{0.2}, imu_retention_sec_{1.0}, deskew_min_angle_deg_{0.0};
  Eigen::Vector3f gyro_bias_{Eigen::Vector3f::Zero()};
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  struct ImuSample { double t; Eigen::Vector3f w; };
  std::deque<ImuSample> imu_buf_;          // rolling gyro buffer (no lock: single-threaded executor)
  Eigen::Matrix3f R_lidar_imu_{Eigen::Matrix3f::Identity()};  // rotates gyro from imu frame → cloud frame
  bool extrinsic_valid_{false};
  struct DeskewKnot { float dt; Eigen::Quaternionf q; };       // dt since scan-start, ΔR(t0→t0+dt)
  std::vector<DeskewKnot> deskew_table_;   // per-scan scratch; rebuilt each scan
  size_t deskew_frames_{0}, deskew_fallback_{0};
  // Phase 2: intra-scan translation (sensor velocity differenced from scan poses).
  bool deskew_translation_{false};
  Eigen::Vector3f prev_scan_trans_{Eigen::Vector3f::Zero()};
  double prev_scan_t_{0.0};
  bool prev_scan_valid_{false};
  // TF placement timing (raw-cloud path): wait for the exact-stamp pose instead
  // of silently falling back to Time(0) (the previous scan's pose).
  double tf_lookup_timeout_sec_{0.2};
  bool tf_require_exact_{false};
  size_t tf_fallback_count_{0};
  // Uniform per-scan voxel-grid downsample (sensor frame), 0 = off. See decl in
  // declareNodeParams. Replicates GLIM's preprocess downsample inside scovox.
  double downsample_voxel_size_{0.0};
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> d_sub_, s_sub_;
  using ISP = message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image,sensor_msgs::msg::Image>;
  std::shared_ptr<message_filters::Synchronizer<ISP>> sync_;
  rclcpp::TimerBase::SharedPtr sm_timer_;
  rclcpp::Publisher<scovox_msgs::msg::ScovoxMap>::SharedPtr sm_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pc_pub_;
  // Reused scratch for publishPointCloud's single grid walk (collect matching
  // voxels once, then fill the message from this) so the Beta grid is traversed
  // once per publish, not twice. clear() retains capacity across publishes.
  // The bool marks a transient (dynamic-class) voxel so the fill step reads its
  // semantics from the transient Dir grid instead of the persistent one.
  std::vector<std::tuple<Bonxai::CoordT, scovox::BetaVoxel, bool>> pc_scratch_;
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
