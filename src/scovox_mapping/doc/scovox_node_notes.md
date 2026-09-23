# scovox_node.cpp — design notes and history

The long comments of `src/scovox_node.cpp`, moved out of the code on 2026-09-23 so the source carries short comments only. Where a comment was moved, the code keeps a short gist ending in `(notes: <id>)`; the section headed `<id>` below holds the original comment, word for word.

Sections follow the order of the source file and are grouped by the function (or section) they sit in. Each gives the line of code the comment was attached to and its original line number. Line numbers, dates, generation numbers and cross-references inside the moved text are as they were when written; they record history and are not maintained.

## Contents

- [SCovoxNode](#scovoxnode) — 4
- [rclcpp::create_timer](#rclcppcreate_timer) — 1
- [SCovoxNode (part 2)](#scovoxnode-part-2) — 3
- [~SCovoxNode](#scovoxnode) — 1
- [declareMapParams](#declaremapparams) — 3
- [declareNodeParams](#declarenodeparams) — 26
- [SCovoxNode — declarations](#scovoxnode--declarations) — 1
- [buildFusionProfiles](#buildfusionprofiles) — 2
- [setupSubscribers](#setupsubscribers) — 1
- [setupPublishers](#setuppublishers) — 1
- [SCovoxNode — declarations (part 2)](#scovoxnode--declarations-part-2) — 2
- [onImages](#onimages) — 2
- [SCovoxNode — declarations (part 3)](#scovoxnode--declarations-part-3) — 2
- [integrateLidarSnapshot](#integratelidarsnapshot) — 6
- [onPointCloud](#onpointcloud) — 2
- [scheduleMemUsage](#schedulememusage) — 1
- [integrateHit](#integratehit) — 1
- [carveNoReturnRays](#carvenoreturnrays) — 1
- [SCovoxNode — declarations (part 4)](#scovoxnode--declarations-part-4) — 1
- [publishBinaryMap](#publishbinarymap) — 6
- [SCovoxNode — declarations (part 5)](#scovoxnode--declarations-part-5) — 2
- [projectPlanningMap](#projectplanningmap) — 1
- [SCovoxNode — declarations (part 6)](#scovoxnode--declarations-part-6) — 1
- [publishPointCloud](#publishpointcloud) — 2
- [SCovoxNode — declarations (part 7)](#scovoxnode--declarations-part-7) — 7

## SCovoxNode

### params-cache-launch-block

**Why the launch param block is cached** — attached to `map_params_ = P;` (line 57)

```text
Cache the launch param block. scovox::Params still carries the
node-level sensor filters (range_decay_length, min_range, max_range,
grazing_angle_threshold, semantic_occ_gate, resolution, top_k) that the
integration + publish paths read; with the legacy map_ object gone this
is their owner.
```

### tsdf-enabled-intent

**Threading the TSDF-off intent to the substrate** — attached to `SP.tsdf_enabled = (sdf_trunc_launch_ > 0.f);` (line 89)

```text
enable_tsdf:false sets sdf_trunc=0, but TsdfMap::sanitise re-clamps that
back up to 0.15 — silently keeping the fused walker's per-voxel TSDF band
writes (TsdfVoxel alloc + Curless-Levoy average + touched push) running
even though the occupancy/LiDAR config never reads the TsdfMap grid
(tsdf_pub_ null, no extract_mesh, share_tsdf_=false). Thread the real
intent through so those dead writes are skipped. Default true keeps every
TSDF-on config — and all scovox_core gtests, which build with
sdf_trunc=0.15 — byte-identical; only the (unread) TsdfMap grid changes.
```

### semsplit-param-mapping

**SemSplitMap params mapped from launch** — attached to `SP.semsplit.w_free                  = P.w_free;` (line 98)

```text
SemSplitMap (de-unified BetaVoxel ∥ DirVoxel substrate): every
Bayesian / sparse-Dirichlet knob from launch P maps 1:1.
semantic_occ_gate / min_range / max_range / grazing_angle_threshold are
node-level sensor filters consumed BEFORE integrateHit so they are not
mirrored. evidence_saturation widens uint16→float.
```

### semsplit-dataset-priors

**Dataset priors num_classes and alpha_0** — attached to `SP.semsplit.num_classes             = num_classes_;` (line 116)

```text
num_classes / alpha_0 — dataset-dependent priors that govern the
OTHER bucket's prior mass `(C − K_TOP)·α_0`. Wrong num_classes shifts
the semantic prior and the eviction capacity (KITTI=20 vs NYU13=14).
Defaults match SemSplitMap::Params defaults (NYU13 / 0.01). KITTI
launches override num_classes:=20.
```

## rclcpp::create_timer

### timer-shared-lock

**One shared lock per publish timer tick** — attached to `std::shared_lock<std::shared_mutex> lock(map_mtx_);` (line 165)

```text
Hold one shared_lock for the whole timer tick so the ScovoxMap and
PointCloud always represent the same map state, and so neither
helper races against scheduleMemUsage's detached reader thread.
SingleThreadedExecutor already serializes us against onImages, but
wrapping here makes the contract explicit and survives a future
switch to MultiThreadedExecutor. publishScovoxMap and
publishPointCloud must NOT take map_mtx_ themselves —
std::shared_mutex is non-recursive, so re-locking here would be UB.
```

## SCovoxNode (part 2)

### share-timer-binary-publish

**The timer-owned binary publish** — attached to `bin_timer_ = rclcpp::create_timer(this, get_clock(),` (line 180)

```text
Timer-owned binary publish (share_rate_hz > 0): the sensor callbacks
skip their inline publishBinaryMap and touched coords accumulate until
this tick. Unique lock — publishBinaryMap drains the touched-sets and
writes the change-gate shadow grids, both mutations. Under the
SingleThreadedExecutor this also serializes us against integration.
```

### fine-region-sub-qos

**Refinement region subscription QoS** — attached to `fine_region_sub_ = create_subscription<scovox_msgs::msg::RefinementRegion>(` (line 203)

```text
Region registration interface. Reliable + latched (transient_local),
deep history — registrations are rare, small, and must not be lost:
latching replays the publisher's retained add/remove history to a
subscription that (re)joins late, and the replay converges because
adds are keyed-replace and removes idempotent. NB a transient_local
sub only matches transient_local publishers — manual `ros2 topic pub`
needs `--qos-durability transient_local`.
```

### semantic-deposit-log-line

**The sanitised deposit-rule log line** — attached to `{` (line 255)

```text
Print the SANITISED deposit rule, read back out of the constructed map,
never the launch argument. A binary predating either knob accepts the
parameter and ignores it, which is how a sweep silently measures nothing;
this line is the only thing that distinguishes "band ran" from "band was
spelled correctly". Drivers should grep for it, not for the launch arg.
```

## ~SCovoxNode

### dtor-join-memlog-worker

**Joining the memlog worker at shutdown** — attached to `if (mem_log_thread_.joinable()) mem_log_thread_.join();` (line 274)

```text
Join the diagnostic memlog worker (scheduleMemUsage) before any node
member is destroyed. The worker holds a shared_lock on map_mtx_ and
dereferences map_/split_map_/logger via the captured `this`; if it
were detached (the prior behaviour) it could still be walking the grid
when rclcpp::spin() returns and this object is torn down → UAF. Joining
here makes shutdown deterministic; the worker is read-only and bounded
by one grid walk, so the join is short.
```

## declareMapParams

### param-dir-leaf-bits

**Dir grid block size** — attached to `P.dir_leaf_bits = (uint8_t)std::clamp((int)dp("dir_leaf_bits", 2), 1, 4);` (line 290)

```text
Semantic (Dir) grid block size. Independent of leaf_bits because the Dir
grid is hit-only + Stream-B gated — ~18× sparser than the full-ray Beta
grid — so it wastes ~9 of every 10 slots in an 8³ block. SemSplitMap
clamps this to <= leaf_bits; pass dir_leaf_bits:=3 to restore the old
shared geometry exactly. See SemSplitMap::Params::dir_leaf_bits.
```

### param-sdf-trunc-voxels

**TSDF truncation in voxel units** — attached to `{` (line 308)

```text
TSDF: truncation distance is set in voxel units so it scales with
resolution across launch files. The whole legacy fused path treats
sdf_trunc==0 as "TSDF off" — no band walk in fused_integrate_ray_static,
no ~/tsdf_pointcloud publisher, no ~/extract_mesh service. `enable_tsdf`
(default true) is the explicit off-switch that forces it there; split
mode (use_split=true) can't honor it (TsdfMap re-clamps sdf_trunc<=0 in
tsdf_map.cpp), so the constructor warns. `carve_band` is independent.
```

### param-semantic-mechanism-knobs

**Semantic mechanism knobs origin** — attached to `semantic_topk_trunc_    = dp("semantic_topk_trunc", 0);` (line 349)

```text
Semantic mechanism knobs (SceneNN mechanism round). Shipped defaults are
off/0, so an unmodified launch is byte-identical to before.
```

## declareNodeParams

### param-fuse-lidar-rgbd

**The LiDAR plus RGB-D fusion switch** — attached to `fuse_lidar_rgbd_ = dp("fuse_lidar_rgbd", false);` (line 390)

```text
Master switch. Default false → today's either/or single-sensor path: every
integrateHit passes prof=nullptr, so the substrate reads the global map
weights and behaviour is byte-identical. true → the node subscribes BOTH
streams and hands each its own HitWeights profile (built in buildFusionProfiles),
so LiDAR and RGB-D write the ONE SemSplitMap with their own sensor models.
```

### param-mode-rolling-persistent

**Rolling versus persistent mode** — attached to `mode_ = dp("mode", std::string("rolling"));` (line 409)

```text
mode = "persistent": single-robot, no binary publish to dscovox.
mode = "rolling":    publishes ScovoxMapBinary updates for the merger
                     and the planning_map is a rolling crop around the
                     robot (Phase 2). The underlying voxel grid is
                     fully persistent in both modes — no pruning.
```

### planning-terrain-relative

**Terrain-relative planning map projection** — attached to `plan_terrain_rel_ = dp("planning_map_terrain_relative", false);` (line 427)

```text
Terrain-relative projection (3D/hilly sites). When true the absolute
[planning_map_min_z, planning_map_max_z] band is IGNORED; instead each
XY column is classified against a band RELATIVE to that column's own
ground elevation (lowest occupied voxel + contiguous stack walk capped
at planning_map_ground_stack_m, absorbing residual vertical smear).
A column whose relative band [rel_min_z, rel_max_z] above the ground
top contains an occupied voxel is blocked (100); a column with
observed ground and a clear band is free (0); columns with no occupied
voxel (free-only / unobserved) stay unknown (-1). Obstacles shorter
than ~ground_stack_m above the detected ground merge into the ground
stack and read traversable. NB: rel_min_z must stay > 0 or the ground
itself blocks every cell.
```

### planning-global-map

**Why a second world-fixed planning map** — attached to `pub_plan_glob_ = dp("publish_global_planning_map", false);` (line 448)

```text
The planning_map above is sized for the local navigation planner: in
mode=rolling it is a small robot-centred crop, and its extent IS that
planner's window (simple_nav_3d has no separate window param). An
exploration planner needs the opposite: a fixed envelope covering the
whole ROI, because it rejects candidates whose cell is out of bounds
(isCellOccupied treats out-of-bounds as occupied) and measures its
coverage-termination unknown fraction over the ROI clipped to the grid.
Point both at one topic and you must pick which consumer to break, so
this is a SECOND publisher over the same voxel grid with its own
envelope, resolution and rate. Off by default: nothing that does not ask
for it pays the cost, and mode=persistent (the other way to get a fixed
envelope) is not an option because it disables the ScovoxMapBinary
publish that multi-robot map sharing depends on.
```

### planning-global-period

**Global planning map publish period** — attached to `plan_glob_period_ = dp("global_planning_map_period_sec", 1.0);` (line 467)

```text
Publish period, seconds. Unlike planning_map this one is NOT free to
emit per integration frame: the inflation pass is O(occupied * (r/res)^2)
over the whole envelope and the message is O(size^2/res^2) bytes, both of
which run on the integration thread. An exploration planner re-reads the
latched map about once per planning step, so ~1 Hz is already generous.
```

### share-tsdf-toggle

**The share_tsdf wire toggle** — attached to `share_tsdf_ = dp("share_tsdf", false);` (line 480)

```text
Sender-side wire toggle for the TSDF stream:
  share_tsdf=false (default): emit Beta + Dir only (dscovox-fusion-only
    path; each robot keeps its local TSDF).
  share_tsdf=true: also emit the TSDF stream (opt-in for fused-geometry
    consensus). Maps to BinarySerializer::Options.share_tsdf.
  The rev-7 fine-TSDF stream rides the same toggle: it ships iff
  share_tsdf is on AND the fine band is enabled (fine_ratio_log2 > 0).
```

### share-dir-toggle

**The share_dir wire toggle** — attached to `share_dir_ = dp("share_dir", true);` (line 488)

```text
Sender-side wire toggle for the Dir (semantics) stream — share_tsdf's
mirror, default ON so the wire is unchanged unless a robot opts into
geometry-only sharing (E6.9):
  share_dir=false: elide the Dir section (dir_count=0 — a zero-length
    stream is already legal wire, same codec rev). Receiver contract,
    verified in dscovox onBinaryMap: ingest is snapshot-replace per
    RECORD and the refold walks only arrived cells, so an absent stream
    means "no semantic update", never "erase semantics" — a peer keeps
    any semantics it already holds, it just stops receiving updates.
```

### fine-band-params

**Fine TSDF band parameters** — attached to `fine_ratio_log2_      = std::clamp<int>((int)dp("fine_ratio_log2", 0), 0, 8);` (line 499)

```text
docs/design/fine_tsdf_band_dbh_2026_07_30.md. fine_ratio_log2 = 0 (the
default) disables everything (no fine grid, no subscription). k = 2 at
resolution 0.10 → 2.5 cm fine voxels. The node's job ends at PRODUCING
the fine lattice; measurement (e.g. the DBH circle fit, dbh_fit.hpp) is
post-processing on the shared/saved map, not a mapping responsibility.
```

### fine-raw-returns

**Raw returns for refinement regions** — attached to `fine_raw_returns_  = dp("fine_raw_returns", true);` (line 514)

```text
Full sensor density for refinement regions: when the per-scan voxel-grid
downsample is active, in-region raw (deskewed) returns are ALSO routed
straight to the fine lattice (refineHit — fine-band-only, no coarse
write), so the fine band sees the sensor's native point density instead
of one return per downsample cell. No effect when downsampling is off
(the full cloud already reaches integrateHit).
```

### share-rate-hz

**Share publish rate and coalescing** — attached to `share_rate_hz_ = dp("share_rate_hz", 0.0);` (line 524)

```text
share_rate_hz: cadence of the binary delta publish. <=0 (default) keeps
the legacy per-scan publish inline in the sensor callbacks. >0 moves the
publish onto a wall timer at this rate; touched coords accumulate
between ticks and drainTouched* sort+uniques them, so slower rates
coalesce repeated writes of the same voxel into ONE wire record. The
receiver merge is snapshot-replace per (source, coord), so coalescing is
lossless — the merger converges to the same state either way.
```

### share-change-gate

**The per-voxel change gate** — attached to `share_change_gate_ = dp("share_change_gate", true);` (line 532)

```text
share_change_gate: per-voxel change gate against the LAST-EMITTED wire
state. A touched voxel is re-emitted only when its posterior actually
moved: |Δp_occ| > share_gate_p_eps, relative total-evidence growth >
share_gate_evidence_rel, or (Dir) a top-K class slot changed. This kills
the dominant waste stream — saturated / effectively-unchanged free-space
carve voxels re-shipped every scan forever. A voxel's FIRST observation
always emits (it has no gate entry), so planner frontiers are never
delayed, and full snapshots (new-subscriber path) bypass the gate.
Costs a shadow copy of the emitted Beta/Dir state (~8/16 B per emitted
voxel). false = legacy wire, byte-identical to before this gate existed.
```

### share-gate-mode

**Gate trigger modes** — attached to `share_gate_mode_ = dp("share_gate_mode", std::string("significance"));` (line 553)

```text
share_gate_mode selects the emit TRIGGER when share_change_gate is on:
  "significance"      (default) — |Δp_occ| > τ OR relative evidence
                      growth > share_gate_evidence_rel (κ = 1 + rel).
                      This is the OR-form significance gate; the wire
                      carries the full posterior either way.
  "state_flip"        — OctoMap-equivalent baseline: emit only when the
                      thresholded state flips (Beta: p_occ crosses
                      share_stateflip_p_occ; Dir: dominantClass changes).
                      Payload unchanged (full posterior).
  "state_flip_binary" — MARBLE-equivalent baseline: state_flip trigger
                      AND the payload is binarized — Beta collapsed to
                      prior + share_binarize_evidence on the winning
                      side, Dir to a one-hot argmax slot; a voxel with
                      no dominant class ships no Dir record at all (the
                      baseline cannot express "uncertain"). Wire FORMAT
                      is unchanged, so the byte advantage a purpose-built
                      binary codec would add must be credited
                      analytically in E6.6's equal-bandwidth comparison.
The any-change baseline row is share_change_gate:=false, not a mode.
```

### share-gate-tau-n

**Evidence-dependent gate threshold** — attached to `share_gate_tau_ref_n_ = dp("share_gate_tau_ref_n", 0.0);` (line 582)

```text
τ(n) ablation (E6.6 req. ④): with tau_ref_n > 0 the mean-arm threshold
shrinks once last-sent evidence n exceeds it:
  τ_eff = τ · (tau_ref_n / n)^tau_n_pow.
pow = 1 is the design doc's 1/n; pow = 0.5 is the constant-KL rate in
the quadratic regime (KL ≈ n·Δp²/2p(1−p) ⇒ Δp* ∝ n^-1/2). 0 = fixed τ.
```

### share-heartbeat

**Per-voxel heartbeat re-emit** — attached to `share_heartbeat_sec_ = dp("share_heartbeat_sec", 0.0);` (line 589)

```text
share_heartbeat_sec: per-voxel re-emit period. Loss healing plus the
liveness half of the negative-information contract — a receiver that
heard nothing about an emitted voxel for longer than this may treat the
silence as "unchanged within τ", not "unheard". Needs the gate's
last-emit state, so it requires share_change_gate:=true. 0 = off.
```

### share-max-voxels-per-msg

**Chunking a tick into messages** — attached to `share_max_voxels_per_msg_ = static_cast<int>(dp("share_max_voxels_per_msg", 0));` (line 604)

```text
share_max_voxels_per_msg: >0 splits one publish tick's deltas across
ceil(total/N) self-contained ScovoxMapBinary messages (each carries the
full envelope + pose; each LZ4-compressed separately). The receiver merge
is snapshot-replace per (source, coord), so chunk boundaries cannot
change the converged state — only delivery dynamics (loss blast radius
vs per-message overhead and LZ4 ratio). 0 (default) = one message per
tick, the legacy wire behaviour.
```

### share-chunk-interleave

**Chunk interleave versus section drain** — attached to `share_chunk_interleave_ = dp("share_chunk_interleave", true);` (line 612)

```text
share_chunk_interleave: how the chunker DIVIDES a tick between the four
sections. true (default) = proportional interleave, so any prefix of the
chunk sequence carries each section in proportion to its size and a
receiver gets semantics from the first message. false = the legacy
section-at-a-time drain (tsdf → beta → dir → fine), which puts every
semantic record behind the entire occupancy stream. Inert unless
share_max_voxels_per_msg > 0, and it cannot change the converged state
either way (merge is replace-per-(source, coord)) — kept as a knob so
the two orderings can be A/B'd in one binary.
```

### share-max-bytes-per-tick

**Per-tick wire byte budget** — attached to `share_max_bytes_per_tick_ =` (line 622)

```text
share_max_bytes_per_tick: > 0 caps the COMPRESSED bytes put on the wire
in one publish tick; chunks past the cap are deferred FIFO to later
ticks. This is the burst shaper the voxel cap above is not: chunking
bounds message SIZE but still emits every chunk back-to-back within the
tick. Deferral cannot change the converged state — chunks carry
absolute state, the publisher preserves order, and the receiver merge
is replace-per-(source, coord) — it trades burst height for convergence
delay. Every tick sends at least one message even if that message alone
exceeds the budget (progress guarantee), so without chunking the cap
degenerates to one-whole-tick-message-per-tick — hence the warning.
0 (default) = unlimited: the legacy publish path, wire-identical.
```

### share-roi-z-band

**Shared z band and its couplings** — attached to `share_roi_z_min_ = dp("share_roi_z_min", 0.0);` (line 641)

```text
share_roi_z_min/max: vertical band (integration frame, metres) outside
which voxels stay OFF the wire (both Beta and Dir streams). The LOCAL
map is untouched — this filters only what is shared, so out-of-band
structure survives in each robot's own map. min >= max (the 0/0
default) disables the band.
KEEP IN SYNC with dscovox_node share_roi_z_min/share_roi_z_max (the
receiver-side defensive clip) and with explo_planner
shared_params.yaml roi_min_z/roi_max_z: the shared band must be a
SUPERSET of the planner band, or free voxels at the band edge arrive
clipped and read as unknown to the planner.
```

### param-semantic-priors

**Semantic prior parameters** — attached to `num_classes_ = std::max<int>(scovox::K_TOP + 1,` (line 656)

```text
Semantic priors. num_classes is the dataset's total class count — sets
the OTHER bucket's prior mass to (num_classes − K_TOP) · alpha_0 so the
implicit Dirichlet still marginalises onto the true (C+1)-category
distribution. Defaults match SemSplitMap::Params (NYU13 / 0.01). KITTI
launches override num_classes:=20; Replica/SceneNet stay at 14.
```

### tsdf-dump-path

**TsdfMap audit dump format** — attached to `tsdf_dump_path_ = dp("tsdf_dump_path", std::string{});` (line 666)

```text
Audit hook (split-grid only): when non-empty, every periodic memlog
tick overwrites this path with a flat binary snapshot of TsdfMap:
  uint64_t n; then n × {float x, float y, float z, float distance,
  float weight} = 20 B/voxel, voxel-centre coords in scovox_node's
  world frame. Lets a parity-test harness compare the TsdfMap voxel
  set against SLIM-VDB's voxels.bin after a Tr_inv frame conversion
  (see tools/tsdf_parity_test.py). Empty default → no-op.
```

### deskew-mode

**Deskew mode parameter** — attached to `deskew_mode_ = dp("deskew_mode", std::string("auto"));` (line 676)

```text
deskew_mode: "auto" (deskew iff the cloud has a per-point time field),
"on"/"gyro" (force; warn if the field is missing), or "off" (never — set
this for already-deskewed feeds like /glim_ros/points, which still carry a
`t` field and would otherwise be double-corrected). Phase 1 is rotation
only: each point is rotated from its capture time back to scan-start using
gyro integrated across the scan, then placed by the single scan-start pose.
```

### tf-lookup-timeout

**Waiting for the exact-stamp TF** — attached to `tf_lookup_timeout_sec_ = dp("tf_lookup_timeout_sec", 0.2);` (line 695)

```text
TF placement timing. The raw /ouster/points scan reaches scovox at the same
instant it reaches GLIM, so GLIM has not yet computed/broadcast the
odom<-os_lidar pose for that stamp. With a short timeout the exact-stamp
lookup fails and we fall back to Time(0) (the PREVIOUS scan's pose) →
mis-placed scan → accumulation smear. tf_lookup_timeout_sec lets scovox
WAIT for GLIM's TF (the TransformListener fills the buffer on its own
thread, so this blocks only the main loop, not TF intake). tf_require_exact
drops a scan rather than integrating it at a stale Time(0) pose.
```

### downsample-voxel-size

**Per-scan voxel-grid downsample** — attached to `downsample_voxel_size_ = dp("downsample_voxel_size", 0.5);` (line 705)

```text
Uniform voxel-grid downsample, applied per-scan in the SENSOR frame BEFORE
integration (after deskew) — this is what GLIM does in preprocessing
(config_preprocess.json: voxel-grid @ downsample_resolution). The raw
full-res cloud over-samples the noisy surface so every scan fills the tails
of the per-column z-distribution → thick smear; collapsing points to one
centroid per voxel cuts that tail-sampling without throwing away coverage.
0.0 = off (full per-point path, unchanged). Geometric only: when >0 the
per-point semantic/top-k labels are dropped (fine for the raw LiDAR path).
Default 0.5 = the swept optimum for coarse-map thinness (see the sweep
table in scovox_lidar_raw_deskew.yaml: 4x thinner shared columns than
0.1 for only -7% footprint). Configs may override (geometric/fused run
0.1 = map resolution); refinement regions are unaffected either way —
in-region raw returns bypass this via refineHit (fine_raw_returns).
```

### runtime-tf-gate

**Runtime TF divergence guard** — attached to `runtime_tf_gate_ = dp("runtime_tf_gate", true);` (line 728)

```text
Runtime divergence guard. Once the startup gate has declared the pose
stable, keep watching frame-to-frame pose jumps. A jump larger than
runtime_tf_jump_threshold means localization has diverged / teleported
(e.g. the NDT track lost lock): drop that frame AND re-arm the startup
stabilization so we stop integrating against the bad pose until it
settles again. Set runtime_tf_gate=false to keep the legacy
startup-only behaviour. The runtime threshold should sit above real
frame-to-frame motion (walking ~0.1-0.15 m at 10 Hz) so normal travel
never trips it.
```

### reject-gate

**Localization reject gate** — attached to `reject_gate_enable_ = dp("reject_gate_enable", false);` (line 739)

```text
Localization reject gate. A frame-to-frame jump gate cannot see a pose
that is *frozen* — when an external localizer (e.g. NDT map-matcher)
loses lock it rejects scans, stops updating its pose, but keeps
re-broadcasting the stale transform on a timer. The TF therefore looks
fresh and jump-free while the robot keeps moving, so scovox would smear
every new scan onto the stuck pose. This gate subscribes to the
localizer's /alignment_status (diagnostic_msgs/DiagnosticArray) and skips
integration whenever it reports the pose is stale: accepted_gap_sec (time
since the last accepted update) exceeds reject_gate_max_accepted_gap_sec,
or consecutive_rejected_updates reaches reject_gate_min_consecutive.
```

### topk-probs-dir

**Soft-probability top-K directory** — attached to `topk_probs_dir_ = dp("topk_probs_dir", std::string(""));` (line 753)

```text
Soft-probability ablation: directory of <frame>.topk flat-binary blobs
produced by topk_npz_to_bin.py. When non-empty, scovox_node uses the
frame index (low 16 bits of header.stamp.nanosec, set by the replay
node) to look up per-point/per-pixel top-K class distributions and
feeds them into the Dirichlet update instead of the one-hot built
from the hard label. Must contain only zero-padded names like
"000000.topk". Empty string = legacy hard-label path.
```

## SCovoxNode — declarations

### fusion-profiles

**Per-source HitWeights profiles** — attached to `void buildFusionProfiles(const scovox::Params& P) {` (line 777)

```text
Build the per-source HitWeights profiles used when fuse_lidar_rgbd_ is on.
LiDAR defaults fall back to the global map weights (parity with the single-
sensor path when only the master switch is flipped); override lidar_w_occ:=8
etc. for the high-evidence ToF calibration (map_interface.hpp). RGB-D is
"pure LiDAR authority": w_occ=0 (Stream A skipped → occupancy stays LiDAR-
built, Stream B gates on it), w_free=0 (no carve onto the shared Beta grid),
geometry_off=true (no TSDF band). A/B "zero vs small w_occ" is a param flip.
```

## buildFusionProfiles

### fusion-rgbd-min-p-occ

**RGB-D semantic gate above the prior** — attached to `rgbd_prof_.dirichlet_min_p_occ = (float)dp("rgbd_dirichlet_min_p_occ", 0.55);` (line 795)

```text
Gate MUST be strictly above the Beta(1,1) prior (p_occ=0.5): with
rgbd_w_occ=0 an RGB-D hit on a voxel LiDAR never touched allocates a Beta
voxel at prior, and the DIRICHLET gate is `p_occ_post >= min_p_occ`, so a
0.5 default would commit semantics on prior-only geometry — defeating pure
LiDAR authority. 0.55 rejects the prior AND LiDAR-carved-free voxels while a
single LiDAR hit (p_occ≈0.9, even a weak q≈0.05 hit ≈0.58) admits. Raise
toward 0.6 for stricter authority (worsens leading-edge temporal recall).
```

### fusion-rgbd-kernel-radius

**RGB-D to LiDAR spread radius** — attached to `rgbd_prof_.kernel_radius = (float)dp("rgbd_kernel_radius", 0.0);` (line 804)

```text
RGB-D→LiDAR BKI spread radius `l` (metres). 0 = classic exact-voxel gate
(RGB-D labels a voxel only if its own endpoint coincides with a LiDAR-
occupied voxel — starved by the LiDAR downsample). >0 spreads each RGB-D
label onto every LiDAR-occupied voxel within `l` via the S-BKI kernel.
Start ~0.4 (LiDAR length-scale from Gan et al.); ~1 voxel at resolution
0.10. Cost scales as (2·l/res+1)³ persistent-Beta lookups per RGB-D point.
```

## setupSubscribers

### lidar-input-qos

**LiDAR input subscription QoS** — attached to `const bool reliable_input = this->declare_parameter<bool>("input_reliable_qos", false);` (line 854)

```text
Best-effort by default to match the real robot's LiDAR driver, which
publishes best-effort. A reliable sub would refuse a best-effort
publisher and we would silently get no cloud. Best-effort still connects
to reliable publishers too (e.g. a bag replay), so this is safe.
BUT: over localhost with a small OS UDP buffer (net.core.rmem_max), a
best-effort link silently drops fragments of large (2+ MB) PointCloud2
scans, losing most frames on a KITTI replay. input_reliable_qos:=true
selects a reliable sub so a reliable publisher retransmits lost
fragments — full frame delivery for offline eval.
```

## setupPublishers

### binary-pub-qos

**Binary delta publisher QoS** — attached to `auto bin_qos = rclcpp::QoS(rclcpp::KeepLast(50)).reliable();` (line 908)

```text
Explicit reliable + deeper queue for binary deltas. dscovox_node
mirrors this. The int-overload (just `, 10`) was nominally
RELIABLE but combined with the subscriber's SystemDefaultsQoS
(which resolved to BEST_EFFORT here) the connection downgraded
and silently dropped large submap payloads. Pinning both ends
explicitly removes the ambiguity.
```

## SCovoxNode — declarations (part 2)

### tf-gate-pass

**The TF quality gate** — attached to `bool tfGatePass(const Eigen::Vector3f& O) {` (line 980)

```text
TF quality gate shared by the depth and LiDAR paths. Returns true when the
observer pose `O` (sensor origin in the integration frame) is trustworthy
enough to integrate this frame. Two stages:
  (1) Startup stabilization — wait until the pose has been jump-free
      (< startup_tf_jump_threshold) for startup_tf_stable_sec before EVER
      integrating. Guards against ghost voxels at the origin while TF is
      briefly wrong at startup.
  (2) Runtime divergence guard — once stable, a frame-to-frame jump larger
      than runtime_tf_jump_threshold means localization diverged: drop the
      frame and re-arm stage (1) so integration pauses until the pose
      settles again.
Records tf_prev_pos_ on every frame (gated or not) so the jump is always
measured against the immediately preceding pose (the legacy code froze
tf_prev_pos_ once stable).
```

### rgbd-snapshot-seam

**RGB-D snapshot and integrate seam** — attached to `struct DepthSnapshot {` (line 1078)

```text
The sensor callback captures the frame and the capture-time pose into a
DepthSnapshot, then hands it to integrateDepthSnapshot below. Integration
reads pose from the snapshot only — no TF touch — so the map update depends
solely on the bundled pose, mirroring dscovox's pose-rides-with-the-data.
```

## onImages

### rgbd-exact-stamp-pose

**RGB-D exact-stamp pose policy** — attached to `Eigen::Isometry3f T_oo;` (line 1201)

```text
RGB-D semantic frames MUST integrate at the EXACT capture-time pose. The old
code fell back to Time(0) (latest pose) on an exact-stamp miss, but seg adds
~250 ms of inference latency, so the depth stamp is ~250 ms old and "latest"
is ahead by the robot's motion — those semantic points smear into the map
(~5% of frames in testing). Reject instead: drop the frame on any exact-stamp
miss. (LiDAR onPointCloud keeps its own fallback; this policy is RGB-D only.)
```

### rgbd-clear-touched-persistent

**Clearing touched buffers in persistent mode** — attached to `split_map_->clearTouchedTsdf();` (line 1249)

```text
No bin_pub_ in persistent mode → publishBinaryMap is never
called → TsdfMap/SemSplitMap touched buffers grow unbounded
(every integrated ray appends coords). Clear via the O(n)
path: drainTouched* sorts+uniques, but the result is unused here,
so a plain clear is ~µs. The bin_pub_ branch above still uses
drainTouched* for the wire-format dedup it needs.
```

## SCovoxNode — declarations (part 3)

### deskew-table

**The deskew rotation table** — attached to `bool buildDeskewTable(double t0, double window) {` (line 1346)

```text
Build the per-scan cumulative-rotation table over [t0, t0+window] by
strapdown-integrating buffered gyro expressed in the sensor frame:
  ΔR(t0→τ+dτ) = ΔR(t0→τ) · Exp(ω_s·dτ),  ω_s = R_lidar_imu · ω_imu.
Knot k stores (τ_k−t0, ΔR(t0→τ_k)); the point loop slerps between knots and
applies ΔR to map each point back to the scan-start sensor frame. Returns
false (→ no deskew) when the buffer can't cover the scan.
```

### lidar-snapshot-seam

**LiDAR snapshot and integrate seam** — attached to `struct LidarSnapshot {` (line 1387)

```text
onPointCloud snapshots the cloud + capture-time pose into a LidarSnapshot,
then hands it here. Integration reads pose from the snapshot only. The one
TF touch left inside is ensureLidarImuExtrinsic — a static sensor<-imu
calibration (not a robot pose), the documented carve-out from the seam.
```

## integrateLidarSnapshot

### lidar-range-gate-squared

**Early squared range gate** — attached to `const bool need_rng = (P.range_decay_length > 0) || (carve_band_ > 0);` (line 1427)

```text
Range gate in SQUARED distance, applied to the RAW per-point range
(x^2+y^2+z^2) at parse time — BEFORE any deskew, binning, or transform
work. |p| in the sensor frame IS the sensor's measured range (min/max
range are sensor-relative specs: min kills self-returns/dust at the
housing, max kills clamp returns), and the rotation deskew preserves it,
so the early gate is exact and an out-of-range return costs three
multiplies and nothing else. The old post-transform gate measured from
the observer O instead — off by the static base->sensor offset plus the
v*dt translation-deskew term — and let out-of-range points into the
downsample medoid statistics, where a just-out-of-range medoid dropped
its voxel's valid returns. `rng` (range-decay weight / carve) is still
taken lazily from (Hp-O).norm() only when a consumer needs it. The
(>0?sq:raw) guard keeps the degenerate negative-threshold case
bit-identical: x->x^2 is monotonic only on [0,inf), and range>=0 always,
so a negative min/max threshold must be compared as-is (it can only ever
always-pass/always-fail).
```

### lidar-validate-buffer

**Validating PointCloud2 buffer geometry** — attached to `if (step <= 0) { RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,` (line 1451)

```text
Validate the buffer geometry before any reinterpret_cast read. A
malformed or truncated PointCloud2 (point_step too small for the declared
field offsets, or data shorter than width*height*point_step) would
otherwise drive an out-of-bounds read in the per-point loop below — a
crash or silent garbage integration on adversarial / buggy input. Valid
clouds always satisfy these (data.size() == height*row_step >=
width*height*point_step, and every field offset+size <= point_step).
```

### lidar-deskew-decision

**Per-scan deskew decision** — attached to `const double t0_sec = last_input_stamp_.seconds();` (line 1487)

```text
auto/on: deskew iff the cloud carries a per-point time field AND a gyro
table can be built for this scan. off: never (already-deskewed feeds).
Any missing prerequisite → fall back to the legacy single-pose path so we
never integrate a half-built correction.
```

### lidar-translation-deskew

**Translation deskew** — attached to `Eigen::Vector3f v_odom = Eigen::Vector3f::Zero();` (line 1516)

```text
Shift each point's endpoint by the sensor's odom-frame velocity × its time
offset, so a point captured at t_i is placed at the sensor position at t_i
(not at scan-start). Velocity is differenced from consecutive scan poses
(no IMU accel, no latency). Endpoints only; the carve origin O stays at
scan-start (a ~0.1 m shift over a scan, negligible for free-space carving).
```

### lidar-downsample-medoid

**Medoid voxel-grid downsample** — attached to `size_t ds_in = 0, ds_out = 0;` (line 1556)

```text
Uniform voxel-grid downsample (sensor frame): integrate one *real* return
per voxel — the measured point nearest the voxel centroid (a medoid, not the
synthetic centroid, which drifts off-surface into free space). GLIM does this
in preprocessing; on the raw cloud it collapses the dense over-sampling that
fills the per-column z-tails (the smear). Geometric only (no semantics/top-k)
— used for the raw LiDAR path.
```

### lidar-fine-raw-returns

**Full-density returns for the fine band** — attached to `if (fine_raw_returns_ && split_map_->fineEnabled() &&` (line 1628)

```text
Full sensor density for refinement regions: the medoid downsample
above is the coarse map's diet, but the fine lattice wants every
return the sensor produced. Route each raw (deskewed) return through
refineHit — fine-band-only (one O(1) region lookup, no coarse write),
so out-of-region points cost a hash probe and nothing else. The
medoid itself is skipped here: it reaches the fine band through
integrateHit below, so no return is fused twice.
```

## onPointCloud

### lidar-tf-exact-then-fallback

**LiDAR TF exact stamp then fallback** — attached to `const auto tf_to = rclcpp::Duration::from_seconds(tf_lookup_timeout_sec_);` (line 1730)

```text
TF: sensor frame -> integration frame (NO kR rotation — LiDAR is already ROS convention).
Wait up to tf_lookup_timeout_sec_ for the EXACT-stamp pose; only fall back
to Time(0) (the previous scan's pose) if tf_require_exact_ is false. A
Time(0) fallback mis-places the whole scan and is the prime suspect for the
accumulation smear, so it is counted + warned.
```

### lidar-clear-touched-persistent

**Clearing touched buffers in persistent mode (LiDAR)** — attached to `split_map_->clearTouchedTsdf();` (line 1783)

```text
No bin_pub_ in persistent mode → publishBinaryMap is never
called → TsdfMap/SemSplitMap touched buffers grow unbounded
(every integrated ray appends coords). Clear via the O(n)
path: drainTouched* sorts+uniques, but the result is unused here,
so a plain clear is ~µs. The bin_pub_ branch above still uses
drainTouched* for the wire-format dedup it needs.
```

## scheduleMemUsage

### memlog-single-inflight

**Single in-flight memlog worker** — attached to `bool expected = false;` (line 1820)

```text
Single-in-flight guard: if the previous walk is still running (a grid
walk can outlast the ~10-frame relaunch cadence on large maps), skip
this tick instead of spawning a second worker. Two concurrent workers
would both open ${tsdf_dump_path_}.tmp and race on std::rename,
producing a truncated/corrupt dump. CAS so only one launcher wins.
```

## integrateHit

### integrate-hit-carve-band

**Split-grid hit integration and carve band** — attached to `Eigen::Vector3f co = O;` (line 1918)

```text
Split-grid path. TsdfMap walks the SDF band, SemSplitMap walks the carve
band leading up to the hit. `q` already bakes in the range/grazing
weights (rw*aw) at the call site.

carve_band: when `carve_band_ > 0` (Replica / KITTI launch default =
0.1), walk the semantic carve along only the last `carve_band` metres
before the surface, matching the production mIoU baselines. carve_band
<= 0 falls back to full-ray.
```

## carveNoReturnRays

### carve-no-return-rays

**Carving along no-return rays** — attached to `for (auto& hf : nr_eps) split_map_->integrateMiss(O, hf, 1.0f, prof);` (line 1956)

```text
Beta-only carve along no-return rays (no TSDF surface to anchor).
q=1.0f matches the legacy carve which doesn't apply rw/aw. `prof` carries
the per-source w_free — a semantics-only source (RGB-D, w_free=0) deposits
NO a_free here, so its many no-return (sky/far) rays can't erode LiDAR
occupancy on the shared Beta grid.
```

## SCovoxNode — declarations (part 4)

### binary-change-gate-predicates

**Binary publish and change-gate predicates** — attached to `double gateTauEff(float n_last) const {` (line 2022)

```text
Publish only the voxels that have been touched since the last call. The
dscovox merger keys per-source grids by header.frame_id and overwrites
the matching voxels per binary; voxels not in the binary are kept, so
the merger's view stays exactly in sync with this robot's persistent
grid as long as it sees every dirty voxel at least once.

To handle a fresh dscovox connecting after this node has already started,
we detect subscriber-count transitions from 0 to >0 and re-mark every
non-prior cell as dirty so the next publish carries a full snapshot.
Split-substrate binary publish path. Drains touched TSDF + Beta + Dir
coords from the SemSplitMap substrate, reads each voxel's current state,
builds a BinarySerializer::Frame (three streams), optionally elides the
TSDF section per share_tsdf_, LZ4-compresses, and publishes with
msg->version=5. Beta (occupancy) and Dir (semantics) cross the wire as
SEPARATE streams — the receiver merges each with its own conjugate rule
(consensus_merge.hpp), losslessly.

Snapshot-on-resub + at-prior elision are applied per grid. This is the
node's only wire path; the SPLIT substrate (semsplit()) is always valid.
Change-gate predicates: has this voxel moved enough since its LAST-EMITTED
wire state to justify re-shipping? (share_change_gate, declareNodeParams.)
Evidence growth is measured RELATIVE to the emitted state, so a voxel that
keeps accumulating same-p carve evidence re-emits at a geometric (not
per-scan) cadence, and a saturated voxel (evidence cap reached, value
frozen) never re-emits at all.
τ(n) ablation: effective mean-arm threshold given the LAST-SENT evidence
(the receiver's belief mass — that is what the trigger's KL is against).
```

## publishBinaryMap

### binary-deferred-drain

**Draining deferred chunks first** — attached to `const size_t byte_budget = share_max_bytes_per_tick_ > 0` (line 2105)

```text
Deferred chunks drain FIRST, FIFO: every chunk carries absolute state
and the publisher preserves order, so an older record for a voxel hits
the wire before any newer one and snapshot-replace at the receiver
makes the newest win. Drained before the TF lookup — each deferred
message already pins its build-time pose, so a TF outage must not
stall the backlog. A tick that has sent nothing yet always sends one
message even over budget, so an oversized chunk cannot wedge the queue.
```

### binary-bundled-pose

**Bundling the map pose with each update** — attached to `geometry_msgs::msg::Transform map_from_source;` (line 2131)

```text
Snapshot the source->map pose from TF and carry it with this update so the
merger (dscovox) integrates against the bundled pose and never needs the
source->map transform from its own TF tree. Capture it BEFORE advancing
prev_sub_count_ or draining any touched-set: on a failed lookup we bail
without consuming state, so the snapshot re-trigger and the pending deltas
survive to the next tick — guaranteeing the first delta a merger ever sees
pins a valid pose. (map_frame_ <- int_frame_ is identity under the
integration_frame:"map" presets.)
```

### binary-tf-zero-timeout

**Zero-timeout pose lookup** — attached to `map_from_source = tf_buffer_.lookupTransform(` (line 2141)

```text
Zero timeout: under the SingleThreadedExecutor the TF listener callback
runs on this same thread, so blocking here can never let a new transform
arrive — the pose can only be found if it is already cached. A nonzero
timeout would just burn dead wait while holding map_mtx_. On a miss we
defer and retry next tick (see the catch below).
```

### binary-binarized-no-dominant

**Binarized baseline without a dominant class** — attached to `if (gate_binarize_ &&` (line 2306)

```text
Binarized baseline: a voxel with no dominant class (OTHER veto or
nothing observed) has nothing an argmax-only payload can say — no
record, and the gate entry keeps its previous emitted state. A
receiver that heard class k earlier keeps class k; the baseline has
no retraction, faithfully. (E6.6 measures exactly this blindness.)
```

### binary-heartbeat-arm

**The heartbeat re-emit arm** — attached to `if (!snapshot && share_heartbeat_sec_ > 0.0 && gate_beta_ && gate_dir_) {` (line 2339)

```text
Re-pin every emitted voxel at least once per period: heals a lost delta
(reliable QoS notwithstanding, the E6.4 relay drops whole messages) and
makes silence mean "unchanged within τ" rather than "unheard". Walks the
gate grids — exactly the ever-emitted set, never the whole map. The
touched-path emits above already stamped t_emit = t_now, so a voxel
never rides both paths in one tick. Snapshot ticks skip this: the
snapshot itself re-pins everything.
```

### binary-chunk-interleave

**Chunking and proportional interleave** — attached to `scovox::BinarySerializer::Frame part;` (line 2425)

```text
E6.7 chunking: split one tick's frame across several messages. Every
chunk is a complete, independently decodable ScovoxMapBinary (own
envelope, pose, LZ4 stream) and may mix sections — the receiver merge
is per-(source, coord) replace, so chunk boundaries change delivery
dynamics only, never the converged state. Costs: per-message envelope
overhead + smaller LZ4 windows.

The split is a PROPORTIONAL INTERLEAVE across the four sections, not
the section-at-a-time drain it used to be. Draining tsdf → beta → dir →
fine in full put every semantic record behind the ENTIRE occupancy
stream: on the connect-triggered full snapshot (~1 M Beta voxels) at
cap 500 a late-joining receiver saw ~2 000 pure-geometry messages
before its first class label, and semantics only completed when the
whole transfer did. Interleaving makes any prefix of the chunk
sequence carry each section in proportion to its size, so a receiver
has usable labels from message 0 and its semantic coverage grows with
the transfer instead of stepping in at the end.

Mechanism: give record `i` of a section of size `n` the position key
(2i+1)/(2n) on a normalized [0,1) axis and 4-way merge the sections by
that key. Keys within a section are already ascending, so this is a
linear merge (no sort), it preserves within-section order exactly, and
after m records each section has contributed within 1 record of its
proportional share m·n/N. Ties break in the historical section order,
so the split stays deterministic. Nothing here touches the wire
format, the envelope, or the serializer — only which records ride in
which message.
```

## SCovoxNode — declarations (part 5)

### planning-map-publish

**Publishing the planning map** — attached to `void publishPlanningMap() {` (line 2519)

```text
Publish planning_map as a 2D projection of the persistent voxel grid.

mode=persistent: fixed (plan_ox_, plan_oy_, plan_sz_) envelope.
mode=rolling:    robot-centered crop of side plan_window_size_m_, snapped
                 to grid resolution to avoid sub-cell jitter as the robot
                 moves. The underlying grid is unchanged — only the
                 publication is windowed.
```

### planning-project-body

**Shared planning map projection body** — attached to `void projectPlanningMap(rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>& pub,` (line 2567)

```text
Shared body: 2D projection of the persistent voxel grid over the axis-
aligned envelope [ox, ox+sz) x [oy, oy+sz), at `res` metres per cell, with
occupied cells dilated by `infl` metres. The z band, terrain-relative mode
and occupancy threshold are shared node state — only the envelope,
resolution and inflation differ between the two publishers.
```

## projectPlanningMap

### planning-terrain-columns

**Terrain-relative column classification** — attached to `struct Col { double x = 0.0, y = 0.0; std::vector<float> occ_z; };` (line 2589)

```text
Terrain-relative projection: per-column ground elevation, then a
band relative to it (see the param comment). One forEachCell pass
collects the occupied voxels per beta-grid XY column; the column map
is then classified without further grid access. Free voxels are not
consulted: an observed ground with a clear band IS the free
evidence (occupied still wins across beta columns sharing a plan
cell). coordToPos returns voxel CORNERS; the ground surface is the
top face (corner + one voxel) of the ground stack.
```

## SCovoxNode — declarations (part 6)

### pointcloud-publish

**The split-substrate point cloud** — attached to `void publishPointCloud() {` (line 2658)

```text
Caller must hold map_mtx_ (shared). The timer body locks once for both
publishScovoxMap and publishPointCloud so they see the same map state.
Split-substrate pointcloud publisher. Occupancy comes from the Beta
grid; semantics from the Dir grid at the same coord. The two are projected
into a SemBetaVoxel so the shared viz helpers (argmaxClassConfidence /
variance / expectedInformationGain) and the 16-field schema stay stable
for pointcloud_to_npz.py / RViz / eval scripts.
```

## publishPointCloud

### pointcloud-transient-overlay

**Transient dynamic-class overlay** — attached to `const auto& tbgrid = ss.transientBetaGrid();` (line 2700)

```text
Transient (dynamic-class) overlay. Dynamic hits land in a parallel
decaying grid; at publish we overlay them on the persistent cloud so they
appear while live and fade as they decay. A transient voxel with
occupancy evidence overrides its persistent counterpart at the same coord
(faithful to the legacy query-time picker). Empty when no dynamic classes
are configured, in which case the override check + extra walk are skipped.
```

### pointcloud-single-walk

**Single grid walk for the point cloud** — attached to `pc_scratch_.clear();` (line 2735)

```text
Single grid walk: collect the voxels that pass the publish gate, then size
and fill the message from the scratch list (was two full forEachCell walks
re-evaluating the same predicate). forEachCell order is deterministic, so
the emitted cloud is byte-identical. Persistent voxels overridden by a
transient voxel at the same coord are dropped; the transient grid walk
then re-emits them with their (decaying) dynamic evidence.
```

## SCovoxNode — declarations (part 7)

### tsdf-pointcloud-publish

**The TSDF zero-crossing cloud** — attached to `void publishTSDFPointCloud() {` (line 2800)

```text
Publish a thin shell at the TSDF zero-crossing. Caller must hold
map_mtx_ (shared). Walks TsdfMap for the surface geometry then runs
labelPointCloud against the Dir (semantics) grid to attach the per-point
semantic class. The cross-grid join uses the 0xFFFF sentinel where the Dir
grid has no voxel at the surface coord (same convention labelMesh /
extractZeroCrossing already produce). 5-field schema.
```

### fine-band-callbacks

**Fine band callbacks scope** — attached to `void onRefinementRegion(const scovox_msgs::msg::RefinementRegion& m) {` (line 2851)

```text
(docs/design/fine_tsdf_band_dbh_2026_07_30.md; all no-ops when
fine_ratio_log2 = 0 — none of these callbacks are created then.)
Measurement on the fine lattice (e.g. the DBH circle fit) is deliberately
NOT here: the node's responsibility ends at generating the map. Consumers
post-process the shared rev-7 fine stream or the saved map.
```

### member-memlog-thread

**Owned memlog worker thread** — attached to `std::thread mem_log_thread_;` (line 2963)

```text
The memlog worker is OWNED (not detached) so the destructor can join it:
a detached thread captures `this` and the grid/mutex/logger, and would
use-after-free if it were still walking the grid when rclcpp::spin()
returns and the node is torn down. mem_log_inflight_ enforces a single
worker at a time — if a grid walk outlasts 10 frames of integration we
skip launching a second one rather than letting two threads race on the
${tsdf_dump_path_}.tmp file (interleaved writes + a racing std::rename
corrupt the dump). The previous worker is joined before a new one starts.
```

### member-topk-probs-dir

**Top-K probability directory member** — attached to `std::string topk_probs_dir_;` (line 3008)

```text
Soft-probability mode: directory of <frame>.topk flat-binary blobs, with
file names matching the low 16 bits of header.stamp.nanosec the replay
node sets (zero-padded to 6 digits). Empty = legacy hard-label path.
topk_probs_dir_ survives only as the param sink + construction input for
topk_; the per-frame cache + loader telemetry now live in TopkProvider.
```

### member-gate-shadow-grids

**Change-gate shadow grids** — attached to `struct GateBeta { scovox::BetaVoxel v; double t_emit; };` (line 3085)

```text
Last-EMITTED wire state per voxel (the change gate's memory) plus its emit
time — node-clock seconds, double not float: wall-clock epoch seconds are
outside float's exact-integer range, and the heartbeat compares differences
of these. Shadow Bonxai grids with the live grids' geometry; allocated only
when share_change_gate is on in mode=rolling.
```

### member-last-input-stamp

**Stamping republished maps** — attached to `rclcpp::Time last_input_stamp_{0, 0, RCL_ROS_TIME};` (line 3110)

```text
Timestamp of the most recent integrated input (scan / depth). The full-map
republishers (publishPointCloud + the TSDF cloud) stamp with THIS, not
now(): the localizer (e.g. GLIM) has a valid integration_frame<-...<-map TF
at each scan time, but its TF can lag wall/playback time. Stamping the
republished map at now() makes RViz fail "Could not transform <int_frame> ->
map" (lookup into the future). Stamping at the last scan time keeps the map
transformable into map/any frame at any playback rate.
```

### member-pc-scratch

**Point cloud scratch buffer** — attached to `std::vector<std::tuple<Bonxai::CoordT, scovox::BetaVoxel, bool>> pc_scratch_;` (line 3153)

```text
Reused scratch for publishPointCloud's single grid walk (collect matching
voxels once, then fill the message from this) so the Beta grid is traversed
once per publish, not twice. clear() retains capacity across publishes.
The bool marks a transient (dynamic-class) voxel so the fill step reads its
semantics from the transient Dir grid instead of the persistent one.
```
