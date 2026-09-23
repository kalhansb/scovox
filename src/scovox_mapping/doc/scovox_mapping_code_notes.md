# scovox_mapping — code comment notes

Long comments from files in the `scovox_mapping` package, moved out of the code on 2026-09-23 so the sources carry short comments only. Where a comment was moved, the code keeps a short gist ending in `(notes: <id>)`; the section headed `<id>` below holds the original comment, word for word. Files with many moved comments have their own notes doc next to this one.

One section per source file, in file order. Each entry names the function (or section) the comment sat in, the line of code it was attached to and its original line number. Line numbers, dates, generation numbers and cross-references inside the moved text are as they were when written; they record history and are not maintained.

## Contents

- [config/default_params.yaml](#configdefault_paramsyaml) — 4
- [config/dscovox_params.yaml](#configdscovox_paramsyaml) — 1
- [include/scovox/dscovox_consensus.hpp](#includescovoxdscovox_consensushpp) — 5
- [include/scovox/node_utils.hpp](#includescovoxnode_utilshpp) — 7
- [include/scovox/scovoxmap.hpp](#includescovoxscovoxmaphpp) — 1
- [include/scovox/topk_provider.hpp](#includescovoxtopk_providerhpp) — 5
- [launch/dscovox_multi_robot.launch.py](#launchdscovox_multi_robotlaunchpy) — 3
- [launch/dscovox_single_robot.launch.py](#launchdscovox_single_robotlaunchpy) — 1
- [launch/scenenet_eval.launch.py](#launchscenenet_evallaunchpy) — 4
- [launch/scovox_single_robot.launch.py](#launchscovox_single_robotlaunchpy) — 8
- [launch/semantickitti_eval.launch.py](#launchsemantickitti_evallaunchpy) — 5
- [src/scovoxmap.cpp](#srcscovoxmapcpp) — 9
- [test/test_beta_update.cpp](#testtest_beta_updatecpp) — 2
- [test/test_consensus.cpp](#testtest_consensuscpp) — 9
- [test/test_marching_cubes.cpp](#testtest_marching_cubescpp) — 1
- [test/test_topk_provider.cpp](#testtest_topk_providercpp) — 1
- [test/test_tsdf_band.cpp](#testtest_tsdf_bandcpp) — 2

## config/default_params.yaml

### params-w-occ-derivation

**Deriving w_occ from sensor hit probability** — in `Map algorithm params [M]`, attached to `w_free:                        # float, default 1.0, pseudo-counts` (line 30)

```text
Evidence added to a_occ per hit observation.
Sensor-physics derivation: w_occ = (2·prob_hit − 1) / (1 − prob_hit).
  RGB-D conservative (OctoMap prob_hit=0.7):  w_occ = 1.33
    → noisy depth, many edge-spurious hits; low evidence per ray.
  RGB-D moderate (prob_hit≈0.75, current):    w_occ = 2.0
    → typical RealSense / Azure Kinect indoor.
  LiDAR (prob_hit≈0.9):                       w_occ = 8.0
    → physical TOF returns; very low FP rate, high evidence per ray.
```

### params-w-free-derivation

**Deriving w_free from sensor miss probability** — in `Map algorithm params [M]`, attached to `kappa0:                        # float, default 2.0, pseudo-counts` (line 40)

```text
Evidence added to a_free per free-space ray traversal.
Sensor-physics derivation: w_free = 1/prob_miss − 2.
  RGB-D conservative (OctoMap prob_miss=0.4):  w_free = 0.5
    → frequent missed returns (dark/specular surfaces).
  RGB-D moderate (prob_miss≈0.33, current):    w_free = 1.0
    → moderate confidence in clear free-space rays.
  LiDAR (prob_miss≈0.15):                      w_free ≈ 4.67
    → clear-air rays are highly trustworthy.
Ratio w_occ/w_free matters more than absolute scale for steady-state.
```

### params-dscovox-undeclared-keys

**Keys dscovox_mapping_node does not declare** — in `Node behaviour params [N]`, attached to `input_topics:                   # string[], default []` (line 250)

```text
NOTE: dscovox_mapping_node declares NO planning-map, pose-correction
(pose_source/correction_*) or consensus-tuning (epsilon_w/Lsat/
k_conflict/epsilon_sem/lambda_sem) parameters — the planning map is the
mapper's (SCovoxNode), and consensus constants are compile-time in
scovox_core. Any such keys in a params-file are silently ignored.
```

### params-core-params-struct

**The scovox Params struct summary** — in `scovox::Params struct (scovox_core library)`, attached to `(end of file)` (line 272)

```text
Key framework-agnostic parameters in the Params struct (authoritative
list and defaults: scovox_core/include/scovox/map_interface.hpp).
They control the Beta-conjugate occupancy model, Dirichlet semantics,
occupancy gate, evidence saturation, range/angle weighting, and
consensus fusion. All have sensible defaults. No ROS dependency.

 1. resolution              double  0.05   m
 2. w_free                  float   1.0    pseudo-counts
 3. w_occ                   float   2.0    pseudo-counts
 4. semantic_mode           enum    DIRICHLET
 5. kappa0                  float   2.0    pseudo-counts
 6. top_k                   int     2      classes
 7. semantic_occ_gate       float   0.5    dimensionless  (publish threshold)
 8. consensus_kl_threshold  float   5.0    nats
 9. consensus_tau_occ_gate  float   0.6    dimensionless
10. range_decay_length      float  -1.0    m  (disabled)
11. min_range               float   0.3    m
12. max_range               float  10.0    m
13. grazing_angle_threshold float  -1.0    dimensionless  (disabled)
```

## config/dscovox_params.yaml

### dscovox-params-file-origin

**Where dscovox merger params come from** — in `Top level`, attached to `/**:` (line 1)

```text
Live DSCovoxNode (merger) configuration. Until this file existed, dscovox
params were set only inline in scovox_multi_robot.launch.py /
scovox_single_robot.launch.py and experiment-script CLI overrides — those
still win when passed explicitly. Full per-param reference:
default_params.yaml (DSCovoxNode section).
```

## include/scovox/dscovox_consensus.hpp

### consensus-prior-slop

**Receiver at-prior slop matches sender gate** — in `File scope`, attached to `static constexpr float kPriorSlop = 1e-4f;` (line 30)

```text
Shared at-prior epsilon for the wire receiver isPrior* tests. This MUST
match the sender's at-prior emit gate (scovox_node.cpp uses + 1e-4f on the
alpha_free / alpha_other / beta priors). If the receiver slop is looser than
the sender gate (the old max(0.01, 0.01·α_0)), a barely-observed voxel that
the sender deliberately put on the wire — e.g. a_free = α_0 + 0.005, which
clears the sender's 1e-4 gate — is classified "at prior" here and silently
dropped from the fused map. Keeping the two epsilons identical guarantees
every emitted voxel survives the refold.
```

### consensus-viz-raw-evidence

**Raw-evidence convention in the viz projection** — in `projectBetaDirToSemBetaForViz`, attached to `const int residual_dims = static_cast<int>(num_classes) - scovox::K_TOP;` (line 85)

```text
RAW-evidence convention: subtract the OTHER bucket's (C-K)*alpha_0 prior
(clamped at 0 for C<=K_TOP, matching defaultDirVoxel) just as the RPC
projectBetaDirToVoxel does. argmaxClassConfidence / effectiveResidual /
semanticVariance all assume a_unk holds raw evicted mass with no prior;
leaving the prior in (out.a_unk = d->other) inflated the confidence
denominator by (C-K)*alpha_0 and made the published semantic_confidence
disagree with the GetRegion RPC for the identical voxel.
```

### consensus-rpc-projection-priors

**Prior handling in the RPC projection** — in `projectBetaDirToVoxel`, attached to `inline scovox::Voxel projectBetaDirToVoxel(` (line 112)

```text
scovox::Voxel stores RAW semantic evidence (the Dirichlet prior is applied
at query time, not in storage), whereas DirVoxel stores prior-inflated
counts. So the projection subtracts the per-class α_0 from each tracked
slot and the OTHER bucket's (C−K)·α_0 prior, yielding the same raw-evidence
convention selectTopKSemantics / argmaxClassConfidence expect — the SEMANTIC
fields are byte-identical to what the fused path produced. Empty Dir slots
(cnt == α_0) collapse to sem_cnt == 0 and are skipped by every consumer's
`sem_cnt > 0` test.

OCCUPANCY now uses the SAME prior as the fused Voxel: a_occ / a_free are copied
verbatim from the BetaVoxel, which ships the symmetric Beta(1,1) prior
(a_occ = a_free = 1.0 → prior p_occ = 0.5) — identical to the unified/fused Voxel
(defaultVoxel). So there is no longer a prior-induced p_occ / variance / EIG /
SSMI gap vs the fused Voxel at the prior. (Historically the split path used a calibrated
Beta(C·α_0, α_0) prior, p_occ = C/(C+1) ≈ 0.933; that was switched to
Beta(1,1) — see docs/occupancy_prior.md.)
```

### consensus-other-prior-clamp

**Clamping the OTHER prior at zero** — in `projectBetaDirToVoxel`, attached to `const int residual_dims = static_cast<int>(num_classes) - scovox::K_TOP;` (line 135)

```text
Clamp residual_dims at 0 to mirror defaultDirVoxel: when num_classes <=
K_TOP there are no residual classes, so the OTHER prior is 0 (defaultDir
stored other=0). An unclamped (C-K)*alpha_0 < 0 would make the subtraction
d->other - other_prior = d->other + |prior| ADD a phantom alpha_0 of
unknown mass, skewing every projected voxel's entropy/EIG/argmax.
```

### consensus-refold-beta

**The per-cell occupancy refold** — in `refoldBeta`, attached to `inline scovox::BetaVoxel refoldBeta(` (line 152)

```text
Pure core of the per-cell occupancy refold (BetaVoxel stream). Reset the
fused cell to the symmetric Beta(1,1) prior, then fold every NON-prior source
via mergeBeta (seed-copying the first non-prior source). `sources[i] ==
nullptr` means that source has no voxel at this cell; sources at prior are
skipped via isPriorBeta (a pure optimisation — folding the prior is a no-op).

This is the safeguard that makes the incremental refold bit-for-bit
equivalent to a from-scratch rebuild and immune to double-counting: the
result depends only on the current *set* of source values, never on how many
times a snapshot was received. The node calls this with one entry per source
grid accessor; tests call it with an explicit pointer list.
```

## include/scovox/node_utils.hpp

### topk-semantics-struct

**The TopKSemantics result** — in `TopKSemantics — declarations`, attached to `struct TopKSemantics {` (line 42)

```text
Result of selecting the strongest top-K semantic slots from a Voxel.
`kept` holds the chosen (class_id, count) pairs in descending count order;
only the first `kept_count` entries are valid. `dropped_mass` is the sum
of the counts that were *not* kept and must be folded into the consumer's
`a_unk` to preserve total semantic mass.
```

### topk-select-why-helper

**Why top-K selection is centralised** — in `selectTopKSemantics`, attached to `inline TopKSemantics selectTopKSemantics(const Voxel& v, int top_k) {` (line 56)

```text
`sparse_add` does not maintain slot order, so the K_TOP slots in a Voxel
are in arbitrary order with respect to count. A naive "first K non-zero
slots" loop can both promote a weak slot over a strong one (label bug) and,
if the dropped counts aren't folded into `a_unk`, leak total semantic mass
(mass-conservation bug). This helper exists so every consumer that wants
fewer than K_TOP classes uses the same correct selection rule.
```

### topk-insertion-sort

**Insertion sort instead of std::sort** — in `selectTopKSemantics`, attached to `for (size_t i = 1; i < n; ++i) {` (line 70)

```text
Hand-rolled insertion sort (n ≤ K_TOP) instead of std::sort: at -O3 the
inlined std::sort pulls in its 16-element introsort fallback, whose dead
`__first + 16` access trips a -Warray-bounds false positive when this is
inlined into the dscovox refold path. Descending by count; identical output
for this n.
```

### topk-negative-top-k

**Negative top_k means zero slots** — in `selectTopKSemantics`, attached to `const size_t cap = std::min<size_t>(static_cast<size_t>(K_TOP),` (line 83)

```text
Clamp the request to [0, K_TOP]. A negative `top_k` is treated as 0 (not 1):
a caller asking for "no semantic slots" (occupancy-only emit) must get
kept_count == 0 and have the *entire* semantic mass folded into
`dropped_mass`, rather than silently keeping one class and under-folding by
one slot. The previous std::max(1, top_k) floor violated that intent and
leaked the strongest slot's mass past the dropped_mass accumulation.
```

### argmax-confidence-consistency

**Confidence matches entropy and variance** — in `File scope`, attached to `template <typename V>` (line 105)

```text
Matches the categorical used by `semanticEntropy` / `semanticVariance`
in scovox_core, so all three uncertainty signals (entropy, variance,
and the published confidence) stay mutually consistent. Replaces the
pre-2026-05-03 inline `cf = best_cnt / Σ tracked` rule, which dropped
`a_unk` and the Dirichlet prior from the denominator and was therefore
systematically over-confident whenever K_TOP eviction had occurred.
```

### argmax-confidence-edge-cases

**Edge cases of argmaxClassConfidence** — in `File scope`, attached to `template <typename V>` (line 112)

```text
Edge cases:
  - n_active == 0 (Beta evidence but no semantic observations): returns
    (0, 0.f). Callers gating on `p_best >= threshold` will not colour /
    emit a label — correct behaviour.
  - Tied counts: first-index-wins from the inner strict-`>` test, which
    matches `sparse_add`'s eviction rule (also strict-`>`). The argmax
    is therefore *not unique* under exact ties; this propagates the
    same first-arrival bias flagged by C4 in the ablations punch list,
    and the paper should not claim uniqueness without that caveat.
Template form (D6): shared body for Voxel and SemBetaVoxel. Both
expose `sem_cnt[K_TOP]`, `sem_cls[K_TOP]`, and the
`effectiveResidual(v)` Hutter helper accepts both via its own
template overload (uncertainty.hpp). Existing call sites pass a
Voxel and the deduction picks the same body that used to be the
non-template overload — no behavioural change for the legacy path.
```

### argmax-residual-bound

**Why no second residual cap** — in `argmaxClassConfidence`, attached to `const float denom = sum_cnt + static_cast<float>(n_active)` (line 144)

```text
The residual term here is `effectiveResidual(v)`, which is now bounded above
by N = Σ sem_cnt + a_unk (the Hutter escape mass is clamped to N in
uncertainty.hpp). That bound is what keeps this confidence denominator sane
for weakly-observed voxels: e.g. sem_cnt={0.05}, a_unk=0 gives residual 0.05
(not the unbounded ~10 the raw m/(2 ln ratio) term used to produce), so
p_best ≈ 1.05/2.1 ≈ 0.5 rather than collapsing to ~0.087 below the labelling
threshold. We deliberately do NOT add a second, tighter cap (e.g. clamp the
residual to sum_cnt) here: this categorical must stay identical to the one
used by semanticEntropy / semanticVariance so all three uncertainty signals
remain mutually consistent, and the bound that fixes the collapse lives in
the shared effectiveResidual helper.
```

## include/scovox/scovoxmap.hpp

### map-fused-integrate-ray

**The fused single-DDA ray walk** — in `Map — declarations`, attached to `void fused_integrate_ray_static(const Eigen::Vector3f& origin,` (line 146)

```text
Single fused DDA walk for non-dynamic rays:
  origin → posToCoord(hit + sdf_trunc * u)
per-voxel fuses Beta free / Beta occupied + semantics / TSDF in band.
Replaces the legacy two-pass `carve_free` + `update_endpoint`, paying
one DDA per ray. `updated_coords` may be nullptr.
```

## include/scovox/topk_provider.hpp

### topk-provider-extraction

**Why TopkProvider is standalone** — in `File scope`, attached to `#include <algorithm>` (line 6)

```text
Extracted verbatim from SCovoxNode (Tier 2 refactor): the node used to carry
the per-frame cache, the binary parser, the fill helpers, and the loader
telemetry inline. Pulling them into a standalone unit makes the binary parser
unit-testable without spinning up ROS (the only ROS dependency is the
injected logger/clock used for the same throttled diagnostics as before).
```

### topk-header-guard

**Guarding the .topk header read** — in `loadFrame`, attached to `static constexpr size_t kMaxTopkBytes = size_t(1) << 30;  // 1 GiB` (line 97)

```text
Guard the header reads before sizing the buffer. An empty or truncated
.topk file leaves the dimension fields at their zero-init (failed reads
no-op), and a garbage `total` from a partially-read header would throw
bad_alloc out of resize() — uncaught in the integration callback, that
calls std::terminate and kills the mapper. Require a clean header read
and a payload within a sane absolute ceiling.
```

### topk-short-read

**Detecting a short .topk payload** — in `loadFrame`, attached to `if (static_cast<size_t>(f.gcount()) != total) {` (line 114)

```text
Detect a truncated payload by the bytes actually delivered, NOT by the
stream flags: a short read sets BOTH failbit and eofbit, so the old
`!f.good() && !f.eof()` test was (true && false) == false and silently
accepted the partial buffer — the zero-filled tail (from resize) would
then be fed to the Dirichlet update as legitimate zero-probability
classes, corrupting semantics for the whole frame. gcount() is the only
reliable short-read signal here.
```

### topk-loader-summary

**The soft-prob loader summary log** — in `TopkProvider — declarations`, attached to `void logSummary(int throttle_ms) {` (line 145)

```text
Emit a running tally of soft-prob loader outcomes. Guards the
silent-fallback footgun: when the probs dir is set but loads intermittently
fail, the per-frame WARN is throttled at 5 s and easy to miss in long batch
runs; this INFO is throttled at the caller's rate and shows running totals
so the operator (or a smoke-gate assert) can verify soft-prob dispatched on
every expected frame. No-op when topk is disabled.
```

### topk-truncate-no-renorm

**Truncation does not renormalise** — in `TopkProvider — declarations`, attached to `void truncate(std::vector<float>& cp) const {` (line 214)

```text
Zero every probability outside the `topk_trunc_` largest. The dropped mass
is deliberately NOT renormalized onto the survivors: dirichletUpdate turns
whatever is missing from 1.0 into OTHER, so a truncated observation reads
as "this class, and I decline to guess about the rest" rather than as a
more confident version of the same distribution.
```

## launch/dscovox_multi_robot.launch.py

### launch-multi-max-range

**Why max_range defaults to 20 m** — in `Per-robot deployment table`, attached to `_ARGS = {` (line 81)

```text
max_range is 20 m, matching scovox_lidar_geometric.yaml (the single-robot
launch still defaults to 40). carve_band is -1.0 = FULL-RAY free-space carve,
so carve cost scales linearly with max_range: at 0.10 m resolution a 40 m ray
traverses ~400 voxels vs ~200 at 20 m. Two mappers run concurrently on one
host here, and a 128-beam scan is already near the real-time budget (see the
downsample_voxel_size note below) — so raising max_range to widen the
footprint costs carve time on EVERY mapper. Keep downsample_voxel_size on
when you do, or bound the carve with a positive carve_band instead.
```

### launch-multi-downsample

**Downsample must be set in the launch** — in `launch_setup`, attached to `"downsample_voxel_size": 0.1,` (line 137)

```text
Per-scan sensor-frame voxel downsample BEFORE integration:
near-lossless for occupancy but keeps the full-ray carve
real-time. The node's own default is 0.0 (OFF) and this launch
loads no params file, so it MUST be set here — without it a
128-beam scan (~115k points) misses real time by ~700 ms/frame.
0.1 fleet-wide, matching every shipped config and
dscovox_single_robot.launch.py.
```

### launch-multi-tf-gates

**TF stability gates left at defaults** — in `launch_setup`, attached to `}],` (line 153)

```text
TF STABILITY GATES: left at the NODE DEFAULTS on purpose
(startup_tf_stable_sec 2.0, startup_tf_jump_threshold 0.5,
runtime_tf_gate true, runtime_tf_jump_threshold 1.0).
scovox_lidar_geometric.yaml disables all three ("NDT owns
map->odom, no SLAM jump to guard against") — do NOT copy that
here. Both robots' NDT self-localizes from a near-origin seed
(docs/dscovox_multi_robot_run.md), so the first convergence is
a LARGE pose jump that the startup gate should absorb rather
than integrate; and 1.0 m frame-to-frame sits well above real
ground-robot motion (~0.1-0.15 m at 10 Hz), so the runtime
gate only trips on genuine NDT divergence. Cost of leaving
them on: ~2 s of scans gated per robot at startup, and a 2 s
re-settle after any real divergence — both visible in the
mapper console ("Waiting for TF stabilization").
```

## launch/dscovox_single_robot.launch.py

### dscovox-single-launch-purpose

**What the single-robot dscovox launch is for** — in `dscovox — SINGLE-ROBOT dscovox map bring-up (rolling mapper + merger).`, attached to `from launch import LaunchDescription` (line 4)

```text
One rolling-mode mapper feeds one merger, which reconstructs the fused
`dscovox` map on `/<robot>/dscovox_node/pointcloud`. Correctly namespaced so
the delta topic is exactly `/<robot>/scovox_node/scovox_bin` (what the merger
subscribes to). Doubles as the KNOWN-GOOD reference when a mapper is NOT
publishing scovox_bin: if the topic flows HERE but not in your setup, the
difference is your config (almost always
mode != rolling, or a namespace/node-name mismatch). See
docs/scovox_bin_manual_bringup.md for the full step-by-step diagnosis.
```

## launch/scenenet_eval.launch.py

### scenenet-publish-occ-gate

**Publish-time occupancy gate plumbed through** — in `_launch_setup`, attached to `occ_vis_arg = float(context.launch_configurations.get("occupancy_vis_threshold", "0.5"))` (line 26)

```text
Publish-time occupancy gate — used by Phase 2.5 to vary the labelling
envelope. Was hardcoded 0.5 pre-2026-05-14; now plumbed through
context so phase2_5_gate_threshold_sweep.sh can integrate at 0.0
and see the unfiltered grid.
```

### scenenet-a9-sensor-weights

**A9 inverse-sensor-model weights** — in `_launch_setup`, attached to `w_occ_arg = float(context.launch_configurations.get("w_occ", "6.0"))` (line 34)

```text
A9 inverse-sensor-model weights. Were hardcoded 6.0/1.0 below; plumbed
through context (as semantickitti_eval.launch.py already does) so the A9
sweep can vary them per cell. Defaults reproduce the previous constants
exactly, so every run that does not pass them is bit-identical to before.
```

### scenenet-e1-capture-knobs

**E1 uncertainty capture knobs** — in `_launch_setup`, attached to `map_mode_arg = context.launch_configurations.get("map_mode", "persistent")` (line 44)

```text
E1 uncertainty capture (mirrors semantickitti_eval.launch.py): rolling mode
enables the ScovoxMapBinary publisher (bin_pub_ exists only when
mode==rolling); share_rate_hz>0 gives a timer-owned binary publish so a
snapshot fires when a capture subscriber connects AFTER replay (with no
subscriber the timer is a cheap no-op, so replay recv stays 300/300).
scovox_publish_rate is overridable so the E1 runner can slow the pointcloud
republish timer down for offline capture. Defaults preserve the paper runs.
```

### scenenet-perf-attribution-knobs

**Perf-attribution knobs for frame time** — in `_launch_setup`, attached to `"carve_band": float(` (line 164)

```text
Perf-attribution knobs (SLIM-VDB frame-time comparison). Every
default below is the value the node already used, so an unset
launch is byte-identical; they exist so the cost of the full-ray
carve, the TSDF band and the carve staging can be measured
SEPARATELY instead of inferred from a single frame_ms.
```

## launch/scovox_single_robot.launch.py

### single-launch-semantics-params

**Semantics parameter table** — in `generate_launch_description`, attached to `"kappa0":                   2.0,` (line 42)

```text
kappa0                   double  2.0   -     Dirichlet base pseudo-count per hit
semantic_occ_gate        double  0.5   -     Hard p_occ threshold for semantic updates
(top-K width per voxel is the compile-time K_TOP in scovox_core
 voxel.hpp — not a parameter)
```

### single-launch-range-params

**Range weighting parameter table** — in `generate_launch_description`, attached to `"range_decay_length": -1.0,` (line 61)

```text
range_decay_length <= 0 disables range weighting.
range_decay_length   double  -1.0   m    Distance at which w decays to ~37% (<=0 disables)
min_range            double  0.3    m    Discard returns closer than this
max_range            double  10.0   m    Discard returns farther than this
```

### single-launch-transient-params

**Transient layer parameter table** — in `generate_launch_description`, attached to `"max_semantic_classes": 10,` (line 75)

```text
max_semantic_classes is the total label space; the per-voxel
top-K width is the compile-time K_TOP (scovox_core voxel.hpp).
max_semantic_classes  int            10   -    Total class count (label space)
transient_decay_rate  double         0.8  -    Per-frame decay for dynamic voxels
dynamic_classes       int[]          []   -    Class ids routed to the transient
                                                decaying grid (argmax match); empty
                                                = feature off. Uncomment to enable,
                                                e.g. "dynamic_classes": [11, 12].
                                                Leave unset (not []) to keep it off:
                                                an empty list breaks ROS 2 param
                                                type inference.
```

### single-launch-frames-topics-params

**Input frames and topics parameter table** — in `generate_launch_description`, attached to `"base_frame":          ["", robot_name, "/base_link"],` (line 90)

```text
base_frame          string  "base_link"  -   Robot base frame
integration_frame   string  "odom"       -   Frame for map integration
map_frame           string  "map"        -   Global map frame
depth_topic         string  ...          -   Depth image topic
depth_info_topic    string  ...          -   CameraInfo topic for depth
seg_topic           string  ...          -   Semantic segmentation topic
stride              int     1            px  Pixel stride when subsampling depth
min_depth           double  0.1          m   Depth clip minimum
max_depth           double  10.0         m   Depth clip maximum
trace_no_return_rays bool   false        -   Carve free space for no-return pixels
```

### single-launch-mode-params

**Mode and identity parameter table** — in `generate_launch_description`, attached to `"mode":                "rolling",` (line 112)

```text
mode      string  "rolling"  -   "rolling" (publishes ScovoxMapBinary
                                  snapshots, rolling planning_map crop)
                                  | "persistent" (no binary)
robot_id  string  ""         -   Robot identifier (informational)
```

### single-launch-output-params

**Output and visualisation parameter table** — in `generate_launch_description`, attached to `"publish_pointcloud":       True,` (line 120)

```text
The 8 planning_map_* params could be moved to a dedicated
planning-map server node, leaving just publish_pointcloud,
occupancy_vis_threshold, and scovox_publish_rate here.
publish_pointcloud        bool    true    -    Publish coloured occupancy pointcloud
pointcloud_topic          string  ~/pc    -    Topic for pointcloud output
scovox_topic              string  ~/sec   -    Topic for full ScovoxMap (a_occ/a_free)
occupancy_vis_threshold        double  0.7     -    Minimum P(occ) to include in outputs
scovox_publish_rate       double  1.0     Hz   Rate of full ScovoxMap publication
publish_planning_map      bool    true    -    Publish 2-D OccupancyGrid for planners
planning_map_topic        string  ~/pm    -    Topic for planning OccupancyGrid
planning_map_resolution   double  0.20    m    Grid cell size
planning_map_size_m       double  80.0    m    Square grid side length
planning_map_origin_x     double  -40.0   m    Grid origin X (world frame)
planning_map_origin_y     double  -40.0   m    Grid origin Y (world frame)
planning_map_min_z        double  -1.0    m    Min voxel Z projected into grid
planning_map_max_z        double   2.0    m    Max voxel Z projected into grid
planning_map_inflation_m  double  0.0     m    Obstacle inflation radius
```

### single-launch-dscovox-declared-params

**Only declared dscovox parameters are set** — in `generate_launch_description`, attached to `dscovox_node = Node(` (line 155)

```text
Only parameters dscovox_mapping_node actually declares are set here. The
planning-map, pose-correction (pose_source/correction_*) and
consensus-tuning (epsilon_w/Lsat/k_conflict/epsilon_sem/lambda_sem) keys
formerly listed were never declared by the node and were silently
ignored; consensus constants live in scovox_core, and the per-voxel
top-K width is the compile-time K_TOP (scovox_core voxel.hpp).
```

### single-launch-dscovox-output-params

**dscovox output parameter table** — in `generate_launch_description`, attached to `"pointcloud_topic":         "/dscovox_mapping/pointcloud",` (line 174)

```text
pointcloud_topic   string  ~/pointcloud  -    Fused pointcloud output
map_frame          string  "map"         -    Frame for fused map
occupancy_vis_threshold double  0.7      -    Min P(occ) for outputs
semantic_occ_gate  double  0.5           -    p_occ gate for semantic merge
publish_rate_hz    double  1.0           Hz   Fused map publish rate (= code default)
```

## launch/semantickitti_eval.launch.py

### kitti-e1-capture-knobs

**E1 uncertainty capture knobs (KITTI)** — in `_launch_setup`, attached to `map_mode_arg = context.launch_configurations.get("map_mode", "persistent")` (line 57)

```text
E1 uncertainty capture: rolling mode enables the ScovoxMapBinary publisher
(bin_pub_ is created only when mode==rolling); share_rate_hz>0 gives a
timer-owned binary publish so a snapshot fires when a capture subscriber
connects AFTER replay (with no subscriber the timer is a cheap no-op, so
replay recv stays 100/100). Defaults preserve the paper persistent runs.
```

### kitti-downsample-pinned-off

**Per-scan downsample pinned off** — in `_launch_setup`, attached to `"downsample_voxel_size": 0.0,` (line 85)

```text
Per-scan medoid downsample: the node default flipped 0.0 -> 0.5
upstream (for live raw-LiDAR configs). When > 0 the scan is
thinned AND integrated geometry-only — the semantic_label field
and the topk soft-prob table are both dropped — so the eval pins
it off: full per-point path, the config every capture ran.
```

### kitti-min-range-history

**KITTI min_range raised to 5 m** — in `_launch_setup`, attached to `"range_decay_length": range_decay_length_arg,` (line 131)

```text
Pre-2026-05-11 value was min=1.0; flipped to 5.0 to eliminate
the only meaningful KITTI head-to-head config asymmetry.
```

### kitti-runtime-tf-gate-off

**Runtime TF-divergence gate disabled** — in `_launch_setup`, attached to `"runtime_tf_gate": False,` (line 159)

```text
The scovox node adds a runtime TF-divergence gate (default
runtime_tf_jump_threshold=1.0 m) that the paper node lacked. KITTI's
~1.2 m/frame motion exceeds it, so after frame 2 stabilizes every
subsequent frame is flagged as "localization diverged" and gated
forever -> empty map. The replay feeds exact GT poses (no real
divergence to guard against), so disable the runtime gate and raise
the threshold well above the per-frame vehicle step.
```

### kitti-slow-publish-timer

**Slow publish timer for offline eval** — in `_launch_setup`, attached to `"scovox_publish_rate": 0.2,` (line 181)

```text
The publish timer holds a shared map lock and walks the whole map
every tick; integration needs the unique lock, so a fast timer on a
growing 600k-voxel map starves the queue-depth-1 cloud subscription
and most replay frames are dropped (recv ~22/100 at 1 Hz). Slow the
timer right down for offline eval — we only need one publish at the
end for the npz capture. Also drop the TSDF cloud republisher (unused
by the eval, another full-grid walk per tick).
```

## src/scovoxmap.cpp

### saturation-shared-beta-factor

**Single shared factor for Beta saturation** — in `Map::apply_evidence_saturation`, attached to `const float max_beta = std::max(v->a_occ, v->a_free);` (line 33)

```text
Beta proportional saturation: scale (a_occ, a_free) by a SINGLE factor so
the larger bucket lands at `cap`, preserving the ratio (and therefore
p_occ). Must be one shared factor: chaining two independent
scale-and-floor blocks (one per bucket) would double-scale when both
buckets exceed cap (neither bucket lands at cap, ratio drifts) and lets
the per-block 1.0 floor distort p_occ on lopsided voxels. We compute the
factor from max(a_occ, a_free) and apply it to both at once.
```

### semantics-mode-occupancy-gate

**Occupancy gating per semantic mode** — in `Map::apply_semantics`, attached to `const float p_occ = v->p_occ();` (line 147)

```text
Bayesian-soft for DIRICHLET: weight by p_occ directly (no hard gate).
NAIVE and MAJORITY_VOTE are ablation baselines and use a hard `p_occ > 0.5`
cutoff (their accumulators don't take a continuous weight) so that the
only ablation variable across the three modes remains the per-observation
accumulation rule, not whether updates fire at all.
```

### fused-carve-weight-caller-range

**Carve weight comes from the caller** — in `Map::fused_integrate_ray_static`, attached to `const float carve_w = range_w;` (line 250)

```text
Beta free-update weight matches the legacy carve_free behavior exactly:
it uses the node-supplied range_w (computed from the FULL sensor→hit
distance), not a fresh exp(-segment_depth/decay) over the carve segment.
For the partial-ray (carve_band > 0) case `origin` is a truncated
origin and depth ≈ carve_band, so an in-function recomputation would
diverge from the unified model by orders of magnitude. Use the caller's value.
```

### fused-band-only-dda-start

**Band-only integration DDA start** — in `Map::fused_integrate_ray_static`, attached to `const CoordT k0 = (params_.band_only_integration && trunc > 0.f)` (line 262)

```text
Band-only mode (benchmarking / TSDF-only): start the DDA at the near
edge of the truncation band rather than the sensor origin. Skips the
long Beta-free carve from origin to (hit - trunc) and matches SLIM-VDB's
[depth-trunc, depth+trunc] DDA. Falls back to full-ray when trunc=0
(TSDF disabled) since there is no band to confine the walk to.
```

### fused-per-voxel-independence

**Per-voxel independence, not reach_prob** — in `Map::fused_integrate_ray_static`, attached to `bool past_wall = false;` (line 273)

```text
Per-voxel independence assumption (OctoMap / log_odds-node style). Joint
ray-cast `reach_prob` was tried (commit 513c969) and reverted: cost
~6 mIoU points on Replica m2f from cold-start damping (every voxel
starts at p_occ=0.5, so reach_prob ≈ 0.5^N along uninitialised rays).
KITTI was bit-flat either way. Through-wall carving is gated by
`carve_skip_occ_threshold` instead — cheap, well-tested, sufficient.
```

### fused-dda-skipped-hit-voxel

**Visiting a hit voxel the DDA skipped** — in `Map::fused_integrate_ray_static`, attached to `bool k_hit_visited = false;` (line 282)

```text
Bresenham DDA can skip k_hit when it's a corner-crossing voxel on the
line (it picks one voxel per dominant-axis step, so an oblique ray's
true hit voxel may not lie on the picked path even though it's on the
line). Track whether k_hit was visited and explicitly visit it after
the loop if not — guarantees the endpoint Beta-occupied + semantics +
surface TSDF mass always lands.
```

### consensus-beta-merge-rule

**Beta consensus merge rule** — in `Map::consensusMerge`, attached to `dst.a_occ  = dst.a_occ  + src.a_occ  - 1.f;` (line 448)

```text
Beta-conjugate posterior under conditional independence given θ:
  Beta(α₁, β₁) ⊕ Beta(α₂, β₂) = Beta(α₁ + α₂ − 1, β₁ + β₂ − 1)
with the shared Beta(1, 1) prior subtracted once.

Conditional independence is upheld by the dscovox publish topology:
each ScovoxMapBinary is sent from a robot's *local* scovox map (its
own sensor observations), never from the merged dscovox map, so a
robot's own evidence cannot be echoed back into its source grid via
the fused output. The one residual correlation source — shared
classifier error if multiple robots run the same segmenter on similar
RGB — is a measurement-model concern on the *semantic* path; the
Bayesian fix is a per-source evidence discount on the Dirichlet
counts, not a change of merge rule, and is unaddressed here.

No `max(1, ·)` floor: every Voxel in this codebase is constructed with
a_occ, a_free ≥ 1 (defaultVoxel() and the integration paths preserve
this), so α₁ + α₂ − 1 ≥ 1 algebraically. The previous floor was a
non-Bayesian guard against malformed inputs that the type system never
produces; removed 2026-05-03.
```

### consensus-dirichlet-merge-ungated

**Dirichlet merge has no occupancy gate** — in `Map::consensusMerge`, attached to `for (int i = 0; i < K_TOP; ++i) {` (line 470)

```text
Dirichlet semantic merge: always combine src's evidence regardless of
post-merge occupancy. Under conditional independence given the latent
(occ, class) the additive Dirichlet has no occupancy condition. The
"don't put semantic colour on free voxels" intent of the legacy gate is
a *display* concern — downstream consumers filter on p_occ at
query/visualization time (e.g. dscovox_node's pointcloud publish gates
colour by `cf >= sem_gate_`).
```

### consensus-no-kl-conflict

**No KL conflict check in the merge** — in `Map::consensusMerge`, attached to `}` (line 484)

```text
No betaKL conflict computation. The previous code computed and returned
a `conflict` bool but every caller discarded it; the threshold knob
(`consensus_kl_threshold`) was log-only and never gated fusion.
betaKL() itself remains in scovox/uncertainty.hpp for callers that want
an explicit disagreement metric outside the merge path.
```

## test/test_beta_update.cpp

### test-beta-order-commutativity-note

**Removed ray-order commutativity test** — in `BetaUpdate.MultipleHitsAreAdditive`, attached to `TEST(BetaUpdate, MultipleHitsAreAdditive) {` (line 183)

```text
NOTE: under the joint ray-casting likelihood, ray-order commutativity
no longer holds — each ray's update is conditional on the current state
of upstream voxels. The earlier `OrderOfHitsDoesNotMatter` test was
removed when reach_prob was introduced.
```

### test-beam-through-wall-stays-solid

**A beam through a wall leaves it solid** — in `BetaUpdate.CarvingAttenuatedPastOccupied`, attached to `TEST(BetaUpdate, CarvingAttenuatedPastOccupied) {` (line 210)

```text
The wall guard is OFF by default (we trust the most recent scan). A beam that
passes through the wall to a farther return therefore deposits ONE free
increment (≤ w_free) on the wall voxel — but the wall stays confidently
occupied because its accumulated a_occ (100 rays) dwarfs a single free update.
This is the intended dynamic-clearing behaviour: repeated pass-through
eventually clears a stale obstacle, one scan does not.
```

## test/test_consensus.cpp

### test-betakl-utility-banner

**Beta-KL utility kept outside the merge** — in `Consensus.BetaKLSymmetryProperty`, attached to `TEST(Consensus, BetaKLSymmetryProperty) {` (line 114)

```text
Beta-KL utility (the function itself is preserved in scovox_core for
callers that want an explicit disagreement metric; consensusMerge no
longer consults it — return value used to be a discarded `conflict`
bool, removed 2026-05-03)
```

### test-semantic-merge-ignores-occupancy

**Semantics merge regardless of occupancy** — in `Consensus.SemanticMergedRegardlessOfOccupancy`, attached to `TEST(Consensus, SemanticMergedRegardlessOfOccupancy) {` (line 158)

```text
2026-05-03: gate removed from consensusMerge — Dirichlet evidence is
additive under conditional independence given the latent (occ, class).
This test now asserts the *new* invariant: semantics merge regardless of
post-merge occupancy. Even when the merged Beta says "free", the source's
semantic counts must still be folded into dst.
```

### test-split-helpers-real-code

**Split tests call the node's real helpers** — in `Semantic additive Dirichlet merge`, attached to `namespace {` (line 193)

```text
Findings 18/19/20 flagged that the receiver path in dscovox_node.cpp had
ZERO symbol-level coverage. Its helpers — projectBetaDirToVoxel /
isPriorBeta / isPriorDir + the refold core (refoldBeta / refoldDir) — were
extracted from dscovox_node.cpp's anonymous namespace into
scovox/dscovox_consensus.hpp, so the tests below now call the REAL functions
the node runs (refoldCellBeta/refoldCellDir are thin Bonxai-accessor wrappers
over refoldBeta/refoldDir). Coverage includes the Dir==null occupancy-only
branch and the num_classes <= K_TOP edge (finding 19).
```

### test-refold-reset-then-fold

**Refold resets to prior, then folds** — in `SplitRefold.BetaFoldIntoPriorReproducesSource`, attached to `TEST(SplitRefold, BetaFoldIntoPriorReproducesSource) {` (line 214)

```text
Finding 20: the refold safeguard rests on "reset fused[c] to prior, then fold
each source's CURRENT value once". A single-source refold therefore reproduces
the source exactly (the reset-to-prior is seed-copied over). These call the
REAL scovox::refoldBeta / refoldDir cores the node runs (refoldCellBeta/Dir are
thin Bonxai-accessor wrappers over them).
```

### test-refold-duplicate-snapshot

**Duplicate snapshots cannot drift** — in `SplitRefold.DuplicateSnapshotIsIdempotent`, attached to `TEST(SplitRefold, DuplicateSnapshotIsIdempotent) {` (line 237)

```text
Finding 20 (idempotency, end-to-end): a source re-publishes the SAME snapshot
twice. A re-sent snapshot overwrites that source's grid in place — it does NOT
append a second source — so both receipts refold the same current set {A}, and
because refoldBeta/refoldDir reset-to-prior before folding, the fused state is
a pure function of {A} and cannot drift. Pinning Beta and Dir together because
the two grids refold separately.
```

### test-refold-order-summary

**Multi-source refold order guarantees** — in `Semantic additive Dirichlet merge`, attached to `namespace {` (line 272)

```text
The idempotence test above is SINGLE-source. The three below cover what the
merger actually runs: several sources refolded together, in an order the node
chooses rather than the network. Read together they say:

  Beta  (addition only)          — order-free up to float rounding
  Dir   at 2 sources             — order-free EXACTLY  (the shipped rig)
  Dir   at ≥3 sources            — NOT order-free; the node pins the order
  duplicate receipt, any N       — a no-op, because refold resets first
```

### test-dir-refold-order-limitation

**Dir refold order-dependent at three sources** — in `SplitRefold.DirRefoldDependsOnSourceOrderAtThreeSources`, attached to `TEST(SplitRefold, DirRefoldDependsOnSourceOrderAtThreeSources) {` (line 334)

```text
⚠ At three or more sources the refold is NOT order-independent, and this test
PINS THAT LIMITATION rather than asserting it away. mergeDir truncates at each
pairwise step and a class dumped to OTHER can never climb back, so the fused
slots depend on which sources met first (consensus_merge.hpp says so in situ).

Fixture: classes 7 and 2 compete for the top slot. Folding A+B first keeps
{2, 7}, so C's extra 1.5 lands ON class 7 and carries it past class 2 → 7.
Folding A+C first evicts class 7 (its 1.5 loses to class 4's 2.5 and class 0's
1.9), and evicted evidence cannot come back, so class 2 keeps the slot → 2.
Both are *confident* labels, not abstentions: this is a real label flip, not a
degradation to OTHER. A 500k-draw random search over 3-source configurations
put these at ~1.0% of draws (with a further ~13.5% flipping class↔abstain and
20.6% differing in the fused slots), so the fixture is representative, not a
hand-built pathology.

The node's mitigation is to fix the order, NOT to make the merge commutative:
dscovox_node.cpp sorts sources by id before folding Dir, so the fused result
is reproducible across runs and rehashes even though it is not permutation-
invariant. The last two assertions are that actual guarantee.

Scope: every E6 cell ran N=2, where the fold is exactly order-free
(DirRefoldIsOrderFreeAtTwoSources above), so no campaign result depends on
this. E6.5's N∈{3,4} scaling arm was dropped 2026-08-06.
```

### test-beta-refold-float-order

**Beta refold order and float rounding** — in `SplitRefold.BetaRefoldOrderInvariantToFloatTolerance`, attached to `TEST(SplitRefold, BetaRefoldOrderInvariantToFloatTolerance) {` (line 381)

```text
Beta carries no truncation, so it is order-free semantically — but float
addition is not associative, so it is NOT bit-exact across fold orders. The
node folds Beta in unordered_map order (unsorted, unlike Dir), which is safe
precisely because the spread is rounding-scale: measured at ≤2 ULP
(relative ~1.5e-7) over 200k random 4-source draws in all 24 orders.
EXPECT_FLOAT_EQ's 4-ULP tolerance is the right assertion here; EXPECT_EQ on
the bits would be wrong and would flake.
```

### test-split-projection-raw-evidence

**Split projection matches fused raw evidence** — in `SplitProjection.RawEvidenceMatchesFused`, attached to `TEST(SplitProjection, RawEvidenceMatchesFused) {` (line 431)

```text
Finding 18: the split RPC projection must hand the planner the SAME raw
semantic evidence the unified fused voxel would carry for the identical
observation history. Build a unified voxel via the sparse_add path and a
split Dir voxel via the sparse_add_class path for the same two observations,
project the split voxel, and assert the raw evidence matches slot-for-slot
(and that a_unk is the OTHER bucket's observed mass, prior subtracted).
```

## test/test_marching_cubes.cpp

### test-mc-mid-voxel-surface

**Why the test plane sits mid-voxel** — in `TEST`, attached to `auto grid = makePlanarGrid(0.10, 5, 4.5f, 10.0f);` (line 109)

```text
Surface at z=4.5 — mid-voxel, between z=4 and z=5, both interior to the
half_extent=5 grid. Two reasons to avoid integer surface_z_coord:
  1. Voxels at the surface coord get exactly d=0.0; standard MC sign
     convention (f[i] < 0 → inside) treats that as positive, so no
     sign change occurs and cube_index stays 0.
  2. The anchor needs corners at both z and z+1 to be present in the grid.
```

## test/test_topk_provider.cpp

### test-topk-provider-scope

**TopkProvider unit test scope** — in `File scope`, attached to `#include <cstdint>` (line 1)

```text
Unit tests for scovox::TopkProvider — the file-backed .topk soft-prob loader
extracted from SCovoxNode. Exercises the binary parser, the cache, the fill
helpers, and the corrupt-file fallbacks (the bugs the inline code's comments
document: bad/empty header → bad_alloc, and the gcount short-read footgun)
without spinning up a ROS graph: logging only needs a logger + clock, which
work without rclcpp::init().
```

## test/test_tsdf_band.cpp

### test-tsdf-beta-unperturbed

**TSDF must not perturb Beta state** — in `TEST`, attached to `Map m_off = makeTsdfMap(0.0f);` (line 120)

```text
With TSDF disabled (trunc=0), a single ray puts a known a_occ on the
surface voxel: defaultVoxel().a_occ + w_occ * range_w * angle_w.
Default Params: w_occ=2, range_decay disabled here, angle_w=1, so
a_occ should be 1 + 2 = 3, a_free should be 1. Same numbers must hold
when TSDF is enabled — the fused walk must not perturb Beta state.
```

### test-tsdf-beam-through-occluder

**TSDF through an occluder** — in `TSDFBand.JointRaycastAttenuation`, attached to `TEST(TSDFBand, JointRaycastAttenuation) {` (line 139)

```text
7. Carving a beam through an occluder — wall stays solid, TSDF still lands ---

The wall guard is OFF by default (trust the recent scan), so a beam passing
through the wall carves the whole segment: front voxels get free evidence,
the wall gets one free increment but stays solid (accumulated a_occ
dominates), and TSDF (geometric, not Bayesian) lands at full strength at the
far surface and its band regardless of occupancy.
```
