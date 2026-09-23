# scovox_eval — code comment notes

Long comments from files in the `scovox_eval` package, moved out of the code on 2026-09-23 so the sources carry short comments only. Where a comment was moved, the code keeps a short gist ending in `(notes: <id>)`; the section headed `<id>` below holds the original comment, word for word. Files with many moved comments have their own notes doc next to this one.

One section per source file, in file order. Each entry names the function (or section) the comment sat in, the line of code it was attached to and its original line number. Line numbers, dates, generation numbers and cross-references inside the moved text are as they were when written; they record history and are not maintained.

## Contents

- [scovox_eval/render_qualitative_compare.py](#scovox_evalrender_qualitative_comparepy) — 2
- [scovox_eval/replica_replay_node.py](#scovox_evalreplica_replay_nodepy) — 3
- [scovox_eval/scenenet_replay_node.py](#scovox_evalscenenet_replay_nodepy) — 1
- [scovox_eval/semantickitti_replay_node.py](#scovox_evalsemantickitti_replay_nodepy) — 1
- [scripts/b6_full_flip_simulation.py](#scriptsb6_full_flip_simulationpy) — 2
- [scripts/download_replica_v1.sh](#scriptsdownload_replica_v1sh) — 1
- [scripts/eval_scovox_kitti_miou.py](#scriptseval_scovox_kitti_mioupy) — 1
- [scripts/phase0_smoke_post_semdir.sh](#scriptsphase0_smoke_post_semdirsh) — 2
- [scripts/phase1_kitti_soft_sweep.sh](#scriptsphase1_kitti_soft_sweepsh) — 2
- [scripts/phase2_5_gate_threshold_sweep.sh](#scriptsphase2_5_gate_threshold_sweepsh) — 1
- [scripts/phase2_5_v2_admission_gate_sweep.sh](#scriptsphase2_5_v2_admission_gate_sweepsh) — 1
- [scripts/phase2_component_ablation.sh](#scriptsphase2_component_ablationsh) — 2
- [scripts/phase3_scenenet_fusion_batch.sh](#scriptsphase3_scenenet_fusion_batchsh) — 3
- [scripts/plot_f2_head_to_head.py](#scriptsplot_f2_head_to_headpy) — 1
- [scripts/render_qualitative_compare.py](#scriptsrender_qualitative_comparepy) — 2
- [scripts/render_replica_semantic.py](#scriptsrender_replica_semanticpy) — 1
- [scripts/run_e13_split_batch.sh](#scriptsrun_e13_split_batchsh) — 2
- [scripts/run_e21_fusion_batch.sh](#scriptsrun_e21_fusion_batchsh) — 1
- [scripts/run_e6_matched_kitti.sh](#scriptsrun_e6_matched_kittish) — 1
- [scripts/run_e6_matched_replica.sh](#scriptsrun_e6_matched_replicash) — 1
- [scripts/scenenet_run_batch_iter6.sh](#scriptsscenenet_run_batch_iter6sh) — 1
- [scripts/scovox_ablation_kitti_seq08_polarseg.sh](#scriptsscovox_ablation_kitti_seq08_polarsegsh) — 1
- [scripts/scovox_ablation_ng_m2f.sh](#scriptsscovox_ablation_ng_m2fsh) — 1
- [scripts/scovox_ablation_ng_nr.sh](#scriptsscovox_ablation_ng_nrsh) — 1
- [scripts/scovox_ablation_replica_room0_m2f.sh](#scriptsscovox_ablation_replica_room0_m2fsh) — 1
- [scripts/scovox_b1_ktop_softprob_fanout.sh](#scriptsscovox_b1_ktop_softprob_fanoutsh) — 1
- [scripts/scovox_b7_sensor_physics_kitti.sh](#scriptsscovox_b7_sensor_physics_kittish) — 1
- [scripts/scovox_b7_sensor_physics_replica.sh](#scriptsscovox_b7_sensor_physics_replicash) — 2
- [scripts/scovox_bc_prelim.sh](#scriptsscovox_bc_prelimsh) — 1
- [scripts/scovox_dir_gate_replica.sh](#scriptsscovox_dir_gate_replicash) — 1
- [scripts/scovox_final_replica_all_scenes.sh](#scriptsscovox_final_replica_all_scenessh) — 1
- [scripts/scovox_final_run_all.sh](#scriptsscovox_final_run_allsh) — 1
- [scripts/scovox_iterate_residual_replica.sh](#scriptsscovox_iterate_residual_replicash) — 1
- [scripts/scovox_replica_ng_extend.sh](#scriptsscovox_replica_ng_extendsh) — 1
- [scripts/scovox_residual_attribution_replica.sh](#scriptsscovox_residual_attribution_replicash) — 1
- [scripts/semantickitti_build_gt.py](#scriptssemantickitti_build_gtpy) — 5
- [scripts/semantickitti_run_nofreespace.sh](#scriptssemantickitti_run_nofreespacesh) — 1
- [scripts/slimvdb_harness/run_slimvdb_scenenet_all.sh](#scriptsslimvdb_harnessrun_slimvdb_scenenet_allsh) — 1
- [scripts/slimvdb_harness/run_slimvdb_scenenet_trunc_sweep.sh](#scriptsslimvdb_harnessrun_slimvdb_scenenet_trunc_sweepsh) — 1
- [scripts/slimvdb_harness/slimvdb_build_inside.sh](#scriptsslimvdb_harnessslimvdb_build_insidesh) — 1
- [scripts/step8_fusion_smoke_v3.sh](#scriptsstep8_fusion_smoke_v3sh) — 1
- [scripts/step8_scenenet_fusion_smoke_v3.sh](#scriptsstep8_scenenet_fusion_smoke_v3sh) — 1
- [scripts/tsdf_full_protocol_v2.sh](#scriptstsdf_full_protocol_v2sh) — 1

## scovox_eval/render_qualitative_compare.py

### render-kitti-clip-window

**KITTI render height and square clip** — in `render_kitti`, attached to `y_clip: Optional[Tuple[float, float]] = (-5.0, 2.0),` (line 511)

```text
In cam0 frame: X=right, Y=down (gravity), Z=forward.
Keep ~5 m above road to ~2 m below; clip a 50 m square
around the trajectory mean. Tight enough that SLIM-VDB's
puffy shell stays under ~1M voxels (Filament rendering
budget) while still showing a recognisable urban block.
```

### render-kitti-slimvdb-frame

**Aligning SLIM-VDB voxels to cam0 frame** — in `render_kitti`, attached to `Tr = np.eye(4, dtype=np.float64)` (line 536)

```text
SCovox accumulates voxels in the cam0 world frame (T_world_velo = P @ Tr)
while SLIM-VDB accumulates in the velodyne world frame
(T_world_velo = Tr_inv @ P @ Tr) — see eval_scovox_kitti_miou.py header.
The two frames are related by Tr (velo→cam0), so left-multiply Tr to
bring SLIM-VDB voxels into SCovox/GT's frame for a fair visual overlay.
```

## scovox_eval/replica_replay_node.py

### replica-category-colors

**Replica category color table** — in `Module scope`, attached to `CATEGORY_COLORS = {` (line 49)

```text
Map category names to distinct RGB colors for SCovox semantic input.
These colors must match semantic_color_map_keys/classes in the SCovox launch config.
Class 0 = unknown (black). Unmapped categories also get black.

Format: "category_substring": (R, G, B) — matched case-insensitively.
The RGB value is packed as (R<<16 | G<<8 | B) for the SCovox color map key.
```

### replica-semantic-lookup

**Semantic pixel value to color lookup** — in `ReplicaReplayNode._load_new_format`, attached to `self.sem_class_map = {}` (line 229)

```text
Semantic mapping: build pixel-value → RGB color lookup.
When semantic_subdir == "semantic", pixel values are object/instance IDs
and we map through info_semantic.json objects[] (obj_id → class_name → color).
When semantic_subdir is anything else (e.g. "semantic_gt_fixed",
"semantic_m2f_ade"), pixel values are CLASS IDs directly and we map
through info_semantic.json classes[] (class_id → class_name → color).
```

### replica-camera-pose-tf

**TF rotation for camera-to-world poses** — in `ReplicaReplayNode._publish_tf`, attached to `kR = np.array([[0,0,1],[-1,0,0],[0,-1,0]], dtype=np.float64)` (line 398)

```text
NICE-SLAM: camera-to-world in mesh native frame.
Publish directly — no coordinate conversion.
SCovoxNode's kR converts optical→body; the TF rotation
must undo kR and apply the camera-to-world rotation.
kR converts optical (Z-fwd,X-right,Y-down) → body (X-fwd,Y-left,Z-up)
We need: T_world = T_cam2world * kR^(-1) * p_optical
So TF rotation = R_cam2world * kR^T  (since kR is orthogonal)
But SCovoxNode does: T_oo.linear() = TF_rotation * kR
So: T_oo = R_cam2world * kR^T * kR = R_cam2world ✓
Therefore: TF_rotation = R_cam2world * kR^T
```

## scovox_eval/scenenet_replay_node.py

### scenenet-trajectory-split

**Trajectory split parameters for fusion** — in `SceneNetReplayNode.__init__`, attached to `self.declare_parameter("start_frame", 0)` (line 73)

```text
Step 8 / NEW_EXPERIMENT_PLAN Phase 3 — trajectory split for
multi-robot fusion. start_frame defaults to 0 (full sequence,
matches pre-Step-8 behaviour byte-for-byte); n_scans defaults
to -1 (no cap). For fusion: robot A → start=0 n=200,
robot B → start=100 n=200  (50% overlap convention).
```

## scovox_eval/semantickitti_replay_node.py

### kitti-soft-prob-passthrough

**Soft-prob passthrough keeps raw order** — in `SemanticKITTIReplayNode.__init__`, attached to `self.declare_parameter("soft_prob_passthrough", False)` (line 96)

```text
Soft-prob passthrough: when True, skip the range mask so the
PointCloud2 publishes points in raw .bin order. scovox_node
then indexes into the matching .topk file by point index. The
C++ side still applies the same range filter, so the integration
results match the masked path bitwise.
```

## scripts/b6_full_flip_simulation.py

### b6-log-odds-semantic-rule

**Log-odds semantic baseline constants** — in `Paths`, attached to `L_HIT  = float(np.log(0.7 / 0.3))` (line 79)

```text
Log-odds-semantic rule (OctoMap-style):
  L[obs] += L_HIT;   L[c' seen] += L_MISS  for c' != obs
L_HIT  = log(0.7/0.3)  ≈ +0.847  (consistent with log_odds_map.hpp `l_hit=0.85`)
L_MISS = log(0.4/0.6)  ≈ -0.405  (consistent with `l_miss=-0.40`)
These match the geometric log-odds parameters scovox uses elsewhere, so the
semantic baseline is "the natural per-class extension" rather than tuned.
```

### b6-per-voxel-state

**Per-voxel state of the flip simulator** — in `FlipSim`, attached to `class FlipSim:` (line 115)

```text
State per voxel:
  d_cnt   : float[K_TOP]    Dirichlet sem counts
  d_cls   : uint16[K_TOP]   Dirichlet sem class IDs
  d_unk   : float           Dirichlet a_unk residual
  m_cnt   : float[K_TOP]    MV sem counts
  m_cls   : uint16[K_TOP]   MV sem class IDs
  np_cls  : uint16          NP last-class (0 = no obs yet)
  lo_L    : dict[cls -> L]  log-odds per class

Plus per-rule:
  {rule}_argmax  : uint16   current argmax class (0 = unknown / no obs)
  {rule}_flips   : uint32

We also track total observation count per voxel (same across rules).

We use parallel numpy arrays keyed by a per-voxel index. The
voxel-key→index map is built lazily as we encounter new keys.
```

## scripts/download_replica_v1.sh

### replica-download-streaming

**Streaming extract of Replica v1 scenes** — in `Top level`, attached to `set -e` (line 4)

```text
Source: facebookresearch/Replica-Dataset release v1.0 (17-part tarball).
The upstream download.sh extracts the entire ~100 GB dataset; this wrapper
streams through the tarball and keeps only the scenes we need, then deletes
the .part?? archives to reclaim disk.
```

## scripts/eval_scovox_kitti_miou.py

### kitti-replay-to-yaml-lut

**REPLAY to yaml class remap** — in `Module scope`, attached to `_REPLAY_TO_YAML_LUT = np.arange(256, dtype=np.int32)` (line 54)

```text
REPLAY → yaml learning-class LUT (per `[[kitti-miou-replay-bug]]` memo).
PolarSeg's .topk files use the REPLAY scheme which inserts class 15 =
"lane-marking" and shifts veg/trunk/terrain/pole/traffic-sign +1 vs
yaml's learning_map (which has no lane-marking dim). Predicted voxels
in REPLAY space (16..20) must be remapped to yaml space (15..19) before
bucket-IoU against yaml-space GT; otherwise those classes score 0.
Lane-marking (REPLAY 15) merges into road (yaml 9 = "road") — the
PolarSeg convention treats it as a road sub-class.
```

## scripts/phase0_smoke_post_semdir.sh

### smoke-phase0-purpose

**What the Phase 0 smoke gate guards** — in `Top level`, attached to `set -o pipefail  # don't enable -u: ROS setup.bash references unbound AMENT_TRACE_SETUP_FILES` (line 4)

```text
Validates Step 7.5 didn't regress the integration substrate on the two
canonical anchors AND that the soft-prob loader is actually dispatching
(guards against the silent-fallback footgun pinned by
[[softprob-pipeline-2026-05-04]] — the SemBeta-era room0 0.352 vs 0.461
regression that turned out to be the topk_probs_dir parameter not
reaching the loader).
```

### smoke-phase0-assertions-gates

**Smoke gate assertions and mIoU bands** — in `Top level`, attached to `set -o pipefail  # don't enable -u: ROS setup.bash references unbound AMENT_TRACE_SETUP_FILES` (line 16)

```text
Sharp assertions (catch silent fallbacks regardless of mIoU magnitude):
  - Cell B's log MUST contain "topk loader: loaded=N" with N >= 95
  - Cell A's log MUST NOT contain any "topk loader:" line
  - Cell B's mIoU MUST exceed Cell A's by ≥ 0.02 (proves soft probs
    actually changed the posterior; equality means silent fallback)

mIoU gates per NEW_EXPERIMENT_PLAN.md:
  - Cell B: SemBeta baseline 0.3030 ± 0.02 → pass if ∈ [0.283, 0.323]
  - Cell C: SemBeta baseline 0.3624 ± 0.02 → pass if ∈ [0.342, 0.382]
  - Cell A: no fixed gate (hard-label is just the comparator for B)
```

## scripts/phase1_kitti_soft_sweep.sh

### ktop-soft-needs-both-params

**Soft-prob ingestion needs both parameters** — in `Top level`, attached to `set -o pipefail` (line 4)

```text
The original phase1_ktop_sweep.sh accidentally ran KITTI with HARD labels
because it set `soft_prob_passthrough:=true` on the replay node but did
NOT set `topk_probs_dir:=...` on scovox_node — both are required for
soft-prob ingestion per [[project-softprob-pipeline-2026-05-04]]. The
datasets table says "PolarSeg soft" but the executed numbers were hard.
```

### eval-pythonpath-bypass-pip

**Why PYTHONPATH points at the source** — in `Top level`, attached to `export PYTHONPATH="${EVAL_PKG}:${PYTHONPATH}"` (line 33)

```text
scovox_eval is a pip-editable package whose install was previously broken
(egg-link pointed at /home/kalhan/Projects/... — capital P — but actual ws
is /home/kalhan/projects/...). Bypass pip by injecting the source dir
into PYTHONPATH directly so `python3 -m scovox_eval.*` resolves.
```

## scripts/phase2_5_gate_threshold_sweep.sh

### gate-sweep-publish-vs-integration

**Publish-time versus integration-time gate** — in `Top level`, attached to `set -o pipefail` (line 9)

```text
Note on naming: NEW_EXPERIMENT_PLAN.md Phase 2.5 spec calls this knob
`dirichlet_min_p_occ`, but in the code that name is the INTEGRATION-
time semantic-commit gate (semdir_map.cpp:applyHitUpdate). The publish-
time / labelling-envelope gate — which is what Phase 2.5 actually
ablates per the spec text ("the publish-time threshold for `is this
voxel occupied?`") — is `occupancy_vis_threshold` / `min_occ_`
(scovox_node.cpp:publishPointCloud). This script varies the latter.
```

## scripts/phase2_5_v2_admission_gate_sweep.sh

### admission-gate-sweep-design

**Integration-time admission gate sweep** — in `Top level`, attached to `set -o pipefail` (line 4)

```text
Companion to the publish-time Phase 2.5 (which returned a null result
because low-p_occ voxels carry class=Unknown which the scorer ignores).
This sweeps the integration-time gate `dirichlet_min_p_occ` instead,
which actually controls which voxels get a real class commit.

At t=0.0 every hit commits a class regardless of occupancy posterior
(SLIM-VDB-like envelope). Predicts mIoU drops monotonically as threshold
decreases, with the largest drop between 0.3 and 0.0.

Anchors: KITTI seq08 (LiDAR + PolarSeg soft) and SceneNet 0_223 (RGB-D
GT labels). 4 thresholds × 2 anchors = 8 cells.

Wall-clock budget: ~50 min (each cell needs a fresh integration, unlike
Phase 2.5 which reused the same SemDirMap state through different
publish thresholds).
```

## scripts/phase2_component_ablation.sh

### ablation-phase2-anchors-modes

**Phase 2 anchors, modes and label caveat** — in `Top level`, attached to `set -o pipefail` (line 2)

```text
NEW_EXPERIMENT_PLAN.md Phase 2 — component ablation on the post-SemDir
substrate. **Expanded from the plan's 4 anchors to 6** by adding two
more KITTI sequences (seq06, seq10) alongside the canonical seq08 —
same coverage convention as Phase 4 h2h, which uses 5 KITTI seqs.
With 3 modes that's 6 × 3 = 18 cells; run on use_split=true to
exercise the SemDirMap path.

Anchors:
  - KITTI seq06 (added; suburban, fast traffic — different distribution)
  - KITTI seq08 (plan's canonical anchor)
  - KITTI seq10 (added; highway/long-straight, less semantic clutter)
  - SceneNet 0_789  (high SCovox mIoU = 0.385)
  - SceneNet 0_723  (low  SCovox mIoU = 0.269)
  - SceneNet 0_485  (mid  SCovox mIoU = 0.303)
Modes: dirichlet (D), majority_vote (MV), naive (NP)

⚠️ SceneNet uses GT one-hot labels → soft-prob reduces to hard. The
"history matters" finding (D vs NP, D vs MV) still tests cleanly on
both datasets. The "soft-prob recovers calibration signal" finding
is KITTI-only (PolarSeg .topk).
```

### kitti-replay-lut-flag

**Why scoring uses the REPLAY-to-yaml LUT** — in `Top level`, attached to `miou=$(python3 "${EVAL_PKG}/scripts/eval_scovox_kitti_miou.py" \` (line 155)

```text
`--replay_to_yaml_lut` remaps PolarSeg's REPLAY class layout (lane-
marking at id 15, veg/trunk/terrain/pole +1 vs yaml learning_map)
back to yaml space before bucket-IoU. Without it 4 classes score 0
and KITTI mIoU drops by ~0.10 absolute. See kitti-miou-replay-bug
2026-05-11 memo + Phase 0 smoke for the canonical invocation.
```

## scripts/phase3_scenenet_fusion_batch.sh

### fusion-batch-split-and-skip

**Per-trajectory robot split and skipping** — in `Top level`, attached to `set -o pipefail` (line 4)

```text
Per-trajectory split: robot A=[0,200), robot B=[100,300) (50% overlap
per the plan). wire_format=v3 end-to-end (validated by Step 8 fusion
smoke commit a958172).

Iterates the existing step8_scenenet_fusion_smoke_v3.sh logic per
trajectory rather than calling it as a black box — gives finer control
over result-dir layout (one cell per trajectory) and per-trajectory
logging.

Default trajectory list = the 13 from the SceneNet head-to-head batch
(project-scenenet-first-batch-2026-05-12). Idempotent: cells with all
3 NPZs present are skipped.
```

### fusion-batch-default-trajs

**Default SceneNet trajectory list** — in `Top level`, attached to `TRAJS_DEFAULT=(0_175 0_178 0_182 0_223 0_279` (line 29)

```text
13 val trajs actually staged on disk under scenenet_val_layout/train/
(2026-05-14 audit). The prior default referenced the head-to-head 13
from [[project-scenenet-first-batch-2026-05-12]], but only 4 of those
overlap with what's on disk. The 13 below are all immediately usable
without re-downloading the 262 GB train shards.
```

### fusion-batch-skip-missing-trajs

**Skipping trajectories with no data** — in `Top level`, attached to `_PRESENT=()` (line 45)

```text
Filter to trajectories that actually have data staged. Per the
[[project-scenenet-rgbd-mirror]] memo, only a subset of the head-to-
head 13 has data on disk — full train shards are 262 GB and won't fit.
Missing trajs would just silently NaN out the scoring; skipping them
up front keeps the matrix honest.
```

## scripts/plot_f2_head_to_head.py

### f2-kitti-scoring-protocol

**KITTI numbers use the strict protocol** — in `Module scope`, attached to `KITTI_SEQS  = ["seq06", "seq07", "seq08", "seq09", "seq10"]` (line 20)

```text
2026-05-09 protocol fix: SCovox numbers re-scored with eval_scovox_kitti_miou.py
(strict bucket-IoU on pred ∪ gt) — same protocol as the SLIM-VDB column.
Pre-fix numbers (lenient KD-Tree pred→GT match @10cm via eval_ablations_kitti_seq08.py)
preserved in git as the prior values: hard=[0.6253,0.5461,0.4328,0.4447,0.4117],
soft=[0.6309,0.5536,0.4394,0.4730,0.4124] — DO NOT REUSE for SLIM-VDB head-to-head.
```

## scripts/render_qualitative_compare.py

### render-kitti-crop-window

**KITTI render crop window** — in `render_kitti`, attached to `y_clip: Optional[Tuple[float, float]] = (-5.0, 2.0),` (line 511)

```text
In cam0 frame: X=right, Y=down (gravity), Z=forward.
Keep ~5 m above road to ~2 m below; clip a 50 m square
around the trajectory mean. Tight enough that SLIM-VDB's
puffy shell stays under ~1M voxels (Filament rendering
budget) while still showing a recognisable urban block.
```

### render-kitti-slim-frame

**Bringing SLIM-VDB into the cam0 frame** — in `render_kitti`, attached to `Tr = np.eye(4, dtype=np.float64)` (line 536)

```text
SCovox accumulates voxels in the cam0 world frame (T_world_velo = P @ Tr)
while SLIM-VDB accumulates in the velodyne world frame
(T_world_velo = Tr_inv @ P @ Tr) — see eval_scovox_kitti_miou.py header.
The two frames are related by Tr (velo→cam0), so left-multiply Tr to
bring SLIM-VDB voxels into SCovox/GT's frame for a fair visual overlay.
```

## scripts/render_replica_semantic.py

### replica-pose-frame-conversion

**NICE-SLAM to Habitat pose conversion** — in `Module scope`, attached to `_WORLD_REPLICA_TO_HABITAT = np.array([` (line 94)

```text
NICE-SLAM traj.txt: c2w, camera axes OpenCV (y-down, z-fwd), world axes Replica
native (z-up, gravity along -Z). Habitat expects: camera axes OpenGL
(y-up, -z-fwd), world axes y-up (gravity along -Y). Compose two transforms:
  (a) world rotation: Replica world (z-up) -> Habitat world (y-up): Rx(-90°)
  (b) camera-axis flip: OpenCV -> OpenGL: diag(1,-1,-1,1) applied on the right.
```

## scripts/run_e13_split_batch.sh

### e13-cell-sheet

**E1.3 cell sheet and outputs** — in `Top level`, attached to `set -o pipefail` (line 4)

```text
Per Step-12 cell sheet:
  8 Replica scenes (M2F soft, K=2)
  5 KITTI sequences (PolarSeg soft, K=2)
  use_split:=true, share_tsdf:=false

Per cell capture:
  results/e13_split_2026_05_08/{replica,kitti}/<scene>/
    scovox.npz, scovox_run.log, summary.txt

After all 13 cells: scoring + summary CSV with mIoU / FPS / TsdfMap MB /
SemBetaMap MB. Compares TsdfMap bytes against SLIM-VDB bytes per the
E1.3 spec ("expect within 5%").

Idempotent: cells whose scovox.npz exists are skipped.
Sequential by user preference (feedback_sequential_execution.md).
```

### e13-topk-rotation

**Rotating the Replica topk dirs** — in `Replica cells`, attached to `if [[ -d "${topk}" ]]; then` (line 95)

```text
Free disk for next scene (Replica topk dirs are ~29 GB each — without
rotation the second cell ENOSPCs on regen). The original "spare room0"
exclusion was wrong; topk is always re-derivable from ade_probs/.
```

## scripts/run_e21_fusion_batch.sh

### e21-fusion-topology

**E2.1 fusion topology and outputs** — in `Top level`, attached to `set -o pipefail` (line 4)

```text
Per-scene topology (replica_eval_fusion.launch.py + use_split=true):
  /robotA/scovox_node  ← replay frames [0, 1000)
  /robotB/scovox_node  ← replay frames [1000, 2000)
  /dscovox_node        ← merges both via wire-format v2 (37 B/voxel SemBeta)

Per cell capture:
  results/e21_fusion_2026_05_08/<scene>/{solo_a,solo_b,fused}.npz

Score: eval_e21_fusion.py — voxel-mIoU + Chamfer + F@5cm vs replica GT.
Verdict per spec: fused beats max(solo_a, solo_b) on mIoU, F@5cm, Chamfer.

Idempotent: scenes with all 3 npz present are skipped.
Sequential execution per user preference.
```

## scripts/run_e6_matched_kitti.sh

### e6-kitti-matched-config

**E6.1 KITTI matched-config run** — in `Top level`, attached to `set -o pipefail` (line 4)

```text
Per sequence (06,07,08,09,10):
  1. Run scovox_node with matched config (band_only=true, range_decay=-1,
     min/max_range=5/30, voxel=0.10, sdf_trunc=0.30, K_TOP=2,
     min_tsdf_weight_publish=10.0, soft PolarSeg topk).
  2. Replay 100 scans at rate_hz=0.5.
  3. Capture ~/pointcloud → scovox.npz.

Per cell:  results/matched_config_2026_05_08/kitti/<seq>/scovox.npz

Idempotent: seqs whose scovox.npz exists are skipped. predictions_topk
already exists per seq (no rotation needed).
```

## scripts/run_e6_matched_replica.sh

### e6-replica-matched-config

**E6.1 Replica matched-config run** — in `Top level`, attached to `set -o pipefail` (line 4)

```text
Per scene:
  1. (re)generate semantic_m2f_topk/ from semantic_m2f_ade_probs/
  2. Run scovox_node with matched config (band_only=true, range_decay=-1,
     min/max_range=0.2/8.0, voxel=0.05, sdf_trunc=0.15, K_TOP=2,
     min_tsdf_weight_publish=5.0, soft m2f topk).
  3. Replay 2000 frames at rate_hz=2.0.
  4. Capture ~/pointcloud → scovox.npz.
  5. Delete the topk dir to free disk for the next scene.

Per cell saved under
  results/matched_config_2026_05_08/replica/<scene>/
in scovox.npz, scovox_run.log.

Idempotent: scenes whose scovox.npz exists are skipped.
```

## scripts/scenenet_run_batch_iter6.sh

### scenenet-iter6-steps

**SceneNet iter6 per-cell steps** — in `Top level`, attached to `set -o pipefail` (line 4)

```text
For each preprocessed trajectory under data/scenenet_val_layout/train/<seq>/:
  1. Launch scovox_mapping_node with scenenet_eval.launch.py
  2. Replay 300 frames via scenenet_replay_node at 10 Hz
  3. Capture ~/pointcloud → NPZ
  4. Build GT voxel grid via scenenet_build_gt.py
  5. Score with scenenet_compute_metrics.py at the end

Defaults: use_split:=true, share_tsdf:=false, fused_walker:=true,
          semantic_mode:=dirichlet, resolution:=0.05, K_TOP=2 (compile-time).
```

## scripts/scovox_ablation_kitti_seq08_polarseg.sh

### ablation-kitti-baseline

**KITTI ablation baseline and candidates** — in `Top level`, attached to `set -eo pipefail` (line 4)

```text
See docs/issues/ablations_punch_list.md for the candidate list.

Baseline (post-NG): range_decay=50, w_occ=6, w_free=1, kappa0=2,
  evidence_saturation=1000, semantic_min_confidence=0.1, carve_skip=0.4,
  semantic_occ_gate=0 (NG).
```

## scripts/scovox_ablation_ng_m2f.sh

### ablation-ng-m2f-goal

**NG ablation under Mask2Former labels** — in `Top level`, attached to `set -eo pipefail` (line 4)

```text
Mirrors scovox_ablation_ng_nr.sh's NG block but feeds Mask2Former ADE-150
predictions (semantic_subdir=semantic_m2f_ade) instead of GT labels. Goal:
show whether the occupancy gate's protective value materialises under
noisy segmentation, complementing the GT finding (gate=0 helped + 0.32 mIoU
on GT — gate hurts under clean labels).
```

## scripts/scovox_ablation_ng_nr.sh

### ablation-ng-nr-hypotheses

**NG and NR ablation hypotheses** — in `Top level`, attached to `set -eo pipefail` (line 6)

```text
Hypothesis:
  NG  — without the gate, free-space rays vote on labels → semantic ECE worse.
  NR  — without range decay, far-range noisy observations carry equal weight
        → mIoU degrades at map edges.
```

## scripts/scovox_ablation_replica_room0_m2f.sh

### ablation-replica-baseline

**Replica room0 ablation baseline** — in `Top level`, attached to `set -eo pipefail` (line 3)

```text
See docs/issues/ablations_punch_list.md for the candidate list.

Each cell: name=<knob>=<val>, with all other knobs at the post-NG baseline:
  range_decay=-1, w_occ=2, w_free=1, kappa0=2, evidence_saturation=1000,
  semantic_min_confidence=0.1, carve_skip=0.7, semantic_occ_gate=0 (NG).
(Note: replica_eval default w_occ is 6 in the launch file but the paper
 baseline for indoor RGB-D is 2.0 — see default_params.yaml. We pass 2.0
 explicitly to align with the documented baseline.)
```

## scripts/scovox_b1_ktop_softprob_fanout.sh

### ktop-fanout-scene-seq

**Fan-out variant scene and seq env** — in `Top level`, attached to `set -eo pipefail` (line 4)

```text
Like scovox_b1_ktop_softprob.sh but parameterised by Replica scene name
and KITTI seq number so it can drive E1.0+E1.1 (per-scene rotation) and
E1.2 (per-seq fan-out). Reads SCENE / SEQ from environment; defaults to
room0 / 8 to stay backward-compatible with the single-anchor wrapper.
```

## scripts/scovox_b7_sensor_physics_kitti.sh

### b7-kitti-cells

**KITTI B7 cells and their purpose** — in `Top level`, attached to `set -eo pipefail` (line 4)

```text
Key purpose: confirm whether the post-cleanup mIoU regression seen on
Replica room0 also appears on LiDAR. Pre-cleanup baseline lives in
results/ablations_kitti_seq08_polarseg/baseline/ (2026-04-26).

Three cells:
  b7_kitti_old_default     (6.00, 1.00)  — pre-cleanup KITTI default; recheck
  b7_kitti_lidar_physics   (8.00, 4.67)  — sensor-physics LiDAR derivation
  b7_kitti_moderate        (2.00, 1.00)  — RGB-D modality mismatch (negative ctrl)

100 scans @ 0.5 Hz ≈ 4-5 min/cell.
```

## scripts/scovox_b7_sensor_physics_replica.sh

### b7-replica-cells

**Replica B7 cells** — in `Top level`, attached to `set -eo pipefail` (line 4)

```text
Runs three cells:
  b7_moderate_recheck    (2.00, 1.00)  — current default; sanity check vs baseline_newgate_05
  b7_conservative        (1.33, 0.50)  — OctoMap defaults (prob_hit=0.7, prob_miss=0.4)
  b7_lidar               (8.00, 4.67)  — LiDAR-tuned (off-modality, sanity for KITTI param)
```

### b7-replica-post-cleanup-baseline

**Replica B7 post-cleanup baseline args** — in `Top level`, attached to `BASE_ARGS=(` (line 27)

```text
Post-cleanup baseline. Only params that still exist after the
bayesian-mapping-cleanup branch (tau/k_gate/s_min/sat/smc/csk removed).
```

## scripts/scovox_bc_prelim.sh

### bc-prelim-scope

**B/C prelim sweep scope and cells** — in `Top level`, attached to `set -eo pipefail` (line 4)

```text
Goal: identify which open B/C ablations actually move mIoU on
Replica room0 m2f and KITTI seq08 PolarSeg. Items not exercised on these
anchors (B2 needs recompile, B6 needs instrumentation, C1–C4 need
multi-robot fusion) are deferred and not run here.

Replica cells (added to results/ablations_replica_room0_m2f/):
  grazon_03    B4 — grazing_angle_threshold=0.3 (default value, on)
  q_both_03    B5 — range_decay=10 AND grazing=0.3 (full quality scaling)

KITTI cells (added to results/ablations_kitti_seq08_polarseg/):
  trans_678    B3 — dynamic_classes=[6,7,8] routed to transient layer
```

## scripts/scovox_dir_gate_replica.sh

### dir-gate-replica-probe-origin

**Origin of the p_occ 0.6 gate probe** — in `Top level`, attached to `set -eo pipefail` (line 2)

```text
Final residual probe: add the Dirichlet early-return back at p_occ=0.6
(matching the OLD gate_value < semantic_occ_gate=0.6 default). Tests
whether this closes the remaining −0.0252 mIoU gap from baseline.
```

## scripts/scovox_final_replica_all_scenes.sh

### final-replica-scene-loop

**Final Replica per-scene loop and disk** — in `Top level`, attached to `set -eo pipefail` (line 4)

```text
  1. Run M2F dense inference (collapse_n_classes=19) → NPZ
  2. Convert NPZ → .topk (~29 GB peak)
  3. Run SCovox soft sweep
  4. Run SCovox hard sweep (no .topk needed)
  5. Delete this scene's .topk + NPZ to free disk for next scene

Disk budget: ~30 GB peak/scene. Free disk should be ≥ 35 GB at start.
Total compute: ~50 min/scene × 8 = ~6.5 hours.
```

## scripts/scovox_final_run_all.sh

### final-run-phases

**Final run phases, resume and disk** — in `Top level`, attached to `set -eo pipefail` (line 6)

```text
Phase 1:  KITTI SCovox (hard + soft) for all 5 seqs            ~35 min
Phase 2:  Replica SCovox per-scene loop (M2F → topk → soft →   ~50 min × 8 = 6.7h
          hard → cleanup)
Phase 3:  KITTI SLIM-VDB all seqs (existing docker runner)     ~25 min
Phase 4:  Replica SLIM-VDB all scenes (existing docker runner) ~4 h

Total: ~11 hours. Idempotent — each cell has a `[[ -f scovox.npz ]]`
guard so partial runs resume safely. Logs to:
  /tmp/softprob_logs/final_run_phase{1..4}.log

Disk: peak ~35 GB during a Replica scene (.topk + NPZ). Free ≥ 35 GB.
```

## scripts/scovox_iterate_residual_replica.sh

### residual-iterate-cells

**Residual iteration cells** — in `Top level`, attached to `set -eo pipefail` (line 5)

```text
  res_all_old_knobs   sat=1000 + smc=0.1 + gate_k=12 + dir_min_p_occ=0.6
                      (closest reproduction of pre-cleanup pipeline)
  res_jitter_check    identical config to post_revert_default
                      (re-run to measure single-cell run jitter)
```

## scripts/scovox_replica_ng_extend.sh

### ng-extend-scope

**Extending the NG ablation to 8 scenes** — in `Top level`, attached to `set -eo pipefail` (line 2)

```text
Extend the SCovox-NG ablation to all 8 Replica scenes for both label
regimes (semantic_gt_fixed and semantic_m2f_ade). Skips scenes that
already have an NPZ. The 3-scene baseline NG runs (room0, office0,
office3) live in:
  replica_ng_5cm/      (GT-oracle labels)
  replica_ng_m2f_5cm/  (m2f predictions)
This script adds the remaining 5 scenes (room1, room2, office1, office2,
office4) to each, so the NG mIoU column is comparable to the 8-scene
baseline (gate=0.6) numbers in exp3 / paper_experiments tables.
```

## scripts/scovox_residual_attribution_replica.sh

### residual-attribution-cells

**Residual attribution cells** — in `Top level`, attached to `set -eo pipefail` (line 3)

```text
Compares against `post_revert_default` (mIoU 0.3394). Each cell restores
one pre-cleanup feature so we can isolate which contributes to the
remaining −0.0252 gap vs the original baseline (0.3646).

  res_smc          + semantic_min_confidence = 0.1   (rest off)
  res_sat          + evidence_saturation = 1000      (rest off)
  res_smooth_gate  + gate_k = 12 (smooth sigmoid)    (rest off)
```

## scripts/semantickitti_build_gt.py

### gt-build-in-memory-options

**Rejected in-memory GT accumulators** — in `main`, attached to `import tempfile, os` (line 109)

```text
Running accumulator: key(int64) -> [counts_per_class] as numpy array
Use a simple approach: collect all (key, label) from all scans,
then do ONE final sort + groupby. But stream to disk to limit RAM.

Actually: just use a Python dict[int, np.uint16[20]].
50M voxels × (40 bytes array + ~120 bytes dict overhead) ≈ 8 GB. Too much.

Compromise: use dict[int, int] mapping key -> packed(best_label << 16 | best_count).
This is a simple running max — NOT a true majority vote, but close enough
when one label dominates (which it does for most voxels at 10cm).

Better compromise: dict[int, int] mapping key -> (label << 16 | count) for the
top label only. Update: if new label == stored label, increment count. If new
label != stored label, decrement count; if count reaches 0, replace with new.
This is the Boyer-Moore majority vote algorithm — exact majority in one pass.
```

### gt-build-boyer-moore

**Boyer-Moore vote memory estimate** — in `main`, attached to `import tempfile, os` (line 125)

```text
Boyer-Moore streaming majority vote per voxel
Store: voxel_key -> (candidate_label, count)
Python int is 28 bytes, tuple is 56 bytes, dict entry ~100 bytes
50M entries × ~180 bytes = 9 GB. Still too much.
```

### gt-build-external-sort

**External sort option for GT pairs** — in `main`, attached to `import tempfile, os` (line 130)

```text
NUCLEAR OPTION: just write per-scan (key, label) arrays to disk,
then process with external sort in chunks.
```

### gt-build-temp-file-option

**Sort and groupby over temp files** — in `main`, attached to `import tempfile, os` (line 133)

```text
SIMPLEST OPTION THAT WORKS: Process scans, build per-scan voxelized
arrays, concatenate keys+labels, sort, groupby. But do it with
memory-mapped temp files.
```

### gt-build-disk-pairs

**On-disk voxel-label pairs approach** — in `main`, attached to `import tempfile, os` (line 137)

```text
Let's try: accumulate per-scan unique (key, label) pairs.
~50k unique voxels/scan × 4071 scans = ~200M pairs × 9 bytes = 1.8 GB on disk.
Then mmap, sort by key, groupby.
```

## scripts/semantickitti_run_nofreespace.sh

### freespace-ablation-scope

**Freespace ablation variants and outputs** — in `Top level`, attached to `set -eo pipefail` (line 4)

```text
The existing 4.9 runs (in results/semantickitti_<res>cm/) used carve_band=0
(endpoint-only, NO freespace carving). This script re-runs SCovox (Dirichlet)
under two explicit freespace configs so we can compare.

Variant:
  scovox_dirichlet_fullray  — carve_band=-1 (full ray freespace carving)

The nocarve baseline already exists as scovox.npz in results/semantickitti_<res>cm/
(task 4.9, carve_band=0). No need to re-run it.

Output directory:
  results/semantickitti_<res>cm_freespace_ablation/
    scovox_dirichlet_nocarve.npz       — captured voxel map
    scovox_dirichlet_nocarve_run.log   — mapping node log (timing/memory)
    scovox_dirichlet_fullray.npz
    scovox_dirichlet_fullray_run.log
    gt.npz                             — symlinked from main results dir

This is a separate directory from the main 4.9 results to keep experiments
cleanly isolated. File names include the semantic mode (dirichlet) and the
carving config (nocarve/fullray) so they are self-documenting.

Metrics are computed via semantickitti_compute_metrics.sh pointed at
this directory, with VARIANTS overridden to match the file names above.
```

## scripts/slimvdb_harness/run_slimvdb_scenenet_all.sh

### scenenet-docker-mounts

**SceneNet container cwd, mounts and display** — in `Top level`, attached to `docker run --rm --gpus all \` (line 94)

```text
The pipeline binary's CWD must contain config/scenenet.yaml; we cd into
examples/cpp/ (where config/ lives in the source tree) and run from there.
We bind-mount the host workspace path twice — once at /workspace (legacy
convention shared with all other runners) and once at the host path
itself, so the host-absolute symlinks under data/scenenet_val_layout/...
and data/scenenet/val_preprocessed/.../depth/*.png resolve correctly
inside the container without rewriting ~10k symlinks to be relative.
scenenet_pipeline's Render() step uses Open3D/OpenCV which want a GUI
backend (GTK) — even for offscreen rendering. Forward the host X
display so cvInitSystem can find a usable connection.
```

## scripts/slimvdb_harness/run_slimvdb_scenenet_trunc_sweep.sh

### trunc-sweep-hypothesis

**Why sweep SLIM-VDB truncation distance** — in `Top level`, attached to `set -euo pipefail` (line 4)

```text
Tests the hypothesis: SCovox's mIoU advantage over SLIM-VDB comes mostly from
SLIM-VDB writing semantic labels into the ±sdf_trunc TSDF shell around every
depth measurement, which leaks FPs into the immediate free-space envelope.

We re-run SLIM-VDB at sdf_trunc ∈ {0.05, 0.075, 0.15} and score with the
same strict bucket-IoU. Narrower trunc → fewer free-space FPs → mIoU ↑.
```

## scripts/slimvdb_harness/slimvdb_build_inside.sh

### slimvdb-build-openvdb-module-path

**Finding OpenVDB when building SLIM-VDB** — in `2/3  SLIM-VDB library`, attached to `export CPLUS_INCLUDE_PATH="${OVDB_PREFIX}/include${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"` (line 71)

```text
SLIM-VDB uses find_package(OpenVDB) in CMake MODULE mode — OpenVDB ships
FindOpenVDB.cmake (not OpenVDBConfig.cmake), so CMAKE_MODULE_PATH must include
our OpenVDB prefix's cmake/OpenVDB dir. CMAKE_PREFIX_PATH alone isn't enough.
Export CPLUS_INCLUDE_PATH because SLIM-VDB's CMakeLists overwrites
CMAKE_CXX_FLAGS (kills our -I), so CXX compiles can't find openvdb headers.
```

## scripts/step8_fusion_smoke_v3.sh

### step8-v3-smoke-pass-criteria

**Step 8 v3 fusion smoke checks** — in `Top level`, attached to `set -o pipefail` (line 4)

```text
Validates the v3 publish/receive loop landed in dscovox_node 3a35eae:
  - scovox_node A integrates frames [0, 200)   on SemDirMap, emits v3 frames
  - scovox_node B integrates frames [200, 400) on SemDirMap, emits v3 frames
  - dscovox_node deserialises v3 frames, merges per consensus_merge_v3.hpp
  - all three publish ~/pointcloud with the same 11-field schema

Pass criteria:
  1. fused.npz exists and contains > 0.5 × min(solo_a.points, solo_b.points)
     — proves dscovox didn't silently drop everything via the version guard
  2. dscovox log shows "v3 receive: pinned num_classes=14 alpha_0=0.0100"
     — proves at least one v3 frame deserialised
  3. dscovox log has NO "Bad version 3" or "v3 prior mismatch" lines
     — proves both robots' frames reached the SemDir merge
  4. fused mIoU ≥ max(solo_a, solo_b) − 0.02 (per Step 11 spec, with
     tolerance for a short-trajectory split that under-samples geometry)

Wall-clock: ~3 minutes at N=200 frames per robot.
```

## scripts/step8_scenenet_fusion_smoke_v3.sh

### scenenet-smoke-trajectory-split

**SceneNet smoke trajectory split** — in `Top level`, attached to `set -o pipefail` (line 10)

```text
Trajectory split (NEW_EXPERIMENT_PLAN.md Phase 3 convention): 50%
overlap. Robot A=[0..200), Robot B=[100..300). Short for smoke; the
full Phase 3 batch will use [0..200) / [100..300) too but on 13 trajs.
```

## scripts/tsdf_full_protocol_v2.sh

### tsdf-v2-carve-band-fused-walk

**Why carve_band now fills the band** — in `Top level`, attached to `set -eo pipefail` (line 6)

```text
Patched scovox_node now routes carve_band>0 through the fused walk so the
band is actually populated (no longer single-cell at the hit).
```
