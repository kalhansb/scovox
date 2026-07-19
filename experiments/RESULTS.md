# SCovox reproduction on the `scovox` repo — results

Experiments from `scovox_paper_code` rerun against the `scovox` repository
(system under test). Datasets staged under `scovox_ws/data/`, harness under
`scovox_ws/experiments/`.

## Environment
- ROS 2 Humble, GCC 11, colcon; scovox built Release (`scovox_mapping_node`).
- Segmentation inference on **CPU** — the Quadro P2000 (sm_61, Pascal) is
  unsupported by the installed PyTorch 2.10 (needs sm_70+).

## Datasets staged
- **SemanticKITTI** seqs 06–10, first 100 frames each (velodyne fetched via
  HTTP range from the 85 GB odometry zip — ~1 GB actual) + labels/poses/calib.
- **SceneNet RGB-D val**, 13 paper trajectories, converted to SLIM-VDB layout
  (`data/scenenet_val_layout/train/<seq>/`), GT voxel grids at 5 cm.

## SceneNet Phase 0 smoke (GT one-hot input, paper protocol)
0_223 mIoU = **0.3750**, inside the plan's gate [0.342, 0.382]
(SemBeta baseline 0.3624). Pipeline validated end-to-end.

## SceneNet GT-input, all 13 trajs (paper protocol)
mean mIoU = **0.351** (range 0.304–0.397). Matches the paper's SceneNet range
(e.g. 0_789 ≈ 0.397 vs paper's high-anchor ~0.385).

## SceneNet SOFT labels (NEW — beyond the paper)
The paper used GT one-hot for SceneNet. This adds a predicted-label pipeline:
Mask2Former (ADE20k, swin-small) on the RGB frames → per-pixel 150-class
posterior → collapsed to SceneNet's 14 classes
(`soft_scenenet/ade150_to_scenenet14.json`) → quantised `.topk` blobs → fed to
scovox's RGB-D soft-prob path (`topk_probs_dir`, 0 % loader fallback).

| Trajectory | GT-input mIoU | Soft mIoU | Δ (soft−GT) |
|---|---|---|---|
| 0_175 | 0.304 | 0.050 | −0.254 |
| 0_178 | 0.358 | 0.156 | −0.201 |
| 0_182 | 0.370 | 0.130 | −0.240 |
| 0_223 | 0.375 | 0.225 | −0.150 |
| 0_279 | 0.370 | 0.176 | −0.195 |
| 0_485 | 0.314 | 0.090 | −0.225 |
| 0_490 | 0.316 | 0.110 | −0.206 |
| 0_571 | 0.304 | 0.103 | −0.201 |
| 0_682 | 0.390 | 0.132 | −0.258 |
| 0_723 | 0.352 | 0.120 | −0.233 |
| 0_789 | 0.397 | 0.088 | −0.309 |
| 0_867 | 0.371 | 0.061 | −0.310 |
| 0_977 | 0.340 | 0.091 | −0.248 |
| **mean** | **0.351** | **0.118** | **−0.233** |

**Reading:** a generic ADE20k segmenter (not fine-tuned on NYUv2/SceneNet)
loses ~0.23 mIoU vs ground-truth labels. Driven by (1) the sim-to-real domain
gap between synthetic SceneNet renders and ADE20k's real-photo training set,
and (2) the coarse 150→14 class collapse. Structural classes (wall, floor,
ceiling, picture) survive; fine indoor classes (chair, furniture, table, sofa,
tv) largely do not. A NYUv2-fine-tuned segmenter would close much of the gap.

## SemanticKITTI: PolarSeg-input vs GT-input (paper protocol)
Paper KITTI protocol = PolarSeg predicted labels (not GT), seqs 06-10 × first
100 scans, 10 cm voxels, 19-class mIoU vs a voxelized GT grid. PolarNet
(`SemKITTI_PolarSeg.pt`) was run on the P2000 GPU (torch 2.1+cu118, Pascal
sm_61) to produce per-scan raw-ID `.label` predictions; scovox then maps them
via its LEARNING_MAP exactly as it does GT.

| Sequence | GT-input mIoU | PolarSeg mIoU | Δ (PolarSeg−GT) |
|---|---|---|---|
| 06 | 0.8704 | 0.6290 | −0.2415 |
| 07 | 0.8296 | 0.6203 | −0.2093 |
| 08 | 0.8316 | 0.4845 | −0.3471 |
| 09 | 0.8615 | 0.4723 | −0.3892 |
| 10 | 0.8396 | 0.4118 | −0.4278 |
| **mean** | **0.8465 ± 0.017** | **0.5236 ± 0.086** | **−0.3230** |

**Reading:** feeding PolarSeg's predicted labels, scovox reconstructs a 10 cm
semantic voxel map at **0.524 mean mIoU** over seqs 06-10 — a realistic
SemanticKITTI number (PolarNet itself scores ~0.54 val mIoU; the small extra
drop is voxel fusion + the 100-scan subset). GT-as-input (0.847) is the
upper bound: not a paper number, only an integrity check that the pipeline and
GT grids are sound. All 100 frames integrate per sequence (recv=100).

### scovox-node fixes required to run KITTI (system-under-test defects)
The KITTI eval launch was ported from the paper, but three latent issues in the
`scovox` node blocked fast-moving LiDAR — each fixed in the eval launch / node:
1. **Runtime TF-divergence gate** (`runtime_tf_jump_threshold=1.0 m`, absent in
   the paper node): KITTI moves ~1.2 m/frame, so after frame 2 every frame was
   flagged "localization diverged" and gated forever → empty map. Disabled for
   eval (replay feeds exact GT poses).
2. **`evidence_saturation` type**: node declares it `int`; launch passed a
   double → node aborted on start. Pass an int.
3. **Best-effort input subscription + 208 KB OS UDP buffer**: 2.4 MB KITTI
   clouds lost UDP fragments on the best-effort link → ~60 % of frames dropped
   (recv 37/100). Added an `input_reliable_qos` node param (reliable retransmits
   lost fragments) + set it in the eval launch → recv 100/100. Publish timer
   also slowed (`scovox_publish_rate=0.2`, TSDF cloud off) to keep the executor
   free during replay.

## Semantic-mode ablation (dirichlet vs majority_vote vs naive)
Paper ablation of the fusion rule, on the predicted-label inputs where fusion
matters (on clean GT labels all three modes converge). Same scenes/voxels as
above; only `semantic_mode` changes.

**SemanticKITTI (PolarSeg input, seqs 06-10, mean mIoU):**

| dirichlet | majority_vote | naive |
|---|---|---|
| **0.5236** | 0.5223 | 0.4884 |

**SceneNet (Mask2Former soft input, 13 trajs, mean mIoU):**

| dirichlet | majority_vote | naive |
|---|---|---|
| **0.1178** | 0.1155 | 0.1056 |

**Reading:** the ordering **dirichlet ≥ majority_vote > naive** holds on both
datasets. The decisive gap is *any* multi-observation fusion vs naive last-write
(naive loses −0.035 mIoU on KITTI, −0.012 on SceneNet); Dirichlet's edge over
majority_vote is real but small on these single-frame predictors (+0.0013 KITTI,
+0.0023 SceneNet), i.e. Dirichlet's advantage would widen with noisier / more
redundant observations. Full per-seq/per-traj tables:
`results/kitti_semantic_mode_ablation.md`,
`results/scenenet_semantic_mode_ablation.md`.

## Two-robot Dirichlet fusion — DISJOINT coverage (no shared frames)
Paper's fusion experiment used a 50%-overlap trajectory split (robot A=[0,200),
B=[100,300)). This variant uses **disjoint** frame sets — **A=[0,150),
B=[150,300), no shared observations** — so each robot maps only half the scene
and fusion must reconstruct the whole. Two `scovox_node`s (rolling mode) publish
binary maps; a `dscovox_node` fuses both via Dirichlet/Beta-KL consensus (pins
num_classes=14, α₀=0.01). Each map scored vs the full-scene GT.

| | solo_a [0,150) | solo_b [150,300) | fused | Δ fused−max(solo) |
|---|---|---|---|---|
| **mean (n=13)** | 0.2568 | 0.2863 | **0.4051** | **+0.1028** |

Fusion beats the better single robot in **12/13** trajectories. Full per-traj
table: `results/fusion_disjoint/fusion_disjoint_table.md`.

**Reading:** with no shared frames each robot sees only half the trajectory, so
each solo map is missing large regions/classes (low mIoU ≈ 0.26–0.29). Dirichlet
fusion merges the two complementary halves into a full-scene map at **0.405 mIoU
— +0.103 over the better robot**, roughly 1.6× the paper's 50%-overlap gain
(+0.065). The margin is largest where the two halves are genuinely complementary
(0_223 +0.177, 0_682 +0.175) and near-zero where one robot already covered the
scene (0_175 −0.002). This is the stronger multi-robot argument: fusion is not
just refining redundant observations, it is the only way to obtain a complete
map from disjoint agents.

## E1 — Occupancy calibration & aleatoric/epistemic decomposition (NEW: uncertainty round)
First experiment of the uncertainty-assessment plan (`experiments/UNCERTAINTY_PLAN.md`):
is scovox's **split Beta occupancy** substrate *calibrated*, and does its BALD
decomposition (EIG epistemic, E_H aleatoric) behave? Run on the split substrate
(`use_split=true`) the plan concerns, seqs 06-10, 10 cm, deterministic replay
(**recv=100/100 every sequence** — reliable QoS + a no-op binary timer during
replay). The full split map (a_occ/a_free for occupied AND carved-free voxels)
is captured via a `ScovoxMapBinary` snapshot and scored **offline** against a
raycast reference — occupied = voxel with ≥1 LiDAR endpoint, free = ≥3 ray
traversals with 0 hits (GT poses, 5-30 m gate matching the node). All uncertainty
signals are a numpy port of scovox's own closed forms (`uncertainty.cpp`),
self-tested (`experiments/uncertainty/selftest.py`).

| seq | matched voxels | occ% | ECE all | ECE 0.2<p<0.8 | Brier | NLL | AUSE Var | AUSE EIG |
|---|---|---|---|---|---|---|---|---|
| 06 | 1,390,792 | 94 | 0.1051 | 0.2899 | 0.0211 | 0.124 | 0.0749 | 0.1330 |
| 07 | 661,645 | 95 | 0.0846 | 0.2954 | 0.0175 | 0.105 | 0.1503 | 0.2764 |
| 08 | 961,631 | 95 | 0.0999 | 0.3054 | 0.0208 | 0.123 | 0.1056 | 0.1861 |
| 09 | 1,442,869 | 94 | 0.1041 | 0.2931 | 0.0205 | 0.123 | 0.0987 | 0.1828 |
| 10 | 1,372,037 | 94 | 0.1072 | 0.3012 | 0.0215 | 0.128 | 0.0343 | 0.0537 |
| **pooled** | **5,828,974** | **94** | **0.1021** | **0.3004** | **0.0206** | **0.122** | **0.1027** | **0.1881** |

**Reading (three findings, all reproducible across 5 sequences):**
1. **Aggregate ECE (0.102) is a near-tautology; stratify by evidence.** Overall
   accuracy is 0.999 because the reference is defined by the same endpoints that
   drive a_occ (see caveat). Split by evidence s=a_occ+a_free, ECE falls
   monotonically **0.206 (s≤6.4) → 0.023 (s>40)**, and at **intermediate p
   (0.2-0.8) ECE=0.300** — the boundary / thin-structure / under-observed voxels
   where scovox is genuinely **overconfident**. This is the honest calibration
   signal; the headline number hides it. (Confirms the plan-assessment's
   reference-circularity caveat — report stratified, not aggregate.)
2. **EIG (epistemic) decays with evidence** — median EIG 0.079 (s≈5) → 0.004
   (s≈118), i.e. C2's epistemic term behaves. E_H (aleatoric) also falls
   cross-sectionally, but that is *confounded* (high-s voxels are confidently
   occupied, low intrinsic entropy), so the clean aleatoric test needs the
   matched-voxel evidence-scaling design (E3a), not this cross-section.
3. **Posterior variance out-ranks EIG for error detection** — AUSE 0.103 vs
   0.188 pooled (lower = closer to oracle), on every sequence. Total occupancy
   uncertainty beats the epistemic-only signal at flagging misclassifications,
   matching the plan's H2.1 intuition (MI/epistemic is expected to help more for
   out-of-support detection, tested later in E2).

**Caveats.** (a) The reference is a near-surface / visibility proxy: scovox's Beta
grid only leaves the prior in the carve band, so the map represents 12-16 % of the
full free reference — calibration is measured on the ~1.4 M voxels the map
actually estimates. (b) Occupied/free reference and a_occ/a_free share the same
ray endpoints → occupancy ECE is comparative, lean on the stratified/intermediate
rows and (later) paired baselines, not the absolute aggregate. Per-seq detail +
evidence/EIG tables: `results/kitti_e1/<seq>/e1_score.md`; combined table:
`results/kitti_e1/e1_kitti_table.md`.

### SceneNet (13 val trajectories, 5 cm) — RGB-D generalization
E1 rerun on the indoor RGB-D dataset (300 frames/traj, deterministic replay via
reliable depth QoS; **integration ≥99 % — recv=300/300 on 8/13, 297-299/300 on the
other 5**, the recv-plateau detector firing a few frames early, immaterial at this
frame count). Same split substrate, same offline pipeline; the only new component
is a depth-camera raycaster (`build_occ_reference_scenenet.py`) that back-projects
each Euclidean-range depth image into the optical frame and casts the **same rays
scovox integrates**, then reuses the KITTI Amanatides-Woo DDA + scorer verbatim.
Because the reference reconstructs scovox's own observed set densely, **coverage is
94.8-99.9 % of the reference** (vs 12-16 % for LiDAR) — calibration is measured on
essentially the whole map.

| traj | matched voxels | occ% | ECE all | ECE 0.2<p<0.8 | Brier | NLL | AUSE Var | AUSE EIG |
|---|---|---|---|---|---|---|---|---|
| 0_175 | 435,544 | 9 | 0.0321 | 0.2734 | 0.0030 | 0.037 | 0.1050 | 0.6140 |
| 0_178 | 708,307 | 9 | 0.0384 | 0.2507 | 0.0058 | 0.052 | 0.1331 | 0.6249 |
| 0_182 | 1,159,348 | 8 | 0.0332 | 0.2788 | 0.0056 | 0.042 | 0.1696 | 0.7283 |
| 0_223 | 989,728 | 7 | 0.0396 | 0.2675 | 0.0062 | 0.050 | 0.5855 | 0.9140 |
| 0_279 | 1,078,836 | 7 | 0.0353 | 0.2436 | 0.0054 | 0.045 | 0.2361 | 0.7345 |
| 0_485 | 599,427 | 9 | 0.0207 | 0.2700 | 0.0033 | 0.032 | 0.0585 | 0.4359 |
| 0_490 | 590,192 | 9 | 0.0259 | 0.2628 | 0.0047 | 0.039 | 0.0893 | 0.4339 |
| 0_571 | 703,985 | 10 | 0.0262 | 0.2705 | 0.0038 | 0.036 | 0.1190 | 0.6674 |
| 0_682 | 343,934 | 11 | 0.0269 | 0.3095 | 0.0042 | 0.038 | 0.2630 | 1.0757 |
| 0_723 | 459,670 | 11 | 0.0465 | 0.2657 | 0.0090 | 0.061 | 0.1764 | 0.3755 |
| 0_789 | 490,301 | 10 | 0.0482 | 0.2761 | 0.0084 | 0.058 | 0.2089 | 0.7828 |
| 0_867 | 242,905 | 12 | 0.0636 | 0.2816 | 0.0171 | 0.082 | 0.2162 | 0.4727 |
| 0_977 | 561,985 | 9 | 0.0295 | 0.2631 | 0.0052 | 0.042 | 0.1923 | 0.7927 |
| **pooled** | **8,364,162** | **9** | **0.0339** | **0.2566** | **0.0057** | **0.045** | **0.2291** | **0.7116** |

**Reading (pooled, 8.36 M voxels; each finding holds on all 13 trajectories):**
1. **Same calibration signature as KITTI.** Aggregate ECE 0.034 is bulk-dominated
   (occ 9 %); stratified by evidence it falls monotonically **0.100 (s≤24) → 0.010
   (s>103)**, and at **intermediate p (0.2-0.8) ECE=0.257** — the boundary /
   under-observed voxels where scovox is **overconfident** (KITTI: 0.300). The
   boundary-overconfidence is therefore *sensor-independent*, not a LiDAR artefact.
2. **EIG (epistemic) decays with evidence** 0.053 (s≈8) → 0.002 (s≈300); E_H
   (aleatoric) also falls cross-sectionally (0.324 → 0.033) — the same confound as
   KITTI (high-s voxels are confidently occupied), so the clean aleatoric test
   still needs the matched-voxel evidence-scaling design (E3a).
3. **Posterior variance out-ranks EIG for error detection on every trajectory** —
   pooled AUSE 0.229 vs 0.712. Absolute AUSE is high because indoor RGB-D geometry
   is easy (base misclassification only **0.18 %**, a needle-in-haystack for
   sparsification), but the Var<EIG ordering is unambiguous and matches KITTI's H2.1.

**Cross-dataset.** SceneNet is better-calibrated in aggregate than KITTI (ECE 0.034
vs 0.102; Brier 0.006 vs 0.021; NLL 0.045 vs 0.122) — denser evidence,
geometrically-easy indoor depth. What *transfers* is the shape: boundary
overconfidence (intermediate-p ECE ≈ 0.26-0.30) and Var > EIG error-ranking hold on
both sensors. Caveat (b) is even stronger here: at ~99 % coverage the reference and
a_occ share nearly all endpoints, so lean on the stratified / intermediate rows and
(later) paired baselines. Per-traj detail: `results/scenenet_e1/<traj>/e1_score.md`;
table: `results/scenenet_e1/e1_scenenet_table.md`.

### scovox changes for E1 (backward-compatible)
- `semantickitti_eval.launch.py` / `scenenet_eval.launch.py`: added `map_mode`
  (default `persistent`) and `share_rate_hz` (default `0.0`) launch args (scenenet
  also exposes `scovox_publish_rate`). E1 uses `map_mode:=rolling`
  `share_rate_hz:=1.0` to enable the `ScovoxMapBinary` publisher and a timer-owned
  snapshot; the defaults reproduce the paper's persistent runs exactly (no
  regression to the mIoU experiments above).
- No node C++ change is needed for the uncertainty math: the wire format already
  exports the raw per-voxel sufficient statistics, so all functionals are computed
  offline in Python. Deterministic KITTI replay relies on the pre-existing
  `input_reliable_qos` node toggle (reliable PointCloud2 sub → no dropped UDP
  fragments); the SceneNet replay already publishes reliable depth QoS.
- These launch configs + the QoS toggle are pushed on the **`new_experiments`**
  branch of the scovox repo (`git@github.com:kalhansb/scovox.git`).

## Re-run
```
# --- E1 occupancy calibration (SemanticKITTI, split substrate) ---
# Foundation self-test:   python3 experiments/uncertainty/selftest.py
# One seq (replay+cap):   bash experiments/uncertainty/run_kitti_e1.sh 06
# Reference (raycast):    python3 experiments/uncertainty/build_occ_reference.py --sequence 6 -o <out>/occ_ref.npz
# Score:                  python3 experiments/uncertainty/score_e1.py --map <out>/map_bin.npz --ref <out>/occ_ref.npz -o <out>/e1_score.md
# Full batch 06-10:       bash experiments/uncertainty/run_all_e1.sh
# Table:                  python3 experiments/uncertainty/build_e1_table.py
#
# --- E1 occupancy calibration (SceneNet RGB-D, split substrate) ---
# One traj (replay+cap):  bash experiments/uncertainty/run_scenenet_e1.sh 0_223
# Reference (raycast):    python3 experiments/uncertainty/build_occ_reference_scenenet.py --sequence 0_223 -o <out>/occ_ref.npz
# Score:                  python3 experiments/uncertainty/score_e1.py --map <out>/map_bin.npz --ref <out>/occ_ref.npz -o <out>/e1_score.md
# Full batch (13 trajs):  bash experiments/uncertainty/run_all_scenenet_e1.sh
# Table:                  python3 experiments/uncertainty/build_e1_table_scenenet.py
#
# --- Two-robot DISJOINT fusion (SceneNet) ---
# Batch (13 trajs):       bash experiments/fusion/batch_scenenet_fusion.sh
# One traj:               bash experiments/fusion/run_scenenet_fusion.sh <traj> 150 150 <out>
# Table:                  python experiments/fusion/build_fusion_table.py
#
# --- Semantic-mode ablation (both datasets) ---
# Full:                   bash experiments/run_all_ablation.sh
# Tables:                 python experiments/kitti/build_ablation_table.py
#                         python experiments/build_scenenet_ablation_table.py
#
# --- SceneNet ---
# GT-input (fast):        bash experiments/batch_scenenet_gt.sh
# Soft inference (CPU):   python experiments/soft_scenenet/run_mask2former_scenenet.py \
#                           --data_root data/scenenet_val_layout --seqs <csv> --device cpu \
#                           --model facebook/mask2former-swin-small-ade-semantic
# Soft scovox:            bash experiments/batch_scenenet_soft.sh
# Table:                  python experiments/build_soft_vs_gt_table.py
#
# --- SemanticKITTI (GT grids already built under results/kitti_gt/<seq>/gt.npz) ---
# Full pipeline:          bash experiments/kitti/run_all_kitti.sh
#   (PolarSeg inference [GPU] -> GT scovox batch -> PolarSeg scovox batch -> table)
# PolarSeg inference only: bash experiments/kitti/run_polarseg_inference.sh 06,07,08,09,10 cuda:0
# Table:                   python experiments/kitti/build_kitti_table.py
```
