# SCovox

Sparse-Covariance Voxel mapping with a unified per-voxel Dirichlet
over `{top-K classes, OTHER, FREE}` — semantic occupancy mapping with
calibrated uncertainty, multi-robot fusion, and edge-device runtime.

This repository accompanies the paper *"SCovox: Hit-Driven Semantic
Voxel Mapping with Unified-Dirichlet Fusion."*  It contains the
code and configurations used to produce every number in the paper's
results section and the Jetson Nano edge-feasibility experiment.

## Headline results

| Result | Value | Source |
|--------|-------|--------|
| K_TOP Pareto knee (per-voxel memory) | K=2 matches K=19 ±0.001 mIoU, 6.2× less memory | [STEP8 §2 / Phase 1](docs/STEP8_EXPERIMENT_RESULTS_2026_05_14.md) |
| SCovox vs SLIM-VDB, SceneNet h2h (n=13, sdf_trunc=0.10) | Δ mIoU **+0.092 ± 0.016**, 13/13 wins | [STEP8 §5 / Phase 4](docs/STEP8_EXPERIMENT_RESULTS_2026_05_14.md) |
| SCovox vs SLIM-VDB, KITTI h2h (n=5, PolarSeg soft) | Δ mIoU **+0.096 ± 0.018**, 5/5 wins | [STEP8 §5 / Phase 4](docs/STEP8_EXPERIMENT_RESULTS_2026_05_14.md) |
| Two-robot Dirichlet fusion (SceneNet, n=13) | Δ mIoU **+0.071**, F1@5cm +0.013, Chamfer −1.4 cm | [STEP8 §4 / Phase 3](docs/STEP8_EXPERIMENT_RESULTS_2026_05_14.md) |
| Jetson Nano 4 GB edge runtime | 0.42 Hz sustained, 1.22 Hz integrate-only | [jetbot_experiment_2026_05_21.md](docs/jetbot_experiment_2026_05_21.md) |

The gap-mechanism analysis (SceneNet sweep over SLIM-VDB's
`sdf_trunc`) and per-voxel memory ratios are in the same Step 8 doc.

## Layout

```
src/
  scovox_core/      Voxel types, Dirichlet semantics, ray casting (C++17, zero ROS deps)
  scovox_mapping/   ROS 2 nodes: scovox_mapping_node, dscovox_mapping_node
  scovox_msgs/      BinaryMap v3 wire format + service definitions
  scovox_eval/      Replay drivers, batch scripts, evaluation tooling, Jetson runner

docs/
  STEP8_EXPERIMENT_RESULTS_2026_05_14.md   Full Step-8 results write-up
  jetbot_experiment_2026_05_21.md          Jetson Nano edge-feasibility report
  NEW_EXPERIMENT_PLAN.md                   Phase 0–7 protocol the results follow
  METHODS.md                               Math + per-voxel storage layout
  design/unified_dirichlet_design_2026_05_13.md   v3 Dirichlet design doc
```

## Build

ROS 2 Humble (or newer), Ubuntu 22.04, GCC ≥ 11.

```bash
mkdir -p ~/scovox_ws/src && cd ~/scovox_ws/src
git clone https://github.com/<owner>/scovox.git .
cd ..
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

Release builds are mandatory — debug builds inflate timing numbers
by ~3–4× and bias any wall-clock measurement.

Runtime dependencies: Eigen 3.4+, lz4, standard ROS 2 stack.
Evaluation scripts additionally require `numpy`, `scipy`, and
`open3d` (for Chamfer / F-score computations).

## Reproducing the paper

The reproduction workflow follows [docs/NEW_EXPERIMENT_PLAN.md](docs/NEW_EXPERIMENT_PLAN.md).
Each phase has a driver shell script under `src/scovox_eval/scripts/`:

| Phase | Question | Driver |
|-------|----------|--------|
| 0     | Substrate smoke test                     | `phase0_smoke_post_semdir.sh` |
| 1     | K_TOP Pareto sweep (KITTI + SceneNet)    | `phase1_ktop_sweep.sh`, `phase1_kitti_soft_sweep.sh` |
| 2     | Dirichlet vs no-prior / majority-vote    | `phase2_component_ablation.sh` |
| 2.5   | Publish- and admission-gate sweeps       | `phase2_5_gate_threshold_sweep.sh`, `phase2_5_v2_admission_gate_sweep.sh` |
| 3     | Two-robot Dirichlet fusion (SceneNet)    | `phase3_scenenet_fusion_batch.sh` |
| 4     | Head-to-head vs SLIM-VDB                 | `eval_scovox_kitti_miou.py`, `eval_slimvdb_kitti_miou.py`, scenenet equivalents |
| 5     | Per-voxel memory accounting              | inline in `eval_slimvdb_runtime_memory.py` |

The scripts assume KITTI seq 06–10, SceneNet-RGBD val split, and
PolarSeg `.topk` soft-probability outputs are staged at the paths
documented at the top of each script.

### Jetson Nano edge experiment

Two-machine setup with a Jetson Nano 4 GB running scovox + dscovox
and the laptop replaying SceneNet trajectories.  See
[`src/scovox_eval/jetbot/README.md`](src/scovox_eval/jetbot/README.md)
for DDS configuration and the runner scripts on both sides.

## Determinism and run-to-run variance

* SCovox is bit-deterministic across x86_64 and aarch64 when no DDS
  messages are dropped (verified — 44,808 / 66,197 voxels exact
  match between desktop and Jetson on SceneNet 0_223).
* SceneNet two-robot fusion has higher run-to-run variance than the
  single-robot KITTI pipeline; published Phase 3 deltas average
  over 13 trajectories at 0.25 Hz replay (4 Hz introduces
  `message_filters` drop bias).

## Citation

If you use this code, please cite the paper (BibTeX block to be added
once the camera-ready DOI is assigned).

## License

Apache-2.0.  See [LICENSE](LICENSE).
