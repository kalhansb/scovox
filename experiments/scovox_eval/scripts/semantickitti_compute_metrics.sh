#!/bin/bash
# Compute Experiment 2 metrics from SemanticKITTI captured npz + GT.
# Run AFTER semantickitti_create_dataset.sh and semantickitti_build_gt.py.
#
# Metrics: mIoU (ignore_label=0, num_classes=20),
#          Semantic ECE (overall + stratified by a_occ evidence),
#          Per-class IoU, timing/memory from run logs
#
# Usage: bash semantickitti_compute_metrics.sh [RESOLUTION]
#   RESOLUTION: voxel size in meters (default: 0.10)

RESOLUTION="${1:-0.10}"
RES_CM=$(python3 -c "print(int(float('$RESOLUTION')*100))")

KITTI_ROOT="$HOME/Projects/HMR_Exploration_Experiment/hmr_exploration_ws/data/semantickitti/dataset"
BASE_RESULTS="$HOME/Projects/HMR_Exploration_Experiment/hmr_exploration_ws/src/robot_sw/distributed_mapping/scovox_eval/results"
RESULTS_ROOT="$BASE_RESULTS/semantickitti_${RES_CM}cm"
WS="$HOME/Projects/HMR_Exploration_Experiment/hmr_exploration_ws"
EVAL_PKG="$WS/src/robot_sw/distributed_mapping/scovox_eval"

export PATH=$(echo $PATH | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
export PYTHONPATH="$EVAL_PKG:$PYTHONPATH"

VARIANTS=(scovox scovox_mv scovox_np)
VARIANT_LABELS=("dirichlet" "majority_vote" "naive")

OUTFILE="$BASE_RESULTS/semantickitti_metrics_${RES_CM}cm.txt"

echo "============================================"
echo "  SemanticKITTI Metrics (Experiment 2) — ${RES_CM}cm"
echo "  Results: $RESULTS_ROOT"
echo "  Output:  $OUTFILE"
echo "============================================"
echo ""

export _RESULTS_ROOT="$RESULTS_ROOT"
export _RES_CM="$RES_CM"

python3 << 'PYEOF' | tee "$OUTFILE"
import numpy as np
import sys
import os

sys.path.insert(0, os.environ.get("PYTHONPATH", "").split(":")[0])

from scipy.spatial import KDTree
from scovox_eval.metrics.miou import compute_miou
from scovox_eval.metrics.semantic_ece import compute_semantic_ece, compute_stratified_semantic_ece

RESULTS_ROOT = os.environ["_RESULTS_ROOT"]
RES_CM = int(os.environ["_RES_CM"])
VARIANTS = ["scovox", "scovox_mv", "scovox_np"]
VARIANT_LABELS = {"scovox": "dirichlet", "scovox_mv": "majority_vote", "scovox_np": "naive"}
NUM_CLASSES = 20
MAX_DIST = float(RES_CM) / 100.0  # match distance = voxel resolution

SEMANTICKITTI_NAMES = {
    0: "unlabeled", 1: "car", 2: "bicycle", 3: "motorcycle", 4: "truck",
    5: "other-vehicle", 6: "person", 7: "bicyclist", 8: "motorcyclist",
    9: "road", 10: "parking", 11: "sidewalk", 12: "other-ground",
    13: "building", 14: "fence", 15: "other-struct", 16: "vegetation",
    17: "trunk", 18: "terrain", 19: "pole",
}

# ─── Load GT ────────────────────────────────────────────────────────
gt_file = os.path.join(RESULTS_ROOT, "gt.npz")
if not os.path.exists(gt_file):
    print(f"ERROR: GT not found at {gt_file}")
    print("Run: python3 semantickitti_build_gt.py <dataset> --resolution <res> -o gt.npz")
    sys.exit(1)

print(f"Loading GT from {gt_file} ...")
gt_data = np.load(gt_file)
gt_pts = gt_data["points"]
gt_labels = gt_data["semantic_class"].astype(int)
gt_tree = KDTree(gt_pts)
print(f"GT: {len(gt_pts)} voxels")

print()
print("=" * 100)
print(f"  SemanticKITTI Seq 08 — {RES_CM}cm — Semantic Mapping Evaluation")
print("=" * 100)

# ─── Per-variant evaluation ─────────────────────────────────────────
all_results = {}

for variant in VARIANTS:
    npz_file = os.path.join(RESULTS_ROOT, f"{variant}.npz")
    log_file = os.path.join(RESULTS_ROOT, f"{variant}_run.log")

    if not os.path.exists(npz_file):
        print(f"\n  {variant}: MISSING {npz_file}")
        continue

    pred_data = np.load(npz_file)
    pred_pts = pred_data["points"]
    pred_labels = pred_data["semantic_class"].astype(int)
    pred_conf = pred_data["semantic_confidence"]
    a_occ = pred_data["a_occ"]

    # NN matching
    dists, indices = gt_tree.query(pred_pts)
    match_mask = dists < MAX_DIST
    matched_pred = pred_labels[match_mask]
    matched_gt = gt_labels[indices[match_mask]]
    matched_conf = pred_conf[match_mask]
    matched_a_occ = a_occ[match_mask]

    n_matched = match_mask.sum()
    n_total = len(pred_pts)

    # ── mIoU (ignore class 0) ──
    miou_result = compute_miou(matched_pred, matched_gt,
                               num_classes=NUM_CLASSES, ignore_label=0)

    # ── ECE (ignore class 0) ──
    valid = matched_gt != 0
    ece_conf = matched_conf[valid]
    ece_correct = (matched_pred[valid] == matched_gt[valid]).astype(float)
    ece_a_occ = matched_a_occ[valid]

    ece_result = compute_semantic_ece(ece_conf, ece_correct)
    strat_result = compute_stratified_semantic_ece(ece_conf, ece_correct, ece_a_occ)

    # ── Timing + memory from log ──
    import re as _re
    frame_times, rss_values = [], []
    integrate_times, publish_times = [], []
    final_voxels, final_bonxai_mb, final_bpv = 0, 0.0, 0
    recv_count = 0
    if os.path.exists(log_file):
        with open(log_file) as f:
            for line in f:
                if "frame_ms=" in line:
                    try:
                        frame_times.append(float(_re.search(r'frame_ms=([\d.]+)', line).group(1)))
                        integrate_times.append(float(_re.search(r'integrate_ms=([\d.]+)', line).group(1)))
                        publish_times.append(float(_re.search(r'publish_ms=([\d.]+)', line).group(1)))
                        rss_values.append(float(_re.search(r'rss_mb=([\d.]+)', line).group(1)))
                        recv_count = int(_re.search(r'recv=(\d+)', line).group(1))
                    except:
                        pass
                if "memUsage" in line and "voxels=" in line:
                    try:
                        final_voxels = int(_re.search(r'voxels=(\d+)', line).group(1))
                        final_bonxai_mb = float(_re.search(r'bonxai_mb=([\d.]+)', line).group(1))
                        final_bpv = int(_re.search(r'bpv=(\d+)', line).group(1))
                    except:
                        pass

    ms_med, ms_p95 = 0.0, 0.0
    fps_med, fps_mean, fps_std = 0.0, 0.0, 0.0
    if frame_times:
        arr = np.array(frame_times)
        ms_med = float(np.median(arr))
        ms_p95 = float(np.percentile(arr, 95))
        fps_arr = 1000.0 / arr
        fps_med = float(np.median(fps_arr))
        fps_mean = float(fps_arr.mean())
        fps_std = float(fps_arr.std())

    all_results[variant] = {
        "n_voxels": n_total,
        "n_matched": n_matched,
        "miou": miou_result["miou"],
        "per_class_iou": miou_result["per_class_iou"],
        "per_class_count": miou_result["per_class_count"],
        "ece": ece_result["semantic_ece"],
        "mce": ece_result["semantic_mce"],
        "mean_conf": float(ece_conf.mean()),
        "mean_acc": float(ece_correct.mean()),
        "strat_ece": strat_result,
        "ms_med": ms_med,
        "ms_p95": ms_p95,
        "bonxai_gb": final_bonxai_mb / 1024.0,
        "fps_med": fps_med,
        "fps_mean": fps_mean,
        "fps_std": fps_std,
        "recv_count": recv_count,
        "final_voxels": final_voxels,
        "final_bpv": final_bpv,
    }

# ─── Summary table ──────────────────────────────────────────────────
print()
print(f"{'Variant':<12s} {'Mode':<14s} {'Voxels':>10s} {'Matched':>10s} {'mIoU':>8s} {'ECE':>8s} {'MCE':>8s} {'Conf':>8s} {'Acc':>8s} {'FPS_med':>8s} {'ms_med':>8s} {'Map_GB':>8s}")
print("─" * 130)

for variant in VARIANTS:
    if variant not in all_results:
        continue
    r = all_results[variant]
    print(f"{variant:<12s} {VARIANT_LABELS[variant]:<14s} {r['n_voxels']:>10d} {r['n_matched']:>10d} "
          f"{r['miou']:>8.4f} {r['ece']:>8.4f} {r['mce']:>8.4f} "
          f"{r['mean_conf']:>8.4f} {r['mean_acc']:>8.4f} "
          f"{r['fps_med']:>8.1f} {r['ms_med']:>8.1f} {r['bonxai_gb']:>8.2f}")

# ─── ECE stratified by evidence (a_occ) ─────────────────────────────
print()
print("=" * 100)
print("  ECE Stratified by Evidence Level (a_occ)")
print("=" * 100)
print(f"{'Variant':<12s} {'Stratum':<14s} {'N voxels':>10s} {'ECE':>8s} {'MCE':>8s}")
print("─" * 54)

for variant in VARIANTS:
    if variant not in all_results:
        continue
    strat = all_results[variant]["strat_ece"]
    for sname in ["low (<5)", "medium (5-20)", "high (>20)"]:
        if sname in strat:
            s = strat[sname]
            print(f"{variant:<12s} {sname:<14s} {s['n_voxels']:>10d} "
                  f"{s['semantic_ece']:>8.4f} {s.get('semantic_mce', float('nan')):>8.4f}")
    print()

# ─── Per-class IoU ──────────────────────────────────────────────────
print("=" * 100)
print("  Per-Class IoU")
print("=" * 100)

header = f"{'Class':<16s}"
for variant in VARIANTS:
    if variant in all_results:
        header += f" {variant:>12s}"
header += f" {'GT voxels':>12s}"
print(header)
print("─" * len(header))

for c in range(1, NUM_CLASSES):
    name = SEMANTICKITTI_NAMES.get(c, f"class_{c}")
    line = f"{name:<16s}"
    gt_count = 0
    for variant in VARIANTS:
        if variant in all_results:
            iou = all_results[variant]["per_class_iou"][c]
            gt_count = all_results[variant]["per_class_count"][c]
            if np.isnan(iou):
                line += f" {'N/A':>12s}"
            else:
                line += f" {iou:>12.4f}"
    line += f" {gt_count:>12d}"
    print(line)

# ─── Runtime & Memory ───────────────────────────────────────────────
print()
print("=" * 100)
print("  Runtime & Memory (vs published SLIM-VDB values)")
print("=" * 100)
print()
print("SCovox variants (CPU-only, Bonxai sparse voxel grid):")
print(f"{'Variant':<14s} {'Mode':<16s} {'FPS (med)':>10s} {'FPS (mean+/-std)':>18s} {'ms_med':>8s} {'ms_p95':>8s} {'Map (GB)':>10s} {'Voxels':>12s} {'bpv':>5s}")
print("─" * 110)
for variant in VARIANTS:
    if variant not in all_results:
        continue
    r = all_results[variant]
    fps_str = f"{r['fps_mean']:.1f}+/-{r['fps_std']:.1f}"
    print(f"{variant:<14s} {VARIANT_LABELS[variant]:<16s} {r['fps_med']:>10.1f} {fps_str:>18s} "
          f"{r['ms_med']:>8.1f} {r['ms_p95']:>8.1f} {r['bonxai_gb']:>10.2f} "
          f"{r['final_voxels']:>12,d} {r['final_bpv']:>5d}")

print()
print("Published reference (GPU, SLIM-VDB RA-L 2026, SemanticKITTI 10cm):")
print(f"{'System':<14s} {'Type':<16s} {'FPS (mean+/-std)':>18s} {'Memory (GB)':>12s}")
print("─" * 64)
for name, typ, fps, mem in [
    ("SLIM-VDB*",  "Non-Bayesian",    "6.25+/-1.12",  "0.67+/-0.05"),
    ("SLIM-VDB_C", "Bayesian (Dir.)", "1.69+/-0.69",  "2.60+/-0.80"),
    ("ConvBKI",    "Bayesian (Dir.)", "0.53+/-0.26", "27.73+/-0.20"),
    ("SEE-CSOM",   "Bayesian",        "0.10+/-0.01", "10.98+/-0.35"),
]:
    print(f"{name:<14s} {typ:<16s} {fps:>18s} {mem:>12s}")

print()
print("=== DONE ===")

PYEOF

echo ""
echo "Results saved to: $OUTFILE"
