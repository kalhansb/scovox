#!/bin/bash
# Compute all Exp 1 metrics from captured npz files + log files.
# Run AFTER replica_create_dataset.sh completes.
#
# Metrics: F-score @5cm, Chamfer-L1, Normal consistency,
#          Integration time (median/P95), Peak memory,
#          Per-component timing breakdown, Wilcoxon signed-rank test
#
# Usage: bash replica_compute_metrics.sh [RESOLUTION]
#   RESOLUTION: voxel size in meters (default: 0.05)

RESOLUTION="${1:-0.05}"
RES_CM=$(python3 -c "print(int(float('$RESOLUTION')*100))")

NICESLAM_ROOT="$HOME/projects/HMR_Exploration_Experiment/hmr_exploration_ws/data/replica_niceslam"
BASE_RESULTS="$HOME/Projects/HMR_Exploration_Experiment/hmr_exploration_ws/src/robot_sw/distributed_mapping/scovox_eval/results"
RESULTS_ROOT="$BASE_RESULTS/replica_${RES_CM}cm"
WS="$HOME/Projects/HMR_Exploration_Experiment/hmr_exploration_ws"
EVAL_PKG="$WS/src/robot_sw/distributed_mapping/scovox_eval"

export PATH=$(echo $PATH | tr ':' '\n' | grep -v miniconda | tr '\n' ':' | sed 's/:$//')
export PYTHONPATH="$EVAL_PKG:$PYTHONPATH"

SCENES=(room0 room1 room2 office0 office1 office2 office3 office4)
VARIANTS=(scovox covox logodds)

echo "============================================"
echo "  Replica Metrics (Experiment 1) — ${RES_CM}cm"
echo "  Results: $RESULTS_ROOT"
echo "============================================"
echo ""

export _RESULTS_ROOT="$RESULTS_ROOT"
export _NICESLAM_ROOT="$NICESLAM_ROOT"
export _RES_CM="$RES_CM"

python3 << 'PYEOF'
import numpy as np
import sys
import os

sys.path.insert(0, os.environ.get("PYTHONPATH", "").split(":")[0])

from scovox_eval.metrics.fscore import compute_fscore
from scovox_eval.metrics.chamfer import compute_chamfer_l1
from scovox_eval.metrics.normal_consistency import compute_normal_consistency

RESULTS_ROOT = os.environ["_RESULTS_ROOT"]
NICESLAM_ROOT = os.environ["_NICESLAM_ROOT"]
RES_CM = int(os.environ["_RES_CM"])
SCENES = ["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]
VARIANTS = ["scovox", "covox", "logodds"]

# ─── Per-scene results ───────────────────────────────────────────────
header = f"{'Scene':<10} {'Variant':<8} {'Voxels':>7} {'F@5cm':>7} {'Chamf':>7} {'NormC':>7} {'ms_med':>7} {'tf_ms':>7} {'integ':>7} {'pub_ms':>7} {'ms_p95':>7} {'Mem_MB':>7}"
print(header)
print("─" * len(header))

all_results = {v: {"fscore": [], "chamfer": [], "normal": [], "ms_med": [], "ms_p95": [], "tf_med": [], "integrate_med": [], "publish_med": [], "mem": [], "scenes": []} for v in VARIANTS}

for scene in SCENES:
    gt_file = f"{RESULTS_ROOT}/{scene}/gt_sampled_ros.npz"
    gt_mesh = f"{NICESLAM_ROOT}/{scene}/mesh.ply"

    if not os.path.exists(gt_file):
        print(f"{scene:<10} {'':>8} MISSING GT")
        continue

    gt_pts = np.load(gt_file)["points"]

    # Load GT mesh for normal consistency (if available)
    gt_mesh_pts = None
    gt_mesh_normals = None
    try:
        import trimesh
        mesh = trimesh.load(gt_mesh)
        gt_mesh_pts = np.asarray(mesh.vertices, dtype=np.float32)
        if hasattr(mesh, "vertex_normals") and len(mesh.vertex_normals) > 0:
            gt_mesh_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    except:
        pass

    for variant in VARIANTS:
        npz_file = f"{RESULTS_ROOT}/{scene}/{variant}.npz"
        log_file = f"{RESULTS_ROOT}/{scene}/{variant}_run.log"

        if not os.path.exists(npz_file):
            print(f"{scene:<10} {variant:<8} MISSING")
            continue

        pred = np.load(npz_file)
        pred_pts = pred["points"]
        n_voxels = len(pred_pts)

        # F-score @5cm
        fs = compute_fscore(pred_pts, gt_pts, 0.05)
        fscore = fs["fscore"]

        # Chamfer-L1
        ch = compute_chamfer_l1(pred_pts, gt_pts)
        chamfer_cm = ch["chamfer_l1"] * 100  # m → cm

        # Normal consistency
        norm_c = 0.0
        if gt_mesh_pts is not None:
            try:
                nc = compute_normal_consistency(
                    pred_pts[::10],  # subsample for speed
                    gt_mesh_pts,
                    gt_mesh_normals,
                    k=20,
                )
                norm_c = nc["normal_consistency"]
            except:
                norm_c = -1.0

        # Timing + memory from log file
        ms_med, ms_p95, peak_mem = 0.0, 0.0, 0.0
        tf_med, integrate_med, publish_med = 0.0, 0.0, 0.0
        if os.path.exists(log_file):
            frame_times = []
            tf_times = []
            integrate_times = []
            publish_times = []
            mem_values = []
            with open(log_file) as f:
                for line in f:
                    if "frame_ms=" in line:
                        try:
                            ms_str = line.split("frame_ms=")[1].split()[0]
                            frame_times.append(float(ms_str))
                        except:
                            pass
                    if "tf_ms=" in line:
                        try:
                            tf_times.append(float(line.split("tf_ms=")[1].split()[0]))
                        except:
                            pass
                    if "integrate_ms=" in line:
                        try:
                            integrate_times.append(float(line.split("integrate_ms=")[1].split()[0]))
                        except:
                            pass
                    if "publish_ms=" in line:
                        try:
                            publish_times.append(float(line.split("publish_ms=")[1].split()[0]))
                        except:
                            pass
                    if "mem_mb=" in line:
                        try:
                            mem_str = line.split("mem_mb=")[1].split()[0]
                            mem_values.append(float(mem_str))
                        except:
                            pass
            if frame_times:
                arr = np.array(frame_times)
                ms_med = float(np.median(arr))
                ms_p95 = float(np.percentile(arr, 95))
            if tf_times:
                tf_med = float(np.median(tf_times))
            if integrate_times:
                integrate_med = float(np.median(integrate_times))
            if publish_times:
                publish_med = float(np.median(publish_times))
            if mem_values:
                peak_mem = max(mem_values)

        print(f"{scene:<10} {variant:<8} {n_voxels:7d} {fscore:7.4f} {chamfer_cm:7.2f} {norm_c:7.4f} {ms_med:7.1f} {tf_med:7.1f} {integrate_med:7.1f} {publish_med:7.1f} {ms_p95:7.1f} {peak_mem:7.1f}")

        all_results[variant]["fscore"].append(fscore)
        all_results[variant]["chamfer"].append(chamfer_cm)
        if norm_c >= 0:
            all_results[variant]["normal"].append(norm_c)
        all_results[variant]["ms_med"].append(ms_med)
        all_results[variant]["ms_p95"].append(ms_p95)
        all_results[variant]["tf_med"].append(tf_med)
        all_results[variant]["integrate_med"].append(integrate_med)
        all_results[variant]["publish_med"].append(publish_med)
        all_results[variant]["mem"].append(peak_mem)
        all_results[variant]["scenes"].append(scene)

# ─── Summary ─────────────────────────────────────────────────────────
print()
print("=" * len(header))
print("SUMMARY (mean +/- std)")
print("=" * len(header))
print(f"{'Variant':<10} {'N':>3} {'F@5cm':>14} {'Chamf(cm)':>14} {'NormC':>14} {'ms_med':>14} {'ms_p95':>14} {'Mem_MB':>14}")
print("─" * 95)

for variant in VARIANTS:
    r = all_results[variant]
    n = len(r["fscore"])
    def fmt(vals):
        if not vals:
            return "     N/A      "
        a = np.array(vals)
        return f"{a.mean():.4f}+/-{a.std():.4f}"

    print(f"{variant:<10} {n:3d} {fmt(r['fscore']):>14} {fmt(r['chamfer']):>14} {fmt(r['normal']):>14} {fmt(r['ms_med']):>14} {fmt(r['ms_p95']):>14} {fmt(r['mem']):>14}")

# ─── Per-component timing breakdown ─────────────────────────────────
has_breakdown = any(all_results[v]["integrate_med"] and any(t > 0 for t in all_results[v]["integrate_med"]) for v in VARIANTS)
if has_breakdown:
    print()
    print("=" * 70)
    print("TIMING BREAKDOWN (median ms, mean across scenes)")
    print("=" * 70)
    print(f"{'Variant':<10} {'tf_ms':>8} {'raycast':>10} {'semantic':>10} {'publish':>10} {'total':>10}")
    print("─" * 58)
    # CoVox integrate_ms = pure raycast+occupancy (no semantics)
    # SCovox integrate_ms = raycast+occupancy+semantics
    # Semantic overhead = SCovox.integrate - CoVox.integrate (on shared scenes)
    covox_integ_by_scene = {}
    if "covox" in all_results and all_results["covox"]["integrate_med"]:
        for i, sc in enumerate(all_results["covox"]["scenes"]):
            covox_integ_by_scene[sc] = all_results["covox"]["integrate_med"][i]

    for variant in VARIANTS:
        r = all_results[variant]
        if not r["tf_med"]:
            continue
        tf = np.mean(r["tf_med"])
        integ = np.mean(r["integrate_med"])
        pub = np.mean(r["publish_med"])
        tot = np.mean(r["ms_med"])

        # Derive semantic overhead for SCovox
        sem = 0.0
        raycast = integ
        if variant == "scovox" and covox_integ_by_scene:
            sem_vals = []
            ray_vals = []
            for i, sc in enumerate(r["scenes"]):
                if sc in covox_integ_by_scene:
                    sem_vals.append(r["integrate_med"][i] - covox_integ_by_scene[sc])
                    ray_vals.append(covox_integ_by_scene[sc])
            if sem_vals:
                sem = np.mean(sem_vals)
                raycast = np.mean(ray_vals)

        if variant == "scovox" and sem > 0:
            print(f"{variant:<10} {tf:8.1f} {raycast:10.1f} {sem:10.1f} {pub:10.1f} {tot:10.1f}")
        else:
            print(f"{variant:<10} {tf:8.1f} {integ:10.1f} {'—':>10} {pub:10.1f} {tot:10.1f}")

# ─── Pairwise comparison + Wilcoxon signed-rank test ─────────────────
print()
from scipy.stats import wilcoxon

baselines = ["covox", "logodds"]
for bl in baselines:
    if all_results["scovox"]["fscore"] and all_results[bl]["fscore"]:
        # Compare on shared scenes only
        s_scenes = set(all_results["scovox"]["scenes"])
        b_scenes = set(all_results[bl]["scenes"])
        shared = sorted(s_scenes & b_scenes)
        if not shared:
            continue
        s_fs = [all_results["scovox"]["fscore"][all_results["scovox"]["scenes"].index(sc)] for sc in shared]
        b_fs = [all_results[bl]["fscore"][all_results[bl]["scenes"].index(sc)] for sc in shared]
        s_mean, b_mean = np.mean(s_fs), np.mean(b_fs)
        diff_pct = (s_mean - b_mean) / b_mean * 100
        print(f"SCovox vs {bl:<8} ({len(shared)} shared scenes): F@5cm {s_mean:.4f} vs {b_mean:.4f} ({diff_pct:+.1f}%)")

# ─── Wilcoxon signed-rank tests ─────────────────────────────────────
print()
print("=" * 70)
print("WILCOXON SIGNED-RANK TESTS (paired across scenes)")
print("=" * 70)

def get_paired(metric, v1, v2):
    """Get paired values for shared scenes."""
    s1 = set(all_results[v1]["scenes"])
    s2 = set(all_results[v2]["scenes"])
    shared = sorted(s1 & s2)
    if len(shared) < 5:
        return None, None, shared
    vals1 = [all_results[v1][metric][all_results[v1]["scenes"].index(sc)] for sc in shared]
    vals2 = [all_results[v2][metric][all_results[v2]["scenes"].index(sc)] for sc in shared]
    return vals1, vals2, shared

pairs = [("scovox", "logodds"), ("scovox", "covox"), ("covox", "logodds")]
metrics_to_test = [("fscore", "F@5cm", True), ("chamfer", "Chamfer(cm)", False)]

for metric, label, higher_better in metrics_to_test:
    print(f"\n  {label}:")
    for v1, v2 in pairs:
        vals1, vals2, shared = get_paired(metric, v1, v2)
        if vals1 is None:
            print(f"    {v1} vs {v2}: <5 shared scenes, skipped")
            continue
        diffs = np.array(vals1) - np.array(vals2)
        if np.all(diffs == 0):
            print(f"    {v1} vs {v2}: identical (all diffs = 0)")
            continue
        try:
            stat, p = wilcoxon(vals1, vals2)
            m1, m2 = np.mean(vals1), np.mean(vals2)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"    {v1} vs {v2}: {m1:.4f} vs {m2:.4f}, W={stat:.0f}, p={p:.4f} {sig}  (n={len(shared)})")
        except Exception as e:
            print(f"    {v1} vs {v2}: test failed ({e})")

PYEOF

echo ""
echo "=== DONE ==="
