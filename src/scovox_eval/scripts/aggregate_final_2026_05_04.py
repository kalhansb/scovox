#!/usr/bin/env python3
"""Aggregate the 2026-05-04 final-run results across all KITTI sequences
(06–10) and Replica scenes (room0..2, office0..4) for three systems:

  - SCovox hard-label (argmax PNG / .label one-hot path)
  - SCovox soft-prob (dense .topk Dirichlet path)
  - SLIM-VDB CLOSED (existing outputs, same M2F / PolarSeg input)

Emits two CSVs:
  results/final_2026_05_04_kitti.csv
  results/final_2026_05_04_replica.csv

with columns: dataset, scene_or_seq, system, miou, fps, grid_mb,
voxels, chamfer_cm (Replica only), f_at_5cm (Replica only).
"""
import argparse
import collections
import csv
import re
import statistics
from pathlib import Path

WS = Path("/home/kalhan/projects/HMR_Exploration_Experiment/hmr_exploration_ws")
EVAL_PKG = WS / "src/robot_sw/distributed_mapping/scovox_eval"
KITTI_RESULTS = EVAL_PKG / "results/final_kitti_2026_05_04"
REPLICA_RESULTS = EVAL_PKG / "results/final_replica_2026_05_04"
SLIMVDB_KITTI = WS / "third_party_sw/slim_vdb/outputs/kitti"
SLIMVDB_REPLICA = WS / "third_party_sw/slim_vdb/outputs/replica_m2f"

KITTI_SEQS = ["06", "07", "08", "09", "10"]
SCENES = ["room0", "room1", "room2",
          "office0", "office1", "office2", "office3", "office4"]


def parse_scovox_log(log_path: Path):
    """Extract mean FPS over last 50 frame_ms entries and final bonxai_mb."""
    if not log_path.exists():
        return None, None
    fps_vals = []
    grid_mb = None
    with log_path.open() as f:
        for line in f:
            m = re.search(r"frame_ms=([\d.]+)", line)
            if m:
                fps_vals.append(float(m.group(1)))
            m = re.search(r"bonxai_mb=([\d.]+)", line)
            if m:
                grid_mb = float(m.group(1))
    if not fps_vals:
        return None, grid_mb
    last = fps_vals[-50:] if len(fps_vals) >= 50 else fps_vals
    avg = sum(last) / len(last)
    fps = 1000.0 / avg
    return fps, grid_mb


def parse_slimvdb_log(log_path: Path):
    """Mean integrate_ms and final grid memory from per-frame TIMING line."""
    if not log_path.exists():
        return None, None
    int_ms = []
    last_grid = None
    with log_path.open() as f:
        for line in f:
            if "TIMING idx=" not in line:
                continue
            m = re.search(r"integrate_ms=([\d.]+)", line)
            if m:
                int_ms.append(float(m.group(1)))
            tsdf = re.search(r"vdb_tsdf_mb=([\d.]+)", line)
            sem = re.search(r"vdb_sem_mb=([\d.]+)", line)
            if tsdf and sem:
                last_grid = float(tsdf.group(1)) + float(sem.group(1))
    if not int_ms:
        return None, last_grid
    last = int_ms[-50:] if len(int_ms) >= 50 else int_ms
    avg = sum(last) / len(last)
    fps = 1000.0 / avg
    return fps, last_grid


def kitti_scovox_miou_single(npz_path: Path, gt_path: Path):
    """Run eval_ablations_kitti_seq08 against a one-cell dir; parse stdout."""
    import subprocess, tempfile, shutil, os
    if not npz_path.exists() or not gt_path.exists():
        return None, None
    tmp = tempfile.mkdtemp(prefix="agg_kitti_")
    try:
        cell = Path(tmp) / "cell"
        cell.mkdir()
        shutil.copy(npz_path, cell / "scovox.npz")
        out = subprocess.run(
            ["python3",
             str(EVAL_PKG / "scripts/eval_ablations_kitti_seq08.py"),
             "--gt", str(gt_path), "--results_root", tmp],
            capture_output=True, text=True, timeout=600)
        m = re.search(r"mIoU=([\d.]+)", out.stdout)
        miou = float(m.group(1)) if m else None
        m2 = re.search(r"pred=(\d+)", out.stdout)
        n_voxels = int(m2.group(1)) if m2 else None
        return miou, n_voxels
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def replica_scovox_metrics(npz_path: Path, scene: str):
    """Returns (miou, chamfer_cm, f_at_5cm, n_voxels) — calls eval_ablations_replica_room0
    against a one-cell dir."""
    import subprocess, tempfile, shutil
    if not npz_path.exists():
        return None, None, None, None
    tmp = tempfile.mkdtemp(prefix="agg_rep_")
    try:
        cell = Path(tmp) / "cell"
        cell.mkdir()
        shutil.copy(npz_path, cell / "scovox.npz")
        out = subprocess.run(
            ["python3",
             str(EVAL_PKG / "scripts/eval_ablations_replica_room0.py"),
             "--replica_root", str(WS / "data/replica_niceslam"),
             "--results_root", tmp, "--scene", scene],
            capture_output=True, text=True, timeout=600)
        m = re.search(r"mIoU=([\d.]+).*chamfer=([\d.]+)cm.*F@5=([\d.]+).*pred=(\d+)",
                      out.stdout)
        if m:
            return (float(m.group(1)), float(m.group(2)),
                    float(m.group(3)), int(m.group(4)))
        return None, None, None, None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def slimvdb_kitti_miou(seq: str):
    """Mean per-class IoU (skip class 0 unlabeled + NaN classes) from
    miou_per_class.csv."""
    csv_path = SLIMVDB_KITTI / "miou_per_class.csv"
    if not csv_path.exists():
        return None
    ious = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if row["sequence"] != seq:
                continue
            if int(row["class_id"]) == 0:
                continue
            if row["iou"] == "":
                continue
            ious.append(float(row["iou"]))
    if not ious:
        return None
    return sum(ious) / len(ious)


def slimvdb_replica_miou(scene: str):
    """Pull per-scene mIoU from /tmp/slimvdb_replica_m2f_18cls.csv if exists,
    else fall back to known room0 = 0.1293."""
    fallback = {"room0": 0.1293}
    cache = Path("/tmp/slimvdb_replica_m2f_18cls.csv")
    if cache.exists():
        with cache.open() as f:
            for row in csv.DictReader(f):
                if row.get("scene") == scene:
                    return float(row["miou"])
    return fallback.get(scene)


# ── KITTI table ───────────────────────────────────────────────────────
def build_kitti_table():
    rows = []
    for seq in KITTI_SEQS:
        gt = EVAL_PKG / f"results/semantickitti_polarseg_10cm/{seq}/gt.npz"
        for mode in ("hard", "soft"):
            cell = KITTI_RESULTS / mode / seq
            miou, voxels = kitti_scovox_miou_single(cell / "scovox.npz", gt)
            fps, grid_mb = parse_scovox_log(cell / "scovox_run.log")
            rows.append({
                "dataset": "kitti", "scene_or_seq": seq,
                "system": f"scovox_{mode}",
                "miou": miou, "fps": fps,
                "grid_mb": grid_mb, "voxels": voxels,
            })
        # SLIM-VDB
        fps, grid_mb = parse_slimvdb_log(SLIMVDB_KITTI / seq / "timing.log")
        miou = slimvdb_kitti_miou(seq)
        rows.append({
            "dataset": "kitti", "scene_or_seq": seq, "system": "slimvdb",
            "miou": miou, "fps": fps, "grid_mb": grid_mb, "voxels": None,
        })
    return rows


# ── Replica table ─────────────────────────────────────────────────────
def build_replica_table():
    rows = []
    for scene in SCENES:
        for mode in ("hard", "soft"):
            cell = REPLICA_RESULTS / mode / scene
            miou, ch, f5, voxels = replica_scovox_metrics(cell / "scovox.npz", scene)
            fps, grid_mb = parse_scovox_log(cell / "scovox_run.log")
            rows.append({
                "dataset": "replica", "scene_or_seq": scene,
                "system": f"scovox_{mode}",
                "miou": miou, "fps": fps,
                "grid_mb": grid_mb, "voxels": voxels,
                "chamfer_cm": ch, "f_at_5cm": f5,
            })
        # SLIM-VDB
        fps, grid_mb = parse_slimvdb_log(SLIMVDB_REPLICA / scene / "timing.log")
        miou = slimvdb_replica_miou(scene)
        rows.append({
            "dataset": "replica", "scene_or_seq": scene, "system": "slimvdb",
            "miou": miou, "fps": fps, "grid_mb": grid_mb, "voxels": None,
            "chamfer_cm": None, "f_at_5cm": None,
        })
    return rows


def write_csv(rows, out_path: Path):
    if not rows:
        return
    fields = sorted({k for r in rows for k in r.keys()})
    # Stable column order
    head = ["dataset", "scene_or_seq", "system", "miou", "fps",
            "grid_mb", "voxels"]
    extras = [f for f in fields if f not in head]
    fields = head + extras
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {out_path}")


def print_summary(rows, dataset_name):
    by_sys = collections.defaultdict(list)
    by_sys_grid = collections.defaultdict(list)
    by_sys_fps = collections.defaultdict(list)
    for r in rows:
        if r["miou"] is not None:
            by_sys[r["system"]].append(r["miou"])
        if r["grid_mb"] is not None:
            by_sys_grid[r["system"]].append(r["grid_mb"])
        if r["fps"] is not None:
            by_sys_fps[r["system"]].append(r["fps"])
    print(f"\n=== {dataset_name} mean over {len(SCENES if dataset_name=='Replica' else KITTI_SEQS)} cells ===")
    for sys in ("scovox_hard", "scovox_soft", "slimvdb"):
        m = by_sys[sys]
        g = by_sys_grid[sys]
        fps = by_sys_fps[sys]
        if m:
            print(f"  {sys:14s} mIoU = {statistics.mean(m):.4f} "
                  f"(±{statistics.stdev(m):.4f} over {len(m)})")
        if fps:
            print(f"  {sys:14s} FPS  = {statistics.mean(fps):.2f}")
        if g:
            print(f"  {sys:14s} grid = {statistics.mean(g):.1f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path,
                    default=EVAL_PKG / "results")
    args = ap.parse_args()

    print("[KITTI] aggregating…")
    kitti = build_kitti_table()
    write_csv(kitti, args.out_dir / "final_2026_05_04_kitti.csv")
    print_summary(kitti, "KITTI")

    print("\n[Replica] aggregating…")
    replica = build_replica_table()
    write_csv(replica, args.out_dir / "final_2026_05_04_replica.csv")
    print_summary(replica, "Replica")


if __name__ == "__main__":
    main()
