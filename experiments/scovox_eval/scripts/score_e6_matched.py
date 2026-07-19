#!/usr/bin/env python3
"""E6.1 — Matched-config head-to-head scoring.

For each cell (8 Replica scenes + 5 KITTI seqs):
  - mIoU(SCovox)        from results/matched_config_2026_05_08/<dataset>/<cell>/scovox.npz
  - mIoU(SLIM-VDB)      from third_party_sw/slim_vdb/outputs/<dataset>/<cell>/voxels.bin
  - FPS(SCovox)         parsed from scovox_run.log per-frame integrate_ms
  - FPS(SLIM-VDB)       parsed from third_party_sw/slim_vdb/outputs/<cell>/timing.log
  - grid_mb(SCovox)     parsed from scovox_run.log [memUsage] last bonxai_mb
  - grid_mb(SLIM-VDB)   parsed from timing.log final TIMING idx= line
                        (vdb_tsdf_mb + vdb_sem_mb)
  - peak_rss_mb(SCovox) parsed from scovox_run.log

Output: results/matched_config_2026_05_08/summary.csv (one row per cell).
Also prints per-locus table and aggregate mean ratios.
"""
from __future__ import annotations
import argparse, csv, json, re, statistics, subprocess, sys
from pathlib import Path

WS = Path(__file__).resolve().parents[5]
EVAL_PKG = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
RESULTS_ROOT = EVAL_PKG / "results" / "matched_config_2026_05_08"
SLIM_OUT = WS / "third_party_sw" / "slim_vdb" / "outputs"
KITTI_YAML = WS / "src" / "sem_seg_pipeline" / "polarseg" / "semantic-kitti.yaml"
REPLICA_ROOT = WS / "data" / "replica_niceslam"
KITTI_ROOT = WS / "data" / "semantickitti" / "dataset"

REPLICA_SCENES = ["room0", "room1", "room2",
                  "office0", "office1", "office2", "office3", "office4"]
KITTI_SEQS = ["06", "07", "08", "09", "10"]


# =============================================================
# Log parsers
# =============================================================

def parse_scovox_log(log_path: Path, warmup: int = 5) -> dict:
    """Per-frame integrate_ms, RSS, [memUsage] bonxai_mb from scovox_node log."""
    integrate_ms, frame_ms, rss_mb = [], [], []
    bonxai_mb_series = []
    voxels_series = []
    if not log_path.exists():
        return {"missing": True}
    for line in log_path.read_text(errors="replace").splitlines():
        m = re.search(
            r"frame_ms=([\d.]+) tf_ms=[\d.]+ integrate_ms=([\d.]+) "
            r"publish_ms=[\d.]+ rss_mb=([\d.]+)", line)
        if m:
            frame_ms.append(float(m.group(1)))
            integrate_ms.append(float(m.group(2)))
            rss_mb.append(float(m.group(3)))
            continue
        g = re.search(r"\[memUsage\] voxels=(\d+) tvox=\d+ bonxai_mb=([\d.]+)", line)
        if g:
            voxels_series.append(int(g.group(1)))
            bonxai_mb_series.append(float(g.group(2)))
    if not integrate_ms:
        return {"missing": True}
    steady = integrate_ms[warmup:] or integrate_ms
    return {
        "n_frames_logged": len(integrate_ms),
        "integrate_ms_mean": statistics.mean(steady),
        "integrate_ms_median": statistics.median(steady),
        "fps_mean": 1000.0 / statistics.mean(steady),
        "fps_median": 1000.0 / statistics.median(steady),
        "frame_ms_mean": statistics.mean(frame_ms[warmup:] or frame_ms),
        "rss_peak_mb": max(rss_mb) if rss_mb else None,
        "rss_final_mb": rss_mb[-1] if rss_mb else None,
        "bonxai_mb_final": bonxai_mb_series[-1] if bonxai_mb_series else None,
        "voxels_final": voxels_series[-1] if voxels_series else None,
        # SLIMVDB-mode TSDF storage equivalent (8/32 of full Voxel struct).
        # Under SCovox-mode (this experiment), the full 32 B is in use; the
        # 8/32 number is the apples-to-apples vs SLIM-VDB's vdb_tsdf_mb.
        "bonxai_tsdf_equiv_mb": (bonxai_mb_series[-1] * 8/32) if bonxai_mb_series else None,
    }


def parse_slimvdb_log(log_path: Path, warmup: int = 5) -> dict:
    """SLIM-VDB timing.log: per-frame integrate_ms + final vdb_tsdf_mb/vdb_sem_mb."""
    integrate_ms, render_ms = [], []
    last_tsdf_mb, last_sem_mb = None, None
    if not log_path.exists():
        return {"missing": True}
    for line in log_path.read_text(errors="replace").splitlines():
        m = re.search(
            r"TIMING idx=\d+ integrate_ms=([\d.]+) render_ms=([\d.]+) "
            r"prune_ms=[\d.]+ vdb_tsdf_mb=([\d.]+) vdb_sem_mb=([\d.]+)", line)
        if m:
            integrate_ms.append(float(m.group(1)))
            render_ms.append(float(m.group(2)))
            last_tsdf_mb = float(m.group(3))
            last_sem_mb = float(m.group(4))
    if not integrate_ms:
        return {"missing": True}
    steady = integrate_ms[warmup:] or integrate_ms
    return {
        "n_frames_logged": len(integrate_ms),
        "integrate_ms_mean": statistics.mean(steady),
        "integrate_ms_median": statistics.median(steady),
        "render_ms_mean": statistics.mean(render_ms[warmup:] or render_ms),
        # FPS_integrate is the apples-to-apples SCovox comparator (SCovox does
        # no separate render step; depth->3D is part of integrate_ms there).
        "fps_mean": 1000.0 / statistics.mean(steady),
        "fps_median": 1000.0 / statistics.median(steady),
        "fps_with_render_mean": 1000.0 / (statistics.mean(steady) +
                                          statistics.mean(render_ms[warmup:] or render_ms)),
        "vdb_tsdf_mb_final": last_tsdf_mb,
        "vdb_sem_mb_final": last_sem_mb,
        "vdb_total_mb_final": (last_tsdf_mb + last_sem_mb) if last_tsdf_mb is not None else None,
    }


# =============================================================
# mIoU runners
# =============================================================

def run_eval(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout + "\n" + proc.stderr


def parse_per_scene_miou(stdout: str, scene: str) -> float | None:
    """Pull 'scene  mIoU=X.XXXX' / '∩=NN  mIoU=X.XXXX' lines."""
    lines = stdout.splitlines()
    cur_scene = None
    for line in lines:
        if scene in line and re.match(rf"\s*===.*{re.escape(scene)}|\[\s*{re.escape(scene)}|=== {re.escape(scene)}", line):
            cur_scene = scene
        m = re.search(r"mIoU=([\d.]+)", line)
        if m and (cur_scene == scene or scene in line):
            return float(m.group(1))
    return None


def parse_global_miou(stdout: str) -> dict[str, float]:
    """Pull all '<key> mIoU=X.XXXX' values keyed by preceding scene/seq."""
    out = {}
    cur = None
    for line in stdout.splitlines():
        m = re.match(r"\[\s*(?:scene\s*)?(\w+)\s*\]", line)
        if m:
            cur = m.group(1)
        m2 = re.search(r"\b(\w+):?\s+mIoU=([\d.]+)", line)
        if m2:
            key = m2.group(1)
            try:
                v = float(m2.group(2))
                if v != v:  # NaN
                    continue
                # Use either explicit key or current scene context
                k = key if key not in ("mIoU",) else cur
                if k:
                    out[k] = v
            except Exception:
                pass
        # Accept patterns like "    pred=...  mIoU=0.1234" — use cur context
        m3 = re.search(r"^\s+(?:pred=\S+\s+gt=\S+\s+).*mIoU=([\d.]+)", line)
        if m3 and cur:
            out[cur] = float(m3.group(1))
    return out


def eval_scovox_replica() -> dict[str, float]:
    cmd = [
        "python3", str(SCRIPTS / "eval_scovox_replica.py"),
        "--replica_root", str(REPLICA_ROOT),
        "--npz_root", str(RESULTS_ROOT / "replica"),
        "--do_miou",
    ]
    rc, out = run_eval(cmd)
    print("[eval scovox replica] rc=", rc, "tail:")
    print("\n".join(out.splitlines()[-20:]))
    return parse_global_miou(out)


def eval_scovox_kitti() -> dict[str, float]:
    cmd = [
        "python3", str(SCRIPTS / "eval_scovox_kitti_miou.py"),
        "--kitti_root", str(KITTI_ROOT),
        "--npz_root", str(RESULTS_ROOT / "kitti"),
        "--semantic_kitti_yaml", str(KITTI_YAML),
        "--variant", "scovox_e6_matched",
    ]
    rc, out = run_eval(cmd)
    print("[eval scovox kitti] rc=", rc, "tail:")
    print("\n".join(out.splitlines()[-20:]))
    return parse_global_miou(out)


def eval_slimvdb_replica() -> dict[str, float]:
    cmd = [
        "python3", str(SCRIPTS / "eval_slimvdb_replica_miou.py"),
        "--replica_root", str(REPLICA_ROOT),
        "--voxels_dir", str(SLIM_OUT / "replica_m2f"),
    ]
    rc, out = run_eval(cmd)
    print("[eval slimvdb replica] rc=", rc, "tail:")
    print("\n".join(out.splitlines()[-20:]))
    return parse_global_miou(out)


def eval_slimvdb_kitti() -> dict[str, float]:
    cmd = [
        "python3", str(SCRIPTS / "eval_slimvdb_kitti_miou.py"),
        "--kitti_root", str(KITTI_ROOT),
        "--voxels_dir", str(SLIM_OUT / "kitti"),
        "--semantic_kitti_yaml", str(KITTI_YAML),
    ]
    rc, out = run_eval(cmd)
    print("[eval slimvdb kitti] rc=", rc, "tail:")
    print("\n".join(out.splitlines()[-20:]))
    return parse_global_miou(out)


# =============================================================
# Main
# =============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_miou", action="store_true",
                    help="Skip slow mIoU (timing/memory only)")
    args = ap.parse_args()

    rows = []

    miou_scov_replica = {} if args.skip_miou else eval_scovox_replica()
    miou_slim_replica = {} if args.skip_miou else eval_slimvdb_replica()
    miou_scov_kitti   = {} if args.skip_miou else eval_scovox_kitti()
    miou_slim_kitti   = {} if args.skip_miou else eval_slimvdb_kitti()

    # ── Replica ──
    for scene in REPLICA_SCENES:
        scov_dir = RESULTS_ROOT / "replica" / scene
        slim_dir = SLIM_OUT / "replica_m2f" / scene
        scov_log = scov_dir / "scovox_run.log"
        slim_log = slim_dir / "timing.log"
        s_perf = parse_scovox_log(scov_log)
        v_perf = parse_slimvdb_log(slim_log)
        rows.append({
            "dataset": "replica", "cell": scene,
            "scovox_miou":  miou_scov_replica.get(scene),
            "slimvdb_miou": miou_slim_replica.get(scene),
            "scovox_fps":   s_perf.get("fps_median"),
            "slimvdb_fps":  v_perf.get("fps_median"),
            "scovox_grid_mb_full":      s_perf.get("bonxai_mb_final"),
            "scovox_grid_mb_tsdf_equiv": s_perf.get("bonxai_tsdf_equiv_mb"),
            "slimvdb_grid_mb_tsdf":     v_perf.get("vdb_tsdf_mb_final"),
            "slimvdb_grid_mb_total":    v_perf.get("vdb_total_mb_final"),
            "scovox_peak_rss_mb":       s_perf.get("rss_peak_mb"),
            "scovox_voxels":            s_perf.get("voxels_final"),
            "scovox_n_frames":          s_perf.get("n_frames_logged"),
            "slimvdb_n_frames":         v_perf.get("n_frames_logged"),
            "scovox_integrate_ms":      s_perf.get("integrate_ms_median"),
            "slimvdb_integrate_ms":     v_perf.get("integrate_ms_median"),
        })

    # ── KITTI ──
    for seq in KITTI_SEQS:
        scov_dir = RESULTS_ROOT / "kitti" / seq
        slim_dir = SLIM_OUT / "kitti" / seq
        scov_log = scov_dir / "scovox_run.log"
        slim_log = slim_dir / "timing.log"
        s_perf = parse_scovox_log(scov_log)
        v_perf = parse_slimvdb_log(slim_log)
        rows.append({
            "dataset": "kitti", "cell": seq,
            "scovox_miou":  miou_scov_kitti.get(seq),
            "slimvdb_miou": miou_slim_kitti.get(seq),
            "scovox_fps":   s_perf.get("fps_median"),
            "slimvdb_fps":  v_perf.get("fps_median"),
            "scovox_grid_mb_full":      s_perf.get("bonxai_mb_final"),
            "scovox_grid_mb_tsdf_equiv": s_perf.get("bonxai_tsdf_equiv_mb"),
            "slimvdb_grid_mb_tsdf":     v_perf.get("vdb_tsdf_mb_final"),
            "slimvdb_grid_mb_total":    v_perf.get("vdb_total_mb_final"),
            "scovox_peak_rss_mb":       s_perf.get("rss_peak_mb"),
            "scovox_voxels":            s_perf.get("voxels_final"),
            "scovox_n_frames":          s_perf.get("n_frames_logged"),
            "slimvdb_n_frames":         v_perf.get("n_frames_logged"),
            "scovox_integrate_ms":      s_perf.get("integrate_ms_median"),
            "slimvdb_integrate_ms":     v_perf.get("integrate_ms_median"),
        })

    # ── Write CSV ──
    out_csv = RESULTS_ROOT / "summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with out_csv.open("w") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {out_csv}\n")

    # ── Print ──
    print("=== Per-cell head-to-head ===")
    print(f"{'dataset':8s} {'cell':8s}  scov_mIoU  slim_mIoU   scov_FPS  slim_FPS  scov_grid_TSDF  slim_grid_TSDF  scov_grid_total  slim_grid_total")
    for r in rows:
        def f(v, w=8, p=4): return ("--" if v is None else f"{v:.{p}f}").rjust(w)
        print(f"{r['dataset']:<8s} {r['cell']:<8s}  "
              f"{f(r['scovox_miou'])}  {f(r['slimvdb_miou'])}    "
              f"{f(r['scovox_fps'], 6, 2)}  {f(r['slimvdb_fps'], 6, 2)}      "
              f"{f(r['scovox_grid_mb_tsdf_equiv'], 6, 2)}          "
              f"{f(r['slimvdb_grid_mb_tsdf'], 6, 2)}           "
              f"{f(r['scovox_grid_mb_full'], 6, 2)}           "
              f"{f(r['slimvdb_grid_mb_total'], 6, 2)}")

    # ── Aggregate ratios ──
    def agg(rows, ds):
        sub = [r for r in rows if r["dataset"] == ds]
        def ratio(a, b):
            xs = [(x[a], x[b]) for x in sub if x[a] is not None and x[b] is not None and x[b] != 0]
            return statistics.mean([va / vb for va, vb in xs]) if xs else None
        def mean(k):
            xs = [x[k] for x in sub if x[k] is not None]
            return statistics.mean(xs) if xs else None
        return {
            "scovox_miou_mean": mean("scovox_miou"),
            "slimvdb_miou_mean": mean("slimvdb_miou"),
            "scovox_fps_mean": mean("scovox_fps"),
            "slimvdb_fps_mean": mean("slimvdb_fps"),
            "fps_speedup_scov_over_slim": ratio("scovox_fps", "slimvdb_fps"),
            "scov_grid_tsdf_mean": mean("scovox_grid_mb_tsdf_equiv"),
            "slim_grid_tsdf_mean": mean("slimvdb_grid_mb_tsdf"),
            "tsdf_grid_ratio_scov_over_slim": ratio("scovox_grid_mb_tsdf_equiv", "slimvdb_grid_mb_tsdf"),
            "scov_grid_total_mean": mean("scovox_grid_mb_full"),
            "slim_grid_total_mean": mean("slimvdb_grid_mb_total"),
            "total_grid_ratio_scov_over_slim": ratio("scovox_grid_mb_full", "slimvdb_grid_mb_total"),
        }

    rep = agg(rows, "replica")
    kit = agg(rows, "kitti")
    print("\n=== Aggregate (means across cells) ===")
    for ds, a in [("replica", rep), ("kitti", kit)]:
        print(f"\n[{ds}]")
        for k, v in a.items():
            print(f"  {k:35s}: {('--' if v is None else f'{v:.4f}'):>10s}")

    # ── Save aggregate JSON ──
    (RESULTS_ROOT / "summary_agg.json").write_text(json.dumps(
        {"replica": rep, "kitti": kit, "rows": rows}, indent=2, default=str))
    print(f"\nSaved {RESULTS_ROOT / 'summary_agg.json'}")


if __name__ == "__main__":
    main()
