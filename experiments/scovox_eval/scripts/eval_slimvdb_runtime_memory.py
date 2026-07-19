#!/usr/bin/env python3
"""Parse SLIM-VDB per-run `timing.log` + `telemetry.csv` files and emit
Table I-style summary rows: mean/median FPS, P95 frame time, peak GPU memory,
peak system memory, OpenVDB TSDF + semantic grid sizes (final frame).

Usage:
    python eval_slimvdb_runtime_memory.py \
        --runs third_party_sw/slim_vdb/outputs/kitti/06 \
               third_party_sw/slim_vdb/outputs/kitti/07 ... \
        --label slimvdb-kitti

or:
    python eval_slimvdb_runtime_memory.py --auto --label all
"""
import argparse, csv, re, statistics as st
from pathlib import Path


TIMING_RE = re.compile(
    r"TIMING idx=(\d+) integrate_ms=([\d.]+) render_ms=([\d.]+) "
    r"prune_ms=([\d.]+) vdb_tsdf_mb=([\d.]+) vdb_sem_mb=([\d.]+)"
)


def parse_run(run_dir: Path) -> dict | None:
    tlog = run_dir / "timing.log"
    tcsv = run_dir / "telemetry.csv"
    if not tlog.exists():
        return None

    text = tlog.read_text()
    samples = []
    for m in TIMING_RE.finditer(text):
        idx = int(m.group(1))
        integ = float(m.group(2))
        rend = float(m.group(3))
        prune = float(m.group(4))
        tsdf_mb = float(m.group(5))
        sem_mb = float(m.group(6))
        samples.append((idx, integ, rend, prune, tsdf_mb, sem_mb))
    if not samples:
        return None

    # Warm-up: drop first 5% of frames (model / kernel caches settle).
    n = len(samples)
    drop = max(1, n // 20)
    warm = samples[drop:]
    integrate = [s[1] for s in warm]
    render = [s[2] for s in warm]
    frame_total = [s[1] + s[2] + s[3] for s in warm]

    def p95(xs):
        return sorted(xs)[int(0.95 * (len(xs) - 1))] if xs else 0.0

    # Peak map memory is the last-frame value (monotonically growing).
    last_tsdf = samples[-1][4]
    last_sem = samples[-1][5]

    # Telemetry: GPU + system RSS peaks during the run.
    gpu_peak_mb = sys_peak_mb = 0.0
    gpu_delta_mb = sys_delta_mb = 0.0
    tel_rows = 0
    if tcsv.exists():
        with open(tcsv) as f:
            for r in csv.DictReader(f):
                gpu_peak_mb = max(gpu_peak_mb, float(r["gpu_mem_mb"]))
                sys_peak_mb = max(sys_peak_mb, float(r["sys_used_mb"]))
                gpu_delta_mb = max(gpu_delta_mb, float(r["gpu_delta_mb"]))
                sys_delta_mb = max(sys_delta_mb, float(r["sys_delta_mb"]))
                tel_rows += 1

    fps_per_frame = [1000.0 / max(ft, 1e-6) for ft in frame_total]

    return {
        "run_dir": str(run_dir),
        "n_frames": n,
        "n_warm": len(warm),
        "integrate_ms_median": st.median(integrate),
        "integrate_ms_mean": st.mean(integrate),
        "render_ms_median": st.median(render),
        "render_ms_mean": st.mean(render),
        "frame_ms_median": st.median(frame_total),
        "frame_ms_p95": p95(frame_total),
        "fps_median": st.median(fps_per_frame),
        "fps_mean": st.mean(fps_per_frame),
        "vdb_tsdf_mb_final": last_tsdf,
        "vdb_sem_mb_final": last_sem,
        "gpu_peak_mb": gpu_peak_mb,
        "gpu_delta_mb": gpu_delta_mb,
        "sys_peak_mb": sys_peak_mb,
        "sys_delta_mb": sys_delta_mb,
        "telemetry_samples": tel_rows,
    }


def aggregate(rows):
    """Mean ± std across a set of runs (scene-level averaging like the paper)."""
    keys_for_stats = ["fps_median", "fps_mean", "integrate_ms_median",
                      "render_ms_median", "frame_ms_median", "frame_ms_p95",
                      "vdb_tsdf_mb_final", "vdb_sem_mb_final",
                      "gpu_peak_mb", "gpu_delta_mb",
                      "sys_peak_mb", "sys_delta_mb"]
    agg = {}
    for k in keys_for_stats:
        vals = [r[k] for r in rows if r is not None]
        if not vals:
            continue
        agg[f"{k}_mean"] = st.mean(vals)
        agg[f"{k}_std"] = st.stdev(vals) if len(vals) > 1 else 0.0
    agg["n_runs"] = len(rows)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", type=Path, default=[],
                    help="Run directories (each containing timing.log + telemetry.csv).")
    ap.add_argument("--auto", action="store_true",
                    help="Auto-scan third_party_sw/slim_vdb/outputs/ for all runs.")
    ap.add_argument("--label", default="",
                    help="Optional group label for the summary row.")
    ap.add_argument("--csv", type=Path, default=None,
                    help="Write detailed per-run CSV to this path.")
    args = ap.parse_args()

    runs = list(args.runs)
    if args.auto:
        for root in Path("third_party_sw/slim_vdb/outputs").glob("*"):
            if not root.is_dir():
                continue
            for sub in sorted(root.iterdir()):
                if sub.is_dir() and (sub / "timing.log").exists():
                    runs.append(sub)

    if not runs:
        print("no runs given; use --runs or --auto")
        return

    parsed = [parse_run(r) for r in runs]
    parsed = [p for p in parsed if p]
    if not parsed:
        print("no valid runs parsed")
        return

    # Per-run table
    print(f"\n=== {args.label or 'runs'} — per-run detail ({len(parsed)} runs) ===")
    cols = ["run", "frames", "fps_med", "integ_ms", "render_ms",
            "frame_p95ms", "tsdf_MB", "sem_MB", "gpuPeak_MB", "sysPeak_MB"]
    print("  " + "  ".join(f"{c:>11}" for c in cols))
    for r in parsed:
        name = Path(r["run_dir"]).name
        print(f"  {name:>11s}  "
              f"{r['n_frames']:>11d}  "
              f"{r['fps_median']:>11.2f}  "
              f"{r['integrate_ms_median']:>11.1f}  "
              f"{r['render_ms_median']:>11.1f}  "
              f"{r['frame_ms_p95']:>11.1f}  "
              f"{r['vdb_tsdf_mb_final']:>11.1f}  "
              f"{r['vdb_sem_mb_final']:>11.1f}  "
              f"{r['gpu_peak_mb']:>11.0f}  "
              f"{r['sys_peak_mb']:>11.0f}")

    agg = aggregate(parsed)
    print(f"\n=== {args.label or 'runs'} — aggregate (mean ± std across {agg['n_runs']} runs) ===")
    def show(k, unit=""):
        m = agg.get(f"{k}_mean"); s = agg.get(f"{k}_std")
        if m is None:
            return
        print(f"  {k:<25s} {m:>10.2f} ± {s:>8.2f} {unit}")

    show("fps_median", "FPS")
    show("integrate_ms_median", "ms")
    show("render_ms_median", "ms")
    show("frame_ms_median", "ms")
    show("frame_ms_p95", "ms")
    show("vdb_tsdf_mb_final", "MB")
    show("vdb_sem_mb_final", "MB")
    show("gpu_peak_mb", "MB")
    show("gpu_delta_mb", "MB")
    show("sys_peak_mb", "MB")
    show("sys_delta_mb", "MB")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in parsed for k in r.keys()}))
            w.writeheader()
            for r in parsed:
                w.writerow(r)
        print(f"\nwrote per-run CSV → {args.csv}")


if __name__ == "__main__":
    main()
