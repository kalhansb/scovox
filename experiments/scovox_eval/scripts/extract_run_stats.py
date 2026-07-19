#!/usr/bin/env python3
"""Parse SCovox `recv=N replay=M frame_ms=X tf_ms=Y integrate_ms=Z publish_ms=W rss_mb=R`
log lines into per-run stats: median / mean ± std / p95 of frame_ms and
integrate_ms, derived FPS, peak RSS, voxel count + bonxai memory if the
[memUsage] line is present.

Usage:
    python3 extract_run_stats.py LOG [LOG ...] [--csv OUT]

Each LOG argument may be a file or a directory; directories are searched
recursively for `*scovox_run.log` and similar.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from statistics import mean, median, pstdev

PATTERN_RECV = re.compile(
    r"recv=(\d+)\s+replay=\d+\s+frame_ms=([\d.]+)\s+tf_ms=([\d.]+)\s+"
    r"integrate_ms=([\d.]+)\s+publish_ms=([\d.]+)\s+rss_mb=([\d.]+)"
)
PATTERN_MEMUSAGE = re.compile(
    r"\[memUsage\]\s+voxels=(\d+)\s+tvox=\d+\s+bonxai_mb=([\d.]+)\s+rss_mb=([\d.]+)\s+bpv=(\d+)"
)


def percentile(values, q):
    if not values:
        return float("nan")
    s = sorted(values)
    idx = int(round((len(s) - 1) * q))
    return s[idx]


def parse_log(path: Path) -> dict | None:
    frame_ms = []
    integrate_ms = []
    publish_ms = []
    rss_mb = []
    voxels = None
    bonxai_mb = None
    bpv = None
    n_frames = 0

    try:
        with path.open("r", errors="replace") as f:
            for line in f:
                m = PATTERN_RECV.search(line)
                if m:
                    n_frames = int(m.group(1))
                    frame_ms.append(float(m.group(2)))
                    integrate_ms.append(float(m.group(4)))
                    publish_ms.append(float(m.group(5)))
                    rss_mb.append(float(m.group(6)))
                    continue
                m = PATTERN_MEMUSAGE.search(line)
                if m:
                    voxels = int(m.group(1))
                    bonxai_mb = float(m.group(2))
                    bpv = int(m.group(4))
    except FileNotFoundError:
        return None

    if not frame_ms:
        return None

    # Drop the first 5 frames (warm-up) for timing aggregates.
    fms = frame_ms[5:] if len(frame_ms) > 5 else frame_ms
    ims = integrate_ms[5:] if len(integrate_ms) > 5 else integrate_ms
    pms = publish_ms[5:] if len(publish_ms) > 5 else publish_ms

    return {
        "log": str(path),
        "frames": n_frames,
        "frame_ms_mean": round(mean(fms), 2),
        "frame_ms_median": round(median(fms), 2),
        "frame_ms_std": round(pstdev(fms), 2) if len(fms) > 1 else 0.0,
        "frame_ms_p95": round(percentile(fms, 0.95), 2),
        "integrate_ms_mean": round(mean(ims), 2),
        "integrate_ms_median": round(median(ims), 2),
        "integrate_ms_p95": round(percentile(ims, 0.95), 2),
        "publish_ms_mean": round(mean(pms), 2),
        "fps_mean": round(1000.0 / mean(fms), 2) if mean(fms) > 0 else 0.0,
        "fps_median": round(1000.0 / median(fms), 2) if median(fms) > 0 else 0.0,
        "rss_peak_mb": round(max(rss_mb), 1),
        "rss_final_mb": round(rss_mb[-1], 1),
        "voxels": voxels if voxels is not None else "",
        "bonxai_mb": bonxai_mb if bonxai_mb is not None else "",
        "bpv": bpv if bpv is not None else "",
    }


def discover(paths):
    out = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for child in sorted(p.rglob("*scovox*run.log")):
                out.append(child)
        elif p.exists():
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", help="Log files or directories")
    ap.add_argument("--csv", type=Path, help="Optional CSV output path")
    args = ap.parse_args()

    logs = discover(args.paths)
    rows = []
    for log in logs:
        stats = parse_log(log)
        if stats is None:
            print(f"[skip] {log} (no recv= lines)", file=sys.stderr)
            continue
        rows.append(stats)

    if not rows:
        print("No usable logs found.", file=sys.stderr)
        sys.exit(1)

    cols = list(rows[0].keys())
    print("\t".join(cols))
    for r in rows:
        print("\t".join(str(r[c]) for c in cols))

    if args.csv:
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"\n[csv] wrote {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
