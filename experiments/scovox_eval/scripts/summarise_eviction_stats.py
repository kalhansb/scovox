"""E5.2 — summarise per-frame sparse_add branch counters.

Reads `eviction_stats.csv` (columns: frame, match, empty, evict, drop)
written by scovox_node when `eviction_stats_csv` is set, prints totals
+ % per branch, and optionally writes a stacked-area chart PNG.

Usage:
    python summarise_eviction_stats.py path/to/eviction_stats.csv
    python summarise_eviction_stats.py csv1 csv2 --plot out.png --labels Replica KITTI
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def _load(csv: Path):
    data = np.genfromtxt(csv, delimiter=",", names=True, dtype=np.int64)
    if data.size == 0:
        raise SystemExit(f"empty CSV: {csv}")
    return data


def _summarise(name: str, data):
    cols = ["match", "empty", "evict", "drop"]
    totals = {c: int(data[c].sum()) for c in cols}
    total_calls = sum(totals.values())
    if total_calls == 0:
        print(f"{name}: 0 sparse_add calls"); return totals, total_calls
    print(f"\n=== {name} — frames={data.size} ===")
    print(f"  sparse_add calls: {total_calls:,}")
    for c in cols:
        v = totals[c]
        pct = 100 * v / total_calls
        print(f"    {c:>5s}: {v:>14,d}  ({pct:6.2f} %)")
    overflow = totals["evict"] + totals["drop"]
    print(f"    overflow share (evict+drop) / calls: {100 * overflow / total_calls:.3f} %")
    return totals, total_calls


def _plot(datas, labels, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(datas), 1, figsize=(9, 3 * len(datas)), squeeze=False)
    cols = ["match", "empty", "evict", "drop"]
    colors = ["#4c72b0", "#55a868", "#dd8452", "#c44e52"]
    for ax, data, lbl in zip(axes[:, 0], datas, labels):
        x = data["frame"]
        # Cumulative running totals (visualises growth + relative shares)
        stacks = np.cumsum([data[c] for c in cols], axis=1)
        prev = np.zeros_like(x, dtype=np.float64)
        for c, color, totals in zip(cols, colors, stacks):
            ax.fill_between(x, prev, totals, label=c, color=color, alpha=0.85)
            prev = totals
        ax.set_title(f"{lbl} — sparse_add branch share (cumulative per frame)")
        ax.set_xlabel("frame")
        ax.set_ylabel("cumulative sparse_add calls")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"\nwrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", type=Path, nargs="+")
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--plot", type=Path, default=None)
    args = ap.parse_args()

    datas = []
    for csv in args.csvs:
        if not csv.exists():
            print(f"[skip] {csv}: missing", file=sys.stderr); continue
        d = _load(csv)
        datas.append(d)
    if not datas:
        sys.exit("no inputs loaded")

    labels = args.labels or [c.parent.name or c.stem for c in args.csvs]
    if len(labels) < len(datas):
        labels = labels + [c.stem for c in args.csvs[len(labels):]]

    grand = {"match": 0, "empty": 0, "evict": 0, "drop": 0}
    grand_calls = 0
    for csv, data, lbl in zip(args.csvs, datas, labels):
        totals, calls = _summarise(lbl, data)
        for k in grand:
            grand[k] += totals[k]
        grand_calls += calls
    if len(datas) > 1:
        print(f"\n=== combined ({len(datas)} runs) ===")
        print(f"  total sparse_add calls: {grand_calls:,}")
        for c in ("match", "empty", "evict", "drop"):
            print(f"    {c:>5s}: {grand[c]:>14,d}  ({100*grand[c]/max(grand_calls,1):6.2f} %)")

    if args.plot:
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        _plot(datas, labels, args.plot)


if __name__ == "__main__":
    main()
