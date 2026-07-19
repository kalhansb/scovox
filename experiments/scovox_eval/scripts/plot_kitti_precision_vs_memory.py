#!/usr/bin/env python3
"""Precision-vs-memory Pareto figure for KITTI head-to-head.

X axis: grid-only memory in MB (vdb_tsdf + vdb_sem for SLIM-VDB,
        tsdf_grid + semdir_grid for SCovox).
Y axis: bucket-IoU mIoU at 10 cm voxel res.
Markers: one per (system, config) at the per-sequence level (light/small)
         plus a bold mean ± SE marker per config.

SLIM-VDB at multiple sdf_trunc values traces a curve; SCovox K=2 is a
single point per sequence. The combined "tighter trunc → higher mIoU"
sweep on SLIM-VDB pushes accuracy up but memory barely shrinks (most
mass is in the dense Dirichlet counters, not the TSDF band), so SCovox
Pareto-dominates at every measured SLIM-VDB config.

Input: third_party_sw/slim_vdb/outputs/kitti_precision_vs_memory.csv
        (built by build_kitti_pareto.py / aggregate script).
"""
import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def mean_se(xs):
    m = statistics.mean(xs)
    if len(xs) < 2:
        return m, float("nan")
    return m, statistics.stdev(xs) / math.sqrt(len(xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path,
                    default=Path("third_party_sw/slim_vdb/outputs/kitti_precision_vs_memory.csv"))
    ap.add_argument("--out", type=Path,
                    default=Path("figures/qualitative_2026_05_15/kitti_precision_vs_memory.png"))
    args = ap.parse_args()

    by_config = defaultdict(list)
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            by_config[(row["system"], row["config"])].append({
                "seq": row["seq"],
                "mb": float(row["total_mb"]),
                "miou": float(row["miou"]),
            })

    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=140)

    SLIM_COLOR = "#d62728"  # red
    SC_COLOR = "#1f77b4"    # blue

    # Per-cell scatter (light)
    for (sys_name, cfg), rows in by_config.items():
        color = SC_COLOR if sys_name.startswith("SCovox") else SLIM_COLOR
        ax.scatter([r["mb"] for r in rows], [r["miou"] for r in rows],
                   color=color, alpha=0.25, s=22, edgecolors="none")

    # Aggregate point + error bars per (system, config)
    slim_xy = []
    slim_xy_se = []
    slim_cfgs = []
    for (sys_name, cfg), rows in by_config.items():
        mb_m, mb_se = mean_se([r["mb"] for r in rows])
        mi_m, mi_se = mean_se([r["miou"] for r in rows])
        color = SC_COLOR if sys_name.startswith("SCovox") else SLIM_COLOR
        marker = "*" if sys_name.startswith("SCovox") else "o"
        size = 280 if sys_name.startswith("SCovox") else 110
        zorder = 5 if sys_name.startswith("SCovox") else 4
        label = sys_name if sys_name.startswith("SCovox") else f"SLIM-VDB {cfg}"
        ax.errorbar(mb_m, mi_m, xerr=mb_se, yerr=mi_se,
                    fmt="none", ecolor=color, alpha=0.7,
                    elinewidth=1.5, capsize=4, zorder=zorder - 1)
        ax.scatter([mb_m], [mi_m], color=color, marker=marker, s=size,
                   edgecolors="black", linewidths=1.0, zorder=zorder,
                   label=label)
        if sys_name == "SLIM-VDB":
            slim_xy.append((mb_m, mi_m))
            slim_xy_se.append((mb_se, mi_se))
            slim_cfgs.append(cfg)

    # Connect SLIM-VDB aggregate points with a trend line (trunc sweep curve)
    slim_xy.sort(key=lambda p: p[0])  # by memory
    ax.plot([p[0] for p in slim_xy], [p[1] for p in slim_xy],
            color=SLIM_COLOR, alpha=0.4, linestyle="--", linewidth=1.4,
            zorder=3, label="SLIM-VDB sdf_trunc sweep")

    # Annotate trunc values
    for (mb, mi), cfg in zip(sorted(zip(slim_xy, slim_cfgs)), slim_cfgs):
        pass

    # Re-walk to annotate by label
    for ((sys_name, cfg), rows) in by_config.items():
        if sys_name != "SLIM-VDB":
            continue
        mb_m, _ = mean_se([r["mb"] for r in rows])
        mi_m, _ = mean_se([r["miou"] for r in rows])
        t = cfg.split("=")[1]
        ax.annotate(f"trunc={t}", xy=(mb_m, mi_m),
                    xytext=(8, -4), textcoords="offset points",
                    fontsize=9, color="black", alpha=0.85)

    # SCovox annotation
    for (sys_name, cfg), rows in by_config.items():
        if sys_name != "SCovox":
            continue
        mb_m, _ = mean_se([r["mb"] for r in rows])
        mi_m, _ = mean_se([r["miou"] for r in rows])
        ax.annotate("SCovox K=2", xy=(mb_m, mi_m),
                    xytext=(10, 8), textcoords="offset points",
                    fontsize=10, weight="bold", color=SC_COLOR)

    ax.set_xlabel("Grid memory (MB, log scale) — vdb_tsdf+vdb_sem  /  tsdf_grid+semdir_grid",
                  fontsize=10)
    ax.set_ylabel("KITTI mIoU (bucket-IoU, 10 cm voxels)", fontsize=11)
    ax.set_xscale("log")
    ax.set_title("KITTI seq06-10 — precision vs. grid memory (n=5 per config, "
                 "mean ± SE)\nSCovox Pareto-dominates SLIM-VDB at every sdf_trunc",
                 fontsize=11, weight="bold")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.set_ylim(0.10, 0.36)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
