"""F3 — Fusion bar chart (solo_a / solo_b / fused per scene).

Reads `results/fusion_replica_m2f/e2_1_summary.csv` and produces a
3-panel figure (mIoU, F@5cm, Chamfer) with grouped bars per scene plus
a final "aggregate" group showing the mean.
"""
import argparse
from pathlib import Path

import numpy as np


SCENES = ["room0", "room1", "room2",
          "office0", "office1", "office2", "office3", "office4"]
LABELS = ["solo_a", "solo_b", "fused"]
COLORS = {"solo_a": "#a0c4e8", "solo_b": "#7eb1d6", "fused": "#d04a4a"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path,
                    default=Path("/home/kalhan/projects/HMR_Exploration_Experiment/hmr_exploration_ws/"
                                 "src/robot_sw/distributed_mapping/scovox_eval/results/fusion_replica_m2f/"
                                 "e2_1_summary.csv"))
    ap.add_argument("--out", type=Path,
                    default=Path("/home/kalhan/projects/HMR_Exploration_Experiment/hmr_exploration_ws/"
                                 "src/robot_sw/distributed_mapping/scovox_eval/results/fusion_replica_m2f/"
                                 "f3_fusion_bars.png"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = np.genfromtxt(args.csv, delimiter=",", names=True, dtype=None, encoding="utf-8")
    # Build {(scene, label): {metric: value}}
    data = {}
    for r in rows:
        data[(r["scene"], r["label"])] = (r["miou"], r["f_at_5cm"], float(r["chamfer_l2_m"]) * 100.0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
    metrics = [
        ("mIoU (↑)", 0),
        ("F@5cm (↑)", 1),
        ("Chamfer L2 (cm) (↓)", 2),
    ]
    x_groups = SCENES + ["mean"]
    x = np.arange(len(x_groups))
    bar_w = 0.27

    for ax, (title, mi) in zip(axes, metrics):
        for j, lbl in enumerate(LABELS):
            vals = [data.get((s, lbl), (np.nan, np.nan, np.nan))[mi] for s in SCENES]
            mean_v = float(np.nanmean(vals))
            xs = x + (j - 1) * bar_w
            ax.bar(xs[:-1], vals, bar_w, label=lbl, color=COLORS[lbl],
                   edgecolor="black", linewidth=0.4)
            ax.bar(xs[-1], mean_v, bar_w, color=COLORS[lbl],
                   edgecolor="black", linewidth=0.8, hatch="//")
        ax.set_xticks(x)
        ax.set_xticklabels(x_groups, rotation=35, ha="right", fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.legend(loc="upper right" if mi != 2 else "upper left", fontsize=9)
        # Visual separator before mean
        ax.axvline(len(SCENES) - 0.5, linestyle="--", color="grey", alpha=0.6, linewidth=0.7)

    fig.suptitle("F3 — Multi-robot trajectory-split fusion vs single-robot solos (Replica, 5 cm, 8 scenes)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
