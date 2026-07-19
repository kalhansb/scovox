"""F2 — SCovox vs SLIM-VDB head-to-head mIoU per scene/sequence.

Hard-coded from the canonical headline tables in experiments_and_results.md
(post-2026-05-04 final run). Two side-by-side panels: Replica + KITTI.
Bars: SCovox hard / SCovox soft / SLIM-VDB CLOSED, with a 'mean' group.
"""
import argparse
from pathlib import Path

import numpy as np


# Replica (8 scenes, M2F→19cls, 5 cm)
REPLICA_SCENES = ["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]
REPLICA_HARD   = [0.3487, 0.2799, 0.2813, 0.2593, 0.1596, 0.3655, 0.3908, 0.1897]
REPLICA_SOFT   = [0.4600, 0.3502, 0.3075, 0.2331, 0.1486, 0.3133, 0.4080, 0.1820]
REPLICA_SLIM   = [0.1293, 0.0923, 0.1037, 0.0972, 0.0597, 0.1321, 0.1228, 0.0942]

# KITTI (5 seqs, PolarSeg→20cls, 10 cm, 100 frames each)
# 2026-05-09 protocol fix: SCovox numbers re-scored with eval_scovox_kitti_miou.py
# (strict bucket-IoU on pred ∪ gt) — same protocol as the SLIM-VDB column.
# Pre-fix numbers (lenient KD-Tree pred→GT match @10cm via eval_ablations_kitti_seq08.py)
# preserved in git as the prior values: hard=[0.6253,0.5461,0.4328,0.4447,0.4117],
# soft=[0.6309,0.5536,0.4394,0.4730,0.4124] — DO NOT REUSE for SLIM-VDB head-to-head.
KITTI_SEQS  = ["seq06", "seq07", "seq08", "seq09", "seq10"]
KITTI_HARD  = [0.2467, 0.2309, 0.2159, 0.1934, 0.1452]
KITTI_SOFT  = [0.2502, 0.2345, 0.2243, 0.1982, 0.1510]
KITTI_SLIM  = [0.2508, 0.2369, 0.1388, 0.2394, 0.1779]

COLORS = {
    "SCovox hard":     "#a0c4e8",
    "SCovox soft":     "#c44e52",
    "SLIM-VDB CLOSED": "#5e887d",
}


def _panel(ax, scenes, hard, soft, slim, title):
    x_groups = list(scenes) + ["mean"]
    x = np.arange(len(x_groups))
    bar_w = 0.27

    series = [
        ("SCovox hard",     hard),
        ("SCovox soft",     soft),
        ("SLIM-VDB CLOSED", slim),
    ]
    for j, (lbl, vals) in enumerate(series):
        m = float(np.mean(vals))
        xs = x + (j - 1) * bar_w
        ax.bar(xs[:-1], vals, bar_w, label=lbl, color=COLORS[lbl],
               edgecolor="black", linewidth=0.4)
        ax.bar(xs[-1], m, bar_w, color=COLORS[lbl],
               edgecolor="black", linewidth=0.8, hatch="//")

    ax.axvline(len(scenes) - 0.5, linestyle="--", color="grey", alpha=0.6, linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(x_groups, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("mIoU (↑)")
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("/home/kalhan/projects/HMR_Exploration_Experiment/hmr_exploration_ws/"
                                 "src/robot_sw/distributed_mapping/scovox_eval/results/f2_head_to_head.png"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.4), sharey=False)
    _panel(axes[0], REPLICA_SCENES, REPLICA_HARD, REPLICA_SOFT, REPLICA_SLIM,
           "Replica — M2F soft-prob → 19 cls @ 5 cm")
    _panel(axes[1], KITTI_SEQS, KITTI_HARD, KITTI_SOFT, KITTI_SLIM,
           "KITTI — PolarSeg → 20 cls @ 10 cm")

    fig.suptitle("F2 — Head-to-head mIoU: SCovox vs SLIM-VDB CLOSED (same inputs both systems)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
