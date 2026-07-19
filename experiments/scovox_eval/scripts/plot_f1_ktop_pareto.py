"""F1 — K_TOP Pareto plot (Replica + KITTI dual-panel).

For each anchor, reads `results/softprob_<anchor>/summary.csv` (cells:
k_top_K) and plots mIoU vs K. Per-anchor faint curves + cross-anchor
mean ± std bold. Shows that K=2 sits within noise of K=full at a much
smaller per-voxel memory footprint.
"""
import argparse
import re
from pathlib import Path

import numpy as np


REPLICA_ANCHORS = [
    "softprob_replica_room0_m2f",
    "softprob_replica_room1_m2f",
    "softprob_replica_room2_m2f",
    "softprob_replica_office0_m2f",
    "softprob_replica_office1_m2f",
    "softprob_replica_office2_m2f",
    "softprob_replica_office3_m2f",
    "softprob_replica_office4_m2f",
]
KITTI_ANCHORS = [
    "softprob_kitti_seq06_polarseg",
    "softprob_kitti_seq07_polarseg",
    "softprob_kitti_seq08_polarseg",
    "softprob_kitti_seq09_polarseg",
    "softprob_kitti_seq10_polarseg",
]


def load_anchor(results_root: Path, anchor: str) -> dict[int, float]:
    path = results_root / anchor / "summary.csv"
    out = {}
    if not path.exists():
        return out
    with open(path) as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split(",")
            cell, miou = parts[0], float(parts[1])
            m = re.match(r"k_top_(\d+)$", cell)
            if m:
                out[int(m.group(1))] = miou
    return out


def per_voxel_bytes(K: int) -> int:
    """Per-slot semantic cost in the wire-format Voxel layout: each slot is
    4 B (float sem_cnt) + 2 B (uint16 sem_cls) = 6 B. K=2 → 12 B, K=18 →
    108 B, K=19 → 114 B. Matches the "12 B/voxel" claim in EXPERIMENT_PLAN.md."""
    return 6 * K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", type=Path,
                    default=Path("/home/kalhan/projects/HMR_Exploration_Experiment/hmr_exploration_ws/"
                                 "src/robot_sw/distributed_mapping/scovox_eval/results"))
    ap.add_argument("--out", type=Path,
                    default=Path("/home/kalhan/projects/HMR_Exploration_Experiment/hmr_exploration_ws/"
                                 "src/robot_sw/distributed_mapping/scovox_eval/results/f1_ktop_pareto.png"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), sharey=False)
    panels = [
        ("Replica (8 scenes, M2F soft-prob, 5 cm)", REPLICA_ANCHORS, axes[0], 18),
        ("KITTI (5 seqs, PolarSeg soft-prob, 10 cm)", KITTI_ANCHORS, axes[1], 19),
    ]

    for title, anchors, ax, K_full in panels:
        all_K = set()
        per_anchor = []
        for a in anchors:
            d = load_anchor(args.results_root, a)
            if not d:
                continue
            per_anchor.append(d)
            all_K.update(d.keys())
        Ks = sorted(all_K)

        # Faint per-anchor curves.
        for d in per_anchor:
            xs = sorted(d.keys())
            ys = [d[k] for k in xs]
            ax.plot(xs, ys, "-o", color="#888888", alpha=0.40, linewidth=1.0, markersize=3.5)

        # Cross-anchor mean ± std.
        means, stds = [], []
        ks_for_mean = []
        for K in Ks:
            vals = [d[K] for d in per_anchor if K in d]
            if len(vals) >= max(3, len(per_anchor) - 1):
                ks_for_mean.append(K)
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals)))
        means = np.array(means); stds = np.array(stds)
        ks_for_mean = np.array(ks_for_mean)
        ax.fill_between(ks_for_mean, means - stds, means + stds,
                        color="#c44e52", alpha=0.18, label="±1 σ across anchors")
        ax.plot(ks_for_mean, means, "-o", color="#c44e52", linewidth=2.2,
                markersize=6, label="cross-anchor mean")

        # Highlight K=2.
        if 2 in ks_for_mean:
            i2 = list(ks_for_mean).index(2)
            ax.annotate(f"K=2: {means[i2]:.3f} mIoU\n{per_voxel_bytes(2)} B/voxel",
                        xy=(2, means[i2]), xytext=(4, means[i2] - 0.025),
                        fontsize=9, color="#c44e52",
                        arrowprops=dict(arrowstyle="->", color="#c44e52", lw=1))
        # Highlight K=full.
        if K_full in ks_for_mean:
            iF = list(ks_for_mean).index(K_full)
            ax.annotate(f"K={K_full}: {means[iF]:.3f} mIoU\n{per_voxel_bytes(K_full)} B/voxel",
                        xy=(K_full, means[iF]), xytext=(K_full - 8, means[iF] + 0.025),
                        fontsize=9, color="black",
                        arrowprops=dict(arrowstyle="->", color="black", lw=1))

        ax.set_xlabel("K_TOP (truncation slots)")
        ax.set_ylabel("mIoU (↑)")
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
        ax.set_xticks([1, 2, 3, 4, 6, 9, 10, 18, 19])
        ax.set_xticklabels([str(k) for k in [1, 2, 3, 4, 6, 9, 10, 18, 19]])

    fig.suptitle("F1 — K_TOP truncation Pareto: accuracy vs per-voxel memory",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
