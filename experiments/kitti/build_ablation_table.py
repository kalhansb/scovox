#!/usr/bin/env python3
"""3-way semantic-mode ablation table for KITTI PolarSeg input:
dirichlet vs majority_vote vs naive (per-seq mIoU + mean)."""
import csv
from pathlib import Path

RES = Path("/home/kalhan/Projects/scovox_ws/experiments/results")
SEQS = ["06", "07", "08", "09", "10"]
# display label -> results dir
MODES = [("dirichlet", "kitti_polarseg"),
         ("majority_vote", "kitti_polarseg_majority_vote"),
         ("naive", "kitti_polarseg_naive")]


def load(p):
    d = {}
    if not p.exists():
        return d
    with open(p) as f:
        for row in csv.DictReader(f):
            try:
                d[row["seq"]] = float(row["miou"])
            except (ValueError, KeyError, TypeError):
                pass
    return d


def main():
    data = {m: load(RES / d / "summary.csv") for m, d in MODES}
    header = "| Sequence | " + " | ".join(m for m, _ in MODES) + " |"
    sep = "|---" * (len(MODES) + 1) + "|"
    lines = [header, sep]
    means = {m: [] for m, _ in MODES}
    for s in SEQS:
        cells = []
        for m, _ in MODES:
            v = data[m].get(s)
            cells.append(f"{v:.4f}" if v is not None else "—")
            if v is not None:
                means[m].append(v)
        lines.append(f"| {s} | " + " | ".join(cells) + " |")

    def mean(x):
        return sum(x) / len(x) if x else float("nan")

    mcells = [f"**{mean(means[m]):.4f}**" for m, _ in MODES]
    lines.append("| **mean** | " + " | ".join(mcells) + " |")
    table = "\n".join(lines)
    out = RES / "kitti_semantic_mode_ablation.md"
    out.write_text("# SemanticKITTI semantic-mode ablation (PolarSeg input, "
                   "scovox 10cm, seqs 06-10)\n\n" + table + "\n")
    print(table)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
