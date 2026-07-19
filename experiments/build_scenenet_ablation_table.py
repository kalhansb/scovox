#!/usr/bin/env python3
"""3-way semantic-mode ablation table for SceneNet SOFT input:
dirichlet vs majority_vote vs naive (per-traj mIoU + mean)."""
import csv
from pathlib import Path

RES = Path("/home/kalhan/Projects/scovox_ws/experiments/results")
TRAJS = ["0_175", "0_178", "0_182", "0_223", "0_279", "0_485", "0_490",
         "0_571", "0_682", "0_723", "0_789", "0_867", "0_977"]
MODES = [("dirichlet", "scenenet_soft"),
         ("majority_vote", "scenenet_soft_majority_vote"),
         ("naive", "scenenet_soft_naive")]


def load(p):
    d = {}
    if not p.exists():
        return d
    with open(p) as f:
        for row in csv.DictReader(f):
            seq = row.get("seq")
            val = next((row[k] for k in row if k.endswith("miou")), "")
            try:
                d[seq] = float(val)
            except (ValueError, TypeError):
                pass
    return d


def main():
    data = {m: load(RES / d / "summary.csv") for m, d in MODES}
    header = "| Trajectory | " + " | ".join(m for m, _ in MODES) + " |"
    sep = "|---" * (len(MODES) + 1) + "|"
    lines = [header, sep]
    means = {m: [] for m, _ in MODES}
    for t in TRAJS:
        cells = []
        for m, _ in MODES:
            v = data[m].get(t)
            cells.append(f"{v:.4f}" if v is not None else "—")
            if v is not None:
                means[m].append(v)
        lines.append(f"| {t} | " + " | ".join(cells) + " |")

    def mean(x):
        return sum(x) / len(x) if x else float("nan")

    lines.append("| **mean** | " + " | ".join(f"**{mean(means[m]):.4f}**" for m, _ in MODES) + " |")
    table = "\n".join(lines)
    out = RES / "scenenet_semantic_mode_ablation.md"
    out.write_text("# SceneNet semantic-mode ablation (Mask2Former SOFT input, "
                   "scovox 5cm, 13 trajs)\n\n" + table + "\n")
    print(table)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
