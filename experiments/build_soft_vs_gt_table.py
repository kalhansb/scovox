#!/usr/bin/env python3
"""Combine the GT-input and soft-label SceneNet summaries into one table."""
import csv
from pathlib import Path

RES = Path("/home/kalhan/Projects/scovox_ws/experiments/results")


def load(csv_path):
    d = {}
    if not csv_path.exists():
        return d
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            seq = row["seq"]
            val = next((row[k] for k in row if k.endswith("miou")), "")
            try:
                d[seq] = float(val)
            except (ValueError, TypeError):
                d[seq] = None
    return d


def main():
    gt = load(RES / "scenenet_gt" / "summary.csv")
    soft = load(RES / "scenenet_soft" / "summary.csv")
    seqs = sorted(set(gt) | set(soft))

    lines = ["| Trajectory | GT-input mIoU | Soft mIoU | Δ (soft−GT) |",
             "|---|---|---|---|"]
    gts, softs = [], []
    for s in seqs:
        g, so = gt.get(s), soft.get(s)
        d = f"{so - g:+.4f}" if (g is not None and so is not None) else "—"
        lines.append(f"| {s} | {g if g is not None else '—'} | "
                     f"{so if so is not None else '—'} | {d} |")
        if g is not None:
            gts.append(g)
        if so is not None:
            softs.append(so)

    def mean(x):
        return sum(x) / len(x) if x else float("nan")

    lines.append(f"| **mean (n={len(gts)}/{len(softs)})** | "
                 f"**{mean(gts):.4f}** | **{mean(softs):.4f}** | "
                 f"**{mean(softs) - mean(gts):+.4f}** |")
    table = "\n".join(lines)
    out = RES / "soft_vs_gt_table.md"
    out.write_text("# SceneNet: soft (Mask2Former swin-small) vs GT-input\n\n" + table + "\n")
    print(table)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
