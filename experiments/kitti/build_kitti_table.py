#!/usr/bin/env python3
"""Combine KITTI GT-input and PolarSeg-input scovox mIoU into one table."""
import csv
from pathlib import Path

RES = Path("/home/kalhan/Projects/scovox_ws/experiments/results")
SEQS = ["06", "07", "08", "09", "10"]


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
    gt = load(RES / "kitti_gt" / "summary.csv")
    ps = load(RES / "kitti_polarseg" / "summary.csv")

    lines = ["| Sequence | GT-input mIoU | PolarSeg mIoU | Δ (PolarSeg−GT) |",
             "|---|---|---|---|"]
    gv, pv = [], []
    for s in SEQS:
        g, p = gt.get(s), ps.get(s)
        d = f"{p - g:+.4f}" if (g is not None and p is not None) else "—"
        gstr = f"{g:.4f}" if g is not None else "—"
        pstr = f"{p:.4f}" if p is not None else "—"
        lines.append(f"| {s} | {gstr} | {pstr} | {d} |")
        if g is not None:
            gv.append(g)
        if p is not None:
            pv.append(p)

    def mean(x):
        return sum(x) / len(x) if x else float("nan")

    dmean = (mean(pv) - mean(gv)) if (gv and pv) else float("nan")
    lines.append(f"| **mean (n={len(gv)}/{len(pv)})** | **{mean(gv):.4f}** | "
                 f"**{mean(pv):.4f}** | **{dmean:+.4f}** |")
    table = "\n".join(lines)
    out = RES / "kitti_gt_vs_polarseg_table.md"
    out.write_text("# SemanticKITTI: GT-input vs PolarSeg-input (scovox, 10cm, seqs 06-10, 100 scans)\n\n"
                   + table + "\n")
    print(table)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
