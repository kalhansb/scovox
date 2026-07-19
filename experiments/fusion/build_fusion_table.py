#!/usr/bin/env python3
"""Two-robot disjoint-fusion table: per-traj solo_a / solo_b / fused mIoU and
fused - max(solo). Rebuilds rows from each cell's scores.txt (robust)."""
import re
from pathlib import Path

RES = Path("/home/kalhan/Projects/scovox_ws/experiments/results/fusion_disjoint")
TRAJS = ["0_175", "0_178", "0_182", "0_223", "0_279", "0_485", "0_490",
         "0_571", "0_682", "0_723", "0_789", "0_867", "0_977"]


def score(cell, tag):
    f = cell / "scores.txt"
    if not f.exists():
        return None
    m = re.search(rf"{tag}\s+([0-9.]+)", f.read_text())
    return float(m.group(1)) if m else None


def main():
    lines = ["| Trajectory | solo_a [0,150) | solo_b [150,300) | fused | Δ fused−max(solo) |",
             "|---|---|---|---|---|"]
    sa, sb, sf, sd = [], [], [], []
    for t in TRAJS:
        cell = RES / t
        a, b, f = score(cell, "solo_a"), score(cell, "solo_b"), score(cell, "fused")
        if None in (a, b, f):
            lines.append(f"| {t} | {a or '—'} | {b or '—'} | {f or '—'} | — |")
            continue
        d = f - max(a, b)
        lines.append(f"| {t} | {a:.4f} | {b:.4f} | {f:.4f} | {d:+.4f} |")
        sa.append(a); sb.append(b); sf.append(f); sd.append(d)

    def mean(x):
        return sum(x) / len(x) if x else float("nan")

    lines.append(f"| **mean (n={len(sf)})** | **{mean(sa):.4f}** | **{mean(sb):.4f}** | "
                 f"**{mean(sf):.4f}** | **{mean(sd):+.4f}** |")
    wins = sum(1 for d in sd if d > 0)
    table = "\n".join(lines)
    out = RES / "fusion_disjoint_table.md"
    out.write_text("# SceneNet two-robot DISJOINT fusion (no shared frames: "
                   "A=[0,150), B=[150,300)), scovox 5cm\n\n" + table +
                   f"\n\nFused beats the better single robot in **{wins}/{len(sd)}** "
                   f"trajectories.\n")
    print(table)
    print(f"\nfused>max(solo) in {wins}/{len(sd)} trajs")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
