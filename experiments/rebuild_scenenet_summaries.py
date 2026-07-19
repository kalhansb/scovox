#!/usr/bin/env python3
"""Rebuild SceneNet soft-mode summary.csv files directly from each score.log's
'mIoU = X' line — robust against any batch-time grep race."""
import re
from pathlib import Path

RES = Path("/home/kalhan/Projects/scovox_ws/experiments/results")
TRAJS = ["0_175", "0_178", "0_182", "0_223", "0_279", "0_485", "0_490",
         "0_571", "0_682", "0_723", "0_789", "0_867", "0_977"]
DIRS = ["scenenet_soft", "scenenet_soft_majority_vote", "scenenet_soft_naive"]
PAT = re.compile(r"mIoU = ([0-9.]+)")


def main():
    for d in DIRS:
        base = RES / d
        if not base.exists():
            print(f"skip {d} (missing)"); continue
        rows = []
        for t in TRAJS:
            sl = base / t / "score.log"
            m = None
            if sl.exists():
                mo = PAT.search(sl.read_text())
                if mo:
                    m = mo.group(1)
            rows.append((t, m))
        with open(base / "summary.csv", "w") as f:
            f.write("seq,soft_miou\n")
            for t, m in rows:
                f.write(f"{t},{m if m is not None else ''}\n")
        got = sum(1 for _, m in rows if m is not None)
        print(f"{d}: {got}/{len(TRAJS)} mIoU values -> {base/'summary.csv'}")


if __name__ == "__main__":
    main()
