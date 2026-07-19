#!/usr/bin/env python3
"""Compute occupancy ECE on Replica per scene × variant × resolution.

Reads {scene}/{variant}_gt_occ.npz (produced by replica_gt_binary.py) which
contains parallel arrays `occupancy_prob` and `gt_binary` aligned to the
predicted voxels.
"""
import os, sys, numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scovox_eval.metrics.ece import compute_ece

SCENES = ["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]
VARIANTS = ["scovox", "covox", "logodds"]
RES_DIRS = {"5cm": 0.05, "10cm": 0.10, "20cm": 0.20}

RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results"


def main():
    out_path = RESULTS_ROOT / "replica_occupancy_ece.txt"
    lines = []
    rows = []
    for res_name in RES_DIRS:
        per_variant = {v: {"ece": [], "mce": [], "brier": []} for v in VARIANTS}
        print(f"\n=== Replica occupancy ECE @ {res_name} ===")
        for scene in SCENES:
            for v in VARIANTS:
                f = RESULTS_ROOT / f"replica_{res_name}" / scene / f"{v}_gt_occ.npz"
                if not f.exists():
                    print(f"  MISS {scene}/{v}"); continue
                d = np.load(f)
                r = compute_ece(d["occupancy_prob"], d["gt_binary"].astype(float), n_bins=20)
                per_variant[v]["ece"].append(r["ece"])
                per_variant[v]["mce"].append(r["mce"])
                per_variant[v]["brier"].append(r["brier"])
                rows.append((res_name, scene, v, r["ece"], r["mce"], r["brier"]))
                print(f"  {scene:<8s} {v:<8s} ECE={r['ece']:.4f}  MCE={r['mce']:.4f}  Brier={r['brier']:.4f}")
        print(f"\n  -- mean ({res_name}) --")
        for v in VARIANTS:
            ece = per_variant[v]["ece"]; mce = per_variant[v]["mce"]; brier = per_variant[v]["brier"]
            if not ece: continue
            print(f"  {v:<8s} ECE={np.mean(ece):.4f} ± {np.std(ece):.4f}  "
                  f"MCE={np.mean(mce):.4f} ± {np.std(mce):.4f}  "
                  f"Brier={np.mean(brier):.4f} ± {np.std(brier):.4f}  (n={len(ece)})")

    with open(out_path, "w") as f:
        f.write("resolution,scene,variant,ece,mce,brier\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"\nWrote CSV: {out_path}")


if __name__ == "__main__":
    main()
