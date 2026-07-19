"""E5.1 — voxel mass conservation check.

For each voxel, mass = a_unk + Σ sem_cnt[k].
This is the sum of all semantic increments ever applied to that voxel —
the eviction/drop branches in sparse_add deliberately route evicted or
dropped mass into a_unk to preserve the total. So total_mass across the
whole grid should equal Σ_frames Σ_observations(inc) (an externally
known quantity if you instrument it).

Usage:
    python verify_mass_conservation.py path/to/scovox.npz
    python verify_mass_conservation.py scovox.npz --expected-mass 12345.0
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=Path)
    ap.add_argument("--expected-mass", type=float, default=None,
                    help="If provided, assert |total - expected| / expected < 1e-5.")
    args = ap.parse_args()

    d = np.load(args.npz)
    print(f"=== {args.npz} ===")
    print(f"  fields: {list(d.files)}")
    if "a_unk" not in d.files:
        print("  ❌ FAIL: a_unk not in NPZ — publish path predates the E5 patch")
        print("           re-run with the 2026-05-06+ scovox build to get raw mass fields")
        sys.exit(2)
    if "sem_cnt0" not in d.files or "sem_cnt1" not in d.files:
        print("  ❌ FAIL: sem_cnt0/sem_cnt1 not in NPZ")
        sys.exit(2)

    a_unk = d["a_unk"].astype(np.float64)
    sem_cnt = np.stack([d["sem_cnt0"], d["sem_cnt1"]], axis=1).astype(np.float64)
    voxel_mass = a_unk + sem_cnt.sum(axis=1)

    # Each non-zero sem_cnt slot must have an associated valid sem_cls.
    sem_cls0 = d["sem_cls0"].astype(np.int64)
    sem_cls1 = d["sem_cls1"].astype(np.int64)
    bad_slot = ((d["sem_cnt0"] > 0) & (sem_cls0 == 0xFFFF)).sum() \
             + ((d["sem_cnt1"] > 0) & (sem_cls1 == 0xFFFF)).sum()

    n = voxel_mass.size
    total = float(voxel_mass.sum())
    nonneg = bool(np.all(voxel_mass >= 0))
    finite = bool(np.all(np.isfinite(voxel_mass)))

    print(f"  voxels: {n}")
    print(f"  Σ mass: {total:.4f}")
    print(f"  mean mass / voxel: {voxel_mass.mean():.4f}")
    print(f"  min / max:          {voxel_mass.min():.4f} / {voxel_mass.max():.4f}")
    print(f"  Σ a_unk:            {float(a_unk.sum()):.4f}")
    print(f"  Σ sem_cnt0:         {float(sem_cnt[:, 0].sum()):.4f}")
    print(f"  Σ sem_cnt1:         {float(sem_cnt[:, 1].sum()):.4f}")
    print(f"  voxels with all-zero mass: {int((voxel_mass == 0).sum())}")
    print(f"  bad slot (count > 0 but cls = 0xFFFF): {int(bad_slot)}")
    print(f"  mass non-negative everywhere: {nonneg}")
    print(f"  mass finite everywhere:       {finite}")

    failed = []
    if not nonneg:
        failed.append("negative mass present")
    if not finite:
        failed.append("non-finite mass present")
    if bad_slot > 0:
        failed.append(f"{bad_slot} voxels with cnt > 0 but cls = 0xFFFF")

    if args.expected_mass is not None:
        rel = abs(total - args.expected_mass) / max(args.expected_mass, 1e-9)
        print(f"  expected: {args.expected_mass:.4f}  rel_err: {rel:.6e}")
        if rel >= 1e-5:
            failed.append(f"mass mismatch (rel {rel:.2e} ≥ 1e-5)")

    if failed:
        print("  ❌ FAIL: " + "; ".join(failed))
        sys.exit(1)
    print("  ✅ PASS")


if __name__ == "__main__":
    main()
