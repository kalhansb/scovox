#!/usr/bin/env python3
"""Per-class precision/recall decomposition + distance-to-GT histogram.

For each cell + each NYU13 class:
  precision_c = TP_c / (TP_c + FP_c)     # of voxels predicted c, fraction in GT
  recall_c    = TP_c / (TP_c + FN_c)     # of GT voxels c, fraction predicted

Then a histogram of FP voxels by L1-grid-distance to the nearest GT voxel:
  d=0 (on GT): the prediction is in a GT voxel but a different class → "label noise"
  d=1 (~5cm): adjacent to GT → "near-surface FP" (TSDF band edge)
  d>=2 (>=10cm): genuine free-space FP

Tests the hypothesis: SLIM-VDB's FP excess is mostly at d=1 and d=2+ (TSDF band
spill into free space), while SCovox's FPs are mostly at d=0 (class confusion
on the surface itself).

Usage:
  python3 scenenet_pr_per_class.py \
    --scovox_root src/robot_sw/distributed_mapping/scovox_eval/results/scenenet_val_iter6 \
    --slim_root   third_party_sw/slim_vdb/outputs/scenenet_val \
    --gt_root     data/scenenet/val_preprocessed
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

NUM_CLASSES = 14
NYU_13 = ["Unknown","Bed","Books","Ceiling","Chair","Floor","Furniture",
          "Objects","Picture","Sofa","Table","TV","Wall","Window"]


def _unpack_keys(keys: np.ndarray):
    """int64 packed key → (N,3) int64 coords."""
    x = ((keys >> 40) & 0xFFFFFF)
    y = ((keys >> 20) & 0xFFFFF)
    z = (keys & 0xFFFFF)
    # Sign-extend the 24-bit / 20-bit fields (they can be negative)
    x = np.where(x & (1 << 23), x - (1 << 24), x)
    y = np.where(y & (1 << 19), y - (1 << 20), y)
    z = np.where(z & (1 << 19), z - (1 << 20), z)
    return np.stack([x, y, z], axis=1).astype(np.int64)


def _pack_keys(coords: np.ndarray):
    c = coords
    return ((c[:, 0] & 0xFFFFFF) << 40
            | (c[:, 1] & 0xFFFFF) << 20
            |  (c[:, 2] & 0xFFFFF))


def _xyz_to_keys(xyz, res):
    return _pack_keys(np.floor(xyz / res).astype(np.int64))


def _load_scovox(path: Path, res: float):
    z = np.load(path)
    keys = _xyz_to_keys(z["points"].astype(np.float64), res)
    lbl = z["semantic_class"].astype(np.int32)
    if len(np.unique(keys)) != len(keys):
        u, idx = np.unique(keys, return_index=True); keys = u; lbl = lbl[idx]
    return keys, lbl


def _load_slimvdb(path: Path, res: float):
    raw = np.fromfile(path, dtype=np.uint8)
    rec = raw.view(np.dtype([("x","<f4"),("y","<f4"),("z","<f4"),("c","<i4")]))
    xyz = np.stack([rec["x"], rec["y"], rec["z"]], axis=1).astype(np.float64)
    lbl = rec["c"].astype(np.int32)
    keys = _xyz_to_keys(xyz, res)
    if len(np.unique(keys)) != len(keys):
        u, idx = np.unique(keys, return_index=True); keys = u; lbl = lbl[idx]
    return keys, lbl


def _distance_to_gt(pred_coords, gt_coords_set, max_check=3):
    """For each pred coord, find min L_inf distance to any GT voxel (capped at max_check).

    Builds a set lookup on tuples — slow for huge sets, fine for ~100k voxels.
    """
    n = len(pred_coords)
    dist = np.full(n, max_check + 1, dtype=np.int8)
    for i in range(n):
        cx, cy, cz = pred_coords[i]
        for d in range(max_check + 1):
            found = False
            # Check all coords with L_inf distance == d (boundary of cube)
            if d == 0:
                if (cx, cy, cz) in gt_coords_set:
                    found = True
            else:
                # Iterate boundary of cube of radius d
                for dx in range(-d, d + 1):
                    for dy in range(-d, d + 1):
                        for dz in range(-d, d + 1):
                            if max(abs(dx), abs(dy), abs(dz)) != d:
                                continue
                            if (cx + dx, cy + dy, cz + dz) in gt_coords_set:
                                found = True
                                break
                        if found: break
                    if found: break
            if found:
                dist[i] = d
                break
    return dist


def _distance_to_gt_vectorized(pred_keys, gt_keys, max_d=3):
    """Vectorised L_inf distance check using key-set lookups in shells.

    Returns: dist array, where dist[i] in {0,1,2,3,>max_d}.
    """
    gt_set = set(int(k) for k in gt_keys)
    pred_coords = _unpack_keys(pred_keys)
    n = len(pred_keys)
    dist = np.full(n, max_d + 1, dtype=np.int8)

    # d=0
    in_gt = np.array([int(k) in gt_set for k in pred_keys], dtype=bool)
    dist[in_gt] = 0

    # d>=1 shells — only check voxels not yet assigned
    for d in range(1, max_d + 1):
        unassigned = dist > max_d  # equiv to "still default"
        if not unassigned.any():
            break
        idxs = np.where(unassigned)[0]
        coords_u = pred_coords[idxs]
        # Generate shell offsets
        offsets = []
        for dx in range(-d, d + 1):
            for dy in range(-d, d + 1):
                for dz in range(-d, d + 1):
                    if max(abs(dx), abs(dy), abs(dz)) == d:
                        offsets.append((dx, dy, dz))
        # For each voxel, check if any offset hits GT
        found_mask = np.zeros(len(idxs), dtype=bool)
        for dx, dy, dz in offsets:
            shifted = coords_u.copy()
            shifted[:, 0] += dx; shifted[:, 1] += dy; shifted[:, 2] += dz
            shifted_keys = _pack_keys(shifted)
            for i, k in enumerate(shifted_keys):
                if not found_mask[i] and int(k) in gt_set:
                    found_mask[i] = True
        for i, h in enumerate(found_mask):
            if h:
                dist[idxs[i]] = d
    return dist


def analyze_cell(seq, gt_npz, scovox_npz, slim_bin, dist_sample=2000):
    g = np.load(gt_npz)
    res = float(g["resolution"])
    gt_keys = g["keys"].astype(np.int64)
    gt_lbl = g["labels"].astype(np.int32)
    sc_keys, sc_lbl = _load_scovox(scovox_npz, res)
    sl_keys, sl_lbl = _load_slimvdb(slim_bin, res)

    # Build per-key dicts
    gt_dict = {int(k): int(l) for k, l in zip(gt_keys, gt_lbl)}
    sc_dict = {int(k): int(l) for k, l in zip(sc_keys, sc_lbl)}
    sl_dict = {int(k): int(l) for k, l in zip(sl_keys, sl_lbl)}

    # Per-class TP/FP/FN
    def tally(pred_dict):
        tp = np.zeros(NUM_CLASSES, dtype=np.int64)
        fp = np.zeros(NUM_CLASSES, dtype=np.int64)
        fn = np.zeros(NUM_CLASSES, dtype=np.int64)
        # TPs and class-FPs: iterate predictions
        for k, p in pred_dict.items():
            g_ = gt_dict.get(k, 0)
            if p == g_:
                tp[p] += 1
            else:
                fp[p] += 1  # FP for predicted class
        # FNs: iterate GT
        for k, g_ in gt_dict.items():
            p = pred_dict.get(k, 0)
            if p != g_:
                fn[g_] += 1
        return tp, fp, fn

    tp_sc, fp_sc, fn_sc = tally(sc_dict)
    tp_sl, fp_sl, fn_sl = tally(sl_dict)

    # Distance-to-GT histogram for FPs (predictions outside GT support)
    sc_fp_keys = np.array([k for k in sc_dict if k not in gt_dict], dtype=np.int64)
    sl_fp_keys = np.array([k for k in sl_dict if k not in gt_dict], dtype=np.int64)

    rng = np.random.default_rng(42)
    if len(sc_fp_keys) > dist_sample:
        sc_fp_sample = rng.choice(sc_fp_keys, dist_sample, replace=False)
    else:
        sc_fp_sample = sc_fp_keys
    if len(sl_fp_keys) > dist_sample:
        sl_fp_sample = rng.choice(sl_fp_keys, dist_sample, replace=False)
    else:
        sl_fp_sample = sl_fp_keys

    sc_dist = _distance_to_gt_vectorized(sc_fp_sample, gt_keys, max_d=3)
    sl_dist = _distance_to_gt_vectorized(sl_fp_sample, gt_keys, max_d=3)

    sc_dist_hist = np.bincount(np.clip(sc_dist, 0, 4), minlength=5)
    sl_dist_hist = np.bincount(np.clip(sl_dist, 0, 4), minlength=5)

    return {
        "seq": seq, "res": res,
        "tp_sc": tp_sc, "fp_sc": fp_sc, "fn_sc": fn_sc,
        "tp_sl": tp_sl, "fp_sl": fp_sl, "fn_sl": fn_sl,
        "sc_fp_total": len(sc_fp_keys),
        "sl_fp_total": len(sl_fp_keys),
        "sc_dist_hist": sc_dist_hist,   # bins 0,1,2,3,>=4 (capped at 4)
        "sl_dist_hist": sl_dist_hist,
        "sc_fp_sample_size": len(sc_fp_sample),
        "sl_fp_sample_size": len(sl_fp_sample),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scovox_root", required=True)
    ap.add_argument("--slim_root", required=True)
    ap.add_argument("--gt_root", required=True)
    args = ap.parse_args()

    scovox_root = Path(args.scovox_root)
    slim_root = Path(args.slim_root)
    gt_root = Path(args.gt_root)

    seqs = sorted([d.name for d in scovox_root.iterdir()
                   if d.is_dir() and (slim_root / d.name / "voxels.bin").exists()])
    print(f"analyzing {len(seqs)} cells (distance hist sampled @ 2000 FPs/cell)")

    rows = []
    for s in seqs:
        gt_npz = gt_root / s / "gt_5cm.npz"
        sc_npz = scovox_root / s / "scovox_dirichlet.npz"
        sl_bin = slim_root / s / "voxels.bin"
        if not (gt_npz.exists() and sc_npz.exists() and sl_bin.exists()):
            print(f"  SKIP {s}"); continue
        print(f"  analyzing {s}...", flush=True)
        r = analyze_cell(s, gt_npz, sc_npz, sl_bin)
        rows.append(r)

    print("\n=== PER-CLASS PRECISION/RECALL (mean across cells) ===")
    print(f"  {'class':10s} {'SC P':>7s} {'SL P':>7s} {'SC R':>7s} {'SL R':>7s} "
          f"{'ΔP':>7s} {'ΔR':>7s}")
    for c in range(NUM_CLASSES):
        if c == 0: continue
        sc_p = np.mean([r["tp_sc"][c] / max(r["tp_sc"][c] + r["fp_sc"][c], 1) for r in rows])
        sl_p = np.mean([r["tp_sl"][c] / max(r["tp_sl"][c] + r["fp_sl"][c], 1) for r in rows])
        sc_r = np.mean([r["tp_sc"][c] / max(r["tp_sc"][c] + r["fn_sc"][c], 1) for r in rows])
        sl_r = np.mean([r["tp_sl"][c] / max(r["tp_sl"][c] + r["fn_sl"][c], 1) for r in rows])
        print(f"  {NYU_13[c]:10s} {sc_p:7.4f} {sl_p:7.4f} {sc_r:7.4f} {sl_r:7.4f} "
              f"{sc_p-sl_p:+7.4f} {sc_r-sl_r:+7.4f}")

    print("\n=== L_inf DISTANCE OF FPs TO NEAREST GT (mean across cells) ===")
    print("    Each row is mean fraction of FPs at that distance (bins: 0=on-GT/wrong-class, 1=5cm, 2=10cm, 3=15cm, ≥4=≥20cm)")
    print(f"  {'method':10s} {'d=0':>7s} {'d=1':>7s} {'d=2':>7s} {'d=3':>7s} {'d≥4':>7s} {'sample':>8s}")
    sc_hist_norm = [r["sc_dist_hist"] / max(r["sc_dist_hist"].sum(), 1) for r in rows]
    sl_hist_norm = [r["sl_dist_hist"] / max(r["sl_dist_hist"].sum(), 1) for r in rows]
    sc_mean = np.mean(sc_hist_norm, axis=0)
    sl_mean = np.mean(sl_hist_norm, axis=0)
    print(f"  {'SCovox':10s} {sc_mean[0]:7.3f} {sc_mean[1]:7.3f} {sc_mean[2]:7.3f} {sc_mean[3]:7.3f} {sc_mean[4]:7.3f}  per-cell mean")
    print(f"  {'SLIM-VDB':10s} {sl_mean[0]:7.3f} {sl_mean[1]:7.3f} {sl_mean[2]:7.3f} {sl_mean[3]:7.3f} {sl_mean[4]:7.3f}  per-cell mean")

    sc_fp_total = np.mean([r["sc_fp_total"] for r in rows])
    sl_fp_total = np.mean([r["sl_fp_total"] for r in rows])
    print(f"\n  Mean FPs per cell: SCovox={sc_fp_total:.0f}  SLIM-VDB={sl_fp_total:.0f}  (ratio {sl_fp_total/max(sc_fp_total,1):.1f}×)")

    # Absolute (extrapolated to total population)
    print(f"\n  Extrapolated FP counts by distance bin (per cell):")
    print(f"  {'method':10s} {'d=0':>7s} {'d=1':>7s} {'d=2':>7s} {'d=3':>7s} {'d≥4':>7s} {'total':>8s}")
    sc_abs = sc_mean * sc_fp_total
    sl_abs = sl_mean * sl_fp_total
    print(f"  {'SCovox':10s} {sc_abs[0]:7.0f} {sc_abs[1]:7.0f} {sc_abs[2]:7.0f} {sc_abs[3]:7.0f} {sc_abs[4]:7.0f}  {sc_fp_total:8.0f}")
    print(f"  {'SLIM-VDB':10s} {sl_abs[0]:7.0f} {sl_abs[1]:7.0f} {sl_abs[2]:7.0f} {sl_abs[3]:7.0f} {sl_abs[4]:7.0f}  {sl_fp_total:8.0f}")


if __name__ == "__main__":
    main()
