#!/usr/bin/env python3
"""Decompose the SCovox-vs-SLIM-VDB mIoU gap on SceneNet val.

For each cell, partition the voxel key universe `GT ∪ SCovox ∪ SLIM` into 7
non-empty regions and tally:
  A. GT only            → both miss (FN for both)
  B. SCovox only        → SCovox FP (clutter), SLIM-VDB correctly silent
  C. SLIM-VDB only      → SLIM-VDB FP (clutter), SCovox correctly silent
  D. GT ∩ SCovox only   → SLIM-VDB missed; SCovox may be TP or class-FP
  E. GT ∩ SLIM only     → SCovox missed; SLIM-VDB may be TP or class-FP
  F. SCovox ∩ SLIM only → both FP; question is symmetric labelling agreement
  G. All three          → both labelled; per-voxel argmax accuracy

This isolates coverage effects (D vs E) from per-voxel accuracy (G).

We also report a "common-keys mIoU": score both methods only on voxels both
labelled (F ∪ G). If SCovox still wins on common keys, the gap is per-voxel
accuracy; if it ties, the gap is coverage.

Usage:
  python3 scenenet_decompose_gap.py \
    --scovox_root src/robot_sw/distributed_mapping/scovox_eval/results/scenenet_val_iter6 \
    --slim_root   third_party_sw/slim_vdb/outputs/scenenet_val \
    --gt_root     data/scenenet/val_preprocessed \
    --out_csv     src/robot_sw/distributed_mapping/scovox_eval/results/scenenet_val_iter6/decompose_gap.csv
"""

from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np

NUM_CLASSES = 14
NYU_13 = ["Unknown","Bed","Books","Ceiling","Chair","Floor","Furniture",
          "Objects","Picture","Sofa","Table","TV","Wall","Window"]


def _pack_keys(coords_floor: np.ndarray) -> np.ndarray:
    c = coords_floor
    return ((c[:, 0] & 0xFFFFFF) << 40
            | (c[:, 1] & 0xFFFFF) << 20
            |  (c[:, 2] & 0xFFFFF))


def _xyz_to_keys(xyz: np.ndarray, res: float) -> np.ndarray:
    c = np.floor(xyz / res).astype(np.int64)
    return _pack_keys(c)


def _load_scovox(path: Path, res: float):
    z = np.load(path)
    xyz = z["points"].astype(np.float64)
    lbl = z["semantic_class"].astype(np.int32)
    keys = _xyz_to_keys(xyz, res)
    # collapse duplicates (shouldn't happen but safe)
    if len(np.unique(keys)) != len(keys):
        u, idx = np.unique(keys, return_index=True)
        keys = u
        lbl = lbl[idx]
    return keys, lbl


def _load_slimvdb(path: Path, res: float):
    raw = np.fromfile(path, dtype=np.uint8)
    rec = raw.view(np.dtype([("x","<f4"),("y","<f4"),("z","<f4"),("c","<i4")]))
    xyz = np.stack([rec["x"], rec["y"], rec["z"]], axis=1).astype(np.float64)
    lbl = rec["c"].astype(np.int32)
    keys = _xyz_to_keys(xyz, res)
    if len(np.unique(keys)) != len(keys):
        u, idx = np.unique(keys, return_index=True)
        keys = u
        lbl = lbl[idx]
    return keys, lbl


def _iou_from_pairs(gt_lbl, pr_lbl, num_classes=NUM_CLASSES, ignore=(0,)):
    """gt_lbl and pr_lbl are aligned arrays over the union universe; 0 = absent."""
    tp = np.zeros(num_classes, dtype=np.int64)
    fp = np.zeros(num_classes, dtype=np.int64)
    fn = np.zeros(num_classes, dtype=np.int64)
    for c in range(num_classes):
        tp[c] = int(np.sum((gt_lbl == c) & (pr_lbl == c)))
        fp[c] = int(np.sum((gt_lbl != c) & (pr_lbl == c)))
        fn[c] = int(np.sum((gt_lbl == c) & (pr_lbl != c)))
    denom = tp + fp + fn
    iou = np.where(denom > 0, tp / np.maximum(denom, 1), 0.0)
    has_data = denom > 0
    mask = has_data.copy()
    for c in ignore: mask[c] = False
    miou = float(iou[mask].mean()) if mask.any() else 0.0
    return miou, iou, tp, fp, fn


def _align_to_universe(universe_keys, src_keys, src_lbl):
    """Return an array of length len(universe_keys) with src labels (0 where absent)."""
    out = np.zeros(len(universe_keys), dtype=np.int32)
    # universe is sorted; use searchsorted
    idx = np.searchsorted(universe_keys, src_keys)
    # Filter to where src_keys actually exist in universe (should be all)
    valid = (idx < len(universe_keys)) & (universe_keys[np.clip(idx, 0, len(universe_keys)-1)] == src_keys)
    out[idx[valid]] = src_lbl[valid]
    return out


def analyze_cell(seq: str, gt_npz: Path, scovox_npz: Path, slim_bin: Path):
    g = np.load(gt_npz)
    res = float(g["resolution"])
    gt_keys = g["keys"].astype(np.int64)
    gt_lbl  = g["labels"].astype(np.int32)
    # ensure GT keys sorted
    order = np.argsort(gt_keys)
    gt_keys = gt_keys[order]; gt_lbl = gt_lbl[order]

    sc_keys, sc_lbl = _load_scovox(scovox_npz, res)
    sl_keys, sl_lbl = _load_slimvdb(slim_bin, res)

    universe = np.union1d(np.union1d(gt_keys, sc_keys), sl_keys)
    universe.sort()

    gt_full = _align_to_universe(universe, gt_keys, gt_lbl)
    sc_full = _align_to_universe(universe, sc_keys, sc_lbl)
    sl_full = _align_to_universe(universe, sl_keys, sl_lbl)

    has_gt = gt_full > 0
    has_sc = sc_full > 0
    has_sl = sl_full > 0

    # Regions
    A = has_gt & ~has_sc & ~has_sl
    B = ~has_gt & has_sc & ~has_sl
    C = ~has_gt & ~has_sc & has_sl
    D = has_gt & has_sc & ~has_sl
    E = has_gt & ~has_sc & has_sl
    F = ~has_gt & has_sc & has_sl
    G = has_gt & has_sc & has_sl

    # Class-correct flags
    sc_correct = (gt_full == sc_full) & has_gt & has_sc
    sl_correct = (gt_full == sl_full) & has_gt & has_sl

    # SCovox advantage in D: GT-covered, only SCovox labels → SCovox can be TP, SLIM is FN
    D_sc_tp = int(np.sum(D & sc_correct))
    D_sc_wrong = int(np.sum(D)) - D_sc_tp
    # SLIM advantage in E: GT-covered, only SLIM labels → SLIM can be TP, SCovox is FN
    E_sl_tp = int(np.sum(E & sl_correct))
    E_sl_wrong = int(np.sum(E)) - E_sl_tp
    # Both label, both could be right: G
    G_sc_tp = int(np.sum(G & sc_correct))
    G_sl_tp = int(np.sum(G & sl_correct))
    G_both_tp = int(np.sum(G & sc_correct & sl_correct))
    G_neither = int(np.sum(G & ~sc_correct & ~sl_correct))

    # Class-restricted scoring on common keys (F ∪ G):
    common = F | G
    if common.any():
        # Pad gt to 0 where it's not gt (which is the F region)
        miou_sc_common, _, _, _, _ = _iou_from_pairs(gt_full[common], sc_full[common])
        miou_sl_common, _, _, _, _ = _iou_from_pairs(gt_full[common], sl_full[common])
    else:
        miou_sc_common = miou_sl_common = 0.0

    # Full union mIoU (sanity check)
    miou_sc_full, iou_sc, tp_sc, fp_sc, fn_sc = _iou_from_pairs(gt_full, sc_full)
    miou_sl_full, iou_sl, tp_sl, fp_sl, fn_sl = _iou_from_pairs(gt_full, sl_full)

    return {
        "seq": seq,
        "miou_scovox": miou_sc_full,
        "miou_slim":   miou_sl_full,
        "miou_scovox_common": miou_sc_common,
        "miou_slim_common":   miou_sl_common,
        "n_gt": int(has_gt.sum()),
        "n_sc": int(has_sc.sum()),
        "n_sl": int(has_sl.sum()),
        "n_universe": int(len(universe)),
        "A_gt_only": int(A.sum()),
        "B_sc_only": int(B.sum()),
        "C_sl_only": int(C.sum()),
        "D_gt_sc_only": int(D.sum()),
        "D_sc_tp": D_sc_tp,
        "D_sc_wrong": D_sc_wrong,
        "E_gt_sl_only": int(E.sum()),
        "E_sl_tp": E_sl_tp,
        "E_sl_wrong": E_sl_wrong,
        "F_sc_sl_only": int(F.sum()),
        "G_all": int(G.sum()),
        "G_sc_tp": G_sc_tp,
        "G_sl_tp": G_sl_tp,
        "G_both_tp": G_both_tp,
        "G_neither": G_neither,
        # Per-class
        "iou_scovox": iou_sc.tolist(),
        "iou_slim":   iou_sl.tolist(),
        "tp_scovox":  tp_sc.tolist(),
        "tp_slim":    tp_sl.tolist(),
        "fn_scovox":  fn_sc.tolist(),
        "fn_slim":    fn_sl.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scovox_root", required=True)
    ap.add_argument("--slim_root", required=True)
    ap.add_argument("--gt_root", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    scovox_root = Path(args.scovox_root)
    slim_root = Path(args.slim_root)
    gt_root = Path(args.gt_root)

    seqs = sorted([d.name for d in scovox_root.iterdir()
                   if d.is_dir() and (slim_root / d.name / "voxels.bin").exists()])
    print(f"analyzing {len(seqs)} cells")

    rows = []
    for s in seqs:
        gt_npz = gt_root / s / "gt_5cm.npz"
        sc_npz = scovox_root / s / "scovox_dirichlet.npz"
        sl_bin = slim_root / s / "voxels.bin"
        if not (gt_npz.exists() and sc_npz.exists() and sl_bin.exists()):
            print(f"  SKIP {s}")
            continue
        r = analyze_cell(s, gt_npz, sc_npz, sl_bin)
        rows.append(r)
        print(f"  {s}  "
              f"mIoU sc={r['miou_scovox']:.3f} sl={r['miou_slim']:.3f}  "
              f"common sc={r['miou_scovox_common']:.3f} sl={r['miou_slim_common']:.3f}  "
              f"|sc|={r['n_sc']:>6d} |sl|={r['n_sl']:>6d} |gt|={r['n_gt']:>6d}  "
              f"D={r['D_gt_sc_only']:>5d}(tp {r['D_sc_tp']:>4d}) "
              f"E={r['E_gt_sl_only']:>5d}(tp {r['E_sl_tp']:>4d}) "
              f"G={r['G_all']:>5d}(sc {r['G_sc_tp']:>4d} sl {r['G_sl_tp']:>4d})")

    if not rows:
        print("no rows")
        return

    # Aggregate
    print("\n=== AGGREGATE ACROSS {} CELLS ===".format(len(rows)))
    keys_scalar = [k for k in rows[0] if k not in ("seq","iou_scovox","iou_slim",
                                                    "tp_scovox","tp_slim","fn_scovox","fn_slim")]
    means = {k: float(np.mean([r[k] for r in rows])) for k in keys_scalar}

    print(f"  mIoU SCovox     : {means['miou_scovox']:.4f}")
    print(f"  mIoU SLIM-VDB   : {means['miou_slim']:.4f}")
    print(f"  mIoU SCovox (common keys F∪G): {means['miou_scovox_common']:.4f}")
    print(f"  mIoU SLIM-VDB (common keys F∪G): {means['miou_slim_common']:.4f}")
    print(f"  Δ on full union : {means['miou_scovox']-means['miou_slim']:+.4f}")
    print(f"  Δ on common keys: {means['miou_scovox_common']-means['miou_slim_common']:+.4f}")

    print(f"\n  Region counts (per-cell mean):")
    print(f"    GT voxels:                  {means['n_gt']:>9.0f}")
    print(f"    SCovox voxels:              {means['n_sc']:>9.0f}")
    print(f"    SLIM voxels:                {means['n_sl']:>9.0f}")
    print(f"    A  (GT only, both miss):    {means['A_gt_only']:>9.0f}")
    print(f"    B  (SCovox FP, no SLIM):    {means['B_sc_only']:>9.0f}")
    print(f"    C  (SLIM FP, no SCovox):    {means['C_sl_only']:>9.0f}")
    print(f"    D  (GT∩SCovox, no SLIM):    {means['D_gt_sc_only']:>9.0f}  "
          f"of which SCovox-correct: {means['D_sc_tp']:>5.0f}  "
          f"({100*means['D_sc_tp']/max(means['D_gt_sc_only'],1):.1f}%)")
    print(f"    E  (GT∩SLIM, no SCovox):    {means['E_gt_sl_only']:>9.0f}  "
          f"of which SLIM-correct:   {means['E_sl_tp']:>5.0f}  "
          f"({100*means['E_sl_tp']/max(means['E_gt_sl_only'],1):.1f}%)")
    print(f"    F  (both label, no GT):     {means['F_sc_sl_only']:>9.0f}")
    print(f"    G  (all three):             {means['G_all']:>9.0f}")
    print(f"       └ SCovox-correct:        {means['G_sc_tp']:>9.0f}  "
          f"({100*means['G_sc_tp']/max(means['G_all'],1):.1f}%)")
    print(f"       └ SLIM-correct:          {means['G_sl_tp']:>9.0f}  "
          f"({100*means['G_sl_tp']/max(means['G_all'],1):.1f}%)")
    print(f"       └ both correct:          {means['G_both_tp']:>9.0f}")
    print(f"       └ neither correct:       {means['G_neither']:>9.0f}")

    # Attribution: where do SCovox's wins come from?
    # SCovox TP count: D_sc_tp + G_sc_tp
    # SLIM    TP count: E_sl_tp + G_sl_tp
    sc_tp_total = means['D_sc_tp'] + means['G_sc_tp']
    sl_tp_total = means['E_sl_tp'] + means['G_sl_tp']
    print(f"\n  TP totals (per cell, semantic TPs ignoring class 0):")
    print(f"    SCovox total TP: {sc_tp_total:>9.0f}  (D-region {means['D_sc_tp']:>5.0f} + G-region {means['G_sc_tp']:>5.0f})")
    print(f"    SLIM    total TP: {sl_tp_total:>9.0f}  (E-region {means['E_sl_tp']:>5.0f} + G-region {means['G_sl_tp']:>5.0f})")
    print(f"    ΔTP (SC - SL):  {sc_tp_total - sl_tp_total:+.0f}")
    print(f"      from coverage gain D-E: {means['D_sc_tp']-means['E_sl_tp']:+.0f}")
    print(f"      from G-region argmax:   {means['G_sc_tp']-means['G_sl_tp']:+.0f}")

    # Per-class summary
    print(f"\n  Per-class IoU (mean across cells):")
    print(f"    {'class':12s} {'SC':>7s} {'SL':>7s} {'Δ':>7s}")
    for c in range(NUM_CLASSES):
        sc_c = float(np.mean([r["iou_scovox"][c] for r in rows]))
        sl_c = float(np.mean([r["iou_slim"][c] for r in rows]))
        print(f"    {NYU_13[c]:12s} {sc_c:7.4f} {sl_c:7.4f} {sc_c-sl_c:+7.4f}")

    # Write CSV
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["seq","miou_scovox","miou_slim","miou_scovox_common","miou_slim_common",
            "n_gt","n_sc","n_sl","n_universe",
            "A_gt_only","B_sc_only","C_sl_only",
            "D_gt_sc_only","D_sc_tp","D_sc_wrong",
            "E_gt_sl_only","E_sl_tp","E_sl_wrong",
            "F_sc_sl_only","G_all","G_sc_tp","G_sl_tp","G_both_tp","G_neither"]
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] for c in cols])
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
