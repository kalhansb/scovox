#!/usr/bin/env python3
"""Compute calibration metrics for flatforest simulation experiment (Exp 2).

Runs on the outputs of flatforest_create_dataset.sh + flatforest_gt_query.sh:
  - scovox.npz, covox.npz, logodds.npz  (predicted voxels)
  - gt_voxels.npz  (ground truth from WorldQuerySystem)

Produces:
  - Occupancy ECE, MCE, Brier (overall + stratified by evidence) — task 4.6
  - Unknown vs ambiguous ROC-AUC — task 4.7
  - Semantic ECE for SCovox — task 4.6b

Usage:
    python3 flatforest_calibration.py [RESULTS_DIR]
    Default: scovox_eval/results/flatforest_10cm
"""

import os
import sys
import time

import numpy as np
from scipy.spatial import KDTree

# Add parent package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scovox_eval.metrics.ece import compute_ece, compute_stratified_ece
from scovox_eval.metrics.semantic_ece import (
    compute_semantic_ece,
    compute_stratified_semantic_ece,
)
from scovox_eval.metrics.unknown_vs_ambiguous import (
    beta_variance,
    compute_unknown_vs_ambiguous,
    compute_logodds_baseline,
)


def match_to_gt(pred_points, gt_points, gt_binary, gt_semantic, max_dist=0.10):
    """Match predicted voxels to GT by nearest-neighbour lookup.

    Returns matched GT arrays aligned to pred_points.
    Voxels with no GT match within max_dist get gt_binary=NaN.
    """
    t0 = time.time()
    tree = KDTree(gt_points)
    dists, idxs = tree.query(pred_points)
    dt = time.time() - t0

    matched_binary = gt_binary[idxs].copy()
    matched_semantic = gt_semantic[idxs].copy()

    # Mark unmatched voxels
    too_far = dists > max_dist
    matched_binary[too_far] = np.nan
    matched_semantic[too_far] = -1

    n_matched = (~too_far).sum()
    print(f"    Matched {n_matched:,}/{len(pred_points):,} "
          f"({100*n_matched/len(pred_points):.1f}%) in {dt:.1f}s")
    return matched_binary, matched_semantic, ~too_far


def print_ece_result(label, result, indent="  "):
    print(f"{indent}{label}: ECE={result['ece']:.4f}  MCE={result['mce']:.4f}  "
          f"Brier={result['brier']:.4f}")


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "results", "flatforest_10cm"
    )
    results_dir = os.path.abspath(results_dir)
    resolution = 0.10  # metres
    out_file = os.path.join(results_dir, "flatforest_calibration_metrics.txt")

    print(f"=== Flatforest Calibration Analysis ===")
    print(f"Results dir: {results_dir}")
    print()

    # Load GT
    gt = np.load(os.path.join(results_dir, "gt_voxels.npz"))
    gt_points = gt["points"]
    gt_binary = gt["gt_binary"]
    gt_semantic = gt["semantic_class"]
    print(f"GT: {len(gt_points):,} points, {int(gt_binary.sum()):,} occupied, "
          f"labels={sorted(set(gt_semantic))}")

    # GT surface points for unknown-vs-ambiguous (occupied voxels only)
    gt_surface_points = gt_points[gt_binary > 0.5]
    print(f"GT surface points: {len(gt_surface_points):,}")
    print()

    lines = []  # accumulate output lines

    def log(s=""):
        print(s)
        lines.append(s)

    log("=" * 70)
    log("FLATFOREST CALIBRATION METRICS")
    log(f"Resolution: {resolution}m")
    log(f"GT: {len(gt_points):,} points, {int(gt_binary.sum()):,} occupied")
    log("=" * 70)

    # ──────────────────────────────────────────────────────────────────
    # Process each variant
    # ──────────────────────────────────────────────────────────────────
    variants = [
        ("scovox", True, True),   # (name, has_beta_params, has_semantics)
        ("covox", True, False),
        ("logodds", False, False),
    ]

    for name, has_beta, has_sem in variants:
        npz_path = os.path.join(results_dir, f"{name}.npz")
        if not os.path.exists(npz_path):
            log(f"\n--- {name.upper()}: SKIPPED (file not found) ---")
            continue

        pred = np.load(npz_path)
        pred_points = pred["points"]
        pred_prob = pred["occupancy_prob"]

        log(f"\n{'─'*70}")
        log(f"  {name.upper()}: {len(pred_points):,} voxels")
        log(f"{'─'*70}")

        # Match to GT
        print(f"  Matching {name} to GT...")
        matched_binary, matched_semantic, valid_mask = match_to_gt(
            pred_points, gt_points, gt_binary, gt_semantic, max_dist=resolution
        )

        # Filter to valid matches
        v_prob = pred_prob[valid_mask]
        v_gt = matched_binary[valid_mask]
        n_valid = valid_mask.sum()

        log(f"  Matched: {n_valid:,} voxels")
        log(f"  GT occupied in matched: {int(v_gt.sum()):,} "
            f"({100*v_gt.mean():.1f}%)")

        # ── Task 4.6: Occupancy ECE ──────────────────────────────────
        log(f"\n  OCCUPANCY CALIBRATION (task 4.6):")
        overall = compute_ece(v_prob, v_gt, n_bins=20)
        log(f"    Overall:  ECE={overall['ece']:.4f}  "
            f"MCE={overall['mce']:.4f}  Brier={overall['brier']:.4f}")

        if has_beta:
            v_a_occ = pred["a_occ"][valid_mask]
            v_a_free = pred["a_free"][valid_mask]
            evidence = v_a_occ + v_a_free - 2  # subtract prior

            strat = compute_stratified_ece(v_prob, v_gt, evidence, n_bins=20)
            for stratum, res in strat.items():
                n_s = int((strat_mask := (
                    (evidence < 5) if "low" in stratum else
                    ((evidence >= 5) & (evidence <= 20)) if "medium" in stratum else
                    (evidence > 20)
                )).sum())
                log(f"    {stratum:20s}: ECE={res['ece']:.4f}  "
                    f"MCE={res['mce']:.4f}  Brier={res['brier']:.4f}  "
                    f"n={n_s:,}")

            # ── Task 4.7: Unknown vs Ambiguous ────────────────────────
            log(f"\n  UNKNOWN vs AMBIGUOUS (task 4.7):")
            uva = compute_unknown_vs_ambiguous(
                pred_points[valid_mask], v_prob,
                v_a_occ, v_a_free,
                gt_surface_points, resolution,
                prob_lo=0.4, prob_hi=0.6,
            )
            log(f"    ROC-AUC: {uva['auc']:.4f}")
            log(f"    Uncertain voxels (p∈[0.4,0.6]): {uva['n_total_uncertain']:,}")
            log(f"    Unknown  (far from surface): n={uva['n_unknown']:,}"
                + (f"  var_mean={uva['variance_unknown_mean']:.6f}  "
                   f"evidence_mean={uva['evidence_unknown_mean']:.1f}"
                   if 'variance_unknown_mean' in uva else ""))
            log(f"    Boundary (near surface):      n={uva['n_boundary']:,}"
                + (f"  var_mean={uva['variance_boundary_mean']:.6f}  "
                   f"evidence_mean={uva['evidence_boundary_mean']:.1f}"
                   if 'variance_boundary_mean' in uva else ""))
        else:
            # LogOdds: no stratification, but run unknown-vs-ambiguous baseline
            log(f"\n  UNKNOWN vs AMBIGUOUS (task 4.7) — LogOdds baseline:")
            uva = compute_logodds_baseline(
                pred_points[valid_mask], v_prob,
                gt_surface_points, resolution,
                prob_lo=0.4, prob_hi=0.6,
            )
            log(f"    ROC-AUC: {uva['auc']:.4f}  (expected ~0.50)")
            log(f"    Uncertain voxels: {uva['n_total_uncertain']:,}")
            log(f"    Unknown: n={uva['n_unknown']:,}  "
                f"Boundary: n={uva['n_boundary']:,}")

        # ── Task 4.6b: Semantic ECE ──────────────────────────────────
        if has_sem and "semantic_class" in pred.files and "semantic_confidence" in pred.files:
            log(f"\n  SEMANTIC CALIBRATION (task 4.6b):")

            v_sem_cls = pred["semantic_class"][valid_mask]
            v_sem_conf = pred["semantic_confidence"][valid_mask]
            v_gt_sem = matched_semantic[valid_mask]

            # Filter to voxels with valid GT semantic label (>0)
            sem_valid = (v_gt_sem > 0) & (v_sem_cls > 0)
            if sem_valid.sum() > 0:
                correct = (v_sem_cls[sem_valid] == v_gt_sem[sem_valid]).astype(float)
                conf = v_sem_conf[sem_valid]

                sem_overall = compute_semantic_ece(conf, correct, n_bins=15)
                log(f"    Overall:  ECE={sem_overall['semantic_ece']:.4f}  "
                    f"MCE={sem_overall['semantic_mce']:.4f}  "
                    f"n={sem_valid.sum():,}")

                # Accuracy
                acc = correct.mean()
                mean_conf = conf.mean()
                log(f"    Accuracy: {acc:.4f}  Mean confidence: {mean_conf:.4f}  "
                    f"Overconfidence: {mean_conf - acc:+.4f}")

                # Stratify by evidence (use a_occ as proxy for observation count)
                if has_beta:
                    alpha0_proxy = pred["a_occ"][valid_mask][sem_valid]
                    strat_sem = compute_stratified_semantic_ece(
                        conf, correct, alpha0_proxy, n_bins=15
                    )
                    for stratum, res in strat_sem.items():
                        log(f"    {stratum:20s}: ECE={res['semantic_ece']:.4f}  "
                            f"n={res['n_voxels']:,}")

                # Per-class accuracy
                log(f"    Per-class accuracy:")
                for cls in sorted(set(v_gt_sem[sem_valid])):
                    cls_mask = v_gt_sem[sem_valid] == cls
                    cls_correct = correct[cls_mask].mean()
                    cls_conf = conf[cls_mask].mean()
                    log(f"      class {int(cls):2d}: acc={cls_correct:.4f}  "
                        f"conf={cls_conf:.4f}  n={int(cls_mask.sum()):,}")
            else:
                log(f"    No voxels with valid GT semantic labels")

    # ──────────────────────────────────────────────────────────────────
    # Save results
    # ──────────────────────────────────────────────────────────────────
    log(f"\n{'='*70}")
    with open(out_file, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nResults saved to: {out_file}")


if __name__ == "__main__":
    main()
