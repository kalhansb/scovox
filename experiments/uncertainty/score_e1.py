#!/usr/bin/env python3
"""E1 scorer: occupancy calibration + decomposition on the captured split map
against the raycast free/occupied reference.

Addresses assessment issue #4 (reference circularity): ECE is reported OVERALL
and STRATIFIED BY EVIDENCE s = a_occ+a_free, because aggregate ECE is dominated
by trivially-classified bulk voxels (p_occ~0/1). The interesting calibration is
at low/intermediate evidence.

Inputs:
  --map   map_bin.npz  (beta_xyz int32 Nx3, a_occ, a_free)
  --ref   ref npz      (coord int32 Mx3, hit uint32, free uint32)
Outputs: markdown table + npz of curve data for figures.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import functionals as F

EPS = 1e-6


def pack(c):
    x = c[:, 0].astype(np.int64); y = c[:, 1].astype(np.int64); z = c[:, 2].astype(np.int64)
    return ((x & 0xFFFFFF) << 40) | ((y & 0xFFFFF) << 20) | (z & 0xFFFFF)


def reliability(p, y, nbins=15):
    """Equal-mass bins on p. Returns mean_p, emp_freq, weight per bin, ECE, MCE."""
    order = np.argsort(p)
    p, y = p[order], y[order]
    idx = np.array_split(np.arange(len(p)), nbins)
    mp, ef, w = [], [], []
    for ii in idx:
        if len(ii) == 0:
            continue
        mp.append(p[ii].mean()); ef.append(y[ii].mean()); w.append(len(ii) / len(p))
    mp, ef, w = np.array(mp), np.array(ef), np.array(w)
    ece = float(np.sum(w * np.abs(mp - ef)))
    mce = float(np.max(np.abs(mp - ef))) if len(mp) else float("nan")
    return mp, ef, w, ece, mce


def metrics(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    brier = float(np.mean((p - y) ** 2))
    nll = float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))
    _, _, _, ece, mce = reliability(p, y)
    acc = float(np.mean((p > 0.5) == (y > 0.5)))
    return dict(n=len(y), occ_frac=float(y.mean()), acc=acc,
                ece=ece, mce=mce, brier=brier, nll=nll)


def ause(signal, err):
    """Area under sparsification error, normalized vs oracle. Reject highest
    `signal` first; measure mean err on retained. Lower AUSE = better ranking."""
    n = len(signal)
    fr = np.linspace(0, 0.99, 50)
    def curve(rank_desc):
        order = np.argsort(-rank_desc)   # highest signal first (rejected first)
        e = err[order]
        out = []
        for f in fr:
            k = int(f * n)
            out.append(e[k:].mean() if k < n else 0.0)
        return np.array(out)
    model = curve(signal)
    oracle = curve(err.astype(float))    # oracle ranks by true error
    # normalize by random (no rejection) baseline area
    denom = err.mean() if err.mean() > 0 else 1.0
    d = np.diff(fr)
    area = float(np.sum((model[1:] + model[:-1] - oracle[1:] - oracle[:-1]) / 2 * d))
    return area / denom, fr, model, oracle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--free-m", type=int, default=3)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    m = np.load(args.map); r = np.load(args.ref)
    mk = pack(m["beta_xyz"]); a_occ = m["a_occ"].astype(float); a_free = m["a_free"].astype(float)
    rk = pack(r["coord"]); hit = r["hit"]; free = r["free"]

    # reference label: occupied(1)=hit>=1 ; free(0)=hit==0 & free>=m ; else excluded
    ref_lab = np.full(len(rk), -1, np.int8)
    ref_lab[hit >= 1] = 1
    ref_lab[(hit == 0) & (free >= args.free_m)] = 0
    keep = ref_lab >= 0
    rk, ref_lab = rk[keep], ref_lab[keep]

    # align map <-> reference by voxel key
    order = np.argsort(mk); mk_s = mk[order]
    pos = np.searchsorted(mk_s, rk)
    valid = (pos < len(mk_s)) & (mk_s[np.clip(pos, 0, len(mk_s) - 1)] == rk)
    ridx = np.where(valid)[0]
    midx = order[pos[valid]]
    y = ref_lab[ridx].astype(float)
    ao, af = a_occ[midx], a_free[midx]
    s = ao + af
    p = ao / np.where(s > 0, s, 1.0)
    eig = F.expected_information_gain(ao, af)
    e_h = F.beta_expected_entropy(ao, af)
    var = F.beta_variance(ao, af)

    cover = valid.mean()
    lines = [f"# E1 occupancy calibration {args.tag}".rstrip(), "",
             f"- reference voxels (occ/free): {len(rk):,}  "
             f"(occupied {int((ref_lab==1).sum()):,}, free {int((ref_lab==0).sum()):,})",
             f"- matched in map: {len(y):,} ({cover:.1%} of reference)",
             f"- map beta voxels: {len(mk):,}", ""]

    ov = metrics(p, y)
    lines += ["## Calibration (overall + stratified by evidence s=a_occ+a_free)", "",
              "| stratum | n | occ% | acc | ECE | MCE | Brier | NLL |",
              "|---|---|---|---|---|---|---|---|"]
    def row(tag, sel):
        if sel.sum() < 20:
            return f"| {tag} | {int(sel.sum())} | – | – | – | – | – | – |"
        mm = metrics(p[sel], y[sel])
        return (f"| {tag} | {mm['n']:,} | {mm['occ_frac']*100:.0f} | {mm['acc']:.3f} | "
                f"{mm['ece']:.4f} | {mm['mce']:.4f} | {mm['brier']:.4f} | {mm['nll']:.3f} |")
    lines.append(row("all", np.ones(len(y), bool)))
    qs = np.quantile(s, [0.25, 0.5, 0.75])
    lines.append(row(f"s<={qs[0]:.1f} (Q1)", s <= qs[0]))
    lines.append(row(f"{qs[0]:.1f}<s<={qs[1]:.1f}", (s > qs[0]) & (s <= qs[1])))
    lines.append(row(f"{qs[1]:.1f}<s<={qs[2]:.1f}", (s > qs[1]) & (s <= qs[2])))
    lines.append(row(f"s>{qs[2]:.1f} (Q4)", s > qs[2]))
    lines.append(row("intermediate 0.2<p<0.8", (p > 0.2) & (p < 0.8)))

    # sparsification of occupancy misclassification by variance and EIG
    err = ((p > 0.5) != (y > 0.5)).astype(float)
    ause_var, fr, mv, orc = ause(var, err)
    ause_eig, _, me, _ = ause(eig, err)
    lines += ["", "## Sparsification (occupancy misclassification)",
              f"- AUSE by Var[p_occ]: **{ause_var:.4f}**   (0 = oracle ranking)",
              f"- AUSE by EIG:        **{ause_eig:.4f}**",
              f"- base error rate:    {err.mean():.4f}"]

    # EIG / E_H vs evidence (cross-sectional; H1.1/H1.2 signature)
    sb = np.quantile(s, np.linspace(0, 1, 9))
    lines += ["", "## EIG (epistemic) and E_H (aleatoric) vs evidence s",
              "| s-bin | median s | median EIG | median E_H |",
              "|---|---|---|---|"]
    for i in range(len(sb) - 1):
        sel = (s >= sb[i]) & (s < sb[i + 1]) if i < len(sb) - 2 else (s >= sb[i])
        if sel.sum() < 20:
            continue
        lines.append(f"| {i} | {np.median(s[sel]):.1f} | {np.median(eig[sel]):.4f} | "
                     f"{np.median(e_h[sel]):.4f} |")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    np.savez_compressed(str(Path(args.out).with_suffix(".npz")),
                        p=p, y=y, s=s, eig=eig, e_h=e_h, var=var,
                        rel_fr=fr, spars_var=mv, spars_eig=me, spars_oracle=orc)
    print("\n".join(lines))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
