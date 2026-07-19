#!/usr/bin/env python3
"""Aggregate SceneNet E1 per-traj curve dumps (e1_score.npz) into one table:
per-traj calibration (all + intermediate-p), sparsification AUSE, and a pooled
row. Recomputes from the saved (p,y,s,eig,var) so it's robust to md formatting."""
from pathlib import Path
import numpy as np
import score_e1 as S

RES = Path("/home/kalhan/Projects/scovox_ws/experiments/results/scenenet_e1")
# 13 SceneNet validation trajectories (staged under data/scenenet_val_layout)
SEQS = ["0_175", "0_178", "0_182", "0_223", "0_279", "0_485", "0_490",
        "0_571", "0_682", "0_723", "0_789", "0_867", "0_977"]


def load(seq):
    f = RES / seq / "e1_score.npz"
    return np.load(f) if f.exists() else None


def main():
    rows = ["| traj | n(occ/free) | occ% | ECE all | ECE 0.2<p<0.8 | Brier | NLL | AUSE Var | AUSE EIG |",
            "|---|---|---|---|---|---|---|---|---|"]
    pool_p, pool_y, pool_s, pool_eig, pool_var = [], [], [], [], []
    for seq in SEQS:
        d = load(seq)
        if d is None:
            rows.append(f"| {seq} | – | – | – | – | – | – | – | – |"); continue
        p, y, s = d["p"], d["y"], d["s"]
        eig, var = d["eig"], d["var"]
        pool_p.append(p); pool_y.append(y); pool_s.append(s)
        pool_eig.append(eig); pool_var.append(var)
        m = S.metrics(p, y)
        inter = (p > 0.2) & (p < 0.8)
        ece_i = f"{S.metrics(p[inter], y[inter])['ece']:.4f}" if inter.sum() > 20 else "–"
        err = ((p > 0.5) != (y > 0.5)).astype(float)
        av = S.ause(var, err)[0]; ae = S.ause(eig, err)[0]
        rows.append(f"| {seq} | {m['n']:,} | {m['occ_frac']*100:.0f} | {m['ece']:.4f} | "
                    f"{ece_i} | {m['brier']:.4f} | {m['nll']:.3f} | {av:.4f} | {ae:.4f} |")
    if pool_p:
        p = np.concatenate(pool_p); y = np.concatenate(pool_y)
        eig = np.concatenate(pool_eig); var = np.concatenate(pool_var)
        m = S.metrics(p, y)
        inter = (p > 0.2) & (p < 0.8); mi = S.metrics(p[inter], y[inter])
        err = ((p > 0.5) != (y > 0.5)).astype(float)
        av = S.ause(var, err)[0]; ae = S.ause(eig, err)[0]
        rows.append(f"| **pooled** | **{m['n']:,}** | **{m['occ_frac']*100:.0f}** | "
                    f"**{m['ece']:.4f}** | **{mi['ece']:.4f}** | **{m['brier']:.4f}** | "
                    f"**{m['nll']:.3f}** | **{av:.4f}** | **{ae:.4f}** |")
    out = RES / "e1_scenenet_table.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# E1 SceneNet occupancy calibration (split substrate, 13 val trajectories)\n\n"
                   + "\n".join(rows) + "\n")
    print("\n".join(rows))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
