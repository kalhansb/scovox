#!/usr/bin/env python3
"""Register the SceneNN trajectory world frame onto the annotated mesh frame.

Why this exists: scenenn_check_relpose.py shows trajectory.log reproduces the
inter-frame motion to ~1.3 cm, so replayed depth is self-consistent and SCovox
can map from it directly. But the same frames land ~30 cm off the annotated
mesh, i.e. trajectory.log and <id>_nyu.ply do not share a world frame. Semantic
labelling needs them in one frame, so we solve for the rigid transform once,
offline, and reuse it for every frame.

Outputs a JSON with T_mesh_from_traj (4x4) plus the achieved residual, consumed
by scenenn_make_labels.py.

Usage:
  python3 scenenn_register_mesh.py --scene /home/jetsondevkit/datasets/scenenn/016 \
      --intrinsics /home/jetsondevkit/datasets/scenenn/asus.ini \
      --out /home/jetsondevkit/datasets/scenenn/016/mesh_align.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scovox_eval.scenenn_data import (  # noqa: E402
    deproject, load_annotated_ply, load_depth_m, load_intrinsics, load_trajectory, to_world,
)


def kabsch(P, Q):
    """Rigid transform mapping P onto Q (both (N,3)), least squares."""
    cp, cq = P.mean(0), Q.mean(0)
    H = (P - cp).T @ (Q - cq)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = cq - R @ cp
    return T


def icp(src, tree, T0, per_stage=8, thresholds=(0.60, 0.40, 0.25, 0.15, 0.08, 0.05), verbose=False):
    """Point-to-point ICP with a coarse-to-fine outlier gate.

    Cost is dominated by tree.query, so keep `src` in the tens of thousands --
    ICP converges just as well on a 40k subsample as on 250k, and 250k made a
    single restart take longer than the whole registration should.
    """
    T = T0.copy()
    med = np.inf
    for thr in thresholds:
        for _ in range(per_stage):
            cur = src @ T[:3, :3].T + T[:3, 3]
            dist, idx = tree.query(cur, k=1, workers=-1)
            keep = dist < thr
            if keep.sum() < 50:
                break
            T = kabsch(cur[keep], tree.data[idx[keep]]) @ T
            med = np.median(dist[keep])
        if verbose:
            print(f"      thr={thr:.2f} -> inlier median {med:.4f} m "
                  f"({keep.sum()}/{len(src)} pts)", flush=True)
    cur = src @ T[:3, :3].T + T[:3, 3]
    dist, _ = tree.query(cur, k=1, workers=-1)
    return T, np.median(dist), float(np.mean(dist < 0.05))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--intrinsics", required=True)
    ap.add_argument("--mesh", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-frames", type=int, default=40)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--pose-offset", type=int, default=0)
    ap.add_argument("--max-points", type=int, default=40000)
    ap.add_argument("--per-frame", action="store_true",
                    help="after the global fit, track each frame onto the mesh and write "
                         "mesh_align_perframe.npz (needed whenever the trajectory drifts)")
    ap.add_argument("--frame-stride", type=int, default=6,
                    help="pixel stride for the per-frame tracking clouds")
    args = ap.parse_args()

    scene = Path(args.scene)
    mesh_path = Path(args.mesh) if args.mesh else next(scene.glob("*_nyu.ply"))
    K = load_intrinsics(args.intrinsics)
    poses = load_trajectory(scene / "trajectory.log")
    df = sorted((scene / "frames" / "depth").glob("depth*.png"))

    xyz, _, _ = load_annotated_ply(mesh_path)
    tree = cKDTree(xyz)
    print(f"mesh {mesh_path.name}: {len(xyz)} vertices")

    n_avail = min(len(df), len(poses) - args.pose_offset)
    picks = np.linspace(0, n_avail - 1, args.n_frames).astype(int)
    clouds = []
    for fi in picks:
        d = load_depth_m(df[fi], flip_vertical=False)
        pts, _ = deproject(d, K, stride=args.stride)
        if len(pts):
            clouds.append(to_world(pts, poses[fi + args.pose_offset]))
    src = np.concatenate(clouds)
    if len(src) > args.max_points:
        sel = np.random.default_rng(0).choice(len(src), args.max_points, replace=False)
        src = src[sel]
    print(f"trajectory-world cloud: {len(src)} points from {len(clouds)} frames")

    d0, _ = tree.query(src, k=1, workers=-1)
    print(f"before ICP: median {np.median(d0):.4f} m, inliers<5cm {np.mean(d0 < 0.05):.1%}\n")

    # Yaw restarts about the mesh up-axis (SceneNN rooms are Y-up) in case the
    # two frames differ by more than ICP's basin of convergence.
    best = None
    c_src, c_dst = src.mean(0), xyz.mean(0)
    for yaw_deg in (0, 90, 180, 270):
        a = np.deg2rad(yaw_deg)
        R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        T0 = np.eye(4)
        T0[:3, :3] = R
        T0[:3, 3] = c_dst - R @ c_src
        print(f"  yaw {yaw_deg:>3}deg:", flush=True)
        T, med, inl = icp(src, tree, T0, verbose=True)
        print(f"  yaw {yaw_deg:>3}deg -> median {med:.4f} m, inliers<5cm {inl:.1%}", flush=True)
        if best is None or med < best[1]:
            best = (T, med, inl)

    T, med, inl = best
    print(f"\nBEST: median {med:.4f} m, inliers<5cm {inl:.1%}")
    print("T_mesh_from_traj =\n", np.array2string(T, precision=6, suppress_small=True))

    Path(args.out).write_text(json.dumps({
        "T_mesh_from_traj": T.tolist(),
        "median_residual_m": float(med),
        "inlier_frac_5cm": float(inl),
        "n_frames": int(len(clouds)),
        "pose_offset": args.pose_offset,
        "mesh": mesh_path.name,
    }, indent=2))
    print(f"\nwrote {args.out}")

    if args.per_frame:
        _per_frame(scene, df, poses, K, tree, T, args)


def _track(fi, df, poses, K, tree, T_init, args):
    """One frame onto the mesh, seeded from T_init."""
    depth = load_depth_m(df[fi], flip_vertical=False)
    pts, _ = deproject(depth, K, stride=args.frame_stride)
    if len(pts) < 200:
        return None
    world = to_world(pts, poses[fi + args.pose_offset])
    return icp(world, tree, T_init, per_stage=4, thresholds=(0.20, 0.10, 0.05))


def _per_frame(scene, df, poses, K, tree, T_global, args):
    """Track every frame onto the mesh.

    A single rigid fit cannot absorb trajectory drift -- on scene 016 the global
    transform leaves 0.077 m median and only 36% inliers, while tracking frame
    to frame holds ~0.006 m at ~100%. Two passes (forward, then backward keeping
    whichever seed scored better) so the first frames, which have only the poor
    global transform to start from, get repaired from their good neighbours.
    """
    n = min(len(df), len(poses) - args.pose_offset)
    Ts = np.repeat(T_global[None], n, axis=0)
    meds = np.full(n, np.inf)
    inls = np.zeros(n)

    for label, order in (("forward", range(n)), ("backward", range(n - 1, -1, -1))):
        seed = T_global
        for k, fi in enumerate(order):
            got = _track(fi, df, poses, K, tree, seed, args)
            if got is None:
                continue
            T_i, med_i, inl_i = got
            if inl_i > inls[fi]:
                Ts[fi], meds[fi], inls[fi] = T_i, med_i, inl_i
            seed = T_i if inl_i > 0.5 else Ts[fi]
            if k % 100 == 0:
                print(f"  {label} {k}/{n}: med {med_i:.4f} m inl {inl_i:.1%}", flush=True)

    out = Path(args.out).with_name("mesh_align_perframe.npz")
    np.savez_compressed(out, T=Ts, median_residual_m=meds, inlier_frac_5cm=inls,
                        pose_offset=args.pose_offset)
    print(f"\nper-frame: median residual {np.median(meds):.4f} m, "
          f"mean inliers<5cm {inls.mean():.1%}, "
          f"frames below 80% inliers: {(inls < 0.8).sum()}/{n}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
