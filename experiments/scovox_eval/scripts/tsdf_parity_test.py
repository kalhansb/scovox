#!/usr/bin/env python3
"""TsdfMap vs SLIM-VDB voxel-set parity test (KITTI).

Resolves the two known parity gaps between SCovox and SLIM-VDB on KITTI:

  (a) World-frame mismatch
        SCovox: T_world_velo = pose @ Tr   (cam0_world)
        SLIM:   T_world_velo = Tr_inv @ pose @ Tr  (velo_world)
        Conversion: x_slim = Tr_inv @ x_scovox.

  (b) min_range mismatch
        SCovox launch default 1.0 m; SLIM-VDB kitti.yaml 5.0 m.
        Resolved at the SCovox side by re-running the producer with
        min_range:=5.0 (no code change — this script just consumes the dump).

The remaining DDA difference (Bresenham RayIterator vs OpenVDB DDA) and
the SemBeta-vs-TSDF subset asymmetry (the regular ~/pointcloud walk only
emits SemBeta voxels, missing the +trunc TSDF tail) are NOT addressed
here — instead we consume the new ``tsdf_dump_path`` snapshot which
walks TsdfMap directly via ``forEachVoxel``.

Inputs
------
- ``--tsdf_bin``  : flat binary written by scovox_node when the
                    ``tsdf_dump_path`` param is set (split mode only).
                    Format: uint64_t n; then n × {f32 x, f32 y, f32 z,
                    f32 distance, f32 weight} = 20 B/voxel; voxel-centre
                    coords in scovox_node's pose@Tr (cam0_world) frame.
- ``--voxels_bin``: SLIM-VDB's ``vdb_to_voxels`` output for the same
                    sequence (16 B/voxel: f32 wx, f32 wy, f32 wz,
                    int32 class) — already in Tr_inv@pose@Tr (velo_world)
                    frame.
- ``--calib``     : KITTI calib.txt for the sequence (pulls Tr).

Output
------
Per-section voxel-set IoU + asymmetric containment. Prints a single
table; no files written.
"""
import argparse
from pathlib import Path

import numpy as np


def read_tr(calib_file: Path) -> np.ndarray:
    """Read the velodyne→cam0 4×4 rigid Tr from a KITTI calib.txt."""
    Tr = np.eye(4, dtype=np.float64)
    with calib_file.open() as f:
        for line in f:
            if line.startswith("Tr:"):
                vals = np.array(
                    [float(x) for x in line.split()[1:]], dtype=np.float64
                )
                Tr[:3, :4] = vals.reshape(3, 4)
                return Tr
    raise ValueError(f"Tr line not found in {calib_file}")


def load_tsdf_dump(path: Path) -> np.ndarray:
    """Returns (N, 5) float32 array: [x, y, z, distance, weight]."""
    raw = np.fromfile(path, dtype=np.uint8)
    n = int(np.frombuffer(raw[:8], dtype=np.uint64)[0])
    body = raw[8:].view(np.float32).reshape(-1, 5)
    if body.shape[0] != n:
        raise ValueError(
            f"voxel count mismatch in {path}: header={n} body={body.shape[0]}"
        )
    return body


def load_slim_voxels_bin(path: Path) -> np.ndarray:
    """Returns (N, 4) float32 array: [x, y, z, class]."""
    raw = np.fromfile(path, dtype=np.uint8)
    rec = raw.view(np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("c", "<i4"),
    ]))
    return np.stack(
        [rec["x"], rec["y"], rec["z"], rec["c"].astype(np.float32)], axis=1,
    )


def voxel_keys(xyz: np.ndarray, voxel_size: float = 0.10) -> np.ndarray:
    """Pack floor-quantised (i, j, k) coords into 64-bit voxel keys."""
    ix = np.floor(xyz[:, 0] / voxel_size).astype(np.int64)
    iy = np.floor(xyz[:, 1] / voxel_size).astype(np.int64)
    iz = np.floor(xyz[:, 2] / voxel_size).astype(np.int64)
    mask = (1 << 24) - 1
    return ((ix & mask) << 48) | ((iy & mask) << 24) | (iz & mask)


def report(label: str, A: set, B: set) -> None:
    inter = A & B
    union = A | B
    iou = len(inter) / len(union) if union else float("nan")
    A_in_B = len(inter) / len(A) if A else float("nan")
    B_in_A = len(inter) / len(B) if B else float("nan")
    print(f"  {label}")
    print(
        f"    |A|={len(A):>10,}  |B|={len(B):>10,}  "
        f"|A∩B|={len(inter):>10,}  IoU={iou:.4f}"
    )
    print(
        f"    A-in-B={A_in_B:.4f}  B-in-A={B_in_A:.4f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tsdf_bin", type=Path, required=True,
                    help="SCovox TsdfMap snapshot from tsdf_dump_path")
    ap.add_argument("--voxels_bin", type=Path, required=True,
                    help="SLIM-VDB voxels.bin from vdb_to_voxels")
    ap.add_argument("--calib", type=Path, required=True,
                    help="KITTI calib.txt for the sequence")
    ap.add_argument("--voxel_size", type=float, default=0.10)
    ap.add_argument("--min_weight", type=float, default=0.0,
                    help="Drop SCovox voxels with weight below this "
                         "(matches SLIM-VDB's --min_weight Prune/Extract). "
                         "Default 0 = no filter (Prune is disabled in "
                         "SLIM-VDB's KITTI config; voxels.bin is unfiltered).")
    args = ap.parse_args()

    Tr = read_tr(args.calib)
    Tr_inv = np.linalg.inv(Tr).astype(np.float32)

    sc = load_tsdf_dump(args.tsdf_bin)
    print(f"SCovox TsdfMap dump: {sc.shape[0]:,} voxels "
          f"({args.tsdf_bin})")
    if args.min_weight > 0:
        keep = sc[:, 4] >= args.min_weight
        print(f"  filter weight >= {args.min_weight}: keep "
              f"{int(keep.sum()):,} ({keep.mean()*100:.1f}%)")
        sc = sc[keep]
    sc_xyz = sc[:, :3]

    sv = load_slim_voxels_bin(args.voxels_bin)
    print(f"SLIM-VDB voxels.bin: {sv.shape[0]:,} voxels "
          f"({args.voxels_bin})")
    sv_xyz = sv[:, :3]

    # Apply Tr_inv to SCovox xyz: x_velo_world = Tr_inv @ x_cam0_world.
    ones = np.ones((sc_xyz.shape[0], 1), dtype=np.float32)
    hom = np.concatenate([sc_xyz, ones], axis=1)
    sc_xyz_velo = (Tr_inv @ hom.T).T[:, :3]

    print()
    print("Bbox check after frame conversion (min, max):")
    for axis, idx in (("x", 0), ("y", 1), ("z", 2)):
        sc_min, sc_max = sc_xyz_velo[:, idx].min(), sc_xyz_velo[:, idx].max()
        sv_min, sv_max = sv_xyz[:, idx].min(), sv_xyz[:, idx].max()
        print(f"  {axis}: SCovox=[{sc_min:7.2f},{sc_max:7.2f}]  "
              f"SLIM=[{sv_min:7.2f},{sv_max:7.2f}]")

    print()
    print("Voxel-set overlap (10 cm grid, both in velo_world frame):")
    sc_keys = set(voxel_keys(sc_xyz_velo, args.voxel_size).tolist())
    sv_keys = set(voxel_keys(sv_xyz, args.voxel_size).tolist())
    report("(a) all voxels — SCovox TsdfMap vs SLIM-VDB voxels.bin",
           sc_keys, sv_keys)


if __name__ == "__main__":
    main()
