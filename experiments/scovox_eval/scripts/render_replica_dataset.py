#!/usr/bin/env python3
"""Package a clean, self-contained Replica dataset from existing sources.

Combines three existing data sources into one folder:
  - Depth PNGs from NICE-SLAM         (Replica/{scene}/results/depth*.png)
  - Color JPGs from NICE-SLAM         (Replica/{scene}/results/frame*.jpg)
  - Semantic NPYs from Habitat-sim    (replica_rendered/{scene}/semantic/*.npy)
    -> converted to compact uint16 PNGs
  - Trajectories from NICE-SLAM       (Replica/{scene}/traj.txt)
  - GT meshes from NICE-SLAM          (Replica/{scene}_mesh.ply)
  - Semantic info from Replica v1      (replica_v1/{scene}/habitat/info_semantic.json)
  - Camera params from NICE-SLAM       (Replica/cam_params.json)

Output:
  replica_niceslam/{scene}/
    depth/000000.png          uint16 PNG (meters * 6553.5), copied from NICE-SLAM
    color/000000.jpg          copied from NICE-SLAM
    semantic/000000.png       uint16 PNG (per-pixel object_id), converted from .npy
    poses.txt                 copied from NICE-SLAM traj.txt
    mesh.ply                  copied GT mesh
    info_semantic.json        object_id -> class_name mapping
  replica_niceslam/cam_params.json

Usage:
    python render_replica_dataset.py                     # all 8 scenes
    python render_replica_dataset.py --scenes room0      # single scene
    python render_replica_dataset.py --dry-run            # preview only
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import cv2
import numpy as np


SCENE_MAP = {
    "room0": "room_0",
    "room1": "room_1",
    "room2": "room_2",
    "office0": "office_0",
    "office1": "office_1",
    "office2": "office_2",
    "office3": "office_3",
    "office4": "office_4",
}

ALL_SCENES = list(SCENE_MAP.keys())

DEPTH_SCALE = 6553.5


def process_scene(
    scene_name: str,
    niceslam_root: Path,
    rendered_root: Path,
    replica_v1_root: Path,
    output_root: Path,
    max_frames: int = -1,
):
    """Package one Replica scene from existing data sources."""
    v1_name = SCENE_MAP[scene_name]
    niceslam_dir = niceslam_root / scene_name
    rendered_dir = rendered_root / scene_name
    v1_dir = replica_v1_root / v1_name / "habitat"
    out_dir = output_root / scene_name

    # Source paths
    traj_path = niceslam_dir / "traj.txt"
    color_dir = niceslam_dir / "results"
    depth_dir = niceslam_dir / "results"
    semantic_dir = rendered_dir / "semantic"
    info_path = v1_dir / "info_semantic.json"
    gt_mesh = niceslam_root / f"{scene_name}_mesh.ply"
    cam_path = niceslam_root / "cam_params.json"

    # Validate inputs
    sources = [
        (traj_path, "trajectory"),
        (color_dir, "color images"),
        (depth_dir, "depth images"),
        (semantic_dir, "semantic labels"),
        (info_path, "semantic info"),
        (gt_mesh, "GT mesh"),
        (cam_path, "camera params"),
    ]
    for p, desc in sources:
        if not p.exists():
            print(f"  ERROR: {desc} not found at {p}")
            return False

    # Count frames
    n_traj = sum(1 for line in open(traj_path) if line.strip())
    n_sem = len(list(semantic_dir.glob("*.npy")))
    n_depth = len(list(depth_dir.glob("depth*.png")))
    n_color = len(list(color_dir.glob("frame*.jpg")))
    n_frames = min(n_traj, n_sem, n_depth, n_color)
    if max_frames > 0:
        n_frames = min(n_frames, max_frames)
    print(f"  Frames: {n_frames} (traj={n_traj}, sem={n_sem}, depth={n_depth}, color={n_color})")

    # Create output directories
    (out_dir / "depth").mkdir(parents=True, exist_ok=True)
    (out_dir / "color").mkdir(parents=True, exist_ok=True)
    (out_dir / "semantic").mkdir(parents=True, exist_ok=True)

    # Copy metadata files
    shutil.copy2(info_path, out_dir / "info_semantic.json")
    shutil.copy2(traj_path, out_dir / "poses.txt")
    shutil.copy2(gt_mesh, out_dir / "mesh.ply")

    # Process frames
    t0 = time.time()
    for i in range(n_frames):
        # Depth: copy PNG directly (already uint16, same encoding)
        src_depth = depth_dir / f"depth{i:06d}.png"
        dst_depth = out_dir / "depth" / f"{i:06d}.png"
        if not dst_depth.exists():
            shutil.copy2(src_depth, dst_depth)

        # Color: copy JPG directly
        src_color = color_dir / f"frame{i:06d}.jpg"
        dst_color = out_dir / "color" / f"{i:06d}.jpg"
        if not dst_color.exists():
            shutil.copy2(src_color, dst_color)

        # Semantic: convert .npy (int32, 3.3 MB) -> uint16 PNG (~18 KB)
        dst_sem = out_dir / "semantic" / f"{i:06d}.png"
        if not dst_sem.exists():
            sem_npy = np.load(semantic_dir / f"{i:06d}.npy")
            sem_uint16 = sem_npy.astype(np.uint16)
            cv2.imwrite(str(dst_sem), sem_uint16)

        if i % 200 == 0 or i == n_frames - 1:
            elapsed = time.time() - t0
            fps = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  Frame {i+1}/{n_frames} ({fps:.0f} fps)")

    elapsed = time.time() - t0
    print(f"  Done: {n_frames} frames in {elapsed:.1f}s")
    return True


def verify_scene(scene_name: str, output_root: Path):
    """Quick integrity check on a packaged scene."""
    out_dir = output_root / scene_name

    n_depth = len(list((out_dir / "depth").glob("*.png")))
    n_color = len(list((out_dir / "color").glob("*.jpg")))
    n_sem = len(list((out_dir / "semantic").glob("*.png")))
    n_poses = sum(1 for line in open(out_dir / "poses.txt") if line.strip())
    has_mesh = (out_dir / "mesh.ply").exists()
    has_info = (out_dir / "info_semantic.json").exists()

    ok = (n_depth == n_color == n_sem == n_poses) and has_mesh and has_info
    status = "OK" if ok else "MISMATCH"

    print(f"  VERIFY {scene_name}: depth={n_depth} color={n_color} sem={n_sem} "
          f"poses={n_poses} mesh={'Y' if has_mesh else 'N'} info={'Y' if has_info else 'N'} [{status}]")

    # Check semantic coverage
    if n_sem > 0:
        n_with_labels = 0
        for sf in sorted((out_dir / "semantic").glob("*.png"))[:n_sem]:
            s = cv2.imread(str(sf), cv2.IMREAD_UNCHANGED)
            if np.any(s > 0):
                n_with_labels += 1
        pct = 100 * n_with_labels / n_sem
        print(f"  VERIFY {scene_name}: {n_with_labels}/{n_sem} frames have semantic labels ({pct:.0f}%)")

    # Spot-check depth
    depth_files = sorted((out_dir / "depth").glob("*.png"))
    if depth_files:
        d = cv2.imread(str(depth_files[0]), cv2.IMREAD_UNCHANGED)
        d_m = d.astype(np.float32) / DEPTH_SCALE
        valid = d_m > 0
        print(f"  VERIFY {scene_name}: depth range [{d_m[valid].min():.2f}, {d_m[valid].max():.2f}]m, "
              f"{valid.sum()}/{d.size} valid pixels")

    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Package clean Replica dataset from existing sources"
    )
    parser.add_argument(
        "--niceslam-root",
        default=os.path.expanduser("~/Projects/datasets/Replica"),
        help="Path to NICE-SLAM Replica data (depth, color, traj, meshes)",
    )
    parser.add_argument(
        "--rendered-root",
        default=os.path.expanduser("~/Projects/datasets/replica_rendered"),
        help="Path to Habitat-sim rendered semantic labels",
    )
    parser.add_argument(
        "--replica-v1-root",
        default=os.path.expanduser("~/Projects/datasets/replica_v1"),
        help="Path to Replica v1 source assets (info_semantic.json)",
    )
    parser.add_argument(
        "--output",
        default=os.path.expanduser("~/Projects/datasets/replica_niceslam"),
        help="Output directory",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=ALL_SCENES,
        choices=ALL_SCENES,
        help="Scenes to process (default: all 8)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=-1, help="Max frames per scene (-1 = all)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate inputs only, don't copy"
    )
    parser.add_argument(
        "--verify-only", action="store_true", help="Only verify existing output"
    )
    args = parser.parse_args()

    niceslam_root = Path(args.niceslam_root)
    rendered_root = Path(args.rendered_root)
    v1_root = Path(args.replica_v1_root)
    output_root = Path(args.output)

    print("=" * 60)
    print("  Replica Dataset Packager")
    print(f"  NICE-SLAM:       {niceslam_root}")
    print(f"  Habitat-sim sem: {rendered_root}")
    print(f"  Replica v1:      {v1_root}")
    print(f"  Output:          {output_root}")
    print(f"  Scenes:          {args.scenes}")
    print("=" * 60)

    if args.verify_only:
        for scene in args.scenes:
            verify_scene(scene, output_root)
        return

    if args.dry_run:
        for scene in args.scenes:
            v1_name = SCENE_MAP[scene]
            traj = niceslam_root / scene / "traj.txt"
            sem_dir = rendered_root / scene / "semantic"
            n_poses = sum(1 for _ in open(traj)) if traj.exists() else 0
            n_sem = len(list(sem_dir.glob("*.npy"))) if sem_dir.exists() else 0
            mesh = niceslam_root / f"{scene}_mesh.ply"
            info = v1_root / v1_name / "habitat" / "info_semantic.json"
            print(f"  {scene}: poses={n_poses} sem={n_sem} "
                  f"mesh={'OK' if mesh.exists() else 'MISSING'} "
                  f"info={'OK' if info.exists() else 'MISSING'}")
        print("\nDry run complete. Remove --dry-run to generate.")
        return

    # Copy cam_params.json to output root
    output_root.mkdir(parents=True, exist_ok=True)
    cam_src = niceslam_root / "cam_params.json"
    if cam_src.exists():
        shutil.copy2(cam_src, output_root / "cam_params.json")

    total_t0 = time.time()
    results = {}
    for scene in args.scenes:
        print(f"\n{'─' * 40}")
        print(f"  SCENE: {scene}")
        print(f"{'─' * 40}")
        ok = process_scene(
            scene, niceslam_root, rendered_root, v1_root, output_root, args.max_frames
        )
        results[scene] = ok
        if ok:
            verify_scene(scene, output_root)

    total_elapsed = time.time() - total_t0
    print(f"\n{'=' * 60}")
    print(f"  COMPLETE: {sum(results.values())}/{len(results)} scenes OK "
          f"in {total_elapsed / 60:.1f} min")
    for scene, ok in results.items():
        print(f"    {scene}: {'OK' if ok else 'FAILED'}")

    # Report sizes
    total_bytes = sum(
        f.stat().st_size
        for f in output_root.rglob("*")
        if f.is_file()
    )
    print(f"  Output: {output_root} ({total_bytes / 1e9:.1f} GB)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
