"""Convert NICE-SLAM Replica dataset format to SCovox replay format.

NICE-SLAM provides:
  - traj.txt: 2000 lines, each a flattened 4×4 camera-to-world matrix
  - results/depth000000.png ... depth001999.png (16-bit PNG, scale=6553.5)
  - results/frame000000.jpg ... frame001999.jpg (RGB)
  - cam_params.json: {camera: {w, h, fx, fy, cx, cy, scale}}

SCovox replay expects:
  - poses.json: list of {position, rotation_quat_wxyz, T_world_camera}
  - depth/000000.npy ... (float32, meters)
  - color/000000.npy ... (uint8, H×W×3)
  - semantic/000000.npy ... (int32, H×W) — set to 0 (unknown) since no GT semantics
  - camera.json: {fx, fy, cx, cy, width, height}

Usage:
    python -m scovox_eval.niceslam_to_scovox \
        --input /path/to/datasets/Replica/room0 \
        --cam-params /path/to/datasets/Replica/cam_params.json \
        --output /path/to/datasets/replica_rendered/room_0
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def load_traj(path: Path) -> list[np.ndarray]:
    """Load NICE-SLAM trajectory: each line is a flattened 4×4 matrix."""
    matrices = []
    with open(path) as f:
        for line in f:
            vals = [float(v) for v in line.strip().split()]
            T = np.array(vals).reshape(4, 4)
            matrices.append(T)
    return matrices


def rotation_matrix_to_quat_wxyz(R: np.ndarray) -> list[float]:
    """Convert 3×3 rotation matrix to quaternion [w, x, y, z]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return [float(w), float(x), float(y), float(z)]


def main():
    parser = argparse.ArgumentParser(description="Convert NICE-SLAM Replica to SCovox format")
    parser.add_argument("--input", required=True, help="NICE-SLAM scene dir (e.g. Replica/room0)")
    parser.add_argument("--cam-params", required=True, help="Path to cam_params.json")
    parser.add_argument("--output", required=True, help="Output dir for SCovox format")
    parser.add_argument("--max-frames", type=int, default=2000, help="Max frames to convert")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    # Load camera params
    with open(args.cam_params) as f:
        cam = json.load(f)["camera"]
    depth_scale = cam["scale"]

    # Load trajectory
    traj = load_traj(input_dir / "traj.txt")
    n_frames = min(len(traj), args.max_frames)
    print(f"Converting {n_frames} frames from {input_dir}")

    # Create output dirs
    for subdir in ["depth", "color", "semantic"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Save camera.json
    camera = {
        "fx": cam["fx"],
        "fy": cam["fy"],
        "cx": cam["cx"],
        "cy": cam["cy"],
        "width": cam["w"],
        "height": cam["h"],
    }
    with open(output_dir / "camera.json", "w") as f:
        json.dump(camera, f, indent=2)

    # Convert frames
    poses = []
    for i in range(n_frames):
        # Depth: 16-bit PNG → float32 meters
        depth_path = input_dir / "results" / f"depth{i:06d}.png"
        try:
            from PIL import Image
            depth_png = np.array(Image.open(depth_path), dtype=np.float32)
        except ImportError:
            import cv2
            depth_png = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth_m = depth_png / depth_scale
        np.save(output_dir / "depth" / f"{i:06d}.npy", depth_m)

        # RGB
        rgb_path = input_dir / "results" / f"frame{i:06d}.jpg"
        try:
            from PIL import Image
            rgb = np.array(Image.open(rgb_path))
        except ImportError:
            import cv2
            rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
        np.save(output_dir / "color" / f"{i:06d}.npy", rgb)

        # Semantic: no GT semantics in NICE-SLAM format, fill with zeros
        semantic = np.zeros((cam["h"], cam["w"]), dtype=np.int32)
        np.save(output_dir / "semantic" / f"{i:06d}.npy", semantic)

        # Pose: camera-to-world 4×4 matrix
        T = traj[i]
        pos = T[:3, 3].tolist()
        R = T[:3, :3]
        quat_wxyz = rotation_matrix_to_quat_wxyz(R)

        poses.append({
            "frame_id": i,
            "position": pos,
            "rotation_quat_wxyz": quat_wxyz,
            "T_world_camera": T.tolist(),
        })

        if i % 200 == 0:
            print(f"  Frame {i}/{n_frames}")

    # Save poses
    with open(output_dir / "poses.json", "w") as f:
        json.dump(poses, f)

    # Save dummy semantic class map (all unknown)
    with open(output_dir / "semantic_class_map.json", "w") as f:
        json.dump({}, f)

    print(f"Done. Saved {n_frames} frames to {output_dir}")
    print(f"  Camera: fx={cam['fx']} fy={cam['fy']} cx={cam['cx']} cy={cam['cy']}")
    print(f"  Resolution: {cam['w']}x{cam['h']}")


if __name__ == "__main__":
    main()
