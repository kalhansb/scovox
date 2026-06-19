#!/usr/bin/env python3
"""Generate camera trajectories for the flatforest simulation world.

Outputs poses.json in the same format as the Replica pipeline so the
recording can be replayed through replica_replay_node → SCovox.

Usage:
    python generate_trajectory.py --type lawnmower --output /tmp/traj/poses.json
    python generate_trajectory.py --type random --num-frames 500 --output /tmp/traj/poses.json
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def rotation_matrix_to_quat_wxyz(R):
    """Convert 3x3 rotation matrix to [w, x, y, z] quaternion."""
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
    return [w, x, y, z]


def yaw_pitch_to_T(x, y, z, yaw, pitch=0.0):
    """Create a 4x4 transform from position + yaw + pitch (radians).

    In Gazebo Fortress, a model with a camera sensor:
    - Default orientation: camera looks along +X
    - Yaw: rotation around Z axis
    - Pitch: rotation around Y axis (positive pitch = tilt down toward ground)

    We compose R = Rz(yaw) @ Ry(pitch) where positive pitch tilts
    the +X forward vector downward (nose down).
    """
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    # Rz(yaw) @ Ry(pitch): positive pitch tilts +X axis downward
    R = np.array([
        [cy * cp,  -sy,   cy * sp],
        [sy * cp,   cy,   sy * sp],
        [-sp,        0,   cp     ],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[0, 3] = x;  T[1, 3] = y;  T[2, 3] = z
    return T


def make_pose(frame_id, x, y, z, yaw, pitch=0.0):
    """Build a pose dict matching the Replica pipeline format."""
    T = yaw_pitch_to_T(x, y, z, yaw, pitch)
    R = T[:3, :3]
    quat = rotation_matrix_to_quat_wxyz(R)
    return {
        "frame_id": frame_id,
        "position": [float(x), float(y), float(z)],
        "rotation_quat_wxyz": [float(q) for q in quat],
        "T_world_camera": T.tolist(),
    }


def generate_lawnmower(bounds, height, spacing, step_size, pitch=0.4):
    """Lawnmower (boustrophedon) sweep pattern.

    The camera faces forward with a downward pitch to see ground + objects.
    pitch: radians, positive = look down (default 0.4 rad ≈ 23°).
    """
    x_min, y_min, x_max, y_max = bounds
    poses = []
    frame_id = 0
    y = y_min
    forward = True

    while y <= y_max:
        if forward:
            x_range = np.arange(x_min, x_max, step_size)
            yaw = 0.0
        else:
            x_range = np.arange(x_max, x_min, -step_size)
            yaw = math.pi

        for x in x_range:
            poses.append(make_pose(frame_id, x, y, height, yaw, pitch))
            frame_id += 1

        # Turn: add a few frames looking in the turn direction
        turn_yaw = math.pi / 2 if forward else -math.pi / 2
        poses.append(make_pose(frame_id, x_range[-1], y, height, turn_yaw, pitch))
        frame_id += 1

        y += spacing
        forward = not forward

    return poses


def generate_random_walk(bounds, height, num_frames, seed=42, pitch=0.4):
    """Random walk with smooth heading changes."""
    rng = np.random.RandomState(seed)
    x_min, y_min, x_max, y_max = bounds
    poses = []

    # Start at centre
    x = (x_min + x_max) / 2
    y = (y_min + y_max) / 2
    yaw = rng.uniform(-math.pi, math.pi)

    step = 0.5  # metres per frame
    max_turn = 0.2  # radians per frame

    for i in range(num_frames):
        poses.append(make_pose(i, x, y, height, yaw, pitch))

        # Step forward
        nx = x + step * math.cos(yaw)
        ny = y + step * math.sin(yaw)

        # Bounce off bounds
        if nx < x_min or nx > x_max:
            yaw = math.pi - yaw
            nx = np.clip(nx, x_min, x_max)
        if ny < y_min or ny > y_max:
            yaw = -yaw
            ny = np.clip(ny, y_min, y_max)

        x, y = nx, ny
        yaw += rng.uniform(-max_turn, max_turn)

    return poses


def main():
    parser = argparse.ArgumentParser(
        description="Generate camera trajectories for flatforest simulation"
    )
    parser.add_argument("--type", choices=["lawnmower", "random"],
                        default="lawnmower")
    parser.add_argument("--bounds", type=float, nargs=4,
                        default=[-45, -45, 45, 45],
                        metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"))
    parser.add_argument("--height", type=float, default=1.5,
                        help="Camera height (m)")
    parser.add_argument("--spacing", type=float, default=3.0,
                        help="Line spacing for lawnmower (m)")
    parser.add_argument("--step-size", type=float, default=0.5,
                        help="Step size along each line (m)")
    parser.add_argument("--num-frames", type=int, default=500,
                        help="Number of frames for random walk")
    parser.add_argument("--pitch", type=float, default=0.4,
                        help="Downward pitch in radians (default: 0.4 ≈ 23°)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True,
                        help="Output poses.json path")
    args = parser.parse_args()

    if args.type == "lawnmower":
        poses = generate_lawnmower(
            args.bounds, args.height, args.spacing, args.step_size, args.pitch
        )
    elif args.type == "random":
        poses = generate_random_walk(
            args.bounds, args.height, args.num_frames, args.seed, args.pitch
        )

    # Save
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(poses, f, indent=2)

    print(f"Generated {len(poses)} poses ({args.type})")
    print(f"Bounds: {args.bounds}, height={args.height}m")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
