#!/usr/bin/env python3
"""Generate RGB-D frames + GT poses from Replica scenes using Habitat-sim.

Usage (run inside the 'habitat' conda env):
    python -m scovox_eval.replica_data_generator \
        --scene /path/to/replica_v1/room_0/habitat/mesh_semantic.ply \
        --output /path/to/datasets/replica_rendered/room_0 \
        --num-frames 300 --width 640 --height 480 --hfov 60

Output structure:
    <output>/
        color/000000.npy  ... (H, W, 3) uint8 RGB
        depth/000000.npy  ... (H, W) float32 meters
        semantic/000000.npy ... (H, W) int32 semantic instance IDs
        poses.json        ... list of {frame_id, position, rotation_quat_wxyz, T_world_camera(4x4)}
        camera.json       ... {width, height, hfov, fx, fy, cx, cy}
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

# habitat_sim is only available in the conda 'habitat' env
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis


def make_sim(scene_path: str, width: int, height: int, hfov: int):
    """Create a Habitat simulator with RGB, depth, and semantic sensors."""
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_path
    backend_cfg.enable_physics = False
    backend_cfg.gpu_device_id = 0

    rgb_sensor = habitat_sim.CameraSensorSpec()
    rgb_sensor.uuid = "color"
    rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor.resolution = [height, width]
    rgb_sensor.hfov = hfov
    rgb_sensor.position = [0.0, 1.5, 0.0]  # sensor at 1.5m height

    depth_sensor = habitat_sim.CameraSensorSpec()
    depth_sensor.uuid = "depth"
    depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor.resolution = [height, width]
    depth_sensor.hfov = hfov
    depth_sensor.position = [0.0, 1.5, 0.0]

    semantic_sensor = habitat_sim.CameraSensorSpec()
    semantic_sensor.uuid = "semantic"
    semantic_sensor.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor.resolution = [height, width]
    semantic_sensor.hfov = hfov
    semantic_sensor.position = [0.0, 1.5, 0.0]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_sensor, depth_sensor, semantic_sensor]
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.15)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
    }

    cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)
    return sim


def agent_state_to_T(state) -> np.ndarray:
    """Convert AgentState (position + quaternion) to a 4x4 homogeneous matrix."""
    import quaternion as qt

    q = state.rotation  # habitat quaternion (scalar-first internally)
    pos = state.position
    # Convert to scipy-style rotation matrix
    R = qt.as_rotation_matrix(q)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


def generate_trajectory(sim, num_frames: int, seed: int = 42):
    """Walk through the scene using navmesh-guided waypoints for good coverage.

    Generates a sequence of random navigable goals and follows shortest paths
    between them, capturing frames at regular intervals along the way.
    """
    rng = np.random.RandomState(seed)
    agent = sim.get_agent(0)

    if not sim.pathfinder.is_loaded:
        print("WARNING: navmesh not loaded, falling back to random walk")
        return _random_walk_trajectory(sim, num_frames, rng)

    # Find a navigable starting point
    start = sim.pathfinder.get_random_navigable_point()
    state = agent.get_state()
    state.position = start
    agent.set_state(state)

    frames = []

    while len(frames) < num_frames:
        # Pick a random navigable goal
        goal = sim.pathfinder.get_random_navigable_point()
        path = habitat_sim.ShortestPath()
        path.requested_start = agent.get_state().position
        path.requested_end = goal
        if not sim.pathfinder.find_path(path) or len(path.points) < 2:
            continue

        # Walk along the path waypoints
        for wp_idx in range(1, len(path.points)):
            if len(frames) >= num_frames:
                break

            target = path.points[wp_idx]
            state = agent.get_state()
            current = state.position

            # Face the target: compute yaw angle
            dx = target[0] - current[0]
            dz = target[2] - current[2]
            target_yaw = math.atan2(-dx, -dz)  # habitat: -Z is forward

            # Rotate agent to face target
            import quaternion as qt
            state.rotation = qt.from_euler_angles([0, target_yaw, 0])
            state.position = current
            agent.set_state(state)

            # Move towards waypoint in steps, capturing frames
            dist = np.linalg.norm(np.array(target) - np.array(current))
            n_steps = max(1, int(dist / 0.15))  # 0.15m per step

            for step in range(n_steps):
                if len(frames) >= num_frames:
                    break

                obs = sim.get_sensor_observations()
                state = agent.get_state()
                T = agent_state_to_T(state)

                pose = {
                    "frame_id": len(frames),
                    "position": state.position.tolist(),
                    "rotation_quat_wxyz": [
                        float(state.rotation.w),
                        float(state.rotation.x),
                        float(state.rotation.y),
                        float(state.rotation.z),
                    ],
                    "T_world_camera": T.tolist(),
                }

                frames.append(
                    {
                        "color": obs["color"][:, :, :3].copy(),
                        "depth": obs["depth"].copy(),
                        "semantic": obs["semantic"].astype(np.int32).copy(),
                        "pose": pose,
                    }
                )

                if len(frames) % 50 == 0:
                    print(f"  Frame {len(frames)}/{num_frames} pos={state.position}")

                # Step forward
                agent.act("move_forward")

    return frames


def save_dataset(frames, output_dir: str, width: int, height: int, hfov: int):
    """Save frames as .npy files + poses as JSON."""
    out = Path(output_dir)
    (out / "color").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(parents=True, exist_ok=True)
    (out / "semantic").mkdir(parents=True, exist_ok=True)

    poses = []
    for f in frames:
        fid = f["pose"]["frame_id"]
        np.save(out / "color" / f"{fid:06d}.npy", f["color"])
        np.save(out / "depth" / f"{fid:06d}.npy", f["depth"])
        np.save(out / "semantic" / f"{fid:06d}.npy", f["semantic"])
        poses.append(f["pose"])

    with open(out / "poses.json", "w") as fp:
        json.dump(poses, fp, indent=2)

    # Save camera intrinsics
    hfov_rad = math.radians(hfov)
    fx = width / (2.0 * math.tan(hfov_rad / 2.0))
    fy = fx  # square pixels
    cx = width / 2.0
    cy = height / 2.0

    camera = {
        "width": width,
        "height": height,
        "hfov_deg": hfov,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
    }
    with open(out / "camera.json", "w") as fp:
        json.dump(camera, fp, indent=2)

    print(f"Saved {len(frames)} frames to {out}")
    print(f"  Camera: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")


def export_semantic_class_map(sim, output_dir: str):
    """Export the scene's semantic class mapping (instance_id -> class name + color)."""
    scene = sim.semantic_scene
    class_map = {}
    for obj in scene.objects:
        if obj is not None:
            class_map[int(obj.semantic_id)] = {
                "category_name": obj.category.name() if obj.category else "unknown",
                "category_index": obj.category.index() if obj.category else -1,
            }

    out = Path(output_dir)
    with open(out / "semantic_class_map.json", "w") as fp:
        json.dump(class_map, fp, indent=2)

    print(f"Exported {len(class_map)} semantic classes")


def main():
    parser = argparse.ArgumentParser(description="Generate Replica RGB-D dataset for SCovox")
    parser.add_argument("--scene", required=True, help="Path to Replica .ply scene file")
    parser.add_argument("--output", required=True, help="Output directory for rendered frames")
    parser.add_argument("--num-frames", type=int, default=300, help="Number of frames to render")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--hfov", type=int, default=60, help="Horizontal FOV in degrees")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for trajectory")
    args = parser.parse_args()

    print(f"Loading scene: {args.scene}")
    sim = make_sim(args.scene, args.width, args.height, args.hfov)

    print(f"Generating {args.num_frames} frames...")
    frames = generate_trajectory(sim, args.num_frames, args.seed)

    print("Saving dataset...")
    save_dataset(frames, args.output, args.width, args.height, args.hfov)
    export_semantic_class_map(sim, args.output)

    sim.close()
    print("Done.")


if __name__ == "__main__":
    main()
