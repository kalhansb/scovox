#!/usr/bin/env python3
"""Render class-ID semantic PNGs for Replica along NICE-SLAM trajectories.

Inputs:
  - replica_v1/<scene>/habitat/mesh_semantic.ply + info_semantic.json
  - data/replica_niceslam/<scene>/poses.txt    (NICE-SLAM traj, 4x4 c2w row-major)
  - data/replica_niceslam/cam_params.json      (1200x680, fx=fy=600, hfov=90)

Outputs per scene:
  - data/replica_niceslam/<scene>/semantic_gt_fixed/NNNNNN.png   uint16 class IDs
  - (optional --also-rgb)  semantic_gt_fixed_rgb/NNNNNN.jpg for visual diff
  - (optional --also-depth) semantic_gt_fixed_depth/NNNNNN.png for geometry diff

Coordinate convention: NICE-SLAM's traj.txt was itself rendered by Habitat, so
poses are already in Habitat agent convention (OpenGL: y-up, -z forward). We
pass position + quaternion straight through.

Run with the `habitat` conda env:
  conda run -n habitat python render_replica_semantic.py --scene office0 \
      --frames 0 500 1000 1500 --preview
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

import habitat_sim
from habitat_sim.utils.common import quat_from_magnum
import magnum as mn


SCENE_MAP = {
    "room0": "room_0", "room1": "room_1", "room2": "room_2",
    "office0": "office_0", "office1": "office_1", "office2": "office_2",
    "office3": "office_3", "office4": "office_4",
}
DEPTH_SCALE_OUT = 6553.5  # match existing NICE-SLAM depth encoding


def read_poses(path: Path) -> np.ndarray:
    poses = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        vals = list(map(float, line.split()))
        poses.append(np.asarray(vals, dtype=np.float64).reshape(4, 4))
    return np.stack(poses, axis=0)


def read_cam_params(path: Path) -> dict:
    return json.loads(path.read_text())["camera"]


def build_lut(info_semantic_path: Path) -> np.ndarray:
    info = json.loads(info_semantic_path.read_text())
    id_to_label = info["id_to_label"]
    obj_max = max(o["id"] for o in info["objects"])
    n = max(len(id_to_label), obj_max + 1)
    lut = np.zeros(n, dtype=np.uint16)
    for i, c in enumerate(id_to_label):
        lut[i] = c if c > 0 else 0
    return lut


def make_sim(scene_path: Path, H: int, W: int, hfov_deg: float) -> habitat_sim.Simulator:
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(scene_path)
    sim_cfg.enable_physics = False

    def sensor(uuid: str, stype: habitat_sim.SensorType) -> habitat_sim.CameraSensorSpec:
        s = habitat_sim.CameraSensorSpec()
        s.uuid = uuid
        s.sensor_type = stype
        s.resolution = [H, W]
        s.position = [0.0, 0.0, 0.0]
        s.orientation = [0.0, 0.0, 0.0]
        s.hfov = hfov_deg
        return s

    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.sensor_specifications = [
        sensor("rgb", habitat_sim.SensorType.COLOR),
        sensor("depth", habitat_sim.SensorType.DEPTH),
        sensor("semantic", habitat_sim.SensorType.SEMANTIC),
    ]

    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


# NICE-SLAM traj.txt: c2w, camera axes OpenCV (y-down, z-fwd), world axes Replica
# native (z-up, gravity along -Z). Habitat expects: camera axes OpenGL
# (y-up, -z-fwd), world axes y-up (gravity along -Y). Compose two transforms:
#   (a) world rotation: Replica world (z-up) -> Habitat world (y-up): Rx(-90°)
#   (b) camera-axis flip: OpenCV -> OpenGL: diag(1,-1,-1,1) applied on the right.
_WORLD_REPLICA_TO_HABITAT = np.array([
    [1.0,  0.0, 0.0, 0.0],
    [0.0,  0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0,  0.0, 0.0, 1.0],
])
_CAM_CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])


def pose_to_agent_state(T_c2w_cv: np.ndarray) -> habitat_sim.AgentState:
    T_c2w = _WORLD_REPLICA_TO_HABITAT @ T_c2w_cv @ _CAM_CV_TO_GL
    state = habitat_sim.AgentState()
    state.position = T_c2w[:3, 3].astype(np.float32)
    R = T_c2w[:3, :3]
    mat = mn.Matrix3(
        mn.Vector3(*R[:, 0].tolist()),
        mn.Vector3(*R[:, 1].tolist()),
        mn.Vector3(*R[:, 2].tolist()),
    )
    state.rotation = quat_from_magnum(mn.Quaternion.from_matrix(mat))
    return state


def render_scene(scene: str, replica_v1_root: Path, niceslam_root: Path,
                 out_subdir: str, frames: Optional[List[int]],
                 also_rgb: bool, also_depth: bool) -> None:
    v1_name = SCENE_MAP[scene]
    mesh_path = replica_v1_root / v1_name / "habitat" / "mesh_semantic.ply"
    info_path = replica_v1_root / v1_name / "habitat" / "info_semantic.json"
    scene_dir = niceslam_root / scene
    poses_path = scene_dir / "poses.txt"
    cam_path = niceslam_root / "cam_params.json"

    for p, desc in [(mesh_path, "mesh_semantic.ply"), (info_path, "info_semantic.json"),
                    (poses_path, "poses.txt"), (cam_path, "cam_params.json")]:
        if not p.exists():
            raise FileNotFoundError(f"{desc} not found at {p}")

    cam = read_cam_params(cam_path)
    H, W = cam["h"], cam["w"]
    hfov = float(np.degrees(2.0 * np.arctan(W / (2.0 * cam["fx"]))))
    poses = read_poses(poses_path)
    n_frames = poses.shape[0]
    indices = frames if frames else list(range(n_frames))

    lut = build_lut(info_path)

    sem_out = scene_dir / out_subdir
    sem_out.mkdir(parents=True, exist_ok=True)
    rgb_out = scene_dir / f"{out_subdir}_rgb" if also_rgb else None
    depth_out = scene_dir / f"{out_subdir}_depth" if also_depth else None
    if rgb_out: rgb_out.mkdir(parents=True, exist_ok=True)
    if depth_out: depth_out.mkdir(parents=True, exist_ok=True)

    sim = make_sim(mesh_path, H, W, hfov)
    try:
        for i in indices:
            if i >= n_frames:
                print(f"  skip {i}: only {n_frames} poses")
                continue
            sim.agents[0].set_state(pose_to_agent_state(poses[i]))
            obs = sim.get_sensor_observations()
            sem_obj = obs["semantic"].astype(np.int64)
            cls = lut[np.clip(sem_obj, 0, len(lut) - 1)].astype(np.uint16)
            cv2.imwrite(str(sem_out / f"{i:06d}.png"), cls)
            if rgb_out is not None:
                rgb = obs["rgb"][..., :3][..., ::-1]  # RGBA->BGR
                cv2.imwrite(str(rgb_out / f"{i:06d}.jpg"), rgb)
            if depth_out is not None:
                d_m = obs["depth"].astype(np.float32)
                d_u16 = np.clip(d_m * DEPTH_SCALE_OUT, 0, 65535).astype(np.uint16)
                cv2.imwrite(str(depth_out / f"{i:06d}.png"), d_u16)
            if i % 200 == 0:
                print(f"  {scene}: frame {i}/{n_frames}")
    finally:
        sim.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--replica-v1-root", type=Path,
                   default=Path("data/replica_v1"))
    p.add_argument("--niceslam-root", type=Path,
                   default=Path("data/replica_niceslam"))
    p.add_argument("--scenes", nargs="+",
                   default=list(SCENE_MAP.keys()), choices=list(SCENE_MAP.keys()))
    p.add_argument("--frames", nargs="+", type=int, default=None,
                   help="Specific frame indices; default = all")
    p.add_argument("--out-subdir", default="semantic_gt_fixed")
    p.add_argument("--also-rgb", action="store_true",
                   help="Also save rendered RGB for visual sanity check")
    p.add_argument("--also-depth", action="store_true",
                   help="Also save rendered depth for geometry sanity check")
    args = p.parse_args()

    for scene in args.scenes:
        print(f"\n=== {scene} ===")
        render_scene(scene, args.replica_v1_root, args.niceslam_root,
                     args.out_subdir, args.frames, args.also_rgb, args.also_depth)


if __name__ == "__main__":
    main()
