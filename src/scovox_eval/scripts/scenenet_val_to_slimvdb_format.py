#!/usr/bin/env python3
"""Convert raw SceneNet RGB-D val trajectories to the SLIM-VDB format used by
scenenet_replay_node.py.

Input layout (raw upstream HF mirror — `haotongl/SceneNetRGBD`):
  data/scenenet/val/0/{traj_id}/{depth,photo,instance}/{T}.png         # T = 0, 25, 50, ...
  data/scenenet/val/scenenet_rgbd_val.pb                              # trajectory metadata

Output layout (SLIM-VDB format — matches existing data/scenenet/train/2/):
  <out_root>/<seq>/depth/NNNN.png             # uint16 mm Euclidean
  <out_root>/<seq>/photo/NNNN.jpg             # RGB 320×240
  <out_root>/<seq>/ground_truth_labels/NNNN.png  # uint8 NYUv2 14-class IDs (0..13)
  <out_root>/<seq>/poses.txt                  # 300 lines, 12 floats (camera-to-world 3×4)
  <out_root>/<seq>/intrinsics.txt             # fx, fy, cx, cy

NNNN = zero-padded frame_num (still 0..7475 step 25 — matches train/2 convention).

Notes:
- depth/photo are symlinked from the raw tree to save 31 GB. ground_truth_labels
  is materialised (per-pixel instance→class mapping).
- Intrinsics are FIXED for SceneNet RGB-D (vfov=45°, hfov=60°, 320×240):
  fx=277.128, fy=289.706, cx=160, cy=120.
- Camera pose: interpolate shutter_open ↔ shutter_close at α=0.5, then standard
  look-at math (up=[0,1,0]) to build world_to_camera, invert to camera_to_world.
- Background class (instance_id 0 or any instance_type=BACKGROUND) → 0 (Unknown).
- WNIDs missing from NYU_WNID_TO_CLASS → 0 (Unknown), with a warning.

Usage:
  ./scenenet_val_to_slimvdb_format.py --traj 0/223 --out_root data/scenenet/val_preprocessed
  ./scenenet_val_to_slimvdb_format.py --trajs 0/223,0/110,0/57 --out_root ...
  ./scenenet_val_to_slimvdb_format.py --n_random 12 --seed 42 --out_root ...
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Make scenenet_pb2 importable (generated from pySceneNetRGBD/scenenet.proto).
SCENENET_ROOT = Path(__file__).resolve().parents[5] / "data" / "scenenet"
sys.path.insert(0, str(SCENENET_ROOT / "pySceneNetRGBD"))
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
import scenenet_pb2 as sn  # noqa: E402  pylint: disable=wrong-import-position

# ------------------------------------------------------------
# WNID → NYUv2 14-class mapping (copied verbatim from
# pySceneNetRGBD/convert_instance2class.py — keep both in sync).
# ------------------------------------------------------------
NYU_WNID_TO_CLASS = {
    "04593077": 4, "03262932": 4, "02933112": 6, "03207941": 7, "03063968": 10,
    "04398044": 7, "04515003": 7, "00017222": 7, "02964075": 10, "03246933": 10,
    "03904060": 10, "03018349": 6, "03786621": 4, "04225987": 7, "04284002": 7,
    "03211117": 11, "02920259": 1, "03782190": 11, "03761084": 7, "03710193": 7,
    "03367059": 7, "02747177": 7, "03063599": 7, "04599124": 7, "20000036": 10,
    "03085219": 7, "04255586": 7, "03165096": 1, "03938244": 1, "14845743": 7,
    "03609235": 7, "03238586": 10, "03797390": 7, "04152829": 11, "04553920": 7,
    "04608329": 10, "20000016": 4, "02883344": 7, "04590933": 4, "04466871": 7,
    "03168217": 4, "03490884": 7, "04569063": 7, "03071021": 7, "03221720": 12,
    "03309808": 7, "04380533": 7, "02839910": 7, "03179701": 10, "02823510": 7,
    "03376595": 4, "03891251": 4, "03438257": 7, "02686379": 7, "03488438": 7,
    "04118021": 5, "03513137": 7, "04315948": 7, "03092883": 10, "15101854": 6,
    "03982430": 10, "02920083": 1, "02990373": 3, "03346455": 12, "03452594": 7,
    "03612814": 7, "06415419": 7, "03025755": 7, "02777927": 12, "04546855": 12,
    "20000040": 10, "20000041": 10, "04533802": 7, "04459362": 7, "04177755": 9,
    "03206908": 7, "20000021": 4, "03624134": 7, "04186051": 7, "04152593": 11,
    "03643737": 7, "02676566": 7, "02789487": 6, "03237340": 6, "04502670": 7,
    "04208936": 7, "20000024": 4, "04401088": 7, "04372370": 12, "20000025": 4,
    "03956922": 7, "04379243": 10, "04447028": 7, "03147509": 7, "03640988": 7,
    "03916031": 7, "03906997": 7, "04190052": 6, "02828884": 4, "03962852": 1,
    "03665366": 7, "02881193": 7, "03920867": 4, "03773035": 12, "03046257": 12,
    "04516116": 7, "00266645": 7, "03665924": 7, "03261776": 7, "03991062": 7,
    "03908831": 7, "03759954": 7, "04164868": 7, "04004475": 7, "03642806": 7,
    "04589593": 13, "04522168": 7, "04446276": 7, "08647616": 4, "02808440": 7,
    "08266235": 10, "03467517": 7, "04256520": 9, "04337974": 7, "03990474": 7,
    "03116530": 6, "03649674": 4, "04349401": 7, "01091234": 7, "15075141": 7,
    "20000028": 9, "02960903": 7, "04254009": 7, "20000018": 4, "20000020": 4,
    "03676759": 11, "20000022": 4, "20000023": 4, "02946921": 7, "03957315": 7,
    "20000026": 4, "20000027": 4, "04381587": 10, "04101232": 7, "03691459": 7,
    "03273913": 7, "02843684": 7, "04183516": 7, "04587648": 13, "02815950": 3,
    "03653583": 6, "03525454": 7, "03405725": 6, "03636248": 7, "03211616": 11,
    "04177820": 4, "04099969": 4, "03928116": 7, "04586225": 7, "02738535": 4,
    "20000039": 10, "20000038": 10, "04476259": 7, "04009801": 11, "03909406": 12,
    "03002711": 7, "03085602": 11, "03233905": 6, "20000037": 10, "02801938": 7,
    "03899768": 7, "04343346": 7, "03603722": 7, "03593526": 7, "02954340": 7,
    "02694662": 7, "04209613": 7, "02951358": 7, "03115762": 9, "04038727": 6,
    "03005285": 7, "04559451": 7, "03775636": 7, "03620967": 10, "02773838": 7,
    "20000008": 6, "04526964": 7, "06508816": 7, "20000009": 6, "03379051": 7,
    "04062428": 7, "04074963": 7, "04047401": 7, "03881893": 13, "03959485": 7,
    "03391301": 7, "03151077": 12, "04590263": 13, "20000006": 1, "03148324": 6,
    "20000004": 1, "04453156": 7, "02840245": 2, "04591713": 7, "03050864": 7,
    "03727837": 5, "06277280": 11, "03365592": 5, "03876519": 8, "03179910": 7,
    "06709442": 7, "03482252": 7, "04223580": 7, "02880940": 7, "04554684": 7,
    "20000030": 9, "03085013": 7, "03169390": 7, "04192858": 7, "20000029": 9,
    "04331277": 4, "03452741": 7, "03485997": 7, "20000007": 1, "02942699": 7,
    "03231368": 10, "03337140": 7, "03001627": 4, "20000011": 6, "20000010": 6,
    "20000013": 6, "04603729": 10, "20000015": 4, "04548280": 12, "06410904": 2,
    "04398951": 10, "03693474": 9, "04330267": 7, "03015149": 9, "04460038": 7,
    "03128519": 7, "04306847": 7, "03677231": 7, "02871439": 6, "04550184": 6,
    "14974264": 7, "04344873": 9, "03636649": 7, "20000012": 6, "02876657": 7,
    "03325088": 7, "04253437": 7, "02992529": 7, "03222722": 12, "04373704": 4,
    "02851099": 13, "04061681": 10, "04529681": 7,
}

# Fixed intrinsics — SceneNet RGB-D, vfov=45°, hfov=60°, 320×240.
FX = 160.0 / math.tan(math.radians(30.0))   # 277.1281292…
FY = 120.0 / math.tan(math.radians(22.5))   # 289.7056274…
CX = 160.0
CY = 120.0


def _vec3(p):
    return np.array([p.x, p.y, p.z], dtype=np.float64)


def _interp(pose_a, pose_b, alpha=0.5):
    cam = alpha * _vec3(pose_b.camera) + (1.0 - alpha) * _vec3(pose_a.camera)
    look = alpha * _vec3(pose_b.lookat) + (1.0 - alpha) * _vec3(pose_a.lookat)
    return cam, look


def _camera_to_world(camera, lookat):
    """Match pySceneNetRGBD world_to_camera_with_pose() then invert."""
    up = np.array([0.0, 1.0, 0.0])
    R = np.eye(4)
    R[2, :3] = (lookat - camera) / np.linalg.norm(lookat - camera)
    R[0, :3] = np.cross(R[2, :3], up); R[0, :3] /= np.linalg.norm(R[0, :3])
    R[1, :3] = -np.cross(R[0, :3], R[2, :3]); R[1, :3] /= np.linalg.norm(R[1, :3])
    T = np.eye(4)
    T[:3, 3] = -camera
    world_to_cam = R @ T
    return np.linalg.inv(world_to_cam)


def _instance_to_class_map(trajectory):
    """Build {instance_id: nyuv2_class} dict for one trajectory.

    Background, missing wnids, and instance types we don't know about all → 0.
    """
    mapping = {0: 0}  # instance 0 is always BACKGROUND
    missing = []
    for inst in trajectory.instances:
        if inst.instance_type == sn.Instance.BACKGROUND:
            mapping[inst.instance_id] = 0
            continue
        wnid = inst.semantic_wordnet_id
        if wnid in NYU_WNID_TO_CLASS:
            mapping[inst.instance_id] = NYU_WNID_TO_CLASS[wnid]
        else:
            missing.append((inst.instance_id, wnid, inst.semantic_english))
            mapping[inst.instance_id] = 0
    return mapping, missing


def _render_instance_to_class(instance_png_path, class_map):
    """Per-pixel instance ID → NYUv2 class (uint8 PNG)."""
    inst_img = np.asarray(Image.open(instance_png_path))
    # Vectorised LUT: max instance_id is ~50, build a lookup array
    max_id = int(inst_img.max())
    lut = np.zeros(max(max_id + 1, max(class_map.keys()) + 1), dtype=np.uint8)
    for iid, cls in class_map.items():
        if iid < lut.size:
            lut[iid] = cls
    return lut[inst_img]


def _convert_one(traj, raw_root: Path, out_dir: Path, link_imgs=True, verbose=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "depth").mkdir(exist_ok=True)
    (out_dir / "photo").mkdir(exist_ok=True)
    (out_dir / "ground_truth_labels").mkdir(exist_ok=True)

    raw_traj_dir = raw_root / "val" / traj.render_path  # e.g. data/scenenet/val/0/223
    if not raw_traj_dir.exists():
        raise FileNotFoundError(f"raw trajectory dir not found: {raw_traj_dir}")

    class_map, missing = _instance_to_class_map(traj)
    if verbose and missing:
        print(f"  [{traj.render_path}] WARN: {len(missing)} instances with unmapped wnids "
              f"(first 3: {missing[:3]})")

    # Poses + intrinsics
    poses_lines = []
    n_frames = len(traj.views)
    for view in traj.views:
        cam, look = _interp(view.shutter_open, view.shutter_close, 0.5)
        c2w = _camera_to_world(cam, look)  # 4×4
        row = c2w[:3, :4].reshape(-1)      # 12 floats, row-major
        poses_lines.append(" ".join(f"{v:.6f}" for v in row))

    (out_dir / "poses.txt").write_text("\n".join(poses_lines) + "\n")
    (out_dir / "intrinsics.txt").write_text(
        f"fx: {FX}\nfy: {FY}\ncx: {CX}\ncy: {CY}\n"
    )

    # Per-frame: ground_truth_labels + symlinked depth + symlinked photo
    for i, view in enumerate(traj.views):
        T = view.frame_num
        nnnn = f"{T:04d}"

        raw_depth = raw_traj_dir / "depth" / f"{T}.png"
        raw_photo = raw_traj_dir / "photo" / f"{T}.jpg"
        raw_inst  = raw_traj_dir / "instance" / f"{T}.png"
        assert raw_depth.exists() and raw_photo.exists() and raw_inst.exists(), \
            f"missing raw frame {T} for {traj.render_path}"

        # depth + photo: symlink to save 31 GB
        out_depth = out_dir / "depth" / f"{nnnn}.png"
        out_photo = out_dir / "photo" / f"{nnnn}.jpg"
        out_label = out_dir / "ground_truth_labels" / f"{nnnn}.png"

        if link_imgs:
            for src, dst in [(raw_depth, out_depth), (raw_photo, out_photo)]:
                if dst.is_symlink() or dst.exists():
                    dst.unlink()
                dst.symlink_to(src.resolve())
        else:
            import shutil
            shutil.copyfile(raw_depth, out_depth)
            shutil.copyfile(raw_photo, out_photo)

        # ground_truth_labels: materialise (per-pixel LUT)
        cls_img = _render_instance_to_class(raw_inst, class_map)
        Image.fromarray(cls_img).save(out_label)

        if verbose and (i % 50 == 0 or i == n_frames - 1):
            print(f"  [{traj.render_path}] frame {i+1}/{n_frames} (T={T})")

    if verbose:
        print(f"  [{traj.render_path}] done → {out_dir}")


def load_trajectories(pb_path: Path):
    trajs = sn.Trajectories()
    with open(pb_path, "rb") as f:
        trajs.ParseFromString(f.read())
    return trajs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_root", required=True,
                   help="Output root, e.g. data/scenenet/val_preprocessed/  (one subdir per traj)")
    p.add_argument("--raw_root", default=str(SCENENET_ROOT),
                   help="SceneNet raw root (default: data/scenenet)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--traj", help="Single trajectory render_path, e.g. 0/223")
    g.add_argument("--trajs", help="Comma-separated render_paths, e.g. 0/223,0/110,0/57")
    g.add_argument("--n_random", type=int, help="Number of trajectories to pick at random")
    p.add_argument("--seed", type=int, default=42, help="Random seed when --n_random")
    p.add_argument("--copy", action="store_true",
                   help="Copy depth/photo instead of symlinking (slower, +30 GB)")
    args = p.parse_args()

    raw_root = Path(args.raw_root)
    pb_path = raw_root / "val" / "scenenet_rgbd_val.pb"
    if not pb_path.exists():
        sys.exit(f"FATAL: {pb_path} not found")

    print(f"loading {pb_path} ...")
    trajs = load_trajectories(pb_path)
    by_path = {t.render_path: t for t in trajs.trajectories}
    print(f"  {len(trajs.trajectories)} trajectories available")

    # Build the work list
    if args.traj:
        wanted = [args.traj]
    elif args.trajs:
        wanted = [s.strip() for s in args.trajs.split(",") if s.strip()]
    else:
        rng = np.random.default_rng(args.seed)
        wanted = list(rng.choice(sorted(by_path.keys()), size=args.n_random, replace=False))
    print(f"  converting {len(wanted)} trajectories: {wanted[:5]}{'…' if len(wanted) > 5 else ''}")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for rp in wanted:
        if rp not in by_path:
            print(f"  SKIP {rp} (not in protobuf)")
            continue
        # Safe directory name: "0/223" → "0_223"
        seq_name = rp.replace("/", "_")
        out_dir = out_root / seq_name
        if (out_dir / "poses.txt").exists() and (out_dir / "ground_truth_labels").exists():
            n_labels = len(list((out_dir / "ground_truth_labels").glob("*.png")))
            if n_labels == 300:
                print(f"  SKIP {rp} (already converted: {out_dir})")
                continue
        print(f"\n--- converting {rp} → {out_dir} ---")
        _convert_one(by_path[rp], raw_root, out_dir, link_imgs=not args.copy)

    print("\nall done.")


if __name__ == "__main__":
    main()
