#!/usr/bin/env python3
"""Render paper-quality 3-panel qualitative comparisons: GT | SCovox | SLIM-VDB.

Voxel-cube rendering via Open3D's EGL OffscreenRenderer (Filament). One
panel per system at a fixed shared camera, then PIL-stitched with titles.

Inputs are auto-discovered from the canonical experiment-output layout:
  - SceneNet GT:    data/scenenet/val_preprocessed/<seq>/gt_5cm.npz
  - SCovox NPZ:     .../phase1_ktop_sweep_2026_05_14/K2/(scenenet_<seq>|kitti_<seq>_soft)/scovox.npz
  - SLIM-VDB:       third_party_sw/slim_vdb/outputs/(scenenet_val|kitti)/<seq>/voxels.bin
  - KITTI GT:       built on-the-fly from data/semantickitti/dataset

Usage
-----
  source install/setup.bash   # not actually needed; pure-python
  python3 render_qualitative_compare.py --scene scenenet:0_223
  python3 render_qualitative_compare.py --scene kitti:08
  python3 render_qualitative_compare.py --scene scenenet:0_223 --scene kitti:08 \\
      --out_dir figures/qualitative_2026_05_15
"""
from __future__ import annotations

# ---- Open3D 0.19 + scipy version conflict: pre-block open3d.ml / _ml3d ----
import sys as _sys

class _OpenMLBlocker:
    def find_module(self, name, path=None):
        if name.startswith("open3d.ml") or name.startswith("open3d._ml3d"):
            return self
        return None

    def load_module(self, name):
        m = type(_sys)(name)
        _sys.modules[name] = m
        return m


_sys.meta_path.insert(0, _OpenMLBlocker())
# ---------------------------------------------------------------------------

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import open3d as o3d
from open3d.visualization import rendering
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Class palettes
# ---------------------------------------------------------------------------

# SceneNet RGB-D (NYUv2-13 + unknown). 14 classes total. Colors chosen for
# legibility against the white render background and to match the SceneNet
# RGB-D paper's qualitative figures where possible.
SCENENET_PALETTE = np.array([
    [0,   0,   0  ],   # 0  unknown
    [255, 140, 0  ],   # 1  bed       (vivid orange)
    [255, 0,   255],   # 2  books     (magenta)
    [200, 200, 200],   # 3  ceiling   (light grey)
    [255, 100, 0  ],   # 4  chair     (saturated orange)
    [140, 60,  240],   # 5  floor     (vivid purple)
    [0,   200, 0  ],   # 6  furniture (saturated green)
    [255, 30,  30 ],   # 7  objects   (saturated red)
    [255, 0,   180],   # 8  picture   (bright pink-magenta)
    [255, 220, 0  ],   # 9  sofa      (saturated yellow)
    [0,   220, 220],   # 10 table     (saturated cyan)
    [200, 100, 0  ],   # 11 tv        (deep orange)
    [80,  140, 240],   # 12 wall      (saturated blue)
    [50,  50,  220],   # 13 window    (deep blue)
], dtype=np.float32) / 255.0

SCENENET_NAMES = ["unknown", "bed", "books", "ceiling", "chair", "floor",
                  "furniture", "objects", "picture", "sofa", "table", "tv",
                  "wall", "window"]

# SemanticKITTI 20-class learning palette (yaml's color_map composed through
# learning_map, BGR → RGB). 0 = unlabeled, 1..19 = semantic.
SEMKITTI_PALETTE = np.array([
    [0,   0,   0  ],   # 0  unlabeled
    [100, 150, 245],   # 1  car            (10:[245,150,100] BGR→RGB)
    [100, 230, 245],   # 2  bicycle
    [30,  60,  150],   # 3  motorcycle
    [80,  30,  180],   # 4  truck
    [0,   0,   255],   # 5  other-vehicle
    [255, 30,  30 ],   # 6  person
    [255, 40,  200],   # 7  bicyclist
    [150, 30,  90 ],   # 8  motorcyclist
    [255, 0,   255],   # 9  road
    [255, 150, 255],   # 10 parking
    [75,  0,   75 ],   # 11 sidewalk
    [175, 0,   75 ],   # 12 other-ground
    [255, 200, 0  ],   # 13 building
    [255, 120, 50 ],   # 14 fence
    [0,   175, 0  ],   # 15 vegetation
    [135, 60,  0  ],   # 16 trunk
    [150, 240, 80 ],   # 17 terrain
    [255, 240, 150],   # 18 pole
    [255, 0,   0  ],   # 19 traffic-sign
], dtype=np.float32) / 255.0

SEMKITTI_NAMES = ["unlabeled", "car", "bicycle", "motorcycle", "truck",
                  "other-vehicle", "person", "bicyclist", "motorcyclist",
                  "road", "parking", "sidewalk", "other-ground", "building",
                  "fence", "vegetation", "trunk", "terrain", "pole",
                  "traffic-sign"]


# REPLAY → yaml LUT (PolarSeg .topk uses REPLAY layout with lane-marking
# inserted at id 15). SCovox NPZs predicted via PolarSeg-soft are in REPLAY
# space; SLIM-VDB voxels.bin and our GT are in yaml learning-class space.
_REPLAY_TO_YAML = np.arange(256, dtype=np.int32)
_REPLAY_TO_YAML[15] = 9   # lane-marking → road
_REPLAY_TO_YAML[16] = 15  # vegetation
_REPLAY_TO_YAML[17] = 16  # trunk
_REPLAY_TO_YAML[18] = 17  # terrain
_REPLAY_TO_YAML[19] = 18  # pole
_REPLAY_TO_YAML[20] = 19  # traffic-sign


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_scovox_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    f = np.load(path)
    return np.asarray(f["points"], dtype=np.float32), \
           np.asarray(f["semantic_class"], dtype=np.int32)


def load_slim_voxelsbin(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    dt = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4"), ("cls", "i4")])
    rec = np.fromfile(path, dtype=dt)
    xyz = np.column_stack([rec["x"], rec["y"], rec["z"]]).astype(np.float32)
    return xyz, rec["cls"].astype(np.int32)


def load_scenenet_gt(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    f = np.load(path)
    return np.asarray(f["coords"], dtype=np.float32), \
           np.asarray(f["labels"], dtype=np.int32)


def build_kitti_gt(kitti_root: Path, sequence: str, n_scans: int,
                   voxel_size: float, yaml_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Build KITTI GT voxel grid in SCovox world frame (T_world_velo = P @ Tr)."""
    import yaml
    cfg = yaml.safe_load(yaml_path.read_text())
    lmap = np.zeros(max(cfg["learning_map"]) + 1, dtype=np.int32)
    for k, v in cfg["learning_map"].items():
        lmap[int(k)] = int(v)

    seq_dir = kitti_root / "sequences" / sequence
    # KITTI calib.txt: Tr line is the velo→cam0 transform (3x4).
    Tr = np.eye(4, dtype=np.float64)
    for line in (seq_dir / "calib.txt").read_text().splitlines():
        if line.startswith("Tr"):
            vals = [float(x) for x in line.split()[1:]]
            Tr[:3, :4] = np.array(vals).reshape(3, 4)
            break
    poses = []
    for line in (seq_dir / "poses.txt").read_text().splitlines():
        vals = [float(x) for x in line.split()]
        if len(vals) != 12:
            continue
        P = np.eye(4)
        P[:3, :4] = np.array(vals).reshape(3, 4)
        poses.append((P @ Tr).astype(np.float32))

    velo = sorted((seq_dir / "velodyne").glob("*.bin"))[:n_scans]
    lbls = sorted((seq_dir / "labels").glob("*.label"))[:n_scans]
    inv = 1.0 / voxel_size
    OFF = np.int64(1 << 20)
    SH1 = np.int64(42)
    SH2 = np.int64(21)
    packed_batches = []
    lbl_batches = []
    for i, (vf, lf) in enumerate(zip(velo, lbls)):
        pts = np.fromfile(vf, dtype=np.float32).reshape(-1, 4)[:, :3]
        lbl_raw = np.fromfile(lf, dtype=np.uint32) & 0xFFFF
        lbl = lmap[lbl_raw]
        keep = lbl > 0
        if not keep.any():
            continue
        pts, lbl = pts[keep], lbl[keep]
        T = poses[i]
        pw = (T @ np.concatenate(
            [pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1).T
              ).T[:, :3]
        keys = np.floor(pw * inv).astype(np.int64)
        packed = ((keys[:, 0] + OFF) << SH1) \
               | ((keys[:, 1] + OFF) << SH2) \
               |  (keys[:, 2] + OFF)
        packed_batches.append(packed)
        lbl_batches.append(lbl.astype(np.uint8))
        del pts, pw, keys, packed, lbl
    packed_all = np.concatenate(packed_batches); del packed_batches
    labels_all = np.concatenate(lbl_batches); del lbl_batches
    print(f"    GT raw points after filter: {packed_all.size:,}", flush=True)

    uniq, inverse = np.unique(packed_all, return_inverse=True)
    n_vox = uniq.shape[0]
    n_cls = int(labels_all.max()) + 1
    print(f"    unique voxels: {n_vox:,}  classes: {n_cls}", flush=True)
    flat = inverse.astype(np.int64) * n_cls + labels_all
    del inverse
    hist = np.bincount(flat, minlength=n_vox * n_cls)
    del flat
    hist = hist.reshape(n_vox, n_cls)
    maj = hist.argmax(axis=1).astype(np.int32)
    del hist

    kz = (uniq & ((1 << 21) - 1)) - OFF
    ky = ((uniq >> SH2) & ((1 << 21) - 1)) - OFF
    kx = (uniq >> SH1) - OFF
    coords = np.column_stack([
        (kx + 0.5) * voxel_size,
        (ky + 0.5) * voxel_size,
        (kz + 0.5) * voxel_size,
    ]).astype(np.float32)
    print(f"    GT done: {coords.shape[0]:,} voxels", flush=True)
    return coords, maj


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def points_to_voxelgrid(points: np.ndarray, labels: np.ndarray,
                       palette: np.ndarray, voxel_size: float,
                       drop_label_zero: bool = True,
                       max_voxels: int = 600_000) -> Tuple[o3d.geometry.VoxelGrid, float, int]:
    """Build an Open3D VoxelGrid from labelled points.

    If the count of unique cells at `voxel_size` would exceed `max_voxels`
    (Filament's practical render budget — ~600k cubes hits ~7M triangles),
    automatically coarsen to a larger display voxel size. The majority-label
    per coarse cell is kept. Returns (vg, display_voxel_size, n_voxels_used).
    """
    if drop_label_zero:
        mask = labels > 0
        points, labels = points[mask], labels[mask]
    labels = np.clip(labels, 0, palette.shape[0] - 1)

    # Cheap unique-cell count at the requested size
    inv = 1.0 / voxel_size
    keys = np.floor(points * inv).astype(np.int64)
    flat = keys[:, 0] * np.int64(1_000_000_007) + keys[:, 1] * np.int64(1_000_003) + keys[:, 2]
    n_unique = int(np.unique(flat).size)

    display_voxel_size = voxel_size
    if n_unique > max_voxels:
        # Solve for ratio so n_unique / ratio^3 ≤ max_voxels.
        ratio = (n_unique / max_voxels) ** (1.0 / 3.0)
        display_voxel_size = max(voxel_size, voxel_size * ratio)
        display_voxel_size = float(np.ceil(display_voxel_size / 0.05) * 0.05)

        # Vectorized majority-vote re-bin at the coarser resolution. Pack
        # 3D keys into int64 (21-bit slots) and bincount per-(voxel,class).
        inv2 = 1.0 / display_voxel_size
        keys2 = np.floor(points * inv2).astype(np.int64)
        OFF = np.int64(1 << 20)
        packed = ((keys2[:, 0] + OFF) << 42) \
               | ((keys2[:, 1] + OFF) << 21) \
               |  (keys2[:, 2] + OFF)
        uniq, inverse = np.unique(packed, return_inverse=True)
        n_vox = uniq.shape[0]
        n_cls = palette.shape[0]
        flat = inverse.astype(np.int64) * n_cls + labels.astype(np.int64)
        hist = np.bincount(flat, minlength=n_vox * n_cls).reshape(n_vox, n_cls)
        new_lbls = hist.argmax(axis=1).astype(np.int32)
        kz = (uniq & ((1 << 21) - 1)) - OFF
        ky = ((uniq >> 21) & ((1 << 21) - 1)) - OFF
        kx = (uniq >> 42) - OFF
        new_pts = np.column_stack([
            (kx + 0.5) * display_voxel_size,
            (ky + 0.5) * display_voxel_size,
            (kz + 0.5) * display_voxel_size,
        ]).astype(np.float32)
        points = new_pts
        labels = new_lbls

    colors = palette[labels]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    vg = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=display_voxel_size)
    return vg, display_voxel_size, len(points)


def render_panel(voxel_grid: o3d.geometry.VoxelGrid,
                 width: int, height: int,
                 eye: np.ndarray, center: np.ndarray, up: np.ndarray,
                 fov: float = 35.0,
                 background: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)) -> np.ndarray:
    r = rendering.OffscreenRenderer(width, height)
    r.scene.set_background(background)
    mat = rendering.MaterialRecord()
    # Unlit shader — every face renders at base_color × per-vertex color,
    # regardless of face orientation. No directional sun, no IBL — so the
    # three panels read identically lit and the only visual difference is
    # voxel class + voxel count.
    mat.shader = "defaultUnlit"
    mat.base_color = [1.0, 1.0, 1.0, 1.0]
    r.scene.add_geometry("vg", voxel_grid, mat)
    r.scene.scene.enable_sun_light(False)
    r.scene.scene.enable_indirect_light(False)
    r.setup_camera(fov, center.astype(np.float32),
                   eye.astype(np.float32), up.astype(np.float32))
    img = np.asarray(r.render_to_image()).copy()
    # OffscreenRenderer holds a Filament FEngine; drop the ref to free the
    # GPU context before opening the next one (sequential per panel).
    del r
    return img


def compute_camera(points: np.ndarray, view: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (eye, center, up) for a shared camera. `view` controls angle."""
    bb_min = points.min(axis=0)
    bb_max = points.max(axis=0)
    center = 0.5 * (bb_min + bb_max)
    extent = bb_max - bb_min
    diag = float(np.linalg.norm(extent))
    if view == "indoor_iso":
        # Y-up world (SceneNet/Replica). After ceiling-clip, look down into
        # the room at ~55° elevation from one corner — high enough that the
        # floor + furniture are visible, low enough that wall thickness
        # (SLIM-VDB's TSDF puff) is also legible.
        center = np.array([0.5 * (bb_min[0] + bb_max[0]),
                           bb_min[1] + 0.30 * extent[1],
                           0.5 * (bb_min[2] + bb_max[2])], dtype=np.float32)
        eye = center + np.array([0.8 * extent[0],
                                 1.6 * extent[1],
                                 0.8 * extent[2]], dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    elif view == "indoor_topdown":
        # Y-up world, straight overhead. Requires ceiling-clip upstream.
        # "Up" in image space points toward +Z so the room reads landscape.
        center = np.array([0.5 * (bb_min[0] + bb_max[0]),
                           0.5 * (bb_min[1] + bb_max[1]),
                           0.5 * (bb_min[2] + bb_max[2])], dtype=np.float32)
        height_above = max(extent[0], extent[2]) * 1.1 + extent[1]
        eye = center + np.array([0.0, height_above, 0.0], dtype=np.float32)
        up = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    elif view == "kitti_bev_angle":
        # cam0 frame: X=right, Y=down (gravity), Z=forward.
        # Camera hovers above the trajectory (negative Y in cam0) and back
        # along -Z, looking down at ~55° elevation.
        eye = center + np.array([0.1 * extent[0],
                                 -1.4 * max(extent[1], 15.0),
                                 -0.9 * max(extent[2], 40.0)], dtype=np.float32)
        up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    elif view == "kitti_topdown":
        # cam0 frame top-down: look straight down along +Y (gravity).
        # Image-up is +Z (the forward axis), so the trajectory reads
        # "north-up" with the lateral X axis going across the frame.
        height_above = max(extent[0], extent[2]) * 1.05 + 5.0
        eye = center + np.array([0.0, -height_above, 0.0], dtype=np.float32)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        raise ValueError(f"unknown view: {view}")
    return eye.astype(np.float32), center.astype(np.float32), up


def stitch_panels(imgs, titles, palette_used,
                  class_names, header: Optional[str] = None) -> Image.Image:
    h, w, _ = imgs[0].shape
    title_h = 38
    legend_h = 70
    gap = 12
    n = len(imgs)
    canvas_w = n * w + (n + 1) * gap
    canvas_h = title_h + h + legend_h + 3 * gap
    if header:
        canvas_h += 32
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))

    try:
        title_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        legend_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        header_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except OSError:
        title_font = legend_font = header_font = ImageFont.load_default()

    draw = ImageDraw.Draw(canvas)
    y_off = 0
    if header:
        draw.text((gap, 4), header, fill=(50, 50, 50), font=header_font)
        y_off = 32

    for i, (img, title) in enumerate(zip(imgs, titles)):
        x = gap + i * (w + gap)
        # title
        bbox = draw.textbbox((0, 0), title, font=title_font)
        tw = bbox[2] - bbox[0]
        draw.text((x + (w - tw) // 2, y_off + 4), title,
                  fill=(0, 0, 0), font=title_font)
        canvas.paste(Image.fromarray(img), (x, y_off + title_h))

    # Class-color legend at the bottom
    used_ids = sorted(set(palette_used.tolist()))
    legend_y = y_off + title_h + h + gap
    swatch = 14
    cursor_x = gap
    line_h = swatch + 6
    for cid in used_ids:
        if cid == 0:
            continue
        if cid >= len(class_names):
            continue
        color = tuple(int(255 * c) for c in palette_used_color(cid))
        label = f"{cid} {class_names[cid]}"
        bbox = draw.textbbox((0, 0), label, font=legend_font)
        lw = bbox[2] - bbox[0]
        if cursor_x + swatch + 6 + lw + 12 > canvas_w - gap:
            cursor_x = gap
            legend_y += line_h
        draw.rectangle([cursor_x, legend_y, cursor_x + swatch, legend_y + swatch],
                       fill=color, outline=(0, 0, 0))
        draw.text((cursor_x + swatch + 4, legend_y - 2), label,
                  fill=(0, 0, 0), font=legend_font)
        cursor_x += swatch + 6 + lw + 16

    return canvas


# tiny closure-ish trick: the legend needs the actual palette colors for
# the ids it draws. We attach the palette via a module-level "current"
# pointer set by render_scene() right before stitch_panels().
_CURRENT_PALETTE = SCENENET_PALETTE


def palette_used_color(cid: int) -> np.ndarray:
    return _CURRENT_PALETTE[cid]


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------

def render_scenenet(ws_root: Path, seq: str, out_dir: Path,
                    width: int, height: int,
                    view: str = "indoor_iso",
                    slim_subdir: str = "scenenet_val",
                    slim_label: str = "sdf_trunc=0.10") -> Path:
    global _CURRENT_PALETTE
    _CURRENT_PALETTE = SCENENET_PALETTE
    voxel_size = 0.05

    gt_pts, gt_lbl = load_scenenet_gt(
        ws_root / f"data/scenenet/val_preprocessed/{seq}/gt_5cm.npz")
    sc_pts, sc_lbl = load_scovox_npz(
        ws_root / "src/robot_sw/distributed_mapping/scovox_eval/results"
        / "phase1_ktop_sweep_2026_05_14/K2"
        / f"scenenet_{seq}/scovox.npz")
    sl_pts, sl_lbl = load_slim_voxelsbin(
        ws_root / f"third_party_sw/slim_vdb/outputs/{slim_subdir}/{seq}/voxels.bin")

    # Ceiling clip: SceneNet world is Y-up, ceiling sits at the top of the
    # room. Drop the upper ~0.8 m of the kept slab so the camera sees the
    # interior. Using GT bbox so all three panels match.
    y_floor = gt_pts[:, 1].min()
    y_ceil = gt_pts[:, 1].max()
    y_cut = y_ceil - 0.8

    def yclip(pts, lbl):
        m = pts[:, 1] <= y_cut
        return pts[m], lbl[m]

    gt_pts, gt_lbl = yclip(gt_pts, gt_lbl)
    sc_pts, sc_lbl = yclip(sc_pts, sc_lbl)
    sl_pts, sl_lbl = yclip(sl_pts, sl_lbl)

    # Shared camera based on union of all three bounding boxes (post-clip).
    all_pts = np.vstack([gt_pts, sc_pts, sl_pts])
    eye, center, up = compute_camera(all_pts, view=view)
    fov = 30.0 if view == "indoor_topdown" else 40.0

    raw_counts = [len(gt_pts), len(sc_pts), len(sl_pts)]
    panels, display_sizes = [], []
    for pts, lbl in [(gt_pts, gt_lbl), (sc_pts, sc_lbl), (sl_pts, sl_lbl)]:
        vg, dvs, _ = points_to_voxelgrid(pts, lbl, SCENENET_PALETTE, voxel_size)
        display_sizes.append(dvs)
        panels.append(render_panel(vg, width, height, eye, center, up, fov=fov))

    used = np.unique(np.concatenate([gt_lbl, sc_lbl, sl_lbl])).astype(np.int32)
    titles = [f"Ground truth ({raw_counts[0]:,} voxels)",
              f"SCovox K=2 ({raw_counts[1]:,} voxels)",
              f"SLIM-VDB {slim_label} ({raw_counts[2]:,} voxels)"]
    suffix = "_topdown" if view == "indoor_topdown" else ""
    if slim_subdir != "scenenet_val":
        suffix += "_" + slim_subdir.replace("scenenet_val_", "")
    canvas = stitch_panels(panels, titles, used, SCENENET_NAMES,
                           header=f"SceneNet RGB-D val/{seq} · 5 cm voxels · n_scans=300")
    out_path = out_dir / f"qualitative_scenenet_{seq}{suffix}.png"
    canvas.save(out_path)
    canvas.convert("RGB").save(out_path.with_suffix(".pdf"), "PDF", resolution=300.0)
    return out_path


def render_kitti(ws_root: Path, seq: str, out_dir: Path,
                 width: int, height: int,
                 view: str = "kitti_bev_angle",
                 # In cam0 frame: X=right, Y=down (gravity), Z=forward.
                 # Keep ~5 m above road to ~2 m below; clip a 50 m square
                 # around the trajectory mean. Tight enough that SLIM-VDB's
                 # puffy shell stays under ~1M voxels (Filament rendering
                 # budget) while still showing a recognisable urban block.
                 y_clip: Optional[Tuple[float, float]] = (-5.0, 2.0),
                 horizontal_half_extent: float = 25.0) -> Path:
    global _CURRENT_PALETTE
    _CURRENT_PALETTE = SEMKITTI_PALETTE
    voxel_size = 0.10

    print(f"  building KITTI seq{seq} GT (100 scans, voxel=0.10 m)...", flush=True)
    gt_pts, gt_lbl = build_kitti_gt(
        ws_root / "data/semantickitti/dataset",
        sequence=seq, n_scans=100, voxel_size=voxel_size,
        yaml_path=ws_root / "src/sem_seg_pipeline/polarseg/semantic-kitti.yaml")

    sc_pts, sc_lbl_replay = load_scovox_npz(
        ws_root / "src/robot_sw/distributed_mapping/scovox_eval/results"
        / "phase1_ktop_sweep_2026_05_14/K2"
        / f"kitti_{seq}_soft/scovox.npz")
    sc_lbl = _REPLAY_TO_YAML[np.clip(sc_lbl_replay, 0, 255)]

    sl_pts, sl_lbl = load_slim_voxelsbin(
        ws_root / f"third_party_sw/slim_vdb/outputs/kitti/{seq}/voxels.bin")
    # SCovox accumulates voxels in the cam0 world frame (T_world_velo = P @ Tr)
    # while SLIM-VDB accumulates in the velodyne world frame
    # (T_world_velo = Tr_inv @ P @ Tr) — see eval_scovox_kitti_miou.py header.
    # The two frames are related by Tr (velo→cam0), so left-multiply Tr to
    # bring SLIM-VDB voxels into SCovox/GT's frame for a fair visual overlay.
    Tr = np.eye(4, dtype=np.float64)
    for line in (ws_root / "data/semantickitti/dataset/sequences"
                 / seq / "calib.txt").read_text().splitlines():
        if line.startswith("Tr"):
            vals = [float(x) for x in line.split()[1:]]
            Tr[:3, :4] = np.array(vals).reshape(3, 4)
            break
    sl_h = np.column_stack([sl_pts, np.ones(len(sl_pts), dtype=np.float32)])
    sl_pts = (Tr @ sl_h.T).T[:, :3].astype(np.float32)

    # Height clip on cam0 Y axis (gravity-down) to drop sky and below-road
    # outliers — large extent (~30 m) otherwise dominates the camera.
    def yclip(pts, lbl):
        if y_clip is None:
            return pts, lbl
        m = (pts[:, 1] >= y_clip[0]) & (pts[:, 1] <= y_clip[1])
        return pts[m], lbl[m]

    gt_pts, gt_lbl = yclip(gt_pts, gt_lbl)
    sc_pts, sc_lbl = yclip(sc_pts, sc_lbl)
    sl_pts, sl_lbl = yclip(sl_pts, sl_lbl)

    # Horizontal square window centered on the GT trajectory's X/Z mean —
    # seq08 covers ~80 m in each horizontal axis over the first 100 scans.
    cx = float(np.mean([gt_pts[:, 0].mean(), sc_pts[:, 0].mean(),
                        sl_pts[:, 0].mean()]))
    cz = float(np.mean([gt_pts[:, 2].mean(), sc_pts[:, 2].mean(),
                        sl_pts[:, 2].mean()]))
    he = horizontal_half_extent

    def hclip(pts, lbl):
        m = ((pts[:, 0] >= cx - he) & (pts[:, 0] <= cx + he)
             & (pts[:, 2] >= cz - he) & (pts[:, 2] <= cz + he))
        return pts[m], lbl[m]

    gt_pts, gt_lbl = hclip(gt_pts, gt_lbl)
    sc_pts, sc_lbl = hclip(sc_pts, sc_lbl)
    sl_pts, sl_lbl = hclip(sl_pts, sl_lbl)

    all_pts = np.vstack([gt_pts, sc_pts, sl_pts])
    eye, center, up = compute_camera(all_pts, view=view)

    raw_counts = [len(gt_pts), len(sc_pts), len(sl_pts)]
    panels, display_sizes = [], []
    for pts, lbl in [(gt_pts, gt_lbl), (sc_pts, sc_lbl), (sl_pts, sl_lbl)]:
        vg, dvs, n_used = points_to_voxelgrid(pts, lbl, SEMKITTI_PALETTE, voxel_size)
        display_sizes.append(dvs)
        print(f"    panel voxels={n_used:,} at display_size={dvs:.2f} m", flush=True)
        panels.append(render_panel(vg, width, height, eye, center, up, fov=30.0))

    used = np.unique(np.concatenate([gt_lbl, sc_lbl, sl_lbl])).astype(np.int32)
    titles = [f"Ground truth ({raw_counts[0]:,} voxels)",
              f"SCovox K=2 PolarSeg-soft ({raw_counts[1]:,} voxels)",
              f"SLIM-VDB sdf_trunc=0.10 ({raw_counts[2]:,} voxels)"]
    suffix = "_topdown" if view == "kitti_topdown" else ""
    # Display voxel size may differ per panel if the auto-coarsener fires;
    # surface that in the header so the reader knows the cubes shown are not
    # the metric's voxels.
    if max(display_sizes) > voxel_size + 1e-3:
        size_str = " · ".join(f"{s*100:.0f} cm" for s in display_sizes)
        size_note = f"display cubes [{size_str}]"
    else:
        size_note = f"{int(voxel_size*100)} cm voxels"
    canvas = stitch_panels(panels, titles, used, SEMKITTI_NAMES,
                           header=f"SemanticKITTI seq{seq} · {size_note} · "
                                  f"first 100 scans, 50 m square slice")
    out_path = out_dir / f"qualitative_kitti_{seq}{suffix}.png"
    canvas.save(out_path)
    canvas.convert("RGB").save(out_path.with_suffix(".pdf"), "PDF", resolution=300.0)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", action="append", required=True,
                    help="scenenet:<seq> or kitti:<seq> (repeatable)")
    ap.add_argument("--ws_root", type=Path,
                    default=Path(__file__).resolve().parents[5])
    ap.add_argument("--out_dir", type=Path, default=Path("figures/qualitative"))
    ap.add_argument("--width", type=int, default=900)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--kitti_view", default="kitti_bev_angle",
                    choices=["kitti_bev_angle", "kitti_topdown"])
    ap.add_argument("--scenenet_view", default="indoor_iso",
                    choices=["indoor_iso", "indoor_topdown"])
    ap.add_argument("--scenenet_slim_subdir", default="scenenet_val",
                    help="third_party_sw/slim_vdb/outputs/<subdir>/<seq>/voxels.bin")
    ap.add_argument("--scenenet_slim_label", default="sdf_trunc=0.10",
                    help="legend label for the SLIM-VDB panel")
    args = ap.parse_args()

    out_dir = (args.ws_root / args.out_dir) if not args.out_dir.is_absolute() else args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"ws_root  = {args.ws_root}")
    print(f"out_dir  = {out_dir}")

    for s in args.scene:
        ds, _, seq = s.partition(":")
        if not seq:
            print(f"  SKIP {s}: expected '<dataset>:<seq>'")
            continue
        print(f"\n[{ds}:{seq}] rendering 3-panel...")
        if ds == "scenenet":
            out = render_scenenet(args.ws_root, seq, out_dir,
                                  args.width, args.height,
                                  view=args.scenenet_view,
                                  slim_subdir=args.scenenet_slim_subdir,
                                  slim_label=args.scenenet_slim_label)
        elif ds == "kitti":
            out = render_kitti(args.ws_root, seq, out_dir,
                               args.width, args.height,
                               view=args.kitti_view)
        else:
            print(f"  SKIP {s}: unknown dataset {ds!r}")
            continue
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
