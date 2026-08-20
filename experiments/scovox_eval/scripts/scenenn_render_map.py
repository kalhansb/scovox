#!/usr/bin/env python3
"""Render a SCovox SceneNN map (.npz from pointcloud_to_npz) to a PNG.

Pure numpy painter's-algorithm splatting -- the Jetson has no open3d, and this
only needs to answer "does the map look like the room". Produces a top-down
view plus two obliques, coloured by NYU-40 class.

Usage:
  python3 scenenn_render_map.py --npz experiments/results/scenenn/016/scovox.npz \
      --colormap /home/jetsondevkit/datasets/scenenn/nyu_color.xml --out map.png
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scovox_eval.scenenn_data import load_nyu_colormap, NYU_NUM_CLASSES  # noqa: E402


BG = np.array([26, 28, 34], np.uint8)


def _view(pts, cols, azim_deg, elev_deg, size=560, margin=0.07, radius=1):
    """Orthographic splat with a real z-buffer.

    Dark background and depth-darkened splats: the NYU-40 palette is pastel, so
    on white the map washes out to near-invisible and structure cannot be read.
    """
    a, e = np.deg2rad(azim_deg), np.deg2rad(elev_deg)
    # SceneNN rooms are Y-up: yaw about Y, then pitch toward the viewer.
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    P = pts @ Ry.T @ Rx.T

    u, v, d = P[:, 0], -P[:, 1], P[:, 2]
    lo_u, lo_v = u.min(), v.min()
    span = max(max(u.max() - lo_u, v.max() - lo_v), 1e-6)
    scale = size * (1 - 2 * margin) / span
    px = np.clip(((u - lo_u) * scale + size * margin).astype(np.int32), 0, size - 1)
    py = np.clip(((v - lo_v) * scale + size * margin).astype(np.int32), 0, size - 1)

    # Nearer = brighter. Without normals this is what gives the cloud relief.
    rng = max(d.max() - d.min(), 1e-6)
    shade = (1.10 - 0.55 * (d - d.min()) / rng)[:, None]
    rgb = np.clip(cols * shade, 0, 255).astype(np.uint8)

    img = np.tile(BG, (size, size, 1))
    zbuf = np.full((size, size), np.inf)
    order = np.argsort(-d)  # far first, so near overwrites
    px, py, rgb, dd = px[order], py[order], rgb[order], d[order]
    for ox in range(-radius, radius + 1):
        for oy in range(-radius, radius + 1):
            xx = np.clip(px + ox, 0, size - 1)
            yy = np.clip(py + oy, 0, size - 1)
            img[yy, xx] = rgb
            zbuf[yy, xx] = dd
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--colormap", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-occ", type=float, default=0.5)
    args = ap.parse_args()

    z = np.load(args.npz)
    print("npz fields:", list(z.keys()))
    # pointcloud_to_npz writes a packed (N,3) 'points'; older captures kept x/y/z apart.
    if "points" in z:
        pts = np.asarray(z["points"], dtype=np.float64)
    else:
        pts = np.stack([z["x"], z["y"], z["z"]], axis=1).astype(np.float64)

    occ_key = next((k for k in ("occupancy_prob", "occupancy", "p_occ", "prob") if k in z), None)
    if occ_key is not None:
        keep = z[occ_key] >= args.min_occ
        pts, sel = pts[keep], keep
        print(f"occupancy field '{occ_key}': kept {keep.sum()}/{len(keep)} voxels >= {args.min_occ}")
    else:
        sel = np.ones(len(pts), bool)

    cls_key = next((k for k in ("class_id", "semantic_class", "label", "class") if k in z), None)
    palette = load_nyu_colormap(args.colormap)
    if cls_key is not None:
        cls = np.clip(z[cls_key][sel].astype(int), 0, NYU_NUM_CLASSES - 1)
        cols = palette[cls].astype(np.float64)
        names = {int(m.group(1)): m.group(2) for m in
                 re.finditer(r'<class id="(\d+)" text="([^"]+)"', Path(args.colormap).read_text())}
        u, c = np.unique(cls, return_counts=True)
        top = sorted(zip(c, u), reverse=True)[:10]
        print(f"\nvoxels: {len(pts)}   semantic field: '{cls_key}'")
        for n, i in top:
            print(f"  {names.get(int(i), i):<16} {n:>7}  {n / len(cls):>6.1%}")
    else:
        print("no semantic field found -- colouring by height")
        h = (pts[:, 1] - pts[:, 1].min()) / max(np.ptp(pts[:, 1]), 1e-6)
        cols = (np.stack([h, 0.4 + 0.4 * h, 1 - h], 1) * 255)

    ext = pts.max(0) - pts.min(0)
    print(f"extent: {ext[0]:.2f} x {ext[1]:.2f} x {ext[2]:.2f} m")

    views = [_view(pts, cols, 0, 88), _view(pts, cols, 40, 22), _view(pts, cols, 220, 22)]
    gap = np.tile(BG, (views[0].shape[0], 6, 1))
    strip = np.concatenate([views[0], gap, views[1], gap, views[2]], axis=1)
    PILImage.fromarray(strip).save(args.out)
    print(f"wrote {args.out}  (top-down | oblique A | oblique B)")


if __name__ == "__main__":
    main()
