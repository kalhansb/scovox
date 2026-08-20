#!/usr/bin/env python3
"""SceneNN (HKUST) data access helpers.

Layout produced by the SceneNN `playback` tool + the HTTP server at
hkust-vgd.ust.hk/scenenn:

    <scene>/frames/depth/depth%05d.png   uint16 mm, 640x480, 1-indexed
    <scene>/frames/image/image%05d.png   RGB8 640x480, 1-indexed
    <scene>/frames/timestamp.txt         "frameID colorTs depthTs"
    <scene>/trajectory.log               Redwood .log, 4x4 camera-to-world
    <scene>/<id>_nyu.ply                 annotated mesh, per-vertex nyu_class
    asus.ini                             fx/fy/cx/cy

Two SceneNN quirks this module exists to encapsulate:

  * `PlaybackSync.cpp` applies FreeImage_FlipVertical to the *colour* bitmap
    but not to the depth bitmap, while writing depth via FreeImage_GetScanLine
    (which counts from the bottom). The saved depth PNG is therefore flipped
    vertically relative to colour. `load_depth_m` undoes it. Verified
    empirically by mesh-residual (see scenenn_check_geometry.py) rather than
    assumed -- flip it the wrong way and every label is garbage.

  * Depth frames are 1-indexed (k starts at 1 in the playback loop) while
    trajectory.log poses are 0-indexed, so frame k pairs with pose k-1.
"""

import re
from pathlib import Path

import numpy as np

# nyu_class ids run 0 (unknown) .. 40. Class 0 must stay "no evidence".
NYU_NUM_CLASSES = 41


def load_intrinsics(ini_path):
    """Parse asus.ini / kinect2.ini -> dict with fx, fy, cx, cy, width, height."""
    vals = {}
    for line in Path(ini_path).read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                vals[parts[0].strip()] = float(parts[1])
            except ValueError:
                pass
    return {
        "fx": vals["fx"],
        "fy": vals["fy"],
        "cx": vals["cx"],
        "cy": vals["cy"],
        "width": int(vals.get("depth_width", 640)),
        "height": int(vals.get("depth_height", 480)),
    }


def load_trajectory(log_path):
    """Redwood .log -> (N,4,4) float64 camera-to-world matrices."""
    nums, poses = [], []
    lines = Path(log_path).read_text().splitlines()
    i = 0
    while i < len(lines):
        header = lines[i].split()
        if len(header) != 3:
            i += 1
            continue
        rows = []
        for j in range(1, 5):
            rows.append([float(v) for v in lines[i + j].split()])
        poses.append(np.array(rows, dtype=np.float64))
        nums.append(int(header[0]))
        i += 5
    return np.stack(poses) if poses else np.zeros((0, 4, 4))


def load_depth_m(png_path, flip_vertical=True):
    """Depth PNG -> float32 metres, corrected for the playback tool's flip."""
    from PIL import Image as PILImage

    d = np.array(PILImage.open(str(png_path)))
    if flip_vertical:
        d = d[::-1, :]
    return np.ascontiguousarray(d).astype(np.float32) / 1000.0


def load_color(png_path):
    from PIL import Image as PILImage

    return np.array(PILImage.open(str(png_path)).convert("RGB"))


def _ply_header(fh):
    """Read an ASCII PLY header; return (n_vertex, vertex_props, data_offset)."""
    props, n_vertex, element = [], 0, None
    while True:
        line = fh.readline().decode("ascii", "replace").strip()
        if line.startswith("element vertex"):
            element, n_vertex = "vertex", int(line.split()[2])
        elif line.startswith("element "):
            element = line.split()[1]
        elif line.startswith("property ") and element == "vertex":
            parts = line.split()
            if parts[1] != "list":
                props.append((parts[2], parts[1]))
        elif line == "end_header":
            return n_vertex, props, fh.tell()


_PLY_DTYPE = {
    "float": "<f4", "float32": "<f4", "double": "<f8",
    "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
    "ushort": "<u2", "uint16": "<u2", "short": "<i2", "int16": "<i2",
    "uint": "<u4", "uint32": "<u4", "int": "<i4", "int32": "<i4",
}


def load_annotated_ply(ply_path):
    """Read a SceneNN binary_little_endian PLY vertex block.

    Returns (xyz float64 (N,3), nyu_class uint16 (N,) or None, label uint32 (N,) or None).
    Only the vertex element is parsed -- faces are not needed for labelling,
    and skipping them keeps a 43 MB mesh cheap to load on the Jetson.
    """
    with open(ply_path, "rb") as fh:
        n_vertex, props, offset = _ply_header(fh)
        dtype = np.dtype([(name, _PLY_DTYPE[typ]) for name, typ in props])
        fh.seek(offset)
        verts = np.frombuffer(fh.read(n_vertex * dtype.itemsize), dtype=dtype, count=n_vertex)

    xyz = np.stack([verts["x"], verts["y"], verts["z"]], axis=1).astype(np.float64)
    nyu = verts["nyu_class"].astype(np.uint16) if "nyu_class" in dtype.names else None
    lab = verts["label"].astype(np.uint32) if "label" in dtype.names else None
    return xyz, nyu, lab


def load_nyu_colormap(xml_path):
    """nyu_color.xml -> (41,3) uint8 palette indexed by class id."""
    text = Path(xml_path).read_text()
    palette = np.zeros((NYU_NUM_CLASSES, 3), dtype=np.uint8)
    for m in re.finditer(r'<class\s+id="(\d+)"[^>]*color="(\d+)\s+(\d+)\s+(\d+)"', text):
        cid = int(m.group(1))
        if 0 <= cid < NYU_NUM_CLASSES:
            palette[cid] = [int(m.group(2)), int(m.group(3)), int(m.group(4))]
    return palette


def deproject(depth_m, K, stride=1):
    """Depth (H,W) metres -> (M,3) camera-frame points + (M,) flat pixel indices.

    OpenNI depth is already perspective Z-depth (not ray length), so no
    ray->Z rescaling here -- unlike the SceneNet replay path.
    """
    h, w = depth_m.shape
    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    z = depth_m[::stride, ::stride]
    valid = z > 0
    z, us, vs = z[valid], us[valid], vs[valid]
    x = (us - K["cx"]) * z / K["fx"]
    y = (vs - K["cy"]) * z / K["fy"]
    return np.stack([x, y, z], axis=1).astype(np.float64), (vs * w + us).astype(np.int64)


def to_world(pts_cam, T_cam2world):
    return pts_cam @ T_cam2world[:3, :3].T + T_cam2world[:3, 3]
