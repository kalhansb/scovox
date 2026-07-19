#!/usr/bin/env python3
"""Decode a scovox ScovoxMapBinary blob (LZ4 + the SCVX v5 wire format) into
numpy arrays — the full split Beta (occupancy) + Dir (semantics) grids.

Mirrors scovox/src/scovox_core/include/scovox/binary_serializer.hpp (FORMAT
5) and lz4_codec.hpp. A snapshot frame (triggered when a new subscriber
connects — scovox_node.cpp:1608) carries EVERY non-prior voxel, so this yields
a_occ/a_free for occupied AND carved-free voxels — the input to E1.

Wire (little-endian, after LZ4 decompress):
  [MAGIC u32=0x53435658][VERSION u8=5][resolution f32][num_classes u16]
  [K_TOP u8][alpha_0 f32]
  [tsdf_count u32] (x,y,z i32; distance,weight f32)*         # 20 B, usually 0
  [beta_count u32] (x,y,z i32; a_occ,a_free f32)*            # 20 B
  [dir_count  u32] (x,y,z i32; other f32; cnt[K] f32; cls[K] u16)*  # 28 B @K=2

LZ4 framing (lz4_codec.hpp): 4-byte BIG-endian original size, then a raw LZ4
block (LZ4_compress_default). Python lz4.block.decompress handles the block.
"""
from __future__ import annotations
import struct
import numpy as np

MAGIC = 0x53435658           # "SCVX"
FORMAT_VERSION = 5
K_TOP = 2

_BETA_DT = np.dtype([("x", "<i4"), ("y", "<i4"), ("z", "<i4"),
                     ("a_occ", "<f4"), ("a_free", "<f4")])
_DIR_DT = np.dtype([("x", "<i4"), ("y", "<i4"), ("z", "<i4"),
                    ("other", "<f4"), ("cnt", "<f4", (K_TOP,)),
                    ("cls", "<u2", (K_TOP,))])
assert _BETA_DT.itemsize == 20, _BETA_DT.itemsize
assert _DIR_DT.itemsize == 28, _DIR_DT.itemsize


def lz4_unwrap(blob: bytes) -> bytes:
    """Undo compressLZ4: 4-byte big-endian size header + raw LZ4 block."""
    import lz4.block
    if len(blob) <= 4:
        return b""
    orig = struct.unpack_from(">I", blob, 0)[0]
    return lz4.block.decompress(blob[4:], uncompressed_size=orig)


def decode_frame(raw: bytes) -> dict:
    """Parse an *uncompressed* SCVX frame -> dict of arrays + header."""
    off = 0
    magic, ver = struct.unpack_from("<IB", raw, off); off += 5
    if magic != MAGIC:
        raise ValueError(f"bad MAGIC {magic:#x}")
    if ver != FORMAT_VERSION:
        raise ValueError(f"bad VERSION {ver}")
    resolution, num_classes, k_top, alpha_0 = struct.unpack_from("<fHBf", raw, off)
    off += 4 + 2 + 1 + 4
    if k_top != K_TOP:
        raise ValueError(f"K_TOP mismatch: wire={k_top} decoder={K_TOP}")

    def read_stream(off, dt):
        (count,) = struct.unpack_from("<I", raw, off); off += 4
        arr = np.frombuffer(raw, dtype=dt, count=count, offset=off).copy()
        off += count * dt.itemsize
        return arr, off

    tsdf, off = read_stream(off, np.dtype([("x", "<i4"), ("y", "<i4"), ("z", "<i4"),
                                           ("distance", "<f4"), ("weight", "<f4")]))
    beta, off = read_stream(off, _BETA_DT)
    dir_, off = read_stream(off, _DIR_DT)
    return {"resolution": resolution, "num_classes": int(num_classes),
            "alpha_0": alpha_0, "k_top": int(k_top),
            "tsdf": tsdf, "beta": beta, "dir": dir_}


def decode_blob(blob: bytes) -> dict:
    """Full path: LZ4-compressed ScovoxMapBinary.data -> parsed frame."""
    return decode_frame(lz4_unwrap(bytes(blob)))


def to_npz_dict(frame: dict) -> dict:
    """Flatten to plain arrays for np.savez (coords in voxel units)."""
    b, d = frame["beta"], frame["dir"]
    out = {
        "resolution": np.float32(frame["resolution"]),
        "num_classes": np.int32(frame["num_classes"]),
        "alpha_0": np.float32(frame["alpha_0"]),
        "beta_xyz": np.stack([b["x"], b["y"], b["z"]], axis=1) if len(b) else np.zeros((0, 3), np.int32),
        "a_occ": b["a_occ"] if len(b) else np.zeros(0, np.float32),
        "a_free": b["a_free"] if len(b) else np.zeros(0, np.float32),
        "dir_xyz": np.stack([d["x"], d["y"], d["z"]], axis=1) if len(d) else np.zeros((0, 3), np.int32),
        "dir_other": d["other"] if len(d) else np.zeros(0, np.float32),
        "dir_cnt": d["cnt"] if len(d) else np.zeros((0, K_TOP), np.float32),
        "dir_cls": d["cls"] if len(d) else np.zeros((0, K_TOP), np.uint16),
    }
    return out


if __name__ == "__main__":
    import sys
    with open(sys.argv[1], "rb") as f:
        blob = f.read()
    fr = decode_blob(blob)
    print(f"res={fr['resolution']} C={fr['num_classes']} a0={fr['alpha_0']} "
          f"beta={len(fr['beta'])} dir={len(fr['dir'])} tsdf={len(fr['tsdf'])}")
    if len(fr["beta"]):
        b = fr["beta"]
        s = b["a_occ"] + b["a_free"]
        p = b["a_occ"] / s
        print(f"  p_occ: min={p.min():.3f} med={np.median(p):.3f} max={p.max():.3f} "
              f"| free(p<.5)={np.mean(p<0.5):.2%} occ(p>.5)={np.mean(p>0.5):.2%}")
