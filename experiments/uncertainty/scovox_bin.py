#!/usr/bin/env python3
"""Decode a scovox ScovoxMapBinary blob (LZ4 + the SCVX wire format) into
numpy arrays — the full split Beta (occupancy) + Dir (semantics) grids.

Mirrors scovox/src/scovox_core/include/scovox/binary_serializer.hpp and
lz4_codec.hpp. Handles codec revisions 5 (flat 20/28 B records) AND 6
(block-run coordinate coding + optional u16 payload quantization,
docs/design/comms_design_2026_07_30.md Part 1), so bags recorded on either
side of the format bump stay readable. A snapshot frame (triggered when a new
subscriber connects) carries EVERY non-prior voxel, so this yields
a_occ/a_free for occupied AND carved-free voxels — the input to E1.

Wire v5 (little-endian, after LZ4 decompress):
  [MAGIC u32=0x53435658][VERSION u8=5][resolution f32][num_classes u16]
  [K_TOP u8][alpha_0 f32]
  [tsdf_count u32] (x,y,z i32; distance,weight f32)*         # 20 B, usually 0
  [beta_count u32] (x,y,z i32; a_occ,a_free f32)*            # 20 B
  [dir_count  u32] (x,y,z i32; other f32; cnt[K] f32; cls[K] u16)*  # 28 B @K=2

Wire v6 adds [quant_step f32] after alpha_0 (0.0 = f32 payloads, else u16
quantized as q = round((a - prior)/step)); the Beta/Dir streams become block
runs — records grouped per 8x8x8 leaf block:
  [bx,by,bz i32][mode u8]
    mode 0: [bitmask 64 B]        # 512 bits, bit=(lx<<6)|(ly<<3)|lz, LSB-first
    mode 1: [n u16][idx u16]*n    # strictly ascending
  [payload]*n                     # ascending bit order
  payload: beta  [q_occ,q_free u16] | [a_occ,a_free f32]
           dir   [q_other u16][q_cnt[K] u16][cls[K] u16]
               | [other f32][cnt[K] f32][cls[K] u16]
Priors for dequantization: Beta occ/free = 1.0 (symmetric Beta(1,1)); Dir
slot = alpha_0, OTHER = max(0, C - K_TOP) * alpha_0. The TSDF stream stays
flat 20 B records in both revisions.

LZ4 framing (lz4_codec.hpp): 4-byte BIG-endian original size, then a raw LZ4
block (LZ4_compress_default). Python lz4.block.decompress handles the block.
"""
from __future__ import annotations
import struct
import numpy as np

MAGIC = 0x53435658           # "SCVX"
SUPPORTED_VERSIONS = (5, 6)
K_TOP = 2

BETA_OCC_PRIOR = 1.0         # kBetaOccPrior / kBetaFreePrior (beta_voxel.hpp)
BETA_FREE_PRIOR = 1.0

_BETA_DT = np.dtype([("x", "<i4"), ("y", "<i4"), ("z", "<i4"),
                     ("a_occ", "<f4"), ("a_free", "<f4")])
_DIR_DT = np.dtype([("x", "<i4"), ("y", "<i4"), ("z", "<i4"),
                    ("other", "<f4"), ("cnt", "<f4", (K_TOP,)),
                    ("cls", "<u2", (K_TOP,))])
_TSDF_DT = np.dtype([("x", "<i4"), ("y", "<i4"), ("z", "<i4"),
                     ("distance", "<f4"), ("weight", "<f4")])
assert _BETA_DT.itemsize == 20, _BETA_DT.itemsize
assert _DIR_DT.itemsize == 28, _DIR_DT.itemsize

# v6 payload dtypes (coords live in the block run, not the record).
_BETA_Q_DT = np.dtype([("q_occ", "<u2"), ("q_free", "<u2")])
_BETA_F_DT = np.dtype([("a_occ", "<f4"), ("a_free", "<f4")])
_DIR_Q_DT = np.dtype([("q_other", "<u2"), ("q_cnt", "<u2", (K_TOP,)),
                      ("cls", "<u2", (K_TOP,))])
_DIR_F_DT = np.dtype([("other", "<f4"), ("cnt", "<f4", (K_TOP,)),
                      ("cls", "<u2", (K_TOP,))])


def lz4_unwrap(blob: bytes) -> bytes:
    """Undo compressLZ4: 4-byte big-endian size header + raw LZ4 block."""
    import lz4.block
    if len(blob) <= 4:
        return b""
    orig = struct.unpack_from(">I", blob, 0)[0]
    return lz4.block.decompress(blob[4:], uncompressed_size=orig)


def _read_block_stream(raw: bytes, off: int, count: int, payload_dt: np.dtype):
    """Parse v6 block runs -> (coords [n,3] i32, payloads structured, new off)."""
    coords = np.empty((count, 3), np.int32)
    payloads = np.empty(count, payload_dt)
    got = 0
    while got < count:
        bx, by, bz, mode = struct.unpack_from("<iiiB", raw, off)
        off += 13
        if mode == 0:
            mask = np.frombuffer(raw, np.uint8, 64, off)
            off += 64
            idx = np.nonzero(np.unpackbits(mask, bitorder="little"))[0]
        elif mode == 1:
            (n,) = struct.unpack_from("<H", raw, off)
            off += 2
            idx = np.frombuffer(raw, "<u2", n, off).astype(np.int64)
            off += 2 * n
        else:
            raise ValueError(f"bad block mode {mode}")
        n = len(idx)
        if n == 0 or got + n > count:
            raise ValueError("malformed block run (empty or overshoots count)")
        coords[got:got + n, 0] = bx * 8 + (idx >> 6)
        coords[got:got + n, 1] = by * 8 + ((idx >> 3) & 7)
        coords[got:got + n, 2] = bz * 8 + (idx & 7)
        payloads[got:got + n] = np.frombuffer(raw, payload_dt, n, off)
        off += n * payload_dt.itemsize
        got += n
    return coords, payloads, off


def decode_frame(raw: bytes) -> dict:
    """Parse an *uncompressed* SCVX frame (v5 or v6) -> dict of arrays."""
    off = 0
    magic, ver = struct.unpack_from("<IB", raw, off); off += 5
    if magic != MAGIC:
        raise ValueError(f"bad MAGIC {magic:#x}")
    if ver not in SUPPORTED_VERSIONS:
        raise ValueError(f"unsupported VERSION {ver} (know {SUPPORTED_VERSIONS})")
    resolution, num_classes, k_top, alpha_0 = struct.unpack_from("<fHBf", raw, off)
    off += 4 + 2 + 1 + 4
    if k_top != K_TOP:
        raise ValueError(f"K_TOP mismatch: wire={k_top} decoder={K_TOP}")
    quant_step = 0.0
    if ver >= 6:
        (quant_step,) = struct.unpack_from("<f", raw, off)
        off += 4

    def read_flat(off, dt):
        (count,) = struct.unpack_from("<I", raw, off); off += 4
        arr = np.frombuffer(raw, dtype=dt, count=count, offset=off).copy()
        off += count * dt.itemsize
        return arr, off

    # TSDF stream: flat records in both revisions.
    tsdf, off = read_flat(off, _TSDF_DT)

    if ver == 5:
        beta, off = read_flat(off, _BETA_DT)
        dir_, off = read_flat(off, _DIR_DT)
    else:
        quant = quant_step > 0.0
        other_prior = max(0, num_classes - K_TOP) * alpha_0

        (count,) = struct.unpack_from("<I", raw, off); off += 4
        bxyz, bpay, off = _read_block_stream(
            raw, off, count, _BETA_Q_DT if quant else _BETA_F_DT)
        beta = np.empty(count, _BETA_DT)
        beta["x"], beta["y"], beta["z"] = bxyz[:, 0], bxyz[:, 1], bxyz[:, 2]
        if quant:
            beta["a_occ"] = BETA_OCC_PRIOR + bpay["q_occ"].astype(np.float32) * quant_step
            beta["a_free"] = BETA_FREE_PRIOR + bpay["q_free"].astype(np.float32) * quant_step
        else:
            beta["a_occ"], beta["a_free"] = bpay["a_occ"], bpay["a_free"]

        (count,) = struct.unpack_from("<I", raw, off); off += 4
        dxyz, dpay, off = _read_block_stream(
            raw, off, count, _DIR_Q_DT if quant else _DIR_F_DT)
        dir_ = np.empty(count, _DIR_DT)
        dir_["x"], dir_["y"], dir_["z"] = dxyz[:, 0], dxyz[:, 1], dxyz[:, 2]
        dir_["cls"] = dpay["cls"]
        if quant:
            dir_["other"] = other_prior + dpay["q_other"].astype(np.float32) * quant_step
            dir_["cnt"] = alpha_0 + dpay["q_cnt"].astype(np.float32) * quant_step
        else:
            dir_["other"], dir_["cnt"] = dpay["other"], dpay["cnt"]

    return {"resolution": resolution, "num_classes": int(num_classes),
            "alpha_0": alpha_0, "k_top": int(k_top), "version": int(ver),
            "quant_step": float(quant_step),
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
    print(f"v{fr['version']} res={fr['resolution']} C={fr['num_classes']} "
          f"a0={fr['alpha_0']} step={fr['quant_step']:.6g} "
          f"beta={len(fr['beta'])} dir={len(fr['dir'])} tsdf={len(fr['tsdf'])}")
    if len(fr["beta"]):
        b = fr["beta"]
        s = b["a_occ"] + b["a_free"]
        p = b["a_occ"] / s
        print(f"  p_occ: min={p.min():.3f} med={np.median(p):.3f} max={p.max():.3f} "
              f"| free(p<.5)={np.mean(p<0.5):.2%} occ(p>.5)={np.mean(p>0.5):.2%}")
