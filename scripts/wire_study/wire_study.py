#!/usr/bin/env python3
"""Offline re-encoding study of recorded scovox_bin (ScovoxMapBinary v5) frames.

Read-only: decodes captured frames, re-encodes the payload several ways,
and compares LZ4-compressed sizes. Touches no scovox source.

Wire framing (lz4_codec.hpp): 4-byte BIG-endian original size + raw LZ4 block.
Frame layout (binary_serializer.hpp v5, little-endian, packed):
  MAGIC u32 | VERSION u8 | resolution f32 | num_classes u16 | K_TOP u8 | alpha_0 f32
  tsdf_count u32 + 20B records
  beta_count u32 + 20B records (x,y,z i32 | a_occ,a_free f32)
  dir_count  u32 + 28B records (x,y,z i32 | other f32 | cnt[2] f32 | cls[2] u16)
"""
import struct
import sys

import numpy as np
import lz4.block

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from scovox_msgs.msg import ScovoxMapBinary

BAG = sys.argv[1] if len(sys.argv) > 1 else "/tmp/scovox_bin_cap"
LEAF_BITS = 3          # 8x8x8 = 512 voxels per Bonxai leaf block (from node log)
BLOCK = 1 << LEAF_BITS
BLOCK_VOX = BLOCK ** 3

BETA_DT = np.dtype([("x", "<i4"), ("y", "<i4"), ("z", "<i4"),
                    ("a_occ", "<f4"), ("a_free", "<f4")])
DIR_DT = np.dtype([("x", "<i4"), ("y", "<i4"), ("z", "<i4"),
                   ("other", "<f4"), ("cnt", "<f4", (2,)), ("cls", "<u2", (2,))])

assert BETA_DT.itemsize == 20, BETA_DT.itemsize
assert DIR_DT.itemsize == 28, DIR_DT.itemsize


def lz4_size(buf: bytes) -> int:
    """Compressed size under the same codec the wire uses (raw block + 4B size)."""
    if not buf:
        return 0
    return 4 + len(lz4.block.compress(buf, mode="default", store_size=False))


def parse_frame(blob: bytes):
    magic, ver = struct.unpack_from("<IB", blob, 0)
    assert magic == 0x53435658, f"bad magic {magic:#x}"
    res, nclass, ktop = struct.unpack_from("<fHB", blob, 5)
    alpha0, = struct.unpack_from("<f", blob, 12)
    off = 16
    tsdf_n, = struct.unpack_from("<I", blob, off); off += 4
    off += tsdf_n * 20
    beta_n, = struct.unpack_from("<I", blob, off); off += 4
    beta = np.frombuffer(blob, dtype=BETA_DT, count=beta_n, offset=off)
    off += beta_n * 20
    dir_n, = struct.unpack_from("<I", blob, off); off += 4
    dirv = np.frombuffer(blob, dtype=DIR_DT, count=dir_n, offset=off)
    off += dir_n * 28
    return dict(ver=ver, res=res, nclass=nclass, ktop=ktop, alpha0=alpha0,
                tsdf_n=tsdf_n, beta=beta, dir=dirv, end=off)


def block_pack(coords, payload_bytes_per_vox, payloads):
    """Block-run coord coding: group voxels by Bonxai leaf block.

    Per block: 12B block coord + 1B mode + occupancy set + payloads.
    Mode 0 = 64B bitmask (512 bits); mode 1 = n * 2B in-block indices.
    Adaptive: whichever is smaller. Returns (nbytes, packed_bytes).
    """
    bx = coords[:, 0] >> LEAF_BITS
    by = coords[:, 1] >> LEAF_BITS
    bz = coords[:, 2] >> LEAF_BITS
    # stable key for grouping
    key = (bx.astype(np.int64) << 42) ^ (by.astype(np.int64) << 21) ^ bz.astype(np.int64)
    order = np.argsort(key, kind="stable")
    key_s = key[order]
    bounds = np.flatnonzero(np.diff(key_s)) + 1
    groups = np.split(order, bounds)

    out = bytearray()
    for g in groups:
        n = len(g)
        i0 = g[0]
        out += struct.pack("<iii", int(bx[i0]), int(by[i0]), int(bz[i0]))
        mask_cost, idx_cost = 64, 2 * n
        if mask_cost <= idx_cost:
            out += b"\x00"
            mask = bytearray(64)
            for i in g:
                lx = int(coords[i, 0]) & (BLOCK - 1)
                ly = int(coords[i, 1]) & (BLOCK - 1)
                lz_ = int(coords[i, 2]) & (BLOCK - 1)
                bit = (lx << 6) | (ly << 3) | lz_
                mask[bit >> 3] |= 1 << (bit & 7)
            out += bytes(mask)
        else:
            out += b"\x01"
            out += struct.pack("<H", n)
            for i in g:
                lx = int(coords[i, 0]) & (BLOCK - 1)
                ly = int(coords[i, 1]) & (BLOCK - 1)
                lz_ = int(coords[i, 2]) & (BLOCK - 1)
                out += struct.pack("<H", (lx << 6) | (ly << 3) | lz_)
        out += payloads[g].tobytes()
    return len(out), bytes(out)


def main():
    reader = SequentialReader()
    reader.open(StorageOptions(uri=BAG, storage_id="mcap"),
                ConverterOptions("", ""))

    stats = []
    all_beta = []
    all_dir = []
    wire_total = 0
    meta = None

    while reader.has_next():
        _topic, raw, _t = reader.read_next()
        msg = deserialize_message(raw, ScovoxMapBinary)
        comp = bytes(msg.data)
        if len(comp) <= 4:
            continue
        wire_total += len(comp)
        orig = struct.unpack(">I", comp[:4])[0]
        blob = lz4.block.decompress(comp[4:], uncompressed_size=orig)
        f = parse_frame(blob)
        meta = meta or f
        stats.append((len(comp), orig, len(f["beta"]), len(f["dir"]), f["tsdf_n"]))
        all_beta.append(f["beta"])
        all_dir.append(f["dir"])

    if not stats:
        print("no frames")
        return

    beta = np.concatenate([b for b in all_beta if len(b)]) if any(len(b) for b in all_beta) else np.empty(0, BETA_DT)
    dirv = np.concatenate([d for d in all_dir if len(d)]) if any(len(d) for d in all_dir) else np.empty(0, DIR_DT)

    n_frames = len(stats)
    wire_uncomp = sum(s[1] for s in stats)
    print("=" * 74)
    print(f"CAPTURE  frames={n_frames}  res={meta['res']:.3f}  C={meta['nclass']} "
          f"K_TOP={meta['ktop']}  alpha_0={meta['alpha0']:.4f}  codec_v{meta['ver']}")
    print(f"  beta voxels total : {len(beta):,}   ({len(beta)/n_frames:,.0f}/frame)")
    print(f"  dir  voxels total : {len(dirv):,}   ({len(dirv)/n_frames:,.0f}/frame)")
    print(f"  tsdf voxels total : {sum(s[4] for s in stats):,}  (share_tsdf off => 0)")
    print(f"  ON-WIRE (lz4)     : {wire_total/1e6:.2f} MB   uncompressed {wire_uncomp/1e6:.2f} MB"
          f"   ratio {wire_uncomp/max(wire_total,1):.2f}x")
    print()

    # ---- prior-relative evidence -------------------------------------------
    BETA_PRIOR = 1.0                                   # kBetaOccPrior/kBetaFreePrior
    other_prior = max(meta["nclass"] - meta["ktop"], 0) * meta["alpha0"]
    e_occ = beta["a_occ"] - BETA_PRIOR
    e_free = beta["a_free"] - BETA_PRIOR
    print("EVIDENCE ABOVE PRIOR (what a quantizer must actually represent)")
    for nm, arr in (("a_occ-1", e_occ), ("a_free-1", e_free)):
        if len(arr):
            print(f"  {nm:9s} min={arr.min():9.3f} p50={np.percentile(arr,50):8.3f} "
                  f"p99={np.percentile(arr,99):8.3f} max={arr.max():9.3f}")
    if len(dirv):
        e_other = dirv["other"] - other_prior
        e_cnt = dirv["cnt"] - meta["alpha0"]
        print(f"  {'other-p':9s} min={e_other.min():9.3f} p50={np.percentile(e_other,50):8.3f} "
              f"p99={np.percentile(e_other,99):8.3f} max={e_other.max():9.3f}")
        print(f"  {'cnt-a0':9s} min={e_cnt.min():9.3f} p50={np.percentile(e_cnt,50):8.3f} "
              f"p99={np.percentile(e_cnt,99):8.3f} max={e_cnt.max():9.3f}")
    print()

    # ---- choose quantization steps -----------------------------------------
    def pick_step(arr, levels):
        if not len(arr):
            return 1.0 / 256
        need = float(np.abs(arr).max()) / (levels - 1)
        k = 0
        step = 1.0
        while step / 2 >= need and k < 20:
            step /= 2
            k += 1
        while step < need:
            step *= 2
        return step

    step16 = pick_step(np.concatenate([e_occ, e_free]) if len(beta) else np.array([1.0]), 65536)
    step8 = pick_step(np.concatenate([e_occ, e_free]) if len(beta) else np.array([1.0]), 256)
    print(f"QUANTIZER  u16 step={step16:g}  ({step16:.6f} counts/LSB)   "
          f"u8 step={step8:g}")

    def quant(arr, step, dt, lo=0):
        q = np.rint(np.clip(arr, 0, None) / step)
        q = np.clip(q, 0, np.iinfo(dt).max)
        return q.astype(dt)

    # fidelity: p_occ error under u16 and u8
    if len(beta):
        for label, st, dt in (("u16", step16, np.uint16), ("u8", step8, np.uint8)):
            qo = quant(e_occ, st, dt).astype(np.float64) * st + BETA_PRIOR
            qf = quant(e_free, st, dt).astype(np.float64) * st + BETA_PRIOR
            p0 = beta["a_occ"].astype(np.float64) / (beta["a_occ"] + beta["a_free"]).astype(np.float64)
            p1 = qo / (qo + qf)
            err = np.abs(p1 - p0)
            print(f"  {label}: p_occ err  mean={err.mean():.2e}  p99={np.percentile(err,99):.2e}  max={err.max():.2e}")
    print()

    # ---- build variants -----------------------------------------------------
    beta_coords = np.stack([beta["x"], beta["y"], beta["z"]], axis=1) if len(beta) else np.zeros((0, 3), np.int32)
    dir_coords = np.stack([dirv["x"], dirv["y"], dirv["z"]], axis=1) if len(dirv) else np.zeros((0, 3), np.int32)

    variants = {}

    # V0 baseline: exactly what ships (coord i32*3 + f32 payloads)
    v0 = beta.tobytes() + dirv.tobytes()
    variants["V0 shipped (f32, raw coords)"] = v0

    # V1 u16 payloads, raw coords
    if len(beta):
        b16 = np.empty(len(beta), dtype=np.dtype([("c", "<i4", (3,)), ("o", "<u2"), ("f", "<u2")]))
        b16["c"] = beta_coords
        b16["o"] = quant(e_occ, step16, np.uint16)
        b16["f"] = quant(e_free, step16, np.uint16)
    else:
        b16 = np.empty(0, dtype=np.dtype([("c", "<i4", (3,)), ("o", "<u2"), ("f", "<u2")]))
    if len(dirv):
        d16 = np.empty(len(dirv), dtype=np.dtype([("c", "<i4", (3,)), ("ot", "<u2"),
                                                  ("cnt", "<u2", (2,)), ("cls", "<u2", (2,))]))
        d16["c"] = dir_coords
        d16["ot"] = quant(dirv["other"] - other_prior, step16, np.uint16)
        d16["cnt"] = quant(dirv["cnt"] - meta["alpha0"], step16, np.uint16)
        d16["cls"] = dirv["cls"]
    else:
        d16 = np.empty(0, dtype=np.dtype([("c", "<i4", (3,)), ("ot", "<u2"),
                                          ("cnt", "<u2", (2,)), ("cls", "<u2", (2,))]))
    variants["V1 u16 payloads, raw coords"] = b16.tobytes() + d16.tobytes()

    # V2 u8 payloads, raw coords (beta only quantized to u8; dir keeps u16 counts)
    if len(beta):
        b8 = np.empty(len(beta), dtype=np.dtype([("c", "<i4", (3,)), ("o", "u1"), ("f", "u1")]))
        b8["c"] = beta_coords
        b8["o"] = quant(e_occ, step8, np.uint8)
        b8["f"] = quant(e_free, step8, np.uint8)
    else:
        b8 = np.empty(0, dtype=np.dtype([("c", "<i4", (3,)), ("o", "u1"), ("f", "u1")]))
    variants["V2 u8 payloads, raw coords"] = b8.tobytes() + d16.tobytes()

    # V3 block coords + f32 payloads (isolates the coord win)
    pb_f32 = np.empty(len(beta), dtype=np.dtype([("o", "<f4"), ("f", "<f4")]))
    if len(beta):
        pb_f32["o"] = beta["a_occ"]; pb_f32["f"] = beta["a_free"]
    pd_f32 = np.empty(len(dirv), dtype=np.dtype([("ot", "<f4"), ("cnt", "<f4", (2,)), ("cls", "<u2", (2,))]))
    if len(dirv):
        pd_f32["ot"] = dirv["other"]; pd_f32["cnt"] = dirv["cnt"]; pd_f32["cls"] = dirv["cls"]
    nb3, bb3 = block_pack(beta_coords, 8, pb_f32) if len(beta) else (0, b"")
    nd3, db3 = block_pack(dir_coords, 16, pd_f32) if len(dirv) else (0, b"")
    variants["V3 block coords + f32"] = bb3 + db3

    # V4 block coords + u16 payloads (combined)
    pb_u16 = np.empty(len(beta), dtype=np.dtype([("o", "<u2"), ("f", "<u2")]))
    if len(beta):
        pb_u16["o"] = quant(e_occ, step16, np.uint16); pb_u16["f"] = quant(e_free, step16, np.uint16)
    pd_u16 = np.empty(len(dirv), dtype=np.dtype([("ot", "<u2"), ("cnt", "<u2", (2,)), ("cls", "<u2", (2,))]))
    if len(dirv):
        pd_u16["ot"] = quant(dirv["other"] - other_prior, step16, np.uint16)
        pd_u16["cnt"] = quant(dirv["cnt"] - meta["alpha0"], step16, np.uint16)
        pd_u16["cls"] = dirv["cls"]
    nb4, bb4 = block_pack(beta_coords, 4, pb_u16) if len(beta) else (0, b"")
    nd4, db4 = block_pack(dir_coords, 10, pd_u16) if len(dirv) else (0, b"")
    variants["V4 block coords + u16"] = bb4 + db4

    # ---- report -------------------------------------------------------------
    base_raw = len(variants["V0 shipped (f32, raw coords)"])
    base_lz4 = lz4_size(variants["V0 shipped (f32, raw coords)"])
    print("=" * 74)
    print(f"{'variant':34s} {'raw MB':>9s} {'lz4 MB':>9s} {'vs V0 raw':>10s} {'vs V0 lz4':>10s}")
    print("-" * 74)
    for name, buf in variants.items():
        raw = len(buf)
        c = lz4_size(buf)
        print(f"{name:34s} {raw/1e6:9.2f} {c/1e6:9.2f} "
              f"{raw/base_raw:9.2f}x {c/base_lz4:9.2f}x")
    print("-" * 74)
    print(f"per-voxel bytes on wire (lz4, beta+dir mixed):")
    nv = len(beta) + len(dirv)
    for name, buf in variants.items():
        print(f"  {name:34s} {lz4_size(buf)/max(nv,1):6.2f} B/voxel")
    print()
    secs = n_frames / 2.0   # share timer 2 Hz
    print(f"link rate @2Hz over {secs:.0f}s of sharing:")
    for name, buf in variants.items():
        print(f"  {name:34s} {lz4_size(buf)/max(secs,1)/1e3:8.1f} kB/s")


if __name__ == "__main__":
    main()
