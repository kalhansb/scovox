#!/usr/bin/env python3
"""Does an INT representation unlock delta-vs-last-sent coding for the beta stream?

Floats can't be delta-coded cleanly; quantized ints can. 86% of records are
re-sends of an already-known voxel with a slightly changed value, so the
per-record delta should be tiny. Read-only, beta stream only (the dominant one).
"""
import struct
import sys

import numpy as np
import lz4.block

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from scovox_msgs.msg import ScovoxMapBinary

BAG = sys.argv[1] if len(sys.argv) > 1 else "/tmp/scovox_bin_cap"
LEAF_BITS = 3
BLOCK = 1 << LEAF_BITS
STEP = 1.0 / 64          # u16 step from wire_study.py
PRIOR = 1.0              # kBetaOccPrior / kBetaFreePrior

BETA_DT = np.dtype([("x", "<i4"), ("y", "<i4"), ("z", "<i4"),
                    ("a_occ", "<f4"), ("a_free", "<f4")])


def lz4_size(buf):
    return 4 + len(lz4.block.compress(buf, mode="default", store_size=False)) if buf else 0


def parse(blob):
    # Rev-5 flat layout only; rev-6 captures decode with
    # experiments/uncertainty/scovox_bin.py (see wire_study.py).
    assert blob[4] == 5, f"codec revision {blob[4]}: rev-5 captures only"
    off = 16
    tsdf_n, = struct.unpack_from("<I", blob, off); off += 4 + tsdf_n * 20
    beta_n, = struct.unpack_from("<I", blob, off); off += 4
    return np.frombuffer(blob, dtype=BETA_DT, count=beta_n, offset=off)


def block_pack(coords, payloads):
    bx = coords[:, 0] >> LEAF_BITS
    by = coords[:, 1] >> LEAF_BITS
    bz = coords[:, 2] >> LEAF_BITS
    key = (bx.astype(np.int64) << 42) ^ (by.astype(np.int64) << 21) ^ bz.astype(np.int64)
    order = np.argsort(key, kind="stable")
    key_s = key[order]
    groups = np.split(order, np.flatnonzero(np.diff(key_s)) + 1)
    out = bytearray()
    for g in groups:
        i0 = g[0]
        out += struct.pack("<iii", int(bx[i0]), int(by[i0]), int(bz[i0]))
        n = len(g)
        if 64 <= 2 * n:
            out += b"\x00"
            mask = bytearray(64)
            for i in g:
                bit = ((int(coords[i, 0]) & 7) << 6) | ((int(coords[i, 1]) & 7) << 3) | (int(coords[i, 2]) & 7)
                mask[bit >> 3] |= 1 << (bit & 7)
            out += bytes(mask)
        else:
            out += b"\x01" + struct.pack("<H", n)
            for i in g:
                out += struct.pack("<H", ((int(coords[i, 0]) & 7) << 6) | ((int(coords[i, 1]) & 7) << 3) | (int(coords[i, 2]) & 7))
        out += payloads[g].tobytes()
    return bytes(out)


def zigzag(a):
    return ((a << 1) ^ (a >> 63)).astype(np.uint64)


def varint_bytes(vals):
    """Total byte count if each zigzag value were LEB128 varint-encoded."""
    v = vals.astype(np.uint64)
    n = np.ones(len(v), np.int64)
    t = v >> np.uint64(7)
    while np.any(t > 0):
        n += (t > 0)
        t = t >> np.uint64(7)
    return int(n.sum())


def main():
    reader = SequentialReader()
    reader.open(StorageOptions(uri=BAG, storage_id="mcap"), ConverterOptions("", ""))

    last = {}
    tot_v0 = tot_abs = tot_delta = 0
    varint_total = 0
    delta_hist = []
    nvox = 0

    while reader.has_next():
        _t, raw, _ts = reader.read_next()
        msg = deserialize_message(raw, ScovoxMapBinary)
        comp = bytes(msg.data)
        if len(comp) <= 4:
            continue
        orig = struct.unpack(">I", comp[:4])[0]
        beta = parse(lz4.block.decompress(comp[4:], uncompressed_size=orig))
        if not len(beta):
            continue
        nvox += len(beta)
        coords = np.stack([beta["x"], beta["y"], beta["z"]], axis=1)

        qo = np.clip(np.rint((beta["a_occ"] - PRIOR) / STEP), 0, 65535).astype(np.int64)
        qf = np.clip(np.rint((beta["a_free"] - PRIOR) / STEP), 0, 65535).astype(np.int64)
        k = (beta["x"].astype(np.int64) << 42) ^ (beta["y"].astype(np.int64) << 21) ^ beta["z"].astype(np.int64)

        po = np.empty(len(k), np.int64)
        pf = np.empty(len(k), np.int64)
        for i in range(len(k)):
            kk = int(k[i])
            p = last.get(kk)
            if p is None:
                po[i] = 0; pf[i] = 0
            else:
                po[i] = p[0]; pf[i] = p[1]
            last[kk] = (qo[i], qf[i])

        do = qo - po
        df = qf - pf
        delta_hist.append(np.abs(np.concatenate([do, df])))

        # V0 shipped
        tot_v0 += lz4_size(beta.tobytes())
        # V4 block + absolute u16
        pab = np.empty(len(beta), dtype=np.dtype([("o", "<u2"), ("f", "<u2")]))
        pab["o"] = qo.astype(np.uint16); pab["f"] = qf.astype(np.uint16)
        tot_abs += lz4_size(block_pack(coords, pab))
        # V5 block + zigzag u16 delta
        pdl = np.empty(len(beta), dtype=np.dtype([("o", "<u2"), ("f", "<u2")]))
        pdl["o"] = np.clip(zigzag(do), 0, 65535).astype(np.uint16)
        pdl["f"] = np.clip(zigzag(df), 0, 65535).astype(np.uint16)
        tot_delta += lz4_size(block_pack(coords, pdl))
        varint_total += varint_bytes(zigzag(do)) + varint_bytes(zigzag(df))

    dh = np.concatenate(delta_hist)
    print("=" * 70)
    print(f"BETA delta-vs-last-sent (quantized, step={STEP})   voxel records={nvox:,}")
    print(f"  |delta| in LSB : p50={np.percentile(dh,50):.0f}  p90={np.percentile(dh,90):.0f} "
          f" p99={np.percentile(dh,99):.0f}  max={dh.max():.0f}")
    print(f"  fits in 1 varint byte (|dq|<64) : {100*np.mean(dh < 64):.1f}% of values")
    print()
    print(f"  V0 shipped   (f32, raw coords)        {tot_v0/1e6:8.2f} MB   1.00x")
    print(f"  V4 block coords + u16 ABSOLUTE        {tot_abs/1e6:8.2f} MB   {tot_abs/tot_v0:.2f}x")
    print(f"  V5 block coords + u16 DELTA           {tot_delta/1e6:8.2f} MB   {tot_delta/tot_v0:.2f}x")
    print(f"  V6 varint delta payload only (no lz4) {varint_total/1e6:8.2f} MB payload")
    print()
    print(f"  link rate @2Hz:  V0 {tot_v0/77/1e3:.0f} kB/s   V4 {tot_abs/77/1e3:.0f} kB/s   "
          f"V5 {tot_delta/77/1e3:.0f} kB/s")


if __name__ == "__main__":
    main()
