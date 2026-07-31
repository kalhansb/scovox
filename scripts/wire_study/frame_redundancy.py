#!/usr/bin/env python3
"""How much of each scovox_bin frame is a re-send of a value the peer already has?

Read-only. The merger is snapshot-replace per source coord, so any (coord,value)
identical to the last value sent for that coord is a pure no-op on the wire.
"""
import struct
import sys

import numpy as np
import lz4.block

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from scovox_msgs.msg import ScovoxMapBinary

BAG = sys.argv[1] if len(sys.argv) > 1 else "/tmp/scovox_bin_cap"

BETA_DT = np.dtype([("x", "<i4"), ("y", "<i4"), ("z", "<i4"),
                    ("a_occ", "<f4"), ("a_free", "<f4")])
DIR_DT = np.dtype([("x", "<i4"), ("y", "<i4"), ("z", "<i4"),
                   ("other", "<f4"), ("cnt", "<f4", (2,)), ("cls", "<u2", (2,))])


def lz4_size(buf):
    if not buf:
        return 0
    return 4 + len(lz4.block.compress(buf, mode="default", store_size=False))


def parse(blob):
    off = 16
    tsdf_n, = struct.unpack_from("<I", blob, off); off += 4 + tsdf_n * 20
    beta_n, = struct.unpack_from("<I", blob, off); off += 4
    beta = np.frombuffer(blob, dtype=BETA_DT, count=beta_n, offset=off)
    off += beta_n * 20
    dir_n, = struct.unpack_from("<I", blob, off); off += 4
    dirv = np.frombuffer(blob, dtype=DIR_DT, count=dir_n, offset=off)
    return beta, dirv


def ckey(a):
    return (a["x"].astype(np.int64) << 42) ^ (a["y"].astype(np.int64) << 21) ^ a["z"].astype(np.int64)


def main():
    reader = SequentialReader()
    reader.open(StorageOptions(uri=BAG, storage_id="mcap"), ConverterOptions("", ""))

    seen = {}          # coord key -> (a_occ, a_free) last SENT
    tot = new = changed = redundant = 0
    wire = 0
    novel_lz4 = 0
    per_frame = []

    while reader.has_next():
        _t, raw, _ts = reader.read_next()
        msg = deserialize_message(raw, ScovoxMapBinary)
        comp = bytes(msg.data)
        if len(comp) <= 4:
            continue
        wire += len(comp)
        orig = struct.unpack(">I", comp[:4])[0]
        beta, _dirv = parse(lz4.block.decompress(comp[4:], uncompressed_size=orig))
        if not len(beta):
            continue
        k = ckey(beta)
        vo = beta["a_occ"]
        vf = beta["a_free"]

        is_new = np.empty(len(k), bool)
        is_same = np.zeros(len(k), bool)
        for i in range(len(k)):
            kk = int(k[i])
            prev = seen.get(kk)
            if prev is None:
                is_new[i] = True
            else:
                is_new[i] = False
                if prev[0] == vo[i] and prev[1] == vf[i]:
                    is_same[i] = True
            seen[kk] = (vo[i], vf[i])

        n_new = int(is_new.sum())
        n_same = int(is_same.sum())
        n_chg = len(k) - n_new - n_same
        tot += len(k); new += n_new; changed += n_chg; redundant += n_same
        novel = beta[~is_same]
        novel_lz4 += lz4_size(novel.tobytes())
        per_frame.append((len(k), n_new, n_chg, n_same))

    print("=" * 70)
    print(f"BETA STREAM re-send analysis over {len(per_frame)} frames")
    print(f"  voxel records sent      : {tot:,}")
    print(f"  ... first time (new)    : {new:,}   ({100*new/tot:.1f}%)")
    print(f"  ... value CHANGED       : {changed:,}   ({100*changed/tot:.1f}%)")
    print(f"  ... IDENTICAL re-send   : {redundant:,}   ({100*redundant/tot:.1f}%)  <-- no-op at merger")
    print(f"  unique coords touched   : {len(seen):,}")
    print(f"  avg re-sends per coord  : {tot/len(seen):.1f}x")
    print()
    print(f"  actual beta+dir on wire (lz4)      : {wire/1e6:8.2f} MB")
    print(f"  beta-only if identical re-sends    : {novel_lz4/1e6:8.2f} MB"
          f"   ({100*novel_lz4/max(wire,1):.0f}% of current wire)")
    print()
    print("  first 12 frames (records / new / changed / identical):")
    for i, (n, a, c, s) in enumerate(per_frame[:12]):
        print(f"    f{i:<3d} {n:8,}  {a:8,}  {c:8,}  {s:8,}")


if __name__ == "__main__":
    main()
