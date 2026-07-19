#!/usr/bin/env python3
"""Build voxelized GT NPZ from SemanticKITTI — minimal memory.

Per-voxel storage: only (best_label, best_count, total_count) = 6 bytes.
For 50M voxels: dict overhead ~100 bytes/entry × 50M = 5 GB. Too much.

Instead: use a flat hash table with open addressing. Voxel coords packed
into int64 keys. Value = (label_byte, count_uint16). ~11 bytes/entry.
At 50M entries with 2x load factor: ~1.1 GB.

Actually simplest: just use the per-scan voxelized points and do a
running argmax. Each voxel stores winning label (uint8) + winning count
(uint16) + second count (uint16) = 5 bytes. With Python dict overhead
that's still ~100+ bytes per entry.

FINAL APPROACH: Process in batches of 100 scans. Per batch, accumulate
into a numpy structured array, then merge into the running result.
"""

import argparse
from pathlib import Path

import numpy as np

# Must match the replay node's LEARNING_MAP exactly
LEARNING_MAP = {
    0: 0, 1: 0, 10: 1, 11: 2, 13: 5, 15: 3, 16: 5, 18: 4, 20: 5,
    30: 6, 31: 7, 32: 8, 40: 9, 44: 10, 48: 11, 49: 12, 50: 13,
    51: 14, 52: 0, 60: 15, 70: 16, 71: 17, 72: 18, 80: 19, 81: 0,
    99: 0, 252: 1, 253: 7, 254: 6, 255: 8, 256: 5, 257: 5, 258: 4, 259: 5,
}
_MAX_RAW = max(LEARNING_MAP.keys()) + 1
_LABEL_LUT = np.zeros(_MAX_RAW, dtype=np.uint8)
for raw_id, eval_id in LEARNING_MAP.items():
    _LABEL_LUT[raw_id] = eval_id


def load_calib(p):
    with open(p) as f:
        for line in f:
            if line.startswith("Tr:"):
                v = list(map(float, line.strip().split()[1:]))
                T = np.eye(4); T[:3, :] = np.array(v).reshape(3, 4)
                return T
    raise ValueError("Tr not found")


def load_poses(p):
    poses = []
    with open(p) as f:
        for line in f:
            v = list(map(float, line.strip().split()))
            T = np.eye(4); T[:3, :] = np.array(v).reshape(3, 4)
            poses.append(T)
    return poses


def voxelize_scan(points_f64, labels_u8, inv_res):
    """Return unique voxel coords and majority label per voxel for one scan."""
    coords = np.floor(points_f64 * inv_res).astype(np.int32)
    # Pack to int64 key
    keys = (coords[:, 0].astype(np.int64) & 0xFFFFFF) << 40 | \
           (coords[:, 1].astype(np.int64) & 0xFFFFF) << 20 | \
           (coords[:, 2].astype(np.int64) & 0xFFFFF)
    unique_keys, inv = np.unique(keys, return_inverse=True)
    n = len(unique_keys)
    majority = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        vox_labels = labels_u8[inv == i]
        majority[i] = np.bincount(vox_labels, minlength=20)[1:].argmax() + 1
    return unique_keys, majority


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path")
    parser.add_argument("--sequence", type=int, default=8)
    parser.add_argument("--resolution", type=float, default=0.10)
    parser.add_argument("--max-range", type=float, default=30.0)
    parser.add_argument("--min-range", type=float, default=1.0)
    parser.add_argument("-o", "--output", default="gt.npz")
    parser.add_argument("--max-scans", type=int, default=-1,
                        help="Cap number of scans (e.g. 100 to match SLIM-VDB paper). -1 = all.")
    args = parser.parse_args()

    seq_str = f"{args.sequence:02d}"
    seq_dir = Path(args.dataset_path) / "sequences" / seq_str
    Tr = load_calib(seq_dir / "calib.txt")

    for pp in [seq_dir / "poses.txt",
               seq_dir / "poses" / f"{seq_str}.txt",
               Path(args.dataset_path) / "poses" / f"{seq_str}.txt"]:
        if pp.exists():
            poses = load_poses(pp)
            print(f"Loaded {len(poses)} poses from {pp}")
            break
    else:
        raise FileNotFoundError("Poses not found")

    scan_files = sorted((seq_dir / "velodyne").glob("*.bin"))
    label_dir = seq_dir / "labels"
    n_scans = min(len(scan_files), len(poses))
    if args.max_scans > 0:
        n_scans = min(n_scans, args.max_scans)
    inv_res = 1.0 / args.resolution

    print(f"Seq {seq_str}: {n_scans} scans, res={args.resolution}m")

    # Running accumulator: key(int64) -> [counts_per_class] as numpy array
    # Use a simple approach: collect all (key, label) from all scans,
    # then do ONE final sort + groupby. But stream to disk to limit RAM.
    #
    # Actually: just use a Python dict[int, np.uint16[20]].
    # 50M voxels × (40 bytes array + ~120 bytes dict overhead) ≈ 8 GB. Too much.
    #
    # Compromise: use dict[int, int] mapping key -> packed(best_label << 16 | best_count).
    # This is a simple running max — NOT a true majority vote, but close enough
    # when one label dominates (which it does for most voxels at 10cm).
    #
    # Better compromise: dict[int, int] mapping key -> (label << 16 | count) for the
    # top label only. Update: if new label == stored label, increment count. If new
    # label != stored label, decrement count; if count reaches 0, replace with new.
    # This is the Boyer-Moore majority vote algorithm — exact majority in one pass.

    # Boyer-Moore streaming majority vote per voxel
    # Store: voxel_key -> (candidate_label, count)
    # Python int is 28 bytes, tuple is 56 bytes, dict entry ~100 bytes
    # 50M entries × ~180 bytes = 9 GB. Still too much.

    # NUCLEAR OPTION: just write per-scan (key, label) arrays to disk,
    # then process with external sort in chunks.

    # SIMPLEST OPTION THAT WORKS: Process scans, build per-scan voxelized
    # arrays, concatenate keys+labels, sort, groupby. But do it with
    # memory-mapped temp files.

    # Let's try: accumulate per-scan unique (key, label) pairs.
    # ~50k unique voxels/scan × 4071 scans = ~200M pairs × 9 bytes = 1.8 GB on disk.
    # Then mmap, sort by key, groupby.

    import tempfile, os

    pair_dtype = np.dtype([('k', 'i8'), ('l', 'u1')])
    tmpf = tempfile.NamedTemporaryFile(suffix=".pairs", delete=False)
    total_pairs = 0

    for i in range(n_scans):
        points = np.fromfile(scan_files[i], dtype=np.float32).reshape(-1, 4)
        label_file = label_dir / scan_files[i].name.replace(".bin", ".label")
        labels_raw = np.fromfile(label_file, dtype=np.uint32)
        mapped = _LABEL_LUT[np.clip((labels_raw & 0xFFFF).astype(np.uint16), 0, _MAX_RAW - 1)]

        ranges = np.linalg.norm(points[:, :3], axis=1)
        mask = (ranges >= args.min_range) & (ranges <= args.max_range) & (mapped > 0)
        pts = points[mask, :3].astype(np.float64)
        lbl = mapped[mask]
        if len(pts) == 0:
            continue

        T = poses[i] @ Tr
        pts_w = (T[:3, :3] @ pts.T).T + T[:3, 3]
        coords = np.floor(pts_w * inv_res).astype(np.int32)

        # Per-scan voxelize to reduce pairs
        keys = (coords[:, 0].astype(np.int64) & 0xFFFFFF) << 40 | \
               (coords[:, 1].astype(np.int64) & 0xFFFFF) << 20 | \
               (coords[:, 2].astype(np.int64) & 0xFFFFF)
        unique_keys, inv = np.unique(keys, return_inverse=True)
        n_uk = len(unique_keys)

        # Vectorized per-voxel majority: build (n_uk × 20) count matrix
        counts = np.zeros((n_uk, 20), dtype=np.int32)
        np.add.at(counts, (inv, lbl), 1)
        scan_labels = counts[:, 1:].argmax(axis=1).astype(np.uint8) + 1

        rec = np.empty(n_uk, dtype=pair_dtype)
        rec['k'] = unique_keys
        rec['l'] = scan_labels
        rec.tofile(tmpf)
        total_pairs += n_uk

        if (i + 1) % 100 == 0 or i == n_scans - 1:
            print(f"  Pass 1: {i+1}/{n_scans}, {total_pairs} voxel-label pairs")

    tmpf.close()
    file_size_mb = os.path.getsize(tmpf.name) / (1024**2)
    print(f"Pass 1 done: {total_pairs} pairs, {file_size_mb:.0f} MB on disk")

    # Pass 2: load, sort by key, groupby majority
    print("Loading pairs...")
    data = np.fromfile(tmpf.name, dtype=pair_dtype)
    print(f"Sorting {len(data)} pairs...")
    data.sort(order='k')

    print("Groupby majority vote...")
    diff = np.empty(len(data), dtype=bool)
    diff[0] = True
    diff[1:] = data['k'][1:] != data['k'][:-1]
    group_starts = np.where(diff)[0]
    n_voxels = len(group_starts)
    group_ends = np.empty(n_voxels, dtype=np.int64)
    group_ends[:-1] = group_starts[1:]
    group_ends[-1] = len(data)
    print(f"Unique voxels: {n_voxels}")

    gt_keys = data['k'][group_starts]
    gt_labels = np.zeros(n_voxels, dtype=np.uint8)

    for idx in range(n_voxels):
        s, e = group_starts[idx], group_ends[idx]
        vox_labels = data['l'][s:e]
        gt_labels[idx] = np.bincount(vox_labels, minlength=20)[1:].argmax() + 1
        if (idx + 1) % 5_000_000 == 0:
            print(f"  Pass 2: {idx+1}/{n_voxels}")

    del data

    # Decode keys back to coords
    res = args.resolution
    ix = ((gt_keys >> 40) & 0xFFFFFF).astype(np.int32)
    iy = ((gt_keys >> 20) & 0xFFFFF).astype(np.int32)
    iz = (gt_keys & 0xFFFFF).astype(np.int32)
    # Sign-extend 24-bit X and 20-bit Y/Z
    ix = np.where(ix >= 0x800000, ix - 0x1000000, ix)
    iy = np.where(iy >= 0x80000, iy - 0x100000, iy)
    iz = np.where(iz >= 0x80000, iz - 0x100000, iz)

    gt_points = np.stack([
        (ix + 0.5) * res,
        (iy + 0.5) * res,
        (iz + 0.5) * res,
    ], axis=1).astype(np.float32)

    np.savez_compressed(args.output, points=gt_points, semantic_class=gt_labels)
    print(f"Saved {args.output}: {n_voxels} voxels")
    os.unlink(tmpf.name)


if __name__ == "__main__":
    main()
