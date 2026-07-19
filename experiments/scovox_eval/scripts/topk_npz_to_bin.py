#!/usr/bin/env python3
"""Convert per-frame DENSE class-probability NPZs (from the post-hoc class
merge in run_mask2former_replica.py / run_polarseg_inference.py) to flat
binary blobs that scovox_node can mmap without a zip/npz parser.

Per-pixel/per-point probabilities are stored as a dense uint8 vector of
length C (number of SCovox target classes), quantized linearly with
full-scale = 1.0 (resolution ≈ 0.4%, well below network noise). The slot
index *is* the SCovox class id — slot 0 = unknown/unlabeled, slot 1..C-1 =
valid target classes.

Layout:
  Pointcloud (KITTI):
      header: uint32 N, uint8 C
      body:   N*C * uint8 probs (×255)
      ext:    .topk

  Image (Replica):
      header: uint16 H, uint16 W, uint8 C
      body:   H*W*C * uint8 probs (×255)
      ext:    .topk
"""
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm


def convert_pointcloud(npz_path: Path, out_path: Path) -> None:
    d = np.load(npz_path)
    probs = d["probs"]                                   # (N, C) uint8 (already quantized)
    if probs.dtype != np.uint8:
        # Float input — quantize ourselves.
        p = np.clip(probs.astype(np.float32), 0.0, 1.0)
        probs = np.round(p * 255.0).astype(np.uint8)
    N, C = probs.shape
    if C > 255:
        raise ValueError(f"too many classes ({C}) — header uses uint8")
    with open(out_path, "wb") as f:
        f.write(np.array([N], dtype=np.uint32).tobytes())
        f.write(np.array([C], dtype=np.uint8).tobytes())
        f.write(probs.tobytes())


def convert_image(npz_path: Path, out_path: Path) -> None:
    d = np.load(npz_path)
    probs = d["probs"]                                   # (H, W, C) uint8
    if probs.dtype != np.uint8:
        p = np.clip(probs.astype(np.float32), 0.0, 1.0)
        probs = np.round(p * 255.0).astype(np.uint8)
    H, W, C = probs.shape
    if max(H, W) > 0xFFFF:
        raise ValueError(f"H or W out of uint16 range in {npz_path}: {(H,W)}")
    if C > 255:
        raise ValueError(f"too many classes ({C}) — header uses uint8")
    with open(out_path, "wb") as f:
        f.write(np.array([H, W], dtype=np.uint16).tobytes())
        f.write(np.array([C], dtype=np.uint8).tobytes())
        f.write(probs.tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src_dir", type=Path, required=True,
                    help="Directory of <idx>.npz files")
    ap.add_argument("--dst_dir", type=Path, required=True,
                    help="Output directory for <idx>.topk files")
    ap.add_argument("--mode", choices=["pointcloud", "image"], required=True)
    args = ap.parse_args()

    args.dst_dir.mkdir(parents=True, exist_ok=True)
    npzs = sorted(args.src_dir.glob("*.npz"))
    if not npzs:
        print(f"[convert] no .npz files in {args.src_dir}")
        return

    fn = convert_pointcloud if args.mode == "pointcloud" else convert_image
    for npz in tqdm(npzs, desc=f"npz→topk ({args.mode})"):
        fn(npz, args.dst_dir / (npz.stem + ".topk"))
    print(f"[convert] wrote {len(npzs)} .topk files to {args.dst_dir}")


if __name__ == "__main__":
    main()
