#!/usr/bin/env python3
"""Generate SOFT per-pixel semantic .topk blobs for SceneNet trajectories by
running Mask2Former (ADE20k) on the RGB photo frames and collapsing the
150-class per-pixel posterior to SceneNet's 14 classes.

Output: <out_dir>/<seq>/predictions_topk/NNNNNN.topk  (sequential frame index,
matching scenenet_replay_node's depth-stamp low-16-bit convention), in the
image .topk layout scovox_node reads:
    [u16 H][u16 W][u8 C][H*W*C u8 probs(x255)]   slot j == SCovox class id.

This is the SceneNet analogue of the paper's Mask2Former-on-Replica soft path;
the paper itself used GT one-hot for SceneNet, so there is no published baseline.
"""
import argparse, json, struct
from pathlib import Path

import numpy as np
import torch
from PIL import Image

C_SCENENET = 14  # slots 0..13; slot 0 = Unknown/escape mass


def load_collapse(path):
    d = json.loads(Path(path).read_text())
    lut = np.zeros(150, dtype=np.int64)
    for k, v in d.items():
        lut[int(k)] = int(v)
    return lut


def per_pixel_probs(model, processor, rgb, device, H, W):
    """Return (C=150, H, W) per-pixel probability distribution."""
    inputs = processor(images=rgb, return_tensors="pt").to(device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16,
                                         enabled=(device == "cuda")):
        out = model(**inputs)
    # Mask2Former semantic inference: sum over queries of class-softmax x mask-sigmoid.
    cls = out.class_queries_logits.float().softmax(-1)[..., :-1]   # (1,Q,150) drop no-object
    msk = out.masks_queries_logits.float().sigmoid()               # (1,Q,h,w)
    seg = torch.einsum("bqc,bqhw->bchw", cls, msk)                 # (1,150,h,w)
    seg = torch.nn.functional.interpolate(seg, size=(H, W), mode="bilinear",
                                          align_corners=False)[0]  # (150,H,W)
    seg = seg / seg.sum(0, keepdim=True).clamp_min(1e-6)           # per-pixel L1 norm
    return seg.cpu().numpy()


def collapse_to_14(probs150, lut):
    """(150,H,W) -> (14,H,W) by summing ADE groups into SceneNet classes."""
    H, W = probs150.shape[1:]
    out = np.zeros((C_SCENENET, H, W), dtype=np.float32)
    for ade in range(150):
        out[lut[ade]] += probs150[ade]
    return out


def write_topk(path, probs14_hw):
    """probs14_hw: (14,H,W) float in [0,1]. Writes image .topk (H,W,C order)."""
    C, H, W = probs14_hw.shape
    q = np.clip(np.round(probs14_hw * 255.0), 0, 255).astype(np.uint8)
    q = np.transpose(q, (1, 2, 0))  # (H,W,C) row-major -> [v][u][c]
    with open(path, "wb") as f:
        f.write(struct.pack("<HHB", H, W, C))
        f.write(q.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="SceneNet SLIM-VDB root (has train/<seq>)")
    ap.add_argument("--seqs", required=True, help="comma-separated, e.g. 0_223,0_175")
    ap.add_argument("--collapse", default=str(Path(__file__).resolve().parent /
                    "ade150_to_scenenet14.json"))
    ap.add_argument("--model", default="facebook/mask2former-swin-large-ade-semantic")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                    help="'auto' uses cuda if usable; the P2000 (sm_61) is not "
                         "supported by torch>=2.5, so force 'cpu' here.")
    ap.add_argument("--depth_h", type=int, default=240)
    ap.add_argument("--depth_w", type=int, default=320)
    args = ap.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
    print(f"loading {args.model} on {device} ...", flush=True)
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(args.model).to(device).eval()
    lut = load_collapse(args.collapse)

    root = Path(args.data_root) / "train"
    for seq in args.seqs.split(","):
        seq = seq.strip()
        photo_dir = root / seq / "photo"
        out_dir = root / seq / "predictions_topk"
        out_dir.mkdir(exist_ok=True)
        photos = sorted(photo_dir.glob("*.jpg"))
        print(f"[{seq}] {len(photos)} frames -> {out_dir}", flush=True)
        for idx, pf in enumerate(photos):
            outp = out_dir / f"{idx:06d}.topk"
            if outp.exists() and outp.stat().st_size > 0:
                continue
            rgb = Image.open(pf).convert("RGB")
            p150 = per_pixel_probs(model, processor, rgb, device, args.depth_h, args.depth_w)
            p14 = collapse_to_14(p150, lut)
            write_topk(outp, p14)
            if (idx + 1) % 50 == 0:
                print(f"  [{seq}] {idx+1}/{len(photos)}", flush=True)
        print(f"[{seq}] done", flush=True)


if __name__ == "__main__":
    main()
