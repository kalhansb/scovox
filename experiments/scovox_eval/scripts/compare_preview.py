#!/usr/bin/env python3
"""Build side-by-side comparison PNGs for Replica render preview.

Layout per frame:
  [existing color | rendered color | |existing-rendered| diff | semantic overlay + legend]
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def colorize(labels: np.ndarray, max_id: int = 101) -> np.ndarray:
    rng = np.random.default_rng(42)
    palette = rng.integers(40, 230, size=(max_id + 2, 3), dtype=np.uint8)
    palette[0] = (0, 0, 0)
    return palette[labels.clip(0, max_id + 1).astype(np.int32)]


def annotate(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def legend(labels: np.ndarray, class_name, palette, h, w=320):
    ids, cnt = np.unique(labels, return_counts=True)
    order = np.argsort(-cnt)
    ids, cnt = ids[order], cnt[order]
    panel = np.full((h, w, 3), 30, dtype=np.uint8)
    y = 20
    for lid, c in zip(ids, cnt):
        if y > h - 24:
            break
        col = palette[int(lid)][::-1].tolist()
        cv2.rectangle(panel, (12, y - 12), (30, y + 4), col, -1)
        name = class_name.get(int(lid), f"id{int(lid)}")
        pct = 100.0 * c / labels.size
        cv2.putText(panel, f"{int(lid):3d} {name} ({pct:.1f}%)",
                    (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (230, 230, 230), 1, cv2.LINE_AA)
        y += 22
    return panel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene-dir", type=Path, required=True)
    p.add_argument("--frames", nargs="+", type=int, required=True)
    p.add_argument("--preview-subdir", default="semantic_gt_preview")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    info = json.loads((args.scene_dir / "info_semantic.json").read_text())
    class_name = {c["id"]: c["name"] for c in info["classes"]}
    class_name[0] = "unlabeled"

    palette = np.vstack([np.random.default_rng(42)
                         .integers(40, 230, size=(102, 3), dtype=np.uint8)])
    palette[0] = (0, 0, 0)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for f in args.frames:
        stem = f"{f:06d}"
        old = cv2.imread(str(args.scene_dir / "color" / f"{stem}.jpg"))
        new = cv2.imread(str(args.scene_dir / f"{args.preview_subdir}_rgb" / f"{stem}.jpg"))
        sem = cv2.imread(str(args.scene_dir / args.preview_subdir / f"{stem}.png"),
                         cv2.IMREAD_UNCHANGED).astype(np.int32)
        if old is None or new is None or sem is None:
            print(f"skip {stem}")
            continue

        diff = cv2.absdiff(old, new)

        sem_rgb = palette[sem.clip(0, 101)][..., ::-1]
        overlay = cv2.addWeighted(new, 0.45, sem_rgb.astype(np.uint8), 0.55, 0)

        H = old.shape[0]
        panels = [
            annotate(old, f"existing color  {stem}"),
            annotate(new, "rendered color (habitat)"),
            annotate(diff, f"|diff|  mean={diff.mean():.1f}  max={diff.max()}"),
            annotate(overlay, f"rendered semantic overlay  uniq={len(np.unique(sem))}"),
            legend(sem, class_name, palette, H),
        ]
        combined = np.hstack(panels)
        out_path = args.out_dir / f"compare_{stem}.png"
        cv2.imwrite(str(out_path), combined)
        print(f"wrote {out_path}  (diff mean={diff.mean():.1f})")


if __name__ == "__main__":
    main()
