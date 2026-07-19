#!/usr/bin/env python3
"""Build the ADE150 -> SceneNet-14 (NYUv2) class-collapse lookup table.

SceneNet's 14 evaluation classes (see scenenet_replay_node.py):
  0 Unknown  1 Bed  2 Books  3 Ceiling  4 Chair  5 Floor  6 Furniture
  7 Objects  8 Picture  9 Sofa  10 Table  11 TV  12 Wall  13 Window

Mask2Former is trained on ADE20k (150 classes). We map each ADE id to the
closest SceneNet class. Rules (first match wins, checked in list order):
  - explicit per-id overrides for the semantically load-bearing classes
  - keyword groups for the rest
  - outdoor / nature / vehicle classes -> 0 (Unknown, absent indoors)
  - any remaining indoor thing -> 7 (Objects, the catch-all)

Emits ade150_to_scenenet14.json: {"<ade_id>": scenenet_class, ...} for 0..149.
The mapping is a defensible default, not ground truth — edit the groups below
to taste and re-run.
"""
import argparse, json
from pathlib import Path

# SceneNet class ids.
UNKNOWN, BED, BOOKS, CEILING, CHAIR, FLOOR, FURNITURE, OBJECTS, \
    PICTURE, SOFA, TABLE, TV, WALL, WINDOW = range(14)

# Explicit ADE-id -> SceneNet-class overrides (ADE20k canonical 150 ordering).
OVERRIDES = {
    0: WALL, 3: FLOOR, 5: CEILING, 7: BED, 8: WINDOW, 10: FURNITURE,
    14: WALL,            # door -> wall/structure (SceneNet has no door class)
    15: TABLE, 18: WINDOW,  # curtain -> window-associated
    19: CHAIR, 22: PICTURE, 23: SOFA, 24: FURNITURE, 27: OBJECTS,
    28: FLOOR,           # rug -> floor
    30: SOFA,            # armchair
    31: CHAIR, 33: TABLE, 35: FURNITURE, 36: OBJECTS, 39: OBJECTS,
    41: OBJECTS, 42: WALL, 43: PICTURE, 44: FURNITURE, 45: FURNITURE,
    47: OBJECTS, 49: FURNITURE, 50: FURNITURE, 55: FURNITURE, 56: TABLE,
    57: OBJECTS, 62: FURNITURE, 63: WINDOW, 64: TABLE, 67: BOOKS,
    69: CHAIR, 70: FURNITURE, 71: FURNITURE, 73: FURNITURE, 74: TV,
    75: CHAIR, 77: TABLE, 89: TV, 97: SOFA, 99: FURNITURE, 100: PICTURE,
    107: FURNITURE, 110: CHAIR, 117: BED, 118: FURNITURE, 124: OBJECTS,
    129: FURNITURE, 130: TV, 141: TV, 143: TV, 144: PICTURE,
    58: WINDOW,          # screen door
}

# ADE ids that are outdoor / nature / vehicles -> Unknown (never indoors).
OUTDOOR = {
    1, 2, 4, 6, 9, 11, 13, 16, 20, 21, 25, 26, 29, 32, 34, 46, 48, 51, 52,
    54, 60, 61, 68, 72, 76, 80, 83, 84, 86, 90, 91, 94, 102, 103, 104, 105,
    109, 113, 114, 116, 121, 122, 126, 127, 128, 140, 145,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ade", default=str(Path(__file__).resolve().parents[1] /
                    "scovox_eval" / "scripts" / "ade150_labels.json"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent /
                    "ade150_to_scenenet14.json"))
    args = ap.parse_args()

    ade = json.loads(Path(args.ade).read_text())
    table = {}
    for i in range(150):
        if i in OVERRIDES:
            table[str(i)] = OVERRIDES[i]
        elif i in OUTDOOR:
            table[str(i)] = UNKNOWN
        else:
            table[str(i)] = OBJECTS  # remaining indoor thing -> catch-all
    Path(args.out).write_text(json.dumps(table, indent=2))

    names = ["Unknown", "Bed", "Books", "Ceiling", "Chair", "Floor",
             "Furniture", "Objects", "Picture", "Sofa", "Table", "TV",
             "Wall", "Window"]
    from collections import Counter
    hist = Counter(table.values())
    print(f"wrote {args.out}")
    for c in range(14):
        ex = [ade[str(i)].strip() for i in range(150) if table[str(i)] == c][:6]
        print(f"  {c:2d} {names[c]:10s} <- {hist[c]:3d} ADE ids  e.g. {ex}")


if __name__ == "__main__":
    main()
