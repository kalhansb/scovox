#!/usr/bin/env python3
"""Emit the ADE150 → SCovox-18-class lookup table that the Replica replay
node implicitly applies when collapsing M2F predictions to the 18-class
evaluation space.

The table is the function `ade_id (0..149) → scovox_class (0..18)`, where
0 means "no match → unknown" and 1..18 are the SCovox indoor categories
(wall, floor, ceiling, door, window, chair, table, sofa, bed, cushion,
lamp, cabinet, blinds, book, picture, plant, rug, pillar).

Output JSON shape: {"ade_id": scovox_class, ...} for ADE IDs 0..149.

Usage:
    python build_ade_collapse_map.py --ade ade150_labels.json --out collapse_ade150_to_scovox18.json
"""
import argparse
import json
from pathlib import Path

# Order matters here: SCovox_class index = position in this list (1-based).
SCOVOX_CATEGORIES = [
    "wall", "floor", "ceiling", "door", "window", "chair", "table", "sofa",
    "bed", "cushion", "lamp", "cabinet", "blinds", "book", "picture",
    "plant", "rug", "pillar",
]

# Aliases for SCovox category names that don't substring-match common ADE150
# labels (`blind` vs `blinds`, `painting` vs `picture`, `column` vs `pillar`).
# Each alias is matched as a substring of the lower-cased ADE name, in
# addition to the canonical key. Keys here must match SCOVOX_CATEGORIES.
SCOVOX_ALIASES = {
    "blinds":  ["blind"],          # ADE 63 = "blind"
    "picture": ["painting"],       # ADE 22 = "painting"
    "pillar":  ["column"],         # ADE 42 = "column"
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ade", type=Path, required=True,
                    help="Path to ade150_labels.json ({\"0\":\"wall\", ...})")
    ap.add_argument("--out", type=Path, required=True,
                    help="Path for output collapse map JSON")
    args = ap.parse_args()

    with open(args.ade) as f:
        ade = json.load(f)

    # Match longer SCovox category names first so e.g. "indoor plant"
    # hits "plant" not "door". Same rule as replica_replay_node.py:225.
    name_to_class = {n: (SCOVOX_CATEGORIES.index(n) + 1) for n in SCOVOX_CATEGORIES}
    # Build (search_term, scovox_class) pairs from canonical names + aliases,
    # sorted by length so longer terms win. Keep a stable order for ties.
    search_terms: list[tuple[str, int]] = []
    for n in SCOVOX_CATEGORIES:
        cls = name_to_class[n]
        search_terms.append((n, cls))
        for alias in SCOVOX_ALIASES.get(n, []):
            search_terms.append((alias, cls))
    search_terms.sort(key=lambda x: -len(x[0]))

    table = {}
    for aid_str, name in ade.items():
        aid = int(aid_str)
        lname = name.lower()
        parts = [p.strip() for p in lname.replace(";", ",").split(",")]
        cls = 0  # default = unknown
        for p in parts:
            for term, target in search_terms:
                if term in p:
                    cls = target
                    break
            if cls:
                break
        table[aid] = cls

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({str(k): v for k, v in sorted(table.items())}, f, indent=2)

    matched = sum(1 for v in table.values() if v != 0)
    print(f"[collapse] wrote {args.out}: {matched}/{len(table)} ADE IDs mapped, "
          f"{len(table) - matched} → unknown (0)")


if __name__ == "__main__":
    main()
