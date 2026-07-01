"""Outdoor semantic palette + Mapillary-Vistas(65) -> compact(14) collapse.

The SCovox RGB-D path decodes each segmentation pixel's packed 0xRRGGBB through
`semantic_color_map_keys -> semantic_color_map_classes`, where class 0 means
"unknown" (NOT integrated semantically) and classes 1..num_classes-1 are real.

We use a compact 13-class outdoor set (ids 1..13) + 0 = other/unknown, so
SCovox should run with `num_classes := 14` and `max_semantic_classes := 14`.

This module:
  * defines the compact class names + colors (id -> RGB),
  * builds a per-model-class collapse LUT (model id -> compact id) by keyword
    matching the model's `id2label` names (priority-ordered, first match wins),
  * exposes helpers to emit the SCovox `semantic_color_map_*` param lists so the
    seg node's palette and SCovox stay in lock-step.

Keep this file the SINGLE source of truth for the palette: the seg node colorizes
with COMPACT_COLORS and SCovox decodes with the keys/classes emitted here.
"""
from __future__ import annotations

# Compact outdoor classes. Index = SCovox class id. 0 is reserved for
# other/unknown (black, not integrated by SCovox).
COMPACT_NAMES = [
    "other",        # 0  (unknown / not integrated)
    "road",         # 1
    "sidewalk",     # 2
    "terrain",      # 3
    "building",     # 4
    "vegetation",   # 5
    "pole",         # 6
    "sign_light",   # 7
    "person",       # 8
    "rider",        # 9
    "vehicle",      # 10
    "sky",          # 11  (won't materialize as voxels — no valid depth — harmless)
    "fence",        # 12
    "wall",         # 13
]

# id -> (R, G, B). Cityscapes-style colors for familiarity in RViz.
COMPACT_COLORS = [
    (0, 0, 0),         # 0 other/unknown
    (128, 64, 128),    # 1 road
    (244, 35, 232),    # 2 sidewalk
    (152, 251, 152),   # 3 terrain
    (70, 70, 70),      # 4 building
    (107, 142, 35),    # 5 vegetation
    (153, 153, 153),   # 6 pole
    (250, 170, 30),    # 7 sign_light
    (220, 20, 60),     # 8 person
    (255, 0, 0),       # 9 rider
    (0, 0, 142),       # 10 vehicle
    (70, 130, 180),    # 11 sky
    (190, 153, 153),   # 12 fence
    (102, 102, 156),   # 13 wall
]

NUM_CLASSES = len(COMPACT_NAMES)  # 14 (0..13)

# Priority-ordered keyword groups: (compact_id, [substrings]). The FIRST group
# whose any-substring is found in the lowercased model class name wins, so list
# the more specific groups first (e.g. sidewalk/curb before wall/barrier;
# rider/bicyclist before vehicle/bicycle). Names that match nothing -> 0 (other).
# Substrings are matched against the model's own class names, and this table
# covers BOTH label sets we test with:
#   * Mapillary Vistas v1.2 ("construction--flat--road", "object--vehicle--car")
#   * ADE20K semantic ("tree", "grass", "earth", "path", "streetlight", ...)
# Ordering note: the POLE group precedes the VEGETATION group on purpose, because
# ADE's "streetlight" contains the substring "tree" — pole must claim it first so
# it isn't mis-collapsed to vegetation.
_KEYWORD_GROUPS = [
    (8,  ["person"]),                                   # person (before rider/bicyclist)
    (9,  ["rider", "bicyclist", "motorcyclist"]),       # rider (before vehicle)
    (7,  ["traffic-sign", "traffic sign", "traffic-light",
          "traffic light", "sign-frame", "signboard"]), # sign / light (ADE: signboard)
    (2,  ["sidewalk", "curb", "pedestrian"]),           # sidewalk (curb before wall/barrier)
    (1,  ["road", "lane", "crosswalk", "bike-lane", "bike lane",
          "parking", "service-lane", "service lane",
          "rail-track", "rail track",
          "path", "runway"]),                           # road/path surface + markings (ADE: path, runway)
    (4,  ["building", "bridge", "tunnel",
          "house", "skyscraper"]),                      # building / structure (ADE: house, skyscraper)
    (6,  ["pole", "street-light", "street light", "streetlight",
          "banner", "billboard"]),                      # vertical poles / signage posts (before vegetation)
    (5,  ["vegetation", "tree", "plant", "palm", "flower"]),  # vegetation / tree (ADE: tree, plant, palm, flower)
    (12, ["fence", "guard-rail", "guard rail", "railing"]),   # fence (ADE: railing)
    (13, ["wall", "barrier"]),                          # wall / generic barrier
    (10, ["car", "truck", "bus", "caravan", "trailer", "van",
          "on-rails", "on rails", "motorcycle", "bicycle",
          "boat", "vehicle", "wheeled-slow", "minibike"]),   # vehicles (ADE: minibike)
    (11, ["sky"]),                                      # sky (skyscraper already claimed by building)
    (3,  ["terrain", "sand", "snow", "mountain",
          "grass", "earth", "field", "hill", "land",
          "dirt", "rock"]),                             # natural ground (NOT bare "ground"
                                                        # -> would catch "Ground Animal";
                                                        # ADE: grass, earth, field, hill, land, dirt track, rock)
]


def collapse_for_name(name: str) -> int:
    """Map a single model class name to a compact id (0 = other)."""
    n = name.lower().replace("_", "-")
    for compact_id, keys in _KEYWORD_GROUPS:
        for k in keys:
            if k in n:
                return compact_id
    return 0


def build_collapse_lut(id2label: dict) -> list[int]:
    """Build model_class_id -> compact_id LUT from a model `config.id2label`.

    `id2label` is {int_or_str_id: name}. Returns a list indexed by model id.
    """
    n = max(int(k) for k in id2label.keys()) + 1
    lut = [0] * n
    for k, name in id2label.items():
        lut[int(k)] = collapse_for_name(str(name))
    return lut


def scovox_color_map():
    """Return (keys, classes) for SCovox `semantic_color_map_keys/classes`.

    keys: packed 0xRRGGBB ints; classes: matching compact ids. Class 0 (black) is
    included last to mirror the default-param convention.
    """
    keys, classes = [], []
    for cid in range(1, NUM_CLASSES):  # skip 0 here; append it last
        r, g, b = COMPACT_COLORS[cid]
        keys.append((r << 16) | (g << 8) | b)
        classes.append(cid)
    keys.append(0)       # black -> unknown
    classes.append(0)
    return keys, classes


if __name__ == "__main__":
    # Print the SCovox params to paste into the RGB-D YAML.
    keys, classes = scovox_color_map()
    print("num_classes:", NUM_CLASSES)
    print("max_semantic_classes:", NUM_CLASSES)
    print("semantic_color_map_keys: [" + ", ".join(hex(k) for k in keys) + "]")
    print("semantic_color_map_classes:", classes)
    print("\nclass id -> name -> color:")
    for i, (nm, col) in enumerate(zip(COMPACT_NAMES, COMPACT_COLORS)):
        print(f"  {i:2d}  {nm:11s} {col}")
