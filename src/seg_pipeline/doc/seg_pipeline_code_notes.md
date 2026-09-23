# outdoor_palette.py — design notes and history

The long comments of `seg_pipeline/outdoor_palette.py`, moved out of the code on 2026-09-23 so the source carries short comments only. Where a comment was moved, the code keeps a short gist ending in `(notes: <id>)`; the section headed `<id>` below holds the original comment, word for word.

Sections follow the order of the source file and are grouped by the function (or section) they sit in. Each gives the line of code the comment was attached to and its original line number. Line numbers, dates, generation numbers and cross-references inside the moved text are as they were when written; they record history and are not maintained.

## Contents

- [Module scope](#module-scope) — 1

## Module scope

### palette-keyword-group-priority

**Keyword group priority for class collapse** — attached to `_KEYWORD_GROUPS = [` (line 61)

```text
Priority-ordered keyword groups: (compact_id, [substrings]). The FIRST group
whose any-substring is found in the lowercased model class name wins, so list
the more specific groups first (e.g. sidewalk/curb before wall/barrier;
rider/bicyclist before vehicle/bicycle). Names that match nothing -> 0 (other).
Substrings are matched against the model's own class names, and this table
covers BOTH label sets we test with:
  * Mapillary Vistas v1.2 ("construction--flat--road", "object--vehicle--car")
  * ADE20K semantic ("tree", "grass", "earth", "path", "streetlight", ...)
Ordering note: the POLE group precedes the VEGETATION group on purpose, because
ADE's "streetlight" contains the substring "tree" — pole must claim it first so
it isn't mis-collapsed to vegetation.
```
