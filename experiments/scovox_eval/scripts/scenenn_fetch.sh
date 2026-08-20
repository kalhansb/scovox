#!/bin/bash
# Fetch one SceneNN scene and extract its RGB-D frames.
# Usage: scenenn_fetch.sh <scene_id> [dest]      e.g. scenenn_fetch.sh 016
#
# The SceneNN README points at a Google Drive folder (0B-aa7y5Ox4eZWE8yMkRkNkU4Tk0)
# that now 404s, and the download script it ships needs PyDrive + an interactive
# OAuth flow. Everything is still served over plain HTTP from HKUST, including
# the .oni raw video the Drive copy was needed for, so this pulls from there.
#
# Per-scene cost: the .oni dominates (016 is 229 MB; some scenes exceed 1.7 GB --
# check with `curl -sIL $BASE/main/oni/<id>.oni | grep -i content-length` first).
# Extraction roughly quadruples that in PNGs.
set -euo pipefail
SCENE="${1:?usage: scenenn_fetch.sh <scene_id> [dest]}"
DEST="${2:-/home/jetsondevkit/datasets/scenenn}"
BASE=https://hkust-vgd.ust.hk/scenenn
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${DEST}/${SCENE}"
cd "${DEST}"

echo "[scenenn_fetch] scene ${SCENE} -> ${DEST}"
SZ=$(curl -sIL "${BASE}/main/oni/${SCENE}.oni" | grep -i '^content-length' | tail -1 | tr -d '\r' | awk '{print $2}')
echo "[scenenn_fetch] oni size: $(( ${SZ:-0} / 1048576 )) MB"
AVAIL=$(df --output=avail -m . | tail -1)
echo "[scenenn_fetch] free space: $(( AVAIL / 1024 )) GB"
if [ $(( ${SZ:-0} / 1048576 * 5 )) -gt "${AVAIL}" ]; then
  echo "[scenenn_fetch] ABORT: need ~5x the oni size for oni + extracted PNGs"; exit 1
fi

# Shared across scenes.
[ -f asus.ini ]      || curl -# -o asus.ini      "${BASE}/main/intrinsic/asus.ini"
[ -f nyu_color.xml ] || curl -# -o nyu_color.xml "${BASE}/contrib/nyu_class/nyu_color.xml"

cd "${SCENE}"
# trajectory.log: Redwood .log, 4x4 camera-to-world, one per extracted frame.
[ -f trajectory.log ]        || curl -# -o trajectory.log        "${BASE}/main/${SCENE}/trajectory.log"
# The nyu_class pair is the annotation actually used for labelling: its PLY
# carries a per-vertex nyu_class, and its instance ids differ from the main
# annotation's (walls are merged), so the two must not be mixed.
[ -f "${SCENE}_nyu.ply" ]    || curl -# -o "${SCENE}_nyu.ply"    "${BASE}/contrib/nyu_class/${SCENE}/${SCENE}.ply"
[ -f "${SCENE}_nyu.xml" ]    || curl -# -o "${SCENE}_nyu.xml"    "${BASE}/contrib/nyu_class/${SCENE}/${SCENE}.xml"
[ -f "${SCENE}.oni" ]        || curl -# -o "${SCENE}.oni"        "${BASE}/main/oni/${SCENE}.oni"

# Build SceneNN's OpenNI 1.5 extractor once. Needs libopenni-dev, libfreeimage-dev,
# libboost-{system,filesystem}-dev -- all present in JetPack's Ubuntu 22.04.
PB="${DEST}/playback"
if [ ! -x "${PB}/playback" ]; then
  echo "[scenenn_fetch] building playback tool"
  mkdir -p "${PB}" && cd "${PB}"
  curl -sL -O https://raw.githubusercontent.com/hkust-vgd/scenenn/master/playback/Makefile
  curl -sL -O https://raw.githubusercontent.com/hkust-vgd/scenenn/master/playback/PlaybackSync.cpp
  make
  cd "${DEST}/${SCENE}"
fi

if [ ! -d frames/depth ]; then
  echo "[scenenn_fetch] extracting frames (this takes a few minutes)"
  "${PB}/playback" "${SCENE}.oni" frames | tail -3
fi
echo "[scenenn_fetch] frames: $(ls frames/depth | wc -l), poses: $(awk 'NF==3' trajectory.log | wc -l)"

cat <<EOF

Next:
  python3 ${HERE}/scenenn_register_mesh.py --scene ${DEST}/${SCENE} \\
      --intrinsics ${DEST}/asus.ini --out ${DEST}/${SCENE}/mesh_align.json --per-frame
  python3 ${HERE}/scenenn_make_labels.py --scene ${DEST}/${SCENE} \\
      --intrinsics ${DEST}/asus.ini --align ${DEST}/${SCENE}/mesh_align.json
  ${HERE}/../../run_scenenn.sh ${SCENE}
EOF
