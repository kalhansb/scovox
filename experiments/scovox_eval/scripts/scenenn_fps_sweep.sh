#!/bin/bash
# Hunt for >=6 Hz at 5 cm on SceneNN WITHOUT degrading the map.
#
# Ground rules (from the perf target, not arbitrary):
#   * resolution stays 0.05 -- it is the deliverable, not a knob.
#   * `stride` and `carve_band` are LAST RESORT: both trade map quality
#     (stride drops thin structure, carve_band drops free space). They are not
#     in this sweep. `max_range` is the same kind of trade and is also out.
#   * Everything here is either a different code path or pure container
#     geometry. Bonxai coordinates are a function of `resolution` alone
#     (VoxelGrid::posToCoord), so leaf_bits/inner_bits/dir_leaf_bits change how
#     voxels are packed, never a voxel's identity or value.
#
# Reports Hz from the node's own frame_ms, plus voxel count so a config that
# quietly changes the map is visible rather than being credited as a win.
#
# Usage: scenenn_fps_sweep.sh [n_frames] [scene_id]
set -o pipefail
N="${1:-400}"; SCENE_ID="${2:-016}"
WS=/home/jetsondevkit/jetbot-slam/scovox
EVAL_PKG="${WS}/experiments/scovox_eval"; DATA=/home/jetsondevkit/datasets/scenenn
SCENE="${DATA}/${SCENE_ID}"
OUTDIR="${WS}/experiments/results/scenenn/${SCENE_ID}/${FPS_TAG:-fps}"
ROBOT=scenenn; RATE=25.0
[ "${N}" -lt 1000 ] || { echo "ABORT: N must stay under the 1000-message queue"; exit 1; }
mkdir -p "${OUTDIR}"
source /opt/ros/humble/setup.bash; source "${WS}/install/setup.bash"
export PYTHONPATH="${EVAL_PKG}:${PYTHONPATH:-}"

run_one() {
  local name="$1"; shift
  local log="${OUTDIR}/${name}.log" npz="${OUTDIR}/${name}.npz"
  [ -s "${log}" ] && { echo "=== ${name}: cached, skipping"; return 0; }
  echo "=== ${name}: $*"
  pkill -9 -f 'scovox_mapping[_]node' 2>/dev/null; sleep 1

  ros2 launch scovox_mapping scenenn_eval.launch.py robot_name:=${ROBOT} \
      semantic_mode:=dirichlet resolution:=0.05 stride:=1 \
      nyu_colormap:="${DATA}/nyu_color.xml" "$@" > "${log}" 2>&1 &
  local lpid=$!; sleep 6

  python3 -m scovox_eval.scenenn_replay_node --ros-args -p scene_dir:="${SCENE}" \
      -p intrinsics:="${DATA}/asus.ini" -p nyu_colormap:="${DATA}/nyu_color.xml" \
      -p labels_dir:="${SCENE}/labels" -p rate_hz:=${RATE} -p robot_name:=${ROBOT} \
      -p n_scans:=${N} > /dev/null 2>&1

  local prev=-1 stable=0 waited=0 last
  while [ ${waited} -lt 900 ]; do
    last=$(grep -oP 'recv=\K[0-9]+' "${log}" 2>/dev/null | tail -1)
    if [ "${last:-0}" -eq "${prev}" ]; then stable=$((stable+1)); else stable=0; fi
    [ ${stable} -ge 4 ] && break
    prev=${last:-0}; sleep 2; waited=$((waited+2))
  done

  timeout 120 python3 -m scovox_eval.pointcloud_to_npz --ros-args \
      -p topic:=/${ROBOT}/scovox_node/pointcloud -p output:="${npz}" >> "${log}" 2>&1 \
      || echo "  WARN: capture failed"
  kill ${lpid} 2>/dev/null; wait ${lpid} 2>/dev/null
  pkill -9 -f 'scovox_mapping[_]node' 2>/dev/null
  pkill -9 -f 'scenenn_replay[_]node' 2>/dev/null
  sleep 2
  python3 - "${log}" "${npz}" "${name}" <<'PY'
import re,sys,statistics,os
log,npz,name=sys.argv[1],sys.argv[2],sys.argv[3]
t=open(log,errors="ignore").read()
fr=[float(x) for x in re.findall(r'frame_ms=([\d.]+)',t)]
if not fr: print("  NO FRAMES"); raise SystemExit
tail=fr[-200:] if len(fr)>200 else fr
m=statistics.mean(tail); v="?"
if os.path.exists(npz):
    import numpy as np; v=str(np.load(npz)['points'].shape[0])
print(f"  frames={len(fr)} mean={m:.1f}ms  HZ={1000/m:.2f}  voxels={v}")
PY
}

if [ -n "${ONLY:-}" ]; then
  for c in ${ONLY}; do
    case "${c}" in
      base)   run_one base   enable_tsdf:=false ;;
      nofuse) run_one nofuse enable_tsdf:=false fused_walker:=false ;;
      tsdf_on) run_one tsdf_on enable_tsdf:=true ;;
      range35) run_one range35 enable_tsdf:=false min_range:=0.5 max_range:=3.5 ;;
      rangectl) run_one rangectl enable_tsdf:=false min_range:=1.5 max_range:=2.5 ;;
      tsdf_r35)         run_one tsdf_r35         enable_tsdf:=true min_range:=0.5 max_range:=3.5 ;;
      tsdf_nocarve_r35) run_one tsdf_nocarve_r35 enable_tsdf:=true min_range:=0.5 max_range:=3.5 batch_free_carve:=false ;;
      tsdf_nocarve) run_one tsdf_nocarve enable_tsdf:=true batch_free_carve:=false ;;
      nocarve)      run_one nocarve      enable_tsdf:=false batch_free_carve:=false ;;
      # TSDF off at the SAME range as the trunc arms. Needed because disabling
      # TSDF does NOT shrink the skip window: sanitise rewrites sdf_trunc 0 ->
      # 0.15, so the window stays at 5 voxels while doing no TSDF work at all.
      # Without matching the range this comparison is confounded.
      nocarve_r35) run_one nocarve_r35 enable_tsdf:=false batch_free_carve:=false \
                             min_range:=0.5 max_range:=3.5 ;;
      # Far-skip A/B: det1/det2 = identical arms (determinism control — with the
      # replay's subscriber-wait they must produce bit-identical npz); noskip =
      # same config, skip disabled via env on the same binary. Keep noskip LAST:
      # a VAR=x fn-call assignment can outlive the call in bash.
      det1)   run_one det1   enable_tsdf:=true min_range:=0.5 max_range:=3.5 batch_free_carve:=false ;;
      det2)   run_one det2   enable_tsdf:=true min_range:=0.5 max_range:=3.5 batch_free_carve:=false ;;
      noskip) SCOVOX_DISABLE_FAR_SKIP=1 run_one noskip \
                             enable_tsdf:=true min_range:=0.5 max_range:=3.5 batch_free_carve:=false ;;
      # Is sdf_trunc a usable width knob for the far-skip window? trunc sets
      # BOTH the TSDF band written and the skip radius, so unlike the skip
      # itself this is not free: it narrows the reconstructed surface band.
      trunc1) run_one trunc1 enable_tsdf:=true min_range:=0.5 max_range:=3.5 \
                             batch_free_carve:=false sdf_trunc_voxels:=1 ;;
      trunc2) run_one trunc2 enable_tsdf:=true min_range:=0.5 max_range:=3.5 \
                             batch_free_carve:=false sdf_trunc_voxels:=2 ;;
    esac
  done
  echo; echo "results -> ${OUTDIR}"; exit 0
fi
# --- reference ------------------------------------------------------------
run_one base          enable_tsdf:=false
run_one tsdf_on       enable_tsdf:=true
# --- different integration code paths -------------------------------------
run_one nofuse        enable_tsdf:=false fused_walker:=false
run_one split         enable_tsdf:=false use_split:=true
run_one split_nofuse  enable_tsdf:=false use_split:=true fused_walker:=false
# --- container geometry: cannot change the map ----------------------------
run_one leaf4         enable_tsdf:=false leaf_bits:=4
run_one leaf2         enable_tsdf:=false leaf_bits:=2
run_one leaf5         enable_tsdf:=false leaf_bits:=5
run_one inner3        enable_tsdf:=false inner_bits:=3
run_one leaf4_inner3  enable_tsdf:=false leaf_bits:=4 inner_bits:=3
run_one dirleaf3      enable_tsdf:=false dir_leaf_bits:=3
# --- best geometry + best path, stacked -----------------------------------
run_one leaf4_nofuse  enable_tsdf:=false leaf_bits:=4 fused_walker:=false

echo; echo "results -> ${OUTDIR}"
