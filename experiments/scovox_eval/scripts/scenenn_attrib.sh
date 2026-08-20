#!/bin/bash
# Attribute SCovox frame time, with the controls the first attempt lacked.
#
# Two questions, one harness, ALL runs on whatever build is installed right now
# and inside one session -- the earlier version of this experiment subtracted a
# `batch_free_carve:=false` run on one build from a control captured on another,
# which makes the difference uninterpretable.
#
#   1. What does the carve staging container actually cost?
#      `off` vs `ctl` -- but note this is NOT a clean isolation: with staging
#      disabled no free-space voxels are ever allocated, so `off` also runs
#      against a ~4x smaller grid. Read the delta as an upper bound.
#   2. Does the per-frame INFO log distort the timings?
#      `ctl` (log to file) vs `devnull` (log formatted+published, not written)
#      vs `quiet` (--log-level warn: no format, no /rosout, no write).
#      frame_ms cannot answer this -- scovox_node.cpp:1233 takes t_end BEFORE
#      the getVmRSSKB() call and the RCLCPP_INFO, so logging is outside the
#      measured window but inside the callback. The logging-independent metric
#      is total process CPU time (utime+stime from /proc/<pid>/stat), which is
#      also the only metric available at all in the `quiet` arm.
#
# Usage: scenenn_attrib.sh [n_frames] [scene_id]
set -o pipefail
N="${1:-400}"; SCENE_ID="${2:-016}"
WS=/home/jetsondevkit/jetbot-slam/scovox
EVAL_PKG="${WS}/experiments/scovox_eval"; DATA=/home/jetsondevkit/datasets/scenenn
SCENE="${DATA}/${SCENE_ID}"
OUTDIR="${WS}/experiments/results/scenenn/${SCENE_ID}/attrib2"
ROBOT=scenenn; RATE=25.0
[ "${N}" -lt 1000 ] || { echo "ABORT: N must stay under the 1000-message queue"; exit 1; }
mkdir -p "${OUTDIR}"
source /opt/ros/humble/setup.bash; source "${WS}/install/setup.bash"
export PYTHONPATH="${EVAL_PKG}:${PYTHONPATH:-}"
HZ=$(getconf CLK_TCK)

run_one() {
  local name="$1"; local sink="$2"; shift 2
  local log="${OUTDIR}/${name}.log"
  echo "=== ${name}: $* ${sink:+(stdout -> ${sink})}"
  local target="${log}"; [ -n "${sink}" ] && target="${sink}"

  # A survivor from a previous arm is fatal, not untidy: `pgrep | head -1` would
  # bind to the STALE pid, so cpu_s/VmHWM would describe the dead run and its
  # already-flat CPU would trip the drain immediately -- killing this arm
  # mid-integration. Clear first, then bind to the NEWEST match (-n).
  # The bracket in the pattern keeps it from matching this script's own cmdline.
  pkill -9 -f 'scovox_mapping[_]node' 2>/dev/null; sleep 1

  ros2 launch scovox_mapping scenenn_eval.launch.py robot_name:=${ROBOT} \
      semantic_mode:=dirichlet stride:=1 enable_tsdf:=false \
      nyu_colormap:="${DATA}/nyu_color.xml" "$@" > "${target}" 2>&1 &
  local lpid=$!
  sleep 6
  local npid; npid=$(pgrep -n -f 'scovox_mapping[_]node')
  [ -n "${npid}" ] || { echo "  ABORT: node not found"; kill ${lpid}; return 1; }

  python3 -m scovox_eval.scenenn_replay_node --ros-args -p scene_dir:="${SCENE}" \
      -p intrinsics:="${DATA}/asus.ini" -p nyu_colormap:="${DATA}/nyu_color.xml" \
      -p labels_dir:="${SCENE}/labels" -p rate_hz:=${RATE} -p robot_name:=${ROBOT} \
      -p n_scans:=${N} > /dev/null 2>&1

  # Drain on CPU-time plateau, not on log lines -- the quiet arm has no log
  # lines, and using a different completion test per arm would confound them.
  #
  # "Plateau" must be a RATE threshold, not exact equality: an idle ROS executor
  # still burns a few jiffies per interval, so `cpu == prev` never becomes true,
  # every arm sits out the full timeout, and that idle spin inflates the very
  # cpu_s being measured. Trip at < 5 jiffies / 2 s (~2.5% of a core) and record
  # cpu_s AT the trip, so at most one poll interval of idle is included.
  local prev=-1 stable=0 waited=0 cpu cpu_s=0
  while [ ${waited} -lt 900 ]; do
    cpu=$(awk '{print $14+$15}' /proc/${npid}/stat 2>/dev/null || echo -1)
    [ "${cpu}" = "-1" ] && break
    # Guard against tripping during start-up or a lull: never before 90 s, so
    # every arm gets through the whole backlog rather than stopping at whatever
    # frame the queue happened to be quiet at (the arms must integrate
    # comparable frame counts or cpu_s is not comparable).
    if [ ${waited} -ge 90 ] && [ ${prev} -ge 0 ] && [ $((cpu - prev)) -lt 5 ]; then stable=$((stable+1)); else stable=0; fi
    cpu_s=$(awk -v h="${HZ}" -v c="${cpu}" 'BEGIN{printf "%.2f", c/h}')
    [ ${stable} -ge 3 ] && break
    prev=${cpu}; sleep 2; waited=$((waited+2))
  done
  echo "  (drained after ${waited}s)"
  local rss_pk; rss_pk=$(awk '/VmHWM/{printf "%.1f", $2/1024}' /proc/${npid}/status 2>/dev/null || echo 0)

  kill ${lpid} 2>/dev/null; wait ${lpid} 2>/dev/null
  pkill -9 -f 'scovox_mapping[_]node' 2>/dev/null
  pkill -9 -f 'scenenn_replay[_]node' 2>/dev/null
  sleep 2
  echo "  cpu_s=${cpu_s} peak_rss=${rss_pk} MB $(
    [ -s "${log}" ] && python3 - "${log}" <<'PY'
import re,sys,statistics
t=open(sys.argv[1],errors="ignore").read()
fr=[float(x) for x in re.findall(r'frame_ms=([\d.]+)',t)]
rc=[int(x) for x in re.findall(r'recv=(\d+)',t)]
if fr:
    tail=fr[-200:] if len(fr)>200 else fr
    print(f"frames={len(fr)} recv={rc[-1] if rc else 0} tail200={statistics.mean(tail):.1f}ms")
PY
  )"
  echo "${name} cpu_s=${cpu_s} rss=${rss_pk}" >> "${OUTDIR}/summary.txt"
}

: > "${OUTDIR}/summary.txt"
# --- Q1: staging cost, paired, same build, same session -------------------
run_one ctl  ""                                        # control
run_one off  ""          batch_free_carve:=false       # staging disabled
# --- Q2: does the per-frame INFO log distort the numbers? -----------------
run_one devnull /dev/null                              # format+rosout, no file write
run_one quiet   ""       log_level:=warn               # no INFO at all
echo; echo "results -> ${OUTDIR}"; cat "${OUTDIR}/summary.txt"
