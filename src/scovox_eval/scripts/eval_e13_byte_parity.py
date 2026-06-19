#!/usr/bin/env python3
"""E1.3 — TsdfMap-vs-SLIM-VDB byte-parity eval (D11).

Reads the SLIM-VDB measurement script's `perf.json` (produced by
`run_slimvdb_replica_m2f.sh` / `run_slimvdb_kitti_all.sh`) for the
`vdb_tsdf_mb_final` field, and compares it against the post-refactor
scovox_node's `[memSplit] tsdf_grid_mb=…` log line (emitted at
shutdown in scheduleMemUsage when use_split=true).

Acceptance gate: |delta| / vdb < 0.15 (paper headline reports the
actually-measured ratio rather than pre-committing to a number — see
D11 in the resume-grilling lock-in).

Usage
-----
  eval_e13_byte_parity.py \
      --slimvdb-perf path/to/slimvdb/perf.json \
      --scovox-log   path/to/scovox/run.log \
      [--threshold 0.15] [--scenario "Replica room0 (5 cm, 100 frames)"]

Exits 0 on PASS (within threshold), 1 on FAIL. Prints one line per
input + a summary line for shell-script consumption.

Reference
---------
- D11 (resume-grilling lock-in): smoke-gate predicate at 15%; paper
  headline reports actuals.
- User memory ("SLIM-VDB memory comparison must use grid-only metric"):
  vdb_tsdf_mb_final is the correct apples-to-apples metric — not the
  system-wide /proc/meminfo readouts that gave the false 115× ratio.
- scovox_eval/results/slimvdb_smoke_step6/perf.json is the existing
  schema reference. vdb_tsdf_mb_final lives under perf['slim_vdb'].
"""

import argparse
import json
import re
import sys
from pathlib import Path


_TSDF_MB_RE = re.compile(
    r"\[memSplit\]\s+tsdf_voxels=(\d+)\s+tsdf_grid_mb=([0-9]*\.?[0-9]+)\s+"
    r"sembeta_voxels=(\d+)\s+sembeta_grid_mb=([0-9]*\.?[0-9]+)"
)


def _last_memsplit(log_path: Path) -> dict:
    """Pull the LAST [memSplit] line out of scovox_node's run.log.

    scheduleMemUsage fires every 10 frames, so the final line reflects
    the steady-state shutdown counts. Earlier lines are noisier
    (bootstrapping pool allocations).
    """
    last = None
    with log_path.open() as f:
        for line in f:
            m = _TSDF_MB_RE.search(line)
            if m is not None:
                last = m
    if last is None:
        sys.exit(
            f"FAIL: no [memSplit] line found in {log_path}. "
            "Ensure use_split=true was set and the run produced at least "
            "one scheduleMemUsage tick (every 10 frames)."
        )
    return {
        "tsdf_voxels":    int(last.group(1)),
        "tsdf_grid_mb":   float(last.group(2)),
        "sembeta_voxels": int(last.group(3)),
        "sembeta_grid_mb": float(last.group(4)),
    }


def _slimvdb_tsdf_mb(perf_json: Path) -> float:
    """Pull vdb_tsdf_mb_final out of the SLIM-VDB perf.json.

    Schema reference: scovox_eval/results/slimvdb_smoke_step6/perf.json
        perf['slim_vdb']['vdb_tsdf_mb_final']
    """
    with perf_json.open() as f:
        perf = json.load(f)
    try:
        return float(perf["slim_vdb"]["vdb_tsdf_mb_final"])
    except (KeyError, TypeError) as e:
        sys.exit(
            f"FAIL: {perf_json} missing perf['slim_vdb']['vdb_tsdf_mb_final'] "
            f"({e}). Update SLIM-VDB run script if schema drifted."
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slimvdb-perf", required=True, type=Path,
                   help="Path to SLIM-VDB perf.json with vdb_tsdf_mb_final.")
    p.add_argument("--scovox-log", required=True, type=Path,
                   help="Path to scovox_node run.log "
                        "(must contain [memSplit] line — set use_split=true).")
    p.add_argument("--threshold", type=float, default=0.15,
                   help="PASS if |delta|/vdb < threshold. Default 0.15 per D11.")
    p.add_argument("--scenario", default="(unspecified)",
                   help="Free-form scenario tag for log output.")
    args = p.parse_args()

    if not args.slimvdb_perf.exists():
        sys.exit(f"FAIL: SLIM-VDB perf JSON not found: {args.slimvdb_perf}")
    if not args.scovox_log.exists():
        sys.exit(f"FAIL: scovox log not found: {args.scovox_log}")

    vdb_mb     = _slimvdb_tsdf_mb(args.slimvdb_perf)
    split_meta = _last_memsplit(args.scovox_log)
    sc_mb      = split_meta["tsdf_grid_mb"]

    if vdb_mb <= 0:
        sys.exit(f"FAIL: SLIM-VDB vdb_tsdf_mb_final={vdb_mb} is non-positive — "
                 "check perf.json for run-time errors.")

    delta_pct  = (sc_mb - vdb_mb) / vdb_mb
    abs_pct    = abs(delta_pct)
    verdict    = "PASS" if abs_pct < args.threshold else "FAIL"

    print(f"scenario:       {args.scenario}")
    print(f"slimvdb_tsdf_mb={vdb_mb:.3f}  (perf.json: slim_vdb.vdb_tsdf_mb_final)")
    print(f"scovox_tsdf_mb ={sc_mb:.3f}  (run.log: [memSplit] tsdf_grid_mb)")
    print(f"scovox tsdf voxels   = {split_meta['tsdf_voxels']}")
    print(f"scovox sembeta voxels= {split_meta['sembeta_voxels']}")
    print(f"scovox_sembeta_mb    = {split_meta['sembeta_grid_mb']:.3f} (paper headline contribution)")
    print(f"delta_pct={delta_pct:+.4%}  threshold=±{args.threshold:.0%}  verdict={verdict}")

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
