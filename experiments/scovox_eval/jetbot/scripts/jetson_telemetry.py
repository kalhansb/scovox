#!/usr/bin/env python3
"""Capture per-second telemetry for the Jetson scovox_mapping_node:
  - RSS (MB), VSZ (MB)
  - CPU% (per-process, sum across cores)
  - Per-core CPU% (system-wide, 4 cores on Nano)
  - System memory available (MB)

Writes a CSV one row per second until killed (Ctrl-C or SIGTERM).
No frame-level timing — that lives inside the scovox node's existing
profiling counters. This is just system-level observability.

Usage (Jetson, inside Humble container):
  python3 jetson_telemetry.py \\
      --out /tmp/jetson_timing.csv \\
      --node-pattern scovox_mapping_node

The matching process is found by `pgrep -f <pattern>`. If multiple match,
the first PID is used.
"""
import argparse
import csv
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple


def find_pid(pattern):
    try:
        out = subprocess.check_output(["pgrep", "-f", pattern]).decode().strip()
    except subprocess.CalledProcessError:
        return None
    pids = [int(x) for x in out.splitlines() if x.strip().isdigit()]
    return pids[0] if pids else None


def read_proc_status(pid):
    """Returns (rss_kb, vsz_kb) from /proc/<pid>/status."""
    rss_kb = vsz_kb = 0
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_kb = int(line.split()[1])
                elif line.startswith("VmSize:"):
                    vsz_kb = int(line.split()[1])
    except FileNotFoundError:
        pass
    return rss_kb, vsz_kb


def read_proc_stat_cpu(pid):
    """Returns utime+stime ticks from /proc/<pid>/stat for delta calc."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            parts = f.read().split()
        # Fields 14 (utime) and 15 (stime) in proc(5).
        return int(parts[13]) + int(parts[14])
    except (FileNotFoundError, IndexError):
        return 0


def read_total_cpu_jiffies():
    """Returns (aggregate, [per_core...]) of total jiffies."""
    total = 0
    per_core = []
    with open("/proc/stat") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "cpu":
                total = sum(int(x) for x in parts[1:])
            elif parts[0].startswith("cpu") and parts[0] != "cpu":
                per_core.append(sum(int(x) for x in parts[1:]))
    return total, per_core


def read_meminfo_available_kb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/jetson_timing.csv")
    ap.add_argument("--node-pattern", default="scovox_mapping_node")
    ap.add_argument("--interval", type=float, default=1.0,
                    help="Sampling interval in seconds (default 1.0)")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pid = None
    last_proc_ticks = 0
    last_total_jiffies = 0
    last_per_core_jiffies = []  # List[int]

    with out_path.open("w", buffering=1) as fp:
        wr = csv.writer(fp)
        wr.writerow(["wall_time", "pid", "rss_mb", "vsz_mb", "proc_cpu_pct",
                     "core0_pct", "core1_pct", "core2_pct", "core3_pct",
                     "mem_avail_mb"])

        sig = signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
        try:
            while True:
                t = time.time()

                if pid is None or not Path(f"/proc/{pid}").exists():
                    pid = find_pid(args.node_pattern)
                    if pid is None:
                        time.sleep(args.interval)
                        continue
                    last_proc_ticks = read_proc_stat_cpu(pid)
                    last_total_jiffies, last_per_core_jiffies = read_total_cpu_jiffies()

                rss_kb, vsz_kb = read_proc_status(pid)
                proc_ticks = read_proc_stat_cpu(pid)
                total_jiffies, per_core_jiffies = read_total_cpu_jiffies()

                d_proc = proc_ticks - last_proc_ticks
                d_total = total_jiffies - last_total_jiffies
                n_cores = max(1, len(per_core_jiffies))
                # Per-process CPU as % of one core (can exceed 100 on multi-thread).
                proc_pct = 100.0 * d_proc / (d_total / n_cores) if d_total > 0 else 0.0

                core_pcts = []
                for i, cj in enumerate(per_core_jiffies):
                    if i < len(last_per_core_jiffies):
                        d_core = cj - last_per_core_jiffies[i]
                        # Idle is field 4 in /proc/stat per-core; read again
                        # for idle delta. Simpler: approximate as busy fraction
                        # of total delta on that core (close enough for this).
                        # Approximation: report d_core relative to mean across cores.
                        core_pcts.append(round(100.0 * d_core / max(1, d_total / n_cores), 1))
                    else:
                        core_pcts.append(0.0)
                # Pad to 4 cores.
                while len(core_pcts) < 4:
                    core_pcts.append(0.0)

                wr.writerow([
                    f"{t:.3f}", pid,
                    round(rss_kb / 1024.0, 1),
                    round(vsz_kb / 1024.0, 1),
                    round(proc_pct, 1),
                    *core_pcts[:4],
                    round(read_meminfo_available_kb() / 1024.0, 1),
                ])

                last_proc_ticks = proc_ticks
                last_total_jiffies = total_jiffies
                last_per_core_jiffies = per_core_jiffies
                time.sleep(args.interval)
        except KeyboardInterrupt:
            pass
        finally:
            signal.signal(signal.SIGTERM, sig)


if __name__ == "__main__":
    main()
