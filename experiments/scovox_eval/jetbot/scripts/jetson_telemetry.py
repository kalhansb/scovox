#!/usr/bin/env python3
"""Capture per-second telemetry for the Jetson scovox_mapping_node:
  - RSS (MB), VSZ (MB)
  - CPU% (per-process, sum across cores)
  - Per-core CPU% busy (system-wide; core count read from /proc/stat)
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
    """Returns (aggregate_total, [(busy, total) per core]).

    /proc/stat fields are user nice system idle iowait irq softirq steal ...
    Busy must exclude idle+iowait -- summing every field gives each core's
    elapsed time, which is identical for all cores and so measures nothing.
    """
    total = 0
    per_core = []
    with open("/proc/stat") as f:
        for line in f:
            parts = line.split()
            if not parts or not parts[0].startswith("cpu"):
                continue
            vals = [int(x) for x in parts[1:]]
            core_total = sum(vals)
            idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
            if parts[0] == "cpu":
                total = core_total
            else:
                per_core.append((core_total - idle, core_total))
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

    # Core count is read from the machine, not assumed -- this was written for
    # the 4-core Nano but also runs on the 12-core AGX Orin.
    n_cpu = len(read_total_cpu_jiffies()[1])

    with out_path.open("w", buffering=1) as fp:
        wr = csv.writer(fp)
        wr.writerow(["wall_time", "pid", "rss_mb", "vsz_mb", "proc_cpu_pct"]
                    + [f"core{i}_pct" for i in range(n_cpu)]
                    + ["mem_avail_mb"])

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
                for i, (busy, tot) in enumerate(per_core_jiffies):
                    if i < len(last_per_core_jiffies):
                        d_busy = busy - last_per_core_jiffies[i][0]
                        d_tot = tot - last_per_core_jiffies[i][1]
                        core_pcts.append(round(100.0 * d_busy / d_tot, 1) if d_tot > 0 else 0.0)
                    else:
                        core_pcts.append(0.0)
                while len(core_pcts) < n_cpu:
                    core_pcts.append(0.0)

                wr.writerow([
                    f"{t:.3f}", pid,
                    round(rss_kb / 1024.0, 1),
                    round(vsz_kb / 1024.0, 1),
                    round(proc_pct, 1),
                    *core_pcts[:n_cpu],
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
