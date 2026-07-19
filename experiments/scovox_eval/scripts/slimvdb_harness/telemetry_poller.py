#!/usr/bin/env python3
"""Logs GPU memory (nvidia-smi) + system memory at a fixed period until killed
or `--duration` elapses. Writes a CSV: time_s, gpu_mem_mb, sys_used_mb.

Designed to run alongside a SLIM-VDB job (inside or outside Docker doesn't
matter — nvidia-smi sees all GPU memory consumers regardless of container).
The parent script SIGTERMs this poller when the SLIM-VDB run finishes.

Usage:
    python telemetry_poller.py --period 0.5 --out run.csv [--duration 3600]
"""
import argparse, csv, subprocess, time


def poll_gpu_mem_mb() -> float:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        if r.returncode != 0:
            return 0.0
        return float(sum(int(x.strip()) for x in r.stdout.splitlines() if x.strip()))
    except Exception:
        return 0.0


def poll_sys_used_mb() -> float:
    try:
        with open("/proc/meminfo") as f:
            info = {}
            for line in f:
                k, _, v = line.partition(":")
                info[k.strip()] = int(v.split()[0])  # kB
        total = info.get("MemTotal", 0)
        avail = info.get("MemAvailable", 0)
        return (total - avail) / 1024.0
    except Exception:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=float, default=float("inf"),
                    help="max duration seconds (default: until killed)")
    args = ap.parse_args()

    t0 = time.time()
    baseline_sys = poll_sys_used_mb()
    baseline_gpu = poll_gpu_mem_mb()
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "gpu_mem_mb", "sys_used_mb",
                    "gpu_delta_mb", "sys_delta_mb"])
        f.flush()
        while (time.time() - t0) < args.duration:
            now = time.time() - t0
            gpu = poll_gpu_mem_mb()
            sysm = poll_sys_used_mb()
            w.writerow([f"{now:.3f}", f"{gpu:.1f}", f"{sysm:.1f}",
                        f"{gpu - baseline_gpu:.1f}",
                        f"{sysm - baseline_sys:.1f}"])
            f.flush()
            time.sleep(args.period)


if __name__ == "__main__":
    main()
