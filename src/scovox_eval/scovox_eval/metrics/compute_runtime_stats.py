"""Parse SCovox run logs and compute runtime summary statistics.

Extracts per-frame timing from log lines of the form:
    integrated: voxels=5135 frame_ms=141.4 tf_ms=0.0 integrate_ms=140.3 publish_ms=1.1 mem_mb=37.5

Produces median, P95, P99 frame time, per-stage breakdown, and memory stats.
"""

from __future__ import annotations

import argparse
import re

import numpy as np

# Matches key=value pairs in log lines (e.g. frame_ms=141.4, voxels=5135)
_KV_RE = re.compile(r"(\w+)=([\d.]+)")


def parse_log_timing(log_path: str) -> dict[str, np.ndarray]:
    """Parse a SCovox run log and extract per-frame timing arrays.

    Parameters
    ----------
    log_path : path to a .log file

    Returns
    -------
    dict with keys: frame_ms, tf_ms, integrate_ms, publish_ms, mem_mb, voxels.
    Each is a 1D float64 array with one entry per frame.
    """
    fields: dict[str, list[float]] = {
        "frame_ms": [],
        "tf_ms": [],
        "integrate_ms": [],
        "publish_ms": [],
        "mem_mb": [],
        "voxels": [],
    }

    with open(log_path) as f:
        for line in f:
            if "frame_ms=" not in line:
                continue
            kvs = dict(_KV_RE.findall(line))
            for key in fields:
                if key in kvs:
                    fields[key].append(float(kvs[key]))

    return {k: np.array(v, dtype=np.float64) for k, v in fields.items()}


def _summarise(arr: np.ndarray) -> dict[str, float]:
    """Compute summary stats for a 1D timing array."""
    if len(arr) == 0:
        return {"median": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "std": 0.0,
                "min": 0.0, "max": 0.0}
    return {
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def compute_runtime_stats(
    timing: dict[str, np.ndarray],
) -> dict[str, float | dict[str, float]]:
    """Compute summary statistics from per-frame timing arrays.

    Parameters
    ----------
    timing : dict as returned by parse_log_timing

    Returns
    -------
    dict with:
        n_frames, frame_ms (summary), stages (per-stage summaries),
        memory (peak_mb, mean_mb, final_mb), voxels (final, peak)
    """
    result: dict = {"n_frames": len(timing.get("frame_ms", []))}

    result["frame_ms"] = _summarise(timing.get("frame_ms", np.array([])))

    stages = {}
    for key in ("tf_ms", "integrate_ms", "publish_ms"):
        if key in timing and len(timing[key]) > 0:
            stages[key] = _summarise(timing[key])
    result["stages"] = stages

    mem = timing.get("mem_mb", np.array([]))
    if len(mem) > 0:
        result["memory"] = {
            "peak_mb": float(np.max(mem)),
            "mean_mb": float(np.mean(mem)),
            "final_mb": float(mem[-1]),
        }
    else:
        result["memory"] = {"peak_mb": 0.0, "mean_mb": 0.0, "final_mb": 0.0}

    voxels = timing.get("voxels", np.array([]))
    if len(voxels) > 0:
        result["voxels"] = {
            "final": int(voxels[-1]),
            "peak": int(np.max(voxels)),
        }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Compute runtime statistics from SCovox run logs"
    )
    parser.add_argument("input", help="SCovox run log (.log)")
    args = parser.parse_args()

    timing = parse_log_timing(args.input)
    stats = compute_runtime_stats(timing)

    print(f"Frames: {stats['n_frames']}")
    if "voxels" in stats:
        print(f"Voxels: final={stats['voxels']['final']}  peak={stats['voxels']['peak']}")

    # Frame time
    fs = stats["frame_ms"]
    print(f"\nFrame time (ms):")
    print(f"  median={fs['median']:.1f}  P95={fs['p95']:.1f}  P99={fs['p99']:.1f}  "
          f"mean={fs['mean']:.1f}+/-{fs['std']:.1f}  range=[{fs['min']:.1f}, {fs['max']:.1f}]")

    # Per-stage breakdown
    if stats["stages"]:
        print(f"\nStage breakdown (median ms):")
        for name, s in stats["stages"].items():
            print(f"  {name:<16} {s['median']:7.1f}")

    # Memory
    m = stats["memory"]
    if m["peak_mb"] > 0:
        print(f"\nMemory: peak={m['peak_mb']:.1f} MB  mean={m['mean_mb']:.1f} MB  "
              f"final={m['final_mb']:.1f} MB")


if __name__ == "__main__":
    main()
