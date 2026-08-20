#!/usr/bin/env python3
"""Summarise E7 embedded-feasibility runs into the table the plan asks for.

Reads the logs + telemetry CSVs written by scenenn_e7_embedded.sh and reports,
per config: frame_ms p50/p95/p99, sustained Hz, RTF against the sensor period,
the stage split, process CPU%, per-core busy%, RSS, and -- where
log_mem_usage was on -- the [memSplit]/[memBlocks] grid-byte breakdown.

Two conventions matter and are stated in the output rather than buried:
  * the analysis window is the LAST 200 frames (warm-up discarded), the same
    window scenenn_fps_sweep.sh uses, so numbers are comparable to it;
  * CPU/RSS are taken only from telemetry samples whose wall clock falls
    inside that frame window, so post-run idle does not dilute the mean.

Usage: scenenn_e7_report.py <results_dir> [sensor_rate_hz]
"""
import csv
import re
import statistics
import sys
from pathlib import Path

WINDOW = 200            # trailing frames analysed; matches the fps sweep
FRAME_RE = re.compile(
    r"\[INFO\] \[(\d+\.\d+)\].*?frame_ms=([\d.]+) tf_ms=([\d.]+) "
    r"integrate_ms=([\d.]+) publish_ms=([\d.]+) rss_mb=([\d.]+) "
    r"tsdf_ms=([\d.]+) sembeta_ms=([\d.]+)")
MEMSPLIT_RE = re.compile(
    r"\[memSplit\] tsdf_voxels=(\d+) tsdf_grid_mb=([\d.]+) "
    r"semdir_voxels=(\d+) semdir_grid_mb=([\d.]+) "
    r"fine_voxels=(\d+) fine_grid_mb=([\d.]+)")
MEMBLOCKS_RE = re.compile(
    r"\[memBlocks\] .*?beta_vox=(\d+) .*?beta_fill=([\d.]+) beta_mb=([\d.]+) \| "
    r".*?dir_vox=(\d+) .*?dir_fill=([\d.]+) dir_mb=([\d.]+)")


def pct(xs, q):
    """Nearest-rank percentile -- no interpolation, so every reported value is
    an actually-observed frame time."""
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(q / 100.0 * len(s) + 0.5)) - 1))
    return s[k]


def parse_log(path):
    rows = [m.groups() for m in FRAME_RE.finditer(path.read_text(errors="ignore"))]
    if not rows:
        return None
    win = rows[-WINDOW:] if len(rows) > WINDOW else rows
    f = lambda i: [float(r[i]) for r in win]
    txt = path.read_text(errors="ignore")
    ms = MEMSPLIT_RE.findall(txt)
    mb = MEMBLOCKS_RE.findall(txt)
    return {
        "n_total": len(rows), "n_win": len(win),
        "t0": float(win[0][0]), "t1": float(win[-1][0]),
        "frame": f(1), "tf": f(2), "integ": f(3), "pub": f(4),
        "rss": f(5), "tsdf": f(6), "sem": f(7),
        "memsplit": [float(x) if "." in x else int(x) for x in ms[-1]] if ms else None,
        "memblocks": [float(x) for x in mb[-1]] if mb else None,
    }


def parse_telem(path, t0, t1):
    if not path.exists():
        return None
    with path.open() as fp:
        rows = list(csv.DictReader(fp))
    cores = [k for k in (rows[0].keys() if rows else []) if re.fullmatch(r"core\d+_pct", k)]
    sel = [r for r in rows if t0 <= float(r["wall_time"]) <= t1]
    if not sel:
        return None
    proc = [float(r["proc_cpu_pct"]) for r in sel]
    rss = [float(r["rss_mb"]) for r in sel]
    avail = [float(r["mem_avail_mb"]) for r in sel]
    # Integrate CPU% over real sample gaps -> process CPU-seconds in the window.
    # Compared against sum(frame_ms) by the caller: a ratio near 1.0 is the
    # evidence that frame_ms is single-threaded CPU-bound work rather than
    # time spent blocked. Anything well under 1.0 means the walker is stalling.
    cpu_s = 0.0
    for a, b in zip(sel, sel[1:]):
        dt = float(b["wall_time"]) - float(a["wall_time"])
        cpu_s += float(b["proc_cpu_pct"]) / 100.0 * dt
    # Per-core busy averaged over the window, then sorted: the shape of this
    # list answers "how many cores is this actually using", which is the
    # embedded question. A single-threaded integrator shows one hot core.
    per_core = sorted((statistics.mean(float(r[c]) for r in sel) for c in cores),
                      reverse=True)
    return {"n": len(sel), "proc_mean": statistics.mean(proc), "proc_max": max(proc),
            "rss_max": max(rss), "rss_final": rss[-1], "avail_min": min(avail),
            "per_core": per_core, "n_cores": len(cores), "cpu_s": cpu_s,
            "wall_s": float(sel[-1]["wall_time"]) - float(sel[0]["wall_time"])}


def main():
    d = Path(sys.argv[1])
    rate = float(sys.argv[2]) if len(sys.argv) > 2 else 25.0
    period_ms = 1000.0 / rate

    plat = (d / "platform.txt")
    if plat.exists():
        print("--- platform " + "-" * 55)
        print(plat.read_text().rstrip())
    print(f"\n--- timing (last {WINDOW} frames; sensor {rate:g} Hz = {period_ms:.0f} ms) "
          + "-" * 8)
    print(f"{'run':<12}{'n':>5}{'mean':>8}{'p50':>8}{'p95':>8}{'p99':>8}"
          f"{'Hz_int':>8}{'Hz_e2e':>8}{'RTF':>6}{'tsdf':>7}{'sem':>6}{'pub':>6}")

    runs = {}
    for log in sorted(d.glob("*.log")):
        r = parse_log(log)
        if not r:
            print(f"{log.stem:<12}  NO FRAMES")
            continue
        runs[log.stem] = r
        mean = statistics.mean(r["frame"])
        span = r["t1"] - r["t0"]
        e2e = r["n_win"] / span if span > 0 else 0.0
        print(f"{log.stem:<12}{r['n_win']:>5}{mean:>8.1f}{pct(r['frame'],50):>8.1f}"
              f"{pct(r['frame'],95):>8.1f}{pct(r['frame'],99):>8.1f}"
              f"{1000/mean:>8.2f}{e2e:>8.2f}{mean/period_ms:>6.2f}"
              f"{statistics.mean(r['tsdf']):>7.1f}{statistics.mean(r['sem']):>6.1f}"
              f"{statistics.mean(r['pub']):>6.1f}")
    print("  Hz_int = 1000/mean(frame_ms), the mapper's own cost -- the number")
    print("           comparable to E5. Hz_e2e = frames/wall, which in this harness")
    print("           measures the PYTHON REPLAY LOADER (PIL decode + ~2 MB/frame of")
    print("           byte copies), not SCovox: the node idles waiting for input.")
    print("  RTF is against the nominal sensor period and uses Hz_int.")

    fast = [runs[k] for k in sorted(runs) if k.startswith("fast_r")]
    if len(fast) > 1:
        means = [statistics.mean(r["frame"]) for r in fast]
        m, sd = statistics.mean(means), statistics.stdev(means)
        print(f"\nfast_r n={len(fast)}: mean {m:.1f} +/- {sd:.1f} ms "
              f"(SD {100*sd/m:.2f}%) -> {1000/m:.2f} Hz, RTF {m/period_ms:.2f}")
        if "fast_mem" in runs:
            mm = statistics.mean(runs["fast_mem"]["frame"])
            print(f"  log_mem_usage perturbation: {mm:.1f} vs {m:.1f} ms "
                  f"({100*(mm-m)/m:+.1f}%) -- grid walk every 10 frames")

    print("\n--- cpu / memory (telemetry, same window) " + "-" * 26)
    print(f"{'run':<12}{'cpu%':>8}{'cpumax':>8}{'cores>50':>10}{'rss_mb':>9}"
          f"{'avail_mb':>10}{'hot cores (busy%)':>22}")
    n_cores = 0
    for name, r in runs.items():
        t = parse_telem(d / f"{name}.telem.csv", r["t0"], r["t1"])
        if not t:
            print(f"{name:<12}  no telemetry in window")
            continue
        n_cores = t["n_cores"]
        hot = " ".join(f"{c:.0f}" for c in t["per_core"][:4])
        n_busy = sum(1 for c in t["per_core"] if c > 50)
        print(f"{name:<12}{t['proc_mean']:>8.1f}{t['proc_max']:>8.1f}{n_busy:>10}"
              f"{t['rss_max']:>9.1f}{t['avail_min']:>10.0f}{hot:>22}")
        # Is frame_ms actually CPU-bound single-thread work?
        frame_s = sum(r["frame"]) / 1000.0
        print(f"{'':<12}  cpu {t['cpu_s']:.1f}s vs frames {frame_s:.1f}s over "
              f"{t['wall_s']:.0f}s wall -> cpu/frame-time = {t['cpu_s']/frame_s:.2f}, "
              f"duty {100*frame_s/t['wall_s']:.0f}%")
    if n_cores:
        print(f"  cpu% is % of ONE core (>100 = multi-threaded); {n_cores} cores total.")

    print("\n--- grid bytes ([memSplit], runs with log_mem_usage:=true) " + "-" * 9)
    any_mem = False
    for name, r in runs.items():
        if not r["memsplit"]:
            continue
        any_mem = True
        tv, tmb, sv, smb, fv, fmb = r["memsplit"]
        print(f"{name}:")
        print(f"  tsdf   {int(tv):>9} voxels  {tmb:>8.3f} MB   <- optional module, "
              "never folded into the core number")
        print(f"  semdir {int(sv):>9} voxels  {smb:>8.3f} MB   <- Beta+Dirichlet core")
        print(f"  fine   {int(fv):>9} voxels  {fmb:>8.3f} MB")
        print(f"  core+tsdf total          {smb+tmb:>8.3f} MB")
        if r["memblocks"]:
            # Per-grid, each against ITS OWN active count -- the Beta and Dir
            # grids hold different voxel sets (carving allocates Beta-only
            # free voxels), so semdir_grid_mb / semdir_voxels would mix them.
            bv, bf, bmb, dv, df, dmb = r["memblocks"]
            bpv = bmb * 1024 * 1024 / bv if bv else 0.0
            dpv = dmb * 1024 * 1024 / dv if dv else 0.0
            print(f"  beta {int(bv):>8} vox fill {bf:.3f} {bmb:>7.3f} MB "
                  f"= {bpv:6.1f} B/vox allocated")
            print(f"  dir  {int(dv):>8} vox fill {df:.3f} {dmb:>7.3f} MB "
                  f"= {dpv:6.1f} B/vox allocated")
            print(f"  (B/vox is allocated blocks over active voxels, so it "
                  "tracks fill, not struct size)")
    if not any_mem:
        print("  none -- no run had log_mem_usage:=true")


if __name__ == "__main__":
    main()
