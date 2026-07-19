#!/usr/bin/env python3
"""B6 full experiment — per-voxel semantic flip churn across 4 update rules.

Mechanism test for the paper's claim "Dirichlet stabilises noisy semantics."
Currently defended only by aggregate mIoU/ECE; B6 measures the per-voxel
behaviour directly.

Why offline: the per-voxel argmax-flip count depends only on (a) the stream
of (voxel, class) observations and (b) the update rule. Whether scovox's
Beta carving keeps a voxel or not is irrelevant — flips happen at the
semantic update, not the geometric one. So we can simulate all four rules
on identical streams without recompiling four scovox builds.

The four update rules:

  scovox-Dirichlet — sparse K_TOP=2 Dirichlet posterior over class IDs,
    with `kappa0=2.0` and `semantic_min_confidence=0.1` mirroring the
    paper config. Each observation contributes `kappa0` pseudocounts split
    between covered classes (one-hot here) and the residual `a_unk`.
    Slot eviction follows scovox's strict-`>` admission rule (the lowest
    slot is replaced only if the incoming evidence STRICTLY beats it).
    Argmax: highest of {sem_cnt[0], sem_cnt[1], a_unk}, where a_unk maps
    to class 0 ("unknown"). Flip on argmax change.

  scovox-MV — same K_TOP=2 sparse store, but each observation casts a
    single +1 vote for its argmax class. `kappa0` and confidence are
    ignored, exactly mirroring scovox::majority_vote_semantics.

  scovox-NP — "naive / last-observation-wins". Each observation overwrites
    state; argmax = last class observed. Memoryless. Mirrors
    scovox::naive_update_semantics (which wipes prior slots on every
    update).

  log_odds-semantic — dense per-class log-odds, OctoMap-style. Each
    observation increments L[obs_class] by log(p_hit/(1−p_hit)) and
    decrements L[c'] by log((1−p_miss)/p_miss) for *all* c' != obs that
    have been seen at least once. Argmax = max-L class. Capacity is
    unlimited (this is the strongest log-odds baseline; scovox's K_TOP=2
    sparse cap is the differentiator).

Anchors:
  - Replica room0, m2f ADE150→scovox18 LUT, 2000 frames, voxel res 0.05 m
  - KITTI seq08, PolarSeg learning IDs, 100 scans, voxel res 0.10 m

Reported per anchor and per method:
  - total voxels with ≥1 observation
  - % voxels that ever flip
  - mean / median / p99 flips per voxel
  - flips-per-observation (total flips / total observations)
  - same, stratified by per-voxel observation count {≥5, ≥20, ≥100}

Output: ``results/b6_full/<anchor>_flip_stats.npz`` (per-voxel
flip_count_<method> + obs_count) and a markdown table in stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# ----- Paths ---------------------------------------------------------------

WS = Path.home() / "projects/HMR_Exploration_Experiment/hmr_exploration_ws"
EVAL_PKG = WS / "src/robot_sw/distributed_mapping/scovox_eval"
KITTI_ROOT = WS / "data/semantickitti/dataset"
REPLICA_ROOT = WS / "data/replica_niceslam"

# Match scovox baseline params for both anchors (post-NG, A1–A6 baseline).
KAPPA0 = 2.0
SEMANTIC_MIN_CONFIDENCE = 0.1
K_TOP = 2

# Log-odds-semantic rule (OctoMap-style):
#   L[obs] += L_HIT;   L[c' seen] += L_MISS  for c' != obs
# L_HIT  = log(0.7/0.3)  ≈ +0.847  (consistent with log_odds_map.hpp `l_hit=0.85`)
# L_MISS = log(0.4/0.6)  ≈ -0.405  (consistent with `l_miss=-0.40`)
# These match the geometric log-odds parameters scovox uses elsewhere, so the
# semantic baseline is "the natural per-class extension" rather than tuned.
L_HIT  = float(np.log(0.7 / 0.3))
L_MISS = float(np.log(0.4 / 0.6))

# KITTI raw -> learning ID LUT (mirrors semantickitti_replay_node.py)
LEARNING_MAP = {
    0: 0, 1: 0, 10: 1, 11: 2, 13: 5, 15: 3, 16: 5, 18: 4, 20: 5,
    30: 6, 31: 7, 32: 8, 40: 9, 44: 10, 48: 11, 49: 12,
    50: 13, 51: 14, 52: 0, 60: 15, 70: 16, 71: 17, 72: 18, 80: 19,
    81: 0, 99: 0, 252: 1, 253: 7, 254: 6, 255: 8,
    256: 5, 257: 5, 258: 4, 259: 5,
}
_KITTI_LUT = np.zeros(260, dtype=np.uint8)
for raw, mapped in LEARNING_MAP.items():
    _KITTI_LUT[raw] = mapped


# ----- Voxel-key packing (same as b6_prelim) -------------------------------

def voxelize_pack(points: np.ndarray, res: float) -> np.ndarray:
    BIAS = 1 << 20
    ijk = np.floor(points / res).astype(np.int64) + BIAS
    if (ijk < 0).any() or (ijk >= (1 << 21)).any():
        raise RuntimeError("voxel coord overflow — increase BIAS")
    keys = (ijk[:, 0].astype(np.uint64) << 42) \
         | (ijk[:, 1].astype(np.uint64) << 21) \
         | ijk[:, 2].astype(np.uint64)
    return keys


# ----- Per-voxel state machines for the four rules -------------------------
#
# State per voxel:
#   d_cnt   : float[K_TOP]    Dirichlet sem counts
#   d_cls   : uint16[K_TOP]   Dirichlet sem class IDs
#   d_unk   : float           Dirichlet a_unk residual
#   m_cnt   : float[K_TOP]    MV sem counts
#   m_cls   : uint16[K_TOP]   MV sem class IDs
#   np_cls  : uint16          NP last-class (0 = no obs yet)
#   lo_L    : dict[cls -> L]  log-odds per class
#
# Plus per-rule:
#   {rule}_argmax  : uint16   current argmax class (0 = unknown / no obs)
#   {rule}_flips   : uint32
#
# We also track total observation count per voxel (same across rules).
#
# We use parallel numpy arrays keyed by a per-voxel index. The
# voxel-key→index map is built lazily as we encounter new keys.


class FlipSim:
    """Streaming flip simulator. One instance per anchor."""

    def __init__(self, n_classes_max: int):
        self.n_classes_max = n_classes_max
        self.key_to_idx: dict[int, int] = {}
        self.obs_count: list[int] = []          # int per voxel
        # Dirichlet
        self.d_cnt: list[np.ndarray] = []       # float[K_TOP]
        self.d_cls: list[np.ndarray] = []       # uint16[K_TOP]
        self.d_unk: list[float] = []
        self.d_argmax: list[int] = []
        self.d_flips: list[int] = []
        # MV
        self.m_cnt: list[np.ndarray] = []
        self.m_cls: list[np.ndarray] = []
        self.m_argmax: list[int] = []
        self.m_flips: list[int] = []
        # NP
        self.np_cls: list[int] = []
        self.np_flips: list[int] = []
        # Log-odds dense
        self.lo_L: list[dict] = []
        self.lo_argmax: list[int] = []
        self.lo_flips: list[int] = []

    def _new_voxel(self):
        self.obs_count.append(0)
        self.d_cnt.append(np.zeros(K_TOP, dtype=np.float32))
        self.d_cls.append(np.zeros(K_TOP, dtype=np.uint16))
        self.d_unk.append(0.0)
        self.d_argmax.append(0)
        self.d_flips.append(0)
        self.m_cnt.append(np.zeros(K_TOP, dtype=np.float32))
        self.m_cls.append(np.zeros(K_TOP, dtype=np.uint16))
        self.m_argmax.append(0)
        self.m_flips.append(0)
        self.np_cls.append(0)
        self.np_flips.append(0)
        self.lo_L.append({})
        self.lo_argmax.append(0)
        self.lo_flips.append(0)

    def _idx(self, key: int) -> int:
        idx = self.key_to_idx.get(key)
        if idx is None:
            idx = len(self.obs_count)
            self.key_to_idx[key] = idx
            self._new_voxel()
        return idx

    @staticmethod
    def _sparse_add(cnt: np.ndarray, cls_arr: np.ndarray, cls_id: int,
                    inc: float, unk_ref: list, idx: int) -> None:
        """Mirror scovox `sparse_add`: strict-`>` admission, lowest-slot evict."""
        for i in range(K_TOP):
            if cnt[i] > 0 and cls_arr[i] == cls_id:
                cnt[i] += inc
                return
        # Free slot?
        for i in range(K_TOP):
            if cnt[i] == 0:
                cls_arr[i] = cls_id
                cnt[i] = inc
                return
        # All slots full — evict the lowest if `inc` strictly exceeds it.
        # Otherwise route the increment to a_unk (overflow).
        lo = int(np.argmin(cnt))
        if inc > cnt[lo]:
            unk_ref[idx] += cnt[lo]   # evicted slot's mass joins residual
            cls_arr[lo] = cls_id
            cnt[lo] = inc
        else:
            unk_ref[idx] += inc

    def _argmax_with_unk(self, cnt: np.ndarray, cls_arr: np.ndarray,
                         a_unk: float) -> int:
        """Argmax over slots vs unknown residual; class 0 == unknown."""
        best_v = a_unk
        best_c = 0
        for i in range(K_TOP):
            if cnt[i] > best_v:
                best_v = float(cnt[i])
                best_c = int(cls_arr[i])
        return best_c

    def update(self, key: int, cls_obs: int) -> None:
        """Apply one observation `(key, cls_obs)` to all four state machines."""
        if cls_obs == 0:        # ignore "unknown" observations
            return
        idx = self._idx(key)
        self.obs_count[idx] += 1

        # A "flip" requires a *prior* non-unknown argmax. The 0→class
        # transition on the very first observation is initialization, not
        # a flip. Without this guard every voxel registers exactly one
        # spurious flip on its first hit.

        # ---- Dirichlet (kappa0 split between covered & residual) ----
        # one-hot input with confidence 1.0 → all kappa0 mass goes to cls_obs,
        # 0 to residual. Mirrors dirichlet_update_semantics for one-hot probs
        # with confidence ≥ semantic_min_confidence (1.0 ≥ 0.1).
        d_cnt = self.d_cnt[idx]; d_cls = self.d_cls[idx]
        prev_d = self.d_argmax[idx]
        self._sparse_add(d_cnt, d_cls, cls_obs, KAPPA0, self.d_unk, idx)
        new_d = self._argmax_with_unk(d_cnt, d_cls, self.d_unk[idx])
        if new_d != prev_d:
            self.d_argmax[idx] = new_d
            if prev_d != 0:
                self.d_flips[idx] += 1

        # ---- Majority-vote (+1 per observation, K_TOP=2 sparse) ----
        m_cnt = self.m_cnt[idx]; m_cls = self.m_cls[idx]
        prev_m = self.m_argmax[idx]
        unk_dummy = [0.0]   # MV doesn't track residual; throwaway sink
        self._sparse_add(m_cnt, m_cls, cls_obs, 1.0, unk_dummy, 0)
        if m_cnt[0] >= m_cnt[1]:
            new_m = int(m_cls[0]) if m_cnt[0] > 0 else 0
        else:
            new_m = int(m_cls[1])
        if new_m != prev_m:
            self.m_argmax[idx] = new_m
            if prev_m != 0:
                self.m_flips[idx] += 1

        # ---- NP (last-observation-wins) ----
        prev_np = self.np_cls[idx]
        if cls_obs != prev_np:
            self.np_cls[idx] = cls_obs
            if prev_np != 0:
                self.np_flips[idx] += 1

        # ---- Log-odds-semantic (dense, OctoMap-style) ----
        lo = self.lo_L[idx]
        prev_lo = self.lo_argmax[idx]
        for c, L in lo.items():
            if c != cls_obs:
                lo[c] = L + L_MISS
        lo[cls_obs] = lo.get(cls_obs, 0.0) + L_HIT
        new_lo = max(lo.items(), key=lambda kv: kv[1])
        new_lo_cls = new_lo[0] if new_lo[1] > 0.0 else 0
        if new_lo_cls != prev_lo:
            self.lo_argmax[idx] = new_lo_cls
            if prev_lo != 0:
                self.lo_flips[idx] += 1


# ----- Anchor loaders (mirror b6_prelim layout) ----------------------------

def load_kitti_calib(seq_dir: Path):
    Tr = None
    for line in (seq_dir / "calib.txt").read_text().splitlines():
        if line.startswith("Tr:"):
            vals = np.array([float(x) for x in line.split()[1:]], dtype=np.float64)
            Tr = np.eye(4, dtype=np.float64)
            Tr[:3, :4] = vals.reshape(3, 4)
    return Tr


def kitti_frame_obs(seq_dir: Path, idx: int, Tr, pose_cam, res: float,
                    max_range=30.0, min_range=1.0):
    bin_path = seq_dir / "velodyne" / f"{idx:06d}.bin"
    lab_path = seq_dir / "predictions" / f"{idx:06d}.label"
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)[:, :3].astype(np.float64)
    lab = np.fromfile(lab_path, dtype=np.uint32) & 0xFFFF
    sem = _KITTI_LUT[np.minimum(lab, 259)]
    r = np.linalg.norm(pts, axis=1)
    keep = (r >= min_range) & (r <= max_range) & (sem > 0)
    pts = pts[keep]; sem = sem[keep]
    if pts.size == 0:
        return np.empty(0, np.uint64), np.empty(0, np.uint8)
    Tw = pose_cam @ Tr
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    pts_w = (Tw @ pts_h.T).T[:, :3]
    return voxelize_pack(pts_w, res), sem.astype(np.uint8)


def replica_frame_obs(scene_dir: Path, idx: int, K, pose, lut: np.ndarray,
                      res: float, pixel_stride=2, max_depth=6.0):
    import cv2
    depth_path = scene_dir / "depth" / f"{idx:06d}.png"
    sem_path = scene_dir / "semantic_m2f_ade" / f"{idx:06d}.png"
    if not depth_path.exists() or not sem_path.exists():
        return np.empty(0, np.uint64), np.empty(0, np.uint8)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 6553.5
    sem = cv2.imread(str(sem_path), cv2.IMREAD_UNCHANGED).astype(np.uint8)
    depth = depth[::pixel_stride, ::pixel_stride]
    sem = sem[::pixel_stride, ::pixel_stride]
    H, W = depth.shape
    fx = K[0, 0] / pixel_stride; fy = K[1, 1] / pixel_stride
    cx = K[0, 2] / pixel_stride; cy = K[1, 2] / pixel_stride
    valid = (depth > 0.05) & (depth < max_depth)
    ys, xs = np.where(valid)
    z = depth[ys, xs].astype(np.float64)
    x = (xs.astype(np.float64) - cx) * z / fx
    y = (ys.astype(np.float64) - cy) * z / fy
    pts_cam = np.stack([x, y, z], axis=1)
    cls_raw = sem[ys, xs]
    cls_mapped = lut[np.minimum(cls_raw, 150)]
    keep = cls_mapped > 0
    pts_cam = pts_cam[keep]; cls_mapped = cls_mapped[keep]
    if pts_cam.size == 0:
        return np.empty(0, np.uint64), np.empty(0, np.uint8)
    pts_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_w = (pose @ pts_h.T).T[:, :3]
    return voxelize_pack(pts_w, res), cls_mapped.astype(np.uint8)


# ----- Streaming runners ---------------------------------------------------

def run_kitti(seq="08", res=0.10, n_scans=100):
    seq_dir = KITTI_ROOT / "sequences" / seq
    Tr = load_kitti_calib(seq_dir)
    poses = np.loadtxt(seq_dir / "poses.txt").reshape(-1, 3, 4)
    poses4 = np.tile(np.eye(4, dtype=np.float64), (poses.shape[0], 1, 1))
    poses4[:, :3, :4] = poses
    sim = FlipSim(n_classes_max=20)
    print(f"[kitti seq{seq}] {n_scans} scans, res={res} m", flush=True)
    t0 = time.time()
    for i in range(n_scans):
        keys, cls = kitti_frame_obs(seq_dir, i, Tr, poses4[i], res)
        for k, c in zip(keys.tolist(), cls.tolist()):
            sim.update(int(k), int(c))
        if (i + 1) % 25 == 0:
            print(f"  scan {i+1}/{n_scans}  ({time.time()-t0:.1f}s, "
                  f"voxels={len(sim.obs_count)})", flush=True)
    print(f"[kitti] done {time.time()-t0:.1f}s", flush=True)
    return sim


def run_replica(scene="room0", res=0.05, n_frames=2000, pixel_stride=2):
    scene_dir = REPLICA_ROOT / scene
    cam = json.loads((REPLICA_ROOT / "cam_params.json").read_text())["camera"]
    K = np.array([[cam["fx"], 0, cam["cx"]],
                  [0, cam["fy"], cam["cy"]],
                  [0, 0, 1]], dtype=np.float64)
    poses = np.loadtxt(scene_dir / "poses.txt").reshape(-1, 4, 4)
    n_avail = min(n_frames, poses.shape[0])
    lut = np.load(EVAL_PKG / "scripts/ade150_to_scovox18.npz")["lut"]
    sim = FlipSim(n_classes_max=18)
    print(f"[replica {scene}] {n_avail} frames, pixel_stride={pixel_stride}, "
          f"res={res} m", flush=True)
    t0 = time.time()
    for i in range(n_avail):
        keys, cls = replica_frame_obs(scene_dir, i, K, poses[i], lut, res,
                                      pixel_stride=pixel_stride)
        for k, c in zip(keys.tolist(), cls.tolist()):
            sim.update(int(k), int(c))
        if (i + 1) % 200 == 0:
            print(f"  frame {i+1}/{n_avail}  ({time.time()-t0:.1f}s, "
                  f"voxels={len(sim.obs_count)})", flush=True)
    print(f"[replica] done {time.time()-t0:.1f}s", flush=True)
    return sim


# ----- Reporting -----------------------------------------------------------

def summarise(sim: FlipSim, name: str, out_path: Path):
    obs = np.asarray(sim.obs_count, dtype=np.int64)
    flips_d = np.asarray(sim.d_flips, dtype=np.int64)
    flips_m = np.asarray(sim.m_flips, dtype=np.int64)
    flips_n = np.asarray(sim.np_flips, dtype=np.int64)
    flips_l = np.asarray(sim.lo_flips, dtype=np.int64)
    n = len(obs)
    total_obs = int(obs.sum())

    print(f"\n=== {name} ===", flush=True)
    print(f"voxels: {n:,}    total observations: {total_obs:,}", flush=True)
    print(f"obs/voxel mean={obs.mean():.1f} median={np.median(obs):.0f} "
          f"p99={np.percentile(obs, 99):.0f} max={obs.max()}", flush=True)

    def stats(flips: np.ndarray, mask: np.ndarray):
        sub_f = flips[mask]; sub_o = obs[mask]
        if len(sub_f) == 0:
            return None
        return {
            "n": int(len(sub_f)),
            "ever_flipped_pct": 100.0 * float((sub_f > 0).mean()),
            "mean": float(sub_f.mean()),
            "median": float(np.median(sub_f)),
            "p99": float(np.percentile(sub_f, 99)),
            "flips_per_obs": (float(sub_f.sum()) / float(sub_o.sum())) if sub_o.sum() else 0.0,
        }

    strata = [
        ("all",      np.ones(n, dtype=bool)),
        (">=5 obs",  obs >= 5),
        (">=20 obs", obs >= 20),
        (">=100 obs",obs >= 100),
    ]

    methods = [
        ("scovox-Dirichlet", flips_d),
        ("scovox-MV",        flips_m),
        ("scovox-NP",        flips_n),
        ("log_odds-semantic",flips_l),
    ]

    for label, mask in strata:
        n_in_stratum = int(mask.sum())
        if n_in_stratum == 0:
            continue
        print(f"\n  stratum: {label:<10} ({n_in_stratum:,} voxels)", flush=True)
        print(f"    {'method':<22} {'%flipped':>9} {'mean':>7} {'median':>7} "
              f"{'p99':>5} {'flips/obs':>10}", flush=True)
        for mname, flips in methods:
            s = stats(flips, mask)
            if s is None:
                continue
            print(f"    {mname:<22} {s['ever_flipped_pct']:>8.2f}% "
                  f"{s['mean']:>7.2f} {s['median']:>7.0f} {s['p99']:>5.0f} "
                  f"{s['flips_per_obs']:>10.4f}", flush=True)

    # Persist per-voxel arrays for later analysis
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path,
                         obs_count=obs,
                         flips_dirichlet=flips_d,
                         flips_mv=flips_m,
                         flips_np=flips_n,
                         flips_logodds=flips_l)
    print(f"\n  wrote {out_path}", flush=True)


# ----- CLI -----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("anchor", choices=["kitti", "replica", "both"], nargs="?", default="both")
    ap.add_argument("--kitti-scans", type=int, default=100)
    ap.add_argument("--kitti-res", type=float, default=0.10)
    ap.add_argument("--replica-frames", type=int, default=2000)
    ap.add_argument("--replica-res", type=float, default=0.05)
    ap.add_argument("--replica-pix-stride", type=int, default=2)
    ap.add_argument("--out_dir", type=Path,
                    default=EVAL_PKG / "results/b6_full")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.anchor in ("kitti", "both"):
        sim = run_kitti(res=args.kitti_res, n_scans=args.kitti_scans)
        summarise(sim, f"KITTI seq08 polarseg, {args.kitti_scans} scans, "
                       f"res={args.kitti_res} m",
                  args.out_dir / "kitti_seq08_flip_stats.npz")

    if args.anchor in ("replica", "both"):
        sim = run_replica(n_frames=args.replica_frames, res=args.replica_res,
                          pixel_stride=args.replica_pix_stride)
        summarise(sim, f"Replica room0 m2f, {args.replica_frames} frames, "
                       f"res={args.replica_res} m, pix_stride={args.replica_pix_stride}",
                  args.out_dir / "replica_room0_flip_stats.npz")


if __name__ == "__main__":
    main()
