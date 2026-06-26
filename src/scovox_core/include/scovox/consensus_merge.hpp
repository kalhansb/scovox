#pragma once

/// @file consensus_merge.hpp
/// @brief Per-voxel and per-frame consensus merge for the split Beta/Dirichlet
/// pipeline (BetaVoxel occupancy grid ∥ DirVoxel semantics grid).
///
/// The two grids merge INDEPENDENTLY — occupancy and semantics never cross —
/// which is the whole point of the split (vs. a single unified Dirichlet).
///
///   BetaVoxel (occupancy): generalises the Beta-consensus rule
///       a_fused = a_A + a_B − prior
///     to both dimensions, with the SYMMETRIC Beta(1,1) prior (p_occ=0.5):
///       a_occ  ← A.a_occ  + B.a_occ  − occ_prior   (occ_prior  = kBetaOccPrior  = 1)
///       a_free ← A.a_free + B.a_free − free_prior  (free_prior = kBetaFreePrior = 1)
///
///   DirVoxel (semantics): the slot-reconciling Dirichlet rule with no FREE
///     dimension:
///       other ← A.other + B.other − (C − K_TOP)·α₀
///       top-K slots: union, sum coinciding counts (subtract one duplicated α₀
///         prior per coincidence), sort by count, re-truncate to K_TOP;
///         the remainder's observed evidence flows to OTHER.
///
/// The α₀ subtractions correspond to the Beta rule's `−1`: every dimension
/// carries its prior on both sides of the merge, so we subtract one prior once.
/// Both frames must share `α_0` and `num_classes`; `mergeFrames` asserts this.
///
/// TSDF merge is the Curless–Levoy weighted average.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "scovox/beta_voxel.hpp"
#include "scovox/binary_serializer.hpp"
#include "scovox/dir_voxel.hpp"
#include "scovox/tsdf_voxel.hpp"

namespace scovox {

// =====================================================================
// TSDF merge + coordinate hashers. TSDF merge is wire-agnostic and shared
// by mergeFrames below.
// =====================================================================

/// Per-voxel TSDF merge: weighted average distance, summed weights
/// (Curless–Levoy). Depends only on TsdfVoxel.
inline TsdfVoxel mergeTsdf(const TsdfVoxel& a, const TsdfVoxel& b) {
  const float w = a.weight + b.weight;
  if (w <= 0.f) return {0.0f, 0.0f};
  return {(a.distance * a.weight + b.distance * b.weight) / w, w};
}

namespace detail {

struct CoordHash {
  size_t operator()(const Bonxai::CoordT& c) const noexcept {
    return std::hash<int64_t>{}(
        (int64_t(c.x) * 73856093) ^
        (int64_t(c.y) * 19349669) ^
        (int64_t(c.z) * 83492791));
  }
};

struct CoordEq {
  bool operator()(const Bonxai::CoordT& a, const Bonxai::CoordT& b) const noexcept {
    return a.x == b.x && a.y == b.y && a.z == b.z;
  }
};

}  // namespace detail

/// Per-voxel BetaVoxel merge under the symmetric Beta(1,1) occupancy prior.
/// The `num_classes` / `alpha_0` params are retained for call-site symmetry with
/// `mergeDir` but are UNUSED for occupancy: the prior is now the decoupled
/// constant `kBetaOccPrior` = `kBetaFreePrior` = 1.0 (docs/occupancy_prior.md),
/// not the old calibrated `C·α₀`. Sender and receiver share this compile-time
/// constant, so the prior-subtraction below stays consistent across nodes.
inline BetaVoxel mergeBeta(const BetaVoxel& a,
                           const BetaVoxel& b,
                           uint16_t         num_classes,
                           float            alpha_0) {
  (void)num_classes; (void)alpha_0;  // occupancy prior is the symmetric constant
  const float occ_prior  = kBetaOccPrior;
  const float free_prior = kBetaFreePrior;
  BetaVoxel f{};
  // Floor each fused α at its prior. For valid inputs (each source at-or-above
  // prior) a+b−prior ≥ prior already, so this is a no-op; it only guards a
  // corrupt/below-prior source from producing a negative α and an out-of-[0,1]
  // p_occ that would poison downstream entropy/EIG.
  f.a_occ  = std::max(occ_prior,  a.a_occ  + b.a_occ  - occ_prior);
  f.a_free = std::max(free_prior, a.a_free + b.a_free - free_prior);
  return f;
}

/// Per-voxel DirVoxel merge under symmetric Dirichlet prior — the
/// slot-reconciling Dirichlet rule with no FREE dimension.
inline DirVoxel mergeDir(const DirVoxel& a,
                         const DirVoxel& b,
                         uint16_t        num_classes,
                         float           alpha_0) {
  // Clamp at 0 exactly as defaultDirVoxel does: when num_classes == K_TOP there
  // are no residual dimensions, and a num_classes < K_TOP header would make
  // other_prior negative — turning the subtraction below into prior INFLATION
  // on every fold.
  const float other_prior = std::max(
      0.f, static_cast<float>(static_cast<int>(num_classes) - K_TOP) * alpha_0);

  DirVoxel f{};
  f.other = std::max(other_prior, a.other + b.other - other_prior);

  // Union dict of (class -> count); at most 2·K_TOP entries. Empty slots
  // (cls == 0xFFFF) carry only the per-slot α₀ prior and are skipped.
  struct Entry { uint16_t cls; float cnt; };
  Entry merged[2 * K_TOP];
  int n = 0;
  auto upsert = [&](uint16_t cls, float cnt) {
    for (int i = 0; i < n; ++i) {
      if (merged[i].cls == cls) {
        merged[i].cnt += cnt - alpha_0;  // sum, subtract one duplicated prior
        return;
      }
    }
    merged[n].cls = cls;
    merged[n].cnt = cnt;
    ++n;
  };
  for (int i = 0; i < K_TOP; ++i) if (a.cls[i] != 0xFFFF) upsert(a.cls[i], a.cnt[i]);
  for (int i = 0; i < K_TOP; ++i) if (b.cls[i] != 0xFFFF) upsert(b.cls[i], b.cnt[i]);

  // Deterministic, fold-order-invariant ordering: count desc, then class id
  // asc as a tie-break. std::sort is NOT stable, so without the secondary key
  // two classes with equal counts at the K_TOP truncation boundary would be
  // kept-or-dumped depending on source iteration order, making the fused slots
  // (and dominantClass) nondeterministic across runs/rehashes. NOTE: this makes
  // the merge DETERMINISTIC but not fully order-INDEPENDENT — a class dumped to
  // OTHER in one pairwise fold cannot climb back into a slot in a later fold.
  // True commutativity requires accumulating every source's evidence before a
  // single truncation; callers needing that must fold in a fixed source order.
  std::sort(merged, merged + n,
            [](const Entry& x, const Entry& y) {
              return x.cnt != y.cnt ? x.cnt > y.cnt : x.cls < y.cls;
            });

  // Initialise fused slots to prior (matches defaultDirVoxel).
  for (int i = 0; i < K_TOP; ++i) {
    f.cnt[i] = alpha_0;
    f.cls[i] = 0xFFFF;
  }
  const int keep = std::min(n, K_TOP);
  for (int i = 0; i < keep; ++i) {
    f.cls[i] = merged[i].cls;
    f.cnt[i] = std::max(alpha_0, merged[i].cnt);
  }
  // Tail past K_TOP: dump observed-evidence (cnt − α₀) into OTHER; the α₀
  // prior stays in the slot — consistent with sparse_add_class eviction.
  for (int i = K_TOP; i < n; ++i) {
    f.other += std::max(0.f, merged[i].cnt - alpha_0);
  }
  return f;
}

/// Per-frame merge: union of coords from a and b, per-voxel rule at
/// overlaps, copy-through for singletons. TSDF, Beta, and Dir grids merge
/// independently. Both frames must agree on `num_classes` and `alpha_0`.
inline BinarySerializer::Frame mergeFrames(
    const BinarySerializer::Frame& a,
    const BinarySerializer::Frame& b)
{
  if (a.num_classes != b.num_classes) {
    throw std::runtime_error(
        "mergeFrames: num_classes mismatch ("
        + std::to_string(a.num_classes) + " vs " + std::to_string(b.num_classes) + ")");
  }
  if (a.num_classes < static_cast<uint16_t>(K_TOP)) {
    throw std::runtime_error(
        "mergeFrames: num_classes (" + std::to_string(a.num_classes)
        + ") < K_TOP (" + std::to_string(K_TOP) + ")");
  }
  if (!std::isfinite(a.alpha_0) || a.alpha_0 <= 0.f) {
    throw std::runtime_error("mergeFrames: alpha_0 must be finite and > 0");
  }
  if (std::abs(a.alpha_0 - b.alpha_0) > 1e-7f) {
    throw std::runtime_error("mergeFrames: alpha_0 mismatch");
  }
  // Both frames must share a voxel lattice: coords are integer lattice indices,
  // so a resolution mismatch makes union-by-coord misplace every b voxel. A
  // zero resolution means "empty side" (the non-zero side is taken below).
  if (a.resolution > 0.f && b.resolution > 0.f &&
      std::abs(a.resolution - b.resolution) > 1e-6f) {
    throw std::runtime_error(
        "mergeFrames: resolution mismatch ("
        + std::to_string(a.resolution) + " vs " + std::to_string(b.resolution) + ")");
  }

  BinarySerializer::Frame fused;
  fused.resolution  = (a.resolution > 0.f) ? a.resolution : b.resolution;
  fused.num_classes = a.num_classes;
  fused.alpha_0     = a.alpha_0;

  // TSDF — Curless–Levoy merge.
  std::unordered_map<Bonxai::CoordT, TsdfVoxel,
                     detail::CoordHash, detail::CoordEq> tsdf_map;
  for (const auto& d : a.tsdf_deltas) tsdf_map[d.coord] = d.data;
  for (const auto& d : b.tsdf_deltas) {
    auto it = tsdf_map.find(d.coord);
    if (it == tsdf_map.end()) tsdf_map[d.coord] = d.data;
    else                       it->second = mergeTsdf(it->second, d.data);
  }
  fused.tsdf_deltas.reserve(tsdf_map.size());
  for (auto& kv : tsdf_map) fused.tsdf_deltas.push_back({kv.first, kv.second});

  // Beta (occupancy) — conjugate Beta merge.
  std::unordered_map<Bonxai::CoordT, BetaVoxel,
                     detail::CoordHash, detail::CoordEq> beta_map;
  for (const auto& d : a.beta_deltas) beta_map[d.coord] = d.data;
  for (const auto& d : b.beta_deltas) {
    auto it = beta_map.find(d.coord);
    if (it == beta_map.end()) beta_map[d.coord] = d.data;
    else                       it->second = mergeBeta(it->second, d.data,
                                                      a.num_classes, a.alpha_0);
  }
  fused.beta_deltas.reserve(beta_map.size());
  for (auto& kv : beta_map) fused.beta_deltas.push_back({kv.first, kv.second});

  // Dir (semantics) — slot-reconciling Dirichlet merge, independent of Beta.
  std::unordered_map<Bonxai::CoordT, DirVoxel,
                     detail::CoordHash, detail::CoordEq> dir_map;
  for (const auto& d : a.dir_deltas) dir_map[d.coord] = d.data;
  for (const auto& d : b.dir_deltas) {
    auto it = dir_map.find(d.coord);
    if (it == dir_map.end()) dir_map[d.coord] = d.data;
    else                     it->second = mergeDir(it->second, d.data,
                                                   a.num_classes, a.alpha_0);
  }
  fused.dir_deltas.reserve(dir_map.size());
  for (auto& kv : dir_map) fused.dir_deltas.push_back({kv.first, kv.second});

  return fused;
}

}  // namespace scovox
