#pragma once

/// @file consensus_merge_v4.hpp
/// @brief Per-voxel and per-frame consensus merge for the split Beta/Dirichlet
/// v4 pipeline (BetaVoxel occupancy grid ∥ DirVoxel semantics grid).
///
/// The two grids merge INDEPENDENTLY — occupancy and semantics never cross —
/// which is the whole point of the split (vs. v3's single unified Dirichlet).
///
///   BetaVoxel (occupancy): generalises the Beta-consensus rule
///       a_fused = a_A + a_B − prior
///     to both dimensions:
///       a_occ  ← A.a_occ  + B.a_occ  − occ_prior   (occ_prior  = C·α₀)
///       a_free ← A.a_free + B.a_free − free_prior  (free_prior = α₀)
///
///   DirVoxel (semantics): identical to v3's `mergeSemDir` minus the FREE
///     dimension:
///       other ← A.other + B.other − (C − K_TOP)·α₀
///       top-K slots: union, sum coinciding counts (subtract one duplicated α₀
///         prior per coincidence), sort by count, re-truncate to K_TOP;
///         the remainder's observed evidence flows to OTHER.
///
/// The α₀ subtractions correspond to the Beta v2 rule's `−1`: every dimension
/// carries its prior on both sides of the merge, so we subtract one prior once.
/// Both frames must share `α_0` and `num_classes`; `mergeFramesV4` asserts this.
///
/// TSDF merge is identical to v3 (Curless–Levoy weighted average); reused.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "scovox/beta_voxel.hpp"
#include "scovox/binary_serializer_v4.hpp"
#include "scovox/consensus_merge_v3.hpp"  // mergeTsdfV3 (reused), detail_v3 hashers
#include "scovox/dir_voxel.hpp"
#include "scovox/tsdf_voxel.hpp"

namespace scovox {

/// Per-voxel BetaVoxel merge under the calibrated occupancy prior.
/// `num_classes` / `alpha_0` reconstruct the prior the sender used
/// (`occ_prior = C·α₀`, `free_prior = α₀`), matching `defaultBetaVoxel`.
inline BetaVoxel mergeBeta(const BetaVoxel& a,
                           const BetaVoxel& b,
                           uint16_t         num_classes,
                           float            alpha_0) {
  const float occ_prior  = static_cast<float>(num_classes) * alpha_0;
  const float free_prior = alpha_0;
  BetaVoxel f{};
  f.a_occ  = a.a_occ  + b.a_occ  - occ_prior;
  f.a_free = a.a_free + b.a_free - free_prior;
  return f;
}

/// Per-voxel DirVoxel merge under symmetric Dirichlet prior. Mirrors
/// `mergeSemDir` (consensus_merge_v3.hpp) with no FREE dimension.
inline DirVoxel mergeDir(const DirVoxel& a,
                         const DirVoxel& b,
                         uint16_t        num_classes,
                         float           alpha_0) {
  const float other_prior =
      static_cast<float>(static_cast<int>(num_classes) - K_TOP) * alpha_0;

  DirVoxel f{};
  f.other = a.other + b.other - other_prior;

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

  std::sort(merged, merged + n,
            [](const Entry& x, const Entry& y) { return x.cnt > y.cnt; });

  // Initialise fused slots to prior (matches defaultDirVoxel).
  for (int i = 0; i < K_TOP; ++i) {
    f.cnt[i] = alpha_0;
    f.cls[i] = 0xFFFF;
  }
  const int keep = std::min(n, K_TOP);
  for (int i = 0; i < keep; ++i) {
    f.cls[i] = merged[i].cls;
    f.cnt[i] = merged[i].cnt;
  }
  // Tail past K_TOP: dump observed-evidence (cnt − α₀) into OTHER; the α₀
  // prior stays in the slot — consistent with sparse_add_class eviction.
  for (int i = K_TOP; i < n; ++i) {
    f.other += merged[i].cnt - alpha_0;
  }
  return f;
}

/// Per-frame merge for v4: union of coords from a and b, per-voxel rule at
/// overlaps, copy-through for singletons. TSDF, Beta, and Dir grids merge
/// independently. Both frames must agree on `num_classes` and `alpha_0`.
inline BinarySerializerV4::Frame mergeFramesV4(
    const BinarySerializerV4::Frame& a,
    const BinarySerializerV4::Frame& b)
{
  if (a.num_classes != b.num_classes) {
    throw std::runtime_error(
        "mergeFramesV4: num_classes mismatch ("
        + std::to_string(a.num_classes) + " vs " + std::to_string(b.num_classes) + ")");
  }
  if (std::abs(a.alpha_0 - b.alpha_0) > 1e-7f) {
    throw std::runtime_error("mergeFramesV4: alpha_0 mismatch");
  }

  BinarySerializerV4::Frame fused;
  fused.resolution  = (a.resolution > 0.f) ? a.resolution : b.resolution;
  fused.num_classes = a.num_classes;
  fused.alpha_0     = a.alpha_0;

  // TSDF — reuse the v3 Curless–Levoy merge.
  std::unordered_map<Bonxai::CoordT, TsdfVoxel,
                     detail_v3::CoordHash, detail_v3::CoordEq> tsdf_map;
  for (const auto& d : a.tsdf_deltas) tsdf_map[d.coord] = d.data;
  for (const auto& d : b.tsdf_deltas) {
    auto it = tsdf_map.find(d.coord);
    if (it == tsdf_map.end()) tsdf_map[d.coord] = d.data;
    else                       it->second = mergeTsdfV3(it->second, d.data);
  }
  fused.tsdf_deltas.reserve(tsdf_map.size());
  for (auto& kv : tsdf_map) fused.tsdf_deltas.push_back({kv.first, kv.second});

  // Beta (occupancy) — conjugate Beta merge.
  std::unordered_map<Bonxai::CoordT, BetaVoxel,
                     detail_v3::CoordHash, detail_v3::CoordEq> beta_map;
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
                     detail_v3::CoordHash, detail_v3::CoordEq> dir_map;
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
