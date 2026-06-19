#pragma once

/// @file consensus_merge_v3.hpp
/// @brief Per-voxel and per-frame consensus merge for the unified-Dirichlet
/// v3 pipeline.
///
/// TSDF merge is unchanged from v2 (Curless–Levoy). SemDir merge generalises
/// the Beta-consensus formula `a_fused = a_A + a_B − 1` to all K_TOP + 2
/// Dirichlet dimensions:
///   alpha_free  ← A.alpha_free  + B.alpha_free  − α_0
///   alpha_other ← A.alpha_other + B.alpha_other − (C − K_TOP − 1) · α_0
///   top-K class slots: build union, sum coinciding counts (subtracting α_0
///     once per duplicated prior), sort by count, re-truncate to K_TOP;
///     remainder flows to OTHER.
///
/// The α_0 subtractions correspond exactly to the Beta v2 rule's `−1`: each
/// dimension carries its prior on both sides of the merge, so we subtract
/// one prior once. Both sides must share the same `α_0` and `num_classes`
/// (otherwise consensus is ill-defined). The merge functions assert this.

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "scovox/binary_serializer_v3.hpp"
#include "scovox/semdir_voxel.hpp"
#include "scovox/tsdf_voxel.hpp"

namespace scovox {

/// Per-voxel TSDF merge: weighted average distance, summed weights.
/// Identical to v2; duplicated here so v3 has no v2 dependency.
inline TsdfVoxel mergeTsdfV3(const TsdfVoxel& a, const TsdfVoxel& b) {
  const float w = a.weight + b.weight;
  if (w <= 0.f) return {0.0f, 0.0f};
  return {(a.distance * a.weight + b.distance * b.weight) / w, w};
}

/// Per-voxel SemDir merge under symmetric Dirichlet prior `(α_0, …)`.
/// `num_classes` is `C` from the wire header. Both robots' frames must
/// share these — the consensus node verifies at frame level.
inline SemDirVoxel mergeSemDir(const SemDirVoxel& a,
                               const SemDirVoxel& b,
                               uint16_t           num_classes,
                               float              alpha_0) {
  const float other_prior =
      static_cast<float>(static_cast<int>(num_classes) - K_TOP) * alpha_0;

  SemDirVoxel f{};
  f.alpha_free  = a.alpha_free  + b.alpha_free  - alpha_0;
  f.alpha_other = a.alpha_other + b.alpha_other - other_prior;

  // Build a small union dict of (class -> count). At most 2·K_TOP entries.
  // Empty slots (cls == 0xFFFF) carry only the per-slot prior α_0 in cnt[]
  // and contribute nothing useful here — skipped.
  struct Entry { uint16_t cls; float cnt; };
  Entry merged[2 * K_TOP];
  int n = 0;
  auto upsert = [&](uint16_t cls, float cnt) {
    for (int i = 0; i < n; ++i) {
      if (merged[i].cls == cls) {
        // Coinciding class on both sides: sum, subtract one duplicated α_0
        // prior so the slot's "prior + evidence" structure is preserved.
        merged[i].cnt += cnt - alpha_0;
        return;
      }
    }
    merged[n].cls = cls;
    merged[n].cnt = cnt;
    ++n;
  };
  for (int i = 0; i < K_TOP; ++i) {
    if (a.cls[i] != 0xFFFF) upsert(a.cls[i], a.cnt[i]);
  }
  for (int i = 0; i < K_TOP; ++i) {
    if (b.cls[i] != 0xFFFF) upsert(b.cls[i], b.cnt[i]);
  }

  // Sort by count descending — top-K_TOP stay; tail flows to OTHER.
  std::sort(merged, merged + n,
            [](const Entry& x, const Entry& y) { return x.cnt > y.cnt; });

  // Initialise fused slots to the prior (matches defaultSemDirVoxel).
  for (int i = 0; i < K_TOP; ++i) {
    f.cnt[i] = alpha_0;
    f.cls[i] = 0xFFFF;
  }
  const int keep = std::min(n, K_TOP);
  for (int i = 0; i < keep; ++i) {
    f.cls[i] = merged[i].cls;
    f.cnt[i] = merged[i].cnt;
  }
  // Anything past K_TOP gets dumped into OTHER (observed-evidence portion
  // only; the α_0 prior stays in the slot — fully consistent with the
  // eviction routing in `sparse_add_unified`).
  for (int i = K_TOP; i < n; ++i) {
    f.alpha_other += merged[i].cnt - alpha_0;
  }
  return f;
}

// v3-scoped subnamespace so this header can be included alongside
// consensus_merge_v2.hpp without colliding with its identically-named
// CoordHash / CoordEq in `scovox::detail`. (Both headers existed in
// isolation pre-Step 8; dscovox_node now pulls in both.)
namespace detail_v3 {

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

}  // namespace detail_v3

/// Per-frame merge: produces a fused frame containing the union of coords
/// from a and b. For coords present in both, applies the per-voxel rule;
/// for coords in only one, copies through unchanged.
inline BinarySerializerV3::Frame mergeFramesV3(
    const BinarySerializerV3::Frame& a,
    const BinarySerializerV3::Frame& b)
{
  if (a.num_classes != b.num_classes) {
    throw std::runtime_error(
        "mergeFramesV3: num_classes mismatch ("
        + std::to_string(a.num_classes) + " vs " + std::to_string(b.num_classes) + ")");
  }
  if (std::abs(a.alpha_0 - b.alpha_0) > 1e-7f) {
    throw std::runtime_error("mergeFramesV3: alpha_0 mismatch");
  }

  BinarySerializerV3::Frame fused;
  fused.resolution  = (a.resolution > 0.f) ? a.resolution : b.resolution;
  fused.num_classes = a.num_classes;
  fused.alpha_0     = a.alpha_0;

  // TSDF
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

  // SemDir
  std::unordered_map<Bonxai::CoordT, SemDirVoxel,
                     detail_v3::CoordHash, detail_v3::CoordEq> sd_map;
  for (const auto& d : a.semdir_deltas) sd_map[d.coord] = d.data;
  for (const auto& d : b.semdir_deltas) {
    auto it = sd_map.find(d.coord);
    if (it == sd_map.end()) sd_map[d.coord] = d.data;
    else                     it->second = mergeSemDir(it->second, d.data,
                                                      a.num_classes, a.alpha_0);
  }
  fused.semdir_deltas.reserve(sd_map.size());
  for (auto& kv : sd_map) fused.semdir_deltas.push_back({kv.first, kv.second});

  return fused;
}

}  // namespace scovox
