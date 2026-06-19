#pragma once

/// @file consensus_merge_v2.hpp
/// @brief Per-voxel and per-frame merge rules for the dSCovox v2 pipeline.
///
/// Pure-C++ free functions; no ROS dependencies. dscovox_node plumbs these
/// in once the full split-grid pipeline lands; until then they're
/// unit-tested standalone here.
///
/// Per Q2 of the 2026-05-08 grill-me design review:
///   TSDF merge:    Curless-Levoy weighted average.
///                    d_fused = (d_A·w_A + d_B·w_B) / (w_A + w_B)
///                    w_fused =  w_A + w_B
///   SemBeta merge: Beta consensus + sparse_add replay.
///                    a_fused  = a_A + a_B - 1   (Beta(1,1) prior shared)
///                    Dirichlet: replay each tracked slot of B through
///                    sparse_add against A's slots.

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "scovox/binary_serializer_v2.hpp"
#include "scovox/sembeta_voxel.hpp"
#include "scovox/tsdf_voxel.hpp"

namespace scovox {

/// Per-voxel TSDF merge: weighted average distance, summed weights.
/// Symmetric in (a, b). w == 0 voxels still merge cleanly (the equation
/// degenerates to copying the non-zero side).
inline TsdfVoxel mergeTsdf(const TsdfVoxel& a, const TsdfVoxel& b) {
  const float w = a.weight + b.weight;
  if (w <= 0.f) return {0.0f, 0.0f};
  return {(a.distance * a.weight + b.distance * b.weight) / w, w};
}

/// Per-voxel SemBeta merge: Beta consensus + sparse_add replay.
/// Both robots carry the same Beta(1,1) prior; the consensus rule
/// `a_fused = a_A + a_B - 1` removes the duplicated prior so the result
/// reflects only accumulated evidence. a_unk sums (no shared prior).
/// Dirichlet slots: start from a's slots, then sparse_add each of b's.
inline SemBetaVoxel mergeSemBeta(const SemBetaVoxel& a, const SemBetaVoxel& b) {
  SemBetaVoxel f;
  f.a_occ  = a.a_occ  + b.a_occ  - 1.0f;
  f.a_free = a.a_free + b.a_free - 1.0f;
  f.a_unk  = a.a_unk  + b.a_unk;
  for (int i = 0; i < K_TOP; ++i) {
    f.sem_cnt[i] = a.sem_cnt[i];
    f.sem_cls[i] = a.sem_cls[i];
  }
  for (int i = 0; i < K_TOP; ++i) {
    if (b.sem_cnt[i] > 0.f && b.sem_cls[i] != 0xFFFF) {
      sparse_add(f.sem_cnt, f.sem_cls, b.sem_cls[i], b.sem_cnt[i], &f.a_unk);
    }
  }
  return f;
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

/// Per-frame merge: produces a fused frame containing the union of coords
/// from a and b. For coords present in both, applies the per-voxel merge
/// rule; for coords in only one, copies through unchanged.
inline BinarySerializerV2::Frame mergeFrames(
    const BinarySerializerV2::Frame& a,
    const BinarySerializerV2::Frame& b)
{
  BinarySerializerV2::Frame fused;
  fused.resolution = (a.resolution > 0.f) ? a.resolution : b.resolution;

  // TSDF
  std::unordered_map<Bonxai::CoordT, TsdfVoxel,
                     detail::CoordHash, detail::CoordEq> tsdf_map;
  for (const auto& d : a.tsdf_deltas) tsdf_map[d.coord] = d.data;
  for (const auto& d : b.tsdf_deltas) {
    auto it = tsdf_map.find(d.coord);
    if (it == tsdf_map.end()) tsdf_map[d.coord] = d.data;
    else                       it->second = mergeTsdf(it->second, d.data);
  }
  fused.tsdf_deltas.reserve(tsdf_map.size());
  for (auto& kv : tsdf_map) {
    fused.tsdf_deltas.push_back({kv.first, kv.second});
  }

  // SemBeta
  std::unordered_map<Bonxai::CoordT, SemBetaVoxel,
                     detail::CoordHash, detail::CoordEq> sb_map;
  for (const auto& d : a.sembeta_deltas) sb_map[d.coord] = d.data;
  for (const auto& d : b.sembeta_deltas) {
    auto it = sb_map.find(d.coord);
    if (it == sb_map.end()) sb_map[d.coord] = d.data;
    else                     it->second = mergeSemBeta(it->second, d.data);
  }
  fused.sembeta_deltas.reserve(sb_map.size());
  for (auto& kv : sb_map) {
    fused.sembeta_deltas.push_back({kv.first, kv.second});
  }

  return fused;
}

}  // namespace scovox
