#pragma once

/// @file
/// @brief 8-byte TSDF-only voxel for the split-grid SCovox refactor.
///
/// Holds Curless–Levoy state for a single voxel of the TSDF grid:
/// truncated signed distance + accumulated weight. **No semantics, no Beta.**
///
/// Used by `scovox::TsdfMap` (band-only DDA, SLIM-VDB-equivalent integration).
/// The semantic and occupancy state lives in a parallel
/// `Bonxai::VoxelGrid<scovox::SemBetaVoxel>`; see `sembeta_voxel.hpp`.

#include <cstddef>
#include <type_traits>

namespace scovox {

/// 8-byte TSDF voxel. Bonxai's pool allocator zero-initialises new leaf
/// blocks, so `weight == 0` is the unobserved sentinel.
struct TsdfVoxel {
  /// Truncated signed distance in metres, clamped to [-sdf_trunc, +sdf_trunc]
  /// at integration time. Sign convention: positive in front of the surface,
  /// negative behind (matches SLIM-VDB / VDBFusion). Valid iff `weight > 0`.
  float distance;

  /// Accumulated weight from Curless–Levoy running average. Zero indicates
  /// the voxel has never been observed (default-constructed leaf-block state).
  float weight;
};

static_assert(sizeof(TsdfVoxel) == 8,
    "TsdfVoxel must be exactly 8 bytes — the paper claim and the Bonxai "
    "memory footprint both depend on this.");
static_assert(std::is_trivial_v<TsdfVoxel>,
    "TsdfVoxel must be trivial for Bonxai's pool allocator (zero-init).");
static_assert(std::is_standard_layout_v<TsdfVoxel>,
    "TsdfVoxel must have standard layout for the wire format's "
    "byte-for-byte reinterpret_cast emit path.");

/// Default-constructed TSDF voxel: distance=0, weight=0 (= unobserved).
/// Matches Bonxai's pool zero-init, so existing leaf-block construction
/// already produces this state. Provided for explicit-construction sites
/// (e.g. unit tests) and as the canonical "what does an unobserved voxel
/// look like?" reference.
inline TsdfVoxel defaultTsdfVoxel() noexcept { return {0.0f, 0.0f}; }

}  // namespace scovox
