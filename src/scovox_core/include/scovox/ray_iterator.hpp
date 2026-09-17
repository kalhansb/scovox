#pragma once
/// @file ray_iterator.hpp
/// @brief Voxel ray traversal — header-only, zero ROS dependencies.
///
/// ONE traversal: Amanatides & Woo, "A Fast Voxel Traversal Algorithm for Ray
/// Tracing" (1987). 6-connected, steps one axis at a time, and visits every
/// voxel the continuous segment crosses.
///
/// Visiting EVERY crossed voxel is the requirement, not an optimisation, and it
/// is why there is only one traversal here rather than a fast/exact choice.
/// Free-space carving deposits its evidence per visited voxel, so any walk that
/// advances more than one axis in a step skips voxels the segment really
/// crosses, and the evidence for exactly those voxels is never deposited. The
/// damage is not uniform either: which voxels get skipped depends on the ray's
/// direction relative to the grid axes, so carving strength becomes a function
/// of viewing angle. An approximate walk is therefore not offered as an option.
/// The exactness is paid for in walk time.

#include <cmath>
#include <limits>

#include <Eigen/Core>
#include <bonxai/bonxai.hpp>

namespace scovox {

/// Visits every voxel crossed by the segment from `from` (continuous world
/// coordinates, inside voxel `coord_from`) to the CENTRE of voxel `coord_to`;
/// `coord_from` is included, `coord_to` is excluded. Voxels are the half-open
/// boxes [coord * resolution, (coord + 1) * resolution) — the convention
/// Bonxai::PosToCoord's floor() already establishes.
///
/// Vendored VERBATIM from Bonxai `bonxai_map/include/bonxai_map/
/// probabilistic_map.hpp` (MPL-2.0, Copyright Contributors to the Bonxai
/// Project) so it can be diffed against upstream. Only the enclosing namespace
/// differs: upstream declares it in `Bonxai`, this copy in `scovox`.
///
/// Caveat carried over from upstream: the segment aims at the CENTRE of
/// `coord_to`, not at a caller-supplied continuous end point. Where the true
/// ray exits through some other part of that voxel, the direction differs by
/// at most half a voxel diagonal over the walk length. `coord_to` is excluded
/// either way, so this can only perturb which voxels are visited immediately
/// before it.
template <class Functor>
inline void ExactRayIterator(const Eigen::Vector3d& from,
                             const Bonxai::CoordT&  coord_from,
                             const Bonxai::CoordT&  coord_to,
                             double                 resolution,
                             const Functor&         func) {
  if (coord_from == coord_to) return;
  if (!func(coord_from)) return;

  const Eigen::Vector3d to((coord_to.x + 0.5) * resolution,
                           (coord_to.y + 0.5) * resolution,
                           (coord_to.z + 0.5) * resolution);
  const Eigen::Vector3d delta = to - from;

  Bonxai::CoordT coord = coord_from;
  int32_t  step_x, step_y, step_z;
  double   tmax_x, tmax_y, tmax_z;
  double   tdel_x, tdel_y, tdel_z;
#define SCOVOX_DDA_AXIS_SETUP(I, C, S, M, D)                                  \
  if (delta[I] != 0.0) {                                                      \
    const double inv_delta = 1.0 / delta[I];                                  \
    S = (delta[I] > 0.0) ? 1 : -1;                                            \
    const double boundary = (C + (S > 0 ? 1 : 0)) * resolution;               \
    M = (boundary - from[I]) * inv_delta;                                     \
    D = resolution * std::abs(inv_delta);                                     \
  } else {                                                                    \
    S = 0;                                                                    \
    M = std::numeric_limits<double>::infinity();                              \
    D = std::numeric_limits<double>::infinity();                              \
  }
  SCOVOX_DDA_AXIS_SETUP(0, coord.x, step_x, tmax_x, tdel_x)
  SCOVOX_DDA_AXIS_SETUP(1, coord.y, step_y, tmax_y, tdel_y)
  SCOVOX_DDA_AXIS_SETUP(2, coord.z, step_z, tmax_z, tdel_z)
#undef SCOVOX_DDA_AXIS_SETUP
  while (true) {
    if (tmax_x < tmax_y) {
      if (tmax_x < tmax_z) { if (tmax_x > 1.0) return; coord.x += step_x; tmax_x += tdel_x; }
      else                 { if (tmax_z > 1.0) return; coord.z += step_z; tmax_z += tdel_z; }
    } else {
      if (tmax_y < tmax_z) { if (tmax_y > 1.0) return; coord.y += step_y; tmax_y += tdel_y; }
      else                 { if (tmax_z > 1.0) return; coord.z += step_z; tmax_z += tdel_z; }
    }
    if (coord == coord_to) return;
    if (!func(coord)) return;
  }
}

} // namespace scovox
