#pragma once
/// @file ray_iterator.hpp
/// @brief Voxel ray traversal — header-only, zero ROS dependencies.
///
/// ONE traversal: Amanatides & Woo, "A Fast Voxel Traversal Algorithm for Ray
/// Tracing" (1987). 6-connected, steps one axis at a time, and visits every
/// voxel the continuous segment crosses.
///
/// This replaced an integer-Bresenham sibling that stepped up to three axes at
/// once and therefore SKIPPED voxels the segment genuinely crosses — 1.84x
/// fewer voxels per ray, and on 82% of random oblique rays it visited a voxel
/// the true segment never enters, so the two sets were not nested either way.
/// Upstream states the consequence for that traversal: "carving is weaker and
/// depends on the direction of the ray relative to the grid axes." Free-space
/// carving is the measurement most exposed to it — the skipped voxels are
/// exactly the free-space evidence that never gets deposited — so the
/// approximate walk is gone rather than selectable. It costs ~46% more walk
/// time, well under the 1.84x voxel ratio.

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
  int32_t        step[3];
  double         t_max[3];
  double         t_delta[3];
  // parametrized along the unnormalized segment: t = 1 at the endpoint center
  for (int i = 0; i < 3; i++) {
    if (delta[i] != 0.0) {
      const double inv_delta = 1.0 / delta[i];
      step[i] = (delta[i] > 0.0) ? 1 : -1;
      const double boundary = (coord[i] + (step[i] > 0 ? 1 : 0)) * resolution;
      t_max[i] = (boundary - from[i]) * inv_delta;
      t_delta[i] = resolution * std::abs(inv_delta);
    } else {
      step[i] = 0;
      t_max[i] = std::numeric_limits<double>::infinity();
      t_delta[i] = std::numeric_limits<double>::infinity();
    }
  }
  while (true) {
    const int axis = (t_max[0] < t_max[1]) ? ((t_max[0] < t_max[2]) ? 0 : 2)
                                           : ((t_max[1] < t_max[2]) ? 1 : 2);
    if (t_max[axis] > 1.0) {
      return;  // no boundary crossing left before the end of the segment
    }
    coord[axis] += step[axis];
    t_max[axis] += t_delta[axis];
    if (coord == coord_to) {
      return;  // the endpoint voxel is excluded
    }
    if (!func(coord)) {
      return;
    }
  }
}

} // namespace scovox
