#pragma once
/// @file ray_iterator.hpp
/// @brief Voxel ray traversal — header-only, zero ROS dependencies.
///
/// Two traversals, both vendored verbatim from Bonxai (MPL-2.0):
///   RayIterator      — integer Bresenham, 26-connected, SKIPS crossed voxels
///   ExactRayIterator — Amanatides & Woo, 6-connected, skips none
/// Identical include/exclude semantics, so they are interchangeable at a
/// call site. Selected per instance via envExactRay(); default Bresenham.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>

#include <Eigen/Core>
#include <bonxai/bonxai.hpp>

namespace scovox {

template <class Functor>
inline void RayIterator(const Bonxai::CoordT& key_origin,
                        const Bonxai::CoordT& key_end,
                        const Functor& func) {
  if (key_origin == key_end) return;
  if (!func(key_origin)) return;

  Bonxai::CoordT error{0, 0, 0};
  Bonxai::CoordT coord = key_origin;
  Bonxai::CoordT delta = (key_end - coord);
  const Bonxai::CoordT step{delta.x < 0 ? -1 : 1,
                            delta.y < 0 ? -1 : 1,
                            delta.z < 0 ? -1 : 1};

  delta = {delta.x < 0 ? -delta.x : delta.x,
           delta.y < 0 ? -delta.y : delta.y,
           delta.z < 0 ? -delta.z : delta.z};

  const int maxc = std::max(std::max(delta.x, delta.y), delta.z);
  if (maxc <= 0) return;

  for (int i = 0; i < maxc - 1; ++i) {
    error = error + delta;
    if ((error.x << 1) >= maxc) { coord.x += step.x; error.x -= maxc; }
    if ((error.y << 1) >= maxc) { coord.y += step.y; error.y -= maxc; }
    if ((error.z << 1) >= maxc) { coord.z += step.z; error.z -= maxc; }
    if (!func(coord)) return;
  }
}

/// Exact voxel traversal (Amanatides & Woo, "A Fast Voxel Traversal Algorithm
/// for Ray Tracing", 1987). Visits every voxel crossed by the segment from
/// `from` (continuous world coordinates, inside voxel `coord_from`) to the
/// CENTRE of voxel `coord_to`; `coord_from` is included, `coord_to` is
/// excluded. Voxels are the half-open boxes
/// [coord * resolution, (coord + 1) * resolution) — the convention
/// Bonxai::PosToCoord's floor() already establishes.
///
/// Vendored VERBATIM from Bonxai `bonxai_map/include/bonxai_map/
/// probabilistic_map.hpp` (MPL-2.0, Copyright Contributors to the Bonxai
/// Project) so it can be diffed against upstream. Only the enclosing namespace
/// differs: upstream declares it in `Bonxai`, this copy in `scovox`, matching
/// the sibling `RayIterator` above — which is itself a verbatim copy of
/// upstream's Bresenham `RayIterator`.
///
/// Include/exclude semantics are IDENTICAL to that Bresenham sibling (start
/// included, end excluded), so it is a drop-in at every call site: callers
/// that already visit `key_end` explicitly after the loop keep working
/// unchanged.
///
/// What DOES change is the visit SET. Bresenham steps 26-connected (up to
/// three axes per step) and therefore skips voxels the continuous segment
/// genuinely crosses; this steps 6-connected (one axis per step) and skips
/// none. Upstream documents the consequence for the Bresenham path: "it skips
/// part of the crossed voxels, so carving is weaker and depends on the
/// direction of the ray relative to the grid axes."
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

/// Per-instance A/B latch for the exact traversal, read ONCE from the
/// environment. Default OFF: absent `SCOVOX_EXACT_RAY`, every walker keeps the
/// Bresenham `RayIterator` and behaviour is byte-identical to before this
/// function existed. Same latch contract as ScovoxMapSplit's
/// SCOVOX_DISABLE_FAR_SKIP / _FAR_CARVE — latched per instance at
/// construction, not per process, so one binary can host an approximate map
/// and an exact map side by side.
inline bool envExactRay() noexcept {
  const char* e = std::getenv("SCOVOX_EXACT_RAY");
  return e && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0');
}

} // namespace scovox
