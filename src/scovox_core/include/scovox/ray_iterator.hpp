#pragma once
/// @file ray_iterator.hpp
/// @brief DDA voxel ray traversal — header-only, zero ROS dependencies.

#include <algorithm>
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

} // namespace scovox
