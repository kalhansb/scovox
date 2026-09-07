#pragma once

/// @file walker_timers.hpp
/// @brief Compile-time switch for the ray walkers' wall-clock brackets.
///
/// The walkers bracket themselves with `steady_clock::now()` so a caller can
/// read walk time apart from its own per-pixel preparation. That read is a
/// vDSO call and it happens twice per RAY, not per frame — cheap beside a full
/// free-space carve, and not cheap beside a walk that only covers the band. So
/// its price is a share of the frame rather than a constant, and a deployment
/// that never reads `tsdfTimeUs()` / `semdirTimeUs()` should not pay it.
///
/// At 1 (the default) every bracket is exactly the code it has always been.
/// At 0 the clock reads, the subtractions and the accumulator updates all fold
/// away, and both accessors report 0. Either way no grid is ever touched, so
/// the two builds produce byte-identical maps — the switch buys time, never
/// accuracy.
///
/// A 0 build reports 0 µs, which is indistinguishable from "the walk was
/// instant" to anything that just prints the number. That is why the switch is
/// carried in `buildSwitches()` alongside the others: the banner is where a
/// run says which program it is, and a timing column of 0.00 is only readable
/// next to `WALKER_TIMERS=0`.

#include <chrono>
#include <cstdint>

#ifndef SCOVOX_WALKER_TIMERS
#define SCOVOX_WALKER_TIMERS 1
#endif

namespace scovox {

#if SCOVOX_WALKER_TIMERS

using WalkClock = std::chrono::steady_clock;
inline WalkClock::time_point walkNow() noexcept { return WalkClock::now(); }
inline std::int64_t walkNs(WalkClock::time_point a,
                           WalkClock::time_point b) noexcept {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
}

#else

/// Empty stand-in: the brackets keep their shape at the call sites, and the
/// optimiser removes them because `walkNs` is a constant 0 and `x += 0` is not
/// an observable write.
struct WalkClock {
  struct time_point {};
};
inline WalkClock::time_point walkNow() noexcept { return {}; }
inline std::int64_t walkNs(WalkClock::time_point,
                           WalkClock::time_point) noexcept { return 0; }

#endif  // SCOVOX_WALKER_TIMERS

}  // namespace scovox
