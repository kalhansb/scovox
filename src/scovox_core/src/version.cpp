/// @file version.cpp
/// @brief Assembles the compiled-in build-switch line declared in version.hpp.
///
/// Lives in a .cpp rather than the header for one reason: reading a switch's
/// value requires including the header that declares its default, and the two
/// heaviest headers in the library are both on that list — `sem_split_map.hpp`
/// for `SCOVOX_DEPOSIT_TRACE`, and `scovox_map_split.hpp`, which pulls it in
/// again along with the mesher, for `SCOVOX_WALK_MARGIN_VOX`. Every
/// translation unit in the tree would pay for them to answer a question that
/// is asked once, at startup. Here the cost is paid by exactly one TU.
///
/// Including the owning headers — rather than restating the `#ifndef` defaults
/// locally — is the whole point. A local copy of a default would report what
/// this file believes instead of what the rest of the library compiled with,
/// and would go stale silently the first time a default moved.

#include "scovox/version.hpp"

#include "scovox/beta_voxel.hpp"        // SCOVOX_BETA_U16,
                                        // SCOVOX_BETA_U16_SCALE
#include "scovox/dir_voxel.hpp"         // SCOVOX_TRACK_QMAX, SCOVOX_TRACK_NHIT
#include "scovox/e0_counters.hpp"       // SCOVOX_E0_COUNTERS
#include "scovox/scovox_map_split.hpp"  // SCOVOX_WALK_MARGIN_VOX
#include "scovox/sem_split_map.hpp"     // SCOVOX_DEPOSIT_TRACE
#include "scovox/voxel.hpp"             // SCOVOX_K_TOP,
                                        // SCOVOX_SPARSE_BRANCH_COUNTERS
#include "scovox/walker_timers.hpp"     // SCOVOX_WALKER_TIMERS

// Two-step stringification: the inner macro must see the argument already
// expanded, or `SCOVOX_STR(SCOVOX_K_TOP)` yields the literal "SCOVOX_K_TOP".
#define SCOVOX_STR2(x) #x
#define SCOVOX_STR(x) SCOVOX_STR2(x)

#ifdef NDEBUG
#define SCOVOX_NDEBUG_STR "1"
#else
#define SCOVOX_NDEBUG_STR "0"
#endif

namespace scovox {

const char* buildSwitches() {
  // Adjacent string literals are concatenated by the compiler, so the whole
  // line is a single constant in .rodata — no formatting, no allocation, and
  // nothing that can fail at runtime.
  return "scovox " SCOVOX_VERSION_STRING
         " K_TOP=" SCOVOX_STR(SCOVOX_K_TOP)
         " BETA_U16=" SCOVOX_STR(SCOVOX_BETA_U16)
         " BETA_U16_SCALE=" SCOVOX_STR(SCOVOX_BETA_U16_SCALE)
         " TRACK_QMAX=" SCOVOX_STR(SCOVOX_TRACK_QMAX)
         " TRACK_NHIT=" SCOVOX_STR(SCOVOX_TRACK_NHIT)
         " DEPOSIT_TRACE=" SCOVOX_STR(SCOVOX_DEPOSIT_TRACE)
         " E0_COUNTERS=" SCOVOX_STR(SCOVOX_E0_COUNTERS)
         " WALKER_TIMERS=" SCOVOX_STR(SCOVOX_WALKER_TIMERS)
         " SPARSE_BRANCH_COUNTERS=" SCOVOX_STR(SCOVOX_SPARSE_BRANCH_COUNTERS)
         " WALK_MARGIN_VOX=" SCOVOX_STR(SCOVOX_WALK_MARGIN_VOX)
         " NDEBUG=" SCOVOX_NDEBUG_STR;
}

}  // namespace scovox
