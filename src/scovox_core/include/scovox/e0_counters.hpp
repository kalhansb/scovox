/// @file
/// @brief E0 instrumentation — the history quantities a finished map cannot
/// answer.  NEW_EXPERIMENTS_PLAN.md §3.
///
/// Everything here is compiled out unless `-DSCOVOX_E0_COUNTERS=1`, so the
/// shipped build's bytes are unchanged and E0's own gate (a) — instrumented
/// and archived builds must produce byte-identical dumps — is a statement
/// about added *observation*, never about added behaviour.  Nothing in this
/// header is read by the mapper: every function is a sink.
///
/// WHY THIS EXISTS WHEN voxel.hpp ALREADY HAS COUNTERS.  `g_sparse_*_count`
/// (voxel.hpp:39-42) already counts four of the five branches, and the ROS
/// node already reads them (scovox_node.cpp:860-863).  Two gaps make them
/// insufficient for E0, and both are traps rather than omissions:
///
///   1. `g_sparse_drop_count` is incremented from TWO branches — the sentinel
///      guard at dir_voxel.hpp:250 (outcome 0, class id 0xFFFF, which never
///      reached the comparator) and the real drop-to-OTHER at :459 (outcome 4,
///      which did).  E0's `n_admit_tests` is outcome 3 + outcome 4.  Reading
///      it off `g_sparse_drop_count` would silently fold in arrivals that were
///      never tested, in an unknown proportion.  The counters below keep the
///      five outcomes apart and report the sentinel count so the conflation is
///      measurable rather than assumed to be zero.
///   2. They are scalars.  `evictions_by_class` and `n_argmax_contested` are
///      not derivable from any scalar, and are the two quantities E3 has to
///      move.
///
/// The existing counters are still used, as a CROSS-CHECK: `verify()` reports
/// both sets, and their disagreement is itself a finding.
#ifndef SCOVOX_E0_COUNTERS_HPP
#define SCOVOX_E0_COUNTERS_HPP

#if SCOVOX_E0_COUNTERS

#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "scovox/voxel.hpp"  // K_TOP + the g_sparse_* counters cross-checked below

namespace scovox {
namespace e0 {

/// One process-wide accumulator.  The replay is single-threaded
/// (`OMP_NUM_THREADS=1`, replay_scene.sh:38), but the mapper is not required
/// to be, so the aggregate paths take a mutex.  That is affordable *because*
/// this build is never the one under test for anything but counts: gate (a)
/// compares its dumps to the archived build's, not its wall clock.
struct Counters {
  // --- deposits offered to the slot store -------------------------------
  // One per class with p > 0 per ray under `soft`; one per ray under `hard`
  // (sem_split_map.cpp:130-168).  Mass that never reached a slot at all — the
  // `d->other += class_share` early returns for an absent or all-zero
  // observation — is NOT counted, because no deposit was offered.
  uint64_t n_deposits_total = 0;

  // --- outcome histogram, dir_voxel.hpp:233-236 -------------------------
  //   0 = sentinel routed to OTHER (class id 0xFFFF; never tested)
  //   1 = matched a tracked slot
  //   2 = filled an empty slot
  //   3 = evicted the weakest slot
  //   4 = dropped to OTHER (tested and refused)
  uint64_t n_outcome[5] = {0, 0, 0, 0, 0};

  // --- the eviction diagnostic ------------------------------------------
  // Which class lost its slot, and which took it.  Keyed (evicted, admitted),
  // sparse because most of a 14x14 grid is empty and a dense one invites the
  // reader to believe the zeroes were measured.
  std::unordered_map<uint32_t, uint64_t> evictions_by_class;

  // --- contention, defined on what the comparator was asked -------------
  // Voxels that received at least one deposit whose ARGMAX class was held in
  // no slot at the moment it arrived.  Not "voxels that ever saw a (K+1)-th
  // class": under `soft` every class with p > 0 arrives on every ray, so that
  // definition is true of nearly every voxel on its first ray at K=2 and
  // cannot move (plan §1).
  std::unordered_set<uint64_t> argmax_contested_voxels;
  uint64_t n_argmax_arrivals_uncovered = 0;  // the same test, per arrival
  uint64_t n_argmax_arrivals           = 0;  // denominator

  std::mutex mu;
};

inline Counters& counters() {
  static Counters c;
  return c;
}

inline uint64_t packVoxel(int32_t x, int32_t y, int32_t z) {
  // 21 bits each, biased — the grids in play are far inside +/-1e6 voxels.
  const uint64_t ux = static_cast<uint64_t>(static_cast<uint32_t>(x + 1048576)) & 0x1FFFFF;
  const uint64_t uy = static_cast<uint64_t>(static_cast<uint32_t>(y + 1048576)) & 0x1FFFFF;
  const uint64_t uz = static_cast<uint64_t>(static_cast<uint32_t>(z + 1048576)) & 0x1FFFFF;
  return (ux << 42) | (uy << 21) | uz;
}

/// Called once per deposit offered, after `sparse_add_class` has reported its
/// branch.  `victim` is the class that lost its slot (valid only for outcome
/// 3); `admitted` is the arriving class.
inline void onDeposit(uint8_t outcome, uint16_t admitted, uint16_t victim) {
  Counters& c = counters();
  std::lock_guard<std::mutex> g(c.mu);
  ++c.n_deposits_total;
  if (outcome < 5) ++c.n_outcome[outcome];
  if (outcome == 3) {
    const uint32_t key = (static_cast<uint32_t>(victim) << 16) | admitted;
    ++c.evictions_by_class[key];
  }
}

/// Called once per `dirichletUpdate` that had a usable observation, BEFORE any
/// deposit from it lands — the slot state tested is the one the arrival saw.
inline void onArgmaxArrival(int32_t x, int32_t y, int32_t z, bool covered) {
  Counters& c = counters();
  std::lock_guard<std::mutex> g(c.mu);
  ++c.n_argmax_arrivals;
  if (!covered) {
    ++c.n_argmax_arrivals_uncovered;
    c.argmax_contested_voxels.insert(packVoxel(x, y, z));
  }
}

inline void reset() {
  Counters& c = counters();
  std::lock_guard<std::mutex> g(c.mu);
  c.n_deposits_total = 0;
  for (int i = 0; i < 5; ++i) c.n_outcome[i] = 0;
  c.evictions_by_class.clear();
  c.argmax_contested_voxels.clear();
  c.n_argmax_arrivals_uncovered = 0;
  c.n_argmax_arrivals = 0;
}

/// Write the counters as JSON.  `class_names` may be empty, in which case the
/// eviction matrix is keyed by integer id.
///
/// The rate E0-P1 grades is written out explicitly rather than left for a
/// reader to divide, and the DENOMINATOR is named in the file: a rate whose
/// denominator has to be guessed is how "eviction fraction 0.0000" became a
/// claim about the mapper instead of a claim about `evid`.
inline bool writeJson(const std::string& path,
                      const std::vector<std::string>& class_names = {}) {
  Counters& c = counters();
  std::lock_guard<std::mutex> g(c.mu);
  FILE* f = std::fopen(path.c_str(), "w");
  if (!f) return false;
  const double den = c.n_deposits_total ? static_cast<double>(c.n_deposits_total) : 1.0;
  std::fprintf(f, "{\n");
  std::fprintf(f, "  \"n_deposits_total\": %llu,\n",
               static_cast<unsigned long long>(c.n_deposits_total));
  std::fprintf(f, "  \"n_outcome\": {\"sentinel_0\": %llu, \"match_1\": %llu, "
                  "\"empty_2\": %llu, \"evict_3\": %llu, \"drop_4\": %llu},\n",
               (unsigned long long)c.n_outcome[0], (unsigned long long)c.n_outcome[1],
               (unsigned long long)c.n_outcome[2], (unsigned long long)c.n_outcome[3],
               (unsigned long long)c.n_outcome[4]);
  std::fprintf(f, "  \"n_evictions_total\": %llu,\n",
               (unsigned long long)c.n_outcome[3]);
  std::fprintf(f, "  \"n_admit_tests\": %llu,\n",
               (unsigned long long)(c.n_outcome[3] + c.n_outcome[4]));
  std::fprintf(f, "  \"eviction_rate\": %.12g,\n", c.n_outcome[3] / den);
  std::fprintf(f, "  \"eviction_rate_denominator\": \"n_deposits_total\",\n");
  std::fprintf(f, "  \"admit_test_rate\": %.12g,\n",
               (c.n_outcome[3] + c.n_outcome[4]) / den);
  std::fprintf(f, "  \"refusal_rate_given_test\": %.12g,\n",
               (c.n_outcome[3] + c.n_outcome[4])
                   ? static_cast<double>(c.n_outcome[4]) /
                     static_cast<double>(c.n_outcome[3] + c.n_outcome[4])
                   : 0.0);
  std::fprintf(f, "  \"n_argmax_contested_voxels\": %llu,\n",
               (unsigned long long)c.argmax_contested_voxels.size());
  std::fprintf(f, "  \"n_argmax_arrivals\": %llu,\n",
               (unsigned long long)c.n_argmax_arrivals);
  std::fprintf(f, "  \"n_argmax_arrivals_uncovered\": %llu,\n",
               (unsigned long long)c.n_argmax_arrivals_uncovered);

  // Cross-check against the pre-existing voxel.hpp counters.  Agreement is
  // not assumed: it is written down per run so a divergence is visible.
  std::fprintf(f, "  \"crosscheck_g_sparse\": {\"match\": %llu, \"empty\": %llu, "
                  "\"evict\": %llu, \"drop_plus_sentinel\": %llu},\n",
               (unsigned long long)g_sparse_match_count.load(std::memory_order_relaxed),
               (unsigned long long)g_sparse_empty_count.load(std::memory_order_relaxed),
               (unsigned long long)g_sparse_evict_count.load(std::memory_order_relaxed),
               (unsigned long long)g_sparse_drop_count.load(std::memory_order_relaxed));

  std::fprintf(f, "  \"evictions_by_class\": [");
  bool first = true;
  for (const auto& kv : c.evictions_by_class) {
    const uint16_t victim   = static_cast<uint16_t>(kv.first >> 16);
    const uint16_t admitted = static_cast<uint16_t>(kv.first & 0xFFFF);
    if (!first) std::fprintf(f, ",");
    first = false;
    std::fprintf(f, "\n    {\"evicted\": %u, \"admitted\": %u, \"n\": %llu",
                 victim, admitted, (unsigned long long)kv.second);
    if (victim < class_names.size() && admitted < class_names.size()) {
      std::fprintf(f, ", \"evicted_name\": \"%s\", \"admitted_name\": \"%s\"",
                   class_names[victim].c_str(), class_names[admitted].c_str());
    }
    std::fprintf(f, "}");
  }
  std::fprintf(f, "\n  ]\n}\n");
  std::fclose(f);
  return true;
}

}  // namespace e0
}  // namespace scovox

#define SCOVOX_E0_ON_DEPOSIT(outcome, admitted, victim) \
  ::scovox::e0::onDeposit((outcome), (admitted), (victim))
#define SCOVOX_E0_ON_ARGMAX(x, y, z, covered) \
  ::scovox::e0::onArgmaxArrival((x), (y), (z), (covered))

#else  // !SCOVOX_E0_COUNTERS

#define SCOVOX_E0_ON_DEPOSIT(outcome, admitted, victim) ((void)0)
#define SCOVOX_E0_ON_ARGMAX(x, y, z, covered)           ((void)0)

#endif  // SCOVOX_E0_COUNTERS
#endif  // SCOVOX_E0_COUNTERS_HPP
