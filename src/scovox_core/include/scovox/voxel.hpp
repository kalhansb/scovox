#pragma once
#include <cstdint>
#include <cstddef>
#include <array>
#include <algorithm>
#include <atomic>
#include <type_traits>

namespace scovox {

constexpr int K_TOP = 2;

/// Default symmetric Dirichlet prior `α₀` applied per underlying class
/// dimension. **Recommended ship value `0.01`** — matches the "Beta starts
/// near zero" behaviour of the legacy code and minimises behavioural drift
/// across the SemBeta / unified-SemDir / split-Beta+Dir substrates, all of
/// which share this default. The launch-file knob `dirichlet_prior` exposes it
/// for the one-shot Jeffreys-prior ablation (`1 / (C + 1)`).
constexpr float kDefaultDirichletPrior = 0.01f;

// Process-wide counters for the four sparse_add branches. All paths are
// instrumented for E5.2 — match/empty are common (hot) and slow the path
// negligibly (one relaxed atomic add per integration). Read at any time via
// dumpEvictStats() or sampled per-frame from scovox_node.
inline std::atomic<uint64_t> g_sparse_match_count{0};  // incoming class matched a tracked slot
inline std::atomic<uint64_t> g_sparse_empty_count{0};  // incoming class filled an empty slot
inline std::atomic<uint64_t> g_sparse_evict_count{0};  // incoming evicted a smaller slot
inline std::atomic<uint64_t> g_sparse_drop_count{0};   // incoming routed to a_unk (no eviction)

struct Voxel {
  float a_occ;
  float a_free;
  float a_unk;
  float sem_cnt[K_TOP];
  uint16_t sem_cls[K_TOP];
  // Truncated signed distance field: distance from surface in metres,
  // clamped to [-sdf_trunc, +sdf_trunc] at integration time. Valid only
  // when tsdf_weight > 0 (zero weight = unobserved). Populated by the
  // fused walk in Map::integrateRay; never shipped over the wire format.
  float tsdf_distance;
  float tsdf_weight;

  inline float p_occ() const {
    const float s = a_occ + a_free;
    return (s > 0.f) ? (a_occ / s) : 0.5f;
  }

  /// Total Dirichlet concentration: sum of all semantic counts + unknown.
  inline float a0() const {
    float s = a_unk;
    for (int i = 0; i < K_TOP; ++i) s += sem_cnt[i];
    return s;
  }
};

static_assert(std::is_trivial_v<Voxel>,
    "Voxel must be trivial for Bonxai pool allocator");
static_assert(offsetof(Voxel, sem_cnt) == 3 * sizeof(float),
    "Voxel layout: sem_cnt must follow a_occ/a_free/a_unk with no padding");
static_assert(offsetof(Voxel, sem_cls) == 3 * sizeof(float) + K_TOP * sizeof(float),
    "Voxel layout: sem_cls must follow sem_cnt with no padding");
// tsdf_distance follows sem_cls + at most (sizeof(float)-1) bytes of natural
// alignment padding (sem_cls is uint16_t, tsdf_distance is float). For odd
// K_TOP the compiler inserts up to 2 bytes of padding here; that's fine.
static_assert(offsetof(Voxel, tsdf_distance)
    >= 3 * sizeof(float) + K_TOP * sizeof(float) + K_TOP * sizeof(uint16_t),
    "Voxel layout: tsdf_distance must come after sem_cls");
static_assert(offsetof(Voxel, tsdf_distance)
    <  3 * sizeof(float) + K_TOP * sizeof(float) + K_TOP * sizeof(uint16_t)
       + sizeof(float),
    "Voxel layout: at most one float of alignment padding before tsdf_distance");
static_assert(offsetof(Voxel, tsdf_weight)
    == offsetof(Voxel, tsdf_distance) + sizeof(float),
    "Voxel layout: tsdf_weight must follow tsdf_distance with no padding");

/// Beta(1,1) prior with zero semantic counts.
/// Note: a_unk and sem_cnt start at 0 (raw evidence). The Dirichlet prior
/// (+1 per category) is applied at query time in semanticEntropy(), not stored.
inline Voxel defaultVoxel() {
  Voxel v{};  // zero-initialise all fields
  v.a_occ  = 1.0f;
  v.a_free = 1.0f;
  return v;
}

inline void sparse_add(float* sem_cnt, uint16_t* sem_cls, uint16_t cls, float inc,
                       float* a_unk = nullptr) {
  for (int i = 0; i < K_TOP; ++i) {
    if (sem_cnt[i] > 0.0f && sem_cls[i] == cls) {
      sem_cnt[i] += inc;
      g_sparse_match_count.fetch_add(1, std::memory_order_relaxed);
      return;
    }
  }
  for (int i = 0; i < K_TOP; ++i) {
    if (sem_cnt[i] <= 0.0f) {
      sem_cls[i] = cls; sem_cnt[i] = inc;
      g_sparse_empty_count.fetch_add(1, std::memory_order_relaxed);
      return;
    }
  }
  int min_i = 0;
  for (int i = 1; i < K_TOP; ++i) { if (sem_cnt[i] < sem_cnt[min_i]) min_i = i; }
  // Posterior-predictive swap test (Dirichlet-Multinomial model).
  //
  // The question: "Should incoming class c (with evidence `inc`) replace
  // tracked class j (with evidence `sem_cnt[min_i]`)?"
  //
  // Under a symmetric Dirichlet prior (α₀ equal for all classes), the
  // posterior predictive probability of class i is:
  //
  //   P(next = i) = (α_i) / (Σα)
  //
  // where α_i = sem_cnt[i] + α₀ for tracked classes. Swapping c into
  // the tracking set is optimal when:
  //
  //   (inc + α₀) / (Σα + inc) > (sem_cnt[min_i] + α₀) / (Σα)
  //
  // For small inc relative to Σα (typical: inc ~ 1-2, Σα ~ 10-50),
  // this simplifies to:
  //
  //   inc > sem_cnt[min_i]
  //
  // This is exactly the Space-Saving criterion (Metwally et al. 2005),
  // which is near-optimal for heavy-hitter tracking (Cormode 2016).
  //
  // Strict `>` (not `>=`) is a deliberate stability choice: a tied
  // newcomer is dropped to a_unk rather than allowed to evict. This
  // prevents thrashing under noisy classifiers emitting equal-weight
  // observations. The trade-off is a first-arrival bias for exact
  // ties — tolerable because exact ties are rare once any meaningful
  // evidence has accumulated.
  //
  // The residual a_unk receives a principled interpretation at query
  // time via the Hutter (2013) adaptive escape mass — see
  // effectiveResidual() in uncertainty.hpp.
  if (inc > sem_cnt[min_i]) {
    if (a_unk) *a_unk += sem_cnt[min_i];  // conserve evicted mass
    sem_cls[min_i] = cls; sem_cnt[min_i] = inc;
    g_sparse_evict_count.fetch_add(1, std::memory_order_relaxed);
  } else {
    if (a_unk) *a_unk += inc;  // conserve dropped mass
    g_sparse_drop_count.fetch_add(1, std::memory_order_relaxed);
  }
}

} // namespace scovox
