#pragma once
#include <cstdint>
#include <cstddef>
#include <array>
#include <algorithm>
#include <atomic>
#include <type_traits>

namespace scovox {

/// Number of tracked class slots per voxel. **Ship value 2** — the paper /
/// production configuration; every default build is byte-identical to the
/// hard-coded constant this replaced.
///
/// Overridable at *build* time only (`-DSCOVOX_K_TOP=n`), for the sufficiency
/// sweep, which needs K ∈ {1,2,3,full} as four separate builds. It is a
/// struct-layout constant: every translation unit in the workspace must see
/// the same value, so a sweep build must pass the flag to `colcon build` as a
/// whole and install into its own base — never mix objects across values.
#ifndef SCOVOX_K_TOP
#define SCOVOX_K_TOP 2
#endif
constexpr int K_TOP = SCOVOX_K_TOP;
static_assert(K_TOP >= 1, "K_TOP must be >= 1");

/// The "no class here" class id. It is a value no real taxonomy uses; the
/// neighbouring sentinel in the same family is `SemSplitMap::kNoHitProbs`.
///
/// WHICH ARRAYS ACTUALLY USE IT — the three substrates do not agree, and
/// assuming they do is the way to misread an empty slot as class 0:
///
///   `DirVoxel::cls[]`        LIVE marker. Written on eviction, tested to find
///                            a free slot, and carried across the wire.
///   `SemBetaVoxel::sem_cls[]` INITIAL value only, set by
///                            `defaultSemBetaVoxel`. Nothing restores or tests
///                            it afterwards.
///   `Voxel::sem_cls[]`       DOES NOT USE IT. The legacy fused substrate
///                            zero-initialises the class ids and marks a slot
///                            empty with `sem_cnt[i] <= 0`, so its idle slots
///                            read as class 0, not as this sentinel.
///
/// It is also the "no class" return of `dominantClass` and of the mesh /
/// marching-cubes label joins.
///
/// Not a policy knob and not overridable: it is baked into the wire (the codec
/// maps it to 0xFF when the taxonomy fits in u8) and `sparse_add_class`
/// refuses to admit an observation whose class id equals it, precisely so a
/// real id can never be mistaken for "empty".
///
/// `inline` is load-bearing, not decoration: a namespace-scope `constexpr` is
/// implicitly `const` and therefore internal-linkage, so header-inline
/// functions that odr-use it (`labelMesh` and `extractMesh` bind it to the
/// `const uint16_t&` of a `push_back`) would each refer to a different entity
/// per translation unit — an ODR violation no compiler is required to report.
inline constexpr uint16_t kEmptySlot = 0xFFFF;

/// Default symmetric Dirichlet prior `α₀` applied per underlying class
/// dimension. **Recommended ship value `0.01`** — matches the "Beta starts
/// near zero" behaviour of the legacy code and minimises behavioural drift
/// across the SemBeta / unified-SemDir / split-Beta+Dir substrates, all of
/// which share this default. The launch-file knob `dirichlet_prior` exposes it
/// for the one-shot Jeffreys-prior ablation (`1 / (C + 1)`).
constexpr float kDefaultDirichletPrior = 0.01f;

// Process-wide counters for the four sparse_add branches. All paths are
// instrumented for E5.2. Read at any time via dumpEvictStats() or sampled
// per-frame from scovox_node.
//
// COMPILED OUT BY DEFAULT, and the default is the design choice.
//
// These are `std::atomic`, and on x86-64 every atomic read-modify-write is a
// `lock`-prefixed instruction whatever the memory order — `relaxed` relaxes the
// compiler's reordering, not the CPU's bus. It costs tens of cycles and it
// drains the store buffer, so it does not overlap with the work either side of
// it. An earlier comment here called that negligible "per integration", which
// was true when a deposit happened once per hit ray. That is not the shape of
// the path any more: the semantic band deposits once per NON-ZERO CLASS per
// BAND VOXEL, so a single ray issues a double-digit number of these and a frame
// issues them by the million. Nothing on any code path reads the counters to
// make a decision, and they are members of no voxel struct, so they reach no
// dump — the deposit is bit-identical either way.
//
// The declarations stay unconditional so every reader still compiles; only the
// increments are switched. E0 builds force this on: dumpEvictStats() writes
// these four as a cross-check against its own outcome tally, and a cross-check
// that silently reads zero is worse than no cross-check at all.
//
// External deposit-bench harnesses that reset and read these globals directly
// must be built with SCOVOX_SPARSE_BRANCH_COUNTERS=1. Set it for the whole
// build, never per translation unit: sparse_add / sparse_add_class are inline,
// so a mixed setting is an ODR violation rather than a partial measurement.
#ifndef SCOVOX_SPARSE_BRANCH_COUNTERS
#  if defined(SCOVOX_E0_COUNTERS) && SCOVOX_E0_COUNTERS
#    define SCOVOX_SPARSE_BRANCH_COUNTERS 1
#  else
#    define SCOVOX_SPARSE_BRANCH_COUNTERS 0
#  endif
#endif

#if SCOVOX_SPARSE_BRANCH_COUNTERS
#  define SCOVOX_SPARSE_BUMP(c_) (c_).fetch_add(1, std::memory_order_relaxed)
#else
#  define SCOVOX_SPARSE_BUMP(c_) ((void)0)
#endif

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
      SCOVOX_SPARSE_BUMP(g_sparse_match_count);
      return;
    }
  }
  for (int i = 0; i < K_TOP; ++i) {
    if (sem_cnt[i] <= 0.0f) {
      sem_cls[i] = cls; sem_cnt[i] = inc;
      SCOVOX_SPARSE_BUMP(g_sparse_empty_count);
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
    SCOVOX_SPARSE_BUMP(g_sparse_evict_count);
  } else {
    if (a_unk) *a_unk += inc;  // conserve dropped mass
    SCOVOX_SPARSE_BUMP(g_sparse_drop_count);
  }
}

} // namespace scovox
