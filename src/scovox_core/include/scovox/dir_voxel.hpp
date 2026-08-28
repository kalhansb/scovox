#pragma once

/// @file dir_voxel.hpp
/// @brief 16-byte sparse-Dirichlet semantics voxel for the split Beta/Dirichlet
/// refactor.
///
/// Holds *only* the semantic state — a Dirichlet over `{top-K classes, OTHER}`
/// — with no occupancy. Occupancy lives in the parallel
/// `Bonxai::VoxelGrid<BetaVoxel>` (see `beta_voxel.hpp`).
///
/// Relationship to the unified `SemDirVoxel`: this is `SemDirVoxel` with the
/// `FREE` dimension removed (`FREE` is now the Beta grid's `a_free`). What
/// remains is the *occupied-class* Dirichlet: the top-K class slots plus the
/// `OTHER` bucket that lumps the `C − K_TOP` collapsed class dimensions and
/// any evicted slot mass. De-unifying this way means class evidence no longer
/// feeds back into the occupancy marginal — occupancy and semantics are
/// conditionally independent given the observation (the SemBeta two-stream
/// model), which is the intended meaning of "split Beta and Dirichlet".
///
/// Layout (K_TOP = 2):
///   offset 0:   other  (float, 4 B)  — lumped OTHER / evicted mass
///   offset 4:   cnt[0] (float, 4 B)  — α for top-K slot 0
///   offset 8:   cnt[1] (float, 4 B)  — α for top-K slot 1
///   offset 12:  cls[0] (uint16, 2 B) — class id at slot 0 (0xFFFF = empty)
///   offset 14:  cls[1] (uint16, 2 B) — class id at slot 1
///   total: 16 B at K_TOP=2.
///
/// Mass conservation: `sparse_add_class` preserves the strict invariant
///   Δ(other + Σ cnt) == Σ Δ inputs
/// — every increment lands somewhere (matched slot / empty slot / evicted-to-
/// OTHER / dropped-to-OTHER), never lost. This is the same eviction-to-OTHER
/// discipline as `sparse_add_unified` in semdir_map.cpp, ported to the
/// occupancy-free Dirichlet; it is *not* the legacy `voxel.hpp::sparse_add`
/// with its `≥ 0` slack.
///
/// One bounded exception, from evidence saturation
/// (`SemSplitMap::applyDirSaturation`): the rescale multiplies an EMPTY slot's
/// α₀ placeholder down to k·α₀ (only FILLED slots are floored back to α₀ —
/// flooring empty ones would re-inflate `s_class` past the cap). The next
/// empty-slot fill overwrites that placeholder with `α₀ + inc`, so the fill's
/// Δ exceeds `inc` by α₀·(1−k) ≤ α₀. This merely restores the eroded prior,
/// is bounded by α₀ per fill, and needs no downstream guard.

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "scovox/voxel.hpp"  // K_TOP + g_sparse_*_count counters

/// Per-slot confidence track for the max-probability eviction comparator.
/// Build-time only (`-DSCOVOX_TRACK_QMAX=1`), following the `SCOVOX_K_TOP`
/// one-install-tree-per-variant convention, because it changes
/// `sizeof(DirVoxel)` (4 + 6·K_TOP → 4 + 8·K_TOP; 16 B → 20 B at K_TOP=2) and
/// we must be able to quote the shipped build's bytes unchanged.
///
/// Never serialized: binary_serializer.hpp emits only `other`, `cnt[]` and
/// `cls[]`, so a `SCOVOX_TRACK_QMAX=1` sender stays wire-compatible with a
/// `=0` receiver. The comparator only changes *which* class wins a slot, and
/// that outcome is already fully visible in the `cls`/`cnt` that do go out.
#ifndef SCOVOX_TRACK_QMAX
#define SCOVOX_TRACK_QMAX 0
#endif

// Per-slot DEPOSIT COUNT. `cnt[i]` is accumulated mass, not a tally: one
// confident visit and twenty unsure ones can leave the same `cnt`. `nhit[i]`
// separates those two, and `cnt[i]/nhit[i]` is then the slot's mean deposit
// strength -- a scale-free quantity, unlike `cnt`, which grows with how often
// the voxel happened to be looked at. Off by default; the wire format never
// carries it.
#ifndef SCOVOX_TRACK_NHIT
#define SCOVOX_TRACK_NHIT 0
#endif

/// VICTIM axis, build-time only (`-DSCOVOX_VICTIM_MEAN=1`), off by default so
/// the shipped build's bytes and behaviour are unchanged.  Selects the victim
/// slot by least MEAN deposit strength `(cnt[i] - alpha_0) / nhit[i]` rather
/// than by least accumulated evidence.  Requires SCOVOX_TRACK_NHIT: the rule
/// reads `nhit[]`, and with the field absent it degrades to the shipped `cnt`
/// search at run time rather than reading uninitialised memory.
#ifndef SCOVOX_VICTIM_MEAN
#define SCOVOX_VICTIM_MEAN 0
#endif

/// ADMIT axis, build-time only (`-DSCOVOX_ADMIT_NORM=1`), off by default.
/// Replaces the admit test `inc > (cnt[m] - alpha_0)` with the scale-free
/// `inc > (cnt[m] - alpha_0) / nhit[m]` -- strength against strength rather
/// than one deposit against a slot's whole history.  Also requires
/// SCOVOX_TRACK_NHIT, and degrades to the shipped test when `nhit` is null.
/// Independent of SCOVOX_VICTIM_MEAN; the two compose.
#ifndef SCOVOX_ADMIT_NORM
#define SCOVOX_ADMIT_NORM 0
#endif

// What an evicting newcomer's counter is set to. 0 is the shipped reset (fresh
// counter, victim's evidence to OTHER); 1 is canonical Space-Saving (the
// newcomer inherits the victim's counter, which is what makes the algorithm's
// no-under-estimate guarantee hold). Rationale and the full table are at the
// eviction branch itself. Off by default; costs no bytes either way.
#ifndef SCOVOX_EVICT_INHERIT
#define SCOVOX_EVICT_INHERIT 0
#endif

namespace scovox {

/// Per-voxel occupied-class Dirichlet state. `cls[i] == 0xFFFF` marks an empty
/// slot; `cnt[i]` then holds the per-slot prior `α₀` (set by
/// `defaultDirVoxel()` — never zero, to keep closed-form variance valid).
struct DirVoxel {
  /// Lumped pseudo-counts for the `C − K_TOP` classes outside the top-K slots
  /// plus any evicted slot evidence. Conserved by `sparse_add_class`.
  float    other;

  /// Top-K class slots, by accumulated `cnt[i]`. `cls[i] == 0xFFFF` is the
  /// empty-slot sentinel; an empty slot's `cnt[i]` holds the prior `α₀`.
  float    cnt[K_TOP];
  uint16_t cls[K_TOP];

#if SCOVOX_TRACK_QMAX
  /// Running maximum of the *deposit probability* each slot has ever seen, as
  /// u16 fixed point (`round(q · 65535)`). Read only by the confidence
  /// eviction comparator in `sparse_add_class`; zero-init means "no deposit
  /// yet", which is only ever the state of an EMPTY slot.
  uint16_t qmax[K_TOP];
#endif

#if SCOVOX_TRACK_NHIT
  /// Number of deposits this slot has absorbed since it was last (re)filled.
  /// Saturates at 65535 rather than wrapping -- a wrapped tally would read as
  /// a nearly-unseen slot. Reset to 1 on fill and on eviction, so it always
  /// counts deposits of the class currently in `cls[i]`, never a predecessor's.
  uint16_t nhit[K_TOP];
#endif

  /// Total class (occupied-semantic) evidence: `other + Σ cnt`. The analogue
  /// of `SemDirVoxel::s_occ()` (which additionally folds in occupancy mass);
  /// here it is purely semantic.
  inline float s_class() const noexcept {
    float s = other;
    for (int i = 0; i < K_TOP; ++i) s += cnt[i];
    return s;
  }
};

// Layout invariants. K_TOP=2 (production / paper default):
//   sizeof == 4 (other) + 4·K_TOP (cnt) + 2·K_TOP (cls) = 4 + 12 = 16.
// General K_TOP: 4 + 6·K_TOP, rounded up to 4-byte alignment for the trailing
// uint16_t pair.
/// 6 B per slot (4 cnt + 2 cls), or 8 B when the qmax confidence track is
/// compiled in.
constexpr std::size_t kDirSlotBytes =
    6u + (SCOVOX_TRACK_QMAX ? 2u : 0u) + (SCOVOX_TRACK_NHIT ? 2u : 0u);
constexpr std::size_t kDirExpectedSize =
    ((4u + kDirSlotBytes * static_cast<std::size_t>(K_TOP) + 3u) / 4u) * 4u;
static_assert(sizeof(DirVoxel) == kDirExpectedSize,
    "DirVoxel size mismatch — layout is 4 B fixed + 6 B per K_TOP slot "
    "(8 B with SCOVOX_TRACK_QMAX) rounded up to 4-byte alignment.");
static_assert(SCOVOX_TRACK_QMAX || SCOVOX_TRACK_NHIT || K_TOP != 2 || sizeof(DirVoxel) == 16,
    "Production K_TOP=2 invariant: DirVoxel must be exactly 16 B "
    "(SemDirVoxel 20 B minus the 4 B FREE dimension moved to BetaVoxel).");
static_assert(!SCOVOX_TRACK_QMAX || SCOVOX_TRACK_NHIT || K_TOP != 2 || sizeof(DirVoxel) == 20,
    "K_TOP=2 with SCOVOX_TRACK_QMAX: 16 B + 2 B per slot confidence = 20 B.");
static_assert(!SCOVOX_VICTIM_MEAN || SCOVOX_TRACK_NHIT,
    "SCOVOX_VICTIM_MEAN reads nhit[]; build with -DSCOVOX_TRACK_NHIT=1.");
static_assert(!SCOVOX_ADMIT_NORM || SCOVOX_TRACK_NHIT,
    "SCOVOX_ADMIT_NORM reads nhit[]; build with -DSCOVOX_TRACK_NHIT=1.");
static_assert(std::is_trivial_v<DirVoxel>,
    "DirVoxel must be trivial for Bonxai's pool allocator (zero-init).");
static_assert(std::is_standard_layout_v<DirVoxel>,
    "DirVoxel must have standard layout for byte-for-byte wire emit.");
static_assert(offsetof(DirVoxel, cnt) == offsetof(DirVoxel, other) + sizeof(float),
    "DirVoxel layout: cnt[] must immediately follow other.");
static_assert(offsetof(DirVoxel, cls) == offsetof(DirVoxel, cnt) + K_TOP * sizeof(float),
    "DirVoxel layout: cls[] must immediately follow cnt[K_TOP] with no padding.");

/// Default symmetric prior. The top-K slots and OTHER carry the same per-dim
/// prior `α₀` that the unified `SemDirVoxel` uses, minus the FREE dimension:
///   - each `cnt[i] = α₀`  (K_TOP slot placeholders)
///   - `other = (C − K_TOP) · α₀`  (the collapsed out-of-K dimensions)
/// Total class prior `= C · α₀`, equal to `SemDirVoxel::s_occ()` at prior and
/// to `BetaVoxel`'s `a_occ` prior (`C·α₀`) — keeping the split consistent with
/// the live path at the prior.
inline DirVoxel defaultDirVoxel(uint16_t num_classes = 14,
                                float    alpha_0     = kDefaultDirichletPrior) noexcept {
  DirVoxel v{};                     // zero-init
  const int residual_dims = static_cast<int>(num_classes) - K_TOP;
  v.other = (residual_dims > 0) ? (residual_dims * alpha_0) : 0.f;
  for (int i = 0; i < K_TOP; ++i) {
    v.cnt[i] = alpha_0;             // per-dim prior on each top-K slot
    v.cls[i] = 0xFFFF;             // empty-slot sentinels
  }
  return v;
}

/// Heavy-hitter sparse-add into the occupied-class Dirichlet, parametrised by
/// the per-dim prior `α₀`. Routes `inc` into one of the top-K_TOP class slots
/// (Space-Saving / Metwally 2005) or the OTHER bucket — never lost. Preserves
/// the strict mass invariant `Δ(other + Σcnt) == inc`.
///
/// Direct port of `sparse_add_unified` (semdir_map.cpp), with `alpha_other`
/// renamed `other` and no FREE interaction (FREE is in the Beta grid).
/// `q` is the deposit's own class probability in [0,1] (for a Dirichlet update
/// that is `inc / class_share`). It is *not* evidence and never changes what is
/// deposited — it only feeds the optional confidence eviction comparator, and
/// is ignored unless `qmax != nullptr` (which requires SCOVOX_TRACK_QMAX).
inline void sparse_add_class(float*    cnt,
                             uint16_t* cls,
                             uint16_t  c,
                             float     inc,
                             float*    other,
                             float     alpha_0,
                             float     q    = -1.0f,
                             uint16_t* qmax = nullptr,
                             bool      evict_by_conf = true,
                             uint16_t* nhit = nullptr,
                             uint8_t*  outcome = nullptr) {
  // Fixed-point form of the incoming confidence, computed once.
  // RECORDING the per-slot confidence and USING it as the eviction comparator
  // are separate decisions. `track` governs the former, `use_q` the latter, so
  // a run may keep evidence eviction (the shipped rule) and still leave a
  // readable confidence trail behind for an offline readout to consult.
  // `evict_by_conf` defaults to true so every pre-existing caller — which only
  // ever passed a non-null `qmax` when it wanted confidence eviction — keeps
  // its exact previous behaviour.
  const bool     track = (qmax != nullptr) && (q >= 0.0f);
  const bool     use_q = track && evict_by_conf;
  const uint16_t q_fx  = !track            ? uint16_t{0}
                       : (q >= 1.0f)       ? uint16_t{65535}
                                           : static_cast<uint16_t>(q * 65535.0f + 0.5f);
  // Which of the four branches below consumed this deposit. An eviction rule
  // can only be judged on the deposits it REJECTS as well as the ones it keeps,
  // and neither `cnt` nor `nhit` remembers a rejection -- dropped mass is
  // indistinguishable from mass that was never offered. Reporting the branch is
  // the only way an offline trace can see the losing arrivals at all.
  //   0 = sentinel routed to OTHER, 1 = matched a slot, 2 = filled an empty
  //   slot, 3 = evicted the weakest slot, 4 = dropped to OTHER.
  const auto say = [outcome](uint8_t code) { if (outcome) *outcome = code; };
  const auto bump = [nhit](int i) {
    if (nhit && nhit[i] != 65535) ++nhit[i];
  };
  // (0) Sentinel guard. `0xFFFF` is the empty-slot marker in `cls[]`, so a real
  // observation of class id 0xFFFF (e.g. a 65535-class taxonomy, or a classifier
  // whose argmax index hits 0xFFFF) must NOT be written into a slot: it would
  // fill `cls[i] = 0xFFFF` with real mass yet still read as EMPTY, so the next
  // add re-fills the slot from scratch (losing the prior inc) and isPriorDir /
  // dominantClass mis-treat it as unfilled — breaking the strict invariant
  // Δ(other + Σcnt) == inc. Route the (untrackable) sentinel class straight to
  // OTHER, which both conserves mass and keeps every slot's sentinel meaning intact.
  if (c == 0xFFFF) {
    *other += inc;
    g_sparse_drop_count.fetch_add(1, std::memory_order_relaxed);
    say(0);
    return;
  }
  // (1) Match — incoming class already tracked in a slot.
  for (int i = 0; i < K_TOP; ++i) {
    if (cls[i] != 0xFFFF && cls[i] == c) {
      cnt[i] += inc;
      if (track && q_fx > qmax[i]) qmax[i] = q_fx;
      bump(i);
      g_sparse_match_count.fetch_add(1, std::memory_order_relaxed);
      say(1);
      return;
    }
  }
  // (2) Empty slot available — fill it. The slot's α₀ prior stays; add on top.
  for (int i = 0; i < K_TOP; ++i) {
    if (cls[i] == 0xFFFF) {
      cls[i] = c;
      cnt[i] = alpha_0 + inc;
      if (track) qmax[i] = q_fx;
      if (nhit) nhit[i] = 1;
      g_sparse_empty_count.fetch_add(1, std::memory_order_relaxed);
      say(2);
      return;
    }
  }
  // (3) All slots filled — evict-or-drop using observed-evidence (cnt − α₀) as
  // the comparison key (posterior-predictive Space-Saving; see voxel.hpp).
  int min_i = 0;
#if SCOVOX_VICTIM_MEAN
  // VICTIM axis, build-time only (`-DSCOVOX_VICTIM_MEAN=1`), off by default so
  // the shipped build's bytes are unchanged.  Requires SCOVOX_TRACK_NHIT.
  //
  // The shipped search picks the slot holding the least ACCUMULATED evidence.
  // That key is not scale-free: a slot filled early has been offered hundreds
  // of deposits and a slot filled late has been offered a handful, so the
  // shipped rule reads "arrived recently" nearly as much as it reads "is
  // weak".  Dividing by the number of deposits that built the slot asks the
  // scale-free question instead -- how strong was this class EACH time it was
  // seen -- which is the question a heavy-hitter victim search is meant to ask.
  //
  // Neither operand works alone.  Offline replay of a recorded deposit stream
  // over one contiguous 29^3 block of scene 016 (1536 GT-labelled voxels,
  // arrival ceiling 0.9831, noise floor 1/1536 = 0.00065) ranks the quotient
  // FIRST and both of its operands below the incumbent, under the confidence
  // admit at K_TOP=2:
  //
  //     victim      retention   acc_all
  //     mean          0.9225     0.8919   <- this rule
  //     qmax          0.9167     0.8854      (SCOVOX_VICTIM_QMAX)
  //     cnt           0.9141     0.8757      (shipped)
  //     nhit          0.9128     0.8613   <- worst of eight measured
  //     keeptrue      0.9818     0.9323      clairvoyant upper bound
  //
  // THE GAIN IS ENTIRELY RETENTION, NOT READOUT: acc_retained moves 0.0009
  // between the shipped victim and this one while retention moves 0.0058.
  //
  // AND IT VANISHES EXACTLY AT K_TOP=4.  Every victim measured there -- the
  // clairvoyant one included -- lands on the same 1365 of 1536 voxels, to the
  // bit.  Retention still spans twelve voxels across the rules; the correct
  // count does not move by one, because each voxel a better victim rescues at
  // K=4 is one the readout then labels wrong.  Capacity and victim choice are
  // SUBSTITUTES here, not complements.  Like SCOVOX_VICTIM_QMAX, this is worth
  // enabling only for builds pinned to K_TOP=2.
  //
  // Only the CHOICE of victim moves.  The displaced mass is still
  // `cnt[min_i] - alpha_0`, so `Delta(other + sum(cnt)) == inc` still holds,
  // and nothing new is serialized.
  if (nhit != nullptr) {
    float best = (cnt[0] - alpha_0) /
                 static_cast<float>(nhit[0] > 0 ? nhit[0] : 1);
    for (int i = 1; i < K_TOP; ++i) {
      const float m = (cnt[i] - alpha_0) /
                      static_cast<float>(nhit[i] > 0 ? nhit[i] : 1);
      if (m < best) { best = m; min_i = i; }
    }
  } else {
    for (int i = 1; i < K_TOP; ++i) if (cnt[i] < cnt[min_i]) min_i = i;
  }
#elif SCOVOX_VICTIM_QMAX
  // VICTIM axis, build-time only (`-DSCOVOX_VICTIM_QMAX=1`), off by default so
  // the shipped build's bytes are unchanged.
  //
  // The shipped search picks the slot holding the least ACCUMULATED evidence,
  // which is the right key for the shipped comparator and the wrong one for the
  // confidence comparator: `--evict-by-confidence` then asks whether the
  // arrival beats the peak confidence of a slot chosen for having been seen
  // least often. Those are two different questions, and the slot with the
  // weakest evidence is not generally the slot with the weakest belief.
  // Selecting the victim by the same key the comparator will judge it on makes
  // the rule internally consistent. Offline replay of a recorded deposit stream
  // ranks it first of 24 AT K_TOP=2 (retention 0.9052 against 0.8979 for the
  // shipped victim under the same comparator, and 0.8680 for the shipped rule)
  // and TWELFTH of 24 at K_TOP=4 (0.9505 against 0.9557 and 0.9485). Capacity
  // dominates the rule, and the spread across all 24 rules collapses from 0.24
  // to 0.08 between the two. Worth enabling only at K_TOP=2.
  //
  // Only the CHOICE of victim moves. The displaced mass is still
  // `cnt[min_i] - alpha_0`, so `Delta(other + sum(cnt)) == inc` still holds.
  if (use_q) {
    for (int i = 1; i < K_TOP; ++i) if (qmax[i] < qmax[min_i]) min_i = i;
  } else {
    for (int i = 1; i < K_TOP; ++i) if (cnt[i] < cnt[min_i]) min_i = i;
  }
#else
  for (int i = 1; i < K_TOP; ++i) if (cnt[i] < cnt[min_i]) min_i = i;
#endif
  // Clamp at 0. A filled slot should always hold >= α₀ (prior + observed
  // evidence), but an evidence-saturation rescale can erode it below α₀; an
  // unclamped `cnt[min_i] − α₀` would then be NEGATIVE and turn the
  // `*other += evicted_evidence` below into a mass SUBTRACTION (driving OTHER
  // negative, breaking the Δ(other + Σcnt) == inc invariant). The saturation
  // path also floors filled slots at α₀, so this is normally a no-op safety net.
  const float raw_evicted = cnt[min_i] - alpha_0;
  const float evicted_evidence = raw_evicted > 0.f ? raw_evicted : 0.f;
  // Comparator choice changes only WHICH deposits win a contested slot, never
  // how much mass moves: both branches below conserve Δ(other + Σcnt) == inc.
  // Shipped rule weighs accumulated evidence, so a class that arrives often
  // beats one that arrives certain; the confidence rule inverts that, letting a
  // single high-probability observation displace a pile of low-probability
  // ones. On noisy real-camera labels the latter measured better.
#if SCOVOX_ADMIT_NORM
  // ADMIT axis, build-time only (`-DSCOVOX_ADMIT_NORM=1`).  The shipped test
  // `inc > evicted_evidence` compares ONE deposit against a slot's entire
  // accumulated history, so once a voxel has been looked at a few hundred
  // times no arrival can ever win: the measured eviction rate on real data is
  // 0.0000 and the rule has degenerated into "the first K_TOP classes to
  // arrive own this voxel forever".  Dividing the incumbent's evidence by the
  // number of deposits that built it makes the comparison scale-free, so a
  // strong arrival can still displace a long-accumulated weak incumbent.
  //
  // This branch takes precedence over the confidence comparator when both are
  // compiled in, because the two are alternative answers to the same question
  // and silently ANDing them would be a third rule nobody measured.
  const bool evict_now =
      (nhit != nullptr)
          ? (inc > (evicted_evidence /
                    static_cast<float>(nhit[min_i] > 0 ? nhit[min_i] : 1)))
          : (inc > evicted_evidence);
#else
  const bool evict_now = use_q ? (q_fx > qmax[min_i])
                               : (inc > evicted_evidence);
#endif
  if (evict_now) {
    // Evict: incoming class displaces slot min_i.
    //
    // INHERIT axis, build-time only (`-DSCOVOX_EVICT_INHERIT=n`), 0 by default
    // so the shipped build's behaviour and bytes are unchanged.
    //
    // The shipped answer (0) gives the newcomer a fresh counter and hands the
    // victim's accumulated evidence to OTHER. That has a consequence nobody
    // chose: a slot that was just filled holds the least evidence in the voxel
    // almost by construction, so it is the slot the NEXT contested arrival
    // selects as victim under the `cnt` key. Two classes then take turns owning
    // one slot, neither accumulates, and which of them holds it when the run
    // ends is close to a coin flip. That is a retention loss no readout can
    // undo, and retention is what final quality tracks.
    //
    // Space-Saving (Metwally, Agrawal & El Abbadi, ICDT 2005) -- which the
    // comment above says this is -- avoids it by giving the newcomer the
    // VICTIM's counter plus its own increment rather than a fresh one. That
    // inheritance is precisely what makes the algorithm's guarantee hold: a
    // counter over-estimates its class's true mass but never under-estimates
    // it, so an item that genuinely arrives often cannot be permanently locked
    // out by an early incumbent. Setting 0 pays the eviction cost without
    // taking the guarantee.
    //
    //   0  cnt = α₀ + inc                 evidence to OTHER   (shipped)
    //   1  cnt = α₀ + ev + inc            nhit inherited      (canonical SS)
    //   2  cnt = α₀ + ev + inc            nhit reset to 1
    //   3  cnt = α₀ + ev/2 + inc          nhit halved
    //
    // 2 exists because inheritance is not obviously a single decision: a
    // scale-free comparator divides accumulated evidence by `nhit`, so
    // inheriting `cnt` while resetting `nhit` hands the newcomer a
    // per-observation strength it never earned and makes it near-immovable.
    // `qmax` is never inherited under any setting -- it is a peak CONFIDENCE,
    // and one class's confidence is not evidence about another's.
    //
    // MASS IS CONSERVED IN EVERY SETTING, which is the invariant this file
    // asserts. Under 0 the victim's evidence leaves the slot and arrives in
    // OTHER; under 1 and 2 it stays in the slot and OTHER receives nothing;
    // under 3 it splits. In all four Δ(other + Σcnt) == inc.
#if   SCOVOX_EVICT_INHERIT == 1 || SCOVOX_EVICT_INHERIT == 2
    const float kept = evicted_evidence;
#elif SCOVOX_EVICT_INHERIT == 3
    const float kept = 0.5f * evicted_evidence;
#else
    const float kept = 0.0f;
#endif
    *other += evicted_evidence - kept;
    cls[min_i] = c;
    cnt[min_i] = alpha_0 + kept + inc;
    if (track) qmax[min_i] = q_fx;
    if (nhit) {
#if   SCOVOX_EVICT_INHERIT == 1
      if (nhit[min_i] != 65535) ++nhit[min_i];
#elif SCOVOX_EVICT_INHERIT == 3
      nhit[min_i] = static_cast<uint16_t>((nhit[min_i] >> 1) + 1);
#else
      nhit[min_i] = 1;
#endif
    }
    g_sparse_evict_count.fetch_add(1, std::memory_order_relaxed);
    say(3);
  } else {
    // Drop: incoming evidence smaller than every tracked class. Mass to OTHER.
    *other += inc;
    g_sparse_drop_count.fetch_add(1, std::memory_order_relaxed);
    say(4);
  }
}

/// Argmax of the top-K class slots by observed evidence (`cnt − α₀`). Returns
/// 0xFFFF if no slot is filled, or if `OTHER`'s *observed* evidence exceeds
/// every slot's evidence (the bulk of the class mass is on out-of-K classes,
/// so committing to a tracked class would be misleading). Mirrors the
/// `SemDirVoxel` overload in mesh_labelling.hpp, restricted to the
/// occupied-class Dirichlet.
///
/// The slot key subtracts each slot's `α₀` placeholder; the OTHER key must
/// subtract OTHER's own prior `(C − K_TOP)·α₀` (the collapsed out-of-K
/// dimensions, set by `defaultDirVoxel`) for an apples-to-apples comparison —
/// otherwise OTHER's prior mass alone (0.12 at C=14, α₀=0.01) spuriously
/// dominates a legitimately-observed slot and the only seen class is hidden.
/// `num_classes` defaults to 14 (matching `defaultDirVoxel`); the residual is
/// clamped at 0 to match `defaultDirVoxel`'s `num_classes < K_TOP` convention.
inline uint16_t dominantClass(const DirVoxel& v,
                              float    alpha_0     = kDefaultDirichletPrior,
                              uint16_t num_classes = 14) noexcept {
  uint16_t cls = 0xFFFF;
  float best_evidence = 0.f;
  for (int i = 0; i < K_TOP; ++i) {
    if (v.cls[i] == 0xFFFF) continue;
    const float evidence = v.cnt[i] - alpha_0;
    if (evidence > best_evidence) {
      best_evidence = evidence;
      cls           = v.cls[i];
    }
  }
  // OTHER's observed evidence = bucket minus its (C − K_TOP)·α₀ prior, clamped
  // at 0 (defaultDirVoxel clamps the residual when num_classes ≤ K_TOP). Only
  // genuine out-of-K evidence — not the prior — may veto the tracked argmax.
  const int   residual_dims = static_cast<int>(num_classes) - K_TOP;
  const float other_prior   = (residual_dims > 0) ? (residual_dims * alpha_0) : 0.f;
  const float other_evidence = v.other - other_prior;
  if (other_evidence > best_evidence) return 0xFFFF;
  return cls;
}

}  // namespace scovox
