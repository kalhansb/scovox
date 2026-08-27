#pragma once
/// @file node_utils.hpp
/// @brief Shared utilities for scovox_node and dscovox_node.

#include <algorithm>
#include <array>
#include <cstdint>
#include <utility>
#include <vector>
#include <cmath>

#include <bonxai/bonxai.hpp>

#include "scovox/voxel.hpp"
#include "scovox/uncertainty.hpp"

namespace scovox {

// Window-restricted variant of Bonxai::VoxelGrid::forEachCell (audit items
// 4/5) — shared by scovox_node's planning-map walk and dscovox_node's
// region/occupancy-service walks. Bonxai root-map keys are block-base voxel
// coords with the low (inner_bits + leaf_bits) bits zero (getRootKey), so a
// whole root block — 2^(inner_bits+leaf_bits) voxels per axis — can be
// rejected with six integer compares before its masks are touched; surviving
// blocks descend with the same bit arithmetic as forEachCell. `box_min`/
// `box_max` are INCLUSIVE voxel-coord bounds and must be conservative
// (over-cover the region of interest): the visitor's own predicates stay the
// sole authority on cell membership, so the output is identical to a full
// walk — only the iteration domain shrinks. Unlike forEachCell this has no
// EmptyVoxel special case (Grid::cell static_asserts on it) — extend before
// reusing on such a grid.
template <typename DataT, typename VisitorFn>
void forEachCellInBox(
    const Bonxai::VoxelGrid<DataT>& grid, const Bonxai::CoordT& box_min,
    const Bonxai::CoordT& box_max, VisitorFn&& fn) {
  const int32_t leaf_bits = int32_t(grid.leafBits());
  const int32_t inner_bits = int32_t(grid.innetBits());  // [sic] Bonxai API
  const int32_t mask_leaf = (1 << leaf_bits) - 1;
  const int32_t mask_inner = (1 << inner_bits) - 1;
  const int32_t root_span = 1 << (inner_bits + leaf_bits);
  const int32_t leaf_span = 1 << leaf_bits;
  for (const auto& [root_key, inner_grid] : grid.rootMap()) {
    if (root_key.x + root_span <= box_min.x || root_key.x > box_max.x ||
        root_key.y + root_span <= box_min.y || root_key.y > box_max.y ||
        root_key.z + root_span <= box_min.z || root_key.z > box_max.z) {
      continue;
    }
    for (auto inner_it = inner_grid.mask().beginOn(); inner_it; ++inner_it) {
      const int32_t inner_index = *inner_it;
      const int32_t xB = root_key.x | ((inner_index & mask_inner) << leaf_bits);
      const int32_t yB =
          root_key.y | (((inner_index >> inner_bits) & mask_inner) << leaf_bits);
      const int32_t zB =
          root_key.z |
          (((inner_index >> (inner_bits * 2)) & mask_inner) << leaf_bits);
      if (xB + leaf_span <= box_min.x || xB > box_max.x ||
          yB + leaf_span <= box_min.y || yB > box_max.y ||
          zB + leaf_span <= box_min.z || zB > box_max.z) {
        continue;
      }
      const auto& leaf_grid = inner_grid.cell(inner_index);
      for (auto leaf_it = leaf_grid->mask().beginOn(); leaf_it; ++leaf_it) {
        const int32_t leaf_index = *leaf_it;
        const Bonxai::CoordT c = {
            xB | (leaf_index & mask_leaf),
            yB | ((leaf_index >> leaf_bits) & mask_leaf),
            zB | ((leaf_index >> (leaf_bits * 2)) & mask_leaf)};
        if (c.x < box_min.x || c.x > box_max.x || c.y < box_min.y ||
            c.y > box_max.y || c.z < box_min.z || c.z > box_max.z) {
          continue;
        }
        fn(leaf_grid->cell(leaf_index), c);
      }
    }
  }
}

// Heartbeat re-emit sweep for the change-gate wire path (audit item 7 split
// the last-emit timestamp out of the gate grids so it is allocated only when
// share_heartbeat_sec > 0). `stamps` is written one-for-one with `gate` at
// every wire emit, so its active set and iteration order ARE the ever-emitted
// set in the order the pre-split single-grid walk used to yield. For each
// voxel whose stamp is at least `period` old: fetch the live voxel via
// `live`; if absent, leave the stamp untouched (retried next tick, as
// before); if `accept(live)` rejects (the Dir binarize veto), leave the stamp
// too — the voxel is re-checked every tick until it becomes emittable;
// otherwise refresh the gate copy, stamp t_now, and hand the voxel to `emit`.
// The stamp stays a double: node-clock epoch seconds are outside float's
// exact-integer range and the sweep compares differences of these.
template <typename GateV, typename LiveAcc, typename AcceptFn, typename EmitFn>
void heartbeatReemit(Bonxai::VoxelGrid<GateV>& gate,
                     Bonxai::VoxelGrid<double>& stamps, LiveAcc& live,
                     double t_now, double period, AcceptFn&& accept,
                     EmitFn&& emit) {
  auto gacc = gate.createAccessor();
  stamps.forEachCell([&](double& t_emit, const Bonxai::CoordT& c) {
    if (t_now - t_emit < period) return;
    const auto* v = live.value(c, false);
    if (!v) return;
    if (!accept(*v)) return;
    if (auto* g = gacc.value(c, false)) *g = *v;
    t_emit = t_now;
    emit(*v, c);
  });
}

/// Generate a semantic color palette of the given size.
/// First 10 entries are fixed well-known colors, rest are golden-angle HSV.
inline std::vector<std::array<float, 3>> generateSemanticColors(size_t count) {
  static const std::array<float, 3> kBase[] = {
    {.5f,.5f,.5f}, {0,1,0}, {.5f,.3f,.1f}, {.3f,.3f,.3f}, {.1f,.5f,.1f},
    {1,0,0}, {0,0,1}, {1,1,0}, {1,0,1}, {0,1,1}
  };
  std::vector<std::array<float, 3>> colors;
  colors.reserve(count);
  for (size_t i = 0; i < 10 && i < count; ++i)
    colors.push_back(kBase[i]);
  while (colors.size() < count) {
    float h = std::fmod(colors.size() * 137.508f / 360.f, 1.f);
    float h6 = h * 6, x = 1 - std::fabs(std::fmod(h6, 2.f) - 1.f);
    std::array<float, 3> c;
    if (h6 < 1) c = {1, x, 0};
    else if (h6 < 2) c = {x, 1, 0};
    else if (h6 < 3) c = {0, 1, x};
    else if (h6 < 4) c = {0, x, 1};
    else if (h6 < 5) c = {x, 0, 1};
    else c = {1, 0, x};
    colors.push_back(c);
  }
  return colors;
}

/// Result of selecting the strongest top-K semantic slots from a Voxel.
/// `kept` holds the chosen (class_id, count) pairs in descending count order;
/// only the first `kept_count` entries are valid. `dropped_mass` is the sum
/// of the counts that were *not* kept and must be folded into the consumer's
/// `a_unk` to preserve total semantic mass.
struct TopKSemantics {
  std::array<std::pair<uint16_t, float>, K_TOP> kept{};
  size_t kept_count = 0;
  float dropped_mass = 0.f;
};

/// Select the `top_k` strongest semantic slots from `v` (sorted descending by
/// count) and report any dropped slots' mass via `dropped_mass`.
///
/// `sparse_add` does not maintain slot order, so the K_TOP slots in a Voxel
/// are in arbitrary order with respect to count. A naive "first K non-zero
/// slots" loop can both promote a weak slot over a strong one (label bug) and,
/// if the dropped counts aren't folded into `a_unk`, leak total semantic mass
/// (mass-conservation bug). This helper exists so every consumer that wants
/// fewer than K_TOP classes uses the same correct selection rule.
inline TopKSemantics selectTopKSemantics(const Voxel& v, int top_k) {
  std::array<std::pair<uint16_t, float>, K_TOP> pairs{};
  size_t n = 0;
  for (int i = 0; i < K_TOP; ++i) {
    if (v.sem_cnt[i] > 0.f) {
      pairs[n++] = {v.sem_cls[i], v.sem_cnt[i]};
    }
  }
  // Hand-rolled insertion sort (n ≤ K_TOP) instead of std::sort: at -O3 the
  // inlined std::sort pulls in its 16-element introsort fallback, whose dead
  // `__first + 16` access trips a -Warray-bounds false positive when this is
  // inlined into the dscovox refold path. Descending by count; identical output
  // for this n.
  for (size_t i = 1; i < n; ++i) {
    const std::pair<uint16_t, float> key = pairs[i];
    size_t j = i;
    while (j > 0 && key.second > pairs[j - 1].second) { pairs[j] = pairs[j - 1]; --j; }
    pairs[j] = key;
  }

  TopKSemantics out;
  // Clamp the request to [0, K_TOP]. A negative `top_k` is treated as 0 (not 1):
  // a caller asking for "no semantic slots" (occupancy-only emit) must get
  // kept_count == 0 and have the *entire* semantic mass folded into
  // `dropped_mass`, rather than silently keeping one class and under-folding by
  // one slot. The previous std::max(1, top_k) floor violated that intent and
  // leaked the strongest slot's mass past the dropped_mass accumulation.
  const size_t cap = std::min<size_t>(static_cast<size_t>(K_TOP),
                                      top_k < 0 ? 0u : static_cast<size_t>(top_k));
  out.kept_count = std::min(n, cap);
  for (size_t i = 0; i < out.kept_count; ++i) out.kept[i] = pairs[i];
  for (size_t i = out.kept_count; i < n; ++i) out.dropped_mass += pairs[i].second;
  return out;
}

/// Argmax tracked semantic class + Hutter-framework-consistent posterior
/// probability. Returns (best_class_id, p_best) under the (Kv + 1)-Dirichlet
/// over { active tracked classes } ∪ { unknown bucket }, with +1 prior on
/// every slot. Kv = `n_active` is *per-voxel* (the count of tracked slots
/// with sem_cnt > 0 at this voxel), not the compile-time K_TOP cap:
///
///   p(best) = (best_cnt + 1) / [Σ_{tracked active} (cnt + 1) + (effectiveResidual(v) + 1)]
///
/// Matches the categorical used by `semanticEntropy` / `semanticVariance`
/// in scovox_core, so all three uncertainty signals (entropy, variance,
/// and the published confidence) stay mutually consistent. Replaces the
/// pre-2026-05-03 inline `cf = best_cnt / Σ tracked` rule, which dropped
/// `a_unk` and the Dirichlet prior from the denominator and was therefore
/// systematically over-confident whenever K_TOP eviction had occurred.
///
/// Edge cases:
///   - n_active == 0 (Beta evidence but no semantic observations): returns
///     (0, 0.f). Callers gating on `p_best >= threshold` will not colour /
///     emit a label — correct behaviour.
///   - Tied counts: first-index-wins from the inner strict-`>` test, which
///     matches `sparse_add`'s eviction rule (also strict-`>`). The argmax
///     is therefore *not unique* under exact ties; this propagates the
///     same first-arrival bias flagged by C4 in the ablations punch list,
///     and the paper should not claim uniqueness without that caveat.
/// Template form (D6): shared body for Voxel and SemBetaVoxel. Both
/// expose `sem_cnt[K_TOP]`, `sem_cls[K_TOP]`, and the
/// `effectiveResidual(v)` Hutter helper accepts both via its own
/// template overload (uncertainty.hpp). Existing call sites pass a
/// Voxel and the deduction picks the same body that used to be the
/// non-template overload — no behavioural change for the legacy path.
template <typename V>
inline std::pair<uint16_t, float> argmaxClassConfidence(const V& v) {
  uint16_t best_cls = 0;
  float    best_cnt = 0.f;
  float    sum_cnt  = 0.f;
  int      n_active = 0;
  for (int i = 0; i < K_TOP; ++i) {
    if (v.sem_cnt[i] > 0.f) {
      sum_cnt += v.sem_cnt[i];
      ++n_active;
      if (v.sem_cnt[i] > best_cnt) {
        best_cnt = v.sem_cnt[i];
        best_cls = v.sem_cls[i];
      }
    }
  }
  if (n_active == 0) return {0, 0.f};
  // The residual term here is `effectiveResidual(v)`, which is now bounded above
  // by N = Σ sem_cnt + a_unk (the Hutter escape mass is clamped to N in
  // uncertainty.hpp). That bound is what keeps this confidence denominator sane
  // for weakly-observed voxels: e.g. sem_cnt={0.05}, a_unk=0 gives residual 0.05
  // (not the unbounded ~10 the raw m/(2 ln ratio) term used to produce), so
  // p_best ≈ 1.05/2.1 ≈ 0.5 rather than collapsing to ~0.087 below the labelling
  // threshold. We deliberately do NOT add a second, tighter cap (e.g. clamp the
  // residual to sum_cnt) here: this categorical must stay identical to the one
  // used by semanticEntropy / semanticVariance so all three uncertainty signals
  // remain mutually consistent, and the bound that fixes the collapse lives in
  // the shared effectiveResidual helper.
  const float denom = sum_cnt + static_cast<float>(n_active)
                    + effectiveResidual(v) + 1.f;
  if (denom <= 0.f) return {best_cls, 0.f};
  return {best_cls, (best_cnt + 1.f) / denom};
}

} // namespace scovox
