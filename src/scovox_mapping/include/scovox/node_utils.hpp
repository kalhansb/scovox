#pragma once
/// @file node_utils.hpp
/// @brief Shared utilities for scovox_node and dscovox_node.
/// Moved comments: doc/scovox_mapping_code_notes.md

#include <algorithm>
#include <array>
#include <utility>
#include <vector>
#include <cmath>

#include "scovox/voxel.hpp"
#include "scovox/uncertainty.hpp"

namespace scovox {

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

/// Strongest top-K semantic slots: kept is sorted by descending count, only the
/// first kept_count entries are valid; dropped_mass must be folded into the
/// consumer's a_unk to conserve semantic mass. (notes: topk-semantics-struct)
struct TopKSemantics {
  std::array<std::pair<uint16_t, float>, K_TOP> kept{};
  size_t kept_count = 0;
  float dropped_mass = 0.f;
};

/// Select the `top_k` strongest semantic slots from `v` (sorted descending by
/// count) and report any dropped slots' mass via `dropped_mass`.
///
/// sparse_add keeps slots in arbitrary order, so every consumer wanting fewer
/// than K_TOP classes must use this helper to pick the strongest and account
/// for the dropped mass. (notes: topk-select-why-helper)
inline TopKSemantics selectTopKSemantics(const Voxel& v, int top_k) {
  std::array<std::pair<uint16_t, float>, K_TOP> pairs{};
  size_t n = 0;
  for (int i = 0; i < K_TOP; ++i) {
    if (v.sem_cnt[i] > 0.f) {
      pairs[n++] = {v.sem_cls[i], v.sem_cnt[i]};
    }
  }
  // Insertion sort (n <= K_TOP), descending by count. Do not switch to
  // std::sort: inlined at -O3 into the dscovox refold path it trips a
  // -Warray-bounds false positive. (notes: topk-insertion-sort)
  for (size_t i = 1; i < n; ++i) {
    const std::pair<uint16_t, float> key = pairs[i];
    size_t j = i;
    while (j > 0 && key.second > pairs[j - 1].second) { pairs[j] = pairs[j - 1]; --j; }
    pairs[j] = key;
  }

  TopKSemantics out;
  // Clamp top_k to [0, K_TOP]; a negative top_k means 0, so an occupancy-only
  // caller gets kept_count == 0 and all semantic mass in dropped_mass.
  // (notes: topk-negative-top-k)
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
/// Uses the same categorical as semanticEntropy and semanticVariance, so
/// entropy, variance and the published confidence stay consistent.
/// (notes: argmax-confidence-consistency)
///
/// n_active == 0 returns (0, 0.f). Exact ties go to the first slot (strict >,
/// as in sparse_add), so the argmax is not unique. One template body serves
/// Voxel and SemBetaVoxel. (notes: argmax-confidence-edge-cases)
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
  // effectiveResidual(v) is bounded above by the sum of sem_cnt plus a_unk (in
  // uncertainty.hpp), which keeps this denominator sane. Do not add a tighter
  // cap here: the categorical must match semanticEntropy and semanticVariance.
  // (notes: argmax-residual-bound)
  const float denom = sum_cnt + static_cast<float>(n_active)
                    + effectiveResidual(v) + 1.f;
  if (denom <= 0.f) return {best_cls, 0.f};
  return {best_cls, (best_cnt + 1.f) / denom};
}

} // namespace scovox
