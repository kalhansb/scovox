#pragma once
/// @file semantics.hpp
/// @brief Semantic update modes — zero ROS dependencies.
///
/// The Dirichlet update is Bayesian-soft: the per-observation weight is
/// `kappa0 * p_occ * quality`, which marginalises the class observation
/// over the current Beta posterior on occupancy. Free voxels (low p_occ)
/// contribute small mass; confidently-occupied voxels contribute fully.
/// No hard threshold gate.

#include <vector>
#include <cmath>
#include <algorithm>
#include "scovox/voxel.hpp"

namespace scovox {

enum class SemanticMode : uint8_t {
  DIRICHLET = 0,
  NAIVE = 1,
  MAJORITY_VOTE = 2,
};

inline void dirichlet_update_semantics(Voxel* v,
                                       const std::vector<float>* class_probs,
                                       float quality,
                                       float p_occ,
                                       float kappa0) {
  if (!class_probs) return;
  const auto& obs_probs = *class_probs;
  if (obs_probs.empty()) return;

  // Bayesian-soft attribution: weight by current posterior p_occ.
  // Caller is expected to gate the "skip uncertain voxels" behaviour via
  // `dirichlet_min_p_occ` BEFORE calling this function.
  const float w = kappa0 * p_occ * std::clamp(quality, 0.0f, 1.0f);
  if (w <= 0.0f) return;

  // First pass: sum the input probabilities. Used to normalise un-normalised
  // inputs (Σ > 1) and to compute the residual budget for `a_unk`.
  float sum_p = 0.0f;
  for (size_t i = 0; i < obs_probs.size(); ++i) {
    if (obs_probs[i] > 0.0f) sum_p += obs_probs[i];
  }

  // Mass conservation: each observation contributes exactly w pseudocounts.
  // For Σ p_i ≤ 1 (one-hot/softmax) this is a no-op (norm = 1).
  const float norm = (sum_p > 1.0f) ? (1.0f / sum_p) : 1.0f;

  for (size_t i = 0; i < obs_probs.size(); ++i) {
    if (obs_probs[i] <= 0.0f) continue;
    const float inc = w * obs_probs[i] * norm;
    if (inc <= 0.0f) continue;
    sparse_add(v->sem_cnt, v->sem_cls, static_cast<uint16_t>(i), inc, &v->a_unk);
  }

  // Residual = unobserved-class share = w * (1 - Σ p_i).
  // sparse_add already routes its own eviction/drop overflow into a_unk;
  // the residual here is the probability mass the classifier did NOT
  // explicitly assign to any class.
  const float covered = sum_p * norm;
  const float uinc = w * (1.0f - covered);
  if (uinc > 0.0f) v->a_unk += uinc;
}

/// NAIVE mode — intentional ablation baseline.
///
/// "Last observation wins": every update wipes prior semantic state and
/// stores the argmax class with a fixed count of 1.0. By design this:
///   - ignores `quality`, `kappa0`, and `p_occ` (no weighting),
///   - cannot accumulate confidence over repeated observations,
///   - returns the same `semanticEntropy` regardless of how many times the
///     cell was observed.
/// `apply_semantics` still applies a hard `p_occ > 0.5` cutoff for NAIVE
/// and MAJORITY_VOTE (so they only fire on occupied voxels). This is used
/// as a contrast against the Dirichlet mode in ablations and should NOT
/// be "improved" without removing it from the ablation suite.
inline void naive_update_semantics(Voxel* v,
                                   const std::vector<float>* class_probs) {
  if (!class_probs) return;
  const auto& obs = *class_probs;
  if (obs.empty()) return;

  auto it = std::max_element(obs.begin(), obs.end());
  if (*it <= 0.f) return;
  uint16_t label = static_cast<uint16_t>(std::distance(obs.begin(), it));

  for (int i = 0; i < K_TOP; ++i) { v->sem_cls[i] = 0; v->sem_cnt[i] = 0.f; }
  v->a_unk = 0.f;
  v->sem_cls[0] = label;
  v->sem_cnt[0] = 1.f;
}

/// MAJORITY_VOTE mode — intentional ablation baseline.
///
/// Each admitted observation contributes a single +1 vote to the argmax
/// class. Like NAIVE, this ignores `quality`, `kappa0`, and `p_occ`;
/// unlike NAIVE it accumulates votes across observations. A hard
/// `p_occ > 0.5` cutoff is applied at `apply_semantics`,
/// so MAJORITY_VOTE only fires on occupied voxels — the only ablation
/// variable vs DIRICHLET is the accumulation rule.
inline void majority_vote_semantics(Voxel* v,
                                    const std::vector<float>* class_probs) {
  if (!class_probs) return;
  const auto& obs = *class_probs;
  if (obs.empty()) return;

  auto it = std::max_element(obs.begin(), obs.end());
  if (*it <= 0.f) return;
  uint16_t label = static_cast<uint16_t>(std::distance(obs.begin(), it));

  sparse_add(v->sem_cnt, v->sem_cls, label, 1.0f, &v->a_unk);
}

} // namespace scovox
