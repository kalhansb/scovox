#pragma once
/// @file uncertainty.hpp
/// @brief Beta and Dirichlet uncertainty functions — zero ROS dependencies.
/// Moved comments: doc/scovox_core_code_notes.md

#include <algorithm>
#include <cmath>

#include "scovox/voxel.hpp"
#include "scovox/sembeta_voxel.hpp"

namespace scovox {

float digamma(float x);

// ====================================================================
// Voxel-typed (legacy fused 32-byte struct) overloads — definitions in
// uncertainty.cpp. Kept verbatim for the legacy `scovox::Map` path.
// ====================================================================
float variance(const Voxel& v);
float entropy(const Voxel& v);
float expectedInformationGain(const Voxel& v);
float semanticEntropy(const Voxel& v);
float semanticVariance(const Voxel& v, uint16_t class_id);
float betaKL(const Voxel& a, const Voxel& b);

// ====================================================================
// SemBetaVoxel (24-byte) overloads with the same bodies as the Voxel versions;
// separate overloads rather than templates keep the library symbol surface
// stable. Only variance and expectedInformationGain exist.
// (notes: uncertainty-sembeta-overloads)
// ====================================================================
float variance(const SemBetaVoxel& v);
float expectedInformationGain(const SemBetaVoxel& v);

/// Lower bound on distinct classes ever observed, from existing fields only;
/// underestimating m makes the Hutter floor conservative. Shared by Voxel and
/// SemBetaVoxel. (notes: uncertainty-distinct-classes)
template <typename V>
inline int estimateDistinctClasses(const V& v) {
  int m = 0;
  for (int i = 0; i < K_TOP; ++i)
    if (v.sem_cnt[i] > 0.f) ++m;
  // If a_unk > 0, at least one additional class was seen and evicted/dropped
  if (v.a_unk > 0.f) m += 1;
  return std::max(m, 1);  // floor at 1 to avoid log(0)
}

/// Hutter escape mass m / (2 ln((N+1)/m)), with m distinct classes seen (or a
/// lower bound) and N the total semantic observations; a floor for a_unk as
/// the posterior mass of untracked classes. (notes: uncertainty-hutter-escape)
inline float hutterEscapeMass(int m, float N) {
  if (m <= 0 || N <= 0.f) return 0.f;
  const float ratio = (N + 1.f) / static_cast<float>(m);
  // The escape mass cannot exceed N; the raw formula diverges as ratio nears
  // 1, so clamp to N both when ratio <= 1 (m >= N+1) and near it.
  // (notes: uncertainty-hutter-clamp)
  if (ratio <= 1.f) return N;  // degenerate: m ≥ N+1, escape mass capped at N
  return std::min(static_cast<float>(m) / (2.f * std::log(ratio)), N);
}

/// a_unk with the Hutter floor applied. Query time only (entropy, prediction,
/// visualisation); never in update paths, where sparse_add and
/// dirichlet_update_semantics must keep raw a_unk so evidence is conserved.
/// (notes: uncertainty-effective-residual)
template <typename V>
inline float effectiveResidual(const V& v) {
  const int m = estimateDistinctClasses(v);
  float N = v.a_unk;
  for (int i = 0; i < K_TOP; ++i) N += v.sem_cnt[i];
  return std::max(v.a_unk, hutterEscapeMass(m, N));
}

/// SSMI-style KL divergences for ray-marginalised MI computation.
/// Both return KL(Bern(p) || Bern(p_post)), matching the f(φ,h) function
/// in Asgharivaskasi & Atanasov (TRO 2023) adapted to the Beta model.

/// KL divergence from current estimate to posterior after an "occupied"
/// observation: prior Beta(a,b) → posterior Beta(a+1,b).
float ssmiOccKL(const Voxel& v);

/// KL divergence from current estimate to posterior after a "free"
/// observation: prior Beta(a,b) → posterior Beta(a,b+1).
float ssmiFreeKL(const Voxel& v);

} // namespace scovox
