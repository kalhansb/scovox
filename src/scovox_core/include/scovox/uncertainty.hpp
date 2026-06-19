#pragma once
/// @file uncertainty.hpp
/// @brief Beta and Dirichlet uncertainty functions — zero ROS dependencies.

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
// SemBetaVoxel-typed (split-grid 24-byte struct) overloads — D6 from
// the resume-grilling pass. Bodies are identical to the Voxel versions
// (both structs expose a_occ / a_free / a_unk / sem_cls[] / sem_cnt[]
// with the same semantics); separate overloads are added rather than
// templating the .cpp definitions to keep the existing library symbol
// surface stable. Only the helpers actually consumed by the dscovox_node
// + scovox_node publishers are added now (variance, expectedInformationGain);
// the rest can be added incrementally if a downstream caller needs them.
// ====================================================================
float variance(const SemBetaVoxel& v);
float expectedInformationGain(const SemBetaVoxel& v);

/// Lower-bound estimate of distinct classes ever observed at this voxel.
/// Uses only existing fields — zero extra memory.
/// Underestimates m, which makes the Hutter floor conservative (less
/// unknown mass). Acceptable if noted as a lower bound in the paper.
///
/// Template form (D6): both Voxel and SemBetaVoxel expose `sem_cnt[K_TOP]`
/// and `a_unk` fields with identical semantics, so the body is shared.
template <typename V>
inline int estimateDistinctClasses(const V& v) {
  int m = 0;
  for (int i = 0; i < K_TOP; ++i)
    if (v.sem_cnt[i] > 0.f) ++m;
  // If a_unk > 0, at least one additional class was seen and evicted/dropped
  if (v.a_unk > 0.f) m += 1;
  return std::max(m, 1);  // floor at 1 to avoid log(0)
}

/// Hutter (AISTATS 2013, §3) adaptive escape mass for the sparse Dirichlet.
///
/// β* = m / [2 ln((N+1)/m)]
///
/// m: distinct classes ever observed (or lower bound)
/// N: total semantic observations (sum of all sem_cnt + a_unk)
///
/// Returns a floor for a_unk that gives the residual a principled
/// interpretation as a Dirichlet escape probability — the posterior
/// mass allocated to untracked classes — rather than a dump bucket
/// for evicted evidence.
inline float hutterEscapeMass(int m, float N) {
  if (m <= 0 || N <= 0.f) return 0.f;
  const float ratio = (N + 1.f) / static_cast<float>(m);
  if (ratio <= 1.f) return static_cast<float>(m);  // degenerate: all singletons
  return static_cast<float>(m) / (2.f * std::log(ratio));
}

/// Effective a_unk with Hutter floor applied.
///
/// Use this at QUERY TIME (entropy, class prediction, visualization)
/// instead of raw v.a_unk. Do NOT use in update paths — the raw
/// accumulation in sparse_add and dirichlet_update_semantics must
/// remain unmodified so evidence is conserved exactly.
///
/// Template form (D6): shared body for Voxel and SemBetaVoxel. Both
/// expose `a_unk` and `sem_cnt[K_TOP]` with identical semantics.
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
