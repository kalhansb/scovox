#pragma once
/// @file dscovox_consensus.hpp
/// @brief Pure receiver-side split-consensus helpers shared by dscovox_node and
///        its unit tests.
///
/// These functions are the math the multi-robot merger (dscovox_node.cpp) is
/// built on: the at-prior tests that gate the refold, the Beta+Dir → fused
/// projection used by the RPC query services / viz, and the per-cell refold
/// core itself. They were previously defined in an anonymous namespace inside
/// dscovox_node.cpp, which made them unreachable by symbol from tests (so the
/// suite had to mirror the arithmetic, per findings #18/#19/#20). Hoisting them
/// here lets the tests exercise the SAME code the node runs.
///
/// Everything here is pure (voxels + num_classes + alpha_0 in, voxel/bool out) —
/// no ROS, Bonxai, Eigen, or node member state. The node keeps the Bonxai
/// accessor / grid plumbing; only the per-cell math lives here.
/// Moved comments: doc/scovox_mapping_code_notes.md

#include <algorithm>
#include <cstdint>
#include <vector>

#include "scovox/beta_voxel.hpp"      // BetaVoxel, kBetaOccPrior/Free, defaultBetaVoxel
#include "scovox/dir_voxel.hpp"       // DirVoxel, defaultDirVoxel
#include "scovox/sembeta_voxel.hpp"   // SemBetaVoxel
#include "scovox/voxel.hpp"           // Voxel, K_TOP
#include "scovox/consensus_merge.hpp" // mergeBeta / mergeDir

namespace scovox {

/// At-prior epsilon for the receiver isPrior* tests. Must equal the sender's
/// at-prior emit gate in scovox_node.cpp (1e-4); a looser slop here drops
/// barely-observed voxels the sender emitted. (notes: consensus-prior-slop)
static constexpr float kPriorSlop = 1e-4f;

/// BetaVoxel "is at prior" check for the split consensus refold. A voxel
/// is at prior iff a_occ ≈ kBetaOccPrior and a_free ≈ kBetaFreePrior (the
/// symmetric Beta(1,1) occupancy prior; see docs/occupancy_prior.md). `slop`
/// matches isPriorDir's one-quantum tolerance.
inline bool isPriorBeta(const scovox::BetaVoxel& v,
                        uint16_t num_classes, float alpha_0) {
  (void)num_classes; (void)alpha_0;  // occupancy prior is the symmetric constant
  // Shipped occupancy prior is symmetric Beta(1,1) (kBetaOccPrior=kBetaFreePrior
  // =1, p_occ=0.5) — decoupled from (num_classes, α₀); see docs/occupancy_prior.md.
  // slop = kPriorSlop (1e-4) matches the sender's at-prior emit gate so a
  // barely-observed Beta voxel the sender put on the wire is not dropped on refold.
  const float slop = kPriorSlop;
  return v.a_occ <= scovox::kBetaOccPrior + slop &&
         v.a_free <= scovox::kBetaFreePrior + slop;
}

/// DirVoxel "is at prior" check: OTHER ≈ (C−K)·α_0 and no slot filled.
inline bool isPriorDir(const scovox::DirVoxel& v,
                       uint16_t num_classes, float alpha_0) {
  // Clamp residual_dims at 0 to match defaultDirVoxel: for num_classes <= K_TOP
  // the OTHER prior is 0, not (C-K)*alpha_0 < 0. A negative other_prior would
  // make `v.other > other_prior + slop` true for genuine prior voxels and so
  // misclassify them as observed.
  const int residual_dims = static_cast<int>(num_classes) - scovox::K_TOP;
  const float other_prior =
      (residual_dims > 0) ? (static_cast<float>(residual_dims) * alpha_0) : 0.f;
  // Match the sender's at-prior emit gate (kPriorSlop = 1e-4). See kPriorSlop.
  const float slop = kPriorSlop;
  if (v.other > other_prior + slop) return false;
  for (int i = 0; i < scovox::K_TOP; ++i)
    if (v.cls[i] != 0xFFFF) return false;
  return true;
}

/// Project split Beta(occupancy) + Dir(semantics) → SemBetaVoxel for the wire
/// visualisation path (reuses the shared argmax/variance/EIG helpers). The
/// Dir pointer may be null (occupancy-only voxel). Per-class evidence has the
/// α_0 prior subtracted so empty slots read 0, matching the sender's viz.
inline scovox::SemBetaVoxel projectBetaDirToSemBetaForViz(
    const scovox::BetaVoxel& b, const scovox::DirVoxel* d, float alpha_0,
    uint16_t num_classes) {
  scovox::SemBetaVoxel out{};
  out.a_occ  = b.a_occ;
  out.a_free = b.a_free;
  if (d) {
    // Raw-evidence convention, as in projectBetaDirToVoxel: subtract the OTHER
    // prior (C-K)*alpha_0, clamped at 0 for C <= K_TOP. argmaxClassConfidence,
    // effectiveResidual and semanticVariance expect a_unk without the prior.
    // (notes: consensus-viz-raw-evidence)
    const int residual_dims = static_cast<int>(num_classes) - scovox::K_TOP;
    const float other_prior =
        (residual_dims > 0) ? (static_cast<float>(residual_dims) * alpha_0) : 0.f;
    out.a_unk = std::max(0.f, d->other - other_prior);
    for (int i = 0; i < scovox::K_TOP; ++i) {
      out.sem_cnt[i] = std::max(0.f, d->cnt[i] - alpha_0);
      out.sem_cls[i] = d->cls[i];
    }
  } else {
    out.a_unk = 0.f;
    for (int i = 0; i < scovox::K_TOP; ++i) { out.sem_cnt[i] = 0.f; out.sem_cls[i] = 0xFFFF; }
  }
  return out;
}

/// Project split Beta(occupancy) + Dir(semantics) → the legacy fused
/// scovox::Voxel for the wire RPC query services (GetRegion / GetOccupancyGrid).
/// The Dir pointer may be null (occupancy-only voxel, or a caller that only
/// needs occupancy — EIG/entropy/SSMI are occupancy-only).
///
/// Voxel stores raw semantic evidence, DirVoxel prior-inflated counts: subtract
/// alpha_0 per slot and (C-K)*alpha_0 (clamped at 0) from OTHER; empty slots
/// become sem_cnt 0. a_occ and a_free are copied verbatim.
/// (notes: consensus-rpc-projection-priors)
inline scovox::Voxel projectBetaDirToVoxel(
    const scovox::BetaVoxel& b, const scovox::DirVoxel* d,
    uint16_t num_classes, float alpha_0) {
  scovox::Voxel out{};            // zero-init: a_unk / sem_cnt / sem_cls / tsdf = 0
  out.a_occ  = b.a_occ;
  out.a_free = b.a_free;
  if (d) {
    // Clamp residual_dims at 0 to mirror defaultDirVoxel: with num_classes <=
    // K_TOP the OTHER prior is 0; a negative prior would add phantom unknown
    // mass. (notes: consensus-other-prior-clamp)
    const int residual_dims = static_cast<int>(num_classes) - scovox::K_TOP;
    const float other_prior =
        (residual_dims > 0) ? (static_cast<float>(residual_dims) * alpha_0) : 0.f;
    out.a_unk = std::max(0.f, d->other - other_prior);
    for (int i = 0; i < scovox::K_TOP; ++i) {
      out.sem_cnt[i] = std::max(0.f, d->cnt[i] - alpha_0);
      out.sem_cls[i] = d->cls[i];
    }
  }
  return out;
}

/// Per-cell occupancy refold: start from the Beta(1,1) prior and fold every
/// non-prior source with mergeBeta (nullptr = no voxel). Depends only on the
/// current set of source values, not on how often a snapshot arrived.
/// (notes: consensus-refold-beta)
inline scovox::BetaVoxel refoldBeta(
    const std::vector<const scovox::BetaVoxel*>& sources,
    uint16_t num_classes, float alpha_0) {
  scovox::BetaVoxel fv =
      scovox::defaultBetaVoxel(scovox::kBetaOccPrior, scovox::kBetaFreePrior);
  bool seeded = false;
  for (const auto* sv : sources) {
    if (!sv || isPriorBeta(*sv, num_classes, alpha_0)) continue;
    if (!seeded) { fv = *sv; seeded = true; }
    else         { fv = scovox::mergeBeta(fv, *sv, num_classes, alpha_0); }
  }
  return fv;
}

/// Pure core of the per-cell semantics refold (DirVoxel stream). Mirror of
/// refoldBeta for the Dirichlet stream: reset to the symmetric Dirichlet prior,
/// then fold every non-prior source via mergeDir (slot-reconciling consensus).
inline scovox::DirVoxel refoldDir(
    const std::vector<const scovox::DirVoxel*>& sources,
    uint16_t num_classes, float alpha_0) {
  scovox::DirVoxel fv = scovox::defaultDirVoxel(num_classes, alpha_0);
  bool seeded = false;
  for (const auto* sv : sources) {
    if (!sv || isPriorDir(*sv, num_classes, alpha_0)) continue;
    if (!seeded) { fv = *sv; seeded = true; }
    else         { fv = scovox::mergeDir(fv, *sv, num_classes, alpha_0); }
  }
  return fv;
}

}  // namespace scovox
