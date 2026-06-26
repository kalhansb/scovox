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

#include <algorithm>
#include <cstdint>
#include <vector>

#include "scovox/beta_voxel.hpp"      // BetaVoxel, kBetaOccPrior/Free, defaultBetaVoxel
#include "scovox/dir_voxel.hpp"       // DirVoxel, defaultDirVoxel
#include "scovox/sembeta_voxel.hpp"   // SemBetaVoxel
#include "scovox/voxel.hpp"           // Voxel, K_TOP
#include "scovox/consensus_merge.hpp" // mergeBeta / mergeDir

namespace scovox {

/// Shared at-prior epsilon for the wire receiver isPrior* tests. This MUST
/// match the sender's at-prior emit gate (scovox_node.cpp uses + 1e-4f on the
/// alpha_free / alpha_other / beta priors). If the receiver slop is looser than
/// the sender gate (the old max(0.01, 0.01·α_0)), a barely-observed voxel that
/// the sender deliberately put on the wire — e.g. a_free = α_0 + 0.005, which
/// clears the sender's 1e-4 gate — is classified "at prior" here and silently
/// dropped from the fused map. Keeping the two epsilons identical guarantees
/// every emitted voxel survives the refold.
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
    // RAW-evidence convention: subtract the OTHER bucket's (C-K)*alpha_0 prior
    // (clamped at 0 for C<=K_TOP, matching defaultDirVoxel) just as the RPC
    // projectBetaDirToVoxel does. argmaxClassConfidence / effectiveResidual /
    // semanticVariance all assume a_unk holds raw evicted mass with no prior;
    // leaving the prior in (out.a_unk = d->other) inflated the confidence
    // denominator by (C-K)*alpha_0 and made the published semantic_confidence
    // disagree with the GetRegion RPC for the identical voxel.
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
/// scovox::Voxel for the wire RPC query services (GetRegion / ScoreCandidates /
/// GetOccupancyGrid). The Dir pointer may be null (occupancy-only voxel, or a
/// caller that only needs occupancy — EIG/entropy/SSMI are occupancy-only).
///
/// scovox::Voxel stores RAW semantic evidence (the Dirichlet prior is applied
/// at query time, not in storage), whereas DirVoxel stores prior-inflated
/// counts. So the projection subtracts the per-class α_0 from each tracked
/// slot and the OTHER bucket's (C−K)·α_0 prior, yielding the same raw-evidence
/// convention selectTopKSemantics / argmaxClassConfidence expect — the SEMANTIC
/// fields are byte-identical to what the fused path produced. Empty Dir slots
/// (cnt == α_0) collapse to sem_cnt == 0 and are skipped by every consumer's
/// `sem_cnt > 0` test.
///
/// OCCUPANCY now uses the SAME prior as the fused Voxel: a_occ / a_free are copied
/// verbatim from the BetaVoxel, which ships the symmetric Beta(1,1) prior
/// (a_occ = a_free = 1.0 → prior p_occ = 0.5) — identical to the unified/fused Voxel
/// (defaultVoxel). So there is no longer a prior-induced p_occ / variance / EIG /
/// SSMI gap vs the fused Voxel at the prior. (Historically the split path used a calibrated
/// Beta(C·α_0, α_0) prior, p_occ = C/(C+1) ≈ 0.933; that was switched to
/// Beta(1,1) — see docs/occupancy_prior.md.)
inline scovox::Voxel projectBetaDirToVoxel(
    const scovox::BetaVoxel& b, const scovox::DirVoxel* d,
    uint16_t num_classes, float alpha_0) {
  scovox::Voxel out{};            // zero-init: a_unk / sem_cnt / sem_cls / tsdf = 0
  out.a_occ  = b.a_occ;
  out.a_free = b.a_free;
  if (d) {
    // Clamp residual_dims at 0 to mirror defaultDirVoxel: when num_classes <=
    // K_TOP there are no residual classes, so the OTHER prior is 0 (defaultDir
    // stored other=0). An unclamped (C-K)*alpha_0 < 0 would make the subtraction
    // d->other - other_prior = d->other + |prior| ADD a phantom alpha_0 of
    // unknown mass, skewing every projected voxel's entropy/EIG/argmax.
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

/// Pure core of the per-cell occupancy refold (BetaVoxel stream). Reset the
/// fused cell to the symmetric Beta(1,1) prior, then fold every NON-prior source
/// via mergeBeta (seed-copying the first non-prior source). `sources[i] ==
/// nullptr` means that source has no voxel at this cell; sources at prior are
/// skipped via isPriorBeta (a pure optimisation — folding the prior is a no-op).
///
/// This is the safeguard that makes the incremental refold bit-for-bit
/// equivalent to a from-scratch rebuild and immune to double-counting: the
/// result depends only on the current *set* of source values, never on how many
/// times a snapshot was received. The node calls this with one entry per source
/// grid accessor; tests call it with an explicit pointer list.
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
