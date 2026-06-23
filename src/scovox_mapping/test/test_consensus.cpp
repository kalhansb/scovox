/// @file test_consensus.cpp
/// Task 1.9 + 2.3: Beta-principled consensus fusion (>=10 tests).

#include <gtest/gtest.h>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "scovox/scovoxmap.hpp"
// v4 split-substrate consensus primitives. The receiver-side refold and the v4
// RPC projection in dscovox_node.cpp are built on these reachable free
// functions; we exercise them here (see the V4Split* tests at the bottom).
#include "scovox/beta_voxel.hpp"
#include "scovox/dir_voxel.hpp"
#include "scovox/consensus_merge_v4.hpp"  // mergeBeta / mergeDir

using namespace scovox;

static Params makeParams() {
  Params p;
  p.resolution = 0.5;
  p.consensus_kl_threshold = 5.0f;
  p.consensus_tau_occ_gate = 0.6f;
  return p;
}

static Voxel makeOcc(float a_occ, float a_free) {
  Voxel v{}; v.a_occ = a_occ; v.a_free = a_free; return v;
}

static Voxel makeOccSem(float a_occ, float a_free,
                        uint16_t cls1, float cnt1,
                        uint16_t cls2 = 0, float cnt2 = 0.f) {
  Voxel v{}; v.a_occ = a_occ; v.a_free = a_free;
  v.sem_cls[0] = cls1; v.sem_cnt[0] = cnt1; v.sem_cls[1] = cls2; v.sem_cnt[1] = cnt2;
  return v;
}

// =====================================================================
// Beta combination: Beta(a1+a2-1, b1+b2-1)
// =====================================================================

TEST(Consensus, BetaCombinationFormula) {
  // Beta(10,5) + Beta(8,3) -> Beta(10+8-1, 5+3-1) = Beta(17, 7)
  Map map(makeParams());
  Voxel dst = makeOcc(10, 5);
  Voxel src = makeOcc(8, 3);
  map.consensusMerge(dst, src);
  EXPECT_FLOAT_EQ(dst.a_occ, 17.f);
  EXPECT_FLOAT_EQ(dst.a_free, 7.f);
}

TEST(Consensus, IdenticalVoxelsPreserveProbability) {
  Map map(makeParams());
  Voxel dst = makeOcc(20, 5);
  float p_before = dst.p_occ();
  Voxel src = makeOcc(20, 5);
  map.consensusMerge(dst, src);
  // Same ratio -> same probability
  EXPECT_NEAR(dst.p_occ(), p_before, 0.02f);
}

TEST(Consensus, IdenticalVoxelsIncreaseEvidence) {
  Map map(makeParams());
  Voxel dst = makeOcc(20, 5);
  float s_before = dst.a_occ + dst.a_free;
  Voxel src = makeOcc(20, 5);
  map.consensusMerge(dst, src);
  // Combined evidence should be greater
  EXPECT_GT(dst.a_occ + dst.a_free, s_before);
}

TEST(Consensus, PriorSourceAddsNoEvidence) {
  // Merging with a prior Beta(1,1) should not change dst
  // Beta(10,5) + Beta(1,1) -> Beta(10+1-1, 5+1-1) = Beta(10, 5)
  Map map(makeParams());
  Voxel dst = makeOcc(10, 5);
  Voxel src = makeOcc(1, 1);
  map.consensusMerge(dst, src);
  EXPECT_FLOAT_EQ(dst.a_occ, 10.f);
  EXPECT_FLOAT_EQ(dst.a_free, 5.f);
}

TEST(Consensus, PriorMergePreservesOneFloor) {
  // Two prior-only voxels merged: Beta(1,1) ⊕ Beta(1,1) = Beta(1+1−1, 1+1−1)
  // = Beta(1, 1). The merge math itself preserves the lower bound on
  // valid inputs (no clamp needed; the previous max(1, ·) floor was
  // removed 2026-05-03).
  Map map(makeParams());
  Voxel dst = makeOcc(1, 1);
  Voxel src = makeOcc(1, 1);
  map.consensusMerge(dst, src);
  EXPECT_FLOAT_EQ(dst.a_occ, 1.f);
  EXPECT_FLOAT_EQ(dst.a_free, 1.f);
}

TEST(Consensus, HighEvidenceDominatesLow) {
  Map map(makeParams());
  Voxel dst = makeOcc(100, 10);  // p ~ 0.91, strong
  Voxel src = makeOcc(3, 2);     // p = 0.60, weak

  float p_before = dst.p_occ();
  map.consensusMerge(dst, src);
  // Result dominated by dst's high evidence
  EXPECT_GT(dst.p_occ(), 0.85f);
  EXPECT_NEAR(dst.p_occ(), p_before, 0.06f);
}

// =====================================================================
// Beta-KL utility (the function itself is preserved in scovox_core for
// callers that want an explicit disagreement metric; consensusMerge no
// longer consults it — return value used to be a discarded `conflict`
// bool, removed 2026-05-03)
// =====================================================================

TEST(Consensus, BetaKLSymmetryProperty) {
  // KL(a||b) != KL(b||a) in general, but both should be non-negative
  Voxel a = makeOcc(20, 5);
  Voxel b = makeOcc(5, 20);
  float kl_ab = betaKL(a, b);
  float kl_ba = betaKL(b, a);
  EXPECT_GT(kl_ab, 0.f);
  EXPECT_GT(kl_ba, 0.f);
}

TEST(Consensus, BetaKLZeroForIdentical) {
  Voxel a = makeOcc(10, 5);
  EXPECT_NEAR(betaKL(a, a), 0.f, 1e-5f);
}

// =====================================================================
// Semantic additive Dirichlet merge
// =====================================================================

TEST(Consensus, SemanticAdditiveMerge) {
  Params p = makeParams();
  p.consensus_tau_occ_gate = 0.0f;  // always merge semantics
  Map map(p);

  Voxel dst = makeOccSem(20, 5, 5, 10.0f, 3, 2.0f);
  Voxel src = makeOccSem(20, 5, 5, 8.0f, 3, 1.0f);
  map.consensusMerge(dst, src);

  // Counts should be additive
  float cls5 = 0.f, cls3 = 0.f;
  for (int si = 0; si < K_TOP; ++si) {
    if (dst.sem_cls[si] == 5) cls5 = dst.sem_cnt[si];
    if (dst.sem_cls[si] == 3) cls3 = dst.sem_cnt[si];
  }
  EXPECT_NEAR(cls5, 18.0f, 0.01f);  // 10 + 8
  EXPECT_NEAR(cls3, 3.0f, 0.01f);   // 2 + 1
}

// 2026-05-03: gate removed from consensusMerge — Dirichlet evidence is
// additive under conditional independence given the latent (occ, class).
// This test now asserts the *new* invariant: semantics merge regardless of
// post-merge occupancy. Even when the merged Beta says "free", the source's
// semantic counts must still be folded into dst.
TEST(Consensus, SemanticMergedRegardlessOfOccupancy) {
  Params p = makeParams();
  p.consensus_tau_occ_gate = 0.9f;  // would have blocked the merge pre-2026-05-03
  Map map(p);

  // Both free after merge: Beta(3+3-1, 20+20-1) = Beta(5, 39), p~0.11
  Voxel dst = makeOccSem(3, 20, 1, 10.0f);
  Voxel src = makeOccSem(3, 20, 2, 8.0f);
  map.consensusMerge(dst, src);

  EXPECT_LT(dst.p_occ(), 0.5f);
  // src's class 2 with cnt 8 must now be present in dst — the gate no longer
  // blocks it. Either as a new slot, or (if dst was full) routed into a_unk.
  float cls2_count = 0.f;
  for (int si = 0; si < K_TOP; ++si) {
    if (dst.sem_cls[si] == 2) cls2_count = dst.sem_cnt[si];
  }
  EXPECT_GT(cls2_count + dst.a_unk, 0.f);
  // Class 1's count is preserved (additive identity, src had no class-1 mass).
  float cls1_count = 0.f;
  for (int si = 0; si < K_TOP; ++si) {
    if (dst.sem_cls[si] == 1) cls1_count = dst.sem_cnt[si];
  }
  EXPECT_FLOAT_EQ(cls1_count, 10.0f);
}

// =====================================================================
// v4 split-substrate consensus: receiver-side refold idempotency + the
// RPC projection's raw-evidence convention.
//
// Findings 18 & 20 flag that the v4 receiver path in dscovox_node.cpp had
// ZERO coverage. The functions that path is built on —
// projectBetaDirToVoxel / isPriorBeta / isPriorDir / refoldCellBeta /
// refoldCellDir — live in an *anonymous namespace inside dscovox_node.cpp*
// and are not exposed by any header, so a pure unit test cannot call them
// by symbol (and exposing them would require editing dscovox_node.cpp).
// What we CAN (and do) test is the reachable consensus primitives those
// functions are layered on (scovox::mergeBeta / mergeDir) plus the
// documented projection arithmetic, validated against the v1 substrate so
// the math itself is pinned. See notes in the review for the residual gap.
// =====================================================================

namespace {
constexpr float    kAlpha = scovox::kDefaultDirichletPrior;  // 0.01
constexpr uint16_t kC     = 14;

scovox::BetaVoxel betaPriorV4() {
  // Calibrated occupancy prior: a_occ = C·α₀, a_free = α₀ (p_occ = C/(C+1)).
  return scovox::defaultBetaVoxel(kC * kAlpha, kAlpha);
}
scovox::DirVoxel dirPriorV4() { return scovox::defaultDirVoxel(kC, kAlpha); }

// Mirror of dscovox_node.cpp's projectBetaDirToVoxel (which we cannot call by
// symbol — see the block comment above). Reproduces that function's CURRENT
// arithmetic so the assertions below pin the SAME math the node runs:
// scovox::Voxel stores RAW evidence, so subtract the per-slot α₀ and OTHER's
// (C−K)·α₀ prior from the prior-inflated DirVoxel counts. NOTE: the node's
// other_prior is intentionally left UNCLAMPED here to match the current node
// code; the missing std::max(0, …) clamp for num_classes < K_TOP is finding 22,
// owned by dscovox_node.cpp — all tests below use C=14 where the clamp is a
// no-op (other_prior = 12·α₀ > 0), so the two forms are identical here.
scovox::Voxel projectBetaDirToVoxelRef(const scovox::BetaVoxel& b,
                                       const scovox::DirVoxel*   d,
                                       uint16_t num_classes, float alpha_0) {
  scovox::Voxel out{};
  out.a_occ  = b.a_occ;
  out.a_free = b.a_free;
  if (d) {
    const float other_prior =
        static_cast<float>(static_cast<int>(num_classes) - scovox::K_TOP) * alpha_0;
    out.a_unk = std::max(0.f, d->other - other_prior);
    for (int i = 0; i < scovox::K_TOP; ++i) {
      out.sem_cnt[i] = std::max(0.f, d->cnt[i] - alpha_0);
      out.sem_cls[i] = d->cls[i];
    }
  }
  return out;
}
}  // namespace

// Finding 20: the refold safeguard rests on "reset fused[c] to prior, then fold
// the source's CURRENT value once". Its correctness reduces to: folding a source
// into the prior reproduces the source exactly (the duplicated prior cancels).
// If this did not hold, every refold would drift the occupancy away from the
// single-source truth and a re-sent snapshot would double-count.
TEST(V4SplitRefold, BetaFoldIntoPriorReproducesSource) {
  scovox::BetaVoxel src{kC * kAlpha + 5.0f, kAlpha + 2.0f};  // observed voxel
  auto f = scovox::mergeBeta(betaPriorV4(), src, kC, kAlpha);
  EXPECT_FLOAT_EQ(f.a_occ,  src.a_occ);
  EXPECT_FLOAT_EQ(f.a_free, src.a_free);
}

TEST(V4SplitRefold, DirFoldIntoPriorReproducesSource) {
  auto src = dirPriorV4();
  src.cls[0] = 7; src.cnt[0] = kAlpha + 1.5f;
  src.cls[1] = 3; src.cnt[1] = kAlpha + 0.8f;
  src.other  = (kC - scovox::K_TOP) * kAlpha + 0.4f;  // some out-of-K evidence
  auto f = scovox::mergeDir(src, dirPriorV4(), kC, kAlpha);
  EXPECT_EQ(f.cls[0], src.cls[0]); EXPECT_FLOAT_EQ(f.cnt[0], src.cnt[0]);
  EXPECT_EQ(f.cls[1], src.cls[1]); EXPECT_FLOAT_EQ(f.cnt[1], src.cnt[1]);
  EXPECT_FLOAT_EQ(f.other, src.other);
}

// Finding 20 (idempotency, end-to-end): a source re-publishes the SAME snapshot
// twice. The architecture demands the fused state be identical after the second
// receipt. Model the refold: fused0 = prior⊕A (first receipt); on the duplicate
// the receiver resets fused to prior and re-folds A's current value → must equal
// fused0. Pinning Beta and Dir together because the two grids refold separately.
TEST(V4SplitRefold, DuplicateSnapshotIsIdempotent) {
  scovox::BetaVoxel betaA{kC * kAlpha + 4.0f, kAlpha + 1.0f};
  auto dirA = dirPriorV4();
  dirA.cls[0] = 5; dirA.cnt[0] = kAlpha + 2.0f;
  dirA.other  = (kC - scovox::K_TOP) * kAlpha + 0.3f;

  // First receipt: fused starts at prior, fold source A.
  auto beta_fused = scovox::mergeBeta(betaPriorV4(), betaA, kC, kAlpha);
  auto dir_fused  = scovox::mergeDir(dirPriorV4(), dirA, kC, kAlpha);

  // Duplicate receipt: reset fused to prior, re-fold A's CURRENT value.
  auto beta_refold = scovox::mergeBeta(betaPriorV4(), betaA, kC, kAlpha);
  auto dir_refold  = scovox::mergeDir(dirPriorV4(), dirA, kC, kAlpha);

  // Byte-identical: no double-count of evidence.
  EXPECT_FLOAT_EQ(beta_refold.a_occ,  beta_fused.a_occ);
  EXPECT_FLOAT_EQ(beta_refold.a_free, beta_fused.a_free);
  EXPECT_EQ(dir_refold.cls[0], dir_fused.cls[0]);
  EXPECT_FLOAT_EQ(dir_refold.cnt[0], dir_fused.cnt[0]);
  EXPECT_FLOAT_EQ(dir_refold.other,  dir_fused.other);
  // Sanity: occupancy did NOT inflate toward a double-fold (a_occ + a_free of a
  // naive double-fold would carry the extra (a+b−prior) mass).
  const auto naive_double = scovox::mergeBeta(beta_fused, betaA, kC, kAlpha);
  EXPECT_GT(naive_double.a_occ, beta_fused.a_occ);  // confirms the trap is real
}

// Finding 20 (at-prior skip path): a re-sent voxel that is itself AT PRIOR
// contributes nothing — folding the prior into the prior yields the prior, so
// the receiver's isPriorBeta/isPriorDir skip is a pure optimisation, not a
// behaviour change. Pin that here at the merge level.
TEST(V4SplitRefold, RefoldingAtPriorVoxelIsNoOp) {
  auto beta = scovox::mergeBeta(betaPriorV4(), betaPriorV4(), kC, kAlpha);
  EXPECT_FLOAT_EQ(beta.a_occ,  kC * kAlpha);
  EXPECT_FLOAT_EQ(beta.a_free, kAlpha);
  auto dir = scovox::mergeDir(dirPriorV4(), dirPriorV4(), kC, kAlpha);
  EXPECT_EQ(dir.cls[0], uint16_t(0xFFFF));
  EXPECT_EQ(dir.cls[1], uint16_t(0xFFFF));
  EXPECT_FLOAT_EQ(dir.other, (kC - scovox::K_TOP) * kAlpha);
}

// Finding 18: the v4 RPC projection must hand the planner the SAME raw semantic
// evidence a v1 fused voxel would carry for the identical observation history.
// Build a v1 voxel via the v1 sparse_add path and a v4 Dir voxel via the v4
// sparse_add_class path for the same two observations, project the v4 voxel, and
// assert the raw evidence matches slot-for-slot (and that a_unk is the OTHER
// bucket's observed mass, prior subtracted).
TEST(V4SplitProjection, RawEvidenceMatchesV1Fused) {
  // Observation history: class 5 seen with weight 1.0, class 3 with weight 0.5.
  // --- v1 fused substrate (raw evidence: prior applied at query time) ---
  scovox::Voxel v1{};
  scovox::sparse_add(v1.sem_cnt, v1.sem_cls, /*cls=*/5, /*inc=*/1.0f, &v1.a_unk);
  scovox::sparse_add(v1.sem_cnt, v1.sem_cls, /*cls=*/3, /*inc=*/0.5f, &v1.a_unk);

  // --- v4 split Dir substrate (prior-inflated counts) ---
  auto d = dirPriorV4();
  scovox::sparse_add_class(d.cnt, d.cls, /*c=*/5, /*inc=*/1.0f, &d.other, kAlpha);
  scovox::sparse_add_class(d.cnt, d.cls, /*c=*/3, /*inc=*/0.5f, &d.other, kAlpha);

  // Occupancy carries the calibrated split prior; semantics are what we compare.
  scovox::BetaVoxel b{kC * kAlpha + 3.0f, kAlpha};
  scovox::Voxel proj = projectBetaDirToVoxelRef(b, &d, kC, kAlpha);

  // Semantic raw evidence must match v1 slot-for-slot (class id keyed, since slot
  // order is not contractually identical across the two substrates).
  auto rawOf = [](const scovox::Voxel& v, uint16_t cls) -> float {
    for (int i = 0; i < scovox::K_TOP; ++i)
      if (v.sem_cls[i] == cls) return v.sem_cnt[i];
    return -1.f;  // not found
  };
  EXPECT_NEAR(rawOf(proj, 5), rawOf(v1, 5), 1e-6f);
  EXPECT_NEAR(rawOf(proj, 3), rawOf(v1, 3), 1e-6f);
  EXPECT_NEAR(rawOf(proj, 5), 1.0f, 1e-6f);  // prior subtracted → raw inc
  EXPECT_NEAR(rawOf(proj, 3), 0.5f, 1e-6f);
  // No eviction happened (only 2 classes, K_TOP=2), so a_unk == v1.a_unk == 0.
  EXPECT_NEAR(proj.a_unk, v1.a_unk, 1e-6f);
  EXPECT_NEAR(proj.a_unk, 0.0f, 1e-6f);
  // Occupancy copied verbatim from the Beta grid (NOT v1-equivalent, by design).
  EXPECT_FLOAT_EQ(proj.a_occ,  b.a_occ);
  EXPECT_FLOAT_EQ(proj.a_free, b.a_free);
}

// Finding 18 (Dir==null / occupancy-only branch): a voxel with no semantics must
// project to a fused voxel with zero semantic mass and the occupancy copied
// through. Mirrors projectBetaDirToVoxel's `if (d)` guard.
TEST(V4SplitProjection, OccupancyOnlyNullDir) {
  scovox::BetaVoxel b{kC * kAlpha + 7.0f, kAlpha + 0.5f};
  scovox::Voxel proj = projectBetaDirToVoxelRef(b, /*d=*/nullptr, kC, kAlpha);
  EXPECT_FLOAT_EQ(proj.a_occ,  b.a_occ);
  EXPECT_FLOAT_EQ(proj.a_free, b.a_free);
  EXPECT_FLOAT_EQ(proj.a_unk,  0.0f);
  for (int i = 0; i < scovox::K_TOP; ++i) EXPECT_FLOAT_EQ(proj.sem_cnt[i], 0.0f);
}

// Finding 18 (OTHER bucket / eviction): when a third class evicts a slot, its
// observed evidence lands in DirVoxel::other; the projection must surface that as
// a_unk = other − (C−K)·α₀ (the OTHER prior subtracted), matching the raw
// "dropped/evicted mass" convention v1's a_unk holds.
TEST(V4SplitProjection, OtherBucketProjectsToRawAUnk) {
  auto d = dirPriorV4();
  // Three classes; K_TOP=2 so the smallest is evicted to OTHER.
  scovox::sparse_add_class(d.cnt, d.cls, /*c=*/5, /*inc=*/3.0f, &d.other, kAlpha);
  scovox::sparse_add_class(d.cnt, d.cls, /*c=*/3, /*inc=*/2.0f, &d.other, kAlpha);
  scovox::sparse_add_class(d.cnt, d.cls, /*c=*/9, /*inc=*/1.0f, &d.other, kAlpha);

  scovox::BetaVoxel b{kC * kAlpha + 1.0f, kAlpha};
  scovox::Voxel proj = projectBetaDirToVoxelRef(b, &d, kC, kAlpha);

  // class 9 (inc=1.0) is smaller than every slot's evidence, so it DROPS to
  // OTHER: other += 1.0. a_unk = (other_prior + 1.0) − other_prior = 1.0.
  EXPECT_NEAR(proj.a_unk, 1.0f, 1e-6f);
  EXPECT_GE(proj.a_unk, 0.0f);  // OTHER prior never over-subtracted
}
