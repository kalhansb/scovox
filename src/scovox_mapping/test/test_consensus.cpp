/// @file test_consensus.cpp
/// Task 1.9 + 2.3: Beta-principled consensus fusion (>=10 tests).

#include <gtest/gtest.h>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "scovox/scovoxmap.hpp"
// Split-substrate consensus primitives. The receiver-side refold and the
// RPC projection in dscovox_node.cpp are built on these reachable free
// functions; we exercise them here (see the Split* tests at the bottom).
#include "scovox/beta_voxel.hpp"
#include "scovox/dir_voxel.hpp"
#include "scovox/consensus_merge.hpp"  // mergeBeta / mergeDir
// The receiver-side split-consensus helpers — projectBetaDirToVoxel /
// isPriorBeta / isPriorDir / refoldBeta / refoldDir — now live in this header
// (hoisted out of dscovox_node.cpp's anonymous namespace), so the Split* tests
// below exercise the SAME code the node runs, not a mirror.
#include "scovox/dscovox_consensus.hpp"

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
// Split-substrate consensus: receiver-side refold idempotency + the
// RPC projection's raw-evidence convention.
//
// Findings 18/19/20 flagged that the receiver path in dscovox_node.cpp had
// ZERO symbol-level coverage. Its helpers — projectBetaDirToVoxel /
// isPriorBeta / isPriorDir + the refold core (refoldBeta / refoldDir) — were
// extracted from dscovox_node.cpp's anonymous namespace into
// scovox/dscovox_consensus.hpp, so the tests below now call the REAL functions
// the node runs (refoldCellBeta/refoldCellDir are thin Bonxai-accessor wrappers
// over refoldBeta/refoldDir). Coverage includes the Dir==null occupancy-only
// branch and the num_classes <= K_TOP edge (finding 19).
// =====================================================================

namespace {
constexpr float    kAlpha = scovox::kDefaultDirichletPrior;  // 0.01
constexpr uint16_t kC     = 14;

scovox::BetaVoxel betaPrior() {
  // Shipped occupancy prior: symmetric Beta(1,1) → p_occ=0.5 (docs/occupancy_prior.md).
  return scovox::defaultBetaVoxel(scovox::kBetaOccPrior, scovox::kBetaFreePrior);
}
scovox::DirVoxel dirPrior() { return scovox::defaultDirVoxel(kC, kAlpha); }
}  // namespace

// Finding 20: the refold safeguard rests on "reset fused[c] to prior, then fold
// each source's CURRENT value once". A single-source refold therefore reproduces
// the source exactly (the reset-to-prior is seed-copied over). These call the
// REAL scovox::refoldBeta / refoldDir cores the node runs (refoldCellBeta/Dir are
// thin Bonxai-accessor wrappers over them).
TEST(SplitRefold, BetaFoldIntoPriorReproducesSource) {
  scovox::BetaVoxel src{kC * kAlpha + 5.0f, kAlpha + 2.0f};  // observed voxel
  auto f = scovox::refoldBeta({&src}, kC, kAlpha);
  EXPECT_FLOAT_EQ(f.a_occ,  src.a_occ);
  EXPECT_FLOAT_EQ(f.a_free, src.a_free);
}

TEST(SplitRefold, DirFoldIntoPriorReproducesSource) {
  auto src = dirPrior();
  src.cls[0] = 7; src.cnt[0] = kAlpha + 1.5f;
  src.cls[1] = 3; src.cnt[1] = kAlpha + 0.8f;
  src.other  = (kC - scovox::K_TOP) * kAlpha + 0.4f;  // some out-of-K evidence
  auto f = scovox::refoldDir({&src}, kC, kAlpha);
  EXPECT_EQ(f.cls[0], src.cls[0]); EXPECT_FLOAT_EQ(f.cnt[0], src.cnt[0]);
  EXPECT_EQ(f.cls[1], src.cls[1]); EXPECT_FLOAT_EQ(f.cnt[1], src.cnt[1]);
  EXPECT_FLOAT_EQ(f.other, src.other);
}

// Finding 20 (idempotency, end-to-end): a source re-publishes the SAME snapshot
// twice. A re-sent snapshot overwrites that source's grid in place — it does NOT
// append a second source — so both receipts refold the same current set {A}, and
// because refoldBeta/refoldDir reset-to-prior before folding, the fused state is
// a pure function of {A} and cannot drift. Pinning Beta and Dir together because
// the two grids refold separately.
TEST(SplitRefold, DuplicateSnapshotIsIdempotent) {
  scovox::BetaVoxel betaA{kC * kAlpha + 4.0f, kAlpha + 1.0f};
  auto dirA = dirPrior();
  dirA.cls[0] = 5; dirA.cnt[0] = kAlpha + 2.0f;
  dirA.other  = (kC - scovox::K_TOP) * kAlpha + 0.3f;

  auto beta_fused  = scovox::refoldBeta({&betaA}, kC, kAlpha);  // first receipt
  auto dir_fused   = scovox::refoldDir({&dirA},  kC, kAlpha);
  auto beta_refold = scovox::refoldBeta({&betaA}, kC, kAlpha);  // duplicate receipt
  auto dir_refold  = scovox::refoldDir({&dirA},  kC, kAlpha);

  // Byte-identical: no double-count of evidence.
  EXPECT_FLOAT_EQ(beta_refold.a_occ,  beta_fused.a_occ);
  EXPECT_FLOAT_EQ(beta_refold.a_free, beta_fused.a_free);
  EXPECT_EQ(dir_refold.cls[0], dir_fused.cls[0]);
  EXPECT_FLOAT_EQ(dir_refold.cnt[0], dir_fused.cnt[0]);
  EXPECT_FLOAT_EQ(dir_refold.other,  dir_fused.other);
  // The single-source refold seed-copies A verbatim — the prior was reset, not
  // folded a second time.
  EXPECT_FLOAT_EQ(beta_fused.a_occ, betaA.a_occ);
  // Contrast: a no-reset implementation that folded A into the already-fused
  // state would inflate occupancy — confirms the reset is what prevents it.
  const auto naive_double = scovox::mergeBeta(beta_fused, betaA, kC, kAlpha);
  EXPECT_GT(naive_double.a_occ, beta_fused.a_occ);  // the trap is real
}

// =====================================================================
// E6.3 — consistency of the receiver under network reordering/duplication.
//
// The idempotence test above is SINGLE-source. The three below cover what the
// merger actually runs: several sources refolded together, in an order the node
// chooses rather than the network. Read together they say:
//
//   Beta  (addition only)          — order-free up to float rounding
//   Dir   at 2 sources             — order-free EXACTLY  (the shipped rig)
//   Dir   at ≥3 sources            — NOT order-free; the node pins the order
//   duplicate receipt, any N       — a no-op, because refold resets first
// =====================================================================

namespace {
// Three sources carrying more than K_TOP distinct classes between them, so the
// mergeDir truncation is live rather than a formality.
scovox::DirVoxel dirSrc(uint16_t c0, float n0, uint16_t c1, float n1) {
  auto v = dirPrior();
  v.cls[0] = c0; v.cnt[0] = kAlpha + n0;
  if (scovox::K_TOP > 1) { v.cls[1] = c1; v.cnt[1] = kAlpha + n1; }
  return v;
}
bool sameDir(const scovox::DirVoxel& x, const scovox::DirVoxel& y) {
  for (int i = 0; i < scovox::K_TOP; ++i)
    if (x.cls[i] != y.cls[i] || x.cnt[i] != y.cnt[i]) return false;
  return x.other == y.other;
}
}  // namespace

// A duplicate snapshot from ONE source while OTHER sources are present. The
// receiver replaces that source's grid entry and refolds the whole set, so the
// duplicate cannot accumulate. The second half pins the trap: an implementation
// that appended the duplicate as an extra source would double-count class 1.
TEST(SplitRefold, DuplicateSnapshotIsIdempotentMultiSource) {
  const auto A = dirSrc(1, 5.0f, 2, 4.0f);
  const auto B = dirSrc(3, 4.5f, 4, 1.0f);
  const auto C = dirSrc(2, 3.0f, 5, 0.5f);

  const auto first  = scovox::refoldDir({&A, &B, &C}, kC, kAlpha);
  const auto second = scovox::refoldDir({&A, &B, &C}, kC, kAlpha);
  EXPECT_TRUE(sameDir(first, second)) << "re-refolding the same source set drifted";

  const auto appended = scovox::refoldDir({&A, &B, &C, &A}, kC, kAlpha);
  EXPECT_FALSE(sameDir(first, appended))
      << "append-the-duplicate trap did not fire — this test proves nothing";

  scovox::BetaVoxel bA{4.0f, 1.5f}, bB{2.0f, 3.0f}, bC{1.2f, 6.0f};
  const auto bfirst  = scovox::refoldBeta({&bA, &bB, &bC}, kC, kAlpha);
  const auto bsecond = scovox::refoldBeta({&bA, &bB, &bC}, kC, kAlpha);
  EXPECT_FLOAT_EQ(bfirst.a_occ,  bsecond.a_occ);
  EXPECT_FLOAT_EQ(bfirst.a_free, bsecond.a_free);
}

// Two sources — the configuration every E6 cell actually ran — is exactly
// order-free, because refoldDir reduces to a single commutative mergeDir call.
TEST(SplitRefold, DirRefoldIsOrderFreeAtTwoSources) {
  const auto A = dirSrc(1, 5.0f, 2, 4.0f);
  const auto B = dirSrc(3, 4.5f, 4, 1.0f);
  const auto ab = scovox::refoldDir({&A, &B}, kC, kAlpha);
  const auto ba = scovox::refoldDir({&B, &A}, kC, kAlpha);
  EXPECT_TRUE(sameDir(ab, ba)) << "two-source refold is order-dependent";
  EXPECT_EQ(scovox::dominantClass(ab, kAlpha, kC),
            scovox::dominantClass(ba, kAlpha, kC));
}

// ⚠ At three or more sources the refold is NOT order-independent, and this test
// PINS THAT LIMITATION rather than asserting it away. mergeDir truncates at each
// pairwise step and a class dumped to OTHER can never climb back, so the fused
// slots depend on which sources met first (consensus_merge.hpp says so in situ).
//
// Fixture: classes 7 and 2 compete for the top slot. Folding A+B first keeps
// {2, 7}, so C's extra 1.5 lands ON class 7 and carries it past class 2 → 7.
// Folding A+C first evicts class 7 (its 1.5 loses to class 4's 2.5 and class 0's
// 1.9), and evicted evidence cannot come back, so class 2 keeps the slot → 2.
// Both are *confident* labels, not abstentions: this is a real label flip, not a
// degradation to OTHER. A 500k-draw random search over 3-source configurations
// put these at ~1.0% of draws (with a further ~13.5% flipping class↔abstain and
// 20.6% differing in the fused slots), so the fixture is representative, not a
// hand-built pathology.
//
// The node's mitigation is to fix the order, NOT to make the merge commutative:
// dscovox_node.cpp sorts sources by id before folding Dir, so the fused result
// is reproducible across runs and rehashes even though it is not permutation-
// invariant. The last two assertions are that actual guarantee.
//
// Scope: every E6 cell ran N=2, where the fold is exactly order-free
// (DirRefoldIsOrderFreeAtTwoSources above), so no campaign result depends on
// this. E6.5's N∈{3,4} scaling arm was dropped 2026-08-06.
//
// If a future change makes mergeDir truly commutative (accumulate every source,
// then truncate once), THIS TEST SHOULD FAIL. Replace it with an equality
// assertion; do not silently relax it.
TEST(SplitRefold, DirRefoldDependsOnSourceOrderAtThreeSources) {
  if (scovox::K_TOP != 2) GTEST_SKIP() << "fixture is written for K_TOP == 2";
  const auto A = dirSrc(2, 9.0f, 4, 2.5f);
  const auto B = dirSrc(7, 8.0f, 1, 3.0f);
  const auto C = dirSrc(7, 1.5f, 0, 1.9f);

  const auto abc = scovox::refoldDir({&A, &B, &C}, kC, kAlpha);
  const auto acb = scovox::refoldDir({&A, &C, &B}, kC, kAlpha);

  // The limitation is semantic, not merely numeric: the reported label changes,
  // and both orders report a real class rather than abstaining.
  EXPECT_EQ(scovox::dominantClass(abc, kAlpha, kC), uint16_t(7));
  EXPECT_EQ(scovox::dominantClass(acb, kAlpha, kC), uint16_t(2));
  EXPECT_FALSE(sameDir(abc, acb));

  // What the node guarantees instead: fold in a fixed order, get a fixed answer.
  EXPECT_TRUE(sameDir(abc, scovox::refoldDir({&A, &B, &C}, kC, kAlpha)));
  EXPECT_TRUE(sameDir(acb, scovox::refoldDir({&A, &C, &B}, kC, kAlpha)));
}

// Beta carries no truncation, so it is order-free semantically — but float
// addition is not associative, so it is NOT bit-exact across fold orders. The
// node folds Beta in unordered_map order (unsorted, unlike Dir), which is safe
// precisely because the spread is rounding-scale: measured at ≤2 ULP
// (relative ~1.5e-7) over 200k random 4-source draws in all 24 orders.
// EXPECT_FLOAT_EQ's 4-ULP tolerance is the right assertion here; EXPECT_EQ on
// the bits would be wrong and would flake.
TEST(SplitRefold, BetaRefoldOrderInvariantToFloatTolerance) {
  scovox::BetaVoxel a{4.0f, 1.5f}, b{2.0f, 3.0f}, c{1.2f, 6.0f};
  const auto abc = scovox::refoldBeta({&a, &b, &c}, kC, kAlpha);
  const auto cab = scovox::refoldBeta({&c, &a, &b}, kC, kAlpha);
  const auto bca = scovox::refoldBeta({&b, &c, &a}, kC, kAlpha);
  EXPECT_FLOAT_EQ(abc.a_occ,  cab.a_occ);
  EXPECT_FLOAT_EQ(abc.a_occ,  bca.a_occ);
  EXPECT_FLOAT_EQ(abc.a_free, cab.a_free);
  EXPECT_FLOAT_EQ(abc.a_free, bca.a_free);
  // Additive with one prior removed per extra source: 4.0+2.0+1.2 − 2·prior.
  EXPECT_FLOAT_EQ(abc.a_occ, 7.2f - 2.0f * scovox::kBetaOccPrior);
}

// Finding 20 (at-prior skip path): a source sitting AT PRIOR contributes nothing
// — refoldBeta/refoldDir skip it via isPriorBeta/isPriorDir — so the fused result
// for sources {A, prior} is identical to {A}, and {prior} alone stays at prior.
// This exercises the REAL isPrior* skip inside the refold core.
TEST(SplitRefold, RefoldingAtPriorVoxelIsNoOp) {
  auto bp = betaPrior();
  auto dp = dirPrior();
  auto beta_prior_only = scovox::refoldBeta({&bp}, kC, kAlpha);
  EXPECT_FLOAT_EQ(beta_prior_only.a_occ,  scovox::kBetaOccPrior);
  EXPECT_FLOAT_EQ(beta_prior_only.a_free, scovox::kBetaFreePrior);
  auto dir_prior_only = scovox::refoldDir({&dp}, kC, kAlpha);
  EXPECT_EQ(dir_prior_only.cls[0], uint16_t(0xFFFF));
  EXPECT_EQ(dir_prior_only.cls[1], uint16_t(0xFFFF));
  EXPECT_FLOAT_EQ(dir_prior_only.other, (kC - scovox::K_TOP) * kAlpha);

  scovox::BetaVoxel betaA{kC * kAlpha + 4.0f, kAlpha + 1.0f};
  auto only_a      = scovox::refoldBeta({&betaA}, kC, kAlpha);
  auto a_and_prior = scovox::refoldBeta({&betaA, &bp}, kC, kAlpha);
  EXPECT_FLOAT_EQ(a_and_prior.a_occ,  only_a.a_occ);
  EXPECT_FLOAT_EQ(a_and_prior.a_free, only_a.a_free);

  auto dirA = dirPrior();
  dirA.cls[0] = 5; dirA.cnt[0] = kAlpha + 2.0f;
  auto only_da      = scovox::refoldDir({&dirA}, kC, kAlpha);
  auto da_and_prior = scovox::refoldDir({&dirA, &dp}, kC, kAlpha);
  EXPECT_EQ(da_and_prior.cls[0], only_da.cls[0]);
  EXPECT_FLOAT_EQ(da_and_prior.cnt[0], only_da.cnt[0]);
  EXPECT_FLOAT_EQ(da_and_prior.other, only_da.other);
}

// Finding 18: the split RPC projection must hand the planner the SAME raw
// semantic evidence the unified fused voxel would carry for the identical
// observation history. Build a unified voxel via the sparse_add path and a
// split Dir voxel via the sparse_add_class path for the same two observations,
// project the split voxel, and assert the raw evidence matches slot-for-slot
// (and that a_unk is the OTHER bucket's observed mass, prior subtracted).
TEST(SplitProjection, RawEvidenceMatchesFused) {
  // Observation history: class 5 seen with weight 1.0, class 3 with weight 0.5.
  // --- unified fused substrate (raw evidence: prior applied at query time) ---
  scovox::Voxel unified{};
  scovox::sparse_add(unified.sem_cnt, unified.sem_cls, /*cls=*/5, /*inc=*/1.0f, &unified.a_unk);
  scovox::sparse_add(unified.sem_cnt, unified.sem_cls, /*cls=*/3, /*inc=*/0.5f, &unified.a_unk);

  // --- split Dir substrate (prior-inflated counts) ---
  auto d = dirPrior();
  scovox::sparse_add_class(d.cnt, d.cls, /*c=*/5, /*inc=*/1.0f, &d.other, kAlpha);
  scovox::sparse_add_class(d.cnt, d.cls, /*c=*/3, /*inc=*/0.5f, &d.other, kAlpha);

  // Occupancy carries the calibrated split prior; semantics are what we compare.
  scovox::BetaVoxel b{kC * kAlpha + 3.0f, kAlpha};
  scovox::Voxel proj = scovox::projectBetaDirToVoxel(b, &d, kC, kAlpha);

  // Semantic raw evidence must match the unified voxel slot-for-slot (class id
  // keyed, since slot order is not contractually identical across substrates).
  auto rawOf = [](const scovox::Voxel& v, uint16_t cls) -> float {
    for (int i = 0; i < scovox::K_TOP; ++i)
      if (v.sem_cls[i] == cls) return v.sem_cnt[i];
    return -1.f;  // not found
  };
  EXPECT_NEAR(rawOf(proj, 5), rawOf(unified, 5), 1e-6f);
  EXPECT_NEAR(rawOf(proj, 3), rawOf(unified, 3), 1e-6f);
  EXPECT_NEAR(rawOf(proj, 5), 1.0f, 1e-6f);  // prior subtracted → raw inc
  EXPECT_NEAR(rawOf(proj, 3), 0.5f, 1e-6f);
  // No eviction happened (only 2 classes, K_TOP=2), so a_unk == unified.a_unk == 0.
  EXPECT_NEAR(proj.a_unk, unified.a_unk, 1e-6f);
  EXPECT_NEAR(proj.a_unk, 0.0f, 1e-6f);
  // Occupancy copied verbatim from the Beta grid (NOT unified-equivalent, by design).
  EXPECT_FLOAT_EQ(proj.a_occ,  b.a_occ);
  EXPECT_FLOAT_EQ(proj.a_free, b.a_free);
}

// Finding 18 (Dir==null / occupancy-only branch): a voxel with no semantics must
// project to a fused voxel with zero semantic mass and the occupancy copied
// through. Mirrors projectBetaDirToVoxel's `if (d)` guard.
TEST(SplitProjection, OccupancyOnlyNullDir) {
  scovox::BetaVoxel b{kC * kAlpha + 7.0f, kAlpha + 0.5f};
  scovox::Voxel proj = scovox::projectBetaDirToVoxel(b, /*d=*/nullptr, kC, kAlpha);
  EXPECT_FLOAT_EQ(proj.a_occ,  b.a_occ);
  EXPECT_FLOAT_EQ(proj.a_free, b.a_free);
  EXPECT_FLOAT_EQ(proj.a_unk,  0.0f);
  for (int i = 0; i < scovox::K_TOP; ++i) EXPECT_FLOAT_EQ(proj.sem_cnt[i], 0.0f);
}

// Finding 18 (OTHER bucket / eviction): when a third class evicts a slot, its
// observed evidence lands in DirVoxel::other; the projection must surface that as
// a_unk = other − (C−K)·α₀ (the OTHER prior subtracted), matching the raw
// "dropped/evicted mass" convention the unified voxel's a_unk holds.
TEST(SplitProjection, OtherBucketProjectsToRawAUnk) {
  auto d = dirPrior();
  // Three classes; K_TOP=2 so the smallest is evicted to OTHER.
  scovox::sparse_add_class(d.cnt, d.cls, /*c=*/5, /*inc=*/3.0f, &d.other, kAlpha);
  scovox::sparse_add_class(d.cnt, d.cls, /*c=*/3, /*inc=*/2.0f, &d.other, kAlpha);
  scovox::sparse_add_class(d.cnt, d.cls, /*c=*/9, /*inc=*/1.0f, &d.other, kAlpha);

  scovox::BetaVoxel b{kC * kAlpha + 1.0f, kAlpha};
  scovox::Voxel proj = scovox::projectBetaDirToVoxel(b, &d, kC, kAlpha);

  // class 9 (inc=1.0) is smaller than every slot's evidence, so it DROPS to
  // OTHER: other += 1.0. a_unk = (other_prior + 1.0) − other_prior = 1.0.
  EXPECT_NEAR(proj.a_unk, 1.0f, 1e-6f);
  EXPECT_GE(proj.a_unk, 0.0f);  // OTHER prior never over-subtracted
}

// Finding 19: num_classes <= K_TOP edge. projectBetaDirToVoxel must clamp the
// OTHER prior (C−K)·α₀ at 0 (matching defaultDirVoxel). At C == K_TOP there are
// no residual classes, so other_prior is 0 and a_unk == OTHER's raw mass.
TEST(SplitProjection, NumClassesAtKTopClampsOtherPrior) {
  constexpr uint16_t cAtK = scovox::K_TOP;            // residual_dims == 0
  auto d = scovox::defaultDirVoxel(cAtK, kAlpha);     // other == 0 (clamped)
  d.other = 0.5f;                                     // pretend some evicted mass
  scovox::BetaVoxel b{2.0f, 1.0f};
  auto proj = scovox::projectBetaDirToVoxel(b, &d, cAtK, kAlpha);
  EXPECT_GE(proj.a_unk, 0.0f);
  EXPECT_FLOAT_EQ(proj.a_unk, 0.5f);  // other_prior clamped to 0 → a_unk == other
}

// Finding 19/22: below K_TOP the unclamped prior (C−K)·α₀ < 0 would ADD a phantom
// α₀ of unknown mass (a_unk = other − negative). The clamp must keep a_unk == other.
TEST(SplitProjection, NumClassesBelowKTopNoPhantomUnknown) {
  constexpr uint16_t cBelowK = 1;                     // (1 − K_TOP) < 0
  auto d = scovox::defaultDirVoxel(cBelowK, kAlpha);
  d.other = 0.5f;
  scovox::BetaVoxel b{2.0f, 1.0f};
  auto proj = scovox::projectBetaDirToVoxel(b, &d, cBelowK, kAlpha);
  EXPECT_FLOAT_EQ(proj.a_unk, 0.5f);  // unclamped would be 0.5 + α₀
}

// Finding 19: isPriorDir at and below the K_TOP edge — a genuine prior voxel must
// still read "at prior" (the OTHER prior clamps to 0, not (C−K)·α₀ < 0), and any
// observed slot must read "not prior".
TEST(SplitPrior, IsPriorDirAtAndBelowKTop) {
  for (uint16_t C : {uint16_t(1), uint16_t(scovox::K_TOP)}) {
    auto prior = scovox::defaultDirVoxel(C, kAlpha);
    EXPECT_TRUE(scovox::isPriorDir(prior, C, kAlpha)) << "C=" << C;
    auto obs = prior;
    scovox::sparse_add_class(obs.cnt, obs.cls, /*c=*/0, /*inc=*/1.0f, &obs.other, kAlpha);
    EXPECT_FALSE(scovox::isPriorDir(obs, C, kAlpha)) << "C=" << C;
  }
}

// Finding 19: isPriorBeta detects the symmetric Beta(1,1) prior and rejects real
// occupancy evidence (the kPriorSlop epsilon admits only a one-quantum tolerance).
TEST(SplitPrior, IsPriorBetaDetectsPriorAndObserved) {
  EXPECT_TRUE(scovox::isPriorBeta(betaPrior(), kC, kAlpha));
  scovox::BetaVoxel observed{kC * kAlpha + 3.0f, kAlpha};
  EXPECT_FALSE(scovox::isPriorBeta(observed, kC, kAlpha));
  scovox::BetaVoxel barely{scovox::kBetaOccPrior + 0.5f, scovox::kBetaFreePrior};
  EXPECT_FALSE(scovox::isPriorBeta(barely, kC, kAlpha));
}
