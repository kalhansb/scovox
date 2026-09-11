#pragma once
/// \file sem_obs.hpp
/// A semantic observation, prepared once per ray instead of once per voxel.

#include <algorithm>
#include <cstdint>
#include <vector>

#include "scovox/voxel.hpp"  // K_TOP

namespace scovox {

/// One class of an observation: the class id and its own probability.
struct SemObsEntry {
  uint16_t cls;
  float    p;
};

/// The observation the deposit path reads: the zeros dropped, the
/// normalisation already applied, and the argmax already located.
///
/// WHY THIS EXISTS. The deposit used to open on the raw dense softmax and
/// re-derive the same three things at every voxel the ray deposited into: an
/// O(C) pass to sum the positive entries, the normaliser built from that sum,
/// and — under HARD — a second O(C) pass to find the argmax. None of the three
/// depends on the voxel, and a ray deposits into far more voxels than the frame
/// has pixels, so all three belong on the ray. Dropping the zeros while we are
/// here costs nothing extra and shortens the one pass that does remain.
///
/// ORDER IS PART OF THE CONTRACT. Entries stay in ascending class order, which
/// is the order the dense scan visited them in, so the sequence of deposits a
/// voxel sees — and therefore which class wins a contested slot — is unchanged.
struct SemObs {
  /// Selection-buffer bound for `top_k`, mirroring TopkProvider::kMaxTrunc.
  static constexpr int kMaxTopK = 16;

  std::vector<SemObsEntry> e;           ///< ascending class order
  int                      argmax{-1};  ///< index into `e`, ties to lowest class

  /// Whether the caller supplied a softmax at all, as opposed to supplying one
  /// with no positive class in it. The endpoint deposit does not distinguish
  /// the two: `commitHit` calls `dirichletUpdate` whenever the return passes
  /// the occupancy gate, and that adds `class_share` to `s_total` before it
  /// looks at `e`, so an absent observation still allocates the DirVoxel and
  /// grows OTHER exactly as an all-zero one does. Only the off-endpoint paths
  /// read this flag: the semantic band, the spread kernel and the ray spread
  /// each return before depositing when it is false. Only `e` can tell you
  /// what was attributed; only this can tell you whether a look happened.
  bool                     present{false};

  bool empty() const { return e.empty(); }
  void clear() { e.clear(); argmax = -1; present = false; }

  /// Point `argmax` at the largest entry; a tie goes to the lower class id,
  /// which is how the dense argmax scan this replaces broke its own ties.
  ///
  /// Call this BEFORE normalising. Scaling by a positive constant cannot
  /// reorder two entries, but it can round them together, and a pair that ties
  /// only after scaling would hand the argmax to the lower class where the raw
  /// comparison gave it to the higher one.
  void locateArgmax() {
    argmax = e.empty() ? -1 : 0;
    for (int i = 1; i < static_cast<int>(e.size()); ++i)
      if (e[i].p > e[argmax].p) argmax = i;
  }
};

/// Build `out` from a dense per-class softmax.
///
/// `top_k <= 0` keeps every class with positive probability, which reproduces
/// the dense path's arithmetic exactly: the same additions in the same order
/// give the same `sum_p`, and every surviving class keeps the probability the
/// dense path would have handed it.
///
/// `top_k > 0` keeps only the `top_k` largest, which is TopkProvider's
/// truncation rule moved to where it can also save the scan. The dropped mass
/// is deliberately NOT renormalized onto the survivors: `norm` only ever
/// shrinks a sum that exceeds 1, so a truncated observation reads as "these
/// classes, and I decline to guess about the rest" and the remainder becomes
/// OTHER, rather than reading as a more confident version of the same
/// distribution.
inline void prepareSemObs(const std::vector<float>* raw, int top_k, SemObs& out) {
  out.clear();
  if (!raw || raw->empty()) return;
  out.present = true;
  const std::vector<float>& p = *raw;

  // 1. Drop the zeros, ascending class order.
  for (size_t i = 0; i < p.size(); ++i)
    if (p[i] > 0.f) out.e.push_back({static_cast<uint16_t>(i), p[i]});
  if (out.e.empty()) return;  // no class has positive probability

  // 2. Truncate to the `top_k` largest. Selection is by repeated strict max so
  //    a tie goes to the lower class id, the same way the argmax scan this
  //    replaces broke its ties, and the survivors are then put back in
  //    ascending class order.
  const int n_in = static_cast<int>(out.e.size());
  if (top_k > 0 && n_in > top_k) {
    const int k = std::min(top_k, SemObs::kMaxTopK);
    int sel[SemObs::kMaxTopK];
    int n = 0;
    while (n < k) {
      int best = -1;
      for (int i = 0; i < n_in; ++i) {
        bool taken = false;
        for (int j = 0; j < n; ++j)
          if (sel[j] == i) { taken = true; break; }
        if (taken) continue;
        if (best < 0 || out.e[i].p > out.e[best].p) best = i;
      }
      sel[n++] = best;
    }
    std::sort(sel, sel + n);
    for (int j = 0; j < n; ++j) out.e[j] = out.e[sel[j]];
    out.e.resize(static_cast<size_t>(n));
  }

  // 3. Locate the argmax on the raw probabilities, which is the comparison the
  //    dense path made, and only then normalise.
  out.locateArgmax();
  float sum_p = 0.f;
  for (const SemObsEntry& x : out.e) sum_p += x.p;
  if (sum_p <= 0.f) { out.clear(); out.present = true; return; }
  const float norm = (sum_p > 1.0f) ? (1.0f / sum_p) : 1.0f;
  for (SemObsEntry& x : out.e) x.p *= norm;
}

}  // namespace scovox
