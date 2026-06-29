#include "scovox/uncertainty.hpp"
#include <cmath>
#include <limits>

namespace scovox {

namespace {
inline float lbeta(float a, float b) {
  return std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
}
} // anonymous namespace

float digamma(float xf) {
  if (xf <= 0.f) return -std::numeric_limits<float>::infinity();

  double x = static_cast<double>(xf);
  double result = 0.0;

  while (x < 6.0) {
    result -= 1.0 / x;
    x += 1.0;
  }

  const double inv  = 1.0 / x;
  const double inv2 = inv * inv;
  result += std::log(x) - 0.5 * inv
          - inv2 * (1.0/12.0 - inv2 * (1.0/120.0 - inv2 * (1.0/252.0)));

  return static_cast<float>(result);
}

float variance(const Voxel& v) {
  const float a = v.a_occ;
  const float b = v.a_free;
  const float s = a + b;
  if (s <= 0.f) return 0.f;
  return (a * b) / (s * s * (s + 1.f));
}

float entropy(const Voxel& v) {
  // NOTE: this is the Beta *differential* entropy (closed form below), kept
  // verbatim for the legacy fused `scovox::Map` path so existing fused-path consumers
  // and tests retain their historical scale. It is UNBOUNDED BELOW: for a
  // near-point-mass voxel (e.g. a_occ=100, a_free=alpha_0=0.01, which survives
  // the a<=0/b<=0 guard) it returns a large NEGATIVE value, NOT a bounded
  // [0, ln2] occupancy uncertainty. Do NOT treat the result as Bernoulli/Shannon
  // entropy. Consumers that need a bounded occupancy-uncertainty signal must
  // compute Bernoulli H(p_occ) = -p ln p - (1-p) ln(1-p) at the call site
  // (as expectedInformationGain does for its H_y term, and as occupancy
  // map-stats mean-entropy aggregators do); we intentionally do not
  // change this function's semantics to avoid a silent behavior regression.
  const float a = v.a_occ;
  const float b = v.a_free;
  if (a <= 0.f || b <= 0.f) return 0.f;

  const float s = a + b;
  return lbeta(a, b)
       - (a - 1.f) * digamma(a)
       - (b - 1.f) * digamma(b)
       + (s - 2.f) * digamma(s);
}

float expectedInformationGain(const Voxel& v) {
  const float a = v.a_occ;
  const float b = v.a_free;
  const float s = a + b;
  if (s <= 0.f) return 0.f;
  const float p = a / s;

  float H_y = 0.f;
  if (p > 1e-7f && p < 1.f - 1e-7f)
    H_y = -p * std::log(p) - (1.f - p) * std::log(1.f - p);

  const float E_H = digamma(s + 1.f)
                   - p        * digamma(a + 1.f)
                   - (1.f - p) * digamma(b + 1.f);

  // EIG is a mutual information and must be >= 0. Near saturation (p -> 0 or 1)
  // the bounded H_y term is clamped to 0 at the 1e-7 boundary while E_H stays
  // strictly positive, so the raw H_y - E_H goes slightly negative (e.g.
  // ~-8e-3 at Beta(1000,1)). A negative EIG mis-ranks saturated voxels in the
  // next-best-view / frontier scorers and poisons mean_eig. Clamp to 0, matching
  // the bernoulliKL noise clamp and the EIGAlwaysNonNegative test contract.
  return std::max(0.f, H_y - E_H);
}

float semanticEntropy(const Voxel& v) {
  // Discrete categorical (Shannon) entropy over the mean Dirichlet
  // probabilities p_i = alpha_i / a0, bounded in [0, ln(K)].
  //
  // We deliberately do NOT return the Dirichlet *differential* entropy here:
  // like the Beta differential entropy in entropy() it is unbounded below and
  // dives to large negatives on concentrated/heavily-observed voxels (e.g.
  // sem_cnt={1000,0} → Dirichlet(1001, …) gives a strongly negative value),
  // which is meaningless as a per-voxel "semantic uncertainty" and poisons any
  // map-level mean. The plug-in mean-probability Shannon entropy is the bounded
  // categorical uncertainty downstream consumers (labelling/ranking) expect and
  // is monotone in how peaked the categorical is, matching the documented
  // contract of the existing semanticEntropy tests.
  float alphas[K_TOP + 1];
  int K = 0;

  for (int i = 0; i < K_TOP; ++i) {
    if (v.sem_cnt[i] > 0.f) {
      alphas[K++] = v.sem_cnt[i] + 1.f;
    }
  }
  // Query-time Hutter floor on the residual — see uncertainty.hpp. This is
  // already bounded by N (no effectiveResidual blow-up leaks into an alpha).
  alphas[K++] = effectiveResidual(v) + 1.f;

  if (K < 2) return 0.f;

  float a0 = 0.f;
  for (int i = 0; i < K; ++i) a0 += alphas[i];
  if (a0 <= 0.f) return 0.f;

  float H = 0.f;
  for (int i = 0; i < K; ++i) {
    const float p = alphas[i] / a0;
    if (p > 1e-7f) H -= p * std::log(p);  // skip p≈0 to avoid 0·log0 = NaN
  }
  return H;  // bounded in [0, ln(K)] ⊆ [0, ln(K_TOP+1)]
}

float semanticVariance(const Voxel& v, uint16_t class_id) {
  // Match semanticEntropy: the Voxel stores raw evidence and the Dirichlet
  // +1 prior is added at query time (see voxel.hpp::defaultVoxel doc). The
  // categorical includes only observed slots plus the unknown bucket — the
  // same K used by semanticEntropy. Without this, entropy and variance
  // disagreed on under-observed cells (entropy used Dirichlet(c+1, ...)
  // while variance used Dirichlet(c, ...)) and downstream consumers got
  // mutually inconsistent uncertainty signals.
  float a_k = 0.f;
  bool found = false;
  for (int i = 0; i < K_TOP; ++i) {
    if (v.sem_cnt[i] > 0.f && v.sem_cls[i] == class_id) {
      a_k = v.sem_cnt[i] + 1.f;
      found = true;
      break;
    }
  }
  if (!found) return 0.f;

  // Query-time Hutter floor on the residual — see uncertainty.hpp.
  float a0 = effectiveResidual(v) + 1.f;
  for (int i = 0; i < K_TOP; ++i) {
    if (v.sem_cnt[i] > 0.f) a0 += v.sem_cnt[i] + 1.f;
  }
  if (a0 <= 0.f) return 0.f;

  return (a_k * (a0 - a_k)) / (a0 * a0 * (a0 + 1.f));
}

namespace {
inline float bernoulliKL(float p, float q) {
  // KL(Bern(p) || Bern(q)) — always >= 0 by Gibbs' inequality.
  // Guard against log(0) when p or q are at the boundary.
  if (p < 1e-7f && q < 1e-7f) return 0.f;
  if (p > 1.f - 1e-7f && q > 1.f - 1e-7f) return 0.f;
  float kl = 0.f;
  if (p > 1e-7f)       kl += p * std::log(p / q);
  if (p < 1.f - 1e-7f) kl += (1.f - p) * std::log((1.f - p) / (1.f - q));
  return std::max(kl, 0.f);  // clamp numerical noise
}
} // anonymous namespace

float ssmiOccKL(const Voxel& v) {
  const float a = v.a_occ, b = v.a_free;
  const float s = a + b;
  if (s <= 0.f) return 0.f;
  const float p      = a / s;
  const float p_post = (a + 1.f) / (s + 1.f);
  return bernoulliKL(p, p_post);
}

float ssmiFreeKL(const Voxel& v) {
  const float a = v.a_occ, b = v.a_free;
  const float s = a + b;
  if (s <= 0.f) return 0.f;
  const float p      = a / s;
  const float p_post = a / (s + 1.f);
  return bernoulliKL(p, p_post);
}

// ====================================================================
// SemBetaVoxel-typed overloads (D6 from the resume-grilling pass) —
// bodies are byte-identical to the Voxel versions above; both structs
// expose a_occ / a_free with identical semantics. Separate overloads
// rather than templating to keep the existing library symbol surface
// stable for any downstream linker that explicitly resolves these.
// ====================================================================
float variance(const SemBetaVoxel& v) {
  const float a = v.a_occ;
  const float b = v.a_free;
  const float s = a + b;
  if (s <= 0.f) return 0.f;
  return (a * b) / (s * s * (s + 1.f));
}

float expectedInformationGain(const SemBetaVoxel& v) {
  const float a = v.a_occ;
  const float b = v.a_free;
  const float s = a + b;
  if (s <= 0.f) return 0.f;
  const float p = a / s;

  float H_y = 0.f;
  if (p > 1e-7f && p < 1.f - 1e-7f)
    H_y = -p * std::log(p) - (1.f - p) * std::log(1.f - p);

  const float E_H = digamma(s + 1.f)
                   - p        * digamma(a + 1.f)
                   - (1.f - p) * digamma(b + 1.f);

  // See the Voxel overload: clamp the mutual-information result at 0 so a
  // near-saturated voxel cannot report a (spurious) negative information gain.
  return std::max(0.f, H_y - E_H);
}

float betaKL(const Voxel& a, const Voxel& b) {
  const float a1 = a.a_occ, b1 = a.a_free;
  const float a2 = b.a_occ, b2 = b.a_free;
  if (a1 <= 0.f || b1 <= 0.f || a2 <= 0.f || b2 <= 0.f) return 0.f;

  const float s1 = a1 + b1;
  const float kl = lbeta(a2, b2) - lbeta(a1, b1)
       + (a1 - a2) * digamma(a1)
       + (b1 - b2) * digamma(b1)
       + (a2 - a1 + b2 - b1) * digamma(s1);
  // KL >= 0 by Gibbs; clamp float round-off (the four-term sum can land a hair
  // below 0 for nearly-identical Betas) so callers never see a negative
  // divergence. Matches the bernoulliKL clamp above.
  return std::max(0.f, kl);
}

} // namespace scovox
