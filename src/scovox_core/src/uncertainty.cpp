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

  return H_y - E_H;
}

float semanticEntropy(const Voxel& v) {
  float alphas[K_TOP + 1];
  int K = 0;

  for (int i = 0; i < K_TOP; ++i) {
    if (v.sem_cnt[i] > 0.f) {
      alphas[K++] = v.sem_cnt[i] + 1.f;
    }
  }
  // Query-time Hutter floor on the residual — see uncertainty.hpp.
  alphas[K++] = effectiveResidual(v) + 1.f;

  if (K < 2) return 0.f;

  float a0 = 0.f;
  for (int i = 0; i < K; ++i) a0 += alphas[i];

  float lmvb = -std::lgamma(a0);
  float sum_psi = 0.f;
  for (int i = 0; i < K; ++i) {
    lmvb += std::lgamma(alphas[i]);
    sum_psi += (alphas[i] - 1.f) * digamma(alphas[i]);
  }

  return lmvb + (a0 - static_cast<float>(K)) * digamma(a0) - sum_psi;
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

  return H_y - E_H;
}

float betaKL(const Voxel& a, const Voxel& b) {
  const float a1 = a.a_occ, b1 = a.a_free;
  const float a2 = b.a_occ, b2 = b.a_free;
  if (a1 <= 0.f || b1 <= 0.f || a2 <= 0.f || b2 <= 0.f) return 0.f;

  const float s1 = a1 + b1;
  return lbeta(a2, b2) - lbeta(a1, b1)
       + (a1 - a2) * digamma(a1)
       + (b1 - b2) * digamma(b1)
       + (a2 - a1 + b2 - b1) * digamma(s1);
}

} // namespace scovox
