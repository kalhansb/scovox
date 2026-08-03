#pragma once

/// @file refinement_regions.hpp
/// @brief Refinement-region registry + per-scan anchor re-registration for
/// the fine-resolution TSDF band
/// (docs/design/fine_tsdf_band_dbh_2026_07_30.md).
///
/// A refinement region is a vertical cylinder `(cx, cy, r, z_lo, z_hi)`
/// around a registered trunk. Hits inside `r + margin` route to the fine
/// TSDF lattice in addition to the coarse one. The registry answers
/// "which region contains this endpoint?" in O(1) via a 2D spatial hash
/// over trunk centres — the check runs once per ray hit on the sensor hot
/// path, so no linear scan over regions is acceptable.
///
/// `fitCylinderAnchorShift` is the drift-absorption fit: the 2-DoF
/// horizontal shift that best aligns one scan's in-region endpoints with
/// the region's canonical cylinder. Robust IRLS (Huber) on the radial
/// residual `‖p_xy + Δ − c‖ − r`, with a small damping toward Δ = 0 (the
/// odometry prior) that also conditions thin one-sided arcs whose
/// tangential component is unobservable. See the design doc's
/// "Per-scan anchor re-registration" section for why this makes the fine
/// lattice drift-free relative to its tree.

#include <Eigen/Core>
#include <cmath>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

namespace scovox {

/// One registered refinement region: a vertical cylinder in the map frame.
/// `radius` is the canonical trunk-model radius (the anchor-fit target and
/// the DBH-fit initialisation), WITHOUT the gate margin — the margin is a
/// map-level parameter applied uniformly at registration.
struct RefinementCylinder {
  uint32_t id     = 0;
  float    cx     = 0.f;
  float    cy     = 0.f;
  float    radius = 0.f;
  float    z_lo   = 0.f;
  float    z_hi   = 0.f;
};

class RefinementRegions {
 public:
  /// `cell_size` is the spatial-hash bucket edge in metres. Must exceed the
  /// largest expected gate radius for the bounding-square insert below to
  /// stay a small constant number of cells; 4 m covers any trunk.
  explicit RefinementRegions(float cell_size = 4.0f)
      : cell_(cell_size > 0.1f ? cell_size : 4.0f), inv_cell_(1.f / cell_) {}

  /// Register (or replace, keyed on `cyl.id`) a region. `margin` widens the
  /// gate radius beyond the model radius. Returns the slot index. Slot
  /// indices are stable for the registry's lifetime (removal tombstones,
  /// re-registration of the same id reuses its slot) so per-slot scan
  /// buffers stay valid.
  int add(const RefinementCylinder& cyl, float margin) {
    int idx = indexOf(cyl.id);
    if (idx >= 0) {
      hashRemove(idx);
      slots_[idx] = Slot{cyl, gateRadius(cyl.radius, margin), true};
    } else {
      // Reuse the first tombstone, else append.
      idx = -1;
      for (std::size_t i = 0; i < slots_.size(); ++i)
        if (!slots_[i].active) { idx = static_cast<int>(i); break; }
      if (idx < 0) {
        idx = static_cast<int>(slots_.size());
        slots_.emplace_back();
      }
      slots_[idx] = Slot{cyl, gateRadius(cyl.radius, margin), true};
      ++active_count_;
    }
    hashInsert(idx);
    return idx;
  }

  /// Unregister by id. Returns false if unknown. The slot becomes a
  /// tombstone (index not reused until a later add).
  bool remove(uint32_t id) {
    const int idx = indexOf(id);
    if (idx < 0) return false;
    hashRemove(idx);
    slots_[idx].active = false;
    --active_count_;
    return true;
  }

  /// Update the canonical model of an existing slot in place (the DBH-fit
  /// model-refresh loop). Keeps id and z-band; re-hashes for the new
  /// centre/gate radius.
  void updateModel(int idx, float cx, float cy, float radius, float margin) {
    if (idx < 0 || idx >= static_cast<int>(slots_.size()) ||
        !slots_[idx].active)
      return;
    hashRemove(idx);
    slots_[idx].cyl.cx     = cx;
    slots_[idx].cyl.cy     = cy;
    slots_[idx].cyl.radius = radius;
    slots_[idx].gate_r     = gateRadius(radius, margin);
    hashInsert(idx);
  }

  /// Slot index of the region containing world point (x, y, z), or -1.
  /// One hash lookup + a precise circle/z test per candidate in the bucket.
  /// Regions are not expected to overlap; if they do, the first candidate
  /// wins deterministically (bucket order = insertion order).
  int lookup(float x, float y, float z) const {
    if (active_count_ == 0) return -1;
    const auto it = cells_.find(cellKey(x, y));
    if (it == cells_.end()) return -1;
    for (const int idx : it->second) {
      const Slot& s = slots_[idx];
      if (!s.active) continue;
      if (z < s.cyl.z_lo || z > s.cyl.z_hi) continue;
      const float dx = x - s.cyl.cx, dy = y - s.cyl.cy;
      if (dx * dx + dy * dy <= s.gate_r * s.gate_r) return idx;
    }
    return -1;
  }

  int indexOf(uint32_t id) const {
    for (std::size_t i = 0; i < slots_.size(); ++i)
      if (slots_[i].active && slots_[i].cyl.id == id)
        return static_cast<int>(i);
    return -1;
  }

  bool slotActive(int idx) const {
    return idx >= 0 && idx < static_cast<int>(slots_.size()) &&
           slots_[idx].active;
  }
  const RefinementCylinder& cylinder(int idx) const {
    return slots_[idx].cyl;
  }
  float gateRadius(int idx) const { return slots_[idx].gate_r; }

  std::size_t size()      const noexcept { return active_count_; }
  bool        empty()     const noexcept { return active_count_ == 0; }
  /// Slot count including tombstones — the size scan buffers must have.
  std::size_t slotCount() const noexcept { return slots_.size(); }

  /// Walk active regions as `fn(const RefinementCylinder&, int slot_idx)`.
  template <typename Fn>
  void forEach(Fn&& fn) const {
    for (std::size_t i = 0; i < slots_.size(); ++i)
      if (slots_[i].active) fn(slots_[i].cyl, static_cast<int>(i));
  }

 private:
  struct Slot {
    RefinementCylinder cyl;
    float gate_r = 0.f;
    bool  active = false;
  };

  static float gateRadius(float radius, float margin) {
    // A non-positive model radius (TreeTarget's "treat as a point") still
    // needs a usable gate; the margin alone carries it.
    return std::max(radius, 0.f) + std::max(margin, 0.05f);
  }

  uint64_t cellKey(float x, float y) const {
    const int32_t ix = static_cast<int32_t>(std::floor(x * inv_cell_));
    const int32_t iy = static_cast<int32_t>(std::floor(y * inv_cell_));
    return (static_cast<uint64_t>(static_cast<uint32_t>(ix)) << 32) |
           static_cast<uint64_t>(static_cast<uint32_t>(iy));
  }

  void hashInsert(int idx) {
    const Slot& s = slots_[idx];
    const float r = s.gate_r;
    const int32_t x0 = static_cast<int32_t>(std::floor((s.cyl.cx - r) * inv_cell_));
    const int32_t x1 = static_cast<int32_t>(std::floor((s.cyl.cx + r) * inv_cell_));
    const int32_t y0 = static_cast<int32_t>(std::floor((s.cyl.cy - r) * inv_cell_));
    const int32_t y1 = static_cast<int32_t>(std::floor((s.cyl.cy + r) * inv_cell_));
    for (int32_t ix = x0; ix <= x1; ++ix)
      for (int32_t iy = y0; iy <= y1; ++iy) {
        const uint64_t key =
            (static_cast<uint64_t>(static_cast<uint32_t>(ix)) << 32) |
            static_cast<uint64_t>(static_cast<uint32_t>(iy));
        cells_[key].push_back(idx);
      }
  }

  void hashRemove(int idx) {
    const Slot& s = slots_[idx];
    const float r = s.gate_r;
    const int32_t x0 = static_cast<int32_t>(std::floor((s.cyl.cx - r) * inv_cell_));
    const int32_t x1 = static_cast<int32_t>(std::floor((s.cyl.cx + r) * inv_cell_));
    const int32_t y0 = static_cast<int32_t>(std::floor((s.cyl.cy - r) * inv_cell_));
    const int32_t y1 = static_cast<int32_t>(std::floor((s.cyl.cy + r) * inv_cell_));
    for (int32_t ix = x0; ix <= x1; ++ix)
      for (int32_t iy = y0; iy <= y1; ++iy) {
        const uint64_t key =
            (static_cast<uint64_t>(static_cast<uint32_t>(ix)) << 32) |
            static_cast<uint64_t>(static_cast<uint32_t>(iy));
        auto it = cells_.find(key);
        if (it == cells_.end()) continue;
        auto& v = it->second;
        for (std::size_t i = 0; i < v.size(); ++i)
          if (v[i] == idx) { v.erase(v.begin() + i); break; }
        if (v.empty()) cells_.erase(it);
      }
  }

  std::vector<Slot> slots_;
  std::unordered_map<uint64_t, std::vector<int>> cells_;
  std::size_t active_count_ = 0;
  float cell_, inv_cell_;
};

// =====================================================================
// Per-scan anchor re-registration fit
// =====================================================================

struct AnchorFitParams {
  int   min_points  = 12;    ///< below this, fall back to odometry (Δ = 0)
  int   max_iters   = 10;
  float huber_delta = 0.05f; ///< m; residuals beyond this are down-weighted
  float max_shift   = 0.30f; ///< m; a larger fitted Δ is a mis-fit → reject
  /// Damping toward Δ = 0, as a fraction of the mean per-point information.
  /// The odometry prior: it shrinks the tangential component of Δ on thin
  /// one-sided arcs (where it is barely observable) while leaving the
  /// well-constrained radial component essentially untouched.
  float damping     = 0.10f;
};

/// Fit the 2-DoF horizontal shift Δ minimising
///   Σ_i ρ_huber( ‖p_i + Δ − c‖ − r )
/// over one scan's in-region endpoint XY positions `pts`, against the
/// canonical cylinder centre `c = (cx, cy)` and radius `r`.
///
/// Returns std::nullopt when the fit must not be trusted: too few points,
/// a degenerate normal matrix, or a converged shift beyond `max_shift`
/// (wrong association / not actually the trunk). Callers then integrate
/// uncorrected — losing a scan's correction, never inventing one.
inline std::optional<Eigen::Vector2f> fitCylinderAnchorShift(
    const std::vector<Eigen::Vector2f>& pts,
    const Eigen::Vector2f& c, float r,
    const AnchorFitParams& p = {}) {
  const int n = static_cast<int>(pts.size());
  if (n < p.min_points || r <= 0.f) return std::nullopt;

  Eigen::Vector2f delta(0.f, 0.f);
  for (int iter = 0; iter < p.max_iters; ++iter) {
    // Gauss–Newton with Huber IRLS. Residual e_i = ‖q_i‖ − r with
    // q_i = p_i + Δ − c; Jacobian ∂e/∂Δ = q_iᵀ/‖q_i‖ (unit radial).
    Eigen::Matrix2f H = Eigen::Matrix2f::Zero();
    Eigen::Vector2f g = Eigen::Vector2f::Zero();
    float wsum = 0.f;
    for (const auto& pt : pts) {
      const Eigen::Vector2f q = pt + delta - c;
      const float d = q.norm();
      if (d < 1e-4f) continue;  // on the axis — no radial direction
      const Eigen::Vector2f u = q / d;
      const float e = d - r;
      const float ae = std::fabs(e);
      const float w = (ae <= p.huber_delta) ? 1.f : p.huber_delta / ae;
      H += w * (u * u.transpose());
      g += w * e * u;
      wsum += w;
    }
    if (wsum <= 0.f) return std::nullopt;
    // Levenberg damping toward Δ = 0 (odometry prior), scaled with the
    // total information so its relative strength is sample-size invariant.
    const float lambda = p.damping * wsum * 0.5f;
    H += lambda * Eigen::Matrix2f::Identity();
    // 2×2 solve by hand; reject a (post-damping) near-singular system.
    const float det = H(0, 0) * H(1, 1) - H(0, 1) * H(1, 0);
    if (det < 1e-9f) return std::nullopt;
    const Eigen::Vector2f step(
        (-g.x() * H(1, 1) + g.y() * H(0, 1)) / det,
        ( g.x() * H(1, 0) - g.y() * H(0, 0)) / det);
    delta += step;
    if (step.norm() < 1e-4f) break;
  }

  if (!std::isfinite(delta.x()) || !std::isfinite(delta.y()))
    return std::nullopt;
  if (delta.norm() > p.max_shift) return std::nullopt;
  return delta;
}

}  // namespace scovox
