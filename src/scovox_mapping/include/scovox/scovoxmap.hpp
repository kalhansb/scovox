#pragma once
/// @file scovoxmap.hpp
/// @brief SCovox Beta-conjugate occupancy + Dirichlet semantic map.
///
/// Depends on scovox_core for Voxel, Params, ray traversal, uncertainty.
///
/// @code
///   #include "scovox/scovoxmap.hpp"
///
///   scovox::Params p;
///   p.resolution = 0.05;
///   scovox::Map map(p);
///
///   Eigen::Vector3f origin(0, 0, 0), hit(3, 0, 0);
///   map.integrateRay(origin, hit);
///
///   scovox::Voxel v = map.getUnionVoxel(hit);
///   float eig = scovox::expectedInformationGain(v);
/// @endcode

#include "scovox/map_interface.hpp"

namespace scovox {

// =====================================================================
// Map
// =====================================================================

class Map {
public:
  using Grid   = Bonxai::VoxelGrid<Voxel>;
  using CoordT = Bonxai::CoordT;

  explicit Map(const Params& p);

  // -----------------------------------------------------------------
  // Core — ray integration
  // -----------------------------------------------------------------

  void integrateRay(const Eigen::Vector3f& origin,
                    const Eigen::Vector3f& hit,
                    bool is_dynamic = false,
                    const std::vector<float>* class_probs = nullptr,
                    float quality = 1.f, float range_w = 1.f, float angle_w = 1.f);

  void integrateRay(const Eigen::Vector3f& origin,
                    const Eigen::Vector3f& hit,
                    std::vector<CoordT>& updated_coords,
                    bool is_dynamic = false,
                    const std::vector<float>* class_probs = nullptr,
                    float quality = 1.f, float range_w = 1.f, float angle_w = 1.f);

  /// Update only the endpoint (occupancy + semantics), no free-space carving.
  void integrateEndpointOnly(const Eigen::Vector3f& hit,
                             bool is_dynamic = false,
                             const std::vector<float>* class_probs = nullptr,
                             float quality = 1.f, float range_w = 1.f, float angle_w = 1.f);

  // -----------------------------------------------------------------
  // Query — single-voxel access
  // -----------------------------------------------------------------

  bool getVoxel(const Eigen::Vector3f& pos, Voxel& out) const;
  Voxel getUnionVoxel(const Eigen::Vector3f& pos) const;

  // -----------------------------------------------------------------
  // Query — bulk iteration
  // -----------------------------------------------------------------

  template <typename Func>
  void forEachVoxel(Func&& cb) const {
    grid_.forEachCell([&](const Voxel& v, const Bonxai::CoordT& c) {
      auto pos = grid_.coordToPos(c);
      cb(v, Eigen::Vector3f(static_cast<float>(pos.x),
                             static_cast<float>(pos.y),
                             static_cast<float>(pos.z)));
    });
  }

  template <typename Func>
  void forEachTransientVoxel(Func&& cb) const {
    transient_grid_.forEachCell([&](const Voxel& v, const Bonxai::CoordT& c) {
      auto pos = transient_grid_.coordToPos(c);
      cb(v, Eigen::Vector3f(static_cast<float>(pos.x),
                             static_cast<float>(pos.y),
                             static_cast<float>(pos.z)));
    });
  }

  // -----------------------------------------------------------------
  // Fusion — multi-robot consensus merge (Beta-conjugate)
  // -----------------------------------------------------------------

  /// Fuse `src` into `dst` under Beta–Dirichlet conjugate addition with a
  /// shared Beta(1, 1) prior subtracted once. Assumes conditional
  /// independence of the two sources given the latent voxel state.
  void consensusMerge(Voxel& dst, const Voxel& src) const;

  // -----------------------------------------------------------------
  // Transient-layer management
  // -----------------------------------------------------------------

  void decayTransientGrid(float decay_rate = 0.8f);
  void clearTransientGrid();

  // -----------------------------------------------------------------
  // Accessors
  // -----------------------------------------------------------------

  const Params& params() const { return params_; }
  Grid&       grid()       { return grid_; }
  const Grid& grid() const { return grid_; }
  Grid&       transientGrid()       { return transient_grid_; }
  const Grid& transientGrid() const { return transient_grid_; }

  // -----------------------------------------------------------------
  // Low-level — exposed for nodes that need direct grid access
  // -----------------------------------------------------------------

  void carve_free(const Eigen::Vector3f& origin, const Eigen::Vector3f& hit);
  void carve_free(const Eigen::Vector3f& origin, const Eigen::Vector3f& hit,
                  std::vector<CoordT>& traversed_coords);
  /// Carve with an explicit range_w (overrides the internal range computation).
  void carve_free(const Eigen::Vector3f& origin, const Eigen::Vector3f& hit,
                  float range_w_override);
  void carve_free(const Eigen::Vector3f& origin, const Eigen::Vector3f& hit,
                  std::vector<CoordT>& traversed_coords, float range_w_override);

private:
  Params params_;
  Grid grid_;
  Grid transient_grid_;
  Grid::Accessor acc_;
  Grid::Accessor transient_acc_;
  mutable Grid::ConstAccessor const_acc_;
  mutable Grid::ConstAccessor const_transient_acc_;

  void update_endpoint(const CoordT& c,
                       const std::vector<float>* class_probs,
                       float quality, float range_w, float angle_w);

  void update_endpoint_on(Grid::Accessor& target_acc, const CoordT& c,
                          const std::vector<float>* class_probs,
                          float quality, float range_w, float angle_w);

  /// Single fused DDA walk for non-dynamic rays:
  ///   origin → posToCoord(hit + sdf_trunc * u)
  /// per-voxel fuses Beta free / Beta occupied + semantics / TSDF in band.
  /// Replaces the legacy two-pass `carve_free` + `update_endpoint`, paying
  /// one DDA per ray. `updated_coords` may be nullptr.
  void fused_integrate_ray_static(const Eigen::Vector3f& origin,
                                  const Eigen::Vector3f& hit,
                                  std::vector<CoordT>* updated_coords,
                                  const std::vector<float>* class_probs,
                                  float quality, float range_w, float angle_w);

  void beta_update_occupied(Voxel* v, float range_w, float angle_w) const;
  void beta_update_free(Voxel* v, float range_w) const;
  void apply_semantics(Voxel* v, const std::vector<float>* class_probs,
                       float quality) const;
  void apply_evidence_saturation(Voxel* v) const;

  inline CoordT posToCoord(const Eigen::Vector3f& p) const {
    return grid_.posToCoord(Eigen::Vector3d(p.x(), p.y(), p.z()));
  }
};

} // namespace scovox
