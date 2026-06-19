/// @file split_memory_demo.cpp
/// @brief Standalone CLI tool — integrates synthetic camera frames into
/// ScovoxMapSplit and reports per-grid memory split + voxel counts.
///
/// Purpose: produce the "system characteristics" number for the paper
///   TsdfMap   : X MB  (band-only, byte-equivalent to SLIM-VDB TSDF)
///   SemDirMap: Y MB  (SCovox contribution: Beta + sparse Dirichlet)
///   Total     : X + Y MB
///
/// without depending on the ROS layer (Steps 8-9 of the refactor are
/// deferred). Generates a hemispherical fan of rays from a moving camera
/// — voxel topology and density are representative of indoor scenes
/// (Replica-shaped) but the absolute numbers will scale with
/// `--frames` / `--rays-per-frame` / `--max-range` knobs.
///
/// Usage:
///   ./split_memory_demo --frames 100 --rays-per-frame 4096 \
///       --max-range 5.0 --resolution 0.05 --sdf-trunc 0.15
///
/// Default config matches Replica room0 broad shape:
///   resolution = 0.05 m  (Replica eval default)
///   sdf_trunc  = 0.15 m  (3 voxels, SLIM-VDB default)
///   max_range  = 5.0 m   (Replica indoor average)
///   frames     = 200
///   rays/frame = 4096    (= 64 × 64 ray fan, low-res depth-equivalent)

#include <Eigen/Core>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "scovox/scovox_map_split.hpp"

namespace {

struct Args {
  int    frames        = 200;
  int    rays_per_frame = 4096;
  float  max_range     = 5.0f;
  double resolution    = 0.05;
  float  sdf_trunc     = 0.15f;
  uint64_t seed        = 42;
};

Args parseArgs(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    if (i + 1 >= argc) break;
    std::string val = argv[++i];
    if      (key == "--frames")          a.frames         = std::stoi(val);
    else if (key == "--rays-per-frame")  a.rays_per_frame = std::stoi(val);
    else if (key == "--max-range")       a.max_range      = std::stof(val);
    else if (key == "--resolution")      a.resolution     = std::stod(val);
    else if (key == "--sdf-trunc")       a.sdf_trunc      = std::stof(val);
    else if (key == "--seed")            a.seed           = std::stoull(val);
  }
  return a;
}

/// Generate a hemispherical ray fan of unit directions in camera frame.
/// Equirectangular over (theta in [0, π/2], phi in [0, 2π]) — covers a
/// forward-facing cone, not unlike a depth camera's FOV.
std::vector<Eigen::Vector3f> genRayDirs(int n, std::mt19937& rng) {
  std::vector<Eigen::Vector3f> dirs;
  dirs.reserve(n);
  std::uniform_real_distribution<float> u01(0.f, 1.f);
  for (int i = 0; i < n; ++i) {
    // Uniform on hemisphere via inverse CDF.
    const float u = u01(rng), v = u01(rng);
    const float theta = std::acos(1.0f - u * 0.7f);  // limit cone half-angle
    const float phi   = 2.0f * float(M_PI) * v;
    const float st = std::sin(theta), ct = std::cos(theta);
    dirs.emplace_back(st * std::cos(phi), st * std::sin(phi), ct);
  }
  return dirs;
}

}  // namespace

int main(int argc, char** argv) {
  const Args args = parseArgs(argc, argv);

  scovox::ScovoxMapSplit::Params p;
  p.resolution           = args.resolution;
  p.tsdf.sdf_trunc       = args.sdf_trunc;
  p.tsdf.space_carving   = false;            // SLIM-VDB default
  p.semdir.dirichlet_min_p_occ = 0.5f;
  p.semdir.kappa0       = 1.0f;
  p.semdir.range_decay_length  = 50.0f;
  p.semdir.num_classes   = 4;  // matches the demo's 4-class softmax
  scovox::ScovoxMapSplit map(p);

  std::mt19937 rng(args.seed);
  std::uniform_real_distribution<float> jitter(-0.05f, 0.05f);

  // 4-class semantic distribution to populate sparse Dirichlet slots.
  std::vector<float> sem_probs(4, 0.f);

  std::cout << "ScovoxMapSplit memory demo\n";
  std::cout << "  frames=" << args.frames
            << "  rays/frame=" << args.rays_per_frame
            << "  max_range=" << args.max_range << " m"
            << "  resolution=" << args.resolution << " m"
            << "  sdf_trunc=" << args.sdf_trunc << " m\n";

  const auto t0 = std::chrono::steady_clock::now();

  // Move the camera along a small trajectory and fan rays each frame.
  for (int f = 0; f < args.frames; ++f) {
    const float t = float(f) / float(args.frames);
    Eigen::Vector3f origin(2.0f * t,                        // X drifts 0 → 2
                           jitter(rng),                     // Y small
                           1.5f + 0.5f * std::sin(t * 6.28f)); // Z bobbing
    auto dirs = genRayDirs(args.rays_per_frame, rng);
    for (const auto& d : dirs) {
      // Surface at random distance in [1, max_range] m for hits.
      // 5% rays exit at max_range as no-returns.
      std::uniform_real_distribution<float> du(0.f, 1.f);
      const float u = du(rng);
      if (u < 0.05f) {
        Eigen::Vector3f endpoint = origin + d * args.max_range;
        map.integrateMiss(origin, endpoint, /*quality=*/0.5f);
      } else {
        const float depth = 1.0f + (args.max_range - 1.0f) * u;
        Eigen::Vector3f endpoint = origin + d * depth;
        // Class label: function of endpoint position (deterministic per coord).
        std::fill(sem_probs.begin(), sem_probs.end(), 0.f);
        const int cls = std::abs(int(endpoint.x() * 7) +
                                 int(endpoint.y() * 11) +
                                 int(endpoint.z() * 13)) % 4;
        sem_probs[cls] = 1.0f;
        map.integrateHit(origin, endpoint, &sem_probs, /*quality=*/0.9f);
      }
    }
    // Drain touched-set every 10 frames to mimic the per-publish cycle.
    if ((f + 1) % 10 == 0) {
      auto t = map.drainTouchedTsdf();
      auto s = map.drainTouchedSemDir();
      (void)t; (void)s;
    }
  }

  const auto t1 = std::chrono::steady_clock::now();
  const auto dur_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

  // Memory accounting.
  const std::size_t tsdf_voxels   = map.tsdfVoxelCount();
  const std::size_t semdir_voxels = map.semdirVoxelCount();
  const std::size_t tsdf_bytes    = map.tsdfGridBytes();
  const std::size_t semdir_bytes  = map.semdirGridBytes();

  const double tsdf_raw_mb   = double(tsdf_voxels   * sizeof(scovox::TsdfVoxel))   / (1024.0 * 1024.0);
  const double semdir_raw_mb = double(semdir_voxels * sizeof(scovox::SemDirVoxel)) / (1024.0 * 1024.0);
  const double tsdf_total_mb   = double(tsdf_bytes)   / (1024.0 * 1024.0);
  const double semdir_total_mb = double(semdir_bytes) / (1024.0 * 1024.0);

  std::cout << std::fixed << std::setprecision(2);
  std::cout << "\n  integration: " << dur_ms << " ms ("
            << double(args.frames) / (dur_ms / 1000.0) << " FPS)\n";
  std::cout << "\n  TsdfMap:\n";
  std::cout << "    voxels:        " << tsdf_voxels << "\n";
  std::cout << "    raw cells:     " << tsdf_raw_mb << " MB ("
            << sizeof(scovox::TsdfVoxel) << " B/voxel)\n";
  std::cout << "    Bonxai total:  " << tsdf_total_mb << " MB (incl. leaf metadata)\n";

  std::cout << "\n  SemDirMap:\n";
  std::cout << "    voxels:        " << semdir_voxels << "\n";
  std::cout << "    raw cells:     " << semdir_raw_mb << " MB ("
            << sizeof(scovox::SemDirVoxel) << " B/voxel)\n";
  std::cout << "    Bonxai total:  " << semdir_total_mb << " MB (incl. leaf metadata)\n";

  std::cout << "\n  Total:           " << (tsdf_total_mb + semdir_total_mb) << " MB\n";

  if (semdir_voxels > 0) {
    std::cout << "  TSDF/SemDir voxel ratio:  "
              << double(tsdf_voxels) / double(semdir_voxels) << " ("
              << "TSDF is band-only, SemDir is full-ray)\n";
  }
  if (semdir_voxels >= tsdf_voxels) {
    std::cout << "  Sanity (SemDir >= TSDF): PASS\n";
  } else {
    std::cout << "  Sanity (SemDir >= TSDF): FAIL — bug in carve geometry?\n";
    return 1;
  }

  return 0;
}
