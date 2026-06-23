#pragma once

/// @file binary_serializer_v4.hpp
/// @brief Wire format v4 for the split Beta/Dirichlet SCovox substrate.
///
/// Triple-stream codec carrying:
///   - TSDF deltas      (20 B/voxel on wire — unchanged from v2/v3, optional)
///   - BetaVoxel deltas (20 B/voxel: coord + a_occ + a_free)
///   - DirVoxel deltas  (28 B/voxel at K_TOP=2: coord + other + cnt[K] + cls[K])
///
/// This is the faithful wire counterpart of `SemSplitMap`: occupancy
/// (`BetaVoxel`) and semantics (`DirVoxel`) cross the wire as SEPARATE streams,
/// so the receiver reconstructs two independent grids and merges each with its
/// own conjugate rule (see consensus_merge_v4.hpp). Unlike a v3 projection it
/// is lossless — `p_occ` and the class distribution are carried independently
/// rather than collapsed into one Dirichlet vector.
///
/// Header mirrors v3 (so the receiver reconstructs priors without sharing
/// launch params) and bumps VERSION to 4. Receivers route on the VERSION byte;
/// v1/v2/v3 stay in tree.
///
/// Frame layout (uncompressed, little-endian):
///   [MAGIC: u32 = "SCVX"] [VERSION: u8 = 4]
///   [resolution: f32]
///   [num_classes: u16]   ← C
///   [K_TOP_wire: u8]     ← K_TOP at sender; receiver asserts match
///   [alpha_0_wire: f32]  ← symmetric Dirichlet prior at sender
///   [tsdf_count: u32]
///     ([x:i32][y:i32][z:i32][distance:f32][weight:f32]) × tsdf_count   // 20 B
///   [beta_count: u32]
///     ([x:i32][y:i32][z:i32][a_occ:f32][a_free:f32])    × beta_count   // 20 B
///   [dir_count: u32]
///     ([x:i32][y:i32][z:i32][other:f32]
///      [cnt0:f32][cnt1:f32] (K_TOP) [cls0:u16][cls1:u16] (K_TOP))
///                                                        × dir_count   // 28 B
///
/// `Options{.share_tsdf=false}` (default) writes `tsdf_count=0` and elides the
/// TSDF payload — same wire-elision scheme as v2/v3; each robot keeps a local
/// TSDF for mesh extraction. Beta and Dir streams are always written (they are
/// the split substrate's payload).
///
/// @warning K_TOP is locked at compile time. The Dir record size scales with
/// it (28 B at K_TOP=2), so sender and receiver MUST be compiled with the same
/// K_TOP. A mismatch is detected during `deserialize` (the K_TOP_wire byte) and
/// throws fatally — there is NO forward/backward compatibility across K_TOP
/// values. If K_TOP must change on the wire, bump FORMAT_VERSION to v5 and add
/// a new serializer; all existing v4 frames become invalid.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <bonxai/bonxai.hpp>

#include "scovox/beta_voxel.hpp"
#include "scovox/dir_voxel.hpp"
#include "scovox/tsdf_voxel.hpp"

namespace scovox {

class BinarySerializerV4 {
 public:
  static constexpr uint32_t MAGIC          = 0x53435658;  // "SCVX" (shared v1–v4)
  static constexpr uint8_t  FORMAT_VERSION = 4;

  struct TsdfDelta { Bonxai::CoordT coord; TsdfVoxel data; };
  struct BetaDelta { Bonxai::CoordT coord; BetaVoxel data; };
  struct DirDelta  { Bonxai::CoordT coord; DirVoxel  data; };

  struct Frame {
    float    resolution  = 0.0f;
    uint16_t num_classes = 14;            ///< NYU13 default
    float    alpha_0     = kDefaultDirichletPrior;
    std::vector<TsdfDelta> tsdf_deltas;
    std::vector<BetaDelta> beta_deltas;
    std::vector<DirDelta>  dir_deltas;
  };

  /// `share_tsdf=false` (default) elides the TSDF stream (tsdf_count=0).
  struct Options {
    bool share_tsdf = false;
  };

  /// Convenience: serialize with default options (Beta+Dir, no TSDF).
  static std::string serialize(const Frame& frame) {
    return serialize(frame, Options{});
  }

  static std::string serialize(const Frame& frame, const Options& opts) {
    std::string out;
    const std::size_t tsdf_n = opts.share_tsdf ? frame.tsdf_deltas.size() : 0;
    const std::size_t beta_n = frame.beta_deltas.size();
    const std::size_t dir_n  = frame.dir_deltas.size();
    const std::size_t per_beta = 12 /*coord*/ + 4 + 4 /*a_occ + a_free*/;
    const std::size_t per_dir  = 12 /*coord*/ + 4 /*other*/
                               + 4 * K_TOP /*cnt*/ + 2 * K_TOP /*cls*/;
    out.reserve(4 + 1 + 4 + 2 + 1 + 4              // header
                + 4 + tsdf_n * 20
                + 4 + beta_n * per_beta
                + 4 + dir_n  * per_dir);

    appendBytes(out, &MAGIC, sizeof(MAGIC));
    appendBytes(out, &FORMAT_VERSION, sizeof(FORMAT_VERSION));
    appendBytes(out, &frame.resolution, sizeof(frame.resolution));
    appendBytes(out, &frame.num_classes, sizeof(frame.num_classes));
    const uint8_t k_top_wire = static_cast<uint8_t>(K_TOP);
    appendBytes(out, &k_top_wire, sizeof(k_top_wire));
    appendBytes(out, &frame.alpha_0, sizeof(frame.alpha_0));

    // TSDF stream (optional).
    const uint32_t tsdf_count = static_cast<uint32_t>(tsdf_n);
    appendBytes(out, &tsdf_count, sizeof(tsdf_count));
    if (opts.share_tsdf) {
      for (const auto& d : frame.tsdf_deltas) {
        appendBytes(out, &d.coord.x,       sizeof(int32_t));
        appendBytes(out, &d.coord.y,       sizeof(int32_t));
        appendBytes(out, &d.coord.z,       sizeof(int32_t));
        appendBytes(out, &d.data.distance, sizeof(float));
        appendBytes(out, &d.data.weight,   sizeof(float));
      }
    }

    // Beta stream (occupancy).
    const uint32_t beta_count = static_cast<uint32_t>(beta_n);
    appendBytes(out, &beta_count, sizeof(beta_count));
    for (const auto& d : frame.beta_deltas) {
      appendBytes(out, &d.coord.x,     sizeof(int32_t));
      appendBytes(out, &d.coord.y,     sizeof(int32_t));
      appendBytes(out, &d.coord.z,     sizeof(int32_t));
      appendBytes(out, &d.data.a_occ,  sizeof(float));
      appendBytes(out, &d.data.a_free, sizeof(float));
    }

    // Dir stream (occupied-class semantics).
    const uint32_t dir_count = static_cast<uint32_t>(dir_n);
    appendBytes(out, &dir_count, sizeof(dir_count));
    for (const auto& d : frame.dir_deltas) {
      appendBytes(out, &d.coord.x,    sizeof(int32_t));
      appendBytes(out, &d.coord.y,    sizeof(int32_t));
      appendBytes(out, &d.coord.z,    sizeof(int32_t));
      appendBytes(out, &d.data.other, sizeof(float));
      for (int i = 0; i < K_TOP; ++i) appendBytes(out, &d.data.cnt[i], sizeof(float));
      for (int i = 0; i < K_TOP; ++i) appendBytes(out, &d.data.cls[i], sizeof(uint16_t));
    }

    return out;
  }

  /// Deserialise. Throws on bad MAGIC, bad VERSION, K_TOP mismatch, or
  /// truncated frame. K_TOP mismatch is fatal (the cnt[]/cls[] arrays are
  /// length-implicit, locked by the header — same contract as v3).
  static Frame deserialize(const std::string& data) {
    Frame f;
    std::size_t off = 0;
    auto need = [&](std::size_t n) {
      if (off + n > data.size())
        throw std::runtime_error("BinarySerializerV4: truncated frame");
    };

    need(sizeof(MAGIC));
    uint32_t magic = 0;
    std::memcpy(&magic, data.data() + off, sizeof(magic));
    off += sizeof(magic);
    if (magic != MAGIC)
      throw std::runtime_error("BinarySerializerV4: bad MAGIC");

    need(sizeof(FORMAT_VERSION));
    uint8_t version = 0;
    std::memcpy(&version, data.data() + off, sizeof(version));
    off += sizeof(version);
    if (version != FORMAT_VERSION)
      throw std::runtime_error("BinarySerializerV4: bad VERSION");

    need(sizeof(float));
    std::memcpy(&f.resolution, data.data() + off, sizeof(float));
    off += sizeof(float);

    need(sizeof(uint16_t));
    std::memcpy(&f.num_classes, data.data() + off, sizeof(uint16_t));
    off += sizeof(uint16_t);

    need(sizeof(uint8_t));
    uint8_t k_top_wire = 0;
    std::memcpy(&k_top_wire, data.data() + off, sizeof(uint8_t));
    off += sizeof(uint8_t);
    if (k_top_wire != static_cast<uint8_t>(K_TOP)) {
      throw std::runtime_error(
          "BinarySerializerV4: K_TOP mismatch — receiver compiled with K_TOP="
          + std::to_string(K_TOP) + ", wire says " + std::to_string(k_top_wire));
    }

    need(sizeof(float));
    std::memcpy(&f.alpha_0, data.data() + off, sizeof(float));
    off += sizeof(float);

    // Validate the header-supplied prior parameters before any voxel is
    // reconstructed from them. consensus_merge_v4 rebuilds the Beta/Dirichlet
    // prior as num_classes·alpha_0 and (num_classes − K_TOP)·alpha_0, so a
    // corrupt header — num_classes < K_TOP, or a non-finite/non-positive
    // alpha_0 — would silently inject negative or NaN mass into the live map
    // (and two re-sent snapshots from the same bad sender agree with each
    // other, so the per-frame equality check cannot catch it).
    if (f.num_classes < static_cast<uint16_t>(K_TOP)) {
      throw std::runtime_error(
          "BinarySerializerV4: num_classes (" + std::to_string(f.num_classes)
          + ") < K_TOP (" + std::to_string(K_TOP) + ")");
    }
    if (!std::isfinite(f.alpha_0) || f.alpha_0 <= 0.f) {
      throw std::runtime_error("BinarySerializerV4: alpha_0 must be finite and > 0");
    }

    // TSDF stream.
    need(sizeof(uint32_t));
    uint32_t tsdf_count = 0;
    std::memcpy(&tsdf_count, data.data() + off, sizeof(tsdf_count));
    off += sizeof(tsdf_count);
    f.tsdf_deltas.reserve(tsdf_count);
    for (uint32_t i = 0; i < tsdf_count; ++i) {
      need(20);
      TsdfDelta d{};
      std::memcpy(&d.coord.x,       data.data() + off, sizeof(int32_t)); off += sizeof(int32_t);
      std::memcpy(&d.coord.y,       data.data() + off, sizeof(int32_t)); off += sizeof(int32_t);
      std::memcpy(&d.coord.z,       data.data() + off, sizeof(int32_t)); off += sizeof(int32_t);
      std::memcpy(&d.data.distance, data.data() + off, sizeof(float));   off += sizeof(float);
      std::memcpy(&d.data.weight,   data.data() + off, sizeof(float));   off += sizeof(float);
      f.tsdf_deltas.push_back(d);
    }

    // Beta stream.
    need(sizeof(uint32_t));
    uint32_t beta_count = 0;
    std::memcpy(&beta_count, data.data() + off, sizeof(beta_count));
    off += sizeof(beta_count);
    f.beta_deltas.reserve(beta_count);
    for (uint32_t i = 0; i < beta_count; ++i) {
      need(20);
      BetaDelta d{};
      std::memcpy(&d.coord.x,     data.data() + off, sizeof(int32_t)); off += sizeof(int32_t);
      std::memcpy(&d.coord.y,     data.data() + off, sizeof(int32_t)); off += sizeof(int32_t);
      std::memcpy(&d.coord.z,     data.data() + off, sizeof(int32_t)); off += sizeof(int32_t);
      std::memcpy(&d.data.a_occ,  data.data() + off, sizeof(float));   off += sizeof(float);
      std::memcpy(&d.data.a_free, data.data() + off, sizeof(float));   off += sizeof(float);
      f.beta_deltas.push_back(d);
    }

    // Dir stream.
    need(sizeof(uint32_t));
    uint32_t dir_count = 0;
    std::memcpy(&dir_count, data.data() + off, sizeof(dir_count));
    off += sizeof(dir_count);
    f.dir_deltas.reserve(dir_count);
    const std::size_t per_dir = 12 + 4 + 4 * K_TOP + 2 * K_TOP;
    for (uint32_t i = 0; i < dir_count; ++i) {
      need(per_dir);
      DirDelta d{};
      std::memcpy(&d.coord.x,    data.data() + off, sizeof(int32_t)); off += sizeof(int32_t);
      std::memcpy(&d.coord.y,    data.data() + off, sizeof(int32_t)); off += sizeof(int32_t);
      std::memcpy(&d.coord.z,    data.data() + off, sizeof(int32_t)); off += sizeof(int32_t);
      std::memcpy(&d.data.other, data.data() + off, sizeof(float));   off += sizeof(float);
      for (int j = 0; j < K_TOP; ++j) {
        std::memcpy(&d.data.cnt[j], data.data() + off, sizeof(float));
        off += sizeof(float);
      }
      for (int j = 0; j < K_TOP; ++j) {
        std::memcpy(&d.data.cls[j], data.data() + off, sizeof(uint16_t));
        off += sizeof(uint16_t);
      }
      f.dir_deltas.push_back(d);
    }

    return f;
  }

 private:
  template <typename T>
  static void appendBytes(std::string& out, const T* p, std::size_t n) {
    out.append(reinterpret_cast<const char*>(p), n);
  }
};

}  // namespace scovox
