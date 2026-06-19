#pragma once

/// @file binary_serializer_v3.hpp
/// @brief Wire format v3 for the unified-Dirichlet SCovox refactor (Step 7.5).
///
/// Carries TSDF deltas (band-only, 20 B/voxel — unchanged from v2) and
/// SemDir deltas (unified Dirichlet, 20 B/voxel at K_TOP=2 — down from
/// v2's 37 B/voxel SemBeta block by collapsing `(a_occ, a_free, a_unk, K,
/// [(cls, cnt) × K])` into `(alpha_free, alpha_other, cnt[K_TOP],
/// cls[K_TOP])`).
///
/// Header gains `num_classes` (u16) and `K_TOP` (u8) so the receiver can
/// reconstruct the symmetric Dirichlet prior `(C − K_TOP − 1) · α_0` for
/// the OTHER bucket during merge. `α_0` itself is encoded too (f32) so
/// the receiver doesn't need to share the sender's launch params.
///
/// Frame layout (uncompressed):
///   [MAGIC: u32 = "SCVX"] [VERSION: u8 = 3]
///   [resolution: f32]
///   [num_classes: u16]   ← C
///   [K_TOP_wire: u8]     ← K_TOP at sender; receiver asserts match
///   [alpha_0_wire: f32]  ← symmetric Dirichlet prior at sender
///   [tsdf_count: u32]
///     ([coord_x: i32][coord_y: i32][coord_z: i32]
///      [distance: f32][weight: f32])  × tsdf_count     // 20 B / voxel
///   [semdir_count: u32]
///     ([coord_x: i32][coord_y: i32][coord_z: i32]
///      [alpha_free: f32][alpha_other: f32]
///      [cnt0: f32][cnt1: f32]    (K_TOP entries; spec frozen at K_TOP=2)
///      [cls0: u16][cls1: u16])
///                                       × semdir_count // 32 B / voxel (=12+20)
///
/// SemDir-only mode (the default, `Options{.share_tsdf=false}`): writes
/// `tsdf_count=0` and elides the entire TSDF payload — same wire-elision
/// scheme as v2. Each robot keeps a local TSDF for mesh extraction; only
/// SemDir deltas are shared. Pass `Options{.share_tsdf=true}` to opt into
/// the dual-stream payload when fused-geometry consensus is needed.
///
/// v1 and v2 readers stay in tree (no deletion). Receivers route on the
/// VERSION byte.

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <bonxai/bonxai.hpp>

#include "scovox/semdir_voxel.hpp"
#include "scovox/tsdf_voxel.hpp"

namespace scovox {

class BinarySerializerV3 {
 public:
  static constexpr uint32_t MAGIC          = 0x53435658;  // "SCVX"
  static constexpr uint8_t  FORMAT_VERSION = 3;

  struct TsdfDelta   { Bonxai::CoordT coord; TsdfVoxel   data; };
  struct SemDirDelta { Bonxai::CoordT coord; SemDirVoxel data; };

  struct Frame {
    float    resolution  = 0.0f;
    uint16_t num_classes = 14;            ///< NYU13 default
    float    alpha_0     = kDefaultDirichletPrior;
    std::vector<TsdfDelta>   tsdf_deltas;
    std::vector<SemDirDelta> semdir_deltas;
  };

  /// Serializer options. The default `share_tsdf=false` writes
  /// `tsdf_count=0` and skips the TSDF deltas section — production
  /// dSCovox path: each robot keeps its own local TSDF for mesh
  /// extraction, only semantic state crosses the wire.
  struct Options {
    bool share_tsdf = false;
  };

  /// Convenience: serialize with default options (SemDir-only).
  static std::string serialize(const Frame& frame) {
    return serialize(frame, Options{});
  }

  static std::string serialize(const Frame& frame, const Options& opts) {
    std::string out;
    const std::size_t tsdf_n   = opts.share_tsdf ? frame.tsdf_deltas.size() : 0;
    const std::size_t semdir_n = frame.semdir_deltas.size();
    const std::size_t per_semdir = 12 /*coord*/
                                 + 4 + 4 /*alpha_free + alpha_other*/
                                 + 4 * K_TOP /*cnt*/
                                 + 2 * K_TOP /*cls*/;
    out.reserve(4 + 1 + 4 + 2 + 1 + 4 + 4 + tsdf_n * 20 + 4 + semdir_n * per_semdir);

    appendBytes(out, &MAGIC, sizeof(MAGIC));
    appendBytes(out, &FORMAT_VERSION, sizeof(FORMAT_VERSION));
    appendBytes(out, &frame.resolution, sizeof(frame.resolution));
    appendBytes(out, &frame.num_classes, sizeof(frame.num_classes));
    const uint8_t k_top_wire = static_cast<uint8_t>(K_TOP);
    appendBytes(out, &k_top_wire, sizeof(k_top_wire));
    appendBytes(out, &frame.alpha_0, sizeof(frame.alpha_0));

    const uint32_t tsdf_count = static_cast<uint32_t>(tsdf_n);
    appendBytes(out, &tsdf_count, sizeof(tsdf_count));
    if (opts.share_tsdf) {
      for (const auto& d : frame.tsdf_deltas) {
        appendBytes(out, &d.coord.x,      sizeof(int32_t));
        appendBytes(out, &d.coord.y,      sizeof(int32_t));
        appendBytes(out, &d.coord.z,      sizeof(int32_t));
        appendBytes(out, &d.data.distance, sizeof(float));
        appendBytes(out, &d.data.weight,   sizeof(float));
      }
    }

    const uint32_t semdir_count = static_cast<uint32_t>(semdir_n);
    appendBytes(out, &semdir_count, sizeof(semdir_count));
    for (const auto& d : frame.semdir_deltas) {
      appendBytes(out, &d.coord.x,         sizeof(int32_t));
      appendBytes(out, &d.coord.y,         sizeof(int32_t));
      appendBytes(out, &d.coord.z,         sizeof(int32_t));
      appendBytes(out, &d.data.alpha_free,  sizeof(float));
      appendBytes(out, &d.data.alpha_other, sizeof(float));
      for (int i = 0; i < K_TOP; ++i) {
        appendBytes(out, &d.data.cnt[i], sizeof(float));
      }
      for (int i = 0; i < K_TOP; ++i) {
        appendBytes(out, &d.data.cls[i], sizeof(uint16_t));
      }
    }

    return out;
  }

  /// Deserialise. Throws on bad MAGIC, bad VERSION, mismatched K_TOP, or
  /// truncated frame. `K_TOP` mismatch is fatal in v3 — unlike v2's
  /// forward-compat skipping, the unified-Dirichlet wire layout depends
  /// on a fixed K_TOP because the cnt[] and cls[] arrays are length-
  /// implicit (no per-record K byte; the header asserts it).
  static Frame deserialize(const std::string& data) {
    Frame f;
    std::size_t off = 0;
    auto need = [&](std::size_t n) {
      if (off + n > data.size())
        throw std::runtime_error("BinarySerializerV3: truncated frame");
    };

    need(sizeof(MAGIC));
    uint32_t magic = 0;
    std::memcpy(&magic, data.data() + off, sizeof(magic));
    off += sizeof(magic);
    if (magic != MAGIC)
      throw std::runtime_error("BinarySerializerV3: bad MAGIC");

    need(sizeof(FORMAT_VERSION));
    uint8_t version = 0;
    std::memcpy(&version, data.data() + off, sizeof(version));
    off += sizeof(version);
    if (version != FORMAT_VERSION)
      throw std::runtime_error("BinarySerializerV3: bad VERSION");

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
          "BinarySerializerV3: K_TOP mismatch — receiver compiled with K_TOP="
          + std::to_string(K_TOP) + ", wire says " + std::to_string(k_top_wire));
    }

    need(sizeof(float));
    std::memcpy(&f.alpha_0, data.data() + off, sizeof(float));
    off += sizeof(float);

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

    need(sizeof(uint32_t));
    uint32_t semdir_count = 0;
    std::memcpy(&semdir_count, data.data() + off, sizeof(semdir_count));
    off += sizeof(semdir_count);
    f.semdir_deltas.reserve(semdir_count);
    const std::size_t per_record = 12 + 8 + 4 * K_TOP + 2 * K_TOP;
    for (uint32_t i = 0; i < semdir_count; ++i) {
      need(per_record);
      SemDirDelta d{};
      std::memcpy(&d.coord.x,         data.data() + off, sizeof(int32_t)); off += sizeof(int32_t);
      std::memcpy(&d.coord.y,         data.data() + off, sizeof(int32_t)); off += sizeof(int32_t);
      std::memcpy(&d.coord.z,         data.data() + off, sizeof(int32_t)); off += sizeof(int32_t);
      std::memcpy(&d.data.alpha_free,  data.data() + off, sizeof(float));   off += sizeof(float);
      std::memcpy(&d.data.alpha_other, data.data() + off, sizeof(float));   off += sizeof(float);
      for (int j = 0; j < K_TOP; ++j) {
        std::memcpy(&d.data.cnt[j], data.data() + off, sizeof(float));
        off += sizeof(float);
      }
      for (int j = 0; j < K_TOP; ++j) {
        std::memcpy(&d.data.cls[j], data.data() + off, sizeof(uint16_t));
        off += sizeof(uint16_t);
      }
      f.semdir_deltas.push_back(d);
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
