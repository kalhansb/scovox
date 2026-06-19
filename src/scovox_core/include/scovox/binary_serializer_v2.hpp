#pragma once

/// @file binary_serializer_v2.hpp
/// @brief Wire format v2 for the split-grid SCovox refactor.
///
/// Per Q2 of the design plan: dual-stream framing carrying both TSDF
/// deltas (8 B/voxel) and SemBeta deltas (24 B/voxel) so dSCovox
/// consensus_node can fuse both grids by Curless–Levoy weighted average
/// (TSDF) and Beta + sparse_add (SemBeta) respectively.
///
/// Frame layout (uncompressed):
///   [MAGIC: u32 = "SCVX"] [VERSION: u8 = 2]
///   [resolution: f32]
///   [tsdf_count: u32]
///     ([coord_x: i32][coord_y: i32][coord_z: i32]
///      [distance: f32][weight: f32])  × tsdf_count        // 20 B / voxel
///   [sembeta_count: u32]
///     ([coord_x: i32][coord_y: i32][coord_z: i32]
///      [a_occ: f32][a_free: f32][a_unk: f32]
///      [K: u8 = 2]
///      [cls0: u16][cnt0: f32][cls1: u16][cnt1: f32])
///                                       × sembeta_count   // 37 B / voxel
///
/// The header bytes (`MAGIC` + `VERSION`) are byte-compatible with v1 — a
/// v1 receiver inspecting the version byte can refuse cleanly. v1 senders
/// are NOT modified; this is a separate function in a separate header.
///
/// SemBeta-only mode (the default): `Options{.share_tsdf=false}` writes
/// `tsdf_count=0` and elides the entire TSDF payload (53.3% savings vs.
/// the dual-stream payload at the indoor 0.84 voxel ratio measured in
/// `split_memory_demo`). Each robot keeps a local TSDF for mesh
/// extraction; only SemBeta deltas are shared on the wire. Pass
/// `Options{.share_tsdf=true}` to opt into the dual-stream payload when
/// fused-geometry consensus is needed (TSDF + SemBeta both cross the wire,
/// +54% B/voxel).

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <bonxai/bonxai.hpp>

#include "scovox/sembeta_voxel.hpp"
#include "scovox/tsdf_voxel.hpp"

namespace scovox {

class BinarySerializerV2 {
 public:
  static constexpr uint32_t MAGIC          = 0x53435658;  // "SCVX"
  static constexpr uint8_t  FORMAT_VERSION = 2;

  struct TsdfDelta    { Bonxai::CoordT coord; TsdfVoxel    data; };
  struct SemBetaDelta { Bonxai::CoordT coord; SemBetaVoxel data; };

  struct Frame {
    float resolution = 0.0f;
    std::vector<TsdfDelta>    tsdf_deltas;
    std::vector<SemBetaDelta> sembeta_deltas;
  };

  /// Serializer options. The default `share_tsdf=false` writes
  /// `tsdf_count=0` and skips the TSDF deltas section entirely — this
  /// is the production dSCovox path: each robot keeps its own local
  /// TSDF for mesh extraction, only semantic state crosses the wire.
  /// Pass `share_tsdf=true` to opt into the dual-stream payload when
  /// fused-geometry consensus across robots is needed.
  ///
  /// The wire format itself is identical in both modes; `share_tsdf`
  /// just controls payload elision, so the receiver path needs no
  /// special-casing (an empty TSDF section is the natural deserialize
  /// behaviour).
  struct Options {
    bool share_tsdf = false;
  };

  /// Serialize a frame with default options (`share_tsdf=false`,
  /// SemBeta-only). Convenience overload — opt into the dual-stream
  /// payload by passing `Options{.share_tsdf=true}` explicitly.
  static std::string serialize(const Frame& frame) {
    return serialize(frame, Options{});
  }

  /// Serialize a frame of TSDF + SemBeta deltas into a contiguous byte
  /// buffer suitable for publishing as `bytes` on a ROS topic. The caller
  /// is expected to LZ4-wrap if needed (matching v1 behaviour).
  static std::string serialize(const Frame& frame, const Options& opts) {
    std::string out;
    const std::size_t tsdf_n    = opts.share_tsdf ? frame.tsdf_deltas.size() : 0;
    const std::size_t sembeta_n = frame.sembeta_deltas.size();
    out.reserve(4 + 1 + 4 + 4 + tsdf_n * 20 + 4 + sembeta_n * 37);

    appendBytes(out, &MAGIC, sizeof(MAGIC));
    appendBytes(out, &FORMAT_VERSION, sizeof(FORMAT_VERSION));
    appendBytes(out, &frame.resolution, sizeof(frame.resolution));

    const uint32_t tsdf_count = static_cast<uint32_t>(tsdf_n);
    appendBytes(out, &tsdf_count, sizeof(tsdf_count));
    if (opts.share_tsdf) {
      for (const auto& d : frame.tsdf_deltas) {
        appendBytes(out, &d.coord.x,    sizeof(int32_t));
        appendBytes(out, &d.coord.y,    sizeof(int32_t));
        appendBytes(out, &d.coord.z,    sizeof(int32_t));
        appendBytes(out, &d.data.distance, sizeof(float));
        appendBytes(out, &d.data.weight,   sizeof(float));
      }
    }

    const uint32_t sembeta_count = static_cast<uint32_t>(sembeta_n);
    appendBytes(out, &sembeta_count, sizeof(sembeta_count));
    for (const auto& d : frame.sembeta_deltas) {
      appendBytes(out, &d.coord.x, sizeof(int32_t));
      appendBytes(out, &d.coord.y, sizeof(int32_t));
      appendBytes(out, &d.coord.z, sizeof(int32_t));
      appendBytes(out, &d.data.a_occ,  sizeof(float));
      appendBytes(out, &d.data.a_free, sizeof(float));
      appendBytes(out, &d.data.a_unk,  sizeof(float));
      const uint8_t k = static_cast<uint8_t>(K_TOP);
      appendBytes(out, &k, sizeof(k));
      for (int i = 0; i < K_TOP; ++i) {
        appendBytes(out, &d.data.sem_cls[i], sizeof(uint16_t));
        appendBytes(out, &d.data.sem_cnt[i], sizeof(float));
      }
    }

    return out;
  }

  /// Deserialize a v2 frame. Throws std::runtime_error on bad magic /
  /// version. The caller is expected to LZ4-unwrap before calling.
  static Frame deserialize(const std::string& data) {
    Frame f;
    std::size_t off = 0;
    auto need = [&](std::size_t n) {
      if (off + n > data.size())
        throw std::runtime_error("BinarySerializerV2: truncated frame");
    };

    need(sizeof(MAGIC));
    uint32_t magic = 0;
    std::memcpy(&magic, data.data() + off, sizeof(magic));
    off += sizeof(magic);
    if (magic != MAGIC)
      throw std::runtime_error("BinarySerializerV2: bad MAGIC");

    need(sizeof(FORMAT_VERSION));
    uint8_t version = 0;
    std::memcpy(&version, data.data() + off, sizeof(version));
    off += sizeof(version);
    if (version != FORMAT_VERSION)
      throw std::runtime_error("BinarySerializerV2: bad VERSION");

    need(sizeof(float));
    std::memcpy(&f.resolution, data.data() + off, sizeof(float));
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
    uint32_t sembeta_count = 0;
    std::memcpy(&sembeta_count, data.data() + off, sizeof(sembeta_count));
    off += sizeof(sembeta_count);
    f.sembeta_deltas.reserve(sembeta_count);
    for (uint32_t i = 0; i < sembeta_count; ++i) {
      need(12 + 12 + 1 + K_TOP * (sizeof(uint16_t) + sizeof(float)));
      SemBetaDelta d{};
      std::memcpy(&d.coord.x,    data.data() + off, sizeof(int32_t)); off += sizeof(int32_t);
      std::memcpy(&d.coord.y,    data.data() + off, sizeof(int32_t)); off += sizeof(int32_t);
      std::memcpy(&d.coord.z,    data.data() + off, sizeof(int32_t)); off += sizeof(int32_t);
      std::memcpy(&d.data.a_occ, data.data() + off, sizeof(float));   off += sizeof(float);
      std::memcpy(&d.data.a_free, data.data() + off, sizeof(float));  off += sizeof(float);
      std::memcpy(&d.data.a_unk, data.data() + off, sizeof(float));   off += sizeof(float);
      uint8_t k = 0;
      std::memcpy(&k, data.data() + off, sizeof(uint8_t));            off += sizeof(uint8_t);
      // Tolerate K mismatch: read min(k, K_TOP) entries.
      for (int j = 0; j < k; ++j) {
        if (j < K_TOP) {
          std::memcpy(&d.data.sem_cls[j], data.data() + off, sizeof(uint16_t));
          off += sizeof(uint16_t);
          std::memcpy(&d.data.sem_cnt[j], data.data() + off, sizeof(float));
          off += sizeof(float);
        } else {
          off += sizeof(uint16_t) + sizeof(float);  // skip
        }
      }
      // Empty-slot sentinels for any unfilled K_TOP entries.
      for (int j = k; j < K_TOP; ++j) {
        d.data.sem_cls[j] = 0xFFFF;
        d.data.sem_cnt[j] = 0.0f;
      }
      f.sembeta_deltas.push_back(d);
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
