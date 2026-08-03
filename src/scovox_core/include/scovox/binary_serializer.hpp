#pragma once

/// @file binary_serializer.hpp
/// @brief Wire format for the split Beta/Dirichlet SCovox substrate.
///
/// Triple-stream codec carrying:
///   - TSDF deltas      (20 B/voxel on wire, optional, flat records)
///   - BetaVoxel deltas (block-run coords + u16-quantized evidence)
///   - DirVoxel deltas  (block-run coords + u16-quantized evidence)
///
/// This is the faithful wire counterpart of `SemSplitMap`: occupancy
/// (`BetaVoxel`) and semantics (`DirVoxel`) cross the wire as SEPARATE streams,
/// so the receiver reconstructs two independent grids and merges each with its
/// own conjugate rule (see consensus_merge.hpp).
///
/// Revision 6 (docs/design/comms_design_2026_07_30.md, Part 1) reworks the
/// Beta/Dir record layout only — the merge rules, the priors, and the shaper
/// are untouched:
///
///  1. **Block-run coordinate coding** (lossless). Records are grouped by
///     their 8×8×8 Bonxai leaf block (`leaf_bits=3`); one block coord
///     amortizes over up to 512 records:
///       [bx:i32][by:i32][bz:i32][mode:u8]
///         mode 0: [bitmask: 64 B]        // 512 bits; bit = (lx<<6)|(ly<<3)|lz,
///                                        // stored LSB-first (byte b>>3, bit b&7)
///         mode 1: [n:u16][idx:u16] × n   // strictly ascending, same numbering
///       [payload] × n                    // ascending bit order
///     Mode is chosen per block by size: the 64 B bitmask wins at n ≥ 32, the
///     index list (2 + 2n B) below that. Blocks are emitted in ascending
///     (bx, by, bz) order, so serialization is deterministic.
///
///  2. **u16 payload quantization** of evidence ABOVE the prior:
///       q = clamp(round((a − prior) / quant_step), 0, 65535)   (sender)
///       a = prior + q · quant_step                              (receiver)
///     Encoding relative to the prior keeps an at-prior value at exactly q=0,
///     reconstructing bit-exactly — the sender's at-prior emit gate and the
///     receiver's isPrior*/refold tests (dscovox_consensus.hpp) keep working.
///     The occupancy priors are compile-time constants (kBetaOccPrior /
///     kBetaFreePrior); the Dir priors (α₀ per slot, (C − K_TOP)·α₀ for OTHER)
///     are derived from existing header fields. The sender sets
///     `quant_step = evidence_saturation / 65535` so the u16 range spans
///     exactly the representable evidence; quant_step = 0.0 is the sentinel
///     for "payloads are f32" (evidence_saturation=0 disables the cap, so
///     evidence is unbounded and quantization is unsafe).
///
/// Quantization edge semantics (all deliberate):
///   - Below-prior inputs clamp to the prior. These occur on normal input —
///     the total-cap saturation rescale erodes the minority bucket below its
///     prior on saturated lopsided voxels — and mergeBeta/mergeDir already
///     floor them at the prior on the receiver (consensus_merge.hpp), so the
///     clamp moves an existing floor from merge time to encode time; the
///     fused-p error stays bounded by the prior mass, same argument.
///   - Evidence within quant_step/2 of the prior quantizes to exactly the
///     prior and is classified at-prior (dropped) by the receiver's refold.
///     At the default cap (step ≈ 0.0153) that band is < 0.008 pseudo-counts,
///     far below a single observation's increment.
///   - Evidence above 65535·quant_step (= evidence_saturation) clamps to the
///     cap; the live grid never exceeds it, so the clamp is unreachable on
///     well-formed input.
///   - Duplicate coords within one stream are deduplicated last-wins. The
///     receiver's ingest is replace-per-(source, coord), so only the latest
///     value is meaningful; the emit path never produces duplicates anyway.
///
/// The header carries the priors so the receiver reconstructs them without
/// sharing launch params. The blob VERSION byte is the codec revision: now 6
/// (was 5) for the block-run/quantized record layout. deserialize rejects a
/// mismatched revision, so a mixed-revision fleet fails loud (the frame is
/// dropped with a warning) instead of silently corrupting fused mass — same
/// drill as the 4→5 occupancy-prior bump. The ROS envelope `version` (=5)
/// routes to this codec and is unchanged.
///
/// Revision 7 (docs/design/fine_tsdf_band_dbh_2026_07_30.md) adds the fine
/// TSDF band: a `fine_ratio_log2` header byte (0 = sender has no fine grid)
/// and one fine-TSDF stream after the Dir stream. Fine coords are indices on
/// the FINE lattice (`resolution / 2^fine_ratio_log2`); payload layout is
/// identical to the coarse TSDF stream. Everything else is unchanged from 6.
///
/// Frame layout (uncompressed, little-endian):
///   [MAGIC: u32 = "SCVX"] [VERSION: u8 = 7]
///   [resolution: f32]
///   [num_classes: u16]   ← C
///   [K_TOP_wire: u8]     ← K_TOP at sender; receiver asserts match
///   [alpha_0_wire: f32]  ← symmetric Dirichlet prior at sender
///   [quant_step: f32]    ← u16 quantization step; 0.0 = f32 payloads
///   [fine_ratio_log2: u8] ← fine lattice ratio k; 0 = no fine grid
///   [tsdf_count: u32]
///     ([x:i32][y:i32][z:i32][distance:f32][weight:f32]) × tsdf_count   // 20 B
///   [beta_count: u32]  <block runs>   // payload: [q_occ:u16][q_free:u16] (4 B)
///                                     //      or  [a_occ:f32][a_free:f32] (8 B)
///   [dir_count: u32]   <block runs>   // payload: [q_other:u16][q_cnt:u16 ×K]
///                                     //          [cls:u16 ×K]   (2+4K B)
///                                     //      or  [other:f32][cnt:f32 ×K]
///                                     //          [cls:u16 ×K]   (4+6K B)
///   [fine_tsdf_count: u32]            // fine-lattice coords, 20 B records
///     ([x:i32][y:i32][z:i32][distance:f32][weight:f32]) × fine_tsdf_count
///
/// `Options{.share_tsdf=false}` (default) writes `tsdf_count=0` and elides the
/// TSDF payload; each robot keeps a local TSDF for mesh extraction. Beta and
/// Dir streams are always written (they are the split substrate's payload).
/// The fine stream ships whatever `frame.fine_tsdf_deltas` holds (the node
/// fills it only when share_tsdf is on and the fine band is enabled); a
/// non-empty fine stream with `fine_ratio_log2 == 0` is rejected on decode.
///
/// @warning K_TOP is locked at compile time. The Dir payload size scales with
/// it, so sender and receiver MUST be compiled with the same K_TOP. A mismatch
/// is detected during `deserialize` (the K_TOP_wire byte) and throws fatally —
/// there is NO forward/backward compatibility across K_TOP values.

#include <algorithm>
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

class BinarySerializer {
 public:
  static constexpr uint32_t MAGIC          = 0x53435658;  // "SCVX"
  // Sane upper bound on the header-supplied class count. num_classes drives the
  // reconstructed Dirichlet OTHER prior (num_classes − K_TOP)·alpha_0 in
  // consensus_merge; a forged u16 near 65535 with alpha_0≈0.01 injects ~650
  // pseudo-counts of "unknown" mass per voxel, swamping real evidence (and two
  // re-sent snapshots from the same bad sender agree, so the per-frame equality
  // check can't catch it). No real taxonomy approaches this ceiling.
  static constexpr uint16_t MAX_NUM_CLASSES = 4096;
  // Blob codec revision (distinct from the ROS envelope `version`=5 that routes
  // to this codec). Bumped 5→6 for the block-run coordinate coding + u16
  // payload quantization (comms_design_2026_07_30.md Part 1); 6→7 for the
  // fine-TSDF band (fine_ratio_log2 header byte + trailing fine stream,
  // fine_tsdf_band_dbh_2026_07_30.md). Any layout change means a
  // mixed-revision fleet fails loud (deserialize rejects the VERSION byte and
  // the frame is dropped with a warning) instead of silently misparsing.
  static constexpr uint8_t  FORMAT_VERSION = 7;
  // Sanity ceiling on the fine-lattice ratio: k=8 is res/256 — far beyond any
  // deployable fine resolution. A forged large k would reconstruct absurd
  // fine lattices downstream (res_fine underflow in consumers).
  static constexpr uint8_t  MAX_FINE_RATIO_LOG2 = 8;

  struct TsdfDelta { Bonxai::CoordT coord; TsdfVoxel data; };
  struct BetaDelta { Bonxai::CoordT coord; BetaVoxel data; };
  struct DirDelta  { Bonxai::CoordT coord; DirVoxel  data; };

  struct Frame {
    float    resolution  = 0.0f;
    uint16_t num_classes = 14;            ///< NYU13 default
    float    alpha_0     = kDefaultDirichletPrior;
    /// u16 quantization step for Beta/Dir payloads. The sender sets
    /// evidence_saturation / 65535; 0.0 (the default) keeps payloads f32.
    float    quant_step  = 0.0f;
    /// Fine-lattice ratio k (res_fine = resolution / 2^k); 0 = the sender
    /// has no fine grid. Coords in `fine_tsdf_deltas` index the FINE lattice.
    uint8_t  fine_ratio_log2 = 0;
    std::vector<TsdfDelta> tsdf_deltas;
    std::vector<BetaDelta> beta_deltas;
    std::vector<DirDelta>  dir_deltas;
    std::vector<TsdfDelta> fine_tsdf_deltas;
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
    if (!std::isfinite(frame.quant_step) || frame.quant_step < 0.f)
      throw std::runtime_error(
          "BinarySerializer: quant_step must be finite and >= 0");
    const float step  = frame.quant_step;
    const bool  quant = step > 0.f;
    // Clamp the residual at 0 to match defaultDirVoxel's num_classes ≤ K_TOP
    // convention (see dir_voxel.hpp).
    const int   residual_dims = static_cast<int>(frame.num_classes) - K_TOP;
    const float other_prior = residual_dims > 0
        ? static_cast<float>(residual_dims) * frame.alpha_0 : 0.f;

    const std::size_t beta_payload = quant ? 4u : 8u;
    const std::size_t dir_payload  = quant
        ? (2u + 4u * static_cast<std::size_t>(K_TOP))
        : (4u + 6u * static_cast<std::size_t>(K_TOP));
    const std::size_t tsdf_n = opts.share_tsdf ? frame.tsdf_deltas.size() : 0;
    if (frame.fine_ratio_log2 > MAX_FINE_RATIO_LOG2)
      throw std::runtime_error(
          "BinarySerializer: fine_ratio_log2 > " +
          std::to_string(MAX_FINE_RATIO_LOG2));
    if (frame.fine_ratio_log2 == 0 && !frame.fine_tsdf_deltas.empty())
      throw std::runtime_error(
          "BinarySerializer: fine_tsdf_deltas without fine_ratio_log2");

    std::string out;
    // Upper bound: every Beta/Dir record in its own single-entry block
    // (12 coord + 1 mode + 2 n + 2 idx = 17 B of overhead). Real frames
    // amortize the block header over up to 512 records.
    out.reserve(22                                                  // header
                + 4 + tsdf_n * 20
                + 4 + frame.beta_deltas.size() * (beta_payload + 17)
                + 4 + frame.dir_deltas.size()  * (dir_payload  + 17)
                + 4 + frame.fine_tsdf_deltas.size() * 20);

    appendBytes(out, &MAGIC, sizeof(MAGIC));
    appendBytes(out, &FORMAT_VERSION, sizeof(FORMAT_VERSION));
    appendBytes(out, &frame.resolution, sizeof(frame.resolution));
    appendBytes(out, &frame.num_classes, sizeof(frame.num_classes));
    const uint8_t k_top_wire = static_cast<uint8_t>(K_TOP);
    appendBytes(out, &k_top_wire, sizeof(k_top_wire));
    appendBytes(out, &frame.alpha_0, sizeof(frame.alpha_0));
    appendBytes(out, &frame.quant_step, sizeof(frame.quant_step));
    appendBytes(out, &frame.fine_ratio_log2, sizeof(frame.fine_ratio_log2));

    // TSDF stream (optional; flat records, unchanged from revision 5).
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

    // Beta stream (occupancy), block-run coded.
    writeBlockStream(out, frame.beta_deltas, [&](const BetaDelta& d) {
      if (quant) {
        const uint16_t q_occ  = quantize(d.data.a_occ,  kBetaOccPrior,  step);
        const uint16_t q_free = quantize(d.data.a_free, kBetaFreePrior, step);
        appendBytes(out, &q_occ,  sizeof(q_occ));
        appendBytes(out, &q_free, sizeof(q_free));
      } else {
        appendBytes(out, &d.data.a_occ,  sizeof(float));
        appendBytes(out, &d.data.a_free, sizeof(float));
      }
    });

    // Dir stream (occupied-class semantics), block-run coded. Class ids stay
    // u16 and are never quantized; empty slots (cls=0xFFFF, cnt=α₀) quantize
    // to q=0 and reconstruct their placeholder exactly.
    writeBlockStream(out, frame.dir_deltas, [&](const DirDelta& d) {
      if (quant) {
        const uint16_t q_other = quantize(d.data.other, other_prior, step);
        appendBytes(out, &q_other, sizeof(q_other));
        for (int i = 0; i < K_TOP; ++i) {
          const uint16_t q_cnt = quantize(d.data.cnt[i], frame.alpha_0, step);
          appendBytes(out, &q_cnt, sizeof(q_cnt));
        }
      } else {
        appendBytes(out, &d.data.other, sizeof(float));
        for (int i = 0; i < K_TOP; ++i)
          appendBytes(out, &d.data.cnt[i], sizeof(float));
      }
      for (int i = 0; i < K_TOP; ++i)
        appendBytes(out, &d.data.cls[i], sizeof(uint16_t));
    });

    // Fine TSDF stream (rev 7; flat records like the coarse TSDF stream,
    // coords on the fine lattice). Content-gated: the sender fills
    // fine_tsdf_deltas only when it means to share them.
    const uint32_t fine_count =
        static_cast<uint32_t>(frame.fine_tsdf_deltas.size());
    appendBytes(out, &fine_count, sizeof(fine_count));
    for (const auto& d : frame.fine_tsdf_deltas) {
      appendBytes(out, &d.coord.x,       sizeof(int32_t));
      appendBytes(out, &d.coord.y,       sizeof(int32_t));
      appendBytes(out, &d.coord.z,       sizeof(int32_t));
      appendBytes(out, &d.data.distance, sizeof(float));
      appendBytes(out, &d.data.weight,   sizeof(float));
    }

    return out;
  }

  /// Deserialise. Throws on bad MAGIC, bad VERSION, K_TOP mismatch, malformed
  /// block runs, or a truncated frame. K_TOP mismatch is fatal (the payload
  /// size is length-implicit, locked by the header).
  static Frame deserialize(const std::string& data) {
    Frame f;
    Reader r{data, 0};

    const uint32_t magic = r.get<uint32_t>();
    if (magic != MAGIC)
      throw std::runtime_error("BinarySerializer: bad MAGIC");

    const uint8_t version = r.get<uint8_t>();
    if (version != FORMAT_VERSION)
      throw std::runtime_error("BinarySerializer: bad VERSION");

    f.resolution  = r.get<float>();
    f.num_classes = r.get<uint16_t>();

    const uint8_t k_top_wire = r.get<uint8_t>();
    if (k_top_wire != static_cast<uint8_t>(K_TOP)) {
      throw std::runtime_error(
          "BinarySerializer: K_TOP mismatch — receiver compiled with K_TOP="
          + std::to_string(K_TOP) + ", wire says " + std::to_string(k_top_wire));
    }

    f.alpha_0         = r.get<float>();
    f.quant_step      = r.get<float>();
    f.fine_ratio_log2 = r.get<uint8_t>();
    if (f.fine_ratio_log2 > MAX_FINE_RATIO_LOG2) {
      throw std::runtime_error(
          "BinarySerializer: fine_ratio_log2 (" +
          std::to_string(f.fine_ratio_log2) + ") > " +
          std::to_string(MAX_FINE_RATIO_LOG2));
    }

    // Validate the header-supplied prior parameters before any voxel is
    // reconstructed from them. consensus_merge rebuilds the Dirichlet
    // (semantic) prior as (num_classes − K_TOP)·alpha_0 with per-slot alpha_0,
    // so a corrupt header — num_classes < K_TOP, or a non-finite/non-positive
    // alpha_0 — would silently inject negative or NaN mass into the live map
    // (and two re-sent snapshots from the same bad sender agree with each
    // other, so the per-frame equality check cannot catch it). The OCCUPANCY
    // Beta prior is the symmetric constant kBetaOccPrior, independent of these
    // fields.
    if (f.num_classes < static_cast<uint16_t>(K_TOP)) {
      throw std::runtime_error(
          "BinarySerializer: num_classes (" + std::to_string(f.num_classes)
          + ") < K_TOP (" + std::to_string(K_TOP) + ")");
    }
    if (f.num_classes > MAX_NUM_CLASSES) {
      throw std::runtime_error(
          "BinarySerializer: num_classes (" + std::to_string(f.num_classes)
          + ") > MAX_NUM_CLASSES (" + std::to_string(MAX_NUM_CLASSES) + ")");
    }
    if (!std::isfinite(f.alpha_0) || f.alpha_0 <= 0.f) {
      throw std::runtime_error("BinarySerializer: alpha_0 must be finite and > 0");
    }
    // resolution is reconstructed into world coords downstream (coord*res);
    // a forged NaN/0/negative would yield NaN positions or a divide-by-zero.
    if (!std::isfinite(f.resolution) || f.resolution <= 0.f) {
      throw std::runtime_error("BinarySerializer: resolution must be finite and > 0");
    }
    // A forged NaN/negative step would reconstruct NaN/shrinking evidence in
    // every quantized payload below.
    if (!std::isfinite(f.quant_step) || f.quant_step < 0.f) {
      throw std::runtime_error("BinarySerializer: quant_step must be finite and >= 0");
    }

    const float step  = f.quant_step;
    const bool  quant = step > 0.f;
    const int   residual_dims = static_cast<int>(f.num_classes) - K_TOP;
    const float other_prior = residual_dims > 0
        ? static_cast<float>(residual_dims) * f.alpha_0 : 0.f;

    // TSDF stream (flat records, unchanged from revision 5).
    const uint32_t tsdf_count = r.get<uint32_t>();
    // Bound the reserve by the bytes actually present: each record is a fixed
    // 20 B on the wire, so a count claiming more records than the remaining
    // buffer can hold is a truncated/forged frame. Validate BEFORE reserve so
    // an attacker-controlled count (e.g. 0xFFFFFFFF) raises the documented
    // runtime_error rather than a multi-GB length_error/bad_alloc.
    if (tsdf_count > r.remaining() / 20)
      throw std::runtime_error("BinarySerializer: truncated frame");
    f.tsdf_deltas.reserve(tsdf_count);
    for (uint32_t i = 0; i < tsdf_count; ++i) {
      TsdfDelta d{};
      d.coord.x       = r.get<int32_t>();
      d.coord.y       = r.get<int32_t>();
      d.coord.z       = r.get<int32_t>();
      d.data.distance = r.get<float>();
      d.data.weight   = r.get<float>();
      f.tsdf_deltas.push_back(d);
    }

    // Beta stream. Same DoS guard as TSDF: every record carries at least
    // payload_bytes on the wire (block headers only add more), so a count
    // exceeding the remaining byte budget is rejected before the reserve.
    const std::size_t beta_payload = quant ? 4u : 8u;
    const uint32_t beta_count = r.get<uint32_t>();
    if (beta_count > r.remaining() / beta_payload)
      throw std::runtime_error("BinarySerializer: truncated frame");
    f.beta_deltas.reserve(beta_count);
    readBlockStream(r, beta_count, [&](const Bonxai::CoordT& c) {
      BetaDelta d{};
      d.coord = c;
      if (quant) {
        d.data.a_occ  = dequantize(r.get<uint16_t>(), kBetaOccPrior,  step);
        d.data.a_free = dequantize(r.get<uint16_t>(), kBetaFreePrior, step);
      } else {
        d.data.a_occ  = r.get<float>();
        d.data.a_free = r.get<float>();
      }
      f.beta_deltas.push_back(d);
    });

    // Dir stream.
    const std::size_t dir_payload = quant
        ? (2u + 4u * static_cast<std::size_t>(K_TOP))
        : (4u + 6u * static_cast<std::size_t>(K_TOP));
    const uint32_t dir_count = r.get<uint32_t>();
    if (dir_count > r.remaining() / dir_payload)
      throw std::runtime_error("BinarySerializer: truncated frame");
    f.dir_deltas.reserve(dir_count);
    readBlockStream(r, dir_count, [&](const Bonxai::CoordT& c) {
      DirDelta d{};
      d.coord = c;
      if (quant) {
        d.data.other = dequantize(r.get<uint16_t>(), other_prior, step);
        for (int j = 0; j < K_TOP; ++j)
          d.data.cnt[j] = dequantize(r.get<uint16_t>(), f.alpha_0, step);
      } else {
        d.data.other = r.get<float>();
        for (int j = 0; j < K_TOP; ++j)
          d.data.cnt[j] = r.get<float>();
      }
      for (int j = 0; j < K_TOP; ++j)
        d.data.cls[j] = r.get<uint16_t>();
      f.dir_deltas.push_back(d);
    });

    // Fine TSDF stream (rev 7). Same 20 B/record DoS guard as coarse.
    const uint32_t fine_count = r.get<uint32_t>();
    if (fine_count > r.remaining() / 20)
      throw std::runtime_error("BinarySerializer: truncated frame");
    if (fine_count > 0 && f.fine_ratio_log2 == 0)
      throw std::runtime_error(
          "BinarySerializer: fine stream without fine_ratio_log2");
    f.fine_tsdf_deltas.reserve(fine_count);
    for (uint32_t i = 0; i < fine_count; ++i) {
      TsdfDelta d{};
      d.coord.x       = r.get<int32_t>();
      d.coord.y       = r.get<int32_t>();
      d.coord.z       = r.get<int32_t>();
      d.data.distance = r.get<float>();
      d.data.weight   = r.get<float>();
      f.fine_tsdf_deltas.push_back(d);
    }

    return f;
  }

 private:
  // Block-run geometry: one run covers an 8×8×8 Bonxai leaf block
  // (leaf_bits=3), 512 voxels, bit = (lx<<6)|(ly<<3)|lz.
  static constexpr int         kBlockVoxels    = 512;
  static constexpr std::size_t kBitmaskBytes   = 64;
  // 64 B bitmask vs (2 + 2n) B index list: the mask wins at n ≥ 32.
  static constexpr std::size_t kBitmaskMinCount = 32;

  static uint16_t quantize(float a, float prior, float step) {
    // Below-prior clamps to 0 (→ prior); above-cap clamps to 65535 (→ cap).
    const long q = std::lround((a - prior) / step);
    return static_cast<uint16_t>(std::clamp(q, 0L, 65535L));
  }
  static float dequantize(uint16_t q, float prior, float step) {
    return prior + static_cast<float>(q) * step;
  }

  /// Emit one stream as [count:u32] followed by block runs. Records are
  /// sorted by (block, bit) — deterministic output — and deduplicated
  /// last-wins on coord (the receiver ingest is replace-per-coord). The
  /// written count is the POST-dedup record count.
  template <typename DeltaT, typename PayloadWriter>
  static void writeBlockStream(std::string& out,
                               const std::vector<DeltaT>& deltas,
                               PayloadWriter&& write_payload) {
    struct Entry { int32_t bx, by, bz; uint16_t bit; uint32_t src; };
    std::vector<Entry> entries;
    entries.reserve(deltas.size());
    for (uint32_t i = 0; i < static_cast<uint32_t>(deltas.size()); ++i) {
      const auto& c = deltas[i].coord;
      // `& 7` keeps locals in [0,7] for negative coords too (two's
      // complement); subtracting them first makes the block division exact,
      // with no implementation-defined shift of a negative value.
      const int32_t lx = c.x & 7, ly = c.y & 7, lz = c.z & 7;
      entries.push_back({(c.x - lx) / 8, (c.y - ly) / 8, (c.z - lz) / 8,
                         static_cast<uint16_t>((lx << 6) | (ly << 3) | lz), i});
    }
    std::sort(entries.begin(), entries.end(),
              [](const Entry& a, const Entry& b) {
      if (a.bx  != b.bx)  return a.bx  < b.bx;
      if (a.by  != b.by)  return a.by  < b.by;
      if (a.bz  != b.bz)  return a.bz  < b.bz;
      if (a.bit != b.bit) return a.bit < b.bit;
      return a.src < b.src;
    });
    // Last-wins dedup: within an equal (block, bit) run the ties are sorted
    // by src ascending, so the final element is the latest value.
    std::size_t w = 0;
    for (std::size_t i = 0; i < entries.size(); ++i) {
      if (w > 0 && entries[w - 1].bx == entries[i].bx
                && entries[w - 1].by == entries[i].by
                && entries[w - 1].bz == entries[i].bz
                && entries[w - 1].bit == entries[i].bit) {
        entries[w - 1] = entries[i];
      } else {
        entries[w++] = entries[i];
      }
    }
    entries.resize(w);

    const uint32_t count = static_cast<uint32_t>(entries.size());
    appendBytes(out, &count, sizeof(count));

    std::size_t i = 0;
    while (i < entries.size()) {
      std::size_t j = i;
      while (j < entries.size() && entries[j].bx == entries[i].bx
                                && entries[j].by == entries[i].by
                                && entries[j].bz == entries[i].bz) ++j;
      const std::size_t n = j - i;
      appendBytes(out, &entries[i].bx, sizeof(int32_t));
      appendBytes(out, &entries[i].by, sizeof(int32_t));
      appendBytes(out, &entries[i].bz, sizeof(int32_t));
      if (n >= kBitmaskMinCount) {
        const uint8_t mode = 0;
        appendBytes(out, &mode, sizeof(mode));
        uint8_t mask[kBitmaskBytes] = {};
        for (std::size_t k = i; k < j; ++k)
          mask[entries[k].bit >> 3] |=
              static_cast<uint8_t>(1u << (entries[k].bit & 7));
        appendBytes(out, mask, sizeof(mask));
      } else {
        const uint8_t mode = 1;
        appendBytes(out, &mode, sizeof(mode));
        const uint16_t n16 = static_cast<uint16_t>(n);
        appendBytes(out, &n16, sizeof(n16));
        for (std::size_t k = i; k < j; ++k)
          appendBytes(out, &entries[k].bit, sizeof(uint16_t));
      }
      for (std::size_t k = i; k < j; ++k)
        write_payload(deltas[entries[k].src]);
      i = j;
    }
  }

  struct Reader {
    const std::string& d;
    std::size_t off;
    std::size_t remaining() const { return d.size() - off; }
    void need(std::size_t n) const {
      if (off + n > d.size())
        throw std::runtime_error("BinarySerializer: truncated frame");
    }
    template <typename T>
    T get() {
      T v;
      need(sizeof(T));
      std::memcpy(&v, d.data() + off, sizeof(T));
      off += sizeof(T);
      return v;
    }
    void copy(void* dst, std::size_t n) {
      need(n);
      std::memcpy(dst, d.data() + off, n);
      off += n;
    }
  };

  /// Read block runs until exactly `count` records have been consumed.
  /// `read_payload(coord)` pulls its own payload bytes from the reader.
  /// Structural violations — bad mode byte, empty block, out-of-range or
  /// non-ascending indices, a block overshooting `count` — all throw, so a
  /// corrupt frame is dropped whole rather than half-integrated.
  template <typename PayloadReader>
  static void readBlockStream(Reader& r, uint32_t count,
                              PayloadReader&& read_payload) {
    uint16_t bits[kBlockVoxels];
    uint32_t consumed = 0;
    while (consumed < count) {
      const int32_t bx  = r.get<int32_t>();
      const int32_t by  = r.get<int32_t>();
      const int32_t bz  = r.get<int32_t>();
      const uint8_t mode = r.get<uint8_t>();
      std::size_t n = 0;
      if (mode == 0) {
        uint8_t mask[kBitmaskBytes];
        r.copy(mask, kBitmaskBytes);
        for (int b = 0; b < kBlockVoxels; ++b)
          if (mask[b >> 3] & (1u << (b & 7)))
            bits[n++] = static_cast<uint16_t>(b);
        if (n == 0)
          throw std::runtime_error("BinarySerializer: empty block bitmask");
      } else if (mode == 1) {
        const uint16_t n16 = r.get<uint16_t>();
        if (n16 == 0 || n16 > kBlockVoxels)
          throw std::runtime_error("BinarySerializer: bad in-block index count");
        int last = -1;
        for (uint16_t k = 0; k < n16; ++k) {
          const uint16_t idx = r.get<uint16_t>();
          if (idx >= kBlockVoxels || static_cast<int>(idx) <= last)
            throw std::runtime_error(
                "BinarySerializer: in-block indices must be strictly "
                "ascending and < 512");
          last = idx;
          bits[n++] = idx;
        }
      } else {
        throw std::runtime_error("BinarySerializer: bad block mode");
      }
      if (consumed + n > count)
        throw std::runtime_error(
            "BinarySerializer: block exceeds stream record count");
      for (std::size_t k = 0; k < n; ++k) {
        const int32_t lx = bits[k] >> 6;
        const int32_t ly = (bits[k] >> 3) & 7;
        const int32_t lz = bits[k] & 7;
        read_payload(Bonxai::CoordT{bx * 8 + lx, by * 8 + ly, bz * 8 + lz});
      }
      consumed += static_cast<uint32_t>(n);
    }
  }

  template <typename T>
  static void appendBytes(std::string& out, const T* p, std::size_t n) {
    out.append(reinterpret_cast<const char*>(p), n);
  }
};

}  // namespace scovox
