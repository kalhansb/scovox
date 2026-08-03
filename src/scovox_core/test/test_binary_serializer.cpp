/// @file
/// @brief Wire format (revision 6: block-run coords + u16 quantization)
/// round-trip, quantization edge semantics, and header/structure validation.

#include <gtest/gtest.h>

#include <algorithm>
#include <bonxai/bonxai.hpp>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include "scovox/binary_serializer.hpp"

namespace {

// Default-cap quantization step: evidence_saturation(1000) / 65535.
constexpr float kStep = 1000.f / 65535.f;

scovox::BinarySerializer::Frame makeFrame() {
  scovox::BinarySerializer::Frame f;
  f.resolution  = 0.05f;
  f.num_classes = 14;
  f.alpha_0     = scovox::kDefaultDirichletPrior;
  // quant_step left at the 0.0 default → f32 payloads, lossless round-trip.

  f.tsdf_deltas.push_back({Bonxai::CoordT{1, 2, 3}, scovox::TsdfVoxel{0.04f, 1.5f}});
  f.tsdf_deltas.push_back({Bonxai::CoordT{4, 5, 6}, scovox::TsdfVoxel{-0.02f, 2.0f}});

  f.beta_deltas.push_back({Bonxai::CoordT{1, 2, 3}, scovox::BetaVoxel{1.14f, 0.5f}});
  f.beta_deltas.push_back({Bonxai::CoordT{4, 5, 6}, scovox::BetaVoxel{2.0f, 0.01f}});

  scovox::DirVoxel a = scovox::defaultDirVoxel(14, scovox::kDefaultDirichletPrior);
  a.other  = 0.30f;        // 0.12 prior + 0.18 evidence
  a.cls[0] = 5; a.cnt[0] = 1.20f;
  a.cls[1] = 9; a.cnt[1] = 0.80f;
  f.dir_deltas.push_back({Bonxai::CoordT{1, 2, 3}, a});

  scovox::DirVoxel b = scovox::defaultDirVoxel(14, scovox::kDefaultDirichletPrior);
  b.other  = 0.12f;        // prior only
  b.cls[0] = 7; b.cnt[0] = 0.50f;
  // slot 1 left at the empty-sentinel default.
  f.dir_deltas.push_back({Bonxai::CoordT{4, 5, 6}, b});

  return f;
}

// Sort both frames' streams by coord before comparing: the serializer emits
// records in (block, bit) order, which is a permutation of the input order.
template <typename DeltaT>
void sortByCoord(std::vector<DeltaT>& v) {
  std::sort(v.begin(), v.end(), [](const DeltaT& a, const DeltaT& b) {
    if (a.coord.x != b.coord.x) return a.coord.x < b.coord.x;
    if (a.coord.y != b.coord.y) return a.coord.y < b.coord.y;
    return a.coord.z < b.coord.z;
  });
}

void expectFrameEq(scovox::BinarySerializer::Frame a,
                   scovox::BinarySerializer::Frame b,
                   bool expect_tsdf) {
  EXPECT_FLOAT_EQ(a.resolution, b.resolution);
  EXPECT_EQ(a.num_classes,      b.num_classes);
  EXPECT_FLOAT_EQ(a.alpha_0,    b.alpha_0);
  EXPECT_FLOAT_EQ(a.quant_step, b.quant_step);

  if (expect_tsdf) {
    ASSERT_EQ(a.tsdf_deltas.size(), b.tsdf_deltas.size());
    for (std::size_t i = 0; i < a.tsdf_deltas.size(); ++i) {
      EXPECT_EQ(a.tsdf_deltas[i].coord.x, b.tsdf_deltas[i].coord.x);
      EXPECT_EQ(a.tsdf_deltas[i].coord.y, b.tsdf_deltas[i].coord.y);
      EXPECT_EQ(a.tsdf_deltas[i].coord.z, b.tsdf_deltas[i].coord.z);
      EXPECT_FLOAT_EQ(a.tsdf_deltas[i].data.distance, b.tsdf_deltas[i].data.distance);
      EXPECT_FLOAT_EQ(a.tsdf_deltas[i].data.weight,   b.tsdf_deltas[i].data.weight);
    }
  } else {
    EXPECT_EQ(b.tsdf_deltas.size(), 0u) << "share_tsdf=false must drop TSDF section";
  }

  sortByCoord(a.beta_deltas);
  sortByCoord(b.beta_deltas);
  ASSERT_EQ(a.beta_deltas.size(), b.beta_deltas.size());
  for (std::size_t i = 0; i < a.beta_deltas.size(); ++i) {
    EXPECT_EQ(a.beta_deltas[i].coord.x, b.beta_deltas[i].coord.x);
    EXPECT_EQ(a.beta_deltas[i].coord.y, b.beta_deltas[i].coord.y);
    EXPECT_EQ(a.beta_deltas[i].coord.z, b.beta_deltas[i].coord.z);
    EXPECT_FLOAT_EQ(a.beta_deltas[i].data.a_occ,  b.beta_deltas[i].data.a_occ);
    EXPECT_FLOAT_EQ(a.beta_deltas[i].data.a_free, b.beta_deltas[i].data.a_free);
  }

  sortByCoord(a.dir_deltas);
  sortByCoord(b.dir_deltas);
  ASSERT_EQ(a.dir_deltas.size(), b.dir_deltas.size());
  for (std::size_t i = 0; i < a.dir_deltas.size(); ++i) {
    EXPECT_EQ(a.dir_deltas[i].coord.x, b.dir_deltas[i].coord.x);
    EXPECT_EQ(a.dir_deltas[i].coord.y, b.dir_deltas[i].coord.y);
    EXPECT_EQ(a.dir_deltas[i].coord.z, b.dir_deltas[i].coord.z);
    EXPECT_FLOAT_EQ(a.dir_deltas[i].data.other, b.dir_deltas[i].data.other);
    for (int j = 0; j < scovox::K_TOP; ++j) {
      EXPECT_FLOAT_EQ(a.dir_deltas[i].data.cnt[j], b.dir_deltas[i].data.cnt[j]);
      EXPECT_EQ(a.dir_deltas[i].data.cls[j],       b.dir_deltas[i].data.cls[j]);
    }
  }
}

}  // namespace

TEST(BinarySerializer, TripleStreamRoundTripWithShareTsdf) {
  auto f = makeFrame();
  auto blob = scovox::BinarySerializer::serialize(
      f, scovox::BinarySerializer::Options{/*share_tsdf=*/true});
  auto g = scovox::BinarySerializer::deserialize(blob);
  expectFrameEq(f, g, /*expect_tsdf=*/true);
}

TEST(BinarySerializer, BetaDirOnlyDefaultElidesTsdfSection) {
  auto f = makeFrame();
  auto blob = scovox::BinarySerializer::serialize(f);  // default: share_tsdf=false
  auto g = scovox::BinarySerializer::deserialize(blob);
  expectFrameEq(f, g, /*expect_tsdf=*/false);
}

TEST(BinarySerializer, BadMagicThrows) {
  std::string blob("garbage payload here");
  EXPECT_THROW(scovox::BinarySerializer::deserialize(blob), std::runtime_error);
}

TEST(BinarySerializer, BadVersionThrows) {
  auto f = makeFrame();
  auto blob = scovox::BinarySerializer::serialize(f);
  blob[4] = static_cast<char>(5);  // a revision-5 blob must be rejected loud
  EXPECT_THROW(scovox::BinarySerializer::deserialize(blob), std::runtime_error);
}

TEST(BinarySerializer, TruncatedFrameThrows) {
  auto f = makeFrame();
  auto blob = scovox::BinarySerializer::serialize(
      f, scovox::BinarySerializer::Options{/*share_tsdf=*/true});
  blob.resize(blob.size() - 5);  // chop the tail of the last Dir payload
  EXPECT_THROW(scovox::BinarySerializer::deserialize(blob), std::runtime_error);
}

// Regression for the forged-record-count DoS (review finding 38 / 8): a frame
// with a valid MAGIC/VERSION/header but a record count (tsdf/beta/dir) far
// larger than the bytes actually present must be REJECTED, not silently
// accepted nor allowed to drive an unbounded reserve(). We forge each of the
// three counts in turn (with an otherwise-empty body) and assert deserialize
// throws so the receiver can drop the frame instead of integrating garbage.
// All three stream guards validate the count against the remaining byte budget
// BEFORE the reserve, so the forged count surfaces as the documented
// runtime_error; std::exception is asserted as the (looser) contract.
TEST(BinarySerializer, ForgedRecordCountIsRejected) {
  // Build a minimal, otherwise-valid header so the forged count is the only
  // anomaly. Layout (little-endian, 21 B): MAGIC u32, VERSION u8, resolution
  // f32, num_classes u16, K_TOP u8, alpha_0 f32, quant_step f32,
  // fine_ratio_log2 u8 (rev 7) — then the stream counts.
  auto makeHeader = []() {
    std::string h;
    const uint32_t magic   = scovox::BinarySerializer::MAGIC;
    const uint8_t  version = scovox::BinarySerializer::FORMAT_VERSION;
    const float    res     = 0.05f;
    const uint16_t nclass  = 14;
    const uint8_t  k_top   = static_cast<uint8_t>(scovox::K_TOP);
    const float    alpha0  = scovox::kDefaultDirichletPrior;
    const float    step    = 0.0f;   // f32 payloads
    const uint8_t  fine_k  = 0;      // no fine grid
    auto app = [&h](const void* p, std::size_t n) {
      h.append(reinterpret_cast<const char*>(p), n);
    };
    app(&magic, sizeof(magic));
    app(&version, sizeof(version));
    app(&res, sizeof(res));
    app(&nclass, sizeof(nclass));
    app(&k_top, sizeof(k_top));
    app(&alpha0, sizeof(alpha0));
    app(&step, sizeof(step));
    app(&fine_k, sizeof(fine_k));
    return h;
  };
  auto appendU32 = [](std::string& s, uint32_t v) {
    s.append(reinterpret_cast<const char*>(&v), sizeof(v));
  };

  const uint32_t kForged = 0xFFFFFFFFu;  // ~4 G records, ~120 GB of payload

  // (a) Forged TSDF count, no payload following it.
  {
    std::string blob = makeHeader();
    appendU32(blob, kForged);  // tsdf_count = huge, then nothing
    EXPECT_THROW(scovox::BinarySerializer::deserialize(blob), std::exception);
  }
  // (b) Valid empty TSDF stream, forged Beta count, no payload.
  {
    std::string blob = makeHeader();
    appendU32(blob, 0u);       // tsdf_count = 0
    appendU32(blob, kForged);  // beta_count = huge, then nothing
    EXPECT_THROW(scovox::BinarySerializer::deserialize(blob), std::exception);
  }
  // (c) Valid empty TSDF + Beta streams, forged Dir count, no payload.
  {
    std::string blob = makeHeader();
    appendU32(blob, 0u);       // tsdf_count = 0
    appendU32(blob, 0u);       // beta_count = 0
    appendU32(blob, kForged);  // dir_count = huge, then nothing
    EXPECT_THROW(scovox::BinarySerializer::deserialize(blob), std::exception);
  }
  // (d) Valid empty TSDF/Beta/Dir streams, forged FINE count (rev 7), no
  // payload. Same byte-budget guard as the coarse TSDF stream.
  {
    std::string blob = makeHeader();
    appendU32(blob, 0u);       // tsdf_count = 0
    appendU32(blob, 0u);       // beta_count = 0
    appendU32(blob, 0u);       // dir_count  = 0
    appendU32(blob, kForged);  // fine_tsdf_count = huge, then nothing
    EXPECT_THROW(scovox::BinarySerializer::deserialize(blob), std::exception);
  }
}

// Companion check: a modestly-forged count whose payload is short but whose
// reserve() would succeed must still throw the documented std::runtime_error.
// With the byte-budget guard the 1000-record claim against a ~100 B body is
// caught before any block run is read.
TEST(BinarySerializer, ModestForgedCountTruncationThrowsRuntimeError) {
  auto f = makeFrame();
  // Serialize a real frame (share_tsdf=false): header(20) + tsdf_count(4,=0)
  // + beta_count(4) + beta block runs + dir_count(4) + dir block runs.
  auto blob = scovox::BinarySerializer::serialize(f);
  const std::size_t beta_count_off = 20 + 4;
  const uint32_t modest = 1000u;
  std::memcpy(&blob[beta_count_off], &modest, sizeof(modest));
  EXPECT_THROW(scovox::BinarySerializer::deserialize(blob), std::runtime_error);
}

TEST(BinarySerializer, EmitSizeMatchesSpec) {
  auto f = makeFrame();
  // Both Beta records — coords (1,2,3), (4,5,6) — share block (0,0,0), bits
  // 83 < 302, n=2 → mode 1 index list. Same for the Dir records.
  // Header: 4 + 1 + 4 + 2 + 1 + 4 + 4 + 1 = 21 B     (rev 7: +fine_ratio_log2)
  // TSDF:   4 (count) + 2 × 20 = 44 B                        (flat records)
  // Block:  12 (coord) + 1 (mode) + 2 (n) + 2·2 (idx) = 19 B
  // Beta:   4 (count) + 19 + 2 × 8  = 39 B                   (f32 payloads)
  // Dir:    4 (count) + 19 + 2 × 16 = 55 B                   (f32: other +
  //                                                cnt 4·K + cls 2·K, K=2)
  // Fine:   4 (count, empty here) = 4 B                      (rev 7 tail)
  if (scovox::K_TOP == 2) {
    auto full = scovox::BinarySerializer::serialize(
        f, scovox::BinarySerializer::Options{/*share_tsdf=*/true});
    EXPECT_EQ(full.size(), 21u + 44u + 39u + 55u + 4u) << "share_tsdf=true emit size";  // 163

    auto no_tsdf = scovox::BinarySerializer::serialize(f);
    EXPECT_EQ(no_tsdf.size(), 21u + 4u + 39u + 55u + 4u) << "share_tsdf=false emit size";  // 123

    // Quantized payloads: Beta 4 B (q_occ, q_free), Dir 10 B (q_other,
    // q_cnt·K, cls·K).
    f.quant_step = kStep;
    auto quant = scovox::BinarySerializer::serialize(f);
    EXPECT_EQ(quant.size(), 21u + 4u + (4u + 19u + 8u) + (4u + 19u + 20u) + 4u)
        << "quantized emit size";  // 103
  }
}

// ---------------------------------------------------------------------------
// Revision 6: quantization semantics.
// ---------------------------------------------------------------------------

// Above-prior evidence must round-trip within quant_step/2; class ids and the
// empty-slot α₀ placeholder must be exact.
TEST(BinarySerializer, QuantizedRoundTripWithinHalfStep) {
  scovox::BinarySerializer::Frame f;
  f.resolution  = 0.05f;
  f.num_classes = 14;
  f.alpha_0     = scovox::kDefaultDirichletPrior;
  f.quant_step  = kStep;

  f.beta_deltas.push_back(
      {Bonxai::CoordT{1, 2, 3}, scovox::BetaVoxel{1.f + 0.14f, 1.f + 12.3f}});
  f.beta_deltas.push_back(
      {Bonxai::CoordT{4, 5, 6}, scovox::BetaVoxel{1.f + 500.25f, 1.f + 999.9f}});

  scovox::DirVoxel a = scovox::defaultDirVoxel(14, f.alpha_0);
  a.other  = 0.12f + 0.18f;
  a.cls[0] = 5; a.cnt[0] = f.alpha_0 + 1.19f;
  // slot 1 stays empty: cls=0xFFFF, cnt=α₀ placeholder.
  f.dir_deltas.push_back({Bonxai::CoordT{1, 2, 3}, a});

  auto g = scovox::BinarySerializer::deserialize(
      scovox::BinarySerializer::serialize(f));
  const float half = kStep / 2.f + 1e-4f;

  ASSERT_EQ(g.beta_deltas.size(), 2u);
  for (std::size_t i = 0; i < 2; ++i) {
    EXPECT_NEAR(g.beta_deltas[i].data.a_occ,  f.beta_deltas[i].data.a_occ,  half);
    EXPECT_NEAR(g.beta_deltas[i].data.a_free, f.beta_deltas[i].data.a_free, half);
  }
  ASSERT_EQ(g.dir_deltas.size(), 1u);
  const auto& d = g.dir_deltas[0].data;
  EXPECT_NEAR(d.other,  a.other,  half);
  EXPECT_NEAR(d.cnt[0], a.cnt[0], half);
  EXPECT_EQ(d.cls[0], 5);
  // Empty slot: q=0 reconstructs the α₀ placeholder bit-exactly.
  EXPECT_EQ(d.cls[1], 0xFFFF);
  EXPECT_FLOAT_EQ(d.cnt[1], f.alpha_0);
}

// At-prior values quantize to q=0 and reconstruct BIT-EXACTLY — the receiver's
// isPrior*/refold gates (dscovox_consensus.hpp) depend on this.
TEST(BinarySerializer, QuantizedAtPriorReconstructsExactly) {
  scovox::BinarySerializer::Frame f;
  f.resolution  = 0.05f;
  f.num_classes = 14;
  f.alpha_0     = scovox::kDefaultDirichletPrior;
  f.quant_step  = kStep;

  f.beta_deltas.push_back({Bonxai::CoordT{7, 7, 7},
      scovox::BetaVoxel{scovox::kBetaOccPrior, scovox::kBetaFreePrior}});
  f.dir_deltas.push_back({Bonxai::CoordT{7, 7, 7},
      scovox::defaultDirVoxel(14, f.alpha_0)});

  auto g = scovox::BinarySerializer::deserialize(
      scovox::BinarySerializer::serialize(f));
  ASSERT_EQ(g.beta_deltas.size(), 1u);
  EXPECT_FLOAT_EQ(g.beta_deltas[0].data.a_occ,  scovox::kBetaOccPrior);
  EXPECT_FLOAT_EQ(g.beta_deltas[0].data.a_free, scovox::kBetaFreePrior);
  ASSERT_EQ(g.dir_deltas.size(), 1u);
  const auto def = scovox::defaultDirVoxel(14, f.alpha_0);
  EXPECT_FLOAT_EQ(g.dir_deltas[0].data.other, def.other);
  for (int j = 0; j < scovox::K_TOP; ++j) {
    EXPECT_FLOAT_EQ(g.dir_deltas[0].data.cnt[j], def.cnt[j]);
    EXPECT_EQ(g.dir_deltas[0].data.cls[j], def.cls[j]);
  }
}

// Below-prior inputs (normal on saturated lopsided voxels — the total-cap
// rescale erodes the minority bucket) clamp to the prior, which is exactly the
// floor mergeBeta applies on the receiver anyway. Above-cap inputs clamp to
// prior + 65535·step (= evidence_saturation).
TEST(BinarySerializer, QuantizedClampsBelowPriorAndAboveCap) {
  scovox::BinarySerializer::Frame f;
  f.resolution  = 0.05f;
  f.num_classes = 14;
  f.alpha_0     = scovox::kDefaultDirichletPrior;
  f.quant_step  = kStep;

  f.beta_deltas.push_back(
      {Bonxai::CoordT{0, 0, 0}, scovox::BetaVoxel{999.5f, 0.62f}});   // a_free < prior
  f.beta_deltas.push_back(
      {Bonxai::CoordT{0, 0, 1}, scovox::BetaVoxel{1.f + 2000.f, 1.f}});  // above cap

  auto g = scovox::BinarySerializer::deserialize(
      scovox::BinarySerializer::serialize(f));
  ASSERT_EQ(g.beta_deltas.size(), 2u);
  EXPECT_FLOAT_EQ(g.beta_deltas[0].data.a_free, scovox::kBetaFreePrior);
  EXPECT_NEAR(g.beta_deltas[0].data.a_occ, 999.5f, kStep / 2.f + 1e-3f);
  EXPECT_NEAR(g.beta_deltas[1].data.a_occ, 1.f + 1000.f, 1e-2f);
}

// ---------------------------------------------------------------------------
// Revision 6: block-run coordinate coding.
// ---------------------------------------------------------------------------

// Negative coords exercise the sign-correct block/local decomposition
// (lx = x & 7 ∈ [0,7]; bx = (x − lx)/8), including block-boundary values.
TEST(BinarySerializer, NegativeCoordsRoundTrip) {
  scovox::BinarySerializer::Frame f;
  f.resolution  = 0.05f;
  f.num_classes = 14;
  f.alpha_0     = scovox::kDefaultDirichletPrior;

  const Bonxai::CoordT coords[] = {
      {-1, -8, -9}, {-17, 5, -1000}, {-8, -8, -8}, {7, -7, 0}, {123, -456, 789}};
  for (const auto& c : coords)
    f.beta_deltas.push_back({c, scovox::BetaVoxel{2.f, 1.f}});

  auto g = scovox::BinarySerializer::deserialize(
      scovox::BinarySerializer::serialize(f));
  ASSERT_EQ(g.beta_deltas.size(), 5u);
  sortByCoord(f.beta_deltas);
  sortByCoord(g.beta_deltas);
  for (std::size_t i = 0; i < 5; ++i) {
    EXPECT_EQ(g.beta_deltas[i].coord.x, f.beta_deltas[i].coord.x);
    EXPECT_EQ(g.beta_deltas[i].coord.y, f.beta_deltas[i].coord.y);
    EXPECT_EQ(g.beta_deltas[i].coord.z, f.beta_deltas[i].coord.z);
  }
}

// A dense block (n ≥ 32) must switch to the 64 B bitmask (mode 0) — the size
// assertion proves the mode choice, the round trip proves the bit numbering.
TEST(BinarySerializer, DenseBlockUsesBitmaskMode) {
  scovox::BinarySerializer::Frame f;
  f.resolution  = 0.05f;
  f.num_classes = 14;
  f.alpha_0     = scovox::kDefaultDirichletPrior;

  // 40 voxels in block (0,0,0): the index list would cost 2 + 80 B, the mask 64 B.
  for (int y = 0; y < 8; ++y)
    for (int z = 0; z < 5; ++z)
      f.beta_deltas.push_back({Bonxai::CoordT{0, y, z},
                               scovox::BetaVoxel{2.f + y, 1.f + z}});

  auto blob = scovox::BinarySerializer::serialize(f);
  // header 21 (rev 7) + tsdf_count 4 + beta: 4 + (12 + 1 + 64) + 40·8
  // + dir_count 4 + fine_count 4.
  EXPECT_EQ(blob.size(), 21u + 4u + 4u + 77u + 320u + 4u + 4u);

  auto g = scovox::BinarySerializer::deserialize(blob);
  ASSERT_EQ(g.beta_deltas.size(), 40u);
  auto e = f;
  expectFrameEq(e, g, /*expect_tsdf=*/false);
}

// Duplicate coords in one stream dedup last-wins — the receiver's ingest is
// replace-per-(source, coord), so only the latest value is meaningful.
TEST(BinarySerializer, DuplicateCoordKeepsLastValue) {
  scovox::BinarySerializer::Frame f;
  f.resolution  = 0.05f;
  f.num_classes = 14;
  f.alpha_0     = scovox::kDefaultDirichletPrior;
  f.beta_deltas.push_back({Bonxai::CoordT{3, 3, 3}, scovox::BetaVoxel{2.f, 1.f}});
  f.beta_deltas.push_back({Bonxai::CoordT{3, 3, 3}, scovox::BetaVoxel{5.f, 1.f}});

  auto g = scovox::BinarySerializer::deserialize(
      scovox::BinarySerializer::serialize(f));
  ASSERT_EQ(g.beta_deltas.size(), 1u);
  EXPECT_FLOAT_EQ(g.beta_deltas[0].data.a_occ, 5.f);
}

// Structural violations in a block run must throw (frame dropped whole), never
// misparse. Offsets (rev 7 header = 21 B): tsdf_count 4, beta_count 4 → block
// coord at 29, mode byte at 41, mode-1 n:u16 at 42, idx[0] at 44, idx[1] at 46.
TEST(BinarySerializer, MalformedBlockRunsThrow) {
  scovox::BinarySerializer::Frame f;
  f.resolution  = 0.05f;
  f.num_classes = 14;
  f.alpha_0     = scovox::kDefaultDirichletPrior;
  f.beta_deltas.push_back({Bonxai::CoordT{1, 2, 3}, scovox::BetaVoxel{2.f, 1.f}});
  f.beta_deltas.push_back({Bonxai::CoordT{4, 5, 6}, scovox::BetaVoxel{2.f, 1.f}});
  const auto blob = scovox::BinarySerializer::serialize(f);

  {  // Unknown mode byte.
    auto bad = blob;
    bad[41] = static_cast<char>(7);
    EXPECT_THROW(scovox::BinarySerializer::deserialize(bad), std::runtime_error);
  }
  {  // Mode-1 indices must be strictly ascending: swap them.
    auto bad = blob;
    std::swap(bad[44], bad[46]);
    std::swap(bad[45], bad[47]);
    EXPECT_THROW(scovox::BinarySerializer::deserialize(bad), std::runtime_error);
  }
  {  // A block whose n overshoots the stream's record count.
    auto bad = blob;
    const uint32_t one = 1;
    std::memcpy(&bad[25], &one, sizeof(one));  // beta_count 2 → 1
    EXPECT_THROW(scovox::BinarySerializer::deserialize(bad), std::runtime_error);
  }
}
