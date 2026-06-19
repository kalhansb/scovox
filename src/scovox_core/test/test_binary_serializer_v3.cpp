/// @file
/// @brief Step-7.5 gate: wire format v3 round-trip and header validation.

#include <gtest/gtest.h>

#include <bonxai/bonxai.hpp>
#include <stdexcept>
#include <string>

#include "scovox/binary_serializer_v3.hpp"

namespace {

scovox::BinarySerializerV3::Frame makeFrame(bool with_tsdf = true) {
  scovox::BinarySerializerV3::Frame f;
  f.resolution  = 0.05f;
  f.num_classes = 14;
  f.alpha_0     = scovox::kDefaultDirichletPrior;

  if (with_tsdf) {
    f.tsdf_deltas.push_back({Bonxai::CoordT{1, 2, 3}, scovox::TsdfVoxel{0.04f, 1.5f}});
    f.tsdf_deltas.push_back({Bonxai::CoordT{4, 5, 6}, scovox::TsdfVoxel{-0.02f, 2.0f}});
  }

  scovox::SemDirVoxel a = scovox::defaultSemDirVoxel(14, scovox::kDefaultDirichletPrior);
  a.alpha_free  = 0.5f;
  a.alpha_other = 0.30f;  // includes the 12·α_0 = 0.12 prior + 0.18 evidence
  a.cls[0] = 5;  a.cnt[0] = 1.20f;
  a.cls[1] = 9;  a.cnt[1] = 0.80f;
  f.semdir_deltas.push_back({Bonxai::CoordT{1, 2, 3}, a});

  scovox::SemDirVoxel b = scovox::defaultSemDirVoxel(14, scovox::kDefaultDirichletPrior);
  b.alpha_free  = 2.0f;
  b.alpha_other = 0.12f;  // prior only
  b.cls[0] = 7;  b.cnt[0] = 0.50f;
  // slot 1 left at the empty-sentinel default.
  f.semdir_deltas.push_back({Bonxai::CoordT{4, 5, 6}, b});

  return f;
}

void expectFrameEq(const scovox::BinarySerializerV3::Frame& a,
                   const scovox::BinarySerializerV3::Frame& b,
                   bool expect_tsdf = true) {
  EXPECT_FLOAT_EQ(a.resolution,  b.resolution);
  EXPECT_EQ(a.num_classes,       b.num_classes);
  EXPECT_FLOAT_EQ(a.alpha_0,     b.alpha_0);
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
  ASSERT_EQ(a.semdir_deltas.size(), b.semdir_deltas.size());
  for (std::size_t i = 0; i < a.semdir_deltas.size(); ++i) {
    EXPECT_EQ(a.semdir_deltas[i].coord.x, b.semdir_deltas[i].coord.x);
    EXPECT_EQ(a.semdir_deltas[i].coord.y, b.semdir_deltas[i].coord.y);
    EXPECT_EQ(a.semdir_deltas[i].coord.z, b.semdir_deltas[i].coord.z);
    EXPECT_FLOAT_EQ(a.semdir_deltas[i].data.alpha_free,  b.semdir_deltas[i].data.alpha_free);
    EXPECT_FLOAT_EQ(a.semdir_deltas[i].data.alpha_other, b.semdir_deltas[i].data.alpha_other);
    for (int j = 0; j < scovox::K_TOP; ++j) {
      EXPECT_FLOAT_EQ(a.semdir_deltas[i].data.cnt[j], b.semdir_deltas[i].data.cnt[j]);
      EXPECT_EQ(a.semdir_deltas[i].data.cls[j],       b.semdir_deltas[i].data.cls[j]);
    }
  }
}

}  // namespace

TEST(BinarySerializerV3, DualStreamRoundTripWithShareTsdf) {
  auto f = makeFrame();
  auto blob = scovox::BinarySerializerV3::serialize(
      f, scovox::BinarySerializerV3::Options{/*share_tsdf=*/true});
  auto g = scovox::BinarySerializerV3::deserialize(blob);
  expectFrameEq(f, g, /*expect_tsdf=*/true);
}

TEST(BinarySerializerV3, SemDirOnlyDefaultElidesTsdfSection) {
  auto f = makeFrame();
  // Default options (share_tsdf=false) — TSDF count must be 0 on the wire.
  auto blob = scovox::BinarySerializerV3::serialize(f);
  auto g = scovox::BinarySerializerV3::deserialize(blob);
  expectFrameEq(f, g, /*expect_tsdf=*/false);
}

TEST(BinarySerializerV3, BadMagicThrows) {
  std::string blob("garbage payload here");
  EXPECT_THROW(scovox::BinarySerializerV3::deserialize(blob), std::runtime_error);
}

TEST(BinarySerializerV3, BadVersionThrows) {
  auto f = makeFrame();
  auto blob = scovox::BinarySerializerV3::serialize(f);
  // Stomp version byte to v2.
  blob[4] = static_cast<char>(2);
  EXPECT_THROW(scovox::BinarySerializerV3::deserialize(blob), std::runtime_error);
}

TEST(BinarySerializerV3, EmitSizeMatchesSpec) {
  auto f = makeFrame();
  auto blob_full = scovox::BinarySerializerV3::serialize(
      f, scovox::BinarySerializerV3::Options{/*share_tsdf=*/true});
  // Header: 4 (magic) + 1 (version) + 4 (resolution) + 2 (num_classes)
  //       + 1 (K_TOP) + 4 (alpha_0) = 16 B
  // TSDF:   4 (count) + 2 records × 20 B = 44 B
  // SemDir: 4 (count) + 2 records × (12 + 8 + 4·K_TOP + 2·K_TOP)
  //       = 4 + 2 × (12 + 8 + 4·2 + 2·2) = 4 + 2 × 32 = 68 B at K_TOP=2
  // Total at K_TOP=2: 16 + 44 + 68 = 128 B
  if (scovox::K_TOP == 2) {
    EXPECT_EQ(blob_full.size(), 128u) << "share_tsdf=true emit size";
  }
  auto blob_semdir = scovox::BinarySerializerV3::serialize(f);
  // share_tsdf=false → tsdf section is just the 4-byte zero count.
  if (scovox::K_TOP == 2) {
    EXPECT_EQ(blob_semdir.size(), 16u + 4u + 68u) << "share_tsdf=false emit size";
  }
}
