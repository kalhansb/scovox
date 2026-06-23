/// @file
/// @brief Wire format v4 (triple-stream: TSDF + BetaVoxel + DirVoxel) round-trip
/// and header validation.

#include <gtest/gtest.h>

#include <bonxai/bonxai.hpp>
#include <stdexcept>
#include <string>

#include "scovox/binary_serializer_v4.hpp"

namespace {

scovox::BinarySerializerV4::Frame makeFrame() {
  scovox::BinarySerializerV4::Frame f;
  f.resolution  = 0.05f;
  f.num_classes = 14;
  f.alpha_0     = scovox::kDefaultDirichletPrior;

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

void expectFrameEq(const scovox::BinarySerializerV4::Frame& a,
                   const scovox::BinarySerializerV4::Frame& b,
                   bool expect_tsdf) {
  EXPECT_FLOAT_EQ(a.resolution, b.resolution);
  EXPECT_EQ(a.num_classes,      b.num_classes);
  EXPECT_FLOAT_EQ(a.alpha_0,    b.alpha_0);

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

  ASSERT_EQ(a.beta_deltas.size(), b.beta_deltas.size());
  for (std::size_t i = 0; i < a.beta_deltas.size(); ++i) {
    EXPECT_EQ(a.beta_deltas[i].coord.x, b.beta_deltas[i].coord.x);
    EXPECT_EQ(a.beta_deltas[i].coord.y, b.beta_deltas[i].coord.y);
    EXPECT_EQ(a.beta_deltas[i].coord.z, b.beta_deltas[i].coord.z);
    EXPECT_FLOAT_EQ(a.beta_deltas[i].data.a_occ,  b.beta_deltas[i].data.a_occ);
    EXPECT_FLOAT_EQ(a.beta_deltas[i].data.a_free, b.beta_deltas[i].data.a_free);
  }

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

TEST(BinarySerializerV4, TripleStreamRoundTripWithShareTsdf) {
  auto f = makeFrame();
  auto blob = scovox::BinarySerializerV4::serialize(
      f, scovox::BinarySerializerV4::Options{/*share_tsdf=*/true});
  auto g = scovox::BinarySerializerV4::deserialize(blob);
  expectFrameEq(f, g, /*expect_tsdf=*/true);
}

TEST(BinarySerializerV4, BetaDirOnlyDefaultElidesTsdfSection) {
  auto f = makeFrame();
  auto blob = scovox::BinarySerializerV4::serialize(f);  // default: share_tsdf=false
  auto g = scovox::BinarySerializerV4::deserialize(blob);
  expectFrameEq(f, g, /*expect_tsdf=*/false);
}

TEST(BinarySerializerV4, BadMagicThrows) {
  std::string blob("garbage payload here");
  EXPECT_THROW(scovox::BinarySerializerV4::deserialize(blob), std::runtime_error);
}

TEST(BinarySerializerV4, BadVersionThrows) {
  auto f = makeFrame();
  auto blob = scovox::BinarySerializerV4::serialize(f);
  blob[4] = static_cast<char>(3);  // stomp version to v3
  EXPECT_THROW(scovox::BinarySerializerV4::deserialize(blob), std::runtime_error);
}

TEST(BinarySerializerV4, TruncatedFrameThrows) {
  auto f = makeFrame();
  auto blob = scovox::BinarySerializerV4::serialize(
      f, scovox::BinarySerializerV4::Options{/*share_tsdf=*/true});
  blob.resize(blob.size() - 5);  // chop the tail of the last Dir record
  EXPECT_THROW(scovox::BinarySerializerV4::deserialize(blob), std::runtime_error);
}

TEST(BinarySerializerV4, EmitSizeMatchesSpec) {
  auto f = makeFrame();
  // Header: 4 + 1 + 4 + 2 + 1 + 4 = 16 B
  // TSDF:   4 (count) + 2 × 20 = 44 B
  // Beta:   4 (count) + 2 × 20 = 44 B   (coord 12 + a_occ 4 + a_free 4)
  // Dir:    4 (count) + 2 × 28 = 60 B   (coord 12 + other 4 + cnt 4·K + cls 2·K)
  if (scovox::K_TOP == 2) {
    auto full = scovox::BinarySerializerV4::serialize(
        f, scovox::BinarySerializerV4::Options{/*share_tsdf=*/true});
    EXPECT_EQ(full.size(), 16u + 44u + 44u + 60u) << "share_tsdf=true emit size";  // 164

    auto no_tsdf = scovox::BinarySerializerV4::serialize(f);
    EXPECT_EQ(no_tsdf.size(), 16u + 4u + 44u + 60u) << "share_tsdf=false emit size";  // 124
  }
}
