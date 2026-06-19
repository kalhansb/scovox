/// @file
/// @brief Step-6 gate: dual-stream wire format v2 round-trip tests.

#include <gtest/gtest.h>
#include "scovox/binary_serializer_v2.hpp"

namespace {

scovox::BinarySerializerV2::Frame makeFrame() {
  scovox::BinarySerializerV2::Frame f;
  f.resolution = 0.05f;
  f.tsdf_deltas.push_back({{1, 2, 3}, {-0.05f, 1.5f}});
  f.tsdf_deltas.push_back({{4, 5, 6}, {+0.10f, 2.5f}});

  scovox::SemBetaVoxel sb1 = scovox::defaultSemBetaVoxel();
  sb1.a_occ = 3.0f; sb1.a_free = 1.0f; sb1.a_unk = 0.5f;
  sb1.sem_cls[0] = 7;  sb1.sem_cnt[0] = 2.5f;
  sb1.sem_cls[1] = 11; sb1.sem_cnt[1] = 0.8f;
  f.sembeta_deltas.push_back({{1, 2, 3}, sb1});

  scovox::SemBetaVoxel sb2 = scovox::defaultSemBetaVoxel();
  sb2.a_occ = 1.0f; sb2.a_free = 5.0f;
  f.sembeta_deltas.push_back({{4, 5, 6}, sb2});
  return f;
}

}  // namespace

TEST(BinarySerializerV2, RoundTripPreservesAllFields) {
  auto orig = makeFrame();
  // Pin share_tsdf=true: this test exercises the dual-stream round-trip
  // (TSDF + SemBeta). The default is now share_tsdf=false (SemBeta-only),
  // which is covered separately by ShareTsdfFalseElidesTsdfPayload.
  scovox::BinarySerializerV2::Options opts;
  opts.share_tsdf = true;
  auto bytes = scovox::BinarySerializerV2::serialize(orig, opts);
  auto rt = scovox::BinarySerializerV2::deserialize(bytes);

  EXPECT_FLOAT_EQ(rt.resolution, orig.resolution);
  ASSERT_EQ(rt.tsdf_deltas.size(),    orig.tsdf_deltas.size());
  ASSERT_EQ(rt.sembeta_deltas.size(), orig.sembeta_deltas.size());

  for (size_t i = 0; i < orig.tsdf_deltas.size(); ++i) {
    EXPECT_EQ(rt.tsdf_deltas[i].coord.x, orig.tsdf_deltas[i].coord.x);
    EXPECT_EQ(rt.tsdf_deltas[i].coord.y, orig.tsdf_deltas[i].coord.y);
    EXPECT_EQ(rt.tsdf_deltas[i].coord.z, orig.tsdf_deltas[i].coord.z);
    EXPECT_FLOAT_EQ(rt.tsdf_deltas[i].data.distance, orig.tsdf_deltas[i].data.distance);
    EXPECT_FLOAT_EQ(rt.tsdf_deltas[i].data.weight,   orig.tsdf_deltas[i].data.weight);
  }

  for (size_t i = 0; i < orig.sembeta_deltas.size(); ++i) {
    EXPECT_EQ(rt.sembeta_deltas[i].coord.x, orig.sembeta_deltas[i].coord.x);
    EXPECT_FLOAT_EQ(rt.sembeta_deltas[i].data.a_occ,  orig.sembeta_deltas[i].data.a_occ);
    EXPECT_FLOAT_EQ(rt.sembeta_deltas[i].data.a_free, orig.sembeta_deltas[i].data.a_free);
    EXPECT_FLOAT_EQ(rt.sembeta_deltas[i].data.a_unk,  orig.sembeta_deltas[i].data.a_unk);
    for (int j = 0; j < scovox::K_TOP; ++j) {
      EXPECT_EQ(rt.sembeta_deltas[i].data.sem_cls[j], orig.sembeta_deltas[i].data.sem_cls[j]);
      EXPECT_FLOAT_EQ(rt.sembeta_deltas[i].data.sem_cnt[j], orig.sembeta_deltas[i].data.sem_cnt[j]);
    }
  }
}

TEST(BinarySerializerV2, EmptyFrameRoundTrips) {
  scovox::BinarySerializerV2::Frame empty;
  empty.resolution = 0.10f;
  auto bytes = scovox::BinarySerializerV2::serialize(empty);
  auto rt = scovox::BinarySerializerV2::deserialize(bytes);
  EXPECT_FLOAT_EQ(rt.resolution, 0.10f);
  EXPECT_TRUE(rt.tsdf_deltas.empty());
  EXPECT_TRUE(rt.sembeta_deltas.empty());
}

TEST(BinarySerializerV2, RejectsBadMagic) {
  std::string garbage(20, '\0');
  EXPECT_THROW(scovox::BinarySerializerV2::deserialize(garbage),
               std::runtime_error);
}

TEST(BinarySerializerV2, RejectsBadVersion) {
  // Use share_tsdf=true so makeFrame's TSDF deltas survive the round-trip
  // setup; the assertion only cares about the version byte being corrupted.
  scovox::BinarySerializerV2::Options opts;
  opts.share_tsdf = true;
  auto bytes = scovox::BinarySerializerV2::serialize(makeFrame(), opts);
  // Corrupt the version byte (offset = sizeof(MAGIC) = 4).
  bytes[4] = 99;
  EXPECT_THROW(scovox::BinarySerializerV2::deserialize(bytes),
               std::runtime_error);
}

TEST(BinarySerializerV2, FrameSizeIsBoundedAsAdvertised) {
  // Pin share_tsdf=true to verify the full dual-stream payload sizing.
  // The default-options (share_tsdf=false) sizing is covered by
  // ShareTsdfFalseElidesTsdfPayload below.
  scovox::BinarySerializerV2::Options opts;
  opts.share_tsdf = true;
  auto bytes = scovox::BinarySerializerV2::serialize(makeFrame(), opts);
  // Header is 4 (MAGIC) + 1 (VERSION) + 4 (resolution) = 9 bytes,
  // plus 4 bytes per count (TSDF + SemBeta) = 17 bytes total framing.
  // 2 TSDF * 20 + 2 SemBeta * 37 = 40 + 74 = 114 byte payload.
  // Total expected = 17 + 114 = 131 bytes.
  EXPECT_EQ(bytes.size(), size_t(131));
}

// share_tsdf=false: TSDF section elided, SemBeta payload preserved.
// Wire-bandwidth contract — what the dSCovox SemBeta-only mode buys.
TEST(BinarySerializerV2, ShareTsdfFalseElidesTsdfPayload) {
  auto orig = makeFrame();
  scovox::BinarySerializerV2::Options opts;
  opts.share_tsdf = false;
  auto bytes = scovox::BinarySerializerV2::serialize(orig, opts);

  // Same header as full frame: 9 bytes. tsdf_count=0 (4 B), no TSDF
  // payload, sembeta_count=2 (4 B), 2 * 37 = 74 B SemBeta payload.
  // Total = 9 + 4 + 0 + 4 + 74 = 91 bytes.
  EXPECT_EQ(bytes.size(), size_t(91));

  auto rt = scovox::BinarySerializerV2::deserialize(bytes);
  EXPECT_FLOAT_EQ(rt.resolution, orig.resolution);
  EXPECT_TRUE(rt.tsdf_deltas.empty());
  ASSERT_EQ(rt.sembeta_deltas.size(), orig.sembeta_deltas.size());
  for (size_t i = 0; i < orig.sembeta_deltas.size(); ++i) {
    EXPECT_EQ(rt.sembeta_deltas[i].coord.x, orig.sembeta_deltas[i].coord.x);
    EXPECT_FLOAT_EQ(rt.sembeta_deltas[i].data.a_occ,
                    orig.sembeta_deltas[i].data.a_occ);
    EXPECT_FLOAT_EQ(rt.sembeta_deltas[i].data.a_free,
                    orig.sembeta_deltas[i].data.a_free);
    EXPECT_FLOAT_EQ(rt.sembeta_deltas[i].data.a_unk,
                    orig.sembeta_deltas[i].data.a_unk);
    for (int j = 0; j < scovox::K_TOP; ++j) {
      EXPECT_EQ(rt.sembeta_deltas[i].data.sem_cls[j],
                orig.sembeta_deltas[i].data.sem_cls[j]);
      EXPECT_FLOAT_EQ(rt.sembeta_deltas[i].data.sem_cnt[j],
                     orig.sembeta_deltas[i].data.sem_cnt[j]);
    }
  }
}

// Default Options{} → share_tsdf=true → identical bytes to the
// no-Options overload. Ensures backwards compatibility.
TEST(BinarySerializerV2, DefaultOptionsMatchUnoptionedSerialize) {
  auto orig = makeFrame();
  auto bytes_default     = scovox::BinarySerializerV2::serialize(orig);
  auto bytes_with_opts   = scovox::BinarySerializerV2::serialize(
      orig, scovox::BinarySerializerV2::Options{});
  EXPECT_EQ(bytes_default, bytes_with_opts);
}
