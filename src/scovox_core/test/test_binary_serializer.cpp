/// @file test_binary_serializer.cpp
/// Round-trip and edge-case tests for ScovoxBinarySerializer.

#include <gtest/gtest.h>
#include <cmath>
#include "scovox/binary_serializer.hpp"

using Ser = scovox::ScovoxBinarySerializer;

// =====================================================================
// Single voxel round-trip
// =====================================================================

TEST(BinarySerializer, SingleVoxelRoundTrip) {
  std::ostringstream out(std::ios::binary);
  std::vector<std::pair<uint16_t, float>> sem = {{3, 5.5f}, {7, 2.1f}};
  Ser::serializeVoxel(10.0f, 4.0f, 1.5f, sem, out);

  std::string data = out.str();
  std::istringstream in(data, std::ios::binary);
  auto wire = Ser::deserializeVoxel(in);

  EXPECT_FLOAT_EQ(wire.a_occ, 10.0f);
  EXPECT_FLOAT_EQ(wire.a_free, 4.0f);
  EXPECT_FLOAT_EQ(wire.a_unk, 1.5f);
  ASSERT_EQ(wire.semantic_pairs.size(), 2u);
  EXPECT_EQ(wire.semantic_pairs[0].class_id, 3);
  EXPECT_NEAR(wire.semantic_pairs[0].count, 5.5f, 1e-5f);
  EXPECT_EQ(wire.semantic_pairs[1].class_id, 7);
  EXPECT_NEAR(wire.semantic_pairs[1].count, 2.1f, 1e-5f);
}

// =====================================================================
// Incremental serialize + deserialize round-trip
// =====================================================================

TEST(BinarySerializer, IncrementalRoundTrip) {
  std::vector<Ser::CoordVoxelPair> voxels;

  Ser::CoordVoxelPair v1;
  v1.x = 10; v1.y = 20; v1.z = 30;
  v1.a_occ = 15.0f; v1.a_free = 3.0f; v1.a_unk = 0.5f;
  v1.semantics = {{1, 8.0f}, {4, 2.0f}};
  voxels.push_back(v1);

  Ser::CoordVoxelPair v2;
  v2.x = 100; v2.y = 200; v2.z = 50;
  v2.a_occ = 5.0f; v2.a_free = 20.0f; v2.a_unk = 1.0f;
  v2.semantics = {{9, 3.0f}};
  voxels.push_back(v2);

  auto data = Ser::serializeIncremental(0.05f, voxels);
  auto result = Ser::deserializeIncremental(data);

  EXPECT_NEAR(result.resolution, 0.05f, 1e-6f);
  ASSERT_EQ(result.voxels.size(), 2u);

  // Find v1 and v2 in output (order may differ due to sorting)
  bool found_v1 = false, found_v2 = false;
  for (auto& rv : result.voxels) {
    if (rv.x == 10 && rv.y == 20 && rv.z == 30) {
      found_v1 = true;
      EXPECT_FLOAT_EQ(rv.a_occ, 15.0f);
      EXPECT_FLOAT_EQ(rv.a_free, 3.0f);
      EXPECT_FLOAT_EQ(rv.a_unk, 0.5f);
    }
    if (rv.x == 100 && rv.y == 200 && rv.z == 50) {
      found_v2 = true;
      EXPECT_FLOAT_EQ(rv.a_occ, 5.0f);
      EXPECT_FLOAT_EQ(rv.a_free, 20.0f);
      EXPECT_FLOAT_EQ(rv.a_unk, 1.0f);
    }
  }
  EXPECT_TRUE(found_v1);
  EXPECT_TRUE(found_v2);
}

// =====================================================================
// LZ4 compress + decompress round-trip
// =====================================================================

TEST(BinarySerializer, LZ4RoundTrip) {
  std::vector<Ser::CoordVoxelPair> voxels;
  for (int i = 0; i < 100; ++i) {
    Ser::CoordVoxelPair v;
    v.x = i; v.y = i * 2; v.z = i * 3;
    v.a_occ = float(i) + 1.0f; v.a_free = 1.0f; v.a_unk = 0.0f;
    v.semantics = {{static_cast<uint16_t>(i % 10), float(i)}};
    voxels.push_back(v);
  }

  auto data = Ser::serializeIncremental(0.1f, voxels);
  auto compressed = Ser::compressLZ4(data);
  EXPECT_FALSE(compressed.empty());
  EXPECT_LT(compressed.size(), data.size());  // should compress

  auto decompressed = Ser::decompressLZ4(compressed);
  EXPECT_EQ(decompressed, data);

  auto result = Ser::deserializeIncremental(decompressed);
  EXPECT_EQ(result.voxels.size(), 100u);
}

// =====================================================================
// Negative coordinates
// =====================================================================

TEST(BinarySerializer, NegativeCoordinates8bit) {
  // Extents < 256 -> uses 8-bit encoding
  std::vector<Ser::CoordVoxelPair> voxels;
  Ser::CoordVoxelPair v1;
  v1.x = -10; v1.y = -20; v1.z = -30;
  v1.a_occ = 5.0f; v1.a_free = 2.0f; v1.a_unk = 0.0f;
  voxels.push_back(v1);

  Ser::CoordVoxelPair v2;
  v2.x = 10; v2.y = 20; v2.z = 30;
  v2.a_occ = 3.0f; v2.a_free = 7.0f; v2.a_unk = 0.0f;
  voxels.push_back(v2);

  auto data = Ser::serializeIncremental(0.1f, voxels);
  auto result = Ser::deserializeIncremental(data);

  ASSERT_EQ(result.voxels.size(), 2u);
  bool found_neg = false, found_pos = false;
  for (auto& rv : result.voxels) {
    if (rv.x == -10 && rv.y == -20 && rv.z == -30) found_neg = true;
    if (rv.x == 10 && rv.y == 20 && rv.z == 30) found_pos = true;
  }
  EXPECT_TRUE(found_neg) << "Negative coordinate voxel not recovered";
  EXPECT_TRUE(found_pos) << "Positive coordinate voxel not recovered";
}

TEST(BinarySerializer, NegativeCoordinates16bit) {
  // Extents >= 256, < 65536 -> uses 16-bit encoding
  std::vector<Ser::CoordVoxelPair> voxels;
  Ser::CoordVoxelPair v1;
  v1.x = -200; v1.y = -300; v1.z = -100;
  v1.a_occ = 5.0f; v1.a_free = 2.0f; v1.a_unk = 0.0f;
  voxels.push_back(v1);

  Ser::CoordVoxelPair v2;
  v2.x = 200; v2.y = 300; v2.z = 100;
  v2.a_occ = 3.0f; v2.a_free = 7.0f; v2.a_unk = 0.0f;
  voxels.push_back(v2);

  auto data = Ser::serializeIncremental(0.1f, voxels);
  auto result = Ser::deserializeIncremental(data);

  ASSERT_EQ(result.voxels.size(), 2u);
  bool found_neg = false, found_pos = false;
  for (auto& rv : result.voxels) {
    if (rv.x == -200 && rv.y == -300 && rv.z == -100) found_neg = true;
    if (rv.x == 200 && rv.y == 300 && rv.z == 100) found_pos = true;
  }
  EXPECT_TRUE(found_neg);
  EXPECT_TRUE(found_pos);
}

TEST(BinarySerializer, NegativeCoordinates32bit) {
  // Extents >= 65536 -> uses 32-bit encoding
  std::vector<Ser::CoordVoxelPair> voxels;
  Ser::CoordVoxelPair v1;
  v1.x = -50000; v1.y = -40000; v1.z = -30000;
  v1.a_occ = 5.0f; v1.a_free = 2.0f; v1.a_unk = 0.0f;
  voxels.push_back(v1);

  Ser::CoordVoxelPair v2;
  v2.x = 50000; v2.y = 40000; v2.z = 30000;
  v2.a_occ = 3.0f; v2.a_free = 7.0f; v2.a_unk = 0.0f;
  voxels.push_back(v2);

  auto data = Ser::serializeIncremental(0.1f, voxels);
  auto result = Ser::deserializeIncremental(data);

  ASSERT_EQ(result.voxels.size(), 2u);
  bool found_neg = false, found_pos = false;
  for (auto& rv : result.voxels) {
    if (rv.x == -50000 && rv.y == -40000 && rv.z == -30000) found_neg = true;
    if (rv.x == 50000 && rv.y == 40000 && rv.z == 30000) found_pos = true;
  }
  EXPECT_TRUE(found_neg);
  EXPECT_TRUE(found_pos);
}

// =====================================================================
// Edge cases
// =====================================================================

TEST(BinarySerializer, EmptyVoxelList) {
  std::vector<Ser::CoordVoxelPair> empty;
  auto data = Ser::serializeIncremental(0.1f, empty);
  auto result = Ser::deserializeIncremental(data);
  EXPECT_TRUE(result.voxels.empty());
}

TEST(BinarySerializer, ParseResolution) {
  std::vector<Ser::CoordVoxelPair> voxels;
  Ser::CoordVoxelPair v;
  v.x = 1; v.y = 2; v.z = 3;
  v.a_occ = 1.0f; v.a_free = 1.0f; v.a_unk = 0.0f;
  voxels.push_back(v);

  auto data = Ser::serializeIncremental(0.05f, voxels);
  float res = Ser::parseResolution(data);
  EXPECT_NEAR(res, 0.05f, 1e-6f);
}

TEST(BinarySerializer, LZ4EmptyData) {
  auto compressed = Ser::compressLZ4("");
  EXPECT_TRUE(compressed.empty());

  auto decompressed = Ser::decompressLZ4({});
  EXPECT_TRUE(decompressed.empty());
}

TEST(BinarySerializer, MagicValidation) {
  // Corrupt data should return empty result
  std::string garbage = "this is not a valid scovox payload";
  auto result = Ser::deserializeIncremental(garbage);
  EXPECT_TRUE(result.voxels.empty());

  float res = Ser::parseResolution(garbage);
  EXPECT_FLOAT_EQ(res, 0.0f);
}

TEST(BinarySerializer, VersionMismatchRejects) {
  // Serialize valid data, then corrupt the version byte
  std::vector<Ser::CoordVoxelPair> voxels;
  Ser::CoordVoxelPair v;
  v.x = 1; v.y = 2; v.z = 3;
  v.a_occ = 5.0f; v.a_free = 2.0f; v.a_unk = 0.0f;
  voxels.push_back(v);

  auto data = Ser::serializeIncremental(0.1f, voxels);
  ASSERT_GT(data.size(), 5u);
  // Byte 4 is the version (after 4-byte magic)
  data[4] = 99;  // invalid version
  auto result = Ser::deserializeIncremental(data);
  EXPECT_TRUE(result.voxels.empty());
}

TEST(BinarySerializer, SemanticMassConservation) {
  // Voxel with >K_MAX semantic classes — overflow goes to a_unk
  std::vector<std::pair<uint16_t, float>> sem = {{1, 10.0f}, {2, 5.0f}, {3, 2.0f}};
  Ser::WireVoxel wire;
  Ser::prepareWireVoxel(10.0f, 4.0f, 1.0f, sem, wire);

  // K_MAX = 2, so class 3 (2.0f) should be added to a_unk
  EXPECT_EQ(wire.k, 2u);
  EXPECT_FLOAT_EQ(wire.a_unk, 1.0f + 2.0f);  // original + dropped
  float total_sem = wire.a_unk;
  for (auto& p : wire.semantic_pairs) total_sem += p.count;
  // Total = 10 + 5 + 1 + 2 = 18 (original a_unk=1 + all sem evidence)
  EXPECT_NEAR(total_sem, 18.0f, 1e-5f);
}

// =====================================================================
// top_k mass-conservation regression tests
//
// Pre-fix bugs:
//   - serializeIncremental pre-truncated v.semantics at top_k_limit
//     before prepareWireVoxel could fold the overflow into a_unk, so
//     a producer with top_k=1 silently lost the second slot's mass.
//   - deserializeIncremental dropped trailing wire slots beyond
//     top_k_limit without folding into wv.a_unk, so a consumer with
//     top_k=1 silently lost mass on every binary.
// =====================================================================

namespace {
inline float voxelTotalSem(const Ser::IncrementalVoxel& v) {
  float s = v.a_unk;
  for (auto& p : v.top) s += p.second;
  return s;
}
}  // namespace

TEST(BinarySerializer, ProducerTopKBelowKMaxPreservesMass) {
  // Producer with top_k=1 used to truncate v.semantics before prepareWireVoxel
  // saw the dropped slot. Post-fix: top_k is ignored on the producer side and
  // K_MAX governs the wire format, so all slots survive (overflow → a_unk).
  std::vector<Ser::CoordVoxelPair> voxels;
  Ser::CoordVoxelPair v;
  v.x = 1; v.y = 2; v.z = 3;
  v.a_occ = 10.0f; v.a_free = 1.0f; v.a_unk = 0.0f;
  v.semantics = {{1, 8.0f}, {2, 5.0f}};  // two slots, both above zero
  voxels.push_back(v);

  // Producer top_k=1 (ignored post-fix), consumer top_k=2 (default).
  auto data = Ser::serializeIncremental(0.1f, voxels, /*top_k=*/1);
  auto result = Ser::deserializeIncremental(data, /*top_k=*/2);

  ASSERT_EQ(result.voxels.size(), 1u);
  // Total semantic mass on the consumer side must equal the producer's
  // input total: 8 + 5 + 0 = 13. Pre-fix this was 8 (slot 2 lost).
  EXPECT_NEAR(voxelTotalSem(result.voxels[0]), 13.0f, 1e-5f)
      << "producer top_k truncation must not lose semantic mass";
}

TEST(BinarySerializer, ConsumerTopKBelowKMaxFoldsOverflowIntoAUnk) {
  // Consumer with top_k=1 used to drop trailing wire slots without folding
  // into wv.a_unk. Post-fix: trailing slots are added to a_unk.
  std::vector<Ser::CoordVoxelPair> voxels;
  Ser::CoordVoxelPair v;
  v.x = 7; v.y = 8; v.z = 9;
  v.a_occ = 10.0f; v.a_free = 1.0f; v.a_unk = 0.5f;
  v.semantics = {{1, 8.0f}, {2, 5.0f}};
  voxels.push_back(v);

  // Producer default (top_k=2), consumer requests only the strongest slot.
  auto data = Ser::serializeIncremental(0.1f, voxels);
  auto result = Ser::deserializeIncremental(data, /*top_k=*/1);

  ASSERT_EQ(result.voxels.size(), 1u);
  const auto& rv = result.voxels[0];

  // Strongest slot is class 1 with count 8.
  EXPECT_EQ(rv.top[0].first, 1);
  EXPECT_NEAR(rv.top[0].second, 8.0f, 1e-5f);
  // The second slot (class 2, count 5) must have been folded into a_unk
  // along with the original 0.5: a_unk = 0.5 + 5.0 = 5.5.
  EXPECT_NEAR(rv.a_unk, 5.5f, 1e-5f)
      << "consumer top_k truncation must fold dropped slots into a_unk";
  // Total mass invariant: 8 + 5 + 0.5 = 13.5.
  EXPECT_NEAR(voxelTotalSem(rv), 13.5f, 1e-5f);
}

TEST(BinarySerializer, RoundTripPreservesMassAcrossTopKConfigs) {
  // Sweep producer x consumer top_k combinations and verify total mass
  // is preserved end-to-end in every case.
  std::vector<Ser::CoordVoxelPair> voxels;
  Ser::CoordVoxelPair v;
  v.x = 4; v.y = 5; v.z = 6;
  v.a_occ = 12.0f; v.a_free = 3.0f; v.a_unk = 1.0f;
  v.semantics = {{10, 7.0f}, {20, 3.0f}};
  voxels.push_back(v);

  const float expected_total = 1.0f + 7.0f + 3.0f;  // a_unk + slots = 11

  for (int prod_k : {1, 2}) {
    for (int cons_k : {1, 2}) {
      auto data = Ser::serializeIncremental(0.1f, voxels, prod_k);
      auto result = Ser::deserializeIncremental(data, cons_k);
      ASSERT_EQ(result.voxels.size(), 1u)
          << "prod_k=" << prod_k << " cons_k=" << cons_k;
      EXPECT_NEAR(voxelTotalSem(result.voxels[0]), expected_total, 1e-5f)
          << "mass leak at prod_k=" << prod_k << " cons_k=" << cons_k;
    }
  }
}
