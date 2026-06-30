// Unit tests for scovox::TopkProvider — the file-backed .topk soft-prob loader
// extracted from SCovoxNode. Exercises the binary parser, the cache, the fill
// helpers, and the corrupt-file fallbacks (the bugs the inline code's comments
// document: bad/empty header → bad_alloc, and the gcount short-read footgun)
// without spinning up a ROS graph: logging only needs a logger + clock, which
// work without rclcpp::init().

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <rclcpp/clock.hpp>
#include <rclcpp/logger.hpp>

#include "scovox/topk_provider.hpp"

namespace {

std::string topk_path(const std::string& dir, uint16_t frame) {
  char path[1024];
  std::snprintf(path, sizeof(path), "%s/%06u.topk", dir.c_str(), (unsigned)frame);
  return path;
}

// Pointcloud layout: [u32 N][u8 C][N*C u8 probs]
void write_point_topk(const std::string& dir, uint16_t frame, uint32_t N,
                      uint8_t C, const std::vector<uint8_t>& probs) {
  std::ofstream f(topk_path(dir, frame), std::ios::binary | std::ios::trunc);
  f.write(reinterpret_cast<const char*>(&N), 4);
  f.write(reinterpret_cast<const char*>(&C), 1);
  f.write(reinterpret_cast<const char*>(probs.data()),
          static_cast<std::streamsize>(probs.size()));
}

// Image layout: [u16 H][u16 W][u8 C][H*W*C u8 probs]
void write_image_topk(const std::string& dir, uint16_t frame, uint16_t H,
                      uint16_t W, uint8_t C, const std::vector<uint8_t>& probs) {
  std::ofstream f(topk_path(dir, frame), std::ios::binary | std::ios::trunc);
  f.write(reinterpret_cast<const char*>(&H), 2);
  f.write(reinterpret_cast<const char*>(&W), 2);
  f.write(reinterpret_cast<const char*>(&C), 1);
  f.write(reinterpret_cast<const char*>(probs.data()),
          static_cast<std::streamsize>(probs.size()));
}

scovox::TopkProvider make_provider(const std::string& dir, int max_sem) {
  return scovox::TopkProvider(rclcpp::get_logger("test_topk"),
                              rclcpp::Clock::make_shared(), dir, max_sem);
}

}  // namespace

TEST(TopkProvider, DisabledWhenDirEmpty) {
  auto p = make_provider("", 14);
  EXPECT_FALSE(p.enabled());
}

TEST(TopkProvider, EnabledWhenDirSet) {
  auto p = make_provider(::testing::TempDir(), 14);
  EXPECT_TRUE(p.enabled());
}

TEST(TopkProvider, PointModeLoadAndFill) {
  const std::string dir = ::testing::TempDir();
  // N=2 rows, C=4 slots. row0: class 1 @ 1.0; row1: class 2 @ ~0.502.
  write_point_topk(dir, /*frame=*/10, /*N=*/2, /*C=*/4,
                   {0, 255, 0, 0,
                    0, 0, 128, 0});
  auto p = make_provider(dir, /*max_sem=*/4);

  ASSERT_TRUE(p.loadFrame(10, /*image_mode=*/false));
  EXPECT_EQ(p.loadSuccess(), 1u);

  std::vector<float> cp(4, -1.f);
  ASSERT_TRUE(p.fillPoint(0, cp));
  EXPECT_FLOAT_EQ(cp[0], 0.f);          // slot 0 (unknown) always skipped
  EXPECT_FLOAT_EQ(cp[1], 1.f);
  EXPECT_FLOAT_EQ(cp[2], 0.f);

  ASSERT_TRUE(p.fillPoint(1, cp));
  EXPECT_FLOAT_EQ(cp[2], 128.f / 255.f);
  EXPECT_FLOAT_EQ(cp[1], 0.f);

  // Out-of-range row → no observation.
  EXPECT_FALSE(p.fillPoint(2, cp));
}

TEST(TopkProvider, Slot0OnlyRowIsDegenerate) {
  const std::string dir = ::testing::TempDir();
  // Only slot 0 carries mass → fill must report "no semantic observation".
  write_point_topk(dir, /*frame=*/11, /*N=*/1, /*C=*/4, {255, 0, 0, 0});
  auto p = make_provider(dir, 4);
  ASSERT_TRUE(p.loadFrame(11, false));
  std::vector<float> cp(4, -1.f);
  EXPECT_FALSE(p.fillPoint(0, cp));
}

TEST(TopkProvider, MaxSemCapsSlots) {
  const std::string dir = ::testing::TempDir();
  // C=4 in file but max_sem=2 → only slot 1 is read; slot 2/3 ignored.
  write_point_topk(dir, /*frame=*/12, /*N=*/1, /*C=*/4, {0, 200, 255, 255});
  auto p = make_provider(dir, /*max_sem=*/2);
  ASSERT_TRUE(p.loadFrame(12, false));
  std::vector<float> cp(2, -1.f);
  ASSERT_TRUE(p.fillPoint(0, cp));
  EXPECT_FLOAT_EQ(cp[1], 200.f / 255.f);
  EXPECT_EQ(cp.size(), 2u);             // caller-sized vector untouched in length
}

TEST(TopkProvider, ImageModeLoadAndFill) {
  const std::string dir = ::testing::TempDir();
  // H=1, W=2, C=4. pixel(0,0): class1@1.0; pixel(1,0): class3@1.0.
  write_image_topk(dir, /*frame=*/20, /*H=*/1, /*W=*/2, /*C=*/4,
                   {0, 255, 0, 0,
                    0, 0, 0, 255});
  auto p = make_provider(dir, 4);
  ASSERT_TRUE(p.loadFrame(20, /*image_mode=*/true));

  std::vector<float> cp(4, -1.f);
  ASSERT_TRUE(p.fillImage(/*u=*/0, /*v=*/0, cp));
  EXPECT_FLOAT_EQ(cp[1], 1.f);
  ASSERT_TRUE(p.fillImage(/*u=*/1, /*v=*/0, cp));
  EXPECT_FLOAT_EQ(cp[3], 1.f);

  // Out-of-bounds pixel → false; point-mode fill on an image cache → false.
  EXPECT_FALSE(p.fillImage(/*u=*/2, /*v=*/0, cp));
  EXPECT_FALSE(p.fillPoint(0, cp));
}

TEST(TopkProvider, CacheHitDoesNotDoubleCount) {
  const std::string dir = ::testing::TempDir();
  write_point_topk(dir, /*frame=*/30, 1, 4, {0, 255, 0, 0});
  auto p = make_provider(dir, 4);
  ASSERT_TRUE(p.loadFrame(30, false));
  ASSERT_TRUE(p.loadFrame(30, false));   // cache hit, same frame + mode
  EXPECT_EQ(p.loadSuccess(), 1u);        // counted once, not twice
  EXPECT_EQ(p.loadFailure(), 0u);
}

TEST(TopkProvider, MissingFileFallsBack) {
  auto p = make_provider(::testing::TempDir(), 4);
  EXPECT_FALSE(p.loadFrame(/*frame=*/9999, false));  // no such file
  EXPECT_EQ(p.loadFailure(), 1u);
  EXPECT_EQ(p.loadSuccess(), 0u);
  std::vector<float> cp(4, 0.f);
  EXPECT_FALSE(p.fillPoint(0, cp));      // invalid cache → no fill
}

TEST(TopkProvider, EmptyHeaderFallsBack) {
  const std::string dir = ::testing::TempDir();
  { std::ofstream f(topk_path(dir, 40), std::ios::binary | std::ios::trunc); }  // 0 bytes
  auto p = make_provider(dir, 4);
  EXPECT_FALSE(p.loadFrame(40, false));  // total==0 / bad header guard
  EXPECT_EQ(p.loadFailure(), 1u);
}

TEST(TopkProvider, TruncatedPayloadFallsBack) {
  const std::string dir = ::testing::TempDir();
  // Header promises N=4, C=4 (16 payload bytes) but only 8 are present.
  {
    std::ofstream f(topk_path(dir, 41), std::ios::binary | std::ios::trunc);
    uint32_t N = 4; uint8_t C = 4;
    f.write(reinterpret_cast<const char*>(&N), 4);
    f.write(reinterpret_cast<const char*>(&C), 1);
    std::vector<uint8_t> partial(8, 7);
    f.write(reinterpret_cast<const char*>(partial.data()), 8);
  }
  auto p = make_provider(dir, 4);
  EXPECT_FALSE(p.loadFrame(41, false));  // gcount() != total → short-read guard
  EXPECT_EQ(p.loadFailure(), 1u);
}

TEST(TopkProvider, Dequantize) {
  EXPECT_FLOAT_EQ(scovox::TopkProvider::dequantize(0), 0.f);
  EXPECT_FLOAT_EQ(scovox::TopkProvider::dequantize(255), 1.f);
}
