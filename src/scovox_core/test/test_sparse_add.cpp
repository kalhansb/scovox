/// @file test_sparse_add.cpp
/// Tests for sparse_add eviction and mass conservation.

#include <gtest/gtest.h>
#include "scovox/voxel.hpp"

using namespace scovox;

static Voxel makeEmpty() {
  Voxel v{};
  return v;
}

// =====================================================================
// Basic insertion
// =====================================================================

TEST(SparseAdd, InsertIntoEmptySlot) {
  Voxel v = makeEmpty();
  sparse_add(v.sem_cnt, v.sem_cls, 5, 3.0f, &v.a_unk);
  EXPECT_EQ(v.sem_cls[0], 5);
  EXPECT_FLOAT_EQ(v.sem_cnt[0], 3.0f);
  EXPECT_FLOAT_EQ(v.a_unk, 0.0f);
}

TEST(SparseAdd, AccumulateExistingClass) {
  Voxel v = makeEmpty();
  sparse_add(v.sem_cnt, v.sem_cls, 3, 2.0f, &v.a_unk);
  sparse_add(v.sem_cnt, v.sem_cls, 3, 5.0f, &v.a_unk);
  float cnt = 0.f;
  for (int i = 0; i < K_TOP; ++i)
    if (v.sem_cls[i] == 3) cnt = v.sem_cnt[i];
  EXPECT_FLOAT_EQ(cnt, 7.0f);
  EXPECT_FLOAT_EQ(v.a_unk, 0.0f);
}

TEST(SparseAdd, FillAllSlots) {
  Voxel v = makeEmpty();
  for (int i = 0; i < K_TOP; ++i)
    sparse_add(v.sem_cnt, v.sem_cls, static_cast<uint16_t>(i + 1), float(i + 1), &v.a_unk);
  for (int i = 0; i < K_TOP; ++i) {
    EXPECT_EQ(v.sem_cls[i], static_cast<uint16_t>(i + 1));
    EXPECT_FLOAT_EQ(v.sem_cnt[i], float(i + 1));
  }
  EXPECT_FLOAT_EQ(v.a_unk, 0.0f);
}

// =====================================================================
// Eviction — new class replaces weakest
// =====================================================================

TEST(SparseAdd, EvictsWeakestWhenNewIsStronger) {
  Voxel v = makeEmpty();
  // Fill: cls 1 -> 2.0, cls 2 -> 5.0
  sparse_add(v.sem_cnt, v.sem_cls, 1, 2.0f, &v.a_unk);
  sparse_add(v.sem_cnt, v.sem_cls, 2, 5.0f, &v.a_unk);

  // New class 3 with 10.0 > min(2.0, 5.0) -> evicts cls 1
  sparse_add(v.sem_cnt, v.sem_cls, 3, 10.0f, &v.a_unk);

  bool has3 = false;
  for (int i = 0; i < K_TOP; ++i)
    if (v.sem_cls[i] == 3 && v.sem_cnt[i] == 10.0f) has3 = true;
  EXPECT_TRUE(has3);
  // Evicted mass (2.0 from cls 1) should go to a_unk
  EXPECT_FLOAT_EQ(v.a_unk, 2.0f);
}

TEST(SparseAdd, DropsWeakNewClassToAUnk) {
  Voxel v = makeEmpty();
  // Fill: cls 1 -> 10.0, cls 2 -> 20.0
  sparse_add(v.sem_cnt, v.sem_cls, 1, 10.0f, &v.a_unk);
  sparse_add(v.sem_cnt, v.sem_cls, 2, 20.0f, &v.a_unk);

  // New class 3 with 5.0 <= min(10.0, 20.0) -> dropped entirely
  sparse_add(v.sem_cnt, v.sem_cls, 3, 5.0f, &v.a_unk);

  // Slots unchanged
  bool has3 = false;
  for (int i = 0; i < K_TOP; ++i)
    if (v.sem_cls[i] == 3) has3 = true;
  EXPECT_FALSE(has3);
  // Dropped mass should go to a_unk
  EXPECT_FLOAT_EQ(v.a_unk, 5.0f);
}

// =====================================================================
// Mass conservation
// =====================================================================

TEST(SparseAdd, MassConservedOnEviction) {
  Voxel v = makeEmpty();
  v.a_unk = 1.0f;
  sparse_add(v.sem_cnt, v.sem_cls, 1, 3.0f, &v.a_unk);
  sparse_add(v.sem_cnt, v.sem_cls, 2, 4.0f, &v.a_unk);

  float total_before = v.a_unk;
  for (int i = 0; i < K_TOP; ++i) total_before += v.sem_cnt[i];

  // Evict cls 1 (3.0) with cls 3 (10.0)
  sparse_add(v.sem_cnt, v.sem_cls, 3, 10.0f, &v.a_unk);

  float total_after = v.a_unk;
  for (int i = 0; i < K_TOP; ++i) total_after += v.sem_cnt[i];

  // total_after should equal total_before + 10.0 (the new evidence added)
  EXPECT_NEAR(total_after, total_before + 10.0f, 1e-5f);
}

TEST(SparseAdd, MassConservedOnDrop) {
  Voxel v = makeEmpty();
  v.a_unk = 0.5f;
  sparse_add(v.sem_cnt, v.sem_cls, 1, 10.0f, &v.a_unk);
  sparse_add(v.sem_cnt, v.sem_cls, 2, 20.0f, &v.a_unk);

  float total_before = v.a_unk;
  for (int i = 0; i < K_TOP; ++i) total_before += v.sem_cnt[i];

  // Drop cls 3 (2.0 <= 10.0)
  sparse_add(v.sem_cnt, v.sem_cls, 3, 2.0f, &v.a_unk);

  float total_after = v.a_unk;
  for (int i = 0; i < K_TOP; ++i) total_after += v.sem_cnt[i];

  EXPECT_NEAR(total_after, total_before + 2.0f, 1e-5f);
}

TEST(SparseAdd, NullAUnkStillWorks) {
  // Backward compat: nullptr a_unk should not crash
  Voxel v = makeEmpty();
  sparse_add(v.sem_cnt, v.sem_cls, 1, 5.0f);
  sparse_add(v.sem_cnt, v.sem_cls, 2, 3.0f);
  sparse_add(v.sem_cnt, v.sem_cls, 3, 10.0f);  // evicts, but no a_unk to update
  EXPECT_FLOAT_EQ(v.a_unk, 0.0f);  // unchanged
}

// =====================================================================
// Edge cases
// =====================================================================

TEST(SparseAdd, ZeroIncrementLandsInEmptySlot) {
  // Pin the zero-increment contract. Tracing sparse_add() on an empty voxel:
  //   - match loop:  needs sem_cnt[i] > 0.0f, false for every empty slot -> no match.
  //   - empty loop:  sem_cnt[i] <= 0.0f is true for slot 0 (0.0f) -> the slot is
  //                  claimed with sem_cls[0] = cls and sem_cnt[0] = inc (== 0.0f).
  // So a zero increment is NOT dropped: it occupies slot 0 with the given class id
  // and a zero count, and leaves a_unk untouched (no eviction/drop path is hit).
  Voxel v = makeEmpty();
  sparse_add(v.sem_cnt, v.sem_cls, 1, 0.0f, &v.a_unk);
  EXPECT_EQ(v.sem_cls[0], 1);
  EXPECT_FLOAT_EQ(v.sem_cnt[0], 0.0f);
  EXPECT_FLOAT_EQ(v.a_unk, 0.0f);

  // Because the slot now holds a zero count, the match guard (sem_cnt[i] > 0.0f)
  // still treats it as empty: a second zero-increment for a different class
  // re-claims the SAME slot rather than accumulating, and a_unk stays 0.
  sparse_add(v.sem_cnt, v.sem_cls, 7, 0.0f, &v.a_unk);
  EXPECT_EQ(v.sem_cls[0], 7);
  EXPECT_FLOAT_EQ(v.sem_cnt[0], 0.0f);
  EXPECT_FLOAT_EQ(v.a_unk, 0.0f);
}

TEST(SparseAdd, RepeatedEvictionsAccumulateAUnk) {
  Voxel v = makeEmpty();
  sparse_add(v.sem_cnt, v.sem_cls, 1, 1.0f, &v.a_unk);
  sparse_add(v.sem_cnt, v.sem_cls, 2, 2.0f, &v.a_unk);

  // Evict cls 1 (1.0) with cls 3 (3.0)
  sparse_add(v.sem_cnt, v.sem_cls, 3, 3.0f, &v.a_unk);
  EXPECT_FLOAT_EQ(v.a_unk, 1.0f);

  // Evict cls 2 (2.0) with cls 4 (4.0)
  sparse_add(v.sem_cnt, v.sem_cls, 4, 4.0f, &v.a_unk);
  EXPECT_FLOAT_EQ(v.a_unk, 1.0f + 2.0f);  // accumulated
}
