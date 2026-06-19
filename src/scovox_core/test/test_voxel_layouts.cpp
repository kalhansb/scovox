/// @file
/// @brief Step-1 gate test for the split-grid refactor.
///
/// All real assertions are `static_assert`s in the headers themselves —
/// this TU exists to *force* both headers through the compiler in one
/// invocation, then run a couple of trivial runtime checks so the gate
/// shows up in `colcon test` output.
///
/// If this test compiles, Step 1 of the refactor is done.

#include <gtest/gtest.h>

#include "scovox/tsdf_voxel.hpp"
#include "scovox/sembeta_voxel.hpp"

namespace {

TEST(TsdfVoxelLayout, DefaultIsUnobserved) {
    auto v = scovox::defaultTsdfVoxel();
    EXPECT_FLOAT_EQ(v.distance, 0.0f);
    EXPECT_FLOAT_EQ(v.weight,   0.0f);
    EXPECT_EQ(sizeof(v), size_t(8));
}

TEST(SemBetaVoxelLayout, DefaultIsBeta11Prior) {
    // Q5 invariant: every freshly-allocated SemBetaVoxel must read back
    // as the Beta(1,1) prior with sentinel-empty slots. This factory is
    // the only path that produces that state — Bonxai's pool zero-init
    // alone does not.
    auto v = scovox::defaultSemBetaVoxel();
    EXPECT_FLOAT_EQ(v.a_occ,  1.0f);
    EXPECT_FLOAT_EQ(v.a_free, 1.0f);
    EXPECT_FLOAT_EQ(v.a_unk,  0.0f);
    for (int i = 0; i < scovox::K_TOP; ++i) {
        EXPECT_FLOAT_EQ(v.sem_cnt[i], 0.0f);
        EXPECT_EQ(v.sem_cls[i], uint16_t(0xFFFF));
    }
    EXPECT_FLOAT_EQ(v.p_occ(), 0.5f);
    EXPECT_FLOAT_EQ(v.a0(),    0.0f);
    EXPECT_EQ(sizeof(v), size_t(24));
}

TEST(SemBetaVoxelLayout, ZeroInitializedHasEmptyEvidence) {
    // Bonxai pool zero-init: leaves a_occ = a_free = 0, sem_cls = {0,0}.
    // p_occ() must return 0.5 (the s == 0 fallback), not NaN.
    // This is the exact state that defaultSemBetaVoxel() rescues us from
    // — keeping it under test makes the contrast visible if someone ever
    // breaks the factory invariant.
    scovox::SemBetaVoxel v{};
    EXPECT_FLOAT_EQ(v.a_occ,  0.0f);
    EXPECT_FLOAT_EQ(v.a_free, 0.0f);
    EXPECT_FLOAT_EQ(v.p_occ(), 0.5f);
}

}  // namespace
