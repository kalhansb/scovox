/// @file
/// @brief Gate tests for SemObs — the per-ray semantic observation.
///
/// The deposit path used to re-read a dense per-class softmax at every voxel a
/// ray touches. It now reads a compact list prepared once per ray. Two claims
/// are pinned here:
///
///   - with the cap off, the preparation is a pure change of REPRESENTATION.
///     The same classes survive, in the same order, carrying the same
///     probabilities, so the arithmetic downstream cannot move;
///   - with the cap on, it is a change of CONTENT, and exactly one: the classes
///     outside the cap are dropped, and their mass is not handed to the
///     survivors.

#include <gtest/gtest.h>

#include <vector>

#include "scovox/sem_obs.hpp"

namespace {

using scovox::SemObs;
using scovox::prepareSemObs;

std::vector<float> softmax13(std::initializer_list<std::pair<int, float>> v) {
  std::vector<float> p(13, 0.f);
  for (const auto& kv : v) p[kv.first] = kv.second;
  return p;
}

}  // namespace

// ---------------------------------------------------------------------------
// Cap off: representation only
// ---------------------------------------------------------------------------

TEST(SemObs, DropsZerosAndKeepsAscendingClassOrder) {
  const auto p = softmax13({{9, 0.2f}, {2, 0.5f}, {5, 0.3f}});
  SemObs o;
  prepareSemObs(&p, 0, o);

  ASSERT_EQ(o.e.size(), 3u);
  EXPECT_EQ(o.e[0].cls, 2);
  EXPECT_EQ(o.e[1].cls, 5);
  EXPECT_EQ(o.e[2].cls, 9);
  EXPECT_TRUE(o.present);
}

TEST(SemObs, ASumAtOrBelowOneIsLeftAlone) {
  // `norm` only ever shrinks a sum that exceeds 1. An under-confident
  // observation keeps its own probabilities and the missing mass becomes OTHER
  // downstream, so scaling it up here would invent confidence.
  const auto p = softmax13({{1, 0.2f}, {4, 0.3f}});
  SemObs o;
  prepareSemObs(&p, 0, o);

  ASSERT_EQ(o.e.size(), 2u);
  EXPECT_FLOAT_EQ(o.e[0].p, 0.2f);
  EXPECT_FLOAT_EQ(o.e[1].p, 0.3f);
}

TEST(SemObs, ASumAboveOneIsScaledToOne) {
  const auto p = softmax13({{1, 1.0f}, {4, 1.0f}});
  SemObs o;
  prepareSemObs(&p, 0, o);

  ASSERT_EQ(o.e.size(), 2u);
  EXPECT_FLOAT_EQ(o.e[0].p, 0.5f);
  EXPECT_FLOAT_EQ(o.e[1].p, 0.5f);
}

TEST(SemObs, ArgmaxTiesGoToTheLowerClass) {
  const auto p = softmax13({{3, 0.4f}, {7, 0.4f}});
  SemObs o;
  prepareSemObs(&p, 0, o);

  ASSERT_GE(o.argmax, 0);
  EXPECT_EQ(o.e[o.argmax].cls, 3);
}

TEST(SemObs, ArgmaxIsTheLargestClass) {
  const auto p = softmax13({{0, 0.1f}, {6, 0.7f}, {11, 0.2f}});
  SemObs o;
  prepareSemObs(&p, 0, o);

  ASSERT_GE(o.argmax, 0);
  EXPECT_EQ(o.e[o.argmax].cls, 6);
}

// ---------------------------------------------------------------------------
// present vs empty: two different kinds of nothing
// ---------------------------------------------------------------------------

TEST(SemObs, NoObservationIsNotAnEmptyOne) {
  // A ray that carried no softmax and a ray that carried one naming no class
  // are both `empty()`, but only the second is `present()`. The deposit path
  // reads the difference: an observation that happened still commits its share
  // (unattributed, into OTHER), one that did not happen commits nothing.
  SemObs absent;
  prepareSemObs(nullptr, 0, absent);
  EXPECT_TRUE(absent.empty());
  EXPECT_FALSE(absent.present);
  EXPECT_EQ(absent.argmax, -1);

  const std::vector<float> all_zero(13, 0.f);
  SemObs blank;
  prepareSemObs(&all_zero, 0, blank);
  EXPECT_TRUE(blank.empty());
  EXPECT_TRUE(blank.present);
  EXPECT_EQ(blank.argmax, -1);
}

TEST(SemObs, AnEmptyVectorIsAnAbsentObservation) {
  const std::vector<float> none;
  SemObs o;
  prepareSemObs(&none, 0, o);
  EXPECT_FALSE(o.present);
}

TEST(SemObs, PreparingReusesTheBufferWithoutCarryingStateOver) {
  const auto wide = softmax13({{1, 0.3f}, {2, 0.3f}, {3, 0.3f}});
  SemObs o;
  prepareSemObs(&wide, 0, o);
  ASSERT_EQ(o.e.size(), 3u);

  prepareSemObs(nullptr, 0, o);
  EXPECT_TRUE(o.empty());
  EXPECT_FALSE(o.present);
  EXPECT_EQ(o.argmax, -1);
}

// ---------------------------------------------------------------------------
// Cap on
// ---------------------------------------------------------------------------

TEST(SemObs, CapKeepsTheLargestClassesInAscendingOrder) {
  const auto p = softmax13({{1, 0.05f}, {4, 0.40f}, {8, 0.25f}, {12, 0.30f}});
  SemObs o;
  prepareSemObs(&p, 2, o);

  ASSERT_EQ(o.e.size(), 2u);
  EXPECT_EQ(o.e[0].cls, 4);
  EXPECT_EQ(o.e[1].cls, 12);
  EXPECT_EQ(o.e[o.argmax].cls, 4);
}

TEST(SemObs, CapDoesNotRenormalizeOntoTheSurvivors) {
  // The dropped mass is not redistributed: a capped observation reads as
  // "these classes, and I decline to guess about the rest". Downstream the
  // remainder of `class_share` stays in OTHER.
  const auto p = softmax13({{1, 0.5f}, {2, 0.3f}, {3, 0.2f}});
  SemObs o;
  prepareSemObs(&p, 1, o);

  ASSERT_EQ(o.e.size(), 1u);
  EXPECT_EQ(o.e[0].cls, 1);
  EXPECT_FLOAT_EQ(o.e[0].p, 0.5f);  // NOT 1.0
}

TEST(SemObs, CapTiesGoToTheLowerClass) {
  const auto p = softmax13({{2, 0.3f}, {6, 0.3f}, {9, 0.1f}});
  SemObs o;
  prepareSemObs(&p, 1, o);

  ASSERT_EQ(o.e.size(), 1u);
  EXPECT_EQ(o.e[0].cls, 2);
}

TEST(SemObs, ACapWiderThanTheObservationSelectsNothing) {
  const auto p = softmax13({{1, 0.4f}, {5, 0.35f}, {6, 0.25f}});
  SemObs wide, off;
  prepareSemObs(&p, 8, wide);
  prepareSemObs(&p, 0, off);

  ASSERT_EQ(wide.e.size(), off.e.size());
  for (size_t i = 0; i < off.e.size(); ++i) {
    EXPECT_EQ(wide.e[i].cls, off.e[i].cls);
    EXPECT_FLOAT_EQ(wide.e[i].p, off.e[i].p);
  }
  EXPECT_EQ(wide.argmax, off.argmax);
}

TEST(SemObs, CapNormalisesOnlyWhatSurvivedTheCut) {
  // Truncation runs before the sum, so an observation whose survivors already
  // sum to at most 1 is left alone even when the full softmax exceeded it.
  const auto p = softmax13({{0, 0.6f}, {1, 0.6f}, {2, 0.6f}});
  SemObs o;
  prepareSemObs(&p, 1, o);

  ASSERT_EQ(o.e.size(), 1u);
  EXPECT_FLOAT_EQ(o.e[0].p, 0.6f);
}
