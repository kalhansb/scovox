// Test-plan 9 (DESIGN_gen33.md §7) — COUNTER LIVENESS.
//
// The per-source fusion counters must keep publishing while nothing arrives.
// The planner's drain-release test reads them as a level against a hold-start
// baseline, and a counter that stops publishing when the count stops moving
// makes "this peer went quiet" and "dscovox went quiet" the same observation —
// the ambiguity the counters exist to remove.
//
// BLACK BOX, ON PURPOSE. The property is about which timer drives the publish
// and what gates it, so the thing under test is the built dscovox binary with
// its real timers, not a helper extracted from it: a unit test of the message
// builder would pass against every defect this test is here for. The node is
// run as a child process in a private namespace (/t9_<pid>), sent one frame,
// and then left alone.
//
// THREE DETAILS DECIDE WHETHER THIS TEST CAN FAIL. Each one, got wrong, makes
// it pass against the fused_dirty_-gated publish that caused the bug:
//
//   1. It subscribes to the FUSED MAP, as the planner does in every run.
//      fused_dirty_ is consumed only by publishFusedMap, and publishFusedMap
//      returns before consuming it when nobody is subscribed. With no
//      subscriber the flag stays set forever after the first frame, and a
//      dirty-gated counter publish keeps firing — live, but only because the
//      test failed to recreate the condition that silences it.
//   2. Freshness is judged by header.stamp from a VOLATILE subscription. The
//      counters are latched (transient_local); a latched subscription replays
//      the stored sample, and a test that only checks receipt passes against a
//      node that published once and fell silent.
//   3. The silent interval starts only after the fused map has been received,
//      and is required to contain no further fused map. That is what makes it
//      silent in the sense that matters — the dirty-gated channel has nothing
//      to say — rather than merely a pause the test chose.
//
// Values are checked as well as arrival: every fresh sample must carry the
// same deltas_received and cells_touched, because a sample during silence
// that moved would mean the interval was not silent at all.
//
// Mutations, each shown to fail this test (ledger continues the gen-33
// sequence in explo_planner: M1-M26, M32-M34 and M38-M40 in
// test_gen21_latched_hold.cpp, M27-M31, M35-M37 and M43 in
// test_exchange_drain.cpp):
//   M41  publishFusionCounters() returns unless fused_dirty_ is set — the
//        dirty-gated publish of test-plan 9's wording. Fails the fresh-sample
//        count: 0 of the required 5.
//   M42  the dedicated timer never created (fusion_counters_hz forced to 0).
//        Fails earlier, at "never reported the delivered frame": with no
//        timer there is no publish at all.
// and one guard shown to be load-bearing by removing it from the test:
//   G1   without the fused-map subscription (detail 1 above), M41 passes.
// Each result held on three consecutive runs.

#include <gtest/gtest.h>

#include <rclcpp/rclcpp.hpp>
#include <scovox_msgs/msg/scovox_fusion_counters.hpp>
#include <scovox_msgs/msg/scovox_map.hpp>
#include <scovox_msgs/msg/scovox_map_binary.hpp>
#include <scovox/binary_serializer.hpp>
#include <scovox/lz4_codec.hpp>

#include <signal.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

using Counters = scovox_msgs::msg::ScovoxFusionCounters;

constexpr char   kSource[]        = "atlas/odom";
constexpr double kCountersHz      = 10.0;
constexpr double kSilentSec       = 2.0;
constexpr int    kMinFreshSamples = 5;   // ~20 expected at 10 Hz over 2 s
constexpr size_t kBetaDeltas      = 3;   // the one frame this test sends

// The real dscovox binary as a child process. argv is built before fork()
// because the child may only call async-signal-safe functions before exec.
// PR_SET_PDEATHSIG so a test that crashes cannot leave the node running.
class ChildNode {
 public:
  explicit ChildNode(std::vector<std::string> args) : args_(std::move(args)) {
    std::vector<char*> argv;
    for (auto& a : args_) argv.push_back(a.data());
    argv.push_back(nullptr);
    const pid_t parent = getpid();
    pid_ = fork();
    if (pid_ == 0) {
      prctl(PR_SET_PDEATHSIG, SIGKILL);
      if (getppid() != parent) _exit(126);
      execv(argv[0], argv.data());
      _exit(127);
    }
  }
  ~ChildNode() { stop(); }
  ChildNode(const ChildNode&) = delete;
  ChildNode& operator=(const ChildNode&) = delete;

  bool alive() {
    if (pid_ <= 0) return false;
    int st = 0;
    if (waitpid(pid_, &st, WNOHANG) == pid_) { pid_ = -1; return false; }
    return true;
  }

  void stop() {
    if (pid_ <= 0) return;
    kill(pid_, SIGINT);
    for (int i = 0; i < 50; ++i) {
      if (waitpid(pid_, nullptr, WNOHANG) == pid_) { pid_ = -1; return; }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    kill(pid_, SIGKILL);
    waitpid(pid_, nullptr, 0);
    pid_ = -1;
  }

 private:
  std::vector<std::string> args_;
  pid_t pid_{-1};
};

template <class Pred>
bool spinUntil(rclcpp::Executor& ex, Pred done, double timeout_s) {
  const auto end = std::chrono::steady_clock::now() +
                   std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                       std::chrono::duration<double>(timeout_s));
  while (std::chrono::steady_clock::now() < end) {
    if (done()) return true;
    ex.spin_some(std::chrono::milliseconds(20));
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  return done();
}

// One frame of kBetaDeltas occupancy deltas on distinct cells, no semantics.
scovox_msgs::msg::ScovoxMapBinary makeFrame() {
  scovox::BinarySerializer::Frame f;
  f.resolution  = 0.1f;
  f.num_classes = 14;
  f.alpha_0     = scovox::kDefaultDirichletPrior;
  for (size_t i = 0; i < kBetaDeltas; ++i)
    f.beta_deltas.push_back(
        {Bonxai::CoordT{static_cast<int32_t>(i), 0, 5}, scovox::BetaVoxel{3.0f, 1.0f}});

  scovox_msgs::msg::ScovoxMapBinary m;
  m.header.frame_id = kSource;
  m.version = 5;
  m.little_endian = (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__);
  m.map_from_source.rotation.w = 1.0;
  m.data = scovox::ScovoxBinarySerializer::compressLZ4(
      scovox::BinarySerializer::serialize(f));
  return m;
}

struct Sample {
  rclcpp::Time stamp;
  bool     has_source{false};
  uint64_t deltas{0};
  uint64_t cells{0};
};

class FusionCountersLiveness : public ::testing::Test {
 protected:
  void SetUp() override {
    ns_ = "/t9_" + std::to_string(getpid());
    in_topic_ = ns_ + "/peer_bin";
    // Child first: fork() before rclcpp::init starts any threads here.
    dscovox_ = std::make_unique<ChildNode>(std::vector<std::string>{
        DSCOVOX_NODE_EXE, "--ros-args",
        "-r", "__ns:=" + ns_,
        "-p", "input_topics:=['" + in_topic_ + "']",
        "-p", "publish_rate_hz:=10.0",
        "-p", "fusion_counters_hz:=" + std::to_string(kCountersHz),
        "--log-level", "warn"});
    rclcpp::init(0, nullptr);
  }
  void TearDown() override {
    dscovox_.reset();
    rclcpp::shutdown();
  }

  std::string ns_, in_topic_;
  std::unique_ptr<ChildNode> dscovox_;
};

TEST_F(FusionCountersLiveness, SilenceStillProducesFreshSamplesWithTheCountUnchanged) {
  auto probe = std::make_shared<rclcpp::Node>("fusion_counters_probe", ns_);
  rclcpp::executors::SingleThreadedExecutor ex;
  ex.add_node(probe);

  std::vector<Sample> samples;
  int maps = 0;

  auto bin_pub = probe->create_publisher<scovox_msgs::msg::ScovoxMapBinary>(
      in_topic_, rclcpp::QoS(rclcpp::KeepLast(10)).reliable());
  // Detail 2: volatile, so nothing latched is replayed as if it were fresh.
  auto counters_sub = probe->create_subscription<Counters>(
      ns_ + "/dscovox_node/fusion_counters",
      rclcpp::QoS(rclcpp::KeepLast(50)).reliable(),
      [&](const Counters::SharedPtr m) {
        Sample s;
        s.stamp = rclcpp::Time(m->header.stamp);
        const auto it = std::find(m->source_frame.begin(), m->source_frame.end(), kSource);
        if (it != m->source_frame.end()) {
          const size_t i = static_cast<size_t>(it - m->source_frame.begin());
          s.has_source = true;
          s.deltas = m->deltas_received.at(i);
          s.cells  = m->cells_touched.at(i);
        }
        samples.push_back(s);
      });
  // Detail 1: the planner's subscription, with the planner's QoS. This is what
  // consumes fused_dirty_ and so what lets a dirty-gated publish fall silent.
  auto map_sub = probe->create_subscription<scovox_msgs::msg::ScovoxMap>(
      ns_ + "/dscovox_node/scovox",
      rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local(),
      [&](const scovox_msgs::msg::ScovoxMap::SharedPtr) { ++maps; });

  ASSERT_TRUE(spinUntil(ex, [&] {
    return bin_pub->get_subscription_count() > 0 &&
           counters_sub->get_publisher_count() > 0 &&
           map_sub->get_publisher_count() > 0;
  }, 20.0)) << "dscovox (" << DSCOVOX_NODE_EXE << ") never came up in " << ns_
            << "; alive=" << dscovox_->alive();

  bin_pub->publish(makeFrame());

  // Wait for the frame to be reported AND for the fused map it dirtied to go
  // out, so the interval that follows has nothing left to publish on the map.
  ASSERT_TRUE(spinUntil(ex, [&] {
    return maps >= 1 && !samples.empty() && samples.back().has_source &&
           samples.back().deltas == kBetaDeltas;
  }, 10.0)) << "dscovox never reported the delivered frame on its counters "
               "(samples=" << samples.size() << ", fused maps=" << maps << ")";

  const Sample base = samples.back();
  const size_t first = samples.size();
  const int maps_before = maps;
  // Counted as presented: every delta the frame carried.
  ASSERT_EQ(base.deltas, kBetaDeltas);
  ASSERT_GT(base.cells, 0u);

  spinUntil(ex, [] { return false; }, kSilentSec);

  // Detail 3 — the precondition, not the claim. If the fused map moved, the
  // interval was not silent and passing would prove nothing.
  ASSERT_EQ(maps, maps_before)
      << "the fused map was republished during the silent interval";

  int fresh = 0;
  rclcpp::Time last = base.stamp;
  for (size_t i = first; i < samples.size(); ++i) {
    const Sample& s = samples[i];
    if (s.stamp <= last) continue;   // replay or duplicate, not a new sample
    last = s.stamp;
    ++fresh;
    EXPECT_TRUE(s.has_source) << "a source that has gone quiet must keep its entry";
    EXPECT_EQ(s.deltas, base.deltas) << "deltas_received moved with nothing sent";
    EXPECT_EQ(s.cells,  base.cells)  << "cells_touched moved with nothing sent";
  }
  EXPECT_GE(fresh, kMinFreshSamples)
      << "only " << fresh << " fresh counter sample(s) in " << kSilentSec
      << " s of silence at " << kCountersHz << " Hz: the counters fell silent "
         "with the fused map, which makes a quiet peer and a quiet dscovox the "
         "same observation";
}

}  // namespace
