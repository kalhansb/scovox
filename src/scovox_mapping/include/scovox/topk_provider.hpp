#pragma once

// TopkProvider — file-backed loader/cache for per-frame soft-probability
// (.topk) blobs produced by topk_npz_to_bin.py.
//
// Extracted verbatim from SCovoxNode (Tier 2 refactor): the node used to carry
// the per-frame cache, the binary parser, the fill helpers, and the loader
// telemetry inline. Pulling them into a standalone unit makes the binary parser
// unit-testable without spinning up ROS (the only ROS dependency is the
// injected logger/clock used for the same throttled diagnostics as before).
//
// Pointcloud layout: [u32 N][u8 C][N*C u8 probs(×255)] — slot j == SCovox class id.
// Image      layout: [u16 H][u16 W][u8 C][H*W*C u8 probs(×255)] — slot j == SCovox class id.
// Slot 0 is unknown/unlabeled by convention.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <ios>
#include <string>
#include <utility>
#include <vector>

#include <rclcpp/clock.hpp>
#include <rclcpp/logger.hpp>
#include <rclcpp/logging.hpp>

namespace scovox {

class TopkProvider {
public:
  // logger/clock drive the same throttled diagnostics the inline code emitted.
  // probs_dir empty == soft-prob mode disabled (caller stays on the one-hot path).
  // max_sem caps the class slots consumed from each blob (slot index IS class id).
  TopkProvider(rclcpp::Logger logger, rclcpp::Clock::SharedPtr clock,
               std::string probs_dir, int max_sem)
      : logger_(std::move(logger)),
        clock_(std::move(clock)),
        probs_dir_(std::move(probs_dir)),
        max_sem_(max_sem) {}

  // True when a probs directory is configured (soft-prob mode active).
  bool enabled() const { return !probs_dir_.empty(); }

  // Loader telemetry (running totals; exposed for tests + shutdown summaries).
  uint64_t loadSuccess() const { return load_success_; }
  uint64_t loadFailure() const { return load_failure_; }

  static inline float dequantize(uint8_t q) { return float(q) * (1.0f / 255.0f); }

  // Load the .topk flat-binary blob for `frame_id` from the probs directory.
  // Returns true on success and populates the cache; false on missing/corrupt
  // file (caller falls back to the legacy hard-label one-hot path).
  bool loadFrame(uint16_t frame_id, bool image_mode) {
    if (cache_valid_ && cache_frame_ == frame_id &&
        cache_is_image_ == image_mode) {
      // Cache hit — the per-frame counter already incremented on the
      // miss-and-load. Don't double-count.
      return true;
    }
    char path[1024];
    std::snprintf(path, sizeof(path), "%s/%06u.topk",
                  probs_dir_.c_str(), (unsigned)frame_id);
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
      RCLCPP_WARN_THROTTLE(logger_, *clock_, 5000,
                           "topk: cannot open %s — falling back to one-hot", path);
      cache_valid_ = false;
      ++load_failure_;
      return false;
    }
    size_t total;
    uint8_t C = 0;
    if (image_mode) {
      uint16_t H = 0, W = 0;
      f.read(reinterpret_cast<char*>(&H), 2);
      f.read(reinterpret_cast<char*>(&W), 2);
      f.read(reinterpret_cast<char*>(&C), 1);
      total = (size_t)H * W * C;
      cache_h_ = H; cache_w_ = W; cache_c_ = C;
      cache_n_ = 0;
    } else {
      uint32_t N = 0;
      f.read(reinterpret_cast<char*>(&N), 4);
      f.read(reinterpret_cast<char*>(&C), 1);
      total = (size_t)N * C;
      cache_n_ = N; cache_c_ = C;
      cache_h_ = 0; cache_w_ = 0;
    }
    // Guard the header reads before sizing the buffer. An empty or truncated
    // .topk file leaves the dimension fields at their zero-init (failed reads
    // no-op), and a garbage `total` from a partially-read header would throw
    // bad_alloc out of resize() — uncaught in the integration callback, that
    // calls std::terminate and kills the mapper. Require a clean header read
    // and a payload within a sane absolute ceiling.
    static constexpr size_t kMaxTopkBytes = size_t(1) << 30;  // 1 GiB
    if (!f.good() || total == 0 || total > kMaxTopkBytes) {
      RCLCPP_WARN_THROTTLE(logger_, *clock_, 5000,
                           "topk: bad/empty header in %s (total=%zu) — falling back to one-hot",
                           path, total);
      cache_valid_ = false;
      ++load_failure_;
      return false;
    }
    cache_probs_.resize(total);
    f.read(reinterpret_cast<char*>(cache_probs_.data()), total);
    // Detect a truncated payload by the bytes actually delivered, NOT by the
    // stream flags: a short read sets BOTH failbit and eofbit, so the old
    // `!f.good() && !f.eof()` test was (true && false) == false and silently
    // accepted the partial buffer — the zero-filled tail (from resize) would
    // then be fed to the Dirichlet update as legitimate zero-probability
    // classes, corrupting semantics for the whole frame. gcount() is the only
    // reliable short-read signal here.
    if (static_cast<size_t>(f.gcount()) != total) {
      RCLCPP_WARN(logger_, "topk: short read for %s (got %zu of %zu bytes)",
                  path, static_cast<size_t>(f.gcount()), total);
      cache_valid_ = false;
      ++load_failure_;
      return false;
    }
    cache_frame_ = frame_id;
    cache_is_image_ = image_mode;
    cache_valid_ = true;
    ++load_success_;
    if (!first_load_logged_) {
      // One-shot INFO so an operator can confirm soft-prob mode is live
      // without grepping for the per-frame trace. After this, only the
      // throttled summary reports counters.
      RCLCPP_INFO(logger_,
                  "topk: first frame loaded from %s (image_mode=%d, C=%u%s)",
                  probs_dir_.c_str(), (int)image_mode, (unsigned)cache_c_,
                  image_mode ? "" : ", point-mode");
      first_load_logged_ = true;
    }
    return true;
  }

  // Emit a running tally of soft-prob loader outcomes. Guards the
  // silent-fallback footgun: when the probs dir is set but loads intermittently
  // fail, the per-frame WARN is throttled at 5 s and easy to miss in long batch
  // runs; this INFO is throttled at the caller's rate and shows running totals
  // so the operator (or a smoke-gate assert) can verify soft-prob dispatched on
  // every expected frame. No-op when topk is disabled.
  void logSummary(int throttle_ms) {
    if (probs_dir_.empty()) return;
    const uint64_t total = load_success_ + load_failure_;
    if (total == 0) return;
    const double fail_pct = 100.0 * static_cast<double>(load_failure_)
                          / static_cast<double>(total);
    RCLCPP_INFO_THROTTLE(logger_, *clock_, throttle_ms,
        "topk loader: loaded=%lu fallback_to_one_hot=%lu (%.1f%% miss) dir=%s",
        (unsigned long)load_success_,
        (unsigned long)load_failure_,
        fail_pct,
        probs_dir_.c_str());
  }

  // Fill `cp` (size max_sem) with the top-K probabilities for the given
  // pointcloud row index. Returns true if any class with id in [1, max_sem)
  // got non-zero probability — false means the row is degenerate (all zero
  // or all out-of-range), caller should treat as no semantic observation.
  bool fillPoint(size_t row, std::vector<float>& cp) const {
    if (!cache_valid_ || cache_is_image_) return false;
    if (row >= cache_n_) return false;
    std::fill(cp.begin(), cp.end(), 0.f);
    const size_t base = row * cache_c_;
    const int C = std::min<int>(cache_c_, max_sem_);
    bool any = false;
    // Slot index *is* the SCovox class id. Skip slot 0 (=unknown) so it
    // contributes to a_unk (escape mass) rather than getting mapped to a
    // real class.
    for (int j = 1; j < C; ++j) {
      const float p = dequantize(cache_probs_[base + j]);
      if (p > 0.f) {
        cp[j] = p;
        any = true;
      }
    }
    return any;
  }

  // Replica image variant: fill `cp` from per-pixel dense distribution at (u, v).
  bool fillImage(int u, int v, std::vector<float>& cp) const {
    if (!cache_valid_ || !cache_is_image_) return false;
    if (u < 0 || v < 0 || u >= (int)cache_w_ || v >= (int)cache_h_) return false;
    std::fill(cp.begin(), cp.end(), 0.f);
    const size_t base = ((size_t)v * cache_w_ + u) * cache_c_;
    const int C = std::min<int>(cache_c_, max_sem_);
    bool any = false;
    for (int j = 1; j < C; ++j) {
      const float p = dequantize(cache_probs_[base + j]);
      if (p > 0.f) {
        cp[j] = p;
        any = true;
      }
    }
    return any;
  }

private:
  rclcpp::Logger logger_;
  rclcpp::Clock::SharedPtr clock_;
  std::string probs_dir_;
  int max_sem_;

  // Cache the most recently loaded frame so we don't read the file twice
  // (once for the hit lookup, once for the integration loop).
  uint16_t cache_frame_{0xFFFF};
  bool     cache_valid_{false};
  bool     cache_is_image_{false};
  uint32_t cache_n_{0};
  uint16_t cache_h_{0}, cache_w_{0};
  // Number of dense class slots per pixel/point. Slot j IS the SCovox
  // class id (slot 0 = unknown). Quantized to uint8 (×255) for storage.
  uint8_t  cache_c_{0};
  std::vector<uint8_t> cache_probs_;
  // Soft-prob loader telemetry — guards the silent-fallback footgun the
  // 5-s-throttled "topk: cannot open" warn alone cannot catch.
  uint64_t load_success_{0};
  uint64_t load_failure_{0};
  bool     first_load_logged_{false};
};

}  // namespace scovox
