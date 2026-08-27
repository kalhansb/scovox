#pragma once

/// @file carve_stage.hpp
/// @brief Per-scan batched-carve accumulator keyed by Beta leaf block —
/// the sort-free replacement for `SemSplitMap`'s `unordered_map<CoordT,float>`
/// staging (efficiency_audit_2026_08_26.md, item 1).
///
/// The batched carve path stages the strongest free vote per voxel during the
/// scan walk and writes each voxel once at flush, leaf-block-ordered so the
/// Bonxai accessor's cached-block fast path is hit on consecutive writes. The
/// retired implementation recovered that order with a per-scan `std::sort` of
/// EVERY staged voxel (hundreds of thousands of 16-byte pairs on KITTI).
///
/// `CarveStage` stages by construction instead:
///   - voxels are keyed by leaf block (`coord >> leaf_bits`, the Beta grid's
///     own block key) into a small open-addressed index (Teschner-prime hash,
///     linear probe, load <= 1/2, grown by doubling);
///   - each block owns one dense SLOT in a reused pool: `2^(3·leaf_bits)`
///     floats (max `w_free·quality` per cell) plus an occupancy bitmask, so
///     staging a voxel is a bit-set + max — no node allocation, no rehash of
///     voxel keys;
///   - flush sorts only the BLOCK keys (thousands, not hundreds of thousands)
///     ascending — the identical `(x>>lb, y>>lb, z>>lb)` lexicographic order
///     the retired sort produced — and walks each block's bitmask.
///
/// Behaviour identity with the retired staging (the flush invariants):
///   - per-voxel value: max-accumulate is order-independent — identical;
///   - block visit order: ascending block key == the retired comparator's
///     total order over blocks (keys are unique per block, so the retired
///     non-stable sort was deterministic AT BLOCK granularity) — identical,
///     which preserves the Beta root map's first-touch insertion order and
///     therefore the serialized wire bytes;
///   - WITHIN-block voxel order changes (x-major ascending here, unspecified
///     before) — structurally invisible: leaf blocks are dense arrays (cell
///     position is coord-derived, not insertion-derived), and the flush
///     `touched_beta_` order is normalized by `drainTouchedBeta`'s sortUnique
///     before anything external sees it;
///   - staged-voxel SET (and hence the flush count): identical.
///
/// All capacity — index table, slot pool, key list, sort scratch — is retained
/// across `beginFrame()`, so steady-state per-scan framing allocates nothing.
/// The trade: the pool holds its HIGH-WATER footprint for the map's lifetime
/// (bounded by the peak per-scan distinct-block count), and a far-field block
/// grazed by a single ray pays a full dense slot where the retired map paid
/// ~50 B per staged voxel — an RSS floor of order tens of MB on outdoor
/// LiDAR, amortized in the dense near field where the staging cost lives.
///
/// Not thread-safe (same contract as the SemSplitMap members it replaces).

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <bonxai/grid_coord.hpp>

namespace scovox {

class CarveStage {
 public:
  using CoordT = Bonxai::CoordT;

  /// @param leaf_bits block edge = 2^leaf_bits voxels — MUST equal the Beta
  /// grid's `Params::leaf_bits` (post-sanitise) so the block order reproduced
  /// at flush is the accessor's own block geometry. Precondition: 1..10
  /// (a slot's dense arrays are ≈4⅛·8^leaf_bits bytes — 4 B of weight per
  /// cell + 1 bit of mask, so ~2.1 KB at leaf_bits=3; the shipped grids
  /// use <= 3).
  explicit CarveStage(int leaf_bits)
      : lb_(leaf_bits),
        cells_(1u << (3 * leaf_bits)),
        // Round UP: at leaf_bits=1 a block has 8 cells — one partial word,
        // not zero words.
        words_((cells_ + 63u) / 64u),
        low_mask_((int32_t(1) << leaf_bits) - 1),
        index_(kInitialIndexCap) {
    assert(leaf_bits >= 1 && leaf_bits <= 10);
  }

  /// Stage the free vote `w` for voxel `c`, keeping the per-voxel MAX.
  void add(const CoordT& c, float w) {
    // Arithmetic >> on int32 is floor division by 2^lb (two's complement) —
    // the same block key the Beta grid derives for `c`.
    const int32_t kx = c.x >> lb_;
    const int32_t ky = c.y >> lb_;
    const int32_t kz = c.z >> lb_;
    uint32_t slot;
    if (have_last_ && kx == last_kx_ && ky == last_ky_ && kz == last_kz_) {
      slot = last_slot_;  // ray-coherent fast path: same leaf as previous add
    } else {
      slot = findOrCreateSlot(kx, ky, kz);
      last_kx_ = kx;
      last_ky_ = ky;
      last_kz_ = kz;
      last_slot_ = slot;
      have_last_ = true;
    }
    // In-block cell index, x-major. `c & low_mask_` is the floor remainder
    // (non-negative) for any sign of `c`, pairing with the >> above.
    const uint32_t idx = (uint32_t(c.x & low_mask_) << (2 * lb_)) |
                         (uint32_t(c.y & low_mask_) << lb_) |
                         uint32_t(c.z & low_mask_);
    uint64_t&      word = mask_[std::size_t(slot) * words_ + (idx >> 6)];
    const uint64_t bit  = uint64_t(1) << (idx & 63u);
    float&         cell = w_[std::size_t(slot) * cells_ + idx];
    if (word & bit) {
      if (w > cell) cell = w;
    } else {
      word |= bit;
      cell = w;
      ++staged_;
    }
  }

  /// Drop all staged state; every buffer keeps its capacity. A reused slot's
  /// mask words are re-zeroed when the slot is claimed (newSlot), so nothing
  /// per-cell is touched here.
  void beginFrame() {
    std::fill(index_.begin(), index_.end(), IndexEntry{});
    slot_keys_.clear();
    n_slots_ = 0;
    staged_ = 0;
    have_last_ = false;
  }

  /// Visit every staged (voxel, weight) pair, blocks in ascending
  /// `(x>>lb, y>>lb, z>>lb)` lexicographic order (== the retired per-voxel
  /// sort's block order), voxels within a block x-major ascending.
  /// `fn(const CoordT&, float)`.
  template <typename Fn>
  void forEachStagedBlockOrdered(Fn&& fn) {
    order_.resize(n_slots_);
    for (uint32_t s = 0; s < n_slots_; ++s) order_[s] = s;
    std::sort(order_.begin(), order_.end(), [this](uint32_t a, uint32_t b) {
      const Key& ka = slot_keys_[a];
      const Key& kb = slot_keys_[b];
      if (ka.kx != kb.kx) return ka.kx < kb.kx;
      if (ka.ky != kb.ky) return ka.ky < kb.ky;
      return ka.kz < kb.kz;
    });
    for (const uint32_t s : order_) {
      const Key& k = slot_keys_[s];
      // key * 2^lb + offset reconstructs the coord exactly for any sign
      // (multiply, NOT <<: left-shifting a negative key is UB in C++17).
      const int32_t     bx    = k.kx * (int32_t(1) << lb_);
      const int32_t     by    = k.ky * (int32_t(1) << lb_);
      const int32_t     bz    = k.kz * (int32_t(1) << lb_);
      const std::size_t wbase = std::size_t(s) * words_;
      const std::size_t cbase = std::size_t(s) * cells_;
      for (uint32_t wi = 0; wi < words_; ++wi) {
        uint64_t bits = mask_[wbase + wi];
        while (bits) {
          const uint32_t b = uint32_t(__builtin_ctzll(bits));
          bits &= bits - 1;
          const uint32_t idx = (wi << 6) | b;
          const CoordT   c{bx + int32_t(idx >> (2 * lb_)),
                           by + int32_t((idx >> lb_) & uint32_t(low_mask_)),
                           bz + int32_t(idx & uint32_t(low_mask_))};
          fn(c, w_[cbase + idx]);
        }
      }
    }
  }

  /// Number of distinct voxels staged this frame.
  [[nodiscard]] std::size_t size() const noexcept { return staged_; }
  [[nodiscard]] bool        empty() const noexcept { return staged_ == 0; }
  /// Number of distinct leaf blocks staged this frame.
  [[nodiscard]] std::size_t blockCount() const noexcept { return n_slots_; }

 private:
  struct Key {
    int32_t kx, ky, kz;
  };
  /// `slot1` = slot id + 1; 0 marks an empty index entry.
  struct IndexEntry {
    int32_t  kx = 0, ky = 0, kz = 0;
    uint32_t slot1 = 0;
  };

  static constexpr std::size_t kInitialIndexCap = 1024;  // power of two

  [[nodiscard]] static uint64_t hashKey(int32_t kx, int32_t ky, int32_t kz) noexcept {
    // Teschner et al. spatial-hash primes; cast through uint32 so negative
    // keys wrap instead of sign-extending into the multiply.
    return (uint64_t(uint32_t(kx)) * 73856093u) ^
           (uint64_t(uint32_t(ky)) * 19349663u) ^
           (uint64_t(uint32_t(kz)) * 83492791u);
  }

  uint32_t findOrCreateSlot(int32_t kx, int32_t ky, int32_t kz) {
    // Grow BEFORE probing so the load factor never exceeds 1/2 — probes stay
    // short and the table always holds an empty entry to terminate on.
    if (2 * (std::size_t(n_slots_) + 1) > index_.size()) growIndex();
    const std::size_t capmask = index_.size() - 1;
    std::size_t       i = std::size_t(hashKey(kx, ky, kz)) & capmask;
    for (;; i = (i + 1) & capmask) {
      IndexEntry& e = index_[i];
      if (e.slot1 == 0) {
        const uint32_t slot = newSlot(kx, ky, kz);
        e = IndexEntry{kx, ky, kz, slot + 1};
        return slot;
      }
      if (e.kx == kx && e.ky == ky && e.kz == kz) return e.slot1 - 1;
    }
  }

  uint32_t newSlot(int32_t kx, int32_t ky, int32_t kz) {
    const uint32_t slot = n_slots_++;
    slot_keys_.push_back(Key{kx, ky, kz});
    const std::size_t wbase = std::size_t(slot) * words_;
    const std::size_t cbase = std::size_t(slot) * cells_;
    if (mask_.size() < wbase + words_) {
      mask_.resize(wbase + words_);  // value-init: new words are zero
    } else {
      // Reused pool storage: zero this slot's mask words (its w_ cells stay
      // dirty — every read is gated on the mask, so stale weights are dead).
      std::fill(mask_.begin() + wbase, mask_.begin() + wbase + words_, uint64_t(0));
    }
    if (w_.size() < cbase + cells_) w_.resize(cbase + cells_);
    return slot;
  }

  void growIndex() {
    std::vector<IndexEntry> old;
    old.swap(index_);
    index_.assign(old.size() * 2, IndexEntry{});
    const std::size_t capmask = index_.size() - 1;
    for (const IndexEntry& e : old) {
      if (e.slot1 == 0) continue;
      std::size_t i = std::size_t(hashKey(e.kx, e.ky, e.kz)) & capmask;
      while (index_[i].slot1 != 0) i = (i + 1) & capmask;
      index_[i] = e;
    }
  }

  int      lb_;
  uint32_t cells_;     ///< voxels per block = 8^lb
  uint32_t words_;     ///< 64-bit mask words per block = ceil(cells_/64)
  int32_t  low_mask_;  ///< in-block coord mask = 2^lb - 1

  std::vector<IndexEntry> index_;      ///< open-addressed block-key -> slot+1
  std::vector<Key>        slot_keys_;  ///< block key per live slot
  std::vector<float>      w_;          ///< slot pool: cells_ weights per slot
  std::vector<uint64_t>   mask_;       ///< slot pool: words_ mask words per slot
  std::vector<uint32_t>   order_;      ///< flush sort scratch (slot ids)

  uint32_t    n_slots_ = 0;
  std::size_t staged_ = 0;

  // Last-leaf cache: consecutive adds along a ray usually land in one block.
  // Survives index growth (slot ids are stable); invalidated per frame.
  bool     have_last_ = false;
  int32_t  last_kx_ = 0, last_ky_ = 0, last_kz_ = 0;
  uint32_t last_slot_ = 0;
};

}  // namespace scovox
