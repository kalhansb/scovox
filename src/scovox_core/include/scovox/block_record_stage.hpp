#pragma once

/// @file block_record_stage.hpp
/// @brief Per-scan record accumulator keyed by leaf block — the sort-free
/// replacement for `SemSplitMap`'s `unordered_map<CoordT, …>` staging, and the
/// record-carrying counterpart to `CarveStage`.
///
/// Two of `SemSplitMap`'s three batched paths stage a RECORD per voxel rather
/// than `CarveStage`'s single float: the semantic band
/// (`SemSplitParams::batch_band`) keeps the scan's most confident look, and the
/// surface hits (`SemSplitParams::batch_hits`) keep its strongest ray. Both
/// retired implementations hashed every individual voxel coordinate on every
/// visit and then recovered the flush order with a per-scan `std::sort` of
/// EVERY staged voxel — the same pattern `CarveStage` was introduced to remove
/// from the carve, left behind on the two paths that stage a record rather
/// than a float. The record is the only thing that differed between them, so
/// the machinery is parameterised on it and the pattern is retired once.
///
/// `BlockRecordStage` stages by construction, exactly as `CarveStage` does:
///   - voxels are keyed by leaf block (`coord >> leaf_bits`) into a small
///     open-addressed index (Teschner-prime hash, linear probe, load <= 1/2,
///     grown by doubling), with a last-leaf cache for ray-coherent adds;
///   - each block owns one dense SLOT in a reused pool: `2^(3·leaf_bits)`
///     u32 record indices plus an occupancy bitmask, so staging a voxel is a
///     bit test + an array index — no node allocation, no per-voxel rehash;
///   - the records themselves live in one flat, frame-local pool, so a cell
///     costs 4 B here rather than the record's full width. That keeps a slot
///     the same size as `CarveStage`'s (≈2.1 KB at leaf_bits=3) whatever `RecT`
///     weighs, which matters because a slot is paid in full even for a block a
///     single ray grazed;
///   - flush sorts only the BLOCK keys (thousands, not hundreds of thousands)
///     and walks each block's bitmask.
///
/// ORDER IDENTITY with the retired staging — stronger than `CarveStage`'s,
/// which had to give up within-block order. Both retired sorts were
/// `(x>>lb, y>>lb, z>>lb)` ascending and then `(x, y, z)` ascending within a
/// block; walking a slot's bitmask by ascending cell index yields x-major, then
/// y, then z over the in-block low bits, and within one block the high bits are
/// equal, so ascending low-bit order IS ascending global order. The flush
/// therefore visits the identical voxel sequence, not merely the identical
/// block sequence: the staged set, the per-voxel record and the visit order all
/// match, so the grids are first-touched in the same order and the serialized
/// bytes are unchanged.
///
/// Per-voxel record semantics are unchanged from the retired maps: `slotFor`
/// default-constructs on first touch and returns the existing record after
/// that, applying no comparison of its own, so each caller keeps the
/// supersede-vs-tie rule it already had.
///
/// All capacity — index table, slot pool, record pool, sort scratch — is
/// retained across `beginFrame()`, so steady-state per-scan framing allocates
/// nothing. The trade is `CarveStage`'s: the pool holds its high-water
/// footprint for the map's lifetime, bounded by the peak per-scan distinct
/// block count for that path, which is a subset of the carve's.
///
/// Not thread-safe (same contract as the SemSplitMap members it replaces).

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <bonxai/grid_coord.hpp>

namespace scovox {

template <typename RecT>
class BlockRecordStage {
 public:
  using CoordT = Bonxai::CoordT;
  using Rec    = RecT;

  /// Sentinel for a record's `probs_off` — no block claimed in the caller's
  /// observation pool.
  static constexpr uint32_t kNoProbs = 0xFFFFFFFFu;

  /// @param leaf_bits block edge = 2^leaf_bits voxels — MUST equal the
  /// `leaf_bits` the retired per-voxel sort used, so the block order
  /// reproduced at flush is the one the grids were first-touched in.
  /// Precondition: 1..10 (a slot's dense arrays are ≈4⅛·8^leaf_bits bytes —
  /// 4 B of record index per cell + 1 bit of mask, so ~2.1 KB at leaf_bits=3).
  explicit BlockRecordStage(int leaf_bits)
      : lb_(leaf_bits),
        cells_(1u << (3 * leaf_bits)),
        // Round UP: at leaf_bits=1 a block has 8 cells — one partial word.
        words_((cells_ + 63u) / 64u),
        low_mask_((int32_t(1) << leaf_bits) - 1),
        index_(kInitialIndexCap) {
    assert(leaf_bits >= 1 && leaf_bits <= 10);
  }

  /// The record for voxel `c`, created default-constructed on first touch this
  /// frame. `*created` (optional) reports whether this call claimed the voxel.
  ///
  /// The reference is valid until the NEXT `slotFor` call — the record pool can
  /// reallocate — which is the same contract the retired `operator[]` on an
  /// `unordered_map` gave, and matches the callers' use (claim, write, return).
  Rec& slotFor(const CoordT& c, bool* created = nullptr) {
    // Arithmetic >> on int32 is floor division by 2^lb (two's complement) —
    // the same block key the retired sort's comparator derived for `c`.
    const int32_t kx = c.x >> lb_;
    const int32_t ky = c.y >> lb_;
    const int32_t kz = c.z >> lb_;
    uint32_t slot;
    if (have_last_ && kx == last_kx_ && ky == last_ky_ && kz == last_kz_) {
      slot = last_slot_;  // ray-coherent fast path: same leaf as previous call
    } else {
      slot = findOrCreateSlot(kx, ky, kz);
      last_kx_ = kx;
      last_ky_ = ky;
      last_kz_ = kz;
      last_slot_ = slot;
      have_last_ = true;
    }
    const uint32_t idx  = cellIndex(c);
    uint64_t&      word = mask_[std::size_t(slot) * words_ + (idx >> 6)];
    const uint64_t bit  = uint64_t(1) << (idx & 63u);
    uint32_t&      ri   = rec_idx_[std::size_t(slot) * cells_ + idx];
    if (word & bit) {
      if (created) *created = false;
      return recs_[ri];
    }
    word |= bit;
    ri = static_cast<uint32_t>(recs_.size());
    recs_.emplace_back();
    ++staged_;
    if (created) *created = true;
    return recs_[ri];
  }

  /// Is `c` staged this frame? Read-only — unlike `slotFor` it never claims a
  /// block, so probing a voxel the scan never touched costs no slot.
  ///
  /// This is the membership test the hit path's occupied-wins rule needs, and
  /// it is asked once per staged CARVE voxel — the hot query in the map. It
  /// keeps its own last-block cache rather than sharing `slotFor`'s, because
  /// what it must remember is usually an ABSENT block (most carved blocks hold
  /// no hit at all) and a slot-id cache cannot represent absence. The carve
  /// flush walks block-ordered, so consecutive queries land in the same block
  /// and the cache answers nearly all of them without a probe.
  [[nodiscard]] bool contains(const CoordT& c) noexcept {
    const int32_t kx = c.x >> lb_;
    const int32_t ky = c.y >> lb_;
    const int32_t kz = c.z >> lb_;
    if (!have_q_ || kx != q_kx_ || ky != q_ky_ || kz != q_kz_) {
      q_slot_ = findSlot(kx, ky, kz);
      q_kx_ = kx;
      q_ky_ = ky;
      q_kz_ = kz;
      have_q_ = true;
    }
    if (q_slot_ == kNoSlot) return false;
    const uint32_t idx = cellIndex(c);
    return (mask_[std::size_t(q_slot_) * words_ + (idx >> 6)] >>
            (idx & 63u)) & 1u;
  }

  /// Drop all staged state; every buffer keeps its capacity. A reused slot's
  /// mask words are re-zeroed when the slot is claimed (newSlot), so nothing
  /// per-cell is touched here.
  void beginFrame() {
    std::fill(index_.begin(), index_.end(), IndexEntry{});
    slot_keys_.clear();
    recs_.clear();
    n_slots_ = 0;
    staged_ = 0;
    have_last_ = false;
    have_q_ = false;
  }

  /// Visit every staged (voxel, record) pair in the retired sort's exact
  /// order: blocks ascending `(x>>lb, y>>lb, z>>lb)`, voxels within a block
  /// ascending `(x, y, z)`. `fn(const CoordT&, const Rec&)`.
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
          fn(c, recs_[rec_idx_[cbase + idx]]);
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
  static constexpr uint32_t    kNoSlot = 0xFFFFFFFFu;

  /// In-block cell index, x-major. `c & low_mask_` is the floor remainder
  /// (non-negative) for any sign of `c`, pairing with the `>> lb_` above.
  [[nodiscard]] uint32_t cellIndex(const CoordT& c) const noexcept {
    return (uint32_t(c.x & low_mask_) << (2 * lb_)) |
           (uint32_t(c.y & low_mask_) << lb_) |
           uint32_t(c.z & low_mask_);
  }

  [[nodiscard]] static uint64_t hashKey(int32_t kx, int32_t ky, int32_t kz) noexcept {
    // Teschner et al. spatial-hash primes; cast through uint32 so negative
    // keys wrap instead of sign-extending into the multiply.
    return (uint64_t(uint32_t(kx)) * 73856093u) ^
           (uint64_t(uint32_t(ky)) * 19349663u) ^
           (uint64_t(uint32_t(kz)) * 83492791u);
  }

  /// Probe only — `kNoSlot` if this block was never staged. The table always
  /// holds an empty entry (load <= 1/2), so the loop terminates.
  [[nodiscard]] uint32_t findSlot(int32_t kx, int32_t ky, int32_t kz) const noexcept {
    const std::size_t capmask = index_.size() - 1;
    std::size_t       i = std::size_t(hashKey(kx, ky, kz)) & capmask;
    for (;; i = (i + 1) & capmask) {
      const IndexEntry& e = index_[i];
      if (e.slot1 == 0) return kNoSlot;
      if (e.kx == kx && e.ky == ky && e.kz == kz) return e.slot1 - 1;
    }
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
      // Reused pool storage: zero this slot's mask words (its rec_idx_ cells
      // stay dirty — every read is gated on the mask, so stale indices are
      // dead).
      std::fill(mask_.begin() + wbase, mask_.begin() + wbase + words_, uint64_t(0));
    }
    if (rec_idx_.size() < cbase + cells_) rec_idx_.resize(cbase + cells_);
    // A cached ABSENT answer for this key is now wrong. Staging and querying
    // are disjoint phases in today's callers, so this never actually fires —
    // it is here so that interleaving them stays correct rather than subtly
    // stale.
    have_q_ = false;
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
  std::vector<uint32_t>   rec_idx_;    ///< slot pool: cells_ record indices per slot
  std::vector<uint64_t>   mask_;       ///< slot pool: words_ mask words per slot
  std::vector<Rec>        recs_;       ///< frame-local record pool
  std::vector<uint32_t>   order_;      ///< flush sort scratch (slot ids)

  uint32_t    n_slots_ = 0;
  std::size_t staged_ = 0;

  // Last-leaf cache: consecutive calls along a ray usually land in one block.
  // Survives index growth (slot ids are stable); invalidated per frame.
  bool     have_last_ = false;
  int32_t  last_kx_ = 0, last_ky_ = 0, last_kz_ = 0;
  uint32_t last_slot_ = 0;

  // `contains`'s own cache — holds `kNoSlot` for an absent block, which the
  // slot-id cache above cannot.
  bool     have_q_ = false;
  int32_t  q_kx_ = 0, q_ky_ = 0, q_kz_ = 0;
  uint32_t q_slot_ = kNoSlot;
};

/// One scan's staged band look for a voxel (see `SemSplitParams::batch_band`).
/// Field-for-field the retired `SemSplitMap::BandStage`, so the flush
/// arithmetic is untouched.
struct BandRec {
  float    kappa0       = 0.f;
  float    min_p_occ    = 0.f;
  float    q            = -1.f;      ///< argmax probability of the holding look
  uint32_t probs_off    = 0xFFFFFFFFu;  ///< start of its entries in the caller's pool
  uint32_t probs_len    = 0;
  uint32_t probs_cap    = 0;
  int      probs_argmax = -1;
};

using BandStage = BlockRecordStage<BandRec>;

}  // namespace scovox
