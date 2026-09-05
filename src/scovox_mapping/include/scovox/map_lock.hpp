#pragma once
/// @file map_lock.hpp
/// @brief The `map_mtx_` contract as a parameter the caller cannot forge.
///
/// The node's map is guarded by one `std::shared_mutex`, and several publish
/// helpers deliberately do NOT take it. The viz timer holds a single lock
/// across four of them so they all describe one map state, and
/// `std::shared_mutex` is non-recursive, so a helper re-locking would be
/// undefined behaviour rather than a stall. That contract used to live only in
/// a comment above each helper, in a class large enough that a caller can be
/// added without ever meeting the comment; nothing diagnoses the omission, and
/// the race it opens is silent, timing-dependent, and will not reproduce under
/// a debugger.
///
/// So the contract is a parameter instead. `MapLockHeld` has no public
/// constructor and the only things that can produce one are the two guards
/// below, each of which owns an engaged lock for its whole lifetime. A helper
/// taking `const MapLockHeld&` therefore cannot be called at all from a context
/// that has not locked — not by oversight, and not deliberately either.
/// `MapWriteHeld` derives from `MapLockHeld` because exclusive ownership does
/// satisfy a shared requirement; the reverse is not true and does not compile.
///
/// Both witnesses are empty and travel by reference, so the parameter costs
/// nothing: the guards are the same `std::shared_lock` / `std::unique_lock`
/// under names that also hand out proof.
///
/// A parameter, rather than clang's `-Wthread-safety` attributes, because the
/// attributes are read by exactly one compiler and the shipped compile line is
/// gcc — which would carry them as decoration and check nothing. A parameter is
/// enforced by whatever compiler builds the tree.
#include <mutex>
#include <shared_mutex>

namespace scovox {

class MapReadLock;
class MapWriteLock;
class MapWriteHeld;

/// Proof that `map_mtx_` is held, at least shared. Not copyable: a witness
/// that outlived its guard would prove nothing.
class MapLockHeld {
 public:
  MapLockHeld(const MapLockHeld&) = delete;
  MapLockHeld& operator=(const MapLockHeld&) = delete;
 private:
  MapLockHeld() = default;
  friend class MapReadLock;
  friend class MapWriteHeld;
};

/// Proof that `map_mtx_` is held exclusively. Binds to `const MapLockHeld&`.
class MapWriteHeld : public MapLockHeld {
 private:
  MapWriteHeld() = default;
  friend class MapWriteLock;
};

/// Shared ownership of `map_mtx_`, plus the witness that proves it.
class MapReadLock {
 public:
  explicit MapReadLock(std::shared_mutex& m) : lk_(m) {}
  const MapLockHeld& held() const { return held_; }
 private:
  std::shared_lock<std::shared_mutex> lk_;
  MapLockHeld held_;
};

/// Exclusive ownership of `map_mtx_`, plus the witness that proves it.
class MapWriteLock {
 public:
  explicit MapWriteLock(std::shared_mutex& m) : lk_(m) {}
  const MapWriteHeld& held() const { return held_; }
 private:
  std::unique_lock<std::shared_mutex> lk_;
  MapWriteHeld held_;
};

}  // namespace scovox
