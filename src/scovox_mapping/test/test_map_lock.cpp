// The map_mtx_ witness types. These tests assert the properties the node's
// lock contract actually leans on — that a witness cannot be manufactured
// without taking the lock, that exclusive satisfies shared, and that shared
// does not satisfy exclusive. Every assertion is a compile-time trait, so a
// regression here is a build failure in the node too, not a runtime surprise.
#include <shared_mutex>
#include <type_traits>

#include <gtest/gtest.h>

#include "scovox/map_lock.hpp"

namespace {

// The load-bearing property: no caller can conjure the proof. If either of
// these ever became true, every `const MapLockHeld&` parameter in the node
// would degrade to documentation again.
TEST(MapLock, WitnessIsNotConstructibleByCallers) {
  EXPECT_FALSE(std::is_default_constructible_v<scovox::MapLockHeld>);
  EXPECT_FALSE(std::is_default_constructible_v<scovox::MapWriteHeld>);
}

// A witness that outlived its guard would prove nothing, so it must not be
// possible to keep one past the lock's scope by copying it out.
TEST(MapLock, WitnessIsNotCopyable) {
  EXPECT_FALSE(std::is_copy_constructible_v<scovox::MapLockHeld>);
  EXPECT_FALSE(std::is_copy_assignable_v<scovox::MapLockHeld>);
  EXPECT_FALSE(std::is_copy_constructible_v<scovox::MapWriteHeld>);
}

// Exclusive ownership does satisfy a shared requirement; the reverse is a
// genuine error and stays uncompilable.
TEST(MapLock, ExclusiveSatisfiesSharedButNotViceVersa) {
  EXPECT_TRUE((std::is_convertible_v<const scovox::MapWriteHeld&,
                                     const scovox::MapLockHeld&>));
  EXPECT_FALSE((std::is_convertible_v<const scovox::MapLockHeld&,
                                      const scovox::MapWriteHeld&>));
}

// The guards are the only source of a witness, and they are real locks: the
// witness is reachable exactly while the guard is alive.
TEST(MapLock, GuardsHandOutTheWitnessAndActuallyLock) {
  std::shared_mutex m;
  {
    scovox::MapReadLock r(m);
    const scovox::MapLockHeld& h = r.held();
    (void)h;
    EXPECT_FALSE(m.try_lock());          // a reader holds it
    EXPECT_TRUE(m.try_lock_shared());    // ...but not exclusively
    m.unlock_shared();
  }
  EXPECT_TRUE(m.try_lock());
  m.unlock();
  {
    scovox::MapWriteLock w(m);
    const scovox::MapWriteHeld& h = w.held();
    const scovox::MapLockHeld& shared_view = h;   // exclusive implies shared
    (void)shared_view;
    EXPECT_FALSE(m.try_lock_shared());   // a writer holds it exclusively
  }
  EXPECT_TRUE(m.try_lock());
  m.unlock();
}

// A guard must not be copyable either — two guards for one acquisition would
// unlock twice.
TEST(MapLock, GuardsAreNotCopyable) {
  EXPECT_FALSE(std::is_copy_constructible_v<scovox::MapReadLock>);
  EXPECT_FALSE(std::is_copy_constructible_v<scovox::MapWriteLock>);
}

}  // namespace
