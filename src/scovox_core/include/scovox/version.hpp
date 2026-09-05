#pragma once
/// @file version.hpp
/// @brief SCovox library version, and what the binary was actually compiled as.

#define SCOVOX_VERSION_MAJOR 0
#define SCOVOX_VERSION_MINOR 1
#define SCOVOX_VERSION_PATCH 0
#define SCOVOX_VERSION_STRING "0.1.0"

namespace scovox {
  constexpr int version_major = SCOVOX_VERSION_MAJOR;
  constexpr int version_minor = SCOVOX_VERSION_MINOR;
  constexpr int version_patch = SCOVOX_VERSION_PATCH;
  constexpr const char* version_string = SCOVOX_VERSION_STRING;

/// The compiled-in value of every live `SCOVOX_*` build switch, as one stable
/// line of `NAME=value` pairs.
///
/// "Live" means the seven switches that declare an `#ifndef` default and can
/// therefore hold any value in a binary that exists. The four removed-arm traps
/// (`SCOVOX_VICTIM_MEAN`, `SCOVOX_VICTIM_QMAX`, `SCOVOX_ADMIT_NORM` and
/// `SCOVOX_EVICT_INHERIT`, all in `dir_voxel.hpp`) are deliberately absent:
/// each is a hard `#error` when nonzero, so a binary that links cannot
/// disagree with them, and printing a constant 0 would only pad the line. Note
/// that the offline replay harness still puts `-DSCOVOX_EVICT_INHERIT=0` on its
/// command line, so a reader diffing the harness's flags against this line will
/// find one that is missing here on purpose.
///
/// This exists because an md5 of the binary records that two builds differ, not
/// how, and a build manifest that lists the flags a script *intended* to pass
/// records an intention, not a fact. A `-D` misspelled on a command line is
/// accepted silently by the preprocessor, the switch keeps its header default,
/// and nothing downstream can tell. Every live switch is a `#ifndef` default in
/// its own header, so asking the translation unit what it saw is the only
/// answer that cannot drift from what was compiled: whatever the header
/// defaulted to or the command line overrode is what appears here.
///
/// Callers should log it once at startup so it lands in the run log next to the
/// results it produced. Deliberately one line, `NAME=value` space-separated,
/// stable order — a run log is grepped and diffed far more often than it is
/// read.
///
/// Returns a pointer to a string literal with static storage duration; it is
/// assembled by the preprocessor, so there is no initialisation order to worry
/// about and it is safe to call before `main`.
const char* buildSwitches();
}
