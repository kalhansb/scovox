#pragma once
/// @file version.hpp
/// @brief SCovox library version.

#define SCOVOX_VERSION_MAJOR 0
#define SCOVOX_VERSION_MINOR 1
#define SCOVOX_VERSION_PATCH 0
#define SCOVOX_VERSION_STRING "0.1.0"

namespace scovox {
  constexpr int version_major = SCOVOX_VERSION_MAJOR;
  constexpr int version_minor = SCOVOX_VERSION_MINOR;
  constexpr int version_patch = SCOVOX_VERSION_PATCH;
  constexpr const char* version_string = SCOVOX_VERSION_STRING;
}
