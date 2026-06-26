#pragma once

/// @file lz4_codec.hpp
/// @brief Standalone LZ4 (de)compression for the binary map wire payload.
///
/// These two helpers used to live on the v1 `ScovoxBinarySerializer` class in
/// `binary_serializer.hpp`. The v4 wire path (the only surviving codec) reuses
/// them to compress the serialized blob before it goes on the ROS topic, so
/// they were lifted here verbatim when the v1/v2/v3 headers were removed. The
/// enclosing struct name is kept (`ScovoxBinarySerializer`) so the call sites
/// `scovox::ScovoxBinarySerializer::compressLZ4 / ::decompressLZ4` are byte
/// identical to before.
///
/// Wire framing: a 4-byte little-endian original-size header is prepended to
/// the LZ4 payload so the decompressor knows the exact allocation. A 256 MB
/// sanity cap rejects obviously corrupt sizes.

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <lz4.h>

namespace scovox {

struct ScovoxBinarySerializer {
  static std::vector<uint8_t> compressLZ4(const std::string& data) {
    if (data.empty()) return {};
    const uint32_t orig_size = static_cast<uint32_t>(data.size());
    const int max_compressed_size = LZ4_compressBound(static_cast<int>(data.size()));
    // Prepend 4-byte original size so decompressor knows exact allocation
    std::vector<uint8_t> compressed(sizeof(uint32_t) + static_cast<size_t>(max_compressed_size));
    std::memcpy(compressed.data(), &orig_size, sizeof(orig_size));
    const int compressed_size = LZ4_compress_default(
        data.c_str(), reinterpret_cast<char*>(compressed.data() + sizeof(uint32_t)),
        static_cast<int>(data.size()), max_compressed_size);
    if (compressed_size <= 0) return {};
    compressed.resize(sizeof(uint32_t) + static_cast<size_t>(compressed_size));
    return compressed;
  }

  static std::string decompressLZ4(const std::vector<uint8_t>& compressed) {
    const int comp_size = static_cast<int>(compressed.size());
    if (comp_size <= static_cast<int>(sizeof(uint32_t))) return {};
    // Read prepended original size
    uint32_t orig_size = 0;
    std::memcpy(&orig_size, compressed.data(), sizeof(orig_size));
    const int payload_size = comp_size - static_cast<int>(sizeof(uint32_t));
    // Sanity cap: reject obviously corrupt sizes (>256 MB)
    if (orig_size == 0 || orig_size > 256u * 1024u * 1024u) return {};
    std::vector<char> buf(orig_size);
    const int decomp_size = LZ4_decompress_safe(
        reinterpret_cast<const char*>(compressed.data() + sizeof(uint32_t)),
        buf.data(), payload_size, static_cast<int>(orig_size));
    if (decomp_size < 0) return {};
    return std::string(buf.data(), static_cast<size_t>(decomp_size));
  }
};

}  // namespace scovox
