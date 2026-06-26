#pragma once

/// @file lz4_codec.hpp
/// @brief Standalone LZ4 (de)compression for the binary map wire payload.
///
/// These two helpers used to live on the `ScovoxBinarySerializer` class in
/// `binary_serializer.hpp`. The wire path (the only codec) reuses
/// them to compress the serialized blob before it goes on the ROS topic, so
/// they were lifted here verbatim when the older serializer headers were removed. The
/// enclosing struct name is kept (`ScovoxBinarySerializer`) so the call sites
/// `scovox::ScovoxBinarySerializer::compressLZ4 / ::decompressLZ4` are byte
/// identical to before.
///
/// Wire framing: a 4-byte big-endian (network byte order) original-size header
/// is prepended to the LZ4 payload so the decompressor knows the exact
/// allocation independent of host endianness. A 256 MB sanity cap rejects
/// obviously corrupt sizes.

#include <cstdint>
#include <string>
#include <vector>

#include <lz4.h>

namespace scovox {

struct ScovoxBinarySerializer {
  static std::vector<uint8_t> compressLZ4(const std::string& data) {
    if (data.empty()) return {};
    const uint32_t orig_size = static_cast<uint32_t>(data.size());
    const int max_compressed_size = LZ4_compressBound(static_cast<int>(data.size()));
    // Prepend the original size as a 4-byte big-endian (network byte order)
    // header so the decompressor sizes its buffer independent of host endianness.
    std::vector<uint8_t> compressed(sizeof(uint32_t) + static_cast<size_t>(max_compressed_size));
    compressed[0] = static_cast<uint8_t>((orig_size >> 24) & 0xFFu);
    compressed[1] = static_cast<uint8_t>((orig_size >> 16) & 0xFFu);
    compressed[2] = static_cast<uint8_t>((orig_size >> 8) & 0xFFu);
    compressed[3] = static_cast<uint8_t>(orig_size & 0xFFu);
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
    // Read the prepended 4-byte big-endian (network byte order) original size.
    const uint32_t orig_size =
        (static_cast<uint32_t>(compressed[0]) << 24) |
        (static_cast<uint32_t>(compressed[1]) << 16) |
        (static_cast<uint32_t>(compressed[2]) << 8) |
        static_cast<uint32_t>(compressed[3]);
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
