#pragma once

#include <vector>
#include <array>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <ostream>
#include <istream>
#include <sstream>
#include <string>
#include <type_traits>
#include <stdexcept>
#include <utility>
#include <lz4.h>

namespace scovox {

/**
 * @brief SCovox map binary format serializer optimized for map merging
 *
 * Binary format per voxel:
 * [a_occ:f32][a_free:f32][a_unk:f32][K:u8] then K × ([class_id:u16][cnt:f32])
 *
 * Mass conservation: Any semantic evidence beyond K_MAX is added to a_unk.
 */
class ScovoxBinarySerializer {
public:
    static constexpr uint8_t K_MAX = 2;
    static constexpr uint32_t MAGIC = 0x53435658;  // "SCVX"
    static constexpr uint8_t FORMAT_VERSION = 1;

    struct SemanticPair {
        uint16_t class_id;
        float count;
        SemanticPair(uint16_t id, float cnt) : class_id(id), count(cnt) {}
        bool operator>(const SemanticPair& other) const { return count > other.count; }
    };

    struct WireVoxel {
        float a_occ;
        float a_free;
        float a_unk;
        uint8_t k;
        std::vector<SemanticPair> semantic_pairs;
        WireVoxel() : a_occ(0.0f), a_free(0.0f), a_unk(0.0f), k(0) {}
    };

    struct CoordVoxelPair {
        int32_t x, y, z;
        float a_occ, a_free, a_unk;
        std::vector<std::pair<uint16_t, float>> semantics;
    };

    struct IncrementalVoxel {
        int32_t x, y, z;
        float a_occ, a_free, a_unk;
        std::array<std::pair<uint16_t, float>, 2> top{};
    };

    struct IncrementalUpdate {
        float resolution = 0.0f;
        std::vector<IncrementalVoxel> voxels;
    };

    static void serializeVoxel(
        float a_occ, float a_free, float a_unk,
        const std::vector<std::pair<uint16_t, float>>& semantic_evidence,
        std::ostream& out)
    {
        WireVoxel wire_voxel;
        prepareWireVoxel(a_occ, a_free, a_unk, semantic_evidence, wire_voxel);
        writeWireVoxel(wire_voxel, out);
    }

    static WireVoxel deserializeVoxel(std::istream& in) {
        WireVoxel wire_voxel;
        readWireVoxel(wire_voxel, in);
        return wire_voxel;
    }

    static void prepareWireVoxel(
        float a_occ, float a_free, float a_unk,
        const std::vector<std::pair<uint16_t, float>>& semantic_evidence,
        WireVoxel& wire_voxel)
    {
        wire_voxel.a_occ = a_occ;
        wire_voxel.a_free = a_free;
        wire_voxel.a_unk = a_unk;

        std::vector<SemanticPair> valid_pairs;
        valid_pairs.reserve(semantic_evidence.size());
        for (const auto& pair : semantic_evidence) {
            if (pair.second > 0.0f) valid_pairs.emplace_back(pair.first, pair.second);
        }
        std::sort(valid_pairs.begin(), valid_pairs.end(), std::greater<SemanticPair>());

        float dropped_mass = 0.0f;
        const size_t k_actual = std::min(static_cast<size_t>(K_MAX), valid_pairs.size());
        for (size_t i = k_actual; i < valid_pairs.size(); ++i) dropped_mass += valid_pairs[i].count;

        wire_voxel.a_unk += dropped_mass;
        wire_voxel.k = static_cast<uint8_t>(k_actual);
        wire_voxel.semantic_pairs.assign(valid_pairs.begin(), valid_pairs.begin() + k_actual);
    }

    static std::string serializeIncremental(
        float resolution,
        const std::vector<CoordVoxelPair>& voxels,
        int top_k = 2)
    {
        std::ostringstream oss(std::ios::binary);

        // Header: magic + format version
        oss.write(reinterpret_cast<const char*>(&MAGIC), sizeof(MAGIC));
        oss.write(reinterpret_cast<const char*>(&FORMAT_VERSION), sizeof(FORMAT_VERSION));

        if (voxels.empty()) {
            uint32_t update_count = 0, clear_count = 0;
            oss.write(reinterpret_cast<const char*>(&update_count), sizeof(update_count));
            oss.write(reinterpret_cast<const char*>(&clear_count), sizeof(clear_count));
            return oss.str();
        }

        uint32_t update_count = static_cast<uint32_t>(voxels.size());
        uint32_t clear_count = 0;
        oss.write(reinterpret_cast<const char*>(&update_count), sizeof(update_count));
        oss.write(reinterpret_cast<const char*>(&clear_count), sizeof(clear_count));
        oss.write(reinterpret_cast<const char*>(&resolution), sizeof(resolution));

        int32_t min_x = voxels[0].x, max_x = min_x;
        int32_t min_y = voxels[0].y, max_y = min_y;
        int32_t min_z = voxels[0].z, max_z = min_z;
        for (const auto& v : voxels) {
            min_x = std::min(min_x, v.x); max_x = std::max(max_x, v.x);
            min_y = std::min(min_y, v.y); max_y = std::max(max_y, v.y);
            min_z = std::min(min_z, v.z); max_z = std::max(max_z, v.z);
        }

        int32_t extent_x = max_x - min_x, extent_y = max_y - min_y, extent_z = max_z - min_z;
        oss.write(reinterpret_cast<const char*>(&min_x), sizeof(min_x));
        oss.write(reinterpret_cast<const char*>(&min_y), sizeof(min_y));
        oss.write(reinterpret_cast<const char*>(&min_z), sizeof(min_z));
        oss.write(reinterpret_cast<const char*>(&extent_x), sizeof(extent_x));
        oss.write(reinterpret_cast<const char*>(&extent_y), sizeof(extent_y));
        oss.write(reinterpret_cast<const char*>(&extent_z), sizeof(extent_z));

        bool use_8bit  = (extent_x < 256 && extent_y < 256 && extent_z < 256);
        bool use_16bit = (extent_x < 65536 && extent_y < 65536 && extent_z < 65536);
        uint8_t encoding_type = use_8bit ? 1 : (use_16bit ? 2 : 3);
        oss.write(reinterpret_cast<const char*>(&encoding_type), sizeof(encoding_type));

        std::vector<size_t> indices(voxels.size());
        for (size_t i = 0; i < indices.size(); ++i) indices[i] = i;
        std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
            const auto& va = voxels[a]; const auto& vb = voxels[b];
            if (va.x != vb.x) return va.x < vb.x;
            if (va.y != vb.y) return va.y < vb.y;
            return va.z < vb.z;
        });

        // The `top_k` parameter is intentionally NOT consulted here. The wire
        // format always carries the top-K_MAX strongest classes (and folds any
        // overflow into a_unk inside prepareWireVoxel), so a producer-side
        // pre-truncation would only lose mass without compressing further.
        // Pre-fix this loop did break early at top_k_limit and silently
        // discarded the dropped slots' counts, violating mass conservation.
        (void)top_k;
        for (size_t idx : indices) {
            const auto& v = voxels[idx];
            if (encoding_type == 1) {
                uint8_t dx = static_cast<uint8_t>(v.x - min_x), dy = static_cast<uint8_t>(v.y - min_y), dz = static_cast<uint8_t>(v.z - min_z);
                oss.write(reinterpret_cast<const char*>(&dx), sizeof(dx));
                oss.write(reinterpret_cast<const char*>(&dy), sizeof(dy));
                oss.write(reinterpret_cast<const char*>(&dz), sizeof(dz));
            } else if (encoding_type == 2) {
                uint16_t dx = static_cast<uint16_t>(v.x - min_x), dy = static_cast<uint16_t>(v.y - min_y), dz = static_cast<uint16_t>(v.z - min_z);
                oss.write(reinterpret_cast<const char*>(&dx), sizeof(dx));
                oss.write(reinterpret_cast<const char*>(&dy), sizeof(dy));
                oss.write(reinterpret_cast<const char*>(&dz), sizeof(dz));
            } else {
                oss.write(reinterpret_cast<const char*>(&v.x), sizeof(v.x));
                oss.write(reinterpret_cast<const char*>(&v.y), sizeof(v.y));
                oss.write(reinterpret_cast<const char*>(&v.z), sizeof(v.z));
            }

            // Pass the full semantic evidence vector through. prepareWireVoxel
            // sorts by descending count, keeps the top K_MAX, and folds any
            // overflow mass into a_unk — that's its whole job, do not
            // pre-truncate or filter here.
            serializeVoxel(v.a_occ, v.a_free, v.a_unk, v.semantics, oss);
        }
        return oss.str();
    }

    static IncrementalUpdate deserializeIncremental(const std::string& data, int top_k = 2) {
        IncrementalUpdate result;
        std::istringstream is(data, std::ios::binary);

        // Read and validate header
        uint32_t magic = 0;
        uint8_t version = 0;
        is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        is.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (!is || magic != MAGIC) return result;
        if (version != FORMAT_VERSION) return result;

        uint32_t update_count = 0, clear_count = 0;
        is.read(reinterpret_cast<char*>(&update_count), sizeof(update_count));
        is.read(reinterpret_cast<char*>(&clear_count), sizeof(clear_count));
        if (update_count == 0 && clear_count == 0) return result;

        is.read(reinterpret_cast<char*>(&result.resolution), sizeof(result.resolution));

        int32_t origin_x = 0, origin_y = 0, origin_z = 0;
        int32_t extent_x = 0, extent_y = 0, extent_z = 0;
        uint8_t encoding_type = 0;
        if (update_count > 0) {
            is.read(reinterpret_cast<char*>(&origin_x), sizeof(origin_x));
            is.read(reinterpret_cast<char*>(&origin_y), sizeof(origin_y));
            is.read(reinterpret_cast<char*>(&origin_z), sizeof(origin_z));
            is.read(reinterpret_cast<char*>(&extent_x), sizeof(extent_x));
            is.read(reinterpret_cast<char*>(&extent_y), sizeof(extent_y));
            is.read(reinterpret_cast<char*>(&extent_z), sizeof(extent_z));
            is.read(reinterpret_cast<char*>(&encoding_type), sizeof(encoding_type));
        }
        if (!is) return result;
        result.voxels.reserve(update_count);
        const size_t top_k_limit = static_cast<size_t>(std::max(1, top_k));

        for (uint32_t i = 0; i < update_count; ++i) {
            int32_t cx = 0, cy = 0, cz = 0;
            if (encoding_type == 1) {
                uint8_t dx, dy, dz;
                is.read(reinterpret_cast<char*>(&dx), sizeof(dx));
                is.read(reinterpret_cast<char*>(&dy), sizeof(dy));
                is.read(reinterpret_cast<char*>(&dz), sizeof(dz));
                cx = origin_x + dx; cy = origin_y + dy; cz = origin_z + dz;
            } else if (encoding_type == 2) {
                uint16_t dx, dy, dz;
                is.read(reinterpret_cast<char*>(&dx), sizeof(dx));
                is.read(reinterpret_cast<char*>(&dy), sizeof(dy));
                is.read(reinterpret_cast<char*>(&dz), sizeof(dz));
                cx = origin_x + dx; cy = origin_y + dy; cz = origin_z + dz;
            } else {
                is.read(reinterpret_cast<char*>(&cx), sizeof(cx));
                is.read(reinterpret_cast<char*>(&cy), sizeof(cy));
                is.read(reinterpret_cast<char*>(&cz), sizeof(cz));
            }
            if (!is) break;

            auto wire = deserializeVoxel(is);
            if (!is) break;

            IncrementalVoxel wv;
            wv.x = cx; wv.y = cy; wv.z = cz;
            wv.a_occ = wire.a_occ; wv.a_free = wire.a_free; wv.a_unk = wire.a_unk;
            // wire.semantic_pairs is sorted by descending count (set in
            // prepareWireVoxel), so the first `keep` items are the strongest.
            // Items beyond `keep` are folded into a_unk to preserve total
            // semantic mass — without this, configuring top_k below K_MAX
            // silently leaked the overflow slots' counts on every binary.
            const size_t keep = std::min({wire.semantic_pairs.size(), top_k_limit, size_t(2)});
            for (size_t j = 0; j < keep; ++j) {
                wv.top[j].first = wire.semantic_pairs[j].class_id;
                wv.top[j].second = wire.semantic_pairs[j].count;
            }
            for (size_t j = keep; j < wire.semantic_pairs.size(); ++j) {
                wv.a_unk += wire.semantic_pairs[j].count;
            }
            result.voxels.push_back(wv);
        }

        if (clear_count > 0) {
            is.ignore(static_cast<std::streamsize>(clear_count) * static_cast<std::streamsize>(3 * sizeof(int32_t)));
        }
        return result;
    }

    static float parseResolution(const std::string& data) {
        std::istringstream is(data, std::ios::binary);
        uint32_t magic = 0; uint8_t version = 0;
        is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        is.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (!is || magic != MAGIC || version != FORMAT_VERSION) return 0.0f;
        uint32_t update_count = 0, clear_count = 0;
        is.read(reinterpret_cast<char*>(&update_count), sizeof(update_count));
        is.read(reinterpret_cast<char*>(&clear_count), sizeof(clear_count));
        if (update_count == 0 && clear_count == 0) return 0.0f;
        float resolution = 0.0f;
        is.read(reinterpret_cast<char*>(&resolution), sizeof(resolution));
        return resolution;
    }

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

private:
    static void writeWireVoxel(const WireVoxel& voxel, std::ostream& out) {
        writeFloat32LittleEndian(voxel.a_occ, out);
        writeFloat32LittleEndian(voxel.a_free, out);
        writeFloat32LittleEndian(voxel.a_unk, out);
        writeLittleEndian(voxel.k, out);
        for (const auto& pair : voxel.semantic_pairs) {
            writeLittleEndian(pair.class_id, out);
            writeFloat32LittleEndian(pair.count, out);
        }
    }

    static void readWireVoxel(WireVoxel& voxel, std::istream& in) {
        voxel.a_occ = readFloat32LittleEndian(in);
        voxel.a_free = readFloat32LittleEndian(in);
        voxel.a_unk = readFloat32LittleEndian(in);
        voxel.k = readLittleEndian<uint8_t>(in);
        if (!in) throw std::runtime_error("binary_serializer: stream error reading voxel header");
        voxel.semantic_pairs.clear();
        voxel.semantic_pairs.reserve(voxel.k);
        for (uint8_t i = 0; i < voxel.k; ++i) {
            uint16_t class_id = readLittleEndian<uint16_t>(in);
            float count = readFloat32LittleEndian(in);
            if (!in) throw std::runtime_error("binary_serializer: stream error reading semantic pair");
            voxel.semantic_pairs.emplace_back(class_id, count);
        }
    }

    static void writeFloat32LittleEndian(float value, std::ostream& out) {
        static_assert(sizeof(float) == sizeof(uint32_t), "Expected IEEE-754 float32");
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        writeLittleEndian(bits, out);
    }

    template<typename T>
    static void writeLittleEndian(T value, std::ostream& out) {
        static_assert(std::is_integral<T>::value, "Type must be integral");
        for (size_t i = 0; i < sizeof(T); ++i) {
            out.put(static_cast<char>(value & 0xFF));
            value >>= 8;
        }
    }

    template<typename T>
    static T readLittleEndian(std::istream& in) {
        static_assert(std::is_integral<T>::value, "Type must be integral");
        T value = 0;
        for (size_t i = 0; i < sizeof(T); ++i) {
            T byte = static_cast<unsigned char>(in.get());
            value |= (byte << (i * 8));
        }
        return value;
    }

    static float readFloat32LittleEndian(std::istream& in) {
        static_assert(sizeof(float) == sizeof(uint32_t), "Expected IEEE-754 float32");
        uint32_t bits = readLittleEndian<uint32_t>(in);
        float value = 0.0f;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }
};

} // namespace scovox
