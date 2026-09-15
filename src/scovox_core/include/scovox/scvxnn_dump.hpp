// scvxnn_dump.hpp -- the SCVXNN01 dump, defined once.
//
// replay_scenenn and replay_kitti used to carry a copy each of the three-pass
// collector below, and the fusion tool would have been a third.  One copy,
// because the scorers read one format: a divergence between two writers would
// show up as a scoring difference that no one could attribute to the map.
//
// It lives in scovox_core because scovox_node writes this format too, and the
// node and the replay drivers must not be two writers.  The old header path
// forwards here; the namespace is unchanged so no include site had to move.
//
// The state table (`pad` on the record) is the drivers' own:
//
//   0  Beta voxel, p_occ >= gate       label kept (or none)
//   1  Dir voxel with no Beta entry    label kept, p_occ is the 0.5 prior
//   2  Beta voxel, label from the ray-spread FALLBACK grid, ungated on p_occ
//   3  no Beta entry, label from the fallback grid, p_occ is the prior
//   4  Beta voxel, p_occ < gate        label kept only under
//                                      `below_as_unknown`, else dropped
//
// Occupancy readers treat 2 exactly like 0 and everything else as not
// occupied.  Passes run Beta, then Dir-without-Beta, then fallback-only, in
// the grids' own iteration order; a writer that changed that order would
// still score identically but would break every byte-identity gate, so the
// order is part of the contract.
#pragma once

#include "scovox/beta_voxel.hpp"
#include "scovox/dir_voxel.hpp"
#include "scovox/voxel.hpp"

#include <bonxai/bonxai.hpp>

#include <cstdint>
#include <fstream>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

namespace scovox_scenenn {

struct DumpRec { int32_t x, y, z; uint16_t cls; uint16_t pad; float p_occ, conf; };
static_assert(sizeof(DumpRec) == 24, "SCVXNN01 records are 24 bytes on disk");

struct DumpCounts {
  long long unknown_labelled      = 0;   // state 1
  long long fallback_labelled     = 0;   // states 2 and 3
  long long free_labels_dropped   = 0;   // p_occ < gate, label discarded
  long long below_gate_as_unknown = 0;   // state 4
};

using DumpBetaGrid = Bonxai::VoxelGrid<scovox::BetaVoxel>;
using DumpDirGrid  = Bonxai::VoxelGrid<scovox::DirVoxel>;

// argmax over the sparse top-K Dirichlet slots, with its normalised share.
inline bool bestClass(const scovox::DirVoxel* dv, uint16_t* cls, float* conf) {
  int best = -1;
  float bestc = 0.f;
  for (int i = 0; i < scovox::K_TOP; ++i)
    if (dv->cls[i] != scovox::kEmptySlot && dv->cnt[i] > bestc) { bestc = dv->cnt[i]; best = i; }
  if (best < 0) return false;
  const float s = dv->s_class();
  *cls  = dv->cls[best];
  *conf = (s > 0.f) ? (dv->cnt[best] / s) : 0.f;
  return true;
}

// Walk the grids and hand every record to `on_rec(rec, emitted)`, where
// `emitted` is the DirVoxel the label came from (null when the record carries
// no label).  `fdir` may be null: a map with no ray-spread fallback grid.
template <class OnRec>
DumpCounts collectDump(DumpBetaGrid& beta, DumpDirGrid& dir, DumpDirGrid* fdir,
                       float gate, bool below_as_unknown, OnRec&& on_rec) {
  DumpCounts n;
  auto dacc = dir.createAccessor();
  auto bacc = beta.createAccessor();
  std::optional<DumpDirGrid::Accessor> facc;
  if (fdir) facc.emplace(fdir->createAccessor());
  auto fallback = [&](const Bonxai::CoordT& c) -> const scovox::DirVoxel* {
    return facc ? facc->value(c, false) : nullptr;
  };

  // Pass 1 -- every Beta voxel.
  beta.forEachCell([&](scovox::BetaVoxel& b, const Bonxai::CoordT& c) {
    DumpRec r{c.x, c.y, c.z, scovox::kEmptySlot, 0, b.p_occ(), 0.f};
    const scovox::DirVoxel* dv = dacc.value(c, false);
    const scovox::DirVoxel* emitted = dv;
    uint16_t cls; float conf;
    if (dv && bestClass(dv, &cls, &conf)) {
      if (r.p_occ >= gate) { r.cls = cls; r.conf = conf; }
      else if (below_as_unknown) {
        r.cls = cls; r.conf = conf; r.pad = 4;
        ++n.below_gate_as_unknown;
      }
      else { ++n.free_labels_dropped; }
    } else if (const scovox::DirVoxel* fv = fallback(c)) {
      if (bestClass(fv, &cls, &conf)) {
        r.cls = cls; r.conf = conf; r.pad = 2;
        emitted = fv;
        ++n.fallback_labelled;
      }
    }
    on_rec(r, emitted);
  });

  // Pass 2 -- Dir voxels with no Beta entry.
  dir.forEachCell([&](scovox::DirVoxel& dv, const Bonxai::CoordT& c) {
    if (bacc.value(c, false)) return;
    uint16_t cls; float conf;
    if (bestClass(&dv, &cls, &conf)) {
      on_rec(DumpRec{c.x, c.y, c.z, cls, 1, 0.5f, conf}, &dv);
      ++n.unknown_labelled;
      return;
    }
    const scovox::DirVoxel* fv = fallback(c);
    if (fv && bestClass(fv, &cls, &conf)) {
      on_rec(DumpRec{c.x, c.y, c.z, cls, 3, 0.5f, conf}, fv);
      ++n.fallback_labelled;
    }
  });

  // Pass 3 -- fallback-only voxels.
  if (fdir) {
    fdir->forEachCell([&](scovox::DirVoxel& fv, const Bonxai::CoordT& c) {
      if (bacc.value(c, false) || dacc.value(c, false)) return;
      uint16_t cls; float conf;
      if (!bestClass(&fv, &cls, &conf)) return;
      on_rec(DumpRec{c.x, c.y, c.z, cls, 3, 0.5f, conf}, &fv);
      ++n.fallback_labelled;
    });
  }
  return n;
}

inline void printDumpCounts(std::ostream& os, size_t n_recs, const DumpCounts& n,
                            bool below_as_unknown) {
  os << "dump: " << n_recs << " voxels ("
     << n.unknown_labelled << " unknown-but-labelled, "
     << n.fallback_labelled << " fallback-labelled, "
     << n.free_labels_dropped << " labels dropped as free"
     << (below_as_unknown
           ? ", " + std::to_string(n.below_gate_as_unknown)
             + " below-gate labels re-stated as unknown"
           : std::string())
     << ")\n";
}

// "SCVXNN01" | f64 resolution | u32 n | DumpRec x n.  Returns false on any
// failure, including a short write, which close() reports through badbit
// rather than by throwing.
inline bool writeDump(const std::string& path, double res, const std::vector<DumpRec>& recs) {
  std::ofstream out(path, std::ios::binary);
  if (!out) return false;
  const char magic[8] = {'S','C','V','X','N','N','0','1'};
  out.write(magic, 8);
  const uint32_t n = static_cast<uint32_t>(recs.size());
  out.write(reinterpret_cast<const char*>(&res), 8);
  out.write(reinterpret_cast<const char*>(&n), 4);
  out.write(reinterpret_cast<const char*>(recs.data()),
            static_cast<std::streamsize>(recs.size() * sizeof(DumpRec)));
  out.close();
  return static_cast<bool>(out);
}

}  // namespace scovox_scenenn
