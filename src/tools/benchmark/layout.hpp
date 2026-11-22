#pragma once
#include "shuffle/layout_format.hpp"
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace benchmark::formal {
struct Geometry {
  uint64_t edge_bytes, vector_bytes, header_bytes, vector_start, file_bytes;
};
inline void Check(bool ok, const char *why) {
  if (!ok)
    throw std::runtime_error(why);
}
inline uint64_t CheckedBytes(uint64_t blocks, uint64_t size, uint64_t start) {
  Check(size && blocks <= (std::numeric_limits<uint64_t>::max() - start) / size,
        "layout geometry overflows");
  return start + blocks * size;
}
template <class Index> Geometry ReadGeometry(const std::string &path, const Index &index) {
  std::ifstream file(path, std::ios::binary);
  alliance::GraphLayoutHeader h{};
  file.read(reinterpret_cast<char *>(&h), sizeof(h));
  Check(file && h.magic == alliance::kLayoutMagic &&
            (h.version == alliance::kLayoutVersion || h.version == alliance::kMixedLayoutVersion) &&
            h.curvec_num == index.curvec_num_ && h.vec_d == index.vec_d_ && h.wp == index.wp_ &&
            h.M == index.M_ && h.num_levels > 0 && h.num_levels <= h.cur_max_layer + 1 &&
            h.vec_region_enabled == 1,
        "shuffled layout identity mismatch");
  alliance::MixedVectorGeometry mixed{};
  if (h.version == alliance::kMixedLayoutVersion)
    file.read(reinterpret_cast<char *>(&mixed), sizeof(mixed));
  uint64_t next = 0;
  for (uint64_t i = 0; i < h.num_levels; ++i) {
    alliance::LevelHeaderEntry level{};
    file.read(reinterpret_cast<char *>(&level), sizeof(level));
    Check(file && level.region_start_block == next && level.num_blocks > 0 &&
              level.num_blocks <= h.curvec_num,
          "invalid neighbor region");
    Check(next <= std::numeric_limits<uint64_t>::max() - level.num_blocks,
          "neighbor region overflow");
    next += level.num_blocks;
  }
  alliance::VecHeaderEntry vec{};
  file.read(reinterpret_cast<char *>(&vec), sizeof(vec));
  Check(file && vec.elements_per_block > 0 &&
            vec.num_blocks == (h.curvec_num - 1) / vec.elements_per_block + 1,
        "invalid vector region");
  Geometry g{h.block_size, index.ml_vec_block_size_, index.ml_header_bytes_, 0,
             std::filesystem::file_size(path)};
  Check(g.edge_bytes == index.ml_block_size_ && index.ml_vec_region_loaded_,
        "loaded page geometry mismatch");
  const auto edge_end = CheckedBytes(next, g.edge_bytes, g.header_bytes);
  if (h.version == alliance::kLayoutVersion) {
    Check(vec.region_start_block == next && g.vector_bytes == g.edge_bytes,
          "v2 vector geometry mismatch");
    g.vector_start = edge_end;
  } else {
    Check(mixed.block_size == g.vector_bytes && mixed.region_offset_bytes >= edge_end &&
              mixed.region_offset_bytes - edge_end < g.vector_bytes &&
              mixed.region_offset_bytes % g.vector_bytes == 0,
          "v3 vector geometry mismatch");
    g.vector_start = mixed.region_offset_bytes;
  }
  Check(g.vector_bytes && vec.elements_per_block == g.vector_bytes / (h.vec_d * sizeof(float)) &&
            CheckedBytes(vec.num_blocks, g.vector_bytes, g.vector_start) == g.file_bytes,
        "vector page capacity or file length mismatch");
  return g;
}
} // namespace benchmark::formal
