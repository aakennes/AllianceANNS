#pragma once
#include <cstdint>

namespace alliance {

constexpr uint64_t kLayoutMagic = 0x314D4C4Cull; // "LLM1" layout family
// v2 persists disjoint ownership separately from overlapping stored ranges.
// v1 lacks enough information to recover ownership (including the missing
// bottom replica), so readers must reject it and request a fresh export.
constexpr uint64_t kLayoutVersion = 2;
constexpr uint64_t kMixedLayoutVersion = 3;
// v3 extension immediately follows GraphLayoutHeader. Vector IDs are relative
// to region_offset_bytes; neighbor IDs retain header-relative edge-block units.
struct MixedVectorGeometry {
  uint64_t block_size;
  uint64_t region_offset_bytes;
};

// Self-describing header written at the front of out_graph_path, padded to
// block_size. Lets an external reader locate every level's block region
// (and the vector region, if colocated) without consulting the sidecar.
struct GraphLayoutHeader {
  uint64_t magic;
  uint64_t version;
  uint64_t max_elements;
  uint64_t vec_d;
  uint64_t wp;
  uint64_t M;
  uint64_t curvec_num;
  uint64_t cur_max_layer;
  uint64_t block_size;
  uint64_t num_levels;
  uint64_t vec_region_enabled;
  uint64_t layer_stride; // log2(shuffle_o); 1 = all layers (default/old)
};

struct LevelHeaderEntry {
  int32_t layer_begin;
  int32_t layer_end;
  int32_t owner_begin;
  int32_t owner_end;
  uint64_t record_size;
  uint64_t elements_per_block;
  uint64_t num_blocks;
  uint64_t region_start_block;
};

struct VecHeaderEntry {
  uint64_t elements_per_block;
  uint64_t num_blocks;
  uint64_t region_start_block;
};

} // namespace alliance
