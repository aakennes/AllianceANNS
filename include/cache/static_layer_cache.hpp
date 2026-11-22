#pragma once
// alliance/cache/static_layer_cache.hpp -- Static per-node layer cache for
// AllianceIndexDisk's multi-level layout search path.
//
// Each cached node stores a contiguous range of layers [layer_begin,
// layer_end] in a single variable-size record. A node can span multiple
// disk levels: if selected in adjacent levels with m overlapping layers, the
// ranges merge into one entry and the overlap is stored only once.
//
// Lookup is O(1): index into static_node_meta_[node_id], check the layer
// range covers the requested layer, return the record pointer.
//
// Per-level budget is decided by the training-set LANDING LAYER DISTRIBUTION.
// Which NODES fill a level's slots is decided by the full-level in-edge
// weight (computed independently of shuffle). The only training artifact is a
// single landing-layer histogram (size wp+1).

#include "core/types.hpp"
#include <cstdint>
#include <type_traits>
#include <cstddef>
#include <cstring>
#include <vector>

namespace alliance {

// Per-node static cache metadata. O(1) lookup by node_id.
struct StaticNodeMeta {
  static constexpr uint32_t kInvalid = UINT32_MAX;

  uint32_t offset{kInvalid}; // byte offset into static_cache_buf_ (kInvalid = not cached)
  layer_t layer_begin{0};    // first cached layer for this node
  layer_t layer_end{0};      // last cached layer for this node
  uint32_t record_bytes{0};  // size of this node's record in bytes

  bool IsCached() const {
    return offset != kInvalid;
  }
  bool Covers(layer_t layer) const {
    return offset != kInvalid && layer >= layer_begin && layer <= layer_end;
  }
};

// v2 stays byte-for-byte unchanged. v3 is selected only on actual offset overflow.
inline constexpr uint8_t kStaticCacheV2 = 2, kStaticCacheV3 = 3;
struct StaticNodeMeta64 {
  static constexpr uint64_t kInvalid = UINT64_MAX;
  uint64_t offset{kInvalid};
  layer_t layer_begin{0}, layer_end{0};
  uint32_t record_bytes{0};
  uint32_t reserved{0}; // explicit padding in the v3 disk record
  bool IsCached() const {
    return offset != kInvalid;
  }
  bool Covers(layer_t layer) const {
    return IsCached() && layer >= layer_begin && layer <= layer_end;
  }
};
static_assert(sizeof(StaticNodeMeta) == 16);
static_assert(sizeof(StaticNodeMeta64) == 24);
static_assert(offsetof(StaticNodeMeta64, reserved) == 20);
class StaticCacheOffsetOverflow : public std::length_error {
public:
  StaticCacheOffsetOverflow() : std::length_error("static cache requires 64-bit offsets") {
  }
};

// Per-level budget allocation helper. Used during build only (not at query
// time). Holds the per-level slot count and landing weight for budget
// distribution. No bitset/rank table -- lookup goes through StaticNodeMeta.
struct StaticLayerGroup {
  layer_t layer_begin{0};
  layer_t layer_end{0};
  size_t layer_count{0}; // layer_end - layer_begin + 1
  size_t piece_bytes{0}; // record_size for a FULL (k+m layer) record in this level
  size_t slot_count{0};  // how many nodes allocated for this level
  size_t byte_offset{0}; // (legacy, unused in per-node layout)

  size_t n_elements_{0}; // num_nodes (for bounds checking)

  // Per-level weight for budget allocation, derived from the landing-layer
  // distribution (sum of landing count over the level's top non-overlap layers).
  uint64_t landing_weight{0};

  void Init(layer_t lb, layer_t le, size_t piece_bytes_, size_t num_nodes) {
    layer_begin = lb;
    layer_end = le;
    layer_count = static_cast<size_t>(le - lb + 1);
    piece_bytes = piece_bytes_;
    n_elements_ = num_nodes;
    slot_count = 0;
    byte_offset = 0;
    landing_weight = 0;
  }
};

} // namespace alliance
