#pragma once
// Shared on-disk record geometry, ownership validation and layer routing.
// No dependency on the index implementation or offline shuffle algorithms.
#include "core/types.hpp"
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace alliance {

// Compute the record size for a level (edge/meta record, no distances).
// record = sizeof(tableint) + sizeof(label_t) + sizeof(att_t) + num_layers * (M+1) *
// sizeof(tableint)
inline auto ComputeRecordSize(size_t num_layers, size_t M, size_t att_bytes = sizeof(int))
    -> size_t {
  return sizeof(tableint) + sizeof(label_t) + att_bytes + num_layers * (M + 1) * sizeof(tableint);
}

// Compute elements_per_block from block_size and record_size. At least 1.
inline auto ComputeElementsPerBlock(size_t block_size, size_t record_size) -> size_t {
  return (record_size > 0 && block_size >= record_size) ? (block_size / record_size) : 1;
}

inline size_t LayerStride(size_t shuffle_o) {
  size_t s = 0, v = shuffle_o;
  while (v > 1) {
    v >>= 1;
    ++s;
  }
  return s > 0 ? s : 1;
}

inline size_t CountSelectedLayers(layer_t lb, layer_t le, size_t stride) {
  if (stride == 0 || lb < 0 || le < lb)
    throw std::invalid_argument("CountSelectedLayers: invalid layer grid");
  return static_cast<size_t>((le - lb) / static_cast<layer_t>(stride)) + 1;
}

// Reject gaps/overlaps in ownership, off-grid endpoints, or owned layers absent
// from their physical record. Used both before export and after loading v2.
void ValidateLevelRanges(const std::vector<LevelRange> &levels, layer_t max_layer, size_t stride);

// Route to the level owning the selected layer, never the first stored range
// covering it. Non-selected landing layers round down on the top-aligned grid;
// values below its lowest selected layer clamp to that layer.
auto FindLevelForLayer(layer_t layer, const std::vector<LevelRange> &levels, size_t stride = 1)
    -> size_t;

} // namespace alliance
