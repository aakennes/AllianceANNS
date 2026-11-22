#include "shuffle/shuffle.hpp"
#include <algorithm>
#include <stdexcept>

namespace alliance {
auto GenerateLevelRanges(layer_t max_layer, const MultiLevelConfig &config)
    -> std::vector<LevelRange> {
  if (max_layer < 0 || config.level_k == 0 || config.shuffle_o < 2 ||
      (config.shuffle_o & (config.shuffle_o - 1)) != 0) {
    throw std::invalid_argument(
        "GenerateLevelRanges: require max_layer >= 0, k > 0, and power-of-two shuffle_o >= 2");
  }
  std::vector<LevelRange> levels;
  const size_t stride = LayerStride(config.shuffle_o);

  // Grid is top-aligned: selected layers = max_layer, max_layer-stride, max_layer-2*stride, ...
  // The lowest selected layer >= 0 is max_layer % stride.
  const size_t grid_base = static_cast<size_t>(max_layer) % stride;
  size_t remaining = static_cast<size_t>(max_layer) / stride + 1;
  while (remaining > 0) {
    const size_t owned = std::min(config.level_k, remaining);
    const size_t overlap = std::min(config.level_m, remaining - owned);
    const auto le = static_cast<layer_t>(grid_base + (remaining - 1) * stride);
    const auto owner_lb = static_cast<layer_t>(grid_base + (remaining - owned) * stride);
    const auto lb = static_cast<layer_t>(grid_base + (remaining - owned - overlap) * stride);
    levels.push_back({lb, le, owner_lb, le});
    // Reaching layer zero in the overlapping record does NOT finish ownership.
    // Keep emitting replicas until the bottom selected layers have their own
    // shuffle pass, including a partial final k-layer group.
    remaining -= owned;
  }
  ValidateLevelRanges(levels, max_layer, stride);
  return levels;
}

} // namespace alliance
