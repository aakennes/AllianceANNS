#include "shuffle/level_geometry.hpp"
#include <limits>

namespace alliance {

void ValidateLevelRanges(const std::vector<LevelRange> &levels, layer_t max_layer, size_t stride) {
  if (levels.empty() || max_layer < 0 || stride == 0 ||
      stride > static_cast<size_t>(std::numeric_limits<layer_t>::max())) {
    throw std::runtime_error("Invalid multi-level layout: empty levels or invalid layer grid");
  }
  const auto step = static_cast<layer_t>(stride);
  const layer_t grid_base = max_layer % step;
  layer_t expected_end = max_layer;
  for (const auto &r : levels) {
    if (r.layer_begin < grid_base || r.layer_begin > r.owner_begin || r.owner_begin > r.owner_end ||
        r.owner_end != r.layer_end || r.owner_end != expected_end ||
        (r.layer_begin - grid_base) % step != 0 || (r.owner_begin - grid_base) % step != 0) {
      throw std::runtime_error(
          "Invalid multi-level layout: ownership must partition the selected layers and lie inside each stored range");
    }
    expected_end = r.owner_begin - step;
  }
  if (expected_end != grid_base - step) {
    throw std::runtime_error(
        "Invalid multi-level layout: bottom layers have no owning replica; re-export the layout");
  }
}

auto FindLevelForLayer(layer_t layer, const std::vector<LevelRange> &levels, size_t stride)
    -> size_t {
  if (levels.empty() || stride == 0 || layer < 0 || layer > levels.front().owner_end) {
    throw std::out_of_range("FindLevelForLayer: invalid layer or empty layout");
  }
  const auto step = static_cast<layer_t>(stride);
  const layer_t grid_base = levels.back().owner_begin;
  const layer_t selected =
      layer < grid_base ? grid_base : static_cast<layer_t>(layer - (layer - grid_base) % step);
  for (size_t i = 0; i < levels.size(); ++i) {
    if (selected >= levels[i].owner_begin && selected <= levels[i].owner_end) {
      return i;
    }
  }
  throw std::runtime_error("FindLevelForLayer: selected layer has no owner");
}

} // namespace alliance
