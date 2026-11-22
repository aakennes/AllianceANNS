#pragma once
#include "shuffle/layout.hpp"

namespace alliance::shuffle_detail {
// Pure geometry: no index, files, ordering computation, or O(N) mappings.
struct LayoutPlan {
  LayoutExportResult layout;
  size_t header_padding{0};
  size_t neighbor_end{0};
};
LayoutPlan PlanLayout(size_t nodes, size_t vector_bytes, size_t degree, size_t attribute_bytes,
                      const std::vector<LevelRange> &levels, const LayoutExportConfig &config);
} // namespace alliance::shuffle_detail
