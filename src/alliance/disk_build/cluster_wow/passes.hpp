#pragma once

#include "core/types.hpp"
#include "build/disk_build_plan.hpp"
#include <cstddef>
#include <string>
#include <vector>

namespace alliance::disk_build {
class ClusterScratchDirectory;

// The driver owns both borrowed views until the synchronous passes return.
// Spill record: [rank u32][length u32 x L][ID u32 x L*M], with unused slots zeroed.
// Files are sector-padded; readers consume only their logical record lengths.
struct ClusterSpillPlan {
  const layer_t start_layer;
  const size_t n, L, rec_head, rec_bytes;
  const std::vector<size_t> &half_window;
  const std::string &graph_path;
  const Phase2Plan memory;
  ClusterScratchDirectory &scratch;
};

} // namespace alliance::disk_build
