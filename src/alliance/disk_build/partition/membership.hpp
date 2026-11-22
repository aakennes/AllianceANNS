#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace alliance::disk_build {

// Rank-major membership shared by the Vamana cluster builder and commit pass.
// Search quantization belongs to the index and never enters this workspace.
struct ClusterBuildState {
  size_t K{0};
  static constexpr size_t np = 2;
  std::vector<uint32_t> cluster_rank_offsets_; // [K+1]
  std::vector<uint32_t> cluster_ranks_;        // [n*2], slot -> rank
  std::vector<uint32_t> rank_cids_;            // [n*2], rank -> nearest clusters
};

} // namespace alliance::disk_build
