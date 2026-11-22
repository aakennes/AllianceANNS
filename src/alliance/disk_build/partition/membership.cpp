#include "index/index_disk.hpp"
#include "disk_build/partition/membership.hpp"
#include "io/disk_resource_guard.hpp"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <vector>

namespace alliance {

// ============================================================================
// SetupClusterBuildDisk
// ============================================================================

template <typename att_t, typename vec_t>
void AllianceIndexDisk<att_t, vec_t>::SetupClusterBuildDisk(const uint32_t *cluster_ids, size_t K,
                                                            size_t n) {
  constexpr size_t np = 2;
  if (!cluster_ids || K < np || K > n)
    throw std::invalid_argument("invalid nearest-two CSR geometry");
  TeardownClusterBuildDisk();
  detail::ScopeExit failed_setup([this]() noexcept { TeardownClusterBuildDisk(); });
  BuildWorkspace().cluster = std::make_unique<DiskClusterBuildState>();
  ClusterScratch()->K = K;

  ClusterScratch()->rank_cids_.resize(n * np);

  // Input memberships use original labels; the CSR uses global rank. Counting
  // sort avoids per-cluster append buffers and preserves rank order for both
  // the sequential cluster input and commit streams.
  ClusterScratch()->cluster_rank_offsets_.assign(K + 1, 0);
  for (size_t rank = 0; rank < n; ++rank) {
    const label_t label = labels_[sorted_ids_[rank]];
    for (size_t a = 0; a < np; ++a) {
      const uint32_t cid = cluster_ids[(size_t)label * np + a];
      if (cid >= K)
        throw std::invalid_argument("cluster ID outside CSR range");
      ClusterScratch()->rank_cids_[rank * np + a] = cid;
      ++ClusterScratch()->cluster_rank_offsets_[cid + 1];
    }
  }
  for (size_t c = 0; c < K; ++c) {
    ClusterScratch()->cluster_rank_offsets_[c + 1] += ClusterScratch()->cluster_rank_offsets_[c];
  }
  ClusterScratch()->cluster_ranks_.resize(n * np);
  {
    std::vector<uint32_t> cursor(ClusterScratch()->cluster_rank_offsets_.begin(),
                                 ClusterScratch()->cluster_rank_offsets_.end() - 1);
    for (size_t rank = 0; rank < n; ++rank) {
      for (size_t a = 0; a < np; ++a) {
        const uint32_t dst = cursor[ClusterScratch()->rank_cids_[rank * np + a]]++;
        ClusterScratch()->cluster_ranks_[dst] = (uint32_t)rank;
      }
    }
  }

  size_t min_sz = n * np, max_sz = 0, total = 0;
  for (size_t c = 0; c < K; ++c) {
    size_t sz =
        ClusterScratch()->cluster_rank_offsets_[c + 1] - ClusterScratch()->cluster_rank_offsets_[c];
    min_sz = std::min(min_sz, sz);
    max_sz = std::max(max_sz, sz);
    total += sz;
  }
  // Report the actual skew used by the Vamana memory planner.
  fprintf(stderr,
          "  Cluster stats (disk): K=%zu, np=%zu, slots=%zu (n*np=%zu), min=%zu, max=%zu, avg=%.0f"
          " (skew max/avg=%.2f)\n",
          K, np, total, n * np, min_sz, max_sz, (double)total / K,
          total > 0 ? (double)max_sz * K / (double)total : 1.0);
  assert(total == n * np);
  failed_setup.Release();
}

template void AllianceIndexDisk<int, float>::SetupClusterBuildDisk(const uint32_t *, size_t,
                                                                   size_t);

} // namespace alliance
