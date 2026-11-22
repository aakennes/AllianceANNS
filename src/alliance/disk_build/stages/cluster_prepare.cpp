#include "index/index_disk.hpp"
#include "disk_build/memory/index_adapter.hpp"
#include "disk_build/memory/process.hpp"
#include "io/disk_resource_guard.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace alliance {

template <typename att_t, typename vec_t>
void AllianceIndexDisk<att_t, vec_t>::ClusterPrepDisk(const float *centroids,
                                                      const uint32_t *cluster_ids, size_t n,
                                                      size_t K) {
  if (n < 2)
    throw std::invalid_argument("Vamana partition requires n>=2");
  if ((centroids == nullptr) != (cluster_ids == nullptr))
    throw std::invalid_argument("Vamana partition requires paired external inputs");

  // Generated partition data remains alive through the CSR memory boundary.
  // Releasing it sooner would change the measured floor used by later passes.
  std::vector<float> owned_centroids;
  std::vector<uint32_t> owned_cluster_ids;
  if (centroids == nullptr) {
    K = build_config_.cluster_K_;
    SynthesizeVamanaClustersDisk(K, n, owned_centroids, owned_cluster_ids);
    cluster_ids = owned_cluster_ids.data();
  }

  // Validate both sources under the same nearest-two membership contract.
  if (K < 2 || K > n)
    throw std::invalid_argument("invalid external Vamana K");
  {
    std::vector<size_t> counts(K, 0);
    for (size_t i = 0; i < n; ++i) {
      const auto a = cluster_ids[2 * i];
      const auto b = cluster_ids[2 * i + 1];
      if (a >= K || b >= K || a == b)
        throw std::invalid_argument("invalid nearest-two membership");
      ++counts[a];
      ++counts[b];
    }
    build_config_.cluster_balance_ =
        float(double(*std::max_element(counts.begin(), counts.end())) * K / (2. * n) + 0.001);
    cluster_K_file_ = K;
  }

  // Resolve the accepted partition once; the same model checks the gate and
  // accounts for the CSR allocation below.
  const auto plan = disk_build::MemoryPlanner<att_t, vec_t>::FromIndex(*this).PlanPartition(
      n, K, build_config_.cluster_balance_);
  if (build_config_.build_mem_gb_ > 0.0) {
    if (plan.base_over_budget)
      throw std::runtime_error("ClusterPrepDisk: BASE alone exceeds --build_mem_gb");
    if (plan.user_over_budget) {
      fprintf(stderr,
              "  ERROR: Vamana K=%zu needs %.2f GiB including the process base; "
              "--build_mem_gb=%.2f. Revise the partition or raise the budget.\n",
              K, (plan.base_bytes + plan.residency_bytes) / double(1ull << 30),
              build_config_.build_mem_gb_);
      throw std::runtime_error("ClusterPrepDisk: Phase 2 K over --build_mem_gb");
    }
  }

  fprintf(stderr, "\n[ClusterPrep] K=%zu, np=2\n", K);
  const auto started = std::chrono::steady_clock::now();
  detail::ScopeExit failed_prep([this]() noexcept { TeardownClusterBuildDisk(); });
  SetupClusterBuildDisk(cluster_ids, K, n);
  fprintf(stderr, "  SetupClusterBuildDisk: %.2fs\n",
          std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
  disk_build::CompleteMemoryPhase("ClusterPrep CSR", plan.base_bytes + plan.state_bytes,
                                  build_config_.build_mem_gb_);

  // Account for allocator/thread-local pages retained after partitioning. The
  // upper-stage planner includes this residue in distribution and both passes.
  const size_t floor_bytes = disk_build::CaptureMemorySnapshot().current_bytes;
  const size_t modelled = plan.base_bytes + plan.state_bytes;
  measured_prep_residue_bytes_ = floor_bytes > modelled ? floor_bytes - modelled : 0;
  fprintf(stderr, "  [RSS] Cluster WoW prep residue: %.1f MB\n",
          measured_prep_residue_bytes_ / (1024.0 * 1024.0));
  failed_prep.Release();
}

template void AllianceIndexDisk<int, float>::ClusterPrepDisk(const float *, const uint32_t *,
                                                             size_t, size_t);

} // namespace alliance
