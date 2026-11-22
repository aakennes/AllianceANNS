#include "disk_build/memory/process.hpp"
#include "disk_build/partition/membership.hpp"
#include "disk_build/cluster_wow/passes.hpp"
#include "disk_build/cluster_wow/scratch.hpp"
#include "disk_build/memory/index_adapter.hpp"
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace alliance {

// Phase 2 constructs each Vamana cluster, then merges its ID-only edge records
// in global rank order.
// Keep Sync before O_DIRECT reads and trim/RSS between the two scratch scopes.
template <typename att_t, typename vec_t>
void AllianceIndexDisk<att_t, vec_t>::ClusterBuildUpperLayersDisk(layer_t start_layer,
                                                                  layer_t end_layer, size_t n) {
  if (ClusterScratch() == nullptr) {
    throw std::runtime_error("ClusterBuildUpperLayersDisk: ClusterPrepDisk has not run");
  }
  const size_t L = (size_t)(end_layer - start_layer + 1);
  const size_t K = ClusterScratch()->K;
  const size_t NP = ClusterScratch()->np;
  const size_t vec_bytes = VecBytes();
  if (vec_bytes > VecPageBytes()) {
    throw std::runtime_error(
        "ClusterBuildUpperLayersDisk: VecBytes() exceeds the build vector page");
  }

  std::vector<size_t> half_window(L);
  for (size_t li = 0; li < L; ++li) {
    half_window[li] = window_size_[start_layer + (layer_t)li] / 2;
  }

  const size_t rec_head = sizeof(uint32_t) * (1 + L);
  const size_t rec_bytes = rec_head + L * size_t(M_) * sizeof(tableint);
  fprintf(stderr, "[ClusterSpill] ids_only=1 edge_payload_bytes=%zu distance_payload_bytes=0\n",
          sizeof(tableint));
  fprintf(stderr,
          "\n[Phase2-Cluster-Disk] BuildUpperLayers: layers %d..%d, n=%zu, K=%zu, np=%zu,"
          " rec_bytes=%zu, rule=Vamana (deduplicate/shuffle/truncate), spill_est=%.2f GB\n",
          int(start_layer), int(end_layer), n, K, NP, rec_bytes,
          double(n * NP * rec_bytes) / (1024.0 * 1024 * 1024));

  // Phase 1/1.5 wrote through the buffered PagedFile; flush so the O_DIRECT
  // reads below observe the freshly written lower-layer edges.
  graph_file_->Sync();

  const std::string graph_path = graph_file_->path();

  // Resolve the accepted partition's model once, after CSR and prep residue exist.
  const auto memory = disk_build::MemoryPlanner<att_t, vec_t>::FromIndex(*this).PlanPartition(
      n, K, build_config_.cluster_balance_);
  disk_build::ClusterScratchDirectory scratch(graph_path,
                                              ClusterScratch()->cluster_rank_offsets_.data(), K);
  const disk_build::ClusterSpillPlan plan{start_layer, n,          L,      rec_head, rec_bytes,
                                          half_window, graph_path, memory, scratch};
  BuildClusterWoWEdgesDisk(plan);

  // Pass A buffers have now been destroyed. Start Pass B's RSS window only
  // after trimming their allocator residue; persistent memberships remain live.
  disk_build::CompleteMemoryPhase("Phase2 passA teardown", memory.base_bytes + memory.state_bytes,
                                  build_config_.build_mem_gb_);

  CommitClusterEdgesDisk(plan);
  scratch.Complete();
}

template class AllianceIndexDisk<int, float>;

} // namespace alliance
