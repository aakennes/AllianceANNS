#include "disk_build/stages/internal.hpp"
#include "disk_build/stages/checkpoint.hpp"
#include "disk_build/memory/process.hpp"
#include "index/index_disk.hpp"
#include <chrono>
#include <cstdio>

namespace alliance::disk_build {

template <typename att_t, typename vec_t>
void PrepareUpperMemoryBoundary(AllianceIndexDisk<att_t, vec_t> &index, size_t n,
                                const ResidentStageResult &resident) {
  // One boundary owns sampling, trimming and peak-window reset. A resumed build
  // still needs it even though no resident buffers were allocated this time.
  CompleteMemoryPhase(resident.built ? "Resident LeftStream" : "Resident reused",
                      resident.built ? resident.memory_model_bytes
                                     : index.EstimateBaseBuildBytes(n),
                      index.build_config_.build_mem_gb_);

  // Capture the post-trim process floor for upper-stage budget estimates.
  size_t floor_bytes = 0;
  index.ProcRss(&floor_bytes, nullptr);
  if (floor_bytes > 0) {
    index.measured_base_bytes_ = floor_bytes;
    const size_t estimate = index.EstimateBaseBuildBytes(n);
    fprintf(stderr, "  [RSS] upper-stage base := measured %.1f MB (estimate %.1f MB, %+.1f MB)\n",
            floor_bytes / (1024.0 * 1024.0), estimate / (1024.0 * 1024.0),
            (double(floor_bytes) - double(estimate)) / (1024.0 * 1024.0));
  }
}

template <typename att_t, typename vec_t>
void RunUpperStage(AllianceIndexDisk<att_t, vec_t> &index, size_t n, layer_t resident_max,
                   size_t start_layer, const std::string &checkpoint_meta_path) {
  // Upper construction commits as one stage. A resident checkpoint can resume
  // it wholesale; a completed upper checkpoint skips all partition/build work.
  if (resident_max >= static_cast<layer_t>(index.wp_) ||
      start_layer > static_cast<size_t>(resident_max) + 1)
    return;

  const auto start = std::chrono::high_resolution_clock::now();
  // Preserve the Phase2 log tokens consumed by existing experiment drivers.
  fprintf(stderr, "\n[Phase2] ClusterBuildUpperLayersDisk: layer %d..%zu\n", resident_max + 1,
          index.wp_);
  index.ClusterPrepDisk(index.cluster_centroids_ext_, index.cluster_ids_ext_, n,
                        index.cluster_K_file_);
  index.ClusterBuildUpperLayersDisk(static_cast<layer_t>(resident_max + 1),
                                    static_cast<layer_t>(index.wp_), n);
  index.TeardownClusterBuildDisk();
  index.graph_file_->Sync();

  const auto end = std::chrono::high_resolution_clock::now();
  fprintf(stderr, "  Phase2 (cluster) done: %.2fs\n",
          std::chrono::duration<double>(end - start).count());
  CommitUpperCheckpoint(index, checkpoint_meta_path);
  if (!checkpoint_meta_path.empty()) {
    fprintf(stderr, "  Checkpoint: layer %zu (top) saved to %s\n", index.wp_,
            checkpoint_meta_path.c_str());
  }
}

template void PrepareUpperMemoryBoundary(AllianceIndexDisk<int, float> &, size_t,
                                         const ResidentStageResult &);
template void RunUpperStage(AllianceIndexDisk<int, float> &, size_t, layer_t, size_t,
                            const std::string &);
} // namespace alliance::disk_build
