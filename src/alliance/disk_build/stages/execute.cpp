#include "disk_build/stages/internal.hpp"
#include "index/index_disk.hpp"
#include <cstdio>
#include <stdexcept>

namespace alliance::disk_build {

template <typename att_t, typename vec_t>
void ExecuteBuild(AllianceIndexDisk<att_t, vec_t> &index, const BuildInvocation &request) {
  const int threads = request.threads;
  const size_t start_layer = request.start_layer;
  const auto &checkpoint_meta_path = request.checkpoint_meta_path;
  // This entry starts after population or checkpoint loading. Reject legacy
  // resident state before mutating tuning or acquiring temporary build resources.
  index.RequireRawGraph("ExecuteBuild");
  if (!index.build_config_.left_stream_build_)
    throw std::runtime_error("legacy resident build is retired; checkpoint is load/search-only");
  index.build_config_.merge_build_threads_ = threads;
  size_t n = index.curvec_num_;

  fprintf(
      stderr,
      "Build(disk): n=%zu, dim=%zu, M=%zu, o=%zu, enum_threshold=%zu, wow_threshold=%zu, threads=%d\n",
      n, index.vec_d_, index.M_, index.o_, index.enum_threshold_,
      index.build_config_.wow_threshold_, threads);
  fprintf(stderr, "  wp=%zu, win_anchor=%s, window_size: ", index.wp_,
          WindowAnchorName(index.win_anchor_));
  for (size_t i = 0; i <= index.wp_; ++i)
    fprintf(stderr, "L%zu=%zu ", i, index.window_size_[i]);
  fprintf(stderr, "\n");
  if (start_layer > 1) {
    fprintf(stderr, "  Resuming from layer %zu (layers 0..%zu assumed complete on disk)\n",
            start_layer, start_layer - 1);
  }

  index.ReportBuildMemoryBudget(n);

  // The session releases temporary cluster state on every exit.
  BuildSession<att_t, vec_t> session(index.BuildWorkspace());
  // vecs_per_page_ is normally set by WriteBuildVecFile{,FromGraph}, which the
  // --resume path skips -- recompute it here so every entry into this function
  // has it, or GatherVecsByRank divides by zero (SIGFPE).
  index.vecs_per_page_ = index.VecPageBytes() / index.VecBytes();
  if (index.vecs_per_page_ == 0)
    index.vecs_per_page_ = 1;
  index.build_vec_file_path_ = index.graph_file_->path() + ".buildvecs";
  // Resident construction owns its durability/checkpoint boundary. A completed
  // LeftStream checkpoint reuses the recorded layer boundary instead of rebuilding.
  const auto resident = RunResidentStage(index, n, start_layer, checkpoint_meta_path);
  // Resident buffers are now gone: trim/sample before solving the upper budget.
  PrepareUpperMemoryBoundary(index, n, resident);
  // Build only the remaining upper layers; this stage also Syncs before advancing
  // completion and saving its checkpoint. BuildSession owns exception cleanup.
  RunUpperStage(index, n, resident.maximum, start_layer, checkpoint_meta_path);
  fprintf(stderr, "Build(disk) done.\n");
}

template void ExecuteBuild(AllianceIndexDisk<int, float> &, const BuildInvocation &);
} // namespace alliance::disk_build
