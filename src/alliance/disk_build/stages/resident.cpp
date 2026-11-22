#include "disk_build/stages/internal.hpp"
#include "disk_build/stages/checkpoint.hpp"
#include "index/index_disk.hpp"
#include <cstdio>
#include <stdexcept>

namespace alliance::disk_build {

template <typename att_t, typename vec_t>
ResidentStageResult RunResidentStage(AllianceIndexDisk<att_t, vec_t> &index, size_t n,
                                     size_t start_layer, const std::string &checkpoint_meta_path) {
  // The resident plan retains the recorded boundary on resume. Fresh builds
  // reverse-solve it from RAM unless the user supplied an explicit threshold.
  const Phase15Plan plan = index.PlanPhase15(n);
  const layer_t resident_max = plan.wow_max;
  const size_t model_bytes = plan.base_bytes + plan.residency_bytes + plan.thread_scratch_bytes;
  if (index.left_stream_complete_)
    return {resident_max, model_bytes, false};

  if (index.build_config_.build_mem_gb_ > 0.0) {
    if (plan.base_over_budget)
      throw std::runtime_error("ExecuteBuild: BASE alone exceeds --build_mem_gb; raise the budget");
    if (plan.user_over_budget) {
      fprintf(stderr,
              "  ERROR: --wow_thresh=%zu exceeds the RAM budget (derived max=%zu nodes).\n"
              "         Drop --wow_thresh (auto-derive) or raise --build_mem_gb.\n",
              index.build_config_.wow_threshold_, plan.derived_threshold);
      throw std::runtime_error("ExecuteBuild: --wow_thresh over --build_mem_gb");
    }
    if (resident_max == 0 && index.wp_ > 0) {
      fprintf(stderr, "  ERROR: budget %.2f GB cannot fit even layer 1; need >= %.2f GB.\n",
              index.build_config_.build_mem_gb_, plan.min_budget_gb_l1);
      throw std::runtime_error("ExecuteBuild: --build_mem_gb too small for layer 1");
    }
  }
  if (start_layer > 1)
    throw std::runtime_error("left_stream must start from fresh vectors");
  if (index.build_config_.build_mem_gb_ > 0.0 &&
      model_bytes > size_t(index.build_config_.build_mem_gb_ * 1073741824.0))
    throw std::runtime_error("left_stream working set exceeds build_mem_gb");

  // A failed executor leaves an explicitly incomplete checkpoint. The kernel
  // returns only after all records are synced and its coverage checks pass.
  BeginResidentCheckpoint(index, resident_max, checkpoint_meta_path);
  ExecuteLeftStreamRecords(index, resident_max, n);
  CommitResidentCheckpoint(index, resident_max, n, checkpoint_meta_path);
  return {resident_max, model_bytes, true};
}

template ResidentStageResult RunResidentStage(AllianceIndexDisk<int, float> &, size_t, size_t,
                                              const std::string &);
} // namespace alliance::disk_build

namespace alliance {

// Keep the direct library entry's completion contract, without saving metadata
// or advancing the staged resume cursor. Both entries share the same executor.
template <typename att_t, typename vec_t>
void AllianceIndexDisk<att_t, vec_t>::BuildLeftStreamLayers(layer_t maximum, size_t n) {
  if (n == 0)
    return;
  disk_build::ExecuteLeftStreamRecords(*this, maximum, n);
  disk_build::PublishResidentRecords(*this, maximum);
}

template void AllianceIndexDisk<int, float>::BuildLeftStreamLayers(layer_t, size_t);
} // namespace alliance
