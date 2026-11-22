#pragma once

#include "disk_build/memory/budget.hpp"
#include "index/index_disk.hpp"

namespace alliance::disk_build {

// Adapt the owning index only at build call sites; pure planning has no index dependency.
template <typename att_t, typename vec_t>
auto MemoryPlanner<att_t, vec_t>::FromIndex(const Index &index) -> MemoryPlanner {
  MemoryPlanner plan(index.window_size_);
  static_cast<BuildConfig &>(plan) = index.build_config_;
  plan.vec_d_ = index.vec_d_;
  plan.M_ = index.M_;
  plan.wp_ = index.wp_;
  plan.node_disk_bytes_ = index.node_disk_bytes_;
  plan.left_stream_locked_max_ = index.left_stream_complete_ ? index.left_stream_max_ : SIZE_MAX;
  plan.measured_base_bytes_ = index.measured_base_bytes_;
  plan.measured_prep_residue_bytes_ = index.measured_prep_residue_bytes_;
  plan.cluster_K_file_ = index.cluster_K_file_;
  return plan;
}

} // namespace alliance::disk_build
