#pragma once
#include "build/disk_build_config.hpp"
#include "build/disk_build_plan.hpp"
#include "build/disk_build_geometry.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace alliance {
template <typename att_t, typename vec_t> class AllianceIndexDisk;
}

namespace alliance::disk_build {

// Small non-owning view: no graph, order table, vectors, quantizer or IO resources.
// The caller owns the layer ladder for the duration of planning/reporting.
template <typename att_t, typename vec_t> struct MemoryPlanner : BuildConfig {
  using Index = AllianceIndexDisk<att_t, vec_t>;
  using Phase15Plan = disk_build::Phase15Plan;
  using Phase2Plan = disk_build::Phase2Plan;
  auto VecPageBytes() const -> size_t {
    return BuildVectorPageBytes(VecBytes());
  }
  explicit MemoryPlanner(const std::vector<size_t> &windows) : window_size_(windows) {
  }
  size_t vec_d_{};
  size_t M_{};
  size_t wp_{};
  size_t node_disk_bytes_{};
  size_t left_stream_locked_max_{SIZE_MAX};
  size_t measured_base_bytes_{};
  size_t measured_prep_residue_bytes_{};
  size_t cluster_K_file_{};

  const std::vector<size_t> &window_size_;
  auto VecBytes() const -> size_t {
    return vec_d_ * sizeof(vec_t);
  }
  auto EstimateBaseBuildBytes(size_t n) const -> size_t;
  auto PlanPhase15(size_t n) const -> Phase15Plan;
  auto PlanPhase2(size_t n) const -> Phase2Plan;
  // Evaluate a candidate without mutating the index or solving report-only K tables.
  auto PlanPartition(size_t n, size_t K, float balance) const -> Phase2Plan;
  void ReportBuildMemoryBudget(size_t n) const;

  // Defined only for index consumers in the private memory/index_adapter.hpp.
  static auto FromIndex(const Index &index) -> MemoryPlanner;

private:
  auto PlanUpperMemory(size_t n, bool include_report) const -> Phase2Plan;
};
} // namespace alliance::disk_build
