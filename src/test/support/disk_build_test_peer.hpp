#pragma once
#include "index/index_disk.hpp"
#include "disk_build/cluster_wow/passes.hpp"

namespace alliance {
struct DiskBuildTestPeer {
  // Test-only stage access keeps fault injection away from unrelated earlier IO.
  // The existing friend boundary exposes no new production API.
  template <typename Att, typename Vec>
  static void BuildClusterPass(AllianceIndexDisk<Att, Vec> &index,
                               const disk_build::ClusterSpillPlan &plan) {
    index.BuildClusterWoWEdgesDisk(plan);
  }
  template <typename Att, typename Vec>
  static void CommitClusterPass(AllianceIndexDisk<Att, Vec> &index,
                                const disk_build::ClusterSpillPlan &plan) {
    index.CommitClusterEdgesDisk(plan);
  }
  template <typename Att, typename Vec>
  static bool HasWorkspace(const AllianceIndexDisk<Att, Vec> &index) noexcept {
    return bool(index.build_state_);
  }
};
} // namespace alliance
