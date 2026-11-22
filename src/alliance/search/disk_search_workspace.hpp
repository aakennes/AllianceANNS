#pragma once
// Compatibility access for callers without an explicit worker context.
#include "search/search_worker.hpp"
#include "search/disk_search_context.hpp"

namespace alliance::detail {
using DiskSearchWorkspace = SearchWorker;
using MultiLevelVisited = alliance::MultiLevelVisited;

template <typename Index> DiskSearchWorkspace &GetSearchWorkspace(const Index *index) {
  return GetDiskSearchWorkspace(index, index->ml_cache_config_.generation);
}

template <typename Index>
DiskSearchWorkspace &GetSearchWorkspace(const Index *index, DiskSearchContext *context) {
  return context->Workspace(index, index->ml_cache_config_.generation);
}

template <typename Index>
DiskSearchWorkspace &GetGraphSearchWorkspace(const Index *index,
                                             DiskSearchContext *context = nullptr) {
  auto &workspace = context ? context->Workspace(index, index->ml_cache_config_.generation)
                            : GetSearchWorkspace(index);
  const auto &config = index->ml_cache_config_;
  // These flags can change after loading the layout. Auto growth requires
  // per-query Clear, so derive it on each graph entry, not at layout load.
  workspace.EnsureGraphCache(
      config.capacity_bytes, config.piece_bytes, config.policy, config.num_nodes,
      index->ml_cache_auto_size_ && index->per_query_cache_, index->ml_block_size_);
  return workspace;
}

} // namespace alliance::detail
