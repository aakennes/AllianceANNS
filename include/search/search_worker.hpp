#pragma once
// Per-worker mutable search storage; never shared by concurrent queries.
#include "cache/dynamic_block_cache.hpp"
#include "search/query_visited.hpp"
#include "core/visit_list.hpp"
#include "search/search_memory.hpp"
#include "search/rerank_candidate_pool.hpp"
#include <cstdlib>
#include <memory>
#include <vector>

namespace alliance {

#ifdef ALLIANCE_STAMP_VISITED
using MultiLevelVisited = VisitedList<tableint>;
#else
using MultiLevelVisited = QueryVisited;
#endif

class SearchWorker {
public:
  SearchWorker() = default;
  SearchWorker(const SearchWorker &) = delete;
  SearchWorker &operator=(const SearchWorker &) = delete;

  // Rebinding invalidates graph state but retains pipeline/prefilter capacities.
  // Call only between batches, with no submitted IO still using this storage.
  void Bind(const void *owner, uint64_t generation);
  void EnsureGraphCache(size_t capacity, size_t piece_bytes, DynamicBlockCache::EvictPolicy policy,
                        size_t num_nodes, bool auto_size, size_t block_size);
  void ClearCache() {
    cache_.Clear();
  }
  DynamicBlockCache &Cache() {
    return cache_;
  }
  std::vector<tableint> &PrefilterIDs() {
    return prefilter_ids_;
  }
  std::vector<RerankCandidatePool::Entry> &PrefilterCandidates() {
    return prefilter_candidates_;
  }
  MultiLevelVisited &GetVisited(size_t num_elements);
  MultiLevelThreadMemory MemoryUsage() const;

  // The caller owns IORequest lifetimes and completion/draining. Ensure must
  // precede publishing buffer pointers to slots or submitting their IO. A
  // request within the current dimensions is a no-op and preserves pointers.
  // On allocation failure, the old buffers and dimensions remain usable.
  void EnsureBlockBuf(size_t bytes);
  void EnsurePipelineBuffers(size_t count, size_t block_size);
  char *BlockBuffer() const {
    return block_buf_.get();
  }
  char *PipelineBuffer(size_t slot) const {
    return pipeline_bufs_[slot].get();
  }

private:
  struct FreeBuffer {
    void operator()(char *p) const {
      std::free(p);
    }
  };
  using Buffer = std::unique_ptr<char, FreeBuffer>;
  static_assert(sizeof(Buffer) == sizeof(char *), "retain pointer-table accounting");
  static Buffer AllocateBuffer(size_t bytes, const char *failure_message);

  DynamicBlockCache cache_;
  std::vector<tableint> prefilter_ids_;
  std::vector<RerankCandidatePool::Entry> prefilter_candidates_;
  Buffer block_buf_;
  size_t block_buf_bytes_{0};
  uint64_t generation_{0};
  const void *owner_{nullptr};
  bool graph_cache_initialized_{false};
  std::vector<Buffer> pipeline_bufs_;
  size_t pipeline_buf_block_size_{0};
  std::unique_ptr<MultiLevelVisited> visited_;
  size_t visited_capacity_{0};
};

namespace detail {
SearchWorker &GetDiskSearchWorkspace(const void *owner, uint64_t generation);
}

} // namespace alliance
