#include "search/disk_search_workspace.hpp"
#include "storage/paged_file.hpp"
#include <stdexcept>

namespace alliance {

SearchWorker::Buffer SearchWorker::AllocateBuffer(size_t bytes, const char *failure_message) {
  void *buffer = nullptr;
  if (posix_memalign(&buffer, kDiskPageSize, bytes) != 0)
    throw std::runtime_error(failure_message);
  return Buffer(static_cast<char *>(buffer));
}

void SearchWorker::EnsureBlockBuf(size_t bytes) {
  if (bytes <= block_buf_bytes_)
    return;
  auto replacement = AllocateBuffer(bytes, "posix_memalign failed for search worker block buffer");
  block_buf_.swap(replacement);
  block_buf_bytes_ = bytes;
}

void SearchWorker::EnsurePipelineBuffers(size_t count, size_t block_size) {
  if (count <= pipeline_bufs_.size() && block_size <= pipeline_buf_block_size_)
    return;
  // Reallocate to the requested dimensions, retaining pointer-table capacity.
  // Count may shrink when block size grows; replace only after all allocations succeed.
  std::vector<Buffer> replacement;
  replacement.reserve(pipeline_bufs_.capacity());
  replacement.resize(pipeline_bufs_.size());
  replacement.resize(count);
  for (auto &buffer : replacement)
    buffer = AllocateBuffer(block_size, "posix_memalign failed for pipeline buffer pool");
  pipeline_bufs_.swap(replacement);
  pipeline_buf_block_size_ = block_size;
}

void SearchWorker::Bind(const void *owner, uint64_t generation) {
  if (owner_ == owner && generation_ == generation)
    return;
  cache_.Configure(0, 0, DynamicBlockCache::EvictPolicy::FIFO, 0);
  visited_.reset();
  visited_capacity_ = 0;
  block_buf_.reset();
  block_buf_bytes_ = 0;
  graph_cache_initialized_ = false;
  owner_ = owner;
  generation_ = generation;
}

void SearchWorker::EnsureGraphCache(size_t capacity, size_t piece_bytes,
                                    DynamicBlockCache::EvictPolicy policy, size_t num_nodes,
                                    bool auto_size, size_t block_size) {
  if (graph_cache_initialized_ && cache_.auto_size() == auto_size)
    return;
  graph_cache_initialized_ = false;
  cache_.Configure(capacity, piece_bytes, policy, num_nodes, auto_size);
  EnsureBlockBuf(block_size);
  graph_cache_initialized_ = true;
}

MultiLevelVisited &SearchWorker::GetVisited(size_t num_elements) {
  if (!visited_ || visited_capacity_ < num_elements) {
    visited_ = std::make_unique<MultiLevelVisited>(static_cast<int>(num_elements));
    visited_capacity_ = num_elements;
  }
  visited_->Clear();
  return *visited_;
}

MultiLevelThreadMemory SearchWorker::MemoryUsage() const {
  MultiLevelThreadMemory m;
  m.cache_payload_bytes = cache_.payload_bytes();
  m.cache_lookup_bytes = cache_.lookup_bytes();
  m.cache_metadata_bytes = cache_.resident_bytes() - m.cache_payload_bytes - m.cache_lookup_bytes;
#ifdef ALLIANCE_STAMP_VISITED
  m.visited_bytes = visited_ ? ((visited_capacity_ * sizeof(vl_type) + 63) & ~size_t(63)) : 0;
#else
  m.visited_bytes = visited_ ? visited_->Bytes() : 0;
#endif
  m.io_buffer_bytes = block_buf_bytes_ + pipeline_bufs_.capacity() * sizeof(Buffer);
  for (const auto &buffer : pipeline_bufs_)
    if (buffer)
      m.io_buffer_bytes += pipeline_buf_block_size_;
  m.prefilter_scratch_bytes = prefilter_ids_.capacity() * sizeof(tableint) +
                              prefilter_candidates_.capacity() * sizeof(RerankCandidatePool::Entry);
  return m;
}

namespace detail {
SearchWorker &GetDiskSearchWorkspace(const void *owner, uint64_t generation) {
  thread_local SearchWorker workspace;
  workspace.Bind(owner, generation);
  return workspace;
}

} // namespace detail
} // namespace alliance
