// Synchronous multi-level edge access and public worker-cache adapters.
#include "search/disk_search_cache.hpp"
#include "io/aio_batch.hpp"
#include <cstring>
#include <stdexcept>

namespace alliance {
using detail::DiskSearchWorkspace;
using detail::GetSearchWorkspace;
using detail::GetGraphSearchWorkspace;
using detail::DecodeBlockIntoCache;

template <typename att_t, typename vec_t>
void AllianceIndexDisk<att_t, vec_t>::ClearThreadLocalMultiLevelCache() const {
  auto &workspace = GetSearchWorkspace(this);
  workspace.ClearCache();
  PublishMultiLevelCacheFootprint(workspace.Cache());
}

template <typename att_t, typename vec_t>
void AllianceIndexDisk<att_t, vec_t>::ReadEdgesMultiLevel(tableint id, layer_t layer,
                                                          tableint *out_edges) const {
  ScopedMultiLevelMetrics metric_scope(*this);
  auto &metrics = metric_scope.Get();
  const size_t per_layer_id_bytes = (M_ + 1) * sizeof(tableint);
  const size_t record_header_bytes = sizeof(tableint) + sizeof(label_t) + sizeof(att_t);
  const size_t level_idx = ml_layer_to_level_[static_cast<size_t>(layer)];
  const size_t stride = ml_layer_stride_;
  auto e = StaticCacheLookup(id, layer);
  if (e.found) {
    ++metrics.metric_ml_static_cache_hits_;
    if (level_idx < kMaxMetricLevels)
      ++metrics.metric_ml_static_hits_per_level_[level_idx];
    const size_t layer_off = record_header_bytes + static_cast<size_t>(layer - e.layer_begin) /
                                                       stride * per_layer_id_bytes;
    std::memcpy(out_edges, e.data + layer_off, per_layer_id_bytes);
    return;
  }

  // Thread-local unified cache: lazily initialized, no mutex. A hit only
  // counts when the cached record's layer range covers the requested layer
  // (the record may have come from any level whose range includes it).
  DiskSearchWorkspace &tl = GetGraphSearchWorkspace(this);

  auto &cache = tl.Cache();
  e = cache.Lookup(id);
  if (e.found && layer >= e.layer_begin && layer <= e.layer_end) {
    ++metrics.metric_ml_cache_hits_;
    if (level_idx < kMaxMetricLevels)
      ++metrics.metric_ml_cache_hits_per_level_[level_idx];
    const size_t layer_off = record_header_bytes + static_cast<size_t>(layer - e.layer_begin) /
                                                       stride * per_layer_id_bytes;
    std::memcpy(out_edges, e.data + layer_off, per_layer_id_bytes);
    return;
  }

  ++metrics.metric_ml_cache_misses_;
  if (level_idx < kMaxMetricLevels)
    ++metrics.metric_ml_cache_misses_per_level_[level_idx];
  const layer_t level_lb = ml_levels_[level_idx].layer_begin;
  const int32_t block_id = ml_disk_blocks_per_level_[level_idx][id];
  const size_t record_size = ml_record_size_per_level_[level_idx];
  const layer_t level_le = ml_levels_[level_idx].layer_end;

  const uint64_t offset = static_cast<uint64_t>(ml_header_bytes_) +
                          static_cast<uint64_t>(block_id) * static_cast<uint64_t>(ml_block_size_);
  // This synchronous path reads one physical block per cache miss. AioBatchEngine
  // handles completion validation; concurrent read slots belong to graph traversal.
  TlsAioBatchEngine().ReadOne(ml_fd_, tl.BlockBuffer(), ml_block_size_, offset);
  ++metrics.metric_ml_block_reads_;
  if (level_idx < kMaxMetricLevels)
    ++metrics.metric_ml_block_reads_per_level_[level_idx];

  // Serve the request directly from the read buffer. Cache retention is not a
  // correctness condition: an in-place FIFO update can remain the oldest slot
  // and be evicted by subsequent co-block prefetch, even if a whole block fits.
  // This also allows reads when the configured cache has zero or one slot.
  const char *requested_record = nullptr;
  for (size_t roff = 0; roff + record_size <= ml_block_size_; roff += record_size) {
    tableint rec_id;
    std::memcpy(&rec_id, tl.BlockBuffer() + roff, sizeof(rec_id));
    if (rec_id == static_cast<tableint>(-1))
      break;
    if (rec_id == id) {
      requested_record = tl.BlockBuffer() + roff;
      break;
    }
  }
  if (!requested_record) {
    throw std::runtime_error("ReadEdgesMultiLevel: node " + std::to_string(id) +
                             " absent from its mapped physical block");
  }
  const size_t layer_off =
      record_header_bytes + static_cast<size_t>(layer - level_lb) / stride * per_layer_id_bytes;
  std::memcpy(out_edges, requested_record + layer_off, per_layer_id_bytes);

  const StaticNodeLookup *sm = static_cache_loaded_ ? &static_node_lookup_ : nullptr;
  DecodeBlockIntoCache(cache, tl.BlockBuffer(), record_size, ml_block_size_, level_lb, level_le,
                       &id, 1, nullptr, sm);
}

template auto AllianceIndexDisk<int, float>::StaticCacheLookup(tableint, layer_t) const
    -> DynamicBlockCache::Entry;
template void AllianceIndexDisk<int, float>::ClearThreadLocalMultiLevelCache() const;
template void AllianceIndexDisk<int, float>::ReadEdgesMultiLevel(tableint, layer_t,
                                                                 tableint *) const;

} // namespace alliance
