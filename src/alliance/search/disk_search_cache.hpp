#pragma once
// Internal cache access shared by synchronous edge reads and async traversal.
// Header definitions preserve inlining on the traversal hot path. No storage
// or TLS is owned here; SearchWorker owns storage through a context or TLS fallback.
#include "index/index_disk.hpp"
#include "search/disk_search_workspace.hpp"
#include <cstring>

namespace alliance {

template <typename att_t, typename vec_t>
auto AllianceIndexDisk<att_t, vec_t>::StaticCacheLookup(tableint id, layer_t layer) const
    -> DynamicBlockCache::Entry {
  DynamicBlockCache::Entry e;
  e.found = false;
  if (!static_cache_loaded_ || static_cache_buf_ == nullptr)
    return e;
  if (static_node_lookup_.Wide()) {
    const auto *m = static_node_lookup_.Find64(id);
    if (!m || !m->Covers(layer))
      return e;
    e.found = true;
    e.data = static_cache_buf_ + m->offset;
    e.layer_begin = m->layer_begin;
    e.layer_end = m->layer_end;
    return e;
  }
  const auto *meta = static_node_lookup_.Find(id);
  if (!meta || !meta->Covers(layer))
    return e;
  const auto &m = *meta;
  e.found = true;
  e.data = static_cache_buf_ + m.offset;
  e.layer_begin = m.layer_begin;
  e.layer_end = m.layer_end;
  return e;
}

namespace detail {

// Decodes every record in a freshly-read physical block into `cache`. Shared
// by the synchronous miss path and the async pipeline's IO-completion path.
// Records whose node is listed in `must_update` (the IO's requester and its
// dedup waiters -- they paused precisely because their cached record does NOT
// cover this block's level) overwrite the node's existing slot, otherwise the
// requester would pause -> re-read forever. Every other record is pure
// prefetch and uses InsertIfMissing: it does not replace an existing record for
// that same node at another level. Normal capacity eviction still applies;
// decoded records are not pinned by this helper.
//
// When `visited` is non-null (pipeline path), it denotes discovery, not
// completed expansion. Unrequested records for already discovered nodes are
// skipped as prefetch; their existing cache entries remain untouched, and
// their own pending IO always updates them through `must_update` above.
//
// When `static_meta` is non-null (immutable compact lookup), nodes whose static cache
// range fully covers [level_lb, level_le] are skipped for prefetch -- their
// records are always available via static cache lookup.
inline void DecodeBlockIntoCache(DynamicBlockCache &cache, const char *block_buf,
                                 size_t record_size, size_t block_size, layer_t level_lb,
                                 layer_t level_le, const tableint *must_update,
                                 size_t must_update_count,
                                 const MultiLevelVisited *visited = nullptr,
                                 const StaticNodeLookup *static_meta = nullptr) {
  for (size_t roff = 0; roff + record_size <= block_size; roff += record_size) {
    tableint rec_id;
    std::memcpy(&rec_id, block_buf + roff, sizeof(tableint));
    if (rec_id == static_cast<tableint>(-1))
      break;
    bool update = false;
    for (size_t i = 0; i < must_update_count; ++i) {
      if (must_update[i] == rec_id) {
        update = true;
        break;
      }
    }
    if (update) {
      cache.InsertOrUpdate(rec_id, block_buf + roff, record_size, level_lb, level_le);
    } else {
      if (visited && visited->Test(rec_id))
        continue;
      if (static_meta) {
        if (static_meta->CoversRange(rec_id, level_lb, level_le))
          continue;
      }
      cache.InsertIfMissing(rec_id, block_buf + roff, record_size, level_lb, level_le);
    }
  }
}

} // namespace detail
} // namespace alliance
