#pragma once
#include <cstddef>

namespace alliance {

// Retained user-space allocations owned by the calling worker. This is not
// process RSS: transient query allocations and kernel AIO memory are excluded.
struct MultiLevelThreadMemory {
  size_t cache_payload_bytes{0}, cache_lookup_bytes{0}, cache_metadata_bytes{0};
  size_t visited_bytes{0}, io_buffer_bytes{0}, prefilter_scratch_bytes{0};
  size_t Total() const {
    return cache_payload_bytes + cache_lookup_bytes + cache_metadata_bytes + visited_bytes +
           io_buffer_bytes + prefilter_scratch_bytes;
  }
  MultiLevelThreadMemory &operator+=(const MultiLevelThreadMemory &m) {
    cache_payload_bytes += m.cache_payload_bytes;
    cache_lookup_bytes += m.cache_lookup_bytes;
    cache_metadata_bytes += m.cache_metadata_bytes;
    visited_bytes += m.visited_bytes;
    io_buffer_bytes += m.io_buffer_bytes;
    prefilter_scratch_bytes += m.prefilter_scratch_bytes;
    return *this;
  }
};

} // namespace alliance
