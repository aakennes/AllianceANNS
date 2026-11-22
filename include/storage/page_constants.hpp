#pragma once

#include <cstddef>

namespace alliance {

constexpr size_t kDiskPageSize = 8192;

/// Smallest legal `node_disk_bytes`: one physical page, which is also the
/// O_DIRECT read/write granularity.
constexpr size_t kMinNodePageBytes = 4096;

} // namespace alliance
