#pragma once

namespace alliance {
// Missing raw storage is never inferred. ML-only callers explicitly accept
// only the complete shuffled neighbor/vector layout and ML search routes.
enum class StorageOpenMode { RequireRawGraph, MultiLevelOnly };
} // namespace alliance
