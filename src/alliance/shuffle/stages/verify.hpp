#pragma once
#include "shuffle/layout.hpp"

namespace alliance::shuffle_detail {
// Full sequential readback, with bounded batched reads from the source graph.
// No queries, cache preparation, or source mutation occur during acceptance.
void VerifyLayoutExport(AllianceIndexDisk<int, float> &source, const LayoutExportConfig &config,
                        const LayoutExportResult &layout,
                        const std::vector<LevelShuffleResult> &orders);
} // namespace alliance::shuffle_detail
