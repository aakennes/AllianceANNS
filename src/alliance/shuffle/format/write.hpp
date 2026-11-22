#pragma once
#include "shuffle/layout.hpp"
#include "io/aio_stream.hpp"

namespace alliance::shuffle_detail {

struct LayoutIoStats {
  double read_s{0}, write_s{0};
  size_t read_ops{0}, read_slices{0}, read_bytes{0}, write_ops{0}, write_bytes{0};
};
template <typename att_t, typename vec_t>
void WriteLayoutLevels(AllianceIndexDisk<att_t, vec_t> *idx, const LayoutExportConfig &config,
                       const std::vector<LevelShuffleResult> &level_results,
                       LayoutExportResult &result, AioRegionWriter &gw, LayoutIoStats &stats);
template <typename att_t, typename vec_t>
void WriteLayoutVectors(AllianceIndexDisk<att_t, vec_t> *idx, const LayoutExportConfig &config,
                        const std::vector<LevelShuffleResult> &level_results,
                        size_t vec_region_start_block, LayoutExportResult &result,
                        AioRegionWriter &writer, LayoutIoStats &stats);
template <typename att_t, typename vec_t>
size_t WriteLayoutSidecar(AllianceIndexDisk<att_t, vec_t> *idx, const LayoutExportConfig &config,
                          const LayoutExportResult &result);

} // namespace alliance::shuffle_detail
