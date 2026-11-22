// Export stages: validate geometry, write neighbors/vectors, persist mappings.

#include "shuffle/layout.hpp"
#include "shuffle/format/write.hpp"
#include "shuffle/format/plan.hpp"
#include "index/index_disk.hpp"
#include "io/aio_stream.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <stdexcept>

namespace alliance {

template <typename att_t, typename vec_t>
auto ExportMultiLevelLayout(AllianceIndexDisk<att_t, vec_t> *idx, const LayoutExportConfig &config)
    -> LayoutExportResult {
  idx->RequireRawGraph("ExportMultiLevelLayout");
  const auto ranges =
      GenerateLevelRanges(static_cast<layer_t>(idx->cur_max_layer_), config.level_config);
  shuffle_detail::PlanLayout(idx->curvec_num_, idx->VecBytes(), idx->M_, sizeof(att_t), ranges,
                             config);
  auto level_results = ComputeMultiLevelShuffleOrders(idx, config.level_config, config.method);
  return ExportMultiLevelLayout(idx, config, level_results);
}

template <typename att_t, typename vec_t>
auto ExportMultiLevelLayout(AllianceIndexDisk<att_t, vec_t> *idx, const LayoutExportConfig &config,
                            const std::vector<LevelShuffleResult> &level_results)
    -> LayoutExportResult {
  idx->RequireRawGraph("ExportMultiLevelLayout");
  using clk = std::chrono::steady_clock;
  auto since = [](clk::time_point t0) {
    return std::chrono::duration<double>(clk::now() - t0).count();
  };
  shuffle_detail::LayoutIoStats lv_io, vec_io;
  const auto t_export0 = clk::now();

  const size_t n = idx->curvec_num_;
  const layer_t max_layer = static_cast<layer_t>(idx->cur_max_layer_);
  auto levels = GenerateLevelRanges(max_layer, config.level_config);
  const size_t num_levels = levels.size();
  if (level_results.size() != num_levels) {
    throw std::invalid_argument(
        "ExportMultiLevelLayout: precomputed shuffle results have a different level count; recompute them");
  }
  for (size_t li = 0; li < num_levels; ++li) {
    const auto &actual = level_results[li].range;
    const auto &expected = levels[li];
    if (actual.layer_begin != expected.layer_begin || actual.layer_end != expected.layer_end ||
        actual.owner_begin != expected.owner_begin || actual.owner_end != expected.owner_end ||
        level_results[li].order.size() != n) {
      throw std::invalid_argument(
          "ExportMultiLevelLayout: precomputed shuffle ownership/order mismatch at level " +
          std::to_string(li));
    }
    std::vector<bool> seen(n, false);
    for (const tableint id : level_results[li].order) {
      if (static_cast<size_t>(id) >= n || seen[id]) {
        throw std::invalid_argument("ExportMultiLevelLayout: shuffle order is not a permutation");
      }
      seen[id] = true;
    }
  }

  const auto plan =
      shuffle_detail::PlanLayout(n, idx->VecBytes(), idx->M_, sizeof(att_t), levels, config);
  auto result = plan.layout;
  for (auto &level : result.levels)
    level.disk_blocks.assign(n, -1);
  const size_t vector_block_size = result.vector_block_size;
  const bool mixed = result.layout_version == kMixedLayoutVersion;
  const size_t vector_start = result.vector_region_offset_bytes;
  const size_t graph_total_bytes = result.total_bytes;
  const size_t vec_region_start_block = result.vec_region_start_block;
  const size_t header_padding = plan.header_padding;
  const size_t neighbor_end = plan.neighbor_end;

  AioPreallocate(config.out_graph_path, graph_total_bytes);
  AioRegionWriter gw(config.out_graph_path, 0);

  // ---- Header + level meta entries (+ the colocated vector entry) ----
  GraphLayoutHeader header{};
  header.magic = kLayoutMagic;
  header.version = result.layout_version;
  header.max_elements = idx->max_elements_;
  header.vec_d = idx->vec_d_;
  header.wp = idx->wp_;
  header.M = idx->M_;
  header.curvec_num = n;
  header.cur_max_layer = idx->cur_max_layer_;
  header.block_size = config.block_size;
  header.num_levels = num_levels;
  header.vec_region_enabled = 1;
  header.layer_stride = LayerStride(config.level_config.shuffle_o);
  gw.Append(&header, sizeof(header));
  if (mixed) {
    MixedVectorGeometry geometry{vector_block_size, vector_start};
    gw.Append(&geometry, sizeof(geometry));
  }

  for (const auto &level : result.levels) {
    LevelHeaderEntry e{level.range.layer_begin, level.range.layer_end,   level.range.owner_begin,
                       level.range.owner_end,   level.record_size,       level.elements_per_block,
                       level.num_blocks,        level.region_start_block};
    gw.Append(&e, sizeof(e));
  }
  {
    VecHeaderEntry e{result.vec_elements_per_block, result.vec_region_num_blocks,
                     vec_region_start_block};
    gw.Append(&e, sizeof(e));
  }
  if (header_padding > 0) {
    std::vector<char> pad(header_padding, 0);
    gw.Append(pad.data(), pad.size());
  }

  const auto t_levels0 = clk::now();
  shuffle_detail::WriteLayoutLevels(idx, config, level_results, result, gw, lv_io);
  const double levels_s = since(t_levels0);

  if (vector_start > neighbor_end) {
    std::vector<char> pad(vector_start - neighbor_end, 0);
    gw.Append(pad.data(), pad.size());
  }
  // ---- Write the search-vector region ----
  const auto t_vec0 = clk::now();
  shuffle_detail::WriteLayoutVectors(idx, config, level_results, vec_region_start_block, result, gw,
                                     vec_io);
  const double vec_s = since(t_vec0);

  // Timed separately: Close drains the last write in flight and fdatasyncs, so
  // a large number here means the tail did not overlap, not that the loop above
  // was slow.
  const auto t_close0 = clk::now();
  gw.Close();
  const double graph_close_s = since(t_close0);
  if (gw.written() != graph_total_bytes) {
    throw std::runtime_error("ExportMultiLevelLayout: graph region byte count mismatch");
  }

  // Mapping metadata is completed only after the data writer drains and syncs.
  const auto t_meta0 = clk::now();
  const size_t meta_bytes = shuffle_detail::WriteLayoutSidecar(idx, config, result);
  const double meta_s = since(t_meta0);

  // Report stage and I/O time separately from CPU packing work.
  {
    const double total_s = since(t_export0);
    auto gb = [](size_t bytes) { return static_cast<double>(bytes) / 1073741824.0; };
    auto mbps = [](size_t bytes, double s) {
      return s > 0.0 ? static_cast<double>(bytes) / 1048576.0 / s : 0.0;
    };
    fprintf(stderr,
            "  [ml-export] total %.2fs = levels %.2fs + vec %.2fs + close %.2fs + meta %.2fs\n",
            total_s, levels_s, vec_s, graph_close_s, meta_s);
    fprintf(stderr,
            "  [ml-export]   levels: read %.2fs (%zu slices in %zu batches, %.2f GB, %.0f MB/s) "
            "write %.2fs (%zu blocks, %.2f GB, %.0f MB/s) other %.2fs\n",
            lv_io.read_s, lv_io.read_slices, lv_io.read_ops, gb(lv_io.read_bytes),
            mbps(lv_io.read_bytes, lv_io.read_s), lv_io.write_s, lv_io.write_ops,
            gb(lv_io.write_bytes), mbps(lv_io.write_bytes, lv_io.write_s),
            levels_s - lv_io.read_s - lv_io.write_s);
    fprintf(stderr,
            "  [ml-export]   vec:    read %.2fs (%zu slices in %zu batches, %.2f GB, %.0f MB/s) "
            "write %.2fs (%zu blocks, %.2f GB, %.0f MB/s) other %.2fs\n",
            vec_io.read_s, vec_io.read_slices, vec_io.read_ops, gb(vec_io.read_bytes),
            mbps(vec_io.read_bytes, vec_io.read_s), vec_io.write_s, vec_io.write_ops,
            gb(vec_io.write_bytes), mbps(vec_io.write_bytes, vec_io.write_s),
            vec_s - vec_io.read_s - vec_io.write_s);
    fprintf(stderr, "  [ml-export]   meta:   %.2fs (%.2f GB one-shot sequential, ofstream)\n",
            meta_s, gb(meta_bytes));
  }

  return result;
}

// ============================================================================
// Explicit template instantiation
// ============================================================================

template auto ExportMultiLevelLayout<int, float>(AllianceIndexDisk<int, float> *,
                                                 const LayoutExportConfig &) -> LayoutExportResult;
template auto ExportMultiLevelLayout<int, float>(AllianceIndexDisk<int, float> *,
                                                 const LayoutExportConfig &,
                                                 const std::vector<LevelShuffleResult> &)
    -> LayoutExportResult;

} // namespace alliance
