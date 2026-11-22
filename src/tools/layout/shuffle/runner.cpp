#include "layout/shuffle/options.hpp"
#include "index/index_disk.hpp"
#include "shuffle/layout.hpp"
#include "shuffle/format/plan.hpp"
#include "shuffle/stages/artifacts.hpp"
#include "shuffle/stages/verify.hpp"
#include "shuffle/memory/runtime.hpp"
#include "shuffle/memory/storage_plan.hpp"
#include "storage/disk_metadata.hpp"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>

namespace benchmark::shuffle {
int Run(const Options &options) {
  using Index = alliance::AllianceIndexDisk<int, float>;
  using Clock = std::chrono::steady_clock;
  alliance::shuffle_detail::ShuffleArtifacts artifacts(options.graph, options.meta, options.output,
                                                       options.output_meta);
  try {
    const auto start = Clock::now();
    // The artifact session holds a shared source-build lock and an exclusive
    // output-directory lock before either source or destination is opened.
    const size_t limit =
        alliance::shuffle_detail::EffectiveShuffleBudget(options.ordering.memory_limit_bytes);
    if (options.ordering.memory_limit_bytes && !limit)
      throw std::invalid_argument("effective shuffle memory budget is zero");
    if (limit) {
      // Inspect skips large payloads: reject impossible load/export fixed state
      // before the index constructor allocates metadata or quantization arrays.
      const auto meta = alliance::storage::ReadDiskMetadata<int>(
          artifacts.InputMetadata(), alliance::storage::MetadataReadMode::Inspect);
      const auto groups = alliance::GenerateLevelRanges(meta.max_layer, options.levels).size();
      const long double loaded =
          20.L * meta.count +
          alliance::SortedVectorOrderTable<int>::EntryBytes() * (long double)meta.max_elements +
          (meta.quantized
               ? (meta.code_bytes + 17.L) * meta.count + 4.L * meta.padded_dim * meta.centroid_count
               : 0.L);
      const long double fixed = alliance::shuffle_detail::ShuffleResidentBytes() + loaded +
                                (8.L * groups + 16.L) * meta.count +
                                alliance::shuffle_detail::ShuffleReserve(limit);
      if (fixed > limit)
        throw std::invalid_argument("shuffle load/export fixed state exceeds memory budget");
      std::cerr << "[shuffle-budget] requested_bytes=" << options.ordering.memory_limit_bytes
                << " effective_bytes=" << limit << " fixed_floor_bytes=" << uint64_t(fixed) << '\n';
    }
    auto index =
        std::make_unique<Index>(artifacts.InputGraph(), artifacts.InputMetadata(), options.space);
    index->build_config_.build_mem_gb_ = options.memory_observation_gib;
    index->ResetProcRssPeak();
    auto levels = options.levels;
    const auto ranges = alliance::GenerateLevelRanges(index->cur_max_layer_, levels);
    levels.per_level_params.assign(ranges.size(), options.ordering);
    for (auto &params : levels.per_level_params) {
      // Scratch shares the transaction's private SSD directory and lifetime.
      params.scratch_directory = std::filesystem::path(artifacts.Graph()).parent_path().string();
      params.memory_limit_bytes = limit;
    }
    std::cout << "method=" << alliance::ShuffleMethodName(options.method)
              << " iterations=" << options.ordering.max_iterations << " fixed_margin=1"
              << " shuffle_threads=" << options.ordering.threads
              << " affinity_batch_size=" << options.ordering.affinity_batch_size
              << " storage=" << alliance::ShuffleStorageName(options.ordering.storage)
              << " diagnostics=" << levels.collect_diagnostics << '\n';

    alliance::LayoutExportConfig config;
    config.block_size = options.ordering.block_size;
    config.vector_block_size = options.vector_block_size;
    config.level_config = levels;
    config.method = options.method;
    config.out_graph_path = artifacts.Graph();
    config.out_meta_path = artifacts.Layout();
    // Reject impossible geometry before the expensive all-node ordering pass.
    alliance::shuffle_detail::PlanLayout(index->curvec_num_, index->VecBytes(), index->M_,
                                         sizeof(int), ranges, config);
    // Ordering changes only physical placement; labels and internal edge IDs
    // retain their original meaning. Diagnostics do not select the ordering.
    auto orders = alliance::ComputeMultiLevelShuffleOrders(index.get(), levels, options.method);
    index->ReportPhaseRss("shuffle-orders", 0);
    auto layout = alliance::ExportMultiLevelLayout(index.get(), config, orders);
    index->ReportPhaseRss("shuffle-export", 0);

    // Metadata is not transformed by shuffle. Keep the exact source version,
    // including compatibility extensions, rather than reserializing it.
    std::filesystem::copy_file(artifacts.InputMetadata(), artifacts.Metadata());
    alliance::shuffle_detail::VerifyLayoutExport(*index, config, layout, orders);
    index->ReportPhaseRss("shuffle-readback", 0);
    const size_t nodes = index->curvec_num_, dimensions = index->vec_d_;
    const size_t level_count = layout.levels.size();
    const auto version = layout.layout_version;
    const size_t vector_page = layout.vector_block_size;
    orders.clear();
    orders.shrink_to_fit();
    layout = {}; // Release O(N) export mappings before validating the loader.
    index.reset();

    // Search readiness must be independent of the source raw graph. This
    // performs no search and prepares neither dynamic nor static cache data.
    auto search = std::make_unique<Index>("", artifacts.Metadata(), options.space,
                                          alliance::StorageOpenMode::MultiLevelOnly);
    search->LoadMultiLevelLayout(artifacts.Graph(), artifacts.Layout(), 0);
    if (!search->IsMultiLevelOnly() || search->curvec_num_ != nodes || search->vec_d_ != dimensions)
      throw std::runtime_error("ML-only acceptance geometry mismatch");
    search.reset();

    // Fixed-name graph/layout/meta are durable before the completion receipt
    // becomes visible. Existing output files are never replaced.
    artifacts.Commit(nodes, dimensions, level_count, version, options.ordering.block_size,
                     vector_page);
    std::cout << "Shuffle complete in "
              << std::chrono::duration<double>(Clock::now() - start).count()
              << " seconds; receipt=" << options.output << ".completion.json\n";
    return 0;
  } catch (const std::exception &error) {
    artifacts.Fail(error.what());
    throw;
  }
}
} // namespace benchmark::shuffle
