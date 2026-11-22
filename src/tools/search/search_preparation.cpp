#include "search/search_preparation.hpp"
#include "index/index_disk.hpp"
#include <iostream>

namespace benchmark {
void ConfigureSearchPolicy(alliance::DiskSearchPolicy &policy, const DiskSearchConfig &config,
                           const alliance::WorkerSearchIOBudget &io) {
  policy.top_layer_postfilter = config.top_layer_postfilter;
  policy.entry_samples = config.entry_samples;
  policy.entry_count = config.entry_count;
  policy.navigation_ef = config.navigation_ef;
  policy.navigation_budget = config.navigation_budget;
  policy.navigation_count = config.navigation_count;
  policy.entry_sampling =
      config.entry_sampling == "guided"  ? alliance::DiskSearchPolicy::EntrySampling::Guided
      : config.entry_sampling == "range" ? alliance::DiskSearchPolicy::EntrySampling::Range
                                         : alliance::DiskSearchPolicy::EntrySampling::Global;
  policy.edge_beam_width = policy.vec_beam_width = io.search;
  policy.rerank_beam_width = io.rerank;
  policy.rerank_num = config.rerank_num;
  policy.rerank_factor = config.rerank_factor;
  policy.rerank_policy = config.selected_rerank_policy;
  policy.rerank_error_bound = config.rerank_error_bound;
  policy.rerank_float_refine = config.rerank_float_refine;
  policy.rerank_reuse_estimates = config.rerank_reuse_estimates;
  policy.rerank_candidate_factor = config.rerank_candidate_factor;
  policy.rerank_bound_prescreen = config.rerank_bound_prescreen;
  policy.rerank_bound_rank_factor = config.rerank_bound_rank_factor;
  policy.rerank_warm_blocks = config.rerank_warm_blocks;
  policy.rerank_batch_blocks = std::min(config.rerank_batch_blocks, io.rerank);
  policy.filter_mode = config.selected_filter_mode;
  policy.prefilter_threshold = config.prefilter_threshold;
  policy.prefilter_rerank = config.prefilter_rerank;
}
void PrepareSearchCodes(PreparedSearchIndex &index, const DiskSearchConfig &config) {
  index.PrepareSearchCIDs();
  if (config.packed_rabitq)
    index.PackRaBitQForSearch();
}
void LoadSearchLayout(PreparedSearchIndex &index, const DiskSearchConfig &config) {
  const auto policy = config.ml_cache_policy == "lru"
                          ? alliance::DynamicBlockCache::EvictPolicy::LRU
                          : alliance::DynamicBlockCache::EvictPolicy::FIFO;
  index.LoadMultiLevelLayout(config.ml_graph_file, config.ml_meta_file, config.ml_cache_bytes,
                             policy);
}
void LoadNavigationForSearch(PreparedSearchIndex &index, const DiskSearchConfig &config) {
  if (!config.navigation_file.empty())
    index.LoadNavigation(config.navigation_file);
}
const char *SearchEntryMode(const DiskSearchConfig &config) {
  if (!config.navigation_file.empty())
    return "ram_navigation";
  if (!config.ml_pipeline_mode || !config.entry_samples || !config.entry_count)
    return "attribute_endpoints";
  return config.entry_sampling.c_str();
}
void PrintSearchEntryConfiguration(const PreparedSearchIndex &index, const DiskSearchConfig &config) {
  std::cout << "# entry_config: mode=" << SearchEntryMode(config)
            << " sampling=" << config.entry_sampling << " samples=" << config.entry_samples
            << " count=" << config.entry_count
            << " active=" << (config.ml_pipeline_mode && config.entry_samples && config.entry_count)
            << " guided_centroids=8 guided_entry_bytes=" << index.guided_entries_.Bytes()
            << " navigation_bytes=" << index.NavigationBytes() << " (graph route only)" << std::endl;
  if (index.HasNavigation())
    std::cout << "# navigation_config: file=" << config.navigation_file
              << " bytes=" << index.NavigationBytes() << " ef=" << config.navigation_ef
              << " budget=" << config.navigation_budget << " count=" << config.navigation_count
              << " original_entries=0 guided_entries=0 vectors_loaded=0" << std::endl;
}
void ConfigureSearchCache(PreparedSearchIndex &index, const DiskSearchConfig &config,
                          CachePreparation protocol) {
  index.per_query_cache_ = config.per_query_cache;
  index.ml_cache_auto_size_ =
      config.ml_cache_auto != 0 && (protocol == CachePreparation::Formal || config.per_query_cache);
  if (protocol == CachePreparation::Ordinary && config.ml_cache_auto && !config.per_query_cache)
    std::cerr << "NOTE: --ml_cache_auto ignored without --per_query_cache\n";
}
} // namespace benchmark
