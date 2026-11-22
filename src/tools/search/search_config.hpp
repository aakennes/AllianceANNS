#pragma once
#include "search/disk_search_policy.hpp"
#include <cstddef>
#include <string>
#include <vector>

namespace benchmark {
// Value configuration only: no index, file handles, CLI tokens, or preparation.
struct DiskSearchConfig {
  std::string graph_file, meta_file, query_vec_path, query_rng_path, gt_file, space = "l2";
  std::string ml_graph_file, ml_meta_file, ml_cache_policy = "fifo";
  size_t ml_cache_bytes = 32 * 1024 * 1024;
  size_t ml_cache_total_bytes = 0;
  size_t k = 10;
  int use_pipeline = 0;
  size_t beam_width = 8;
  std::string rabitq_centroids, rabitq_cids; // override search-side RaBitQ centroids
  size_t rabitq_bits = 1;                    // re-encode bit width when requantizing (1/2/3/4/8)
  bool per_query_cache = false;
  bool packed_rabitq = true;
  bool top_layer_postfilter = true;
  size_t entry_samples = 256, entry_count = 4;
  std::string entry_sampling = "navigation";
  // ML graph/auto defaults to navigation.bin beside graph_file (or ml_graph_file
  // for ML-only inputs). Explicit guided/global/range retains rank entries.
  std::string navigation_file;
  size_t navigation_ef = 64, navigation_budget = 256, navigation_count = 16;
  int ml_cache_auto = 1; // size the dynamic cache to one query's working set
  size_t rerank_num = 0;
  size_t rerank_factor = 0;
  std::string rerank_policy = "error_bound";
  bool rerank_float_refine = true;
  bool rerank_reuse_estimates = false;
  using GraphRerankPolicy = alliance::DiskSearchPolicy::GraphRerankPolicy;
  GraphRerankPolicy selected_rerank_policy = GraphRerankPolicy::Fixed;
  bool rerank_error_bound = false;
  size_t rerank_batch_blocks = 32;
  size_t rerank_candidate_factor = 0;
  bool rerank_bound_prescreen = false;
  size_t rerank_bound_rank_factor = 1;
  size_t rerank_warm_blocks = 0;
  size_t single_efs = 0;
  size_t query_limit = 0;
  std::string quant_checkpoint;
  std::vector<size_t> efs_list;
  double recall_stop = 0;
  std::string filter_mode = "legacy";
  size_t prefilter_threshold = 8192;
  size_t prefilter_rerank = 0;
  size_t search_threads = 1;
  size_t io_budget_total = 0;
  size_t rerank_beam_width = 8;

  size_t process_memory_budget_bytes = 0;
  bool has_ml = false, ml_pipeline_mode = false, compare_mode = false, ml_only = false;
  using FilterMode = alliance::DiskSearchPolicy::FilterMode;
  FilterMode selected_filter_mode = FilterMode::Legacy;
};
} // namespace benchmark
