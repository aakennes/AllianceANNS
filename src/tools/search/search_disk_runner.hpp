#pragma once
#include "index/index_disk.hpp"
#include "search/search_results.hpp"
#include <memory>
#include <vector>

namespace benchmark {
struct DiskSearchOptions;
using SearchIndex = alliance::AllianceIndexDisk<int, float>;
using SearchContexts = std::vector<std::unique_ptr<alliance::DiskSearchContext>>;

// Non-owning view of the query corpus; the driver owns vectors and filters.
struct SearchQueries {
  float *vectors;
  size_t dimension, count;
  const std::vector<alliance::attr_range<int>> &filters;
};

// Declare before contexts so their profiling observer outlives every handler.
struct SearchRunProfile {
#ifdef ALLIANCE_PIPELINE_PROFILE
  alliance::IOQueueObserver io_observer;
#endif
};

struct PipelineBatch {
  SearchResults results;
  std::vector<double> latency_ms;
  std::vector<alliance::MultiLevelThreadMemory> worker_memory;
  alliance::MultiLevelThreadMemory retained;
  double seconds = 0;
#ifdef ALLIANCE_PIPELINE_PROFILE
  std::vector<double> worker_enter_ms, worker_first_ms;
#endif
};

struct PassStats {
  float recall{-1.0f};
  float qps{0.0f};
  double dist_comps_avg{0.0};
  double hops_avg{0.0};
  double disk_reads_avg{0.0};
  double ml_block_reads_avg{0.0};
  double ml_cache_hit_rate{-1.0}; // -1 means "not applicable" (legacy pass)
};

PassStats RunSearchPass(SearchIndex &index, bool use_ml, const SearchQueries &queries, size_t efs,
                        size_t k, const SearchResults &gt);
PassStats RunLegacyPass(SearchIndex &index, const SearchQueries &queries, size_t efs,
                        const DiskSearchOptions &options, const SearchContexts &contexts,
                        const SearchResults &gt);
PipelineBatch RunPipelineBatch(SearchIndex &index, const SearchQueries &queries, size_t efs,
                               const DiskSearchOptions &options, const SearchContexts &contexts,
                               SearchRunProfile &profile);
} // namespace benchmark
