#include "search/search_disk_runner.hpp"
#include "search/search_disk_options.hpp"
#include "search/search_disk_report.hpp"
#include <atomic>
#include <chrono>
#include <exception>
#include <omp.h>

namespace benchmark {
namespace {
void ResetLegacyMetrics(SearchIndex &index) {
  index.metric_dist_comps_ = 0;
  index.metric_hops_ = 0;
  index.metric_disk_reads_ = 0;
}

struct SerialBatch {
  SearchResults results;
  float seconds = 0;
};

// Shared by the compare and legacy disk drivers. Keep their historical
// float accumulation and per-query timer (the parallel path uses wall time).
template <typename Search> SerialBatch RunSerialBatch(size_t nq, Search search) {
  SerialBatch batch;
  batch.results.resize(nq);
  for (size_t i = 0; i < nq; ++i) {
    const auto start = std::chrono::high_resolution_clock::now();
    auto result = search(i);
    const auto end = std::chrono::high_resolution_clock::now();
    batch.seconds += std::chrono::duration<float>(end - start).count();
    AppendResultLabels(result, batch.results[i]);
  }
  return batch;
}

PassStats SummarizeSerialPass(const SearchIndex &index, const SerialBatch &batch,
                              const SearchResults &gt) {
  const size_t nq = batch.results.size();
  PassStats stats;
  if (!gt.empty())
    stats.recall = CalculateRecall(gt, batch.results);
  stats.qps = nq / batch.seconds;
  stats.dist_comps_avg = index.metric_dist_comps_ * 1.0 / nq;
  stats.hops_avg = index.metric_hops_ * 1.0 / nq;
  stats.disk_reads_avg = index.metric_disk_reads_ * 1.0 / nq;
  return stats;
}
} // namespace

PassStats RunSearchPass(SearchIndex &index, bool use_ml, const SearchQueries &queries, size_t efs,
                        size_t k, const SearchResults &gt) {
  index.ml_layout_loaded_ = use_ml;
  ResetLegacyMetrics(index);
  if (use_ml) {
    index.metric_ml_block_reads_ = 0;
    index.metric_ml_cache_hits_ = 0;
    index.metric_ml_cache_misses_ = 0;
  }
  auto batch = RunSerialBatch(queries.count, [&](size_t i) {
    return index.searchKNN(queries.vectors + i * queries.dimension, efs, k, queries.filters[i]);
  });
  auto stats = SummarizeSerialPass(index, batch, gt);
  if (use_ml) {
    stats.ml_block_reads_avg = index.metric_ml_block_reads_ * 1.0 / queries.count;
    size_t hits = index.metric_ml_cache_hits_;
    size_t misses = index.metric_ml_cache_misses_;
    stats.ml_cache_hit_rate =
        (hits + misses > 0) ? static_cast<float>(hits) / static_cast<float>(hits + misses) : 0.0f;
  }
  return stats;
}

PassStats RunLegacyPass(SearchIndex &index, const SearchQueries &queries, size_t efs,
                        const DiskSearchOptions &options, const SearchContexts &contexts,
                        const SearchResults &gt) {
  ResetLegacyMetrics(index);
  auto batch = RunSerialBatch(queries.count, [&](size_t i) {
    auto *query = queries.vectors + i * queries.dimension;
    const auto &filter = queries.filters[i];
    std::vector<std::pair<alliance::dist_t, alliance::label_t>> result;
    if (options.use_pipeline) {
      result = index.searchKNNPipeline(query, efs, options.k, filter, contexts.front().get());
    } else {
      result = index.searchKNN(query, efs, options.k, filter);
    }
    return result;
  });
  return SummarizeSerialPass(index, batch, gt);
}

PipelineBatch RunPipelineBatch(SearchIndex &index, const SearchQueries &queries, size_t efs,
                               const DiskSearchOptions &options,
                               const SearchContexts &search_contexts, SearchRunProfile &profile) {
  const size_t nq = queries.count, d = queries.dimension;
  auto *query_vecs = queries.vectors;
  const auto &query_filters = queries.filters;
  PipelineBatch batch;
  batch.results.resize(nq);
  batch.latency_ms.assign(nq, 0.0);
  auto &results = batch.results;
  auto &latency_ms = batch.latency_ms;
  alliance::ResetMultiLevelMetrics(index);
  index.metric_ml_cache_resident_bytes_ = 0;
  index.metric_ml_cache_peak_slots_ = 0;
  index.metric_ml_cache_peak_wanted_ = 0;

  std::atomic<size_t> next_query{0};
  std::atomic<bool> failed{false};
  std::vector<std::exception_ptr> errors(options.search_threads);
  batch.worker_memory.resize(options.search_threads);
  auto &worker_memory = batch.worker_memory;
#ifdef ALLIANCE_PIPELINE_PROFILE
  batch.worker_enter_ms.resize(options.search_threads);
  batch.worker_first_ms.assign(options.search_threads, -1.0);
  auto &worker_enter_ms = batch.worker_enter_ms;
  auto &worker_first_ms = batch.worker_first_ms;
  auto &io_observer = profile.io_observer;
  io_observer.Reset();
  for (const auto &ctx : search_contexts) {
    if (ctx->io_handler)
      ctx->io_handler->ResetQueueProfile(&io_observer);
    if (ctx->ml_io_handler)
      ctx->ml_io_handler->ResetQueueProfile(&io_observer);
  }
#endif
  PrintMemoryPhase("search_start", options.process_memory_budget_bytes);
  const auto batch_start = std::chrono::steady_clock::now();
#pragma omp parallel num_threads(static_cast<int>(options.search_threads))
  {
    alliance::ScopedMultiLevelMetrics metric_batch(index);
    const size_t tid = static_cast<size_t>(omp_get_thread_num());
#ifdef ALLIANCE_PIPELINE_PROFILE
    worker_enter_ms[tid] =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - batch_start)
            .count();
#endif
    while (!failed.load(std::memory_order_relaxed)) {
      const size_t i = next_query.fetch_add(1, std::memory_order_relaxed);
      if (i >= nq)
        break;
      try {
        const auto start = std::chrono::steady_clock::now();
        auto result = index.searchKNNMultiLevelPipeline(
            query_vecs + i * d, efs, options.k, query_filters[i], search_contexts[tid].get());
        latency_ms[i] =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count();
#ifdef ALLIANCE_PIPELINE_PROFILE
        if (worker_first_ms[tid] < 0)
          worker_first_ms[tid] = latency_ms[i];
#endif
        AppendResultLabels(result, results[i]);
      } catch (...) {
        errors[tid] = std::current_exception();
        failed.store(true, std::memory_order_relaxed);
        break;
      }
    }
    worker_memory[tid] =
        search_contexts[tid]->WorkspaceMemory(&index, index.ml_cache_config_.generation);
  }
  batch.seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - batch_start).count();
  PrintMemoryPhase("search_end", options.process_memory_budget_bytes);
  for (const auto &m : worker_memory)
    batch.retained += m;
  PrintWorkerMemory(batch.retained);
  for (const auto &error : errors)
    if (error)
      std::rethrow_exception(error);
  return batch;
}
} // namespace benchmark
