// Query lifecycle only: validate -> plan -> execute route -> rerank -> finalize.
// Private helpers own each route; traversal and rerank retain their IO lifetimes.
#if defined(ALLIANCE_PIPELINE_DIAGNOSTICS) && !defined(ALLIANCE_PIPELINE_PROFILE)
#define ALLIANCE_PIPELINE_PROFILE
#endif

#include "index/index_disk.hpp"
#include "search/query/graph.hpp"
#include "search/query/plan.hpp"
#include "search/query/prefilter.hpp"
#include "search/query/results.hpp"
#include <chrono>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>

namespace alliance {

template <typename att_t, typename vec_t>
template <typename filter_t>
auto AllianceIndexDisk<att_t, vec_t>::searchKNNMultiLevelPipeline(vec_t *query_vec, size_t efs,
                                                                  size_t k, const filter_t &filter,
                                                                  DiskSearchContext *ctx)
    -> std::vector<std::pair<dist_t, label_t>> {
  query_detail::ValidateStorage(*this, ctx);
  ScopedMultiLevelMetrics metric_scope(*this);
  auto &metrics = metric_scope.Get();
#ifdef ALLIANCE_PIPELINE_PROFILE
  QueryPhaseTimer phase_timer(metrics.metric_query_total_ns_, metrics.metric_query_prepare_ns_);
#endif
  // One query-owned timer accounts for exclusive phases. Helpers switch it at
  // route boundaries; disabled profiling adds no runtime dispatch.
  auto enter_phase = [&](size_t &phase) {
#ifdef ALLIANCE_PIPELINE_PROFILE
    phase_timer.SwitchTo(phase);
#endif
  };

  if (!rabitq_initialized_)
    throw std::runtime_error("searchKNNMultiLevelPipeline requires InitRaBitQ to be called first");
  if (k == 0)
    return {};
  query_detail::ValidateBoundedRerank(*this, *ctx);
  ValidateFilterPolicy(*ctx, k);

  const auto plan = query_detail::PlanQuery<att_t>(*this, filter, *ctx);
  if constexpr (std::is_same_v<filter_t, attr_range<att_t>>) {
    if (plan.route == RangeSearchRoute::Empty) {
      ++metrics.metric_filter_empty_queries_;
      return {};
    }
    if (plan.route == RangeSearchRoute::Prefilter)
      return query_detail::ExecutePrefilter(*this, query_vec, efs, k, plan, ctx, metrics,
                                            enter_phase);
    // This raw-vector fallback is available only through explicit Legacy mode.
    if (plan.route == RangeSearchRoute::LegacyExact)
      return BruteForceSearch(query_vec, k, filter);
  }

  const size_t rerank_budget = ctx->GraphRerankBudgetForSearch(k, efs, rerank_times_upper_bound_);
  ++metrics.metric_filter_graph_queries_;
  RaBitQEstimator estimator(query_vec, this);
  auto entry = query_detail::PrepareGraphEntry<att_t>(*this, filter, estimator, *ctx, metrics);

  enter_phase(metrics.metric_query_traversal_ns_);
  auto candidates = query_detail::TraverseGraph<att_t>(
      *this, entry, query_vec, filter, plan.cardinality, efs, k, ctx, estimator, metrics);

  enter_phase(metrics.metric_query_finalize_ns_);
  // Deduplication storage remains live through rerank. Its destruction belongs
  // to the final phase, including the retained allocation in the query peak.
  std::unordered_set<tableint> unique_ids;
  query_detail::FinalizeGraphCandidates(candidates, unique_ids, rerank_budget, metrics);
  query_detail::QueryResults results;

  enter_phase(metrics.metric_query_rerank_ns_);
  {
#ifdef ALLIANCE_PIPELINE_PROFILE
    const auto rerank_begin = std::chrono::steady_clock::now();
#endif
    if (!candidates.empty())
      query_detail::RerankCandidates(*this, query_vec, candidates, k, estimator, ctx, results);
#ifdef ALLIANCE_PIPELINE_PROFILE
    metrics.metric_ml_rerank_ns_ +=
        static_cast<size_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - rerank_begin)
                                .count());
#endif
  }

  enter_phase(metrics.metric_query_finalize_ns_);
  query_detail::FinalizeResults(results, k);
  return results;
}

template auto AllianceIndexDisk<int, float>::searchKNNMultiLevelPipeline<attr_range<int>>(
    float *, size_t, size_t, const attr_range<int> &, DiskSearchContext *)
    -> std::vector<std::pair<dist_t, label_t>>;

template auto
AllianceIndexDisk<int, float>::searchKNNMultiLevelPipeline<attr_range<att_label_t<int>>>(
    float *, size_t, size_t, const attr_range<att_label_t<int>> &, DiskSearchContext *)
    -> std::vector<std::pair<dist_t, label_t>>;

} // namespace alliance
