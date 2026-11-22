#pragma once
#include "search/disk_search_workspace.hpp"
#include "search/query/plan.hpp"
#include "search/query/results.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace alliance::query_detail {

template <typename Index, typename Vector, typename Metrics, typename EnterPhase>
QueryResults ExecutePrefilter(Index &index, Vector *query, size_t efs, size_t k,
                              const QueryPlan &plan, DiskSearchContext *context, Metrics &metrics,
                              EnterPhase &enter_phase) {
  const size_t budget =
      std::min(plan.cardinality,
               context->prefilter_rerank
                   ? context->prefilter_rerank
                   : std::max(k, DiskSearchContext::SaturatingProduct(
                                     efs, std::max(size_t{1}, context->rerank_candidate_factor))));
  auto &scratch = detail::GetSearchWorkspace(&index, context);
  auto &ids = scratch.PrefilterIDs();
  auto &candidates = scratch.PrefilterCandidates();
  candidates.clear();
  candidates.reserve(budget);
  typename Index::RaBitQEstimator estimator(query, &index);

  enter_phase(metrics.metric_query_traversal_ns_);
  constexpr size_t chunk_size = 4096;
  for (size_t begin = plan.rank_begin; begin < plan.rank_end;) {
    const size_t end = begin + std::min(chunk_size, plan.rank_end - begin);
    index.order_table_->GetIDsInRankRange(begin, end, ids);
    for (auto id : ids) {
      const double precise = estimator.dist_estimate_precise(index.cids_[index.labels_[id]], id);
      const float distance = static_cast<float>(precise);
      if (!std::isfinite(distance))
        throw std::runtime_error("non-finite prefilter distance estimate");
      const RerankCandidatePool::Entry candidate{distance, id, precise};
      if (candidates.size() < budget) {
        candidates.push_back(candidate);
        std::push_heap(candidates.begin(), candidates.end(), CandidateDistanceLess);
      } else if (CandidateDistanceLess(candidate, candidates.front())) {
        std::pop_heap(candidates.begin(), candidates.end(), CandidateDistanceLess);
        candidates.back() = candidate;
        std::push_heap(candidates.begin(), candidates.end(), CandidateDistanceLess);
      }
    }
    begin = end;
  }
  metrics.metric_dist_comps_ += plan.cardinality;
  ++metrics.metric_prefilter_queries_;
  metrics.metric_prefilter_scanned_ += plan.cardinality;
  metrics.metric_prefilter_candidates_ += candidates.size();
  QueryResults results;

  enter_phase(metrics.metric_query_rerank_ns_);
  // Grouped exact IO does not require estimated-distance ordering. Bounded
  // rerank retains precise estimates for optional refinement-cache reuse.
  std::vector<dist_id_pair> retained;
  retained.reserve(candidates.size());
  if (context->rerank_error_bound)
    std::sort(candidates.begin(), candidates.end(), CandidateDistanceLess);
  if (context->rerank_error_bound && context->rerank_float_refine &&
      context->rerank_reuse_estimates)
    estimator.refinement_cache_.Reset(candidates.size());
  for (const auto &candidate : candidates) {
    retained.emplace_back(candidate.dist_, candidate.id_);
    estimator.refinement_cache_.Insert(candidate.id_, index.SearchCID(candidate.id_),
                                       candidate.precise_);
  }
  RerankCandidates(index, query, retained, k, estimator, context, results);

  enter_phase(metrics.metric_query_finalize_ns_);
  FinalizeResults(results, k);
  return results;
}

} // namespace alliance::query_detail
