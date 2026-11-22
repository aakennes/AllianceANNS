#pragma once
#include "core/types.hpp"
#include "search/disk_search_context.hpp"
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace alliance::query_detail {

using QueryResults = std::vector<std::pair<dist_t, label_t>>;

inline bool CandidateDistanceLess(const dist_id_pair &a, const dist_id_pair &b) {
  return a.dist_ < b.dist_ || (a.dist_ == b.dist_ && a.id_ < b.id_);
}

template <typename Metrics>
void FinalizeGraphCandidates(std::vector<dist_id_pair> &candidates,
                             std::unordered_set<tableint> &unique_ids, size_t budget,
                             Metrics &metrics) {
  // Deduplicate before truncating: repeated co-block discoveries must not use
  // several rerank slots. Estimated distance, then internal ID, resolves ties.
  std::sort(candidates.begin(), candidates.end(), CandidateDistanceLess);
  const size_t before_dedup = candidates.size();
  candidates.erase(
      std::remove_if(candidates.begin(), candidates.end(),
                     [&](const auto &item) { return !unique_ids.insert(item.id_).second; }),
      candidates.end());
  metrics.metric_ml_rerank_duplicate_ids_ += before_dedup - candidates.size();
  candidates.resize(std::min(candidates.size(), budget));
  metrics.metric_ml_rerank_candidates_ += candidates.size();
}

// Keep exact rerank's storage dispatch separate from traversal and budgeting.
// Legacy layouts still read raw vectors; ML layouts use grouped or bounded IO.
template <typename Index, typename Vector, typename Estimator>
void RerankCandidates(Index &index, const Vector *query,
                      const std::vector<dist_id_pair> &candidates, size_t k, Estimator &estimator,
                      DiskSearchContext *context, QueryResults &results) {
  if (index.ml_vec_region_loaded_) {
    if (context->rerank_error_bound)
      index.RerankMultiLevelBounded(query, candidates, k, estimator, context, results);
    else
      index.RerankMultiLevel(query, candidates, context, results);
  } else {
    std::vector<Vector> vector_cache;
    std::unordered_map<tableint, size_t> vector_cache_index;
    index.RerankPipeline(query, candidates, vector_cache, vector_cache_index, context, results);
  }
}

inline void FinalizeResults(QueryResults &results, size_t k) {
  // Exact distance, then external label, is the public result ordering.
  std::sort(results.begin(), results.end());
  results.resize(std::min(k, results.size()));
}

} // namespace alliance::query_detail
