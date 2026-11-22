#pragma once
#include "core/types.hpp"
#include "storage/order_table.hpp"
#include "search/disk_search_context.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <type_traits>
#include <vector>

namespace alliance::query_detail {

struct GraphEntry {
  attr_range<layer_t> layers;
  std::vector<dist_id_pair> points;
};

template <typename Attribute, typename Index, typename Filter, typename Estimator, typename Metrics>
GraphEntry PrepareGraphEntry(Index &index, const Filter &filter, Estimator &estimator,
                             const DiskSearchPolicy &policy, Metrics &metrics) {
  GraphEntry entry;
  if constexpr (std::is_same_v<Filter, attr_range<Attribute>>) {
    std::vector<tableint> ids;
    entry.layers = index.DecideLayerRange(filter, ids);
    if (index.HasNavigation()) {
      // Navigation is an entry replacement, not an additional guided sample.
      // It shares the query estimator, but its visited/pool state stays in RAM.
      const auto started = std::chrono::steady_clock::now();
      const auto [lo, hi] = index.order_table_->GetRankRange(
          {filter.l_, 0}, {filter.u_, std::numeric_limits<label_t>::max()});
      auto navigation = index.Navigation().Search(
          lo, hi, entry.layers.u_, policy.navigation_ef, policy.navigation_budget,
          policy.navigation_count, [&](uint32_t id) {
            return estimator.dist_estimate(index.SearchCID(id), id);
          });
      for (const auto &[distance, id] : navigation.entries)
        entry.points.emplace_back(distance, id);
      ++metrics.metric_navigation_queries_;
      metrics.metric_navigation_scores_ += navigation.scores;
      metrics.metric_navigation_expansions_ += navigation.expansions;
      metrics.metric_navigation_edges_ += navigation.edge_visits;
      metrics.metric_navigation_entries_ += navigation.entries.size();
      metrics.metric_dist_comps_ += navigation.scores;
      metrics.metric_navigation_ns_ += static_cast<size_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - started).count());
    } else {
      for (auto id : ids) {
        const auto distance = estimator.dist_estimate(index.SearchCID(id), id);
        ++metrics.metric_dist_comps_;
        entry.points.emplace_back(distance, id);
      }
    }
  } else {
    if (index.HasNavigation())
      throw std::invalid_argument("navigation requires attribute-range queries");
    const auto id = static_cast<tableint>(rand() % index.curvec_num_);
    const auto distance = estimator.dist_estimate(index.SearchCID(id), id);
    ++metrics.metric_dist_comps_;
    entry.points.emplace_back(distance, id);
    entry.layers = {static_cast<layer_t>(index.cur_max_layer_),
                    static_cast<layer_t>(index.cur_max_layer_)};
  }
  return entry;
}

template <typename Attribute, typename Index, typename Vector, typename Filter, typename Estimator,
          typename Metrics>
std::vector<dist_id_pair> TraverseGraph(Index &index, GraphEntry &entry, Vector *query,
                                        const Filter &filter, size_t cardinality, size_t efs,
                                        size_t k, DiskSearchContext *context, Estimator &estimator,
                                        Metrics &metrics) {
  if constexpr (std::is_same_v<Filter, attr_range<Attribute>>) {
    attr_range<att_label_t<Attribute>> dedup_filter{
        {filter.l_, 0}, {filter.u_, std::numeric_limits<label_t>::max()}};
    // Only the highest landing layer of a broad range may navigate without
    // range checks. Lower layers and narrower ranges keep constrained traversal.
    const bool top_postfilter = context->top_layer_postfilter &&
                                cardinality >= index.curvec_num_ / 2 && !index.ml_levels_.empty() &&
                                entry.layers.u_ == static_cast<layer_t>(index.cur_max_layer_) &&
                                index.ml_layer_to_level_[entry.layers.u_] == 0 &&
                                entry.layers.u_ == index.ml_levels_[0].owner_end;
    if (top_postfilter) {
      ++metrics.metric_filter_top_postfilter_queries_;
      auto candidates = index.template SearchCandidatesQueryMultiLevelPipeline<Filter, true>(
          entry.points, query, filter, entry.layers, efs, k, context, &estimator);
      // Retry only an underfilled query, with the same ef, estimator, entries,
      // and rerank budget. Global candidate deduplication follows this merge.
      const size_t required = std::min(k, cardinality);
      if (candidates.size() < required) {
        const size_t post_count = candidates.size();
        auto filtered = index.SearchCandidatesQueryMultiLevelPipeline(
            entry.points, query, dedup_filter, entry.layers, efs, k, context, &estimator);
        if (std::getenv("ALLIANCE_TRACE_TOP_FALLBACK"))
          std::fprintf(stderr,
                       "# top_postfilter_underfill: ef=%zu k=%zu retained=%zu filtered=%zu\n", efs,
                       k, post_count, filtered.size());
        candidates.insert(candidates.end(), filtered.begin(), filtered.end());
      }
      return candidates;
    }
    return index.SearchCandidatesQueryMultiLevelPipeline(entry.points, query, dedup_filter,
                                                         entry.layers, efs, k, context, &estimator);
  } else {
    return index.SearchCandidatesQueryMultiLevelPipeline(entry.points, query, filter, entry.layers,
                                                         efs, k, context, &estimator);
  }
}

} // namespace alliance::query_detail
