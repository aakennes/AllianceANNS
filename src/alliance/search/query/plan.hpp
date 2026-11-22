#pragma once
#include "core/types.hpp"
#include "search/disk_search_context.hpp"
#include "search/search_route.hpp"
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace alliance::query_detail {

struct QueryPlan {
  RangeSearchRoute route{RangeSearchRoute::Graph};
  size_t rank_begin{0};
  size_t rank_end{0};
  size_t cardinality{0};
};

template <typename Index>
void ValidateStorage(const Index &index, const DiskSearchContext *context) {
  if (index.IsMultiLevelOnly() || (context && context->IsMultiLevelOnly())) {
    if (!context || !context->ml_io_handler || !index.ml_layout_loaded_ ||
        !index.ml_vec_region_loaded_)
      throw std::runtime_error(
          "ML-only search requires a loaded complete shuffled layout and ML IO context");
    if (context->filter_mode == DiskSearchContext::FilterMode::Legacy)
      throw std::invalid_argument(
          "ML-only search requires graph, auto or prefilter routing; legacy is unavailable");
  }
}

template <typename Index>
void ValidateBoundedRerank(const Index &index, const DiskSearchContext &context) {
  if (context.rerank_error_bound &&
      ((index.rabitq_space_name_ != "l2" && index.rabitq_space_name_ != "ip") ||
       !index.ml_vec_region_loaded_ || context.rerank_batch_blocks == 0 ||
       context.rerank_bound_rank_factor == 0))
    throw std::invalid_argument(
        "bounded rerank requires L2/IP, colocated vectors and positive batch width");
}

// Planning does not create an estimator, allocate graph scratch, or clear caches.
// Wide ranges only pay for the order table's O(log N) cardinality lookup.
template <typename Attribute, typename Index, typename Filter>
QueryPlan PlanQuery(Index &index, const Filter &filter, const DiskSearchContext &context) {
  QueryPlan plan;
  if constexpr (std::is_same_v<Filter, attr_range<Attribute>>) {
    const auto ranks = index.order_table_->GetRankRange(
        {filter.l_, 0}, {filter.u_, std::numeric_limits<label_t>::max()});
    plan.rank_begin = ranks.first;
    plan.rank_end = ranks.second;
    plan.cardinality = plan.rank_end - plan.rank_begin;
    plan.route = ResolveRangeRoute(context, plan.cardinality, filter.u_ < filter.l_,
                                   index.ml_vec_region_loaded_ && context.ml_io_handler,
                                   index.vec_file_available_, index.window_size_[0]);
  } else if (context.filter_mode == DiskSearchContext::FilterMode::Prefilter) {
    throw std::invalid_argument("prefilter currently requires an attribute range");
  }
  return plan;
}

} // namespace alliance::query_detail
