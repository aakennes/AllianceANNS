#pragma once
#include "search/disk_search_policy.hpp"

namespace alliance {

enum class RangeSearchRoute { Empty, Graph, Prefilter, LegacyExact };

inline void ValidateFilterPolicy(const DiskSearchPolicy &policy, size_t k) {
  using Mode = DiskSearchPolicy::FilterMode;
  const bool prefilter_enabled =
      policy.filter_mode == Mode::Prefilter || policy.filter_mode == Mode::Auto;
  if (prefilter_enabled &&
      ((policy.prefilter_rerank != 0 && policy.prefilter_rerank < k) ||
       (policy.filter_mode == Mode::Auto && policy.prefilter_threshold == 0))) {
    throw std::invalid_argument("prefilter requires R=0 or R>=k and a positive auto threshold");
  }
}

// Routing only inspects range cardinality and available storage. It does not
// allocate estimator/scratch, scan vectors or invalidate query caches.
inline RangeSearchRoute ResolveRangeRoute(const DiskSearchPolicy &policy, size_t cardinality,
                                          bool reversed_range, bool ml_vectors_available,
                                          bool raw_vectors_available, size_t base_window) {
  using Mode = DiskSearchPolicy::FilterMode;
  if (cardinality == 0 || reversed_range)
    return RangeSearchRoute::Empty;
  if (policy.filter_mode == Mode::Prefilter ||
      (policy.filter_mode == Mode::Auto && ml_vectors_available &&
       cardinality <= policy.prefilter_threshold)) {
    if (!ml_vectors_available)
      throw std::invalid_argument("prefilter requires a colocated ML vector region and IO context");
    return RangeSearchRoute::Prefilter;
  }
  if (policy.filter_mode == Mode::Legacy && raw_vectors_available && cardinality < base_window)
    return RangeSearchRoute::LegacyExact;
  return RangeSearchRoute::Graph;
}

} // namespace alliance
