#pragma once

#include <algorithm>
#include <limits>
#include <unordered_set>
#include <utility>
#include <vector>
#include "core/types.hpp"

namespace alliance {

// Result retention only: these heaps never schedule graph expansion or IO.
// Main holds the best min(efs, R); reserve holds the best R-main_capacity
// rejected/evicted candidates. Their union is the best R discovered IDs.
// OfferUnique requires the query-wide discovery gate to have claimed the ID.
// Scores are immutable; an ID moved to reserve is never offered again.
class RerankCandidatePool {
public:
  struct Entry : dist_id_pair {
    double precise_;
    Entry(dist_t d, tableint id, double precise = std::numeric_limits<double>::quiet_NaN())
        : dist_id_pair(d, id), precise_(precise) {
    }
  };
  RerankCandidatePool(size_t efs, size_t budget)
      : main_capacity_(std::min(efs, budget)), reserve_capacity_(budget - main_capacity_) {
  }

  static bool Better(const dist_id_pair &a, const dist_id_pair &b) {
    return a.dist_ < b.dist_ || (a.dist_ == b.dist_ && a.id_ < b.id_);
  }

  void OfferUnique(tableint id, dist_t distance,
                   double precise = std::numeric_limits<double>::quiet_NaN()) {
    const Entry candidate{distance, id, precise};
    if (main_.size() < main_capacity_) {
      main_.push_back(candidate);
      std::push_heap(main_.begin(), main_.end(), Better);
    } else if (!main_.empty() && Better(candidate, main_.front())) {
      std::pop_heap(main_.begin(), main_.end(), Better);
      const auto evicted = main_.back();
      main_.back() = candidate;
      std::push_heap(main_.begin(), main_.end(), Better);
      OfferReserve(evicted);
    } else {
      // Also retain candidates rejected at first admission. Saving only
      // formerly admitted IDs would miss potentially good exact neighbors.
      OfferReserve(candidate);
    }
#ifdef ALLIANCE_VALIDATE_CANDIDATES
    std::unordered_set<tableint> ids;
    for (const auto &heap : {&main_, &reserve_}) {
      for (const auto &entry : *heap)
        if (!ids.insert(entry.id_).second)
          throw std::logic_error("duplicate ID across graph rerank heaps");
    }
    if (main_.size() > main_capacity_ || reserve_.size() > reserve_capacity_)
      throw std::logic_error("graph rerank heap capacity exceeded");
#endif
  }

  // Exact admission cutoff across both retained heaps, including ties.
  bool CouldImprove(tableint id, dist_t distance) const {
    if (main_.size() + reserve_.size() < main_capacity_ + reserve_capacity_)
      return true;
    const dist_id_pair *worst = nullptr;
    if (!main_.empty())
      worst = &main_.front();
    if (!reserve_.empty() && (!worst || Better(*worst, reserve_.front())))
      worst = &reserve_.front();
    return worst && Better(dist_id_pair{distance, id}, *worst);
  }
  const std::vector<Entry> &Main() const {
    return main_;
  }
  const std::vector<Entry> &Reserve() const {
    return reserve_;
  }

  // Called once after traversal. The caller performs final sorting/rerank.
  std::vector<dist_id_pair> TakeCandidates() {
    std::vector<dist_id_pair> result;
    result.reserve(main_.size() + reserve_.size());
    for (const auto *heap : {&main_, &reserve_})
      for (const auto &c : *heap)
        result.emplace_back(c.dist_, c.id_);
    main_.clear();
    reserve_.clear();
    return result;
  }

private:
  void OfferReserve(const Entry &candidate) {
    if (reserve_capacity_ == 0)
      return;
    if (reserve_.size() < reserve_capacity_) {
      reserve_.push_back(candidate);
      std::push_heap(reserve_.begin(), reserve_.end(), Better);
    } else if (Better(candidate, reserve_.front())) {
      std::pop_heap(reserve_.begin(), reserve_.end(), Better);
      reserve_.back() = candidate;
      std::push_heap(reserve_.begin(), reserve_.end(), Better);
    }
  }

  size_t main_capacity_, reserve_capacity_;
  std::vector<Entry> main_, reserve_;
};

} // namespace alliance
