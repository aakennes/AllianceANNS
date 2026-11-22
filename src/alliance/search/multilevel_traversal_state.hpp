#pragma once
// Private, per-call traversal state. The worker owns visited bits and read
// buffers; these objects own only candidate lifecycle and submitted-request slots.
#include "io/io_handler.hpp"
#include "search/search_worker.hpp"
#include <algorithm>
#include <limits>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace alliance::detail {

class MultiLevelTraversalCandidates {
public:
  struct Neighbor {
    tableint id{static_cast<tableint>(-1)};
    dist_t dist{std::numeric_limits<dist_t>::max()};
    bool expanded{false};
    bool in_flight{false};
    bool evicted{false};
  };

  MultiLevelTraversalCandidates(MultiLevelVisited &visited, size_t ef, size_t degree)
      : visited_(visited), ef_(ef) {
    storage_.reserve(ef * (degree + 2));
    order_.reserve(ef + 1);
  }
  MultiLevelTraversalCandidates(const MultiLevelTraversalCandidates &) = delete;
  MultiLevelTraversalCandidates &operator=(const MultiLevelTraversalCandidates &) = delete;

  // Discovery is query-wide. A rejected/evicted ID is never readmitted: scores
  // are immutable and both traversal and result cutoffs only tighten.
  bool Claim(tableint id) {
    if (visited_.Test(id)) {
      ++duplicate_discoveries_;
      return false;
    }
    visited_.Set(id);
    ++unique_discoveries_;
    return true;
  }

  bool Rejects(dist_t distance) const {
    return order_.size() >= ef_ && !(distance < storage_[order_.back()].dist);
  }

  // Only a caller holding a successful Claim may insert. Stable storage indices
  // survive beam eviction and vector reallocation; IO slots never borrow a
  // Neighbor pointer. Discovery, outstanding IO and expansion stay independent.
  size_t Insert(tableint id, dist_t distance) {
#ifdef ALLIANCE_VALIDATE_CANDIDATES
    if (!stored_ids_.insert(id).second)
      throw std::logic_error("duplicate ID admitted to traversal storage");
#endif
    if (Rejects(distance))
      return SIZE_MAX;
    const size_t storage_idx = storage_.size();
    storage_.push_back(Neighbor{id, distance, false, false, false});
    size_t lo = 0, hi = order_.size();
    while (lo < hi) {
      const size_t mid = (lo + hi) / 2;
      if (storage_[order_[mid]].dist <= distance)
        lo = mid + 1;
      else
        hi = mid;
    }
    order_.insert(order_.begin() + lo, storage_idx);
    if (order_.size() > ef_) {
      storage_[order_.back()].evicted = true;
      order_.pop_back();
    }
    if (storage_[storage_idx].evicted)
      return SIZE_MAX;
    return storage_idx;
  }

  Neighbor &operator[](size_t index) {
    return storage_[index];
  }
  const Neighbor &operator[](size_t index) const {
    return storage_[index];
  }
  const std::vector<size_t> &Ordered() const {
    return order_;
  }
  size_t UniqueDiscoveries() const {
    return unique_discoveries_;
  }
  size_t DuplicateDiscoveries() const {
    return duplicate_discoveries_;
  }

private:
  MultiLevelVisited &visited_;
  const size_t ef_;
  std::vector<Neighbor> storage_;
  std::vector<size_t> order_;
  size_t unique_discoveries_{0}, duplicate_discoveries_{0};
#ifdef ALLIANCE_VALIDATE_CANDIDATES
  std::unordered_set<tableint> stored_ids_;
#endif
};

// Own request addresses until IOCompletionGuard drains the handler. read_buf
// points into the worker pool, sized before this state is constructed. Cache
// hits enter ready without an IO slot; misses share slots by physical block.
template <typename Compare> struct MultiLevelTraversalReads {
  struct Slot {
    size_t storage_idx{SIZE_MAX};
    uint64_t block_key{UINT64_MAX};
    IORequest req;
    char *read_buf{nullptr};
    bool in_flight{false};
    std::vector<size_t> waiters;
  };

  MultiLevelTraversalReads(size_t beam_width, Compare compare)
      : slots(beam_width), pending(compare), ready(compare),
        pipeline_width(std::min<size_t>(4, beam_width)) {
    block_slots.reserve(beam_width);
    submit_batch.reserve(beam_width);
    must_update_ids.reserve(beam_width);
  }
  MultiLevelTraversalReads(const MultiLevelTraversalReads &) = delete;
  MultiLevelTraversalReads &operator=(const MultiLevelTraversalReads &) = delete;

  std::vector<Slot> slots;
  std::priority_queue<size_t, std::vector<size_t>, Compare> pending;
  std::priority_queue<size_t, std::vector<size_t>, Compare> ready;
  std::unordered_map<uint64_t, size_t> block_slots;
  std::vector<IORequest *> submit_batch;
  std::vector<tableint> must_update_ids;
  size_t in_flight{0};
  // Start narrow, then widen when completed requests show little eviction waste.
  size_t pipeline_width;
  size_t retained_completions{0}, completed_candidates{0};
};

} // namespace alliance::detail
