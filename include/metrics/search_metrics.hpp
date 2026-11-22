#pragma once
// Query/worker-owned counters. Public index atomics remain readable after the
// outer scope exits. Nested helpers share the same batch; different indexes
// get separate scopes. No shared writes occur in the inner search loop.
#include <atomic>
#include <array>
#include <chrono>
#include <cstddef>
#include <optional>
#include <utility>

namespace alliance {
template <typename Counter> struct alignas(64) MultiLevelMetricValues {
#define ALLIANCE_SEARCH_METRIC(name) Counter name{0};
#define ALLIANCE_SEARCH_METRIC_ARRAY(name, count) std::array<Counter, count> name{};
#include "metrics/search_metric_fields.inc"
#undef ALLIANCE_SEARCH_METRIC_ARRAY
#undef ALLIANCE_SEARCH_METRIC
};

// Reset only between batches, with no active metric scopes/workers.
template <typename Index> void ResetMultiLevelMetrics(Index &index) {
#define ALLIANCE_SEARCH_METRIC(name) index.name = 0;
#define ALLIANCE_SEARCH_METRIC_ARRAY(name, count)                                                  \
  for (auto &counter : index.name)                                                                 \
    counter = 0;
#include "metrics/search_metric_fields.inc"
#undef ALLIANCE_SEARCH_METRIC_ARRAY
#undef ALLIANCE_SEARCH_METRIC
}

// An outer worker scope amortizes merging across a whole query batch. Public
// standalone calls create their own scope, so their counters remain observable
// immediately after return. Partial counters are also merged on exceptions.
template <typename Index> class ScopedMultiLevelMetrics {
public:
  explicit ScopedMultiLevelMetrics(const Index &index) : index_(index) {
    if (active_ && &active_->index_ == &index_) {
      values_ = active_->values_;
    } else {
      owned_.emplace();
      values_ = &*owned_;
      previous_ = active_;
      active_ = this;
    }
  }
  ScopedMultiLevelMetrics(const ScopedMultiLevelMetrics &) = delete;
  ScopedMultiLevelMetrics &operator=(const ScopedMultiLevelMetrics &) = delete;
  ~ScopedMultiLevelMetrics() noexcept {
    if (!owned_)
      return;
    const auto &m = *owned_;
#define ALLIANCE_SEARCH_METRIC(name)                                                               \
  if (m.name)                                                                                      \
    index_.name.fetch_add(m.name, std::memory_order_relaxed);
#define ALLIANCE_SEARCH_METRIC_ARRAY(name, count)                                                  \
  for (size_t i = 0; i < count; ++i)                                                               \
    if (m.name[i])                                                                                 \
      index_.name[i].fetch_add(m.name[i], std::memory_order_relaxed);
#include "metrics/search_metric_fields.inc"
#undef ALLIANCE_SEARCH_METRIC_ARRAY
#undef ALLIANCE_SEARCH_METRIC
    active_ = previous_;
  }
  MultiLevelMetricValues<size_t> &Get() noexcept {
    return *values_;
  }

private:
  const Index &index_;
  std::optional<MultiLevelMetricValues<size_t>> owned_;
  MultiLevelMetricValues<size_t> *values_{nullptr};
  ScopedMultiLevelMetrics *previous_{nullptr};
  inline static thread_local ScopedMultiLevelMetrics *active_{nullptr};
};

#ifdef ALLIANCE_PIPELINE_PROFILE
// Exclusive top-level elapsed-time phases. Their sum equals total exactly;
// these are per-query wall times, not CPU time or batch_wall / query_count.
class QueryPhaseTimer {
public:
  QueryPhaseTimer(size_t &total, size_t &phase)
      : total_(total), phase_(&phase), last_(Clock::now()) {
  }
  ~QueryPhaseTimer() {
    SwitchTo(*phase_);
  }
  void SwitchTo(size_t &phase) {
    const auto now = Clock::now();
    const size_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now - last_).count();
    *phase_ += ns;
    total_ += ns;
    phase_ = &phase;
    last_ = now;
  }

private:
  using Clock = std::chrono::steady_clock;
  size_t &total_;
  size_t *phase_;
  Clock::time_point last_;
};
#endif

// Only the blocking completion call is timed here. Existing poll counters also
// include nonblocking completion processing, decoding and cache insertion.
template <typename Fn> inline void ProfileMetricWait(size_t &elapsed_ns, size_t &calls, Fn &&fn) {
#ifdef ALLIANCE_PIPELINE_PROFILE
  const auto start = std::chrono::steady_clock::now();
  ++calls;
#endif
  std::forward<Fn>(fn)();
#ifdef ALLIANCE_PIPELINE_PROFILE
  elapsed_ns +=
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start)
          .count();
#endif
}
} // namespace alliance
