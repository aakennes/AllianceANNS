#include "support/test_check.hpp"
#include "metrics/search_metrics.hpp"
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>
#include "search/search_io_budget.hpp"
#include "metrics/io_queue_profile.hpp"

namespace search_metrics {

struct TestAtomic {
  mutable std::atomic<size_t> value{0};
  TestAtomic(size_t initial = 0) : value(initial) {
  }
  void fetch_add(size_t n, std::memory_order order) const {
    value.fetch_add(n, order);
  }
  size_t load() const {
    return value.load();
  }
};
struct TestIndex : alliance::MultiLevelMetricValues<TestAtomic> {};
static void Check(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}

void TestPublicationAndIsolation() {
  TestIndex a, b;
  {
    alliance::ScopedMultiLevelMetrics outer(a);
    outer.Get().metric_hops_ += 2;
    {
      alliance::ScopedMultiLevelMetrics nested(a);
      nested.Get().metric_hops_ += 3;
      nested.Get().metric_ml_cache_hits_per_level_[7]++;
    }
    Check(a.metric_hops_.load() == 0, "nested scope published before worker boundary");
    {
      alliance::ScopedMultiLevelMetrics other(b);
      other.Get().metric_hops_ = 7;
    }
    Check(b.metric_hops_.load() == 7, "nested different index was not isolated");
    alliance::ScopedMultiLevelMetrics resumed(a);
    resumed.Get().metric_hops_++;
  }
  Check(a.metric_hops_.load() == 6, "nested counters lost or double-counted");
  Check(a.metric_ml_cache_hits_per_level_[7].load() == 1, "array counter not published");
  try {
    alliance::ScopedMultiLevelMetrics standalone(a);
    standalone.Get().metric_hops_++;
    throw std::runtime_error("expected");
  } catch (const std::runtime_error &) {
  }
  Check(a.metric_hops_.load() == 7, "exception discarded partial metrics");

  std::vector<std::thread> workers;
  for (size_t t = 0; t < 16; ++t)
    workers.emplace_back([&, t] {
      alliance::ScopedMultiLevelMetrics batch(a);
      for (size_t q = 0; q < 200; ++q) {
        alliance::ScopedMultiLevelMetrics query(a);
        query.Get().metric_hops_ += 5;
        query.Get().metric_ml_route_hits_per_layer_[t]++;
      }
    });
  for (auto &worker : workers)
    worker.join();
  Check(a.metric_hops_.load() == 16007, "parallel merging lost updates");
  for (size_t t = 0; t < 16; ++t)
    Check(a.metric_ml_route_hits_per_layer_[t].load() == 200,
          "thread-local array leaked across workers");
  {
    alliance::ScopedMultiLevelMetrics fresh(a);
    Check(fresh.Get().metric_hops_ == 0, "new batch inherited old local counts");
  }
  alliance::MultiLevelMetricValues<std::atomic<size_t>> published;
  published.metric_hops_ = 12;
  published.metric_prefilter_queries_ = 5;
  published.metric_query_total_ns_ = 17;
  published.metric_ml_cache_hits_per_level_[7] = 3;
  published.metric_ml_route_misses_per_layer_[15] = 4;
  alliance::ResetMultiLevelMetrics(published);
  Check(published.metric_hops_ == 0 && published.metric_prefilter_queries_ == 0 &&
            published.metric_query_total_ns_ == 0 &&
            published.metric_ml_cache_hits_per_level_[7] == 0 &&
            published.metric_ml_route_misses_per_layer_[15] == 0,
        "next batch inherited published scalar/array/phase counters");
#ifdef ALLIANCE_PIPELINE_PROFILE
  size_t total = 0, prepare = 0, traversal = 0, wait = 0, calls = 0;
  {
    alliance::QueryPhaseTimer timer(total, prepare);
    for (volatile size_t i = 0; i < 10000; ++i) {
    }
    timer.SwitchTo(traversal);
    alliance::ProfileMetricWait(wait, calls, [] {
      for (volatile size_t i = 0; i < 10000; ++i) {
      }
    });
  }
  Check(total == prepare + traversal, "exclusive phases do not conserve elapsed time");
  Check(total > 0 && wait <= traversal && calls == 1, "blocking wait accounting invalid");
#endif
  std::cout
      << "search metrics nesting, isolation, exception, concurrency and phase accounting passed\n";
}

} // namespace search_metrics

namespace search_io_budget {

static void Check(bool ok, const char *what) {
  if (!ok)
    throw std::runtime_error(what);
}
void TestBudgetAndOccupancy() {
  using namespace alliance;
  for (size_t t : {1u, 8u, 16u, 31u}) {
    for (size_t sb : {1u, 8u, 32u, 64u})
      for (size_t rb : {1u, 16u, 64u, 256u}) {
        const auto legacy = PlanSearchIOBudget(t, sb, rb, 0);
        for (auto p : legacy)
          Check(p.search == sb && p.rerank == rb && p.queue_depth == sb + rb,
                "disabled budget changed defaults");
        for (size_t total : {t, t * 5 + 3, t * 64, t * 400}) {
          const auto plans = PlanSearchIOBudget(t, sb, rb, total);
          size_t shares = 0, maximum = 0;
          for (auto p : plans) {
            Check(p.search > 0 && p.search <= sb && p.rerank > 0 && p.rerank <= rb,
                  "invalid phase width");
            Check(p.search <= p.share && p.rerank <= p.share, "worker exceeds share");
            Check(p.queue_depth <= p.share, "queue exceeds worker budget");
            shares += p.share;
            maximum += std::max(p.search, p.rerank);
          }
          Check(shares == total && maximum <= total, "aggregate budget exceeded");
        }
      }
  }
  for (size_t total : {1u, 15u}) {
    bool caught = false;
    try {
      PlanSearchIOBudget(16, 16, 64, total);
    } catch (const std::invalid_argument &) {
      caught = true;
    }
    Check(caught, "too-small budget accepted");
  }
  for (auto p : PlanSearchIOBudget(16, 8, 64, 16 * 24))
    Check(p.search == 8 && p.rerank == 24 && p.queue_depth == 24,
          "sequential phases must independently use their full share");
  IOQueueObserver observer;
  IOQueueTracker tracker;
  tracker.Reset(&observer, 100);
  tracker.Submit(4, false, 110);
  tracker.Submit(2, true, 120);
  tracker.Complete(4, false, 150);
  tracker.Complete(2, true, 170);
  Check(tracker.GraphRequestNs() == 160 && tracker.VectorRequestNs() == 100,
        "time-weighted occupancy incorrect");
  Check(observer.peak == 6 && observer.graph_peak == 4 && observer.vector_peak == 2 &&
            observer.current == 0,
        "peak observer incorrect");
  observer.Reset();
  std::vector<std::thread> workers;
  for (size_t i = 0; i < 16; ++i)
    workers.emplace_back([&] {
      IOQueueTracker local;
      local.Reset(&observer);
      for (size_t q = 0; q < 500; ++q) {
        local.Submit(8, false);
        local.Submit(24, true);
        std::this_thread::yield();
        local.Complete(8, false);
        local.Complete(24, true);
      }
      Check(local.Pending() == 0, "worker leaked requests");
    });
  for (auto &w : workers)
    w.join();
  Check(observer.current == 0 && observer.graph == 0 && observer.vector == 0,
        "concurrent observer lost updates");
  Check(observer.peak <= 16 * 32 && observer.graph_peak <= 16 * 8 &&
            observer.vector_peak <= 16 * 24,
        "observer exceeded possible peak");
  std::cout
      << "IO budget partition, sequential phase bounds, legacy defaults and occupancy accounting passed\n";
}

} // namespace search_io_budget

int main() {
  return alliance::test::RunSuite("search observability contracts", [] {
    search_metrics::TestPublicationAndIsolation();
    search_io_budget::TestBudgetAndOccupancy();
  });
}
