#include "support/test_check.hpp"
#include "search/rerank_candidate_pool.hpp"
#include <cstdio>
#include <numeric>
#include <random>
#include "search/bounded_rerank.hpp"
#include <iostream>
#include <set>

namespace candidate_pool {

using namespace alliance;

void Check(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

void CheckStream(const std::vector<dist_id_pair> &stream, size_t ef, size_t budget) {
  RerankCandidatePool pool(ef, budget);
  std::vector<dist_id_pair> seen;
  for (const auto &entry : stream) {
    pool.OfferUnique(entry.id_, entry.dist_, double(entry.dist_) + double(entry.id_ + 1) * 1e-12);
    seen.push_back(entry);
    auto expected = seen;
    std::sort(expected.begin(), expected.end(), [](const auto &a, const auto &b) {
      return std::pair{a.dist_, a.id_} < std::pair{b.dist_, b.id_};
    });
    expected.resize(std::min(expected.size(), budget));
    auto main = pool.Main();
    auto reserve = pool.Reserve();
    Check(main.size() == std::min(seen.size(), std::min(ef, budget)), "main size mismatch");
    Check(reserve.size() == expected.size() - main.size(), "reserve size mismatch");
    auto actual = main;
    actual.insert(actual.end(), reserve.begin(), reserve.end());
    std::sort(actual.begin(), actual.end(), RerankCandidatePool::Better);
    Check(actual.size() == expected.size(), "union size mismatch");
    for (size_t i = 0; i < actual.size(); ++i)
      Check(actual[i].id_ == expected[i].id_ && actual[i].dist_ == expected[i].dist_,
            "split heaps lost a globally best candidate");
    for (const auto &c : actual)
      Check(c.precise_ == double(c.dist_) + double(c.id_ + 1) * 1e-12,
            "heap lost precise estimate");
    std::sort(main.begin(), main.end(), RerankCandidatePool::Better);
    for (size_t i = 0; i < main.size(); ++i)
      Check(main[i].id_ == expected[i].id_, "main/reserve partition incorrect");
    std::unordered_set<tableint> ids;
    for (const auto &v : actual)
      Check(ids.insert(v.id_).second, "duplicate across heaps");
  }
  const size_t retained = pool.Main().size() + pool.Reserve().size();
  auto all = pool.TakeCandidates();
  Check(all.size() == retained, "merge lost candidates");
}

void TestRetentionAndExactRecovery() {
  std::mt19937 rng(311);
  for (auto setting : std::vector<std::pair<size_t, size_t>>{
           {0, 0}, {0, 7}, {1, 1}, {1, 17}, {5, 17}, {17, 17}, {32, 17}, {8, 200}, {16, 400}}) {
    for (int order = 0; order < 5; ++order) {
      std::vector<dist_id_pair> stream;
      for (tableint id = 0; id < 257; ++id)
        stream.emplace_back(order == 4 ? 1.0f : static_cast<float>(id % 37), id);
      if (order == 1)
        std::reverse(stream.begin(), stream.end());
      if (order >= 2)
        std::shuffle(stream.begin(), stream.end(), rng);
      CheckStream(stream, setting.first, setting.second);
    }
  }
  // Both first-time rejection and later eviction must retain a useful point.
  RerankCandidatePool pool(2, 4);
  pool.OfferUnique(1, 5);
  pool.OfferUnique(2, 6);
  pool.OfferUnique(3, 7); // Immediately rejected by main, retained by reserve.
  pool.OfferUnique(4, 1); // Evicts ID 2 from main to reserve.
  pool.OfferUnique(5, 9); // Worse than both reserve candidates; discarded.
  auto candidates = pool.TakeCandidates();
  std::vector<std::pair<float, tableint>> exact;
  for (const auto &p : candidates)
    exact.emplace_back(p.id_ == 3 ? 0.0f : 10.0f, p.id_);
  std::sort(exact.begin(), exact.end());
  Check(exact.front().second == 3, "reserve failed to recover the exact nearest point");
  Check(candidates.size() == 4, "two heap capacities must sum to R");
  std::puts(
      "Rerank reserve: full-sort prefix oracle, ties, eviction, rejection, sparse discovery PASSED");
}

} // namespace candidate_pool

namespace bounded_rerank {

using alliance::detail::BoundedRerankCandidate;
using alliance::detail::BoundedRerankBlock;

void TestAsync() {
  using alliance::detail::RerankAsyncWithBounds;
  // Complete one slot at a time in reverse slot order. A slow first request
  // remains pending while later completions cause immediate refills.
  for (bool fail : {false, true}) {
    std::vector<BoundedRerankCandidate> candidates;
    for (size_t id = 0; id < 12; ++id)
      candidates.push_back({id, id, float(id), -100});
    std::vector<const BoundedRerankBlock *> active(4, nullptr);
    size_t submitted = 0, completed = 0, max_pending = 0;
    bool saw_refill = false, threw = false, failed = false;
    try {
      auto stats = RerankAsyncWithBounds(
          candidates, 2, 4, 2,
          [&](const auto &batch) {
            ALLIANCE_TEST_CHECK(batch.size() <= 2);
            ALLIANCE_TEST_CHECK(!failed);
            if (completed > 0 && active[0])
              saw_refill = true;
            for (auto &entry : batch) {
              ALLIANCE_TEST_CHECK(!active[entry.first]);
              active[entry.first] = entry.second;
              ++submitted;
            }
            max_pending = std::max(max_pending, submitted - completed);
          },
          [&](auto complete) {
            for (size_t i = active.size(); i-- > 0;)
              if (active[i]) {
                const auto id = active[i]->candidates[0].id;
                const bool ok = !(fail && completed == 0);
                failed |= !ok;
                active[i] = nullptr;
                ++completed;
                complete(i, ok, std::vector<std::pair<size_t, float>>{{id, float(id)}});
                break;
              }
          });
      ALLIANCE_TEST_CHECK(stats.blocks == 12 && stats.pruned == 0);
    } catch (const std::runtime_error &) {
      threw = true;
    }
    ALLIANCE_TEST_CHECK(threw == fail && submitted == completed && max_pending == 4);
    if (fail)
      ALLIANCE_TEST_CHECK(submitted == 4);
    else
      ALLIANCE_TEST_CHECK(saw_refill);
  }
  // Differential oracle: valid lower bounds, shared blocks, out-of-order
  // completion, ties, and invalid bounds that force reads.
  std::mt19937 rng(23);
  for (bool refining : {false, true})
    for (size_t trial = 0; trial < 150; ++trial) {
      const size_t k = 1 + trial % 25, width = 1 + trial % 16, limit = 1 + trial % 8;
      std::vector<BoundedRerankCandidate> candidates;
      std::vector<float> distances(200);
      std::vector<std::pair<float, size_t>> expected, observed;
      for (size_t id = 0; id < 200; ++id) {
        const float d = distances[id] = float(rng() % 100);
        expected.emplace_back(d, id);
        {
          float lb = d - float(rng() % 20);
          if (id % 17 == 0)
            lb = std::numeric_limits<float>::quiet_NaN();
          candidates.push_back({id, id / 3, d + float(rng() % 15), lb, d + float(rng() % 10)});
        }
      }
      std::sort(candidates.begin(), candidates.end(),
                [](const auto &a, const auto &b) { return a.estimate < b.estimate; });
      std::vector<const BoundedRerankBlock *> active(width, nullptr);
      std::set<size_t> reads, ids, refined_ids;
      std::function<std::pair<float, float>(const BoundedRerankCandidate &)> refine;
      if (refining)
        refine = [&](const auto &candidate) {
          ALLIANCE_TEST_CHECK(refined_ids.insert(candidate.id).second);
          const float distance = distances[candidate.id];
          if (candidate.id % 11 == 0)
            return std::pair<float, float>{NAN, NAN};
          return std::pair<float, float>{distance - 0.25f, distance + 0.25f};
        };
      auto stats = RerankAsyncWithBounds(
          candidates, k, width, limit,
          [&](const auto &batch) {
            ALLIANCE_TEST_CHECK(batch.size() <= limit);
            for (auto &entry : batch) {
              ALLIANCE_TEST_CHECK(!active[entry.first]);
              ALLIANCE_TEST_CHECK(reads.insert(entry.second->block).second);
              active[entry.first] = entry.second;
            }
          },
          [&](auto complete) {
            size_t slot = rng() % width;
            while (!active[slot])
              slot = (slot + 1) % width;
            std::vector<std::pair<size_t, float>> exact;
            for (const auto &c : active[slot]->candidates) {
              ALLIANCE_TEST_CHECK(ids.insert(c.id).second);
              exact.emplace_back(c.id, distances[c.id]);
              observed.emplace_back(distances[c.id], c.id);
            }
            active[slot] = nullptr;
            complete(slot, true, exact);
          },
          trial % 3 == 0 ? 0 : k * (trial % 3), trial % 4, refine);
      ALLIANCE_TEST_CHECK(stats.blocks == reads.size() && stats.pruned + ids.size() == 200);
      std::sort(expected.begin(), expected.end());
      std::sort(observed.begin(), observed.end());
      expected.resize(k);
      observed.resize(k);
      ALLIANCE_TEST_CHECK(expected == observed);
    }
}

void TestPrescreenAndWarmup() {
  using namespace alliance::detail;
  const float inf = std::numeric_limits<float>::infinity();
  std::vector<BoundedRerankCandidate> c{
      {0, 0, 0, 0, 1}, {1, 1, 1, 1, 2}, {2, 2, 2, 2, 3}, {3, 3, 3, 10, 11}, {4, 4, 4, -inf, inf}};
  ALLIANCE_TEST_CHECK(RerankUpperThreshold(c, 2) == 2);
  ALLIANCE_TEST_CHECK(RerankUpperThreshold(c, 6) == inf);
  std::vector<const BoundedRerankBlock *> active(8, nullptr);
  std::vector<size_t> reads;
  size_t completed = 0;
  const auto s = RerankAsyncWithBounds(
      c, 2, 8, 8,
      [&](const auto &batch) {
        ALLIANCE_TEST_CHECK(completed >= 2 ||
                            batch.size() == 1); // warm cap prevents eager full-width fill
        for (auto &e : batch) {
          active[e.first] = e.second;
          reads.push_back(e.second->candidates[0].id);
        }
      },
      [&](auto complete) {
        for (size_t i = 0; i < active.size(); ++i)
          if (active[i]) {
            auto id = active[i]->candidates[0].id;
            active[i] = nullptr;
            ++completed;
            complete(i, true, std::vector<std::pair<size_t, float>>{{id, float(id)}});
            break;
          }
      },
      2, 1);
  ALLIANCE_TEST_CHECK(s.prescreen_pruned == 1 && s.pruned == 2 &&
                      reads == std::vector<size_t>({0, 1, 4}));
}

void TestBoundSafetyAndSharedBlocks() {
  using alliance::detail::RerankBoundCanPrune;
  using alliance::detail::RerankAsyncWithBounds;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  ALLIANCE_TEST_CHECK(!RerankBoundCanPrune(100, 2, 3, 10));
  ALLIANCE_TEST_CHECK(!RerankBoundCanPrune(10, 3, 3, 10));
  ALLIANCE_TEST_CHECK(RerankBoundCanPrune(11, 3, 3, 10));
  ALLIANCE_TEST_CHECK(!RerankBoundCanPrune(nan, 3, 3, 10));
  ALLIANCE_TEST_CHECK(!RerankBoundCanPrune(inf, 3, 3, 10));
  ALLIANCE_TEST_CHECK(!RerankBoundCanPrune(11, 3, 3, nan));

  // The production async interface receives unique candidates. Shared blocks
  // require one read; an invalid lower bound must still receive exact IO after
  // completed distances tighten the threshold. A bound equal to tau survives.
  const std::vector<BoundedRerankCandidate> candidates{{0, 0, 0, 0},   {1, 0, 1, 1}, {2, 1, 2, 2},
                                                       {3, 2, 3, nan}, {4, 3, 4, 4}, {5, 4, 5, 1}};
  const std::vector<float> distances{0, 1, 2, 3, 4, 1};
  std::vector<std::pair<float, size_t>> expected, observed;
  for (const auto &candidate : candidates)
    expected.emplace_back(distances[candidate.id], candidate.id);
  std::set<size_t> read_ids, read_blocks;
  const BoundedRerankBlock *active = nullptr;
  const auto stats = RerankAsyncWithBounds(
      candidates, 2, 1, 1,
      [&](const auto &batch) {
        ALLIANCE_TEST_CHECK(batch.size() == 1 && active == nullptr);
        active = batch.front().second;
        ALLIANCE_TEST_CHECK(read_blocks.insert(active->block).second);
      },
      [&](auto complete) {
        ALLIANCE_TEST_CHECK(active != nullptr);
        std::vector<std::pair<size_t, float>> exact;
        for (const auto &candidate : active->candidates) {
          ALLIANCE_TEST_CHECK(read_ids.insert(candidate.id).second);
          exact.emplace_back(candidate.id, distances[candidate.id]);
          observed.emplace_back(distances[candidate.id], candidate.id);
        }
        active = nullptr;
        complete(0, true, exact);
      });
  ALLIANCE_TEST_CHECK(stats.pruned == 2 && stats.blocks == 3);
  ALLIANCE_TEST_CHECK(read_ids == std::set<size_t>({0, 1, 3, 5}));
  std::sort(expected.begin(), expected.end());
  std::sort(observed.begin(), observed.end());
  expected.resize(2);
  observed.resize(2);
  ALLIANCE_TEST_CHECK(expected == observed);
}

void TestAsyncBounds() {
  TestBoundSafetyAndSharedBlocks();
  TestPrescreenAndWarmup();
  TestAsync();
}

} // namespace bounded_rerank

int main() {
  return alliance::test::RunSuite("rerank contracts", [] {
    candidate_pool::TestRetentionAndExactRecovery();
    bounded_rerank::TestAsyncBounds();
  });
}
