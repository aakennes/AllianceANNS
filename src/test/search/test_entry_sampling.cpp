#include "support/test_check.hpp"
#include "search/rank_entry_sampling.hpp"
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <iostream>
#include "search/guided_entry_directory.hpp"
#include <thread>

namespace rank_sampling {

using namespace alliance;
void check(bool b) {
  if (!b)
    throw std::runtime_error("rank sampling contract failed");
}
void TestRangeSampling() {
  // Numeric IDs deliberately disagree with attribute order.
  std::vector<uint32_t> ids{90, 7, 65, 1, 50, 12, 40, 3};
  std::vector<uint32_t> scored;
  auto accept = [](uint32_t id) { return id != 1; };
  auto score = [&](uint32_t id) {
    scored.push_back(id);
    return float(id);
  };
  auto best = SelectRankEntries(ids, 0, 8, 4, 4, accept, score);
  check(scored == std::vector<uint32_t>({7, 12, 3}));
  check(best.size() == 3 && best[0].second == 3 && best[1].second == 7 && best[2].second == 12);
  scored.clear();
  best = SelectRankEntries(ids, 2, 6, 512, 2, accept, score);
  check(scored == std::vector<uint32_t>({65, 50, 12}) && best.size() == 2 && best[0].second == 12);
  for (auto range : std::vector<std::pair<size_t, size_t>>{{0, 0}, {8, 8}})
    check(SelectRankEntries(ids, range.first, range.second, 512, 4, accept, score).empty());
  scored.clear();
  check(SelectRankEntries(ids, 0, 8, 0, 4, accept, score).empty());
  check(scored.empty());
  check(SelectRankEntries(ids, 0, 8, 512, 0, accept, score).empty());
  check(scored.empty());
  // Independent exhaustive membership/uniqueness checks for small spans/budgets.
  std::vector<uint32_t> seq(200);
  std::iota(seq.begin(), seq.end(), 0);
  for (size_t span = 1; span <= 100; ++span)
    for (size_t count = 1; count <= 110; ++count) {
      scored.clear();
      auto all =
          SelectRankEntries(seq, 20, 20 + span, count, 200, [](auto) { return true; }, score);
      const size_t expected = std::min(span, count);
      check(all.size() == expected &&
            std::set<uint32_t>(scored.begin(), scored.end()).size() == expected);
      for (size_t j = 0; j < expected; ++j) {
        const auto rank = size_t((static_cast<long double>(j) + 0.5L) * span / expected) + 20;
        check(scored[j] == rank && scored[j] >= 20 && scored[j] < 20 + span);
      }
    }
  std::cout << "rank entry sampling contracts PASSED\n";
}

} // namespace rank_sampling

namespace guided_sampling {

using alliance::GuidedEntryDirectory;
void check(bool b) {
  if (!b)
    throw std::runtime_error("guided directory contract failed");
}
void TestDirectoryLifecycleAndSelection() {
  GuidedEntryDirectory dir;
  for (size_t n : {size_t(1), size_t(7), size_t(100), size_t(1000)}) {
    dir.Clear();
    std::vector<uint32_t> ids(n);
    std::iota(ids.begin(), ids.end(), 0);
    std::reverse(ids.begin(), ids.end()); // ID order differs from attribute order.
    auto cid = [](uint32_t id) { return (id * 7) % 17; };
    std::vector<std::thread> workers;
    for (int t = 0; t < 8; ++t)
      workers.emplace_back([&] { dir.Prepare(ids, 17, cid); });
    for (auto &t : workers)
      t.join();
    std::vector<float> d(17);
    for (size_t c = 0; c < 17; ++c)
      d[c] = 16 - c;
    for (size_t lo = 0; lo < n; lo += std::max(size_t(1), n / 13))
      for (size_t hi : {lo, std::min(n, lo + 3), n})
        for (size_t budget : {size_t(0), size_t(1), size_t(256)}) {
          auto v = dir.Select(lo, hi, d, budget);
          std::set<uint32_t> s(v.begin(), v.end());
          check(v.size() == std::min(budget, hi - lo) && s.size() == v.size());
          for (auto rank : v)
            check(rank >= lo && rank < hi);
        }
  }
  dir.Clear();
  check(dir.Bytes() == 0);
  std::vector<uint32_t> ids(1600);
  std::iota(ids.begin(), ids.end(), 0);
  std::vector<float> d(16);
  std::iota(d.begin(), d.end(), 0);
  dir.Prepare(ids, 16, [](auto id) { return id / 100; });
  auto v = dir.Select(0, 1600, d, 256);
  std::vector<size_t> counts(16);
  for (auto r : v)
    ++counts[r / 100];
  for (size_t c = 0; c < 16; ++c)
    check(counts[c] == (c < 8 ? 32 : 0));
  dir.Clear();
  dir.Prepare(ids, 16, [](auto id) { return 15 - id / 100; });
  for (auto rank : dir.Select(0, 1600, d, 256))
    check(rank >= 800);
  // Independent index instances must not share directories.
  GuidedEntryDirectory other;
  other.Prepare(ids, 16, [](auto id) { return id / 100; });
  for (auto rank : other.Select(0, 1600, d, 256))
    check(rank < 800);
  bool threw = false;
  try {
    dir.Select(9, 8, d, 256);
  } catch (const std::out_of_range &) {
    threw = true;
  }
  check(threw);
  dir.Clear();
  threw = false;
  try {
    dir.Prepare(ids, 16, [](auto) { return 16; });
  } catch (const std::logic_error &) {
    threw = true;
  }
  check(threw);
  check(dir.Bytes() == 0);
  dir.Prepare(ids, 16, [](auto id) { return id / 100; });
  check(dir.Select(0, 1600, d, 256).size() == 256);
  std::cout << "guided directory contracts PASSED\n";
}

} // namespace guided_sampling

int main() {
  return alliance::test::RunSuite("entry sampling contracts", [] {
    rank_sampling::TestRangeSampling();
    guided_sampling::TestDirectoryLifecycleAndSelection();
  });
}
