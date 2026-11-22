#include "search/memory_navigation.hpp"
#include "support/navigation_fixture.hpp"
#include <cstdlib>
#include <filesystem>
#include <future>
#include <iostream>
#include <numeric>
#include <set>

namespace {
using Navigation = alliance::MemoryNavigationGraph;

void Check(bool condition) {
  if (!condition)
    throw std::runtime_error("RAM navigation contract failed");
}

template <class Action> void Rejects(Action action) {
  bool rejected = false;
  try {
    action();
  } catch (const std::exception &) {
    rejected = true;
  }
  Check(rejected);
}

Navigation Fixture() {
  Navigation graph;
  graph.population = 1024;
  graph.dimension = 2;
  graph.metric = 2;
  graph.degree = 4;
  graph.windows = {64, 256, 1024};
  for (size_t group_index = 0; group_index < 2; ++group_index) {
    Navigation::Group group;
    group.first = group_index ? 0 : 1;
    group.last = group_index ? 0 : 2;
    const size_t count = group_index ? 128 : 64;
    const size_t stride = graph.population / count;
    for (size_t slot = 0; slot < count; ++slot) {
      group.ranks.push_back(slot * stride);
      // Internal ID order deliberately differs from attribute rank order.
      group.ids.push_back((slot * stride * 37) % graph.population);
    }
    group.edges.assign((group.last - group.first + 1) * count * graph.degree,
                       Navigation::invalid);
    for (size_t layer = group.first; layer <= group.last; ++layer) {
      for (size_t slot = 0; slot < count; ++slot) {
        size_t edge = 0;
        for (int delta : {-2, -1, 1, 2}) {
          const int neighbor = int(slot) + delta;
          if (neighbor >= 0 && neighbor < int(count) &&
              std::abs(delta) * stride <= graph.windows[layer] / 2)
            group.edges[((layer - group.first) * count + slot) * graph.degree + edge++] =
                neighbor;
        }
      }
    }
    graph.groups.push_back(std::move(group));
  }
  graph.Validate();
  return graph;
}

void CheckSearch(const Navigation &graph) {
  auto search = [&] {
    std::set<uint32_t> scored;
    auto result = graph.Search(100, 900, 2, 32, 80, 16, [&](uint32_t id) {
      Check(scored.insert(id).second);
      return float(std::abs(int(id) - 500));
    });
    Check(result.scores == scored.size() && result.scores <= 80 && result.expansions > 0);
    Check(result.entries.size() == 16 && std::is_sorted(result.entries.begin(), result.entries.end()));
    // Frozen from the pre-refactor implementation, compiled independently.
    // Lock score/edge/expansion semantics, not only shape and range validity.
    const std::vector<std::pair<float, uint32_t>> historical{
        {12, 488}, {12, 512}, {28, 528}, {36, 464}, {36, 536}, {52, 448},
        {52, 552}, {60, 440}, {60, 560}, {76, 424}, {76, 576}, {84, 416},
        {84, 584}, {100, 400}, {100, 600}, {108, 392}};
    Check(result.entries == historical && result.scores == 78 && result.expansions == 39 &&
          result.edge_visits == 308 && result.layers_visited == 77);
    for (const auto &entry : result.entries) {
      bool in_range = false;
      for (const auto &group : graph.groups)
        for (size_t slot = 0; slot < group.ids.size(); ++slot)
          in_range |= group.ids[slot] == entry.second && group.ranks[slot] >= 100 &&
                      group.ranks[slot] < 900;
      Check(in_range);
    }
    return result.entries;
  };
  auto expected = search();
  std::vector<std::future<decltype(expected)>> workers;
  for (int worker = 0; worker < 4; ++worker)
    workers.push_back(std::async(std::launch::async, search));
  for (auto &worker : workers)
    Check(worker.get() == expected);
  // The coarse group has no rank 8; the denser group supplies it, not an endpoint.
  const auto narrow = graph.Search(8, 9, 2, 4, 4, 1, [](auto) { return 0.f; });
  Check(narrow.entries.size() == 1 && narrow.entries[0].second == 8 * 37);
  const auto one = graph.Search(0, 1024, 2, 1, 1, 1, [](auto) { return 0.f; });
  Check(one.scores == 1 && one.entries.size() == 1);
  Rejects([&] { graph.Search(101, 102, 2, 32, 80, 16, [](auto) { return 0.f; }); });
  Rejects([&] { graph.Search(100, 900, 2, 2, 1, 4, [](auto) { return 0.f; }); });
  Rejects([&] { graph.Search(100, 900, 2, 32, 80, 16, [](auto) { return NAN; }); });
}

void CheckSourceIdentity(Navigation graph) {
  struct Level {
    uint32_t owner_begin, owner_end;
  };
  struct Source {
    size_t curvec_num_ = 1024, vec_d_ = 2, cur_max_layer_ = 2;
    std::vector<uint32_t> sorted_ids_;
    std::vector<int> atts_;
    std::vector<uint64_t> window_size_{64, 256, 1024};
    std::vector<Level> ml_levels_{{1, 2}, {0, 0}};
  } source;
  source.sorted_ids_.resize(1024);
  source.atts_.resize(1024);
  for (size_t rank = 0; rank < 1024; ++rank) {
    source.sorted_ids_[rank] = (rank * 37) % 1024;
    source.atts_[source.sorted_ids_[rank]] = int(rank);
  }
  graph.order_hash = Navigation::OrderHash(source.sorted_ids_, source.atts_);
  graph.Match(source, 2);
  // Build geometry may reserve an unused top window beyond cur_max_layer_.
  // Navigation binds the active layers, not the vector's spare ladder tail.
  source.window_size_.push_back(4096);
  graph.Match(source, 2);
  source.window_size_.resize(2);
  Rejects([&] { graph.Match(source, 2); });
  source.window_size_ = {64, 256, 1024};
  Rejects([&] { graph.Match(source, 1); });
  source.vec_d_ = 3;
  Rejects([&] { graph.Match(source, 2); });
  source.vec_d_ = 2;
  source.atts_[0] = -1;
  Rejects([&] { graph.Match(source, 2); });
  source.atts_[0] = 0;
  source.ml_levels_[0].owner_begin = 2;
  Rejects([&] { graph.Match(source, 2); });
}
} // namespace

int main() {
  const char *temporary = std::getenv("TEST_TMPDIR");
  const auto path = (temporary ? std::filesystem::path(temporary)
                               : std::filesystem::temp_directory_path()) /
                    "legacy-navigation-fixture.bin";
  const auto graph = Fixture();
  test_support::WriteLegacyNavigation(path, graph);
  const auto loaded = Navigation::Load(path.string());
  Navigation::Load(path.string(), 1024, 2);
  Rejects([&] { Navigation::Load(path.string(), 1025, 2); });
  Rejects([&] { Navigation::Load(path.string(), 1024, 3); });
  Check(loaded.Bytes() > 0 && loaded.groups[0].edges == graph.groups[0].edges);
  CheckSearch(loaded);
  CheckSourceIdentity(loaded);
  auto malformed = graph;
  malformed.groups[0].edges[0] = 0;
  test_support::WriteLegacyNavigation(path, malformed); // Valid checksum must not bypass structural validation.
  Rejects([&] { Navigation::Load(path.string()); });
  malformed = graph;
  malformed.groups[1].ids[0] = 1;
  test_support::WriteLegacyNavigation(path, malformed);
  Rejects([&] { Navigation::Load(path.string()); });
  test_support::WriteLegacyNavigation(path, graph);
  {
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    file.seekp(40);
    file.put(17);
  }
  Rejects([&] { Navigation::Load(path.string()); });
  test_support::WriteLegacyNavigation(path, graph);
  std::filesystem::resize_file(path, 120);
  Rejects([&] { Navigation::Load(path.string()); });
  test_support::WriteLegacyNavigation(path, graph);
  {
    std::ofstream file(path, std::ios::binary | std::ios::app);
    file.put(0);
  }
  Rejects([&] { Navigation::Load(path.string()); });
  std::filesystem::remove(path);
  std::cout << "RAM navigation: legacy reader, identity, ranges, budgets, corruption, concurrency PASSED\n";
}
