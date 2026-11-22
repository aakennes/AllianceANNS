#include "shuffle/shuffle.hpp"
#include "shuffle/layout.hpp"
#include "index/index_disk.hpp"
#include "shuffle/adjacency/neighbor_filter.hpp"
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <memory>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <unistd.h>

using namespace alliance;
namespace fs = std::filesystem;
using Index = AllianceIndexDisk<int, float>;
static void Check(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}

static bool TestLevelGeneration() {
  printf("--- Test 1: Level generation ---\n");

  struct Case {
    layer_t max_layer;
    MultiLevelConfig config;
    std::vector<LevelRange> expected;
  };
  const std::vector<Case> cases{
      {5, {2, 2}, {{2, 5, 4, 5}, {0, 3, 2, 3}, {0, 1, 0, 1}}},
      {11, {5, 2}, {{5, 11, 7, 11}, {0, 6, 2, 6}, {0, 1, 0, 1}}},
      {7, {3, 1}, {{4, 7, 5, 7}, {1, 4, 2, 4}, {0, 1, 0, 1}}},
      {0, {5, 2}, {{0, 0, 0, 0}}},
      {4, {2, 0}, {{3, 4, 3, 4}, {1, 2, 1, 2}, {0, 0, 0, 0}}},
      {5, {2, 99}, {{0, 5, 4, 5}, {0, 3, 2, 3}, {0, 1, 0, 1}}},
      {5, {2, 2, 4}, {{1, 5, 3, 5}, {1, 1, 1, 1}}},
      {6, {2, 2, 4}, {{0, 6, 4, 6}, {0, 2, 0, 2}}},
      {8, {2, 1, 8}, {{2, 8, 5, 8}, {2, 2, 2, 2}}},
  };
  for (const auto &c : cases) {
    const auto levels = GenerateLevelRanges(c.max_layer, c.config);
    if (levels.size() != c.expected.size())
      return false;
    const auto stride = static_cast<layer_t>(LayerStride(c.config.shuffle_o));
    for (size_t i = 0; i < levels.size(); ++i) {
      const auto &a = levels[i];
      const auto &b = c.expected[i];
      if (a.layer_begin != b.layer_begin || a.layer_end != b.layer_end ||
          a.owner_begin != b.owner_begin || a.owner_end != b.owner_end)
        return false;
      for (layer_t l = b.owner_begin; l <= b.owner_end; l += stride) {
        if (FindLevelForLayer(l, levels, stride) != i)
          return false;
      }
    }
    // Landing-layer LUT must round down to the selected grid, then use that
    // layer's owner. This includes partial bottom groups/top-aligned grids.
    const layer_t base = c.max_layer % stride;
    for (layer_t l = 0; l <= c.max_layer; ++l) {
      const layer_t selected = l < base ? base : l - (l - base) % stride;
      const size_t owner = FindLevelForLayer(l, levels, stride);
      if (selected < levels[owner].owner_begin || selected > levels[owner].owner_end)
        return false;
    }
  }
  // Configurations that previously hung (k=0) or silently rounded a non-power
  // of two are rejected; malformed persisted ownership cannot silently route.
  for (const auto config : {MultiLevelConfig{0, 2}, MultiLevelConfig{2, 2, 3}}) {
    bool rejected = false;
    try {
      GenerateLevelRanges(5, config);
    } catch (const std::exception &) {
      rejected = true;
    }
    if (!rejected)
      return false;
  }
  auto malformed = GenerateLevelRanges(5, {2, 2});
  malformed.pop_back();
  bool rejected = false;
  try {
    ValidateLevelRanges(malformed, 5, 1);
  } catch (const std::exception &) {
    rejected = true;
  }
  if (!rejected)
    return false;
  malformed = GenerateLevelRanges(5, {2, 2});
  malformed[1].owner_end = 4; // ownership overlaps the top replica
  rejected = false;
  try {
    ValidateLevelRanges(malformed, 5, 1);
  } catch (const std::exception &) {
    rejected = true;
  }
  if (!rejected)
    return false;

  printf("PASS: Level generation\n\n");
  return true;
}

static void CheckNeighborFilter() {
  using Filter = shuffle_detail::NeighborMembershipFilter;
  Filter filter;
  Check(!filter.MayContain(Filter::MaskFor(0)), "empty neighbor filter");
  std::vector<uint64_t> present = {
      0, 1, 7, 63, 64, std::numeric_limits<uint32_t>::max(), std::numeric_limits<uint64_t>::max()};
  for (uint64_t id : present) {
    filter.Add(id);
    filter.Add(id);
  }
  for (uint64_t id : present)
    Check(filter.MayContain(Filter::MaskFor(id)), "neighbor filter false negative");
  for (uint64_t id = 0; id < 512; ++id) {
    const bool exact = std::binary_search(present.begin(), present.end(), id);
    const bool filtered = filter.MayContain(Filter::MaskFor(id)) &&
                          std::binary_search(present.begin(), present.end(), id);
    Check(filtered == exact, "neighbor filter exact fallback");
  }

  // There are only 64 * 64 masks, so 4097 IDs must include a collision.
  // A positive filter result for that absent ID must still fail exact lookup.
  std::map<Filter::Mask, uint64_t> seen;
  uint64_t first = 0, second = 0;
  bool collision = false;
  for (uint64_t id = 0; id <= 4096; ++id) {
    const auto result = seen.emplace(Filter::MaskFor(id), id);
    if (!result.second) {
      first = result.first->second;
      second = id;
      collision = true;
      break;
    }
  }
  Check(collision, "neighbor filter collision fixture");
  Filter collided;
  collided.Add(first);
  const std::array<uint64_t, 1> adjacency{first};
  Check(collided.MayContain(Filter::MaskFor(second)), "neighbor filter collision positive");
  Check(!std::binary_search(adjacency.begin(), adjacency.end(), second),
        "neighbor filter collision requires exact rejection");
}
template <typename T> static void Put(std::ostream &out, const T &value) {
  out.write(reinterpret_cast<const char *>(&value), sizeof(value));
}
static void Permutation(const std::vector<tableint> &order, size_t n) {
  auto sorted = order;
  std::sort(sorted.begin(), sorted.end());
  Check(sorted.size() == n, "order size");
  for (size_t i = 0; i < n; ++i)
    Check(sorted[i] == i, "order permutation");
}
static void DumpOrder(std::ostream &out, const std::vector<tableint> &order) {
  Put(out, order.size());
  for (auto id : order)
    Put(out, id);
}
static void CheckStats(Index &index, const std::vector<tableint> &order, layer_t first,
                       layer_t last, size_t stride, const ShuffleStats &actual) {
  std::vector<size_t> positions(order.size());
  for (size_t i = 0; i < order.size(); ++i)
    positions[order[i]] = i;
  std::vector<tableint> edges(index.M_ + 1);
  std::vector<size_t> spans;
  for (tableint u = 0; u < order.size(); ++u) {
    for (layer_t layer = first; layer <= last; layer += stride) {
      index.ReadEdges(u, layer, edges.data());
      for (size_t j = 0; j < edges[index.M_]; ++j) {
        const tableint v = edges[j];
        if (u == v || v >= order.size())
          continue;
        spans.push_back(positions[u] > positions[v] ? positions[u] - positions[v]
                                                    : positions[v] - positions[u]);
      }
    }
  }
  std::sort(spans.begin(), spans.end());
  const double mean =
      spans.empty() ? 0
                    : double(std::accumulate(spans.begin(), spans.end(), size_t(0))) / spans.size();
  const double p90 = spans.empty() ? 0 : spans[size_t(.9 * (spans.size() - 1))];
  Check(actual.num_edges == spans.size() && actual.avg_edge_span == mean &&
            actual.p90_edge_span == p90 &&
            actual.max_edge_span == (spans.empty() ? 0 : spans.back()),
        "locality oracle");
}

static int shuffle_threads = 1;
static size_t shuffle_batch = 16384;
static void Run(const fs::path &output) {
  fs::create_directories(output);
  std::ofstream summary(output / "orders.bin", std::ios::binary);
  const ShuffleMethod methods[] = {ShuffleMethod::Identity, ShuffleMethod::BlockBNF};
  size_t checks = 0;
  for (size_t n : {0, 1, 73}) {
    const auto folder = output / std::to_string(n);
    fs::create_directory(folder);
    auto idx = std::make_unique<Index>(std::max(n, size_t(1)), 128, 8, 24, "l2",
                                       (folder / "source.graph").string(), 2, 5, false);
    if (n) {
      std::vector<float> values(n * 128);
      std::vector<int> attributes(n);
      for (size_t i = 0; i < n; ++i) {
        attributes[i] = int((i * 29) % n);
        for (size_t d = 0; d < 128; ++d)
          values[i * 128 + d] = float(i * 128 + d) / 128;
      }
      idx->PopulateElements(values.data(), attributes.data(), n);
      for (tableint u = 0; u < n; ++u)
        for (layer_t l = 0; l <= 5; ++l) {
          std::vector<tableint> neighbors;
          if (u % 9)
            neighbors = {tableint((u * 5 + l) % n), tableint((u + 1) % n), tableint((u + 1) % n),
                         u};
          idx->WriteEdges(u, l, neighbors.data(), neighbors.size());
        }
    }
    idx->cur_max_layer_ = 5;
    idx->graph_file_->Sync();
    for (auto method : methods) {
      BNFParams params{512, 3, .001};
      if (method == ShuffleMethod::BlockBNF) {
        params.threads = shuffle_threads;
        params.affinity_batch_size = shuffle_batch;
      }
      for (size_t grid : {2, 4, 8}) {
        MultiLevelConfig config{2, 1, grid};
        params.max_iterations = grid == 4 ? 0 : 3;
        config.per_level_params.assign(3, params);
        // Diagnostics are explicit and must never influence the permutation.
        const auto without_diagnostics = ComputeMultiLevelShuffleOrders(idx.get(), config, method);
        if (method == ShuffleMethod::BlockBNF) {
          auto automatic = config;
          for (auto &p : automatic.per_level_params) {
            p.storage = ShuffleStorage::Auto;
            p.memory_limit_bytes = size_t(1) << 30;
          }
          auto auto_orders = ComputeMultiLevelShuffleOrders(idx.get(), automatic, method);
          for (size_t i = 0; i < auto_orders.size(); ++i)
            Check(auto_orders[i].order == without_diagnostics[i].order, "auto RAM changed order");
        }
        config.collect_diagnostics = true;
        auto levels = ComputeMultiLevelShuffleOrders(idx.get(), config, method);
        Check(levels.size() == without_diagnostics.size(), "diagnostic level count");
        for (size_t i = 0; i < levels.size(); ++i) {
          Check(levels[i].order == without_diagnostics[i].order, "diagnostics changed order");
          Check(levels[i].stats_collected && !without_diagnostics[i].stats_collected,
                "explicit diagnostic state");
          Check(without_diagnostics[i].stats.num_edges == 0 &&
                    without_diagnostics[i].stats.avg_edge_span == 0 &&
                    without_diagnostics[i].stats.p90_edge_span == 0 &&
                    without_diagnostics[i].stats.max_edge_span == 0,
                "disabled diagnostic values");
        }
        for (const auto &level : levels) {
          Permutation(level.order, n);
          DumpOrder(summary, level.order);
          Put(summary, level.range.layer_begin);
          Put(summary, level.range.layer_end);
          Put(summary, level.range.owner_begin);
          Put(summary, level.range.owner_end);
          Put(summary, level.stats.num_edges);
          Put(summary, level.stats.avg_edge_span);
          Put(summary, level.stats.p90_edge_span);
          Put(summary, level.stats.max_edge_span);
          CheckStats(*idx, level.order, level.range.owner_begin, level.range.owner_end,
                     LayerStride(grid), level.stats);
        }
        const std::string stem = std::to_string(int(method)) + "_" + std::to_string(grid);
        LayoutExportConfig export_config;
        export_config.block_size = 512;
        export_config.level_config = config;
        export_config.method = method;
        export_config.out_graph_path = (folder / (stem + ".graph")).string();
        export_config.out_meta_path = (folder / (stem + ".layout")).string();
        if (n == 0) {
          // Empty permutations remain a valid ordering contract, but are not
          // searchable layouts. Both export entry points must reject them
          // before creating any output artifact.
          for (bool precomputed : {false, true}) {
            bool rejected = false;
            try {
              if (precomputed)
                ExportMultiLevelLayout(idx.get(), export_config, levels);
              else
                ExportMultiLevelLayout(idx.get(), export_config);
            } catch (const std::invalid_argument &) {
              rejected = true;
            }
            Check(rejected, "empty layout export must be rejected");
            Check(!fs::exists(export_config.out_graph_path) &&
                      !fs::exists(export_config.out_meta_path),
                  "empty layout export created output artifacts");
          }
          ++checks;
          continue;
        }
        const auto result = ExportMultiLevelLayout(idx.get(), export_config, levels);
        Check(result.total_bytes == fs::file_size(export_config.out_graph_path),
              "layout byte count");
        for (const auto &level : result.levels) {
          Check(level.disk_blocks.size() == n, "mapping count");
          for (auto block : level.disk_blocks)
            Check(block >= 0, "unmapped node");
        }
        // The normal loader validates headers, ownership and sidecar mappings.
        idx->LoadMultiLevelLayout(export_config.out_graph_path, export_config.out_meta_path, 4096);
        ++checks;
      }
    }
  }
  summary.close();
  Check(summary.good(), "summary write");
  std::printf("SHUFFLE_CONTRACT_PASS cases=%zu methods=2 sizes=0/1/73 grids=2/4/8\n", checks);
}

int main(int argc, char **argv) {
  try {
    Check(TestLevelGeneration(), "level geometry and routing");
    CheckNeighborFilter();
    if (argc > 2)
      shuffle_threads = std::stoi(argv[2]);
    if (argc > 3)
      shuffle_batch = std::stoull(argv[3]);
    if (argc > 1)
      Run(argv[1]);
    else {
      char pattern[] = "/tmp/alliance-shuffle-contract-XXXXXX";
      const char *dir = ::mkdtemp(pattern);
      Check(dir, "temporary directory");
      const fs::path path(dir);
      Run(path);
      fs::remove_all(path);
    }
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}
