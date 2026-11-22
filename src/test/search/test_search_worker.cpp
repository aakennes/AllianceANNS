#include "index/index_disk.hpp"
#include "search/search_route.hpp"
#include "shuffle/layout.hpp"
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <thread>
#include <unistd.h>

using namespace alliance;
using Index = AllianceIndexDisk<int, float>;
namespace fs = std::filesystem;
static void Check(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
struct Scratch {
  fs::path path;
  Scratch() {
    const char *base = std::getenv("TEST_TMPDIR");
    auto pattern = ((base ? fs::path(base) : fs::current_path()) / "search-worker-XXXXXX").string();
    std::vector<char> chars(pattern.begin(), pattern.end());
    chars.push_back(0);
    Check(mkdtemp(chars.data()), "mkdtemp");
    path = chars.data();
  }
  ~Scratch() {
    std::error_code error;
    fs::remove_all(path, error);
  }
};
static void TestRoutes() {
  DiskSearchPolicy policy;
  using Mode = DiskSearchPolicy::FilterMode;
  using Route = RangeSearchRoute;
  for (auto mode : {Mode::Legacy, Mode::Graph, Mode::Prefilter, Mode::Auto}) {
    policy.filter_mode = mode;
    Check(ResolveRangeRoute(policy, 0, false, false, false, 32) == Route::Empty, "empty route");
    Check(ResolveRangeRoute(policy, 40, true, false, false, 32) == Route::Empty, "reversed route");
  }
  policy.filter_mode = Mode::Auto;
  policy.prefilter_threshold = 32;
  Check(ResolveRangeRoute(policy, 32, false, true, true, 64) == Route::Prefilter,
        "inclusive auto threshold");
  Check(ResolveRangeRoute(policy, 33, false, true, true, 64) == Route::Graph,
        "auto ignores raw fallback");
  Check(ResolveRangeRoute(policy, 8, false, false, true, 64) == Route::Graph,
        "auto without ML vectors");
  policy.filter_mode = Mode::Legacy;
  Check(ResolveRangeRoute(policy, 31, false, true, true, 32) == Route::LegacyExact,
        "legacy fallback");
  Check(ResolveRangeRoute(policy, 32, false, true, true, 32) == Route::Graph,
        "legacy strict threshold");
  policy.filter_mode = Mode::Graph;
  Check(ResolveRangeRoute(policy, 8, false, true, true, 32) == Route::Graph, "forced graph");
  policy.filter_mode = Mode::Prefilter;
  bool caught = false;
  try {
    ResolveRangeRoute(policy, 8, false, false, true, 32);
  } catch (const std::invalid_argument &) {
    caught = true;
  }
  Check(caught, "forced prefilter requires ML vectors");
  policy.prefilter_rerank = 4;
  caught = false;
  try {
    ValidateFilterPolicy(policy, 5);
  } catch (const std::invalid_argument &) {
    caught = true;
  }
  Check(caught, "prefilter R validation");
}
static void TestWorkspaceOwnership() {
  SearchWorker first, second;
  int owner = 0, other_owner = 0;
  first.Bind(&owner, 1);
  second.Bind(&owner, 1);
  first.EnsureGraphCache(4096, 32, DynamicBlockCache::EvictPolicy::FIFO, 128, false, 4096);
  second.EnsureGraphCache(4096, 32, DynamicBlockCache::EvictPolicy::FIFO, 128, false, 4096);
  char record[32]{};
  first.Cache().InsertIfMissing(7, record, sizeof(record), 0, 0);
  Check(first.Cache().Lookup(7).found && !second.Cache().Lookup(7).found, "workers share cache");
  first.Bind(&owner, 1);
  Check(first.Cache().Lookup(7).found, "query boundary cleared cache");
  first.GetVisited(128).Set(7);
  Check(!first.GetVisited(128).Test(7), "visited survived query boundary");
  first.EnsurePipelineBuffers(2, 8192);
  auto *pipeline = first.PipelineBuffer(0);
  first.PrefilterIDs().reserve(32);
  const auto retained = first.MemoryUsage().prefilter_scratch_bytes;
  first.Bind(&owner, 2);
  Check(!first.Cache().Lookup(7).found, "EF generation retained cache");
  Check(first.PipelineBuffer(0) == pipeline &&
            first.MemoryUsage().prefilter_scratch_bytes == retained,
        "EF boundary changed retained scratch policy");
  first.EnsureGraphCache(4096, 32, DynamicBlockCache::EvictPolicy::FIFO, 128, false, 4096);
  first.Cache().InsertIfMissing(7, record, sizeof(record), 0, 0);
  first.Bind(&other_owner, 2);
  Check(!first.Cache().Lookup(7).found, "index switch retained cache");
}
struct Observation {
  std::vector<std::pair<dist_t, label_t>> results;
  size_t graph_reads, vector_reads;
};
static Observation Query(Index &index, DiskSearchContext &context, float *query) {
  ResetMultiLevelMetrics(index);
  auto result = index.searchKNNMultiLevelPipeline(query, 64, 5, attr_range<int>{0, 127}, &context);
  return {std::move(result), index.metric_ml_io_submits_.load(),
          index.metric_ml_rerank_io_submits_.load()};
}
static void Same(const Observation &a, const Observation &b) {
  Check(a.results == b.results, "worker changed results/distances");
  Check(a.graph_reads == b.graph_reads && a.vector_reads == b.vector_reads,
        "worker changed logical IO");
}
int main() {
  try {
    TestRoutes();
    TestWorkspaceOwnership();
    Scratch scratch;
    constexpr size_t n = 128, d = 64;
    std::vector<float> vectors(n * d);
    std::vector<int> attrs(n);
    for (size_t i = 0; i < n; ++i) {
      attrs[i] = int(i);
      for (size_t j = 0; j < d; ++j)
        vectors[i * d + j] = std::sin(float(i * 17 + j * 11) * .031f);
    }
    const auto centers = (scratch.path / "centers").string(),
               cids = (scratch.path / "cids").string();
    {
      std::ofstream out(centers, std::ios::binary);
      uint32_t dim = d;
      out.write((char *)&dim, 4);
      out.write((char *)vectors.data(), d * 4);
    }
    {
      std::ofstream out(cids, std::ios::binary);
      uint32_t one = 1, zero = 0;
      for (size_t i = 0; i < n; ++i) {
        out.write((char *)&one, 4);
        out.write((char *)&zero, 4);
      }
    }
    const auto graph = (scratch.path / "graph").string(), ml = (scratch.path / "ml").string();
    Index index(n, d, 8, 32, "l2", graph, 2, 2, false, 32);
    index.InitRaBitQ(centers, cids, "l2", 1);
    index.Build(vectors.data(), attrs.data(), n, 1);
    LayoutExportConfig cfg;
    cfg.block_size = 4096;
    cfg.vector_block_size = 8192;
    cfg.level_config = MultiLevelConfig{1, 1, 2, {}};
    cfg.method = ShuffleMethod::Identity;
    cfg.out_graph_path = ml;
    cfg.out_meta_path = ml + ".meta";
    ExportMultiLevelLayout(&index, cfg);
    index.LoadMultiLevelLayout(ml, ml + ".meta", 1024 * 1024, DynamicBlockCache::EvictPolicy::FIFO);
    index.ml_cache_auto_size_ = false;
    DiskSearchContext compatibility(32, graph, ml), owned(32, graph, ml), migrated(32, graph, ml);
    owned.UseOwnedWorkspace();
    migrated.UseOwnedWorkspace();
    for (auto *context : {&compatibility, &owned, &migrated}) {
      context->top_layer_postfilter = false;
      context->rerank_beam_width = 8;
    }
    Check(!compatibility.HasOwnedWorkspace() && owned.HasOwnedWorkspace(), "compatibility opt-in");
    for (auto mode : {DiskSearchPolicy::FilterMode::Graph, DiskSearchPolicy::FilterMode::Auto}) {
      for (bool clear_query : {false, true}) {
        index.per_query_cache_ = clear_query;
        compatibility.filter_mode = owned.filter_mode = migrated.filter_mode = mode;
        index.InvalidateMultiLevelCache();
        auto baseline_first = Query(index, compatibility, vectors.data() + 3 * d);
        auto baseline_second = Query(index, compatibility, vectors.data() + 3 * d);
        Same(baseline_first, Query(index, owned, vectors.data() + 3 * d));
        Same(baseline_second, Query(index, owned, vectors.data() + 3 * d));
        Observation threaded;
        std::exception_ptr error;
        std::thread worker([&] {
          try {
            threaded = Query(index, migrated, vectors.data() + 3 * d);
          } catch (...) {
            error = std::current_exception();
          }
        });
        worker.join();
        if (error)
          std::rethrow_exception(error);
        Same(baseline_first, threaded);
        Same(baseline_second, Query(index, migrated, vectors.data() + 3 * d));
        const auto memory = owned.WorkspaceMemory(&index, index.ml_cache_config_.generation);
        Check(memory.io_buffer_bytes > 0, "owned workspace not used by vector rerank");
      }
    }
    std::cout
        << "SEARCH_WORKER_PASS ownership, query/EF boundaries, routing, IO, cross-thread reuse\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
