#include "index/index_disk.hpp"
#include "shuffle/layout.hpp"
#include "benchmark/io_observer.hpp"
#include "benchmark/layout.hpp"
#include "search/search_disk_options.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <unistd.h>

using namespace alliance;
using Index = AllianceIndexDisk<int, float>;
namespace fs = std::filesystem;
static void Check(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
template <class Action> static void Reject(Action action) {
  bool rejected = false;
  try {
    action();
  } catch (const std::exception &) {
    rejected = true;
  }
  Check(rejected, "unavailable storage or malformed asset was accepted");
}
template <class Action> static void RejectRawWriter(Action action) {
  bool rejected = false;
  try {
    action();
  } catch (const std::runtime_error &error) {
    rejected =
        std::string(error.what()).find("requires raw graph storage; unavailable in ML-only mode") !=
        std::string::npos;
  }
  Check(rejected, "ML-only writer did not reject raw storage before doing work");
}
using FileSnapshot = std::map<std::string, std::vector<char>>;
static FileSnapshot Snapshot(const fs::path &directory) {
  FileSnapshot files;
  for (const auto &entry : fs::recursive_directory_iterator(directory))
    if (entry.is_regular_file()) {
      std::ifstream input(entry.path(), std::ios::binary);
      Check(bool(input), "cannot snapshot fixture file");
      files.emplace(fs::relative(entry.path(), directory).generic_string(),
                    std::vector<char>(std::istreambuf_iterator<char>(input),
                                      std::istreambuf_iterator<char>()));
      Check(!input.bad(), "cannot read fixture snapshot");
    }
  return files;
}
struct Scratch {
  fs::path path;
  Scratch() {
    const char *tmp = std::getenv("TEST_TMPDIR");
    auto pattern = ((tmp ? fs::path(tmp) : fs::current_path()) / "ml-only-XXXXXX").string();
    std::vector<char> name(pattern.begin(), pattern.end());
    name.push_back(0);
    Check(mkdtemp(name.data()) != nullptr, "mkdtemp failed");
    path = name.data();
  }
  ~Scratch() {
    std::error_code error;
    fs::remove_all(path, error);
  }
};
struct Observation {
  std::vector<std::pair<dist_t, label_t>> results;
  uint64_t graph = 0, vectors = 0, bytes = 0;
};
static std::vector<Observation> Run(Index &index, DiskSearchContext &ctx, const std::string &ml,
                                    const std::vector<float> &base, size_t dim, size_t beam,
                                    bool prefilter_only = false) {
  const auto geometry = benchmark::formal::ReadGeometry(ml, index);
  ctx.UseOwnedWorkspace();
  ctx.edge_beam_width = ctx.rerank_beam_width = beam;
  ctx.top_layer_postfilter = false;
  ctx.prefilter_rerank = prefilter_only ? base.size() / dim : 0;
  index.ml_cache_auto_size_ = false;
  std::vector<Observation> observations;
  const std::vector<DiskSearchContext::FilterMode> routes =
      prefilter_only
          ? std::vector<DiskSearchContext::FilterMode>{DiskSearchContext::FilterMode::Prefilter}
          : std::vector<DiskSearchContext::FilterMode>{DiskSearchContext::FilterMode::Graph,
                                                       DiskSearchContext::FilterMode::Auto,
                                                       DiskSearchContext::FilterMode::Prefilter};
  for (auto mode : routes) {
    const std::vector<bool> bounds =
        prefilter_only ? std::vector<bool>{false} : std::vector<bool>{false, true};
    for (bool bounded : bounds)
      for (bool per_query : {false, true}) {
        ctx.filter_mode = mode;
        ctx.rerank_error_bound = bounded;
        ctx.rerank_bound_prescreen = bounded;
        ctx.rerank_candidate_factor = bounded ? 2 : 0;
        ctx.rerank_num = bounded ? 0 : 40;
        ctx.rerank_batch_blocks = 8;
        index.per_query_cache_ = per_query;
        index.InvalidateMultiLevelCache();
        for (size_t query = 0; query < 2; ++query) {
          ResetMultiLevelMetrics(index);
          IOStart(ml.c_str(), geometry.vector_start, geometry.edge_bytes, geometry.vector_bytes,
                  geometry.header_bytes, geometry.file_bytes);
          auto results = index.searchKNNMultiLevelPipeline(
              const_cast<float *>(base.data() + 3 * dim), 64, 5, attr_range<int>{0, 127}, &ctx);
          const auto io = IOStop();
          Check(!io.errors && !io.pending && !io.unknown_fd && !io.bad_layout,
                "ML-only AIO integrity");
          Check(io.requests == io.completions && io.requested_bytes == io.completed_bytes,
                "ML-only completions");
          Check(io.graph_requests == index.metric_ml_io_submits_ &&
                    io.vector_requests == index.metric_ml_rerank_io_submits_,
                "ML-only observer/native mismatch");
          Check(io.requested_bytes == geometry.edge_bytes * io.graph_requests +
                                          geometry.vector_bytes * io.vector_requests,
                "ML-only read units changed");
          Check(results.size() == 5, "ML-only short result");
          observations.push_back(
              {std::move(results), io.graph_requests, io.vector_requests, io.requested_bytes});
        }
      }
  }
  return observations;
}
static void SameResults(const std::vector<Observation> &expected,
                        const std::vector<Observation> &actual) {
  Check(expected.size() == actual.size(), "ML-only query coverage changed");
  for (size_t i = 0; i < actual.size(); ++i)
    Check(expected[i].results == actual[i].results, "ML-only IDs/distances changed");
}
static void Same(const std::vector<Observation> &expected, const std::vector<Observation> &actual) {
  SameResults(expected, actual);
  for (size_t i = 0; i < actual.size(); ++i) {
    if (expected[i].graph != actual[i].graph || expected[i].vectors != actual[i].vectors ||
        expected[i].bytes != actual[i].bytes)
      throw std::runtime_error(
          "ML-only logical reads changed at observation " + std::to_string(i) + ": graph " +
          std::to_string(expected[i].graph) + " -> " + std::to_string(actual[i].graph) +
          ", vectors " + std::to_string(expected[i].vectors) + " -> " +
          std::to_string(actual[i].vectors) + ", bytes " + std::to_string(expected[i].bytes) +
          " -> " + std::to_string(actual[i].bytes));
  }
}
static void Case(const fs::path &root, size_t dim, size_t vector_page) {
  const auto dir = root / ("d" + std::to_string(dim));
  fs::create_directory(dir);
  const auto graph = (dir / "graph.bin").string(), meta = (dir / "index.meta").string();
  const auto raw = (dir / "vectors.raw").string(), ml = (dir / "ml.bin").string(),
             sidecar = (dir / "ml.meta").string();
  const auto edge_only = (dir / "edges.bin").string(), edge_meta = (dir / "edges.meta").string();
  const auto centers = (dir / "centers.fvecs").string(), cids = (dir / "cids.ivecs").string();
  constexpr size_t n = 128;
  std::vector<float> base(n * dim);
  std::vector<int> attrs(n);
  for (size_t i = 0; i < n; ++i) {
    attrs[i] = int(n - 1 - i);
    for (size_t j = 0; j < dim; ++j)
      base[i * dim + j] = std::sin(float(i * 23 + j * 17) * .017f) + float(i % 7) * .03f;
  }
  {
    std::ofstream out(centers, std::ios::binary);
    uint32_t d = dim;
    out.write((char *)&d, 4);
    out.write((char *)base.data(), dim * 4);
  }
  {
    std::ofstream out(cids, std::ios::binary);
    uint32_t one = 1, zero = 0;
    for (size_t i = 0; i < n; ++i) {
      out.write((char *)&one, 4);
      out.write((char *)&zero, 4);
    }
  }
  {
    Index build(n, dim, 8, 32, "l2", graph, 2, 2, false, 32, vector_page);
    build.InitRaBitQ(centers, cids, "l2", 1);
    build.vec_file_path_ = raw;
    build.Build(base.data(), attrs.data(), n, 1);
    build.WriteVecFileFromDisk();
    build.Save(meta);
    build.graph_file_->Sync();
    Check(fs::exists(raw), "fixture raw vector file missing");
    LayoutExportConfig config;
    config.block_size = 4096;
    config.vector_block_size = vector_page;
    config.level_config = MultiLevelConfig{1, 1, 2, {}};
    config.method = ShuffleMethod::Identity;
    const auto bare = sidecar + ".bare";
    config.out_graph_path = ml;
    config.out_meta_path = bare;
    ExportMultiLevelLayout(&build, config);
    build.PrepareStaticCacheSidecars(ml, bare, {{sidecar, 4096}});
    // Legacy edge-only layout is a negative reader fixture, not a writer mode.
    fs::copy_file(ml, edge_only);
    fs::copy_file(bare, edge_meta);
    {
      std::fstream file(edge_only, std::ios::binary | std::ios::in | std::ios::out);
      uint64_t zero = 0;
      file.seekp(offsetof(GraphLayoutHeader, vec_region_enabled));
      file.write((char *)&zero, 8);
    }
  }
  std::vector<Observation> expected, async_expected;
  {
    Index ordinary(graph, meta, "l2");
    ordinary.LoadMultiLevelLayout(ml, sidecar, 4096);
    ordinary.PrepareSearchCIDs();
    ordinary.PackRaBitQForSearch();
    DiskSearchContext ctx(16, graph, ml);
    Check(ordinary.graph_file_ && ctx.io_handler && ordinary.vec_file_available_,
          "default raw capability changed");
    // Full-pool, bound-off prefilter has deterministic grouped vector reads.
    // Graph traversal may expand ready cached nodes while IO is pending even
    // at beam 1, so its completion schedule is validated separately below.
    expected = Run(ordinary, ctx, ml, base, dim, 1, true);
    async_expected = Run(ordinary, ctx, ml, base, dim, 8);
  }
  // No descriptors to the original files survive the scopes above. Remove the
  // names from their recorded locations only inside this disposable fixture.
  fs::rename(graph, graph + ".held");
  fs::rename(raw, raw + ".held");
  Reject([&] { Index ordinary(graph, meta, "l2"); });
  Reject([&] { DiskSearchContext ordinary(16, graph, ml); });
  Reject([&] { Index missing("", meta + ".missing", "l2", StorageOpenMode::MultiLevelOnly); });
  Reject([&] { DiskSearchContext missing(16, "", "", StorageOpenMode::MultiLevelOnly); });
  Index independent("", meta, "l2", StorageOpenMode::MultiLevelOnly);
  DiskSearchContext ctx(16, "", ml, StorageOpenMode::MultiLevelOnly);
  ctx.filter_mode = DiskSearchContext::FilterMode::Graph;
  Check(!independent.graph_file_ && !independent.vec_file_available_ && !ctx.io_handler &&
            ctx.ml_io_handler,
        "ML-only opened raw storage");
  Reject([&] {
    independent.searchKNNMultiLevelPipeline(base.data(), 64, 5, attr_range<int>{0, 127}, &ctx);
  });
  independent.LoadMultiLevelLayout(ml, sidecar, 4096);
  independent.PrepareSearchCIDs();
  independent.PackRaBitQForSearch();
  Same(expected, Run(independent, ctx, ml, base, dim, 1, true));
  // At beam 8, completion order affects cache eviction and in-flight vector
  // dedup even for repeated raw-mode runs. Run validates every actual IO
  // against the observer/native counters; the result comparison stays exact.
  SameResults(async_expected, Run(independent, ctx, ml, base, dim, 8));
  ctx.filter_mode = DiskSearchContext::FilterMode::Legacy;
  Reject([&] {
    independent.searchKNNMultiLevelPipeline(base.data(), 64, 5, attr_range<int>{0, 127}, &ctx);
  });
  Reject([&] { independent.searchKNN(base.data(), 64, 5, attr_range<int>{0, 127}); });
  Reject([&] { independent.searchKNNPipeline(base.data(), 64, 5, attr_range<int>{0, 127}, &ctx); });
  Reject([&] { independent.BruteForceSearch(base.data(), 5, attr_range<int>{0, 127}); });
  Reject([&] { independent.OpenVecFile(); });
  std::vector<float> vector(dim);
  std::vector<tableint> edges(9);
  Reject([&] { independent.ReadVec(0, vector.data()); });
  Reject([&] { independent.ReadEdges(0, 0, edges.data()); });
  Reject([&] { independent.RequantizeRaBitQ(centers, cids, "l2", 1); });
  LayoutExportConfig denied_export;
  denied_export.out_graph_path = (dir / "denied.ml").string();
  denied_export.out_meta_path = (dir / "denied.meta").string();
  // Check absent outputs and existing sentinel contents. Even the public
  // population/resume and precomputed-export entry points must fail first.
  for (bool existing : {false, true}) {
    if (existing)
      for (const auto &path : {raw, denied_export.out_graph_path, denied_export.out_meta_path}) {
        std::ofstream output(path, std::ios::binary);
        output << "ML-only writer sentinel";
        Check(bool(output), "cannot create writer sentinel");
      }
    const auto before = Snapshot(dir);
    RejectRawWriter([&] { independent.WriteVecFile(base.data()); });
    RejectRawWriter([&] { independent.WriteVecFileFromDisk(); });
    RejectRawWriter([&] { independent.WriteBuildVecFile(base.data(), n); });
    RejectRawWriter([&] { independent.WriteBuildVecFileFromGraph(n); });
    RejectRawWriter([&] { independent.Build(base.data(), attrs.data(), n, 1); });
    RejectRawWriter([&] { independent.BuildPostPopulate(1); });
    RejectRawWriter([&] { independent.PopulateElements(base.data(), attrs.data(), n); });
    RejectRawWriter([&] { independent.PopulateElements(centers, attrs.data(), n); });
    RejectRawWriter([&] { ExportMultiLevelLayout(&independent, denied_export); });
    RejectRawWriter([&] {
      ExportMultiLevelLayout(&independent, denied_export, std::vector<LevelShuffleResult>{});
    });
    Check(before == Snapshot(dir), "ML-only rejection created or modified fixture files");
    if (existing)
      for (const auto &path : {raw, denied_export.out_graph_path, denied_export.out_meta_path})
        fs::remove(path);
  }
  Check(!fs::exists(graph) && !fs::exists(raw), "ML-only rejection recreated raw storage");
  auto bad_layout = [&](const std::string &path, const std::string &mappings) {
    Reject([&] {
      Index bad("", meta, "l2", StorageOpenMode::MultiLevelOnly);
      bad.LoadMultiLevelLayout(path, mappings, 4096);
    });
  };
  bad_layout(ml + ".missing", sidecar);
  bad_layout(ml, sidecar + ".missing");
  bad_layout(edge_only, edge_meta);
  const auto bad_graph = ml + ".bad";
  fs::copy_file(ml, bad_graph);
  {
    std::fstream file(bad_graph, std::ios::binary | std::ios::in | std::ios::out);
    uint64_t zero = 0;
    file.write((char *)&zero, 8);
  }
  bad_layout(bad_graph, sidecar);
  const auto truncated = sidecar + ".short";
  fs::copy_file(sidecar, truncated);
  fs::resize_file(truncated, fs::file_size(truncated) - 1);
  bad_layout(ml, truncated);
  const auto trailing = sidecar + ".trailing";
  fs::copy_file(sidecar, trailing);
  {
    std::ofstream out(trailing, std::ios::binary | std::ios::app);
    out.put(0);
  }
  bad_layout(ml, trailing);
  const auto bad_map = sidecar + ".bad";
  fs::copy_file(sidecar, bad_map);
  {
    std::fstream file(bad_map, std::ios::binary | std::ios::in | std::ios::out);
    int32_t invalid = -1;
    file.seekp(32);
    file.write((char *)&invalid, 4);
  }
  bad_layout(ml, bad_map);
  std::cout << "ML_ONLY_PASS dimension=" << dim << " neighbors=4096 vectors=" << vector_page
            << " compared_queries=" << expected.size() + async_expected.size()
            << " exact_io_queries=" << expected.size()
            << " async_integrity_queries=" << async_expected.size() << " exact_io_route=prefilter"
            << " raw_graph_and_rawvec_absent=1 rejected_write_attempts=20 outputs_unchanged=1\n";
}
static void Options() {
  auto parse = [](std::vector<std::string> arguments) {
    std::vector<char *> argv;
    for (auto &text : arguments)
      argv.push_back(text.data());
    benchmark::DiskSearchOptions options;
    return options.ParseAndValidate(argv.size(), argv.data());
  };
  std::vector<std::string> valid = {"search",          "--ml_only", "--meta_file",    "meta",
                                    "--query_vec",     "query",     "--query_rng",    "range",
                                    "--ml_graph_file", "ml",        "--ml_meta_file", "sidecar",
                                    "--pipeline",      "1",         "--filter_mode",  "graph"};
  Check(parse(valid), "valid ML-only options rejected");
  auto invalid = valid;
  invalid.insert(invalid.end(), {"--memory"});
  bool retired_memory_rejected = false;
  try {
    parse(invalid);
  } catch (const std::invalid_argument &error) {
    retired_memory_rejected =
        std::string(error.what()).find("retired: --memory") != std::string::npos;
  }
  Check(retired_memory_rejected, "retired memory option did not fail explicitly");
  invalid = valid;
  invalid.insert(invalid.end(), {"--pipeline", "0"});
  Check(!parse(invalid), "ML-only sync accepted");
  invalid = valid;
  invalid.insert(invalid.end(), {"--filter_mode", "legacy"});
  Check(!parse(invalid), "ML-only legacy accepted");
  invalid = valid;
  invalid.insert(invalid.end(), {"--rabitq_centroids", "centers", "--rabitq_cids", "cids"});
  Check(!parse(invalid), "ML-only raw requantization accepted");
}
int main() {
  try {
    Options();
    Scratch scratch;
    Case(scratch.path, 64, 4096);
    Case(scratch.path, 1536, 8192);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "ML_ONLY_FAIL " << error.what() << '\n';
    return 1;
  }
}
