#include "index/index_disk.hpp"
#include "shuffle/layout.hpp"
#include "build/disk_build_geometry.hpp"
#include <filesystem>
#include <fstream>
#include <random>
#include <set>
#include <cmath>
#include <limits>
#include <utility>
#include <unistd.h>

using namespace alliance;
using Index = AllianceIndexDisk<int, float>;
namespace fs = std::filesystem;
static void Check(bool ok, const char *why) {
  if (!ok)
    throw std::runtime_error(why);
}
template <typename Error, typename Function>
static void GeometryError(Function fn, const char *message) {
  bool rejected = false;
  try {
    fn();
  } catch (const Error &error) {
    rejected = std::string(error.what()) == message;
  }
  Check(rejected, "geometry exception contract changed");
}

static void GeometryBoundaries() {
  // These are format boundaries, not a second implementation of the planner.
  const std::pair<size_t, size_t> vector_pages[] = {{1, 4096},     {4095, 4096}, {4096, 4096},
                                                    {4097, 8192},  {8192, 8192}, {8193, 16384},
                                                    {12800, 16384}};
  for (const auto [bytes, page] : vector_pages) {
    Check(disk_build::BuildVectorPageBytes(bytes) == page, "vector page boundary");
    Check(Index::BuildVectorPageBytes(bytes) == page, "index vector geometry compatibility");
  }
  const size_t limit = std::numeric_limits<size_t>::max();
  const size_t largest_page = size_t{1} << (std::numeric_limits<size_t>::digits - 1);
  Check(disk_build::BuildVectorPageBytes(largest_page) == largest_page,
        "largest representable vector page");
  GeometryError<std::invalid_argument>([] { disk_build::BuildVectorPageBytes(0); },
                                       "invalid build vector size");
  GeometryError<std::overflow_error>([&] { disk_build::BuildVectorPageBytes(largest_page + 1); },
                                     "build vector page size overflow");
  for (size_t requested : {size_t{0}, size_t{4096}}) {
    const auto exact = disk_build::PlanNodeStorage(0, 0, 4092, requested);
    Check(exact.record_bytes == 4096 && exact.page_bytes == 4096, "exact node page boundary");
  }
  const auto above = disk_build::PlanNodeStorage(0, 0, 4093, 0);
  Check(above.record_bytes == 4097 && above.page_bytes == 8192, "node page growth boundary");
  GeometryError<std::runtime_error>([] { disk_build::PlanNodeStorage(0, 0, 4093, 4096); },
                                    "Node record (4097B) exceeds page size (4096B)");
  GeometryError<std::runtime_error>(
      [] { disk_build::PlanNodeStorage(0, 0, 4, 12288); },
      "node_page_bytes (12288B) must be a power of two and a multiple of 4096B");
  GeometryError<std::runtime_error>([&] { disk_build::PlanNodeStorage(limit, 0, 0, 0); },
                                    "Node record geometry overflow");
  GeometryError<std::runtime_error>([&] { disk_build::PlanNodeStorage(0, limit, 0, 0); },
                                    "Node record geometry overflow");
  GeometryError<std::runtime_error>([&] { disk_build::PlanNodeStorage(1, 0, limit, 0); },
                                    "Node record geometry overflow");
  GeometryError<std::runtime_error>([&] { disk_build::PlanNodeStorage(0, 0, largest_page - 3, 0); },
                                    "Node page size overflow");
}

static void Fvecs(const fs::path &p, const std::vector<float> &v, size_t d) {
  std::ofstream out(p, std::ios::binary);
  uint32_t dim = d;
  for (size_t i = 0; i < v.size(); i += d) {
    out.write((char *)&dim, 4);
    out.write((char *)(v.data() + i), d * 4);
  }
}

static void Case(const fs::path &root, size_t d, size_t page, bool final_shape) {
  constexpr size_t n = 512, k = 10;
  const size_t m = final_shape ? 32 : 8, wp = final_shape ? 6 : 3;
  const auto dir = root / (std::to_string(d) + "_" + std::to_string(page));
  fs::create_directory(dir);
  std::mt19937 rng(1234);
  std::normal_distribution<float> normal;
  std::vector<float> base(n * d);
  for (auto &x : base)
    x = normal(rng);
  std::vector<int> attr(n);
  for (size_t i = 0; i < n; ++i)
    attr[i] = int(n - 1 - i);
  std::vector<float> centers(2 * d);
  std::copy_n(base.data(), 2 * d, centers.data());
  Fvecs(dir / "centers", centers, d);
  {
    std::ofstream f(dir / "cids", std::ios::binary);
    uint32_t one = 1, zero = 0;
    for (size_t i = 0; i < n; ++i) {
      f.write((char *)&one, 4);
      f.write((char *)&zero, 4);
    }
  }
  auto graph = (dir / "graph").string(), meta = (dir / "meta").string();
  Index idx(n, d, m, 64, "l2", graph, 4, wp, false, 32, page, WindowAnchor::kTop);
  idx.curvec_num_ = n;
  idx.InitRaBitQ((dir / "centers").string(), (dir / "cids").string(), "l2", 3);
  idx.build_config_.left_stream_build_ = true;
  idx.build_config_.left_stream_batch_ = 16;
  idx.build_config_.wow_threshold_ = 256;
  idx.build_config_.wow_efc_ = 64;
  idx.build_config_.build_mem_gb_ = 1;
  idx.build_config_.cluster_K_ = 2;
  idx.build_config_.cluster_wow_efc_ = 64;
  idx.Build(base.data(), attr.data(), n, 2);
  idx.Save(meta);
  idx.graph_file_->Sync();
  Check(idx.node_disk_bytes_ == page, "node page changed");
  Check(fs::file_size(graph) == idx.header_bytes_ + n * page, "graph length");
  Check(fs::file_size(graph + ".buildvecs") ==
            ((n + idx.vecs_per_page_ - 1) / idx.vecs_per_page_) * idx.VecPageBytes(),
        "buildvec length");
  // Exhaustively compare raw vectors and gather using scattered/duplicate ranks.
  std::vector<float> vec(d), gathered(n * d);
  std::vector<uint32_t> ranks(n);
  for (size_t id = 0; id < n; ++id) {
    idx.ReadVec(id, vec.data());
    Check(!memcmp(vec.data(), base.data() + idx.labels_[id] * d, d * 4), "node vector changed");
    ranks[id] = (id * 37) % n;
  }
  IOHandler io(64);
  io.Open(graph + ".buildvecs");
  idx.GatherVecsByRank(io, ranks.data(), nullptr, n, gathered.data());
  for (size_t i = 0; i < n; ++i)
    Check(!memcmp(gathered.data() + i * d, base.data() + idx.labels_[idx.sorted_ids_[ranks[i]]] * d,
                  d * 4),
          "scattered gather changed vector");
  std::vector<std::vector<std::pair<dist_t, label_t>>> baseline;
  attr_range<int> filter{0, int(n - 1)};
  for (size_t qi = 0; qi < 8; ++qi)
    baseline.push_back(idx.searchKNN(base.data() + qi * d, n, k, filter));
  LayoutExportConfig cfg;
  cfg.block_size = page;
  cfg.level_config = MultiLevelConfig{2, 2, 2, {}};
  cfg.method = ShuffleMethod::BlockBNF;
  cfg.out_graph_path = (dir / "ml").string();
  cfg.out_meta_path = (dir / "layout").string();
  auto layout = ExportMultiLevelLayout(&idx, cfg);
  const auto cached_meta = cfg.out_meta_path + ".cached";
  idx.PrepareStaticCacheSidecars(cfg.out_graph_path, cfg.out_meta_path, {{cached_meta, page * 8}});
  Check(layout.vec_elements_per_block == page / (d * 4), "vector slots per block");
  // Fresh metadata load, async graph traversal, vector rerank, both cache modes.
  Index loaded(graph, meta, "l2");
  Check(loaded.node_disk_bytes_ == page, "metadata page roundtrip");
  for (auto policy : {DynamicBlockCache::EvictPolicy::FIFO, DynamicBlockCache::EvictPolicy::LRU}) {
    loaded.LoadMultiLevelLayout(cfg.out_graph_path, cached_meta, page * 32, policy);
    DiskSearchContext ctx(64, graph, cfg.out_graph_path);
    ctx.edge_beam_width = 8;
    ctx.rerank_num = n;
    for (size_t qi = 0; qi < 8; ++qi) {
      auto sync = loaded.searchKNN(base.data() + qi * d, n, k, filter);
      Check(sync == baseline[qi], "layout changed sync result");
      auto result = loaded.searchKNNMultiLevelPipeline(base.data() + qi * d, n, k, filter, &ctx);
      std::set<label_t> seen;
      Check(result.size() == k, "short async result");
      for (auto [distance, label] : result) {
        Check(label < n && seen.insert(label).second, "invalid/duplicate async ID");
        double exact = 0;
        for (size_t j = 0; j < d; ++j) {
          double delta = double(base[qi * d + j]) - base[label * d + j];
          exact += delta * delta;
        }
        Check(std::abs(distance - exact) < 1e-3 * std::max(1.0, exact), "incorrect async distance");
      }
      Check(result.front().second == qi && result.front().first == 0,
            "self nearest neighbor missing");
    }
  }
  printf("VARIABLE_PAGE_PASS dim=%zu node=%zu buildvec=%zu wp=%zu vectors=%zu async_queries=16\n",
         d, page, idx.VecPageBytes(), wp, n);
}
int main() {
  try {
    GeometryBoundaries();
    const char *tmp = std::getenv("TEST_TMPDIR");
    fs::path root = fs::path(tmp ? tmp : "/tmp") / ("variable_pages_" + std::to_string(getpid()));
    fs::create_directory(root);
    Check(disk_build::PlanNodeStorage(3, 8, 2304 * 4, 0).page_bytes == 16384,
          "automatic power-of-two geometry");
    Case(root, 128, 4096, false);
    Case(root, 1536, 8192, true);
    Case(root, 1536, 16384, false);
    Case(root, 2304, 16384, false);
    Case(root, 3200, 16384, false);
    // Revisit small pages on the same worker thread after the largest gather.
    Case(root, 256, 8192, false);
    bool rejected = false;
    try {
      Index bad(128, 1536, 32, 64, "l2", (root / "bad").string(), 4, 6, false, 32, 4096,
                WindowAnchor::kTop);
    } catch (const std::exception &) {
      rejected = true;
    }
    Check(rejected && !fs::exists(root / "bad"), "undersized page accepted/touched output");
    rejected = false;
    try {
      Index bad(128, 128, 8, 32, "l2", (root / "bad12").string(), 4, 3, false, 32, 12288,
                WindowAnchor::kTop);
    } catch (const std::exception &) {
      rejected = true;
    }
    Check(rejected && !fs::exists(root / "bad12"), "12 KiB page accepted/touched output");
    printf("VARIABLE_PAGES_PASS cases=6\n");
    return 0;
  } catch (const std::exception &e) {
    fprintf(stderr, "VARIABLE_PAGES_FAIL %s\n", e.what());
    return 1;
  }
}
