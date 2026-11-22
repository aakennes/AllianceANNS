#include "index/index_disk.hpp"
#include "shuffle/layout.hpp"
#include "build/disk_build_geometry.hpp"
#include <filesystem>
#include <fstream>
#include <random>
#include <set>
#include <cmath>
#include <unistd.h>
#include "support/io_observer.hpp"
#include <chrono>

using namespace alliance;
using Index = AllianceIndexDisk<int, float>;
namespace fs = std::filesystem;
static void Check(bool ok, const char *why) {
  if (!ok)
    throw std::runtime_error(why);
}
static void Fvecs(const fs::path &p, const std::vector<float> &v, size_t d) {
  std::ofstream out(p, std::ios::binary);
  uint32_t dim = d;
  for (size_t i = 0; i < v.size(); i += d) {
    out.write((char *)&dim, 4);
    out.write((char *)(v.data() + i), d * 4);
  }
}

static IOCounters CheckCompletedIO() {
  const auto io = IOStop();
  Check(!io.errors && !io.unknown_fd && !io.bad_layout && !io.pending, "mixed AIO validation");
  Check(io.requests == io.completions && io.requested_bytes == io.completed_bytes,
        "mixed completion mismatch");
  Check(io.requested_bytes == 4096 * io.graph_requests + 8192 * io.vector_requests,
        "mixed byte count");
  return io;
}

static void Case(const fs::path &root, size_t d, size_t page, bool final_shape, size_t edge_page) {
  constexpr size_t n = 512, k = 10;
  const size_t m = final_shape ? 32 : 8, wp = final_shape ? 6 : 3;
  const auto dir =
      root / (std::to_string(d) + "_" + std::to_string(page) + "_edge" + std::to_string(edge_page));
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
  cfg.block_size = edge_page;
  cfg.vector_block_size = page;
  cfg.level_config = MultiLevelConfig{2, 2, 2, {}};
  cfg.method = ShuffleMethod::BlockBNF;
  auto ranges = GenerateLevelRanges(idx.cur_max_layer_, cfg.level_config);
  BNFParams params;
  params.block_size = edge_page;
  params.max_iterations = 4;
  params.threads = 1;
  cfg.level_config.per_level_params.assign(ranges.size(), params);
  cfg.out_graph_path = (dir / "ml").string();
  cfg.out_meta_path = (dir / "layout").string();
  auto layout = ExportMultiLevelLayout(&idx, cfg);
  Check(layout.vec_elements_per_block == page / (d * 4), "vector slots per block");
  Check(layout.layout_version == kMixedLayoutVersion, "mixed version missing");
  Check(layout.vector_block_size == page && layout.vector_region_offset_bytes % page == 0,
        "vector geometry");
  // Independently read and verify every record and raw vector, including tail slots.
  std::ifstream file(cfg.out_graph_path, std::ios::binary);
  GraphLayoutHeader h{};
  MixedVectorGeometry vg{};
  file.read((char *)&h, sizeof(h));
  file.read((char *)&vg, sizeof(vg));
  Check(h.version == 3 && h.block_size == 4096 && vg.block_size == 8192, "mixed header");
  std::vector<char> block(edge_page);
  std::vector<tableint> expected(m + 1);
  size_t checked = 0;
  for (const auto &lv : layout.levels) {
    Check(lv.elements_per_block == edge_page / lv.record_size, "neighbor packing");
    std::vector<bool> seen(n, false);
    for (size_t b = 0; b < lv.num_blocks; ++b) {
      const uint64_t head =
          ((sizeof(h) + sizeof(vg) + layout.levels.size() * sizeof(LevelHeaderEntry) +
            sizeof(VecHeaderEntry) + edge_page - 1) /
           edge_page) *
          edge_page;
      file.seekg(head + (lv.region_start_block + b) * edge_page);
      file.read(block.data(), edge_page);
      Check(bool(file), "neighbor block read");
      for (size_t slot = 0; slot < lv.elements_per_block; ++slot) {
        const char *rec = block.data() + slot * lv.record_size;
        tableint id;
        memcpy(&id, rec, 4);
        if (id == UINT32_MAX)
          continue;
        Check(id < n && !seen[id], "invalid/duplicate neighbor record");
        seen[id] = true;
        label_t label;
        int value;
        memcpy(&label, rec + 4, 8);
        memcpy(&value, rec + 12, 4);
        Check(label == idx.labels_[id] && value == idx.atts_[id], "label/attribute changed");
        for (layer_t l = lv.range.layer_begin; l <= lv.range.layer_end; ++l) {
          idx.ReadEdges(id, l, expected.data());
          Check(!memcmp(expected.data(), rec + 16 + (l - lv.range.layer_begin) * (m + 1) * 4,
                        (m + 1) * 4),
                "ordered edge changed");
        }
        ++checked;
      }
    }
    Check(std::count(seen.begin(), seen.end(), true) == n, "missing neighbor records");
  }
  std::vector<float> actual(d);
  for (size_t id = 0; id < n; ++id) {
    file.seekg(vg.region_offset_bytes + layout.vec_block_ids[id] * page +
               layout.vec_slots[id] * d * 4);
    file.read((char *)actual.data(), d * 4);
    idx.ReadVec(id, vec.data());
    Check(bool(file) && !memcmp(actual.data(), vec.data(), d * 4), "mixed vector changed");
  }
  printf("MIXED_FULL_READBACK records=%zu vectors=%zu\n", checked, n);
  // Fresh metadata load, async graph traversal, vector rerank, both cache modes.
  Index loaded(graph, meta, "l2");
  Check(loaded.node_disk_bytes_ == page, "metadata page roundtrip");
  loaded.LoadMultiLevelLayout(cfg.out_graph_path, cfg.out_meta_path, 8 * 1024 * 1024);
  std::vector<tableint> got(m + 1);
  for (size_t id = 0; id < n; ++id)
    for (layer_t l = 0; l <= wp; ++l) {
      idx.ReadEdges(id, l, expected.data());
      loaded.ReadEdgesMultiLevel(id, l, got.data());
      Check(got == expected, "production mixed edge readback");
    }
  auto prepared = (dir / "static_prepared.layout").string();
  loaded.PrepareStaticCacheSidecars(cfg.out_graph_path, cfg.out_meta_path, {{prepared, 64 * 1024}});
  const auto bad = cfg.out_graph_path + ".bad";
  fs::copy_file(cfg.out_graph_path, bad, fs::copy_options::overwrite_existing);
  {
    std::fstream f(bad, std::ios::in | std::ios::out | std::ios::binary);
    uint64_t invalid = 12288;
    f.seekp(sizeof(GraphLayoutHeader));
    f.write((char *)&invalid, 8);
  }
  bool bad_rejected = false;
  try {
    loaded.LoadMultiLevelLayout(bad, cfg.out_meta_path, 0);
  } catch (const std::exception &) {
    bad_rejected = true;
  }
  Check(bad_rejected, "invalid mixed vector block accepted");
  fs::remove(bad);
  fs::copy_file(cfg.out_graph_path, bad, fs::copy_options::overwrite_existing);
  fs::resize_file(bad, vg.region_offset_bytes + page);
  bad_rejected = false;
  try {
    loaded.LoadMultiLevelLayout(bad, cfg.out_meta_path, 0);
  } catch (const std::exception &) {
    bad_rejected = true;
  }
  Check(bad_rejected, "truncated mixed vector region accepted");
  fs::remove(bad);
  uint64_t total_graph = 0, total_vector = 0;
  double seconds = 0;
  for (auto policy : {DynamicBlockCache::EvictPolicy::FIFO, DynamicBlockCache::EvictPolicy::LRU}) {
    loaded.LoadMultiLevelLayout(cfg.out_graph_path, prepared, 8 * 1024 * 1024, policy);
    Check(loaded.ml_block_size_ == edge_page && loaded.ml_vec_block_size_ == page,
          "loaded mixed sizes");
    loaded.PrepareSearchCIDs();
    loaded.PackRaBitQForSearch();
    DiskSearchContext ctx(64, graph, cfg.out_graph_path);
    ctx.edge_beam_width = 8;
    ctx.rerank_beam_width = 8;
    ctx.rerank_num = n;
    loaded.per_query_cache_ = true;
    loaded.ml_cache_auto_size_ = true;
    for (size_t qi = 0; qi < 8; ++qi) {
      auto sync = loaded.searchKNN(base.data() + qi * d, n, k, filter);
      Check(sync == baseline[qi], "layout changed sync result");
      auto g = uint64_t(loaded.metric_ml_io_submits_),
           v = uint64_t(loaded.metric_ml_rerank_io_submits_);
      IOStart(cfg.out_graph_path.c_str(), vg.region_offset_bytes);
      auto begin = std::chrono::steady_clock::now();
      auto result = loaded.searchKNNMultiLevelPipeline(base.data() + qi * d, n, k, filter, &ctx);
      seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
      auto io = IOStop();
      Check(!io.errors && !io.unknown_fd && !io.bad_layout && !io.pending, "mixed AIO validation");
      Check(io.requests == io.completions && io.requested_bytes == io.completed_bytes,
            "mixed completion mismatch");
      Check(io.requested_bytes == 4096 * io.graph_requests + 8192 * io.vector_requests,
            "mixed byte count");
      Check(io.graph_requests == uint64_t(loaded.metric_ml_io_submits_) - g &&
                io.vector_requests == uint64_t(loaded.metric_ml_rerank_io_submits_) - v,
            "mixed native counters");
      total_graph += io.graph_requests;
      total_vector += io.vector_requests;
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
      std::vector<std::pair<double, size_t>> exact_top;
      for (size_t id = 0; id < n; ++id) {
        double distance = 0;
        for (size_t j = 0; j < d; ++j) {
          double delta = double(base[qi * d + j]) - base[id * d + j];
          distance += delta * delta;
        }
        exact_top.emplace_back(distance, id);
      }
      std::sort(exact_top.begin(), exact_top.end());
      size_t hits = 0;
      for (auto [distance, label] : result)
        for (size_t j = 0; j < k; ++j)
          hits += label == exact_top[j].second;
      Check(hits == k, "mixed exhaustive recall below one");
      Check(result.front().second == qi && result.front().first == 0,
            "self nearest neighbor missing");
    }
  }
  Check(total_graph > 0 && total_vector > 0, "both read paths must execute");

  // Exercise final exact and bounded vector requests with 8 KiB vector pages.
  DiskSearchContext ctx(64, graph, cfg.out_graph_path);
  ctx.edge_beam_width = 8;
  ctx.rerank_beam_width = 8;
  ctx.rerank_num = n;
  ctx.filter_mode = DiskSearchContext::FilterMode::Graph;
  const auto exact_before = uint64_t(loaded.metric_ml_post_rerank_io_submits_);
  IOStart(cfg.out_graph_path.c_str(), vg.region_offset_bytes);
  const auto exact_results = loaded.searchKNNMultiLevelPipeline(base.data(), n, k, filter, &ctx);
  const auto exact_io = CheckCompletedIO();
  Check(uint64_t(loaded.metric_ml_post_rerank_io_submits_) > exact_before,
        "exact mixed vector requests were not exercised");
  Check(exact_io.vector_requests > 0 && exact_results.size() == k,
        "exact mixed results/reads missing");
  for (const auto &[distance, label] : exact_results) {
    Check(label < n, "exact invalid label");
    double exact = 0;
    for (size_t j = 0; j < d; ++j) {
      const double delta = double(base[j]) - base[label * d + j];
      exact += delta * delta;
    }
    Check(std::abs(distance - exact) < 1e-3 * std::max(1.0, exact),
          "exact mixed distance mismatch");
  }
  Check(exact_results.front().second == 0 && exact_results.front().first == 0,
        "exact self nearest neighbor missing");

  // k=n and no prescreen require every candidate's exact distance, so this
  // checks bounded-rerank IO geometry without relying on probabilistic pruning.
  Index::RaBitQEstimator estimator(base.data(), &loaded);
  std::vector<dist_id_pair> candidates;
  candidates.reserve(n);
  for (size_t id = 0; id < n; ++id)
    candidates.emplace_back(estimator.dist_estimate(loaded.SearchCID(id), id), id);
  std::vector<std::pair<dist_t, label_t>> bounded_results;
  ctx.rerank_bound_prescreen = false;
  IOStart(cfg.out_graph_path.c_str(), vg.region_offset_bytes);
  loaded.RerankMultiLevelBounded(base.data(), candidates, n, estimator, &ctx, bounded_results);
  const auto bounded_io = CheckCompletedIO();
  Check(bounded_io.graph_requests == 0 && bounded_io.vector_requests > 0 &&
            bounded_results.size() == n,
        "bounded mixed vector coverage");
  std::set<label_t> bounded_labels;
  for (const auto &[distance, label] : bounded_results) {
    Check(label < n && bounded_labels.insert(label).second, "bounded duplicate/invalid label");
    double exact = 0;
    for (size_t j = 0; j < d; ++j) {
      const double delta = double(base[j]) - base[label * d + j];
      exact += delta * delta;
    }
    Check(std::abs(distance - exact) < 1e-3 * std::max(1.0, exact),
          "bounded mixed distance mismatch");
  }
  printf("MIXED_RERANK_PATHS_PASS exact_vector8k=%llu bounded_vector8k=%llu\n",
         (unsigned long long)exact_io.vector_requests,
         (unsigned long long)bounded_io.vector_requests);
  printf(
      "MIXED_IO_VERIFIED graph4k=%llu vector8k=%llu errors=0 recall=1 queries=16 search_seconds=%.9f qps=%.3f\n",
      (unsigned long long)total_graph, (unsigned long long)total_vector, seconds, 16 / seconds);
  printf("VARIABLE_PAGE_PASS dim=%zu node=%zu buildvec=%zu wp=%zu vectors=%zu async_queries=16\n",
         d, page, idx.VecPageBytes(), wp, n);
}
int main() {
  try {
    const char *tmp = std::getenv("TEST_TMPDIR");
    fs::path root = fs::path(tmp ? tmp : "/tmp") / ("mixed_pages_" + std::to_string(getpid()));
    fs::create_directory(root);
    Check(disk_build::PlanNodeStorage(3, 8, 2304 * 4, 0).page_bytes == 16384,
          "automatic power-of-two geometry");
    Case(root, 1536, 8192, true, 4096);
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
    printf("MIXED_PAGES_PASS n=512 dim=1536 M=32 wp=6 edges=4096 vectors=8192 queries=16\n");
    return 0;
  } catch (const std::exception &e) {
    fprintf(stderr, "MIXED_PAGES_FAIL %s\n", e.what());
    return 1;
  }
}
