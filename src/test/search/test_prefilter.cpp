#include "index/index_disk.hpp"
#include "shuffle/layout.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <unistd.h>
#include <omp.h>
#include <exception>

using namespace alliance;
using Index = AllianceIndexDisk<int, float>;
using Result = std::vector<std::pair<dist_t, label_t>>;
using Mode = DiskSearchContext::FilterMode;
static void Check(bool yes, const char *message) {
  if (!yes)
    throw std::runtime_error(message);
}
template <class F> void Reject(F f) {
  bool failed = false;
  try {
    f();
  } catch (const std::exception &) {
    failed = true;
  }
  Check(failed, "expected rejected request or failed IO");
}
template <class T> void WriteVecs(const std::string &path, const std::vector<T> &data, size_t d) {
  std::ofstream f(path, std::ios::binary);
  int dim = static_cast<int>(d);
  for (size_t i = 0; i < data.size(); i += d) {
    f.write(reinterpret_cast<char *>(&dim), 4);
    f.write(reinterpret_cast<const char *>(data.data() + i), d * sizeof(T));
  }
  Check(bool(f), "write fixture");
}
static void Equal(const Result &a, const Result &b) {
  Check(a.size() == b.size(), "result count");
  std::set<label_t> unique;
  for (size_t i = 0; i < a.size(); ++i) {
    Check(unique.insert(a[i].second).second, "duplicate output");
    Check(a[i].second == b[i].second, "oracle label mismatch");
    Check(std::abs(a[i].first - b[i].first) < 0.002f, "oracle distance mismatch");
  }
}
// Exercise rerank in another translation unit under an outer metrics scope.
// k equals the candidate count so bounded rerank cannot prune; the independent
// scalar oracle checks vector offsets, labels and shared buffer reuse exactly.
static void CheckRerankModules(Index &idx, DiskSearchContext &ctx, const std::vector<float> &query,
                               const std::vector<float> &base, size_t d, const std::string &space,
                               bool short_read = false) {
  Index::RaBitQEstimator estimator(query.data(), &idx);
  std::vector<dist_id_pair> candidates;
  std::set<size_t> blocks;
  Result expected;
  for (size_t i = 0; i < 43; ++i) {
    const tableint id = (i * 97) % idx.curvec_num_;
    const auto label = idx.labels_[id];
    candidates.emplace_back(estimator.dist_estimate(idx.cids_[label], id), id);
    blocks.insert(idx.ml_vec_block_ids_[id]);
    float distance = 0;
    for (size_t j = 0; j < d; ++j) {
      const float value = base[label * d + j];
      if (space == "l2") {
        const float diff = query[j] - value;
        distance += diff * diff;
      } else
        distance += query[j] * value;
    }
    expected.emplace_back(space == "l2" ? distance : 1 - distance, label);
  }
  std::sort(candidates.begin(), candidates.end());
  std::sort(expected.begin(), expected.end());
  Check(blocks.size() > 16, "fixture must span multiple IO waves");
  const auto saved_width = ctx.rerank_beam_width;
  for (size_t width : {1, 3, 16}) {
    ctx.rerank_beam_width = width;
    ResetMultiLevelMetrics(idx);
    size_t expected_io = 0;
    {
      ScopedMultiLevelMetrics outer(idx);
      for (int bounded = 0; bounded < (space == "l2" ? 2 : 1); ++bounded) {
        Result actual;
        auto run = [&] {
          if (bounded)
            idx.RerankMultiLevelBounded(query.data(), candidates, candidates.size(), estimator,
                                        &ctx, actual);
          else
            idx.RerankMultiLevel(query.data(), candidates, &ctx, actual);
        };
        if (short_read) {
          // A second call reuses the same IO context after exception/drain.
          for (int repeat = 0; repeat < 2; ++repeat)
            Reject(run);
          Check(outer.Get().metric_ml_rerank_io_submits_ > 0, "failed IO not counted");
        } else {
          run();
          std::sort(actual.begin(), actual.end());
          Equal(actual, expected);
          expected_io += blocks.size();
          Check(outer.Get().metric_ml_rerank_io_submits_ == expected_io,
                "block dedup/metric nesting");
        }
        Check(idx.metric_ml_rerank_io_submits_ == 0, "nested rerank published early");
        const auto memory = ctx.WorkspaceMemory(&idx, idx.ml_cache_config_.generation);
        Check(memory.io_buffer_bytes >= (ctx.edge_beam_width + width) * idx.ml_block_size_,
              "rerank pool missing from graph module memory accounting");
      }
      expected_io = outer.Get().metric_ml_rerank_io_submits_;
    }
    Check(idx.metric_ml_rerank_io_submits_ == expected_io, "outer metric publication");
    Check(idx.metric_ml_post_rerank_io_submits_ == expected_io, "post-rerank IO accounting");
  }
  ctx.rerank_beam_width = saved_width;
  std::cout << "RERANK_MODULES_PASSED space=" << space << " short_read=" << short_read << std::endl;
}

int main() {
  const char *env = std::getenv("TEST_TMPDIR");
  auto parent = std::filesystem::path(env ? env : "/tmp");
  std::string pattern = (parent / "alliance_prefilter_XXXXXX").string();
  std::vector<char> name(pattern.begin(), pattern.end());
  name.push_back(0);
  Check(mkdtemp(name.data()) != nullptr, "mkdtemp");
  const std::filesystem::path dir(name.data());
  constexpr size_t n = 4353, d = 64, k = 10, nc = 8;
  std::mt19937 rng(421);
  std::normal_distribution<float> normal;
  std::vector<float> base(n * d), query(d), centroids(nc * d);
  for (auto &v : base)
    v = normal(rng);
  for (auto &v : query)
    v = normal(rng);
  std::vector<int> attrs(n), cids(n);
  // Attributes repeat, labels are deliberately unrelated to rank/internal ID.
  for (size_t i = 0; i < n; ++i) {
    attrs[i] = int((i * 37) % n) / 3;
    cids[i] = i % nc;
  }
  std::copy_n(base.begin(), nc * d, centroids.begin());
  const auto cf = (dir / "centroids.fvecs").string(), ci = (dir / "cids.ivecs").string();
  WriteVecs(cf, centroids, d);
  WriteVecs(ci, cids, 1);
  for (const std::string space : {"l2", "ip"}) {
    const auto graph = (dir / (space + ".graph")).string(), ml = (dir / (space + ".ml")).string();
    Index idx(n, d, 8, 24, space, graph, 4, 4, false);
    idx.curvec_num_ = n;
    idx.InitRaBitQ(cf, ci, space);
    idx.Build(base.data(), attrs.data(), n, 1);
    LayoutExportConfig config;
    config.block_size = 4096;
    config.level_config = MultiLevelConfig{2, 1, 2, {}};
    config.method = ShuffleMethod::Identity;
    const bool test_static = std::getenv("TEST_STATIC_CACHE") != nullptr;
    config.out_graph_path = ml;
    config.out_meta_path = ml + ".meta";
    ExportMultiLevelLayout(&idx, config);
    const auto cached_meta = ml + ".cached";
    if (test_static)
      idx.PrepareStaticCacheSidecars(ml, ml + ".meta", {{cached_meta, 64 * 1024}});
    idx.LoadMultiLevelLayout(ml, test_static ? cached_meta : ml + ".meta", 1024 * 1024,
                             DynamicBlockCache::EvictPolicy::FIFO);
    const auto *shared_static = idx.static_cache_buf_;
    if (test_static)
      Check(shared_static != nullptr && idx.static_cache_bytes_ > 0, "static fixture missing");
    DiskSearchContext ctx(64, graph, ml);
    ctx.filter_mode = Mode::Prefilter;
    CheckRerankModules(idx, ctx, query, base, d, space);
    // An independent full sort enumerates by internal ID, filters by original
    // label attributes, then reranks original in-memory vectors with scalar math.
    auto oracle = [&](attr_range<int> f, size_t R, size_t want) {
      Index::RaBitQEstimator est(query.data(), &idx);
      std::vector<std::pair<float, tableint>> approx;
      for (tableint id = 0; id < n; ++id)
        if (f.Test(attrs[idx.labels_[id]]))
          approx.emplace_back(est.dist_estimate(idx.cids_[idx.labels_[id]], id), id);
      std::sort(approx.begin(), approx.end());
      approx.resize(std::min(R, approx.size()));
      Result result;
      for (auto a : approx) {
        auto label = idx.labels_[a.second];
        float dist = 0;
        for (size_t j = 0; j < d; ++j) {
          float v = base[label * d + j];
          if (space == "l2") {
            float diff = query[j] - v;
            dist += diff * diff;
          } else
            dist += query[j] * v;
        }
        result.emplace_back(space == "l2" ? dist : 1 - dist, label);
      }
      std::sort(result.begin(), result.end());
      result.resize(std::min(want, result.size()));
      return result;
    };
    // Query, traversal and rerank are separate translation units. One worker
    // scope must collect all three without early or duplicate publication.
    ResetMultiLevelMetrics(idx);
    size_t graph_io = 0, rerank_io = 0, dist_comps = 0;
    ctx.prefilter_rerank = 31;
    ctx.prefilter_threshold = n;
    {
      ScopedMultiLevelMetrics worker(idx);
      for (Mode mode : {Mode::Prefilter, Mode::Auto, Mode::Graph}) {
        ctx.filter_mode = mode;
        auto actual =
            idx.searchKNNMultiLevelPipeline(query.data(), 80, k, attr_range<int>{0, 2000}, &ctx);
        if (mode != Mode::Graph)
          Equal(actual, oracle({0, 2000}, 31, k));
        else
          Check(actual.size() == k, "graph route returned short results");
        Check(idx.metric_prefilter_queries_ == 0 && idx.metric_filter_graph_queries_ == 0 &&
                  idx.metric_ml_io_submits_ == 0 && idx.metric_ml_rerank_io_submits_ == 0 &&
                  idx.metric_dist_comps_ == 0,
              "query module published inside worker scope");
      }
      const auto &m = worker.Get();
      Check(m.metric_prefilter_queries_ == 2 && m.metric_filter_graph_queries_ == 1,
            "query routing counters did not reach worker scope");
      Check(m.metric_prefilter_scanned_ == 2 * n, "prefilter scan count");
      graph_io = m.metric_ml_io_submits_;
      rerank_io = m.metric_ml_rerank_io_submits_;
      dist_comps = m.metric_dist_comps_;
      Check((test_static || graph_io > 0) && rerank_io > 0 && dist_comps > 2 * n,
            "nested graph/rerank work missing");
#ifdef ALLIANCE_PIPELINE_PROFILE
      Check(m.metric_query_total_ns_ > 0 &&
                m.metric_query_total_ns_ ==
                    m.metric_query_prepare_ns_ + m.metric_query_traversal_ns_ +
                        m.metric_query_rerank_ns_ + m.metric_query_finalize_ns_,
            "query phase partition");
#endif
    }
    Check(idx.metric_prefilter_queries_ == 2 && idx.metric_filter_graph_queries_ == 1 &&
              idx.metric_ml_io_submits_ == graph_io &&
              idx.metric_ml_rerank_io_submits_ == rerank_io && idx.metric_dist_comps_ == dist_comps,
          "worker publication after split");
    ctx.filter_mode = Mode::Prefilter;
    ctx.prefilter_rerank = 0;
    ctx.prefilter_threshold = 0;
    std::cout << "QUERY_MODULE_SCOPE_PASSED space=" << space << std::endl;
    // Independent oracle plus mixed routes, empty/repeated ranges and every
    // worker's retained allocations. Reuse the same OpenMP threads across
    // graph -> invalidate -> prefilter to detect stale cache allocations.
    for (int threads : {1, 8, 16}) {
      std::vector<std::unique_ptr<DiskSearchContext>> contexts;
      for (int t = 0; t < threads; ++t) {
        contexts.emplace_back(std::make_unique<DiskSearchContext>(32, graph, ml));
        contexts.back()->prefilter_rerank = 31;
        contexts.back()->prefilter_threshold = 93;
      }
      const std::vector<attr_range<int>> filters{
          {19, 49}, {30, 30}, {3000, 3010}, {49, 19}, {0, 2000}};
      std::vector<Result> expected;
      for (auto f : filters)
        expected.push_back(oracle(f, 31, k));
      for (Mode mode : {Mode::Prefilter, Mode::Auto, Mode::Graph, Mode::Prefilter}) {
        idx.InvalidateMultiLevelCache();
        for (auto &c : contexts)
          c->filter_mode = mode;
        std::vector<std::exception_ptr> errors(threads);
        std::vector<MultiLevelThreadMemory> memory(threads);
#pragma omp parallel num_threads(threads)
        {
          const int tid = omp_get_thread_num();
          try {
            for (size_t q = 0; q < filters.size() * 3; ++q) {
              const size_t fi = q % filters.size();
              auto result = idx.searchKNNMultiLevelPipeline(query.data(), 80, k, filters[fi],
                                                            contexts[tid].get());
              if (mode == Mode::Prefilter || (mode == Mode::Auto && fi != 4)) {
                Equal(result, expected[fi]);
              } else {
                // Forced approximate graph traversal can return fewer than k
                // for tiny filters; exhaustive prefilter has a stronger contract.
                Check(result.size() <= expected[fi].size(), "parallel graph result count");
                if (fi == 4)
                  Check(!result.empty(), "parallel graph wide range empty");
                std::set<label_t> seen;
                for (auto r : result) {
                  Check(r.second < n && filters[fi].Test(attrs[r.second]), "parallel graph filter");
                  Check(seen.insert(r.second).second, "parallel graph duplicate");
                  Check(std::isfinite(r.first), "parallel graph distance");
                }
              }
            }
            memory[tid] = contexts[tid]->WorkspaceMemory(&idx, idx.ml_cache_config_.generation);
            Check(idx.static_cache_buf_ == shared_static, "static cache not shared");
            if (mode == Mode::Prefilter) {
              Check(memory[tid].cache_payload_bytes == 0 && memory[tid].cache_lookup_bytes == 0 &&
                        memory[tid].cache_metadata_bytes == 0 && memory[tid].visited_bytes == 0,
                    "pure prefilter allocated graph cache or visited");
              Check(memory[tid].io_buffer_bytes > 0 && memory[tid].prefilter_scratch_bytes > 0,
                    "missing prefilter memory accounting");
            } else
              Check(memory[tid].cache_lookup_bytes >= n * sizeof(tableint),
                    "graph cache was not initialized");
          } catch (...) {
            errors[tid] = std::current_exception();
          }
        }
        for (auto &error : errors)
          if (error)
            std::rethrow_exception(error);
        MultiLevelThreadMemory total;
        for (const auto &m : memory)
          total += m;
        std::cout << "PARALLEL_ROUTE_PASSED space=" << space << " threads=" << threads
                  << " mode=" << int(mode) << " cache_lookup_bytes=" << total.cache_lookup_bytes
                  << " retained_bytes=" << total.Total() << std::endl;
      }
    }
    for (size_t bits : {1, 2, 4, 8}) {
      if (bits != 1)
        idx.RequantizeRaBitQ(cf, ci, space, bits);
      for (auto f : {attr_range<int>{0, 2000}, attr_range<int>{19, 49}, attr_range<int>{30, 30}}) {
        for (size_t budget : {size_t(10), size_t(31), n, n + 100}) {
          ctx.prefilter_rerank = budget;
          const auto edges = idx.metric_ml_io_submits_.load();
          auto actual = idx.searchKNNMultiLevelPipeline(query.data(), 25, k, f, &ctx);
          Equal(actual, oracle(f, budget, k));
          Check(idx.metric_ml_io_submits_ == edges, "prefilter read graph edges");
        }
      }
      ctx.prefilter_rerank = 0;
      Equal(idx.searchKNNMultiLevelPipeline(query.data(), 31, k, attr_range<int>{19, 49}, &ctx),
            oracle({19, 49}, 31, k));
      Equal(idx.searchKNNMultiLevelPipeline(query.data(), 1, k, attr_range<int>{19, 49}, &ctx),
            oracle({19, 49}, k, k));
      const auto reads = idx.metric_ml_rerank_io_submits_.load();
      for (auto f : {attr_range<int>{3000, 3010}, attr_range<int>{49, 19}})
        Check(idx.searchKNNMultiLevelPipeline(query.data(), 31, k, f, &ctx).empty(),
              "empty/inverted filter");
      Check(idx.searchKNNMultiLevelPipeline(query.data(), 31, 0, attr_range<int>{0, 2000}, &ctx)
                .empty(),
            "k=0");
      Check(idx.metric_ml_rerank_io_submits_ == reads, "empty query performed IO");
      // S=93 exactly: threshold is inclusive and unrelated to graph windows.
      ctx.filter_mode = Mode::Auto;
      ctx.prefilter_threshold = 93;
      ctx.prefilter_rerank = 31;
      const auto pq = idx.metric_prefilter_queries_.load();
      Equal(idx.searchKNNMultiLevelPipeline(query.data(), 25, k, attr_range<int>{19, 49}, &ctx),
            oracle({19, 49}, 31, k));
      Check(idx.metric_prefilter_queries_ == pq + 1, "auto inclusive boundary");
      ctx.prefilter_threshold = 92;
      const auto gq = idx.metric_filter_graph_queries_.load();
      auto routed =
          idx.searchKNNMultiLevelPipeline(query.data(), 80, k, attr_range<int>{19, 49}, &ctx);
      Check(idx.metric_filter_graph_queries_ == gq + 1, "auto graph fallback");
      Check(idx.metric_prefilter_queries_ == pq + 1, "wide range was scanned");
      ctx.filter_mode = Mode::Prefilter;
    }
    ctx.prefilter_rerank = k - 1;
    Reject([&] {
      idx.searchKNNMultiLevelPipeline(query.data(), 25, k, attr_range<int>{0, 2000}, &ctx);
    });
    ctx.prefilter_rerank = 31;
    ctx.rerank_error_bound = true;
    ctx.rerank_bound_prescreen = true;
    const auto saved_model_factors = idx.factors_;
    for (size_t f = 2; f < idx.factors_.size(); f += 4)
      idx.factors_[f] *= 1e6f;
    Equal(idx.searchKNNMultiLevelPipeline(query.data(), 25, k, attr_range<int>{0, 2000}, &ctx),
          oracle({0, 2000}, 31, k));
    ctx.rerank_bound_prescreen = false;
    idx.factors_ = saved_model_factors;
    ctx.rerank_error_bound = false;
    ctx.filter_mode = Mode::Auto;
    ctx.prefilter_threshold = 0;
    Reject([&] {
      idx.searchKNNMultiLevelPipeline(query.data(), 25, k, attr_range<int>{0, 2000}, &ctx);
    });
    ctx.filter_mode = Mode::Prefilter;
    idx.ml_vec_region_loaded_ = false;
    Reject([&] {
      idx.searchKNNMultiLevelPipeline(query.data(), 25, k, attr_range<int>{0, 2000}, &ctx);
    });
    idx.ml_vec_region_loaded_ = true;
    // More than one IO wave: no partial/stale answers on short reads, and all
    // queued requests must be drained before exception destroys their slots.
    ctx.prefilter_rerank = n;
    std::filesystem::resize_file(ml, 0);
    Reject([&] {
      idx.searchKNNMultiLevelPipeline(query.data(), 25, k, attr_range<int>{0, 2000}, &ctx);
    });
    CheckRerankModules(idx, ctx, query, base, d, space, true);
    std::cout << "PREFILTER_TEST_PASSED space=" << space << " bits=1,2,4,8" << std::endl;
  }
  std::filesystem::remove_all(dir);
}
