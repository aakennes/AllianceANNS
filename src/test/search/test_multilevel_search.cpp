// test_multilevel_search.cpp -- Verify that searchKNN produces identical
// results whether edges are read via the legacy single-record path or via
// the multi-level layout + dynamic cache path (ReadEdgesMultiLevel).
//
// Same graph, same edges -- only the on-disk representation and the presence
// of a cache differ. Any result difference would indicate a bug in either
// ExportMultiLevelLayout, LoadMultiLevelLayout, or ReadEdgesMultiLevel.
//
// Run: ./bin/test_multilevel_search

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <limits>
#include <unordered_set>
#include "index/index_disk.hpp"
#include "shuffle/shuffle.hpp"
#include "shuffle/layout.hpp"
#include "support/navigation_fixture.hpp"

using namespace alliance;

namespace {

void WriteFvecs(const std::string &path, const float *data, size_t n, size_t d) {
  FILE *f = fopen(path.c_str(), "wb");
  int di = static_cast<int>(d);
  for (size_t i = 0; i < n; ++i) {
    fwrite(&di, sizeof(int), 1, f);
    fwrite(data + i * d, sizeof(float), d, f);
  }
  fclose(f);
}

void WriteIvecs(const std::string &path, const int *data, size_t n, size_t d) {
  FILE *f = fopen(path.c_str(), "wb");
  int di = static_cast<int>(d);
  for (size_t i = 0; i < n; ++i) {
    fwrite(&di, sizeof(int), 1, f);
    fwrite(data + i * d, sizeof(int), d, f);
  }
  fclose(f);
}

// Simple k-means (fixed iteration count) so RaBitQ has non-degenerate centroids.
void KMeans(const std::vector<float> &base_vecs, size_t n, size_t d, size_t nc,
            std::vector<float> &centroids, std::vector<int> &cids) {
  centroids.assign(nc * d, 0.0f);
  cids.assign(n, 0);
  for (size_t c = 0; c < nc; ++c) {
    std::copy(base_vecs.data() + c * d, base_vecs.data() + (c + 1) * d, centroids.data() + c * d);
  }
  for (int iter = 0; iter < 8; ++iter) {
    for (size_t i = 0; i < n; ++i) {
      float best = std::numeric_limits<float>::max();
      int best_c = 0;
      for (size_t c = 0; c < nc; ++c) {
        float dist = 0;
        for (size_t dd = 0; dd < d; ++dd) {
          float diff = base_vecs[i * d + dd] - centroids[c * d + dd];
          dist += diff * diff;
        }
        if (dist < best) {
          best = dist;
          best_c = static_cast<int>(c);
        }
      }
      cids[i] = best_c;
    }
    std::vector<float> new_centroids(nc * d, 0.0f);
    std::vector<int> counts(nc, 0);
    for (size_t i = 0; i < n; ++i) {
      int c = cids[i];
      counts[c]++;
      for (size_t dd = 0; dd < d; ++dd)
        new_centroids[c * d + dd] += base_vecs[i * d + dd];
    }
    for (size_t c = 0; c < nc; ++c) {
      if (counts[c] > 0) {
        for (size_t dd = 0; dd < d; ++dd)
          centroids[c * d + dd] = new_centroids[c * d + dd] / counts[c];
      }
    }
  }
}

using ResultSet = std::vector<std::vector<std::pair<dist_t, label_t>>>;

// `narrow_hi`, when >= 0, overrides the filter's upper bound to something
// smaller than n-1 -- a narrower attribute range causes more neighbor
// candidates to fail the filter check, which forces filter-driven descent
// into lower layers (see the `visit_next_layer` logic in SearchCandidates*).
auto RunQueries(AllianceIndexDisk<int, float> &idx, const std::vector<float> &queries, size_t nq,
                size_t d, size_t efs, size_t k, int n, int narrow_hi = -1) -> ResultSet {
  ResultSet out(nq);
  int hi = narrow_hi >= 0 ? narrow_hi : (n - 1);
  attr_range<int> filter{0, hi};
  for (size_t qi = 0; qi < nq; ++qi) {
    out[qi] = idx.searchKNN(const_cast<float *>(queries.data() + qi * d), efs, k, filter);
  }
  return out;
}

auto RunQueriesPipeline(AllianceIndexDisk<int, float> &idx, DiskSearchContext &ctx,
                        const std::vector<float> &queries, size_t nq, size_t d, size_t efs,
                        size_t k, int n, int narrow_hi = -1) -> ResultSet {
  ResultSet out(nq);
  int hi = narrow_hi >= 0 ? narrow_hi : (n - 1);
  attr_range<int> filter{0, hi};
  for (size_t qi = 0; qi < nq; ++qi) {
    out[qi] = idx.searchKNNMultiLevelPipeline(const_cast<float *>(queries.data() + qi * d), efs, k,
                                              filter, &ctx);
    std::unordered_set<label_t> ids;
    for (const auto &item : out[qi])
      ids.insert(item.second);
    if (out[qi].size() != k || ids.size() != k)
      throw std::runtime_error("pipeline must return K unique results");
  }
  return out;
}

auto ResultsEqual(const ResultSet &a, const ResultSet &b) -> bool {
  if (a.size() != b.size())
    return false;
  for (size_t qi = 0; qi < a.size(); ++qi) {
    if (a[qi].size() != b[qi].size())
      return false;
    for (size_t j = 0; j < a[qi].size(); ++j) {
      if (a[qi][j].second != b[qi][j].second)
        return false;
      if (std::abs(a[qi][j].first - b[qi][j].first) > 1e-4f)
        return false;
    }
  }
  return true;
}

// Recall of `test` against `gt`, treating `gt` as the reference top-k set.
// Pipelined beam search (both AllianceANNS's pre-existing baseline
// searchKNNPipeline and this multi-level variant) expands ready candidates
// via a LIFO ready_queue rather than strict global-nearest-first order, so
// its exact top-k need not bit-match a sequential search's -- that's an
// existing, expected property of the pipeline design, not a correctness bug.
// Recall against the sequential result is the right bar for the pipeline
// paths; exact match is reserved for the two deterministic sync paths above.
auto Recall(const ResultSet &gt, const ResultSet &test) -> double {
  size_t nq = std::min(gt.size(), test.size());
  if (nq == 0)
    return 0.0;
  double sum = 0.0;
  for (size_t qi = 0; qi < nq; ++qi) {
    if (gt[qi].empty())
      continue;
    size_t hit = 0;
    for (auto &p : test[qi]) {
      for (auto &g : gt[qi]) {
        if (p.second == g.second) {
          ++hit;
          break;
        }
      }
    }
    sum += static_cast<double>(hit) / static_cast<double>(gt[qi].size());
  }
  return sum / static_cast<double>(nq);
}

} // namespace

int main() {
  printf("====== Multi-Level Search Correctness Tests ======\n\n");

  // IOHandler opens files with O_DIRECT, which tmpfs does not support -- on
  // machines where /tmp is tmpfs, point TEST_TMPDIR at a real filesystem.
  const char *tmp_env = std::getenv("TEST_TMPDIR");
  const std::string tmp_dir = tmp_env ? tmp_env : "/tmp";

  const size_t n = 5000;
  const size_t d = 128;
  const size_t M = 16;
  const size_t efc = 64;
  const size_t o = 4;
  const size_t wp = 6;
  const size_t nc = 64;
  const size_t k = 10;
  const size_t efs = 150;
  const size_t nq = 50;
  const int threads = 1;

  std::mt19937 rng(99);
  std::normal_distribution<float> gauss(0.0f, 1.0f);
  std::vector<float> base_vecs(n * d);
  for (auto &v : base_vecs)
    v = gauss(rng);
  std::vector<int> att_vec(n);
  std::iota(att_vec.begin(), att_vec.end(), 0);

  std::vector<float> centroids;
  std::vector<int> cids;
  KMeans(base_vecs, n, d, nc, centroids, cids);
  const std::string centroids_path = tmp_dir + "/test_ml_search_centroids.fvecs";
  const std::string cids_path = tmp_dir + "/test_ml_search_cids.ivecs";
  WriteFvecs(centroids_path, centroids.data(), nc, d);
  WriteIvecs(cids_path, cids.data(), n, 1);

  const std::string graph_path = tmp_dir + "/test_ml_search_graph.bin";
  std::remove(graph_path.c_str());

  AllianceIndexDisk<int, float> idx(n, d, M, efc, "l2", graph_path, o, wp, /*auto_raise_wp=*/false);
  // Configure search quantization before the public build entry populates
  // the graph; construction itself does not use RaBitQ codes.
  idx.curvec_num_ = n;
  idx.InitRaBitQ(centroids_path, cids_path, "l2");
  idx.Build(base_vecs.data(), att_vec.data(), n, threads);

  printf("Built index: n=%zu M=%zu wp=%zu cur_max_layer=%zu\n\n", n, M, wp, idx.cur_max_layer_);

  std::vector<float> queries(nq * d);
  for (auto &v : queries)
    v = gauss(rng);

  // --- Baseline: legacy single-record ReadEdges path ---
  auto baseline = RunQueries(idx, queries, nq, d, efs, k, static_cast<int>(n));
  printf("Baseline (legacy ReadEdges path): %zu queries done\n\n", nq);

  // --- Export + load the multi-level layout, FIFO policy ---
  const std::string out_graph = tmp_dir + "/test_ml_search_ml_graph.bin";
  const std::string out_meta = tmp_dir + "/test_ml_search_ml_meta.bin";
  std::remove(out_graph.c_str());
  std::remove(out_meta.c_str());

  LayoutExportConfig config;
  config.block_size = 4096;
  config.level_config = MultiLevelConfig{2, 1, 2, {}};
  config.method = ShuffleMethod::BlockBNF;
  config.out_graph_path = out_graph;
  config.out_meta_path = out_meta;
  auto export_result = ExportMultiLevelLayout(&idx, config);
  const auto cached_meta = out_meta + ".cached";
  std::remove(cached_meta.c_str());
  idx.PrepareStaticCacheSidecars(out_graph, out_meta, {{cached_meta, 1024 * 1024}});
  printf("Exported multi-level layout: %zu levels\n\n", export_result.levels.size());

  bool all_ok = true;

  idx.LoadMultiLevelLayout(out_graph, cached_meta, 32 * 1024 * 1024,
                           DynamicBlockCache::EvictPolicy::FIFO);
  auto fifo_results = RunQueries(idx, queries, nq, d, efs, k, static_cast<int>(n));
  bool fifo_match = ResultsEqual(baseline, fifo_results);
  printf("FIFO policy: results %s baseline\n", fifo_match ? "MATCH" : "MISMATCH");
  printf("  metric_ml_block_reads_=%zu metric_ml_cache_hits_=%zu metric_ml_cache_misses_=%zu\n\n",
         idx.metric_ml_block_reads_.load(), idx.metric_ml_cache_hits_.load(),
         idx.metric_ml_cache_misses_.load());
  all_ok &= fifo_match;

  // --- Reload with LRU policy; must reset ml_layout_loaded_ state via a fresh LoadMultiLevelLayout
  // call ---
  idx.LoadMultiLevelLayout(out_graph, cached_meta, 32 * 1024 * 1024,
                           DynamicBlockCache::EvictPolicy::LRU);
  auto lru_results = RunQueries(idx, queries, nq, d, efs, k, static_cast<int>(n));
  bool lru_match = ResultsEqual(baseline, lru_results);
  printf("LRU policy: results %s baseline\n", lru_match ? "MATCH" : "MISMATCH");
  printf("  metric_ml_block_reads_=%zu metric_ml_cache_hits_=%zu metric_ml_cache_misses_=%zu\n\n",
         idx.metric_ml_block_reads_.load(), idx.metric_ml_cache_hits_.load(),
         idx.metric_ml_cache_misses_.load());
  all_ok &= lru_match;

  // --- Pipeline path (async O_DIRECT IO, block dedup, cross-level pause/resume) ---
  // --- Baseline pipeline (searchKNNPipeline, one-node-per-page) ---
  const double kPipelineRecallThreshold = 0.90;
  {
    DiskSearchContext base_ctx(/*queue_depth=*/64, graph_path);
    base_ctx.edge_beam_width = 4;
    ResultSet base_pipeline(nq);
    attr_range<int> full_filter{0, static_cast<int>(n) - 1};
    for (size_t qi = 0; qi < nq; ++qi) {
      base_pipeline[qi] = idx.searchKNNPipeline(const_cast<float *>(queries.data() + qi * d), efs,
                                                k, full_filter, &base_ctx);
    }
    double base_pipeline_recall = Recall(baseline, base_pipeline);
    bool base_pipeline_ok = base_pipeline_recall >= kPipelineRecallThreshold;
    printf("Baseline pipeline (searchKNNPipeline): recall vs sync = %.3f %s\n\n",
           base_pipeline_recall, base_pipeline_ok ? "OK" : "TOO LOW");
    all_ok &= base_pipeline_ok;
  }

  // --- Multi-level pipeline ---
  idx.LoadMultiLevelLayout(out_graph, cached_meta, 32 * 1024 * 1024,
                           DynamicBlockCache::EvictPolicy::FIFO);
  DiskSearchContext ml_ctx(/*queue_depth=*/64, graph_path, out_graph);
  ml_ctx.edge_beam_width = 4;

  // Disable the WoW-style rerank budget (rerank_times_upper_bound_ * k) for
  // these correctness checks: with this synthetic gaussian data the RaBitQ
  // estimates are noisy, so truncating rerank to the top 8*k estimated
  // candidates costs visible recall -- a policy trade-off, not a bug. The
  // budget's effect is evaluated on real datasets by alliance_search_disk.
  ml_ctx.rerank_num = efs;

  // Full-range filter: never excludes a candidate, so every node's expansion
  // stays within one layer/level. Mainly exercises same-level block dedup
  // (BNF co-locates multiple candidates' records in the same physical block).
  auto pipeline_full = RunQueriesPipeline(idx, ml_ctx, queries, nq, d, efs, k, static_cast<int>(n));
  double pipeline_full_recall = Recall(baseline, pipeline_full);
  bool pipeline_full_ok = pipeline_full_recall >= kPipelineRecallThreshold;
  printf("Pipeline (full-range filter): recall vs baseline = %.3f %s\n", pipeline_full_recall,
         pipeline_full_ok ? "OK" : "TOO LOW");
  printf(
      "  metric_ml_block_reads_=%zu metric_ml_io_submits_=%zu metric_ml_io_dedup_saved_=%zu ml_static_hits=%zu\n\n",
      idx.metric_ml_block_reads_.load(), idx.metric_ml_io_submits_.load(),
      idx.metric_ml_io_dedup_saved_.load(), idx.metric_ml_static_cache_hits_.load());
  all_ok &= pipeline_full_ok;

  // Narrow filter: roughly half of candidates fail the attribute check each
  // layer, forcing multi-layer descent within one hop's expansion -- this is
  // what exercises the pause/resume-across-levels path (a node's expansion
  // can hit a level boundary mid-loop: cached on one level, miss on the
  // next). Baseline uses the sync ml path (already proven == legacy above).
  const int narrow_hi = static_cast<int>(n) / 2;
  auto baseline_narrow = RunQueries(idx, queries, nq, d, efs, k, static_cast<int>(n), narrow_hi);
  auto pipeline_narrow =
      RunQueriesPipeline(idx, ml_ctx, queries, nq, d, efs, k, static_cast<int>(n), narrow_hi);
  double pipeline_narrow_recall = Recall(baseline_narrow, pipeline_narrow);
  bool pipeline_narrow_ok = pipeline_narrow_recall >= kPipelineRecallThreshold;
  printf(
      "Pipeline (narrow filter, forces cross-level pause/resume): recall vs baseline = %.3f %s\n",
      pipeline_narrow_recall, pipeline_narrow_ok ? "OK" : "TOO LOW");
  printf(
      "  metric_ml_block_reads_=%zu metric_ml_io_submits_=%zu metric_ml_io_dedup_saved_=%zu ml_static_hits=%zu\n\n",
      idx.metric_ml_block_reads_.load(), idx.metric_ml_io_submits_.load(),
      idx.metric_ml_io_dedup_saved_.load(), idx.metric_ml_static_cache_hits_.load());
  all_ok &= pipeline_narrow_ok;

  // Stress bounded pools at low ef, where post-search dedup cannot repair
  // candidates lost to duplicate IDs. Exercise cold/warm caches; the optional
  // core checks validate every admission.
  idx.metric_ml_rerank_duplicate_ids_ = 0;
  for (bool cold : {false, true}) {
    idx.per_query_cache_ = cold;
    RunQueriesPipeline(idx, ml_ctx, queries, nq, d, 16, k, static_cast<int>(n), narrow_hi);
  }
  bool no_pre_rerank_duplicates = idx.metric_ml_rerank_duplicate_ids_.load() == 0;
  printf("Low-ef cold/warm: pre-rerank duplicates=%zu %s\n",
         idx.metric_ml_rerank_duplicate_ids_.load(), no_pre_rerank_duplicates ? "OK" : "FAILED");
  all_ok &= no_pre_rerank_duplicates;

  // Repeated entry IDs used to occupy multiple traversal and result slots.
  // Inspect candidates BEFORE final rerank dedup, including eps.size() > ef.
  std::vector<dist_id_pair> repeated_eps;
  for (tableint id = 0; id < 20; ++id) {
    for (int repeat = 0; repeat < 3; ++repeat)
      repeated_eps.emplace_back(static_cast<dist_t>(id), id);
  }
  attr_range<int> full_filter{0, static_cast<int>(n) - 1};
  std::vector<tableint> unused_eps;
  auto layers = idx.DecideLayerRange(full_filter, unused_eps);
  ml_ctx.rerank_num = 2 * k; // ef=16 plus four retained rejected/evicted IDs.
  auto candidates = idx.SearchCandidatesQueryMultiLevelPipeline(
      repeated_eps, queries.data(), full_filter, layers, 16, k, &ml_ctx);
  std::unordered_set<tableint> candidate_ids;
  for (const auto &item : candidates)
    candidate_ids.insert(item.id_);
  bool entry_unique = candidates.size() == 2 * k && candidate_ids.size() == 2 * k;
  printf("Repeated entry IDs with ef=16, R=20: count=%zu unique=%zu %s\n", candidates.size(),
         candidate_ids.size(), entry_unique ? "OK" : "FAILED");
  all_ok &= entry_unique;

  // Compare owned/borrowed estimators using real seed scores. Preload every
  // entry-layer record so expansion is deterministic
  // and estimator equality is not obscured by asynchronous completion order.
  idx.per_query_cache_ = false;
  std::vector<tableint> warm_edges(M + 1);
  for (tableint id = 0; id < n; ++id)
    idx.ReadEdgesMultiLevel(id, layers.u_, warm_edges.data());
  AllianceIndexDisk<int, float>::RaBitQEstimator shared_estimator(queries.data(), &idx);
  std::vector<dist_id_pair> real_eps;
  for (tableint id = 0; id < 20; ++id)
    real_eps.emplace_back(shared_estimator.dist_estimate(idx.cids_[idx.labels_[id]], id), id);
  const auto reads_before = idx.metric_ml_io_submits_.load();
  candidates = idx.SearchCandidatesQueryMultiLevelPipeline(real_eps, queries.data(), full_filter,
                                                           layers, 16, k, &ml_ctx);
  auto shared_candidates = idx.SearchCandidatesQueryMultiLevelPipeline(
      real_eps, queries.data(), full_filter, layers, 16, k, &ml_ctx, &shared_estimator);
  auto sort_candidates = [](auto &v) {
    std::sort(v.begin(), v.end(), [](const auto &a, const auto &b) { return a.id_ < b.id_; });
  };
  sort_candidates(candidates);
  sort_candidates(shared_candidates);
  bool shared_ok = candidates.size() == shared_candidates.size();
  for (size_t i = 0; shared_ok && i < candidates.size(); ++i)
    shared_ok = candidates[i].id_ == shared_candidates[i].id_ &&
                candidates[i].dist_ == shared_candidates[i].dist_;
  shared_ok &= idx.metric_ml_io_submits_.load() == reads_before;
  shared_ok &=
      std::any_of(candidates.begin(), candidates.end(), [](const auto &c) { return c.id_ >= 20; });
  printf("Borrowed/owned query estimator candidates: %s\n", shared_ok ? "MATCH" : "MISMATCH");
  all_ok &= shared_ok;

  // Graph rerank policy: test actual candidate counts and exact top-k against
  // an independent scalar L2 oracle over deterministic, RAM-cached traversal.
  const auto check_policy = [](bool ok, const char *message) {
    if (!ok)
      throw std::runtime_error(message);
  };
  ml_ctx.rerank_num = 0;
  check_policy(ml_ctx.GraphRerankBudget(k) == 8 * k, "automatic graph budget changed");
  ml_ctx.rerank_factor = 3;
  check_policy(ml_ctx.GraphRerankBudget(k) == 3 * k && ml_ctx.GraphRerankBudget(25) == 75,
               "factor budget must follow the current query k");
  check_policy(ml_ctx.GraphRerankBudget(std::numeric_limits<size_t>::max()) ==
                   std::numeric_limits<size_t>::max(),
               "factor budget overflow");
  ml_ctx.rerank_factor = 0;
  ml_ctx.rerank_num = k - 1;
  bool rejected = false;
  try {
    idx.searchKNNMultiLevelPipeline(queries.data(), 40, k, full_filter, &ml_ctx);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  check_policy(rejected, "graph budget below k must be rejected by search API");
  const size_t max_size = std::numeric_limits<size_t>::max();
  ml_ctx.rerank_num = 0;
  check_policy(ml_ctx.GraphRerankBudget(max_size) == max_size, "budget overflow");
  ml_ctx.filter_mode = DiskSearchContext::FilterMode::Graph;
  for (size_t qi = 0; qi < 2; ++qi) {
    auto *q = queries.data() + qi * d;
    AllianceIndexDisk<int, float>::RaBitQEstimator query_estimator(q, &idx);
    std::vector<tableint> seed_ids;
    auto query_layers = idx.DecideLayerRange(full_filter, seed_ids);
    std::vector<dist_id_pair> scored_seeds;
    for (auto id : seed_ids)
      scored_seeds.emplace_back(query_estimator.dist_estimate(idx.cids_[idx.labels_[id]], id), id);
    for (const auto &setting : std::vector<std::pair<size_t, size_t>>{{10, 10},
                                                                      {40, 10},
                                                                      {15, 20},
                                                                      {20, 20},
                                                                      {40, 20},
                                                                      {40, 30},
                                                                      {40, 50},
                                                                      {260, 250},
                                                                      {15, n + 100}}) {
      const auto test_ef = setting.first;
      const auto budget = setting.second;
      ml_ctx.rerank_num = budget;
      auto discovered = idx.SearchCandidatesQueryMultiLevelPipeline(
          scored_seeds, q, full_filter, query_layers, test_ef, k, &ml_ctx, &query_estimator);
      std::sort(discovered.begin(), discovered.end(), [](const auto &a, const auto &b) {
        return a.dist_ < b.dist_ || (a.dist_ == b.dist_ && a.id_ < b.id_);
      });
      if (budget <= n)
        check_policy(discovered.size() == budget,
                     "reserve failed to fill R from enough discoveries");
      else
        check_policy(discovered.size() < budget && discovered.size() > test_ef,
                     "sparse discovery must not invent candidates to fill R");
      discovered.resize(std::min(discovered.size(), budget));
      std::vector<std::pair<float, label_t>> expected;
      for (const auto &c : discovered) {
        auto label = idx.labels_[c.id_];
        float distance = 0;
        for (size_t dim = 0; dim < d; ++dim) {
          const float diff = q[dim] - base_vecs[label * d + dim];
          distance += diff * diff;
        }
        expected.emplace_back(distance, label);
      }
      std::sort(expected.begin(), expected.end());
      expected.resize(std::min(k, expected.size()));
      ml_ctx.rerank_num = budget;
      idx.metric_ml_rerank_candidates_ = 0;
      auto actual = idx.searchKNNMultiLevelPipeline(q, test_ef, k, full_filter, &ml_ctx);
      check_policy(idx.metric_ml_rerank_candidates_.load() == discovered.size(),
                   "actual rerank count did not match combined main/reserve candidates");
      check_policy(actual.size() == expected.size(), "rerank oracle result size mismatch");
      for (size_t j = 0; j < actual.size(); ++j)
        check_policy(actual[j].second == expected[j].second &&
                         std::abs(actual[j].first - expected[j].first) < 0.001f,
                     "rerank result differs from independent scalar L2 oracle");
      // The ordinary direct-output path and the retained bounded/old-record
      // paths must agree on a warm deterministic traversal. A conservative
      // bound suppresses pruning here so the same scalar oracle applies.
      for (int control : {0, 1}) {
        ml_ctx.rerank_error_bound = control == 0;
        const auto saved_model_factors = idx.factors_;
        for (size_t f = 2; f < idx.factors_.size(); f += 4)
          idx.factors_[f] *= 1e6f;
        idx.ml_vec_region_loaded_ = control == 0;
        const auto pruned_before = idx.metric_ml_rerank_bound_pruned_.load();
        auto retained = idx.searchKNNMultiLevelPipeline(q, test_ef, k, full_filter, &ml_ctx);
        ml_ctx.rerank_error_bound = false;
        idx.factors_ = saved_model_factors;
        idx.ml_vec_region_loaded_ = true;
        check_policy(idx.metric_ml_rerank_bound_pruned_.load() == pruned_before,
                     "conservative control unexpectedly pruned");
        check_policy(retained.size() == expected.size(), "retained rerank result count");
        for (size_t j = 0; j < retained.size(); ++j)
          check_policy(retained[j].second == expected[j].second &&
                           std::abs(retained[j].first - expected[j].first) < 0.001f,
                       "bounded/old-record rerank scalar oracle mismatch");
      }
    }
  }
  // Factor and absolute budgets must select the same exact top-k from a
  // deterministic warm traversal, including the newly allowed R=k floor.
  for (size_t factor : {size_t{1}, size_t{3}}) {
    ml_ctx.rerank_num = factor * k;
    auto fixed = idx.searchKNNMultiLevelPipeline(queries.data(), 40, k, full_filter, &ml_ctx);
    ml_ctx.rerank_num = 0;
    ml_ctx.rerank_factor = factor;
    idx.metric_ml_rerank_candidates_ = 0;
    auto scaled = idx.searchKNNMultiLevelPipeline(queries.data(), 40, k, full_filter, &ml_ctx);
    check_policy(fixed == scaled && scaled.size() == k &&
                     idx.metric_ml_rerank_candidates_.load() == factor * k,
                 "factor budget changed exact results or candidate count");
    ml_ctx.rerank_factor = 0;
  }
  using RerankPolicy = DiskSearchContext::GraphRerankPolicy;
  for (auto policy : {RerankPolicy::Step, RerankPolicy::Linear}) {
    ml_ctx.rerank_num = 0;
    ml_ctx.rerank_policy = policy;
    const std::vector<size_t> widths{10, 20, 21, 30, 40, 60, 80, 81, 160, 320};
    const std::vector<size_t> expected =
        policy == RerankPolicy::Step ? std::vector<size_t>{20, 20, 30, 30, 30, 30, 30, 40, 40, 40}
                                     : std::vector<size_t>{20, 20, 21, 22, 23, 26, 29, 29, 40, 40};
    for (size_t i = 0; i < widths.size(); ++i)
      check_policy(ml_ctx.GraphRerankBudgetForSearch(10, widths[i]) == expected[i],
                   "dynamic boundary or rounding");
    check_policy(ml_ctx.GraphRerankBudgetForSearch(25, 50) == 50 &&
                     ml_ctx.GraphRerankBudgetForSearch(25, 400) == 100,
                 "dynamic k scaling");
    check_policy(ml_ctx.GraphRerankBudgetForSearch(max_size, max_size) == max_size,
                 "dynamic budget overflow");
    for (int invalid = 0; invalid < 4; ++invalid) {
      ml_ctx.rerank_num = invalid == 0 ? 20 : 0;
      ml_ctx.rerank_factor = invalid == 1 ? 2 : 0;
      bool threw = false;
      try {
        if (invalid == 2)
          ml_ctx.GraphRerankBudget(10);
        else
          ml_ctx.GraphRerankBudgetForSearch(10, invalid == 3 ? 0 : 40);
      } catch (const std::invalid_argument &) {
        threw = true;
      }
      check_policy(threw, "dynamic invalid budget accepted");
    }
    ml_ctx.rerank_num = 0;
    ml_ctx.rerank_factor = 0;
    for (size_t width : {size_t{10}, size_t{30}, size_t{80}, size_t{160}}) {
      ml_ctx.rerank_policy = policy;
      const size_t budget = ml_ctx.GraphRerankBudgetForSearch(k, width);
      ml_ctx.rerank_policy = RerankPolicy::Fixed;
      ml_ctx.rerank_num = budget;
      auto fixed = idx.searchKNNMultiLevelPipeline(queries.data(), width, k, full_filter, &ml_ctx);
      ml_ctx.rerank_num = 0;
      ml_ctx.rerank_policy = policy;
      idx.metric_ml_rerank_candidates_ = 0;
      auto dynamic =
          idx.searchKNNMultiLevelPipeline(queries.data(), width, k, full_filter, &ml_ctx);
      check_policy(dynamic == fixed && dynamic.size() == k &&
                       idx.metric_ml_rerank_candidates_.load() == budget,
                   "dynamic traversal and exact rerank budget mismatch");
    }
  }
  ml_ctx.rerank_policy = RerankPolicy::Fixed;
  // With warm graph records, growing only the reserve must preserve graph
  // discovery and expansion work. Measure the full discovered set first:
  // Vamana's sparse graph need not discover 200 IDs at ef=15. The admission
  // counter is independent of both result heaps; this full-range query offers
  // every uniquely discovered ID to rerank.
  constexpr size_t reserve_ef = 15;
  const size_t exhaustive_budget = n + 100;
  ml_ctx.rerank_num = exhaustive_budget;
  const auto reference_hops_before = idx.metric_hops_.load();
  const auto reference_comps_before = idx.metric_dist_comps_.load();
  const auto reference_discoveries_before = idx.metric_ml_unique_discoveries_.load();
  idx.metric_ml_rerank_candidates_ = 0;
  idx.searchKNNMultiLevelPipeline(queries.data(), reserve_ef, k, full_filter, &ml_ctx);
  const auto reference_hops = idx.metric_hops_.load() - reference_hops_before;
  const auto reference_comps = idx.metric_dist_comps_.load() - reference_comps_before;
  const auto discovered_count =
      idx.metric_ml_unique_discoveries_.load() - reference_discoveries_before;
  check_policy(discovered_count >= 50 && discovered_count < exhaustive_budget,
               "reserve fixture must cover both full and underfilled budgets");
  check_policy(idx.metric_ml_rerank_candidates_.load() == discovered_count,
               "unbounded reserve lost unique full-range discoveries");
  for (size_t budget : {size_t{20}, size_t{50}, size_t{200}, exhaustive_budget}) {
    ml_ctx.rerank_num = budget;
    const auto hops_before = idx.metric_hops_.load();
    const auto comps_before = idx.metric_dist_comps_.load();
    const auto discoveries_before = idx.metric_ml_unique_discoveries_.load();
    idx.metric_ml_rerank_candidates_ = 0;
    idx.metric_ml_rerank_reserve_candidates_ = 0;
    idx.searchKNNMultiLevelPipeline(queries.data(), reserve_ef, k, full_filter, &ml_ctx);
    const auto hops = idx.metric_hops_.load() - hops_before;
    const auto comps = idx.metric_dist_comps_.load() - comps_before;
    const auto discoveries = idx.metric_ml_unique_discoveries_.load() - discoveries_before;
    check_policy(hops == reference_hops && comps == reference_comps &&
                     discoveries == discovered_count,
                 "reserve changed graph discovery or expansion work");
    const size_t expected = std::min(budget, discovered_count);
    check_policy(idx.metric_ml_rerank_candidates_.load() == expected &&
                     idx.metric_ml_rerank_reserve_candidates_.load() ==
                         expected - std::min(expected, reserve_ef),
                 "main plus reserve counts do not equal min(R, discovered)");
  }
  printf("Reserve accounting: ef=%zu unique_discoveries=%zu full budgets=20,50 sparse budget=%zu\n",
         reserve_ef, discovered_count, exhaustive_budget);
  ml_ctx.rerank_num = 2 * k;
  printf(
      "Rerank policy: reserve fills R>ef, unchanged traversal work, sparse discovery, scalar oracle PASSED\n");

  // Fixed candidates isolate rerank from asynchronous graph traversal.
  // Include shared blocks and more blocks than every tested effective width.
  std::unordered_map<size_t, size_t> block_counts;
  std::vector<dist_id_pair> fixed_candidates;
  for (tableint id = 0; id < n; ++id) {
    const auto block = idx.ml_vec_block_ids_[id];
    if (!block_counts.count(block) && block_counts.size() == 67)
      continue;
    if (block_counts[block]++ < 2)
      fixed_candidates.emplace_back(0, id);
  }
  check_policy(block_counts.size() == 67 && fixed_candidates.size() > 67,
               "rerank fixture requires multiple waves and shared blocks");
  std::reverse(fixed_candidates.begin(), fixed_candidates.end());
  const auto check_exact = [&](const auto &results) {
    check_policy(results.size() == fixed_candidates.size(), "direct rerank count");
    for (size_t i = 0; i < results.size(); ++i) {
      const auto label = idx.labels_[fixed_candidates[i].id_];
      float expected = 0;
      for (size_t j = 0; j < d; ++j) {
        const float diff = queries[j] - base_vecs[label * d + j];
        expected += diff * diff;
      }
      check_policy(results[i].second == label && std::abs(results[i].first - expected) < 0.001f,
                   "direct rerank differs from scalar L2 oracle or candidate order");
    }
  };
  for (size_t width : {size_t{1}, size_t{3}, size_t{32}, size_t{0}}) {
    ml_ctx.rerank_beam_width = width;
    std::vector<std::pair<dist_t, label_t>> results;
    const auto reads = idx.metric_ml_rerank_io_submits_.load();
    idx.RerankMultiLevel(queries.data(), fixed_candidates, &ml_ctx, results);
    check_exact(results);
    check_policy(idx.metric_ml_rerank_io_submits_.load() - reads == block_counts.size(),
                 "rerank must read each unique block exactly once across refills");
    idx.RerankMultiLevel(queries.data(), {}, &ml_ctx, results);
    check_policy(results.empty() &&
                     idx.metric_ml_rerank_io_submits_.load() - reads == block_counts.size(),
                 "empty rerank must clear results without IO");
#ifdef ALLIANCE_PIPELINE_PROFILE
    check_policy(ml_ctx.ml_io_handler->QueueProfile().Pending() == 0, "rerank queue not drained");
#endif
  }
  ml_ctx.rerank_beam_width = 0;
  const std::string short_path = tmp_dir + "/test_ml_rerank_short.bin";
  FILE *short_file = std::fopen(short_path.c_str(), "wb");
  check_policy(short_file != nullptr, "cannot create short-read fixture");
  std::fclose(short_file);
  {
    DiskSearchContext failed_ctx(/*queue_depth=*/64, graph_path, short_path);
    failed_ctx.rerank_beam_width = 3;
    std::vector<std::pair<dist_t, label_t>> results;
    const auto reads = idx.metric_ml_rerank_io_submits_.load();
    bool threw = false;
    try {
      idx.RerankMultiLevel(queries.data(), fixed_candidates, &failed_ctx, results);
    } catch (const std::runtime_error &e) {
      threw = std::string(e.what()) == "short or failed read in multi-level rerank";
    }
    check_policy(threw && idx.metric_ml_rerank_io_submits_.load() - reads == block_counts.size(),
                 "short reads must drain every rerank wave before throwing");
#ifdef ALLIANCE_PIPELINE_PROFILE
    check_policy(failed_ctx.ml_io_handler->QueueProfile().Pending() == 0,
                 "failed rerank queue not drained");
#endif
    failed_ctx.ml_io_handler->Open(out_graph);
    idx.RerankMultiLevel(queries.data(), fixed_candidates, &failed_ctx, results);
    check_exact(results);
  }
  std::remove(short_path.c_str());
  printf(
      "Direct rerank: multi-wave widths, grouped blocks, empty input, short-read drain and recovery PASSED\n");

  // Scheme B oracle with different final-rerank widths, using a warm graph and
  // deliberately wide bounds. Candidate retention must remain ef-scaled.
  ml_ctx.rerank_num = 0;
  ml_ctx.rerank_factor = 0;
  ml_ctx.rerank_policy = DiskSearchContext::GraphRerankPolicy::Fixed;
  ml_ctx.rerank_candidate_factor = 2;
  const auto saved_model_factors = idx.factors_;
  for (size_t f = 2; f < idx.factors_.size(); f += 4)
    idx.factors_[f] *= 1e6f;
  ml_ctx.rerank_bound_prescreen = true;
  ml_ctx.rerank_warm_blocks = 2;
  ml_ctx.edge_beam_width = 8;
  ml_ctx.rerank_beam_width = 8;
  check_policy(ml_ctx.GraphRerankBudgetForSearch(25, 10) == 25, "candidate factor k floor");
  for (bool top : {false, true}) {
    ml_ctx.top_layer_postfilter = top;
    ml_ctx.rerank_error_bound = false;
    auto oracle = idx.searchKNNMultiLevelPipeline(queries.data(), 80, k, full_filter, &ml_ctx);
    for (size_t width : {size_t(1), size_t(2), size_t(4), size_t(8), size_t(16)}) {
      ml_ctx.rerank_beam_width = width;
      check_policy(ml_ctx.RerankWidth() == width, "final rerank width ignored");
      ml_ctx.rerank_error_bound = true;
      auto actual = idx.searchKNNMultiLevelPipeline(queries.data(), 80, k, full_filter, &ml_ctx);
      check_policy(actual == oracle, "scheme B changed warm candidate-pool exact oracle");
    }
  }
  ml_ctx.rerank_candidate_factor = 0;
  ml_ctx.rerank_error_bound = false;
  ml_ctx.rerank_bound_prescreen = false;
  idx.factors_ = saved_model_factors;
  ml_ctx.rerank_warm_blocks = 0;
  ml_ctx.rerank_beam_width = 8;
  printf("Scheme B candidate-pool and final-rerank width oracle PASSED\n");

  // Entry sampling must not bypass range filtering, return duplicate IDs,
  // alter the rerank policy, or allocate a second disk-query path.
  ml_ctx.rerank_num = 2 * k;
  ml_ctx.filter_mode = DiskSearchContext::FilterMode::Graph;
  for (auto sampling :
       {DiskSearchContext::EntrySampling::Global, DiskSearchContext::EntrySampling::Range,
        DiskSearchContext::EntrySampling::Guided}) {
    ml_ctx.entry_sampling = sampling;
    for (bool post : {false, true})
      for (size_t sample_count : {size_t(0), size_t(256)}) {
        ml_ctx.top_layer_postfilter = post;
        ml_ctx.entry_samples = sample_count;
        for (int hi : {int(n) - 1, narrow_hi}) {
          attr_range<int> f{0, hi};
          const auto before = idx.metric_ml_entry_estimates_.load();
          auto result = idx.searchKNNMultiLevelPipeline(queries.data(), 16, k, f, &ml_ctx);
          check_policy(result.size() == k, "rank entries returned short result");
          std::unordered_set<label_t> unique;
          for (const auto &[distance, label] : result) {
            check_policy(label <= label_t(hi) && unique.insert(label).second,
                         "rank entries range/uniqueness");
          }
          const auto scores = idx.metric_ml_entry_estimates_.load() - before;
          check_policy(sample_count ? scores > 0 : scores == 0, "rank entry score accounting");
          if (sampling == DiskSearchContext::EntrySampling::Guided && sample_count)
            check_policy(scores == std::min(sample_count, size_t(hi + 1)),
                         "guided exact score budget");
          check_policy(ml_ctx.GraphRerankBudgetForSearch(k, 16) == 2 * k,
                       "entry sampling changed rerank policy");
        }
      }
  }
  ml_ctx.entry_samples = 256;
  ml_ctx.entry_sampling = DiskSearchContext::EntrySampling::Guided;
  check_policy(idx.guided_entries_.Bytes() > 0, "guided directory missing");
  idx.PrepareSearchCIDs();
  check_policy(idx.guided_entries_.Bytes() == 0, "CID refresh did not invalidate navigation");
  idx.PrepareGuidedEntries();
  check_policy(idx.guided_entries_.Bytes() > 0, "guided directory failed to rebuild");
  {
    attr_range<int> empty{int(n) + 10, int(n) + 20};
    const auto before = idx.metric_ml_entry_estimates_.load();
    check_policy(idx.searchKNNMultiLevelPipeline(queries.data(), 16, k, empty, &ml_ctx).empty(),
                 "empty range entries");
    check_policy(idx.metric_ml_entry_estimates_.load() == before, "empty range must not sample");
  }
  printf("Rank entry integration contracts PASSED\n");

  // Landing weights are offline inputs. Structural routing needs no query IO,
  // and a real search must leave the prepared histogram untouched.
  idx.landing_layer_cnt_.assign(idx.cur_max_layer_ + 1, 7);
  const auto histogram = idx.landing_layer_cnt_;
  const auto reads_before_routing = idx.metric_ml_io_submits_.load();
  const auto distances_before_routing = idx.metric_dist_comps_.load();
  for (const attr_range<int> range :
       {full_filter, attr_range<int>{0, narrow_hi}, attr_range<int>{0, 0}}) {
    std::vector<tableint> entries;
    const auto route_layers = idx.DecideLayerRange(range, entries);
    check_policy(route_layers.l_ == 0 && route_layers.u_ >= 0 &&
                     size_t(route_layers.u_) <= idx.cur_max_layer_,
                 "structural landing lies outside the built ladder");
    check_policy(!entries.empty(), "nonempty range must have structural entries");
    for (const auto id : entries)
      check_policy(range.Test(idx.GetAttByID(id)), "structural entry lies outside its range");
    const size_t owner = idx.ml_layer_to_level_[route_layers.u_];
    check_policy(owner < idx.ml_levels_.size() && FindLevelForLayer(route_layers.u_, idx.ml_levels_,
                                                                    idx.ml_layer_stride_) == owner,
                 "structural landing disagrees with layout ownership");
  }
  check_policy(idx.metric_ml_io_submits_ == reads_before_routing &&
                   idx.metric_dist_comps_ == distances_before_routing,
               "structural routing performed graph IO or distance estimation");
  ml_ctx.rerank_num = 2 * k;
  idx.searchKNNMultiLevelPipeline(queries.data(), 40, k, full_filter, &ml_ctx);
  check_policy(idx.landing_layer_cnt_ == histogram, "search changed offline landing weights");

  // Exercise optional RAM navigation through the real ML pipeline and final
  // exact rerank. This tiny all-rank fixture owns the same layers as the layout;
  // adjacent-rank edges obey every window without rebuilding the disk graph.
  {
    MemoryNavigationGraph navigation;
    navigation.population = n;
    navigation.dimension = d;
    navigation.metric = 2;
    navigation.degree = 2;
    navigation.order_hash = MemoryNavigationGraph::OrderHash(idx.sorted_ids_, idx.atts_);
    navigation.windows.assign(idx.window_size_.begin(),
                              idx.window_size_.begin() + idx.cur_max_layer_ + 1);
    for (const auto &level : idx.ml_levels_) {
      MemoryNavigationGraph::Group group;
      group.first = level.owner_begin;
      group.last = level.owner_end;
      group.ids.assign(idx.sorted_ids_.begin(), idx.sorted_ids_.end());
      group.ranks.resize(n);
      std::iota(group.ranks.begin(), group.ranks.end(), 0);
      group.edges.assign((group.last - group.first + 1) * n * navigation.degree,
                         MemoryNavigationGraph::invalid);
      for (size_t layer = group.first; layer <= group.last; ++layer) {
        for (size_t rank = 0; rank < n; ++rank) {
          size_t offset = ((layer - group.first) * n + rank) * navigation.degree;
          if (rank > 0)
            group.edges[offset++] = rank - 1;
          if (rank + 1 < n)
            group.edges[offset] = rank + 1;
        }
      }
      navigation.groups.push_back(std::move(group));
    }
    const auto navigation_path = tmp_dir + "/test_ml_search_navigation.bin";
    test_support::WriteLegacyNavigation(navigation_path, navigation);
    idx.LoadNavigation(navigation_path);
    check_policy(idx.HasNavigation() && idx.NavigationBytes() > 0 && idx.guided_entries_.Bytes() == 0,
                 "loading RAM navigation must release guided directory");
    bool mixed_entries_rejected = false;
    try {
      idx.PrepareGuidedEntries();
    } catch (const std::exception &) {
      mixed_entries_rejected = true;
    }
    check_policy(mixed_entries_rejected, "RAM navigation must not allocate a guided directory");
    // Leaving guided knobs nonzero tests the core dispatch, not only CLI normalization.
    ml_ctx.entry_sampling = DiskSearchContext::EntrySampling::Guided;
    ml_ctx.entry_samples = 256;
    ml_ctx.entry_count = 4;
    ml_ctx.navigation_ef = 64;
    ml_ctx.navigation_budget = 256;
    ml_ctx.navigation_count = 16;
    ml_ctx.filter_mode = DiskSearchContext::FilterMode::Graph;
    const auto before_guided = idx.metric_ml_entry_estimates_.load();
    const auto before_nav = idx.metric_navigation_queries_.load();
    const auto before_scores = idx.metric_navigation_scores_.load();
    attr_range<int> nav_range{100, 1000};
    for (size_t query = 0; query < 2; ++query) {
      const auto *vector = queries.data() + query * d;
      const auto result = idx.searchKNNMultiLevelPipeline(
          const_cast<float *>(vector), 40, k, nav_range, &ml_ctx);
      std::unordered_set<label_t> labels;
      check_policy(result.size() == k, "navigation search must return full top-k");
      for (const auto &[distance, label] : result) {
        check_policy(label < n && labels.insert(label).second && nav_range.Test(att_vec[label]),
                     "navigation result must satisfy ID uniqueness and predicate");
        float exact = 0;
        for (size_t dimension = 0; dimension < d; ++dimension) {
          const float delta = vector[dimension] - base_vecs[label * d + dimension];
          exact += delta * delta;
        }
        check_policy(std::abs(distance - exact) <= 1e-4f * std::max(1.f, exact),
                     "navigation must retain exact final rerank distances");
      }
    }
    check_policy(idx.metric_navigation_queries_ == before_nav + 2 &&
                     idx.metric_navigation_scores_ > before_scores &&
                     idx.metric_navigation_scores_ <= before_scores + 512 &&
                     idx.metric_ml_entry_estimates_ == before_guided && idx.guided_entries_.Bytes() == 0,
                 "pipeline must use only bounded RAM navigation entries");
    idx.PrepareSearchCIDs();
    check_policy(!idx.HasNavigation(), "CID refresh must invalidate RAM navigation");
    idx.LoadNavigation(navigation_path);
    idx.InitRaBitQ(centroids_path, cids_path, "l2");
    check_policy(!idx.HasNavigation(), "quantization initialization must invalidate RAM navigation");
    idx.LoadNavigation(navigation_path);
    idx.LoadMultiLevelLayout(out_graph, cached_meta, 32 * 1024 * 1024,
                            DynamicBlockCache::EvictPolicy::FIFO);
    check_policy(!idx.HasNavigation(), "layout reload must invalidate RAM navigation");
    idx.PrepareGuidedEntries();
    check_policy(idx.guided_entries_.Bytes() > 0, "guided must remain available after navigation unload");
    std::remove(navigation_path.c_str());
    printf("RAM navigation ML integration and invalidation PASSED\n");
  }

  std::remove(centroids_path.c_str());
  std::remove(cids_path.c_str());

  if (all_ok) {
    printf("====== ALL TESTS PASSED ======\n");
    return 0;
  } else {
    printf("====== SOME TESTS FAILED ======\n");
    return 1;
  }
}
