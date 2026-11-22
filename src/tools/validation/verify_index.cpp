// verify_index.cpp — Diagnostic tool for verifying disk index correctness.
//
// Checks:
// 1. Graph structure: edge counts, valid IDs, layer coverage
// 2. Stored edge distances vs recomputed exact distances
// 3. RaBitQ distance estimate accuracy (rank correlation with exact)
// 4. Entry point selection quality
// 5. Per-query recall breakdown (graph search vs brute force)
//
// Usage:
//   ./verify_index --graph_file <path> --meta_file <path>
//       --query_vec <path> --query_rng <path> --gt_file <path>
//       [--space ip] [--k 10] [--efs 200] [--sample 50]
//       [--centroids <path> --cids <path>]

#include "index/index_disk.hpp"
#include "common/bench_utils.hpp"
#include <chrono>
#include <numeric>
#include <queue>
#include <random>
#include <set>

using namespace alliance;

struct VerifyConfig {
  std::string graph_file, meta_file, query_vec_path, query_rng_path, gt_file;
  std::string space = "ip";
  std::string centroids, cids_path;
  size_t k = 10;
  size_t efs = 200;
  size_t sample = 50;
};

static VerifyConfig parse_args(int argc, char **argv) {
  VerifyConfig c;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--graph_file") == 0)
      c.graph_file = argv[++i];
    else if (strcmp(argv[i], "--meta_file") == 0)
      c.meta_file = argv[++i];
    else if (strcmp(argv[i], "--query_vec") == 0)
      c.query_vec_path = argv[++i];
    else if (strcmp(argv[i], "--query_rng") == 0)
      c.query_rng_path = argv[++i];
    else if (strcmp(argv[i], "--gt_file") == 0)
      c.gt_file = argv[++i];
    else if (strcmp(argv[i], "--space") == 0)
      c.space = argv[++i];
    else if (strcmp(argv[i], "--k") == 0)
      c.k = std::stoul(argv[++i]);
    else if (strcmp(argv[i], "--efs") == 0)
      c.efs = std::stoul(argv[++i]);
    else if (strcmp(argv[i], "--sample") == 0)
      c.sample = std::stoul(argv[++i]);
    else if (strcmp(argv[i], "--centroids") == 0)
      c.centroids = argv[++i];
    else if (strcmp(argv[i], "--cids") == 0)
      c.cids_path = argv[++i];
    else
      throw std::runtime_error("unknown argument: " + std::string(argv[i]));
  }
  return c;
}

// ============================================================================
// Test 1: Graph structure validation
// ============================================================================
static void verify_graph_structure(AllianceIndexDisk<int, float> &idx, size_t sample_n) {
  printf("\n=== Test 1: Graph Structure ===\n");
  size_t N = idx.curvec_num_;
  size_t M = idx.M_;
  size_t wp = idx.wp_;

  printf("  N=%zu, M=%zu, wp=%zu, window_size=[", N, M, wp);
  for (size_t i = 0; i <= wp; ++i) {
    printf("%zu%s", idx.window_size_[i], i < wp ? "," : "");
  }
  printf("]\n");

  std::mt19937 rng(42);
  std::vector<size_t> sample_ranks(N);
  std::iota(sample_ranks.begin(), sample_ranks.end(), 0);
  std::shuffle(sample_ranks.begin(), sample_ranks.end(), rng);
  sample_ranks.resize(std::min(sample_n, N));

  std::vector<tableint> edge_buf(M + 1);
  std::vector<size_t> layer_edge_count(wp + 1, 0);
  std::vector<size_t> layer_node_has_edges(wp + 1, 0);
  size_t invalid_edges = 0;
  size_t self_loops = 0;
  size_t duplicate_edges = 0;

  for (size_t rank : sample_ranks) {
    tableint id = idx.sorted_ids_[rank];
    for (layer_t layer = 0; layer <= (layer_t)wp; ++layer) {
      idx.ReadEdges(id, layer, edge_buf.data());
      tableint cnt = edge_buf[M];
      if (cnt > 0) {
        layer_node_has_edges[layer]++;
        layer_edge_count[layer] += cnt;
      }
      std::set<tableint> seen;
      for (tableint i = 0; i < cnt; ++i) {
        tableint nn = edge_buf[i];
        if (nn < 0 || nn >= (tableint)N) {
          invalid_edges++;
        }
        if (nn == id)
          self_loops++;
        if (seen.count(nn))
          duplicate_edges++;
        seen.insert(nn);
      }
    }
  }

  printf("  Sampled %zu nodes:\n", sample_ranks.size());
  for (layer_t layer = 0; layer <= (layer_t)wp; ++layer) {
    printf("    layer %d: %zu/%zu have edges, avg_degree=%.1f\n", layer,
           layer_node_has_edges[layer], sample_ranks.size(),
           layer_node_has_edges[layer] > 0
               ? (double)layer_edge_count[layer] / layer_node_has_edges[layer]
               : 0.0);
  }
  printf("  Invalid edge IDs: %zu\n", invalid_edges);
  printf("  Self-loops: %zu\n", self_loops);
  printf("  Duplicate edges: %zu\n", duplicate_edges);
}

// ============================================================================
// Test 2: Edge distance correctness
// ============================================================================
static void verify_edge_distances(AllianceIndexDisk<int, float> &idx, size_t sample_n) {
  printf("\n=== Test 2: Edge Distance Correctness ===\n");
  size_t N = idx.curvec_num_;
  size_t M = idx.M_;
  size_t wp = idx.wp_;
  size_t d = idx.vec_d_;

  std::mt19937 rng(123);
  std::vector<size_t> sample_ranks(N);
  std::iota(sample_ranks.begin(), sample_ranks.end(), 0);
  std::shuffle(sample_ranks.begin(), sample_ranks.end(), rng);
  sample_ranks.resize(std::min(sample_n, N));

  std::vector<tableint> edge_ids(M + 1);
  std::vector<dist_t> edge_dists(M);
  std::vector<float> vec_a(d), vec_b(d);

  size_t total_edges = 0;
  size_t mismatch_count = 0;
  double max_rel_error = 0;
  double sum_rel_error = 0;

  for (size_t rank : sample_ranks) {
    tableint id = idx.sorted_ids_[rank];
    idx.ReadVec(id, vec_a.data());

    for (layer_t layer = 0; layer <= (layer_t)wp; ++layer) {
      idx.ReadEdgesWithDist(id, layer, edge_ids.data(), edge_dists.data());
      tableint cnt = edge_ids[M];
      for (tableint i = 0; i < cnt; ++i) {
        tableint nn = edge_ids[i];
        if (nn < 0 || nn >= (tableint)N)
          continue;
        idx.ReadVec(nn, vec_b.data());
        dist_t exact = idx.fstdistfunc_(vec_a.data(), vec_b.data(), idx.dist_func_param_);
        dist_t stored = edge_dists[i];
        total_edges++;
        if (stored != 0.0f) {
          double rel = std::abs((double)(exact - stored) / (std::abs((double)stored) + 1e-10));
          if (rel > 0.01)
            mismatch_count++;
          max_rel_error = std::max(max_rel_error, rel);
          sum_rel_error += rel;
        } else if (exact != 0.0f) {
          mismatch_count++;
        }
      }
    }
  }

  printf("  Checked %zu edges across %zu nodes\n", total_edges, sample_ranks.size());
  printf("  Mismatches (>1%% relative error): %zu (%.2f%%)\n", mismatch_count,
         total_edges > 0 ? 100.0 * mismatch_count / total_edges : 0.0);
  printf("  Max relative error: %.6f\n", max_rel_error);
  printf("  Mean relative error: %.6f\n", total_edges > 0 ? sum_rel_error / total_edges : 0.0);
}

// ============================================================================
// Test 3: RaBitQ distance estimate accuracy
// ============================================================================
static void verify_rabitq_accuracy(AllianceIndexDisk<int, float> &idx, float *queries, size_t nq,
                                   size_t d, size_t sample_n) {
  printf("\n=== Test 3: RaBitQ Distance Estimate Accuracy ===\n");
  if (!idx.rabitq_initialized_) {
    printf("  SKIPPED (RaBitQ not initialized)\n");
    return;
  }

  size_t N = idx.curvec_num_;
  std::mt19937 rng(77);

  size_t num_queries = std::min(sample_n, nq);
  size_t neighbors_per_query = 100;

  size_t total_pairs = 0;
  size_t rank_inversions = 0;
  double sum_ratio = 0;
  double max_ratio = 0;
  double min_ratio = 1e9;

  std::vector<float> vec_buf(d);

  for (size_t qi = 0; qi < num_queries; ++qi) {
    const float *q = queries + qi * d;
    AllianceIndexDisk<int, float>::RaBitQEstimator estimator(q, &idx);

    std::vector<tableint> sample_ids(N);
    std::iota(sample_ids.begin(), sample_ids.end(), 0);
    std::shuffle(sample_ids.begin(), sample_ids.end(), rng);
    sample_ids.resize(std::min(neighbors_per_query, N));

    std::vector<std::pair<dist_t, dist_t>> exact_vs_approx;
    for (tableint id : sample_ids) {
      idx.ReadVec(id, vec_buf.data());
      dist_t exact = idx.fstdistfunc_(q, vec_buf.data(), idx.dist_func_param_);
      auto [approx, error_bound] =
          estimator.dist_estimate_with_errorbound(idx.cids_[idx.labels_[id]], id);
      exact_vs_approx.emplace_back(exact, approx);
      total_pairs++;

      double ratio = (std::abs((double)exact) > 1e-10) ? (double)approx / exact : 1.0;
      sum_ratio += ratio;
      max_ratio = std::max(max_ratio, ratio);
      min_ratio = std::min(min_ratio, ratio);
    }

    // Check rank inversions: pairs where approx ordering disagrees with exact
    for (size_t i = 0; i < exact_vs_approx.size(); ++i) {
      for (size_t j = i + 1; j < exact_vs_approx.size(); ++j) {
        bool exact_order = exact_vs_approx[i].first < exact_vs_approx[j].first;
        bool approx_order = exact_vs_approx[i].second < exact_vs_approx[j].second;
        if (exact_order != approx_order)
          rank_inversions++;
      }
    }
  }

  size_t total_pairs_checked = num_queries * neighbors_per_query * (neighbors_per_query - 1) / 2;
  printf("  Checked %zu query-node pairs across %zu queries\n", total_pairs, num_queries);
  printf("  Approx/Exact ratio: mean=%.4f, min=%.4f, max=%.4f\n", sum_ratio / total_pairs,
         min_ratio, max_ratio);
  printf("  Rank inversions: %zu / %zu pairs (%.2f%%)\n", rank_inversions, total_pairs_checked,
         total_pairs_checked > 0 ? 100.0 * rank_inversions / total_pairs_checked : 0.0);
}

// ============================================================================
// Test 4: Entry point quality
// ============================================================================
static void verify_entry_points(AllianceIndexDisk<int, float> &idx, float *queries, size_t nq,
                                size_t d, std::vector<attr_range<int>> &filters, size_t sample_n) {
  printf("\n=== Test 4: Entry Point Quality ===\n");

  size_t num_queries = std::min(sample_n, nq);
  std::vector<float> vec_buf(d);

  size_t total_eps = 0;
  size_t eps_in_filter = 0;

  for (size_t qi = 0; qi < num_queries; ++qi) {
    std::vector<tableint> eps;
    auto layer_rng = idx.DecideLayerRange(filters[qi], eps);

    for (auto ep_id : eps) {
      total_eps++;
      int att = idx.atts_[ep_id];
      if (att >= filters[qi].l_ && att <= filters[qi].u_) {
        eps_in_filter++;
      }
    }

    if (qi < 5) {
      printf("  Query %zu: filter=[%d,%d], layer_rng=[%d,%d], #eps=%zu", qi, filters[qi].l_,
             filters[qi].u_, layer_rng.l_, layer_rng.u_, eps.size());
      if (!eps.empty()) {
        printf(", ep_atts=[");
        for (size_t i = 0; i < std::min(eps.size(), (size_t)5); ++i) {
          printf("%d%s", idx.atts_[eps[i]], i < std::min(eps.size(), (size_t)5) - 1 ? "," : "");
        }
        printf("]");
      }
      printf("\n");
    }
  }

  printf("  Total entry points: %zu across %zu queries (avg %.1f/query)\n", total_eps, num_queries,
         (double)total_eps / num_queries);
  printf("  EPs within filter range: %zu/%zu (%.1f%%)\n", eps_in_filter, total_eps,
         total_eps > 0 ? 100.0 * eps_in_filter / total_eps : 0.0);
}

// ============================================================================
// Test 5: Graph search with exact distances (bypass RaBitQ)
// Standalone greedy beam search reading edges/vectors from disk.
// ============================================================================
static void verify_graph_search_exact(AllianceIndexDisk<int, float> &idx, float *queries, size_t nq,
                                      size_t d, std::vector<attr_range<int>> &filters,
                                      std::vector<std::vector<label_t>> &gt, size_t efs, size_t k,
                                      size_t sample_n) {
  printf("\n=== Test 5: Graph Search with Exact Distances (no RaBitQ) ===\n");
  printf("  (Pure exact-distance greedy graph traversal, no RaBitQ filtering)\n");

  size_t N = idx.curvec_num_;
  size_t M = idx.M_;
  size_t num_queries = std::min(sample_n, nq);
  size_t correct = 0, total = 0;

  std::vector<float> vec_buf(d);
  std::vector<tableint> edge_buf(M + 1);

  for (size_t qi = 0; qi < num_queries; ++qi) {
    const float *q = queries + qi * d;

    // Get entry points via DecideLayerRange
    std::vector<tableint> ep_ids;
    auto layer_rng = idx.DecideLayerRange(filters[qi], ep_ids);
    if (ep_ids.empty())
      continue;

    // Build dedup filter
    attr_range<att_label_t<int>> dedup_filter{
        {filters[qi].l_, 0}, {filters[qi].u_, std::numeric_limits<label_t>::max()}};

    // Greedy beam search with exact distances (best-first via min-heap)
    VisitedList<tableint> visited(N);
    visited.Clear();

    // Min-heap for expansion order (best-first)
    auto cmp_min = [](const dist_id_pair &a, const dist_id_pair &b) { return a.dist_ > b.dist_; };
    std::priority_queue<dist_id_pair, std::vector<dist_id_pair>, decltype(cmp_min)> expand_q(
        cmp_min);
    // Max-heap for result (top-efs worst at top)
    std::vector<dist_id_pair> result;

    for (auto ep_id : ep_ids) {
      idx.ReadVec(ep_id, vec_buf.data());
      dist_t dist = idx.fstdistfunc_(q, vec_buf.data(), idx.dist_func_param_);
      expand_q.push({dist, ep_id});
      PUSH_HEAP(result, dist, ep_id);
      visited.Set(ep_id);
    }
    dist_t res_max_dist = TOP_HEAP(result).dist_;

    size_t hops = 0;
    while (!expand_q.empty() && hops < efs) {
      auto [cur_dist, cur_id] = expand_q.top();
      expand_q.pop();
      hops++;

      // Expand neighbors across all layers in range
      for (layer_t layer = layer_rng.u_; layer >= layer_rng.l_; --layer) {
        idx.ReadEdges(cur_id, layer, edge_buf.data());
        tableint ll_sz = edge_buf[M];
        if (ll_sz > (tableint)M)
          ll_sz = 0;

        for (tableint i = 0; i < ll_sz; ++i) {
          tableint nn_id = edge_buf[i];
          if (nn_id >= (tableint)N)
            continue;
          if (!dedup_filter.Test({idx.atts_[nn_id], idx.labels_[nn_id]}))
            continue;
          if (visited.Test(nn_id))
            continue;
          visited.Set(nn_id);

          idx.ReadVec(nn_id, vec_buf.data());
          dist_t nn_dist = idx.fstdistfunc_(q, vec_buf.data(), idx.dist_func_param_);

          expand_q.push({nn_dist, nn_id});

          if (result.size() < efs || nn_dist < res_max_dist) {
            PUSH_HEAP(result, nn_dist, nn_id);
            if (result.size() > efs)
              POP_HEAP(result);
            res_max_dist = TOP_HEAP(result).dist_;
          }
        }
      }
    }

    // Extract top-k from result
    std::vector<std::pair<dist_t, label_t>> final_result;
    for (auto &r : result) {
      final_result.emplace_back(r.dist_, idx.labels_[r.id_]);
    }
    std::sort(final_result.begin(), final_result.end());
    final_result.resize(std::min(k, final_result.size()));

    // Check recall
    for (auto &[dd, label] : final_result) {
      if (std::find(gt[qi].begin(), gt[qi].end(), label) != gt[qi].end()) {
        correct++;
      }
    }
    total += gt[qi].size();
  }

  printf("  Recall (exact graph search, efs=%zu): %.4f (%zu/%zu)\n", efs,
         total > 0 ? (double)correct / total : 0.0, correct, total);
  printf("  (If this is high but RaBitQ search recall is low → problem is in RaBitQ)\n");
  printf("  (If this is also low → problem is in graph structure or entry points)\n");
}

// ============================================================================
// Test 6: RaBitQ search recall (normal searchKNN path)
// ============================================================================
static void verify_rabitq_search(AllianceIndexDisk<int, float> &idx, float *queries, size_t nq,
                                 size_t d, std::vector<attr_range<int>> &filters,
                                 std::vector<std::vector<label_t>> &gt, size_t efs, size_t k,
                                 size_t sample_n) {
  printf("\n=== Test 6: RaBitQ Graph Search (normal searchKNN path) ===\n");
  if (!idx.rabitq_initialized_) {
    printf("  SKIPPED (RaBitQ not initialized)\n");
    return;
  }

  size_t num_queries = std::min(sample_n, nq);
  size_t correct = 0, total = 0;

  for (size_t qi = 0; qi < num_queries; ++qi) {
    float *q = queries + qi * d;
    auto result = idx.searchKNN(q, efs, k, filters[qi]);

    for (auto &[d, label] : result) {
      if (std::find(gt[qi].begin(), gt[qi].end(), label) != gt[qi].end()) {
        correct++;
      }
    }
    total += gt[qi].size();
  }

  printf("  Recall (RaBitQ search, efs=%zu): %.4f (%zu/%zu)\n", efs,
         total > 0 ? (double)correct / total : 0.0, correct, total);
}

// ============================================================================
// Test 7: Detailed per-query diagnostic (first few queries)
// ============================================================================
static void detailed_per_query(AllianceIndexDisk<int, float> &idx, float *queries, size_t nq,
                               size_t d, std::vector<attr_range<int>> &filters,
                               std::vector<std::vector<label_t>> &gt, size_t efs, size_t k) {
  printf("\n=== Test 7: Detailed Per-Query Diagnostic (first 5 queries) ===\n");

  size_t num_queries = std::min((size_t)5, nq);
  std::vector<float> vec_buf(d);

  for (size_t qi = 0; qi < num_queries; ++qi) {
    const float *q = queries + qi * d;
    printf("\n  --- Query %zu: filter=[%d,%d] ---\n", qi, filters[qi].l_, filters[qi].u_);

    // Ground truth: show top-k labels and their distances
    printf("    GT labels (first 5): ");
    for (size_t i = 0; i < std::min((size_t)5, gt[qi].size()); ++i) {
      printf("%d ", gt[qi][i]);
    }
    printf("\n");

    // Find GT nodes' internal IDs and compute distances
    printf("    GT distances: ");
    for (size_t i = 0; i < std::min((size_t)5, gt[qi].size()); ++i) {
      label_t gt_label = gt[qi][i];
      // Find internal ID for this label
      tableint gt_id = -1;
      for (size_t j = 0; j < idx.curvec_num_; ++j) {
        if (idx.labels_[j] == gt_label) {
          gt_id = j;
          break;
        }
      }
      if (gt_id >= 0) {
        idx.ReadVec(gt_id, vec_buf.data());
        dist_t exact = idx.fstdistfunc_(q, vec_buf.data(), idx.dist_func_param_);
        printf("%.4f(att=%d) ", exact, idx.atts_[gt_id]);
      }
    }
    printf("\n");

    // Entry points
    std::vector<tableint> ep_ids;
    auto layer_rng = idx.DecideLayerRange(filters[qi], ep_ids);
    printf("    layer_rng=[%d,%d], #eps=%zu\n", layer_rng.l_, layer_rng.u_, ep_ids.size());

    // EP distances
    if (!ep_ids.empty()) {
      printf("    EP exact distances: ");
      for (size_t i = 0; i < std::min(ep_ids.size(), (size_t)5); ++i) {
        idx.ReadVec(ep_ids[i], vec_buf.data());
        dist_t exact = idx.fstdistfunc_(q, vec_buf.data(), idx.dist_func_param_);
        printf("%.4f(id=%d,att=%d) ", exact, ep_ids[i], idx.atts_[ep_ids[i]]);
      }
      printf("\n");
    }

    // Search result via RaBitQ
    if (idx.rabitq_initialized_) {
      float *q_mut = const_cast<float *>(q);
      auto result = idx.searchKNN(q_mut, efs, k, filters[qi]);
      printf("    RaBitQ search result (top 5): ");
      for (size_t i = 0; i < std::min(result.size(), (size_t)5); ++i) {
        printf("(d=%.4f,label=%d) ", result[i].first, result[i].second);
      }
      printf("\n");
      // Check which GT labels are found
      size_t found = 0;
      for (auto &[dd, label] : result) {
        if (std::find(gt[qi].begin(), gt[qi].end(), label) != gt[qi].end())
          found++;
      }
      printf("    Recall for this query: %zu/%zu\n", found, gt[qi].size());
    }
  }
}

// ============================================================================
int main(int argc, char **argv) {
  VerifyConfig cfg = parse_args(argc, argv);

  if (cfg.graph_file.empty() || cfg.meta_file.empty()) {
    fprintf(stderr, "Usage: verify_index --graph_file <path> --meta_file <path>\n"
                    "  --query_vec <path> --query_rng <path> --gt_file <path>\n"
                    "  [--space ip] [--k 10] [--efs 200] [--sample 50]\n"
                    "  [--centroids <path> --cids <path>]\n");
    return 1;
  }

  printf("=== AllianceIndexDisk Verification Tool ===\n");
  printf("  graph_file: %s\n", cfg.graph_file.c_str());
  printf("  meta_file: %s\n", cfg.meta_file.c_str());
  printf("  space: %s\n", cfg.space.c_str());

  AllianceIndexDisk<int, float> idx(cfg.graph_file, cfg.meta_file, cfg.space);
  printf("  Loaded: N=%zu, M=%zu, wp=%zu, d=%zu\n", idx.curvec_num_, idx.M_, idx.wp_, idx.vec_d_);

  // Open vec file if it exists
  std::string vec_file = cfg.graph_file + ".rawvecs";
  idx.vec_file_path_ = vec_file;
  try {
    idx.OpenVecFile();
  } catch (...) {
  }

  // Test 1: Graph structure
  verify_graph_structure(idx, std::min(cfg.sample * 10, idx.curvec_num_));

  // Test 2: Edge distances
  verify_edge_distances(idx, cfg.sample);

  // Load queries if available
  float *queries = nullptr;
  size_t nq = 0, d = 0;
  std::vector<attr_range<int>> filters;
  std::vector<std::vector<label_t>> gt;

  if (!cfg.query_vec_path.empty()) {
    queries = benchmark::fvecs_read(cfg.query_vec_path, d, nq);
    printf("\n  Loaded queries: nq=%zu, d=%zu\n", nq, d);
  }
  if (!cfg.query_rng_path.empty()) {
    filters = benchmark::LoadRange(cfg.query_rng_path);
    nq = std::min(nq, filters.size());
  }
  if (!cfg.gt_file.empty()) {
    gt = benchmark::LoadGroundTruth(cfg.gt_file);
    nq = std::min(nq, gt.size());
  }

  // Initialize RaBitQ if centroids provided
  if (!cfg.centroids.empty() && !cfg.cids_path.empty()) {
    idx.InitRaBitQ(cfg.centroids, cfg.cids_path, cfg.space);
    printf("  RaBitQ initialized.\n");
  }

  if (queries && !filters.empty()) {
    // Test 3: RaBitQ accuracy
    verify_rabitq_accuracy(idx, queries, nq, d, cfg.sample);

    // Test 4: Entry points
    verify_entry_points(idx, queries, nq, d, filters, cfg.sample);

    if (!gt.empty()) {
      // Test 5: Graph search with exact distances
      verify_graph_search_exact(idx, queries, nq, d, filters, gt, cfg.efs, cfg.k, cfg.sample);

      // Test 6: RaBitQ search
      verify_rabitq_search(idx, queries, nq, d, filters, gt, cfg.efs, cfg.k, cfg.sample);

      // Test 7: Per-query diagnostic
      detailed_per_query(idx, queries, nq, d, filters, gt, cfg.efs, cfg.k);
    }
  }

  printf("\n=== Verification Complete ===\n");
  if (queries)
    delete[] queries;
  return 0;
}
