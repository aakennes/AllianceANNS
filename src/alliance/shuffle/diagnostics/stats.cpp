#include "shuffle/diagnostics/stats.hpp"
#include "index/index_disk.hpp"
#include <algorithm>
#include <numeric>
#include <cstdio>
#include <stdexcept>
namespace alliance::shuffle_detail {
void ReportBlockAffinity(const FlatWeightedGraph &graph, const std::vector<size_t> &blocks,
                         const std::vector<size_t> &order) {
  std::vector<double> scores(graph.num_nodes, 0);
  for (size_t base = 0; base < order.size();) {
    const size_t count = graph.external ? graph.external->LoadBatch(base, 16384)
                                        : std::min(size_t(16384), order.size() - base);
    for (size_t i = 0; i < count; ++i) {
      const size_t u = order[base + i];
      const auto row = graph.Row(u, base + i);
      for (size_t e = 0; e < row.size; ++e)
        if (blocks[u] == blocks[row.Neighbor(e)])
          scores[u] += row.Weight(e);
    }
    base += count;
  }
  std::sort(scores.begin(), scores.end());
  const double mean =
      scores.empty() ? 0 : std::accumulate(scores.begin(), scores.end(), 0.0) / scores.size();
  auto quantile = [&](double p) {
    return scores.empty() ? 0.0 : scores[size_t(p * (scores.size() - 1))];
  };
  fprintf(stderr, "  [block-affinity] converged: mean=%.2f p10=%.1f p50=%.1f p90=%.1f p99=%.1f\n",
          mean, quantile(.1), quantile(.5), quantile(.9), quantile(.99));
}
} // namespace alliance::shuffle_detail
namespace alliance {

template <typename att_t, typename vec_t>
auto ComputeEdgeLocalityStats(AllianceIndexDisk<att_t, vec_t> *idx,
                              const std::vector<tableint> &order, layer_t layer_begin,
                              layer_t layer_end, size_t stride) -> ShuffleStats {
  const size_t n = idx->curvec_num_;
  std::vector<size_t> new_pos_of(n);
  for (size_t p = 0; p < n; ++p)
    new_pos_of[order[p]] = p;

  // Spans are integers in [0,n). An n-bucket histogram gives exact percentiles
  // without storing or sorting one entry per graph edge.
  std::vector<uint64_t> hist(n, 0);
  std::vector<tableint> buf(idx->M_ + 1);
  double sum = 0;
  size_t num_edges = 0;

  for (tableint u = 0; u < static_cast<tableint>(n); ++u) {
    for (layer_t layer = layer_begin; layer <= layer_end; layer += static_cast<layer_t>(stride)) {
      idx->ReadEdges(u, layer, buf.data());
      tableint sz = buf[idx->M_];
      for (tableint j = 0; j < sz; ++j) {
        tableint v = buf[j];
        if (v == u || v >= static_cast<tableint>(n))
          continue;
        const int64_t d = static_cast<int64_t>(new_pos_of[u]) - static_cast<int64_t>(new_pos_of[v]);
        const size_t span = static_cast<size_t>(d < 0 ? -d : d);
        ++hist[span];
        sum += static_cast<double>(span);
        ++num_edges;
      }
    }
  }

  ShuffleStats stats;
  stats.num_edges = num_edges;
  if (num_edges == 0)
    return stats;

  stats.avg_edge_span = sum / static_cast<double>(num_edges);

  // max = highest non-empty bucket. p90 matches the old sorted-position
  // semantics exactly: value at sorted index floor(0.9*(num_edges-1)), i.e. the
  // smallest span whose cumulative count strictly exceeds that index.
  const size_t p90_idx = static_cast<size_t>(0.9 * static_cast<double>(num_edges - 1));
  size_t cum = 0;
  bool p90_set = false;
  for (size_t s = 0; s < n; ++s) {
    if (hist[s] == 0)
      continue;
    if (!p90_set) {
      cum += hist[s];
      if (cum > p90_idx) {
        stats.p90_edge_span = static_cast<double>(s);
        p90_set = true;
      }
    }
    stats.max_edge_span = static_cast<double>(s); // last non-empty bucket wins
  }
  return stats;
}

template auto ComputeEdgeLocalityStats<int, float>(AllianceIndexDisk<int, float> *,
                                                   const std::vector<tableint> &, layer_t, layer_t,
                                                   size_t) -> ShuffleStats;

} // namespace alliance
