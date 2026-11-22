#include "shuffle/order/block_bnf.hpp"
#include "index/index_disk.hpp"
#include "shuffle/adjacency/weighted_graph.hpp"
#include "shuffle/order/ordered_block_bnf.hpp"
#include "shuffle/diagnostics/stats.hpp"
#include <algorithm>
#include <cstdio>
#include <stdexcept>

namespace alliance::shuffle_detail {
template <typename att_t, typename vec_t>
auto ShuffleBlockBNF(AllianceIndexDisk<att_t, vec_t> *idx, const BNFParams &params,
                     layer_t layer_begin, layer_t layer_end, size_t record_num_layers,
                     size_t stride, bool collect_diagnostics) -> std::vector<tableint> {
  if (params.threads < 1 || params.threads > 256 || !params.affinity_batch_size ||
      params.affinity_batch_size > 1048576)
    throw std::invalid_argument("invalid BlockBNF threads/batch size");
  const size_t n = idx->curvec_num_;
  if (n == 0)
    return {};

  const size_t graph_layers = CountSelectedLayers(layer_begin, layer_end, stride);
  const size_t num_layers_for_record = record_num_layers > 0 ? record_num_layers : graph_layers;
  const size_t record_size = ComputeRecordSize(num_layers_for_record, idx->M_, sizeof(att_t));
  const size_t elements_per_block = ComputeElementsPerBlock(params.block_size, record_size);
  const size_t num_blocks = (n + elements_per_block - 1) / elements_per_block;

  fprintf(stderr,
          "  BlockBNF: n=%zu blocks=%zu epb=%zu graph_layers=[%d..%d](%zu) "
          "record_layers=%zu record=%zu bytes\n",
          n, num_blocks, elements_per_block, layer_begin, layer_end, graph_layers,
          num_layers_for_record, record_size);

  fprintf(stderr, "  BlockBNF: building weighted graph...\n");
  const double level_start = ShuffleSeconds();
  auto graph = BuildWeightedGraph(idx, layer_begin, layer_end, stride, params.threads, params,
                                  elements_per_block);
  const double csr_end = ShuffleSeconds();
  fprintf(stderr, "  BlockBNF: graph built, %zu edges total\n", graph.offsets[n]);

  // Node processing order: IN-weight descending (fixed across all iterations).
  auto node_order = SortNodesByInWeight(graph.in_weights);
  const double sort_end = ShuffleSeconds();
  // The fixed processing order is its only consumer. Do not retain 8*N bytes
  // across the ordering rounds; clear() alone would retain the allocation.
  std::vector<size_t>().swap(graph.in_weights);
  if (graph.external) {
    // Preparation is a one-time transpose/reordering of adjacency storage.
    // The fixed node order and every later commit decision remain unchanged.
    graph.external->Prepare(node_order, graph.offsets);
    std::vector<size_t>().swap(graph.offsets);
  }

  // Identity initial block assignment: node i → block i/epb.
  OrderedBlockBNF workspace(graph, elements_per_block, params);
  std::vector<size_t> current_block_of_node(n, 0);
  for (size_t u = 0; u < n; ++u) {
    size_t b = std::min(u / elements_per_block, num_blocks - 1);
    current_block_of_node[u] = b;
  }

  const double setup_end = ShuffleSeconds();
  // Fixed node order: materialize final order only once.
  for (size_t iter = 0; iter < params.max_iterations; ++iter) {
    size_t moved = workspace.Round(node_order, current_block_of_node, iter + 1);

    double move_fraction = static_cast<double>(moved) / static_cast<double>(n);
    fprintf(stderr, "  BlockBNF iter %zu: moved=%zu (%.2f%%)\n", iter + 1, moved,
            move_fraction * 100.0);

    if (move_fraction < params.convergence_threshold) {
      fprintf(stderr, "  BlockBNF converged at iter %zu\n", iter + 1);
      break;
    }
    if (moved == 0)
      break;
  }

  auto last_order = workspace.TakeOrder();
  const double rounds_end = ShuffleSeconds();
  if (collect_diagnostics)
    ReportBlockAffinity(graph, current_block_of_node, node_order);
  fprintf(
      stderr,
      "  [shuffle-level-timing] layers=%d..%d csr_s=%.6f sort_s=%.6f setup_s=%.6f rounds_s=%.6f affinity_stats_s=%.6f total_s=%.6f\n",
      layer_begin, layer_end, csr_end - level_start, sort_end - csr_end, setup_end - sort_end,
      rounds_end - setup_end, ShuffleSeconds() - rounds_end, ShuffleSeconds() - level_start);

  return last_order;
}

template auto ShuffleBlockBNF<int, float>(AllianceIndexDisk<int, float> *, const BNFParams &,
                                          layer_t, layer_t, size_t, size_t, bool)
    -> std::vector<tableint>;
} // namespace alliance::shuffle_detail
