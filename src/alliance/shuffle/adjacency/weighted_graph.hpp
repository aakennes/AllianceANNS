#pragma once
#include "shuffle/shuffle.hpp"
#include "shuffle/adjacency/weighted_edges.hpp"
#include "shuffle/adjacency/external_graph.hpp"
#include <cstdint>

namespace alliance::shuffle_detail {

// One immutable row view, shared by RAM and external scoring. It owns no buffer.
struct WeightedRow {
  const WeightedEdges *memory{};
  size_t begin{}, size{};
  EncodedRow encoded;
  tableint Neighbor(size_t i) const {
    return memory ? memory->Neighbor(begin + i) : encoded.Neighbor(i);
  }
  uint8_t Weight(size_t i) const {
    return memory ? memory->Weight(begin + i) : encoded.Weight(i);
  }
};

struct FlatWeightedGraph {
  WeightedEdges edges;
  std::shared_ptr<ExternalGraph> external;
  std::vector<size_t> offsets;    // offsets[u]..offsets[u+1]
  std::vector<size_t> in_weights; // IN-weight sum per node (reverse graph).
  size_t num_nodes{0};
  WeightedRow Row(size_t node, size_t rank) const {
    if (external) {
      auto row = external->Row(rank);
      return {nullptr, 0, row.size, row};
    }
    return {&edges, offsets[node], offsets[node + 1] - offsets[node], {}};
  }
};

template <typename att_t, typename vec_t>
auto BuildWeightedGraph(AllianceIndexDisk<att_t, vec_t> *idx, layer_t layer_begin,
                        layer_t layer_end, size_t stride, int threads = 1,
                        const BNFParams &params = {}, size_t elements_per_block = 1)
    -> FlatWeightedGraph;

auto SortNodesByInWeight(const std::vector<size_t> &in_weights) -> std::vector<size_t>;

} // namespace alliance::shuffle_detail
