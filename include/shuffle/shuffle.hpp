#pragma once
#include "shuffle/config.hpp"
#include "shuffle/level_geometry.hpp"
#include <vector>

namespace alliance {
template <typename att_t, typename vec_t> class AllianceIndexDisk;

// Ranges are ordered top-to-bottom, with disjoint owners and stored overlap.
auto GenerateLevelRanges(layer_t max_layer, const MultiLevelConfig &config)
    -> std::vector<LevelRange>;

struct ShuffleStats {
  size_t num_edges{0};
  double avg_edge_span{0};
  double p90_edge_span{0};
  double max_edge_span{0};
};

// Explicit diagnostic: reads the selected graph edges; not required to reorder.
template <typename att_t, typename vec_t>
auto ComputeEdgeLocalityStats(AllianceIndexDisk<att_t, vec_t> *idx,
                              const std::vector<tableint> &order, layer_t first, layer_t last,
                              size_t stride = 1) -> ShuffleStats;

struct LevelShuffleResult {
  LevelRange range;
  // Physical position -> original internal ID. Edges retain these IDs.
  std::vector<tableint> order;
  ShuffleStats stats;
  bool stats_collected{false};
};

template <typename att_t, typename vec_t>
auto ComputeMultiLevelShuffleOrders(AllianceIndexDisk<att_t, vec_t> *idx,
                                    const MultiLevelConfig &config, ShuffleMethod method)
    -> std::vector<LevelShuffleResult>;
} // namespace alliance
