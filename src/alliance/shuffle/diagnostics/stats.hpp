#pragma once
#include "shuffle/adjacency/weighted_graph.hpp"
namespace alliance::shuffle_detail {
void ReportBlockAffinity(const FlatWeightedGraph &, const std::vector<size_t> &blocks,
                         const std::vector<size_t> &processing_order);
} // namespace alliance::shuffle_detail
