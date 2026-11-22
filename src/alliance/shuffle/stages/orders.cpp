#include "shuffle/shuffle.hpp"
#include "index/index_disk.hpp"
#include "shuffle/order/block_bnf.hpp"
#include <numeric>
#include <cstdio>
#include <stdexcept>

namespace alliance {
using namespace shuffle_detail;
template <typename att_t, typename vec_t>
auto ComputeMultiLevelShuffleOrders(AllianceIndexDisk<att_t, vec_t> *idx,
                                    const MultiLevelConfig &config, ShuffleMethod method)
    -> std::vector<LevelShuffleResult> {
  auto levels = GenerateLevelRanges(static_cast<layer_t>(idx->cur_max_layer_), config);
  std::vector<LevelShuffleResult> results;
  results.reserve(levels.size());

  for (size_t li = 0; li < levels.size(); ++li) {
    BNFParams params;
    if (li < config.per_level_params.size()) {
      params = config.per_level_params[li];
    }

    // Shuffle affinity uses only the k non-overlapping layers of this level.
    // Using all k+m layers adds too many w=1 noise edges that degrade BNF
    // convergence; the k-layer subset has a higher w>=2 signal ratio.
    const layer_t stride = static_cast<layer_t>(LayerStride(config.shuffle_o));
    const layer_t le = levels[li].layer_end;
    const layer_t lb_full = levels[li].layer_begin;
    const layer_t shuffle_lb = levels[li].owner_begin;
    const size_t full_level_layers = CountSelectedLayers(lb_full, le, static_cast<size_t>(stride));

    fprintf(stderr, "[level %zu] full=[%d..%d](%zu sel layers, stride=%d) shuffle=[%d..%d]\n", li,
            lb_full, le, full_level_layers, stride, shuffle_lb, le);

    std::vector<tableint> order;
    switch (method) {
    case ShuffleMethod::Identity:
      order.resize(idx->curvec_num_);
      std::iota(order.begin(), order.end(), tableint(0));
      break;
    case ShuffleMethod::BlockBNF:
      order = ShuffleBlockBNF(idx, params, shuffle_lb, le, full_level_layers, stride,
                              config.collect_diagnostics);
      break;
    default:
      throw std::invalid_argument("retired or invalid shuffle method");
    }

    ShuffleStats stats;
    if (config.collect_diagnostics)
      stats = ComputeEdgeLocalityStats(idx, order, levels[li].owner_begin, levels[li].owner_end,
                                       static_cast<size_t>(stride));
    results.push_back({levels[li], std::move(order), stats, config.collect_diagnostics});
  }
  return results;
}

template auto ComputeMultiLevelShuffleOrders<int, float>(AllianceIndexDisk<int, float> *,
                                                         const MultiLevelConfig &, ShuffleMethod)
    -> std::vector<LevelShuffleResult>;

} // namespace alliance
