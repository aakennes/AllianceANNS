#include "index/index_disk.hpp"
#include "io/aio_stream.hpp"
#include "rabitq/utils/buffer.hpp"
#include <algorithm>

namespace alliance {

// Synchronous compatibility path with its own candidate-retention policy.
// It can read raw or multi-level edges; exact vectors still use raw storage.

// SearchCandidatesQuery -- RaBitQ-driven traversal, disk reads only for edges

template <typename att_t, typename vec_t>
template <typename filter_t>
auto AllianceIndexDisk<att_t, vec_t>::SearchCandidatesQuery(std::vector<dist_id_pair> &eps,
                                                            const vec_t *query_vec,
                                                            const filter_t &filter,
                                                            const attr_range<layer_t> &layer_rng,
                                                            size_t ef, size_t k)
    -> std::vector<dist_id_pair> {
  RequireRawGraph("synchronous search");
  constexpr bool check_filter =
      (std::is_same_v<filter_t, attr_range<att_t>> ||
       std::is_same_v<filter_t, attr_range<att_label_t<att_t>>> ||
       std::is_same_v<filter_t, attr_set<att_t>> || std::is_same_v<filter_t, VisitedBitset<int>> ||
       std::is_same_v<filter_t, VisitedBitset<label_t>>);
  if (eps.empty())
    return {};
  if (per_query_cache_ && ml_layout_loaded_)
    ClearThreadLocalMultiLevelCache();

  RaBitQEstimator estimator(query_vec, this);

  VisitedList<tableint> visited(curvec_num_);
  visited.Clear();

  rabitqlib::buffer::SearchBuffer<float> candidates(ef);
  std::vector<dist_id_pair> result;
  for (auto &ep : eps) {
    candidates.insert(ep.id_, ep.dist_);
    PUSH_HEAP(result, ep.dist_, ep.id_);
    visited.Set(ep.id_);
  }
  auto res_max_dist = TOP_HEAP(result).dist_;

  std::vector<tableint> edge_buf(M_ + 1);

  while (candidates.has_next()) {
    tableint id = candidates.pop();
    metric_hops_++;

    size_t neighbor_cnt = 0;
    const layer_t step = ml_layout_loaded_ ? static_cast<layer_t>(ml_layer_stride_) : 1;
    const layer_t start =
        ml_layout_loaded_
            ? layer_rng.u_ -
                  static_cast<layer_t>((layer_rng.u_ - ml_layer_grid_base_) % ml_layer_stride_)
            : layer_rng.u_;
    for (layer_t layer = start; layer >= layer_rng.l_; layer -= step) {
      if (neighbor_cnt >= M_)
        break;

      if (ml_layout_loaded_) {
        ReadEdgesMultiLevel(id, layer, edge_buf.data());
      } else {
        ReadEdges(id, layer, edge_buf.data());
      }
      metric_disk_reads_++;
      // Clamp degree and skip out-of-range IDs before indexing edge, attribute,
      // label or visited storage. Corrupted records must not cause unchecked reads.
      tableint ll_sz = edge_buf[M_];
      if (ll_sz > static_cast<tableint>(M_))
        ll_sz = static_cast<tableint>(M_);

      bool visit_next_layer = false;
      for (tableint i = 0; i < ll_sz; ++i) {
        if (neighbor_cnt >= M_)
          break;
        tableint nn_id = edge_buf[i];
        if (nn_id >= static_cast<tableint>(curvec_num_))
          continue;

        if constexpr (check_filter) {
          if constexpr (std::is_same_v<filter_t, attr_range<att_label_t<att_t>>>) {
            if (!filter.Test({atts_[nn_id], labels_[nn_id]})) {
              visit_next_layer = true;
              continue;
            }
          } else {
            if (!filter.Test(atts_[nn_id])) {
              visit_next_layer = true;
              continue;
            }
          }
        }
        if (visited.Test(nn_id))
          continue;
        visited.Set(nn_id);

        auto [nn_dist, error_bound] =
            estimator.dist_estimate_with_errorbound(cids_[labels_[nn_id]], nn_id);
        metric_dist_comps_++;
        neighbor_cnt++;

        if (!candidates.is_full(nn_dist)) {
          candidates.insert(nn_id, nn_dist);
        }

        if (result.size() < ef || nn_dist < res_max_dist) {
          PUSH_HEAP(result, nn_dist + error_bound, nn_id);
          if (result.size() > ef) {
            POP_HEAP(result);
          }
          res_max_dist = TOP_HEAP(result).dist_;
        }
      }
      if (!visit_next_layer)
        break;
    }
  }
  return result;
}

// BruteForceSearch -- sequential scan for small filter ranges

template <typename att_t, typename vec_t>
template <typename filter_t>
auto AllianceIndexDisk<att_t, vec_t>::BruteForceSearch(vec_t *query_vec, size_t k,
                                                       const filter_t &filter)
    -> std::vector<std::pair<dist_t, label_t>> {
  if (IsMultiLevelOnly())
    throw std::runtime_error("raw-vector fallback is unavailable in ML-only mode");
  static_assert(std::is_same_v<filter_t, attr_range<att_t>>,
                "BruteForceSearch only supports attr_range<att_t> filters");

  auto [rank_lo, rank_hi] =
      order_table_->GetRankRange({filter.l_, 0}, {filter.u_, std::numeric_limits<label_t>::max()});
  size_t card = rank_hi - rank_lo;
  if (card == 0)
    return {};

  const size_t vec_bytes = vec_d_ * sizeof(vec_t);
  std::vector<vec_t> buf(card * vec_d_);

  const uint64_t offset = static_cast<uint64_t>(rank_lo) * static_cast<uint64_t>(vec_bytes);
  const uint64_t total_bytes = static_cast<uint64_t>(card) * static_cast<uint64_t>(vec_bytes);
  // Attribute-rank order makes the filter range one contiguous vector extent.
  // The stream reader handles aligned O_DIRECT chunks and rejects unexpected EOF.
  if (vec_file_path_.empty()) {
    throw std::runtime_error("BruteForceSearch: no vec file (call OpenVecFile first)");
  }
  AioSeqStreamReader vr(vec_file_path_, offset, offset + total_bytes);
  vr.Read(buf.data(), total_bytes);

  // top-k via max-heap
  using kv = std::pair<dist_t, label_t>;
  std::priority_queue<kv> topk;
  for (size_t i = 0; i < card; ++i) {
    tableint id = sorted_ids_[rank_lo + i];
    dist_t d = fstdistfunc_(query_vec, buf.data() + i * vec_d_, dist_func_param_);
    metric_dist_comps_++;
    if (topk.size() < k) {
      topk.emplace(d, labels_[id]);
    } else if (d < topk.top().first) {
      topk.pop();
      topk.emplace(d, labels_[id]);
    }
  }

  std::vector<kv> result;
  result.reserve(topk.size());
  while (!topk.empty()) {
    result.push_back(topk.top());
    topk.pop();
  }
  std::sort(result.begin(), result.end());
  return result;
}

// searchKNN -- DiskANN-style: RaBitQ traversal + fp32 rerank from disk

template <typename att_t, typename vec_t>
template <typename filter_t>
auto AllianceIndexDisk<att_t, vec_t>::searchKNN(vec_t *query_vec, size_t efs, size_t k,
                                                const filter_t &filter)
    -> std::vector<std::pair<dist_t, label_t>> {
  RequireRawGraph("searchKNN");
  if (!rabitq_initialized_) {
    throw std::runtime_error("searchKNN requires InitRaBitQ to be called first");
  }

  // Brute-force path for small filter ranges (below layer 0 window)
  if constexpr (std::is_same_v<filter_t, attr_range<att_t>>) {
    if (vec_file_available_) {
      auto [rank_lo, rank_hi] = order_table_->GetRankRange(
          {filter.l_, 0}, {filter.u_, std::numeric_limits<label_t>::max()});
      size_t filter_card = rank_hi - rank_lo;
      if (filter_card > 0 && filter_card < window_size_[0]) {
        return BruteForceSearch(query_vec, k, filter);
      }
    }
  }

  attr_range<layer_t> layer_rng;
  std::vector<dist_id_pair> ep_dist_id_pairs;

  RaBitQEstimator estimator(query_vec, this);

  if constexpr (std::is_same_v<filter_t, attr_range<att_t>>) {
    std::vector<tableint> eps;
    layer_rng = DecideLayerRange(filter, eps);
    for (auto ep_id : eps) {
      auto d = estimator.dist_estimate(cids_[labels_[ep_id]], ep_id);
      metric_dist_comps_++;
      ep_dist_id_pairs.emplace_back(d, ep_id);
    }
  } else {
    auto ep_id = static_cast<tableint>(rand() % curvec_num_);
    auto d = estimator.dist_estimate(cids_[labels_[ep_id]], ep_id);
    metric_dist_comps_++;
    ep_dist_id_pairs.emplace_back(d, ep_id);
    layer_rng = {static_cast<layer_t>(cur_max_layer_), static_cast<layer_t>(cur_max_layer_)};
  }

  std::vector<dist_id_pair> result;
  if constexpr (std::is_same_v<filter_t, attr_range<att_t>>) {
    attr_range<att_label_t<att_t>> dedup_filter{{filter.l_, 0},
                                                {filter.u_, std::numeric_limits<label_t>::max()}};
    result = SearchCandidatesQuery(ep_dist_id_pairs, query_vec, dedup_filter, layer_rng, efs, k);
  } else {
    result = SearchCandidatesQuery(ep_dist_id_pairs, query_vec, filter, layer_rng, efs, k);
  }

  // Rerank with fp32 vectors from disk
  size_t rerank_num = result.size();
  std::vector<vec_t> vec_buf(vec_d_);
  std::vector<std::pair<dist_t, label_t>> rerank_list(rerank_num);
  for (size_t i = 0; i < rerank_num; ++i) {
    tableint id = result[i].id_;
    label_t label = labels_[id];
    ReadVec(id, vec_buf.data());
    metric_disk_reads_++;
    rerank_list[i].first = fstdistfunc_(query_vec, vec_buf.data(), dist_func_param_);
    rerank_list[i].second = label;
  }
  std::sort(rerank_list.begin(), rerank_list.end());

  size_t final_k = std::min(k, rerank_num);
  rerank_list.resize(final_k);
  return rerank_list;
}

#define INSTANTIATE_DISK_QUERY(ATT_T, VEC_T, FILTER_T)                                             \
  template auto AllianceIndexDisk<ATT_T, VEC_T>::SearchCandidatesQuery<FILTER_T>(                  \
      std::vector<dist_id_pair> &, const VEC_T *, const FILTER_T &, const attr_range<layer_t> &,   \
      size_t, size_t) -> std::vector<dist_id_pair>;                                                \
  template auto AllianceIndexDisk<ATT_T, VEC_T>::searchKNN<FILTER_T>(                              \
      VEC_T *, size_t, size_t, const FILTER_T &) -> std::vector<std::pair<dist_t, label_t>>;

INSTANTIATE_DISK_QUERY(int, float, attr_range<int>)
INSTANTIATE_DISK_QUERY(int, float, attr_range<att_label_t<int>>)

#undef INSTANTIATE_DISK_QUERY

#define INSTANTIATE_DISK_BRUTE(ATT_T, VEC_T, FILTER_T)                                             \
  template auto AllianceIndexDisk<ATT_T, VEC_T>::BruteForceSearch<FILTER_T>(                       \
      VEC_T *, size_t, const FILTER_T &) -> std::vector<std::pair<dist_t, label_t>>;

INSTANTIATE_DISK_BRUTE(int, float, attr_range<int>)

#undef INSTANTIATE_DISK_BRUTE

} // namespace alliance
