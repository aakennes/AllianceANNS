#include "index/index_disk.hpp"
#include "io/disk_resource_guard.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>

namespace alliance {

// Async traversal of the raw single-record layout.
// RerankPipeline also serves multi-level layouts without a vector region.

// SearchCandidatesQueryPipeline -- async IO pipelined graph traversal
// Ported from StorageAirRFANNS SearchCandidatesWithShuffle (wow/ssd/pipeline.hh)

template <typename att_t, typename vec_t>
template <typename filter_t>
auto AllianceIndexDisk<att_t, vec_t>::SearchCandidatesQueryPipeline(
    std::vector<dist_id_pair> &eps, const vec_t *query_vec, const filter_t &filter,
    const attr_range<layer_t> &layer_rng, size_t ef, size_t k, DiskSearchContext *ctx,
    std::vector<vec_t> &vec_cache_buf, std::unordered_map<tableint, size_t> &vec_cache_idx)
    -> std::vector<dist_id_pair> {
  RequireRawGraph("raw graph pipeline");
  if (!ctx || !ctx->io_handler)
    throw std::runtime_error("raw graph pipeline requires a raw IO context");
  constexpr bool check_filter =
      (std::is_same_v<filter_t, attr_range<att_t>> ||
       std::is_same_v<filter_t, attr_range<att_label_t<att_t>>> ||
       std::is_same_v<filter_t, attr_set<att_t>> || std::is_same_v<filter_t, VisitedBitset<int>> ||
       std::is_same_v<filter_t, VisitedBitset<label_t>>);
  if (eps.empty())
    return {};

  RaBitQEstimator estimator(query_vec, this);

  VisitedList<tableint> visited(curvec_num_);
  visited.Clear();

  const size_t beam_width = ctx->edge_beam_width;
  const size_t bytes_per_read = node_disk_bytes_;

  struct Neighbor {
    tableint id{static_cast<tableint>(-1)};
    dist_t dist{std::numeric_limits<dist_t>::max()};
    bool in_flight{false};
    bool expanded{false};
    bool evicted{false};
  };

  std::vector<Neighbor> storage;
  storage.reserve(ef * (M_ + 2));
  std::vector<size_t> order;
  order.reserve(ef + 1);

  auto insert_candidate = [&](tableint id, dist_t dist) -> size_t {
    if (order.size() >= ef) {
      if (!(dist < storage[order.back()].dist))
        return SIZE_MAX;
    }
    size_t storage_idx = storage.size();
    storage.push_back(Neighbor{id, dist, false, false, false});
    size_t lo = 0, hi = order.size();
    while (lo < hi) {
      size_t mid = (lo + hi) / 2;
      if (storage[order[mid]].dist <= dist)
        lo = mid + 1;
      else
        hi = mid;
    }
    order.insert(order.begin() + lo, storage_idx);
    if (order.size() > ef) {
      storage[order.back()].evicted = true;
      order.pop_back();
    }
    if (storage[storage_idx].evicted)
      return SIZE_MAX;
    return storage_idx;
  };

  std::vector<dist_id_pair> result;
  for (auto &ep : eps) {
    insert_candidate(ep.id_, ep.dist_);
    PUSH_HEAP(result, ep.dist_, ep.id_);
    visited.Set(ep.id_);
  }
  auto res_max_dist = TOP_HEAP(result).dist_;

  struct PipelineSlot {
    size_t storage_idx{SIZE_MAX};
    IORequest req;
    char *read_buf{nullptr};
    bool in_flight{false};
  };

  std::vector<PipelineSlot> slots(beam_width);
  detail::ScopeExit slot_buffers([&]() noexcept {
    for (auto &slot : slots)
      free(slot.read_buf);
  });
  IOCompletionGuard io_completion(*ctx->io_handler);
  for (size_t i = 0; i < beam_width; ++i) {
    void *buf = nullptr;
    if (posix_memalign(&buf, kDiskPageSize, bytes_per_read) != 0) {
      throw std::runtime_error("posix_memalign failed for pipeline slot");
    }
    slots[i].read_buf = reinterpret_cast<char *>(buf);
  }

  auto cmp = [&](size_t a, size_t b) { return storage[a].dist > storage[b].dist; };
  std::priority_queue<size_t, std::vector<size_t>, decltype(cmp)> io_pending(cmp);
  std::vector<std::pair<size_t, size_t>> ready_queue;
  ready_queue.reserve(beam_width * 2);
  size_t in_flight = 0;

  for (size_t i = 0; i < order.size(); ++i) {
    io_pending.push(order[i]);
  }

  auto fill_pipeline = [&]() {
    while (!io_pending.empty() && in_flight < beam_width) {
      size_t si = io_pending.top();
      io_pending.pop();
      if (storage[si].evicted || storage[si].expanded || storage[si].in_flight)
        continue;

      size_t free_slot = beam_width;
      for (size_t i = 0; i < beam_width; ++i) {
        if (!slots[i].in_flight) {
          free_slot = i;
          break;
        }
      }
      if (free_slot == beam_width) {
        io_pending.push(si);
        break;
      }

      storage[si].in_flight = true;
      auto &slot = slots[free_slot];
      slot.storage_idx = si;
      slot.req.offset = graph_file_->node_offset(DiskSlot(storage[si].id));
      slot.req.len = bytes_per_read;
      slot.req.buf = slot.read_buf;
      slot.req.finished = false;
      slot.in_flight = true;
      ctx->io_handler->SubmitRead(slot.req);
      ++in_flight;
      metric_disk_reads_++;
    }
  };

  auto poll_finished = [&]() {
    ctx->io_handler->PollAll();
    for (size_t i = 0; i < beam_width; ++i) {
      auto &slot = slots[i];
      if (!slot.in_flight || !slot.req.finished)
        continue;
      if (slot.req.res != static_cast<long>(bytes_per_read)) {
        // IOCompletionGuard drains the remaining slots before any buffer is
        // released. A completed request can still be a failed or short read.
        throw std::runtime_error("short or failed read in raw graph traversal");
      }

      const size_t si = slot.storage_idx;
      const tableint node_id = storage[si].id;

      size_t cache_pos = vec_cache_buf.size() / vec_d_;
      vec_cache_buf.resize(vec_cache_buf.size() + vec_d_);
      std::memcpy(vec_cache_buf.data() + cache_pos * vec_d_, slot.read_buf + VecOffsetInRecord(),
                  VecBytes());
      vec_cache_idx[node_id] = cache_pos;

      ready_queue.push_back({si, i});
      slot.in_flight = false;
      slot.req.finished = false;
      --in_flight;
    }
  };

  auto expand_node = [&](size_t storage_idx, size_t slot_idx) {
    if (storage[storage_idx].evicted || storage[storage_idx].expanded)
      return;
    storage[storage_idx].expanded = true;
    storage[storage_idx].in_flight = false;

    const char *node_buf = slots[slot_idx].read_buf;
    metric_hops_++;
    size_t neighbor_cnt = 0;

    std::vector<tableint> edge_buf(M_ + 1);
    for (layer_t layer = layer_rng.u_; layer >= layer_rng.l_; --layer) {
      if (neighbor_cnt >= M_)
        break;

      std::memcpy(edge_buf.data(), node_buf + EdgeOffsetInRecord(layer), EdgeIdBytes());
      tableint ll_sz = edge_buf[M_];

      bool visit_next_layer = false;
      for (tableint i = 0; i < ll_sz; ++i) {
        if (neighbor_cnt >= M_)
          break;
        tableint nn_id = edge_buf[i];

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

        const auto nn_dist = estimator.dist_estimate(cids_[labels_[nn_id]], nn_id);
        metric_dist_comps_++;
        neighbor_cnt++;

        size_t new_idx = insert_candidate(nn_id, nn_dist);
        if (new_idx != SIZE_MAX) {
          io_pending.push(new_idx);
        }

        if (result.size() < ef || nn_dist < res_max_dist) {
          PUSH_HEAP(result, nn_dist, nn_id);
          if (result.size() > ef) {
            POP_HEAP(result);
          }
          res_max_dist = TOP_HEAP(result).dist_;
        }
      }
      if (!visit_next_layer)
        break;
    }
  };

  fill_pipeline();

  while (!ready_queue.empty() || !io_pending.empty() || in_flight > 0) {
    poll_finished();

    if (in_flight > 0 && ready_queue.empty()) {
      ctx->io_handler->PollWait();
      poll_finished();
    }

    while (!ready_queue.empty()) {
      auto [si, slot_idx] = ready_queue.back();
      ready_queue.pop_back();
      expand_node(si, slot_idx);
    }

    fill_pipeline();
  }

  return result;
}

// RerankPipeline -- pipeline rerank using cached vectors from traversal

template <typename att_t, typename vec_t>
void AllianceIndexDisk<att_t, vec_t>::RerankPipeline(
    const vec_t *query_vec, const std::vector<dist_id_pair> &candidates,
    const std::vector<vec_t> &vec_cache_buf,
    const std::unordered_map<tableint, size_t> &vec_cache_idx, DiskSearchContext *ctx,
    std::vector<std::pair<dist_t, label_t>> &out_results) {
  RequireRawGraph("raw vector rerank");
  if (!ctx || !ctx->io_handler)
    throw std::runtime_error("raw vector rerank requires a raw IO context");
  out_results.resize(candidates.size());

  std::vector<size_t> uncached_indices;
  for (size_t i = 0; i < candidates.size(); ++i) {
    tableint id = candidates[i].id_;
    auto it = vec_cache_idx.find(id);
    if (it != vec_cache_idx.end()) {
      const vec_t *cached_vec = vec_cache_buf.data() + it->second * vec_d_;
      out_results[i].first = fstdistfunc_(query_vec, cached_vec, dist_func_param_);
      out_results[i].second = labels_[id];
    } else {
      uncached_indices.push_back(i);
    }
  }

  if (uncached_indices.empty())
    return;

  const size_t beam = ctx->vec_beam_width;
  const size_t bytes_per_read = node_disk_bytes_;

  struct RerankSlot {
    size_t result_idx{SIZE_MAX};
    IORequest req;
    char *read_buf{nullptr};
    bool in_flight{false};
  };

  std::vector<RerankSlot> slots(beam);
  detail::ScopeExit slot_buffers([&]() noexcept {
    for (auto &slot : slots)
      free(slot.read_buf);
  });
  IOCompletionGuard io_completion(*ctx->io_handler);
  for (size_t i = 0; i < beam; ++i) {
    void *buf = nullptr;
    if (posix_memalign(&buf, kDiskPageSize, bytes_per_read) != 0) {
      throw std::runtime_error("posix_memalign failed for rerank slot");
    }
    slots[i].read_buf = reinterpret_cast<char *>(buf);
  }

  size_t next_idx = 0;
  size_t rerank_in_flight = 0;

  while (next_idx < uncached_indices.size() || rerank_in_flight > 0) {
    std::vector<IORequest *> batch;
    while (next_idx < uncached_indices.size() && rerank_in_flight < beam) {
      size_t free_slot = beam;
      for (size_t i = 0; i < beam; ++i) {
        if (!slots[i].in_flight) {
          free_slot = i;
          break;
        }
      }
      if (free_slot == beam)
        break;

      size_t ri = uncached_indices[next_idx++];
      tableint id = candidates[ri].id_;
      auto &slot = slots[free_slot];
      slot.result_idx = ri;
      slot.req.offset = graph_file_->node_offset(DiskSlot(id));
      slot.req.len = bytes_per_read;
      slot.req.buf = slot.read_buf;
      slot.req.finished = false;
      slot.in_flight = true;
      batch.push_back(&slot.req);
      ++rerank_in_flight;
      metric_disk_reads_++;
    }

    if (!batch.empty()) {
      ctx->io_handler->SubmitVecReadBatch(batch);
    }

    if (rerank_in_flight == 0)
      break;

    ctx->io_handler->PollVecAll();
    for (auto &slot : slots) {
      if (!slot.in_flight || !slot.req.finished)
        continue;
      if (slot.req.res != static_cast<long>(bytes_per_read)) {
        throw std::runtime_error("short or failed read in raw vector rerank");
      }
      size_t ri = slot.result_idx;
      tableint id = candidates[ri].id_;
      const vec_t *vec_ptr = reinterpret_cast<const vec_t *>(slot.read_buf + VecOffsetInRecord());
      out_results[ri].first = fstdistfunc_(query_vec, vec_ptr, dist_func_param_);
      out_results[ri].second = labels_[id];
      slot.in_flight = false;
      --rerank_in_flight;
    }
  }
}

// searchKNNPipeline -- top-level pipeline search entry point

template <typename att_t, typename vec_t>
template <typename filter_t>
auto AllianceIndexDisk<att_t, vec_t>::searchKNNPipeline(vec_t *query_vec, size_t efs, size_t k,
                                                        const filter_t &filter,
                                                        DiskSearchContext *ctx)
    -> std::vector<std::pair<dist_t, label_t>> {
  RequireRawGraph("searchKNNPipeline");
  if (!ctx || !ctx->io_handler)
    throw std::runtime_error("searchKNNPipeline requires a raw IO context");
  if (!rabitq_initialized_) {
    throw std::runtime_error("searchKNNPipeline requires InitRaBitQ to be called first");
  }

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

  std::vector<vec_t> vec_cache_buf;
  vec_cache_buf.reserve(efs * vec_d_);
  std::unordered_map<tableint, size_t> vec_cache_idx;
  vec_cache_idx.reserve(efs);

  std::vector<dist_id_pair> result;
  if constexpr (std::is_same_v<filter_t, attr_range<att_t>>) {
    attr_range<att_label_t<att_t>> dedup_filter{{filter.l_, 0},
                                                {filter.u_, std::numeric_limits<label_t>::max()}};
    result = SearchCandidatesQueryPipeline(ep_dist_id_pairs, query_vec, dedup_filter, layer_rng,
                                           efs, k, ctx, vec_cache_buf, vec_cache_idx);
  } else {
    result = SearchCandidatesQueryPipeline(ep_dist_id_pairs, query_vec, filter, layer_rng, efs, k,
                                           ctx, vec_cache_buf, vec_cache_idx);
  }

  std::vector<std::pair<dist_t, label_t>> rerank_list;
  RerankPipeline(query_vec, result, vec_cache_buf, vec_cache_idx, ctx, rerank_list);

  std::sort(rerank_list.begin(), rerank_list.end());
  size_t final_k = std::min(k, rerank_list.size());
  rerank_list.resize(final_k);
  return rerank_list;
}

template class AllianceIndexDisk<int, float>;

#define INSTANTIATE_DISK_PIPELINE(ATT_T, VEC_T, FILTER_T)                                          \
  template auto AllianceIndexDisk<ATT_T, VEC_T>::searchKNNPipeline<FILTER_T>(                      \
      VEC_T *, size_t, size_t, const FILTER_T &, DiskSearchContext *)                              \
      -> std::vector<std::pair<dist_t, label_t>>;                                                  \
  template auto AllianceIndexDisk<ATT_T, VEC_T>::SearchCandidatesQueryPipeline<FILTER_T>(          \
      std::vector<dist_id_pair> &, const VEC_T *, const FILTER_T &, const attr_range<layer_t> &,   \
      size_t, size_t, DiskSearchContext *, std::vector<VEC_T> &,                                   \
      std::unordered_map<tableint, size_t> &) -> std::vector<dist_id_pair>;

INSTANTIATE_DISK_PIPELINE(int, float, attr_range<int>)
INSTANTIATE_DISK_PIPELINE(int, float, attr_range<att_label_t<int>>)

#undef INSTANTIATE_DISK_PIPELINE

} // namespace alliance
