#ifdef ALLIANCE_BATCH_EXPAND
#error "ALLIANCE_BATCH_EXPAND was retired; use the default storage/order traversal"
#endif

// Async multi-level graph traversal. Exact rerank begins after traversal drains.
// Query orchestration and post-traversal rerank live in the query/rerank
// modules; all three use the same worker-owned buffers and scratch.

// ALLIANCE_PIPELINE_PROFILE = cheap steady_clock per-phase timers.
// ALLIANCE_PIPELINE_DIAGNOSTICS additionally enables the expensive per-block-read
// co-block neighbour scan, so it implies (but is not implied by) PROFILE.
#if defined(ALLIANCE_PIPELINE_DIAGNOSTICS) && !defined(ALLIANCE_PIPELINE_PROFILE)
#define ALLIANCE_PIPELINE_PROFILE
#endif

#include "index/index_disk.hpp"
#include "search/rerank_candidate_pool.hpp"
#include "search/rank_entry_sampling.hpp"
#include "search/disk_search_cache.hpp"
#include "search/multilevel_traversal_state.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>

namespace alliance {

using detail::DecodeBlockIntoCache;
using detail::DiskSearchWorkspace;
using detail::GetGraphSearchWorkspace;

// SearchCandidatesQueryMultiLevelPipeline -- async IO pipelined traversal
// over the multi-level layout, with block-level IO dedup and cached-level
// expansion. Per-call candidate and IO state lives in multilevel_traversal_state.hpp.

template <typename att_t, typename vec_t>
template <typename filter_t, bool post_filter>
auto AllianceIndexDisk<att_t, vec_t>::SearchCandidatesQueryMultiLevelPipeline(
    std::vector<dist_id_pair> &eps, const vec_t *query_vec, const filter_t &filter,
    const attr_range<layer_t> &layer_rng, size_t ef, size_t k, DiskSearchContext *ctx,
    RaBitQEstimator *query_estimator) -> std::vector<dist_id_pair> {
  ScopedMultiLevelMetrics metric_scope(*this);
  auto &metrics = metric_scope.Get();
  constexpr bool check_filter =
      (std::is_same_v<filter_t, attr_range<att_t>> ||
       std::is_same_v<filter_t, attr_range<att_label_t<att_t>>> ||
       std::is_same_v<filter_t, attr_set<att_t>> || std::is_same_v<filter_t, VisitedBitset<int>> ||
       std::is_same_v<filter_t, VisitedBitset<label_t>>);
  if (eps.empty() || ef == 0)
    return {};
  if (!ctx->ml_io_handler) {
    throw std::runtime_error("SearchCandidatesQueryMultiLevelPipeline requires "
                             "DiskSearchContext to be constructed with a non-empty ml_graph_path");
  }

  // The public standalone candidate API can still own its estimator. Normal
  // KNN search passes the one already prepared for entry points and bounded
  // rerank, avoiding duplicate float-query rotation and centroid-distance preparation.
  std::optional<RaBitQEstimator> owned_estimator;
  if (!query_estimator)
    owned_estimator.emplace(query_vec, this);
  auto &estimator = query_estimator ? *query_estimator : *owned_estimator;

  DiskSearchWorkspace &tl = GetGraphSearchWorkspace(this, ctx);
  if (per_query_cache_) {
    tl.ClearCache();
    PublishMultiLevelCacheFootprint(tl.Cache());
  }
  auto &visited = tl.GetVisited(curvec_num_);

  // Keep traversal width independent of the exact rerank budget. The main
  // result heap retains min(ef,R), and reserve retains the best remaining
  // R-main_capacity rejected/evicted IDs. Reserve never feeds graph queues.
  const size_t rerank_budget = ctx->GraphRerankBudgetForSearch(k, ef, rerank_times_upper_bound_);
  RerankCandidatePool result_pool(ef, rerank_budget);
  auto offer_result = [&](tableint id, dist_t dist,
                          double precise = std::numeric_limits<double>::quiet_NaN()) {
    if constexpr (post_filter) {
      if (!result_pool.CouldImprove(id, dist) || !filter.Test(atts_[id]))
        return;
    }
    result_pool.OfferUnique(id, dist, precise);
  };

  const size_t beam_width = ctx->edge_beam_width;
  const size_t per_layer_id_bytes = (M_ + 1) * sizeof(tableint);
  const layer_t layer_stride = static_cast<layer_t>(ml_layer_stride_);
  const layer_t grid_base = ml_layer_grid_base_;

  detail::MultiLevelTraversalCandidates traversal(visited, ef, M_);

  // Supplemental entries use the same query estimator and attribute order
  // as graph search. They add no disk reads and do not change rerank policy.
  if (!HasNavigation() && ctx->entry_samples && ctx->entry_count) {
    if (sorted_ids_.size() != static_cast<size_t>(curvec_num_))
      throw std::logic_error("entry sampling requires the full attribute order");
    size_t lo = 0, hi = sorted_ids_.size();
    constexpr bool is_range_filter = std::is_same_v<filter_t, attr_range<att_t>> ||
                                     std::is_same_v<filter_t, attr_range<att_label_t<att_t>>>;
    const bool guided =
        ctx->entry_sampling == DiskSearchContext::EntrySampling::Guided && is_range_filter;
    if (ctx->entry_sampling != DiskSearchContext::EntrySampling::Global) {
      if constexpr (std::is_same_v<filter_t, attr_range<att_t>> ||
                    std::is_same_v<filter_t, attr_range<att_label_t<att_t>>>) {
        auto key = [&](tableint id) {
          if constexpr (std::is_same_v<filter_t, attr_range<att_label_t<att_t>>>)
            return att_label_t<att_t>{atts_[id], labels_[id]};
          else
            return atts_[id];
        };
        auto first =
            std::lower_bound(sorted_ids_.begin(), sorted_ids_.end(), filter.l_,
                             [&](tableint id, const auto &value) { return key(id) < value; });
        auto last =
            std::upper_bound(first, sorted_ids_.end(), filter.u_,
                             [&](const auto &value, tableint id) { return value < key(id); });
        lo = first - sorted_ids_.begin();
        hi = last - sorted_ids_.begin();
      }
      // Non-range predicates retain global sampling followed by filtering.
    }
    auto accept = [&](tableint id) {
      // Supplemental seeds stay in range even in top-layer postfilter mode.
      if constexpr (check_filter) {
        if constexpr (std::is_same_v<filter_t, attr_range<att_label_t<att_t>>>)
          return filter.Test({atts_[id], labels_[id]});
        else
          return filter.Test(atts_[id]);
      } else
        return true;
    };
    metrics.metric_ml_entry_samples_ += std::min(ctx->entry_samples, hi - lo);
    auto score = [&](tableint id) {
      ++metrics.metric_dist_comps_;
      ++metrics.metric_ml_entry_estimates_;
      return estimator.dist_estimate(SearchCID(id), id);
    };
    // Production CLI prepares this once before measured batches. Library
    // callers may prepare explicitly; concurrent first use is also safe.
    std::vector<tableint> guided_ids;
    if (guided) {
      PrepareGuidedEntries();
      auto ranks = guided_entries_.Select(lo, hi, estimator.dist_to_centroids_, ctx->entry_samples);
      guided_ids.reserve(ranks.size());
      for (auto rank : ranks)
        guided_ids.push_back(sorted_ids_[rank]);
    }
    auto entries = guided ? SelectRankEntries(guided_ids, 0, guided_ids.size(), guided_ids.size(),
                                              ctx->entry_count, accept, score)
                          : SelectRankEntries(sorted_ids_, lo, hi, ctx->entry_samples,
                                              ctx->entry_count, accept, score);
    for (const auto &[dist, id] : entries) {
      if (!traversal.Claim(id))
        continue;
      if (traversal.Insert(id, dist) != SIZE_MAX)
        ++metrics.metric_ml_entry_admitted_;
      offer_result(id, dist);
    }
  }

  for (auto &ep : eps) {
    if (!traversal.Claim(ep.id_))
      continue;
    traversal.Insert(ep.id_, ep.dist_);
    offer_result(ep.id_, ep.dist_);
  }

  // Reuse one buffer pool for traversal and the later final-rerank phase.
  const size_t total_bufs_needed = beam_width + ctx->RerankWidth();
  // Both request kinds borrow this pool. Size every slot for the larger page
  // before publishing pointers; growing it during traversal invalidates them.
  tl.EnsurePipelineBuffers(total_bufs_needed, std::max(ml_block_size_, ml_vec_block_size_));

  auto compare_distance = [&](size_t a, size_t b) { return traversal[a].dist > traversal[b].dist; };
  detail::MultiLevelTraversalReads reads(beam_width, compare_distance);
  for (size_t i = 0; i < beam_width; ++i) {
    reads.slots[i].read_buf = tl.PipelineBuffer(i);
  }

  auto make_block_key = [](size_t level_idx, int32_t block_id) -> uint64_t {
    return (static_cast<uint64_t>(level_idx) << 40) |
           static_cast<uint64_t>(static_cast<uint32_t>(block_id));
  };

  // Unified cache: a lookup only counts as covering when the cached record's
  // layer range includes the target layer (the record may come from any level).
  auto lookup_covering = [&](tableint id, layer_t layer) -> DynamicBlockCache::Entry {
    auto e = StaticCacheLookup(id, layer);
    if (e.found)
      return e;
    e = tl.Cache().Lookup(id);
    if (e.found && layer >= e.layer_begin && layer <= e.layer_end)
      return e;
    e.found = false;
    return e;
  };

  // Route a candidate to the ready queue on a cache hit or the pending queue
  // on a miss. Every candidate starts at the query landing layer.
  auto route_candidate = [&](size_t si, size_t source_layer = kMaxMetricLayers) {
    auto &nb = traversal[si];
    const layer_t target_layer = layer_rng.u_;
    const size_t tli = ml_layer_to_level_[static_cast<size_t>(target_layer)];
    auto e = StaticCacheLookup(nb.id, target_layer);
    if (e.found) {
      ++metrics.metric_ml_static_cache_hits_;
      if (tli < kMaxMetricLevels)
        ++metrics.metric_ml_static_hits_per_level_[tli];
      if (source_layer < kMaxMetricLayers)
        ++metrics.metric_ml_route_hits_per_layer_[source_layer];
      reads.ready.push(si);
    } else {
      e = tl.Cache().Lookup(nb.id);
      if (e.found && target_layer >= e.layer_begin && target_layer <= e.layer_end) {
        ++metrics.metric_ml_cache_hits_;
        if (tli < kMaxMetricLevels)
          ++metrics.metric_ml_cache_hits_per_level_[tli];
        if (source_layer < kMaxMetricLayers)
          ++metrics.metric_ml_route_hits_per_layer_[source_layer];
        reads.ready.push(si);
      } else {
        ++metrics.metric_ml_cache_misses_;
        if (tli < kMaxMetricLevels)
          ++metrics.metric_ml_cache_misses_per_level_[tli];
        if (source_layer < kMaxMetricLayers)
          ++metrics.metric_ml_route_misses_per_layer_[source_layer];
        reads.pending.push(si);
      }
    }
  };

  for (const size_t candidate : traversal.Ordered()) {
    route_candidate(candidate);
  }

  // Submission is batched once per fill; adaptive width starts narrow to
  // avoid speculative IO for candidates evicted before expansion.
  auto submit_pending_reads = [&]() {
    reads.submit_batch.clear();
    while (!reads.pending.empty() && reads.in_flight < reads.pipeline_width) {
      size_t si = reads.pending.top();
      reads.pending.pop();
      if (traversal[si].evicted || traversal[si].expanded || traversal[si].in_flight)
        continue;

      const layer_t target_layer = layer_rng.u_;

      // Cache coverage is checked when routing, not again on this pending pop.
      // Only in-flight reads are deduplicated below. A block completed since
      // routing may therefore be read again; ready entries recheck coverage.
      const size_t level_idx = ml_layer_to_level_[static_cast<size_t>(target_layer)];
      const int32_t block_id = ml_disk_blocks_per_level_[level_idx][traversal[si].id];
      const uint64_t block_key = make_block_key(level_idx, block_id);

      auto dedup_it = reads.block_slots.find(block_key);
      if (dedup_it != reads.block_slots.end()) {
        traversal[si].in_flight = true;
        reads.slots[dedup_it->second].waiters.push_back(si);
        ++metrics.metric_ml_io_dedup_saved_;
        continue;
      }

      size_t free_slot = beam_width;
      for (size_t i = 0; i < beam_width; ++i) {
        if (!reads.slots[i].in_flight) {
          free_slot = i;
          break;
        }
      }
      if (free_slot == beam_width) {
        reads.pending.push(si);
        break;
      }

      traversal[si].in_flight = true;
      auto &slot = reads.slots[free_slot];
      slot.storage_idx = si;
      slot.block_key = block_key;
      slot.req.offset = static_cast<uint64_t>(ml_header_bytes_) +
                        static_cast<uint64_t>(block_id) * static_cast<uint64_t>(ml_block_size_);
      slot.req.len = ml_block_size_;
      slot.req.buf = slot.read_buf;
      slot.req.finished = false;
      slot.in_flight = true;
      reads.submit_batch.push_back(&slot.req);
      ++reads.in_flight;
      ++metrics.metric_ml_io_submits_;
      reads.block_slots[block_key] = free_slot;
    }
    if (!reads.submit_batch.empty()) {
      ctx->ml_io_handler->SubmitReadBatch(reads.submit_batch);
    }
  };

#ifdef ALLIANCE_PIPELINE_DIAGNOSTICS
  // Optional shuffle diagnostic; it neither admits candidates nor schedules IO.
  auto record_block_utilization = [&](const char *block_buf, tableint req_id, size_t record_size,
                                      layer_t level_lb, layer_t level_le) {
    const size_t rs_d = record_size;
    const size_t hdr = sizeof(tableint) + sizeof(label_t) + sizeof(att_t);
    const size_t plb = per_layer_id_bytes;
    const size_t ls = static_cast<size_t>(ml_layer_stride_);
    const size_t nl = static_cast<size_t>((level_le - level_lb) / static_cast<layer_t>(ls)) + 1;

    const char *req_rec = nullptr;
    tableint coblock[32];
    size_t coblock_cnt = 0;
    for (size_t roff = 0; roff + rs_d <= ml_block_size_; roff += rs_d) {
      tableint rid;
      std::memcpy(&rid, block_buf + roff, sizeof(tableint));
      if (rid == static_cast<tableint>(-1))
        break;
      if (rid == req_id) {
        req_rec = block_buf + roff;
      } else if (coblock_cnt < 32) {
        coblock[coblock_cnt++] = rid;
      }
    }

    if (req_rec && coblock_cnt > 0) {
      size_t neighbor_count = 0;
      for (size_t ci = 0; ci < coblock_cnt; ++ci) {
        bool found = false;
        for (size_t li = 0; li < nl && !found; ++li) {
          const tableint *edges = reinterpret_cast<const tableint *>(req_rec + hdr + li * plb);
          tableint ll_sz = edges[M_];
          for (tableint ei = 0; ei < ll_sz && ei < M_; ++ei) {
            if (edges[ei] == coblock[ci]) {
              found = true;
              break;
            }
          }
        }
        if (found)
          ++neighbor_count;
      }
      metrics.metric_block_coblock_total_ += coblock_cnt;
      metrics.metric_block_coblock_neighbor_ += neighbor_count;
    }
  };
#endif

  // Co-block records use the same discovery gate as graph neighbors. Claim
  // before scoring, and preserve already queued/in-flight candidate state.
  auto admit_block_candidates = [&](const char *block_buf, size_t record_size) {
    const size_t rs = record_size;
    for (size_t roff = 0; roff + rs <= ml_block_size_; roff += rs) {
      tableint rec_id;
      std::memcpy(&rec_id, block_buf + roff, sizeof(tableint));
      if (rec_id == static_cast<tableint>(-1))
        break;
      if constexpr (check_filter && !post_filter) {
        if constexpr (std::is_same_v<filter_t, attr_range<att_label_t<att_t>>>) {
          if (!filter.Test({atts_[rec_id], labels_[rec_id]}))
            continue;
        } else {
          if (!filter.Test(atts_[rec_id]))
            continue;
        }
      }

      if (!traversal.Claim(rec_id))
        continue;
      const double precise = estimator.dist_estimate_precise(SearchCID(rec_id), rec_id);
      const float dist = static_cast<float>(precise);
      metrics.metric_dist_comps_++;

      offer_result(rec_id, dist, precise);

      if (traversal.Rejects(dist))
        continue;
      size_t new_idx = traversal.Insert(rec_id, dist);
      if (new_idx != SIZE_MAX) {
        // A co-block record may cover another level, or have been evicted
        // while the block was decoded. Use the regular coverage check.
        route_candidate(new_idx);
        ++metrics.metric_ml_block_inject_;
      }
    }
  };

  auto complete_reads = [&]() {
    if (reads.in_flight == 0)
      return;
    ctx->ml_io_handler->PollAll();
    for (size_t i = 0; i < beam_width; ++i) {
      auto &slot = reads.slots[i];
      if (!slot.in_flight || !slot.req.finished)
        continue;
      if (slot.req.res != static_cast<long>(ml_block_size_)) {
        // The guard drains all other slots before their requests or buffers
        // leave scope. Never decode a failed/partial block or publish its hits.
        throw std::runtime_error("short or failed read in multi-level traversal");
      }

      const uint64_t level_idx = slot.block_key >> 40;
      auto &cache = tl.Cache();
      const layer_t level_lb = ml_levels_[level_idx].layer_begin;
      const layer_t level_le = ml_levels_[level_idx].layer_end;
      const size_t record_size = ml_record_size_per_level_[level_idx];
      // Requester + dedup waiters paused on this block's level: their cache slots
      // must take this level's record; the rest of the block is prefetch.
      reads.must_update_ids.clear();
      reads.must_update_ids.push_back(traversal[slot.storage_idx].id);
      for (size_t waiter_idx : slot.waiters)
        reads.must_update_ids.push_back(traversal[waiter_idx].id);
      const StaticNodeLookup *sm = static_cache_loaded_ ? &static_node_lookup_ : nullptr;
      DecodeBlockIntoCache(cache, slot.read_buf, record_size, ml_block_size_, level_lb, level_le,
                           reads.must_update_ids.data(), reads.must_update_ids.size(), &visited,
                           sm);
      ++metrics.metric_ml_block_reads_;
      if (level_idx < kMaxMetricLevels)
        ++metrics.metric_ml_block_reads_per_level_[level_idx];

#ifdef ALLIANCE_PIPELINE_DIAGNOSTICS
      record_block_utilization(slot.read_buf, traversal[slot.storage_idx].id, record_size, level_lb,
                               level_le);
#endif

      admit_block_candidates(slot.read_buf, record_size);

      ++reads.completed_candidates;
      if (!traversal[slot.storage_idx].evicted)
        ++reads.retained_completions;
      reads.ready.push(slot.storage_idx);
      for (size_t waiter_idx : slot.waiters) {
        ++reads.completed_candidates;
        if (!traversal[waiter_idx].evicted)
          ++reads.retained_completions;
        reads.ready.push(waiter_idx);
      }

      reads.block_slots.erase(slot.block_key);
      slot.waiters.clear();
      slot.block_key = UINT64_MAX;
      slot.in_flight = false;
      slot.req.finished = false;
      --reads.in_flight;
    }
  };

  // start_layer is a per-query constant: layer_rng.u_ snapped to the stride grid.
  const layer_t start_layer =
      layer_rng.u_ - static_cast<layer_t>((layer_rng.u_ - grid_base) % layer_stride);

  auto expand_cached_candidate = [&](size_t storage_idx, const DynamicBlockCache::Entry &cached_e) {
    if (traversal[storage_idx].evicted || traversal[storage_idx].expanded)
      return;
    traversal[storage_idx].in_flight = false;

    metrics.metric_hops_++;

    const layer_t loop_lower = std::max(layer_rng.l_, cached_e.layer_begin);
    size_t neighbor_cnt = 0;

    for (layer_t layer = start_layer; layer >= loop_lower; layer -= layer_stride) {
      if constexpr (!post_filter)
        if (neighbor_cnt >= M_)
          break;
      metrics.metric_layer_visits_++;

      const size_t layer_off = sizeof(tableint) + sizeof(label_t) + sizeof(att_t) +
                               static_cast<size_t>(layer - cached_e.layer_begin) /
                                   static_cast<size_t>(layer_stride) * per_layer_id_bytes;
      const tableint *edges = reinterpret_cast<const tableint *>(cached_e.data + layer_off);
      tableint ll_sz = edges[M_];

      bool visit_next_layer = false;
      for (tableint i = 0; i < ll_sz; ++i) {
        if constexpr (!post_filter)
          if (neighbor_cnt >= M_)
            break;
        if constexpr (post_filter) {
          const tableint id = edges[std::min<tableint>(i + 4, ll_sz - 1)];
          __builtin_prefetch(atts_.data() + id, 0, 1);
          if (!search_cids_.empty())
            __builtin_prefetch(search_cids_.data() + id, 0, 1);
          __builtin_prefetch(RaBitQCode(id), 0, 1);
          if (packed_rabitq_.Empty()) {
            __builtin_prefetch(RaBitQCode(id) + quant_vec_bytes_ - 1, 0, 1);
            __builtin_prefetch(RaBitQFactors(id), 0, 1);
          } else if (packed_rabitq_.Stride() > 64) {
            for (size_t off = 64; off < quant_vec_bytes_ + 16; off += 64)
              __builtin_prefetch(RaBitQCode(id) + off, 0, 1);
          }
        }
        tableint nn_id = edges[i];

        if constexpr (check_filter && !post_filter) {
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
        if (!traversal.Claim(nn_id))
          continue;

        const double precise = estimator.dist_estimate_precise(SearchCID(nn_id), nn_id);
        const float nn_dist = static_cast<float>(precise);
        metrics.metric_dist_comps_++;
        neighbor_cnt++;

        // Reject before insertion/routing; result retention is independent.
        if (traversal.Rejects(nn_dist)) {
          offer_result(nn_id, nn_dist, precise);
          continue;
        }

        size_t new_idx = traversal.Insert(nn_id, nn_dist);
        if (new_idx != SIZE_MAX) {
          route_candidate(new_idx, static_cast<size_t>(layer));
        }

        offer_result(nn_id, nn_dist, precise);
      }
      if constexpr (!post_filter)
        if (!visit_next_layer)
          break;
    }

    if (traversal[storage_idx].evicted)
      return;

    // This candidate is finished after expanding the cached layer range.
    traversal[storage_idx].expanded = true;
  };

  // Edge request storage must survive exception draining.
  IOCompletionGuard io_completion(*ctx->ml_io_handler);
  submit_pending_reads();

  using ProfClock = std::chrono::steady_clock;
#ifdef ALLIANCE_PIPELINE_PROFILE
  auto prof_ns = [](ProfClock::time_point t0) -> size_t {
    return static_cast<size_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(ProfClock::now() - t0).count());
  };
  size_t prof_submit_ns = 0, prof_poll_ns = 0, prof_expand_ns = 0;
#endif
  size_t prof_loop_iters = 0, prof_inflight_sum = 0;
#ifdef ALLIANCE_PIPELINE_PROFILE
  const auto prof_loop_begin = ProfClock::now();
#endif

  while (!reads.ready.empty() || !reads.pending.empty() || reads.in_flight > 0) {
    ++prof_loop_iters;
#ifdef ALLIANCE_PIPELINE_PROFILE
    prof_inflight_sum += reads.in_flight;
#endif

#ifdef ALLIANCE_PIPELINE_PROFILE
    {
      auto t0 = ProfClock::now();
      complete_reads();
      prof_poll_ns += prof_ns(t0);
    }
#else
    complete_reads();
#endif

    // Adaptive width: widen pipeline when eviction waste is low.
    if (prof_loop_iters >= 5 && reads.completed_candidates > 0) {
      const double waste_ratio =
          static_cast<double>(reads.completed_candidates - reads.retained_completions) /
          static_cast<double>(reads.completed_candidates);
      if (waste_ratio <= 0.1) {
        reads.pipeline_width = std::min(beam_width, reads.pipeline_width + 1);
      }
    }

    while (!reads.ready.empty()) {
      size_t si = reads.ready.top();
      reads.ready.pop();
      auto ce = lookup_covering(traversal[si].id, start_layer);
      if (ce.found) {
#ifdef ALLIANCE_PIPELINE_PROFILE
        {
          auto t0 = ProfClock::now();
          expand_cached_candidate(si, ce);
          prof_expand_ns += prof_ns(t0);
        }
#else
        expand_cached_candidate(si, ce);
#endif
      }
      if (reads.in_flight < beam_width && !reads.pending.empty()) {
#ifdef ALLIANCE_PIPELINE_PROFILE
        {
          auto t0 = ProfClock::now();
          complete_reads();
          prof_poll_ns += prof_ns(t0);
        }
        {
          auto t0 = ProfClock::now();
          submit_pending_reads();
          prof_submit_ns += prof_ns(t0);
        }
#else
        complete_reads();
        submit_pending_reads();
#endif
      }
    }

#ifdef ALLIANCE_PIPELINE_PROFILE
    {
      auto t0 = ProfClock::now();
      complete_reads();
      prof_poll_ns += prof_ns(t0);
    }
    {
      auto t0 = ProfClock::now();
      submit_pending_reads();
      prof_submit_ns += prof_ns(t0);
    }
#else
    complete_reads();
    submit_pending_reads();
#endif

    if (reads.ready.empty() && reads.in_flight > 0) {
#ifdef ALLIANCE_PIPELINE_PROFILE
      {
        auto t0 = ProfClock::now();
        ProfileMetricWait(metrics.metric_ml_search_wait_ns_, metrics.metric_ml_search_wait_calls_,
                          [&] { ctx->ml_io_handler->PollWaitBatch(); });
        prof_poll_ns += prof_ns(t0);
      }
#else
      ProfileMetricWait(metrics.metric_ml_search_wait_ns_, metrics.metric_ml_search_wait_calls_,
                        [&] { ctx->ml_io_handler->PollWaitBatch(); });
#endif
    }
  }

#ifdef ALLIANCE_PIPELINE_PROFILE
  {
    auto total_ns = static_cast<size_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(ProfClock::now() - prof_loop_begin)
            .count());
    metrics.metric_ml_search_total_ns_ += total_ns;
  }
#endif
#ifdef ALLIANCE_PIPELINE_PROFILE
  metrics.metric_ml_search_submit_ns_ += prof_submit_ns;
  metrics.metric_ml_search_poll_ns_ += prof_poll_ns;
  metrics.metric_ml_search_expand_ns_ += prof_expand_ns;
#endif
#ifdef ALLIANCE_PIPELINE_PROFILE
  metrics.metric_ml_loop_iters_ += prof_loop_iters;
  metrics.metric_ml_inflight_sum_ += prof_inflight_sum;
#endif

  metrics.metric_ml_unique_discoveries_ += traversal.UniqueDiscoveries();
  metrics.metric_ml_duplicate_discoveries_ += traversal.DuplicateDiscoveries();
  metrics.metric_ml_rerank_reserve_candidates_ += result_pool.Reserve().size();
  if (ctx->rerank_error_bound && ctx->rerank_float_refine && ctx->rerank_reuse_estimates) {
    estimator.refinement_cache_.Reset(result_pool.Main().size() + result_pool.Reserve().size());
    for (const auto *heap : {&result_pool.Main(), &result_pool.Reserve()})
      for (const auto &c : *heap)
        estimator.refinement_cache_.Insert(c.id_, SearchCID(c.id_), c.precise_);
  }
  return result_pool.TakeCandidates();
}

// Explicit template instantiation

#define INSTANTIATE_DISK_ML_TRAVERSAL(ATT_T, VEC_T, FILTER_T)                                      \
  template auto                                                                                    \
  AllianceIndexDisk<ATT_T, VEC_T>::SearchCandidatesQueryMultiLevelPipeline<FILTER_T>(              \
      std::vector<dist_id_pair> &, const VEC_T *, const FILTER_T &, const attr_range<layer_t> &,   \
      size_t, size_t, DiskSearchContext *, AllianceIndexDisk<ATT_T, VEC_T>::RaBitQEstimator *)     \
      -> std::vector<dist_id_pair>;

INSTANTIATE_DISK_ML_TRAVERSAL(int, float, attr_range<int>)
INSTANTIATE_DISK_ML_TRAVERSAL(int, float, attr_range<att_label_t<int>>)

template auto
AllianceIndexDisk<int, float>::SearchCandidatesQueryMultiLevelPipeline<attr_range<int>, true>(
    std::vector<dist_id_pair> &, const float *, const attr_range<int> &,
    const attr_range<layer_t> &, size_t, size_t, DiskSearchContext *,
    AllianceIndexDisk<int, float>::RaBitQEstimator *) -> std::vector<dist_id_pair>;

#undef INSTANTIATE_DISK_ML_TRAVERSAL

} // namespace alliance
