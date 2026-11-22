// Exact rerank over the layout's colocated vector region. Request lifetimes,
// completion/draining, candidate grouping and width policies stay local here.
// Mutable buffers come from the same workspace used by graph and prefilter.
#if defined(ALLIANCE_PIPELINE_DIAGNOSTICS) && !defined(ALLIANCE_PIPELINE_PROFILE)
#define ALLIANCE_PIPELINE_PROFILE
#endif

#include "index/index_disk.hpp"
#include "search/bounded_rerank.hpp"
#include "search/disk_search_workspace.hpp"
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace alliance {
namespace {
using detail::DiskSearchWorkspace;
using detail::GetSearchWorkspace;
} // namespace

// RerankMultiLevel -- rerank from the layout's colocated vector region
//
// Block grouping follows StorageAirRFANNS's FetchExactDistancesForRawVecs:
// each distinct candidate vector block is read once. The raw-layout fallback
// instead reads a full node record for each candidate absent from its vector cache.

template <typename att_t, typename vec_t>
void AllianceIndexDisk<att_t, vec_t>::RerankMultiLevel(
    const vec_t *query_vec, const std::vector<dist_id_pair> &candidates, DiskSearchContext *ctx,
    std::vector<std::pair<dist_t, label_t>> &out_results) {
  ScopedMultiLevelMetrics metric_scope(*this);
  auto &metrics = metric_scope.Get();
  out_results.resize(candidates.size());
  if (candidates.empty())
    return;
  if (!ml_vec_region_loaded_) {
    throw std::runtime_error("RerankMultiLevel requires a layout with a colocated vector region");
  }
  if (!ctx->ml_io_handler) {
    throw std::runtime_error("RerankMultiLevel requires DiskSearchContext to be constructed "
                             "with a non-empty ml_graph_path");
  }

  const size_t vec_bytes = VecBytes();

  // Group candidates by physical vector block: one IO per distinct block.
  // Pair = (index into candidates/out_results, slot within the block).
  std::unordered_map<size_t, std::vector<std::pair<size_t, size_t>>> block_to_cands;
  block_to_cands.reserve(candidates.size());
  for (size_t i = 0; i < candidates.size(); ++i) {
    const tableint id = candidates[i].id_;
    block_to_cands[ml_vec_block_ids_[id]].emplace_back(i, ml_vec_slots_[id]);
  }

  std::vector<std::pair<size_t, const std::vector<std::pair<size_t, size_t>> *>> block_entries;
  block_entries.reserve(block_to_cands.size());
  for (auto &kv : block_to_cands) {
    block_entries.emplace_back(kv.first, &kv.second);
  }

  struct RerankSlot {
    const std::vector<std::pair<size_t, size_t>> *cands{nullptr};
    IORequest req;
    char *read_buf{nullptr};
    bool in_flight{false};
  };

  // This unbounded path reads every grouped block. Width controls occupancy,
  // not the block set; zero policy width uses the context queue depth.
  const size_t beam = std::max<size_t>(1, std::min(ctx->RerankWidth(), block_entries.size()));
  DiskSearchWorkspace &rtl = GetSearchWorkspace(this, ctx);
  // Rerank buffers start after the traversal slots in the shared pool.
  const size_t rerank_buf_offset = ctx->edge_beam_width;
  rtl.EnsurePipelineBuffers(rerank_buf_offset + beam, ml_vec_block_size_);
  std::vector<RerankSlot> slots(beam);
  IOCompletionGuard io_completion(*ctx->ml_io_handler);
  for (size_t i = 0; i < beam; ++i) {
    slots[i].read_buf = rtl.PipelineBuffer(rerank_buf_offset + i);
  }

  size_t next_block = 0;
  size_t rerank_in_flight = 0;
  bool read_failed = false;
#ifdef ALLIANCE_PIPELINE_PROFILE
  // Accumulate loop timing locally, then add totals to the active metric scope.
  using RrClock = std::chrono::steady_clock;
  auto rr_ns = [](RrClock::time_point t0) -> size_t {
    return static_cast<size_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(RrClock::now() - t0).count());
  };
  size_t rr_submit_ns = 0, rr_poll_ns = 0;
#endif

  // The handler retains request pointers, not this submission list. Reserve
  // before publishing requests, then reuse the list for this rerank only.
  std::vector<IORequest *> batch;
  batch.reserve(beam);
  while (next_block < block_entries.size() || rerank_in_flight > 0) {
    batch.clear();
    while (next_block < block_entries.size() && rerank_in_flight < beam) {
      size_t free_slot = beam;
      for (size_t i = 0; i < beam; ++i) {
        if (!slots[i].in_flight) {
          free_slot = i;
          break;
        }
      }
      if (free_slot == beam)
        break;

      auto &slot = slots[free_slot];
      slot.cands = block_entries[next_block].second;
      slot.req.offset = MultiLevelVectorOffset(block_entries[next_block].first);
      slot.req.len = ml_vec_block_size_;
      slot.req.buf = slot.read_buf;
      slot.req.finished = false;
      slot.in_flight = true;
      batch.push_back(&slot.req);
      ++next_block;
      ++rerank_in_flight;
      ++metrics.metric_ml_rerank_io_submits_;
      ++metrics.metric_ml_post_rerank_io_submits_;
    }

    if (!batch.empty()) {
#ifdef ALLIANCE_PIPELINE_PROFILE
      const auto t_submit = RrClock::now();
#endif
      ctx->ml_io_handler->SubmitVecReadBatch(batch);
#ifdef ALLIANCE_PIPELINE_PROFILE
      rr_submit_ns += rr_ns(t_submit);
#endif
    }

    if (rerank_in_flight == 0)
      break;

    // Wait for at least one completion, then collect other ready events.
    // This supports both blocking and busy-poll completion policies.
#ifdef ALLIANCE_PIPELINE_PROFILE
    const auto t_poll = RrClock::now();
#endif
    ProfileMetricWait(metrics.metric_ml_rerank_wait_ns_, metrics.metric_ml_rerank_wait_calls_,
                      [&] { ctx->ml_io_handler->PollVecWait(); });
    ctx->ml_io_handler->PollVecAll();
#ifdef ALLIANCE_PIPELINE_PROFILE
    rr_poll_ns += rr_ns(t_poll);
#endif
    for (auto &slot : slots) {
      if (!slot.in_flight || !slot.req.finished)
        continue;
      if (slot.req.res != static_cast<long>(ml_vec_block_size_)) {
        read_failed = true;
      } else {
        for (const auto &[ri, vslot] : *slot.cands) {
          const vec_t *vec_ptr = reinterpret_cast<const vec_t *>(slot.read_buf + vslot * vec_bytes);
          out_results[ri].first = fstdistfunc_(query_vec, vec_ptr, dist_func_param_);
          out_results[ri].second = labels_[candidates[ri].id_];
        }
      }
      slot.cands = nullptr;
      slot.in_flight = false;
      --rerank_in_flight;
    }
  }
#ifdef ALLIANCE_PIPELINE_PROFILE
  metrics.metric_ml_rerank_submit_ns_ += rr_submit_ns;
  metrics.metric_ml_rerank_poll_ns_ += rr_poll_ns;
#endif
  // Drain every submitted request before releasing slots/buffers on failure.
  if (read_failed)
    throw std::runtime_error("short or failed read in multi-level rerank");
}

// Bounded post-traversal rerank: continuously refill completed IO slots,
// checking bounds with the latest completed exact top-k before submission.

template <typename att_t, typename vec_t>
void AllianceIndexDisk<att_t, vec_t>::RerankMultiLevelBounded(
    const vec_t *query_vec, const std::vector<dist_id_pair> &candidates, size_t k,
    RaBitQEstimator &estimator, DiskSearchContext *ctx,
    std::vector<std::pair<dist_t, label_t>> &out_results) {
  ScopedMultiLevelMetrics metric_scope(*this);
  auto &metrics = metric_scope.Get();
  if (!ml_vec_region_loaded_ || !ctx->ml_io_handler ||
      (rabitq_space_name_ != "l2" && rabitq_space_name_ != "ip"))
    throw std::invalid_argument("bounded rerank requires an L2/IP multi-level vector region");
  out_results.clear();
  std::vector<detail::BoundedRerankCandidate> bounded;
  bounded.reserve(candidates.size());
  for (const auto &c : candidates) {
    const auto interval = estimator.distance_bounds(SearchCID(c.id_), c.id_, c.dist_);
    bounded.push_back({c.id_, ml_vec_block_ids_[c.id_], c.dist_, interval.first, interval.second});
  }
  const size_t width = std::max<size_t>(1, ctx->RerankWidth());
  const size_t batch_width = std::min(width, ctx->rerank_batch_blocks);
#ifdef ALLIANCE_PIPELINE_PROFILE
  using RrClock = std::chrono::steady_clock;
  auto rr_ns = [](RrClock::time_point t0) -> size_t {
    return static_cast<size_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(RrClock::now() - t0).count());
  };
  size_t rr_submit_ns = 0, rr_poll_ns = 0;
#endif
  DiskSearchWorkspace &rtl = GetSearchWorkspace(this, ctx);
  const size_t offset = ctx->edge_beam_width;
  rtl.EnsurePipelineBuffers(offset + width, ml_vec_block_size_);
  const size_t vec_bytes = VecBytes();
  // Requests and completion scratch live for the entire rerank. Slot buffers
  // are reused only after completion; no per-batch allocation or batch barrier.
  std::vector<IORequest> requests(width);
  IOCompletionGuard io_completion(*ctx->ml_io_handler);
  std::vector<const detail::BoundedRerankBlock *> active(width, nullptr);
  std::vector<IORequest *> batch;
  batch.reserve(width);
  std::vector<std::pair<size_t, float>> exact;
  exact.reserve(candidates.size());
  out_results.reserve(candidates.size());
  auto submit = [&](const auto &blocks) {
    batch.clear();
    for (const auto &entry : blocks) {
      const size_t slot = entry.first;
      active[slot] = entry.second;
      auto &req = requests[slot];
      req.offset = MultiLevelVectorOffset(entry.second->block);
      req.len = ml_vec_block_size_;
      req.buf = rtl.PipelineBuffer(offset + slot);
      req.finished = false;
      batch.push_back(&req);
    }
#ifdef ALLIANCE_PIPELINE_PROFILE
    const auto t_submit = RrClock::now();
#endif
    ctx->ml_io_handler->SubmitVecReadBatch(batch);
#ifdef ALLIANCE_PIPELINE_PROFILE
    rr_submit_ns += rr_ns(t_submit);
#endif
    metrics.metric_ml_rerank_io_submits_ += blocks.size();
    metrics.metric_ml_post_rerank_io_submits_ += blocks.size();
  };
  auto poll = [&](auto complete) {
#ifdef ALLIANCE_PIPELINE_PROFILE
    const auto t_poll = RrClock::now();
#endif
    ProfileMetricWait(metrics.metric_ml_rerank_wait_ns_, metrics.metric_ml_rerank_wait_calls_,
                      [&] { ctx->ml_io_handler->PollVecWait(); });
    ctx->ml_io_handler->PollVecAll();
#ifdef ALLIANCE_PIPELINE_PROFILE
    rr_poll_ns += rr_ns(t_poll);
#endif
    for (size_t slot = 0; slot < width; ++slot) {
      if (!active[slot] || !requests[slot].finished)
        continue;
      exact.clear();
      const bool success = requests[slot].res == static_cast<long>(ml_vec_block_size_);
      if (success) {
        for (const auto &c : active[slot]->candidates) {
          const vec_t *vec = reinterpret_cast<const vec_t *>(rtl.PipelineBuffer(offset + slot) +
                                                             ml_vec_slots_[c.id] * vec_bytes);
          const float distance = fstdistfunc_(query_vec, vec, dist_func_param_);
          exact.emplace_back(c.id, distance);
          out_results.emplace_back(distance, labels_[c.id]);
        }
      }
      active[slot] = nullptr;
      complete(slot, success, exact);
    }
  };
  std::function<std::pair<float, float>(const detail::BoundedRerankCandidate &)> refine;
  if (ctx->rerank_float_refine)
    refine = [&](const auto &c) {
      ++ctx->rerank_float_refined;
      const size_t before = estimator.refinement_reused_,
                   before_recomputed = estimator.refinement_recomputed_;
      const auto interval = estimator.distance_bounds(SearchCID(c.id), c.id, c.estimate, true);
      if (estimator.refinement_reused_ != before)
        ++ctx->rerank_float_reused;
      ctx->rerank_float_recomputed += estimator.refinement_recomputed_ - before_recomputed;
      return interval;
    };
  const auto stats = detail::RerankAsyncWithBounds(
      bounded, k, width, batch_width, submit, poll,
      ctx->rerank_bound_prescreen
          ? DiskSearchContext::SaturatingProduct(k, ctx->rerank_bound_rank_factor)
          : 0,
      ctx->rerank_warm_blocks, refine);
  metrics.metric_ml_rerank_bound_pruned_ += stats.pruned;
  metrics.metric_ml_rerank_prescreen_pruned_ += stats.prescreen_pruned;
  metrics.metric_ml_rerank_exact_ += out_results.size();
#ifdef ALLIANCE_PIPELINE_PROFILE
  metrics.metric_ml_rerank_submit_ns_ += rr_submit_ns;
  metrics.metric_ml_rerank_poll_ns_ += rr_poll_ns;
#endif
}

template void
AllianceIndexDisk<int, float>::RerankMultiLevel(const float *, const std::vector<dist_id_pair> &,
                                                DiskSearchContext *,
                                                std::vector<std::pair<dist_t, label_t>> &);

template void AllianceIndexDisk<int, float>::RerankMultiLevelBounded(
    const float *, const std::vector<dist_id_pair> &, size_t, RaBitQEstimator &,
    DiskSearchContext *, std::vector<std::pair<dist_t, label_t>> &);

} // namespace alliance
