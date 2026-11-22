#include "search/search_disk_report.hpp"
#include "search/search_disk_options.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sys/resource.h>
#include <unordered_set>

namespace benchmark {
void PrintMemoryPhase(const char *phase, size_t process_memory_budget_bytes) {
  size_t rss_kib=0, swap_kib=0;
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status,line)) {
    if (line.rfind("VmRSS:",0)==0) rss_kib=std::stoull(line.substr(6));
    if (line.rfind("VmSwap:",0)==0) swap_kib=std::stoull(line.substr(7));
  }
  struct rusage usage{};getrusage(RUSAGE_SELF,&usage);
  std::cout << "# memory_phase: phase=" << phase << " rss_kib=" << rss_kib
            << " peak_rss_kib=" << usage.ru_maxrss << " swap_kib=" << swap_kib << std::endl;
  // HWM catches transient allocations between phase boundaries. This validates
  // the process budget; it is not an allocator or an OS hard memory limit.
  if(process_memory_budget_bytes &&
     (static_cast<uint64_t>(usage.ru_maxrss)*1024>process_memory_budget_bytes || swap_kib))
    throw std::runtime_error(std::string("process memory budget exceeded at ")+phase);
}

void PrintSharedMemory(const SearchIndex &index) {
  auto vector_bytes=[](const auto &v) { return v.capacity()*sizeof(v[0]); };
  size_t ml_map_bytes=0;
  for (const auto &v:index.ml_disk_blocks_per_level_) ml_map_bytes+=vector_bytes(v);
  const size_t attributes_bytes=vector_bytes(index.labels_)+vector_bytes(index.atts_)+
      vector_bytes(index.sorted_ids_)+vector_bytes(index.position_by_id_)+vector_bytes(index.window_size_);
  const size_t order_bytes=index.order_table_ ? index.order_table_->AllocatedBytes() : 0;
  const size_t quant_aux_bytes=vector_bytes(index.cids_)+vector_bytes(index.search_cids_)+vector_bytes(index.factors_)+
      (index.centroids_ ? index.maxC_*index.padded_dim_*sizeof(float) : 0);
  const size_t guided_entry_bytes=index.guided_entries_.Bytes();
  const size_t navigation_bytes=index.NavigationBytes();
  const size_t vector_map_bytes=vector_bytes(index.ml_vec_block_ids_)+vector_bytes(index.ml_vec_slots_);
  std::cout << "# shared_memory: attributes_bytes=" << attributes_bytes << " order_table_bytes=" << order_bytes
            << " quant_packed_bytes=" << index.packed_rabitq_.Bytes() << " quant_code_bytes=" << vector_bytes(index.quant_vecs_) << " quant_aux_bytes=" << quant_aux_bytes
            << " guided_entry_bytes=" << guided_entry_bytes
            << " entry_navigation_bytes=" << navigation_bytes
            << " ml_map_bytes=" << ml_map_bytes << " vector_map_bytes=" << vector_map_bytes
            << " known_total_bytes=" << attributes_bytes+order_bytes+vector_bytes(index.quant_vecs_)+index.packed_rabitq_.Bytes()+quant_aux_bytes+ml_map_bytes+vector_map_bytes+guided_entry_bytes+navigation_bytes
            << " (excludes_static_rotator_runtime_and_transient_allocations)" << std::endl;
}

void PrintWorkerMemory(const alliance::MultiLevelThreadMemory &retained) {
  struct rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
  std::cout << "# worker_memory: cache_payload_bytes=" << retained.cache_payload_bytes
            << " cache_lookup_bytes=" << retained.cache_lookup_bytes
            << " cache_metadata_bytes=" << retained.cache_metadata_bytes
            << " visited_bytes=" << retained.visited_bytes
            << " io_buffer_bytes=" << retained.io_buffer_bytes
            << " prefilter_scratch_bytes=" << retained.prefilter_scratch_bytes
            << " retained_total_bytes=" << retained.Total()
            << " process_peak_rss_kib=" << usage.ru_maxrss << std::endl;
}

namespace {
void PrintPipelineProfile(const SearchIndex &index, const DiskSearchOptions &options,
    const SearchContexts &contexts, const SearchRunProfile &profile,
    const PipelineBatch &batch, double latency_avg_ms) {
  const size_t nq = batch.results.size();
  const double time = batch.seconds;
  const auto &search_contexts = contexts;
#ifdef ALLIANCE_PIPELINE_PROFILE
  const auto &io_observer = profile.io_observer;
  const auto &worker_enter_ms = batch.worker_enter_ms;
  const auto &worker_first_ms = batch.worker_first_ms;
#endif
#ifdef ALLIANCE_PIPELINE_PROFILE
  {
    uint64_t graph_request_ns = 0, vector_request_ns = 0;
    for (const auto &ctx : search_contexts) {
      for (auto *handler : {ctx->io_handler.get(), ctx->ml_io_handler.get()}) {
        if (!handler) continue;
        const auto &profile = handler->QueueProfile();
        graph_request_ns += profile.GraphRequestNs();
        vector_request_ns += profile.VectorRequestNs();
      }
    }
    const double batch_ns = time * 1e9;
    std::cout << "# io_queue: avg_graph_pending=" << graph_request_ns / batch_ns
              << " avg_vector_pending=" << vector_request_ns / batch_ns
              << " avg_total_pending=" << (graph_request_ns + vector_request_ns) / batch_ns
              << " peak_graph_pending=" << io_observer.graph_peak.load()
              << " peak_vector_pending=" << io_observer.vector_peak.load()
              << " peak_total_pending=" << io_observer.peak.load()
              << " remaining=" << io_observer.current.load()
              << " avg_graph_workers=" << index.metric_query_traversal_ns_ / batch_ns
              << " avg_rerank_workers=" << index.metric_query_rerank_ns_ / batch_ns << std::endl;
    std::cout << "# io_queue_semantics: time_weighted_submitted_not_reaped; includes_completed_unpolled; not_device_queue_depth" << std::endl;
  }
  // Batch throughput and per-query latency have different denominators in
  // concurrent runs. Phase percentages must use mean query latency.
  {
    const double unit = 1.0 / (1e6 * nq);
    const double query_ms = index.metric_query_total_ns_ * unit;
    const double wait_ms = (index.metric_ml_search_wait_ns_ + index.metric_ml_rerank_wait_ns_) * unit;
    std::cout << "# phase_timing: query_ms=" << query_ms
              << " prepare_ms=" << index.metric_query_prepare_ns_ * unit
              << " traversal_ms=" << index.metric_query_traversal_ns_ * unit
              << " rerank_ms=" << index.metric_query_rerank_ns_ * unit
              << " finalize_ms=" << index.metric_query_finalize_ns_ * unit
              << " graph_wait_ms=" << index.metric_ml_search_wait_ns_ * unit
              << " rerank_wait_ms=" << index.metric_ml_rerank_wait_ns_ * unit
              << " graph_wait_calls=" << index.metric_ml_search_wait_calls_ * 1.0 / nq
              << " rerank_wait_calls=" << index.metric_ml_rerank_wait_calls_ * 1.0 / nq
              << " blocking_wait_pct=" << (query_ms > 0 ? 100.0 * wait_ms / query_ms : 0.0)
              << " mean_query_ms=" << latency_avg_ms
              << " batch_wall_ms=" << time * 1000.0
              << " team_enter_max_ms=" << *std::max_element(worker_enter_ms.begin(), worker_enter_ms.end())
              << " first_query_mean_ms=" << std::accumulate(worker_first_ms.begin(), worker_first_ms.end(), 0.0) / options.search_threads
              << std::endl;
    std::cout << "# timing_semantics: wall_ms=mean_query_latency cpu_ms=non_io_elapsed_not_process_CPU avg_inflight=iteration_weighted" << std::endl;
  }
  // Per-query phase breakdown (ms). "#"-prefixed so CSV consumers skip it.
  // wall = mean harness-measured searchKNN latency; traversal+rerank < wall,
  // the difference is per-query setup (estimator init, DecideLayerRange).
  // other = traversal - submit - poll - expand (loop management + busy-spin).
  // avg_inflight = effective IO queue depth sustained during traversal.
  {
    const double per_q_ms = 1.0 / (1e6 * nq);
    const double t_total = index.metric_ml_search_total_ns_ * per_q_ms;
    const double t_submit = index.metric_ml_search_submit_ns_ * per_q_ms;
    const double t_poll = index.metric_ml_search_poll_ns_ * per_q_ms;
    const double t_expand = index.metric_ml_search_expand_ns_ * per_q_ms;
    const double t_rerank = index.metric_ml_rerank_ns_ * per_q_ms;
    const double t_rr_submit = index.metric_ml_rerank_submit_ns_ * per_q_ms;
    const double t_rr_poll = index.metric_ml_rerank_poll_ns_ * per_q_ms;
    const size_t iters = index.metric_ml_loop_iters_;
    const double avg_inflight = iters > 0 ? index.metric_ml_inflight_sum_ * 1.0 / iters : 0.0;
    std::cout << "# profile: wall=" << latency_avg_ms << "ms"
              << " traversal=" << t_total << "ms"
              << " (submit=" << t_submit
              << " poll=" << t_poll
              << " expand=" << t_expand
              << " other=" << (t_total - t_submit - t_poll - t_expand) << ")"
              << " rerank=" << t_rerank << "ms"
              << " (submit=" << t_rr_submit << " poll=" << t_rr_poll
              << " cpu=" << (t_rerank - t_rr_submit - t_rr_poll) << ")"
              << " avg_inflight=" << avg_inflight
              << " loop_iters=" << iters * 1.0 / nq << std::endl;
    // Machine-parseable variant (all values in ms/query, no units suffixed).
    // io = every submit+poll on both phases; cpu = wall - io (distance
    // computation, candidate management, per-query setup).
    const double io_ms = t_poll + t_submit + t_rr_poll + t_rr_submit;
    const double wall_ms = latency_avg_ms;
    std::cout << "# timing: wall_ms=" << wall_ms
              << " search_ms=" << t_total
              << " search_io_ms=" << t_poll
              << " search_cpu_ms=" << t_expand
              << " search_submit_ms=" << t_submit
              << " search_other_ms=" << (t_total - t_submit - t_poll - t_expand)
              << " rerank_ms=" << t_rerank
              << " rerank_io_ms=" << t_rr_poll
              << " rerank_cpu_ms=" << (t_rerank - t_rr_submit - t_rr_poll)
              << " rerank_submit_ms=" << t_rr_submit
              << " setup_ms=" << (wall_ms - t_total - t_rerank)
              << " io_ms=" << io_ms
              << " cpu_ms=" << (wall_ms - io_ms)
              << " io_pct=" << (wall_ms > 0 ? io_ms * 100.0 / wall_ms : 0.0)
              << " cpu_pct=" << (wall_ms > 0 ? (wall_ms - io_ms) * 100.0 / wall_ms : 0.0)
              << " avg_inflight=" << avg_inflight
              << " loop_iters=" << iters * 1.0 / nq << std::endl;
  }

#else
  std::cout << "# profile: disabled (build with ALLIANCE_PIPELINE_PROFILE=ON)" << std::endl;
#endif
}
}  // namespace

void PrintPipelineReport(const SearchIndex &index, const DiskSearchOptions &options,
    size_t efs, const SearchResults &gt, const SearchContexts &contexts,
    const SearchRunProfile &profile, PipelineBatch &batch) {
  const size_t nq = batch.results.size();
  const double time = batch.seconds;
  const auto &results = batch.results;
  auto &latency_ms = batch.latency_ms;
  const auto &retained = batch.retained;
  const double latency_avg_ms = std::accumulate(latency_ms.begin(), latency_ms.end(), 0.0) / nq;
  std::sort(latency_ms.begin(), latency_ms.end());
  const auto percentile = [&](double p) {
    const size_t pos = static_cast<size_t>(std::ceil(p * nq)) - 1;
    return latency_ms[std::min(pos, nq - 1)];
  };
  std::cout << "# latency: avg_ms=" << latency_avg_ms
            << " p95_ms=" << percentile(0.95) << " p99_ms=" << percentile(0.99)
            << " batch_wall_ms=" << time * 1000.0 << std::endl;

  const float recall = gt.empty() ? -1.0f : CalculateRecall(gt, results);
  size_t unique_correct = 0, gt_total = 0, duplicate_results = 0, short_results = 0;
  for (size_t qi = 0; qi < nq; ++qi) {
    std::unordered_set<alliance::label_t> unique(results[qi].begin(), results[qi].end());
    duplicate_results += results[qi].size() - unique.size();
    short_results += unique.size() < options.k;
    if (qi < gt.size()) {
      std::unordered_set<alliance::label_t> truth(gt[qi].begin(), gt[qi].end());
      gt_total += truth.size();
      for (const auto id : unique) unique_correct += truth.count(id);
    }
  }
  const double unique_recall = gt_total > 0 ? unique_correct * 1.0 / gt_total : -1.0;
  size_t hops = index.metric_hops_;
  double layers_per_hop = hops > 0 ? index.metric_layer_visits_ * 1.0 / hops : 0.0;
  size_t static_hits = index.metric_ml_static_cache_hits_;
  size_t dynamic_hits = index.metric_ml_cache_hits_;
  size_t misses = index.metric_ml_cache_misses_;
  size_t total_lookups = static_hits + dynamic_hits + misses;
  double dynamic_hit_rate = (dynamic_hits + misses > 0) ? dynamic_hits * 1.0 / total_lookups : 0.0;
  double static_hit_rate = (total_lookups > 0) ? static_hits * 1.0 / total_lookups : 0.0;
  double overall_hit_rate = (total_lookups > 0) ? (static_hits + dynamic_hits) * 1.0 / total_lookups : 0.0;
  std::cout << efs << "," << recall << "," << nq / time << ","
            << index.metric_dist_comps_ * 1.0 / nq << ","
            << hops * 1.0 / nq << ","
            << layers_per_hop << ","
            << index.metric_ml_io_submits_ * 1.0 / nq << ","
            << index.metric_ml_io_dedup_saved_ * 1.0 / nq << ","
            << index.metric_ml_block_reads_ * 1.0 / nq << ","
            << dynamic_hit_rate << ","
            << static_hit_rate << ","
            << overall_hit_rate << ","
            << index.metric_ml_rerank_io_submits_ * 1.0 / nq << ","
            << static_hits * 1.0 / nq << ","
            << unique_recall << ","
            << index.metric_ml_post_rerank_io_submits_ * 1.0 / nq << ","
            << index.metric_ml_rerank_bound_pruned_ * 1.0 / nq << ","
            << index.metric_ml_rerank_duplicate_ids_ * 1.0 / nq << ","
            << duplicate_results * 1.0 / nq << ","
            << index.metric_ml_rerank_candidates_ * 1.0 / nq << ","
            << index.metric_ml_rerank_reserve_candidates_ * 1.0 / nq << std::endl;
  std::cout << "# validation: unique_recall=" << unique_recall
            << " duplicate_results=" << duplicate_results
            << " short_results=" << short_results << " queries=" << nq << std::endl;
  std::cout << "# filter_routes: prefilter_queries=" << index.metric_prefilter_queries_.load()
            << " graph_queries=" << index.metric_filter_graph_queries_.load()
            << " empty_queries=" << index.metric_filter_empty_queries_.load()
            << " scanned_per_query=" << index.metric_prefilter_scanned_ * 1.0 / nq
            << " candidates_per_query=" << index.metric_prefilter_candidates_ * 1.0 / nq << std::endl;

  if (index.HasNavigation())
    std::cout << "# navigation: queries=" << index.metric_navigation_queries_
              << " scores_per_query=" << index.metric_navigation_scores_ * 1.0 / nq
              << " expansions_per_query=" << index.metric_navigation_expansions_ * 1.0 / nq
              << " edges_per_query=" << index.metric_navigation_edges_ * 1.0 / nq
              << " entries_per_query=" << index.metric_navigation_entries_ * 1.0 / nq
              << " ms_per_query=" << index.metric_navigation_ns_ * 1e-6 / nq << std::endl;

  std::cout << "# bounded_rerank: prescreen_pruned=" << index.metric_ml_rerank_prescreen_pruned_ * 1.0 / nq
            << " all_pruned=" << index.metric_ml_rerank_bound_pruned_ * 1.0 / nq
            << " post_exact=" << index.metric_ml_rerank_exact_ * 1.0 / nq << std::endl;

  PrintPipelineProfile(index, options, contexts, profile, batch, latency_avg_ms);

  // IO efficiency: actual block reads vs expanded nodes.
  // io_per_hop < 1 means some expansions reused cached blocks (hits or dedup).
  // effective_hit_rate includes dedup waiters as "hits" (no fresh IO needed).
  {
    double io_per_q = index.metric_ml_io_submits_ * 1.0 / nq;
    double hops_per_q = hops * 1.0 / nq;
    double dedup_per_q = index.metric_ml_io_dedup_saved_ * 1.0 / nq;
    double io_per_hop = hops_per_q > 0 ? io_per_q / hops_per_q : 0;
    double routed = dynamic_hits + misses + static_hits;
    double eff_miss = routed > 0 ? (misses - index.metric_ml_io_dedup_saved_) * 1.0 / routed : 0;
    std::cout << "# io_efficiency: io_per_hop=" << io_per_hop
              << " free_expand=" << (1 - io_per_hop) * 100 << "%"
              << " dedup_as_hit=" << dedup_per_q
              << " effective_miss=" << eff_miss * 100 << "%"
              << " raw_miss=" << (routed > 0 ? misses * 1.0 / routed * 100 : 0) << "%" << std::endl;
  }

  // Allocated tunable structures, not total process memory. Equal-memory
  // comparisons must also include all shared/worker/runtime allocations
  // and gate the measured process peak against the common budget.
  {
    const size_t rabitq_bytes = index.curvec_num_*index.quant_vec_bytes_;
    const size_t static_payload = index.static_cache_loaded_ ? index.static_cache_bytes_ : 0;
    const size_t static_meta = index.StaticMetadataBytes();
    // Sum actual worker allocations, including unequal auto-sized caches.
    const size_t dyn_bytes = retained.cache_payload_bytes + retained.cache_lookup_bytes +
        retained.cache_metadata_bytes;
    std::cout << "# ml_cache: auto=" << (index.ml_cache_auto_size_ ? 1 : 0)
              << " policy=" << options.ml_cache_policy
              << " budget_bytes=" << options.ml_cache_bytes
              << " peak_slots=" << index.metric_ml_cache_peak_slots_
              << " peak_wanted_slots=" << index.metric_ml_cache_peak_wanted_
              << " resident_bytes_total=" << dyn_bytes
              << std::endl;
    std::cout << "# memory: rabitq_bytes=" << rabitq_bytes
              << " quant_vec_bytes=" << index.quant_vec_bytes_
              << " static_payload_bytes=" << static_payload
              << " static_meta_bytes=" << static_meta
              << " static_resident_bytes=" << index.StaticResidentBytes()
              << " dyn_cache_bytes=" << dyn_bytes
              << " tunable_total_bytes=" << (rabitq_bytes + static_payload + static_meta + dyn_bytes)
              << " tunable_total_mib="
              << (rabitq_bytes + static_payload + static_meta + dyn_bytes) / 1048576.0
              << std::endl;
  }

  // Per-level cache breakdown. "#"-prefixed for CSV consumers.
  {
    std::cout << "# per_level:";
    for (size_t li = 0; li < index.kMaxMetricLevels; ++li) {
      size_t lh = index.metric_ml_cache_hits_per_level_[li];
      size_t lm = index.metric_ml_cache_misses_per_level_[li];
      size_t lb = index.metric_ml_block_reads_per_level_[li];
      size_t ls = index.metric_ml_static_hits_per_level_[li];
      size_t lt = ls + lh + lm;
      if (lt == 0) continue;
      double dhr = lh * 1.0 / lt;
      double shr = ls * 1.0 / lt;
      std::cout << " L" << li
                << "{lookups=" << lt
                << " static=" << ls << "(" << shr * 100 << "%)"
                << " dynamic=" << lh << "(" << dhr * 100 << "%)"
                << " miss=" << lm
                << " blk_reads=" << lb << "}";
    }
    std::cout << " inject=" << index.metric_ml_block_inject_ * 1.0 / nq;
    std::cout << " unique_discoveries=" << index.metric_ml_unique_discoveries_ * 1.0 / nq
              << " duplicate_discoveries=" << index.metric_ml_duplicate_discoveries_ * 1.0 / nq;
#ifdef ALLIANCE_PIPELINE_DIAGNOSTICS
    {
      size_t cb_n = index.metric_block_coblock_neighbor_;
      size_t cb_t = index.metric_block_coblock_total_;
      double cb_pct = cb_t > 0 ? cb_n * 100.0 / cb_t : 0;
      std::cout << " coblock=" << cb_n * 1.0 / nq << "/" << cb_t * 1.0 / nq
                << "(" << cb_pct << "%)";
    }
#else
    std::cout << " coblock=disabled";
#endif
    std::cout << std::endl;
  }

  // Per-source-layer routing breakdown (expand_node Phase 3 only).
  {
    std::cout << "# per_src_layer:";
    for (size_t li = 0; li < index.kMaxMetricLayers; ++li) {
      size_t h = index.metric_ml_route_hits_per_layer_[li];
      size_t m = index.metric_ml_route_misses_per_layer_[li];
      size_t t = h + m;
      if (t == 0) continue;
      double hr = h * 1.0 / t;
      std::cout << " ly" << li
                << "{routed=" << t
                << " hit=" << h << "(" << hr * 100 << "%)"
                << " miss=" << m << "}";
    }
    std::cout << std::endl;
  }
}
}  // namespace benchmark
