// alliance-formal-v1 timing/observer protocol; only stage ownership is split.
#include "benchmark/batch.hpp"
#include "search/search_preparation.hpp"
#include "benchmark/archive.hpp"
#include "benchmark/checks.hpp"
#include "benchmark/io_observer.hpp"
#include "search/disk_search_workspace.hpp"
#include <chrono>
#include <unordered_set>

namespace benchmark::formal {
namespace {
using Clock = std::chrono::steady_clock;
uint64_t current_rss() {
  std::ifstream f("/proc/self/status");
  std::string line;
  while (std::getline(f, line))
    if (line.rfind("VmRSS:", 0) == 0)
      return std::stoull(line.substr(6)) * 1024;
  throw std::runtime_error("RSS unavailable");
}

struct Measurement {
  double wall;
  IOCounters io;
  uint64_t rss_before, rss_after;
  bool rss_ok;
};
struct Validation {
  uint64_t hits = 0, invalid = 0, duplicates = 0, predicate = 0, shorts = 0;
};

std::filesystem::path PrepareBatch(FormalBatchSession &session, const Batch &batch) {
  auto &index = session.index;
  auto &ctx = session.context;
  const auto &protocol = session.protocol;
  const size_t nq = session.inputs.count;
  const auto ef = batch.ef, repeat = batch.repeat, trial = batch.warmup_ordinal;
  const bool warmup = batch.warmup;
  const auto dir = std::filesystem::path(protocol.output) /
                   (warmup ? "warmup_r" + std::to_string(repeat) + "_" + std::to_string(trial)
                           : "r" + std::to_string(repeat)) /
                   ("ef" + std::to_string(ef));
  Require(std::filesystem::create_directories(dir), "point already exists: do not repeat");
  {
    std::ofstream f(dir / "started.json");
    f << "{\"ef\":" << ef << ",\"queries\":" << nq << "}\n";
    Require(bool(f), "started write");
  }
  index.InvalidateMultiLevelCache();
  // Allocate the empty cache before timing, as PipeWoW does. No query or
  // graph block is read here; the first real request remains cold.
  auto &workspace = alliance::detail::GetGraphSearchWorkspace(&index, &ctx);
  workspace.GetVisited(index.curvec_num_);
  Require(workspace.Cache().size() == 0, "cache not empty before batch");
  alliance::ResetMultiLevelMetrics(index);
  return dir;
}

Measurement MeasureBatch(FormalBatchSession &session, size_t ef,
                         std::vector<Observation> &observations) {
  auto &index = session.index;
  auto &ctx = session.context;
  const auto &o = session.options;
  const auto &protocol = session.protocol;
  const auto &geometry = session.geometry;
  const auto &q = session.inputs.vectors;
  const auto &ranges = session.inputs.ranges;
  const size_t d = session.inputs.dimension, nq = session.inputs.count;
  auto &phase_memory = session.phase_memory;
  const auto search_limit = protocol.search_limit;
  IOStart(o.ml_graph_file.c_str(), geometry.vector_start, geometry.edge_bytes,
          geometry.vector_bytes, geometry.header_bytes, geometry.file_bytes);
  const uint64_t rss_before = current_rss();
  Require(!search_limit || current_rss() <= search_limit, "initial search RSS exceeds budget");
  phase_memory.Begin();
  const auto batch_start = Clock::now();
  for (size_t i = 0; i < nq; ++i) {
    auto &z = observations[i];
    auto before = IOSnapshot();
    uint64_t g = index.metric_ml_io_submits_, r = index.metric_ml_rerank_io_submits_,
             h = index.metric_hops_;
    auto start = Clock::now();
    // Keep only returned neighbors in the batch archive. Moving the search
    // vector here would retain its EF-sized capacity for every query.
    auto result = index.searchKNNMultiLevelPipeline(q.get() + i * d, ef, o.k, ranges[i], &ctx);
    z.result.assign(result.begin(), result.end());
    z.latency = std::chrono::duration<double>(Clock::now() - start).count();
    auto after = IOSnapshot();
    z.requests = after.requests - before.requests;
    z.completions = after.completions - before.completions;
    z.bytes = after.requested_bytes - before.requested_bytes;
    z.graph = uint64_t(index.metric_ml_io_submits_) - g;
    z.rerank = uint64_t(index.metric_ml_rerank_io_submits_) - r;
    z.hops = uint64_t(index.metric_hops_) - h;
    const auto observed_graph = after.graph_requests - before.graph_requests;
    const auto observed_vector = after.vector_requests - before.vector_requests;
    if (z.graph != observed_graph || z.rerank != observed_vector) {
      std::cerr << "IO_SOURCE_MISMATCH ef=" << ef << " query=" << i << " native_graph=" << z.graph
                << " native_rerank=" << z.rerank << " observed_neighbor=" << observed_graph
                << " observed_vector=" << observed_vector << '\n';
      throw std::runtime_error("per-query source classification");
    }
    Require(z.bytes == geometry.edge_bytes * z.graph + geometry.vector_bytes * z.rerank,
            "per-query logical read bytes");
  }
  const double wall = std::chrono::duration<double>(Clock::now() - batch_start).count();
  auto io = IOStop();
  phase_memory.End();
  const uint64_t rss_after = current_rss();
  const bool rss_ok = phase_memory.search_peak_rss <= search_limit;
  return {wall, io, rss_before, rss_after, rss_ok};
}

void ValidateBatch(FormalBatchSession &session, size_t ef, const std::filesystem::path &dir,
                   const Measurement &measured, std::vector<Observation> &observations,
                   Validation &validation) {
  auto &index = session.index;
  const auto &o = session.options;
  const auto &geometry = session.geometry;
  const auto &ranges = session.inputs.ranges;
  const auto &gt = session.inputs.ground_truth;
  const auto &attrs = session.inputs.attributes;
  const size_t nq = session.inputs.count;
  const auto &io = measured.io;
  const bool rss_ok = measured.rss_ok;
  auto &hits = validation.hits;
  auto &invalid = validation.invalid;
  auto &duplicates = validation.duplicates;
  auto &predicate = validation.predicate;
  auto &shorts = validation.shorts;
  // Result validation and export are outside the timed search phase.
  for (size_t i = 0; i < nq; ++i) {
    auto &z = observations[i];
    std::unordered_set<size_t> seen;
    shorts += z.result.size() != o.k;
    for (const auto &[dist, id] : z.result) {
      const bool unique = seen.insert(id).second;
      duplicates += !unique;
      if (id >= attrs.size() || !std::isfinite(dist)) {
        ++invalid;
        continue;
      }
      predicate += !ranges[i].Test(attrs[id]);
      if (unique && std::find(gt[i].begin(), gt[i].begin() + o.k, id) != gt[i].begin() + o.k) {
        ++hits;
        ++z.matches;
      }
    }
  }
  Require(io.pending == 0 && io.errors == 0 && io.unknown_fd == 0 && io.bad_layout == 0,
          "AIO integrity");
  Require(io.requests == io.completions && io.requested_bytes == io.completed_bytes &&
              io.requested_bytes == geometry.edge_bytes * io.graph_requests +
                                        geometry.vector_bytes * io.vector_requests,
          "AIO counts");
  Require(io.requests ==
              uint64_t(index.metric_ml_io_submits_) + uint64_t(index.metric_ml_rerank_io_submits_),
          "native/observer I/O mismatch");
  Require(io.graph_requests == uint64_t(index.metric_ml_io_submits_) &&
              io.vector_requests == uint64_t(index.metric_ml_rerank_io_submits_),
          "observer region/native split mismatch");
  Require(!invalid && !duplicates && !predicate, "result correctness");
  Require(uint64_t(index.metric_prefilter_queries_) +
                  uint64_t(index.metric_filter_graph_queries_) ==
              nq,
          "routing coverage");
  if (o.filter_mode == "graph")
    Require(uint64_t(index.metric_prefilter_queries_) == 0, "graph-only routing");
  Require(rss_ok, "search RSS budget exceeded");
  if (shorts) {
    std::ofstream rejected(dir / "rejected.json");
    rejected << "{\"ef\":" << ef << ",\"queries\":" << nq << ",\"short\":" << shorts
             << ",\"postfilter_queries\":" << index.metric_filter_top_postfilter_queries_ << "}\n";
    Require(bool(rejected), "rejected point write");
    std::cout << "REJECTED ef=" << ef << " short=" << shorts << std::endl;
    throw std::runtime_error("short results: rejected point retained, no retry");
  }
}

void ArchiveBatch(FormalBatchSession &session, const Batch &batch, const std::filesystem::path &dir,
                  const Measurement &measured, const Validation &validation,
                  const std::vector<Observation> &observations) {
  auto &index = session.index;
  auto &ctx = session.context;
  const auto &o = session.options;
  const auto &protocol = session.protocol;
  const auto &geometry = session.geometry;
  const size_t nq = session.inputs.count;
  auto &phase_memory = session.phase_memory;
  const auto search_limit = protocol.search_limit;
  const auto wall = measured.wall;
  const auto &io = measured.io;
  const auto rss_before = measured.rss_before, rss_after = measured.rss_after;
  const bool rss_ok = measured.rss_ok;
  const auto ef = batch.ef, repeat = batch.repeat;
  const auto fp = session.fingerprint, hits = validation.hits;
  auto &workspace = alliance::detail::GetGraphSearchWorkspace(&index, &ctx);
  benchmark::formal::WriteResults(dir, observations, o.k, ef);
  std::vector<double> latency;
  for (auto &z : observations)
    latency.push_back(z.latency);
  std::sort(latency.begin(), latency.end());
  auto retained = ctx.WorkspaceMemory(&index, index.ml_cache_config_.generation);
  Require(retained.cache_payload_bytes <= protocol.dynamic_budget,
          "dynamic payload exceeds protocol");
  std::ofstream f(dir / "measured.json");
  f << std::setprecision(17) << "{\"protocol\":\"alliance-formal-v1\",\"repeat\":" << repeat
    << ",\"requested_repeats\":" << protocol.repeats << ",\"quantization_fingerprint\":\"" << fp
    << "\",\"bits\":" << o.rabitq_bits << ",\"search_rss_limit_bytes\":" << search_limit
    << ",\"rss_budget_passed\":" << (rss_ok ? "true" : "false") << ",\"ef\":" << ef
    << ",\"queries\":" << nq << ",\"k\":" << o.k << ",\"matches\":" << hits
    << ",\"recall\":" << double(hits) / (nq * o.k) << ",\"batch_wall_seconds\":" << wall
    << ",\"qps\":" << nq / wall << ",\"p95_ms\":" << latency[size_t(std::ceil(.95 * nq)) - 1] * 1000
    << ",\"io_requests\":" << io.requests << ",\"io_completions\":" << io.completions
    << ",\"io_bytes\":" << io.requested_bytes << ",\"edge_block_bytes\":" << geometry.edge_bytes
    << ",\"vector_block_bytes\":" << geometry.vector_bytes
    << ",\"observer_graph_requests\":" << io.graph_requests
    << ",\"observer_vector_requests\":" << io.vector_requests
    << ",\"graph_io\":" << index.metric_ml_io_submits_
    << ",\"rerank_io\":" << index.metric_ml_rerank_io_submits_
    << ",\"prefilter_queries\":" << index.metric_prefilter_queries_
    << ",\"graph_queries\":" << index.metric_filter_graph_queries_
    << ",\"postfilter_queries\":" << index.metric_filter_top_postfilter_queries_
    << ",\"top_layer_postfilter\":" << (o.top_layer_postfilter ? "true" : "false")
    << ",\"entry_mode\":\"" << benchmark::SearchEntryMode(o) << "\""
    << ",\"entry_samples\":" << o.entry_samples << ",\"entry_count\":" << o.entry_count
    << ",\"guided_entry_bytes\":" << index.guided_entries_.Bytes()
    << ",\"navigation_bytes\":" << index.NavigationBytes()
    << ",\"navigation_ef\":" << (index.HasNavigation() ? o.navigation_ef : 0)
    << ",\"navigation_budget\":" << (index.HasNavigation() ? o.navigation_budget : 0)
    << ",\"navigation_count\":" << (index.HasNavigation() ? o.navigation_count : 0)
    << ",\"navigation_queries\":" << index.metric_navigation_queries_
    << ",\"navigation_scores\":" << index.metric_navigation_scores_
    << ",\"navigation_expansions\":" << index.metric_navigation_expansions_
    << ",\"navigation_edges\":" << index.metric_navigation_edges_
    << ",\"navigation_entries\":" << index.metric_navigation_entries_
    << ",\"navigation_ns\":" << index.metric_navigation_ns_
    << ",\"peak_rss_bytes\":" << phase_memory.lifetime_peak_rss
    << ",\"dynamic_cache_payload_bytes\":" << retained.cache_payload_bytes
    << ",\"dynamic_cache_clear_per_query\":" << (o.per_query_cache ? "true" : "false")
    << ",\"dynamic_cache_clear_per_ef\":true,\"dynamic_cache_auto_size\":"
    << (o.ml_cache_auto ? "true" : "false")
    << ",\"cache_entries_before_timing\":0,\"cache_entries_after_timing\":"
    << workspace.Cache().size() << ",\"rss_before_batch_bytes\":" << rss_before
    << ",\"rss_after_batch_bytes\":" << rss_after
    << ",\"loading_peak_rss_bytes\":" << phase_memory.loading_peak_rss
    << ",\"search_phase_peak_rss_bytes\":" << phase_memory.search_peak_rss
    << ",\"loading_cgroup_peak_bytes\":" << phase_memory.loading_cgroup_peak
    << ",\"search_phase_cgroup_peak_bytes\":" << phase_memory.search_cgroup_peak

    << ",\"dynamic_cache_lookup_bytes\":" << retained.cache_lookup_bytes
    << ",\"dynamic_cache_metadata_bytes\":" << retained.cache_metadata_bytes
    << ",\"static_payload_bytes\":" << index.static_cache_bytes_
    << ",\"static_metadata_bytes\":" << index.StaticMetadataBytes()
    << ",\"static_resident_bytes\":" << index.StaticResidentBytes()
    << ",\"static_cached_nodes\":" << index.static_node_lookup_.Count()
    << ",\"static_hits\":" << index.metric_ml_static_cache_hits_
    << ",\"dynamic_hits\":" << index.metric_ml_cache_hits_
    << ",\"quant_packed_bytes\":" << index.packed_rabitq_.Bytes()
    << ",\"quant_packed_stride\":" << index.packed_rabitq_.Stride()

    << ",\"invalid\":0,\"duplicates\":0,\"predicate\":0,\"short\":0,\"observer_errors\":0,\"unknown_fd\":0,\"bad_layout\":0,\"pending\":0}\n";
  Require(bool(f), "measured write");
  f.close();
  std::cout << "MEASURED ef=" << ef << " recall=" << double(hits) / (nq * o.k)
            << " qps=" << nq / wall << std::endl;
}
} // namespace

void RunFormalBatches(FormalBatchSession &session, const std::vector<size_t> &efs) {
  for (const auto &batch : Schedule(session.protocol, efs)) {
    const auto directory = PrepareBatch(session, batch);
    std::vector<Observation> observations(session.inputs.count);
    Validation validation;
    const auto measured = MeasureBatch(session, batch.ef, observations);
    ValidateBatch(session, batch.ef, directory, measured, observations, validation);
    if (!batch.warmup)
      ArchiveBatch(session, batch, directory, measured, validation, observations);
  }
}
} // namespace benchmark::formal
