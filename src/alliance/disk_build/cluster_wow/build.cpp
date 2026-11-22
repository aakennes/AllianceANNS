#include "index/index_disk.hpp"
#include "disk_build/memory/process.hpp"
#include "io/aio_stream.hpp"
#include "disk_build/cluster_wow/passes.hpp"
#include "disk_build/partition/membership.hpp"
#include "disk_build/cluster_wow/input.hpp"
#include "disk_build/cluster_wow/scratch.hpp"
#include "disk_build/cluster_wow/graph.hpp"
#include <algorithm>
#include <chrono>
#include <exception>
#include <memory>
#include <vector>
#include <cstdio>
#include <cstring>

namespace alliance {
namespace {

// All vectors and seed IDs must be loaded before any seed distance is evaluated.
template <typename vec_t, class Graph>
void LoadClusterVectorsAndSeeds(disk_build::SequentialClusterInputs &inputs, size_t cluster,
                                size_t n, size_t dimension, size_t max_degree, size_t total_nodes,
                                const uint32_t *ranks, const std::vector<uint32_t> &local,
                                const tableint *position_by_id, std::vector<vec_t> &vectors,
                                Graph &graph) {
  auto reader = inputs.Reader(cluster);
  std::vector<char> record(inputs.RecordBytes());
  std::vector<uint32_t> seed;
  seed.reserve(max_degree);
  for (size_t u = 0; u < n; ++u) {
    reader->Read(record.data(), record.size());
    uint32_t rank = 0, degree = 0;
    std::memcpy(&rank, record.data(), 4);
    std::memcpy(&degree, record.data() + 4, 4);
    if (rank != ranks[u] || degree > max_degree)
      throw std::runtime_error("cluster input rank/degree mismatch");
    std::memcpy(vectors.data() + u * dimension, record.data() + 8, dimension * sizeof(vec_t));
    seed.clear();
    for (size_t j = 0; j < degree; ++j) {
      tableint id = 0;
      std::memcpy(&id, record.data() + 8 + dimension * sizeof(vec_t) + j * sizeof(tableint),
                  sizeof(tableint));
      if (id >= total_nodes)
        throw std::runtime_error("invalid cluster input neighbor");
      const auto slot = local[position_by_id[id]];
      if (slot != UINT32_MAX)
        seed.push_back(slot);
    }
    graph.StageSeed(uint32_t(u), seed);
  }
}

template <class Graph> void FinalizeClusterSeeds(Graph &graph, size_t n, int threads) {
  std::exception_ptr error;
#pragma omp parallel for num_threads(threads) schedule(static)
  for (size_t u = 0; u < n; ++u)
    try {
      graph.FinishStagedSeed(uint32_t(u));
    } catch (...) {
#pragma omp critical(alliance_cluster_input_seed_error)
      {
        if (!error)
          error = std::current_exception();
      }
    }
  if (error)
    std::rethrow_exception(error);
}

// Preserve distance/ID order and the ID-only record layout consumed by Pass B.
template <class Graph>
void WriteClusterSpill(const disk_build::ClusterSpillPlan &plan, size_t cluster, size_t n,
                       size_t degree, const uint32_t *ranks, const tableint *sorted_ids,
                       Graph &graph) {
  const std::string path = plan.scratch.File(cluster);
  const uint64_t bytes = uint64_t(n) * plan.rec_bytes;
  AioPreallocate(path, (bytes + kAioAlign - 1) / kAioAlign * kAioAlign);
  {
    AioRegionWriter writer(path, 0);
    std::vector<char> record(plan.rec_bytes);
    for (size_t u = 0; u < n; ++u) {
      std::fill(record.begin(), record.end(), 0);
      uint32_t rank = ranks[u];
      std::memcpy(record.data(), &rank, 4);
      for (size_t l = 0; l < plan.L; ++l) {
        auto edges = graph.Edges(uint32_t(u), l + 1);
        std::sort(edges.begin(), edges.end(), [](const auto &a, const auto &b) {
          return a.dist_ < b.dist_ || (a.dist_ == b.dist_ && a.id_ < b.id_);
        });
        uint32_t size = edges.size();
        std::memcpy(record.data() + 4 * (l + 1), &size, 4);
        for (size_t j = 0; j < edges.size(); ++j) {
          if (!graph.Allowed(uint32_t(u), edges[j].id_, l + 1))
            throw std::runtime_error("cluster WoW edge outside global window");
          const tableint id = sorted_ids[ranks[edges[j].id_]];
          std::memcpy(record.data() + plan.rec_head + (l * degree + j) * sizeof(tableint), &id,
                      sizeof(tableint));
        }
      }
      writer.Append(record.data(), record.size());
    }
    writer.Close();
  }
}

} // namespace

template <typename att_t, typename vec_t>
void AllianceIndexDisk<att_t, vec_t>::BuildClusterWoWEdgesDisk(
    const disk_build::ClusterSpillPlan &plan) {
  if (!build_config_.cluster_wow_passes_ || build_config_.cluster_wow_efc_ < M_ ||
      plan.start_layer < 1)
    throw std::invalid_argument("invalid cluster WoW configuration");
  const auto started = std::chrono::steady_clock::now();
  const auto &p = plan.memory;
  size_t max_size = 0;
  for (size_t c = 0; c < ClusterScratch()->K; ++c)
    max_size = std::max(max_size, size_t(ClusterScratch()->cluster_rank_offsets_[c + 1] -
                                         ClusterScratch()->cluster_rank_offsets_[c]));
  if (max_size > p.cap_slots ||
      (build_config_.build_mem_gb_ > 0 &&
       p.base_bytes + p.residency_bytes > size_t(build_config_.build_mem_gb_ * (1ull << 30))))
    throw std::runtime_error("cluster WoW actual cluster/model exceeds memory budget");
  auto inputs = std::make_unique<disk_build::SequentialClusterInputs>(
      plan.scratch.InputFile(), plan.n, VecBytes(), M_,
      ClusterScratch()->cluster_rank_offsets_.data(), ClusterScratch()->K);
  {
    const auto distribution_start = std::chrono::steady_clock::now();
    inputs->Distribute(plan.graph_path, graph_file_->node_offset(0), node_disk_bytes_,
                       VecOffsetInRecord(), EdgeOffsetInRecord(plan.start_layer - 1),
                       ClusterScratch()->rank_cids_.data(), ClusterScratch()->np);
    fprintf(stderr, "[ClusterInputIO] distribution seconds=%.6f\n",
            std::chrono::duration<double>(std::chrono::steady_clock::now() - distribution_start)
                .count());
    plan.scratch.Sample();
    disk_build::CompleteMemoryPhase("Phase2 input distribute", p.base_bytes + p.prep_bytes,
                                    build_config_.build_mem_gb_);
  }
  std::vector<uint32_t> local(plan.n, UINT32_MAX);
  // Sequential clusters keep only one compact vector/adjacency workspace live.
  for (size_t c = 0; c < ClusterScratch()->K; ++c) {
    const size_t begin = ClusterScratch()->cluster_rank_offsets_[c],
                 n = ClusterScratch()->cluster_rank_offsets_[c + 1] - begin;
    if (!n)
      continue;
    const auto t0 = std::chrono::steady_clock::now();
    auto seconds_since = [](const auto &t) {
      return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
    };
    const auto *ranks = ClusterScratch()->cluster_ranks_.data() + begin;
    for (size_t u = 0; u < n; ++u)
      local[ranks[u]] = uint32_t(u);
    std::vector<vec_t> vectors(n * vec_d_);
    double load_seconds = 0;
    std::vector<size_t> windows{window_size_[plan.start_layer - 1] / 2};
    windows.insert(windows.end(), plan.half_window.begin(), plan.half_window.end());
    auto distance = [&](uint32_t u, uint32_t v) {
      if (u >= n || v >= n)
        throw std::logic_error("cluster-local distance received an invalid slot");
      return fstdistfunc_(vectors.data() + size_t(u) * vec_d_, vectors.data() + size_t(v) * vec_d_,
                          dist_func_param_);
    };
    using Graph = disk_build::ClusterWowGraph<decltype(distance)>;
    std::vector<size_t> caps(windows.size(), M_);
    for (size_t l = 1; l < caps.size(); ++l)
      caps[l] = 2 * M_ / 3;
    auto graph = std::make_unique<Graph>(ranks, n, M_, windows, distance, alpha_, false, caps);
    // Load vectors and lower-layer seeds together before evaluating any distance.
    double seed_seconds = 0;
    {
      const auto input_t0 = std::chrono::steady_clock::now();
      {
        LoadClusterVectorsAndSeeds(*inputs, c, n, vec_d_, M_, plan.n, ranks, local,
                                   position_by_id_.data(), vectors, *graph);
      }
      load_seconds = seconds_since(input_t0);
      const auto seed_t0 = std::chrono::steady_clock::now();
      FinalizeClusterSeeds(*graph, n, build_config_.merge_build_threads_);
      seed_seconds = seconds_since(seed_t0);
    }
    const auto wow_t0 = std::chrono::steady_clock::now();
    graph->Build(build_config_.cluster_wow_efc_, build_config_.cluster_wow_passes_,
                 build_config_.merge_build_threads_);
    const double wow_seconds = seconds_since(wow_t0);
    const auto spill_t0 = std::chrono::steady_clock::now();
    WriteClusterSpill(plan, c, n, M_, ranks, sorted_ids_.data(), *graph);
    plan.scratch.SampleSpill(c);
    const double spill_seconds = seconds_since(spill_t0);
    for (size_t u = 0; u < n; ++u)
      local[ranks[u]] = UINT32_MAX;
    fprintf(stderr, "[wow_stage_times] cluster=%zu vectors=%.6f seed=%.6f refine=%.6f spill=%.6f\n",
            c, load_seconds, seed_seconds, wow_seconds, spill_seconds);
    fprintf(stderr,
            "[ClusterWoW] cluster=%zu n=%zu layers=%zu efc=%zu passes=%zu "
            "distances=%zu expansions=%zu seconds=%.3f\n",
            c, n, plan.L, build_config_.cluster_wow_efc_, build_config_.cluster_wow_passes_,
            graph->distances, graph->expansions,
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    disk_build::CompleteMemoryPhase("Phase2 cluster WoW", p.base_bytes + p.pass_a_bytes,
                                    build_config_.build_mem_gb_);
  }
  inputs->ReportComplete();
  inputs.reset();
  // Every cluster reader is closed; discard the single sequential input file now.
  plan.scratch.ReleaseInput();
  fprintf(stderr, "[ClusterWoW] pass A done: %.3fs\n",
          std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
}
template void
AllianceIndexDisk<int, float>::BuildClusterWoWEdgesDisk(const disk_build::ClusterSpillPlan &);
} // namespace alliance
