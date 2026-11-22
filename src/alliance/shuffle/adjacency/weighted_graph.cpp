#include "shuffle/adjacency/weighted_graph.hpp"
#include "index/index_disk.hpp"
#include "shuffle/level_geometry.hpp"
#include "shuffle/memory/storage_plan.hpp"
#include "shuffle/memory/runtime.hpp"
#include <algorithm>
#include <numeric>
#include <cstdio>
#include <stdexcept>
#include <chrono>
#include <atomic>
#include <exception>
#include <mutex>
#include <omp.h>

namespace alliance::shuffle_detail {

template <typename att_t, typename vec_t>
auto BuildWeightedGraph(AllianceIndexDisk<att_t, vec_t> *idx, layer_t layer_begin,
                        layer_t layer_end, size_t stride, int threads, const BNFParams &params,
                        size_t elements_per_block) -> FlatWeightedGraph {
  if (threads < 1)
    throw std::invalid_argument("CSR threads must be positive");
  const auto now = [] {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
  };
  const double start = now();
  const size_t n = idx->curvec_num_;
  bool external = params.storage == ShuffleStorage::External;
  if (params.memory_limit_bytes &&
      ShuffleResidentBytes() + 8.L * n + ShuffleReserve(params.memory_limit_bytes) >
          params.memory_limit_bytes)
    throw std::invalid_argument("shuffle count-pass working set exceeds memory budget");
  const size_t num_layers = CountSelectedLayers(layer_begin, layer_end, stride);
  const size_t max_edges = num_layers * static_cast<size_t>(idx->M_);
  constexpr size_t chunk = 256;
  const size_t slots = num_layers * (idx->M_ + 1);
  FlatWeightedGraph graph;
  graph.num_nodes = n;
  graph.offsets.resize(n + 1, 0);
  struct Scratch {
    std::vector<tableint> raw, edges;
    std::vector<AioSlice> reads;
    std::vector<uint8_t> encoded;
    uint8_t maximum_weight{1};
    size_t maximum_degree{}, maximum_score{};
  };
  std::vector<Scratch> scratch(threads);
  for (auto &w : scratch) {
    w.raw.resize(chunk * slots);
    w.edges.resize(max_edges);
    w.reads.reserve(chunk * num_layers);
  }
  auto pass = [&](bool fill) {
    std::exception_ptr error;
    std::mutex error_mutex;
    std::atomic<bool> failed{false};
    const size_t chunks = (n + chunk - 1) / chunk;
#pragma omp parallel for num_threads(threads) schedule(static) if (threads > 1)
    for (size_t c = 0; c < chunks; ++c) {
      if (failed.load(std::memory_order_relaxed))
        continue;
      try {
        auto &w = scratch[omp_get_thread_num()];
        const size_t first = c * chunk, count = std::min(chunk, n - first);
        if (fill && external)
          w.encoded.resize((graph.offsets[first + count] - graph.offsets[first]) *
                           graph.external->Codec().Bytes());
        w.reads.clear();
        for (size_t i = 0; i < count; ++i) {
          const uint64_t off =
              idx->graph_file_->node_offset(idx->DiskSlot(static_cast<tableint>(first + i)));
          size_t li = 0;
          for (layer_t layer = layer_begin; layer <= layer_end;
               layer += static_cast<layer_t>(stride), ++li)
            w.reads.push_back({w.raw.data() + i * slots + li * (idx->M_ + 1), idx->EdgeIdBytes(),
                               off + idx->EdgeOffsetInRecord(layer)});
        }
        // Every worker owns a TLS AIO context and private bounded input buffers.
        idx->graph_file_->ReadSlices(w.reads.data(), w.reads.size());
        for (size_t i = 0; i < count; ++i) {
          const size_t u = first + i;
          size_t count_edges = 0;
          for (size_t li = 0; li < num_layers; ++li) {
            const tableint *buf = w.raw.data() + i * slots + li * (idx->M_ + 1);
            if (buf[idx->M_] > idx->M_)
              throw std::runtime_error("invalid graph degree during shuffle CSR build");
            for (size_t j = 0; j < buf[idx->M_]; ++j)
              if (buf[j] < n)
                w.edges[count_edges++] = buf[j];
          }
          std::sort(w.edges.begin(), w.edges.begin() + count_edges);
          size_t unique = 0;
          size_t score = 0;
          for (size_t p = 0; p < count_edges;) {
            size_t q = p + 1;
            while (q < count_edges && w.edges[q] == w.edges[p])
              ++q;
            const auto weight = static_cast<uint8_t>(std::min(q - p, size_t(255)));
            if (fill) {
              const size_t pos = graph.offsets[u] + unique;
              if (external) {
                graph.external->Codec().Encode(w.encoded.data() +
                                                   (pos - graph.offsets[first]) *
                                                       graph.external->Codec().Bytes(),
                                               w.edges[p], weight);
#pragma omp atomic update
                graph.in_weights[w.edges[p]] += weight;
              } else
                graph.edges.Set(pos, w.edges[p], weight);
            } else
              w.maximum_weight = std::max(w.maximum_weight, weight);
            ++unique;
            score += size_t(weight) * weight;
            p = q;
          }
          if (!fill) {
            graph.offsets[u + 1] = unique;
            w.maximum_degree = std::max(w.maximum_degree, unique);
            w.maximum_score = std::max(w.maximum_score, score);
          }
        }
        if (fill && external)
          graph.external->Write(graph.offsets[first], w.encoded.data(), w.encoded.size());
      } catch (...) {
        failed.store(true, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(error_mutex);
        if (!error)
          error = std::current_exception();
      }
    }
    if (error)
      std::rethrow_exception(error);
  };
  pass(false);
  const double count_end = now();
  for (size_t u = 0; u < n; ++u)
    graph.offsets[u + 1] += graph.offsets[u];
  uint8_t maximum_weight = 1;
  size_t maximum_degree = 0, maximum_score = 0;
  for (const auto &worker : scratch) {
    maximum_weight = std::max(maximum_weight, worker.maximum_weight);
    maximum_degree = std::max(maximum_degree, worker.maximum_degree);
    maximum_score = std::max(maximum_score, worker.maximum_score);
  }
  const EdgeCodec codec(n, maximum_weight);
  const auto plan = PlanShuffleStorage(
      {n, graph.offsets[n], codec.Bytes(), maximum_degree, maximum_score, elements_per_block},
      params, ShuffleResidentBytes());
  external = plan.selected == ShuffleStorage::External;
  std::fprintf(
      stderr,
      "  [shuffle-storage-plan] requested=%s selected=%s limit_bytes=%zu resident_bytes=%zu memory_peak_bytes=%zu external_peak_bytes=%zu reserve_bytes=%zu edges=%zu edge_bytes=%zu\n",
      ShuffleStorageName(params.storage), ShuffleStorageName(plan.selected), plan.limit,
      plan.resident_before, plan.memory_peak, plan.external_peak, plan.reserve, graph.offsets[n],
      codec.Bytes());
  if (external) {
    graph.external = std::make_shared<ExternalGraph>(params.scratch_directory, n, maximum_weight,
                                                     graph.offsets[n], plan.external_buffers);
  } else
    graph.edges.Allocate(n, maximum_weight, graph.offsets[n]);
  graph.in_weights.resize(n, 0);
  const double alloc_end = now();
  pass(true);
  const double fill_end = now();
  // Input AIO scratch is no longer needed by the graph-only accumulation pass.
  std::vector<Scratch>().swap(scratch);
  // Integer atomics preserve exact sums without O(N * threads) replicas.
  if (!external) {
#pragma omp parallel for num_threads(threads) schedule(static) if (threads > 1)
    for (size_t u = 0; u < n; ++u) {
      for (size_t e = graph.offsets[u]; e < graph.offsets[u + 1]; ++e) {
        const size_t v = graph.edges.Neighbor(e), value = graph.edges.Weight(e);
#pragma omp atomic update
        graph.in_weights[v] += value;
      }
    }
  }
  fprintf(stderr, "  [shuffle-csr-storage] storage=%s packed=%d max_weight=%u edge_bytes=%zu\n",
          external ? "external" : "memory",
          int(external ? graph.external->Codec().Packed() : graph.edges.Packed()),
          unsigned(maximum_weight),
          external ? graph.offsets[n] * graph.external->Codec().Bytes()
                   : graph.edges.StorageBytes());
  fprintf(
      stderr,
      "  [shuffle-csr-timing] threads=%d count_s=%.6f prefix_alloc_s=%.6f fill_s=%.6f in_weights_s=%.6f total_s=%.6f\n",
      threads, count_end - start, alloc_end - count_end, fill_end - alloc_end, now() - fill_end,
      now() - start);
  return graph;
}

auto SortNodesByInWeight(const std::vector<size_t> &in_weights) -> std::vector<size_t> {
  const size_t n = in_weights.size();
  std::vector<size_t> sorted(n);
  std::iota(sorted.begin(), sorted.end(), size_t(0));
  std::sort(sorted.begin(), sorted.end(), [&](size_t a, size_t b) {
    if (in_weights[a] != in_weights[b])
      return in_weights[a] > in_weights[b];
    return a < b;
  });
  return sorted;
}

template auto BuildWeightedGraph<int, float>(AllianceIndexDisk<int, float> *, layer_t, layer_t,
                                             size_t, int, const BNFParams &, size_t)
    -> FlatWeightedGraph;

} // namespace alliance::shuffle_detail
