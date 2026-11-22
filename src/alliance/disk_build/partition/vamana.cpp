#include "index/index_disk.hpp"
#include "io/aio_stream.hpp"
#include "disk_build/memory/index_adapter.hpp"
#include "io/disk_resource_guard.hpp"
#include "disk_build/partition/kmeans.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
#include <omp.h>

namespace alliance {
namespace {

// These helpers keep partition policy in the caller: they neither choose K nor
// publish build state. Training uses the same OpenMP setting as the graph build.
void TrainPartitionCenters(float *training, size_t count, size_t dimension, float *centers,
                           size_t K, int threads) {
  const int previous_threads = omp_get_max_threads();
  omp_set_num_threads(threads);
  detail::ScopeExit restore_threads(
      [previous_threads]() noexcept { omp_set_num_threads(previous_threads); });
  disk_build::vamana_kmeans::kmeanspp_selecting_pivots(training, count, dimension, centers, K);
  disk_build::vamana_kmeans::run_elkan(training, count, dimension, centers, K, 20);
}

size_t EstimateMaximumMembership(const std::vector<float> &sample, size_t count, size_t dimension,
                                 const std::vector<float> &centers, size_t K, size_t n, int threads,
                                 std::vector<size_t> &counts) {
#pragma omp parallel num_threads(threads)
  {
    std::vector<size_t> local(K, 0);
#pragma omp for schedule(static)
    for (size_t i = 0; i < count; ++i) {
      float best[2] = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
      uint32_t which[2] = {0, 0};
      for (size_t c = 0; c < K; ++c) {
        float d = 0;
        for (size_t j = 0; j < dimension; ++j) {
          const float v = sample[i * dimension + j] - centers[c * dimension + j];
          d += v * v;
        }
        if (d < best[1]) {
          if (d < best[0]) {
            best[1] = best[0];
            which[1] = which[0];
            best[0] = d;
            which[0] = c;
          } else {
            best[1] = d;
            which[1] = c;
          }
        }
      }
      ++local[which[0]];
      ++local[which[1]];
    }
#pragma omp critical(alliance_vamana_counts)
    for (size_t c = 0; c < K; ++c)
      counts[c] += local[c];
  }
  return std::min(n, size_t(std::ceil(*std::max_element(counts.begin(), counts.end()) / .01)));
}

void WritePartitionArtifacts(const std::string &graph_path, size_t dimension, size_t K, size_t n,
                             const std::vector<float> &centers, const std::vector<uint32_t> &ids) {
  const auto prefix = graph_path + ".partition";
  std::ofstream cf(prefix + ".centroids.fvecs", std::ios::binary);
  std::ofstream mf(prefix + ".memberships.ivecs", std::ios::binary);
  const uint32_t dim = dimension, two = 2;
  for (size_t c = 0; c < K; ++c) {
    cf.write(reinterpret_cast<const char *>(&dim), 4);
    cf.write(reinterpret_cast<const char *>(centers.data() + c * dimension), dimension * 4);
  }
  for (size_t i = 0; i < n; ++i) {
    mf.write(reinterpret_cast<const char *>(&two), 4);
    mf.write(reinterpret_cast<const char *>(ids.data() + 2 * i), 8);
  }
  if (!cf || !mf)
    throw std::runtime_error("cannot save Vamana partition artifacts");
}

} // namespace

// PipeANN/Vamana partition policy: 1% random sample, kmeans++/Elkan,
// increasing K, and two nearest L2 centroids with no capacity reassignment.
// The budget test intentionally uses AllianceANN's complete multi-layer RSS
// model, not PipeANN's single-layer shard estimate or a half-budget heuristic.
template <typename att_t, typename vec_t>
void AllianceIndexDisk<att_t, vec_t>::SynthesizeVamanaClustersDisk(size_t &K, size_t n,
                                                                   std::vector<float> &centers,
                                                                   std::vector<uint32_t> &ids) {
  if (n < 2)
    throw std::invalid_argument("Vamana partition requires n>=2");
  const bool pinned = K != 0;
  if (K == 1 || K > n)
    throw std::invalid_argument("invalid Vamana K");
  if (!K)
    K = std::min(n, size_t(3));
  const size_t requested_K = K;
  std::mt19937 random(std::random_device{}());
  std::bernoulli_distribution sample(.01);
  auto sample_ranks = [&](size_t minimum) {
    std::vector<uint32_t> ranks;
    ranks.reserve(std::min(n, std::max(minimum, size_t(std::ceil(n * .011)) + 256)));
    for (size_t i = 0; i < n; ++i)
      if (sample(random))
        ranks.push_back(uint32_t(i));
    // Tiny inputs still need enough training points for kmeans++.
    for (size_t i = 0; ranks.size() < minimum && i < n; ++i) {
      auto at = std::lower_bound(ranks.begin(), ranks.end(), uint32_t(i));
      if (at == ranks.end() || *at != i)
        ranks.insert(at, uint32_t(i));
    }
    return ranks;
  };
  const auto train_ranks = sample_ranks(K);
  const size_t sn = train_ranks.size();
  auto require_room = [&](size_t bytes, const char *stage) {
    if (build_config_.build_mem_gb_ <= 0)
      return;
    size_t current = 0;
    ProcRss(&current, nullptr);
    const size_t budget = size_t(build_config_.build_mem_gb_ * (1ull << 30));
    if (current > budget || bytes > budget - current)
      throw std::runtime_error(std::string("Vamana ") + stage +
                               " exceeds whole-process memory budget");
  };
  require_room(sn * vec_d_ * sizeof(float), "training sample");
  std::vector<float> train(sn * vec_d_);
  {
    IOHandler io(64);
    io.Open(build_vec_file_path_);
    GatherVecsByRank(io, train_ranks.data(), nullptr, sn, train.data());
  }
  const double budget = build_config_.build_mem_gb_ * (1ull << 30);
  auto fits = [&](size_t maximum) {
    // A candidate budget query must not publish a tentative partition. Skew
    // sizes the memory model; it never moves memberships to enforce capacity.
    const float balance = float(double(maximum) * K / (2. * n) + .001);
    const auto plan =
        disk_build::MemoryPlanner<att_t, vec_t>::FromIndex(*this).PlanPartition(n, K, balance);
    return build_config_.build_mem_gb_ <= 0 ||
           (!plan.base_over_budget && !plan.user_over_budget &&
            double(plan.base_bytes) + double(plan.residency_bytes) <= budget);
  };
  for (;;) {
    if (K > sn)
      throw std::runtime_error(
          "Vamana K exceeds random training sample; increase memory or supply external partition");
    require_room((sn * K + 2 * K * K + 3 * K * vec_d_) * sizeof(float) + sn * 64,
                 "Elkan workspace");
    centers.assign(K * vec_d_, 0);
    TrainPartitionCenters(train.data(), sn, vec_d_, centers.data(), K,
                          build_config_.merge_build_threads_);

    // An independent sample screens candidates before expensive full assignment.
    const auto test_ranks = sample_ranks(std::min(n, size_t(3)));
    require_room(test_ranks.size() * vec_d_ * sizeof(float), "size estimation sample");
    std::vector<float> test(test_ranks.size() * vec_d_);
    {
      IOHandler io(64);
      io.Open(build_vec_file_path_);
      GatherVecsByRank(io, test_ranks.data(), nullptr, test_ranks.size(), test.data());
    }
    std::vector<size_t> counts(K, 0);
    const size_t estimate = EstimateMaximumMembership(
        test, test_ranks.size(), vec_d_, centers, K, n, build_config_.merge_build_threads_, counts);
    bool fit = fits(estimate);
    fprintf(
        stderr,
        "[VamanaPartition] K=%zu sample=%zu estimated_max=%zu whole_rss_fit=%d metric=l2 np=2\n", K,
        sn, estimate, int(fit));
    test.clear();
    test.shrink_to_fit();
    if (fit) {
      AssignNearestTwoClustersDisk(centers.data(), K, n, ids);
      std::fill(counts.begin(), counts.end(), 0);
      for (auto c : ids)
        ++counts[c];
      const size_t actual = *std::max_element(counts.begin(), counts.end());
      fit = fits(actual);
      fprintf(stderr, "[VamanaPartition] K=%zu actual_max=%zu slots=%zu whole_rss_fit=%d\n", K,
              actual, ids.size(), int(fit));
      if (fit) {
        // Publish only the accepted partition's statistics. Checkpoint fields
        // are committed later by the upper-stage coordinator.
        build_config_.cluster_balance_ = float(double(actual) * K / (2. * n) + .001);
        cluster_K_file_ = K;
        break;
      }
      ids.clear();
      ids.shrink_to_fit();
    }
    if (pinned)
      throw std::runtime_error("requested Vamana K exceeds whole-process memory budget");
    ++K;
  }
  fprintf(
      stderr,
      "[VamanaPartition] chosen_K=%zu requested_K=%zu memberships=%zu sampling=0.01 init=kmeans++ solver=Elkan iterations=20 capacity_reassignment=0\n",
      K, pinned ? requested_K : 0, ids.size());
  WritePartitionArtifacts(graph_file_->path(), vec_d_, K, n, centers, ids);
}
template void AllianceIndexDisk<int, float>::SynthesizeVamanaClustersDisk(size_t &, size_t,
                                                                          std::vector<float> &,
                                                                          std::vector<uint32_t> &);
} // namespace alliance
