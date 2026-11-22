#include "index/index_disk.hpp"
#include "io/aio_stream.hpp"
#include <algorithm>
#include <limits>
#include <vector>
#include <omp.h>

namespace alliance {

// Exact nearest-two L2 memberships, in centroid-distance order by original label.
// Capacity balancing would change the partition, so it is not applied.
template <typename att_t, typename vec_t>
void AllianceIndexDisk<att_t, vec_t>::AssignNearestTwoClustersDisk(
    const float *centroids, size_t K, size_t n, std::vector<uint32_t> &OUT_cluster_ids) {
  constexpr size_t np = 2;
  if (K < np)
    throw std::invalid_argument("nearest-two assignment requires K>=2");
  OUT_cluster_ids.assign(n * np, 0);
  const size_t kChunk = 256 * std::max(vecs_per_page_, size_t(1));
  const size_t nchunks = (n + kChunk - 1) / kChunk;
#pragma omp parallel num_threads(build_config_.merge_build_threads_)
  {
    IOHandler vio(64);
    vio.Open(build_vec_file_path_);
    std::vector<uint32_t> cranks(kChunk);
    std::vector<vec_t> cvecs(kChunk * vec_d_);
#pragma omp for schedule(static)
    for (size_t ci = 0; ci < nchunks; ++ci) {
      const size_t rbegin = ci * kChunk;
      const size_t rcount = std::min(kChunk, n - rbegin);
      for (size_t k = 0; k < rcount; ++k)
        cranks[k] = uint32_t(rbegin + k);
      GatherVecsByRank(vio, cranks.data(), nullptr, rcount, cvecs.data());
      for (size_t k = 0; k < rcount; ++k) {
        const size_t rank = rbegin + k;
        const vec_t *vec = cvecs.data() + k * vec_d_;
        float bd[np] = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
        uint32_t bc[np] = {0, 0};
        for (size_t c = 0; c < K; ++c) {
          float d = 0;
          for (size_t j = 0; j < vec_d_; ++j) {
            const float v = vec[j] - centroids[c * vec_d_ + j];
            d += v * v;
          }
          if (d >= bd[np - 1])
            continue;
          size_t p = np - 1;
          while (p > 0 && bd[p - 1] > d) {
            bd[p] = bd[p - 1];
            bc[p] = bc[p - 1];
            --p;
          }
          bd[p] = d;
          bc[p] = uint32_t(c);
        }
        const label_t label = labels_[sorted_ids_[rank]];
        for (size_t a = 0; a < np; ++a)
          OUT_cluster_ids[size_t(label) * np + a] = bc[a];
      }
    }
  }
}

template void AllianceIndexDisk<int, float>::AssignNearestTwoClustersDisk(const float *, size_t,
                                                                          size_t,
                                                                          std::vector<uint32_t> &);

} // namespace alliance
