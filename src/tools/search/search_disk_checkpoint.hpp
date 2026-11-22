#pragma once
#include "search/search_config.hpp"
#include "index/index_disk.hpp"
#include <bit>
#include <fstream>
#include <filesystem>
#include <unordered_set>

namespace benchmark {
using CheckpointIndex = alliance::AllianceIndexDisk<int, float>;
inline uint64_t QuantizationFingerprint(const CheckpointIndex &index) {
  uint64_t hash = 14695981039346656037ULL;
  auto add = [&](const void *ptr, size_t n) {
    auto *bytes = static_cast<const uint8_t *>(ptr);
    for (size_t i = 0; i < n; ++i) {
      hash ^= bytes[i];
      hash *= 1099511628211ULL;
    }
  };
  std::vector<uint8_t> canonical(index.quant_vec_bytes_);
  for (size_t i = 0; i < index.curvec_num_; ++i) {
    index.CopyCanonicalRaBitQCode(i, canonical.data());
    add(canonical.data(), canonical.size());
  }
  for (size_t i = 0; i < index.curvec_num_; ++i)
    add(index.RaBitQFactors(i), 16);
  add(index.centroids_, index.maxC_ * index.padded_dim_ * sizeof(float));
  add(index.cids_.data(), index.cids_.size());
  return hash;
}

// Read the existing frozen benchmark checkpoint without changing the index files.
inline uint64_t LoadSearchCheckpoint(CheckpointIndex &index, const DiskSearchConfig &options) {
  if (options.rabitq_centroids.empty() || options.rabitq_cids.empty())
    throw std::invalid_argument("quant_checkpoint requires centroid and CID paths");
  index.InitRaBitQ(options.rabitq_centroids, options.rabitq_cids, options.space,
                   options.rabitq_bits);
  std::ifstream in(options.quant_checkpoint, std::ios::binary);
  uint64_t n = 0, code_bytes = 0, factors = 0, hash = 0;
  double t = 0;
  in.read(reinterpret_cast<char *>(&n), 8);
  in.read(reinterpret_cast<char *>(&code_bytes), 8);
  in.read(reinterpret_cast<char *>(&factors), 8);
  in.read(reinterpret_cast<char *>(&t), 8);
  in.read(reinterpret_cast<char *>(&hash), 8);
  if (!in || n != index.curvec_num_ || code_bytes != index.quant_vecs_.size() ||
      factors != index.factors_.size() || !std::isfinite(t) ||
      (options.rabitq_bits == 1 ? (t != -1.0 || index.rabitq_config_.t_const != -1.0) : t <= 0) ||
      std::filesystem::file_size(options.quant_checkpoint) !=
          40 + code_bytes + factors * sizeof(float))
    throw std::runtime_error("quant_checkpoint shape/header mismatch");
  in.read(reinterpret_cast<char *>(index.quant_vecs_.data()), code_bytes);
  in.read(reinterpret_cast<char *>(index.factors_.data()), factors * sizeof(float));
  index.rabitq_config_.t_const = t;
  if (!in || QuantizationFingerprint(index) != hash)
    throw std::runtime_error("quant_checkpoint fingerprint mismatch");
  return hash;
}

struct CheckpointProbe {
  uint32_t score, error;
  double tolerance;
};
inline std::vector<CheckpointProbe> CheckpointScoreProbe(CheckpointIndex &index, bool prepared) {
  std::vector<float> query(index.vec_d_);
  std::vector<CheckpointProbe> probe;
  std::vector<uint8_t> canonical(index.quant_vec_bytes_);
  for (size_t q = 0; q < 4; ++q) {
    for (size_t d = 0; d < query.size(); ++d)
      query[d] = static_cast<float>(int((d * 17 + q * 31) % 101) - 50) / 51;
    CheckpointIndex::RaBitQEstimator estimator(query.data(), &index);
    for (size_t i = 0; i < std::min(size_t(256), index.curvec_num_); ++i) {
      const size_t id = (i * 2654435761ULL + q * 104729) % index.curvec_num_;
      const auto cid = prepared ? index.SearchCID(id) : index.cids_[index.labels_[id]];
      const auto [value, error] = estimator.dist_estimate_with_errorbound(cid, id);
      index.CopyCanonicalRaBitQCode(id, canonical.data());
      double magnitude = 0;
      for (size_t d = 0; d < index.padded_dim_; ++d) {
        unsigned code = 0;
        for (size_t p = 0; p < index.rabitq_bits_; ++p)
          code |= ((canonical[p * index.padded_dim_ / 8 + d / 8] >> (d % 8)) & 1u) << p;
        magnitude += code * std::abs(double(estimator.rotated_query_[d]));
      }
      const float *f = index.RaBitQFactors(id);
      const double cb = -0.5 * ((1u << index.rabitq_bits_) - 1);
      const double scale =
          std::abs(double(f[0])) + std::abs(double(estimator.dist_to_centroids_[cid])) +
          std::abs(double(f[1])) * (magnitude + std::abs(cb * estimator.float_query_sum_)) + 1;
      // Allow both AVX-512 reduction orders, including the final float cast.
      // Independent code/factor fingerprints remain byte-exact.
      const double tolerance =
          4 * std::numeric_limits<float>::epsilon() * (index.padded_dim_ / 16 + 16) * scale;
      probe.push_back({std::bit_cast<uint32_t>(value), std::bit_cast<uint32_t>(error), tolerance});
    }
  }
  return probe;
}

inline void ValidateCheckpointProbes(const std::vector<CheckpointProbe> &before,
                                     const std::vector<CheckpointProbe> &after) {
  if (before.size() != after.size())
    throw std::runtime_error("checkpoint probe count changed");
  bool bitwise = true;
  double max_delta = 0;
  for (size_t i = 0; i < before.size(); ++i) {
    const auto &a = before[i], &b = after[i];
    if (a.error != b.error)
      throw std::runtime_error("packed quantization changed error factor");
    const double x = std::bit_cast<float>(a.score), y = std::bit_cast<float>(b.score);
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(a.tolerance) ||
        !std::isfinite(b.tolerance))
      throw std::runtime_error("non-finite packed quantization score probe");
    if (a.score == b.score)
      continue;
    bitwise = false;
    const double delta = std::abs(x - y), limit = std::max(a.tolerance, b.tolerance);
    if (delta > limit)
      throw std::runtime_error("packed quantization score exceeds floating-point tolerance");
    max_delta = std::max(max_delta, delta);
  }
  std::cout
      << "# packed_validation: code_factor_bitwise_equal=1 score_tolerance_pass=1 score_bitwise_equal="
      << bitwise << " max_abs_delta=" << max_delta << " samples=" << before.size() << std::endl;
}

inline std::vector<int> DenseCheckpointAttributes(const CheckpointIndex &index) {
  std::vector<int> attributes(index.curvec_num_);
  std::vector<bool> seen(index.curvec_num_);
  for (size_t id = 0; id < index.curvec_num_; ++id) {
    const auto label = index.labels_[id];
    if (label >= attributes.size() || seen[label])
      throw std::runtime_error("frozen benchmark validation requires dense unique labels");
    seen[label] = true;
    attributes[label] = index.atts_[id];
  }
  return attributes;
}
} // namespace benchmark
