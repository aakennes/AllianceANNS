#include "index/index_disk.hpp"
#include "quantization/float_query_dot.hpp"
#include "io/aio_stream.hpp"
#include "io/io.hpp"
#include "rabitq/utils/buffer.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <omp.h>

namespace alliance {

// ============================================================================
// InitRaBitQ
// ============================================================================

template <typename att_t, typename vec_t>
void AllianceIndexDisk<att_t, vec_t>::InitRaBitQ(const std::string &centroids_path,
                                                 const std::string &cids_path,
                                                 const std::string &space_name, size_t total_bits) {
  // Navigation was matched to the previous metric and must be reloaded explicitly.
  ClearNavigation();
  guided_entries_.Clear();
  packed_rabitq_.Clear();
  search_cids_.clear();
  rabitq_space_name_ = space_name;
  rabitq_bits_ = total_bits;
  size_t dim_c = 0, n_c = 0;
  float *raw_centroids = fvecs_read(centroids_path, dim_c, n_c);
  if (dim_c != vec_d_) {
    delete[] raw_centroids;
    throw std::runtime_error("InitRaBitQ: centroid dimension mismatch");
  }
  maxC_ = n_c;

  size_t dim_cid = 0, n_cid = 0;
  int *raw_cids = ivecs_read(cids_path, dim_cid, n_cid);
  if (dim_cid != 1 || n_cid < curvec_num_) {
    delete[] raw_centroids;
    delete[] raw_cids;
    throw std::runtime_error("InitRaBitQ: cids dimension/count mismatch (need >= " +
                             std::to_string(curvec_num_) + ", got " + std::to_string(n_cid) + ")");
  }
  cids_.resize(n_cid);
  for (size_t i = 0; i < n_cid; ++i) {
    cids_[i] = static_cast<uint8_t>(raw_cids[i]);
  }
  delete[] raw_cids;

  rotator_ = rabitqlib::choose_rotator<float>(vec_d_);
  padded_dim_ = rotator_->size();
  quant_vec_bytes_ = padded_dim_ * rabitq_bits_ / 8;

  centroids_ = new float[padded_dim_ * maxC_];
  for (size_t i = 0; i < maxC_; ++i) {
    rotator_->rotate(raw_centroids + i * vec_d_, centroids_ + i * padded_dim_);
  }
  delete[] raw_centroids;

  rabitq_config_ = rabitqlib::quant::faster_config(padded_dim_, rabitq_bits_);

  quant_vecs_.resize(curvec_num_ * quant_vec_bytes_, 0);
  factors_.resize(curvec_num_ * 4, 0.0f);

  if (space_name == "l2") {
    rabitq_space_ = new alliance::L2Space(padded_dim_);
  } else if (space_name == "ip") {
    rabitq_space_ = new alliance::InnerProductSpace(padded_dim_);
  } else {
    throw std::runtime_error("InitRaBitQ: unsupported space " + space_name);
  }
  rabitq_distfunc_ = rabitq_space_->get_dist_func();
  rabitq_dist_func_param_ = rabitq_space_->get_dist_func_param();

  rabitq_initialized_ = true;
  std::cout << "InitRaBitQ: maxC=" << maxC_ << " padded_dim=" << padded_dim_
            << " rabitq_bits=" << rabitq_bits_ << " quant_vec_bytes=" << quant_vec_bytes_ << "\n";
}

// ============================================================================
// PackMultiBitCode — bit-plane packing shared by build + requantize paths
// ============================================================================

template <typename att_t, typename vec_t>
void AllianceIndexDisk<att_t, vec_t>::PackMultiBitCode(const uint8_t *code, uint8_t *packed,
                                                       float &pop_count_out) const {
  const size_t plane_bytes = padded_dim_ / 8;
  if (rabitq_bits_ == 1)
    rabitqlib::quant::rabitq_impl::ex_bits::packing_1bit_excode(code, packed, padded_dim_);
  else
    for (size_t p = 0; p < rabitq_bits_; ++p) {
      uint8_t *plane = packed + p * plane_bytes;
      for (size_t b = 0; b < plane_bytes; ++b) {
        uint8_t byte = 0;
        for (int bit = 0; bit < 8; ++bit) {
          byte |= static_cast<uint8_t>((code[b * 8 + bit] >> p) & 1) << bit;
        }
        plane[b] = byte;
      }
    }
  // Dequant popcount term is sum_d code[d] = sum_p (popcount(plane_p) << p);
  // for rabitq_bits_==1 this is exactly the set-bit count (prior behavior).
  float pc = 0;
  for (size_t d = 0; d < padded_dim_; ++d)
    pc += code[d];
  pop_count_out = pc;
}

// ============================================================================
// RequantizeRaBitQ — swap the search-side quantization centroid set on a
// loaded graph (vectors are read back from the graph records; the graph file
// itself is untouched, codes/factors live in RAM only until the next Save).
// ============================================================================

template <typename att_t, typename vec_t>
void AllianceIndexDisk<att_t, vec_t>::RequantizeRaBitQ(const std::string &centroids_path,
                                                       const std::string &cids_path,
                                                       const std::string &space_name,
                                                       size_t total_bits) {
  RequireRawGraph("RequantizeRaBitQ");
  if (rabitq_initialized_) {
    delete[] centroids_;
    centroids_ = nullptr;
    delete rotator_;
    rotator_ = nullptr;
    delete rabitq_space_;
    rabitq_space_ = nullptr;
    rabitq_initialized_ = false;
  }
  InitRaBitQ(centroids_path, cids_path, space_name, total_bits);

  rabitqlib::MetricType metric = rabitqlib::METRIC_L2;
  if (rabitq_space_name_ == "ip") {
    metric = rabitqlib::METRIC_IP;
  }

  const size_t n = curvec_num_;
  auto rq_start = std::chrono::high_resolution_clock::now();

  // Walk rank order, not id order: ReadVec(id) lands at DiskSlot(id), so the
  // old id-ascending loop was a random sweep of the record array served by one
  // synchronous pread per vector. Rank ascending is the same set of records in
  // file order, which one AioSeqStreamReader delivers in 4 MB O_DIRECT chunks.
  // The outputs stay id-indexed (scattered writes into RAM, which is free).
  graph_file_->Sync();
  const size_t vec_bytes = VecBytes();
  const size_t rec_pre = VecOffsetInRecord();
  const size_t rec_post = node_disk_bytes_ - rec_pre - vec_bytes;
  const size_t rq_batch = std::max<size_t>(1, (size_t)(kAioStreamBytes / node_disk_bytes_));
  std::vector<char> stage(rq_batch * vec_bytes);
  {
    AioSeqStreamReader reader(graph_file_->path(), graph_file_->node_offset(0),
                              graph_file_->node_offset(static_cast<tableint>(n)));
    for (size_t beg = 0; beg < n; beg += rq_batch) {
      const size_t cnt = std::min(rq_batch, n - beg);
      for (size_t k = 0; k < cnt; ++k) {
        reader.Skip(rec_pre);
        reader.Read(stage.data() + k * vec_bytes, vec_bytes);
        reader.Skip(rec_post);
      }

#pragma omp parallel for schedule(static)
      for (size_t k = 0; k < cnt; ++k) {
        const size_t i = sorted_ids_[beg + k];
        const vec_t *raw_vec = reinterpret_cast<const vec_t *>(stage.data() + k * vec_bytes);

        std::vector<float> rotated(padded_dim_);
        rotator_->rotate(raw_vec, rotated.data());
        std::vector<uint8_t> raw_code(padded_dim_, 0);

        // cids_ is label-indexed; quant_vecs_/factors_ are internal-id-indexed.
        uint8_t c_id = cids_[labels_[i]];
        float f_add, f_rescale, f_error;
        rabitqlib::quant::quantize_full_single(rotated.data(), centroids_ + padded_dim_ * c_id,
                                               padded_dim_, rabitq_bits_, raw_code.data(), f_add,
                                               f_rescale, f_error, metric, rabitq_config_);

        uint8_t *packed = quant_vecs_.data() + i * quant_vec_bytes_;
        float pop_count = 0;
        PackMultiBitCode(raw_code.data(), packed, pop_count);

        float *factor = factors_.data() + i * 4;
        factor[0] = f_add;
        factor[1] = f_rescale;
        factor[2] = f_error;
        factor[3] = pop_count;
      }
    }
  }

  auto rq_end = std::chrono::high_resolution_clock::now();
  std::cout << "RequantizeRaBitQ: n=" << n << " maxC=" << maxC_ << " ("
            << std::chrono::duration<double>(rq_end - rq_start).count() << "s)\n";
}

// ============================================================================
// RaBitQEstimator
// ============================================================================

template <typename att_t, typename vec_t>
AllianceIndexDisk<att_t, vec_t>::RaBitQEstimator::RaBitQEstimator(const vec_t *query,
                                                                  AllianceIndexDisk *idx)
    : index_(idx) {
  bits = static_cast<uint32_t>(idx->rabitq_bits_);
  rotated_query_ = new float[idx->padded_dim_];
  idx->rotator_->rotate(query, rotated_query_);

  dist_to_centroids_.resize(idx->maxC_);
  for (size_t i = 0; i < idx->maxC_; ++i) {
    dist_to_centroids_[i] = idx->rabitq_distfunc_(
        rotated_query_, idx->centroids_ + i * idx->padded_dim_, idx->rabitq_dist_func_param_);
  }

  // Query-side state is exclusively floating point. Reused by navigation
  // and rerank recomputation; no SQ buffer, min/max scaling or SQ error term.
  for (size_t d = 0; d < idx->padded_dim_; ++d) {
    float_query_sum_ += static_cast<double>(rotated_query_[d]);
    max_abs_query_ = std::max(max_abs_query_, std::abs(rotated_query_[d]));
  }
}

template <typename att_t, typename vec_t>
AllianceIndexDisk<att_t, vec_t>::RaBitQEstimator::~RaBitQEstimator() {
  delete[] rotated_query_;
}

namespace {
// Keep one source implementation without adding a call to either public API.
template <typename Estimator>
inline __attribute__((always_inline)) double
EstimateRaBitQDistance(const Estimator &estimator, uint8_t cid, tableint internal_id) {
  const double dot = detail::FloatQueryCodeDot(
      estimator.index_->RaBitQCode(internal_id), estimator.rotated_query_,
      estimator.index_->padded_dim_, estimator.bits, estimator.index_->RaBitQLayout());
  const float *f = estimator.index_->RaBitQFactors(internal_id);
  const double cb = -0.5 * static_cast<double>((1u << estimator.bits) - 1);
  // Keep the existing navigation metric convention, including the IP offset.
  return static_cast<double>(f[0]) + estimator.dist_to_centroids_[cid] +
         static_cast<double>(f[1]) * (dot + cb * estimator.float_query_sum_);
}
} // namespace

template <typename att_t, typename vec_t>
auto AllianceIndexDisk<att_t, vec_t>::RaBitQEstimator::dist_estimate_precise(uint8_t cid,
                                                                             tableint internal_id)
    -> double {
  return EstimateRaBitQDistance(*this, cid, internal_id);
}

template <typename att_t, typename vec_t>
auto AllianceIndexDisk<att_t, vec_t>::RaBitQEstimator::dist_estimate(uint8_t cid,
                                                                     tableint internal_id)
    -> float {
  return EstimateRaBitQDistance(*this, cid, internal_id);
}

template <typename att_t, typename vec_t>
auto AllianceIndexDisk<att_t, vec_t>::RaBitQEstimator::dist_estimate_with_errorbound(
    uint8_t cid, tableint internal_id) -> std::tuple<float, float> {
  // Both APIs use the same original RaBitQ float kernel and evaluation order.
  // The synchronous baseline additionally consumes the stored model error.
  return {EstimateRaBitQDistance(*this, cid, internal_id), index_->RaBitQFactors(internal_id)[2]};
}

template <typename att_t, typename vec_t>
auto AllianceIndexDisk<att_t, vec_t>::RaBitQEstimator::l2_lower_bound(uint8_t cid,
                                                                      tableint internal_id)
    -> float {
  return l2_lower_bound(cid, internal_id, dist_estimate(cid, internal_id));
}

template <typename att_t, typename vec_t>
auto AllianceIndexDisk<att_t, vec_t>::RaBitQEstimator::l2_lower_bound(uint8_t cid,
                                                                      tableint internal_id,
                                                                      float known_estimate)
    -> float {
  if (index_->rabitq_space_name_ != "l2")
    return -std::numeric_limits<float>::infinity();
  return distance_bounds(cid, internal_id, known_estimate).first;
}

template <typename att_t, typename vec_t>
auto AllianceIndexDisk<att_t, vec_t>::RaBitQEstimator::distance_bounds(uint8_t cid,
                                                                       tableint internal_id,
                                                                       float known_estimate,
                                                                       bool recompute_float)
    -> std::pair<float, float> {
  const float inf = std::numeric_limits<float>::infinity();
  const std::pair<float, float> invalid{-inf, inf};
  const bool l2 = index_->rabitq_space_name_ == "l2";
  if (!l2 && index_->rabitq_space_name_ != "ip")
    return invalid;
  const float *f = index_->RaBitQFactors(internal_id);
  const double query_centroid_d2 = dist_to_centroids_[cid];
  if (!std::isfinite(query_centroid_d2) || (l2 && query_centroid_d2 < 0) || f[2] < 0 || f[3] < 0)
    return invalid;
  if (norm_to_centroids_.empty())
    norm_to_centroids_.assign(dist_to_centroids_.size(), -1.0);
  double &norm = norm_to_centroids_[cid];
  if (l2) {
    if (norm < 0)
      norm = std::sqrt(query_centroid_d2);
  } else {
    // This quantizer's IP f_add includes <approx_residual-residual,c>.
    // Its error therefore projects on q-c, NOT q or sqrt(1-<q,c>).
    if (norm < 0) {
      double sum = 0;
      for (size_t d = 0; d < index_->padded_dim_; ++d) {
        const double diff = static_cast<double>(rotated_query_[d]) -
                            index_->centroids_[cid * index_->padded_dim_ + d];
        sum += diff * diff;
      }
      norm = std::sqrt(sum);
    }
  }
  // IP f_add and g_add each contain +1 in the existing scoring kernel.
  // Align the navigation score convention with
  // the exact metric 1-<q,x> before comparing it with exact rerank thresholds.
  double estimate = static_cast<double>(known_estimate) - (l2 ? 0.0 : 1.0);
  if (recompute_float) {
    double precise;
    if (refinement_cache_.Find(internal_id, cid, known_estimate, precise))
      ++refinement_reused_;
    else {
      precise = EstimateRaBitQDistance(*this, cid, internal_id);
      ++refinement_recomputed_;
    }
    estimate = precise - (l2 ? 0.0 : 1.0);
  }
  const double model_error = static_cast<double>(f[2]) * norm;
  const double sum_magnitude =
      0.5 * ((1u << bits) - 1) * index_->padded_dim_ * static_cast<double>(max_abs_query_);
  // Include accumulation/centroid/rounding slack in the native distance units.
  // This numeric guard does not turn RaBitQ's probabilistic bound into a
  // deterministic guarantee; observed recall must still be reported.
  const double scale = std::abs(static_cast<double>(f[0])) + std::abs(query_centroid_d2) +
                       std::abs(static_cast<double>(f[1])) *
                           (sum_magnitude + f[3] * static_cast<double>(max_abs_query_));
  const double guard =
      8.0 * std::numeric_limits<float>::epsilon() * (index_->padded_dim_ + 32) * (scale + 1.0);
  // f[2] already includes the library kConstEpsilon. No extra model scaling.
  const double error = model_error + guard;
  if (!std::isfinite(estimate) || !std::isfinite(error) || error < 0)
    return invalid;
  const double lower = estimate - error, upper = estimate + error;
  if (!std::isfinite(lower) || !std::isfinite(upper))
    return invalid;
  return {std::nextafter(static_cast<float>(lower), -inf),
          std::nextafter(static_cast<float>(upper), inf)};
}

template class AllianceIndexDisk<int, float>;

} // namespace alliance
