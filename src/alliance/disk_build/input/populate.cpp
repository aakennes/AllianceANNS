#include "index/index_disk.hpp"
#include "io/aio_stream.hpp"
#include "io/float_vector_input.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>

namespace alliance {

/// RAM the fvecs-streaming PopulateElements may hold for one rank window when no
/// --build_mem_gb is set. See the Phase 2 comment: the window is what turns the
/// record writes from scattered into sequential, and its size sets how many
/// times the input has to be re-streamed.
constexpr uint64_t kRecordWindowBytesDefault = 1ull << 30;

namespace {
// Generate persistent search-side codes/factors only when InitRaBitQ has run.
// This does not select graph edges; LeftStream uses exact vector distances later.
// Both input sources share the code layout and CID/label mapping; encoding scratch
// lives for one population pass. Save serializes the resulting search metadata.
template <typename att_t, typename vec_t> class PopulationQuantizer {
public:
  PopulationQuantizer(AllianceIndexDisk<att_t, vec_t> &index, size_t n) : index_(index) {
    if (!index_.rabitq_initialized_)
      return;
    // InitRaBitQ can precede population, before curvec_num_ is known.
    if (index_.cids_.size() < n)
      throw std::runtime_error(
          "PopulateElements: RaBitQ cids count is smaller than the build input");
    index_.quant_vecs_.resize(n * index_.quant_vec_bytes_);
    index_.factors_.resize(n * 4);
    rotated_.resize(index_.padded_dim_);
    code_.resize(index_.padded_dim_);
    if (index_.rabitq_space_name_ == "ip")
      metric_ = rabitqlib::METRIC_IP;
  }

  void Encode(size_t id, const vec_t *vector) {
    if (!index_.rabitq_initialized_)
      return;
    std::fill(code_.begin(), code_.end(), uint8_t{0});
    index_.rotator_->rotate(vector, rotated_.data());
    const uint8_t cid = index_.cids_[index_.labels_[id]];
    float *factors = index_.factors_.data() + id * 4;
    rabitqlib::quant::quantize_full_single(
        rotated_.data(), index_.centroids_ + index_.padded_dim_ * cid, index_.padded_dim_,
        index_.rabitq_bits_, code_.data(), factors[0], factors[1], factors[2], metric_,
        index_.rabitq_config_);
    index_.PackMultiBitCode(code_.data(), index_.quant_vecs_.data() + id * index_.quant_vec_bytes_,
                            factors[3]);
  }

private:
  AllianceIndexDisk<att_t, vec_t> &index_;
  rabitqlib::MetricType metric_{rabitqlib::METRIC_L2};
  std::vector<float> rotated_;
  std::vector<uint8_t> code_;
};
} // namespace

// ============================================================================
// PopulateElements
// ============================================================================

template <typename att_t, typename vec_t>
void AllianceIndexDisk<att_t, vec_t>::PopulateElements(const vec_t *base_vecs, const att_t *att_vec,
                                                       size_t n) {
  RequireRawGraph("PopulateElements");
  labels_.resize(n);
  atts_.resize(n);

  PopulationQuantizer<att_t, vec_t> quantizer(*this, n);

  // Population pass 1: order table + optional search RaBitQ codes (no disk IO).
  for (size_t i = 0; i < n; ++i) {
    label_t label = static_cast<label_t>(i);
    labels_[i] = label;
    atts_[i] = att_vec[i];

    order_table_->InsertAttInid({att_vec[i], label}, static_cast<tableint>(i));

    quantizer.Encode(i, base_vecs + i * vec_d_);
  }

  // Finalize sort: position_by_id_ now valid for DiskSlot()
  order_table_->Finalize();
  sorted_ids_ = order_table_->GetSortedInternalIDs();
  position_by_id_.resize(n);
  for (size_t i = 0; i < n; ++i) {
    position_by_id_[sorted_ids_[i]] = static_cast<tableint>(i);
  }
  curvec_num_ = n;
  cur_max_layer_ = wp_;

  // Phase 2: Write node records to disk, in rank order.
  {
    AioRegionWriter writer(graph_file_->path(), graph_file_->node_offset(0));
    std::vector<char> record(node_disk_bytes_, 0);
    for (size_t rank = 0; rank < n; ++rank) {
      const tableint id = sorted_ids_[rank];
      std::memcpy(record.data() + VecOffsetInRecord(), base_vecs + static_cast<size_t>(id) * vec_d_,
                  VecBytes());
      writer.Append(record.data(), node_disk_bytes_);
    }
    writer.Close();
  }

  // Phase 3: Write dense build vec file
  WriteBuildVecFile(base_vecs, n);
}

// ============================================================================
// PopulateElements (streaming from fvecs file)
// ============================================================================

template <typename att_t, typename vec_t>
void AllianceIndexDisk<att_t, vec_t>::PopulateElements(const std::string &fvecs_path,
                                                       const att_t *att_vec, size_t n,
                                                       const std::string &format) {
  RequireRawGraph("PopulateElements");
  labels_.resize(n);
  atts_.resize(n);

  PopulationQuantizer<att_t, vec_t> quantizer(*this, n);

  std::vector<vec_t> vec_buf(vec_d_);

  const auto t_pop0 = std::chrono::high_resolution_clock::now();

  // Population pass 1: order table + optional search RaBitQ codes (streamed input).
  {
    AioFloatVectorStreamReader reader(fvecs_path, format);
    for (size_t i = 0; i < n; ++i) {
      if (!reader.ReadNext(vec_buf.data())) {
        throw std::runtime_error("PopulateElements: premature EOF at vector " + std::to_string(i));
      }

      label_t label = static_cast<label_t>(i);
      labels_[i] = label;
      atts_[i] = att_vec[i];

      order_table_->InsertAttInid({att_vec[i], label}, static_cast<tableint>(i));

      quantizer.Encode(i, vec_buf.data());
    }
  }

  // Finalize sort: position_by_id_ now valid for DiskSlot()
  order_table_->Finalize();
  sorted_ids_ = order_table_->GetSortedInternalIDs();
  position_by_id_.resize(n);
  for (size_t i = 0; i < n; ++i) {
    position_by_id_[sorted_ids_[i]] = static_cast<tableint>(i);
  }
  curvec_num_ = n;
  cur_max_layer_ = wp_;
  const auto t_pop1 = std::chrono::high_resolution_clock::now();

  // Phase 2: Write node records to disk, in rank order (stream from fvecs again).

  {
    const size_t vec_bytes = VecBytes();
    const uint64_t win_budget =
        (build_config_.build_mem_gb_ > 0.0)
            ? (uint64_t)(build_config_.build_mem_gb_ * 0.5 * 1024.0 * 1024.0 * 1024.0)
            : kRecordWindowBytesDefault;
    size_t win_ranks = std::max<size_t>(1, (size_t)(win_budget / vec_bytes));
    win_ranks = std::min(win_ranks, n);
    const size_t passes = (n + win_ranks - 1) / win_ranks;
    fprintf(
        stderr,
        "  PopulateElements Phase2: rank-ordered, window %zu ranks (%.1f MB), %zu input pass%s\n",
        win_ranks, (double)win_ranks * vec_bytes / (1024.0 * 1024.0), passes,
        passes == 1 ? "" : "es");

    std::vector<char> window((size_t)win_ranks * vec_bytes);
    std::vector<char> record(node_disk_bytes_, 0);

    double scan_s = 0.0, flush_s = 0.0;
    for (size_t wbeg = 0; wbeg < n; wbeg += win_ranks) {
      const size_t wend = std::min(n, wbeg + win_ranks);
      const auto ts0 = std::chrono::high_resolution_clock::now();
      {
        AioFloatVectorStreamReader reader(fvecs_path, format);
        for (size_t i = 0; i < n; ++i) {
          if (!reader.ReadNext(vec_buf.data())) {
            throw std::runtime_error("PopulateElements Phase2: premature EOF at vector " +
                                     std::to_string(i));
          }
          const size_t rank = DiskSlot(static_cast<tableint>(i));
          if (rank >= wbeg && rank < wend) {
            std::memcpy(window.data() + (rank - wbeg) * vec_bytes, vec_buf.data(), vec_bytes);
          }
        }
      }
      const auto ts1 = std::chrono::high_resolution_clock::now();
      AioRegionWriter writer(graph_file_->path(),
                             graph_file_->node_offset(static_cast<tableint>(wbeg)));
      for (size_t rank = wbeg; rank < wend; ++rank) {
        std::memcpy(record.data() + VecOffsetInRecord(), window.data() + (rank - wbeg) * vec_bytes,
                    vec_bytes);
        writer.Append(record.data(), node_disk_bytes_);
      }
      writer.Close();
      const auto ts2 = std::chrono::high_resolution_clock::now();
      scan_s += std::chrono::duration<double>(ts1 - ts0).count();
      flush_s += std::chrono::duration<double>(ts2 - ts1).count();
    }
    const double in_gb =
        (double)passes * n * (sizeof(int) + vec_bytes) / (1024.0 * 1024.0 * 1024.0);
    fprintf(stderr,
            "  PopulateElements Phase2: input %.2fs (%.2f GB over %zu pass%s, %.0f MB/s), "
            "records %.2fs (%.2f GB, %.0f MB/s)\n",
            scan_s, in_gb, passes, passes == 1 ? "" : "es",
            scan_s > 0.0 ? in_gb * 1024.0 / scan_s : 0.0, flush_s,
            (double)n * node_disk_bytes_ / (1024.0 * 1024.0 * 1024.0),
            flush_s > 0.0 ? (double)n * node_disk_bytes_ / (1024.0 * 1024.0) / flush_s : 0.0);
  }
  const auto t_pop2 = std::chrono::high_resolution_clock::now();

  // Phase 3: Write dense build vec file.
  WriteBuildVecFileFromGraph(n);
  const auto t_pop3 = std::chrono::high_resolution_clock::now();
  fprintf(stderr,
          "  PopulateElements: quantize+sort %.2fs, records %.2fs (%.1f GB), buildvecs %.2fs\n",
          std::chrono::duration<double>(t_pop1 - t_pop0).count(),
          std::chrono::duration<double>(t_pop2 - t_pop1).count(),
          (double)n * node_disk_bytes_ / (1024.0 * 1024.0 * 1024.0),
          std::chrono::duration<double>(t_pop3 - t_pop2).count());
}

// Input sources share encoding while retaining distinct vector IO/staging.
template void AllianceIndexDisk<int, float>::PopulateElements(const float *, const int *, size_t);
template void AllianceIndexDisk<int, float>::PopulateElements(const std::string &, const int *,
                                                              size_t, const std::string &);

} // namespace alliance
