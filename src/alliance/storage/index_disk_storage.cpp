#include "index/index_disk.hpp"
#include "io/aio_stream.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <sys/stat.h>

namespace alliance {

// ============================================================================
// Disk-backed accessors
// ============================================================================

template <typename att_t, typename vec_t>
void AllianceIndexDisk<att_t, vec_t>::ReadEdges(tableint id, layer_t layer,
                                                tableint *out_edges) const {
  RequireRawGraph("ReadEdges");
  graph_file_->ReadNodeSlice(DiskSlot(id), EdgeOffsetInRecord(layer), EdgeIdBytes(), out_edges);
}

template <typename att_t, typename vec_t>
void AllianceIndexDisk<att_t, vec_t>::ReadEdgesWithDist(tableint id, layer_t layer,
                                                        tableint *out_ids,
                                                        dist_t *out_dists) const {
  RequireRawGraph("ReadEdgesWithDist");
  tableint slot = DiskSlot(id);
  graph_file_->ReadNodeSlice(slot, EdgeOffsetInRecord(layer), EdgeIdBytes(), out_ids);
  if (HasStoredEdgeDistances(layer)) {
    graph_file_->ReadNodeSlice(slot, EdgeOffsetInRecord(layer) + EdgeIdBytes(), EdgeDistBytes(),
                               out_dists);
  } else {
    std::fill(out_dists, out_dists + M_, dist_t(0));
    const size_t count = out_ids[M_];
    if (count > M_)
      throw std::runtime_error("invalid ID-only edge degree");
    if (!count)
      return;
    std::vector<vec_t> from(vec_d_), to(vec_d_);
    ReadVec(id, from.data());
    for (size_t i = 0; i < count; ++i) {
      if (out_ids[i] >= curvec_num_)
        throw std::runtime_error("invalid ID-only neighbor");
      ReadVec(out_ids[i], to.data());
      out_dists[i] = fstdistfunc_(from.data(), to.data(), dist_func_param_);
    }
  }
}

template <typename att_t, typename vec_t>
void AllianceIndexDisk<att_t, vec_t>::WriteEdges(tableint id, layer_t layer,
                                                 const tableint *neighbor_ids,
                                                 tableint count) const {
  RequireRawGraph("WriteEdges");
  std::vector<tableint> buf(M_ + 1, 0);
  tableint n = std::min(count, static_cast<tableint>(M_));
  for (tableint i = 0; i < n; ++i)
    buf[i] = neighbor_ids[i];
  buf[M_] = n;
  graph_file_->WriteNodeSlice(DiskSlot(id), EdgeOffsetInRecord(layer), EdgeIdBytes(), buf.data());
}

template <typename att_t, typename vec_t>
void AllianceIndexDisk<att_t, vec_t>::WriteEdgesWithDist(tableint id, layer_t layer,
                                                         const tableint *ids, const dist_t *dists,
                                                         tableint count) const {
  RequireRawGraph("WriteEdgesWithDist");
  std::vector<tableint> id_buf(M_ + 1, 0);
  std::vector<dist_t> dist_buf(M_, 0);
  tableint n = std::min(count, static_cast<tableint>(M_));
  for (tableint i = 0; i < n; ++i) {
    id_buf[i] = ids[i];
    dist_buf[i] = HasStoredEdgeDistances(layer) ? dists[i] : dist_t(0);
  }
  id_buf[M_] = n;
  tableint slot = DiskSlot(id);
  graph_file_->WriteNodeSlice(slot, EdgeOffsetInRecord(layer), EdgeIdBytes(), id_buf.data());
  graph_file_->WriteNodeSlice(slot, EdgeOffsetInRecord(layer) + EdgeIdBytes(), EdgeDistBytes(),
                              dist_buf.data());
}

template <typename att_t, typename vec_t>
void AllianceIndexDisk<att_t, vec_t>::ReadVec(tableint id, vec_t *out_vec) const {
  RequireRawGraph("ReadVec");
  graph_file_->ReadNodeSlice(DiskSlot(id), VecOffsetInRecord(), VecBytes(), out_vec);
}

// ============================================================================
// WriteVecFile -- writes raw vectors in rank order (sorted_ids_ order)
// ============================================================================

template <typename att_t, typename vec_t>
void AllianceIndexDisk<att_t, vec_t>::WriteVecFile(const vec_t *base_vecs) {
  RequireRawGraph("WriteVecFile");
  if (vec_file_path_.empty())
    return;

  // Dense, rank-ordered, no padding: a pure stream. AioRegionWriter needs the
  // extent up front and rounds its final flush out to a sector, so preallocate
  // the padded length and cut the pad off afterwards -- readers bound
  // themselves by curvec_num_ * vec_bytes either way, but the file length is
  // part of this format's contract.
  const size_t vec_bytes = vec_d_ * sizeof(vec_t);
  const uint64_t total_bytes = (uint64_t)curvec_num_ * vec_bytes;
  AioPreallocate(vec_file_path_, AioPadUp(total_bytes));
  {
    AioRegionWriter writer(vec_file_path_, 0);
    for (size_t rank = 0; rank < curvec_num_; ++rank) {
      tableint id = sorted_ids_[rank];
      label_t label = labels_[id];
      writer.Append(base_vecs + static_cast<size_t>(label) * vec_d_, vec_bytes);
    }
    writer.Close();
    if (writer.written() != total_bytes) {
      throw std::runtime_error("WriteVecFile: short write");
    }
  }
  if (AioPadUp(total_bytes) != total_bytes) {
    AioTruncateTo(vec_file_path_, total_bytes);
  }
  vec_file_available_ = true;
  std::cout << "WriteVecFile: wrote " << curvec_num_ << " vectors to " << vec_file_path_ << "\n";
}

// ============================================================================
// WriteVecFileFromDisk -- reads vectors from graph_file_ (on disk) in rank order
// ============================================================================

template <typename att_t, typename vec_t>
void AllianceIndexDisk<att_t, vec_t>::WriteVecFileFromDisk() {
  RequireRawGraph("WriteVecFileFromDisk");
  if (vec_file_path_.empty())
    return;
  const auto t0 = std::chrono::high_resolution_clock::now();

  const size_t vec_bytes = vec_d_ * sizeof(vec_t);
  const uint64_t total_bytes = (uint64_t)curvec_num_ * vec_bytes;
  AioPreallocate(vec_file_path_, AioPadUp(total_bytes));
  // The reader below is O_DIRECT, so anything still sitting dirty in the page
  // cache from a buffered PagedFile write would be invisible to it.
  graph_file_->Sync();
  {
    // DiskSlot(sorted_ids_[rank]) == rank by construction, so the ReadVec loop
    // this replaces was walking the record array strictly forwards -- both sides
    // are streams, and neither needed a per-record blocking call.
    AioSeqStreamReader reader(graph_file_->path(), graph_file_->node_offset(0),
                              graph_file_->node_offset(static_cast<tableint>(curvec_num_)));
    AioRegionWriter writer(vec_file_path_, 0);
    std::vector<char> rec_buf(node_disk_bytes_);
    for (size_t rank = 0; rank < curvec_num_; ++rank) {
      reader.Read(rec_buf.data(), node_disk_bytes_);
      writer.Append(rec_buf.data() + VecOffsetInRecord(), vec_bytes);
    }
    writer.Close();
    if (writer.written() != total_bytes) {
      throw std::runtime_error("WriteVecFileFromDisk: short write");
    }
  }
  if (AioPadUp(total_bytes) != total_bytes) {
    AioTruncateTo(vec_file_path_, total_bytes);
  }
  vec_file_available_ = true;
  std::cout << "WriteVecFileFromDisk: wrote " << curvec_num_ << " vectors to " << vec_file_path_
            << " ("
            << std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count()
            << "s)\n";
}

// ============================================================================
// OpenVecFile -- marks the vec file usable for brute-force search
// ============================================================================

// Existence probe only. It used to hold a long-lived O_RDONLY fd for
// BruteForceSearch's ::pread; that read is now an AioSeqStreamReader which opens
// its own O_DIRECT fd per call, so the fd here had no reader left and the name
// is all that survives of it.
template <typename att_t, typename vec_t> void AllianceIndexDisk<att_t, vec_t>::OpenVecFile() {
  if (IsMultiLevelOnly())
    throw std::runtime_error("raw-vector fallback is unavailable in ML-only mode");
  if (vec_file_path_.empty())
    return;
  struct stat st{};
  if (::stat(vec_file_path_.c_str(), &st) == 0 && st.st_size > 0) {
    vec_file_available_ = true;
  }
}

template void AllianceIndexDisk<int, float>::ReadEdges(tableint, layer_t, tableint *) const;
template void AllianceIndexDisk<int, float>::ReadEdgesWithDist(tableint, layer_t, tableint *,
                                                               dist_t *) const;
template void AllianceIndexDisk<int, float>::WriteEdges(tableint, layer_t, const tableint *,
                                                        tableint) const;
template void AllianceIndexDisk<int, float>::WriteEdgesWithDist(tableint, layer_t, const tableint *,
                                                                const dist_t *, tableint) const;
template void AllianceIndexDisk<int, float>::ReadVec(tableint, float *) const;
template void AllianceIndexDisk<int, float>::WriteVecFile(const float *);
template void AllianceIndexDisk<int, float>::WriteVecFileFromDisk();
template void AllianceIndexDisk<int, float>::OpenVecFile();

} // namespace alliance
