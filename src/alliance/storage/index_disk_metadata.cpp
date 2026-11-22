#include "index/index_disk.hpp"
#include "io/disk_resource_guard.hpp"
#include "storage/disk_metadata.hpp"
#include "storage/atomic_file.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace alliance {

// ============================================================================
// Load constructor
// ============================================================================

template <typename att_t, typename vec_t>
AllianceIndexDisk<att_t, vec_t>::AllianceIndexDisk(const std::string &graph_file_path,
                                                   const std::string &meta_file_path,
                                                   std::string space_name)
    : AllianceIndexDisk(graph_file_path, meta_file_path, std::move(space_name),
                        StorageOpenMode::RequireRawGraph) {
}

template <typename att_t, typename vec_t>
AllianceIndexDisk<att_t, vec_t>::AllianceIndexDisk(const std::string &graph_file_path,
                                                   const std::string &meta_file_path,
                                                   std::string space_name, StorageOpenMode mode)
    : storage_open_mode_(mode) {
  if (mode != StorageOpenMode::RequireRawGraph && mode != StorageOpenMode::MultiLevelOnly)
    throw std::invalid_argument("unknown disk storage open mode");
  detail::ScopeExit failed_construction([this]() noexcept { ReleaseOwnedResources(); });
  auto metadata = storage::ReadDiskMetadata<att_t>(
      meta_file_path, storage::MetadataReadMode::Load, [this](size_t input_dim, size_t padded_dim) {
        rotator_ = rabitqlib::choose_rotator<float>(input_dim);
        if (rotator_->size() != padded_dim)
          throw std::runtime_error("Rotator padded_dim mismatch on load");
      });
  max_elements_ = metadata.max_elements;
  vec_d_ = metadata.vec_d;
  wp_ = metadata.wp;
  o_ = metadata.order;
  M_ = metadata.degree;
  efc_ = metadata.efc;
  curvec_num_ = metadata.count;
  cur_max_layer_ = metadata.max_layer;
  node_record_bytes_ = metadata.record_bytes;
  node_disk_bytes_ = metadata.node_bytes;

  if (space_name == "l2") {
    space_ = new alliance::L2Space(vec_d_);
  } else if (space_name == "ip") {
    space_ = new alliance::InnerProductSpace(vec_d_);
  } else {
    throw std::runtime_error("unsupported space type " + space_name);
  }
  fstdistfunc_ = space_->get_dist_func();
  dist_func_param_ = space_->get_dist_func_param();

  if (!IsMultiLevelOnly())
    graph_file_ = std::make_unique<PagedFile>(graph_file_path, /*create=*/false, header_bytes_,
                                              node_disk_bytes_);

  window_size_ = std::move(metadata.windows);
  labels_ = std::move(metadata.labels);
  atts_ = std::move(metadata.attributes);
  sorted_ids_ = std::move(metadata.sorted_ids);
  position_by_id_ = std::move(metadata.position_by_id);

  // Rebuild order_table_
  order_table_ = new SortedVectorOrderTable<att_t>(max_elements_);
  for (tableint i = 0; i < curvec_num_; ++i) {
    order_table_->InsertAttInid({atts_[i], labels_[i]}, i);
  }
  order_table_->Finalize();

  // Reconstruct runtime RaBitQ objects from the unchanged canonical payload.
  rabitq_initialized_ = metadata.quantized;
  if (rabitq_initialized_) {
    padded_dim_ = metadata.padded_dim;
    quant_vec_bytes_ = metadata.code_bytes;
    maxC_ = metadata.centroid_count;

    // rabitq_bits_ is not stored in meta -- derive it from the packed code size.
    // quant_vec_bytes_ = padded_dim_ * rabitq_bits_ / 8, so bits = 8*bytes/padded_dim.
    rabitq_bits_ = (padded_dim_ > 0) ? (quant_vec_bytes_ * 8 / padded_dim_) : 1;
    if (rabitq_bits_ == 0)
      rabitq_bits_ = 1;

    // The header callback already created and validated the owned rotator.
    centroids_ = metadata.centroids.release();
    cids_ = std::move(metadata.cids);
    quant_vecs_ = std::move(metadata.codes);
    factors_ = std::move(metadata.factors);

    rabitq_config_ = rabitqlib::quant::faster_config(padded_dim_, rabitq_bits_);
    rabitq_space_name_ = space_name;
    if (space_name == "l2") {
      rabitq_space_ = new alliance::L2Space(padded_dim_);
    } else {
      rabitq_space_ = new alliance::InnerProductSpace(padded_dim_);
    }
    rabitq_distfunc_ = rabitq_space_->get_dist_func();
    rabitq_dist_func_param_ = rabitq_space_->get_dist_func_param();
  }

  vec_file_path_ = std::move(metadata.vector_path);
  if (!IsMultiLevelOnly() && !vec_file_path_.empty())
    OpenVecFile();
  last_completed_layer_ = metadata.completed_layer;
  win_anchor_ = metadata.anchor;
  id_only_start_layer_ = metadata.id_only_start_layer;
  build_config_.left_stream_build_ = metadata.left_stream;
  left_stream_max_ = metadata.left_max;
  build_config_.left_stream_batch_ = metadata.left_batch;
  left_stream_complete_ = metadata.left_complete;

  std::cout << "AllianceIndexDisk loaded: N=" << curvec_num_ << " d=" << vec_d_ << " M=" << M_
            << " wp=" << wp_ << " rabitq=" << rabitq_initialized_
            << " vec_file=" << vec_file_available_
            << " win_anchor=" << WindowAnchorName(win_anchor_)
            << " last_completed_layer=" << last_completed_layer_ << "\n";
  failed_construction.Release();
}

// ============================================================================
// Save
// ============================================================================

template <typename att_t, typename vec_t>
void AllianceIndexDisk<att_t, vec_t>::Save(const std::string &meta_file_path) const {
  storage::WriteAtomicFile(meta_file_path, [&](std::ostream &ofs) {
    ofs.write(reinterpret_cast<const char *>(&max_elements_), sizeof(max_elements_));
    ofs.write(reinterpret_cast<const char *>(&vec_d_), sizeof(vec_d_));
    ofs.write(reinterpret_cast<const char *>(&wp_), sizeof(wp_));
    ofs.write(reinterpret_cast<const char *>(&o_), sizeof(o_));
    ofs.write(reinterpret_cast<const char *>(&M_), sizeof(M_));
    ofs.write(reinterpret_cast<const char *>(&efc_), sizeof(efc_));
    ofs.write(reinterpret_cast<const char *>(&curvec_num_), sizeof(curvec_num_));
    ofs.write(reinterpret_cast<const char *>(&cur_max_layer_), sizeof(cur_max_layer_));
    ofs.write(reinterpret_cast<const char *>(&node_record_bytes_), sizeof(node_record_bytes_));
    ofs.write(reinterpret_cast<const char *>(&node_disk_bytes_), sizeof(node_disk_bytes_));

    size_t ws_size = window_size_.size();
    ofs.write(reinterpret_cast<const char *>(&ws_size), sizeof(ws_size));
    ofs.write(reinterpret_cast<const char *>(window_size_.data()), ws_size * sizeof(size_t));

    ofs.write(reinterpret_cast<const char *>(labels_.data()), curvec_num_ * sizeof(label_t));
    ofs.write(reinterpret_cast<const char *>(atts_.data()), curvec_num_ * sizeof(att_t));
    ofs.write(reinterpret_cast<const char *>(sorted_ids_.data()), curvec_num_ * sizeof(tableint));
    ofs.write(reinterpret_cast<const char *>(position_by_id_.data()),
              curvec_num_ * sizeof(tableint));

    ofs.write(reinterpret_cast<const char *>(&rabitq_initialized_), sizeof(rabitq_initialized_));
    if (rabitq_initialized_) {
      ofs.write(reinterpret_cast<const char *>(&padded_dim_), sizeof(padded_dim_));
      ofs.write(reinterpret_cast<const char *>(&quant_vec_bytes_), sizeof(quant_vec_bytes_));
      ofs.write(reinterpret_cast<const char *>(&maxC_), sizeof(maxC_));
      ofs.write(reinterpret_cast<const char *>(centroids_), padded_dim_ * maxC_ * sizeof(float));
      ofs.write(reinterpret_cast<const char *>(cids_.data()), curvec_num_);
      if (!packed_rabitq_.Empty())
        packed_rabitq_.WriteCanonical(ofs);
      else {
        ofs.write(reinterpret_cast<const char *>(quant_vecs_.data()),
                  curvec_num_ * quant_vec_bytes_);
        ofs.write(reinterpret_cast<const char *>(factors_.data()), curvec_num_ * 4 * sizeof(float));
      }
    }

    // Vec file path for brute-force search
    size_t vfp_len = vec_file_path_.size();
    ofs.write(reinterpret_cast<const char *>(&vfp_len), sizeof(vfp_len));
    if (vfp_len > 0) {
      ofs.write(vec_file_path_.data(), vfp_len);
    }

    // Build checkpoint state (see load ctor for backward-compat note).
    ofs.write(reinterpret_cast<const char *>(&last_completed_layer_),
              sizeof(last_completed_layer_));

    // Window anchor precedes the optional tagged extensions.
    const uint8_t anchor_tag = static_cast<uint8_t>(win_anchor_);
    ofs.write(reinterpret_cast<const char *>(&anchor_tag), sizeof(anchor_tag));

    if (id_only_start_layer_ <= wp_) {
      const uint32_t tag = storage::kIdOnlyMetadataTag;
      const uint64_t first = id_only_start_layer_;
      ofs.write(reinterpret_cast<const char *>(&tag), 4);
      ofs.write(reinterpret_cast<const char *>(&first), 8);
    }

    if (build_config_.left_stream_build_) {
      const uint32_t tag = storage::kLeftStreamMetadataTag;
      const uint64_t maximum = left_stream_max_, batch = build_config_.left_stream_batch_;
      const uint8_t complete = left_stream_complete_ ? 1 : 0;
      ofs.write(reinterpret_cast<const char *>(&tag), 4);
      ofs.write(reinterpret_cast<const char *>(&maximum), 8);
      ofs.write(reinterpret_cast<const char *>(&batch), 8);
      ofs.write(reinterpret_cast<const char *>(&complete), 1);
    }
  });
  std::cout << "Saved metadata to " << meta_file_path << "\n";
}

template AllianceIndexDisk<int, float>::AllianceIndexDisk(const std::string &, const std::string &,
                                                          std::string);
template AllianceIndexDisk<int, float>::AllianceIndexDisk(const std::string &, const std::string &,
                                                          std::string, StorageOpenMode);
template void AllianceIndexDisk<int, float>::Save(const std::string &) const;

} // namespace alliance
