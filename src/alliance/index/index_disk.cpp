#include "index/index_disk.hpp"
#include "disk_build/memory/index_adapter.hpp"
#include "build/disk_build_geometry.hpp"
#include "io/disk_resource_guard.hpp"
#include <iostream>
#include <memory>
#include <stdexcept>

namespace alliance {

template <typename att_t, typename vec_t>
AllianceIndexDisk<att_t, vec_t>::AllianceIndexDisk(size_t max_elements, size_t vec_d, size_t M,
                                                   size_t efc, std::string space_name,
                                                   std::string graph_file_path, size_t o, size_t wp,
                                                   bool auto_raise_wp, size_t enum_threshold,
                                                   size_t node_page_bytes, WindowAnchor win_anchor)
    : max_elements_(max_elements), vec_d_(vec_d), wp_(wp), o_(o), M_(M), efc_(efc),
      enum_threshold_(enum_threshold), win_anchor_(win_anchor) {
  detail::ScopeExit failed_construction([this]() noexcept { ReleaseOwnedResources(); });
  if (space_name == "l2") {
    space_ = new alliance::L2Space(vec_d_);
  } else if (space_name == "ip") {
    space_ = new alliance::InnerProductSpace(vec_d_);
  } else {
    throw std::runtime_error("unsupported space type " + space_name + ", supported: l2, ip");
  }
  fstdistfunc_ = space_->get_dist_func();
  dist_func_param_ = space_->get_dist_func_param();

  {
    const WindowLadder ladder =
        BuildWindowLadder(max_elements_, o_, wp_, enum_threshold_, M_, win_anchor_, auto_raise_wp);
    if (ladder.wp_raised) {
      std::cout << "auto_raise_wp is true. to cover all unique values, increased wp_ to "
                << ladder.wp;
      if (win_anchor_ == WindowAnchor::kTop) {
        std::cout << " (reverse-solved from enum_thresh=" << enum_threshold_
                  << ": layer 0 window is " << ladder.base << ")";
      }
      std::cout << std::endl;
    }
    if (ladder.base_clamped) {
      std::cout
          << "WARNING: win_anchor=top derived a layer-0 window of ceil(n/o^wp) < max(M+1,64) for n="
          << max_elements_ << ", o=" << o_ << ", wp=" << wp_ << "; raised it to " << ladder.base
          << ". The upper windows now collide at n -- lower wp or raise n." << std::endl;
    }
    wp_ = ladder.wp;
    window_size_ = ladder.window_size;
  }

  const auto storage =
      disk_build::PlanNodeStorage(wp_, M_, vec_d_ * sizeof(vec_t), node_page_bytes);
  node_record_bytes_ = storage.record_bytes;
  node_disk_bytes_ = storage.page_bytes;

  graph_file_ = std::make_unique<PagedFile>(graph_file_path, /*create=*/true, header_bytes_,
                                            node_disk_bytes_);
  graph_file_->Preallocate(max_elements_);

  struct DiskIndexHeader {
    size_t max_elements, vec_d, wp, o, M, efc, curvec_num, cur_max_layer, node_record_bytes,
        node_disk_bytes;
  } header{max_elements_,      vec_d_,          wp_, o_, M_, efc_, curvec_num_, cur_max_layer_,
           node_record_bytes_, node_disk_bytes_};
  graph_file_->WriteHeader(&header, sizeof(header));

  order_table_ = new SortedVectorOrderTable<att_t>(max_elements_);

  std::cout << "AllianceIndexDisk: node_record_bytes=" << node_record_bytes_
            << " node_disk_bytes=" << node_disk_bytes_ << " total_disk_mb="
            << (header_bytes_ + max_elements_ * node_disk_bytes_) / (1024 * 1024) << "\n";
  failed_construction.Release();
}

template <typename att_t, typename vec_t>
void AllianceIndexDisk<att_t, vec_t>::ReleaseBuildResources() noexcept {
  build_state_.reset();
}

template <typename att_t, typename vec_t>
void AllianceIndexDisk<att_t, vec_t>::ReleaseOwnedResources() noexcept {
  ReleaseBuildResources();
  if (ml_fd_ >= 0) {
    ::close(ml_fd_);
    ml_fd_ = -1;
  }
  delete order_table_;
  order_table_ = nullptr;
  delete space_;
  space_ = nullptr;
  delete rotator_;
  rotator_ = nullptr;
  delete[] centroids_;
  centroids_ = nullptr;
  delete rabitq_space_;
  rabitq_space_ = nullptr;
  if (static_cache_buf_) {
    free(static_cache_buf_);
    static_cache_buf_ = nullptr;
  }
}

template <typename att_t, typename vec_t> AllianceIndexDisk<att_t, vec_t>::~AllianceIndexDisk() {
  ReleaseOwnedResources();
}

template <typename att_t, typename vec_t>
auto AllianceIndexDisk<att_t, vec_t>::EstimateBaseBuildBytes(size_t n) const -> size_t {
  return disk_build::MemoryPlanner<att_t, vec_t>::FromIndex(*this).EstimateBaseBuildBytes(n);
}
template <typename att_t, typename vec_t>
auto AllianceIndexDisk<att_t, vec_t>::PlanPhase15(size_t n) const -> Phase15Plan {
  return disk_build::MemoryPlanner<att_t, vec_t>::FromIndex(*this).PlanPhase15(n);
}
template <typename att_t, typename vec_t>
auto AllianceIndexDisk<att_t, vec_t>::PlanPhase2(size_t n) const -> Phase2Plan {
  return disk_build::MemoryPlanner<att_t, vec_t>::FromIndex(*this).PlanPhase2(n);
}
template class AllianceIndexDisk<int, float>;

} // namespace alliance
