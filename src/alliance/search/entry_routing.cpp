#include "index/index_disk.hpp"
#include <algorithm>
#include <limits>

namespace alliance {

// Choose a landing layer from range cardinality and return attribute-order entries.

template <typename att_t, typename vec_t>
auto AllianceIndexDisk<att_t, vec_t>::DecideLayerRange(const attr_range<att_t> &filter_range,
                                                       std::vector<tableint> &OUT_eps)
    -> attr_range<layer_t> {
  attr_range<layer_t> new_layer_rng;
  size_t filter_card = order_table_->GetRangeCardinality(
      {filter_range.l_, 0}, {filter_range.u_, std::numeric_limits<label_t>::max()}, OUT_eps);
  auto c_it = std::lower_bound(window_size_.begin(), window_size_.end(), filter_card);
  if (c_it != window_size_.begin() && (c_it == window_size_.end() || *c_it > filter_card)) {
    c_it--;
  }
  int c_it_idx = std::distance(window_size_.begin(), c_it);
  if (c_it_idx == 0) {
    new_layer_rng.u_ = c_it_idx + 1;
  } else if (c_it_idx == (int)wp_) {
    new_layer_rng.u_ = c_it_idx;
  } else {
    int c_l = c_it_idx - 1;
    int c_u = c_it_idx + 1;
    float frac_l = 1.0f * window_size_[c_l] / filter_card;
    float frac_u = 1.0f * filter_card / std::min((size_t)window_size_[c_u], max_elements_);
    if (frac_l > frac_u) {
      new_layer_rng.u_ = c_it_idx;
    } else {
      new_layer_rng.u_ = c_u;
    }
  }
  // Small graphs can select layer 1 even when wp_ is zero. Clamp to the
  // stored ladder before any caller computes an edge slot for the landing layer.
  const auto wp_layer = static_cast<layer_t>(wp_);
  if (new_layer_rng.u_ > wp_layer)
    new_layer_rng.u_ = wp_layer;
  new_layer_rng.l_ = 0;
  return new_layer_rng;
}

template class AllianceIndexDisk<int, float>;

} // namespace alliance
