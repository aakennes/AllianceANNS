#pragma once

#include <vector>
#include <mutex>
#include <algorithm>
#include "core/types.hpp"
#include "io/io.hpp"

namespace alliance {

template <typename att_t> struct att_label_t {
  att_t att_;
  label_t label_;
  att_label_t() = default;
  att_label_t(att_t att, label_t label) : att_(att), label_(label) {
  }
  bool operator<(const att_label_t &other) const {
    return att_ < other.att_ || (att_ == other.att_ && label_ < other.label_);
  }
  bool operator>(const att_label_t &other) const {
    return att_ > other.att_ || (att_ == other.att_ && label_ > other.label_);
  }
  bool operator==(const att_label_t &other) const {
    return att_ == other.att_ && label_ == other.label_;
  }
  bool operator!=(const att_label_t &other) const {
    return !(*this == other);
  }
  bool operator<=(const att_label_t &other) const {
    return *this < other || *this == other;
  }
  bool operator>=(const att_label_t &other) const {
    return *this > other || *this == other;
  }
};

template <typename att_t> class OrderTable {

public:
  OrderTable() = default;

  virtual ~OrderTable() = default;

  virtual void InsertAttInid(const att_label_t<att_t> &att_label, tableint id) = 0;

  virtual auto GetWindowedFilterAndEntries(const att_label_t<att_t> &cur_att_label,
                                           int half_window_size,
                                           std::vector<tableint> &entry_points)
      -> attr_range<att_label_t<att_t>> = 0;

  virtual auto GetInWindowCandidates(const std::vector<dist_id_pair> &candidates,
                                     const std::vector<att_label_t<att_t>> &cand_att_label_vec,
                                     const att_label_t<att_t> &center_att_label,
                                     int half_window_size) -> std::vector<dist_id_pair> = 0;

  virtual auto GetRangeCardinality(const att_label_t<att_t> &l, const att_label_t<att_t> &u,
                                   std::vector<tableint> &OUT_eps) -> size_t = 0;

  virtual auto GetRankRange(const att_label_t<att_t> &l, const att_label_t<att_t> &u)
      -> std::pair<size_t, size_t> = 0;

  virtual void GetIDsInRankRange(size_t rank_lo, size_t rank_hi,
                                 std::vector<tableint> &out_ids) = 0;

  virtual void Finalize() {
  }

protected:
  std::mutex lock_{};
};

template <typename att_t> class SortedVectorOrderTable : public OrderTable<att_t> {
  struct Entry {
    att_t att;
    label_t label;
    tableint id;
    bool operator<(const Entry &o) const {
      return att < o.att || (att == o.att && label < o.label);
    }
  };

public:
  SortedVectorOrderTable() = delete;

  explicit SortedVectorOrderTable(size_t max_N) {
    entries_.reserve(max_N);
  }

  /// RAM held per inserted element, for build memory budgeting.
  static constexpr auto EntryBytes() -> size_t {
    return sizeof(Entry);
  }
  auto AllocatedBytes() const -> size_t {
    return entries_.capacity() * sizeof(Entry);
  }

  ~SortedVectorOrderTable() override = default;

  void InsertAttInid(const att_label_t<att_t> &att_label, tableint id) override {
    std::lock_guard<std::mutex> lock(this->lock_);
    Entry e{att_label.att_, att_label.label_, id};
    if (sorted_ && !entries_.empty() && e < entries_.back()) {
      sorted_ = false;
    }
    entries_.push_back(e);
  }

  void Finalize() override {
    std::lock_guard<std::mutex> lock(this->lock_);
    if (!sorted_) {
      std::sort(entries_.begin(), entries_.end());
      sorted_ = true;
    }
  }

  auto GetWindowedFilterAndEntries(const att_label_t<att_t> &cur_att_label, int half_window_size,
                                   std::vector<tableint> &entry_points)
      -> attr_range<att_label_t<att_t>> override {
    std::lock_guard<std::mutex> lock(this->lock_);
    EnsureSorted();
    int n = static_cast<int>(entries_.size());
    if (2 * half_window_size >= n) {
      entry_points.emplace_back(entries_.front().id);
      return {{entries_.front().att, entries_.front().label},
              {entries_.back().att, entries_.back().label}};
    }
    Entry key{cur_att_label.att_, cur_att_label.label_, 0};
    auto it = std::lower_bound(entries_.begin(), entries_.end(), key);
    int pos = static_cast<int>(it - entries_.begin());
    if (pos >= n)
      pos = n - 1;

    int lo = std::max(0, pos - half_window_size);
    int hi = std::min(n - 1, pos + half_window_size);
    if (hi - lo < 2 * half_window_size) {
      if (lo == 0) {
        hi = std::min(n - 1, 2 * half_window_size);
      } else {
        lo = std::max(0, n - 1 - 2 * half_window_size);
      }
    }

    entry_points.emplace_back(entries_[lo].id);
    if (lo != hi) {
      entry_points.emplace_back(entries_[hi].id);
    }
    return {{entries_[lo].att, entries_[lo].label}, {entries_[hi].att, entries_[hi].label}};
  }

  auto GetInWindowCandidates(const std::vector<dist_id_pair> &candidates,
                             const std::vector<att_label_t<att_t>> &cand_att_label_vec,
                             const att_label_t<att_t> &center_att_label, int half_window_size)
      -> std::vector<dist_id_pair> override {
    std::lock_guard<std::mutex> lock(this->lock_);
    EnsureSorted();
    int n = static_cast<int>(entries_.size());
    if (2 * half_window_size >= n) {
      return candidates;
    }
    Entry key{center_att_label.att_, center_att_label.label_, 0};
    auto it = std::lower_bound(entries_.begin(), entries_.end(), key);
    int pos = static_cast<int>(it - entries_.begin());
    if (pos >= n)
      pos = n - 1;
    if (!(pos < n && entries_[pos].att == center_att_label.att_ &&
          entries_[pos].label == center_att_label.label_)) {
      if (pos > 0)
        pos--;
    }

    int lo = std::max(0, pos - half_window_size);
    int hi = std::min(n - 1, pos + half_window_size);
    if (hi - lo < 2 * half_window_size) {
      if (lo == 0) {
        hi = std::min(n - 1, 2 * half_window_size);
      } else {
        lo = std::max(0, n - 1 - 2 * half_window_size);
      }
    }

    att_label_t<att_t> lo_key{entries_[lo].att, entries_[lo].label};
    att_label_t<att_t> hi_key{entries_[hi].att, entries_[hi].label};

    std::vector<dist_id_pair> result;
    for (size_t i = 0; i < candidates.size(); ++i) {
      if (cand_att_label_vec[i] >= lo_key && cand_att_label_vec[i] <= hi_key) {
        result.emplace_back(candidates[i]);
      }
    }
    return result;
  }

  auto GetRangeCardinality(const att_label_t<att_t> &l, const att_label_t<att_t> &u,
                           std::vector<tableint> &OUT_eps) -> size_t override {
    std::lock_guard<std::mutex> lock(this->lock_);
    EnsureSorted();
    Entry lo_key{l.att_, l.label_, 0};
    Entry hi_key{u.att_, u.label_, 0};
    auto it_l = std::lower_bound(entries_.begin(), entries_.end(), lo_key);
    auto it_u = std::upper_bound(it_l, entries_.end(), hi_key);
    if (it_l == it_u)
      return 0;
    OUT_eps.emplace_back(it_l->id);
    auto it_last = std::prev(it_u);
    if (it_last != it_l) {
      OUT_eps.emplace_back(it_last->id);
    }
    return static_cast<size_t>(std::distance(it_l, it_u));
  }

  auto GetElementCount() const -> size_t {
    return entries_.size();
  }

  auto GetRankRange(const att_label_t<att_t> &l, const att_label_t<att_t> &u)
      -> std::pair<size_t, size_t> override {
    std::lock_guard<std::mutex> lock(this->lock_);
    EnsureSorted();
    Entry lo_key{l.att_, l.label_, 0};
    Entry hi_key{u.att_, u.label_, 0};
    auto it_l = std::lower_bound(entries_.begin(), entries_.end(), lo_key);
    auto it_u = std::upper_bound(it_l, entries_.end(), hi_key);
    return {static_cast<size_t>(std::distance(entries_.begin(), it_l)),
            static_cast<size_t>(std::distance(entries_.begin(), it_u))};
  }

  void GetIDsInRankRange(size_t rank_lo, size_t rank_hi, std::vector<tableint> &out_ids) override {
    std::lock_guard<std::mutex> lock(this->lock_);
    EnsureSorted();
    rank_hi = std::min(rank_hi, entries_.size());
    out_ids.resize(rank_hi - rank_lo);
    for (size_t i = rank_lo; i < rank_hi; ++i) {
      out_ids[i - rank_lo] = entries_[i].id;
    }
  }

  auto GetSortedInternalIDs() const -> std::vector<tableint> {
    std::vector<tableint> ids(entries_.size());
    for (size_t i = 0; i < entries_.size(); ++i) {
      ids[i] = entries_[i].id;
    }
    return ids;
  }

private:
  void EnsureSorted() {
    if (!sorted_) {
      std::sort(entries_.begin(), entries_.end());
      sorted_ = true;
    }
  }

  std::vector<Entry> entries_;
  bool sorted_ = true;
};

} // namespace alliance
