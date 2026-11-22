#pragma once
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace alliance {
// Shared immutable navigation metadata after preparation. Clear/Prepare with
// changed membership require exclusive index access, like requantization.
class GuidedEntryDirectory {
public:
  template <class Id, class CID>
  void Prepare(const std::vector<Id> &ids, size_t centroid_count, CID cid) {
    if (ready_.load(std::memory_order_acquire)) {
      if (point_count_ != ids.size() || ranks_.size() != centroid_count)
        throw std::logic_error("stale guided entry directory; clear before rebuilding");
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (ready_.load(std::memory_order_relaxed))
      return;
    if (!centroid_count || centroid_count > 256 ||
        ids.size() > std::numeric_limits<uint32_t>::max())
      throw std::logic_error("invalid guided entry directory dimensions");
    std::vector<size_t> counts(centroid_count, 0);
    for (auto id : ids) {
      size_t c = cid(id);
      if (c >= counts.size())
        throw std::logic_error("invalid entry CID");
      ++counts[c];
    }
    std::vector<std::vector<uint32_t>> next(centroid_count);
    for (size_t c = 0; c < centroid_count; ++c)
      next[c].reserve(counts[c]);
    for (size_t rank = 0; rank < ids.size(); ++rank)
      next[cid(ids[rank])].push_back(uint32_t(rank));
    ranks_.swap(next);
    point_count_ = ids.size();
    ready_.store(true, std::memory_order_release);
  }
  void Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    ready_.store(false, std::memory_order_release);
    decltype(ranks_){}.swap(ranks_);
    point_count_ = 0;
  }
  size_t Bytes() const {
    if (!ready_.load(std::memory_order_acquire))
      return 0;
    size_t n = ranks_.capacity() * sizeof(ranks_[0]);
    for (const auto &v : ranks_)
      n += v.capacity() * sizeof(uint32_t);
    return n;
  }
  // Reuse the query estimator's centroid distances to choose rank samples.
  // This directory stores no query scores; selected nodes are scored by traversal.
  std::vector<uint32_t> Select(size_t lo, size_t hi, const std::vector<float> &distances,
                               size_t budget) const {
    if (!ready_.load(std::memory_order_acquire))
      throw std::logic_error("navigation directory not prepared");
    if (lo > hi || hi > point_count_)
      throw std::out_of_range("invalid guided entry rank range");
    if (distances.size() != ranks_.size())
      throw std::logic_error("centroid count mismatch");
    struct Slice {
      size_t cid, begin, size, quota;
    };
    std::vector<Slice> slices;
    slices.reserve(ranks_.size());
    for (size_t c = 0; c < ranks_.size(); ++c) {
      const auto &list = ranks_[c];
      auto a = std::lower_bound(list.begin(), list.end(), lo);
      auto b = std::lower_bound(a, list.end(), hi);
      if (a != b)
        slices.push_back({c, size_t(a - list.begin()), size_t(b - a), 0});
    }
    std::sort(slices.begin(), slices.end(), [&](auto a, auto b) {
      return distances[a.cid] < distances[b.cid] ||
             (distances[a.cid] == distances[b.cid] && a.cid < b.cid);
    });
    // At least the nearest eight nonempty centroids; extend only if capacity
    // cannot fill the fixed candidate budget. Equal quotas with saturation.
    budget = std::min(budget, hi - lo);
    size_t keep = std::min(size_t(8), slices.size()), capacity = 0;
    for (size_t c = 0; c < keep; ++c)
      capacity += slices[c].size;
    while (keep < slices.size() && capacity < budget)
      capacity += slices[keep++].size;
    slices.resize(keep);
    if (capacity < budget)
      throw std::logic_error("incomplete navigation rank coverage");
    size_t assigned = 0;
    while (assigned < budget)
      for (auto &s : slices) {
        if (assigned == budget)
          break;
        if (s.quota < s.size) {
          ++s.quota;
          ++assigned;
        }
      }
    std::vector<uint32_t> result;
    result.reserve(budget);
    for (const auto &s : slices)
      for (size_t j = 0; j < s.quota; ++j) {
        const size_t offset = (uint64_t(j) * s.size + s.size / 2) / s.quota;
        result.push_back(ranks_[s.cid][s.begin + offset]);
      }
    return result;
  }

private:
  std::vector<std::vector<uint32_t>> ranks_;
  size_t point_count_{0};
  std::atomic<bool> ready_{false};
  std::mutex mutex_;
};
} // namespace alliance
