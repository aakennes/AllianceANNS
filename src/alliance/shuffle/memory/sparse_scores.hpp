#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace alliance::shuffle_detail {

// One row touches no more blocks than it has neighbors. Keep an open-addressed
// table bounded by that degree, plus insertion-ordered slots for legacy ties.
// A positive score marks occupancy; Clear() only visits touched slots.
class SparseScores {
public:
  explicit SparseScores(size_t maximum_degree) {
    if (maximum_degree > std::numeric_limits<size_t>::max() / 4)
      throw std::overflow_error("shuffle sparse score capacity overflow");
    size_t capacity = 1;
    while (capacity < maximum_degree * 2)
      capacity *= 2;
    keys_.resize(capacity);
    values_.resize(capacity, 0);
    touched_.reserve(maximum_degree);
    maximum_degree_ = maximum_degree;
  }

  size_t Add(size_t block, size_t value) {
    if (!value)
      throw std::invalid_argument("shuffle scores must be positive");
    const size_t slot = Find(block);
    if (!values_[slot]) {
      if (touched_.size() == maximum_degree_)
        throw std::logic_error("shuffle score row exceeds graph degree");
      keys_[slot] = block;
      touched_.push_back(slot);
    }
    if (value > std::numeric_limits<size_t>::max() - values_[slot])
      throw std::overflow_error("shuffle score overflow");
    return values_[slot] += value;
  }

  size_t Get(size_t block) const {
    return values_[Find(block)];
  }
  size_t Key(size_t slot) const {
    return keys_[slot];
  }
  size_t Value(size_t slot) const {
    return values_[slot];
  }
  const std::vector<size_t> &Touched() const {
    return touched_;
  }

  void Clear() {
    for (size_t slot : touched_)
      values_[slot] = 0;
    touched_.clear();
  }

  size_t StorageBytes() const {
    return (keys_.capacity() + values_.capacity() + touched_.capacity()) * sizeof(size_t);
  }

private:
  size_t Find(size_t block) const {
    uint64_t mixed = block;
    mixed ^= mixed >> 33;
    mixed *= UINT64_C(0xff51afd7ed558ccd);
    mixed ^= mixed >> 33;
    const size_t mask = values_.size() - 1;
    size_t slot = mixed & mask;
    while (values_[slot] && keys_[slot] != block)
      slot = (slot + 1) & mask;
    return slot;
  }

  std::vector<size_t> keys_, values_, touched_;
  size_t maximum_degree_{};
};

} // namespace alliance::shuffle_detail
