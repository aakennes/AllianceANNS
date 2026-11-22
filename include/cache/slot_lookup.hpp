#pragma once
#include "core/types.hpp"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace alliance {

// Fixed-capacity exact lookup. Backward-shift deletion avoids tombstones even
// after arbitrarily many FIFO/LRU evictions. Only Configure allocates memory.
class SlotLookup {
public:
  static constexpr tableint kInvalid = static_cast<tableint>(-1);
  void Configure(size_t nodes, size_t slots) {
#ifdef ALLIANCE_DENSE_DYNAMIC_CACHE
    if (!slots)
      std::vector<tableint>().swap(entries_);
    else if (entries_.size() != nodes)
      entries_.assign(nodes, kInvalid);
#else
    if (slots >= static_cast<size_t>(kInvalid) || slots > std::numeric_limits<size_t>::max() / 2)
      throw std::length_error("SlotLookup capacity exceeds slot ID range");
    size_t capacity = slots ? 2 : 0;
    while (capacity < slots * 2) {
      if (capacity > std::numeric_limits<size_t>::max() / 2)
        throw std::length_error("SlotLookup table size overflow");
      capacity *= 2;
    }
    std::vector<uint64_t>(capacity, kEmpty).swap(entries_);
    mask_ = capacity ? capacity - 1 : 0;
    (void)nodes;
#endif
  }
  tableint Find(tableint id) const {
#ifdef ALLIANCE_DENSE_DYNAMIC_CACHE
    return entries_[id];
#else
    size_t i = Hash(id) & mask_;
    while (entries_[i] != kEmpty) {
      if (Key(entries_[i]) == id)
        return static_cast<tableint>(entries_[i]);
      i = (i + 1) & mask_;
    }
    return kInvalid;
#endif
  }
  void Insert(tableint id, tableint slot) {
#ifdef ALLIANCE_DENSE_DYNAMIC_CACHE
    entries_[id] = slot;
#else
    size_t i = Hash(id) & mask_;
    while (entries_[i] != kEmpty && Key(entries_[i]) != id)
      i = (i + 1) & mask_;
    entries_[i] = (uint64_t(id) << 32) | slot;
#endif
  }
  void Erase(tableint id) {
#ifdef ALLIANCE_DENSE_DYNAMIC_CACHE
    entries_[id] = kInvalid;
#else
    size_t hole = Hash(id) & mask_;
    while (entries_[hole] != kEmpty && Key(entries_[hole]) != id)
      hole = (hole + 1) & mask_;
    if (entries_[hole] == kEmpty)
      return;
    for (size_t i = (hole + 1) & mask_; entries_[i] != kEmpty; i = (i + 1) & mask_) {
      const size_t home = Hash(Key(entries_[i])) & mask_;
      if (((i - home) & mask_) >= ((i - hole) & mask_)) {
        entries_[hole] = entries_[i];
        hole = i;
      }
    }
    entries_[hole] = kEmpty;
#endif
  }
  void Clear(const std::vector<tableint> &owners, size_t count) {
#ifdef ALLIANCE_DENSE_DYNAMIC_CACHE
    if (count < entries_.size() / (64 / sizeof(tableint))) {
      for (size_t i = 0; i < count; ++i)
        entries_[owners[i]] = kInvalid;
    } else
      std::fill(entries_.begin(), entries_.end(), kInvalid);
#else
    std::fill(entries_.begin(), entries_.end(), kEmpty);
    (void)owners;
    (void)count;
#endif
  }
  size_t Bytes() const {
    return entries_.capacity() * sizeof(entries_[0]);
  }

private:
#ifdef ALLIANCE_DENSE_DYNAMIC_CACHE
  std::vector<tableint> entries_;
#else
  static constexpr uint64_t kEmpty = std::numeric_limits<uint64_t>::max();
  static tableint Key(uint64_t entry) {
    return static_cast<tableint>(entry >> 32);
  }
  static uint32_t Hash(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    return x ^ (x >> 16);
  }
  std::vector<uint64_t> entries_;
  size_t mask_{0};
#endif
};
} // namespace alliance
