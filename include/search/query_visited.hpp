#pragma once
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace alliance {
// Exact, thread-private visited set. Each word is added to dirty_ at most once
// between Clears. No individual Reset is exposed, preserving that invariant.
class QueryVisited {
public:
  explicit QueryVisited(size_t n) : words_(n / 64 + (n % 64 != 0), 0) {
    if (words_.size() > UINT32_MAX)
      throw std::length_error("QueryVisited word index overflow");
    dirty_.reserve(words_.size());
  }
  void Set(size_t id) {
    const size_t i = id >> 6;
    if (words_[i] == 0)
      dirty_.push_back(static_cast<uint32_t>(i));
    words_[i] |= uint64_t(1) << (id & 63);
  }
  bool Test(size_t id) const {
    return (words_[id >> 6] & (uint64_t(1) << (id & 63))) != 0;
  }
  void Clear() {
    for (uint32_t i : dirty_)
      words_[i] = 0;
    dirty_.clear();
  }
  size_t Bytes() const {
    return words_.capacity() * sizeof(uint64_t) + dirty_.capacity() * sizeof(uint32_t);
  }

private:
  std::vector<uint64_t> words_;
  std::vector<uint32_t> dirty_;
};
} // namespace alliance
