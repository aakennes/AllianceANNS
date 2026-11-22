#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
namespace alliance::detail {
class RefinementScoreCache {
  struct Slot {
    double value{};
    uint32_t id{};
    uint8_t cid{};
    bool occupied{false};
  };
  std::vector<Slot> slots_;
  size_t Hash(uint32_t id, uint8_t cid) const {
    return (size_t(id) * 2654435761u + cid) & (slots_.size() - 1);
  }

public:
  void Reset(size_t count) {
    slots_.clear();
    if (!count)
      return;
    if (count > std::numeric_limits<size_t>::max() / 4)
      throw std::length_error("refinement cache too large");
    size_t size = 1;
    while (size < count * 2)
      size *= 2;
    slots_.resize(size);
  }
  void Insert(uint32_t id, uint8_t cid, double value) {
    if (slots_.empty() || !std::isfinite(value))
      return;
    size_t pos = Hash(id, cid);
    for (size_t n = 0; n < slots_.size(); ++n, pos = (pos + 1) & (slots_.size() - 1)) {
      auto &s = slots_[pos];
      if (!s.occupied || (s.id == id && s.cid == cid)) {
        s = {value, id, cid, true};
        return;
      }
    }
    throw std::logic_error("refinement cache capacity exceeded");
  }
  bool Find(uint32_t id, uint8_t cid, float rounded, double &value) const {
    if (slots_.empty())
      return false;
    size_t pos = Hash(id, cid);
    for (size_t n = 0; n < slots_.size(); ++n, pos = (pos + 1) & (slots_.size() - 1)) {
      const auto &s = slots_[pos];
      if (!s.occupied)
        return false;
      if (s.id == id && s.cid == cid) {
        if (static_cast<float>(s.value) != rounded)
          return false;
        value = s.value;
        return true;
      }
    }
    return false;
  }
};
} // namespace alliance::detail
