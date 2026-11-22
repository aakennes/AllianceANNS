#pragma once
#include "core/types.hpp"
#include <bit>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace alliance::shuffle_detail {
// Private, temporary representation only. Persistent graph/layout bytes do not change.
class EdgeCodec {
public:
  EdgeCodec(size_t nodes, uint8_t maximum_weight) : nodes_(nodes), maximum_(maximum_weight) {
    static_assert(sizeof(tableint) == 4);
    if (!nodes || nodes - 1 > UINT32_MAX || !maximum_)
      throw std::invalid_argument("invalid weighted-edge geometry");
    bits_ = std::bit_width(static_cast<uint32_t>(nodes - 1));
    mask_ = bits_ == 32 ? UINT32_MAX : uint32_t((uint64_t(1) << bits_) - 1);
    packed_ = uint64_t(maximum_) <= (uint64_t(1) << (32 - bits_));
  }
  bool Packed() const {
    return packed_;
  }
  size_t Bytes() const {
    return packed_ ? 4 : 5;
  }
  void Check(tableint id, uint8_t weight) const {
    if (id >= nodes_ || !weight || weight > maximum_)
      throw std::invalid_argument("weighted edge exceeds counted geometry");
  }
  uint32_t Word(tableint id, uint8_t weight) const {
    Check(id, weight);
    return packed_ ? id | uint32_t(uint64_t(weight - 1) << bits_) : id;
  }
  tableint Id(uint32_t word) const {
    return packed_ ? word & mask_ : word;
  }
  uint8_t Weight(uint32_t word) const {
    return uint8_t((uint64_t(word) >> bits_) + 1);
  }
  void Encode(uint8_t *out, tableint id, uint8_t weight) const {
    const uint32_t word = Word(id, weight);
    std::memcpy(out, &word, sizeof(word));
    if (!packed_)
      out[4] = weight;
  }
  tableint Id(const uint8_t *data) const {
    uint32_t word;
    std::memcpy(&word, data, 4);
    return Id(word);
  }
  uint8_t Weight(const uint8_t *data) const {
    if (!packed_)
      return data[4];
    uint32_t word;
    std::memcpy(&word, data, 4);
    return Weight(word);
  }

private:
  size_t nodes_;
  uint8_t maximum_;
  unsigned bits_;
  uint32_t mask_;
  bool packed_;
};

struct EncodedRow {
  const uint8_t *data{};
  size_t size{};
  const EdgeCodec *codec{};
  tableint Neighbor(size_t i) const {
    return codec->Id(data + i * codec->Bytes());
  }
  uint8_t Weight(size_t i) const {
    return codec->Weight(data + i * codec->Bytes());
  }
};
} // namespace alliance::shuffle_detail
