#pragma once

#include "shuffle/adjacency/edge_codec.hpp"
#include <bit>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
#include <optional>

namespace alliance::shuffle_detail {

// This is transient shuffle storage, not an on-disk graph format. Spare ID
// bits hold (weight - 1) when the observed maximum fits. Duplicate edges and
// larger node spaces retain the exact legacy uint8 weight through a fallback.
class WeightedEdges {
public:
  void Allocate(size_t nodes, uint8_t maximum_weight, size_t edges) {
    codec_.emplace(nodes, maximum_weight);
    words_.resize(edges);
    if (Packed())
      std::vector<uint8_t>().swap(weights_);
    else
      weights_.resize(edges);
  }

  void Set(size_t position, tableint neighbor, uint8_t weight) {
    words_[position] = codec_->Word(neighbor, weight);
    if (!Packed()) {
      weights_[position] = weight;
    }
  }

  tableint Neighbor(size_t position) const {
    return codec_->Id(words_[position]);
  }

  uint8_t Weight(size_t position) const {
    return Packed() ? codec_->Weight(words_[position]) : weights_[position];
  }

  // Rows remain ordered by decoded neighbor ID. Packed integer order is not
  // ID order because a weight may occupy the most significant bits.
  bool Contains(size_t begin, size_t end, tableint neighbor) const {
    const size_t limit = end;
    while (begin < end) {
      const size_t mid = begin + (end - begin) / 2;
      if (Neighbor(mid) < neighbor)
        begin = mid + 1;
      else
        end = mid;
    }
    return begin < limit && Neighbor(begin) == neighbor;
  }

  bool Packed() const {
    return codec_ && codec_->Packed();
  }
  size_t StorageBytes() const {
    return words_.capacity() * sizeof(tableint) + weights_.capacity();
  }

private:
  std::vector<tableint> words_;
  std::vector<uint8_t> weights_;
  std::optional<EdgeCodec> codec_;
};

} // namespace alliance::shuffle_detail
