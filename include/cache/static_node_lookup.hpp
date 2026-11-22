#pragma once
#include "cache/static_layer_cache.hpp"
#include <algorithm>
#include <array>
#include <istream>
#include <limits>
#include <stdexcept>

namespace alliance {
// Immutable bitmap/rank lookup. v2 retains 16-byte query metadata; v3 uses wide offsets only when
// needed. Two streaming passes avoid allocating a temporary metadata array of N nodes.
class StaticNodeLookup {
public:
  static size_t MetadataBytes(size_t nodes, size_t cached, bool wide = false) {
    if (!cached)
      return 0;
    const size_t words = nodes / 64 + (nodes % 64 != 0);
    return words * (sizeof(uint64_t) + sizeof(uint32_t)) +
           cached * (wide ? sizeof(StaticNodeMeta64) : sizeof(StaticNodeMeta));
  }
  static size_t BudgetBytes(size_t nodes, size_t cached, size_t payload, bool wide = false) {
    if (!cached)
      return 0;
    if (payload > std::numeric_limits<size_t>::max() - 4095)
      throw std::length_error("static payload overflow");
    return ((payload + 4095) & ~size_t(4095)) + MetadataBytes(nodes, cached, wide);
  }
  void Clear() {
    std::vector<uint64_t>().swap(bits_);
    std::vector<uint32_t>().swap(rank_);
    std::vector<StaticNodeMeta>().swap(meta_);
    std::vector<StaticNodeMeta64>().swap(wide_meta_);
    wide_ = false;
    nodes_ = 0;
  }
  bool Wide() const {
    return wide_;
  }
  void LoadDense(std::istream &stream, size_t nodes, size_t payload,
                 uint8_t version = kStaticCacheV2) {
    Clear();
    if (version != kStaticCacheV2 && version != kStaticCacheV3)
      throw std::runtime_error("unsupported static cache version");
    wide_ = version == kStaticCacheV3;
    if (wide_)
      LoadDenseImpl(stream, nodes, payload, wide_meta_);
    else
      LoadDenseImpl(stream, nodes, payload, meta_);
  }

private:
  template <class Meta>
  void LoadDenseImpl(std::istream &stream, size_t nodes, size_t payload, std::vector<Meta> &meta) {
    if (nodes > UINT32_MAX)
      throw std::length_error("static node count exceeds ID range");
    const auto start = stream.tellg();
    if (start == std::streampos(-1))
      throw std::runtime_error("static metadata requires seekable input");
    nodes_ = nodes;
    bits_.assign(nodes / 64 + (nodes % 64 != 0), 0);
    size_t count = 0;
    VisitDense<Meta>(stream, nodes, [&](size_t id, const Meta &m) {
      if constexpr (std::is_same_v<Meta, StaticNodeMeta64>)
        if (m.reserved)
          throw std::runtime_error("nonzero static metadata reserved field");
      if (!m.IsCached())
        return;
      if (m.layer_begin < 0 || m.layer_end < m.layer_begin || !m.record_bytes ||
          m.offset > payload || m.record_bytes > payload - m.offset)
        throw std::runtime_error("invalid static node metadata");
      bits_[id / 64] |= uint64_t(1) << (id % 64);
      ++count;
    });
    if (!count) {
      Clear();
      if (payload)
        throw std::runtime_error("static payload has no owners");
      return;
    }
    rank_.resize(bits_.size());
    uint32_t prefix = 0;
    for (size_t i = 0; i < bits_.size(); ++i) {
      rank_[i] = prefix;
      prefix += __builtin_popcountll(bits_[i]);
    }
    meta.resize(count);
    stream.seekg(start);
    size_t pos = 0;
    VisitDense<Meta>(stream, nodes, [&](size_t, const Meta &m) {
      if (m.IsCached())
        meta[pos++] = m;
    });
    if (pos != count)
      throw std::runtime_error("static metadata changed while loading");
  }

public:
  const StaticNodeMeta *Find(size_t id) const {
    return wide_ ? nullptr : FindImpl(id, meta_);
  }
  const StaticNodeMeta64 *Find64(size_t id) const {
    return wide_ ? FindImpl(id, wide_meta_) : nullptr;
  }
  bool CoversRange(size_t id, layer_t lo, layer_t hi) const {
    if (wide_) {
      const auto *m = Find64(id);
      return m && m->layer_begin <= lo && m->layer_end >= hi;
    }
    const auto *m = Find(id);
    return m && m->layer_begin <= lo && m->layer_end >= hi;
  }

private:
  template <class Meta> const Meta *FindImpl(size_t id, const std::vector<Meta> &meta) const {
    if (id >= nodes_ || bits_.empty())
      return nullptr;
    const size_t w = id / 64;
    const uint64_t bit = uint64_t(1) << (id % 64);
    if (!(bits_[w] & bit))
      return nullptr;
    return &meta[rank_[w] + __builtin_popcountll(bits_[w] & (bit - 1))];
  }

public:
  size_t Bytes() const {
    return bits_.capacity() * sizeof(uint64_t) + rank_.capacity() * sizeof(uint32_t) +
           meta_.capacity() * sizeof(StaticNodeMeta) +
           wide_meta_.capacity() * sizeof(StaticNodeMeta64);
  }
  size_t Count() const {
    return wide_ ? wide_meta_.size() : meta_.size();
  }

private:
  template <class Meta, class F> static void VisitDense(std::istream &stream, size_t nodes, F f) {
    std::array<Meta, 512> buf;
    for (size_t start = 0; start < nodes; start += buf.size()) {
      const size_t count = std::min(buf.size(), nodes - start);
      stream.read(reinterpret_cast<char *>(buf.data()), count * sizeof(Meta));
      if (!stream.good())
        throw std::runtime_error("truncated static node metadata");
      for (size_t i = 0; i < count; ++i)
        f(start + i, buf[i]);
    }
  }
  size_t nodes_{0};
  std::vector<uint64_t> bits_;
  std::vector<uint32_t> rank_;
  std::vector<StaticNodeMeta> meta_;
  std::vector<StaticNodeMeta64> wide_meta_;
  bool wide_{false};
};
} // namespace alliance
