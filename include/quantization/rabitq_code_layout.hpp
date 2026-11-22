#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include "quantization/external_rabitq.hpp"

namespace alliance::detail {
enum class RaBitQCodeLayout : uint8_t {
  BitPlanes = 0,
  Compact2 = 2,
  Compact3 = 3,
  Compact4 = 4,
  Compact8 = 8
};

inline RaBitQCodeLayout NativeCodeLayout(unsigned bits) {
  switch (bits) {
  case 2:
    return RaBitQCodeLayout::Compact2;
  case 3:
    return RaBitQCodeLayout::Compact3;
  case 4:
    return RaBitQCodeLayout::Compact4;
  case 8:
    return RaBitQCodeLayout::Compact8;
  default:
    return RaBitQCodeLayout::BitPlanes;
  }
}
inline const char *CodeLayoutName(RaBitQCodeLayout layout) {
  switch (layout) {
  case RaBitQCodeLayout::Compact2:
    return "external_v071_compact2";
  case RaBitQCodeLayout::Compact3:
    return "external_v071_compact3";
  case RaBitQCodeLayout::Compact4:
    return "external_v071_compact4";
  case RaBitQCodeLayout::Compact8:
    return "external_v071_compact8";
  default:
    return "canonical_bitplanes";
  }
}

// Canonical checkpoints retain little-bit-order planes and 64-coordinate
// padding. Convert only in memory; v0.7.1 compact2 differs from the old WoW layout.
inline void PackCompact(const uint8_t *planes, uint8_t *compact, size_t bytes,
                        RaBitQCodeLayout layout) {
  const unsigned bits = static_cast<unsigned>(layout);
  const size_t plane_bytes = bytes / bits;
  for (size_t base = 0; base < plane_bytes * 8; base += 64) {
    uint8_t raw[64]{};
    for (size_t j = 0; j < 64; ++j)
      for (unsigned p = 0; p < bits; ++p)
        raw[j] |= ((planes[p * plane_bytes + (base + j) / 8] >> (j % 8)) & 1) << p;
    alliance_rabitq_pack(raw, compact + base * bits / 8, 64, bits);
  }
}
inline void UnpackCompact(const uint8_t *compact, uint8_t *planes, size_t bytes,
                          RaBitQCodeLayout layout) {
  const unsigned bits = static_cast<unsigned>(layout);
  const size_t plane_bytes = bytes / bits;
  std::memset(planes, 0, bytes);
  for (size_t base = 0; base < plane_bytes * 8; base += 64) {
    uint8_t raw[64];
    alliance_rabitq_unpack(compact + base * bits / 8, raw, 64, bits);
    for (size_t j = 0; j < 64; ++j)
      for (unsigned p = 0; p < bits; ++p)
        planes[p * plane_bytes + (base + j) / 8] |= ((raw[j] >> p) & 1) << (j % 8);
  }
}
inline void PackCompact3(const uint8_t *planes, uint8_t *compact, size_t bytes) {
  PackCompact(planes, compact, bytes, RaBitQCodeLayout::Compact3);
}
inline void UnpackCompact3(const uint8_t *compact, uint8_t *planes, size_t bytes) {
  UnpackCompact(compact, planes, bytes, RaBitQCodeLayout::Compact3);
}
} // namespace alliance::detail
