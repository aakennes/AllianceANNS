#pragma once

#include <array>
#include <cstdint>

namespace alliance::shuffle_detail {

// A negative-only prefilter for membership in an immutable adjacency list.
// Two 64-bit lanes cost 16 bytes/node. A positive still requires exact lookup.
class NeighborMembershipFilter {
public:
  using Mask = std::array<uint64_t, 2>;

  static Mask MaskFor(uint64_t id) noexcept {
    // Unsigned mixing is deterministic, including at the ID-domain boundaries.
    id += UINT64_C(0x9e3779b97f4a7c15);
    id = (id ^ (id >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    id = (id ^ (id >> 27)) * UINT64_C(0x94d049bb133111eb);
    id ^= id >> 31;
    return {UINT64_C(1) << (id & 63), UINT64_C(1) << ((id >> 32) & 63)};
  }

  void Add(uint64_t id) noexcept {
    const auto mask = MaskFor(id);
    words_[0] |= mask[0];
    words_[1] |= mask[1];
  }

  bool MayContain(const Mask &mask) const noexcept {
    return (words_[0] & mask[0]) != 0 && (words_[1] & mask[1]) != 0;
  }

private:
  Mask words_{};
};

static_assert(sizeof(NeighborMembershipFilter) == 16, "neighbor filter memory budget");

} // namespace alliance::shuffle_detail
