#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace alliance {
// IDs must be unique. Range/global sampling uses (attribute, label) order;
// guided sampling supplies a preselected list and scores every entry.
// Score only eligible samples. Unselected samples are NOT marked visited:
// graph traversal must still be able to discover and admit them later.
template <class Id, class Accept, class Score>
auto SelectRankEntries(const std::vector<Id> &ids, size_t lo, size_t hi, size_t samples,
                       size_t keep, Accept accept, Score score) {
  using Distance = decltype(score(Id{}));
  std::vector<std::pair<Distance, Id>> candidates;
  if (lo > hi || hi > ids.size())
    throw std::out_of_range("invalid entry rank range");
  const size_t span = hi - lo, count = std::min(samples, span);
  if (!count || !keep)
    return candidates;
  candidates.reserve(count);
  // Index IDs and cardinality are uint32_t; the intermediate fits uint64_t.
  for (size_t j = 0; j < count; ++j) {
    const size_t rank = lo + (uint64_t(j) * span + span / 2) / count;
    const Id id = ids[rank];
    if (accept(id))
      candidates.emplace_back(score(id), id);
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });
  if (candidates.size() > keep)
    candidates.resize(keep);
  return candidates;
}
} // namespace alliance
