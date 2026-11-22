#pragma once
#include "core/types.hpp"
#include <algorithm>
#include <vector>

namespace benchmark {
using SearchResults = std::vector<std::vector<alliance::label_t>>;

// Keep label extraction after the per-query timer, as in the original driver.
template <typename Results>
inline void AppendResultLabels(const Results &source, std::vector<alliance::label_t> &labels) {
  for (const auto &result : source)
    labels.emplace_back(result.second);
}

// Historical recall counts result occurrences; unique recall is reported
// separately by the ML benchmark. Preserve that distinction for old drivers.
inline auto CalculateRecall(const std::vector<std::vector<alliance::label_t>> &gt,
                            const std::vector<std::vector<alliance::label_t>> &res) -> float {
  size_t n = std::min(gt.size(), res.size());
  size_t total = 0;
  size_t correct = 0;
  for (size_t i = 0; i < n; ++i) {
    total += gt[i].size();
    for (auto ib : res[i]) {
      if (std::find(gt[i].begin(), gt[i].end(), ib) != gt[i].end()) {
        correct++;
      }
    }
  }
  return static_cast<float>(correct) / total;
}

} // namespace benchmark
