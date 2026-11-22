#pragma once
#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace alliance {
struct WorkerSearchIOBudget {
  size_t share{0};
  size_t search{0};
  size_t rerank{0};
  size_t queue_depth{0};
};

// Fixed shares avoid synchronization on submission/completion. Traversal and
// final rerank run sequentially within each worker; unused shares are not borrowed.
// Total is a process request cap, not bytes/s or a device bandwidth limit.
inline std::vector<WorkerSearchIOBudget> PlanSearchIOBudget(size_t threads, size_t search,
                                                            size_t rerank, size_t total) {
  if (!threads || threads > static_cast<size_t>(std::numeric_limits<int>::max()) || !search ||
      search > 64 || !rerank || rerank > 256)
    throw std::invalid_argument("invalid search threads or beam widths");
  if (total && total / threads < 1)
    throw std::invalid_argument("io_budget_total needs at least one request per worker");
  std::vector<WorkerSearchIOBudget> plans;
  plans.reserve(threads);
  for (size_t t = 0; t < threads; ++t) {
    WorkerSearchIOBudget p;
    if (!total) {
      // Keep existing non-budgeted context allocation and effective phase widths.
      p.search = search;
      p.rerank = rerank;
      p.queue_depth = search + rerank;
    } else {
      p.share = total / threads + (t < total % threads);
      p.search = std::min(search, p.share);
      p.rerank = std::min(rerank, p.share);
      p.queue_depth = std::max(p.search, p.rerank);
    }
    plans.push_back(p);
  }
  return plans;
}
} // namespace alliance
