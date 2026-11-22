#pragma once

#include <algorithm>
#include <functional>
#include <cmath>
#include <cstddef>
#include <limits>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace alliance {
namespace detail {

struct BoundedRerankCandidate {
  size_t id;
  size_t block;
  float estimate;
  float lower_bound;
  float upper_bound{std::numeric_limits<float>::infinity()};
};

struct BoundedRerankBlock {
  size_t block;
  std::vector<BoundedRerankCandidate> candidates;
};

struct BoundedRerankStats {
  size_t pruned{0};
  size_t blocks{0};
  size_t prescreen_pruned{0};
};

// Strict comparison retains distance ties; invalid bounds always cause a read.
inline bool RerankBoundCanPrune(float lower_bound, size_t exact_count, size_t k, float threshold) {
  return k > 0 && exact_count >= k && std::isfinite(lower_bound) && std::isfinite(threshold) &&
         lower_bound > threshold;
}

// kth upper bound is an upper bound on the true kth distance among this
// candidate set. Unknown upper
// bounds count as +inf; they must not be omitted from the rank calculation.
inline float RerankUpperThreshold(const std::vector<BoundedRerankCandidate> &candidates,
                                  size_t rank) {
  const float inf = std::numeric_limits<float>::infinity();
  if (!rank || rank > candidates.size())
    return inf;
  std::vector<float> upper;
  upper.reserve(candidates.size());
  for (const auto &c : candidates)
    upper.push_back(std::isfinite(c.upper_bound) ? c.upper_bound : inf);
  std::nth_element(upper.begin(), upper.begin() + rank - 1, upper.end());
  return upper[rank - 1];
}

// Candidates arrive sorted and unique from the final candidate pool. Group
// once, without sorting or building a second ID set. Submit must drain accepted
// IO before throwing.
// Poll calls complete(slot, success, exact_results) for finished requests.
// A failed read stops new submissions and drains all active slots before throw.
template <typename Submit, typename Poll>
BoundedRerankStats RerankAsyncWithBounds(
    const std::vector<BoundedRerankCandidate> &candidates, size_t k, size_t width,
    size_t submit_limit, Submit submit, Poll poll, size_t prescreen_rank = 0, size_t warm_width = 0,
    std::function<std::pair<float, float>(const BoundedRerankCandidate &)> refine = {}) {
  if (!k || !width || !submit_limit)
    throw std::invalid_argument("bounded rerank requires positive k and widths");
  std::priority_queue<float> topk;
  auto add_exact = [&](const std::pair<size_t, float> &r) {
    if (!std::isfinite(r.second))
      return;
    if (topk.size() < k)
      topk.push(r.second);
    else if (r.second < topk.top()) {
      topk.pop();
      topk.push(r.second);
    }
  };
  BoundedRerankStats stats;
  const float upper_tau = RerankUpperThreshold(candidates, prescreen_rank);
  std::unordered_map<size_t, size_t> positions;
  positions.reserve(candidates.size());
  std::vector<BoundedRerankBlock> blocks;
  blocks.reserve(candidates.size());
  for (const auto &c : candidates) {
    if (RerankBoundCanPrune(c.lower_bound, prescreen_rank, k, upper_tau)) {
      ++stats.pruned;
      ++stats.prescreen_pruned;
      continue;
    }
    auto p = positions.emplace(c.block, blocks.size());
    if (p.second)
      blocks.push_back({c.block, {}});
    blocks[p.first->second].candidates.push_back(c);
  }
  std::vector<bool> active(width, false);
  std::vector<std::pair<size_t, const BoundedRerankBlock *>> batch;
  batch.reserve(std::min(width, submit_limit));
  size_t next = 0, pending = 0;
  bool failed = false;
  while (next < blocks.size() || pending) {
    if (!failed) {
      for (size_t slot = 0; slot < width && next < blocks.size(); ++slot) {
        if (active[slot])
          continue;
        if (warm_width && topk.size() < k && pending >= std::min(width, warm_width))
          break;
        while (next < blocks.size()) {
          auto &block = blocks[next++];
          const float exact_tau =
              topk.size() < k ? std::numeric_limits<float>::infinity() : topk.top();
          const float tau = refine ? std::min(exact_tau, upper_tau) : exact_tau;
          const size_t threshold_count = refine && prescreen_rank >= k ? k : topk.size();
          auto end = std::remove_if(block.candidates.begin(), block.candidates.end(), [&](auto &c) {
            // Refine an ambiguous interval once, immediately before submission.
            // Invalid refinement retains the coarse interval and exact-IO fallback.
            if (refine && std::isfinite(tau) && c.lower_bound <= tau && c.upper_bound > tau) {
              const auto interval = refine(c);
              if (std::isfinite(interval.first) && std::isfinite(interval.second) &&
                  interval.first <= interval.second) {
                c.lower_bound = interval.first;
                c.upper_bound = interval.second;
              }
            }
            if (!RerankBoundCanPrune(c.lower_bound, threshold_count, k, tau))
              return false;
            ++stats.pruned;
            return true;
          });
          block.candidates.erase(end, block.candidates.end());
          if (block.candidates.empty())
            continue;
          batch.emplace_back(slot, &block);
          active[slot] = true;
          ++pending;
          ++stats.blocks;
          break;
        }
        if (batch.size() == submit_limit) {
          submit(batch);
          batch.clear();
        }
      }
      if (!batch.empty()) {
        submit(batch);
        batch.clear();
      }
    }
    if (!pending)
      break;
    poll([&](size_t slot, bool success, const auto &exact) {
      // The IO adapter completes each active slot exactly once.
      active[slot] = false;
      --pending;
      if (!success)
        failed = true;
      else
        for (const auto &r : exact)
          add_exact(r);
    });
  }
  if (failed)
    throw std::runtime_error("short or failed read in bounded rerank");
  return stats;
}

} // namespace detail
} // namespace alliance
