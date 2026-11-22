#pragma once
// Query policy shared by CLI and native drivers. Does not own file handles or scratch.
#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace alliance {
struct DiskSearchPolicy {
  bool top_layer_postfilter{true};
  // Supplemental graph entries; independent of k, ef and rerank policy.
  // Zero samples uses only the layer-routing entries. Guided reuses centroid scores.
  enum class EntrySampling { Global, Range, Guided };
  EntrySampling entry_sampling{EntrySampling::Guided};
  size_t entry_samples{256};
  size_t entry_count{4};
  // Optional RAM graph navigation replaces all ordinary/sampled entry points.
  // These limits affect entry selection only, not disk traversal ef or rerank.
  size_t navigation_ef{64};
  size_t navigation_budget{256};
  size_t navigation_count{16};
  enum class FilterMode { Legacy, Graph, Prefilter, Auto };
  // Runtime policy, independent of build windows and serialized index format.
  FilterMode filter_mode{FilterMode::Legacy};
  size_t prefilter_threshold{8192}; // Auto: inclusive, actual range cardinality.
  // Zero selects max(k, efs * max(1, rerank_candidate_factor)), capped by
  // range cardinality. An explicit nonzero budget must be at least k.
  size_t prefilter_rerank{0};
  size_t edge_beam_width{8};
  size_t vec_beam_width{8};
  // Depth both io_contexts were io_setup() with; hard upper bound for each
  // sequential traversal or final-rerank phase.
  size_t queue_depth{64};
  // Per-worker rerank in-flight cap; zero uses queue_depth. This limits requests,
  // not bandwidth. With bounds enabled, completion order changes when the exact
  // top-k threshold tightens and can change how many later blocks are read.
  size_t rerank_beam_width{0};
  size_t RerankWidth() const {
    return rerank_beam_width != 0 ? std::min(rerank_beam_width, queue_depth) : queue_depth;
  }
  // Graph-only budget. 0 retains the index multiplier (normally 8*k).
  // Explicit budgets must be >=k. An ef-independent reserve retains
  // rejected/evicted candidates; actual count is limited by unique discoveries.
  size_t rerank_num{0};
  // Per-query k multiplier; zero uses the index default. Mutually exclusive
  // with an explicit rerank_num. Changing k recalculates the budget.
  size_t rerank_factor{0};
  enum class GraphRerankPolicy { Fixed, Step, Linear };
  GraphRerankPolicy rerank_policy{GraphRerankPolicy::Fixed};

  size_t GraphRerankBudget(size_t k, size_t default_multiplier = 8) const {
    if (rerank_policy != GraphRerankPolicy::Fixed)
      throw std::invalid_argument("dynamic graph budget requires query efs");
    if (rerank_num != 0) {
      if (rerank_factor != 0)
        throw std::invalid_argument("choose rerank_num or rerank_factor, not both");
      if (rerank_num < k)
        throw std::invalid_argument("graph rerank_num must be 0 (automatic) or >= k");
      return rerank_num;
    }
    if (rerank_factor != 0)
      return SaturatingProduct(k, rerank_factor);
    return SaturatingProduct(k, std::max(size_t{2}, default_multiplier));
  }

  // Shared by candidate retention, exact rerank and benchmark reporting.
  // Thresholds scale with the current query k; traversal width stays at efs.
  size_t GraphRerankBudgetForSearch(size_t k, size_t efs, size_t default_multiplier = 8) const {
    if (rerank_candidate_factor != 0) {
      if (rerank_policy != GraphRerankPolicy::Fixed || rerank_num || rerank_factor)
        throw std::invalid_argument(
            "candidate factor cannot combine with a fixed/dynamic rerank budget");
      return std::max(k, SaturatingProduct(efs, rerank_candidate_factor));
    }
    if (rerank_policy == GraphRerankPolicy::Fixed)
      return GraphRerankBudget(k, default_multiplier);
    if (k == 0 || efs == 0 || rerank_num != 0 || rerank_factor != 0)
      throw std::invalid_argument(
          "dynamic rerank requires positive k/efs and no fixed budget override");
    const size_t lower = SaturatingProduct(k, 2), upper = SaturatingProduct(k, 4);
    if (rerank_policy == GraphRerankPolicy::Step) {
      if (efs <= lower)
        return lower;
      return efs <= SaturatingProduct(k, 8) ? SaturatingProduct(k, 3) : upper;
    }
    if (rerank_policy != GraphRerankPolicy::Linear)
      throw std::invalid_argument("unknown graph rerank policy");
    if (efs <= lower)
      return lower;
    // ceil(k * [2 + 2*clip((efs/k - 2)/14, 0, 1)])
    // = clamp(2*k + ceil((efs - 2*k)/7), 2*k, 4*k).
    // Integer arithmetic avoids rounding errors at thresholds and overflow.
    const size_t delta = efs - lower;
    const size_t increment = delta / 7 + (delta % 7 != 0);
    return increment >= upper - lower ? upper : lower + increment;
  }

  static size_t SaturatingProduct(size_t a, size_t b) {
    const size_t limit = std::numeric_limits<size_t>::max();
    return b != 0 && a > limit / b ? limit : a * b;
  }
  // Candidate retention is independent of actual vector reads. Nonzero keeps
  // max(k, factor*ef) approximate candidates; bound-off is the exact oracle for
  // the same pool. Zero selects the fixed/step/linear budget policy above.
  size_t rerank_candidate_factor{0};
  // Upper-bound rank prescreen, then pruning against completed exact top-k.
  bool rerank_bound_prescreen{false};
  size_t rerank_bound_rank_factor{1}; // upper rank=max(k, factor*k)
  size_t rerank_warm_blocks{0};       // 0: full width before exact top-k fills
  // Probabilistic RaBitQ bound with the unscaled library model error.
  bool rerank_float_refine{true};     // Recompute ambiguous bounds using the float query.
  bool rerank_reuse_estimates{false}; // Reuse retained precise scores during bound refinement.
  bool rerank_error_bound{false};
  size_t rerank_batch_blocks{32};
};
} // namespace alliance
