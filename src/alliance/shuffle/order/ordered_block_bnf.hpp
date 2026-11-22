#pragma once
// Deterministic BlockBNF: parallel immutable-map scoring, ordered capacity commit.
#include "shuffle/adjacency/weighted_graph.hpp"
#include "shuffle/adjacency/neighbor_filter.hpp"
#include "shuffle/memory/sparse_scores.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <omp.h>

namespace alliance::shuffle_detail {

inline double ShuffleSeconds() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

class OrderedBlockBNF {
  // Production policy: require improvement beyond one affinity unit.
  static constexpr size_t kMoveMargin = 1;
  struct Score {
    size_t block, value;
  };
  struct Worker {
    SparseScores scores;
    std::vector<size_t> counts, cursor;
    Worker(size_t degree, size_t max_score)
        : scores(degree), counts(max_score + 1, 0), cursor(max_score + 1, 0) {
    }
  };
  const FlatWeightedGraph &graph_;
  size_t epb_, batch_, max_degree_{0}, max_score_{0};
  int threads_;
  std::vector<tableint> nodes_;
  std::vector<size_t> used_, next_, lengths_, stay_scores_;
  std::vector<Score> rows_;
  std::vector<NeighborMembershipFilter> neighbor_filters_;
  std::vector<Worker> workers_;

public:
  OrderedBlockBNF(const FlatWeightedGraph &graph, size_t epb, const BNFParams &params)
      : graph_(graph), epb_(epb), batch_(std::min(params.affinity_batch_size, graph.num_nodes)),
        threads_(params.threads), nodes_(graph.num_nodes),
        used_((graph.num_nodes + epb - 1) / epb, 0), next_(graph.num_nodes) {
    if (threads_ < 1 || !batch_)
      throw std::invalid_argument("BlockBNF requires positive threads/batch");
    // Immutable graph summaries: compute once per level, not every iteration.
    if (graph.external) {
      max_degree_ = graph.external->MaxDegree();
      max_score_ = graph.external->MaxScore();
    } else
      for (size_t u = 0; u < graph.num_nodes; ++u) {
        max_degree_ = std::max(max_degree_, graph.offsets[u + 1] - graph.offsets[u]);
        size_t value = 0;
        for (size_t e = graph.offsets[u]; e < graph.offsets[u + 1]; ++e)
          value += size_t(graph.edges.Weight(e)) * graph.edges.Weight(e);
        max_score_ = std::max(max_score_, value);
      }
    if (max_degree_ && batch_ > std::numeric_limits<size_t>::max() / max_degree_)
      throw std::overflow_error("BlockBNF score batch overflow");
    rows_.resize(batch_ * max_degree_);
    lengths_.resize(batch_);
    stay_scores_.resize(batch_);
    if (!graph.external) {
      neighbor_filters_.resize(graph.num_nodes);
// Graph membership is immutable across rounds. Each worker owns a row;
// negatives can skip exact reciprocal lookups without changing any tie.
#pragma omp parallel for num_threads(threads_) schedule(static) if (threads_ > 1)
      for (size_t u = 0; u < graph.num_nodes; ++u)
        for (size_t e = graph.offsets[u]; e < graph.offsets[u + 1]; ++e)
          neighbor_filters_[u].Add(graph.edges.Neighbor(e));
    }
    workers_.reserve(threads_);
    for (int t = 0; t < threads_; ++t)
      workers_.emplace_back(max_degree_, max_score_);
    std::iota(nodes_.begin(), nodes_.end(), tableint(0));
    for (size_t b = 0; b < used_.size(); ++b)
      used_[b] = Capacity(b);
    fprintf(
        stderr,
        "  [shuffle-workspace] threads=%d batch=%zu max_degree=%zu max_score=%zu score_rows_MiB=%.2f worker_scores_MiB=%.2f neighbor_filters_MiB=%.2f stay_scores_KiB=%.2f\n",
        threads_, batch_, max_degree_, max_score_, rows_.size() * sizeof(Score) / 1048576.0,
        double(threads_) * workers_.front().scores.StorageBytes() / 1048576.0,
        graph.num_nodes * sizeof(NeighborMembershipFilter) / 1048576.0,
        stay_scores_.size() * sizeof(size_t) / 1024.0);
  }

  size_t Capacity(size_t b) const {
    return std::min(epb_, graph_.num_nodes - b * epb_);
  }

  size_t Round(const std::vector<size_t> &order, std::vector<size_t> &old_map, size_t iteration) {
    const double start = ShuffleSeconds();
    std::fill(used_.begin(), used_.end(), 0);
    size_t first_available = 0, ties = 0, moved = 0;
    size_t filter_rejected = 0, exact_checks = 0;
    double score_seconds = 0, commit_seconds = 0, read_seconds = 0;
    for (size_t base = 0; base < order.size();) {
      size_t count = std::min(batch_, order.size() - base);
      if (graph_.external) {
        const double read_start = ShuffleSeconds();
        count = graph_.external->LoadBatch(base, count);
        read_seconds += ShuffleSeconds() - read_start;
      }
      const double score_start = ShuffleSeconds();
// Workers only read old_map. The ordered commit below is its sole consumer;
// old_map is swapped only after the full round finishes.
#pragma omp parallel for num_threads(threads_) schedule(static) if (threads_ > 1)
      for (size_t i = 0; i < count; ++i) {
        auto &w = workers_[omp_get_thread_num()];
        const size_t u = order[base + i];
        size_t row_max = 0;
        const auto row = graph_.Row(u, base + i);
        for (size_t e = 0; e < row.size; ++e) {
          const size_t b = old_map[row.Neighbor(e)];
          const size_t weight = row.Weight(e);
          row_max = std::max(row_max, w.scores.Add(b, weight * weight));
        }
        // Save before touched scores are reset. An untouched old block is zero.
        stay_scores_[i] = w.scores.Get(old_map[u]);
        std::fill(w.counts.begin(), w.counts.begin() + row_max + 1, 0);
        for (size_t slot : w.scores.Touched())
          ++w.counts[w.scores.Value(slot)];
        size_t offset = 0;
        for (size_t s = row_max; s > 0; --s) {
          w.cursor[s] = offset;
          offset += w.counts[s];
        }
        // Stable counting order exactly matches legacy score_buckets: scores
        // descending, first neighbor occurrence order within equal scores.
        for (size_t slot : w.scores.Touched()) {
          const size_t score = w.scores.Value(slot);
          rows_[i * max_degree_ + w.cursor[score]++] = {w.scores.Key(slot), score};
        }
        lengths_[i] = offset;
        w.scores.Clear();
      }
      score_seconds += ShuffleSeconds() - score_start;
      const double commit_start = ShuffleSeconds();
      for (size_t i = 0; i < count; ++i) {
        const size_t u = order[base + i];
        const size_t old_block = old_map[u];
        size_t chosen = used_.size(), chosen_score = 0;
        const size_t stay_score = stay_scores_[i];
        const Score *row = max_degree_ ? rows_.data() + i * max_degree_ : nullptr;
        for (size_t begin = 0; begin < lengths_[i];) {
          size_t end = begin + 1;
          while (end < lengths_[i] && row[end].value == row[begin].value)
            ++end;
          size_t available = 0;
          for (size_t j = begin; j < end; ++j) {
            const size_t b = row[j].block;
            if (used_[b] < Capacity(b)) {
              if (!available)
                chosen = b;
              ++available;
            }
          }
          if (available) {
            chosen_score = row[begin].value;
            if (available > 1) {
              ++ties;
              const auto query_mask = NeighborMembershipFilter::MaskFor(u);
              size_t best_recip = 0;
              for (size_t j = begin; j < end; ++j) {
                const size_t b = row[j].block;
                if (used_[b] >= Capacity(b))
                  continue;
                size_t reciprocal = 0;
                for (size_t k = 0; k < used_[b]; ++k) {
                  const tableint v = nodes_[b * epb_ + k];
                  const bool may_contain = graph_.external
                                               ? graph_.external->MayContain(v, query_mask)
                                               : neighbor_filters_[v].MayContain(query_mask);
                  if (!may_contain) {
                    ++filter_rejected;
                    continue;
                  }
                  ++exact_checks;
                  reciprocal += graph_.external ? graph_.external->ContainsIncoming(base + i, v)
                                                : graph_.edges.Contains(graph_.offsets[v],
                                                                        graph_.offsets[v + 1],
                                                                        static_cast<tableint>(u));
                }
                if (reciprocal > best_recip) {
                  best_recip = reciprocal;
                  chosen = b;
                }
              }
            }
            break;
          }
          begin = end;
        }
        if (chosen < used_.size() && chosen != old_block &&
            used_[old_block] < Capacity(old_block) && chosen_score <= stay_score + kMoveMargin)
          chosen = old_block;
        if (chosen == used_.size() && used_[old_block] < Capacity(old_block))
          chosen = old_block;
        if (chosen == used_.size()) {
          while (first_available < used_.size() &&
                 used_[first_available] == Capacity(first_available))
            ++first_available;
          chosen = first_available;
        }
        if (chosen >= used_.size())
          throw std::runtime_error("BlockBNF assignment: all blocks full");
        nodes_[chosen * epb_ + used_[chosen]++] = static_cast<tableint>(u);
        next_[u] = chosen;
        moved += chosen != old_block;
      }
      commit_seconds += ShuffleSeconds() - commit_start;
      base += count;
    }
    old_map.swap(next_);
    fprintf(
        stderr,
        "  [shuffle-timing] iteration=%zu score_s=%.6f commit_s=%.6f total_s=%.6f ties=%zu threads=%d filter_rejected=%zu exact_checks=%zu read_s=%.6f\n",
        iteration, score_seconds, commit_seconds, ShuffleSeconds() - start, ties, threads_,
        filter_rejected, exact_checks, read_seconds);
    return moved;
  }

  std::vector<tableint> TakeOrder() {
    return std::move(nodes_);
  }
};

} // namespace alliance::shuffle_detail
