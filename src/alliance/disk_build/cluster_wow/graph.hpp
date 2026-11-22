#pragma once
#include "search/query_visited.hpp"
#include "core/types.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <omp.h>
#include <stdexcept>
#include <vector>

namespace alliance::disk_build {
// Compact cluster-local WoW workspace. IDs here are local slots, ranks are
// GLOBAL ranks. Layer 0 is an immutable projection of the completed lower
// graph. All future nodes have a valid seed graph before concurrent refinement
// starts, so there is no publication/readiness race and no reads of unbuilt
// adjacency.
template <class Distance> class ClusterWowGraph {
public:
  using Edge = dist_id_pair;
  struct Candidate {
    float distance;
    uint32_t id;
    bool expanded;
  };
  struct Scratch {
    QueryVisited visited;
    std::vector<Candidate> pool;
    std::vector<Edge> candidates, selected, reverse, reverse_pruned, carry;
    std::vector<uint32_t> neighbors;
    size_t distances{0}, expansions{0};
    Scratch(size_t n, size_t ef, size_t m) : visited(n) {
      pool.reserve(ef + 1);
      candidates.reserve(ef + 2 * m + 8);
      selected.reserve(m);
      carry.reserve(ef);
      reverse.reserve(m + 1);
      reverse_pruned.reserve(m);
      neighbors.reserve(m);
    }
  };
  ClusterWowGraph(const uint32_t *ranks, size_t n, size_t m, std::vector<size_t> half_windows,
                  Distance distance, float alpha = 1.2f, bool full_carry = false,
                  std::vector<size_t> layer_caps = {})
      : ranks_(ranks), n_(n), m_(m), half_(std::move(half_windows)), distance_(distance),
        edges_(n * half_.size() * m), degrees_(n * half_.size(), 0), locks_(new std::mutex[4096]),
        alpha_(alpha), full_carry_(full_carry), caps_(std::move(layer_caps)) {
    if (caps_.empty())
      caps_.assign(half_.size(), m_);
    if (caps_.size() != half_.size() ||
        std::any_of(caps_.begin(), caps_.end(), [&](size_t c) { return !c || c > m_; }))
      throw std::invalid_argument("invalid layer caps");
    if (!m || half_.size() < 2 || !std::is_sorted(ranks, ranks + n))
      throw std::invalid_argument("cluster WoW: invalid geometry");
  }
  bool Allowed(uint32_t u, uint32_t v, size_t layer) const {
    if (u >= n_ || v >= n_ || u == v)
      return false;
    const size_t a = ranks_[u], b = ranks_[v];
    return (a > b ? a - b : b - a) <= half_[layer];
  }
  // Reuse the existing seed-layer slots while a compact input stream loads all
  // vectors. No distance is evaluated until FinishStagedSeed is called.
  void StageSeed(uint32_t u, const std::vector<uint32_t> &inherited) {
    if (u >= n_ || inherited.size() > m_)
      throw std::invalid_argument("invalid staged cluster seed");
    const size_t slot = size_t(u) * half_.size();
    size_t count = 0;
    for (auto v : inherited)
      if (Allowed(u, v, 0))
        edges_[slot * m_ + count++] = {0, v};
    degrees_[slot] = uint32_t(count);
  }
  void FinishStagedSeed(uint32_t u) {
    if (u >= n_)
      throw std::invalid_argument("invalid staged seed finalization");
    const size_t slot = size_t(u) * half_.size();
    std::vector<uint32_t> inherited;
    inherited.reserve(degrees_[slot]);
    for (size_t j = 0; j < degrees_[slot]; ++j)
      inherited.push_back(edges_[slot * m_ + j].id_);
    Seed(u, inherited);
  }
  // Called before Build: inherited list and rank-local bootstrap.
  void Seed(uint32_t u, const std::vector<uint32_t> &inherited) {
    std::vector<Edge> c, kept;
    for (auto v : inherited)
      if (Allowed(u, v, 0))
        c.push_back({distance_(u, v), v});
    for (size_t d = 1; d <= 2; ++d) {
      if (u >= d && Allowed(u, u - d, 0))
        c.push_back({distance_(u, u - d), uint32_t(u - d)});
      if (u + d < n_ && Allowed(u, u + d, 0))
        c.push_back({distance_(u, u + d), uint32_t(u + d)});
    }
    size_t ignored = 0;
    for (size_t l = 0; l < half_.size(); ++l) {
      Prune(c, kept, ignored, caps_[l]);
      Assign(u, l, kept);
    }
  }
  void Build(size_t ef, size_t passes, int threads) {
    if (ef < m_ || !passes || threads < 1)
      throw std::invalid_argument("cluster WoW: invalid build tuning");
    std::vector<std::unique_ptr<Scratch>> scratch;
    for (int t = 0; t < threads; ++t)
      scratch.emplace_back(new Scratch(n_, ef, m_));
    std::exception_ptr error;
    std::atomic<bool> failed{false};
// Persistent parallel team; batches are ordered, each batch completes all
// its forward and reverse updates before the next rank range begins.
#pragma omp parallel num_threads(threads)
    {
      auto &s = *scratch[omp_get_thread_num()];
      for (size_t pass = 0; pass < passes; ++pass) {
        for (size_t begin = 0; begin < n_; begin += 2048) {
          const size_t end = std::min(n_, begin + 2048);
#pragma omp for schedule(dynamic, 16)
          for (size_t u = begin; u < end; ++u) {
            if (failed.load(std::memory_order_relaxed))
              continue;
            try {
              s.selected.clear();
              s.carry.clear();
              for (size_t l = half_.size() - 1; l > 0; --l)
                Refine(uint32_t(u), l, ef, s);
            } catch (...) {
#pragma omp critical(alliance_cluster_wow_error)
              {
                if (!error)
                  error = std::current_exception();
              }
              failed.store(true, std::memory_order_relaxed);
            }
          }
        }
      }
    }
    if (error)
      std::rethrow_exception(error);
    for (const auto &s : scratch) {
      distances += s->distances;
      expansions += s->expansions;
    }
  }
  std::vector<Edge> Edges(uint32_t u, size_t layer) const {
    const size_t offset = (size_t(u) * half_.size() + layer) * m_;
    return {edges_.begin() + offset,
            edges_.begin() + offset + degrees_[size_t(u) * half_.size() + layer]};
  }
  size_t distances{0}, expansions{0};

private:
  void Assign(uint32_t u, size_t l, const std::vector<Edge> &list) {
    const size_t slot = size_t(u) * half_.size() + l;
    std::copy(list.begin(), list.end(), edges_.begin() + slot * m_);
    degrees_[slot] = uint32_t(list.size());
  }
  void Snapshot(uint32_t u, size_t l, std::vector<uint32_t> &out) {
    std::lock_guard<std::mutex> lock(locks_[u % 4096]);
    const size_t slot = size_t(u) * half_.size() + l;
    out.clear();
    for (size_t i = 0; i < degrees_[slot]; ++i)
      out.push_back(edges_[slot * m_ + i].id_);
  }
  void Prune(std::vector<Edge> &c, std::vector<Edge> &out, size_t &count, size_t cap) {
    // Forward and reverse calculations can differ in floating-point rounding.
    // Deduplicate by ID before distance sorting so equal IDs need not be adjacent
    // in distance order (especially around negative-IP ties).
    std::sort(c.begin(), c.end(), [](const Edge &a, const Edge &b) {
      return a.id_ < b.id_ || (a.id_ == b.id_ && a.dist_ < b.dist_);
    });
    c.erase(std::unique(c.begin(), c.end(),
                        [](const Edge &a, const Edge &b) { return a.id_ == b.id_; }),
            c.end());
    std::sort(c.begin(), c.end(), [](const Edge &a, const Edge &b) {
      return a.dist_ < b.dist_ || (a.dist_ == b.dist_ && a.id_ < b.id_);
    });
    if (c.size() <= cap) {
      out = c;
      return;
    }
    out.clear();
    for (const auto &e : c) {
      bool good = true;
      for (size_t i = 0; i < out.size(); ++i) {
        const auto &a = out[i];
        ++count;
        if (alpha_ * distance_(a.id_, e.id_) <= e.dist_) {
          good = false;
          break;
        }
      }
      if (good)
        out.push_back(e);
      if (out.size() == cap)
        break;
    }
  }
  void Refine(uint32_t u, size_t l, size_t ef, Scratch &s) {
    s.visited.Clear();
    s.visited.Set(u);
    s.pool.clear();
    auto offer = [&](uint32_t v) {
      if (v >= n_ || !Allowed(u, v, l) || s.visited.Test(v))
        return;
      s.visited.Set(v);
      ++s.distances;
      Candidate c{distance_(u, v), v, false};
      if (!std::isfinite(c.distance))
        throw std::runtime_error("nonfinite cluster distance");
      auto less = [](const Candidate &a, const Candidate &b) {
        return a.distance < b.distance || (a.distance == b.distance && a.id < b.id);
      };
      if (s.pool.size() == ef && !less(c, s.pool.back()))
        return;
      s.pool.insert(std::lower_bound(s.pool.begin(), s.pool.end(), c, less), c);
      if (s.pool.size() > ef)
        s.pool.pop_back();
    };
    // Reuse higher-layer candidates, then lower navigation, then rank-near
    // entry points. No cluster-local rank is ever used as a window width.
    for (const auto &e : (full_carry_ ? s.carry : s.selected))
      offer(e.id_);
    Snapshot(u, l, s.neighbors);
    for (auto v : s.neighbors)
      offer(v);
    Snapshot(u, l - 1, s.neighbors);
    for (auto v : s.neighbors)
      offer(v);
    for (size_t d = 1; d <= 2; ++d) {
      if (u >= d)
        offer(u - d);
      if (u + d < n_)
        offer(u + d);
    }
    while (true) {
      auto it = std::find_if(s.pool.begin(), s.pool.end(),
                             [](const Candidate &c) { return !c.expanded; });
      if (it == s.pool.end())
        break;
      const uint32_t v = it->id;
      it->expanded = true;
      ++s.expansions;
      // Walk the current window graph, supplementing it with lower levels.
      for (size_t level = l + 1; level-- > 0;) {
        Snapshot(v, level, s.neighbors);
        for (auto w : s.neighbors)
          offer(w);
      }
    }
    s.candidates.clear();
    for (const auto &c : s.pool)
      s.candidates.push_back({c.distance, c.id});
    if (full_carry_)
      s.carry = s.candidates;
    {
      // Merge concurrent backlinks before publishing the replacement list.
      std::lock_guard<std::mutex> lock(locks_[u % 4096]);
      const size_t slot = size_t(u) * half_.size() + l;
      for (size_t i = 0; i < degrees_[slot]; ++i)
        s.candidates.push_back(edges_[slot * m_ + i]);
      Prune(s.candidates, s.selected, s.distances, caps_[l]);
      Assign(u, l, s.selected);
    }
    for (const auto &e : s.selected) {
      const uint32_t v = e.id_;
      std::lock_guard<std::mutex> lock(locks_[v % 4096]);
      const size_t slot = size_t(v) * half_.size() + l;
      s.reverse.assign(edges_.begin() + slot * m_, edges_.begin() + slot * m_ + degrees_[slot]);
      if (std::any_of(s.reverse.begin(), s.reverse.end(),
                      [&](const Edge &a) { return a.id_ == u; }))
        continue;
      ++s.distances;
      s.reverse.push_back({distance_(v, u), u});
      if (s.reverse.size() <= caps_[l])
        Assign(v, l, s.reverse);
      else {
        Prune(s.reverse, s.reverse_pruned, s.distances, caps_[l]);
        Assign(v, l, s.reverse_pruned);
      }
    }
  }
  const uint32_t *ranks_;
  size_t n_, m_;
  std::vector<size_t> half_;
  Distance distance_;
  std::vector<Edge> edges_;
  std::vector<uint32_t> degrees_;
  std::unique_ptr<std::mutex[]> locks_;
  float alpha_;
  bool full_carry_;
  std::vector<size_t> caps_;
};
} // namespace alliance::disk_build
