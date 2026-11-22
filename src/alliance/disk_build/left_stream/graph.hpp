#pragma once
#include "core/types.hpp"
#include "core/visit_list.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>
#include <omp.h>

namespace alliance::disk_build {
struct LeftStreamEdges {
  tableint *ids;
  dist_t *distances;
  tableint *count;
};

// Storage supplies resident views/vectors, rank mappings and striped locks.
// There is no IO here. L0 active and reserve entries share exactly M slots;
// higher layers retain active edges only and fill from finalized lower layers.
template <class Storage> class LeftStreamGraph {
  using Edge = dist_id_pair;
  struct Candidate {
    dist_t distance;
    tableint id;
    bool expanded;
  };
  struct Scratch {
    VisitedList<tableint> visited;
    std::vector<Edge> raw, bottom, old_pool, back, back_out, carry, candidates, eps, selected,
        merged, merged_out, result;
    std::vector<Candidate> beam;
    std::vector<tableint> neighbors;
    size_t bottom_fill{0}, upper_fill{0};
    Scratch(size_t n, size_t ef, size_t m, size_t h) : visited(n) {
      raw.reserve(h);
      bottom.reserve(m);
      old_pool.reserve(m + 1);
      back.reserve(m + 1);
      back_out.reserve(m);
      carry.reserve(ef);
      candidates.reserve(ef);
      eps.reserve(ef);
      selected.reserve(m);
      merged.reserve(2 * m);
      merged_out.reserve(m);
      result.reserve(ef + 1);
      beam.reserve(ef + 1);
      neighbors.reserve(m);
    }
  };

public:
  LeftStreamGraph(Storage &storage, size_t n, size_t m, size_t top,
                  const std::vector<size_t> &windows, size_t ef, int threads, float alpha)
      : s_(storage), n_(n), m_(m), top_(top), windows_(windows), ef_(ef), threads_(threads),
        alpha_(alpha), ready_(n) {
    if (!n || !m || !ef || ef < m || threads < 1 || top >= windows.size() ||
        windows[0] / 2 < std::min(m, n - 1))
      throw std::invalid_argument("left_stream requires ef>=M and L0 half-window>=min(M,n-1)");
    for (auto &r : ready_)
      r.store(0, std::memory_order_relaxed);
    for (int t = 0; t < threads; ++t)
      scratch_.emplace_back(new Scratch(n, ef, m, windows[0] / 2));
  }
  void Batch(size_t begin, size_t end) {
    if (begin != processed_ || end <= begin || end > n_)
      throw std::logic_error("unordered left_stream batch");
    const size_t count = end - begin;
    forward_.resize(count * m_);
    forward_count_.resize(count);
    // Forward L0 lists are private. No reverse writer runs until every list
    // in the batch has been initialized, preventing overwrite of incoming edges.
    Parallel(begin, end, [&](size_t rank, Scratch &x) {
      const auto u = s_.Id(rank);
      x.raw.clear();
      const size_t h = windows_[0] / 2;
      for (size_t r = rank > h ? rank - h : 0; r < rank; ++r)
        x.raw.push_back({s_.Distance(u, s_.Id(r)), s_.Id(r)});
      if (x.raw.size() <= m_)
        x.bottom = x.raw;
      else
        Prune(x.raw, x.bottom, m_);
      SaveBottom(u, x.bottom, x.raw);
      const size_t local = rank - begin;
      forward_count_[local] = x.bottom.size();
      std::copy(x.bottom.begin(), x.bottom.end(), forward_.begin() + local * m_);
      ready_[u].store(1, std::memory_order_release);
    });
    Parallel(begin, end, [&](size_t rank, Scratch &x) {
      const auto u = s_.Id(rank);
      const size_t local = rank - begin;
      for (size_t k = 0; k < forward_count_[local]; ++k)
        Reverse(u, 0, forward_[local * m_ + k], x);
    });
    // Register all enumerated pairs at both endpoints. Each target has one
    // owner in this phase, avoiding a mutex per unselected candidate pair.
    const size_t h = windows_[0] / 2;
    Parallel(begin > h ? begin - h : 0, end, [&](size_t rank, Scratch &) {
      const auto u = s_.Id(rank);
      auto v = s_.View(u, 0);
      if (*v.count == m_)
        return;
      const size_t right = std::min(end, rank + std::min(h, n_ - 1 - rank) + 1);
      for (size_t r = std::max(begin, rank + 1); r < right; ++r)
        Remember(u, {s_.Distance(u, s_.Id(r)), s_.Id(r)});
    });
    Activate(end);
    Parallel(begin, end, [&](size_t rank, Scratch &x) { BuildUpper(rank, x); });
    processed_ = end;
  }
  void FinishLayers() {
    if (processed_ != n_)
      throw std::logic_error("unfinished left_stream graph");
    // An oversized final window never activates. Its prefix remains resident
    // because 2H>=n. Preserve the existing global-layer copy behavior.
    if (active_top_ > 0)
      for (size_t l = active_top_ + 1; l <= top_; ++l)
        Copy(active_top_, l, n_);
  }
  void Finalize(size_t begin, size_t end) {
    Parallel(begin, end, [&](size_t rank, Scratch &x) {
      const auto u = s_.Id(rank);
      const size_t target = std::min(m_, n_ - 1);
      auto base = s_.View(u, 0);
      const size_t total = s_.BottomTotal(u);
      if (total != target || *base.count > total)
        throw std::runtime_error("left_stream L0 candidate coverage shortfall");
      x.bottom_fill += total - *base.count;
      *base.count = total;
      for (size_t l = 1; l <= top_; ++l) {
        auto dst = s_.View(u, l);
        auto lower = s_.View(u, l - 1);
        for (size_t k = 0; k < *lower.count && *dst.count < target; ++k) {
          const auto id = lower.ids[k];
          if (!Allowed(u, id, l))
            continue;
          bool duplicate = false;
          for (size_t j = 0; j < *dst.count; ++j)
            duplicate |= dst.ids[j] == id;
          if (!duplicate) {
            const size_t p = (*dst.count)++;
            dst.ids[p] = id;
            dst.distances[p] = lower.distances[k];
            ++x.upper_fill;
          }
        }
        if (*dst.count != target)
          throw std::runtime_error("left_stream upper-layer fill shortfall");
      }
    });
    finalized_ += end - begin;
  }
  size_t Finalized() const {
    return finalized_;
  }
  size_t BottomFill() const {
    size_t n = 0;
    for (auto &x : scratch_)
      n += x->bottom_fill;
    return n;
  }
  size_t UpperFill() const {
    size_t n = 0;
    for (auto &x : scratch_)
      n += x->upper_fill;
    return n;
  }

private:
  template <class F> void Parallel(size_t begin, size_t end, F function) {
    std::exception_ptr error;
    std::atomic<bool> failed{false};
    // Small batches must expose enough work for all threads: batch=256
    // with chunk=32 would provide only eight schedulable tasks.
#pragma omp parallel for num_threads(threads_) schedule(dynamic, 1)
    for (size_t rank = begin; rank < end; ++rank) {
      if (failed.load(std::memory_order_relaxed))
        continue;
      try {
        function(rank, *scratch_[omp_get_thread_num()]);
      } catch (...) {
        failed.store(true, std::memory_order_relaxed);
#pragma omp critical(alliance_left_stream_error)
        {
          if (!error)
            error = std::current_exception();
        }
      }
    }
    if (error)
      std::rethrow_exception(error);
  }
  bool Allowed(tableint u, tableint v, size_t l) const {
    if (u == v || v >= n_)
      return false;
    const size_t a = s_.Rank(u), b = s_.Rank(v);
    return (a > b ? a - b : b - a) <= windows_[l] / 2;
  }
  static void Unique(std::vector<Edge> &out, const Edge &e) {
    if (std::none_of(out.begin(), out.end(), [&](const Edge &a) { return a.id_ == e.id_; }))
      out.push_back(e);
  }
  void Prune(std::vector<Edge> &candidates, std::vector<Edge> &out, size_t cap) {
    std::sort(candidates.begin(), candidates.end());
    out.clear();
    for (const auto &e : candidates) {
      if (out.size() >= cap)
        break;
      bool good = true;
      for (const auto &a : out)
        if (alpha_ * s_.Distance(e.id_, a.id_) <= e.dist_) {
          good = false;
          break;
        }
      if (good)
        out.push_back(e);
    }
  }
  void Assign(tableint u, size_t l, const std::vector<Edge> &edges) {
    auto v = s_.View(u, l);
    if (edges.size() > m_)
      throw std::logic_error("left_stream degree overflow");
    for (size_t i = 0; i < edges.size(); ++i) {
      v.ids[i] = edges[i].id_;
      v.distances[i] = edges[i].dist_;
    }
    *v.count = edges.size();
  }
  void SaveBottom(tableint u, const std::vector<Edge> &active, std::vector<Edge> &pool) {
    std::sort(pool.begin(), pool.end());
    Assign(u, 0, active);
    auto v = s_.View(u, 0);
    size_t total = active.size();
    for (const auto &e : pool) {
      if (total == m_)
        break;
      if (!Allowed(u, e.id_, 0))
        continue;
      bool duplicate = false;
      for (size_t j = 0; j < total; ++j)
        duplicate |= v.ids[j] == e.id_;
      if (!duplicate) {
        v.ids[total] = e.id_;
        v.distances[total] = e.dist_;
        ++total;
      }
    }
    s_.BottomTotal(u) = total;
  }
  void Remember(tableint u, const Edge &e) {
    auto v = s_.View(u, 0);
    size_t total = s_.BottomTotal(u), active = *v.count;
    if (active == m_)
      return;
    if (total == m_ && Edge{v.distances[total - 1], v.ids[total - 1]} < e)
      return;
    for (size_t j = 0; j < total; ++j)
      if (v.ids[j] == e.id_)
        return;
    size_t position = active;
    while (position < total && Edge{v.distances[position], v.ids[position]} < e)
      ++position;
    if (position == m_)
      return;
    const size_t new_total = std::min(total + 1, m_);
    for (size_t j = new_total - 1; j > position; --j) {
      v.ids[j] = v.ids[j - 1];
      v.distances[j] = v.distances[j - 1];
    }
    v.ids[position] = e.id_;
    v.distances[position] = e.dist_;
    s_.BottomTotal(u) = new_total;
  }
  void Reverse(tableint source, size_t l, const Edge &edge, Scratch &x) {
    const auto target = edge.id_;
    std::lock_guard<std::mutex> lock(s_.Mutex(target));
    auto v = s_.View(target, l);
    const size_t count = *v.count;
    for (size_t j = 0; j < count; ++j)
      if (v.ids[j] == source)
        return;
    x.back.clear();
    for (size_t j = 0; j < count; ++j)
      if (Allowed(target, v.ids[j], l) && ready_[v.ids[j]].load(std::memory_order_acquire) >= 1)
        x.back.push_back({v.distances[j], v.ids[j]});
    x.back.push_back({edge.dist_, source});
    if (l == 0) {
      x.old_pool.clear();
      for (size_t j = 0; j < s_.BottomTotal(target); ++j)
        x.old_pool.push_back({v.distances[j], v.ids[j]});
      Unique(x.old_pool, {edge.dist_, source});
    }
    if (count < m_) {
      // Existing lists are valid; append without another RNG pass, as in WoW.
      if (l == 0)
        SaveBottom(target, x.back, x.old_pool);
      else
        Assign(target, l, x.back);
    } else {
      Prune(x.back, x.back_out, m_);
      if (l == 0)
        SaveBottom(target, x.back_out, x.old_pool);
      else
        Assign(target, l, x.back_out);
    }
  }
  void Copy(size_t from, size_t to, size_t count) {
    Parallel(0, count, [&](size_t rank, Scratch &) {
      auto a = s_.View(s_.Id(rank), from), b = s_.View(s_.Id(rank), to);
      std::copy(a.ids, a.ids + m_, b.ids);
      std::copy(a.distances, a.distances + m_, b.distances);
      *b.count = *a.count;
    });
  }
  void Activate(size_t end) {
    for (size_t l = active_top_ + 1; l <= top_; ++l) {
      if (windows_[l] - 1 >= end)
        break;
      Copy(l == 1 ? 0 : l - 1, l, windows_[l]);
      active_top_ = l;
    }
  }
  void Search(tableint query, size_t left, size_t right, size_t max_layer, Scratch &x) {
    x.visited.Clear();
    x.visited.Set(query);
    x.beam.clear();
    x.result.clear();
    auto insert = [&](const Edge &e) {
      if (x.beam.size() == ef_ && e.dist_ >= x.beam.back().distance)
        return;
      auto at = std::lower_bound(x.beam.begin(), x.beam.end(), e.dist_,
                                 [](const Candidate &a, dist_t d) { return a.distance < d; });
      x.beam.insert(at, {e.dist_, e.id_, false});
      if (x.beam.size() > ef_)
        x.beam.pop_back();
    };
    for (const auto &e : x.eps) {
      insert(e);
      x.result.push_back(e);
      x.visited.Set(e.id_);
    }
    std::sort(x.result.begin(), x.result.end());
    if (x.result.size() > ef_)
      x.result.resize(ef_);
    while (true) {
      auto at = std::find_if(x.beam.begin(), x.beam.end(),
                             [](const Candidate &c) { return !c.expanded; });
      if (at == x.beam.end())
        break;
      const auto current = at->id;
      at->expanded = true;
      x.neighbors.clear();
      {
        std::lock_guard<std::mutex> lock(s_.Mutex(current));
        const size_t from = ready_[current].load(std::memory_order_acquire) >= 2 ? max_layer : 0;
        for (size_t ll = from + 1; ll > 0 && x.neighbors.size() < m_; --ll) {
          auto v = s_.View(current, ll - 1);
          for (size_t k = 0; k < *v.count && x.neighbors.size() < m_; ++k) {
            const auto id = v.ids[k];
            if (id >= n_)
              throw std::runtime_error("left_stream corrupt neighbor");
            const size_t rank = s_.Rank(id);
            if (rank < left || rank > right || x.visited.Test(id))
              continue;
            x.visited.Set(id);
            if (ready_[id].load(std::memory_order_acquire) < 1)
              continue;
            x.neighbors.push_back(id);
          }
        }
      }
      for (const auto id : x.neighbors) {
        Edge e{s_.Distance(query, id), id};
        if (x.beam.size() == ef_ && e.dist_ >= x.beam.back().distance)
          continue;
        insert(e);
        if (x.result.size() < ef_ || e.dist_ < x.result.back().dist_) {
          x.result.insert(std::lower_bound(x.result.begin(), x.result.end(), e), e);
          if (x.result.size() > ef_)
            x.result.pop_back();
        }
      }
    }
  }
  void BuildUpper(size_t rank, Scratch &x) {
    const auto u = s_.Id(rank);
    size_t top = 0;
    for (size_t l = 1; l <= active_top_; ++l) {
      if (windows_[l] <= rank + 1)
        top = l;
      else
        break;
    }
    x.carry.clear();
    for (size_t ll = top; ll > 0; --ll) {
      const size_t h = windows_[ll] / 2, left = rank > h ? rank - h : 0;
      x.candidates.clear();
      for (const auto &e : x.carry) {
        const size_t r = s_.Rank(e.id_);
        if (r >= left && r < rank)
          x.candidates.push_back(e);
      }
      x.eps = x.candidates;
      if (x.eps.empty())
        for (size_t r = rank; r > left;) {
          --r;
          const auto id = s_.Id(r);
          bool available = false;
          {
            std::lock_guard<std::mutex> lock(s_.Mutex(id));
            available = *s_.View(id, ll).count > 0 || *s_.View(id, 0).count > 0;
          }
          if (available) {
            x.eps.push_back({s_.Distance(u, id), id});
            break;
          }
        }
      if (x.eps.empty())
        continue;
      if (x.candidates.size() < m_) {
        Search(u, left, rank - 1, top, x);
        if (!x.result.empty())
          x.candidates = x.result;
      }
      if (x.candidates.empty())
        continue;
      Prune(x.candidates, x.selected, m_ / 2);
      {
        std::lock_guard<std::mutex> lock(s_.Mutex(u));
        auto v = s_.View(u, ll);
        x.merged = x.selected;
        for (size_t k = 0; k < *v.count; ++k)
          Unique(x.merged, {v.distances[k], v.ids[k]});
        if (*v.count) {
          Prune(x.merged, x.merged_out, m_);
          Assign(u, ll, x.merged_out);
        } else
          Assign(u, ll, x.selected);
      }
      for (const auto &e : x.selected)
        Reverse(u, ll, e, x);
      x.carry = x.candidates;
    }
    ready_[u].store(2, std::memory_order_release);
  }
  Storage &s_;
  size_t n_, m_, top_;
  const std::vector<size_t> &windows_;
  size_t ef_;
  int threads_;
  float alpha_;
  size_t active_top_{0}, processed_{0}, finalized_{0};
  std::vector<std::atomic<uint8_t>> ready_;
  std::vector<std::unique_ptr<Scratch>> scratch_;
  std::vector<Edge> forward_;
  std::vector<uint32_t> forward_count_;
};
} // namespace alliance::disk_build
