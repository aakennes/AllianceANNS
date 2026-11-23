#pragma once
// Immutable, rank-aware navigation graph. No SSD reads and no guided entries.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <bit>
#include <vector>

namespace alliance {
class MemoryNavigationGraph {
public:
  static constexpr uint32_t invalid = UINT32_MAX;
  struct Group {
    uint32_t first{}, last{}; // inclusive basic layers, low -> high
    std::vector<uint32_t> ranks, ids;
    // [layer-first][node][degree], local slots; invalid terminates a list.
    std::vector<uint32_t> edges;
  };
  struct Result {
    std::vector<std::pair<float, uint32_t>> entries;
    size_t scores{}, expansions{}, edge_visits{}, layers_visited{};
  };
  uint64_t population{}, dimension{}, order_hash{}, metric{};
  uint32_t degree{16};
  std::vector<uint64_t> windows;
  std::vector<Group> groups; // top SSD level first; increasingly dense

  static uint64_t Hash(const void *data, size_t bytes, uint64_t h = 14695981039346656037ULL) {
    const auto *p = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < bytes; ++i) {
      h ^= p[i];
      h *= 1099511628211ULL;
    }
    return h;
  }
  template <class IDs, class Attributes>
  static uint64_t OrderHash(const IDs &ids, const Attributes &attributes) {
    uint64_t h = Hash(ids.data(), ids.size() * sizeof(ids[0]));
    return Hash(attributes.data(), attributes.size() * sizeof(attributes[0]), h);
  }
  size_t Bytes() const {
    size_t n = sizeof(*this) + windows.capacity() * 8 + groups.capacity() * sizeof(Group);
    for (const auto &g : groups)
      n += 4 * (g.ranks.capacity() + g.ids.capacity() + g.edges.capacity());
    return n;
  }
  void Validate() const {
    if (!population || population >= invalid || !dimension || !degree || degree > 1024 ||
        windows.empty() || windows.size() > 64 || groups.empty() ||
        groups.size() > windows.size() || (metric != 1 && metric != 2))
      throw std::runtime_error("navigation: invalid header");
    if (!std::is_sorted(windows.begin(), windows.end()) || !windows.front())
      throw std::runtime_error("navigation: invalid windows");
    size_t next = windows.size();
    for (size_t gi = 0; gi < groups.size(); ++gi) {
      const auto &g = groups[gi];
      const size_t n = g.ids.size();
      if (g.last >= windows.size() || size_t(g.last) + 1 != next || g.first > g.last || !n ||
          n > population || g.ranks.size() != n ||
          g.edges.size() != (g.last - g.first + 1) * n * degree)
        throw std::runtime_error("navigation: invalid group shape/ownership");
      next = g.first;
      std::vector<uint32_t> ids = g.ids;
      std::sort(ids.begin(), ids.end());
      if (ids.back() >= population || std::adjacent_find(ids.begin(), ids.end()) != ids.end())
        throw std::runtime_error("navigation: invalid or duplicate global ID");
      for (size_t i = 0; i < n; ++i) {
        if (g.ranks[i] >= population || (i && g.ranks[i] <= g.ranks[i - 1]))
          throw std::runtime_error("navigation: rank order");
        for (size_t l = g.first; l <= g.last; ++l) {
          const auto *edges = g.edges.data() + ((l - g.first) * n + i) * degree;
          bool ended = false;
          for (size_t j = 0; j < degree; ++j) {
            const auto v = edges[j];
            if (v == invalid) {
              ended = true;
              continue;
            }
            if (ended || v >= n || v == i || std::find(edges, edges + j, v) != edges + j ||
                uint64_t(std::max(g.ranks[i], g.ranks[v])) - std::min(g.ranks[i], g.ranks[v]) >
                    windows[l] / 2)
              throw std::runtime_error("navigation: invalid/window-violating edge");
          }
        }
      }
      if (gi) {
        const auto &p = groups[gi - 1];
        for (size_t j = 0; j < p.ranks.size(); ++j) {
          auto it = std::lower_bound(g.ranks.begin(), g.ranks.end(), p.ranks[j]);
          if (it == g.ranks.end() || *it != p.ranks[j] || g.ids[it - g.ranks.begin()] != p.ids[j])
            throw std::runtime_error("navigation: representative sets must be nested");
        }
      }
    }
    // A sparse navigation sidecar may omit basic layer zero only.
    if (next > 1)
      throw std::runtime_error("navigation: uncovered basic layer");
  }
  template <class Index> void Match(const Index &index, uint64_t expected_metric) const {
    if (index.sorted_ids_.size() != population || index.atts_.size() != population ||
        population != index.curvec_num_ || dimension != index.vec_d_ || metric != expected_metric ||
        order_hash != OrderHash(index.sorted_ids_, index.atts_) ||
        windows.size() != index.cur_max_layer_ + 1 || index.window_size_.size() < windows.size() ||
        (groups.size() != index.ml_levels_.size() &&
         !(groups.back().first == 1 && groups.size() + 1 == index.ml_levels_.size() &&
           index.ml_levels_.back().owner_begin == 0 && index.ml_levels_.back().owner_end == 0)))
      throw std::runtime_error("navigation: source index mismatch");
    for (size_t l = 0; l < windows.size(); ++l)
      if (windows[l] != index.window_size_[l])
        throw std::runtime_error("navigation: window mismatch");
    for (size_t gi = 0; gi < groups.size(); ++gi) {
      const auto &g = groups[gi];
      const auto &level = index.ml_levels_[gi];
      if ((g.first != level.owner_begin && !(gi + 1 == groups.size() && g.first == 1 && level.owner_begin == 0)) || g.last != level.owner_end)
        throw std::runtime_error("navigation: level ownership mismatch");
      for (size_t j = 0; j < g.ids.size(); ++j)
        if (index.sorted_ids_[g.ranks[j]] != g.ids[j])
          throw std::runtime_error("navigation: rank/ID mismatch");
    }
  }
  static MemoryNavigationGraph Load(const std::string &path, uint64_t expected_population = 0,
                                    uint64_t expected_dimension = 0) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in || in.tellg() < 80)
      throw std::runtime_error("navigation: missing/truncated file");
    uint64_t remaining = static_cast<uint64_t>(in.tellg()) - 8, hash = 14695981039346656037ULL;
    in.seekg(0);
    auto read = [&](void *p, size_t n) {
      if (n > remaining)
        throw std::runtime_error("navigation: truncated payload");
      in.read(static_cast<char *>(p), n);
      if (!in)
        throw std::runtime_error("navigation: read failed");
      remaining -= n;
      hash = Hash(p, n, hash);
    };
    auto word = [&]() {
      uint64_t n;
      read(&n, 8);
      return n;
    };
    if (word() != 0x314850415247564EULL || word() != 1)
      throw std::runtime_error("navigation: format/version mismatch");
    MemoryNavigationGraph nav;
    nav.population = word();
    nav.dimension = word();
    nav.order_hash = word();
    nav.metric = word();
    uint64_t m = word(), nl = word(), ng = word();
    if (!nav.population || nav.population >= invalid || !nav.dimension ||
        (nav.metric != 1 && nav.metric != 2) || !m || m > 1024 || !nl || nl > 64 || !ng || ng > nl)
      throw std::runtime_error("navigation: invalid counts");
    // Match the loaded index before allocating the serialized graph payload.
    if ((expected_population && nav.population != expected_population) ||
        (expected_dimension && nav.dimension != expected_dimension))
      throw std::runtime_error("navigation: source index dimensions mismatch");
    nav.degree = m;
    nav.windows.resize(nl);
    read(nav.windows.data(), nl * 8);
    nav.groups.resize(ng);
    for (auto &g : nav.groups) {
      uint64_t lo = word(), hi = word(), n = word();
      if (lo > hi || hi >= nl || !n || n > nav.population)
        throw std::runtime_error("navigation: invalid group");
      uint64_t edge_count = (hi - lo + 1) * n * m;
      if ((edge_count + 2 * n) > remaining / 4)
        throw std::runtime_error("navigation: invalid allocation size");
      g.first = lo;
      g.last = hi;
      g.ranks.resize(n);
      g.ids.resize(n);
      g.edges.resize(edge_count);
      read(g.ranks.data(), n * 4);
      read(g.ids.data(), n * 4);
      read(g.edges.data(), edge_count * 4);
    }
    uint64_t expected = 0;
    in.read(reinterpret_cast<char *>(&expected), 8);
    if (remaining || !in || hash != expected)
      throw std::runtime_error("navigation: checksum/trailing data");
    nav.Validate();
    return nav;
  }

private:
  struct Candidate {
    float d;
    uint32_t id, rank;
    bool expanded;
  };

public:
  // Owned by a search worker. Only capacity survives a query boundary.
  class Scratch {
    friend class MemoryNavigationGraph;
    std::vector<Candidate> pool_;
    std::vector<uint32_t> slots_;

    void Reset(size_t ef, size_t budget) {
      pool_.clear();
      pool_.reserve(ef + 1);
      // At most budget distinct IDs can be scored. Keep an empty slot in
      // every probe sequence even when the query consumes its entire budget.
      if (budget > std::bit_floor(slots_.max_size()) / 2)
        throw std::length_error("navigation: score budget too large");
      slots_.assign(std::bit_ceil(2 * budget), invalid);
    }

    bool Mark(uint32_t id) {
      const size_t mask = slots_.size() - 1;
      size_t slot = ((uint64_t(id) * 11400714819323198485ull) >> 32) & mask;
      while (slots_[slot] != invalid) {
        if (slots_[slot] == id)
          return false;
        slot = (slot + 1) & mask;
      }
      slots_[slot] = id;
      return true;
    }
  };

  // All rank bounds are in the ORIGINAL population, upper-exclusive. The
  // bounded query scratch is separate from every SSD visited set/cache.
  template <class Distance>
  Result Search(uint32_t lo, uint32_t hi, uint32_t landing, size_t ef, size_t budget, size_t count,
                Distance distance) const {
    Scratch scratch;
    return Search(lo, hi, landing, ef, budget, count, distance, scratch);
  }

  template <class Distance>
  Result Search(uint32_t lo, uint32_t hi, uint32_t landing, size_t ef, size_t budget, size_t count,
                Distance &&distance, Scratch &scratch) const {
    if (lo >= hi || hi > population || landing >= windows.size() || !ef || !count || count > ef ||
        budget < count)
      throw std::invalid_argument("navigation: invalid query/budget");
    const size_t limit = std::min(budget, size_t(population));
    scratch.Reset(std::min(ef, limit), limit);
    auto &pool = scratch.pool_;
    const auto better = [](const Candidate &a, const Candidate &b) {
      return a.d < b.d || (a.d == b.d && a.id < b.id);
    };
    Result result;
    auto offer = [&](uint32_t id, uint32_t rank) {
      if (rank < lo || rank >= hi || result.scores == budget || !scratch.Mark(id))
        return false;
      const float d = distance(id);
      if (!std::isfinite(d))
        throw std::runtime_error("navigation: nonfinite distance");
      ++result.scores;
      Candidate c{d, id, rank, false};
      auto it = std::lower_bound(pool.begin(), pool.end(), c, better);
      if (pool.size() < ef || it != pool.end()) {
        pool.insert(it, c);
        if (pool.size() > ef)
          pool.pop_back();
      }
      return true;
    };
    // Route layer-zero requests through the finest retained navigation layer.
    // Candidate admission still uses the original query rank bounds.
    landing = std::max<uint32_t>(landing, groups.back().first);
    size_t start = 0;
    while (start < groups.size() &&
           !(groups[start].first <= landing && landing <= groups[start].last))
      ++start;
    if (start == groups.size())
      throw std::runtime_error("navigation: no owner");
    // A very narrow range may have no representative at its owning level.
    // Descend within the navigation index; never fall back to raw endpoints.
    for (size_t gi = start; gi < groups.size() && pool.empty(); ++gi) {
      const auto &g = groups[gi];
      size_t a = std::lower_bound(g.ranks.begin(), g.ranks.end(), lo) - g.ranks.begin();
      size_t b = std::lower_bound(g.ranks.begin(), g.ranks.end(), hi) - g.ranks.begin();
      const size_t seeds = std::min({size_t(4), b - a, budget});
      for (size_t j = 0; j < seeds; ++j) {
        const size_t p = a + ((2 * j + 1) * (b - a)) / (2 * seeds);
        offer(g.ids[p], g.ranks[p]);
      }
    }
    if (pool.empty())
      throw std::runtime_error(
          "navigation: query has no representative; increase density or use prefilter");
    while (result.scores < budget) {
      auto it =
          std::find_if(pool.begin(), pool.end(), [](const Candidate &c) { return !c.expanded; });
      if (it == pool.end())
        break;
      Candidate u = *it;
      it->expanded = true;
      ++result.expansions;
      size_t new_neighbors = 0;
      for (size_t gi = start;
           gi < groups.size() && result.scores < budget && new_neighbors < degree; ++gi) {
        const auto &g = groups[gi];
        auto p = std::lower_bound(g.ranks.begin(), g.ranks.end(), u.rank);
        if (p == g.ranks.end() || *p != u.rank)
          continue;
        const size_t slot = p - g.ranks.begin(), n = g.ids.size();
        for (int l = std::min<uint32_t>(landing, g.last);
             l >= int(g.first) && result.scores < budget && new_neighbors < degree; --l) {
          ++result.layers_visited;
          const auto *edges = g.edges.data() + ((l - g.first) * n + slot) * degree;
          for (size_t j = 0; j < degree && edges[j] != invalid && result.scores < budget; ++j) {
            const auto v = edges[j];
            ++result.edge_visits;
            new_neighbors += offer(g.ids[v], g.ranks[v]);
          }
        }
      }
    }
    for (size_t j = 0; j < std::min(count, pool.size()); ++j)
      result.entries.emplace_back(pool[j].d, pool[j].id);
    return result;
  }
};
} // namespace alliance
