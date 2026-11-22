#pragma once
// alliance/core/types.hpp -- Core type definitions for AllianceANNS
// Adapted from StorageAirRFANNS (WoW) wow/utils.hh

#include <vector>
#include <cstdlib>
#include <cstring>
#include <stdio.h>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <sys/mman.h>

/// Graph-structure invariant checks. Off by default; compile with
/// -DALLIANCE_GRAPH_CHECKS to keep them at -O3, which is the only practical way
/// to catch adjacency corruption during a full-scale build (a Debug build is
/// -O0 and far too slow to reproduce a build-time race). They fail at the
/// corruption site instead of letting a garbage id become a wild pointer
/// several hops later. WoW carries the equivalent checks unconditionally, see
/// StorageAirRFANNS wow/index.hh:309-314.
#ifdef ALLIANCE_GRAPH_CHECKS
#define ALLIANCE_GRAPH_CHECK(cond, msg)                                                            \
  do {                                                                                             \
    if (!(cond)) {                                                                                 \
      throw std::runtime_error(std::string("graph invariant violated: ") + (msg));                 \
    }                                                                                              \
  } while (0)
#else
#define ALLIANCE_GRAPH_CHECK(cond, msg) ((void)0)
#endif

#define PUSH_HEAP(vec, ...)                                                                        \
  vec.emplace_back(__VA_ARGS__);                                                                   \
  std::push_heap(vec.begin(), vec.end())

#define POP_HEAP(vec)                                                                              \
  std::pop_heap(vec.begin(), vec.end());                                                           \
  vec.pop_back();

#define TOP_HEAP(vec) vec.front()

// alignment tests
#define IS_ALIGNED(X, Y) ((uint64_t)(X) % (uint64_t)(Y) == 0)
#define IS_512_ALIGNED(X) IS_ALIGNED(X, 512)
#define IS_4096_ALIGNED(X) IS_ALIGNED(X, 4096)

namespace alliance {
typedef unsigned int tableint;
typedef int layer_t;
typedef size_t label_t;
typedef float dist_t;
typedef unsigned short int vl_type;

template <typename att_t> struct attr_range {
  att_t l_{};
  att_t u_{};

  attr_range(const att_t &l, const att_t &u) : l_(l), u_(u) {
  }
  attr_range() = default;

  inline __attribute__((always_inline)) auto Test(const att_t &att) const -> bool {
    return att >= l_ && att <= u_;
  }
};

template <typename att_t> struct attr_set {
  std::unordered_set<att_t> set_;
  void Set(att_t i) {
    set_.insert(i);
  }
  inline __attribute__((always_inline)) bool Test(att_t i) const {
    return set_.find(i) != set_.end();
  }
};

// A level stores layers [layer_begin, layer_end], but owns/shuffles only
// [owner_begin, owner_end]. Ownership partitions the selected layer grid;
// stored ranges may overlap to support descent from an already cached record.
// Lives here (rather than shuffle.hpp, its logical home) so both
// shuffle.hpp and index_disk.hpp can reference it without a circular include
// (shuffle.hpp already includes index_disk.hpp).
struct LevelRange {
  layer_t layer_begin;
  layer_t layer_end;
  layer_t owner_begin{-1};
  layer_t owner_end{-1};
};

struct dist_id_pair {
  dist_t dist_;
  tableint id_;

  dist_id_pair() = default;

  dist_id_pair(dist_t dist, tableint id) : dist_(dist), id_(id) {
  }

  bool operator<(const dist_id_pair &rhs) const {
    return dist_ < rhs.dist_;
  }

  bool operator>(const dist_id_pair &rhs) const {
    return dist_ > rhs.dist_;
  }
};
} // namespace alliance
