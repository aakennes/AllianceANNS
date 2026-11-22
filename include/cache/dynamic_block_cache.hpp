#pragma once
// alliance/cache/dynamic_block_cache.hpp -- Dynamic FIFO/LRU piece cache for
// AllianceIndexDisk's multi-level layout search path.
//
// Exact lookup proportional to cache slots, no mutex (thread_local ownership),
// FIFO eviction via circular slot allocation.
//
// One unified cache spans all levels (WoW-style): a slot holds ONE level's
// record for a node, tagged with that record's [layer_begin, layer_end].
// Callers must check the requested layer against the entry's range; when a
// node's expansion crosses into another level, InsertOrUpdate overwrites the
// slot in place with the new level's record.
//
// Auto-sizing (`auto_size` in Configure): when the owner clears the cache
// between queries, nothing survives a query boundary, so a budget large enough
// for cross-query reuse is pure waste -- the cache can never hold more than one
// query's working set. In that mode the configured budget becomes an upper
// bound only: the cache starts at kAutoInitialSlots and grows at Clear() time
// to whatever the previous query actually asked for. Growth reallocates
// `storage_`, which is why it happens only at a Clear() boundary, where no
// Entry::data pointer from the previous query can still be live. It converges
// within a handful of queries and then behaves exactly like the full-budget
// cache (same slots occupied, same hit rate) at a fraction of the memory.
//
// auto_size is only meaningful when the owner really does call Clear() every
// query. Without that the cache would stay at its initial size forever.

#include "core/types.hpp"
#include "cache/slot_lookup.hpp"
#include <algorithm>
#include <cstring>
#include <vector>

namespace alliance {

class DynamicBlockCache {
public:
  enum class EvictPolicy { LRU, FIFO };

  struct Entry {
    bool found{false};
    const char *data{nullptr};
    layer_t layer_begin{0};
    layer_t layer_end{0};
  };

  DynamicBlockCache() = default;

  void Configure(size_t capacity_bytes, size_t piece_bytes, EvictPolicy policy, size_t num_nodes,
                 bool auto_size = false) {
#ifdef ALLIANCE_DENSE_DYNAMIC_CACHE
    // Clear the old owners before a full reconfiguration. Auto growth already
    // runs after Clear(), so it need not refill an O(N) dense array each time.
    lookup_.Clear(slot_to_id_, count_);
#endif
    policy_ = policy;
    piece_bytes_ = piece_bytes;
    num_nodes_ = num_nodes;
    auto_size_ = auto_size;
    max_slots_ = (piece_bytes_ > 0) ? (capacity_bytes / piece_bytes_) : 0;
    num_slots_ = auto_size_ ? std::min(max_slots_, kAutoInitialSlots) : max_slots_;
    wanted_slots_ = 0;
    peak_wanted_slots_ = 0;
    peak_count_ = 0;
    count_ = 0;
    clock_hand_ = 0;
    lru_head_ = lru_tail_ = kInvalid;
    if (max_slots_ == 0) {
      num_slots_ = 0;
      // Disabling a cache must release retained vector capacity as well.
      std::vector<char>().swap(storage_);
      lookup_.Configure(0, 0);
      std::vector<tableint>().swap(slot_to_id_);
      std::vector<tableint>().swap(slot_next_);
      std::vector<tableint>().swap(slot_prev_);
      std::vector<layer_t>().swap(slot_layer_begin_);
      std::vector<layer_t>().swap(slot_layer_end_);
      return;
    }

    ResizeSlots(num_slots_);
  }

  auto Lookup(tableint node_id) -> Entry {
    Entry e;
    if (num_slots_ == 0 || static_cast<size_t>(node_id) >= num_nodes_)
      return e;
    tableint slot = lookup_.Find(node_id);
    if (slot == kInvalid)
      return e;

    if (policy_ == EvictPolicy::LRU && slot != lru_head_) {
      LruDetach(slot);
      LruPushFront(slot);
    }
    e.found = true;
    e.data = storage_.data() + static_cast<size_t>(slot) * piece_bytes_;
    e.layer_begin = slot_layer_begin_[slot];
    e.layer_end = slot_layer_end_[slot];
    return e;
  }

  /// Inserts only when the node has no cached record yet. Prefetched records
  /// decoded from a block read go through this: they must never evict another
  /// level's still-needed record for the same node, or a pending candidate
  /// would have to re-read a block that was already fetched for it earlier
  /// (see DecodeBlockIntoCache in disk_search_cache.hpp).
  void InsertIfMissing(tableint node_id, const char *data, size_t size, layer_t layer_begin,
                       layer_t layer_end) {
    if (num_slots_ == 0 || size > piece_bytes_)
      return;
    if (static_cast<size_t>(node_id) >= num_nodes_)
      return;
    if (lookup_.Find(node_id) != kInvalid)
      return;

    tableint slot = AllocateSlot();
    lookup_.Insert(node_id, slot);
    slot_to_id_[slot] = node_id;
    slot_layer_begin_[slot] = layer_begin;
    slot_layer_end_[slot] = layer_end;
    std::memcpy(storage_.data() + static_cast<size_t>(slot) * piece_bytes_, data, size);

    if (policy_ == EvictPolicy::LRU) {
      LruPushFront(slot);
    }
  }

  /// Inserts `node_id`'s record, or overwrites the existing slot in place
  /// when the node is already cached (with another level's record): the slot
  /// keeps at most one level's record per node, retagged to the new range.
  void InsertOrUpdate(tableint node_id, const char *data, size_t size, layer_t layer_begin,
                      layer_t layer_end) {
    if (num_slots_ == 0 || size > piece_bytes_)
      return;
    if (static_cast<size_t>(node_id) >= num_nodes_)
      return;

    tableint slot = lookup_.Find(node_id);
    if (slot != kInvalid) {
      slot_layer_begin_[slot] = layer_begin;
      slot_layer_end_[slot] = layer_end;
      std::memcpy(storage_.data() + static_cast<size_t>(slot) * piece_bytes_, data, size);
      if (policy_ == EvictPolicy::LRU && slot != lru_head_) {
        LruDetach(slot);
        LruPushFront(slot);
      }
      return;
    }

    slot = AllocateSlot();
    lookup_.Insert(node_id, slot);
    slot_to_id_[slot] = node_id;
    slot_layer_begin_[slot] = layer_begin;
    slot_layer_end_[slot] = layer_end;
    std::memcpy(storage_.data() + static_cast<size_t>(slot) * piece_bytes_, data, size);

    if (policy_ == EvictPolicy::LRU) {
      LruPushFront(slot);
    }
  }

  void Clear() {
    if (num_slots_ != 0 && count_ != 0) {
      lookup_.Clear(slot_to_id_, count_);
      count_ = 0;
      clock_hand_ = 0;
      if (policy_ == EvictPolicy::LRU) {
        lru_head_ = kInvalid;
        lru_tail_ = kInvalid;
      }
    }
    // Grow towards the previous query's demand. Safe here and nowhere else:
    // every mapping was just invalidated, so no live Entry::data can point into
    // the old storage_. Overshoot slightly so a query that grows by a few
    // records does not trigger a reallocation every time.
    if (auto_size_ && wanted_slots_ > num_slots_ && num_slots_ < max_slots_) {
      const size_t target =
          std::min(max_slots_, std::max(wanted_slots_ + wanted_slots_ / 8 + 8, num_slots_ * 2));
      ResizeSlots(target);
    }
    wanted_slots_ = 0;
  }

  auto capacity() const -> size_t {
    return num_slots_;
  }
  auto max_capacity() const -> size_t {
    return max_slots_;
  }
  auto size() const -> size_t {
    return count_;
  }
  auto auto_size() const -> bool {
    return auto_size_;
  }
  /// Highest slot occupancy ever reached (== working set when nothing evicted).
  auto peak_slots() const -> size_t {
    return peak_count_;
  }
  /// Highest number of records a single query asked to insert. Above
  /// max_capacity() it means the budget forced evictions that a larger cache
  /// would have avoided.
  auto peak_wanted_slots() const -> size_t {
    return peak_wanted_slots_;
  }
  /// Bytes this instance currently holds, including the direct-mapped index.
  auto payload_bytes() const -> size_t {
    return storage_.capacity();
  }
  auto lookup_bytes() const -> size_t {
    return lookup_.Bytes();
  }
  auto resident_bytes() const -> size_t {
    return storage_.capacity() + lookup_.Bytes() +
           (slot_to_id_.capacity() + slot_next_.capacity() + slot_prev_.capacity()) *
               sizeof(tableint) +
           (slot_layer_begin_.capacity() + slot_layer_end_.capacity()) * sizeof(layer_t);
  }

private:
  static constexpr tableint kInvalid = static_cast<tableint>(-1);
  // Sized so a typical query fits on the very first try: growth only happens
  // at a Clear(), so a query that overflows evicts and loses hits it would
  // otherwise have had. One query's worth of records is a few hundred for the
  // tested efs range; 1024 covers that with room to spare and still costs
  // ~0.5 MiB, two orders of magnitude below a cross-query budget. Anything
  // larger converges after a single query.
  static constexpr size_t kAutoInitialSlots = 1024;

  void ResizeSlots(size_t new_slots) {
    num_slots_ = new_slots;
    lookup_.Configure(num_nodes_, num_slots_);
    storage_.assign(num_slots_ * piece_bytes_, 0);
    slot_to_id_.assign(num_slots_, kInvalid);
    slot_layer_begin_.assign(num_slots_, layer_t(0));
    slot_layer_end_.assign(num_slots_, layer_t(0));
    if (policy_ == EvictPolicy::LRU) {
      slot_next_.assign(num_slots_, kInvalid);
      slot_prev_.assign(num_slots_, kInvalid);
    }
    lru_head_ = kInvalid;
    lru_tail_ = kInvalid;
    count_ = 0;
    clock_hand_ = 0;
  }

  auto AllocateSlot() -> tableint {
    if (++wanted_slots_ > peak_wanted_slots_)
      peak_wanted_slots_ = wanted_slots_;
    if (count_ < num_slots_) {
      // Still have free slots — use next sequential
      if (count_ + 1 > peak_count_)
        peak_count_ = count_ + 1;
      return static_cast<tableint>(count_++);
    }
    // Evict
    if (policy_ == EvictPolicy::LRU) {
      tableint victim = lru_tail_;
      lookup_.Erase(slot_to_id_[victim]);
      LruDetach(victim);
      return victim;
    }
    // FIFO: circular clock hand
    tableint victim = static_cast<tableint>(clock_hand_);
    clock_hand_ = (clock_hand_ + 1) % num_slots_;
    lookup_.Erase(slot_to_id_[victim]);
    return victim;
  }

  void LruDetach(tableint slot) {
    tableint p = slot_prev_[slot];
    tableint n = slot_next_[slot];
    if (p != kInvalid)
      slot_next_[p] = n;
    else
      lru_head_ = n;
    if (n != kInvalid)
      slot_prev_[n] = p;
    else
      lru_tail_ = p;
  }

  void LruPushFront(tableint slot) {
    slot_prev_[slot] = kInvalid;
    slot_next_[slot] = lru_head_;
    if (lru_head_ != kInvalid)
      slot_prev_[lru_head_] = slot;
    else
      lru_tail_ = slot;
    lru_head_ = slot;
  }

  EvictPolicy policy_{EvictPolicy::FIFO};
  size_t piece_bytes_{0};
  size_t num_slots_{0}; // slots currently allocated
  size_t max_slots_{0}; // configured budget / piece_bytes_ (upper bound)
  size_t num_nodes_{0};
  bool auto_size_{false};
  size_t wanted_slots_{0};      // AllocateSlot calls since the last Clear()
  size_t peak_wanted_slots_{0}; // max of wanted_slots_ over all queries
  size_t peak_count_{0};        // max of count_ over all queries
  std::vector<char> storage_;

  SlotLookup lookup_;
  std::vector<tableint> slot_to_id_; // size = num_slots_
  std::vector<layer_t> slot_layer_begin_;
  std::vector<layer_t> slot_layer_end_;

  // LRU-only state (unused in FIFO mode)
  std::vector<tableint> slot_next_;
  std::vector<tableint> slot_prev_;
  tableint lru_head_{kInvalid};
  tableint lru_tail_{kInvalid};

  size_t count_{0};
  size_t clock_hand_{0}; // FIFO circular eviction pointer
};

} // namespace alliance
