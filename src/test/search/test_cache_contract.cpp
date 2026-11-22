#include "support/test_check.hpp"
#include "cache/dynamic_block_cache.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <list>
#include <random>
#include <unordered_map>
#include "cache/static_node_lookup.hpp"
#include <sstream>
#include "search/disk_search_cache.hpp"
#include <array>
#include <stdexcept>
#include "search/query_visited.hpp"
#include <cstdint>
#include <vector>

namespace dynamic_cache {

using namespace alliance;
struct Record {
  int value;
  layer_t lo, hi;
};
// Independent map/list reference, intentionally not based on cache slot IDs.
struct Model {
  size_t capacity = 0;
  bool lru = false;
  std::list<tableint> order;
  std::unordered_map<tableint, Record> entries;
  void clear() {
    order.clear();
    entries.clear();
  }
  void touch(tableint id) {
    order.remove(id);
    order.push_back(id);
  }
  void put(tableint id, Record r, bool update) {
    if (!capacity)
      return;
    if (entries.count(id)) {
      if (update) {
        entries[id] = r;
        if (lru)
          touch(id);
      }
      return;
    }
    if (entries.size() == capacity) {
      entries.erase(order.front());
      order.pop_front();
    }
    entries[id] = r;
    order.push_back(id);
  }
};
void TestReplacementAndReuse() {
  std::mt19937 rng(713);
  for (bool lru : {false, true})
    for (size_t n : {size_t(17), size_t(4096)}) {
      DynamicBlockCache cache;
      Model model;
      model.lru = lru;
      auto configure = [&](size_t cap) {
        cache.Configure(
            cap * 16, 16,
            lru ? DynamicBlockCache::EvictPolicy::LRU : DynamicBlockCache::EvictPolicy::FIFO, n);
        model.capacity = cap;
        model.clear();
        ALLIANCE_TEST_CHECK(cache.size() == 0);
      };
      auto lookup = [&](tableint id) {
        auto actual = cache.Lookup(id);
        auto expected = model.entries.find(id);
        ALLIANCE_TEST_CHECK(actual.found == (expected != model.entries.end()));
        if (actual.found) {
          int value;
          std::memcpy(&value, actual.data, sizeof(value));
          ALLIANCE_TEST_CHECK(value == expected->second.value);
          ALLIANCE_TEST_CHECK(actual.layer_begin == expected->second.lo &&
                              actual.layer_end == expected->second.hi);
          if (lru)
            model.touch(id);
        }
      };
      for (size_t cap : {size_t(0), size_t(1), size_t(7), size_t(300)}) {
        configure(cap);
        for (int step = 0; step < 20000; ++step) {
          tableint id = rng() % n;
          int op = rng() % 10;
          if (op < 6) {
            Record r{int(rng()), int(rng() % 5), 0};
            r.hi = r.lo + 2;
            if (op < 3)
              cache.InsertIfMissing(id, reinterpret_cast<char *>(&r.value), sizeof(r.value), r.lo,
                                    r.hi);
            else
              cache.InsertOrUpdate(id, reinterpret_cast<char *>(&r.value), sizeof(r.value), r.lo,
                                   r.hi);
            model.put(id, r, op >= 3);
          } else if (op < 9)
            lookup(id);
          else {
            cache.Clear();
            model.clear();
            cache.Clear();
            ALLIANCE_TEST_CHECK(cache.size() == 0);
            // Old owners must never become hits when slots are reused.
            for (tableint x = 0; x < n; ++x)
              ALLIANCE_TEST_CHECK(!cache.Lookup(x).found);
          }
          ALLIANCE_TEST_CHECK(cache.size() == model.entries.size());
        }
        for (tableint id = 0; id < n; ++id)
          lookup(id);
        configure(0);
        ALLIANCE_TEST_CHECK(cache.size() == 0 && !cache.Lookup(0).found);
        ALLIANCE_TEST_CHECK(cache.resident_bytes() == 0 && cache.payload_bytes() == 0 &&
                            cache.lookup_bytes() == 0);
      }
      // Fill past capacity, then clear: both eviction orders must invalidate
      // precisely the current owners, including the contiguous-fill branch.
      configure(n / 2);
      for (tableint id = 0; id < n; ++id) {
        int v = id;
        cache.InsertOrUpdate(id, reinterpret_cast<char *>(&v), 4, 1, 3);
      }
      cache.Clear();
      ALLIANCE_TEST_CHECK(cache.size() == 0);
      for (tableint id = 0; id < n; ++id)
        ALLIANCE_TEST_CHECK(!cache.Lookup(id).found);
      configure(7);
      int v = 42;
      cache.InsertIfMissing(3, reinterpret_cast<char *>(&v), 4, 2, 4);
      ALLIANCE_TEST_CHECK(cache.Lookup(3).found && cache.Lookup(3).layer_end == 4);
    }
  // Long eviction chains exercise backward shifts and wrapping probe clusters.
  // Compare with the independent map/list model while every insertion is a
  // possible eviction; unlike the cases above this rarely clears the cache.
  for (bool lru : {false, true})
    for (size_t cap : {size_t(1), size_t(3), size_t(31), size_t(257)}) {
      DynamicBlockCache cache;
      Model model;
      model.capacity = cap;
      model.lru = lru;
      cache.Configure(cap * 16, 16,
                      lru ? DynamicBlockCache::EvictPolicy::LRU
                          : DynamicBlockCache::EvictPolicy::FIFO,
                      1000000);
      for (int step = 0; step < 100000; ++step) {
        tableint id = (rng() % 1021) * 977;
        int value = step;
        if (step % 7 == 0) {
          auto e = cache.Lookup(id);
          auto it = model.entries.find(id);
          ALLIANCE_TEST_CHECK(e.found == (it != model.entries.end()));
          if (e.found) {
            int v;
            std::memcpy(&v, e.data, 4);
            ALLIANCE_TEST_CHECK(v == it->second.value);
            if (lru)
              model.touch(id);
          }
        } else {
          bool update = (step % 3 == 0);
          Record r{value, step % 5, step % 5 + 1};
          if (update)
            cache.InsertOrUpdate(id, reinterpret_cast<char *>(&value), 4, r.lo, r.hi);
          else
            cache.InsertIfMissing(id, reinterpret_cast<char *>(&value), 4, r.lo, r.hi);
          model.put(id, r, update);
        }
        ALLIANCE_TEST_CHECK(cache.size() == model.entries.size());
        if (step % 10000 == 9999) {
          cache.Clear();
          model.clear();
        }
      }
#ifndef ALLIANCE_DENSE_DYNAMIC_CACHE
      ALLIANCE_TEST_CHECK(cache.lookup_bytes() < 32768);
#endif
    }
  // Auto growth only at query boundaries, bounds and retained allocation reset.
  DynamicBlockCache auto_cache;
  auto_cache.Configure(4096 * 16, 16, DynamicBlockCache::EvictPolicy::FIFO, 1000000, true);
  ALLIANCE_TEST_CHECK(auto_cache.capacity() == 1024);
  for (tableint id = 0; id < 2000; ++id) {
    int v = id;
    auto_cache.InsertOrUpdate(id, reinterpret_cast<char *>(&v), 4, 1, 2);
  }
  ALLIANCE_TEST_CHECK(auto_cache.capacity() == 1024);
  auto_cache.Clear();
  ALLIANCE_TEST_CHECK(auto_cache.capacity() >= 2000 && auto_cache.capacity() <= 4096);
  for (tableint id = 0; id < 2000; ++id)
    ALLIANCE_TEST_CHECK(!auto_cache.Lookup(id).found);
  int value = 1;
  auto_cache.InsertOrUpdate(1000000, reinterpret_cast<char *>(&value), 4, 1, 2);
  ALLIANCE_TEST_CHECK(!auto_cache.Lookup(1000000).found && auto_cache.size() == 0);
  auto_cache.Configure(0, 16, DynamicBlockCache::EvictPolicy::FIFO, 1000000, true);
  ALLIANCE_TEST_CHECK(auto_cache.resident_bytes() == 0);
  std::cout << "DynamicBlockCache randomized FIFO/LRU clear/reconfigure tests passed\n";
}

} // namespace dynamic_cache

namespace static_lookup {

using namespace alliance;
static std::stringstream Encode(const std::vector<StaticNodeMeta> &a) {
  std::stringstream s(std::ios::in | std::ios::out | std::ios::binary);
  s.write(reinterpret_cast<const char *>(a.data()), a.size() * sizeof(StaticNodeMeta));
  s.seekg(0);
  return s;
}
void TestSerializationAndBounds() {
  // Offsets across 4GiB are tested without allocating a multi-GiB payload.
  // Sparse virtual memory below verifies the final pointer arithmetic too.
  {
    const uint64_t boundary = uint64_t(1) << 32;
    std::vector<StaticNodeMeta64> dense(4);
    dense[0].offset = boundary - 32;
    dense[1].offset = boundary;
    dense[2].offset = boundary + 64;
    for (size_t i = 0; i < 3; ++i)
      dense[i].record_bytes = 32;
    StaticNodeLookup wide;
    std::stringstream encoded(std::ios::in | std::ios::out | std::ios::binary);
    encoded.write(reinterpret_cast<const char *>(dense.data()),
                  dense.size() * sizeof(StaticNodeMeta64));
    encoded.seekg(0);
    wide.LoadDense(encoded, dense.size(), boundary + 96, kStaticCacheV3);
    for (size_t i = 0; i < 3; ++i)
      ALLIANCE_TEST_CHECK(wide.Find64(i)->offset == dense[i].offset);
    ALLIANCE_TEST_CHECK(!wide.Find64(3));
    auto *memory = static_cast<char *>(mmap(nullptr, boundary + 96, PROT_READ | PROT_WRITE,
                                            MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0));
    ALLIANCE_TEST_CHECK(memory != MAP_FAILED);
    for (size_t i = 0; i < 3; ++i)
      memory[dense[i].offset] = char(i + 17);
    for (size_t i = 0; i < 3; ++i)
      ALLIANCE_TEST_CHECK(memory[wide.Find64(i)->offset] == char(i + 17));
    munmap(memory, boundary + 96);
    for (int variant = 0; variant < 4; ++variant) {
      auto bad = dense;
      if (variant == 0)
        bad[2].offset = UINT64_MAX - 1;
      if (variant == 1)
        bad[2].record_bytes = 33;
      if (variant == 2)
        bad[2].reserved = 1;
      std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
      stream.write(reinterpret_cast<const char *>(bad.data()),
                   bad.size() * sizeof(StaticNodeMeta64));
      stream.seekg(0);
      bool rejected = false;
      try {
        wide.LoadDense(stream, bad.size(), boundary + 96, variant == 3 ? 99 : kStaticCacheV3);
      } catch (const std::exception &) {
        rejected = true;
      }
      ALLIANCE_TEST_CHECK(rejected);
    }
    std::vector<StaticNodeMeta> old(3);
    old[0] = {0, 0, 1, 32};
    old[2] = {64, 2, 3, 32};
    std::stringstream legacy(std::ios::in | std::ios::out | std::ios::binary);
    legacy.write(reinterpret_cast<const char *>(old.data()), old.size() * sizeof(StaticNodeMeta));
    legacy.seekg(0);
    wide.LoadDense(legacy, old.size(), 96, kStaticCacheV2);
    ALLIANCE_TEST_CHECK(wide.Count() == 2 && !wide.Find(1));
    ALLIANCE_TEST_CHECK(wide.Find(2)->offset == 64 && wide.Find(2)->Covers(3));
    ALLIANCE_TEST_CHECK(size_t(legacy.tellg()) == 48);
    ALLIANCE_TEST_CHECK(wide.Bytes() == StaticNodeLookup::MetadataBytes(3, 2));
  }

  std::mt19937 rng(8765);
  StaticNodeLookup lookup;
  for (size_t n : {size_t(0), size_t(1), size_t(63), size_t(64), size_t(65), size_t(10003)})
    for (unsigned stride : {1u, 7u, 211u}) {
      std::vector<StaticNodeMeta> dense(n);
      size_t count = 0;
      for (size_t i = 0; i < n; ++i)
        if (i % stride == 0 || i + 1 == n) {
          dense[i].offset = count++ * 32;
          dense[i].record_bytes = 32;
          dense[i].layer_begin = rng() % 9;
          dense[i].layer_end = dense[i].layer_begin + 3;
        }
      auto stream = Encode(dense);
      lookup.LoadDense(stream, n, count * 32);
      ALLIANCE_TEST_CHECK(lookup.Count() == count);
      ALLIANCE_TEST_CHECK(lookup.Bytes() == StaticNodeLookup::MetadataBytes(n, count));
      ALLIANCE_TEST_CHECK(size_t(stream.tellg()) == n * sizeof(StaticNodeMeta));
      for (size_t i = 0; i < n; ++i) {
        const auto *p = lookup.Find(i);
        ALLIANCE_TEST_CHECK(bool(p) == dense[i].IsCached());
        if (p)
          ALLIANCE_TEST_CHECK(
              p->offset == dense[i].offset && p->layer_begin == dense[i].layer_begin &&
              p->layer_end == dense[i].layer_end && p->record_bytes == dense[i].record_bytes);
      }
      ALLIANCE_TEST_CHECK(!lookup.Find(n) && !lookup.Find(UINT32_MAX));
      ALLIANCE_TEST_CHECK(StaticNodeLookup::BudgetBytes(n, count, count * 32) ==
                          (count ? ((count * 32 + 4095) & ~size_t(4095)) + lookup.Bytes() : 0));
    }
  std::vector<StaticNodeMeta> bad(65);
  bad[64].offset = 63;
  bad[64].record_bytes = 2;
  auto stream = Encode(bad);
  bool rejected = false;
  try {
    lookup.LoadDense(stream, 65, 64);
  } catch (const std::runtime_error &) {
    rejected = true;
  }
  ALLIANCE_TEST_CHECK(rejected);
  std::stringstream truncated(std::ios::in | std::ios::out | std::ios::binary);
  rejected = false;
  try {
    lookup.LoadDense(truncated, 65, 64);
  } catch (const std::runtime_error &) {
    rejected = true;
  }
  ALLIANCE_TEST_CHECK(rejected);
  lookup.Clear();
  ALLIANCE_TEST_CHECK(lookup.Bytes() == 0 && !lookup.Find(0));
  std::cout << "StaticNodeLookup rank, sparse/dense loading, bounds and budget tests passed\n";
}

} // namespace static_lookup

namespace cache_admission {

// Cache admission contracts independent of graph traversal and IO scheduling.

using namespace alliance;
using alliance::detail::DecodeBlockIntoCache;
using alliance::detail::MultiLevelVisited;

namespace {
struct Record {
  tableint id;
  uint32_t marker;
};
void Check(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
void Expect(DynamicBlockCache &cache, tableint id, layer_t lo, layer_t hi, uint32_t marker) {
  const auto e = cache.Lookup(id);
  Check(e.found && e.layer_begin == lo && e.layer_end == hi, "wrong record coverage");
  Record record{};
  std::memcpy(&record, e.data, sizeof(record));
  Check(record.id == id && record.marker == marker, "wrong cached record bytes");
}

void Admission(DynamicBlockCache::EvictPolicy policy, bool with_visited) {
  DynamicBlockCache cache;
  cache.Configure(16 * sizeof(Record), sizeof(Record), policy, 16);
  for (tableint id : {1u, 2u, 4u}) {
    const Record old{id, 99};
    cache.InsertOrUpdate(id, reinterpret_cast<const char *>(&old), sizeof(old), 8, 9);
  }
  // Requester 1 is both visited and statically covered, waiter 2 is visited.
  // Both must replace their stale level; prefetch must never replace node 4.
  MultiLevelVisited visited(16);
  visited.Clear();
  for (tableint id : {1u, 2u, 3u})
    visited.Set(id);
  std::array<StaticNodeMeta, 16> meta{};
  for (tableint id : {1u, 5u, 6u}) {
    meta[id].offset = 0;
    meta[id].record_bytes = sizeof(Record);
    meta[id].layer_begin = 1;
    meta[id].layer_end = id == 6 ? 2 : 3;
  }
  std::istringstream stream(std::string(reinterpret_cast<const char *>(meta.data()), sizeof(meta)));
  StaticNodeLookup lookup;
  lookup.LoadDense(stream, meta.size(), sizeof(Record));
  const std::array<Record, 10> block{{{0, 10},
                                      {1, 11},
                                      {2, 12},
                                      {3, 13},
                                      {4, 14},
                                      {5, 15},
                                      {6, 16},
                                      {7, 17},
                                      {tableint(-1), 0},
                                      {8, 18}}};
  const tableint requested[] = {1, 2};
  DecodeBlockIntoCache(cache, reinterpret_cast<const char *>(block.data()), sizeof(Record),
                       sizeof(block), 1, 3, requested, 2, with_visited ? &visited : nullptr,
                       &lookup);
  Expect(cache, 0, 1, 3, 10);
  Expect(cache, 1, 1, 3, 11);
  Expect(cache, 2, 1, 3, 12);
  if (with_visited)
    Check(!cache.Lookup(3).found, "visited prefetch must be skipped");
  else
    Expect(cache, 3, 1, 3, 13);
  Expect(cache, 4, 8, 9, 99);
  Check(!cache.Lookup(5).found, "fully static record must not occupy dynamic cache");
  Expect(cache, 6, 1, 3, 16);
  Expect(cache, 7, 1, 3, 17);
  Check(!cache.Lookup(8).found, "record after padding sentinel must be ignored");
  // Loading an already cached prefetch must not change its coverage or value.
  DecodeBlockIntoCache(cache, reinterpret_cast<const char *>(block.data()), sizeof(Record),
                       sizeof(block), 4, 7, nullptr, 0);
  Expect(cache, 4, 8, 9, 99);
  cache.Clear();
  Check(cache.size() == 0, "query boundary failed to clear cache");
}

void SmallCapacity(DynamicBlockCache::EvictPolicy policy) {
  const std::array<Record, 2> block{{{1, 11}, {2, 12}}};
  const tableint requested = 1;
  for (size_t slots : {size_t(0), size_t(1)}) {
    DynamicBlockCache cache;
    cache.Configure(slots * sizeof(Record), sizeof(Record), policy, 16);
    DecodeBlockIntoCache(cache, reinterpret_cast<const char *>(block.data()), sizeof(Record),
                         sizeof(block), 1, 3, &requested, 1);
    Check(cache.size() == slots, "cache exceeded configured slot count");
    Check(!cache.Lookup(1).found, "request retention is not a cache guarantee");
    if (slots)
      Expect(cache, 2, 1, 3, 12);
    cache.Clear();
    // A partial trailing record is ignored, without inspecting its ID/payload.
    DecodeBlockIntoCache(cache, reinterpret_cast<const char *>(block.data()), sizeof(Record),
                         sizeof(Record) + sizeof(tableint), 1, 3, &requested, 1);
    if (slots)
      Expect(cache, 1, 1, 3, 11);
    Check(!cache.Lookup(2).found, "partial trailing record was admitted");
  }
}
} // namespace

void TestBlockAdmission() {
  for (auto policy : {DynamicBlockCache::EvictPolicy::FIFO, DynamicBlockCache::EvictPolicy::LRU}) {
    Admission(policy, false);
    Admission(policy, true);
    SmallCapacity(policy);
  }
  std::cout << "disk search cache admission contracts passed\n";
}

} // namespace cache_admission

namespace query_visited {

void TestLifecycleAndIsolation() {
  std::mt19937 rng(532431);
  for (size_t n :
       {size_t(0), size_t(1), size_t(63), size_t(64), size_t(65), size_t(2003), size_t(1000000)}) {
    alliance::QueryVisited a(n), b(n);
    std::vector<uint8_t> expected(n, 0);
    const auto bytes = a.Bytes();
    for (size_t q = 0; q < 1000; ++q) {
      a.Clear();
      a.Clear();
      std::fill(expected.begin(), expected.end(), 0);
      for (size_t j = 0; n && j < 512; ++j) {
        size_t id = rng() % n;
        ALLIANCE_TEST_CHECK(a.Test(id) == bool(expected[id]));
        a.Set(id);
        a.Set(id);
        expected[id] = 1;
        ALLIANCE_TEST_CHECK(a.Test(id));
        ALLIANCE_TEST_CHECK(!b.Test(id));
      }
      ALLIANCE_TEST_CHECK(a.Bytes() == bytes);
    }
    if (n) {
      a.Clear();
      for (size_t id = 0; id < n; ++id) {
        ALLIANCE_TEST_CHECK(!a.Test(id));
        a.Set(id);
        a.Set(id);
      }
      a.Clear();
      for (size_t id = 0; id < n; ++id)
        ALLIANCE_TEST_CHECK(!a.Test(id));
      a.Set(n - 1);
      ALLIANCE_TEST_CHECK(a.Test(n - 1));
      a.Clear();
      ALLIANCE_TEST_CHECK(!a.Test(n - 1));
    }
  }
  std::cout << "QueryVisited exactness, reuse, boundaries and bounded allocation passed\n";
}

} // namespace query_visited

int main() {
  return alliance::test::RunSuite("cache contracts", [] {
    dynamic_cache::TestReplacementAndReuse();
    static_lookup::TestSerializationAndBounds();
    cache_admission::TestBlockAdmission();
    query_visited::TestLifecycleAndIsolation();
  });
}
