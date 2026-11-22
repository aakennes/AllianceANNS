#include "shuffle/order/ordered_block_bnf.hpp"
#include "shuffle/adjacency/external_sort.hpp"
#include "shuffle/memory/storage_plan.hpp"
#include <thread>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <cstdio>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <system_error>
#include <tuple>
#include <cstdlib>

using namespace alliance;
using namespace alliance::shuffle_detail;

static void Check(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}

static void CheckPacking() {
  struct Case {
    size_t n;
    uint8_t weight;
    bool packed;
  };
  for (const auto c : {Case{1, 255, true}, Case{100000000, 32, true}, Case{100000000, 33, false},
                       Case{size_t(1) << 31, 2, true}, Case{size_t(1) << 31, 3, false},
                       Case{size_t(1) << 32, 1, true}, Case{size_t(1) << 32, 255, false}}) {
    WeightedEdges edges;
    EdgeCodec codec(c.n, c.weight);
    std::vector<uint8_t> encoded(3 * codec.Bytes());
    edges.Allocate(c.n, c.weight, 3);
    std::vector<tableint> ids{0, tableint((c.n - 1) / 2), tableint(c.n - 1)};
    const uint8_t weights[] = {c.weight, 1, c.weight};
    for (size_t i = 0; i < ids.size(); ++i) {
      edges.Set(i, ids[i], weights[i]);
      codec.Encode(encoded.data() + i * codec.Bytes(), ids[i], weights[i]);
      Check(codec.Id(encoded.data() + i * codec.Bytes()) == ids[i] &&
                codec.Weight(encoded.data() + i * codec.Bytes()) == weights[i],
            "disk codec round trip");
    }
    Check(edges.Packed() == c.packed, "packing mode boundary");
    Check(edges.StorageBytes() == 3 * (c.packed ? 4 : 5), "packed allocation size");
    for (size_t i = 0; i < ids.size(); ++i) {
      Check(edges.Neighbor(i) == ids[i] && edges.Weight(i) == weights[i], "edge round trip");
      Check(edges.Contains(0, 3, ids[i]), "decoded-ID membership");
    }
    Check(!edges.Contains(1, 1, 0), "empty adjacency membership");
    if (c.n > 4)
      Check(!edges.Contains(0, 3, 1), "absent adjacency membership");
    bool rejected = false;
    try {
      edges.Set(0, 0, 0);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    Check(rejected, "zero weight rejected");
  }
}

static void CheckScores() {
  SparseScores scores(96);
  Check(scores.StorageBytes() < 8192, "sparse scoring must not scale with global block count");
  std::mt19937_64 rng(20261115);
  for (size_t row = 0; row < 200; ++row) {
    std::map<size_t, size_t> expected;
    std::vector<size_t> order;
    for (size_t j = 0; j < 96; ++j) {
      const size_t block = (rng() % 48) * 1048576;
      const size_t value = 1 + rng() % 65025;
      if (!expected.count(block))
        order.push_back(block);
      expected[block] += value;
      Check(scores.Add(block, value) == expected[block], "sparse accumulation");
    }
    Check(scores.Get(std::numeric_limits<size_t>::max()) == 0, "absent score");
    Check(scores.Touched().size() == order.size(), "one touched entry per block");
    for (size_t i = 0; i < order.size(); ++i) {
      const size_t slot = scores.Touched()[i];
      Check(scores.Key(slot) == order[i] && scores.Value(slot) == expected[order[i]],
            "first-occurrence tie order");
    }
    scores.Clear();
    for (size_t block : order)
      Check(scores.Get(block) == 0, "clear collision chain");
  }
}

using Rows = std::vector<std::vector<std::pair<tableint, uint8_t>>>;

// Independent scalar oracle: associative score sums and a stable sort instead
// of packed storage, hashing, counting sort, batching or membership filters.
static std::vector<tableint> ReferenceRound(const Rows &rows, size_t epb,
                                            const std::vector<size_t> &order,
                                            std::vector<size_t> &old_map, size_t &moved) {
  const size_t n = rows.size(), blocks = (n + epb - 1) / epb;
  std::vector<size_t> used(blocks, 0), next(n);
  std::vector<tableint> nodes(n);
  size_t first_available = 0;
  moved = 0;
  auto capacity = [&](size_t b) { return std::min(epb, n - b * epb); };
  for (size_t u : order) {
    std::map<size_t, size_t> scores;
    std::vector<size_t> touched;
    for (auto [v, weight] : rows[u]) {
      const size_t b = old_map[v];
      if (!scores.count(b))
        touched.push_back(b);
      scores[b] += size_t(weight) * weight;
    }
    std::stable_sort(touched.begin(), touched.end(),
                     [&](size_t a, size_t b) { return scores[a] > scores[b]; });
    const size_t old_block = old_map[u];
    const size_t stay = scores.count(old_block) ? scores[old_block] : 0;
    size_t chosen = blocks, chosen_score = 0;
    for (size_t begin = 0; begin < touched.size();) {
      size_t end = begin + 1;
      while (end < touched.size() && scores[touched[end]] == scores[touched[begin]])
        ++end;
      std::vector<size_t> available;
      for (size_t j = begin; j < end; ++j)
        if (used[touched[j]] < capacity(touched[j]))
          available.push_back(touched[j]);
      if (!available.empty()) {
        chosen = available.front();
        chosen_score = scores[touched[begin]];
        size_t best = 0;
        if (available.size() > 1)
          for (size_t b : available) {
            size_t reciprocal = 0;
            for (size_t k = 0; k < used[b]; ++k) {
              const auto &adjacency = rows[nodes[b * epb + k]];
              reciprocal += std::any_of(adjacency.begin(), adjacency.end(),
                                        [&](auto edge) { return edge.first == u; });
            }
            if (reciprocal > best) {
              best = reciprocal;
              chosen = b;
            }
          }
        break;
      }
      begin = end;
    }
    if (chosen < blocks && chosen != old_block && used[old_block] < capacity(old_block) &&
        chosen_score <= stay + 1)
      chosen = old_block;
    if (chosen == blocks && used[old_block] < capacity(old_block))
      chosen = old_block;
    if (chosen == blocks) {
      while (first_available < blocks && used[first_available] == capacity(first_available))
        ++first_available;
      chosen = first_available;
    }
    Check(chosen < blocks, "reference capacity");
    nodes[chosen * epb + used[chosen]++] = static_cast<tableint>(u);
    next[u] = chosen;
    moved += chosen != old_block;
  }
  old_map.swap(next);
  return nodes;
}

static void CheckRounds() {
  ScratchDirectory temporary(std::getenv("TEST_TMPDIR") ? std::getenv("TEST_TMPDIR") : "/tmp");
  std::mt19937 rng(20261115);
  size_t cases = 0;
  for (size_t n : {size_t(1), size_t(7), size_t(73), size_t(257)}) {
    for (size_t fixture = 0; fixture < 3; ++fixture) {
      Rows rows(n);
      FlatWeightedGraph graph;
      graph.num_nodes = n;
      graph.offsets.assign(n + 1, 0);
      graph.in_weights.assign(n, 0);
      uint8_t maximum = 1;
      for (size_t u = 0; u < n; ++u) {
        std::map<tableint, uint8_t> row;
        if (fixture)
          for (size_t j = 0; j < 24; ++j)
            ++row[fixture == 1 ? tableint((u + j) % n) : tableint(rng() % n)];
        for (auto [v, w] : row) {
          rows[u].push_back({v, w});
          graph.in_weights[v] += w;
          maximum = std::max(maximum, w);
        }
        graph.offsets[u + 1] = graph.offsets[u] + rows[u].size();
      }
      // Exercise fallback ordering without allocating a huge node space: only
      // the codec sees a larger address range; every stored ID stays below n.
      graph.edges.Allocate(fixture == 2 ? size_t(1) << 32 : n, maximum, graph.offsets.back());
      for (size_t u = 0; u < n; ++u)
        for (size_t j = 0; j < rows[u].size(); ++j)
          graph.edges.Set(graph.offsets[u] + j, rows[u][j].first, rows[u][j].second);
      const auto order = SortNodesByInWeight(graph.in_weights);
      FlatWeightedGraph disk;
      disk.num_nodes = n;
      ExternalBudget budget;
      budget.sort_bytes = 512;
      budget.fan_in = 2;
      budget.batch_bytes = 128;
      disk.external = std::make_shared<ExternalGraph>(temporary.Path(), n, maximum,
                                                      graph.offsets.back(), budget);
      std::vector<uint8_t> bytes(graph.offsets.back() * disk.external->Codec().Bytes());
      for (size_t u = 0; u < n; ++u)
        for (size_t j = 0; j < rows[u].size(); ++j)
          disk.external->Codec().Encode(bytes.data() +
                                            (graph.offsets[u] + j) * disk.external->Codec().Bytes(),
                                        rows[u][j].first, rows[u][j].second);
      disk.external->Write(0, bytes.data(), bytes.size());
      disk.external->Prepare(order, graph.offsets);
      for (size_t epb : {size_t(1), size_t(5), size_t(10)}) {
        for (auto [threads, batch] :
             {std::pair{1, size_t(1)}, {4, size_t(17)}, {4, size_t(16384)}}) {
          BNFParams params;
          params.threads = threads;
          params.affinity_batch_size = batch;
          for (const FlatWeightedGraph *backend : {&graph, &disk}) {
            OrderedBlockBNF actual(*backend, epb, params);
            std::vector<size_t> expected_map(n), actual_map(n);
            for (size_t u = 0; u < n; ++u)
              expected_map[u] = actual_map[u] = u / epb;
            std::vector<tableint> expected_order;
            for (size_t round = 0; round < 5; ++round) {
              size_t expected_moved;
              expected_order = ReferenceRound(rows, epb, order, expected_map, expected_moved);
              Check(actual.Round(order, actual_map, round + 1) == expected_moved, "moved count");
              Check(actual_map == expected_map, "round assignment differs from scalar reference");
            }
            Check(actual.TakeOrder() == expected_order, "final permutation differs");
            ++cases;
          }
        }
      }
    }
  }
  std::printf("SHUFFLE_MEMORY_PASS scalar_cases=%zu rounds_per_case=5\n", cases);
}

static void CheckExternalBounds() {
  ScratchDirectory temporary(std::getenv("TEST_TMPDIR") ? std::getenv("TEST_TMPDIR") : "/tmp");
  {
    ScratchIo io;
    IncomingSorter sorter(temporary.Path(), io, 64, 2);
    std::vector<uint64_t> expected;
    for (size_t i = 0; i < 1001; ++i) {
      const uint64_t key = (i * 7919) % 997;
      expected.push_back(key);
      sorter.Add(key);
    }
    std::sort(expected.begin(), expected.end());
    std::vector<uint64_t> actual;
    sorter.Finish([&](uint64_t key) { actual.push_back(key); });
    Check(actual == expected, "multi-pass external merge loses keys");
  }
  Check(std::filesystem::is_empty(temporary.Path()), "sort scratch not released");
  {
    // Node zero has 99 incoming neighbors: larger than this 16-byte batch cap.
    ExternalBudget budget;
    budget.batch_bytes = 16;
    budget.sort_bytes = 64;
    budget.fan_in = 2;
    ExternalGraph graph(temporary.Path(), 100, 1, 100, budget);
    std::vector<uint8_t> bytes(400);
    for (size_t i = 0; i < 100; ++i)
      graph.Codec().Encode(bytes.data() + 4 * i, 0, 1);
    graph.Write(0, bytes.data(), bytes.size());
    std::vector<size_t> order(100), offsets(101);
    std::iota(order.begin(), order.end(), 0);
    std::iota(offsets.begin(), offsets.end(), 0);
    graph.Prepare(order, offsets);
    Check(graph.LoadBatch(0, 100) == 1, "oversized incoming row must stay bounded");
    for (size_t v = 0; v < 100; ++v)
      Check(graph.ContainsIncoming(0, v), "hub reciprocal edge lost");
    Check(!graph.ContainsIncoming(0, 100), "hub false reciprocal edge");
  }
  Check(std::filesystem::is_empty(temporary.Path()), "group scratch not released");
  bool rejected = false;
  try {
    BNFParams p;
    p.storage = ShuffleStorage::Auto;
    p.memory_limit_bytes = 600 << 20;
    PlanShuffleStorage({100000000, 9600000000ULL, 4, 96, 192, 5}, p, size_t(4) << 30);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  Check(rejected, "600 MiB cannot admit full SIFT100M metadata");
  {
    ScratchIo io;
    ScratchFile file(temporary.Path() / "short-read", io);
    file.Resize(1);
    uint32_t word;
    rejected = false;
    try {
      file.Read(0, &word, sizeof(word));
    } catch (const std::runtime_error &) {
      rejected = true;
    }
    Check(rejected, "truncated scratch accepted");
  }
  Check(std::filesystem::is_empty(temporary.Path()), "failure scratch not released");
}

static void CheckAuto() {
  ShuffleGraphStats s{1500000, 144000000, 4, 96, 192, 5};
  BNFParams p;
  p.storage = ShuffleStorage::Auto;
  p.threads = 32;
  p.memory_limit_bytes = 600 << 20;
  Check(PlanShuffleStorage(s, p, 160 << 20).selected == ShuffleStorage::External,
        "auto must spill");
  p.memory_limit_bytes = size_t(4) << 30;
  Check(PlanShuffleStorage(s, p, 160 << 20).selected == ShuffleStorage::Memory,
        "auto must retain RAM");
  p.storage = ShuffleStorage::External;
  Check(PlanShuffleStorage(s, p, 160 << 20).selected == ShuffleStorage::External,
        "forced external overridden");
  p.storage = ShuffleStorage::Memory;
  p.memory_limit_bytes = 0;
  Check(PlanShuffleStorage(s, p, 160 << 20).selected == ShuffleStorage::Memory,
        "unlimited memory changed");
  for (auto mode : {ShuffleStorage::Memory, ShuffleStorage::External, ShuffleStorage::Auto}) {
    p.storage = mode;
    p.memory_limit_bytes = 128 << 20;
    bool failed = false;
    try {
      PlanShuffleStorage(s, p, 160 << 20);
    } catch (const std::invalid_argument &) {
      failed = true;
    }
    Check(failed, "impossible budget accepted");
  }
  p.storage = ShuffleStorage::Auto;
  p.memory_limit_bytes = 600 << 20;
  bool failed = false;
  try {
    PlanShuffleStorage(s, p, 700 << 20);
  } catch (const std::invalid_argument &) {
    failed = true;
  }
  Check(failed, "loaded quantization/RSS ignored");
  std::puts("SHUFFLE_AUTO_PASS");
}
static void CheckDirectIo() {
  ScratchDirectory temporary(std::getenv("TEST_TMPDIR") ? std::getenv("TEST_TMPDIR") : "/tmp");
  ScratchIo io;
  ScratchFile file(temporary.Path() / "direct", io);
  constexpr size_t chunk = 5003, chunks = 701;
  std::vector<uint8_t> expected(chunk * chunks);
  for (size_t i = 0; i < expected.size(); ++i)
    expected[i] = (i * 179 + i / 17) % 251;
  file.Resize(expected.size());
  std::vector<std::thread> writers;
  for (size_t t = 0; t < 4; ++t)
    writers.emplace_back([&, t] {
      for (size_t k = t; k < chunks; k += 4)
        file.Write(k * chunk, expected.data() + k * chunk, chunk);
    });
  for (auto &thread : writers)
    thread.join();
  // Unaligned updates must preserve both shared boundary sectors.
  std::fill(expected.begin() + 4093, expected.begin() + 8197, 217);
  file.Write(4093, expected.data() + 4093, 4104);
  file.Seal();
  Check(std::filesystem::file_size(temporary.Path() / "direct") == expected.size(),
        "physical padding persisted");
  std::vector<uint8_t> actual(expected.size());
  file.Read(0, actual.data(), actual.size());
  Check(actual == expected, "concurrent/direct RMW corrupted payload");
  std::vector<uint8_t> gather(64 * 703);
  std::vector<ScratchRead> reads;
  for (size_t i = 0; i < 64; ++i)
    reads.push_back({(i * 7919) % (expected.size() - 703), gather.data() + i * 703, 703});
  file.ReadMany(reads);
  for (size_t i = 0; i < reads.size(); ++i)
    Check(!std::memcmp(reads[i].data, expected.data() + reads[i].offset, reads[i].bytes),
          "queued gather differs");
  Check(io.max_inflight > 1 && io.aio_completed == io.aio_reads + io.aio_writes && io.errors == 0,
        "libaio submission/completion accounting");
  // A failed request must not retain live operations or poison reuse of the TLS engine.
  std::filesystem::resize_file(temporary.Path() / "direct", 0);
  bool failed = false;
  try {
    file.ReadMany(reads);
  } catch (const std::runtime_error &) {
    failed = true;
  }
  Check(failed && io.errors == 1, "short completion was not rejected");
  ScratchFile next(temporary.Path() / "next", io);
  next.Write(0, expected.data(), expected.size());
  next.Seal();
  next.Read(0, actual.data(), actual.size());
  Check(actual == expected, "AIO context failed to recover after drain");
  auto &engine = ThreadScratchAio();
  failed = false;
  try {
    engine.Run({{-1, engine.Buffer(), 4096, 0, 4096, false}}, io);
  } catch (const std::system_error &) {
    failed = true;
  }
  Check(failed, "invalid descriptor accepted");
  next.Read(0, actual.data(), actual.size());
  Check(actual == expected, "AIO failed after submission error");
  std::puts("SHUFFLE_DIRECT_AIO_PASS");
}

int main() {
  try {
    CheckDirectIo();
    CheckAuto();
    CheckPacking();
    CheckScores();
    CheckExternalBounds();
    CheckRounds();
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}
