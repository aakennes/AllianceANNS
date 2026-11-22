#pragma once
#include "shuffle/config.hpp"
#include "shuffle/adjacency/scratch_aio.hpp"
#include "shuffle/memory/external_budget.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace alliance::shuffle_detail {
struct ShuffleGraphStats {
  size_t nodes{}, edges{}, edge_bytes{}, max_degree{}, max_score{}, elements_per_block{};
};
struct ShuffleStoragePlan {
  ShuffleStorage selected;
  size_t resident_before{}, memory_peak{}, external_peak{}, reserve{}, limit{};
  ExternalBudget external_buffers;
};
inline size_t PlannedBytes(long double bytes) {
  if (bytes < 0 || bytes > std::numeric_limits<size_t>::max())
    throw std::overflow_error("shuffle memory estimate overflow");
  return static_cast<size_t>(bytes);
}
inline size_t ShuffleReserve(size_t limit) {
  return std::max<size_t>(64 << 20, limit / 20); // 5%, at least 64 MiB; not an OS cap.
}
// Called after the existing count pass and before E-sized allocations. Current
// RSS already includes loaded quantization, metadata, earlier permutations and
// count-pass offsets/scratch. Only overlapping future lifetimes are added.
inline ShuffleStoragePlan PlanShuffleStorage(const ShuffleGraphStats &s, const BNFParams &p,
                                             size_t resident_before) {
  if (!s.elements_per_block || p.threads < 1 || !p.affinity_batch_size)
    throw std::invalid_argument("invalid shuffle planning geometry");
  ShuffleStoragePlan plan{
      p.storage, resident_before, 0, 0, ShuffleReserve(p.memory_limit_bytes), p.memory_limit_bytes,
      {}};
  const long double n = s.nodes;
  const long double batch = std::min(s.nodes, p.affinity_batch_size);
  const long double scores = 16.L * batch * s.max_degree + 16.L * batch +
                             p.threads * (64.L * s.max_degree + 16.L * (s.max_score + 1.L));
  const long double blocks = s.nodes / s.elements_per_block + (s.nodes % s.elements_per_block != 0);
  const long double common = resident_before + scores + 8.L * blocks + plan.reserve;
  // RAM: node order 8N, old/next maps 16N, output 4N, membership filter 16N.
  plan.memory_peak = PlannedBytes(common + s.edges * (long double)s.edge_bytes + 44.L * n);
  // External preparation and round buffers do not overlap. Count the larger
  // lifetime, retaining a bounded row gather and merge/writer allowance.
  const long double direct_buffers = (p.threads + 1.L) * kScratchAioBufferBytes;
  const long double preparation = resident_before + plan.reserve + direct_buffers + 44.L * n +
                                  plan.external_buffers.sort_bytes + 24.L * (1 << 20);
  const long double rounds = common + direct_buffers + 52.L * n +
                             2.L * plan.external_buffers.batch_bytes + 8.L * (1 << 20);
  plan.external_peak = PlannedBytes(std::max(preparation, rounds));
  const bool external_rows_fit = s.max_degree <= plan.external_buffers.batch_bytes / 5;
  if (p.storage == ShuffleStorage::Auto) {
    if (!plan.limit)
      throw std::invalid_argument("auto shuffle requires a memory budget");
    plan.selected =
        plan.memory_peak <= plan.limit ? ShuffleStorage::Memory : ShuffleStorage::External;
  }
  if (plan.selected == ShuffleStorage::External &&
      (!plan.limit || plan.external_peak > plan.limit || !external_rows_fit))
    throw std::invalid_argument("external shuffle working set exceeds memory budget");
  if (plan.selected == ShuffleStorage::Memory && plan.limit && plan.memory_peak > plan.limit)
    throw std::invalid_argument("memory shuffle working set exceeds memory budget");
  return plan;
}
} // namespace alliance::shuffle_detail
