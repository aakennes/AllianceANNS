#include "disk_build/memory/budget.hpp"
#include "storage/order_table.hpp"
#include "disk_build/cluster_wow/input.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>

namespace alliance::disk_build {

// Pure planning over the index's existing configuration and recorded RSS floors.
// No process sampling, allocator trimming, file creation or build-stage execution.
// ============================================================================
// Build memory budget
// ============================================================================

template <typename att_t, typename vec_t>
auto MemoryPlanner<att_t, vec_t>::EstimateBaseBuildBytes(size_t n) const -> size_t {
  size_t bytes = 0;
  bytes += n * sizeof(label_t);                             // labels_
  bytes += n * sizeof(att_t);                               // atts_
  bytes += n * sizeof(tableint);                            // sorted_ids_
  bytes += n * sizeof(tableint);                            // position_by_id_
  bytes += n * SortedVectorOrderTable<att_t>::EntryBytes(); // order_table_
  return bytes;
}

template <typename att_t, typename vec_t>
auto MemoryPlanner<att_t, vec_t>::PlanPhase15(size_t n) const -> Phase15Plan {
  constexpr double kGB = 1024.0 * 1024.0 * 1024.0;
  Phase15Plan p{};
  p.base_bytes = EstimateBaseBuildBytes(n);
  p.node_bytes = node_disk_bytes_;
  p.batch_size = left_stream_batch_;

  // The LeftStream executor retains left 2H plus the current batch, capped at n.
  auto capacity_at = [&](layer_t l) -> size_t {
    return std::min(n, 2 * (window_size_[l] / 2) + p.batch_size);
  };
  auto residency_at = [&](layer_t l) -> size_t { return capacity_at(l) * p.node_bytes; };
  // L0 totals live beside the node records. Batched extent I/O is fixed scratch.
  const size_t side_per_record = sizeof(uint32_t);
  const size_t p15_threads = static_cast<size_t>(std::max(merge_build_threads_, 1));
  size_t p15_fixed_scratch = p15_threads * n * sizeof(vl_type) + n;
  p15_fixed_scratch += p.batch_size * (M_ * sizeof(dist_id_pair) + sizeof(uint32_t)) +
                       p15_threads * (2 * (window_size_[0] / 2) * sizeof(dist_id_pair) +
                                      64 * std::max(wow_efc_, M_) + 128 * M_ + 4096) +
                       65536; // bounded extent requests/IO bookkeeping

  p.min_budget_gb_l1 = (p.base_bytes + p15_fixed_scratch +
                        (wp_ >= 1 ? capacity_at(1) * (p.node_bytes + side_per_record) : 0)) /
                       kGB;

  // Reverse-solve 2*floor(window/2) + batch, capped at n. Charge the
  // per-record side storage before deriving the number of records that fit.
  if (build_mem_gb_ <= 0.0) {
    p.avail_bytes = 0;
    p.derived_threshold = SIZE_MAX;
  } else {
    const size_t budget_bytes = static_cast<size_t>(build_mem_gb_ * kGB);
    if (budget_bytes <= p.base_bytes) {
      p.base_over_budget = true;
      p.derived_threshold = 0;
    } else if (budget_bytes <= p.base_bytes + p15_fixed_scratch) {
      p.avail_bytes = 0;
      p.derived_threshold = 0;
    } else {
      p.avail_bytes = budget_bytes - p.base_bytes - p15_fixed_scratch;
      const size_t fit = p.avail_bytes / (p.node_bytes + side_per_record);
      if (fit >= n)
        p.derived_threshold = SIZE_MAX;
      else
        p.derived_threshold = fit >= p.batch_size ? 2 * ((fit - p.batch_size) / 2) + 1 : 0;
    }
  }

  // Effective threshold: honour an explicit wow_threshold_, else use derived.
  if (build_mem_gb_ <= 0.0) {
    p.effective_threshold = wow_threshold_; // unlimited: trust the user (0 = L0 only)
  } else if (wow_threshold_ > 0) {
    p.user_over_budget = (wow_threshold_ > p.derived_threshold);
    p.effective_threshold = wow_threshold_; // driver aborts if over budget
  } else {
    p.effective_threshold = p.derived_threshold;
  }

  // Largest layer admitted by the threshold. If the complete n-record
  // resident buffer and its side storage fit, even an oversized window is
  // affordable. Layer 0 is always built by LeftStream; the comparison uses the
  // persisted ladder because a top anchor derives its bottom window from n.
  p.wow_max = 0;
  if (p.effective_threshold > window_size_[0]) {
    for (layer_t l = 1; l <= static_cast<layer_t>(wp_); ++l) {
      if (window_size_[l] <= p.effective_threshold)
        p.wow_max = l;
      else
        break;
    }
  }
  if (left_stream_locked_max_ != SIZE_MAX) {
    if (left_stream_locked_max_ > wp_)
      throw std::runtime_error("invalid locked resident boundary");
    p.wow_max = static_cast<layer_t>(left_stream_locked_max_);
  }
  p.residency_bytes = residency_at(p.wow_max);
  p.thread_scratch_bytes = p15_fixed_scratch + capacity_at(p.wow_max) * side_per_record;

  return p;
}

template <typename att_t, typename vec_t>
auto MemoryPlanner<att_t, vec_t>::PlanPhase2(size_t n) const -> Phase2Plan {
  return PlanUpperMemory(n, true);
}

template <typename att_t, typename vec_t>
auto MemoryPlanner<att_t, vec_t>::PlanPartition(size_t n, size_t K, float balance) const
    -> Phase2Plan {
  if (K < 2 || K > n)
    throw std::invalid_argument("partition memory plan requires 2 <= K <= n");
  auto partition = *this;
  partition.cluster_K_file_ = K;
  partition.cluster_balance_ = balance;
  return partition.PlanUpperMemory(n, false);
}

template <typename att_t, typename vec_t>
auto MemoryPlanner<att_t, vec_t>::PlanUpperMemory(size_t n, bool include_report) const
    -> Phase2Plan {
  constexpr double kGB = 1024.0 * 1024.0 * 1024.0;
  constexpr size_t kAioStream = 4ull << 20;
  constexpr size_t kPassBBlk = 64;
  // Vamana never reassigns members to enforce a cap. Partitioning supplies the
  // measured skew; before it runs the memory report is only provisional.
  const double skew = cluster_balance_ >= 1.0f ? double(cluster_balance_) : 1.8;
  Phase2Plan p{};
  // Retain the measured whole-process floor, including search quantization and
  // allocator residue. Removing build-side codes does not remove search codes.
  p.base_bytes = measured_base_bytes_ > 0 ? measured_base_bytes_ : EstimateBaseBuildBytes(n);
  p.np = 2;
  p.threads = std::max(merge_build_threads_, 1);
  p.vec_bytes = VecBytes();
  const size_t slots_total = n * p.np;
  const size_t vpp = std::max(size_t(1), VecPageBytes() / std::max(p.vec_bytes, size_t(1)));
  const Phase15Plan p15 = PlanPhase15(n);
  p.cluster_layers = wp_ > size_t(p15.wow_max) ? wp_ - size_t(p15.wow_max) : 0;
  const size_t L = std::max(p.cluster_layers, size_t(1));
  const size_t Mv = size_t(M_);
  p.rec_bytes = sizeof(uint32_t) * (1 + L) + L * Mv * sizeof(tableint);
  p.thread_scratch_bytes = (std::max(cluster_wow_efc_, Mv) + 4 * Mv + 32) * 64;
  // Gather staging is thread_local and remains live after partition assignment.
  const size_t gather_stage = p.threads * 256 * VecPageBytes();
  auto state_at = [&](size_t K) -> size_t {
    // rank_cids + cluster_ranks + input memberships retained during preparation.
    return slots_total * 3 * sizeof(uint32_t) + (K + 1) * sizeof(uint32_t) + gather_stage;
  };
  auto cap_slots_at = [&](size_t K) -> size_t {
    if (K <= 1)
      return slots_total;
    const size_t cap = size_t(skew * double(slots_total) / double(K)) + 1;
    return std::min(slots_total, std::max(cap, size_t(1)));
  };
  auto prep_at = [&](size_t K) -> size_t {
    const size_t cap = cap_slots_at(K), state = state_at(K);
    const size_t sample = std::min(n, std::max(K, size_t(std::ceil(n * .011)) + 256));
    const size_t assign_chunk = 256 * vpp;
    // Sample, Elkan workspace and nearest-two assignment are FP32 operations.
    // Count both training/estimation samples conservatively; no padded codes.
    const size_t synth = slots_total * sizeof(uint32_t) + gather_stage +
                         std::max(sample * (2 * p.vec_bytes + K * sizeof(float) + 64) +
                                      (2 * K * K + 3 * K * vec_d_) * sizeof(float),
                                  p.threads * assign_chunk * (p.vec_bytes + sizeof(uint32_t)));
    const size_t distribution =
        state + measured_prep_residue_bytes_ +
        ClusterInputDistributionScratch(K, cap, ClusterInputRecordBytes(p.vec_bytes, Mv),
                                        node_disk_bytes_);
    return std::max({synth, state + K * sizeof(uint32_t), distribution});
  };
  auto pass_a_at = [&](size_t K) -> size_t {
    const size_t cap = cap_slots_at(K);
    const size_t visited = ((cap + 63) / 64) * 12;
    return state_at(K) + measured_prep_residue_bytes_ + n * sizeof(uint32_t) +
           cap * (p.vec_bytes + (L + 1) * (Mv * sizeof(dist_id_pair) + sizeof(uint32_t))) +
           p.threads * (p.thread_scratch_bytes + visited) + 4096 * 64 + 64 * node_disk_bytes_ +
           2 * kAioStream + std::max(p.rec_bytes, ClusterInputRecordBytes(p.vec_bytes, Mv));
  };
  auto pass_b_at = [&](size_t K) -> size_t {
    return state_at(K) + measured_prep_residue_bytes_ + K * 2 * kAioStream +
           kPassBBlk * (node_disk_bytes_ + p.np * p.rec_bytes);
  };
  auto residency_at = [&](size_t K) -> size_t {
    return K == 0 ? 0 : std::max({prep_at(K), pass_a_at(K), pass_b_at(K)});
  };
  p.fixed_bytes = state_at(1) + n * sizeof(uint32_t) + p.threads * p.thread_scratch_bytes;
  p.spill_bytes = slots_total * p.rec_bytes;
  p.k_from_file = cluster_K_file_ > 0;
  p.requested_K = p.k_from_file ? cluster_K_file_ : cluster_K_;
  auto finalise = [&](size_t K) {
    p.state_bytes = K > 0 ? state_at(K) : 0;
    p.cap_slots = K > 0 ? cap_slots_at(K) : 0;
    p.prep_bytes = K > 0 ? prep_at(K) : 0;
    p.pass_a_bytes = K > 0 ? pass_a_at(K) : 0;
    p.pass_b_bytes = K > 0 ? pass_b_at(K) : 0;
    p.residency_bytes = std::max({p.prep_bytes, p.pass_a_bytes, p.pass_b_bytes});
  };
  auto fill_table = [&]() {
    if (!include_report)
      return;
    const size_t ladder[Phase2Plan::kTableSize] = {4, 8, 16, 32, 64};
    for (size_t i = 0; i < Phase2Plan::kTableSize; ++i) {
      p.table_K[i] = ladder[i];
      p.table_residency[i] = residency_at(ladder[i]);
    }
  };
  if (build_mem_gb_ <= 0.0) {
    p.effective_K = p.requested_K;
    finalise(p.effective_K);
    fill_table();
    return p;
  }
  const size_t budget_bytes = size_t(build_mem_gb_ * kGB);
  if (budget_bytes <= p.base_bytes) {
    p.base_over_budget = true;
    p.effective_K = p.requested_K;
    finalise(p.effective_K);
    fill_table();
    p.user_over_budget = true;
    return p;
  }
  p.avail_bytes = budget_bytes - p.base_bytes;
  if (include_report && p.avail_bytes > p.fixed_bytes) {
    // Every cluster needs two merge-stream buffers, a strict lower bound on
    // residency. This bounds the complete search without the old IVF sub-K
    // heuristic, which could hide a feasible Vamana K. Memberships require K>=2.
    const size_t k_hi = std::min(n, p.avail_bytes / (2 * kAioStream));
    for (size_t K = 2; K <= k_hi; ++K) {
      if (residency_at(K) <= p.avail_bytes) {
        p.derived_K = K;
        break;
      }
    }
  }
  p.effective_K = p.requested_K > 0 ? p.requested_K : p.derived_K;
  finalise(p.effective_K);
  fill_table();
  p.user_over_budget = p.requested_K > 0 && p.residency_bytes > p.avail_bytes;
  return p;
}

// Keep the existing supported specialization; do not instantiate unrelated methods.
template auto MemoryPlanner<int, float>::EstimateBaseBuildBytes(size_t) const -> size_t;
template auto MemoryPlanner<int, float>::PlanPhase15(size_t) const
    -> MemoryPlanner<int, float>::Phase15Plan;
template auto MemoryPlanner<int, float>::PlanPhase2(size_t) const
    -> MemoryPlanner<int, float>::Phase2Plan;
template auto MemoryPlanner<int, float>::PlanPartition(size_t, size_t, float) const
    -> MemoryPlanner<int, float>::Phase2Plan;

} // namespace alliance::disk_build
