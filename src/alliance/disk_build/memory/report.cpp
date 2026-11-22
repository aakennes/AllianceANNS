#include "disk_build/memory/index_adapter.hpp"
#include "disk_build/memory/process.hpp"
#include <cstdio>

namespace alliance {

// Compatibility entry points; stage controllers use the explicit memory boundary.
template <typename att_t, typename vec_t>
void AllianceIndexDisk<att_t, vec_t>::ProcRss(size_t *cur_bytes, size_t *peak_bytes) {
  const auto sample = disk_build::CaptureMemorySnapshot();
  if (cur_bytes)
    *cur_bytes = sample.current_bytes;
  if (peak_bytes)
    *peak_bytes = sample.peak_bytes;
}

template <typename att_t, typename vec_t> void AllianceIndexDisk<att_t, vec_t>::ResetProcRssPeak() {
  disk_build::ResetPeakWindow();
}

template <typename att_t, typename vec_t>
void AllianceIndexDisk<att_t, vec_t>::ReportPhaseRss(const char *stage, size_t model_bytes) const {
  disk_build::CompleteMemoryPhase(stage, model_bytes, build_config_.build_mem_gb_);
}

template <typename att_t, typename vec_t>
void disk_build::MemoryPlanner<att_t, vec_t>::ReportBuildMemoryBudget(size_t n) const {
  constexpr double kGB = 1024.0 * 1024.0 * 1024.0;
  const size_t bookkeeping = EstimateBaseBuildBytes(n);
  const Phase15Plan p = PlanPhase15(n);
  fprintf(stderr, "  Build RAM (base bookkeeping estimate): %.2f GB\n", bookkeeping / kGB);
  fprintf(stderr,
          "    Phase 2 uses the measured process floor when available, including search\n"
          "    quantization and retained allocations. No build-side RaBitQ codes are generated.\n");
  const Phase2Plan p2 = PlanPhase2(n);
  fprintf(stderr, "  Phase 2 (ClusterWoW/Vamana) working set:\n");
  fprintf(stderr, "    memberships=%zu cluster_layers=%zu, vector=%zu B, ID spill record=%zu B\n",
          p2.np, p2.cluster_layers, p2.vec_bytes, p2.rec_bytes);
  fprintf(stderr,
          "    scratch: pass A %.2f MB/thread x %zu, retained gather window %.2f MB/thread x %zu\n",
          p2.thread_scratch_bytes / 1048576.0, p2.threads, 256.0 * VecPageBytes() / 1048576.0,
          p2.threads);
  const bool budgeted = build_mem_gb_ > 0.0;
  fprintf(stderr, "    %-6s %-12s %-12s %s\n", "K", "residency", "total", "fits");
  for (size_t i = 0; i < Phase2Plan::kTableSize; ++i) {
    const size_t K = p2.table_K[i], res = p2.table_residency[i];
    const bool fits = !budgeted || (!p2.base_over_budget && res <= p2.avail_bytes);
    fprintf(stderr, "    %-6zu %-12.2f %-12.2f %s\n", K, res / kGB, (p2.base_bytes + res) / kGB,
            fits ? "yes" : "NO");
  }
  fprintf(
      stderr,
      "    WoW Vamana partition: nearest-two L2 memberships, floor(2*M/3) cluster degree, random union merge.\n"
      "    Auto-K table is provisional: random-sample estimation and actual maximum cluster size are checked at build time.\n");
  fprintf(stderr, "    spill (disk, not RAM): %.2f GB pruned-edge spill\n", p2.spill_bytes / kGB);
  const char *k_src = p2.requested_K == 0 ? "provisional memory model"
                                          : (p2.k_from_file ? "pinned by --cluster_centroids"
                                                            : "pinned by --cluster_K");
  fprintf(stderr, "    effective_K=%zu (%s) -> residency %.2f GB, total %.2f GB\n", p2.effective_K,
          k_src, p2.residency_bytes / kGB, (p2.base_bytes + p2.residency_bytes) / kGB);
  if (p2.effective_K > 0) {
    fprintf(stderr,
            "      persistent state %.2f GB; prep %.2f, passA %.2f, passB %.2f GB above base;\n"
            "      max_csize %zu slots (partition skew %.2fx average, no capacity reassignment)\n",
            p2.state_bytes / kGB, p2.prep_bytes / kGB, p2.pass_a_bytes / kGB, p2.pass_b_bytes / kGB,
            p2.cap_slots, double(cluster_balance_ >= 1.0f ? cluster_balance_ : 1.8f));
  }
  if (budgeted && !p2.base_over_budget) {
    if (p2.derived_K > 0) {
      fprintf(stderr, "    derived_K=%zu (smallest fitting provisional K)\n", p2.derived_K);
    } else if (p2.fixed_bytes >= p2.avail_bytes) {
      fprintf(
          stderr,
          "    !! no K fits AVAIL: K-independent scratch (%.2f GB, %zu threads) alone exceeds it\n",
          p2.fixed_bytes / kGB, p2.threads);
    } else {
      fprintf(stderr, "    !! no K fits AVAIL (%.2f GB); raise --build_mem_gb\n",
              p2.avail_bytes / kGB);
    }
    if (p2.user_over_budget) {
      fprintf(
          stderr,
          "    !! K=%zu (%s) needs %.2f GB but AVAIL is %.2f GB; the build will abort at Phase 2.\n"
          "       Use automatic Vamana partitioning or raise --build_mem_gb to >= %.2f GB.\n",
          p2.requested_K, k_src, p2.residency_bytes / kGB, p2.avail_bytes / kGB,
          (p2.base_bytes + p2.residency_bytes) / kGB);
    }
  }

  if (build_mem_gb_ <= 0.0) {
    fprintf(stderr, "    budget: unlimited (--build_mem_gb not set)\n");
    return;
  }

  fprintf(stderr, "  Resident builder: left_stream; left 2H/right 0; batch=%zu; M-slot L0 pool\n",
          p.batch_size);
  fprintf(stderr, "  Resident LeftStream budget reverse-solve:\n");
  if (p.base_over_budget) {
    fprintf(stderr, "    !! BASE %.2f GB already exceeds budget %.2f GB\n", p.base_bytes / kGB,
            build_mem_gb_);
    return;
  }
  fprintf(
      stderr,
      "    scratch peak = %d workers + visited/ready state + L0 totals + batch snapshots + extent I/O = %.2f GB\n",
      merge_build_threads_, p.thread_scratch_bytes / kGB);
  fprintf(
      stderr,
      "    AVAIL = budget - base - fixed scratch (before resident records/L0 totals) = %.2f GB\n",
      p.avail_bytes / kGB);
  fprintf(stderr, "    node_disk_bytes=%zu, batch=%zu\n", p.node_bytes, p.batch_size);
  fprintf(
      stderr,
      "    resident capacity=min(2*floor(window/2)+batch,n); scratch includes L0 totals and batch snapshots\n");
  if (p.avail_bytes == 0) {
    fprintf(stderr,
            "    !! base + per-thread scratch alone exhausts the budget: no resident buffer\n"
            "       fits, so no resident layer can be built. Raise --build_mem_gb to >= %.2f GB\n"
            "       or lower --threads (scratch is linear in it).\n",
            p.min_budget_gb_l1);
  }

  fprintf(stderr, "    max_window_allowed (derived wow_threshold) = %zu nodes\n",
          p.derived_threshold);
  if (wow_threshold_ > 0) {
    fprintf(stderr, "    user --wow_thresh=%zu -> %s\n", wow_threshold_,
            p.user_over_budget ? "OVER BUDGET (build will abort)" : "within budget");
  }
  if (p.wow_max > 0) {
    fprintf(stderr, "    => wow_max=%d (window=%zu), residency %.2f GB, total %.2f GB\n", p.wow_max,
            window_size_[p.wow_max], p.residency_bytes / kGB,
            (p.base_bytes + p.residency_bytes + p.thread_scratch_bytes) / kGB);
  } else {
    fprintf(stderr, "    => wow_max=0: only L0 selected; layer 1 requires >= %.2f GB\n",
            p.min_budget_gb_l1);
  }
}

template <typename att_t, typename vec_t>
void AllianceIndexDisk<att_t, vec_t>::ReportBuildMemoryBudget(size_t n) const {
  disk_build::MemoryPlanner<att_t, vec_t>::FromIndex(*this).ReportBuildMemoryBudget(n);
}
template void disk_build::MemoryPlanner<int, float>::ReportBuildMemoryBudget(size_t) const;

template void AllianceIndexDisk<int, float>::ProcRss(size_t *, size_t *);
template void AllianceIndexDisk<int, float>::ResetProcRssPeak();
template void AllianceIndexDisk<int, float>::ReportPhaseRss(const char *, size_t) const;
template void AllianceIndexDisk<int, float>::ReportBuildMemoryBudget(size_t) const;

} // namespace alliance
