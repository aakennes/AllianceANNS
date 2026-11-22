#include "index/index_disk.hpp"
#include "io/aio_stream.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace alliance {

// ============================================================================
// WriteBuildVecFile — writes vectors in rank order, dimension-sized pages
// ============================================================================

template <typename att_t, typename vec_t>
void AllianceIndexDisk<att_t, vec_t>::WriteBuildVecFile(const vec_t *base_vecs, size_t n) {
  RequireRawGraph("WriteBuildVecFile");
  vecs_per_page_ = VecPageBytes() / VecBytes();
  if (vecs_per_page_ == 0)
    vecs_per_page_ = 1;
  build_vec_file_path_ = graph_file_->path() + ".buildvecs";
  const auto t0 = std::chrono::high_resolution_clock::now();

  // Output is one dense page per vecs_per_page_ vectors, appended front to back:
  // a pure stream, so AioRegionWriter over a preallocated file. Total size is a
  // page multiple, hence already sector-aligned -- no tail truncate needed.
  const size_t total_pages = (n + vecs_per_page_ - 1) / vecs_per_page_;
  AioPreallocate(build_vec_file_path_, (uint64_t)total_pages * VecPageBytes());
  AioRegionWriter writer(build_vec_file_path_, 0);

  std::vector<char> page_buf(VecPageBytes(), 0);
  size_t slot = 0;
  for (size_t rank = 0; rank < n; ++rank) {
    tableint id = sorted_ids_[rank];
    const vec_t *vec_ptr = base_vecs + static_cast<size_t>(labels_[id]) * vec_d_;
    std::memcpy(page_buf.data() + slot * VecBytes(), vec_ptr, VecBytes());
    slot++;
    if (slot == vecs_per_page_) {
      writer.Append(page_buf.data(), VecPageBytes());
      std::memset(page_buf.data(), 0, VecPageBytes());
      slot = 0;
    }
  }
  if (slot > 0) {
    // Zero the unused tail of the last page: the old code relied on the memset
    // after the previous full flush, and so does this one.
    writer.Append(page_buf.data(), VecPageBytes());
  }
  writer.Close();
  fprintf(stderr, "  WriteBuildVecFile: %zu vectors -> %s (%.1f MB, %.2fs)\n", n,
          build_vec_file_path_.c_str(),
          (double)((n + vecs_per_page_ - 1) / vecs_per_page_) * VecPageBytes() / (1024.0 * 1024.0),
          std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count());
}

// ============================================================================
// WriteBuildVecFileFromGraph — same output as WriteBuildVecFile, sourced from
// the graph file instead of an in-RAM copy of the base set.
// ============================================================================

template <typename att_t, typename vec_t>
void AllianceIndexDisk<att_t, vec_t>::WriteBuildVecFileFromGraph(size_t n) {
  RequireRawGraph("WriteBuildVecFileFromGraph");
  const auto t0 = std::chrono::high_resolution_clock::now();
  vecs_per_page_ = VecPageBytes() / VecBytes();
  if (vecs_per_page_ == 0)
    vecs_per_page_ = 1;
  build_vec_file_path_ = graph_file_->path() + ".buildvecs";

  const size_t total_pages = (n + vecs_per_page_ - 1) / vecs_per_page_;
  AioPreallocate(build_vec_file_path_, (uint64_t)total_pages * VecPageBytes());
  AioRegionWriter writer(build_vec_file_path_, 0);
  // The record reader is O_DIRECT and would not see page-cache-dirty records.
  graph_file_->Sync();

  // Graph slot `rank` already holds the vector of the node at that rank, so a
  // forward scan of the graph file yields exactly the order this file needs —
  // no permutation buffer and no extra pass over the input vectors. Both sides
  // are therefore sequential: one AioSeqStreamReader over the record array and
  // one AioRegionWriter over the output, each keeping its next 4 MB in flight,
  // replacing the ReadNodeRange pread loop and the per-page ::write.
  AioSeqStreamReader reader(graph_file_->path(), graph_file_->node_offset(0),
                            graph_file_->node_offset((tableint)n));
  std::vector<char> rec_buf(node_disk_bytes_);
  std::vector<char> page_buf(VecPageBytes(), 0);
  size_t slot = 0;

  for (size_t rank = 0; rank < n; ++rank) {
    reader.Read(rec_buf.data(), node_disk_bytes_);
    std::memcpy(page_buf.data() + slot * VecBytes(), rec_buf.data() + VecOffsetInRecord(),
                VecBytes());
    if (++slot == vecs_per_page_) {
      writer.Append(page_buf.data(), VecPageBytes());
      std::memset(page_buf.data(), 0, VecPageBytes());
      slot = 0;
    }
  }
  if (slot > 0) {
    writer.Append(page_buf.data(), VecPageBytes());
  }
  writer.Close();
  fprintf(stderr,
          "  WriteBuildVecFileFromGraph: %zu vectors -> %s (%.1f MB, %zu vecs/page, %.2fs)\n", n,
          build_vec_file_path_.c_str(),
          (double)((n + vecs_per_page_ - 1) / vecs_per_page_) * VecPageBytes() / (1024.0 * 1024.0),
          vecs_per_page_,
          std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count());
}

template <typename att_t, typename vec_t>
void AllianceIndexDisk<att_t, vec_t>::GatherVecsByRank(IOHandler &io, const uint32_t *ranks,
                                                       const uint32_t *perm, size_t count,
                                                       vec_t *out, bool out_by_slot) const {
  if (count == 0)
    return;
  const size_t vec_bytes = VecBytes();
  if (vec_bytes > VecPageBytes()) {
    throw std::runtime_error("GatherVecsByRank: VecBytes() exceeds the build vector page");
  }
  // One staging window of kBatch *contiguous* pages, so a run of consecutive
  // pages can be read by a single iocb straight into its final position in the
  // window. On a rank-ascending sweep of a dense file that makes the whole
  // window one contiguous read; the first version issued kBatch separate
  // page reads instead and spent more time in the queue than the encode loop it
  // was feeding. kExtCap bounds the iocbs per submit to the queue depth the
  // callers open -- which is also what a fully scattered order degrades to,
  // i.e. exactly the old behaviour, 64 single-page reads per submit.
  constexpr size_t kBatch = 256;
  constexpr size_t kExtCap = 64;
  // One window per thread, allocated once. Every caller reaches this function
  // from inside a parallel region that is re-entered per cluster, so a fresh
  // aligned_alloc/free per call creates per-thread arena churn per generation --
  // below glibc's 32 MB mmap threshold, hence exactly the ratchet the
  // per-cluster buffers had (it is most of pass A's 190 MB residual trim).
  // The dimension-sized windows are retained, and PlanPhase2 charges
  // it as a persistent Phase 2 term rather than pretending it comes and goes.
  static thread_local std::unique_ptr<char, void (*)(void *)> stage_owner(nullptr, &std::free);
  static thread_local size_t stage_bytes = 0;
  const size_t required_bytes = kBatch * VecPageBytes();
  if (stage_bytes != required_bytes) {
    void *p = aligned_alloc(kMinNodePageBytes, required_bytes);
    if (p == nullptr) {
      throw std::runtime_error("GatherVecsByRank: aligned_alloc failed");
    }
    stage_owner.reset(static_cast<char *>(p));
    stage_bytes = required_bytes;
  }
  char *stage = stage_owner.get();
  std::vector<IORequest> reqs(kExtCap);
  IOCompletionGuard io_completion(io);
  std::vector<IORequest *> ptrs;
  std::vector<size_t> pages(kBatch);

  size_t i = 0;
  while (i < count) {
    // Collect the distinct pages of the next run of requests, stopping when the
    // window is full or one more extent would overrun the queue. Only *adjacent*
    // duplicates collapse, so this coalesces fully when the visit order is
    // rank-ascending (what `perm` is for) and degrades to one page per vector
    // otherwise -- never to a wrong answer, since the scatter below replays the
    // exact same visit order over `pages`.
    size_t npg = 0, next = 0;
    size_t j = i;
    while (j < count) {
      const size_t page = (size_t)ranks[perm != nullptr ? perm[j] : j] / vecs_per_page_;
      if (npg != 0 && page == pages[npg - 1]) {
        ++j;
        continue;
      } // served already
      if (npg == kBatch)
        break;
      const bool contig = (npg != 0 && page == pages[npg - 1] + 1);
      if (!contig) {
        if (next == kExtCap)
          break;
        ++next;
      }
      pages[npg++] = page;
      ++j;
    }

    // One request per maximal run of consecutive pages, landing at the run's
    // own offset in the window so the scatter can stay page-indexed.
    ptrs.clear();
    size_t nreq = 0;
    for (size_t p = 0; p < npg;) {
      size_t q = p + 1;
      while (q < npg && pages[q] == pages[q - 1] + 1)
        ++q;
      reqs[nreq].offset = (uint64_t)pages[p] * VecPageBytes();
      reqs[nreq].len = (uint64_t)(q - p) * VecPageBytes();
      reqs[nreq].buf = stage + p * VecPageBytes();
      reqs[nreq].finished = false;
      reqs[nreq].res = 0;
      ptrs.push_back(&reqs[nreq]);
      ++nreq;
      p = q;
    }
    io.SubmitReadBatch(ptrs);
    while (true) {
      bool all = true;
      for (size_t p = 0; p < nreq; ++p) {
        if (!reqs[p].finished) {
          all = false;
          break;
        }
      }
      if (all)
        break;
      io.PollWaitBatch();
    }
    // A multi-page extent can come back short where a single page could not, and
    // the tail would then be stale window bytes rather than an obvious zero.
    for (size_t p = 0; p < nreq; ++p) {
      if (reqs[p].res != (long)reqs[p].len) {
        throw std::runtime_error("GatherVecsByRank: read of " + std::to_string(reqs[p].len) +
                                 " bytes at " + std::to_string(reqs[p].offset) + " returned " +
                                 std::to_string(reqs[p].res));
      }
    }

    size_t pcur = 0;
    for (size_t k = i; k < j; ++k) {
      const size_t slot = (perm != nullptr) ? (size_t)perm[k] : k;
      const size_t rank = ranks[slot];
      const size_t page = rank / vecs_per_page_;
      while (pages[pcur] != page)
        ++pcur; // forward-only: same order as above
      std::memcpy(out + (out_by_slot ? slot : k) * vec_d_,
                  stage + pcur * VecPageBytes() + (rank % vecs_per_page_) * vec_bytes, vec_bytes);
    }
    i = j;
  }
}

// One definition also preserves one GatherVecsByRank staging buffer per thread.
template void AllianceIndexDisk<int, float>::WriteBuildVecFile(const float *, size_t);
template void AllianceIndexDisk<int, float>::WriteBuildVecFileFromGraph(size_t);
template void AllianceIndexDisk<int, float>::GatherVecsByRank(IOHandler &, const uint32_t *,
                                                              const uint32_t *, size_t, float *,
                                                              bool) const;

} // namespace alliance
