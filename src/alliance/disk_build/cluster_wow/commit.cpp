#include "index/index_disk.hpp"
#include "disk_build/memory/process.hpp"
#include "disk_build/partition/membership.hpp"
#include "io/disk_resource_guard.hpp"
#include "io/aio_stream.hpp"
#include "io/io_handler.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>
#include <random>
#include <unistd.h>

#include "disk_build/cluster_wow/passes.hpp"
#include "disk_build/cluster_wow/scratch.hpp"

namespace alliance {

template <typename att_t, typename vec_t>
void AllianceIndexDisk<att_t, vec_t>::CommitClusterEdgesDisk(
    const disk_build::ClusterSpillPlan &plan) {
  const size_t n = plan.n;
  const layer_t start_layer = plan.start_layer;
  const size_t L = plan.L;
  const size_t K = ClusterScratch()->K;
  const uint32_t *coffs = ClusterScratch()->cluster_rank_offsets_.data();
  const size_t NP = ClusterScratch()->np;
  const uint32_t *rank_cids = ClusterScratch()->rank_cids_.data();
  const size_t rec_head = plan.rec_head;
  const size_t rec_bytes = plan.rec_bytes;
  const std::string &graph_path = plan.graph_path;
  auto spill_path = [&](size_t c) { return plan.scratch.File(c); };

  // ================= pass B: streaming merge + commit replay ================
  // PipeANN merge_shards structure: per-file buffered forward readers walked
  // in ascending-id order, finalize per id. Our records are fixed-size and
  // dense, so each cursor is a byte stream that must yield exactly this rank's
  // record; a mismatch means pass A and B disagree about the layout, which is
  // a hard error, not a data problem.
  auto pass_b_t0 = std::chrono::high_resolution_clock::now();
  std::vector<std::unique_ptr<AioSeqStreamReader>> streams(K);
  for (size_t c = 0; c < K; ++c) {
    const size_t csize = coffs[c + 1] - coffs[c];
    if (csize == 0)
      continue;
    streams[c] =
        std::make_unique<AioSeqStreamReader>(spill_path(c), 0, (uint64_t)csize * rec_bytes);
  }

  IOHandler grw(256);
  grw.OpenRW(graph_path);
  constexpr size_t kBlk = 64;
  size_t read_requests = 0, write_requests = 0;
  char *recs = static_cast<char *>(aligned_alloc(kMinNodePageBytes, kBlk * node_disk_bytes_));
  if (recs == nullptr)
    throw std::runtime_error("pass B: aligned_alloc failed");
  detail::ScopeExit buffer_owner([&]() noexcept { free(recs); });
  std::vector<char> rec_store(kBlk * NP * rec_bytes);
  std::vector<uint32_t> cid_order(NP);
  std::vector<tableint> final_edges;
  final_edges.reserve(M_);
  std::vector<size_t> r_dedup(L, 0), r_edges(L, 0);

  for (size_t base = 0; base < n; base += kBlk) {
    const size_t bn = std::min(kBlk, n - base);

    // Fetch this block's spill records: per rank, one record from each of its
    // np clusters, each read sequential within its stream.
    for (size_t k = 0; k < bn; ++k) {
      const size_t rank = base + k;
      const uint32_t *my_cids = rank_cids + rank * NP;
      for (size_t a = 0; a < NP; ++a) {
        const uint32_t cid = my_cids[a];
        if (streams[cid] == nullptr) {
          throw std::runtime_error("pass B: rank belongs to an empty cluster");
        }
        char *dst = rec_store.data() + (k * NP + a) * rec_bytes;
        streams[cid]->Read(dst, rec_bytes);
        if (*reinterpret_cast<uint32_t *>(dst) != (uint32_t)rank) {
          throw std::runtime_error("pass B: spill stream desync on cluster " + std::to_string(cid));
        }
        if (rank == ClusterScratch()->cluster_ranks_[coffs[cid + 1] - 1]) {
          // The final record is in rec_store; drain/close before unlinking its stream.
          streams[cid].reset();
          plan.scratch.ReleaseSpill(cid);
        }
      }
    }

    // Read the block's node records (they carry the vector and the lower
    // layers, which this phase must preserve byte for byte).
    {
      IORequest request;
      request.offset = graph_file_->node_offset(tableint(base));
      request.len = bn * node_disk_bytes_;
      request.buf = recs;
      IOCompletionGuard completion(grw);
      grw.SubmitRead(request);
      while (!request.finished)
        grw.PollWaitBatch();
      if (request.res != static_cast<long>(request.len))
        throw std::runtime_error("short cluster commit extent read");
      ++read_requests;
    }

    // Commit replay per (rank, layer), then patch the staged record.
    for (size_t k = 0; k < bn; ++k) {
      const size_t rank = base + k;
      const uint32_t *my_cids = rank_cids + rank * NP;
      for (size_t a = 0; a < NP; ++a)
        cid_order[a] = (uint32_t)a;
      for (size_t i = 1; i < NP; ++i) {
        const uint32_t v = cid_order[i];
        size_t j = i;
        while (j > 0 && my_cids[cid_order[j - 1]] > my_cids[v]) {
          cid_order[j] = cid_order[j - 1];
          --j;
        }
        cid_order[j] = v;
      }
      char *rec = recs + k * node_disk_bytes_;

      for (size_t li = 0; li < L; ++li) {
        const layer_t layer = start_layer + (layer_t)li;
        final_edges.clear();
        size_t local_dedup = 0;
        for (size_t oi = 0; oi < NP; ++oi) {
          const size_t a = cid_order[oi];
          const char *rec_c = rec_store.data() + (k * NP + a) * rec_bytes;
          const uint32_t *lens = reinterpret_cast<const uint32_t *>(rec_c + sizeof(uint32_t));
          const tableint *ids = reinterpret_cast<const tableint *>(rec_c + rec_head) + li * M_;
          if (lens[li] > M_)
            throw std::runtime_error("invalid cluster spill degree");
          for (size_t ei = 0; ei < size_t(lens[li]); ++ei) {
            const tableint id = ids[ei];
            if (std::find(final_edges.begin(), final_edges.end(), id) != final_edges.end()) {
              ++local_dedup;
              continue;
            }
            final_edges.push_back(id);
          }
        }
        // Fixed per-node seed preserves merge order independently of batching.
        std::seed_seq seed{20261026u, uint32_t(rank), uint32_t(layer)};
        std::mt19937 rng(seed);
        std::shuffle(final_edges.begin(), final_edges.end(), rng);
        if (final_edges.size() > M_)
          final_edges.resize(M_);
        r_dedup[li] += local_dedup;
        r_edges[li] += final_edges.size();

        // Same byte layout as WriteEdgesWithDist: zero-padded id and dist
        // arrays, count at id[M_].
        const tableint cnt = static_cast<tableint>(final_edges.size());
        tableint *idp = reinterpret_cast<tableint *>(rec + EdgeOffsetInRecord(layer));
        dist_t *dpp = reinterpret_cast<dist_t *>(rec + EdgeOffsetInRecord(layer) + EdgeIdBytes());
        for (size_t i = 0; i < (size_t)M_; ++i) {
          idp[i] = (i < cnt) ? final_edges[i] : 0;
          dpp[i] = 0;
        }
        idp[M_] = cnt;
      }
    }

    // Write the patched block back.
    {
      IORequest request;
      request.offset = graph_file_->node_offset(tableint(base));
      request.len = bn * node_disk_bytes_;
      request.buf = recs;
      IOCompletionGuard completion(grw);
      grw.SubmitWrite(request);
      while (!request.finished)
        grw.PollWaitBatch();
      if (request.res != static_cast<long>(request.len))
        throw std::runtime_error("short cluster commit extent write");
      ++write_requests;
    }

    if ((base / kBlk) % 4096 == 4095) {
      fprintf(stderr, "  [P2B] merged %zu/%zu ranks\n", base + bn, n);
    }
  }
  if (::fsync(grw.fd()) != 0)
    throw std::runtime_error("cluster commit sync failed");
  fprintf(
      stderr,
      "[ClusterCommitIO] node_pages=%zu read_requests=%zu write_requests=%zu bytes_each_direction=%llu extent_nodes=%zu\n",
      n, read_requests, write_requests,
      static_cast<unsigned long long>(uint64_t(n) * node_disk_bytes_), kBlk);
  free(recs);
  recs = nullptr;

  auto pass_b_t1 = std::chrono::high_resolution_clock::now();
  fprintf(stderr, "[Phase2-Cluster-Disk] pass B done: %.2fs\n",
          std::chrono::duration<double>(pass_b_t1 - pass_b_t0).count());
  // Report the merge working set: memberships and double-buffered streams.
  // The graph block was released above; Pass A buffers never outlive Pass A.
  disk_build::CompleteMemoryPhase("Phase2 pass B",
                                  plan.memory.base_bytes + plan.memory.pass_b_bytes,
                                  build_config_.build_mem_gb_);
  for (size_t li = 0; li < L; ++li) {
    fprintf(stderr, "  [Quota-Disk] layer %d: edges=%zu avg_deg=%.2f dedup=%zu\n",
            (int)(start_layer + (layer_t)li), r_edges[li], (double)r_edges[li] / (double)n,
            r_dedup[li]);
  }

  id_only_start_layer_ = std::min(id_only_start_layer_, size_t(start_layer));
  // Spill files are pure intermediates; the merged graph is the artifact.
  streams.clear();
  // The stage owner removes exactly this generation after readers close.
}

template void
AllianceIndexDisk<int, float>::CommitClusterEdgesDisk(const disk_build::ClusterSpillPlan &);

} // namespace alliance
