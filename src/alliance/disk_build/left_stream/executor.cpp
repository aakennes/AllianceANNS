#include "index/index_disk.hpp"
#include "disk_build/stages/internal.hpp"
#include "io/disk_resource_guard.hpp"
#include "disk_build/left_stream/graph.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

namespace alliance::disk_build {
template <typename att_t, typename vec_t>
void ExecuteLeftStreamRecords(AllianceIndexDisk<att_t, vec_t> &index, layer_t maximum, size_t n) {
  if (!n)
    return;
  // Keep the previous 2H ranks plus the current batch, capped at the whole input.
  // The highest resident layer sets H; lower layers share these node records.
  const size_t h = index.window_size_[maximum] / 2;
  const size_t batch = index.build_config_.left_stream_batch_;
  const size_t capacity = std::min(n, 2 * h + batch);
  char *buffer =
      static_cast<char *>(aligned_alloc(kMinNodePageBytes, capacity * index.node_disk_bytes_));
  if (!buffer)
    throw std::bad_alloc();
  detail::ScopeExit owner([&]() noexcept { free(buffer); });
  std::vector<uint32_t> bottom_total(capacity, 0);
  size_t left = 0, right = 0, read_nodes = 0, written_nodes = 0;
  // Adapt the bounded record buffer to LeftStreamGraph. Rank checks must precede
  // modulo addressing so an evicted rank cannot alias a newer resident record.
  struct Storage {
    AllianceIndexDisk<att_t, vec_t> *index;
    char *buffer;
    size_t capacity;
    size_t *left;
    size_t *right;
    std::vector<uint32_t> *total;
    size_t Rank(tableint id) const {
      return index->position_by_id_[id];
    }
    tableint Id(size_t rank) const {
      return index->sorted_ids_[rank];
    }
    char *Record(size_t rank) const {
      if (rank < *left || rank >= *right)
        throw std::runtime_error("left_stream access outside resident [p-2H,q)");
      return buffer + (rank % capacity) * index->node_disk_bytes_;
    }
    LeftStreamEdges View(tableint id, size_t layer) const {
      char *row = Record(Rank(id)) + index->EdgeOffsetInRecord(layer);
      auto *ids = reinterpret_cast<tableint *>(row);
      return {ids, reinterpret_cast<dist_t *>(row + index->EdgeIdBytes()), ids + index->M_};
    }
    uint32_t &BottomTotal(tableint id) {
      return (*total)[Rank(id) % capacity];
    }
    std::mutex &Mutex(tableint id) {
      return index->EdgeLockFor(id);
    }
    dist_t Distance(tableint a, tableint b) const {
      // Edge construction uses exact distances from the original vectors.
      // The search-side RaBitQ codes populated earlier are not consulted here.
      auto *va = reinterpret_cast<const vec_t *>(Record(Rank(a)) + index->VecOffsetInRecord());
      auto *vb = reinterpret_cast<const vec_t *>(Record(Rank(b)) + index->VecOffsetInRecord());
      return index->fstdistfunc_(va, vb, index->dist_func_param_);
    }
  } storage{&index, buffer, capacity, &left, &right, &bottom_total};
  LeftStreamGraph<Storage> graph(storage, n, index.M_, maximum, index.window_size_,
                                 std::max(index.build_config_.wow_efc_, index.M_),
                                 index.build_config_.merge_build_threads_, index.alpha_);
  // Flush buffered population writes before reading the same records with O_DIRECT.
  IOHandler io(64);
  index.graph_file_->Sync();
  io.OpenRW(index.graph_file_->path());
  // Contiguous extents go directly into the ring. A transfer ends at a ring
  // wrap; no staging vector or per-resident-node IORequest array is needed.
  auto transfer = [&](size_t begin, size_t end, bool write) {
    constexpr size_t inflight = 32, extent_nodes = 256;
    std::vector<IORequest> requests(inflight);
    while (begin < end) {
      size_t used = 0;
      // Drain submitted IO on exceptions before request storage or buffers die.
      IOCompletionGuard completion(io);
      while (begin < end && used < inflight) {
        const size_t slot = begin % capacity;
        const size_t count = std::min({end - begin, capacity - slot, extent_nodes});
        auto &request = requests[used++];
        request = IORequest{};
        request.offset = index.graph_file_->node_offset(begin);
        request.len = count * index.node_disk_bytes_;
        request.buf = buffer + slot * index.node_disk_bytes_;
        if (write)
          io.SubmitWrite(request);
        else
          io.SubmitRead(request);
        begin += count;
        if (write)
          written_nodes += count;
        else
          read_nodes += count;
      }
      for (size_t k = 0; k < used; ++k)
        while (!requests[k].finished)
          io.PollWaitBatch();
      for (size_t k = 0; k < used; ++k)
        if (requests[k].res != static_cast<long>(requests[k].len))
          throw std::runtime_error("left_stream short or failed sequential IO");
    }
  };
  // Finalize ranks before eviction, preserve the disk layout, and clear obsolete
  // distance slots. Exact edge distances are needed only while records are resident.
  auto close = [&](size_t begin, size_t end) {
    if (begin == end)
      return;
    graph.Finalize(begin, end);
    for (size_t rank = begin; rank < end; ++rank) {
      char *row = storage.Record(rank);
      for (layer_t l = 0; l <= maximum; ++l)
        std::memset(row + index.EdgeOffsetInRecord(l) + index.EdgeIdBytes(), 0,
                    index.EdgeDistBytes());
    }
    transfer(begin, end, true);
  };
  const auto start = std::chrono::steady_clock::now();
  size_t last_percent = SIZE_MAX;
  for (size_t begin = 0; begin < n; begin += batch) {
    const size_t end = std::min(begin + batch, n);
    const size_t needed_left = begin > 2 * h ? begin - 2 * h : 0;
    // Write expired ranks before their modulo slots are reused by the next read.
    close(left, needed_left);
    left = needed_left;
    const size_t old_right = right;
    transfer(right, end, false);
    right = end;
    // Freshly loaded records keep their vectors but start with empty resident edges.
    for (size_t rank = old_right; rank < right; ++rank) {
      bottom_total[rank % capacity] = 0;
      char *row = storage.Record(rank);
      for (layer_t l = 0; l <= maximum; ++l)
        std::memset(row + index.EdgeOffsetInRecord(l), 0, index.EdgeBytes());
    }
    if (right - left > capacity)
      throw std::logic_error("left_stream resident interval exceeds capacity");
    graph.Batch(begin, end);
    const size_t percent = 100 * end / n;
    if (percent != last_percent) {
      last_percent = percent;
      const double seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      fprintf(stderr, "[LeftStream] %zu/%zu (%zu%%) elapsed=%.2fs resident=[%zu,%zu)\n", end, n,
              percent, seconds, left, right);
    }
  }
  // Finish pending layer activation/inheritance while the final window is still
  // resident, then write the tail and establish durability before marking success.
  graph.FinishLayers();
  close(left, right);
  index.graph_file_->Sync();
  if (read_nodes != n || written_nodes != n || graph.Finalized() != n)
    throw std::runtime_error("left_stream violated one-read/one-write invariant");
  fprintf(stderr,
          "[LeftStream] records_read=%zu records_written=%zu finalized=%zu capacity=%zu "
          "bottom_fill=%zu upper_fill=%zu distances_persisted=0\n",
          read_nodes, written_nodes, graph.Finalized(), capacity, graph.BottomFill(),
          graph.UpperFill());
}
template void ExecuteLeftStreamRecords(AllianceIndexDisk<int, float> &, layer_t, size_t);
} // namespace alliance::disk_build
