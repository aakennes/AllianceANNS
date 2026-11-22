#pragma once
#include "io/aio_stream.hpp"
#include "io/io_handler.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <unistd.h>

namespace alliance::disk_build {
inline size_t ClusterInputRecordBytes(size_t vector_bytes, size_t m) {
  return 2 * sizeof(uint32_t) + vector_bytes + m * sizeof(uint32_t);
}
inline size_t ClusterInputChunkBytes(size_t k, size_t cap, size_t record) {
  constexpr size_t budget = 32ull << 20;
  if (!k || k > budget / (2 * kAioAlign))
    throw std::runtime_error("too many clusters for bounded input buffers");
  const size_t file_cap = std::max(size_t(kAioAlign), size_t(AioPadUp(uint64_t(cap) * record)));
  return std::min({size_t(1ull << 20), file_cap, (budget / (2 * k) / kAioAlign) * kAioAlign});
}
inline size_t ClusterInputDistributionScratch(size_t k, size_t cap, size_t record,
                                              size_t node_page) {
  return 2 * k * ClusterInputChunkBytes(k, cap, record) + 2 * kAioStreamBytes + node_page + record +
         k * (2 * sizeof(IORequest) + 128) + (64ull << 10);
}

// One temporary file, with a contiguous, sector-aligned region per cluster.
// A shared IO queue and <=32 MiB of paired buffers replace K independent writers.
// Input records: [global rank u32][seed degree u32][vector][M global IDs].
class SequentialClusterInputs {
public:
  SequentialClusterInputs(std::string owned_path, size_t n, size_t vector_bytes, size_t m,
                          const uint32_t *offsets, size_t k)
      : path_(std::move(owned_path)), n_(n), vb_(vector_bytes), m_(m),
        record_(ClusterInputRecordBytes(vector_bytes, m)), counts_(k), starts_(k), read_(k, 0) {
    if (!n || !k || n > UINT32_MAX)
      throw std::runtime_error("invalid cluster input geometry");
    size_t cap = 0;
    for (size_t c = 0; c < k; ++c) {
      if (offsets[c + 1] < offsets[c])
        throw std::runtime_error("invalid cluster offsets");
      counts_[c] = offsets[c + 1] - offsets[c];
      cap = std::max(cap, counts_[c]);
      starts_[c] = file_bytes_;
      const uint64_t length = AioPadUp(uint64_t(counts_[c]) * record_);
      if (length > UINT64_MAX - file_bytes_)
        throw std::overflow_error("cluster input file too large");
      file_bytes_ += length;
    }
    members_ = offsets[k];
    chunk_ = ClusterInputChunkBytes(k, cap, record_);
  }
  // The stage owns this predeclared scratch file and removes it after all readers close.
  ~SequentialClusterInputs() = default;
  SequentialClusterInputs(const SequentialClusterInputs &) = delete;
  SequentialClusterInputs &operator=(const SequentialClusterInputs &) = delete;
  size_t RecordBytes() const {
    return record_;
  }
  size_t InputBytes() const {
    return size_t(members_) * record_;
  }
  uint64_t RegionBegin(size_t c) const {
    return starts_.at(c);
  }
  uint64_t RegionEnd(size_t c) const {
    return starts_.at(c) + uint64_t(counts_.at(c)) * record_;
  }
  const std::string &Path() const {
    return path_;
  }
  void Distribute(const std::string &graph, uint64_t begin, size_t page, size_t vector_offset,
                  size_t seed_offset, const uint32_t *cids, size_t np) {
    if (owned_ || !np || uint64_t(n_) * np != members_ || vector_offset + vb_ > page ||
        seed_offset + (m_ + 1) * 4 > page)
      throw std::runtime_error("invalid cluster input distribution");
    const size_t k = counts_.size();
    owned_ = true;
    AioPreallocate(path_, file_bytes_);
    std::unique_ptr<char, decltype(&std::free)> buffers(
        static_cast<char *>(aligned_alloc(kAioAlign, 2 * k * chunk_)), &std::free);
    if (!buffers)
      throw std::bad_alloc();
    std::vector<IORequest> requests(2 * k);
    std::vector<uint8_t> pending(2 * k, 0), active(k, 0);
    std::vector<size_t> positions(k, 0), records(k, 0);
    std::vector<uint64_t> written(k, 0);
    IOHandler io(64);
    io.OpenRW(path_);
    IOCompletionGuard drain(io);
    size_t inflight = 0;
    auto ensure = [&](size_t slot) {
      if (!pending[slot])
        return;
      while (!requests[slot].finished)
        io.PollWaitBatch();
      if (requests[slot].res != static_cast<long>(requests[slot].len))
        throw std::runtime_error("short cluster input write");
      pending[slot] = 0;
      --inflight;
    };
    auto flush = [&](size_t c, size_t length) {
      if (!length)
        return;
      if (inflight == 64)
        for (size_t slot = 0; slot < 2 * k; ++slot)
          ensure(slot);
      const size_t slot = 2 * c + active[c];
      ensure(slot);
      const size_t padded = size_t(AioPadUp(length));
      char *data = buffers.get() + slot * chunk_;
      std::memset(data + length, 0, padded - length);
      auto &req = requests[slot];
      req = IORequest{};
      req.offset = starts_[c] + written[c];
      req.len = padded;
      req.buf = data;
      pending[slot] = 1;
      ++inflight;
      io.SubmitWrite(req);
      ++write_requests_;
      written[c] += length;
      active[c] ^= 1;
      positions[c] = 0;
      ensure(2 * c + active[c]);
    };
    std::vector<char> node(page), record(record_);
    AioSeqStreamReader reader(graph, begin, begin + uint64_t(n_) * page);
    for (size_t rank = 0; rank < n_; ++rank) {
      reader.Read(node.data(), page);
      uint32_t degree = 0;
      std::memcpy(&degree, node.data() + seed_offset + m_ * 4, 4);
      if (degree > m_)
        throw std::runtime_error("invalid cluster input seed degree");
      std::fill(record.begin(), record.end(), 0);
      const uint32_t r = uint32_t(rank);
      std::memcpy(record.data(), &r, 4);
      std::memcpy(record.data() + 4, &degree, 4);
      std::memcpy(record.data() + 8, node.data() + vector_offset, vb_);
      std::memcpy(record.data() + 8 + vb_, node.data() + seed_offset, degree * 4);
      for (size_t a = 0; a < np; ++a) {
        const size_t c = cids[rank * np + a];
        if (c >= k || records[c] >= counts_[c])
          throw std::runtime_error("cluster input membership mismatch");
        for (size_t b = 0; b < a; ++b)
          if (cids[rank * np + b] == c)
            throw std::runtime_error("duplicate cluster membership");
        size_t off = 0;
        while (off < record_) {
          const size_t take = std::min(record_ - off, chunk_ - positions[c]);
          std::memcpy(buffers.get() + (2 * c + active[c]) * chunk_ + positions[c],
                      record.data() + off, take);
          positions[c] += take;
          off += take;
          if (positions[c] == chunk_)
            flush(c, chunk_);
        }
        ++records[c];
      }
    }
    for (size_t c = 0; c < k; ++c) {
      flush(c, positions[c]);
      if (records[c] != counts_[c] || written[c] != uint64_t(counts_[c]) * record_)
        throw std::runtime_error("incomplete cluster input distribution");
    }
    for (size_t slot = 0; slot < 2 * k; ++slot)
      ensure(slot);
    if (::fsync(io.fd()) != 0)
      throw std::runtime_error("cluster input sync failed");
    raw_read_requests_ = (uint64_t(n_) * page + kAioStreamBytes - 1) / kAioStreamBytes;
    fprintf(
        stderr,
        "[ClusterInputIO] distribution nodes=%zu records=%llu record_bytes=%zu buffer_bytes=%zu raw_read_requests=%zu input_write_requests=%zu input_bytes=%zu\n",
        n_, static_cast<unsigned long long>(members_), record_, 2 * k * chunk_, raw_read_requests_,
        write_requests_, InputBytes());
  }
  std::unique_ptr<AioSeqStreamReader> Reader(size_t c) {
    if (c >= counts_.size() || read_[c] || !counts_[c])
      throw std::runtime_error("invalid cluster input reader");
    read_[c] = 1;
    const uint64_t bytes = uint64_t(counts_[c]) * record_;
    read_requests_ += (AioPadUp(bytes) + kAioStreamBytes - 1) / kAioStreamBytes;
    read_records_ += counts_[c];
    return std::make_unique<AioSeqStreamReader>(path_, RegionBegin(c), RegionEnd(c));
  }
  void ReportComplete() const {
    if (read_records_ != members_)
      throw std::runtime_error("cluster input records not consumed once");
    fprintf(
        stderr,
        "[ClusterInputIO] loading records=%llu input_read_requests=%zu vector_gather_requests=0 seed_gather_requests=0 input_bytes=%zu\n",
        static_cast<unsigned long long>(read_records_), read_requests_, InputBytes());
  }

private:
  std::string path_;
  size_t n_, vb_, m_, record_, chunk_{0};
  std::vector<size_t> counts_;
  std::vector<uint64_t> starts_;
  std::vector<uint8_t> read_;
  uint64_t file_bytes_{0}, members_{0}, read_records_{0};
  size_t raw_read_requests_{0}, write_requests_{0}, read_requests_{0};
  bool owned_{false};
};
} // namespace alliance::disk_build
