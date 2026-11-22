#pragma once
// alliance/core/paged_file.hpp -- Fixed-size, per-node-exclusive page file on SSD.
//
// One node occupies exactly one page of `node_disk_bytes` bytes, never shared
// with another node. This lets concurrent reads/writes of different nodes
// proceed without any locking, since their byte ranges never overlap.
//
// `node_disk_bytes` must be a multiple of kMinNodePageBytes (the physical page /
// O_DIRECT granularity). Anything smaller would let two nodes land in the same
// physical page and break the no-locking invariant above.
//
// All IO here goes through libaio (aio_batch.hpp), per the standing rule that
// this index issues no synchronous ::pread/::pwrite on index data. The fd is
// buffered, not O_DIRECT: a slice inside a record (one layer's edge list is
// (M+1)*4 + M*4 bytes at a non-sector offset) is not O_DIRECT-legal without a
// read-modify-write on every access. So the honest accounting is that libaio on
// this fd completes inline inside io_submit -- what the conversion buys is one
// syscall per *batch* instead of one per slice, plus a single IO idiom. Callers
// that need genuinely overlapping device IO on this file read whole 4 KB pages
// through IOHandler (O_DIRECT) instead.

#include <string>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include "io/aio_batch.hpp"
#include "core/types.hpp"
#include "storage/page_constants.hpp"

namespace alliance {

class PagedFile {
public:
  /// Opens (create=false) or creates (create=true, truncating) the file at `path`.
  /// `header_bytes` is reserved at the front of the file for a POD header (should
  /// itself be page-aligned, e.g. kDiskPageSize). `node_disk_bytes` is the fixed,
  /// page-aligned number of bytes exclusively occupied by one node's record.
  PagedFile(const std::string &path, bool create, size_t header_bytes, size_t node_disk_bytes)
      : path_(path), header_bytes_(header_bytes), node_disk_bytes_(node_disk_bytes) {
    if (!node_disk_bytes_ || node_disk_bytes_ % kMinNodePageBytes != 0 ||
        (node_disk_bytes_ & (node_disk_bytes_ - 1)) != 0 || header_bytes_ % kMinNodePageBytes != 0)
      throw std::invalid_argument(
          "PagedFile: node size must be a power of two; header/node must be 4 KiB aligned");
    int flags = create ? (O_RDWR | O_CREAT | O_TRUNC) : O_RDWR;
    fd_ = ::open(path.c_str(), flags, 0644);
    if (fd_ < 0) {
      throw std::runtime_error("PagedFile: failed to open " + path);
    }
  }

  ~PagedFile() {
    if (fd_ >= 0) {
      ::close(fd_);
    }
  }

  PagedFile(const PagedFile &) = delete;
  PagedFile &operator=(const PagedFile &) = delete;

  /// Grows the file to hold exactly `n` node records (plus the header).
  void Preallocate(size_t n) const {
    off_t total = static_cast<off_t>(header_bytes_) +
                  static_cast<off_t>(n) * static_cast<off_t>(node_disk_bytes_);
    if (::ftruncate(fd_, total) != 0) {
      throw std::runtime_error("PagedFile: ftruncate failed");
    }
  }

  void WriteHeader(const void *header, size_t header_bytes) const {
    RawWrite(header, header_bytes, 0);
  }

  /// Reads `len` bytes starting at `off_in_record` within node `id`'s record.
  void ReadNodeSlice(tableint id, size_t off_in_record, size_t len, void *buf) const {
    RawRead(buf, len, NodeFileOffset(id) + static_cast<off_t>(off_in_record));
  }

  /// Writes `len` bytes starting at `off_in_record` within node `id`'s record.
  void WriteNodeSlice(tableint id, size_t off_in_record, size_t len, const void *buf) const {
    RawWrite(buf, len, NodeFileOffset(id) + static_cast<off_t>(off_in_record));
  }

  /// Batched gather: arbitrary (buffer, length, file offset) triples submitted as
  /// queue-depth-sized libaio batches instead of one syscall per slice. This is
  /// the shape every per-node gather has -- a chunk of nodes' edge windows, a
  /// rank range's vectors -- and offsets need not be sorted, contiguous, or
  /// uniform in length. `node_offset(id)` is public so callers build the offsets
  /// themselves; that is what lets one element cover several layers of a record.
  void ReadSlices(const AioSlice *slices, size_t count) const {
    if (count == 0)
      return;
    TlsAioBatchEngine().Read(fd_, slices, count);
  }

  void Sync() const {
    if (::fsync(fd_) != 0) {
      throw std::runtime_error("PagedFile: fsync failed");
    }
  }

  auto fd() const -> int {
    return fd_;
  }
  auto node_disk_bytes() const -> size_t {
    return node_disk_bytes_;
  }
  auto header_bytes() const -> size_t {
    return header_bytes_;
  }
  auto path() const -> const std::string & {
    return path_;
  }

  auto node_offset(tableint id) const -> uint64_t {
    return static_cast<uint64_t>(header_bytes_) +
           static_cast<uint64_t>(id) * static_cast<uint64_t>(node_disk_bytes_);
  }

private:
  auto NodeFileOffset(tableint id) const -> off_t {
    return static_cast<off_t>(header_bytes_) +
           static_cast<off_t>(id) * static_cast<off_t>(node_disk_bytes_);
  }

  /// Single-slice libaio submit + drain. The engine handles short completions
  /// and EINTR, so this is the whole of what the ::pread loop used to be.
  void RawRead(void *buf, size_t len, off_t offset) const {
    TlsAioBatchEngine().ReadOne(fd_, buf, len, static_cast<uint64_t>(offset));
  }

  void RawWrite(const void *buf, size_t len, off_t offset) const {
    TlsAioBatchEngine().WriteOne(fd_, buf, len, static_cast<uint64_t>(offset));
  }

  int fd_{-1};
  std::string path_;
  size_t header_bytes_{0};
  size_t node_disk_bytes_{0};
};

} // namespace alliance
