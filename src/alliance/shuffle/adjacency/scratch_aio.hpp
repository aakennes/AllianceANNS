#pragma once
// Private direct-I/O engine. A batch owns its aligned buffers until all accepted
// libaio requests complete; failure destroys/drains the context before unwinding.
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <libaio.h>

namespace alliance::shuffle_detail {
inline constexpr size_t kScratchAlignment = 4096;
inline constexpr size_t kScratchAioDepth = 64;
inline constexpr size_t kScratchAioBufferBytes = 2 << 20;
inline constexpr size_t kScratchAioChunkBytes = 1 << 20;

struct ScratchIo {
  // Logical payload requests, not syscalls or device sectors. RMW/padding are
  // counted separately in the actual libaio request/completion counters below.
  std::atomic<uint64_t> read_calls{}, read_bytes{}, write_calls{}, write_bytes{};
  std::atomic<uint64_t> aio_reads{}, aio_writes{}, aio_read_bytes{}, aio_write_bytes{};
  std::atomic<uint64_t> aio_completed{}, aio_completed_bytes{}, submit_calls{}, wait_calls{};
  std::atomic<uint64_t> max_inflight{}, errors{};
};

struct ScratchAioRequest {
  int fd;
  void *buffer;
  size_t bytes;
  uint64_t offset;
  size_t minimum_bytes; // A short final direct read is valid only beyond payload EOF.
  bool write;
};

class ScratchAio {
public:
  ScratchAio();
  ~ScratchAio();
  ScratchAio(const ScratchAio &) = delete;
  uint8_t *Buffer() {
    return buffer_;
  }
  void Run(const std::vector<ScratchAioRequest> &requests, ScratchIo &counters);

private:
  io_context_t context_{};
  uint8_t *buffer_{};
};
ScratchAio &ThreadScratchAio();
size_t ScratchAlignedSize(size_t bytes);
} // namespace alliance::shuffle_detail
