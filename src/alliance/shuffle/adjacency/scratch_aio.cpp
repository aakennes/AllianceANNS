#include "shuffle/adjacency/scratch_aio.hpp"
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <thread>

namespace alliance::shuffle_detail {
size_t ScratchAlignedSize(size_t bytes) {
  if (bytes > std::numeric_limits<size_t>::max() - (kScratchAlignment - 1))
    throw std::overflow_error("scratch aligned size overflow");
  return (bytes + kScratchAlignment - 1) & ~(kScratchAlignment - 1);
}
ScratchAio::ScratchAio() {
  void *buffer = nullptr;
  const int error = ::posix_memalign(&buffer, kScratchAlignment, kScratchAioBufferBytes);
  if (error)
    throw std::system_error(error, std::generic_category(), "scratch aligned buffer");
  buffer_ = static_cast<uint8_t *>(buffer);
}
ScratchAio::~ScratchAio() {
  if (context_)
    ::io_destroy(context_); // Accepted I/O must stop using buffers before free.
  std::free(buffer_);
}
void ScratchAio::Run(const std::vector<ScratchAioRequest> &requests, ScratchIo &counters) {
  if (requests.empty())
    return;
  if (requests.size() > kScratchAioDepth)
    throw std::invalid_argument("scratch AIO batch depth");
  std::array<iocb, kScratchAioDepth> controls{};
  std::array<iocb *, kScratchAioDepth> pointers{};
  std::array<io_event, kScratchAioDepth> events{};
  std::array<bool, kScratchAioDepth> completed{};
  try {
    if (!context_) {
      const int result = ::io_setup(kScratchAioDepth, &context_);
      if (result < 0)
        throw std::system_error(-result, std::generic_category(), "scratch io_setup");
    }
    for (size_t i = 0; i < requests.size(); ++i) {
      const auto &r = requests[i];
      if (!r.bytes || r.minimum_bytes > r.bytes || r.offset > INT64_MAX - r.bytes ||
          r.offset % kScratchAlignment || r.bytes % kScratchAlignment ||
          reinterpret_cast<uintptr_t>(r.buffer) % kScratchAlignment)
        throw std::invalid_argument("unaligned/invalid scratch AIO request");
      if (r.write)
        io_prep_pwrite(&controls[i], r.fd, r.buffer, r.bytes, r.offset);
      else
        io_prep_pread(&controls[i], r.fd, r.buffer, r.bytes, r.offset);
      controls[i].data = reinterpret_cast<void *>(i + 1);
      pointers[i] = &controls[i];
    }
    size_t submitted = 0, reaped = 0, empty_retries = 0;
    while (reaped < requests.size()) {
      if (submitted < requests.size()) {
        ++counters.submit_calls;
        const int result =
            ::io_submit(context_, requests.size() - submitted, pointers.data() + submitted);
        if (result == -EINTR)
          continue;
        if (result < 0 && result != -EAGAIN)
          throw std::system_error(-result, std::generic_category(), "scratch io_submit");
        if (result > 0) {
          for (size_t i = submitted; i < submitted + size_t(result); ++i) {
            const auto &r = requests[i];
            if (r.write) {
              ++counters.aio_writes;
              counters.aio_write_bytes += r.bytes;
            } else {
              ++counters.aio_reads;
              counters.aio_read_bytes += r.bytes;
            }
          }
          submitted += result;
          empty_retries = 0;
          uint64_t peak = counters.max_inflight.load();
          while (peak < submitted - reaped &&
                 !counters.max_inflight.compare_exchange_weak(peak, submitted - reaped)) {
          }
        } else if (submitted == reaped) {
          if (++empty_retries > 16)
            throw std::runtime_error("scratch AIO made no submission progress");
          std::this_thread::yield();
          continue;
        }
      }
      // Drain partial acceptance before retrying EAGAIN; never spin on a full queue.
      if (submitted == reaped)
        continue;
      ++counters.wait_calls;
      const int got = ::io_getevents(context_, 1, submitted - reaped, events.data(), nullptr);
      if (got == -EINTR)
        continue;
      if (got < 0)
        throw std::system_error(-got, std::generic_category(), "scratch io_getevents");
      for (int e = 0; e < got; ++e) {
        const size_t i = reinterpret_cast<uintptr_t>(events[e].data) - 1;
        if (i >= submitted || completed[i])
          throw std::runtime_error("invalid scratch AIO completion");
        completed[i] = true;
        const auto bytes = static_cast<int64_t>(events[e].res);
        ++counters.aio_completed;
        if (bytes > 0)
          counters.aio_completed_bytes += bytes;
        if (bytes < 0)
          throw std::system_error(-bytes, std::generic_category(), "scratch AIO completion");
        if (events[e].res2 || uint64_t(bytes) < requests[i].minimum_bytes ||
            uint64_t(bytes) > requests[i].bytes)
          throw std::runtime_error("short or invalid scratch AIO completion");
      }
      reaped += got;
    }
  } catch (...) {
    ++counters.errors;
    // Destruction waits for all accepted requests, including those not reaped
    // after a failed completion. No stack iocb or caller buffer can escape live.
    if (context_) {
      ::io_destroy(context_);
      context_ = {};
    }
    throw;
  }
}
ScratchAio &ThreadScratchAio() {
  static thread_local ScratchAio engine;
  return engine;
}
} // namespace alliance::shuffle_detail
