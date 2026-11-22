#pragma once
// alliance/core/aio_batch.hpp -- libaio engine for scattered per-record IO.
//
// aio_stream.hpp covers the sequential shape (one 4 MB chunk after another).
// This covers the other shape the index needs: many small, arbitrarily placed
// reads/writes at once -- a node record's edge slice, a vector slice, a block's
// worth of per-node gathers. Those used to be ::pread/::pwrite loops, one
// syscall per slice; the standing rule for this index is that all index IO is
// async, so they go through io_submit instead.
//
// What batching buys and what it does not:
//   * It buys syscall count. One io_submit carries up to `depth` slices, so a
//     19M-slice gather costs ~300k submits instead of 19M preads.
//   * It does NOT buy device parallelism on a *buffered* fd. Linux has no async
//     path for buffered reads, so io_submit does the work inline in the calling
//     thread and returns already-completed events. The graph file has to stay
//     buffered: slice offsets and lengths inside a record (an edge list is
//     (M+1)*4 + M*4 = 260 B at a non-sector offset) are not O_DIRECT-legal
//     without read-modify-write on every access. Callers that want real
//     overlapping IO on the graph file use the O_DIRECT IOHandler path with
//     whole 4 KB pages; this engine is for the correctness-and-syscall-count
//     case, and the uniformity of having one IO idiom.
//
// One io_context_t per thread (TlsAioBatchEngine), shared across every file:
// an iocb names its own fd, so the context does not need to.

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include <libaio.h>

namespace alliance {

// The kernel's completion ring, mapped into the process by io_setup and pointed
// at by io_context_t. Layout is fixed ABI (libaio itself reads `magic` and
// `head`/`tail` out of it for its zero-timeout fast path), and the guard below
// falls back to the io_getevents syscall whenever the magic does not match.
//
// Why bother: on a buffered fd Linux has no async read path, so io_submit does
// the IO inline and the completion is already in the ring when io_submit
// returns. libaio's io_getevents(timeout=NULL) always enters the kernel anyway,
// which would make every converted slice cost two syscalls where the ::pread it
// replaced cost one. Harvesting the ring directly keeps the count at one, so the
// conversion is not a per-slice regression.
struct AioCompletionRing {
  unsigned id;
  unsigned nr;
  unsigned head;
  unsigned tail;
  unsigned magic;
  unsigned compat_features;
  unsigned incompat_features;
  unsigned header_length;
  struct io_event events[0];
};

constexpr unsigned kAioRingMagic = 0xa10a10a1;

/// Default queue depth of the per-thread engine. 64 matches the IOHandler
/// depth the search path uses, and keeps the per-thread iocb/event arrays at a
/// few KB.
constexpr size_t kAioBatchDepth = 64;

/// One scattered read or write: `len` bytes at file offset `off`, to/from `buf`.
struct AioSlice {
  void *buf{nullptr};
  size_t len{0};
  uint64_t off{0};
};

/// Submits scattered slices in queue-depth-sized batches and drains each batch
/// before returning: every call is complete when it returns, so callers own
/// their buffers for exactly the duration of the call and need no handles.
class AioBatchEngine {
public:
  explicit AioBatchEngine(size_t depth = kAioBatchDepth) : depth_(std::max<size_t>(depth, 1)) {
    if (io_setup(static_cast<int>(depth_), &ctx_) < 0) {
      throw std::runtime_error("AioBatchEngine: io_setup failed");
    }
    cbs_.resize(depth_);
    list_.resize(depth_);
    evs_.resize(depth_);
  }

  ~AioBatchEngine() {
    if (ctx_ != nullptr)
      io_destroy(ctx_);
  }

  AioBatchEngine(const AioBatchEngine &) = delete;
  AioBatchEngine &operator=(const AioBatchEngine &) = delete;

  void Read(int fd, const AioSlice *slices, size_t count) {
    Run(fd, slices, count, /*write=*/false);
  }

  void Write(int fd, const AioSlice *slices, size_t count) {
    Run(fd, slices, count, /*write=*/true);
  }

  /// Single-slice convenience wrappers -- the shape a per-record accessor uses.
  void ReadOne(int fd, void *buf, size_t len, uint64_t off) {
    AioSlice s{buf, len, off};
    Run(fd, &s, 1, /*write=*/false);
  }

  void WriteOne(int fd, const void *buf, size_t len, uint64_t off) {
    AioSlice s{const_cast<void *>(buf), len, off};
    Run(fd, &s, 1, /*write=*/true);
  }

  auto depth() const -> size_t {
    return depth_;
  }

private:
  void Run(int fd, const AioSlice *slices, size_t count, bool write) {
    size_t i = 0;
    while (i < count) {
      const size_t batch = std::min(depth_, count - i);
      for (size_t k = 0; k < batch; ++k) {
        const AioSlice &s = slices[i + k];
        if (s.len == 0) {
          throw std::runtime_error("AioBatchEngine: zero-length slice");
        }
        if (write) {
          io_prep_pwrite(&cbs_[k], fd, s.buf, s.len, static_cast<long long>(s.off));
        } else {
          io_prep_pread(&cbs_[k], fd, s.buf, s.len, static_cast<long long>(s.off));
        }
        cbs_[k].data = reinterpret_cast<void *>(static_cast<intptr_t>(k));
        list_[k] = &cbs_[k];
      }

      size_t submitted = 0;
      while (submitted < batch) {
        int r = io_submit(ctx_, static_cast<long>(batch - submitted), list_.data() + submitted);
        if (r < 0) {
          // libaio returns -errno. EAGAIN means the context is momentarily full
          // (only possible if a previous batch leaked, but retry rather than die).
          if (-r == EINTR || -r == EAGAIN)
            continue;
          throw std::runtime_error("AioBatchEngine: io_submit failed");
        }
        submitted += static_cast<size_t>(r);
      }

      size_t reaped = 0;
      while (reaped < batch) {
        // Ring first (no syscall when the completions are already there, which
        // is always for a buffered fd), syscall only when it comes up empty.
        size_t r = ReapRing(evs_.data(), batch - reaped);
        if (r == 0) {
          int g = io_getevents(ctx_, 1, static_cast<long>(batch - reaped), evs_.data(), nullptr);
          if (g < 0) {
            if (errno == EINTR)
              continue;
            throw std::runtime_error("AioBatchEngine: io_getevents failed");
          }
          r = static_cast<size_t>(g);
        }
        for (size_t e = 0; e < r; ++e) {
          const long long res = static_cast<long long>(evs_[e].res);
          const size_t k = static_cast<size_t>(reinterpret_cast<intptr_t>(evs_[e].data));
          if (res < 0) {
            throw std::runtime_error(write ? "AioBatchEngine: write failed"
                                           : "AioBatchEngine: read failed");
          }
          const AioSlice &s = slices[i + k];
          if (static_cast<size_t>(res) < s.len) {
            // Short completion. Legal for both directions (signals, partial page
            // at EOF), so the remainder still has to be issued -- but not from
            // inside this loop: its submit/reap would race the batch's own
            // outstanding events for the ring. Queue it and run it after the
            // batch has fully drained.
            shorts_.emplace_back(i + k, static_cast<size_t>(res));
          }
        }
        reaped += r;
      }
      for (const auto &sh : shorts_) {
        Finish(fd, slices[sh.first], sh.second, write);
      }
      shorts_.clear();
      i += batch;
    }
  }

  /// Drives the tail of a short completion to zero with single-slice submits.
  void Finish(int fd, const AioSlice &s, size_t done, bool write) {
    struct iocb cb;
    struct iocb *list[1] = {&cb};
    struct io_event ev;
    while (done < s.len) {
      char *p = static_cast<char *>(s.buf) + done;
      const size_t rest = s.len - done;
      if (write) {
        io_prep_pwrite(&cb, fd, p, rest, static_cast<long long>(s.off + done));
      } else {
        io_prep_pread(&cb, fd, p, rest, static_cast<long long>(s.off + done));
      }
      cb.data = nullptr;
      int r = io_submit(ctx_, 1, list);
      if (r < 0) {
        if (-r == EINTR || -r == EAGAIN)
          continue;
        throw std::runtime_error("AioBatchEngine: io_submit failed (tail)");
      }
      if (r != 1)
        continue;
      while (true) {
        if (ReapRing(&ev, 1) == 1)
          break;
        int g = io_getevents(ctx_, 1, 1, &ev, nullptr);
        if (g < 0) {
          if (errno == EINTR)
            continue;
          throw std::runtime_error("AioBatchEngine: io_getevents failed (tail)");
        }
        if (g == 1)
          break;
      }
      const long long res = static_cast<long long>(ev.res);
      if (res < 0) {
        throw std::runtime_error("AioBatchEngine: tail IO failed");
      }
      if (res == 0) {
        throw std::runtime_error("AioBatchEngine: short read (EOF)");
      }
      done += static_cast<size_t>(res);
    }
  }

  /// Copies up to `max` already-posted completions out of the kernel ring
  /// without a syscall. Safe because each engine has exactly one consumer (its
  /// own thread) and every call drains before returning, so nothing else can be
  /// moving `head`. Returns 0 when the ring is unusable or empty, which sends
  /// the caller to io_getevents.
  auto ReapRing(struct io_event *out, size_t max) -> size_t {
    auto *ring = reinterpret_cast<AioCompletionRing *>(ctx_);
    if (ring == nullptr || ring->magic != kAioRingMagic || ring->nr == 0)
      return 0;
    size_t got = 0;
    unsigned head = ring->head;
    while (got < max) {
      __sync_synchronize();
      const unsigned tail = ring->tail;
      if (head == tail)
        break;
      out[got++] = ring->events[head];
      head = (head + 1) % ring->nr;
    }
    if (got > 0) {
      ring->head = head;
      __sync_synchronize();
    }
    return got;
  }

  io_context_t ctx_{nullptr};
  size_t depth_{kAioBatchDepth};
  std::vector<struct iocb> cbs_;
  std::vector<struct iocb *> list_;
  std::vector<struct io_event> evs_;
  std::vector<std::pair<size_t, size_t>> shorts_; // (slice index, bytes already done)
};

/// The calling thread's engine. Every PagedFile op on this thread shares it;
/// each call drains before returning, so there is never more than one batch in
/// flight per thread and no cross-file interference.
inline auto TlsAioBatchEngine() -> AioBatchEngine & {
  static thread_local AioBatchEngine engine(kAioBatchDepth);
  return engine;
}

} // namespace alliance
