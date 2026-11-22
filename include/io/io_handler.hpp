#pragma once

#include <cstdint>
#include <cerrno>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <libaio.h>
#include <unistd.h>
#ifdef ALLIANCE_PIPELINE_PROFILE
#include "metrics/io_queue_profile.hpp"
#endif

namespace alliance {

struct IORequest {
  uint64_t offset{0};
  uint64_t len{0};
  void *buf{nullptr};
  bool finished{false};
  // Bytes transferred, or -errno. Callers that submit multi-page extents must
  // check this: a short read leaves the tail of `buf` holding whatever was
  // there before, and `finished` alone cannot tell that apart from success.
  long res{0};
};

class IOHandler {
public:
  static constexpr int kMaxEvents = 96;

  enum class CompletionWait { Blocking, BusyPoll };

  explicit IOHandler(size_t queue_depth = 64,
                     CompletionWait completion_wait = CompletionWait::Blocking)
      : completion_wait_(completion_wait) {
    std::memset(&ctx_, 0, sizeof(ctx_));
    if (io_setup(static_cast<int>(queue_depth), &ctx_) < 0) {
      throw std::runtime_error("io_setup failed for ctx_");
    }
    std::memset(&ctx_vec_, 0, sizeof(ctx_vec_));
    if (io_setup(static_cast<int>(queue_depth), &ctx_vec_) < 0) {
      DestroyContext(ctx_, pending_, false);
      throw std::runtime_error("io_setup failed for ctx_vec_");
    }
  }

  ~IOHandler() {
    // Do not dereference requests here: their owner must drain before they die.
    DestroyContext(ctx_, pending_, false);
    DestroyContext(ctx_vec_, pending_vec_, true);
    if (fd_ >= 0)
      ::close(fd_);
  }

  IOHandler(const IOHandler &) = delete;
  IOHandler &operator=(const IOHandler &) = delete;

  void Open(const std::string &path) {
    DrainPending(); // Stop the previous file's accesses before replacing its fd.
    CheckContext(ctx_);
    if (fd_ >= 0)
      ::close(fd_);
    fd_ = ::open(path.c_str(), O_DIRECT | O_RDONLY | O_LARGEFILE);
    if (fd_ < 0) {
      throw std::runtime_error("IOHandler::Open failed: " + path);
    }
  }

  void OpenRW(const std::string &path) {
    DrainPending(); // Stop the previous file's accesses before replacing its fd.
    CheckContext(ctx_);
    if (fd_ >= 0)
      ::close(fd_);
    fd_ = ::open(path.c_str(), O_DIRECT | O_RDWR | O_LARGEFILE);
    if (fd_ < 0) {
      throw std::runtime_error("IOHandler::OpenRW failed: " + path);
    }
  }

  void SubmitRead(IORequest &req) {
    SubmitOne(req, ctx_);
  }

  void SubmitReadBatch(const std::vector<IORequest *> &reqs) {
    SubmitBatch(reqs, ctx_);
  }

  void PollAll() {
    PollChecked(ctx_, 0, kMaxEvents);
  }

  void PollWait() {
    PollChecked(ctx_, 1, 1);
  }

  void PollWaitBatch(int max_events = kMaxEvents) {
    PollChecked(ctx_, 1, max_events);
  }

  void SubmitWrite(IORequest &req) {
    SubmitOneWrite(req, ctx_);
  }

  void SubmitWriteBatch(const std::vector<IORequest *> &reqs) {
    SubmitBatchWrite(reqs, ctx_);
  }

  void SubmitVecRead(IORequest &req) {
    SubmitOne(req, ctx_vec_);
  }

  void SubmitVecReadBatch(const std::vector<IORequest *> &reqs) {
    SubmitBatch(reqs, ctx_vec_);
  }

  void PollVecAll() {
    PollChecked(ctx_vec_, 0, kMaxEvents);
  }

  void PollVecWait() {
    PollChecked(ctx_vec_, 1, 1);
  }

  // Request/buffer owners call this before their storage leaves scope. It
  // completes both queues, or destroys a failing context as a lifetime barrier.
  // No callbacks or new allocations; normal empty queues require no syscalls.
  void DrainPending() noexcept {
    DrainContext(ctx_, pending_);
    DrainContext(ctx_vec_, pending_vec_);
  }

  auto fd() const -> int {
    return fd_;
  }
#ifdef ALLIANCE_PIPELINE_PROFILE
  void ResetQueueProfile(IOQueueObserver *observer) {
    queue_profile_.Reset(observer);
  }
  const IOQueueTracker &QueueProfile() const {
    return queue_profile_;
  }
#endif

private:
  void SubmitOne(IORequest &req, io_context_t ctx) {
    CheckContext(ctx);
    req.finished = false;
    struct iocb cb;
    io_prep_pread(&cb, fd_, req.buf, req.len, req.offset);
    cb.data = reinterpret_cast<void *>(&req);
    struct iocb *cbs[1] = {&cb};
    int ret = io_submit(ctx, 1, cbs);
    if (ret > 0) {
      Pending(ctx) += static_cast<size_t>(ret);
#ifdef ALLIANCE_PIPELINE_PROFILE
      queue_profile_.Submit(static_cast<size_t>(ret), ctx == ctx_vec_);
#endif
    }
    if (ret != 1) {
      DrainPending();
      throw std::runtime_error("io_submit failed (single read)");
    }
  }

  void SubmitOneWrite(IORequest &req, io_context_t ctx) {
    CheckContext(ctx);
    req.finished = false;
    struct iocb cb;
    io_prep_pwrite(&cb, fd_, req.buf, req.len, req.offset);
    cb.data = reinterpret_cast<void *>(&req);
    struct iocb *cbs[1] = {&cb};
    int ret = io_submit(ctx, 1, cbs);
    if (ret > 0) {
      Pending(ctx) += static_cast<size_t>(ret);
#ifdef ALLIANCE_PIPELINE_PROFILE
      queue_profile_.Submit(static_cast<size_t>(ret), ctx == ctx_vec_);
#endif
    }
    if (ret != 1) {
      DrainPending();
      throw std::runtime_error("io_submit failed (single write)");
    }
  }

  void SubmitBatch(const std::vector<IORequest *> &reqs, io_context_t ctx) {
    if (reqs.empty())
      return;
    CheckContext(ctx);
    size_t n = reqs.size();
    if (n > batch_cbs_.size()) {
      batch_cbs_.resize(n);
      batch_cb_ptrs_.resize(n);
    }
    for (size_t i = 0; i < n; ++i) {
      reqs[i]->finished = false;
      io_prep_pread(&batch_cbs_[i], fd_, reqs[i]->buf, reqs[i]->len, reqs[i]->offset);
      batch_cbs_[i].data = reinterpret_cast<void *>(reqs[i]);
      batch_cb_ptrs_[i] = &batch_cbs_[i];
    }
    int ret = io_submit(ctx, static_cast<long>(n), batch_cb_ptrs_.data());
    if (ret > 0) {
      Pending(ctx) += static_cast<size_t>(ret);
#ifdef ALLIANCE_PIPELINE_PROFILE
      queue_profile_.Submit(static_cast<size_t>(ret), ctx == ctx_vec_);
#endif
    }
    if (ret != static_cast<int>(n)) {
      DrainPending();
      MarkCancelledAccepted(reqs, ret);
      throw std::runtime_error("io_submit failed (batch read): submitted " + std::to_string(ret) +
                               "/" + std::to_string(n));
    }
  }

  void SubmitBatchWrite(const std::vector<IORequest *> &reqs, io_context_t ctx) {
    if (reqs.empty())
      return;
    CheckContext(ctx);
    size_t n = reqs.size();
    if (n > batch_cbs_.size()) {
      batch_cbs_.resize(n);
      batch_cb_ptrs_.resize(n);
    }
    for (size_t i = 0; i < n; ++i) {
      reqs[i]->finished = false;
      io_prep_pwrite(&batch_cbs_[i], fd_, reqs[i]->buf, reqs[i]->len, reqs[i]->offset);
      batch_cbs_[i].data = reinterpret_cast<void *>(reqs[i]);
      batch_cb_ptrs_[i] = &batch_cbs_[i];
    }
    int ret = io_submit(ctx, static_cast<long>(n), batch_cb_ptrs_.data());
    if (ret > 0) {
      Pending(ctx) += static_cast<size_t>(ret);
#ifdef ALLIANCE_PIPELINE_PROFILE
      queue_profile_.Submit(static_cast<size_t>(ret), ctx == ctx_vec_);
#endif
    }
    if (ret != static_cast<int>(n)) {
      DrainPending();
      MarkCancelledAccepted(reqs, ret);
      throw std::runtime_error("io_submit failed (batch write): submitted " + std::to_string(ret) +
                               "/" + std::to_string(n));
    }
  }

  int PollCtx(io_context_t ctx, int min_nr, int max_nr) {
    if (!ctx)
      return -EINVAL;
    struct io_event evts[kMaxEvents];
    int nr = std::min(max_nr, kMaxEvents);
    // Search contexts may keep their worker runnable while waiting. Build
    // handlers retain blocking waits; PollAll remains a single nonblocking poll.
    const bool busy_wait = completion_wait_ == CompletionWait::BusyPoll && min_nr == 1;
    if (busy_wait && nr < 1)
      return -EINVAL;
    int ret;
    do {
      ret = io_getevents(ctx, busy_wait ? 0 : min_nr, nr, evts, nullptr);
      if (busy_wait && ret == 0) {
#if defined(__i386__) || defined(__x86_64__)
        __builtin_ia32_pause();
#endif
      }
    } while (busy_wait && ret == 0);
    if (ret < 0)
      return ret;
#ifdef ALLIANCE_PIPELINE_PROFILE
    if (ret > 0)
      queue_profile_.Complete(static_cast<size_t>(ret), ctx == ctx_vec_);
#endif
    for (int i = 0; i < ret; ++i) {
      auto *req = reinterpret_cast<IORequest *>(evts[i].data);
      if (req) {
        req->res = static_cast<long>(evts[i].res);
        req->finished = true;
      }
      if (Pending(ctx) > 0)
        --Pending(ctx);
    }
    return ret;
  }

  // A handler has one owner, but its edge and vector contexts may both have
  // outstanding requests. On submission failure drain BOTH before unwinding:
  // io_submit may have accepted only a prefix, and previous batches can still
  // refer to stack-owned IORequest objects in the same caller.
  size_t &Pending(io_context_t ctx) {
    return ctx == ctx_ ? pending_ : pending_vec_;
  }

  void CheckContext(io_context_t ctx) const {
    if (!ctx || poisoned_)
      throw std::runtime_error("IOHandler cannot submit after an unrecoverable AIO failure");
  }

  void PollChecked(io_context_t ctx, int min_nr, int max_nr) {
    CheckContext(ctx);
    for (;;) {
      const int ret = PollCtx(ctx, min_nr, max_nr);
      if (ret >= 0)
        return;
      if (ret == -EINTR || ret == -EAGAIN) {
        if (min_nr == 0)
          return; // The nonblocking caller will poll again.
        continue;
      }
      DrainPending();
      throw std::runtime_error("io_getevents failed: " + std::to_string(ret));
    }
  }

  void DestroyContext(io_context_t &ctx, size_t &pending, bool vector_queue) noexcept {
    if (ctx) {
      int ret;
      do {
        ret = io_destroy(ctx);
      } while (ret == -EINTR);
      if (ret < 0)
        std::terminate(); // Cannot safely release live IO buffers.
    }
#ifdef ALLIANCE_PIPELINE_PROFILE
    queue_profile_.Complete(pending, vector_queue);
#endif
    ctx = nullptr;
    pending = 0;
  }

  void DrainContext(io_context_t &ctx, size_t &pending) noexcept {
    while (pending > 0) {
      int ret = PollCtx(ctx, 1, kMaxEvents);
      if (ret >= 0 || ret == -EINTR || ret == -EAGAIN)
        continue;
      // io_destroy cancels requests or waits for those that cannot be
      // cancelled. It is a lifetime barrier even when completions cannot be
      // collected. Do not permit reuse of a handler that required this path.
      DestroyContext(ctx, pending, &pending == &pending_vec_);
      poisoned_ = true;
    }
  }

  void MarkCancelledAccepted(const std::vector<IORequest *> &reqs, int accepted) {
    // Only needed if the destruction barrier replaced completion collection.
    // Unaccepted requests remain unfinished, and successful completions retain
    // their actual result. The handler is poisoned after this fallback.
    for (int i = 0; i < accepted; ++i) {
      if (!reqs[i]->finished) {
        reqs[i]->res = -ECANCELED;
        reqs[i]->finished = true;
      }
    }
  }

#ifdef ALLIANCE_PIPELINE_PROFILE
  IOQueueTracker queue_profile_;
#endif
  CompletionWait completion_wait_{CompletionWait::Blocking};
  io_context_t ctx_{nullptr};
  io_context_t ctx_vec_{nullptr};
  size_t pending_{0};
  size_t pending_vec_{0};
  bool poisoned_{false};
  int fd_{-1};
  std::vector<struct iocb> batch_cbs_;
  std::vector<struct iocb *> batch_cb_ptrs_;
};

// Declare AFTER every borrowed request/buffer owner and BEFORE the first
// submission. The handler must outlive this guard. Draining does not call
// application completion callbacks or publish a partially computed result.
class IOCompletionGuard {
public:
  explicit IOCompletionGuard(IOHandler &handler) noexcept : handler_(handler) {
  }
  ~IOCompletionGuard() noexcept {
    handler_.DrainPending();
  }
  IOCompletionGuard(const IOCompletionGuard &) = delete;
  IOCompletionGuard &operator=(const IOCompletionGuard &) = delete;

private:
  IOHandler &handler_;
};

} // namespace alliance
