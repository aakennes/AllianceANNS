#pragma once
// alliance/core/aio_stream.hpp -- libaio/O_DIRECT sequential stream primitives.
//
// Both classes were written for the Phase 2 cluster spill and lived in an
// anonymous namespace inside index_disk_cluster_build.cpp. They are the only
// sequential-stream shape the build needs, and every remaining synchronous
// bulk loop elsewhere (node-record writes, buildvecs, rawvecs, the RaBitQ
// quantization passes) is the same shape, so they moved here rather than being
// re-derived per call site. Nothing about them is Phase-2 specific.
//
// The standing rule for this index is that all index IO is async: libaio with
// O_DIRECT, never ::pread/::pwrite/::write. These two classes are what a
// sequential loop uses to obey it without hand-rolling a submit/reap pair.

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include <fcntl.h>
#include <libaio.h>
#include <sys/stat.h>
#include <unistd.h>

namespace alliance {

constexpr uint64_t kAioAlign = 512;              // O_DIRECT granularity
constexpr uint64_t kAioStreamBytes = 4ull << 20; // 4 MB stream buffers

/// Rounds `bytes` up to the O_DIRECT sector multiple.
inline auto AioPadUp(uint64_t bytes) -> uint64_t {
  return (bytes + kAioAlign - 1) / kAioAlign * kAioAlign;
}

/// Creates (or truncates) `path` and sizes it to `bytes`, which must already be
/// a sector multiple -- AioRegionWriter writes at O_DIRECT offsets inside an
/// existing extent and never appends. ::open/::ftruncate are metadata calls,
/// not data IO, so they stay synchronous by design.
inline void AioPreallocate(const std::string &path, uint64_t bytes) {
  int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_LARGEFILE, 0644);
  if (fd < 0) {
    throw std::runtime_error("AioPreallocate: cannot create " + path);
  }
  if (::ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
    ::close(fd);
    throw std::runtime_error("AioPreallocate: ftruncate failed for " + path);
  }
  ::close(fd);
}

/// Cuts a preallocated file back to its logical length, dropping the sector pad
/// the O_DIRECT tail flush had to write. Only needed when the logical length is
/// not itself a sector multiple.
inline void AioTruncateTo(const std::string &path, uint64_t bytes) {
  int fd = ::open(path.c_str(), O_WRONLY | O_LARGEFILE);
  if (fd < 0) {
    throw std::runtime_error("AioTruncateTo: cannot open " + path);
  }
  if (::ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
    ::close(fd);
    throw std::runtime_error("AioTruncateTo: ftruncate failed for " + path);
  }
  ::close(fd);
}

// libaio sequential byte-stream reader over [begin, end) of a file, backed by
// two chunk buffers with the next chunk always in flight. The range need not be
// sector aligned: requests are rounded out to 512 B, and the rounded tail may
// complete short at EOF, which Linux reports as a short io_event res rather
// than an error -- only logical bytes in [begin, end) are ever served.
//
// `direct` picks the fd mode. O_DIRECT is the default and is what every index
// file wants: the data is read once, so caching it only evicts pages another
// stage still needs. Pass false when the same bytes are read several times over
// and the page cache is the cheapest way to serve the repeats -- the fvecs
// input below is the one such case. Buffered libaio completes inside io_submit
// (no device/CPU overlap), so the only thing it saves over a read loop is
// syscalls; that is still the point, since a chunked stream turns a per-record
// read into one submit per chunk.
class AioSeqStreamReader {
public:
  AioSeqStreamReader(const std::string &path, uint64_t begin, uint64_t end, bool direct = true,
                     uint64_t chunk_bytes = kAioStreamBytes)
      : logical_end_(end), chunk_bytes_(chunk_bytes) {
    if (chunk_bytes_ == 0 || chunk_bytes_ % kAioAlign != 0) {
      throw std::runtime_error("AioSeqStreamReader: chunk_bytes must be a nonzero sector multiple");
    }
    int flags = O_RDONLY | O_LARGEFILE;
    if (direct)
      flags |= O_DIRECT;
    fd_ = ::open(path.c_str(), flags);
    if (fd_ < 0) {
      throw std::runtime_error("AioSeqStreamReader: open failed: " + path);
    }
    if (io_setup(4, &ctx_) < 0) {
      ::close(fd_);
      throw std::runtime_error("AioSeqStreamReader: io_setup failed");
    }
    raw_begin_ = begin & ~(kAioAlign - 1);
    skip_ = begin - raw_begin_;
    read_end_ = (end + kAioAlign - 1) & ~(kAioAlign - 1);
    next_submit_ = raw_begin_;
    for (int i = 0; i < 2; ++i) {
      bufs_[i] = static_cast<char *>(aligned_alloc(kAioAlign, chunk_bytes_));
      if (bufs_[i] == nullptr) {
        throw std::runtime_error("AioSeqStreamReader: aligned_alloc failed");
      }
      Submit(i);
    }
    Ensure(0);
    serve_off_ = begin;
  }

  ~AioSeqStreamReader() {
    for (int i = 0; i < 2; ++i)
      Ensure(i); // reap before freeing the buffers
    if (ctx_ != nullptr)
      io_destroy(ctx_);
    if (fd_ >= 0)
      ::close(fd_);
    free(bufs_[0]);
    free(bufs_[1]);
  }

  AioSeqStreamReader(const AioSeqStreamReader &) = delete;
  AioSeqStreamReader &operator=(const AioSeqStreamReader &) = delete;

  // Serves exactly len logical bytes; throws on exhaustion.
  void Read(void *dst, size_t len) {
    Consume(static_cast<char *>(dst), len);
  }

  /// Advances the stream by len logical bytes without copying them. Record
  /// scans that only want one slice (the vector region of a node record, say)
  /// use this for the header/edge/pad bytes around it: the chunks still have to
  /// be read off the device, but the memcpy out of the stream buffer is skipped.
  void Skip(size_t len) {
    Consume(nullptr, len);
  }

private:
  void Consume(char *out, size_t len) {
    if (serve_off_ + len > logical_end_) {
      throw std::runtime_error("AioSeqStreamReader: read past end");
    }
    while (len > 0) {
      if (serve_off_ == off_[cur_] + valid_[cur_]) {
        Submit(cur_); // recycle the consumed buffer
        cur_ ^= 1;
        Ensure(cur_);
        if (valid_[cur_] == 0) {
          throw std::runtime_error("AioSeqStreamReader: unexpected EOF");
        }
      }
      const size_t avail = (size_t)(off_[cur_] + valid_[cur_] - serve_off_);
      const size_t take = std::min(avail, len);
      if (out != nullptr) { // nullptr == Skip: advance the cursor, copy nothing
        std::memcpy(out, bufs_[cur_] + (serve_off_ - off_[cur_]), take);
        out += take;
      }
      serve_off_ += take;
      len -= take;
    }
  }

  void Submit(int bi) {
    if (next_submit_ >= read_end_) {
      pending_[bi] = false;
      done_[bi] = true;
      valid_[bi] = 0;
      off_[bi] = next_submit_;
      return;
    }
    off_[bi] = next_submit_;
    req_len_[bi] = std::min(chunk_bytes_, read_end_ - next_submit_);
    io_prep_pread(&cbs_[bi], fd_, bufs_[bi], req_len_[bi], (long long)off_[bi]);
    cbs_[bi].data = reinterpret_cast<void *>((intptr_t)bi);
    struct iocb *list[1] = {&cbs_[bi]};
    if (io_submit(ctx_, 1, list) != 1) {
      throw std::runtime_error("AioSeqStreamReader: io_submit failed");
    }
    pending_[bi] = true;
    done_[bi] = false;
    next_submit_ += req_len_[bi];
  }

  void Ensure(int bi) {
    while (pending_[bi] && !done_[bi]) {
      struct io_event ev;
      int r = io_getevents(ctx_, 1, 1, &ev, nullptr);
      if (r < 0) {
        if (errno == EINTR)
          continue;
        throw std::runtime_error("AioSeqStreamReader: io_getevents failed");
      }
      const int b = (int)(intptr_t)ev.data;
      // A short completion is only legal on the rounded tail, and even there
      // it must cover every logical byte this chunk owes.
      const uint64_t required = std::min(req_len_[b], logical_end_ - off_[b]);
      if ((long long)ev.res < 0 || (uint64_t)ev.res < required) {
        throw std::runtime_error("AioSeqStreamReader: short read");
      }
      valid_[b] = req_len_[b];
      done_[b] = true;
    }
  }

  int fd_{-1};
  io_context_t ctx_{nullptr};
  uint64_t logical_end_{0};
  uint64_t chunk_bytes_{kAioStreamBytes};
  uint64_t raw_begin_{0}, skip_{0}, read_end_{0}, next_submit_{0};
  char *bufs_[2]{nullptr, nullptr};
  struct iocb cbs_[2];
  uint64_t off_[2]{0, 0}, req_len_[2]{0, 0}, valid_[2]{0, 0};
  bool pending_[2]{false, false}, done_[2]{true, true};
  int cur_{0};
  uint64_t serve_off_{0};
};

// Sequential fvecs reader on libaio -- the input side of PopulateElements.
//
// It replaces an ifstream that did two istream::read calls per vector (4 B dim
// header, then d*4 B of payload). At 2M x D=384 that is 4.3M stream reads per
// pass, and PopulateElements makes 1 + `passes` passes (the Phase 2 rank window
// re-streams the file once per window), so under --build_mem_gb 1.0 it was 34M
// stream reads over 26 GB. A chunked stream serves the same bytes with one
// io_submit per chunk instead.
//
// Deliberately NOT O_DIRECT. The repeats are the whole point: the file is a few
// GB and gets read `passes`+1 times, so the page cache serves every pass after
// the first at memory speed. O_DIRECT here would trade ~1-2 s of cached re-reads
// for real device reads on every pass (at 2M/7 passes: 23 GB, ~7 s) to isolate a
// cache effect nobody is measuring. If the pass count ever drops to 1, revisit
// -- a single pass has no repeats to serve and O_DIRECT stops costing anything.
class AioFvecsStreamReader {
public:
  static constexpr uint64_t kFvecsChunkBytes = 1ull << 20; // 2 x 1 MB staged

  explicit AioFvecsStreamReader(const std::string &path, uint64_t chunk_bytes = kFvecsChunkBytes)
      : path_(path), chunk_bytes_(chunk_bytes) {
    struct stat st{};
    if (::stat(path.c_str(), &st) != 0) {
      throw std::runtime_error("AioFvecsStreamReader: stat failed: " + path);
    }
    file_bytes_ = static_cast<uint64_t>(st.st_size);
    if (file_bytes_ < sizeof(int)) {
      throw std::runtime_error("AioFvecsStreamReader: file too small to hold a dim: " + path);
    }
    // One sector to learn the dimension, before the record size is known. The
    // range is 4 B, so this probe allocates 2 x 512 B and issues one read.
    {
      int d = 0;
      AioSeqStreamReader probe(path, 0, sizeof(int), /*direct=*/false, kAioAlign);
      probe.Read(&d, sizeof(int));
      if (d <= 0) {
        throw std::runtime_error("AioFvecsStreamReader: bad dim " + std::to_string(d) + " in " +
                                 path);
      }
      dim_ = static_cast<size_t>(d);
    }
    rec_bytes_ = sizeof(int) + dim_ * sizeof(float);
    count_ = static_cast<size_t>(file_bytes_ / rec_bytes_);
    Rewind();
  }

  auto dim() const -> size_t {
    return dim_;
  }
  auto count() const -> size_t {
    return count_;
  }

  /// Serves the next vector's payload, dropping its dim header. Returns false
  /// once every whole record in the file has been served (a trailing partial
  /// record is not one, exactly as `count()` counts them).
  bool ReadNext(float *out_vec) {
    if (served_ >= count_)
      return false;
    int d = 0;
    stream_->Read(&d, sizeof(int));
    if (static_cast<size_t>(d) != dim_) {
      // The ifstream version silently accepted a per-record dim change and then
      // read dim_ floats anyway, i.e. it desynchronized the stream and returned
      // garbage. fvecs has one dim by definition, so this is corruption.
      throw std::runtime_error("AioFvecsStreamReader: record " + std::to_string(served_) +
                               " has dim " + std::to_string(d) + ", expected " +
                               std::to_string(dim_));
    }
    stream_->Read(out_vec, dim_ * sizeof(float));
    ++served_;
    return true;
  }

  /// Restarts from the first record. Tears the stream down and rebuilds it, so
  /// callers that re-stream per window may just construct a new reader instead.
  void Rewind() {
    stream_ = std::make_unique<AioSeqStreamReader>(
        path_, 0, static_cast<uint64_t>(count_) * rec_bytes_, /*direct=*/false, chunk_bytes_);
    served_ = 0;
  }

private:
  std::string path_;
  uint64_t chunk_bytes_{kFvecsChunkBytes};
  uint64_t file_bytes_{0};
  size_t dim_{0}, rec_bytes_{0}, count_{0}, served_{0};
  std::unique_ptr<AioSeqStreamReader> stream_;
};

// Buffered libaio pwrite stream over a byte region of a shared, preallocated
// file, O_DIRECT. Two 4 MB staging buffers keep one write in flight while the
// other is being filled.
//
// It used to run buffered, on the argument that dense records and unaligned
// region bases ruled O_DIRECT out. Only the first half of that was ever true:
// `region_base` has been 0 at the sole call site since the spill flush was
// consolidated into one writer per cluster, and the spill file is ftruncate'd
// to a sector multiple before pass A starts. The buffered write therefore
// bought nothing and cost twice: the 1 GB+ it dirtied was fdatasync'd and then
// re-read with O_DIRECT by pass B, so the cached copy was never served, and
// evicting it displaced pages other build IO wanted. What actually blocked
// O_DIRECT was the staging buffers (plain new[], not sector-aligned) and the
// unaligned final flush -- both handled below. `base_` must stay sector-aligned
// for the offsets to be legal, which the constructor now enforces.
class AioRegionWriter {
public:
  AioRegionWriter(const std::string &path, uint64_t region_base) : base_(region_base) {
    if (base_ % kAioAlign != 0) {
      throw std::runtime_error("AioRegionWriter: region_base must be sector-aligned for O_DIRECT");
    }
    fd_ = ::open(path.c_str(), O_WRONLY | O_DIRECT | O_LARGEFILE);
    if (fd_ < 0) {
      throw std::runtime_error("AioRegionWriter: open failed: " + path);
    }
    if (io_setup(4, &ctx_) < 0) {
      ::close(fd_);
      throw std::runtime_error("AioRegionWriter: io_setup failed");
    }
    bufs_[0] = static_cast<char *>(aligned_alloc(kAioAlign, kAioStreamBytes));
    bufs_[1] = static_cast<char *>(aligned_alloc(kAioAlign, kAioStreamBytes));
    if (bufs_[0] == nullptr || bufs_[1] == nullptr) {
      free(bufs_[0]);
      free(bufs_[1]);
      io_destroy(ctx_);
      ::close(fd_);
      throw std::runtime_error("AioRegionWriter: aligned_alloc failed");
    }
  }

  ~AioRegionWriter() {
    if (fd_ >= 0)
      Close();
    if (ctx_ != nullptr)
      io_destroy(ctx_);
    free(bufs_[0]);
    free(bufs_[1]);
  }

  AioRegionWriter(const AioRegionWriter &) = delete;
  AioRegionWriter &operator=(const AioRegionWriter &) = delete;

  void Append(const void *src, size_t len) {
    const char *p = static_cast<const char *>(src);
    while (len > 0) {
      const size_t space = (size_t)kAioStreamBytes - pos_;
      const size_t take = std::min(space, len);
      std::memcpy(bufs_[cur_] + pos_, p, take);
      pos_ += take;
      p += take;
      len -= take;
      if (pos_ == kAioStreamBytes) {
        Flush(cur_, pos_, pos_);
        cur_ ^= 1;
        pos_ = 0;
      }
    }
  }

  void Close() {
    if (fd_ < 0)
      return;
    if (pos_ > 0) {
      // O_DIRECT needs a sector-multiple length, so round the tail up and zero
      // the pad. The pad bytes land inside the file (callers preallocate to the
      // same sector multiple) and past the logical end, which every reader
      // bounds itself by, so they are never served.
      const size_t padded = (pos_ + kAioAlign - 1) / kAioAlign * kAioAlign;
      std::memset(bufs_[cur_] + pos_, 0, padded - pos_);
      Flush(cur_, padded, pos_);
      pos_ = 0;
    }
    Ensure(0);
    Ensure(1);
    // O_DIRECT already put the data on media and the file was preallocated, so
    // no metadata changed; this only guards filesystems that need the barrier
    // anyway, and costs nothing when they don't.
    if (::fdatasync(fd_) != 0) {
      ::close(fd_);
      fd_ = -1;
      throw std::runtime_error("AioRegionWriter: fdatasync failed");
    }
    ::close(fd_);
    fd_ = -1;
  }

  uint64_t written() const {
    return written_;
  }

private:
  /// `len` is what goes to the device (sector multiple); `logical_len` is what
  /// the caller appended. They differ only for the padded final flush, and
  /// `written_` tracks the logical total so callers can verify it. Every
  /// non-final flush is exactly kAioStreamBytes, so the next offset stays
  /// aligned; a short flush is therefore only ever the last one.
  void Flush(int bi, size_t len, size_t logical_len) {
    Ensure(bi); // the buffer's previous write must be done before reuse
    io_prep_pwrite(&cbs_[bi], fd_, bufs_[bi], len, (long long)(base_ + written_));
    cbs_[bi].data = reinterpret_cast<void *>((intptr_t)bi);
    struct iocb *list[1] = {&cbs_[bi]};
    if (io_submit(ctx_, 1, list) != 1) {
      throw std::runtime_error("AioRegionWriter: io_submit failed");
    }
    pending_[bi] = true;
    written_ += logical_len;
  }

  void Ensure(int bi) {
    while (pending_[bi]) {
      struct io_event ev;
      int r = io_getevents(ctx_, 1, 1, &ev, nullptr);
      if (r < 0) {
        if (errno == EINTR)
          continue;
        throw std::runtime_error("AioRegionWriter: io_getevents failed");
      }
      const int b = (int)(intptr_t)ev.data;
      if ((long long)ev.res < 0) {
        throw std::runtime_error("AioRegionWriter: write failed");
      }
      pending_[b] = false;
    }
  }

  int fd_{-1};
  io_context_t ctx_{nullptr};
  uint64_t base_{0}, written_{0};
  char *bufs_[2]{nullptr, nullptr};
  struct iocb cbs_[2];
  bool pending_[2]{false, false};
  int cur_{0};
  size_t pos_{0};
};

} // namespace alliance
