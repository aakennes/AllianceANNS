#include "shuffle/adjacency/scratch_io.hpp"
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <system_error>
#include <unistd.h>

namespace alliance::shuffle_detail {
namespace {
[[noreturn]] void Error(const char *operation) {
  throw std::system_error(errno, std::generic_category(), operation);
}
uint64_t End(uint64_t offset, size_t bytes) {
  if (offset > uint64_t(INT64_MAX) - kScratchAlignment ||
      bytes > uint64_t(INT64_MAX) - kScratchAlignment - offset)
    throw std::overflow_error("scratch range exceeds off_t");
  return offset + bytes;
}
} // namespace
ScratchDirectory::ScratchDirectory(const std::filesystem::path &parent) {
  if (parent.empty() || !std::filesystem::is_directory(parent))
    throw std::invalid_argument("external shuffle requires an existing scratch parent");
  auto pattern = (parent / ".external-adjacency-XXXXXX").string();
  if (!::mkdtemp(pattern.data()))
    Error("mkdtemp shuffle adjacency");
  path_ = pattern;
}
ScratchDirectory::~ScratchDirectory() {
  std::error_code error;
  std::filesystem::remove_all(path_, error);
}
ScratchFile::ScratchFile(const std::filesystem::path &path, ScratchIo &io) : path_(path), io_(io) {
  // No buffered fallback: scratch data I/O must use aligned native libaio.
  fd_ = ::open(path.c_str(), O_CREAT | O_EXCL | O_RDWR | O_DIRECT | O_CLOEXEC, 0600);
  if (fd_ < 0)
    Error("create direct shuffle scratch");
}
ScratchFile::~ScratchFile() {
  if (fd_ >= 0)
    ::close(fd_);
  ::unlink(path_.c_str());
}
void ScratchFile::Resize(uint64_t bytes) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (sealed_)
    throw std::logic_error("resize sealed scratch");
  End(0, bytes);
  if (::ftruncate(fd_, ScratchAlignedSize(bytes)))
    Error("resize shuffle scratch");
  logical_size_ = bytes;
}
void ScratchFile::Write(uint64_t offset, const void *data, size_t bytes) {
  End(offset, bytes);
  if (!bytes)
    return;
  // Logical CSR chunks can share a boundary sector. Serialize each file's RMW
  // transaction so concurrent fill workers never overwrite a neighbor's bytes.
  std::lock_guard<std::mutex> lock(mutex_);
  if (sealed_)
    throw std::logic_error("write sealed scratch");
  auto &engine = ThreadScratchAio();
  const size_t original_bytes = bytes;
  const auto *source = static_cast<const uint8_t *>(data);
  while (bytes) {
    const size_t shift = offset % kScratchAlignment;
    const uint64_t aligned_offset = offset - shift;
    const size_t count = std::min(bytes, kScratchAioBufferBytes - shift);
    const size_t aligned_bytes = ScratchAlignedSize(shift + count);
    auto *buffer = engine.Buffer();
    std::memset(buffer, 0, aligned_bytes);
    std::vector<ScratchAioRequest> requests;
    auto preserve = [&](size_t page) {
      const uint64_t position = aligned_offset + page;
      if (position < logical_size_)
        requests.push_back({fd_, buffer + page, kScratchAlignment, position,
                            size_t(std::min<uint64_t>(kScratchAlignment, logical_size_ - position)),
                            false});
    };
    if (shift)
      preserve(0);
    if ((shift + count) % kScratchAlignment && (!shift || aligned_bytes > kScratchAlignment))
      preserve(aligned_bytes - kScratchAlignment);
    engine.Run(requests, io_);
    std::memcpy(buffer + shift, source, count);
    requests.clear();
    for (size_t pos = 0; pos < aligned_bytes; pos += kScratchAioChunkBytes) {
      const size_t size = std::min(kScratchAioChunkBytes, aligned_bytes - pos);
      requests.push_back({fd_, buffer + pos, size, aligned_offset + pos, size, true});
    }
    engine.Run(requests, io_);
    logical_size_ = std::max(logical_size_, offset + count);
    offset += count;
    source += count;
    bytes -= count;
  }
  ++io_.write_calls;
  io_.write_bytes += original_bytes;
}
int ScratchFile::Reader() const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (fd_ < 0) {
    fd_ = ::open(path_.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
    if (fd_ < 0)
      Error("open direct shuffle scratch reader");
  }
  return fd_;
}
void ScratchFile::Read(uint64_t offset, void *data, size_t bytes) const {
  ReadMany({{offset, data, bytes}});
}
void ScratchFile::ReadMany(const std::vector<ScratchRead> &reads) const {
  for (const auto &r : reads)
    if (End(r.offset, r.bytes) > logical_size_)
      throw std::runtime_error("truncated shuffle scratch payload");
  auto &engine = ThreadScratchAio();
  const int fd = Reader();
  struct Copy {
    uint8_t *destination;
    size_t position, shift, bytes;
  };
  std::vector<ScratchAioRequest> requests;
  std::vector<Copy> copies;
  size_t used = 0;
  auto flush = [&] {
    engine.Run(requests, io_);
    for (const auto &c : copies)
      std::memcpy(c.destination, engine.Buffer() + c.position + c.shift, c.bytes);
    requests.clear();
    copies.clear();
    used = 0;
  };
  uint64_t logical_bytes = 0, logical_calls = 0;
  for (const auto &r : reads) {
    if (!r.bytes)
      continue;
    ++logical_calls;
    logical_bytes += r.bytes;
    size_t done = 0;
    while (done < r.bytes) {
      const uint64_t offset = r.offset + done;
      const size_t shift = offset % kScratchAlignment;
      const size_t count = std::min(r.bytes - done, kScratchAioChunkBytes - shift);
      const size_t physical = ScratchAlignedSize(shift + count);
      if (requests.size() == kScratchAioDepth || used + physical > kScratchAioBufferBytes)
        flush();
      requests.push_back(
          {fd, engine.Buffer() + used, physical, offset - shift, shift + count, false});
      copies.push_back({static_cast<uint8_t *>(r.data) + done, used, shift, count});
      used += physical;
      done += count;
    }
  }
  flush();
  io_.read_calls += logical_calls;
  io_.read_bytes += logical_bytes;
}
void ScratchFile::Seal() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (sealed_)
    throw std::logic_error("scratch already sealed");
  if (::ftruncate(fd_, logical_size_))
    Error("truncate scratch padding");
  if (::fdatasync(fd_))
    Error("sync shuffle scratch");
  if (::close(fd_))
    Error("close shuffle scratch");
  fd_ = -1;
  sealed_ = true;
}
uint64_t ScratchFile::Size() const {
  return logical_size_;
}
ScratchWriter::ScratchWriter(ScratchFile &file, size_t capacity) : file_(file), buffer_(capacity) {
  if (!capacity)
    throw std::invalid_argument("empty scratch write buffer");
}
void ScratchWriter::Flush() {
  if (used_) {
    file_.Write(offset_, buffer_.data(), used_);
    offset_ += used_;
    used_ = 0;
  }
}
void ScratchWriter::Append(const void *data, size_t bytes) {
  const auto *p = static_cast<const uint8_t *>(data);
  while (bytes) {
    const size_t count = std::min(bytes, buffer_.size() - used_);
    std::memcpy(buffer_.data() + used_, p, count);
    used_ += count;
    p += count;
    bytes -= count;
    if (used_ == buffer_.size())
      Flush();
  }
}
void ScratchWriter::Finish() {
  Flush();
  file_.Seal();
}
} // namespace alliance::shuffle_detail
