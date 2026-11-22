#include "storage/atomic_file.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <ostream>
#include <stdexcept>
#include <streambuf>
#include <system_error>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace alliance::storage {
namespace {

[[noreturn]] void Fail(const char* operation, const std::string& path, int error = errno) {
  throw std::system_error(error, std::generic_category(),
                          std::string(operation) + ": " + path);
}

// Destruction never flushes: a failed serializer must not write or commit more data.
class BufferedFdOutput final : public std::streambuf {
 public:
  explicit BufferedFdOutput(int fd) : fd_(fd) { setp(buffer_.data(), buffer_.data() + buffer_.size()); }
  int Error() const { return error_; }

 protected:
  int sync() override { return Flush() ? 0 : -1; }
  int_type overflow(int_type value) override {
    if (!Flush()) return traits_type::eof();
    if (!traits_type::eq_int_type(value, traits_type::eof())) {
      *pptr() = traits_type::to_char_type(value);
      pbump(1);
    }
    return traits_type::not_eof(value);
  }
  std::streamsize xsputn(const char* data, std::streamsize count) override {
    if (count <= 0) return 0;
    const size_t bytes = static_cast<size_t>(count);
    if (bytes >= buffer_.size()) {
      return Flush() && WriteAll(data, bytes) ? count : 0;
    }
    size_t remaining = bytes;
    while (remaining) {
      if (pptr() == epptr() && !Flush()) return 0;
      const size_t amount = std::min(remaining, static_cast<size_t>(epptr() - pptr()));
      std::memcpy(pptr(), data, amount);
      pbump(static_cast<int>(amount));
      data += amount;
      remaining -= amount;
    }
    return count;
  }

 private:
  bool WriteAll(const char* data, size_t bytes) {
    if (error_) return false;
    while (bytes) {
      const ssize_t written = ::write(fd_, data, bytes);
      if (written < 0 && errno == EINTR) continue;
      if (written <= 0) { error_ = written < 0 ? errno : EIO; return false; }
      data += written;
      bytes -= static_cast<size_t>(written);
    }
    return true;
  }
  bool Flush() {
    const size_t bytes = static_cast<size_t>(pptr() - pbase());
    if (!WriteAll(pbase(), bytes)) return false;
    setp(buffer_.data(), buffer_.data() + buffer_.size());
    return true;
  }
  int fd_;
  int error_{0};
  std::array<char, 8 * 1024> buffer_{};
};

struct PendingFile {
  int directory{-1};
  int file{-1};
  std::string temporary;
  ~PendingFile() {
    if (file >= 0) ::close(file);
    if (!temporary.empty()) ::unlinkat(directory, temporary.c_str(), 0);
    if (directory >= 0) ::close(directory);
  }
};

bool Sync(int fd) {
  int result;
  do { result = ::fsync(fd); } while (result < 0 && errno == EINTR);
  return result == 0;
}

}  // namespace

void WriteAtomicFile(const std::string& path,
                     const std::function<void(std::ostream&)>& serialize) {
  const std::filesystem::path destination(path);
  const std::string name = destination.filename().string();
  if (name.empty() || name == "." || name == "..")
    throw std::invalid_argument("invalid checkpoint destination: " + path);
  const std::string parent = destination.has_parent_path() ? destination.parent_path().string() : ".";
  PendingFile pending;
  pending.directory = ::open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (pending.directory < 0) Fail("cannot open checkpoint directory", path);

  struct stat original{};
  const bool exists = ::fstatat(pending.directory, name.c_str(), &original, AT_SYMLINK_NOFOLLOW) == 0;
  if (!exists && errno != ENOENT) Fail("cannot inspect checkpoint destination", path);
  if (exists && !S_ISREG(original.st_mode))
    throw std::runtime_error("checkpoint destination must be a regular file, not a symlink or directory: " + path);

  static std::atomic<unsigned long long> sequence{0};
  for (unsigned attempt = 0; attempt != 128; ++attempt) {
    std::string temporary = ".alliance-checkpoint." + std::to_string(::getpid()) + "." +
        std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)) + ".tmp";
    pending.file = ::openat(pending.directory, temporary.c_str(),
        O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, exists ? 0600 : 0666);
    if (pending.file >= 0) { pending.temporary.swap(temporary); break; }
    if (errno != EEXIST && errno != EINTR) Fail("cannot create temporary checkpoint", path);
  }
  if (pending.file < 0) Fail("cannot reserve unique temporary checkpoint", path, EEXIST);

  BufferedFdOutput buffer(pending.file);
  std::ostream output(&buffer);
  output.exceptions(std::ios::badbit | std::ios::failbit);
  try {
    serialize(output);
    output.flush();
    if (!output) throw std::runtime_error("checkpoint serializer left a failed output stream: " + path);
  } catch (...) {
    if (buffer.Error()) Fail("cannot write temporary checkpoint", path, buffer.Error());
    throw;
  }
  if (exists && ::fchmod(pending.file, original.st_mode & 07777) != 0)
    Fail("cannot preserve checkpoint permissions", path);
  if (!Sync(pending.file)) Fail("cannot sync temporary checkpoint", path);
  const int finished = pending.file;
  pending.file = -1;  // On Linux a failed close must not be retried on a reused descriptor.
  if (::close(finished) != 0) Fail("cannot close temporary checkpoint", path);
  if (::renameat(pending.directory, pending.temporary.c_str(), pending.directory, name.c_str()) != 0)
    Fail("cannot commit checkpoint", path);
  pending.temporary.clear();
  if (!Sync(pending.directory))
    Fail("checkpoint committed but directory fsync failed; new file is visible and cannot be rolled back", path);
}

}  // namespace alliance::storage
