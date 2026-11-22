#pragma once
#include "shuffle/adjacency/scratch_aio.hpp"
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <vector>

namespace alliance::shuffle_detail {
struct ScratchRead {
  uint64_t offset;
  void *data;
  size_t bytes;
};

// Files live only below a private directory owned by the current shuffle group.
// They are disposable: exceptions remove them; original graphs are never touched.
class ScratchDirectory {
public:
  explicit ScratchDirectory(const std::filesystem::path &parent);
  ~ScratchDirectory();
  ScratchDirectory(const ScratchDirectory &) = delete;
  const std::filesystem::path &Path() const {
    return path_;
  }

private:
  std::filesystem::path path_;
};

class ScratchFile {
public:
  ScratchFile(const std::filesystem::path &path, ScratchIo &io);
  ~ScratchFile();
  ScratchFile(const ScratchFile &) = delete;
  void Resize(uint64_t bytes);
  void Write(uint64_t offset, const void *data, size_t bytes);
  void Read(uint64_t offset, void *data, size_t bytes) const;
  // Queue bounded aligned reads together; complete before caller consumes data.
  void ReadMany(const std::vector<ScratchRead> &reads) const;
  // Complete/sync writes, remove physical padding, then close the disposable file.
  void Seal();
  uint64_t Size() const;

private:
  std::filesystem::path path_;
  ScratchIo &io_;
  mutable int fd_{-1};
  mutable std::mutex mutex_;
  uint64_t logical_size_{};
  bool sealed_{};
  int Reader() const;
};

class ScratchWriter {
public:
  explicit ScratchWriter(ScratchFile &file, size_t capacity = 1 << 20);
  void Append(const void *data, size_t bytes);
  void Finish();

private:
  ScratchFile &file_;
  std::vector<uint8_t> buffer_;
  size_t used_{};
  uint64_t offset_{};
  void Flush();
};
} // namespace alliance::shuffle_detail
