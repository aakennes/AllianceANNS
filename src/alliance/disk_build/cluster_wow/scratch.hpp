#pragma once
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_set>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace alliance::disk_build {

// Only this private generation owns these sequential inputs/spills. A durable
// manifest identifies crash leftovers by process start time and file identities.
// These are disposable intermediates, never checkpoint/recovery dependencies.
class ClusterScratchDirectory {
public:
  ClusterScratchDirectory(const std::string &graph, const uint32_t *offsets, size_t clusters) {
    std::string pattern = graph + ".cluster-scratch.XXXXXX";
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back(0);
    if (!::mkdtemp(writable.data()))
      throw std::runtime_error("cannot create cluster scratch directory");
    path_ = writable.data();
    try {
      struct stat source{};
      if (::stat(graph.c_str(), &source))
        throw std::runtime_error("cannot identify source graph");
      input_ = Reserve("input.records");
      files_by_cluster_.resize(clusters, SIZE_MAX);
      for (size_t c = 0; c < clusters; ++c)
        if (offsets[c + 1] != offsets[c])
          files_by_cluster_[c] = Reserve("cluster." + std::to_string(c));
      std::ifstream boot("/proc/sys/kernel/random/boot_id");
      std::string boot_id;
      std::getline(boot, boot_id);
      std::ifstream proc("/proc/self/stat");
      std::string process;
      std::getline(proc, process);
      const auto end = process.rfind(')');
      if (!boot || end == std::string::npos)
        throw std::runtime_error("cannot identify scratch owner");
      std::istringstream fields(process.substr(end + 2));
      std::string token, start_ticks;
      for (size_t field = 3; field <= 22; ++field) {
        if (!(fields >> token))
          throw std::runtime_error("cannot read process start time");
        if (field == 22)
          start_ticks = token;
      }
      const std::string manifest = path_ + "/manifest.json";
      std::ofstream output(manifest, std::ios::binary | std::ios::trunc);
      output << "{\"version\":1,\"kind\":\"alliance-cluster-spill\",\"pid\":" << ::getpid()
             << ",\"boot_id\":" << Quote(boot_id) << ",\"start_ticks\":" << Quote(start_ticks)
             << ",\"graph\":" << Quote(std::filesystem::absolute(graph).lexically_normal().string())
             << ",\"graph_dev\":" << source.st_dev << ",\"graph_ino\":" << source.st_ino
             << ",\"files\":[";
      for (size_t i = 0; i < files_.size(); ++i) {
        if (i)
          output << ',';
        output << "{\"name\":" << Quote(std::filesystem::path(files_[i].path).filename().string())
               << ",\"dev\":" << files_[i].dev << ",\"ino\":" << files_[i].ino
               << ",\"allow_absent\":true}";
      }
      output << "]}\n";
      output.flush();
      if (!output)
        throw std::runtime_error("cannot write cluster scratch manifest");
      output.close();
      struct stat recorded{};
      if (::lstat(manifest.c_str(), &recorded) || !S_ISREG(recorded.st_mode))
        throw std::runtime_error("cannot identify scratch manifest");
      manifest_ = {manifest, recorded.st_dev, recorded.st_ino};
      Sync(manifest, false);
      Sync(path_, true);
      const auto parent = std::filesystem::path(path_).parent_path();
      Sync(parent.empty() ? "." : parent.string(), true);
      fprintf(stderr, "[ClusterScratch] directory=%s manifest=manifest.json\n", path_.c_str());
    } catch (...) {
      CleanupNoThrow();
      throw;
    }
  }
  ~ClusterScratchDirectory() {
    CleanupNoThrow();
  }
  ClusterScratchDirectory(const ClusterScratchDirectory &) = delete;
  ClusterScratchDirectory &operator=(const ClusterScratchDirectory &) = delete;

  const std::string &InputFile() const {
    return files_.at(input_).path;
  }
  const std::string &File(size_t cluster) const {
    return files_.at(files_by_cluster_.at(cluster)).path;
  }
  const std::string &Path() const {
    return path_;
  }
  // Call only after all readers of the file have drained and closed.
  void ReleaseInput() {
    Release(files_.at(input_));
  }
  void ReleaseSpill(size_t cluster) {
    Release(files_.at(files_by_cluster_.at(cluster)));
  }
  void Sample() {
    for (auto &file : files_)
      Refresh(file);
    Report();
  }
  // Spill files grow one at a time. Refresh only the just-closed writer so
  // allocation evidence costs O(K) stats overall, not O(K squared).
  void SampleSpill(size_t cluster) {
    Refresh(files_.at(files_by_cluster_.at(cluster)));
    Report();
  }
  // After graph fsync, dispose remaining metadata; no build resume state changes.
  void Complete() {
    Sample();
    Cleanup();
  }

private:
  struct Owned {
    std::string path;
    dev_t dev{};
    ino_t ino{};
    uint64_t logical{}, allocated{};
  };
  std::string path_;
  std::vector<size_t> files_by_cluster_;
  size_t input_{};
  std::vector<Owned> files_;
  Owned manifest_{};
  uint64_t logical_bytes_{}, allocated_bytes_{}, peak_allocated_{};

  size_t Reserve(const std::string &name) {
    const auto file = path_ + "/" + name;
    const int fd = ::open(file.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
    if (fd < 0)
      throw std::runtime_error("cannot reserve owned cluster scratch file");
    struct stat st{};
    const int rc = ::fstat(fd, &st);
    ::close(fd);
    if (rc)
      throw std::runtime_error("cannot identify owned cluster scratch file");
    files_.push_back({file, st.st_dev, st.st_ino});
    return files_.size() - 1;
  }
  static void RequireIdentity(const Owned &file, const struct stat &st) {
    if (!S_ISREG(st.st_mode) || st.st_nlink != 1 || st.st_dev != file.dev || st.st_ino != file.ino)
      throw std::runtime_error("owned cluster scratch identity changed: " + file.path);
  }
  void Release(Owned &file) {
    struct stat st{};
    if (::lstat(file.path.c_str(), &st)) {
      if (errno == ENOENT)
        return;
      throw std::runtime_error("cannot inspect owned cluster scratch file");
    }
    RequireIdentity(file, st);
    if (::unlink(file.path.c_str()))
      throw std::runtime_error("cannot remove owned cluster scratch file");
    logical_bytes_ -= file.logical;
    allocated_bytes_ -= file.allocated;
    file.logical = file.allocated = 0;
    fprintf(stderr, "[ClusterScratch] released=%s remaining_allocated_bytes=%llu\n",
            std::filesystem::path(file.path).filename().c_str(),
            (unsigned long long)allocated_bytes_);
  }
  void Refresh(Owned &file) {
    struct stat st{};
    if (::lstat(file.path.c_str(), &st)) {
      if (errno != ENOENT)
        throw std::runtime_error("cannot stat owned cluster scratch file");
      logical_bytes_ -= file.logical;
      allocated_bytes_ -= file.allocated;
      file.logical = file.allocated = 0;
      return;
    }
    RequireIdentity(file, st);
    logical_bytes_ = logical_bytes_ - file.logical + st.st_size;
    allocated_bytes_ = allocated_bytes_ - file.allocated + uint64_t(st.st_blocks) * 512;
    file.logical = st.st_size;
    file.allocated = uint64_t(st.st_blocks) * 512;
  }
  void Report() {
    peak_allocated_ = std::max(peak_allocated_, allocated_bytes_);
    fprintf(stderr,
            "[ClusterScratch] logical_bytes=%llu allocated_bytes=%llu peak_allocated_bytes=%llu\n",
            (unsigned long long)logical_bytes_, (unsigned long long)allocated_bytes_,
            (unsigned long long)peak_allocated_);
  }
  static std::string Quote(const std::string &s) {
    std::string out = "\"";
    const char *hex = "0123456789abcdef";
    for (unsigned char c : s) {
      if (c == '"' || c == '\\') {
        out += '\\';
        out += char(c);
      } else if (c < 32) {
        out += "\\u00";
        out += hex[c >> 4];
        out += hex[c & 15];
      } else
        out += char(c);
    }
    return out + '"';
  }
  static void Sync(const std::string &path, bool directory) {
    int fd = ::open(path.c_str(), O_RDONLY | O_NOFOLLOW | (directory ? O_DIRECTORY : 0));
    if (fd < 0)
      throw std::runtime_error("cannot open scratch path for sync");
    const int rc = ::fsync(fd);
    ::close(fd);
    if (rc)
      throw std::runtime_error("cannot sync cluster scratch metadata");
  }
  void CleanupNoThrow() noexcept {
    try {
      Cleanup();
    } catch (...) {
      fprintf(stderr, "[ClusterScratch] cleanup incomplete; retained evidence: %s\n",
              path_.c_str());
    }
  }
  void Cleanup() {
    if (path_.empty())
      return;
    // Validate the complete directory before mutation. Unknown entries preserve
    // the manifest for manual inspection instead of broadening deletion scope.
    std::unordered_set<std::string> known_names{"manifest.json"};
    for (const auto &file : files_)
      known_names.insert(std::filesystem::path(file.path).filename().string());
    for (const auto &entry : std::filesystem::directory_iterator(path_))
      if (!known_names.count(entry.path().filename().string()))
        throw std::runtime_error("unknown file in cluster scratch generation");
    auto validate = [](const Owned &file) {
      struct stat st{};
      if (::lstat(file.path.c_str(), &st)) {
        if (errno == ENOENT)
          return;
        throw std::runtime_error("cannot inspect owned scratch cleanup candidate");
      }
      RequireIdentity(file, st);
    };
    for (const auto &file : files_)
      validate(file);
    if (!manifest_.path.empty())
      validate(manifest_);
    for (auto &file : files_)
      Release(file);
    if (!manifest_.path.empty())
      Release(manifest_);
    if (::rmdir(path_.c_str()))
      throw std::runtime_error("cannot remove cluster scratch directory");
    path_.clear();
  }
};
} // namespace alliance::disk_build
