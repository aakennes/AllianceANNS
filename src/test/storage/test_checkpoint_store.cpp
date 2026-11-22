#include "storage/atomic_file.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>

namespace fs = std::filesystem;
namespace {

void Require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

struct Probe {
  bool active{false};
  bool fail_file_sync{false}, fail_directory_sync{false}, fail_rename{false}, fail_close{false};
  bool interrupt_write{false}, short_writes{false};
  int fail_write_after{-1};
  size_t injections{0};
  std::vector<std::string> events;
} probe;

bool IsDirectory(int fd) {
  struct stat status{};
  return ::fstat(fd, &status) == 0 && S_ISDIR(status.st_mode);
}

std::string Read(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  Require(input.is_open(), "read checkpoint");
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void Write(const fs::path& path, const std::string& content) {
  std::ofstream output(path, std::ios::binary);
  output.write(content.data(), content.size());
  output.close();
  Require(bool(output), "prepare checkpoint");
}

void NoTemporaryFiles(const fs::path& directory) {
  for (const auto& entry : fs::directory_iterator(directory)) {
    Require(entry.path().filename().string().find(".alliance-checkpoint.") != 0,
            "temporary checkpoint must be removed");
  }
}

mode_t Permissions(const fs::path& path) {
  struct stat status{};
  Require(::stat(path.c_str(), &status) == 0, "stat checkpoint permissions");
  return status.st_mode & 07777;
}

size_t Position(const char* event) {
  const auto found = std::find(probe.events.begin(), probe.events.end(), event);
  Require(found != probe.events.end(), "missing commit operation");
  return static_cast<size_t>(found - probe.events.begin());
}

}  // namespace

extern "C" ssize_t __real_write(int, const void*, size_t);
extern "C" int __real_fsync(int);
extern "C" int __real_renameat(int, const char*, int, const char*);
extern "C" int __real_close(int);

extern "C" ssize_t __wrap_write(int fd, const void* bytes, size_t count) {
  if (!probe.active) return __real_write(fd, bytes, count);
  probe.events.emplace_back("write");
  if (probe.interrupt_write) {
    probe.interrupt_write = false;
    ++probe.injections;
    errno = EINTR;
    return -1;
  }
  if (probe.fail_write_after == 0) {
    probe.fail_write_after = -1;
    ++probe.injections;
    errno = ENOSPC;
    return -1;
  }
  if (probe.fail_write_after > 0) --probe.fail_write_after;
  return __real_write(fd, bytes, probe.short_writes ? std::min(count, size_t{4096}) : count);
}

extern "C" int __wrap_fsync(int fd) {
  if (!probe.active) return __real_fsync(fd);
  const bool directory = IsDirectory(fd);
  probe.events.emplace_back(directory ? "directory_sync" : "file_sync");
  if (directory ? probe.fail_directory_sync : probe.fail_file_sync) {
    ++probe.injections;
    errno = EIO;
    return -1;
  }
  return __real_fsync(fd);
}

extern "C" int __wrap_renameat(int source, const char* old_name, int destination, const char* new_name) {
  if (probe.active) {
    probe.events.emplace_back("rename");
    if (probe.fail_rename) { ++probe.injections; errno = EIO; return -1; }
  }
  return __real_renameat(source, old_name, destination, new_name);
}

extern "C" int __wrap_close(int fd) {
  if (!probe.active) return __real_close(fd);
  const bool directory = IsDirectory(fd);
  probe.events.emplace_back(directory ? "directory_close" : "file_close");
  const int result = __real_close(fd);
  if (!directory && probe.fail_close) {
    probe.fail_close = false;
    ++probe.injections;
    errno = EIO;
    return -1;
  }
  return result;
}

int main() {
  try {
    const char* temporary = std::getenv("TEST_TMPDIR");
    std::string directory = std::string(temporary ? temporary : "/tmp") + "/alliance_checkpoint_XXXXXX";
    Require(::mkdtemp(directory.data()) != nullptr, "create temporary directory");
    struct Cleanup { fs::path directory; ~Cleanup() { fs::remove_all(directory); } } cleanup{directory};
    const fs::path target = fs::path(directory) / "graph.meta";
    const std::string original("old\0checkpoint", 14);
    const std::string header("new\0header", 10);
    std::string payload(131117, 'x');
    for (size_t i = 0; i < payload.size(); ++i) payload[i] = static_cast<char>(i % 251);
    const std::string expected = header + payload + "tail";
    auto serialize = [&](std::ostream& output) {
      Require(Read(target) == original, "old checkpoint must remain visible during serialization");
      for (char value : header) output.put(value);
      output.write(payload.data(), payload.size());
      output.write("tail", 4);
    };

    // Every pre-rename failure leaves the original bytes/mode and removes the temporary file.
    for (int scenario = 0; scenario != 7; ++scenario) {
      Write(target, original);
      Require(::chmod(target.c_str(), 0640) == 0, "set original permissions");
      probe = Probe{};
      probe.active = true;
      probe.fail_write_after = scenario == 1 ? 1 : -1;
      probe.fail_file_sync = scenario == 2;
      probe.fail_close = scenario == 3;
      probe.fail_rename = scenario == 4;
      probe.fail_directory_sync = scenario == 5;
      std::string error;
      try {
        alliance::storage::WriteAtomicFile(target.string(), [&](std::ostream& output) {
          serialize(output);
          if (scenario == 0) throw std::runtime_error("serializer fixture failure");
        });
      } catch (const std::exception& failure) { error = failure.what(); }
      probe.active = false;
      Require((scenario == 6) == error.empty(), "unexpected atomic write outcome");
      if (scenario >= 1 && scenario <= 5) Require(probe.injections == 1, "fault must reach real storage operation");
      if (scenario < 5) {
        Require(Read(target) == original, "pre-commit failure damaged original checkpoint");
      } else {
        Require(Read(target) == expected, "committed payload mismatch");
        Require(Position("write") < Position("file_sync") &&
                    Position("file_sync") < Position("file_close") &&
                    Position("file_close") < Position("rename") &&
                    Position("rename") < Position("directory_sync"), "durable commit ordering");
      }
      if (scenario == 5)
        Require(error.find("committed") != std::string::npos && error.find("cannot be rolled back") != std::string::npos,
                "post-commit failure must clearly state that replacement occurred");
      Require(Permissions(target) == 0640, "preserve original checkpoint permissions");
      NoTemporaryFiles(directory);
    }

    Write(target, original);
    probe = Probe{};
    probe.active = true;
    probe.interrupt_write = true;
    probe.short_writes = true;
    alliance::storage::WriteAtomicFile(target.string(), serialize);
    probe.active = false;
    Require(probe.injections == 1 && Read(target) == expected, "short writes and EINTR must complete exactly");
    NoTemporaryFiles(directory);

    const fs::path fresh = fs::path(directory) / "fresh.meta";
    bool failed = false;
    try {
      alliance::storage::WriteAtomicFile(fresh.string(), [](std::ostream& out) {
        out << "partial";
        throw std::runtime_error("new file serializer failure");
      });
    } catch (const std::runtime_error&) { failed = true; }
    Require(failed && !fs::exists(fresh), "failed first checkpoint must not create a destination");
    NoTemporaryFiles(directory);
    const mode_t old_mask = ::umask(0027);
    struct RestoreMask { mode_t value; ~RestoreMask() { ::umask(value); } } restore_mask{old_mask};
    alliance::storage::WriteAtomicFile(fresh.string(), [](std::ostream& out) { out << "complete"; });
    Require(Read(fresh) == "complete" && Permissions(fresh) == 0640, "new checkpoint must honor umask");

    const fs::path link = fs::path(directory) / "link.meta";
    Require(::symlink(target.c_str(), link.c_str()) == 0, "prepare symlink");
    failed = false;
    try { alliance::storage::WriteAtomicFile(link.string(), [](std::ostream& out) { out << "wrong"; }); }
    catch (const std::runtime_error&) { failed = true; }
    Require(failed && fs::is_symlink(link) && Read(target) == expected, "reject symlink without changing target");
    NoTemporaryFiles(directory);
    std::cout << "PASS: atomic checkpoint commit, fault retention, directory durability, permissions and exact writes\n";
    return 0;
  } catch (const std::exception& error) {
    probe.active = false;
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
