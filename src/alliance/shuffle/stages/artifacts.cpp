#include "shuffle/stages/artifacts.hpp"
#include <cerrno>
#include <cstring>
#include <fstream>
#include <sstream>
#include <set>
#include <stdexcept>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/wait.h>
#include <unistd.h>

namespace alliance::shuffle_detail {
namespace {
namespace fs = std::filesystem;
std::string Json(const std::string &s) {
  std::string out = "\"";
  for (unsigned char c : s) {
    if (c < 32)
      throw std::invalid_argument("control characters in artifact paths are unsupported");
    if (c == '\\' || c == '"')
      out += '\\';
    out += c;
  }
  return out + '"';
}
fs::path CheckedPath(const std::string &path) {
  Json(path);
  // Inspect before lexical normalization: link/../file must not hide a
  // symlink whose kernel resolution differs from the lexical path.
  const auto absolute = fs::absolute(path);
  fs::path current;
  for (const auto &part : absolute) {
    current /= part;
    std::error_code ec;
    const auto status = fs::symlink_status(current, ec);
    if (fs::is_symlink(status))
      throw std::invalid_argument("symlink artifact path: " + current.string());
    if (ec && ec != std::errc::no_such_file_or_directory)
      throw std::runtime_error("cannot inspect " + current.string());
    if (ec && current != absolute)
      throw std::invalid_argument("artifact parent must exist: " + current.string());
  }
  return absolute.lexically_normal();
}
struct stat Identity(const fs::path &path) {
  struct stat st{};
  if (::lstat(path.c_str(), &st) || !S_ISREG(st.st_mode) || st.st_nlink != 1)
    throw std::runtime_error("not a private regular artifact: " + path.string());
  return st;
}
bool Same(const struct stat &a, const struct stat &b) {
  return a.st_dev == b.st_dev && a.st_ino == b.st_ino && a.st_size == b.st_size &&
         a.st_mtim.tv_sec == b.st_mtim.tv_sec && a.st_mtim.tv_nsec == b.st_mtim.tv_nsec;
}
void Sync(const fs::path &path, bool directory = false) {
  const int fd =
      ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | (directory ? O_DIRECTORY : 0));
  if (fd < 0)
    throw std::runtime_error("cannot open for sync: " + path.string());
  const int rc = ::fsync(fd);
  const int error = errno;
  ::close(fd);
  if (rc)
    throw std::runtime_error("sync failed: " + path.string() + ": " + std::strerror(error));
}
int Lock(const fs::path &path, int mode) {
  int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0644);
  struct stat st{};
  if (fd < 0 || ::fstat(fd, &st) || !S_ISREG(st.st_mode) || st.st_nlink != 1 ||
      ::flock(fd, mode | LOCK_NB)) {
    if (fd >= 0)
      ::close(fd);
    throw std::runtime_error("artifact lock unavailable: " + path.string());
  }
  return fd;
}
std::string Sha256(const fs::path &path) {
  int pipefd[2];
  if (::pipe2(pipefd, O_CLOEXEC))
    throw std::runtime_error("sha256 pipe failed");
  const pid_t child = ::fork();
  if (child == 0) {
    ::dup2(pipefd[1], STDOUT_FILENO);
    ::close(pipefd[0]);
    ::close(pipefd[1]);
    ::execl("/usr/bin/sha256sum", "sha256sum", "--zero", "--", path.c_str(),
            static_cast<char *>(nullptr));
    ::_exit(127);
  }
  ::close(pipefd[1]);
  if (child < 0) {
    ::close(pipefd[0]);
    throw std::runtime_error("sha256 fork failed");
  }
  std::string output;
  char buffer[4096];
  bool read_ok = true;
  while (true) {
    const auto n = ::read(pipefd[0], buffer, sizeof(buffer));
    if (n < 0 && errno == EINTR)
      continue;
    if (n < 0) {
      read_ok = false;
      break;
    }
    if (!n)
      break;
    output.append(buffer, n);
  }
  ::close(pipefd[0]);
  int status = 0;
  while (::waitpid(child, &status, 0) < 0)
    if (errno != EINTR)
      throw std::runtime_error("sha256 wait failed");
  if (!read_ok || !WIFEXITED(status) || WEXITSTATUS(status) || output.size() < 66 ||
      output.substr(0, 64).find_first_not_of("0123456789abcdef") != std::string::npos)
    throw std::runtime_error("sha256sum failed for " + path.string());
  return output.substr(0, 64);
}
void Write(const fs::path &path, const std::string &data) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream << data;
  stream.flush();
  if (!stream)
    throw std::runtime_error("cannot write receipt: " + path.string());
  stream.close();
  if (!stream)
    throw std::runtime_error("cannot close receipt: " + path.string());
  Sync(path);
}
} // namespace

ShuffleArtifacts::ShuffleArtifacts(const std::string &input_graph, const std::string &input_meta,
                                   const std::string &output_graph,
                                   const std::string &output_meta) {
  try {
    const auto graph = CheckedPath(output_graph), meta = CheckedPath(output_meta);
    directory_ = graph.parent_path();
    if (meta.parent_path() != directory_)
      throw std::invalid_argument("shuffle outputs must share one directory");
    receipt_ = CheckedPath(output_graph + ".completion.json");
    const auto sidecar = CheckedPath(output_graph + ".layout");
    std::set<fs::path> outputs{graph, meta, sidecar, receipt_};
    if (outputs.size() != 4)
      throw std::invalid_argument("shuffle output paths overlap");
    for (const auto &name : {input_graph, input_meta}) {
      const auto path = CheckedPath(name);
      if (outputs.count(path))
        throw std::invalid_argument("shuffle input/output paths overlap");
      inputs_.emplace_back(path, Identity(path));
    }
    source_lock_ = Lock(inputs_[0].first.string() + ".build.lock", LOCK_SH);
    for (const auto &[path, expected] : inputs_)
      if (!Same(Identity(path), expected))
        throw std::runtime_error("shuffle input changed before locking");
    if (inputs_[0].first == inputs_[1].first)
      throw std::invalid_argument("graph and metadata inputs must differ");
    lock_ = Lock(directory_ / ".alliance-shuffle.lock", LOCK_EX);
    for (const auto &path : outputs) {
      struct stat st{};
      if (::lstat(path.c_str(), &st) == 0 || errno != ENOENT)
        throw std::runtime_error("shuffle output already exists; use a fresh output path: " +
                                 path.string());
    }
    std::string pattern = (directory_ / ".alliance-shuffle-XXXXXX").string();
    std::vector<char> temp(pattern.begin(), pattern.end());
    temp.push_back(0);
    if (!::mkdtemp(temp.data()))
      throw std::runtime_error("cannot create shuffle staging directory");
    staging_ = temp.data();
    files_ = {
        {staging_ / "graph", graph}, {staging_ / "layout", sidecar}, {staging_ / "metadata", meta}};
    Write(staging_ / "session.json",
          "{\"status\":\"incomplete\",\"pid\":" + std::to_string(::getpid()) +
              ",\"output_graph\":" + Json(graph.string()) +
              ",\"receipt\":" + Json(receipt_.string()) + "}\n");
    Sync(staging_, true);
    Sync(directory_, true);
  } catch (...) {
    Cleanup();
    throw;
  }
}
ShuffleArtifacts::~ShuffleArtifacts() {
  Cleanup();
}
std::string ShuffleArtifacts::InputGraph() const {
  return inputs_[0].first.string();
}
std::string ShuffleArtifacts::InputMetadata() const {
  return inputs_[1].first.string();
}
std::string ShuffleArtifacts::Graph() const {
  return files_[0].temporary.string();
}
std::string ShuffleArtifacts::Layout() const {
  return files_[1].temporary.string();
}
std::string ShuffleArtifacts::Metadata() const {
  return files_[2].temporary.string();
}
void ShuffleArtifacts::Fail(const std::string &reason) noexcept {
  if (staging_.empty() || complete_)
    return;
  try {
    Write(staging_ / "failure.txt", reason + "\n");
  } catch (...) {
  }
}

void ShuffleArtifacts::Commit(size_t nodes, size_t dimensions, size_t levels, uint64_t version,
                              size_t neighbor_page, size_t vector_page) {
  for (const auto &[path, expected] : inputs_)
    if (!Same(Identity(path), expected))
      throw std::runtime_error("shuffle input changed: " + path.string());
  std::ostringstream json;
  json << "{\"status\":\"complete\",\"protocol_version\":1,\"ml_only\":true,\"nodes\":" << nodes
       << ",\"dimensions\":" << dimensions << ",\"levels\":" << levels
       << ",\"layout_version\":" << version << ",\"neighbor_page_bytes\":" << neighbor_page
       << ",\"vector_page_bytes\":" << vector_page << ",\"files\":[";
  bool first = true;
  for (auto &file : files_) {
    Sync(file.temporary);
    file.identity = Identity(file.temporary);
    const auto hash = Sha256(file.temporary);
    if (file.temporary.filename() == "metadata" && hash != Sha256(inputs_[1].first))
      throw std::runtime_error("copied metadata differs from source");
    if (!Same(file.identity, Identity(file.temporary)))
      throw std::runtime_error("output changed while hashing");
    if (!first)
      json << ',';
    first = false;
    json << "{\"path\":" << Json(file.final.string()) << ",\"bytes\":" << file.identity.st_size
         << ",\"sha256\":" << Json(hash) << '}';
  }
  json << "]}\n";
  const auto temporary_receipt = staging_ / "completion.json";
  Write(temporary_receipt, json.str());
  // link is atomic and refuses every existing destination. Partial sets from a
  // killed process have no receipt and must not be consumed or overwritten.
  for (auto &file : files_) {
    if (::link(file.temporary.c_str(), file.final.c_str()))
      throw std::runtime_error("cannot publish: " + file.final.string());
    file.published = true;
  }
  Sync(directory_, true);
  if (::link(temporary_receipt.c_str(), receipt_.c_str()))
    throw std::runtime_error("cannot publish completion receipt");
  // After the receipt is visible, never remove its outputs even if directory
  // fsync fails: report uncertain durability, preserving the complete set.
  complete_ = true;
  Sync(directory_, true);
}

void ShuffleArtifacts::Cleanup() noexcept {
  if (!complete_)
    for (const auto &file : files_)
      if (file.published) {
        struct stat st{};
        if (::lstat(file.final.c_str(), &st) == 0 && st.st_dev == file.identity.st_dev &&
            st.st_ino == file.identity.st_ino)
          ::unlink(file.final.c_str());
      }
  if (!staging_.empty()) {
    // Only fixed names owned by this session are removed; no wildcard cleanup.
    for (const char *name : {"graph", "layout", "metadata", "completion.json"})
      ::unlink((staging_ / name).c_str());
    if (complete_) {
      ::unlink((staging_ / "session.json").c_str());
      ::unlink((staging_ / "failure.txt").c_str());
      ::rmdir(staging_.c_str());
    }
  }
  if (lock_ >= 0) {
    ::close(lock_);
    lock_ = -1;
  }
  if (source_lock_ >= 0) {
    ::close(source_lock_);
    source_lock_ = -1;
  }
}
} // namespace alliance::shuffle_detail
