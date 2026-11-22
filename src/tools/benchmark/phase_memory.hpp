#pragma once
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <fcntl.h>
#include <unistd.h>
inline uint64_t proc_memory(const char *key) {
  std::ifstream f("/proc/self/status");
  std::string s;
  const std::string prefix = std::string(key) + ":";
  while (std::getline(f, s))
    if (s.rfind(prefix, 0) == 0)
      return std::stoull(s.substr(prefix.size())) * 1024;
  throw std::runtime_error("proc memory counter unavailable");
}
// Counter resets only: no page reclamation, cache warming, process migration,
// memory budget changes, or search algorithm modifications.
class PhaseMemory {
  int fd_ = -1;
  bool separate_;
  std::string scope_prefix_;
  std::string events_path_;
  void CheckEvents() {
    std::ifstream input(events_path_);
    std::string key;
    uint64_t value;
    bool oom = false, killed = false;
    while (input >> key >> value) {
      if (key == "oom") {
        oom = true;
        if (value)
          throw std::runtime_error("cgroup OOM event");
      }
      if (key == "oom_kill") {
        killed = true;
        if (value)
          throw std::runtime_error("cgroup OOM kill event");
      }
    }
    if (!oom || !killed)
      throw std::runtime_error("missing actual cgroup memory.events counters");
  }
  uint64_t read_peak() {
    char b[64]{};
    if (lseek(fd_, 0, SEEK_SET) < 0)
      throw std::runtime_error("peak seek");
    auto n = read(fd_, b, sizeof(b) - 1);
    if (n <= 0)
      throw std::runtime_error("peak read");
    return std::stoull(b);
  }

public:
  uint64_t loading_peak_rss = 0, loading_cgroup_peak = 0, search_peak_rss = 0,
           search_cgroup_peak = 0, lifetime_peak_rss = 0;
  PhaseMemory(std::string prefix, bool separate)
      : separate_(separate), scope_prefix_(std::move(prefix)) {
    loading_peak_rss = lifetime_peak_rss = proc_memory("VmHWM");
    std::ifstream f("/proc/self/cgroup");
    std::string s, path;
    while (std::getline(f, s))
      if (s.rfind("0::", 0) == 0)
        path = "/sys/fs/cgroup" + s.substr(3) + "/memory.peak";
    if (path.empty())
      throw std::runtime_error("missing cgroup v2");
    events_path_ =
        path.substr(0, path.size() - std::string("memory.peak").size()) + "memory.events";
    CheckEvents();
    fd_ = open(path.c_str(), O_RDWR | O_CLOEXEC);
    if (fd_ < 0)
      throw std::runtime_error("memory.peak not writable");
    loading_cgroup_peak = read_peak();
  }
  ~PhaseMemory() {
    if (fd_ >= 0)
      close(fd_);
  }
  void ReleaseLoadingCap() {
    if (!separate_)
      return;
    std::ifstream f("/proc/self/cgroup");
    std::string line, group;
    while (std::getline(f, line))
      if (line.rfind("0::", 0) == 0)
        group = line.substr(3);
    const auto leaf = group.substr(group.find_last_of('/') + 1);
    if (leaf.rfind(scope_prefix_, 0) != 0 || leaf.size() < 6 ||
        leaf.substr(leaf.size() - 6) != ".scope")
      throw std::runtime_error("unexpected cgroup: refusing to change cap");
    const auto path = "/sys/fs/cgroup" + group + "/memory.max";
    std::ofstream out(path);
    out << "max\n";
    out.close();
    if (!out)
      throw std::runtime_error("cannot release loading cap");
    std::ifstream check(path);
    std::string value;
    check >> value;
    if (value != "max")
      throw std::runtime_error("search cgroup must have no memory cap");
  }
  void Begin() {
    CheckEvents();
    if (!separate_)
      return;
    lifetime_peak_rss = std::max(lifetime_peak_rss, proc_memory("VmHWM"));
    std::ofstream reset("/proc/self/clear_refs");
    reset << "5\n";
    reset.close();
    if (!reset)
      throw std::runtime_error("RSS HWM reset failed");
    if (lseek(fd_, 0, SEEK_SET) < 0 || write(fd_, "0", 1) != 1)
      throw std::runtime_error("cgroup FD peak reset failed");
  }
  void End() {
    CheckEvents();
    search_peak_rss = proc_memory("VmHWM");
    search_cgroup_peak = read_peak();
    lifetime_peak_rss = std::max(lifetime_peak_rss, search_peak_rss);
  }
};
