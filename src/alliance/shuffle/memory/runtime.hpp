#pragma once
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace alliance::shuffle_detail {
inline size_t ShuffleResidentBytes() {
  std::ifstream input("/proc/self/statm");
  size_t virtual_pages = 0, resident_pages = 0;
  const long page = ::sysconf(_SC_PAGESIZE);
  if (!(input >> virtual_pages >> resident_pages) || page <= 0 ||
      resident_pages > std::numeric_limits<size_t>::max() / size_t(page))
    throw std::runtime_error("cannot observe shuffle RSS");
  return resident_pages * page;
}
// A requested planning budget cannot override a smaller inherited cgroup-v2
// cap. This only reads limits; it never installs/raises a limit or changes swap.
inline size_t EffectiveShuffleBudget(size_t requested) {
  if (!requested)
    return 0;
  std::ifstream membership("/proc/self/cgroup");
  std::string line;
  while (std::getline(membership, line)) {
    if (line.rfind("0::", 0) != 0)
      continue;
    const std::filesystem::path root("/sys/fs/cgroup");
    auto path = root / std::filesystem::path(line.substr(3)).relative_path();
    for (;;) {
      std::ifstream file(path / "memory.max");
      std::string value;
      if (file >> value && value != "max") {
        size_t consumed = 0;
        const auto limit = std::stoull(value, &consumed);
        if (consumed != value.size())
          throw std::runtime_error("invalid cgroup memory.max");
        requested = std::min<uint64_t>(requested, limit);
      }
      if (path == root || path == path.parent_path())
        break;
      path = path.parent_path();
    }
    break;
  }
  return requested;
}
} // namespace alliance::shuffle_detail
