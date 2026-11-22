#pragma once
#include "search/memory_navigation.hpp"
#include <filesystem>

namespace test_support {
// Emit the historical NVGRAPH1 byte layout independently of production Save.
// This remains a reader compatibility test if the production writer changes.
inline void WriteLegacyNavigation(const std::filesystem::path &path, const alliance::MemoryNavigationGraph &graph) {
  std::vector<uint8_t> bytes;
  auto append = [&](const auto &value) {
    const auto *begin = reinterpret_cast<const uint8_t *>(&value);
    bytes.insert(bytes.end(), begin, begin + sizeof(value));
  };
  auto word = [&](uint64_t value) { append(value); };
  word(0x314850415247564EULL);
  word(1);
  for (auto value : {graph.population, graph.dimension, graph.order_hash, graph.metric})
    word(value);
  word(graph.degree);
  word(graph.windows.size());
  word(graph.groups.size());
  for (auto window : graph.windows)
    word(window);
  for (const auto &group : graph.groups) {
    word(group.first);
    word(group.last);
    word(group.ids.size());
    for (const auto *values : {&group.ranks, &group.ids, &group.edges})
      for (auto value : *values)
        append(value);
  }
  uint64_t checksum = 14695981039346656037ULL;
  for (auto byte : bytes) {
    checksum ^= byte;
    checksum *= 1099511628211ULL;
  }
  word(checksum);
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
  if (!output)
    throw std::runtime_error("cannot write legacy navigation fixture");
}

} // namespace test_support
