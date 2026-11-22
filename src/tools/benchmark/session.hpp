#pragma once
#include "index/index_disk.hpp"
#include "search/search_config.hpp"
#include "benchmark/layout.hpp"
#include "benchmark/protocol.hpp"
#include <memory>

namespace benchmark::formal {
using Index = alliance::AllianceIndexDisk<int, float>;
// Inputs remain alive for every scheduled batch. Strict formal validation is
// deliberately separate from the ordinary CLI's permissive query truncation.
struct FormalInputs {
  size_t dimension = 0, count = 0;
  std::unique_ptr<float[]> vectors;
  std::vector<alliance::attr_range<int>> ranges;
  std::vector<std::vector<alliance::label_t>> ground_truth;
  std::vector<int> attributes;
};
void ValidateFormalOptions(const DiskSearchConfig &o, const Protocol &protocol);
uint64_t LoadFormalQuantization(Index &index, const DiskSearchConfig &o, const Protocol &protocol);
Geometry LoadFormalLayout(Index &index, const DiskSearchConfig &o, const Protocol &protocol);
FormalInputs LoadFormalInputs(Index &index, const DiskSearchConfig &o, const Protocol &protocol);
std::vector<size_t> FormalEFPlan(const DiskSearchConfig &o);
void CheckLoadingMemory(size_t cap);
void WriteFingerprint(const std::string &path, uint64_t fingerprint);
} // namespace benchmark::formal
