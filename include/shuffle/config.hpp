#pragma once
#include "core/types.hpp"
#include <cstddef>
#include <vector>
#include <string>

namespace alliance {

// Preserve the historical numeric values in diagnostic/test artifacts.
enum class ShuffleMethod { Identity = 0, BlockBNF = 5 };
inline const char *ShuffleMethodName(ShuffleMethod method) {
  switch (method) {
  case ShuffleMethod::Identity:
    return "Identity";
  case ShuffleMethod::BlockBNF:
    return "BlockBNF";
  }
  return "Unknown";
}

enum class ShuffleStorage { Memory, External, Auto };
inline const char *ShuffleStorageName(ShuffleStorage storage) {
  switch (storage) {
  case ShuffleStorage::Memory:
    return "memory";
  case ShuffleStorage::External:
    return "external";
  case ShuffleStorage::Auto:
    return "auto";
  }
  return "invalid";
}

struct BNFParams {
  size_t block_size{4096};
  size_t max_iterations{32};
  double convergence_threshold{0.001};
  // Immutable-map scoring is parallel; capacity commits retain node order.
  int threads{1};
  size_t affinity_batch_size{16384};
  // External mode changes adjacency residency only, never ordering policy.
  ShuffleStorage storage{ShuffleStorage::Memory};
  size_t memory_limit_bytes{
      0}; // Selection/admission budget; 0 is allowed only for explicit memory.
  std::string scratch_directory; // Set by the artifact session, not an arbitrary temp root.
};

struct MultiLevelConfig {
  // Each level owns k selected layers and stores up to m lower overlap layers.
  size_t level_k{2};
  size_t level_m{2};
  // Select a top-aligned layer grid with stride log2(shuffle_o).
  size_t shuffle_o{2};
  std::vector<BNFParams> per_level_params;
  // Optional diagnostics never participate in selecting the permutation.
  bool collect_diagnostics{false};
};

} // namespace alliance
