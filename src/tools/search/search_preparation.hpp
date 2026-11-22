#pragma once
#include "search/search_config.hpp"
#include "search/search_io_budget.hpp"

namespace alliance {
template <typename Attribute, typename Vector> class AllianceIndexDisk;
}
namespace benchmark {
using PreparedSearchIndex = alliance::AllianceIndexDisk<int, float>;

// Policy mapping is shared, while each driver retains context/workspace ownership.
void ConfigureSearchPolicy(alliance::DiskSearchPolicy &policy, const DiskSearchConfig &config,
                           const alliance::WorkerSearchIOBudget &io);
void PrepareSearchCodes(PreparedSearchIndex &index, const DiskSearchConfig &config);
void LoadSearchLayout(PreparedSearchIndex &index, const DiskSearchConfig &config);
// Called after layout validation, before any timed query batch.
void LoadNavigationForSearch(PreparedSearchIndex &index, const DiskSearchConfig &config);
const char *SearchEntryMode(const DiskSearchConfig &config);
void PrintSearchEntryConfiguration(const PreparedSearchIndex &index, const DiskSearchConfig &config);

// Ordinary CLI auto-sizing historically requires per-query clearing. Formal
// replay preserves its explicitly frozen auto-size flag independently.
enum class CachePreparation { Ordinary, Formal };
void ConfigureSearchCache(PreparedSearchIndex &index, const DiskSearchConfig &config,
                          CachePreparation protocol);

// Global/range entries do not allocate the full centroid-to-rank directory.
template <class Index>
void PrepareGuidedEntriesForSearch(Index &index, const DiskSearchConfig &config) {
  if (config.navigation_file.empty() && config.ml_pipeline_mode &&
      config.entry_sampling == "guided" && config.entry_samples &&
      config.entry_count)
    index.PrepareGuidedEntries();
}
} // namespace benchmark
