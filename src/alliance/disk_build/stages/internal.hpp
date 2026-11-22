#pragma once

#include "disk_build/stages/execute.hpp"
#include "core/types.hpp"

namespace alliance::disk_build {

struct ResidentStageResult {
  layer_t maximum;
  size_t memory_model_bytes;
  bool built;
};

// The record executor establishes durability, but does not publish progress.
template <typename att_t, typename vec_t>
void ExecuteLeftStreamRecords(AllianceIndexDisk<att_t, vec_t> &index, layer_t maximum, size_t n);

template <typename att_t, typename vec_t>
ResidentStageResult RunResidentStage(AllianceIndexDisk<att_t, vec_t> &index, size_t n,
                                     size_t start_layer, const std::string &checkpoint);

template <typename att_t, typename vec_t>
void PrepareUpperMemoryBoundary(AllianceIndexDisk<att_t, vec_t> &index, size_t n,
                                const ResidentStageResult &resident);

template <typename att_t, typename vec_t>
void RunUpperStage(AllianceIndexDisk<att_t, vec_t> &index, size_t n, layer_t resident_max,
                   size_t start_layer, const std::string &checkpoint);

} // namespace alliance::disk_build
