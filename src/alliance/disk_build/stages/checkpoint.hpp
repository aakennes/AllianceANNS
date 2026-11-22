#pragma once

#include "disk_build/stages/execute.hpp"
#include "core/types.hpp"

namespace alliance::disk_build {

// All persisted progress transitions live here. Callers must finish graph I/O
// and its durability checks before publishing completion. Save atomically
// commits metadata; an empty path changes in-memory progress only.
template <typename att_t, typename vec_t>
void PersistBuildCheckpoint(const AllianceIndexDisk<att_t, vec_t> &index, const std::string &path) {
  if (!path.empty())
    index.Save(path);
}

template <typename att_t, typename vec_t>
void BeginResidentCheckpoint(AllianceIndexDisk<att_t, vec_t> &index, layer_t maximum,
                             const std::string &path) {
  index.left_stream_max_ = maximum;
  PersistBuildCheckpoint(index, path);
}

// The direct BuildLeftStreamLayers API historically publishes record-format
// and resident completion flags without advancing the staged resume cursor.
template <typename att_t, typename vec_t>
void PublishResidentRecords(AllianceIndexDisk<att_t, vec_t> &index, layer_t maximum) {
  index.id_only_start_layer_ = 0;
  index.left_stream_complete_ = true;
  index.left_stream_max_ = maximum;
}

template <typename att_t, typename vec_t>
void CommitResidentCheckpoint(AllianceIndexDisk<att_t, vec_t> &index, layer_t maximum, size_t n,
                              const std::string &path) {
  if (n != 0)
    PublishResidentRecords(index, maximum);
  index.last_completed_layer_ = maximum;
  PersistBuildCheckpoint(index, path);
}

template <typename att_t, typename vec_t>
void CommitUpperCheckpoint(AllianceIndexDisk<att_t, vec_t> &index, const std::string &path) {
  index.last_completed_layer_ = index.wp_;
  PersistBuildCheckpoint(index, path);
}

} // namespace alliance::disk_build
