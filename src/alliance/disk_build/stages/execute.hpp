#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

namespace alliance {
template <typename att_t, typename vec_t> class AllianceIndexDisk;
namespace disk_build {

struct BuildInvocation {
  int threads{1};
  size_t start_layer{1};
  std::string checkpoint_meta_path;
};

// A view of persisted progress, distinct from tuning allowed to change on resume.
// The canonical fields remain on the index to keep the metadata payload unchanged.
struct BuildProgress {
  size_t completed_layer;
  bool left_stream_complete;
  size_t left_stream_max;
};

template <typename att_t, typename vec_t>
BuildProgress CaptureBuildProgress(const AllianceIndexDisk<att_t, vec_t> &index) {
  return {index.last_completed_layer_, index.left_stream_complete_, index.left_stream_max_};
}

inline size_t ResumeStartLayer(const BuildProgress &progress, bool left_stream) {
  if (!left_stream)
    throw std::runtime_error("legacy resident build is retired; checkpoint is load/search-only");
  if (!progress.left_stream_complete)
    throw std::runtime_error("incomplete left_stream checkpoint: restart from base vectors");
  return progress.completed_layer + 1;
}

template <typename att_t, typename vec_t>
void ExecuteBuild(AllianceIndexDisk<att_t, vec_t> &index, const BuildInvocation &request);

} // namespace disk_build
} // namespace alliance
