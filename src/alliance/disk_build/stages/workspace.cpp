#include "disk_build/partition/membership.hpp"
#include "index/index_disk.hpp"

namespace alliance::disk_build {

template <typename att_t, typename vec_t> BuildState<att_t, vec_t>::BuildState() = default;

template <typename att_t, typename vec_t> BuildState<att_t, vec_t>::~BuildState() {
  Reset();
}

template <typename att_t, typename vec_t> void BuildState<att_t, vec_t>::Reset() noexcept {
  cluster.reset();
}

template class BuildState<int, float>;

} // namespace alliance::disk_build

namespace alliance {

template <typename att_t, typename vec_t>
auto AllianceIndexDisk<att_t, vec_t>::BuildWorkspace() -> disk_build::BuildState<att_t, vec_t> & {
  if (!build_state_)
    build_state_ = std::make_unique<disk_build::BuildState<att_t, vec_t>>();
  return *build_state_;
}

template <typename att_t, typename vec_t>
void AllianceIndexDisk<att_t, vec_t>::TeardownClusterBuildDisk() {
  if (build_state_)
    build_state_->cluster.reset();
}

template auto AllianceIndexDisk<int, float>::BuildWorkspace()
    -> disk_build::BuildState<int, float> &;
template void AllianceIndexDisk<int, float>::TeardownClusterBuildDisk();

} // namespace alliance
