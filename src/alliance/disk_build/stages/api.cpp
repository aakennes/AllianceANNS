#include "index/index_disk.hpp"
#include "disk_build/stages/execute.hpp"

namespace alliance {
template <typename att_t, typename vec_t>
void AllianceIndexDisk<att_t, vec_t>::Build(const vec_t *base, const att_t *attributes, size_t n,
                                            int threads) {
  RequireRawGraph("Build");
  build_config_.merge_build_threads_ = threads;
  PopulateElements(base, attributes, n);
  BuildPostPopulate(threads);
}

// Library and CLI callers share the same LeftStream/Vamana build lifecycle.
template <typename att_t, typename vec_t>
void AllianceIndexDisk<att_t, vec_t>::BuildPostPopulate(int threads, size_t start_layer,
                                                        const std::string &checkpoint_meta_path) {
  disk_build::ExecuteBuild(*this, {threads, start_layer, checkpoint_meta_path});
}

template class AllianceIndexDisk<int, float>;
} // namespace alliance
