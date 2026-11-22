#include "search/disk_search_workspace.hpp"

alliance::detail::DiskSearchWorkspace *WorkspaceFromOtherTU(
    const void *owner, uint64_t generation) {
  return &alliance::detail::GetDiskSearchWorkspace(owner, generation);
}
