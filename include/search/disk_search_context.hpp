#pragma once
// Worker-owned query policy and IO queues, independent of index storage/state.

#include <memory>
#include <stdexcept>
#include <string>
#include <fcntl.h>
#include <unistd.h>
#include "io/io_handler.hpp"
#include "search/disk_search_policy.hpp"
#include "search/search_worker.hpp"
#include "search/memory_navigation.hpp"
#include "storage/storage_open_mode.hpp"

namespace alliance {

struct DiskSearchContext : DiskSearchPolicy {
private:
  // Declared first so buffers outlive both IO handlers during destruction.
  std::unique_ptr<SearchWorker> workspace_;
  StorageOpenMode storage_mode_;

public:
  MemoryNavigationGraph::Scratch navigation_scratch;
  std::unique_ptr<IOHandler> io_handler;
  size_t rerank_float_reused{0}, rerank_float_recomputed{0};
  size_t rerank_float_refined{0}; // Per-worker diagnostic counter.

  // Optional second IO context for the multi-level layout file (a separate
  // physical file from graph_path). Only opened when a non-empty
  // ml_graph_path is given -- searchKNNMultiLevelPipeline requires this.
  std::unique_ptr<IOHandler> ml_io_handler;

  DiskSearchContext(size_t queue_depth, const std::string &graph_path,
                    const std::string &ml_graph_path = "",
                    StorageOpenMode storage_mode = StorageOpenMode::RequireRawGraph)
      : storage_mode_(storage_mode) {
    if (storage_mode != StorageOpenMode::RequireRawGraph &&
        storage_mode != StorageOpenMode::MultiLevelOnly)
      throw std::invalid_argument("unknown search storage mode");
    if (IsMultiLevelOnly() && ml_graph_path.empty())
      throw std::invalid_argument("ML-only search requires a shuffled layout path");
    this->queue_depth = queue_depth;
    // Synchronize buffered index writes before opening independent O_DIRECT
    // read handles, including searches immediately after build or layout export.
    if (!IsMultiLevelOnly()) {
      io_handler = std::make_unique<IOHandler>(queue_depth);
      FsyncPath(graph_path);
      io_handler->Open(graph_path);
    }
    if (!ml_graph_path.empty()) {
      FsyncPath(ml_graph_path);
      // Graph/vector completion waits use the validated busy-poll search path.
      ml_io_handler = std::make_unique<IOHandler>(queue_depth, IOHandler::CompletionWait::BusyPoll);
      ml_io_handler->Open(ml_graph_path);
    }
  }

  bool IsMultiLevelOnly() const noexcept {
    return storage_mode_ == StorageOpenMode::MultiLevelOnly;
  }

  // Opt in before the first query. Context-owned cache/visited/IO buffers can
  // move between OS threads with the context, but must not serve concurrent
  // queries. A generation change invalidates graph state between batches.
  void UseOwnedWorkspace() {
    if (!workspace_)
      workspace_ = std::make_unique<SearchWorker>();
  }
  bool HasOwnedWorkspace() const {
    return bool(workspace_);
  }
  SearchWorker &Workspace(const void *owner, uint64_t generation) {
    if (!workspace_)
      return detail::GetDiskSearchWorkspace(owner, generation);
    workspace_->Bind(owner, generation);
    return *workspace_;
  }
  MultiLevelThreadMemory WorkspaceMemory(const void *owner, uint64_t generation) {
    return Workspace(owner, generation).MemoryUsage();
  }

private:
  static void FsyncPath(const std::string &path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
      throw std::runtime_error("DiskSearchContext: failed to open for fsync: " + path);
    }
    ::fsync(fd);
    ::close(fd);
  }
};

} // namespace alliance
