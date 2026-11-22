#include "search/disk_search_workspace.hpp"
#include <array>
#include <atomic>
#include <cerrno>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>

using alliance::detail::DiskSearchWorkspace;
using alliance::detail::GetDiskSearchWorkspace;
using Policy = alliance::DynamicBlockCache::EvictPolicy;
DiskSearchWorkspace *WorkspaceFromOtherTU(const void *, uint64_t);

namespace {
std::atomic<int> fail_after{-1};
std::mutex allocation_mutex;
std::array<void *, 4096> live_buffers{};
size_t LiveBuffers() {
  std::lock_guard<std::mutex> lock(allocation_mutex);
  size_t count = 0;
  for (auto p : live_buffers) count += p != nullptr;
  return count;
}
void Check(bool pass, const char *message) {
  if (!pass) throw std::runtime_error(message);  // Active in Release builds.
}
void Aligned(char *p) {
  Check(p && reinterpret_cast<uintptr_t>(p) % 4096 == 0, "unaligned IO buffer");
}
template <typename F> void AllocationFails(F action) {
  bool failed = false;
  try { action(); } catch (const std::runtime_error &) { failed = true; }
  fail_after = -1;
  Check(failed, "expected allocation failure");
}
}  // namespace

// Linker wrapping affects only this test, not the production library interface.
extern "C" int __real_posix_memalign(void **, size_t, size_t);
extern "C" void __real_free(void *);
extern "C" int __wrap_posix_memalign(void **p, size_t alignment, size_t bytes) {
  if (fail_after.load() >= 0 && fail_after.fetch_sub(1) == 0) return ENOMEM;
  const int error = __real_posix_memalign(p, alignment, bytes);
  if (!error) {
    std::lock_guard<std::mutex> lock(allocation_mutex);
    for (auto &entry : live_buffers) if (!entry) { entry = *p; return error; }
    std::abort();
  }
  return error;
}
extern "C" void __wrap_free(void *p) {
  if (p) {
    std::lock_guard<std::mutex> lock(allocation_mutex);
    for (auto &entry : live_buffers) if (entry == p) { entry = nullptr; break; }
  }
  __real_free(p);
}

namespace {
void TestBufferOwnershipAndResize() {
  const auto initial_live = LiveBuffers();
  {
    DiskSearchWorkspace workspace;
    workspace.EnsureBlockBuf(4096);
    auto *block = workspace.BlockBuffer();
    Aligned(block); block[0] = 42;
    workspace.EnsureBlockBuf(1024);
    Check(workspace.BlockBuffer() == block, "sync buffer shrank");
    fail_after = 0;
    AllocationFails([&] { workspace.EnsureBlockBuf(8192); });
    Check(workspace.BlockBuffer() == block && block[0] == 42 &&
        workspace.MemoryUsage().io_buffer_bytes == 4096, "failed sync grow lost old storage");
    workspace.EnsureBlockBuf(8192); Aligned(workspace.BlockBuffer());

    std::vector<char *> reference;  // Old resize policy, without allocating data.
    size_t block_size = 0;
    for (auto shape : {std::pair<size_t,size_t>{8,4096}, {2,4096}, {2,8192},
                       {12,4096}, {3,16384}, {10,8192}}) {
      const bool grow = shape.first > reference.size() || shape.second > block_size;
      char *old = reference.empty() ? nullptr : workspace.PipelineBuffer(0);
      workspace.EnsurePipelineBuffers(shape.first, shape.second);
      if (grow) { reference.resize(shape.first, nullptr); block_size = shape.second; }
      else Check(workspace.PipelineBuffer(0) == old, "no-op changed published pipeline pointer");
      for (size_t i = 0; i < reference.size(); ++i) Aligned(workspace.PipelineBuffer(i));
      Check(workspace.MemoryUsage().io_buffer_bytes == 8192 +
          reference.capacity()*sizeof(char *) + reference.size()*block_size,
          "pipeline retention/accounting policy changed");
    }
    const auto bytes = workspace.MemoryUsage().io_buffer_bytes;
    const auto live = LiveBuffers();
    auto *slot = workspace.PipelineBuffer(0); slot[0] = 73;
    fail_after = 2;  // Fail after two replacement allocations, exercising unwind.
    AllocationFails([&] { workspace.EnsurePipelineBuffers(16, 16384); });
    Check(LiveBuffers() == live, "partial replacement leaked buffers");
    Check(workspace.PipelineBuffer(0) == slot && slot[0] == 73 &&
        workspace.MemoryUsage().io_buffer_bytes == bytes, "failed pool grow changed old storage");
    workspace.EnsurePipelineBuffers(10,8192);
    Check(workspace.PipelineBuffer(0) == slot, "failure corrupted cached dimensions");
    workspace.EnsurePipelineBuffers(16,16384);
    Aligned(workspace.PipelineBuffer(15));
  }
  Check(LiveBuffers() == initial_live, "workspace destructor leaked aligned buffers");
}

void TestLifecycle() {
  DiskSearchWorkspace workspace;
  int owner1, owner2;
  workspace.Bind(&owner1,1);
  workspace.PrefilterIDs().reserve(4096);
  workspace.PrefilterCandidates().reserve(50);
  workspace.EnsurePipelineBuffers(5,4096);
  const auto retained = workspace.MemoryUsage();
  auto *slot = workspace.PipelineBuffer(0);
  Check(retained.cache_payload_bytes == 0 && retained.visited_bytes == 0,
      "prefilter eagerly allocated graph state");
  workspace.EnsureGraphCache(128*8192,128,Policy::FIFO,20000,true,4096);
  Check(workspace.Cache().auto_size(), "auto cache not enabled");
  char record[128]{};
  const auto initial_capacity = workspace.Cache().capacity();
  for (size_t i=0; i<initial_capacity+32; ++i)
    workspace.Cache().InsertIfMissing(i,record,sizeof(record),0,0);
  workspace.ClearCache();
  Check(workspace.Cache().size() == 0 && workspace.Cache().capacity() > initial_capacity,
      "per-query clear lost adaptive growth");
  workspace.Cache().InsertIfMissing(7,record,sizeof(record),0,0);
  auto &visited = workspace.GetVisited(512); visited.Set(17);
  const auto bytes = workspace.MemoryUsage().visited_bytes;
  auto &again = workspace.GetVisited(128);
  Check(&again == &visited && !again.Test(17) && workspace.MemoryUsage().visited_bytes == bytes,
      "visited clear/reuse changed");
  workspace.GetVisited(1024).Set(900);
  auto *block = workspace.BlockBuffer();
  workspace.Bind(&owner1,1);
  Check(workspace.Cache().Lookup(7).found && workspace.BlockBuffer()==block,
      "same owner/generation reset warm cache");
  workspace.EnsureGraphCache(128*8192,128,Policy::FIFO,20000,false,4096);
  Check(!workspace.Cache().auto_size() && workspace.Cache().capacity() == 8192,
      "runtime auto toggle not applied");
  for (auto binding : {std::pair<const void *,uint64_t>{&owner1,2}, {&owner2,2}}) {
    workspace.Bind(binding.first,binding.second);
    const auto memory = workspace.MemoryUsage();
    Check(memory.cache_payload_bytes == 0 && memory.cache_lookup_bytes == 0 &&
        memory.cache_metadata_bytes == 0 && memory.visited_bytes == 0 && !workspace.BlockBuffer(),
        "index invalidation retained graph state");
    Check(memory.io_buffer_bytes == retained.io_buffer_bytes &&
        memory.prefilter_scratch_bytes == retained.prefilter_scratch_bytes &&
        workspace.PipelineBuffer(0) == slot, "index invalidation changed retained scratch");
    workspace.EnsureGraphCache(128*64,128,Policy::FIFO,20000,false,4096);
    workspace.GetVisited(512);
  }
  // A failed lazy initialization can be retried without losing its block buffer.
  workspace.Bind(&owner2,3);
  fail_after = 0;
  AllocationFails([&] { workspace.EnsureGraphCache(128*64,128,Policy::FIFO,20000,false,4096); });
  workspace.EnsureGraphCache(128*64,128,Policy::FIFO,20000,false,4096);
  Aligned(workspace.BlockBuffer());
}

void TestSingleTLSAcrossTranslationUnits() {
  int owner;
  auto &main = GetDiskSearchWorkspace(&owner,1);
  Check(&main == WorkspaceFromOtherTU(&owner,1), "duplicate TLS across translation units");
  const auto live = LiveBuffers();
  std::array<DiskSearchWorkspace *,2> workers{};
  std::array<std::exception_ptr,2> errors{};
  std::atomic<int> ready{0}; std::atomic<bool> release{false};
  auto run = [&](size_t i) {
    try {
      workers[i] = &GetDiskSearchWorkspace(&owner,1);
      Check(workers[i] == WorkspaceFromOtherTU(&owner,1), "worker has multiple TLS instances");
      workers[i]->EnsurePipelineBuffers(3,4096);
      workers[i]->GetVisited(512).Set(i);
    } catch (...) { errors[i] = std::current_exception(); }
    ++ready;
    while (!release.load()) std::this_thread::yield();
  };
  std::thread first(run,0), second(run,1);
  while (ready.load() != 2) std::this_thread::yield();
  const bool separate = workers[0] != workers[1] && workers[0] != &main && workers[1] != &main;
  release = true; first.join(); second.join();
  for (auto error : errors) if (error) std::rethrow_exception(error);
  Check(separate, "workers share mutable workspace");
  Check(LiveBuffers() == live, "thread exit leaked workspace buffers");
}
}  // namespace

int main() {
  static_assert(!std::is_copy_constructible_v<DiskSearchWorkspace>);
  static_assert(!std::is_move_constructible_v<DiskSearchWorkspace>);
  try {
    TestBufferOwnershipAndResize();
    TestLifecycle();
    TestSingleTLSAcrossTranslationUnits();
    std::cout << "workspace lifecycle, resize failure, accounting and cross-TU TLS passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n'; return 1;
  }
}
