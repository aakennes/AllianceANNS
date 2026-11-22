#pragma once
// Test-only batch-read fixture. It deliberately has no graph-building tail.
#include "io/io_handler.hpp"
#include "storage/page_constants.hpp"
#include "io/disk_resource_guard.hpp"
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace alliance::test_support {

// The caller owns io so the test can inspect outstanding requests before the
// handler's destructor. The hook observes the actual buffer release boundary.
template <typename BeforeRelease>
void ReadRecordBatch(IOHandler &io, uint64_t offset, size_t record_bytes, size_t count,
                     BeforeRelease before_release) {
  static_assert(std::is_nothrow_invocable_v<BeforeRelease>);
  if (!count || count > 64 || !record_bytes || record_bytes % kMinNodePageBytes ||
      count > std::numeric_limits<size_t>::max() / record_bytes)
    throw std::invalid_argument("invalid test record batch geometry");
  auto *buffer = static_cast<char *>(std::aligned_alloc(kMinNodePageBytes, count * record_bytes));
  if (!buffer)
    throw std::bad_alloc();
  detail::ScopeExit buffer_owner([&]() noexcept {
    before_release();
    std::free(buffer);
  });
  std::vector<IORequest> requests(count);
  std::vector<IORequest *> pointers(count);
  IOCompletionGuard completion(io);
  for (size_t i = 0; i < count; ++i) {
    requests[i].offset = offset + i * record_bytes;
    requests[i].len = record_bytes;
    requests[i].buf = buffer + i * record_bytes;
    requests[i].finished = false;
    pointers[i] = &requests[i];
  }
  io.SubmitReadBatch(pointers);
  for (;;) {
    bool finished = true;
    for (const auto &request : requests)
      finished = finished && request.finished;
    if (finished)
      return;
    io.PollWaitBatch();
  }
}

} // namespace alliance::test_support
