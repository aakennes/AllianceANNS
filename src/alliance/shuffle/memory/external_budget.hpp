#pragma once
#include <cstddef>
#include <cstdio>
#include <stdexcept>

namespace alliance::shuffle_detail {
struct ExternalBudget {
  size_t sort_bytes{32 << 20};
  size_t batch_bytes{16 << 20}; // Separate caps for outgoing and incoming payload.
  size_t fan_in{32};

  // Buffer policy only. Shared memory/auto/external admission is in storage_plan.hpp.
};
} // namespace alliance::shuffle_detail
