#pragma once
#include <cstdint>
struct IOCounters {
  uint64_t requests=0, completions=0, requested_bytes=0, completed_bytes=0, graph_requests=0, vector_requests=0;
  uint64_t errors=0, unknown_fd=0, bad_layout=0, pending=0;
};
void IOStart(const char* ml_path, uint64_t vector_base);
IOCounters IOStop();
