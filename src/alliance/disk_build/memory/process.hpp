#pragma once

#include <cstddef>
#include <cstdio>

namespace alliance::disk_build {

struct MemorySnapshot {
  size_t current_bytes{0};
  size_t peak_bytes{0};
};

// Missing /proc fields remain zero; sampling does not trim or reset the peak.
MemorySnapshot CaptureMemorySnapshot();
void TrimReleasedMemory();
void ResetPeakWindow();

// Rendering consumes a captured value and has no allocator/kernel side effects.
void PrintMemorySample(FILE *output, const char *stage, size_t model_bytes, double budget_gib,
                       const MemorySnapshot &sample);
void PrintTrimResult(FILE *output, size_t before_bytes, size_t after_bytes);

// Explicit phase boundary: sample, print, trim, sample, newline, reset.
// With no peak sample, preserve the former no-report/no-trim/no-reset behavior.
void CompleteMemoryPhase(const char *stage, size_t model_bytes, double budget_gib);

} // namespace alliance::disk_build
