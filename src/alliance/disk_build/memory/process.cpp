#include "disk_build/memory/process.hpp"
#if defined(__GLIBC__)
#include <malloc.h>
#endif

namespace alliance::disk_build {

MemorySnapshot CaptureMemorySnapshot() {
  MemorySnapshot sample;
  FILE *f = ::fopen("/proc/self/status", "r");
  if (f == nullptr)
    return sample;
  char line[256];
  while (::fgets(line, sizeof(line), f) != nullptr) {
    unsigned long kb = 0;
    // Both fields are reported in kB by the kernel regardless of page size.
    if (std::sscanf(line, "VmRSS: %lu kB", &kb) == 1) {
      sample.current_bytes = (size_t)kb * 1024;
    } else if (std::sscanf(line, "VmHWM: %lu kB", &kb) == 1) {
      sample.peak_bytes = (size_t)kb * 1024;
    }
  }
  ::fclose(f);
  return sample;
}

void TrimReleasedMemory() {
#if defined(__GLIBC__)
  ::malloc_trim(0);
#endif
}

void ResetPeakWindow() {
  FILE *f = ::fopen("/proc/self/clear_refs", "w");
  if (f == nullptr)
    return;
  ::fputs("5\n", f); // CLEAR_REFS_MM_HIWATER_RSS: VmHWM := VmRSS
  ::fclose(f);
}

void PrintMemorySample(FILE *output, const char *stage, size_t model_bytes, double budget_gib,
                       const MemorySnapshot &sample) {
  constexpr double kMB = 1024.0 * 1024.0;
  // The peak is phase-local when clear_refs succeeded, cumulative otherwise.
  fprintf(output, "  [RSS] %-22s peak %8.1f MB  now %8.1f MB", stage, sample.peak_bytes / kMB,
          sample.current_bytes / kMB);
  if (model_bytes > 0) {
    fprintf(output, "  model %8.1f MB  peak/model %.2fx", model_bytes / kMB,
            (double)sample.peak_bytes / (double)model_bytes);
  } else {
    fprintf(output, "  model      -              -   ");
  }
  if (budget_gib > 0.0) {
    const size_t budget = (size_t)(budget_gib * 1024.0 * 1024.0 * 1024.0);
    fprintf(output, "  budget %.2f GB %s", budget_gib,
            sample.peak_bytes <= budget ? "ok" : "EXCEEDED");
  }
}

void PrintTrimResult(FILE *output, size_t before_bytes, size_t after_bytes) {
  constexpr double kMB = 1024.0 * 1024.0;
  if (after_bytes > 0 && before_bytes > after_bytes) {
    fprintf(output, "  trim -%.1f MB -> %.1f MB", (before_bytes - after_bytes) / kMB,
            after_bytes / kMB);
  }
}

void CompleteMemoryPhase(const char *stage, size_t model_bytes, double budget_gib) {
  const MemorySnapshot sample = CaptureMemorySnapshot();
  if (sample.peak_bytes == 0)
    return;
  PrintMemorySample(stderr, stage, model_bytes, budget_gib, sample);
#if defined(__GLIBC__)
  TrimReleasedMemory();
  const MemorySnapshot after = CaptureMemorySnapshot();
  PrintTrimResult(stderr, sample.current_bytes, after.current_bytes);
#endif
  fprintf(stderr, "\n");
  ResetPeakWindow();
}

} // namespace alliance::disk_build
