#pragma once
#include "search/search_disk_runner.hpp"

namespace benchmark {
void PrintMemoryPhase(const char *phase, size_t process_memory_budget_bytes);
void PrintSharedMemory(const SearchIndex &index);
void PrintWorkerMemory(const alliance::MultiLevelThreadMemory &retained);
// Reporting runs after the wall timer and worker metric publication.
void PrintPipelineReport(const SearchIndex &index, const DiskSearchOptions &options,
    size_t efs, const SearchResults &gt, const SearchContexts &contexts,
    const SearchRunProfile &profile, PipelineBatch &batch);
}  // namespace benchmark
