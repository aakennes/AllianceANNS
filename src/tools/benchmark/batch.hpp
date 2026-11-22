#pragma once
#include "benchmark/session.hpp"
#include "benchmark/phase_memory.hpp"
namespace benchmark::formal {
// Non-owning view: main owns the index, context and phase-memory guard in that
// order. The observer is started/stopped within each measured batch.
struct FormalBatchSession {
  Index &index;
  alliance::DiskSearchContext &context;
  const DiskSearchConfig &options;
  const Protocol &protocol;
  const Geometry &geometry;
  const FormalInputs &inputs;
  PhaseMemory &phase_memory;
  uint64_t fingerprint;
};
void RunFormalBatches(FormalBatchSession &session, const std::vector<size_t> &efs);
} // namespace benchmark::formal
