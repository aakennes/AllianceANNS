// alliance-formal-v1: see docs/formal_benchmark.md for measurement provenance.
#include "benchmark/batch.hpp"
#include "benchmark/checks.hpp"
#include "search/search_disk_options.hpp"
#include "search/search_disk_checkpoint.hpp"
#include "search/search_preparation.hpp"
#include <iomanip>

int main(int argc, char **argv) {
  using namespace benchmark::formal;
  try {
    std::cout << std::setprecision(17);
    benchmark::DiskSearchOptions options;
    Require(options.ParseAndValidate(argc, argv), "arguments");
    const Protocol protocol;
    ValidateFormalOptions(options, protocol);
    omp_set_num_threads(1);
    const auto storage_mode = options.ml_only ? alliance::StorageOpenMode::MultiLevelOnly
                                              : alliance::StorageOpenMode::RequireRawGraph;
    Index index(options.graph_file, options.meta_file, options.space, storage_mode);
    const auto fingerprint = LoadFormalQuantization(index, options, protocol);
    if (protocol.mode == "fingerprint") {
      benchmark::DenseCheckpointAttributes(index);
      WriteFingerprint(protocol.output, fingerprint);
      return 0;
    }

    // Retain canonical probes for the whole session, matching the frozen driver's
    // resident allocation. Packing and navigation preparation precede all queries.
    auto probes = benchmark::CheckpointScoreProbe(index, false);
    benchmark::PrepareSearchCodes(index, options);
    Require(fingerprint == protocol.fingerprint, "quantization fingerprint");
    Require(benchmark::QuantizationFingerprint(index) == fingerprint, "packed fingerprint");
    benchmark::ValidateCheckpointProbes(probes, benchmark::CheckpointScoreProbe(index, true));
    benchmark::PrepareGuidedEntriesForSearch(index, options);
    const auto geometry = LoadFormalLayout(index, options, protocol);
    benchmark::LoadNavigationForSearch(index, options);
    benchmark::PrintSearchEntryConfiguration(index, options);
    const auto plans = alliance::PlanSearchIOBudget(
        1, options.beam_width, options.rerank_beam_width, options.io_budget_total);
    const auto &plan = plans.front();
    // Context/workspace are destroyed before their index. The batch observer
    // attaches only during measurement; no observer pointer escapes a batch.
    alliance::DiskSearchContext context(plan.queue_depth, options.graph_file, options.ml_graph_file,
                                        storage_mode);
    context.UseOwnedWorkspace();
    benchmark::ConfigureSearchPolicy(context, options, plan);
    const auto inputs = LoadFormalInputs(index, options, protocol);
    const auto efs = FormalEFPlan(options);
    CheckLoadingMemory(protocol.loading_limit);
    PhaseMemory phase_memory(protocol.scope_prefix, protocol.memory_mode == "phase");
    phase_memory.ReleaseLoadingCap();
    FormalBatchSession session{index,    context, options,      protocol,
                               geometry, inputs,  phase_memory, fingerprint};
    RunFormalBatches(session, efs);
    Require(benchmark::QuantizationFingerprint(index) == fingerprint,
            "final quantization fingerprint");
    std::cout << "SEARCH_VERIFIED fingerprint=" << fingerprint << std::endl;
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "SEARCH_FAIL " << error.what() << std::endl;
    return 1;
  }
}
