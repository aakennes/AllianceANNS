# Multi-level search metrics and profiling

Multi-level search accumulates hot counters in `ScopedMultiLevelMetrics` instead of updating shared atomics for every distance, hop or cache lookup. The public index counters retain their existing names.

An ordinary standalone API call publishes its counters when it returns, including partial counters on an exception. Nested helpers for the same index share one accumulator. Scopes for a different index remain independent. A worker can amortize merging across its entire batch:

```cpp
#pragma omp parallel
{
  alliance::ScopedMultiLevelMetrics batch(index);
  // Run queries on this worker. Its counters merge when batch leaves scope.
}
// Read aggregate counters after the parallel region has finished.
```

Do not reset or interpret public counters as complete while a batch scope remains active. The index must outlive the scope. Legacy search paths that do not use this scope retain their existing direct accounting.

## Phase timing

Normal builds contain no phase clock probes. Build a separate profiling executable:

```bash
cmake -S . -B build_prof -DALLIANCE_PIPELINE_PROFILE=ON -DALLIANCE_PIPELINE_DIAGNOSTICS=OFF
cmake --build build_prof --target alliance_search_disk -j8
```

`# phase_timing` reports milliseconds per query:

| Field | Meaning |
|---|---|
| `prepare_ms` | Routing, query estimator and entry-point preparation |
| `traversal_ms` | Candidate search, including lazy graph workspace initialization; prefilter uses this phase for its scan |
| `rerank_ms` | Candidate preparation for exact rerank, vector I/O and exact distance evaluation |
| `finalize_ms` | Candidate deduplication/budget selection and final result sorting |
| `query_ms` | Sum of the four exclusive phases |
| `graph_wait_ms`, `rerank_wait_ms` | Elapsed time inside completion waits; subsets of the relevant phases |
| `mean_query_ms` | Independently measured mean API latency |
| `batch_wall_ms` | Wall time for the complete parallel batch, used for QPS |
| `team_enter_max_ms` | Latest worker entry relative to the batch timer; includes parallel-region startup |
| `first_query_mean_ms` | Mean first-query latency across configured workers; includes real search work as well as lazy initialization |

The phases measure elapsed time, including scheduling delays, not thread CPU time.
The production ML handler waits by busy polling; other handlers may use blocking
waits. Neither duration is a direct device-latency measurement. Existing `poll`
fields also include nonblocking completion processing, decoding and cache
insertion. The legacy `cpu_ms` field is retained as a non-I/O elapsed-time residual;
it is not process CPU consumption. `avg_inflight` is iteration-weighted, not a
time-weighted device queue depth.

In concurrent runs, `batch_wall_ms / query_count` is the inverse of throughput, not mean query latency. Phase percentages use mean query latency. The ordinary ML CLI starts timing before the OpenMP parallel region and does not perform an untimed warm-up: thread-team startup and lazy per-worker allocation remain in batch QPS. Index loading and pre-created search contexts are outside the timer.
The formal single-thread observer benchmark has a separate timer and validation
contract; see [measurement boundaries](../search_lifecycle.md).

Use normal builds for throughput A/B comparisons and profile builds for attribution. Keep index, query set, search budget, beams and cache settings fixed; compare Recall and result validity as well as QPS. Profiling clock overhead must not be mistaken for a regression in the normal build.
