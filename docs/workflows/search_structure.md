# Search source map

Start with the [lifecycle contract](../search_lifecycle.md) for ownership,
timing, cache and memory boundaries. This map identifies the current files;
public headers use feature paths under `include/`, with namespace `alliance`.

## Native tools

| Owner | Files and entry functions |
| --- | --- |
| Ordinary entry | `src/tools/search/search_disk.cpp::main` |
| Value configuration and parsing | `search_disk_options.hpp`, `search_disk_options.cpp`: `DiskSearchOptions`, parse and normalize |
| Shared preparation | `search_preparation.hpp/.cpp`: `PrepareSearchCodes`, `PrepareGuidedEntriesForSearch`, `LoadSearchLayout`, `ConfigureSearchCache`, `ConfigureSearchPolicy` |
| Ordinary execution | `search_disk_runner.cpp`: `RunPipelineBatch`, serial/raw/compare runners |
| Ordinary reporting | `search_disk_report.cpp`: post-timer validation, Recall and counters |
| Formal entry and preparation | `src/tools/benchmark/benchmark.cpp`, `session.hpp/.cpp` |
| Formal batches | `batch.hpp/.cpp`: prepare -> measure -> validate -> archive |
| Formal contracts | `protocol.hpp`, `strict_input.hpp`, `layout.hpp`, `phase_memory.hpp`, `io_observer.*`, `archive.hpp` |

`src/tools/CMakeLists.txt` maps these owners to `alliance_search_disk` and
`alliance_benchmark`. Both share options/preparation, not their measurement loop.
The latter calls the same core query API, using a single owned context.

## Core query implementation

Files below are relative to `src/alliance/search/`:

| File/group | Responsibility |
| --- | --- |
| `multilevel_query.cpp` | `searchKNNMultiLevelPipeline`, query orchestration |
| `query/plan.hpp` | Storage/policy validation and empty/graph/prefilter/legacy routing |
| `query/graph.hpp`, `query/prefilter.hpp`, `query/results.hpp` | Entry preparation, route execution, candidate and final-result stages |
| `entry_routing.cpp` | Graph layer/window and initial entry selection |
| `multilevel_traversal.cpp` and traversal helpers | Candidate discovery, EF queues, asynchronous graph I/O and graph-cache state |
| `multilevel_edge_access.cpp` | Multi-level edge record access |
| `multilevel_rerank.cpp` | Exact-vector final rerank and bound-aware request scheduling |
| `search_sync.cpp`, `raw_pipeline.cpp` | Retained synchronous/raw asynchronous compatibility paths |

Private `.hpp` helpers in `src` are implementation boundaries, not duplicate
public APIs. Public policies, context/workspace interfaces and shared types live
under `include/search/`. Static-layout loading/cache preparation belong to
`src/alliance/layout`, and quantization adapters to `src/alliance/quantization`.

## Launch and evidence

[Script entries](entry_points.md) distinguish the 16-thread dataset
workflow from formal single-thread/runtime-bound execution. The formal Python
runner owns SHA validation, sent receipts, independent replay, aggregation and
resume without reissuing measurements. The native binary owns its timed loop.

Memory benchmarks, navigation/converters, inline rerank and query-executing
landing collection are retired. `alliance_collect_landing` remains a separate
structural routing tool. Exact rerank, bounded rerank and raw/v2/v3 readers remain.
