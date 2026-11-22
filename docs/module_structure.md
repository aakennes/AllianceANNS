# Module ownership and entry points

This organization separates public library headers, private implementations,
production tools and regression tests. Executable names and CLI options are
unchanged. Frozen experiment programs and recorded results are not rewritten.

## Public headers (`include`)

| Directory | Responsibility |
| --- | --- |
| `index` | `AllianceIndexDisk` facade only |
| `build` | Build configuration, transient state, pure phase plans and record geometry |
| `search` | Query policy/context, entry selection, visited state, exact rerank support and worker memory |
| `storage` | Paged records, metadata, page constants, rank/order and window geometry |
| `io` | Binary/vector input, AIO requests, streaming and completion handling |
| `metrics` | Search counters and I/O queue profiling |
| `quantization` | Alliance-owned RaBitQ adapter, packed code layout and scoring utilities |
| `cache` | Static and dynamic cache data structures |
| `core` | Shared scalar/container primitives and filters |
| `space` | Distance spaces |
| `rabitq`, `third` | Embedded dependencies; internal paths remain unchanged |

Repository consumers use feature-qualified header paths such as
`#include "index/index_disk.hpp"`. The C++ namespace remains `alliance`. The facade remains at
`index/index_disk.hpp`; its nested `Phase15Plan` and `Phase2Plan` names
remain source-compatible aliases. Previously compiled clients must be rebuilt
when linking the new library. This is not a promise of binary ABI compatibility
or compatibility with retired historical APIs.

## Private implementation (`src/alliance`)

`index` owns facade construction/destruction and delegating adapters; `storage`
owns record access and metadata; `disk_build` owns the existing build stages and
budget solver; `search` owns traversal/rerank/workspaces; `layout` owns layout
loading and static-cache preparation; `shuffle` owns offline ordering/export;
`quantization` owns quantizer integration; `io` owns internal resource guards.

The four existing object-library groups and the `alliance_core` link target are
preserved. Private headers are resolved through target-private include paths.
The embedded RaBitQ and external RaBitQ shared-library boundary is unchanged.

## Production tools (`src/tools`)

| Tool | Source entry | Core call |
| --- | --- | --- |
| `alliance_build_disk` | `build/build_disk.cpp::main` | `RunDiskBuild` then private `ExecuteBuild` (public `BuildPostPopulate` retained) |
| `alliance_shuffle_disk` | `layout/shuffle_disk.cpp::main` | `ComputeMultiLevelShuffleOrders`, `ExportMultiLevelLayout` |
| `alliance_search_disk` | `search/search_disk.cpp::main` | Search runner and selected index query route |
| `alliance_benchmark` | `benchmark/benchmark.cpp::main` | `searchKNNMultiLevelPipeline` under the formal measurement protocol |
| `alliance_collect_landing`, `alliance_prepare_static_cache` | `layout/` | Routing profile and cache sidecar preparation |
| `verify_index`, `alliance_verify_layout` | `validation/` | Index/layout validation |
| `alliance_gengt` | `dataset/gen_ground_truth.cpp::main` | Exact ground truth generation |
| `bench_iops` (opt-in) | `diagnostics/bench_iops.cpp::main` | Storage performance diagnostic |

Value-only search options and shared preparation live in `tools/search`; the
ordinary and formal tools retain separate measurement owners. See the
[search source map](workflows/search_structure.md).

Common CLI utilities live in `common`. `src/test` remains separate, opt-in and
grouped by test responsibility. The optional storage diagnostic is not a search
benchmark and is excluded unless `ALLIANCE_BUILD_DIAGNOSTICS=ON`.

## Build planning boundary

`build/disk_build_plan.hpp` defines value-only `Phase15Plan` and `Phase2Plan`.
`build/disk_build_geometry.hpp` defines `NodeStoragePlan`, `PlanNodeStorage` and
`BuildVectorPageBytes`. They do not depend on the index, PagedFile, AIO, caches or
RaBitQ. Page constants have one definition in `storage/page_constants.hpp`.

The private `disk_build/memory/budget.hpp` declares a non-owning `MemoryPlanner`. Its
solver implementation does not include the index facade. Index-to-planner field
copying is confined to `memory/index_adapter.hpp`; the facade's existing public
planning methods delegate to that solver. The CLI planning path uses the pure
planner and geometry directly, without constructing or opening an index.

Budget formulas, default parameters and node/vector page formats are preserved.
Unused diagnostic fields were removed from the in-memory plan; source clients
must relink. The layer-window vector is still
borrowed by the planner and must outlive planning/reporting.

Build lifecycle lives in private `disk_build/stages/`, with resident and upper
coordination and a common `checkpoint.hpp` for persisted progress transitions.
The CLI owns input validation and fresh/resume preparation in `tools/build/disk/`.
`disk_build/memory/process.*` separates sampling, printing, trim and peak reset.
Cluster preparation is one coordinator; algorithms and I/O live in `left_stream/`,
`partition/` and `cluster_wow/`. See the [execution map](../src/alliance/disk_build/README.md).
Metadata
serialization delegates durable file commit to private `storage/atomic_file.*`.
See [build contracts](workflows/build.md) for the checkpoint
boundary, LeftStream-only resident construction and upper-layer Pass B commit.
