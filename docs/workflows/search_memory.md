# Search memory budgets and shared static caches

The multi-level graph search uses a shared read-only index and static cache,
with a private dynamic cache and exact visited set per worker. RaBitQ bits and
rerank policy are unchanged by the memory optimizations.

## Dynamic cache and visited

`--ml_cache_bytes` is a **per-worker payload limit**. Prefer
`--ml_cache_total_bytes` for a fixed total payload limit across thread counts.
Neither flag includes lookup metadata, visited, candidates, I/O contexts, or
other process memory. With `--per_query_cache --ml_cache_auto 1`, each worker
grows its payload only at query boundaries to its observed working set, up to
its share of the payload limit. Clearing a query does not release capacity.

The lookup now stores exact `(node_id, slot)` entries in a preallocated open
addressed table. Its size is proportional to allocated cache slots, rather
than the number of index nodes. Backward-shift deletion preserves lookup
semantics across FIFO/LRU eviction without tombstones or per-entry allocation.

Graph visited state uses a bitset and a list of dirty 64-bit words. Membership
is exact; query reset clears only touched words. The worst-case reservation is
`ceil(N / 64) * 12` bytes per graph worker. Pure prefilter workers do not allocate
these graph structures.

For reproducible ablations, CMake options `ALLIANCE_DENSE_DYNAMIC_CACHE=ON` and
`ALLIANCE_STAMP_VISITED=ON` restore the previous data representations. Both
default to OFF. These options do not change the index format.

## Static cache budget

The static cache remains shared and immutable during search. Runtime lookup
uses a bitmap/rank directory and metadata only for cached nodes. The old dense
v2 sidecar format remains readable; the loader makes two streaming passes over
metadata so it never retains a full temporary dense metadata array.

Static cache selection budgets include:

```
aligned_payload + ceil(N / 64) * 12 + cached_nodes * sizeof(StaticNodeMeta)
```

An empty cache has zero resident cache allocation. Build-time selection work
and the dense persisted metadata are separate from query-resident memory.
New exports enforce this budget; old sidecars are still loadable but were
created under the old payload-only budget semantics.

Existing multi-level graphs can be reused to prepare several static caches:

```bash
./build/bin/alliance_prepare_static_cache \
  --graph_file "$GRAPH" --meta_file "$BITS3_META" \
  --ml_graph_file "$ML_GRAPH" --ml_meta_file "$ML_META" \
  --space ip --static_landing_file "$TRAIN_LANDING" \
  --out_prefix "$NEW_CACHE_PREFIX" \
  --budgets 33554432,67108864,134217728
```

The output paths append `_<budget bytes>.layout` to the prefix. They must be
new paths; existing files are never overwritten. The graph itself is not
rewritten. One sequential scan computes in-degree scores over each level's
owned layers; subsequent scans copy selected layer records. Node selection
and layer merging retain the existing policy. Without a routing histogram,
the existing equal-per-level allocation is used. Identify whether histogram predicates come from the evaluation ranges or
independent training ranges; this collector does not execute query vectors.

Use the selected sidecar as the search binary's `--ml_meta_file` argument.
`--static_cache_budget` belongs to layout construction, not query search.
Graph static caches do not speed up pure prefilter paths by themselves.

## Total process budget and timing

`alliance_search_disk --memory_budget_bytes B` checks Linux process peak RSS
and swap at loading/search phase boundaries in the multi-level pipeline path
(`--pipeline 1`) and exits with an error if the
peak exceeds B or swap is observed. Zero disables the check. This is a budget
validation guard, **not an allocator limit or cgroup hard limit**. A benchmark
must reject over-budget results even if it saw a QPS line. A transient RSS
peak between checks is included through `ru_maxrss`.
Other search paths currently have the loading-phase checks but do not have an
equivalent search-end guard; do not use this flag alone to certify their full
query-workspace budget.

The logs distinguish:

- `memory_phase`: current RSS, lifetime peak RSS, and swap at phase boundaries.
- `shared_memory`: known shared allocation capacities, excluding static cache.
- `worker_memory`: retained per-worker cache, visited, and I/O/scratch allocations.
- `memory`: quantized codes and configurable caches, including
  `static_resident_bytes`. `tunable_total_bytes` is not total process memory.

The component counters are capacities, not a simultaneous RSS attribution.
Runtime, allocator overhead, candidates, and temporary allocations are covered
by process RSS even where there is no separate component counter. Linux page
cache and kernel AIO memory are outside the process RSS scope.

Index/query/context creation and memory reporting are outside the timed query
batch. OpenMP team entry, first-query lazy allocation, result collection, and
the final worker barrier remain inside. Keep the same timing and warmup policy
when comparing implementations.

For equal-memory comparisons, use the same **total process RSS ceiling**, query
set, predicates, K, thread count, and recall threshold. Allow each system to
use the remaining budget for useful caches, record actual RSS, reject swap and
invalid results, and keep cache0 controls. A common ceiling does not imply
identical actual RSS or globally optimal parameters.
## Per-selectivity static cache allocation

`disk_search_common.sh`, the lower-level compatibility helper, accepts
`--static_cache_use_landing true|false`, default `false`. With a positive
`--static_cache_budget` (bytes), `false` prepares equal **per-level** budgets;
`true` looks up the histogram for the current range file and generates it on a
cache miss, before starting any search process. Cache preparation is outside
the search timer and search RSS measurement. Static cache is shared by all
threads; the total budget is not divided by thread count. Dynamic cache options
are unchanged. No graph rebuild or reshuffle is needed.

For example, append these options to an existing search command with a supplied
ML graph/layout:

```bash
--inputs_dir /SSD/kdr/allianceann/arxiv2m/queries/final \
--selectivities p1,p4 --k 10 \
--static_cache_budget 260046848 \
--static_cache_use_landing true \
--static_cache_dir /SSD/kdr/allianceann/arxiv2m/static_cache
```

Use `--static_cache_use_landing false` for the equal-budget comparison. Each
mode has its own sidecars, even when a supplied base layout already contains
static cache. Search mode creates new shuffle layouts with static budget zero,
then builds separate sidecars for the requested allocation. Existing files are
never overwritten. `--inputs_dir` selects canonical `<dataset>_<label>.ranges.ibin`
and `<dataset>_<label>.k<K>.gt.ivecs`, where dataset is `arxiv2m` or `sift1m`,
and label is `origin`, `mixed`, `p1`, `p4`, `p7`, or `p10`. Existing legacy and
explicit `--query_rng/--gt_file` paths still work.

The new `alliance_collect_landing` utility uses the index's actual
`DecideLayerRange` function and routing conditions. It reads closed int32 range
pairs and index metadata, without query vectors, GT, ANN traversal or rerank.
Only graph routes contribute to the layer histogram; prefilter, empty and
legacy brute-force routes are counted separately in its JSON report. An
all-prefilter range produces zero graph landings: the existing allocator then
falls back to equal budgets, explicitly reported as `equal_no_graph_landings`.

By default, the current range predicates supply the distribution. This is a
range-aware resource policy and must be identified as such in comparisons. To
use **independent training predicates**, pass
`--static_landing_query_rng /path/to/train.ranges.ibin`. To reproduce a combined
p1/p4/p7/p10 training profile like the reference server, pass the existing binary
`--static_landing_file /path/to/combined.landing`; this file takes precedence over
range collection. The format is a little-endian uint64 layer count followed by
that many uint64 counts. An explicit `false` ignores both training options.
For compatibility, an explicit landing file implies `true` only when the boolean
is omitted. PipeANN page selection is unaffected.

Histograms are reused across thread counts and budgets. Keys include the range
content SHA, index metadata SHA, source graph identity, collector SHA and routing
settings. Large immutable graphs/layouts use inode, size, nanosecond times and
header SHA rather than hashing their complete contents on every invocation.
Sidecar keys also include layout identity, base sidecar SHA, preparer SHA,
allocation mode and budget. Manifests verify every generated file's full SHA
before reuse. Changed inputs regenerate managed artifacts; corrupt cached
artifacts and failed attempts are preserved for diagnosis. Linux file locks and
atomic publication prevent duplicate concurrent preparation. Do not modify the
index/ranges concurrently with preparation.

The standalone frontend is also available without launching searches:

```bash
ALLIANCE_BIN_DIR=/home/kdr/allianceann/build/bin \
python3 scripts/prepare_selectivity_static_cache.py \
  --graph_file /path/graph.bin --meta_file /path/meta.bin \
  --ml_graph_file /path/ml_graph.bin --ml_meta_file /path/ml_graph.bin.layout \
  --query_rng /path/p1.ranges.ibin --space ip \
  --static_cache_use_landing true --budgets 260046848,268435456 \
  --cache_dir /path/static_cache
```

Build with `cmake --build <build> --target alliance_collect_landing`; the existing
`alliance_prepare_static_cache` binary remains unchanged. Run regression checks
with `python3 -m unittest discover -s scripts/tests -p test_selectivity_static_cache.py`.
