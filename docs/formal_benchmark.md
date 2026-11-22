# In-tree formal benchmark

`alliance_benchmark` is the dataset-neutral implementation of the final LAION
native-observer protocol. Build it with the main CMake build; no experiment-local
C++ copy, absolute historical include path, or frozen external static library is
needed. Its input options remain the common `DiskSearchOptions` options.

The driver is split into protocol/scheduling, strict input reading, layout
geometry, phase memory, AIO observation and raw archive modules under
`src/tools/benchmark/`. Search still calls `searchKNNMultiLevelPipeline` with an
explicit context-owned workspace. Graph traversal, RaBitQ estimates, routing,
rerank policy and static/dynamic cache implementations remain in the core.

## Preparation and batch owners

`benchmark.cpp::main` establishes the index/context lifetime and phase memory.
`session.cpp` validates formal options, loads the selected quantization source,
checks layout geometry and reads exact-count inputs. Shared
`tools/search/search_preparation.cpp` prepares codes, guided entries, layout,
cache configuration and context policy for both native entry points.
`batch.cpp::RunFormalBatches` owns the schedule and each batch's
`PrepareBatch -> MeasureBatch -> ValidateBatch -> ArchiveBatch` sequence.
The observer attaches only while measuring; the context is destroyed before
its index. Preparation probes remain alive throughout the session, so this
split does not change their measured RSS lifetime.

The native tool is always single-threaded. The phase memory policy is a search
RSS acceptance test after releasing the loading cap, not a hard search-only
cgroup RSS limit. `run_dataset.py search` retains its separate 16-thread,
metric-wise median workflow and must not be substituted for this runner.

## Preserved protocol boundaries

- `alliance-formal-v1` times the native batch and its accepted-AIO observer.
  GT/ID/predicate validation, raw writing and independent replay are outside the
  timed batch. This is the actual final LAION source behavior; older descriptions
  saying its GT validation was timed are incorrect.
- Positive EF values below k remain legal. A short-result batch is rejected and
  its rejection evidence retained; it is never silently repeated or aggregated.
- The checkpoint source requires checkpoint, centroid and CID paths, exact
  fingerprint, expected shape and identity labels. The explicitly selected
  `meta` source uses already persisted quantization, requires a fingerprint and
  validates a dense label bijection; it never re-quantizes or implicitly falls
  back. The one-bit checkpoint sentinel `t_const == -1` is checked explicitly.
- I/O counts are actual accepted requests with completion/source/FD checks.
  Neighbor and vector byte sizes come from the validated v2/v3 layout, so a
  vector read is not incorrectly counted as a neighbor page. A 4 KiB equivalent
  is requested bytes / 4096, not a count of physical device reads.
- The phase policy records loading RSS, releases only this benchmark's scoped
  loading cap, resets RSS/cgroup peak counters for each batch and checks search
  RSS separately. The whole-process policy retains the cap and lifetime HWM.
  Actual `memory.events` must exist with zero OOM/OOM-kill counters. The Python
  runner additionally records the scope identity, memory caps, CPU affinity and
  before/after events; missing evidence does not become zero.
- Dynamic cache invalidation occurs at each EF/repeat boundary. Per-query clear
  and automatic growth remain explicit profile choices. Warmups are explicit:
  EF-first scheduling warms once per EF before its repetitions; repeat-first
  scheduling warms within each repeated EF sequence.

## Dataset profiles and legacy compatibility

`scripts/benchmark.py` defaults to the unchanged CaseLaw schema-1 catalog and
schema-2 execution adapter, including its pinned historical binaries, timings,
five-to-three QPS-paired aggregation and recovery behavior.

New schema-3 dataset profiles are in `scripts/configs/datasets/`:

| Profile | Quantization | Cache | Memory | Warmup/order |
| --- | --- | --- | --- | --- |
| `laion10m.json` | IP, 3-bit, 512D, 128 centers | 196 MiB static, 8 MiB FIFO, reuse across queries | 9 GiB loading / 3 GiB search RSS | none; repeat then EF |
| `arxiv2m_observer.json` | IP, 3-bit, 384D, 256 centers | 480 MiB static, 8 MiB FIFO, per-query clear and auto growth; guided entries | 1 GiB whole-process cap | one batch; EF then repeats |

The Arxiv observer profile deliberately has a distinct protocol name from the
historical `RunPipelineBatch` timer, which had no added AIO observer. Matching
parameters does not make their QPS measurements interchangeable. Its unknown
`wp`/maximum layer values must be bound from the actual index metadata, rather
than inferred from a directory name. LAION P1 enables top-layer postfilter; its
other workloads disable it. Arxiv retains its historical routing configuration.

Profiles contain no production file paths and do not choose new EFs. A runtime
binding JSON contains:

```json
{
  "schema_version": 1,
  "profile_sha256": "<SHA256 of the exact profile JSON>",
  "lock_paths": ["/absolute/path/benchmark.lock"],
  "assets": {
    "binary": {"path": "/absolute/build/bin/alliance_benchmark", "sha256": "<SHA256>"}
  },
  "curves": [{
    "workload": "mixed", "cache_policy": "landing", "efs": [7, 10, 20],
    "quantization_fingerprint": "<unsigned 64-bit decimal>",
    "assets": {"binary": "binary"}
  }]
}
```

The abbreviated example must bind all actual roles before execution: `base`,
`attributes`, `query`, `ranges`, `gt`, `graph`, `meta`, `ml_graph`, `ml_meta`, and
for checkpoint mode `checkpoint`, `centroids`, `cids`. Native profiles default to
`navigation=true` when omitted, meaning EF/budget/count 64/256/16; the current
Arxiv profile explicitly selects it. Each selected curve must bind a `navigation`
role with a fixed SHA256. Missing bindings fail planning; missing files or
changed SHA values fail preflight before queries. An explicit
`navigation=false` with guided entry settings keeps guided sampling; existing
LAION/CaseLaw profiles retain their historical explicit settings. Navigation
must match the selected graph/layout; a new graph cannot directly reuse an old
navigation file, and this change does not restore a generator.
Every asset has an absolute path and SHA256. Each
curve maps role names to entries in the asset ledger. Static layout metadata is
the pinned `ml_meta` file for that landing/equal curve. `off` is allowed only with
a zero static budget. Optional `index_shape` can fill missing profile geometry,
but cannot override a known value.

`ml_only` defaults to `false`. An explicitly enabled profile adds the native
`--ml_only` flag and may omit the `graph` asset; `meta`, `ml_graph` and `ml_meta`
remain mandatory. Independent replay still requires the original base input.
This mode does not open the old graph or its raw-vector file and cannot invoke
legacy/compare search or implicit re-quantization. It does not delete artifacts;
cleanup is a separate, explicitly accepted operation.

```sh
python scripts/benchmark.py --config scripts/configs/datasets/laion10m.json \
  --bindings /absolute/runtime.json --plan
python scripts/benchmark.py --config scripts/configs/datasets/laion10m.json \
  --bindings /absolute/runtime.json --validate
python scripts/benchmark.py --config scripts/configs/datasets/laion10m.json \
  --bindings /absolute/runtime.json --run --job /absolute/new-job
```

Planning does not open production assets or run queries. Preflight validates
identity, complete query/GT headers and shapes. Execution durably marks a curve
sent before launch. Resume may verify/export its archive, but cannot launch a
sent measurement again. Independent replay supports fbin/fvecs base vectors,
raw signed-int32 or headered ibin attributes, raw closed range pairs and Top-k
ivecs. It recomputes exact L2 or `1 - dot` IP for returned IDs, Recall, QPS, P95,
byte geometry and all counter totals. Every repetition must validate before
paired trimming; the same three repetitions supply QPS, Recall and I/O.

For small fixtures, write a separate profile with the actual small shape,
query count and static budget, keeping the selected production algorithm knobs.
`BENCH_MODE=fingerprint` is an explicit zero-ANN-query native operation for
capturing a persisted meta checkpoint fingerprint; measurement always requires
that value as `BENCH_EXPECTED_FP`. The remaining controls are generated by the
dataset adapter (`BENCH_EXPECTED_*`, `BENCH_MEMORY_MODE`, `BENCH_ORDER`, cache and
RSS budgets, repeats, warmups and output). Direct native use still performs
strict input checks and exclusive output claims, but the coordinated runner is required
for complete input SHA and scope evidence.

## Provenance and verification

`formal_benchmark_sources.json` records remote source paths and SHA256 values,
verified against the downloaded source-only copies. The LAION compiler record
pins final source SHA
`172e13e8c05711ba4f243799286854164e8cb293df341907298acd63825970b4`.
Its frozen binary SHA is
`bc72c72506af4960974c02b0770cadd7b48a4eb624a077621ca9971787313bbc`;
the newly compiled in-tree executable necessarily has its own pinned SHA.

The native input tests check malformed row headers, truncated files, reversed
ranges and exclusive output claims. `test_benchmark_datasets.py` checks profile
identity, EF<k planning, the different Arxiv/LAION protocols, exact IP replay,
headered attributes and 4/8 KiB conversion. Existing schema-2 replay and runner
tests remain applicable. Legacy per-metric trim/median wrappers call the shared
named aggregation helpers without switching to QPS-paired trimming.
