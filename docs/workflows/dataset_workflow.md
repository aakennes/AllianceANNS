# Dataset build, shuffle and search workflow

`build.sh`, `shuffle.sh` and `search.sh` delegate to `run_dataset.py`. This is the
existing dataset configuration workflow. Formal single-thread experiments use
[benchmark.py and explicit runtime bindings](../formal_benchmark.md).
Retired one-off PipeWoW/PipeANNFilter launchers are not part of this workflow.

## Configuration ownership

The selected stage JSON recursively overrides `defaults.json[stage]`; arrays
replace as a whole. Dataset paths/environment are merged separately with their
shared defaults. For search, `dataset.json.index` then overrides same-named
stage parameters, while `queries.json` selects query/range/GT/cache paths.
`--index-dir` takes precedence for the index and cache paths recorded in its
manifest. CLI `--efs` and `--repeats` override the corresponding configured
search settings. Build does not inherit search parameters, and search does not
inherit build parameters. Effective configuration and file SHA256 values are
recorded with each run.

Main-driver ML graph/auto search defaults to `entry_sampling=navigation`, with
navigation EF/budget/count 64/256/16. The runner resolves `navigation_file` from
the selected `--index-dir` manifest, or `navigation.bin` beside that manifest's
`graph_file` (beside `ml_graph_file` when ML-only and no raw graph is bound).
A relative manifest navigation path is relative to the manifest directory.
Without `--index-dir`, the configured explicit navigation path or current
graph directory is used. Missing navigation fails preflight even in dry runs.
Selecting a new manifest never inherits a navigation path from the old dataset
configuration. The native loader checks graph/layout identity before queries.

To use guided sampling, put `{"parameters":{"entry_sampling":"guided"}}` in
the copied `search.json`; explicit global/range sampling also disables implicit
navigation. Pure prefilter/legacy routes do not implicitly load navigation.
Build and shuffle do not generate a navigation artifact. Prepare a compatible
artifact before searching a new graph; never copy an old graph's file into a
new build. No navigation generator is restored in this change.

| File | Responsibility |
| --- | --- |
| `defaults.json` | Shared workflow parameters, executable bindings and protocol defaults |
| `dataset.json` | Input paths, current index/layout and memory limits |
| `build.json` | M, efc, windows, threads and partition/build overrides |
| `shuffle.json` | Level geometry, BlockBNF, shuffle threads and static cache |
| `search.json` | Search binary/driver, checkpoint, beams and policy overrides |
| `queries.json` | Query vectors, ranges, GT and workload cache layouts |
| `efs.json` | Workload/k-specific EF values |

`queries.json` has common `defaults` plus `workloads` overrides. `efs.json` maps
workload and k to an array, for example `"p1": {"25": [10,15,20]}`. Missing
combinations use `search_protocol.fallback_efs`; fallback grids are not a new
recall-coverage calibration. A stage may override repetition/stop settings:

```json
{"protocol": {"repeats": 3, "recall_stop": 0.995}}
```

Do not infer current algorithm settings from historical performance notes.
The current build path is LeftStream plus Vamana nearest-two partitioning and
ClusterWoW. IVF construction, direct-inherit, resident two-pass and copy_top
are retired. Search-side quantization is independent of partitioning. See
[build contracts](build.md) and [search policy](../search_policy.md).

## Read-only preview

```bash
cd /home/kdr/AllianceANNS
bash scripts/build.sh sift1m --dry-run
bash scripts/shuffle.sh deep10m --workload mixed --dry-run
bash scripts/search.sh arxiv2m --workload origin --k 100 --dry-run
bash scripts/search.sh sift1m --config-dir /path/to/sift1m --workload p1 --k 25 --dry-run
```

`--config-dir` points to a stage-config directory; `--config` is its alias.
`--defaults` selects another shared defaults file. Preview does not start an
experiment or create outputs. Dataset paths in examples must exist on the host
where an explicitly requested run is performed.

## Build and shuffle artifacts

The native build CLI keeps `rawvec_export=on` for compatibility. The current
shared workflow selects `off` and finalizes successful builds: full acceptance
runs before closing the resume window and removing the exact `.buildvecs` file.
Failures retain build inputs and resumable evidence. This workflow choice does
not delete historical indexes. See [artifact lifecycle](../build_artifact_lifecycle.md).

Shuffle uses `block_bnf` by default, at most 32 iterations, with fixed margin 1;
`identity` is the retained control. It writes a cache-free complete layout,
verifies graph/vector/mapping bytes, and publishes `GRAPH.completion.json` last.
Automatic reuse validates that manifest and file hashes. Static caches are
prepared separately for the chosen workload. The source graph is not deleted.

## Manual command examples

These examples describe explicit launches; reading the guide does not execute
them. Output directories must be new.

```bash
cd /home/kdr/AllianceANNS
RUN=/SSD/kdr/allianceann/sift1m/manual_example_02
bash scripts/build.sh sift1m --out-dir "$RUN/build"
bash scripts/shuffle.sh sift1m --index-dir "$RUN/build" \
  --workload mixed --out-dir "$RUN/shuffle"
bash scripts/search.sh sift1m --index-dir "$RUN/shuffle" \
  --workload mixed --k 25 --out-dir "$RUN/search_k25"
```

To select the configured current index, omit `--index-dir`:

```bash
bash scripts/search.sh deep10m --workload p1 --k 100
bash scripts/search.sh sift1m --workload mixed --k 25 \
  --efs 5,10,15,20,25,30,40,50,65,80,100,125,160,200 --repeats 3
```

## Search protocol boundary

This driver requires **16 search threads**, and its supported k values are
10/25/100. EF values are positive, sorted, unique multiples of 5, bounded by
64*k; EF below k is permitted. Each repeat invokes the configured native driver
for its EF sequence. The driver takes a separate median for each metric across
repeats. It does not use five-to-three QPS-paired trimming.

The configured Recall stop retains the first measured crossing point and stops
larger EFs. CSV repetition counts report actual measurements. Search is scoped
with the configured whole-process `MemoryMax`, no swap, CPU affinity and
`/usr/bin/time -v`; it does not silently adopt the formal benchmark's
loading/search phase protocol. Exact limits and quantization settings come from
the effective dataset files, not a universal value in this guide.

Logs contain commands, environment, raw output, metrics and effective
configuration. New run outputs use the configured `result_root`, under
`allianceann/<dataset>/k10|k25|k100/` with dataset JSON summaries. Keep this output
root outside the repository's paper-data `result/` directory. Imported paper CSVs
are reference data, not output targets for new runs. Frozen driver configurations
remain protocol evidence; never relabel an old measurement as a new result.
