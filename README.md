# AllianceANN

AllianceANN builds a disk graph with LeftStream, Vamana partitioning and
ClusterWoW, exports a multi-level search layout, and searches it with RaBitQ
candidate scoring and final exact-vector reranking. Uniform-page v2 and mixed
neighbor/vector-page v3 layouts remain readable.

## Build the tools

The Linux workflow needs CMake 3.20+, a C++20 compiler with OpenMP, BLAS, libaio,
Git and Python 3. Formal input verification also uses NumPy. Launch scripts use
a working `systemd --user` manager with memory-controller delegation,
`/usr/bin/time` and `taskset`. Dataset paths, allowed CPUs and
`environment.XDG_RUNTIME_DIR` must match the execution host.

```sh
cmake -S . -B build -DBUILD_TESTING=OFF
cmake --build build -j 6
```

CMake fetches the pinned RaBitQ dependency. For an existing checkout at that
revision, add `-DALLIANCE_RABITQ_SOURCE_DIR=/absolute/RaBitQ-Library`.
Executables are written to `build/bin/`. Tests are opt-in with
`-DBUILD_TESTING=ON`; see the [test guide](src/test/README.md).

**Compilation does not change script bindings.** `scripts/defaults.json` pins
verified server binaries, not automatically the local `build/bin` directory.
To use a newly validated build, copy the configuration and update its executable
paths and SHA values:

```sh
CONFIG_ROOT="$HOME/alliance-runs/arxiv-check-01-config"
mkdir -p "$CONFIG_ROOT"
cp -R scripts/configs/arxiv2m "$CONFIG_ROOT/arxiv2m"
cp scripts/defaults.json "$CONFIG_ROOT/defaults.json"
python3 - "$CONFIG_ROOT/defaults.json" "$PWD/build/bin" <<'PY'
import hashlib, json, pathlib, sys
config, binaries = map(pathlib.Path, sys.argv[1:])
data = json.loads(config.read_text())
for stage in ("build", "shuffle", "search"):
    binary = (binaries / ("alliance_" + stage + "_disk")).resolve()
    data[stage].update(binary=str(binary), sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
data["build"]["verifier_binary"] = str((binaries / "alliance_verify_layout").resolve())
data["shuffle"]["collect_binary"] = str((binaries / "alliance_collect_landing").resolve())
data["shuffle"]["cache_binary"] = str((binaries / "alliance_prepare_static_cache").resolve())
config.write_text(json.dumps(data, indent=2) + "\n")
PY
```

Review the copied dataset paths and parameters before launching. Rebinding records
which build is selected; it does not replace validation of that build or its inputs.

## Shuffle storage and memory

The default is compact in-memory adjacency. `--shuffle_storage auto` selects
`memory` or `external` after counting each group's actual edges, using
`--shuffle_memory_mib N` as the admission budget. Explicit modes are also supported.
External scratch uses aligned `O_DIRECT` + libaio batches; BlockBNF decisions and
persistent formats are identical. The planning budget is separate from a cgroup
hard cap. See the [examples, call tree and memory/I/O contract](docs/shuffle_memory.md)
and [implementation map](src/alliance/shuffle/README.md).
Future SIFT100M validation explicitly uses `memory`, no memory cap, and RSS
observation only; it does not use `auto`. The 600 MiB external example targets the
1.57M-node fixture, not all of SIFT100M.

## Quick start: build, shuffle and search

Run from the repository root. Supported dataset names are `arxiv2m`, `sift1m`
and `deep10m`. Each output directory must be new; `--index-dir` consumes the
previous stage's `index.json`.

| Function | Script | Native executable |
| --- | --- | --- |
| Build | `scripts/build.sh` | `alliance_build_disk` |
| Shuffle and workload cache | `scripts/shuffle.sh` | `alliance_shuffle_disk`, landing collector and cache preparer |
| Ordinary search | `scripts/search.sh` | `alliance_search_disk` |
| Formal benchmark | `scripts/benchmark.py` | Profile-bound driver; schema-3 uses `alliance_benchmark` |

Preview the build first. `--dry-run` checks configured paths and binary SHA values
and prints commands without executing them or creating outputs:

```sh
RUN="$HOME/alliance-runs/arxiv-check-01"
bash scripts/build.sh arxiv2m --config-dir "$CONFIG_ROOT/arxiv2m" \
  --defaults "$CONFIG_ROOT/defaults.json" --out-dir "$RUN/build" --dry-run
```

The following commands **execute** the Origin pipeline in order. Search requires
a navigation artifact compatible with the new graph/layout before its final
command: bind `navigation_file` in the selected index manifest or place
`navigation.bin` beside its `graph_file`. Building and shuffling do not generate
navigation. To test guided entries instead, set `parameters.entry_sampling` to
`"guided"` in the copied `search.json` before launching search.

```sh
bash scripts/build.sh arxiv2m --config-dir "$CONFIG_ROOT/arxiv2m" \
  --defaults "$CONFIG_ROOT/defaults.json" --out-dir "$RUN/build"
bash scripts/shuffle.sh arxiv2m --config-dir "$CONFIG_ROOT/arxiv2m" \
  --defaults "$CONFIG_ROOT/defaults.json" --index-dir "$RUN/build" \
  --workload origin --out-dir "$RUN/shuffle_origin"
bash scripts/search.sh arxiv2m --config-dir "$CONFIG_ROOT/arxiv2m" \
  --defaults "$CONFIG_ROOT/defaults.json" --index-dir "$RUN/shuffle_origin" \
  --workload origin --k 10 --efs 10,20,30,50,100 --repeats 1 \
  --out-dir "$RUN/search_origin"
```

Shuffle/search can also be previewed with `--dry-run`, but their referenced
upstream artifacts must already exist. A build preview does not create a usable
build directory. The example EF grid is a quick check, not a calibrated paper curve.

For P4 on the **already configured index**, omit `--index-dir`:

```sh
bash scripts/search.sh arxiv2m --config-dir "$CONFIG_ROOT/arxiv2m" \
  --defaults "$CONFIG_ROOT/defaults.json" --workload p4 --k 10 \
  --efs 10,20,30,50,100 --repeats 1 --out-dir "$RUN/search_configured_p4"
```

That P4 command uses `dataset.json` and `queries.json`, not the newly built
Origin index above. To test P4 on the new graph, first prepare a P4 static-cache
sidecar for its existing shuffled layout with
[`prepare_selectivity_static_cache.py`](scripts/prepare_selectivity_static_cache.py),
then bind that same index/quantization and the P4 sidecar in a separate configuration
or formal runtime binding. Do not re-shuffle the graph just to change the workload.
An Origin `index.json` cannot be reused by merely changing `--workload`: the
runner checks `cache_workload`.

## Adjust parameters

Edit a copied configuration; pass its directory with `--config-dir` and its shared
defaults with `--defaults`. The wrappers do not forward arbitrary native options
such as `--m` or `--wp`.

| File or option | Controls |
| --- | --- |
| `defaults.json` | Shared stage parameters, binary paths/SHA, environment and output roots |
| `dataset.json` | Dataset identity, configured index paths and default `memory_max_bytes` |
| `build.json` / `parameters` | `m`, `efc`, `wp`, `o`, `threads`, `build_mem_gb`, metric, partition and ClusterWoW settings |
| `shuffle.json` | `parameters` for BlockBNF/level geometry/threads; `static_budget_bytes` for the separate cache |
| `search.json` | Search parameters, beams, routing/cache policy, checkpoint and fingerprint |
| `queries.json` | Query vectors, ranges, GT and static layout for each workload |
| `efs.json` | EF arrays per workload and k, e.g. `"p4": {"10": [10,20,30,50,100]}` |
| CLI | `--workload`, `--k`, `--efs`, `--repeats`, `--index-dir`, `--out-dir`, `--dry-run` |

Each stage JSON recursively overrides its section in `defaults.json`; arrays
replace the entire array. Stage `memory_max_bytes` overrides the dataset default.
Search then takes configured index paths from `dataset.json`, workload inputs
from `queries.json`, and, when supplied, index/cache paths from `--index-dir`.
CLI EF/repeat values override their configured values. Missing EF combinations
use the fallback grid, so specify the grid explicitly for comparisons.

Keep algorithm memory budgets (`build_mem_gb`, `shuffle_mem_gb`) distinct from the
process cap (`memory_max_bytes`). Changing a cache budget does not resize an old
sidecar: prepare a matching one. Changing quantization requires matching codes,
centroids/CIDs and fingerprint. New run outputs belong outside the curated
paper CSV directory `result/`. See [configuration details](docs/workflows/dataset_workflow.md).

The ordinary dataset exporter currently uses a fixed `3bit` filename label.
Keep this workflow at 3-bit; use a separately validated formal profile for other
precisions so the exported data is not mislabeled.

## Formal single-thread benchmark

`search.sh` retains the ordinary **16-thread** workflow and computes a separate
median for each metric. Its EF values must be increasing, unique, positive
multiples of five and at most `64*k`. It is not the formal single-thread timer.

Use an explicit profile and runtime binding for the formal runner:

```sh
PROFILE=scripts/configs/datasets/arxiv2m_observer.json
BINDINGS=/absolute/arxiv-origin-p4-runtime.json
python3 scripts/benchmark.py --config "$PROFILE" --bindings "$BINDINGS" \
  --workload origin --workload p4 --algorithm allianceann --plan
python3 scripts/benchmark.py --config "$PROFILE" --bindings "$BINDINGS" \
  --workload origin --workload p4 --algorithm allianceann --validate
python3 scripts/benchmark.py --config "$PROFILE" --bindings "$BINDINGS" \
  --workload origin --workload p4 --algorithm allianceann \
  --run --job "$RUN/formal_origin_p4"
```

Create the runtime file using the [binding schema](docs/formal_benchmark.md): pin
the profile, native binary and every input/index/cache asset by SHA; specify each
curve's EF list and quantization fingerprint. The profile owns thread count,
memory/cache policy, repetitions and aggregation. `--plan` does not inspect
production inputs; `--validate` verifies them without queries; only `--run`
executes. Without explicit configuration, the frontend selects the CaseLaw catalog.
Do not compare the two entry points as if they used the same QPS protocol.

Both native search executables default to a prebuilt RAM navigation graph for
ML graph/auto search. They resolve `navigation.bin` beside `graph_file`, or beside
`ml_graph_file` for ML-only search without a raw graph. An explicit
`--navigation_file /absolute/navigation.bin` selects another compatible asset;
`--navigation_ef 64`,
`--navigation_budget 256` and `--navigation_count 16` control its traversal width,
maximum unique quantized scores and output entries. These are separate from the
disk-search EF grid. Count must not exceed either navigation EF or score budget;
EF and budget can each range up to 65536, with no ordering required between them.
Navigation replaces original/guided entries, reuses loaded
RaBitQ codes and adds its own resident memory; its index identity is checked
before queries. Loading a different graph/layout requires a matching navigation
file. Never copy an old graph's navigation file into a new build. This change
does not restore a navigation generator. Missing navigation fails before queries.
Explicit `--entry_sampling guided` retains guided sampling; `global` and `range`
also disable implicit navigation. Pure prefilter/legacy routes need no navigation.

For the formal dataset runner, omitted `defaults.navigation` means `true`
(64/256/16), and the current Arxiv profile explicitly selects it. Add a `navigation` asset role to each
curve's runtime binding. Pin that asset's SHA-256 like every other index input.
Use `false` with guided entry settings to retain guided sampling. Historical
LAION/CaseLaw profiles keep their explicit entry policies. See [entry policy](docs/search_policy.md)
for range behavior and validation requirements.

## Source and verification guides

Public headers live in `include/<feature>/`, private implementations in
`src/alliance/<feature>/`, tools in `src/tools/`. The namespace remains `alliance`.

- [Module map](docs/module_structure.md)
- [Build contracts](docs/workflows/build.md) and [execution map](src/alliance/disk_build/README.md)
- [Shuffle execution and artifacts](src/alliance/shuffle/README.md)
- [Search source map](docs/workflows/search_structure.md), [lifecycle](docs/search_lifecycle.md) and [policy](docs/search_policy.md)
- [Memory](docs/workflows/search_memory.md), [I/O budget](docs/workflows/search_io_budget.md) and [metrics](docs/workflows/search_metrics.md)
- [Regression tests](src/test/README.md)

Static cache offsets remain 32-bit (v2) while the selected payload fits. Only
actual overflow enables 64-bit offsets (v3), including matching query metadata
and budget accounting. See [static cache format](docs/static_cache_format.md).
