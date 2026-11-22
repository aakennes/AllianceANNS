# Script entry points

The repository [quick-start guide](../README.md) covers compilation, binary SHA
bindings, stage commands and parameter edits. Run commands from the repository
root; the supported dataset wrappers are:

| Function | Entry |
| --- | --- |
| Build | `bash scripts/build.sh DATASET` |
| Shuffle and workload cache | `bash scripts/shuffle.sh DATASET` |
| Ordinary 16-thread search | `bash scripts/search.sh DATASET` |
| Formal single-thread benchmark | `python3 scripts/benchmark.py --config PROFILE --bindings RUNTIME` |

The first three use `run_dataset.py` with `--config-dir`, `--defaults`,
`--out-dir` and optional `--index-dir`. Use `--dry-run` to inspect the effective
commands before execution; upstream index artifacts must exist even for a
preview. Native build/search parameters belong in the stage JSON files, not in
extra wrapper CLI flags.

Ordinary ML graph/auto search defaults to RAM navigation (EF 64, score budget
256, output count 16). `run_dataset.py` checks `navigation_file` from the selected
index manifest, or `navigation.bin` beside its raw graph (beside the ML graph
when ML-only and no raw graph is bound). `--index-dir` discards any navigation
path belonging to the previously configured index. Missing assets fail during
preflight, including `--dry-run`. Set `parameters.entry_sampling` to `"guided"`
in the copied `search.json` for explicit guided sampling. Build parameters are
independent of search entry policy.

`disk_search_common.sh` forwards `--entry_sampling` and `--navigation_file`,
`--navigation_ef`, `--navigation_budget`, `--navigation_count`. Its legacy
`--dry_run` previews navigation paths without requiring files; `--skip_search`
does not require navigation. Pure prefilter/legacy defaults need no navigation.
Every new graph/layout needs compatible navigation; an old graph's file cannot
be reused directly. No navigation generator is restored by this change.

An index manifest binds its static-cache workload. Preparing an Origin layout
does not prepare P4's landing sidecar. Reuse the shuffled graph, prepare the
requested workload cache separately, and bind the matching files.

`benchmark.py` plans by default. Its explicit profile and runtime select the
binary, inputs, caches, memory, EF grid and aggregation. `--validate` runs no
queries; `--run --job NEW_DIRECTORY` executes. Omitting configuration selects
the CaseLaw catalog. The ordinary search median and formal QPS-paired trimming
are different protocols.

- [Configuration and stage options](../docs/workflows/dataset_workflow.md)
- [Entry ownership and supporting helpers](../docs/workflows/entry_points.md)
- [Formal profile and runtime schema](../docs/formal_benchmark.md)
- [Executable script regressions](../src/test/README.md)

`scripts/tests/` contains active regressions. Data preparation and artifact/cache
validation helpers remain separate from the four launch entries. New run outputs
must not overwrite the paper CSVs in `result/`.
