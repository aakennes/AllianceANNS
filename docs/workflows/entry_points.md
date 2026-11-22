# Supported entry points

| Entry | Responsibility |
| --- | --- |
| build.sh / run_dataset.py build | Dataset-configured disk build |
| shuffle.sh / run_dataset.py shuffle | Multi-level layout and static-cache preparation |
| search.sh / run_dataset.py search | Dataset workflow: 16 threads, per-metric median across repeats |
| benchmark.py | Plan by default; explicit validation/run/resume with pinned profiles and runtime bindings |
| finalize_build.py | Verify completed build before cleaning optional vector copies |
| prepare_selectivity_static_cache.py | Validated per-workload landing/equal cache preparation |
| validate_shuffle_completion.py | Read-only validation of committed shuffle outputs before automatic reuse |
| examples/build_shuffle_smoke.py / examples/search_smoke.py | Synthetic acceptance examples |

`run_dataset.py` resolves its binary from the effective search configuration;
the current main-driver binding uses `alliance_search_disk`. `benchmark.py`
defaults to the frozen CaseLaw adapter; schema-3 profiles bind the current
single-thread `alliance_benchmark` through an explicit binary path and SHA.
Its source being in-tree does not silently replace an older pinned executable.
See [formal protocol](../formal_benchmark.md) and
[configuration workflow](dataset_workflow.md).

Dataset-specific shell wrappers have been removed. Use build.sh, shuffle.sh
and search.sh with a dataset argument. Automated regression runs through CTest
with BUILD_TESTING=ON. The lower-level disk_build_common.sh and
disk_search_common.sh helpers retain their separately tested CLI contracts;
they are not aliases for the JSON workflow.

The lower-level shell helpers support only `block_bnf` or `identity`, with 32
iterations by default and the fixed BlockBNF margin of 1. Layout export and
static-cache preparation are separate calls. `--skip_search` still prepares a
requested nonzero static cache, including per-selectivity landing statistics,
without running vector queries or reading ground truth. Automatic layout reuse
requires a complete matching `GRAPH.completion.json`; an explicit
`--skip_shuffle` retains compatibility with validated legacy layouts.

Old mixed memory/disk drivers (test_arxiv.sh, test_sift.sh, arxiv_build/search,
memory cluster/conversion drivers), flat-shuffle harness and RaBitQ/high-selectivity
memory benchmarks are retired. Choose a supported disk entry explicitly; its
defaults must be reviewed rather than translated from removed memory flags.

The navigation-dependent benchmark_allianceann launcher is removed. Its recorded
logs can still be parsed with benchlib.legacy_log. Historical PipeWoW,
PipeANNFilter and manual Arxiv comparison/repair launchers have also been
removed. Keep their paper CSVs separate from new experiment output; a data file
is not an executable profile. Frozen CaseLaw configuration and source records
remain available to validate the supported historical adapter.
