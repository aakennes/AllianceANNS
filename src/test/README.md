# Test responsibilities

Tests are opt-in (`-DBUILD_TESTING=ON`). Production targets never include this
directory. CTest names are stable across the directory migration, so existing
test selections continue to work.

| Directory | Owner |
| --- | --- |
| build | Input, build budgets, cluster/WoW builders, sessions and build CLI |
| storage | Pages, metadata, edge representation and I/O/index lifetimes |
| layout | Multi-level layout export and shuffle contracts |
| quantization | Packed codes, scalar bounds and quantization reload |
| search | Entry selection, cache, routing, rerank, metrics and workspaces |
| benchmark | Native benchmark input contract |
| integration | Mixed-page/ML-only search and build completion/artifact workflows |
| support | Assertions, observation peers, fault reference and test I/O observer |

The scan-only `test_cluster_select` suite is retired. Build CLI, budget,
completion, artifact and resource-lifetime fixtures exercise LeftStream and
ClusterWoW/Vamana. Search-quantization persistence/reload tests remain separate
and must not be removed with build-side quantization.

Script-level Python suites, including `test_disk_scripts.py` and the repository-structure
checks, live under `scripts/tests`. Native CLI and artifact-integration Python
fixtures remain alongside their owning suites in `src/test`. CTest registers
these from the test root and the benchmark/integration groups. There is no
separate root `tests/` directory. Use `ctest --test-dir build-tests -N` for the
configured test count. Platform-specific I/O and `malloc_trim` tests remain
conditional, so compare test inventories on the same platform.

Example commands from the repository root:

```sh
cmake -S . -B build-tests -DBUILD_TESTING=ON
cmake --build build-tests -j 6
ctest --test-dir build-tests --output-on-failure
ctest --test-dir build-tests -L search --output-on-failure
ctest --test-dir build-tests -R '^test_mixed_pages$' --output-on-failure
```

Categories describe ownership, not mutually exclusive coverage. A test can have
multiple CTest labels. Release assertions, per-test temporary directories,
resource locks and link-time failure injection remain enabled. The test observer
is distinct from the formal benchmark observer; their contracts differ.

Search behavior is defined in [search policy](../../docs/search_policy.md) and
[ownership/lifecycle](../../docs/search_lifecycle.md). Retained regression
contracts include:

| Suites | Independent checks |
| --- | --- |
| `test_rerank_contract`, `test_rerank_estimator` | Exact full-pool top-k, ties/invalid bounds, page and ID deduplication, scalar L2/IP quantizer factors |
| `test_prefilter` | >4096 IDs, repeated attributes, bounded selection versus full sort, R>=S exact control, S<k, empty/inverted ranges, threshold equality and missing/truncated storage |
| `test_multilevel_search`, `test_entry_sampling` | Unique admissions under repeated entries, low-EF cold/warm cache, borrowed/owned estimators and supplemental entries |
| `test_memory_navigation` | Historical reader compatibility, source identity, frozen navigation results/counters, score budgets, range constraints and malformed files |
| `test_cache_contract` | FIFO/LRU operations against an independent model, sparse/dense/empty clear and zero/nonzero reconfiguration |
| `test_io_handler_failure`, `test_io_operation_lifetime` | Partial submission/read/allocation/distance failures, draining before release and successful reuse |
| `test_mixed_pages`, `test_variable_pages`, `test_ml_only` | Actual page geometry, standalone layout loading, exact results and logical I/O |
| `test_shuffle_contract`, `test_layout_export` | Identity/BlockBNF, deterministic order across scoring workers/batches, independent edge-span oracle and exported mappings |

Use a filesystem supporting O_DIRECT for storage tests. Generic seeded-threshold
scheduler tests remain valid independently of the retired inline-rerank path.
Correctness tests do not establish real-data QPS or a universal probabilistic
rerank recall guarantee.

Review retention using assertions, callers and recorded test executions. File
modification dates do not establish whether a test has recently run. A failure
path that triggers rarely can still protect data integrity and should not be
deleted merely because its normal executions pass.
