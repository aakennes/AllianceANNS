# CaseLaw frozen protocol plans

`../../plan_caselaw.py` validates the bundled configuration evidence and prints a
plan. It does not import a historical runner, inspect production data, compile,
create caches, generate ground truth, warm up, or start queries. There is no
`--run` option. Python 3.7 or newer and only its standard library are required.

From the repository root:

```bash
python scripts/plan_caselaw.py --dry-run
python scripts/plan_caselaw.py --validate --workload p1 --workload p10
python scripts/plan_caselaw.py --workload origin --paths /path/to/roots.json
```

`--validate` and `--dry-run` both perform the same read-only validation, including
SHA256 checks of the bundled request, EF and native-option evidence. The default
is `--dry-run`. Neither option checks that a server binary or dataset exists.
The plan reports `execution_supported: false` and `production_inputs_checked:
false`; a valid plan is not evidence of a runnable or relocated driver.

An optional roots JSON changes paths printed in the plan:

```json
{"experiments": "/mnt/frozen/experiments", "data": "/mnt/datasets/CaseLaw"}
```

Both roots must be absolute. Overrides do not rewrite historical source code or
its embedded paths. `historical_driver_dependencies` records the original
driver identities and hashes without duplicating their executable source here;
`verified_configuration_sources` records
the exact snapshot hashes and their historical locations. Preserve the original
experiment directories until those dependencies are deliberately migrated.

## Included profiles

This is a selected set of frozen protocols, not a catalog of every final curve.
The separate `caselaw_equal_rss3_final_20261108` experiment is not incorporated.
Here `landing`/`equal` apply only to AllianceANN; PipeWoW and PipeANNFilter retain
their own native cache configuration and are labeled `native` in the plan.

| Profile | Formal source | Algorithms / AllianceANN cache | Prefilter threshold | Top-layer postfilter |
|---|---|---|---:|---:|
| origin | `caselaw_origin_rss3_fixed20_r5_trim3_20261106` | AllianceANN, PipeWoW / landing | 8192 | 1 |
| p1 | `caselaw_p1_postfilter_off_final_20261107` | AllianceANN, PipeWoW, PipeANNFilter / landing | 8192 | 0 |
| p4 | `caselaw_p4_p7_p10_final_20261108/p4` | AllianceANN, PipeWoW, PipeANNFilter / landing | 8192 | 0 |
| p7 | `caselaw_p4_p7_p10_final_20261108/p7` | AllianceANN, PipeWoW, PipeANNFilter / landing | 8192 | 0 |
| p10 | `caselaw_p10_graph_final_20261108` | AllianceANN / landing and equal | 1 | 0 |

P10 uses the later graph-only protocol, not the earlier auto-prefilter P10
curve. P1 and P10 require zero prefilter queries. Origin's top-layer postfilter
was inherited from the frozen native option default; that source snapshot is
included instead of silently assuming the newer disabled setting.

Each algorithm/cache curve has its exact 20 frozen EF values and five measured
repeats. The P1/P4/P7 grids come from their formal `coverage.json` metadata,
downloaded read-only from 9030; no pilot or formal query was rerun. P10's shared
grid comes from its formal request and `selected_efs.json`. Values are checked
against the bundled evidence instead of being regenerated from interpolation.

## Protocol boundaries

- All profiles use 1 bit, 1 thread, beam 8, k=10, L2, and the same 6734-query
  sequence across repeats/EFs for their workload. Each workload retains its
  own ranges and ground truth.
- Loading uses a separate 9 GiB cgroup `memory.max` and no swap. Before timed
  search the frozen driver releases that cap. Search is accepted only when its
  per-batch reset RSS peak is at most 3 GiB. This is not a 3 GiB cgroup limit.
- AllianceANN uses an 8 MiB dynamic cache. The 800 MiB static budget and the
  baseline algorithm budgets are bound to their formal request fields; a
  budget is not the same as measured resident bytes.
- Dynamic cache content survives between queries in a batch and is empty at
  each EF/repeat boundary. Static caches are reused. Auto-sizing and ANN warmup
  are disabled. PipeANNFilter's budget is static; it has no dynamic cache budget
  in these profiles.
- All five repeats must pass validation before trimming. `qps-paired-trim`
  ranks by `(qps, repeat ID)`, discards the lowest/highest, and averages the same
  three repeat rows for every metric. P95 is the mean of batch P95 values,
  not a percentile of pooled queries.
- Native batch timing includes the observer and in-batch GT validation;
  loading, raw output export and independent exact-L2 replay are outside it.

`profiles.json` has shared protocol defaults plus workload overrides. Its
`request_bindings` cross-check parameters and EF grids with immutable snapshots.
Change a profile only when selecting a different explicitly identified protocol;
path relocation belongs in `--paths`, not in frozen evidence.

## Statistics and legacy compatibility

`../../benchlib/metrics.py` exposes two different policies:

- `metric-wise-trim`: sort each metric independently and discard one value at
  each end. Its `trimmed_mean` alias preserves the historical benchmark formula.
- `qps-paired-trim`: retain the same QPS-ranked three repeats for all metrics,
  and return retained/dropped repeat IDs for an audit.

The navigation-era `benchmark_allianceann.py` launcher has been retired.
Read-only log interpretation remains in `benchlib/legacy_log.py`; it cannot
start queries. The metric tests compare the named legacy aggregation rules
without importing a retired launcher. `run_dataset.py` retains its separate
16-thread/metric-wise median protocol. Neither helper migration nor current
source cleanup changes the CaseLaw protocol tables, request snapshots or
recorded results above.

For execution, [benchmark.py](../../benchmark.py) defaults to this historical
adapter. New dataset-native profiles require explicit runtime bindings and a
pinned current binary; see [formal benchmark](../../../docs/formal_benchmark.md).
The read-only `plan_caselaw.py` behavior described above remains unchanged.

`../arxiv2m/benchmark.json` and `benchmark_landing_equal.json` are byte-identical
at this baseline (SHA256
`c4094a1ef2c1901a1e00982dccfc5573ac90213908268f524335ca48e7631013`).
Both filenames remain for compatibility with historical invocations; no alias
schema or implicit config inheritance was introduced in the legacy loader.

Run the synthetic/configuration tests without ANN data:

```bash
python -B -m unittest discover -s scripts/tests -p test_benchmark_metrics.py -v
python -B -m unittest discover -s scripts/tests -p test_caselaw_profiles.py -v
```

The tests exercise disagreeing QPS/IO order, ties, invalid repeats, phase
semantics, cache boundaries, frozen evidence, root overrides and a read-only CLI.
