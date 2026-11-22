# Shuffle implementation map

The executable entry is `src/tools/layout/shuffle_disk.cpp::main`; argument
validation and job orchestration are in `src/tools/layout/shuffle/options.cpp`
and `runner.cpp`. The public configuration and API are under `include/shuffle/`.
See [memory/storage modes and runnable example](../../../docs/shuffle_memory.md).

- `stages/`: level planning, one-group-at-a-time ordering, export verification and
  manifest-last publication. `ShuffleArtifacts` owns locks and output staging.
- `order/`: BlockBNF policy and deterministic score/commit rounds.
- `adjacency/`: exact edge representation, memory/external row access, bounded
  incoming transpose and disposable scratch I/O.
- `memory/`: sparse per-worker scores and external admission/buffer accounting.
- `format/`: pure geometry and persistent layout writers.
- `diagnostics/`: optional statistics; never selects a permutation.

```text
Run -> ComputeMultiLevelShuffleOrders -> ShuffleBlockBNF
  BuildWeightedGraph (count -> PlanShuffleStorage -> fill) -> SortNodesByInWeight
  [external only] ExternalGraph::Prepare -> IncomingSorter
  OrderedBlockBNF::Round -> [external only] LoadBatch -> score -> ordered commit
Run -> ExportMultiLevelLayout -> VerifyLayoutExport -> ML-only acceptance
    -> ShuffleArtifacts::Commit
```

`memory` remains the default. `auto` resolves each group by the supplied budget.
`external` uses O_DIRECT/libaio scratch and stores adjacency on the output SSD and
reads bounded outgoing/incoming batches; metadata/maps/permutations stay in RAM.
Storage mode is independent of the ordering policy. `identity` and `block_bnf`
remain the only methods; BlockBNF keeps 32 default rounds and fixed margin=1.
There are no new public search/layout formats or cache-policy switches.

Output graph and metadata must use fresh names in one directory. The job stages
files privately, verifies every exported vector/edge/mapping, copies unchanged
metadata, then checks ML-only loading without issuing queries. All three output
files are synchronized and hashed before `GRAPH.completion.json` is published.
Readers require the complete receipt, not just existing output files. This is a
manifest-last multi-file protocol, not a single atomic directory rename.
Exceptions clean owned temporary data; crash residues remain attributable to the
job. Source graphs are never removed. External group scratch is likewise scoped
to this transaction and removed after consumption.

The layout writes `static_flag=0`. Separate cache preparation owns landing/equal
allocation. Existing v2/v3 readers retain compatibility with older layouts.
