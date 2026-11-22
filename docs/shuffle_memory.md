# Shuffle memory and external adjacency

## Entry points and policy

`src/tools/layout/shuffle_disk.cpp::main` calls `Options::Parse` and `Run` in
`src/tools/layout/shuffle/`. `scripts/shuffle.sh` uses the JSON dataset runner and
also prepares workload caches; use the native executable for a shuffle-only test.
The default storage is `memory`. Select `external` explicitly or request `auto`; ordering remains
BlockBNF with the same fixed node order, affinity, reciprocal tie break, page
capacity, margin=1 and convergence threshold. Persistent graph/layout formats,
original IDs, vectors, attributes and output permutations are unchanged.

```sh
# Existing raw graph and metadata; OUT is a fresh directory on the chosen SSD.
mkdir -p "$OUT"
systemd-run --user --scope --quiet -p MemoryMax=629145600 -p MemorySwapMax=0 \
  /usr/bin/time -v build/bin/alliance_shuffle_disk \
  --graph_file "$GRAPH" --meta_file "$META" \
  --out_graph_file "$OUT/ml_graph.bin" --out_meta_file "$OUT/ml_meta.bin" \
  --space l2 --method block_bnf --level_k 2 --level_m 2 --shuffle_o 2 \
  --block_size 4096 --vector_block_size 4096 --max_iterations 32 \
  --shuffle_threads 32 --affinity_batch_size 16384 \
  --shuffle_storage external --shuffle_memory_mib 600
```

600 MiB is 629145600 bytes. `shuffle_memory_mib` is a selection/admission
budget, not an OS limit. `auto` and `external` require it. `memory` accepts an
optional budget; omit it for unlimited RAM mode. The effective budget is the
smaller of the requested value and any readable inherited cgroup-v2 limit.

`auto` measures loaded RSS and counts actual edges, maximum degree and exact
codec width for each group before allocating adjacency. It estimates overlapping
maps, score buffers, remaining node arrays and a reserve (5%, at least 64 MiB).
It chooses RAM when that estimate fits, otherwise external when its O(N) working
set and bounded buffers fit, otherwise fails with no adjacency allocation. Groups
may choose different backends. `[shuffle-storage-plan]` records both estimates,
the effective budget and the choice. This is conservative admission, not a strict
RSS guarantee or a mid-round spill/retry mechanism.

For automatic selection, replace the example's storage with `auto`; keep the
600 MiB cgroup only when testing that hard cap. For the later SIFT100M shuffle:

```sh
# Only after the source build is accepted; keep all other geometry parameters.
/usr/bin/time -v build/bin/alliance_shuffle_disk \
  --graph_file "$GRAPH" --meta_file "$META" \
  --out_graph_file "$OUT/ml_graph.bin" --out_meta_file "$OUT/ml_meta.bin" \
  --space l2 --method block_bnf --level_k 2 --level_m 2 --shuffle_o 2 \
  --block_size 4096 --vector_block_size 4096 --max_iterations 32 \
  --shuffle_threads 32 --affinity_batch_size 16384 \
  --shuffle_storage memory --shuffle_mem_gb 0
```

Do not pass `shuffle_memory_mib` or set a cgroup hard cap for that SIFT100M run.
`shuffle_mem_gb=0` disables the older RSS failure threshold. This command is a
future policy, not authorization to resume the paused build. Preserve native
phase RSS logs and process RSS/HWM samples; GNU time alone misses phase resets.
Ancestor limits still apply even when this task installs no limit.

For JSON workflows put storage/budget flags in `shuffle.json.parameters`.
`memory_max_bytes: null` explicitly disables the native shuffle scope cap and
records RSS only. Remove/override `shuffle_memory_mib` to null and set
`shuffle_mem_gb` to 0. The combined wrapper requires a separate positive
`cache_memory_max_bytes` so this choice does not uncap later cache preparation.
Build and search limits are unchanged. Native shuffle-only tests do not prepare
caches or issue queries.

## Call flow and ownership

```text
main -> Options::Parse -> Run
  ShuffleArtifacts: source lock, output lock, private staging directory
  ComputeMultiLevelShuffleOrders: one group at a time; retain final permutations
    ShuffleBlockBNF
      BuildWeightedGraph: count unique edges/weights -> plan -> fill adjacency
      SortNodesByInWeight: freeze processing order -> release incoming totals
      memory: keep packed CSR
      external: ExternalGraph::Prepare
        gather outgoing rows in processing order
        IncomingSorter: bounded sorted runs -> bounded fan-in merges
        stream incoming source IDs by (processing rank, source ID)
        release raw CSR scratch and original-order offsets
      OrderedBlockBNF::Round (same decision kernel for both backends)
        external LoadBatch: sequential outgoing + incoming payload
        parallel scoring against immutable previous-round mapping
        ordered capacity commit with exact reciprocal lookup
      destroy current-group adjacency and scratch
  ExportMultiLevelLayout -> full VerifyLayoutExport -> ML-only loader acceptance
  ShuffleArtifacts::Commit: synchronize and hash outputs; publish receipt last
```

| Private component | Responsibility |
| --- | --- |
| `order/block_bnf.cpp` | Sequence one group's CSR, fixed order and rounds |
| `order/ordered_block_bnf.hpp` | Scoring, ties and deterministic capacity commit |
| `adjacency/weighted_graph.*` | Count/fill source adjacency; expose immutable row views |
| `adjacency/edge_codec.hpp`, `weighted_edges.hpp` | Shared exact codec; RAM storage |
| `adjacency/external_graph.*` | Outgoing/incoming streams, batch views and hub fallback |
| `adjacency/external_sort.*` | Bounded runs and multi-pass incoming transpose |
| `adjacency/scratch_io.*` | Logical ranges, alignment/RMW, scoped scratch and logical I/O counters |
| `adjacency/scratch_aio.*` | Thread-local libaio contexts, aligned buffers, submit/completion validation |
| `adjacency/neighbor_filter.hpp` | Negative-only membership filter; positives still checked |
| `memory/sparse_scores.hpp` | Degree-bounded worker scores; first-touch order preserved |
| `memory/storage_plan.hpp`, `runtime.hpp` | Pure mode selection, loaded RSS and inherited budget |
| `memory/external_budget.hpp` | Bounded sort/stream buffer policy |

Paths in the table are relative to `src/alliance/shuffle/`. Public configuration
is in `include/shuffle/config.hpp`; external scratch/codec classes are private.

## Memory and I/O behavior

RAM adjacency packs ID and `(weight-1)` into 32 bits when the observed ID/weight
ranges fit, with an exact 5-byte ID/weight fallback. The external codec uses the
same rule; it never changes or approximates a weight. Actual multiplicity is
counted, including repeated neighbor slots. Per-worker scores scale with maximum
unique degree rather than the global block count. Input buffers and incoming
weight sums are released after their last consumer.

External mode keeps offsets, node permutations/maps and 16-byte membership
filters resident. It does NOT load all edges or mmap the whole file. Preparation
uses 32 MiB sort runs, at most 32 merge readers of 64 KiB, and buffered writers.
Every round reads at most 16 MiB each of outgoing and incoming payload per batch.
If one incoming hub row exceeds that bound, it uses exact positional binary
search for that row. Outgoing rows too large for the fixed batch cap are rejected.
No partition-local approximation, changed node order or reduced iteration count
is used to make the budget fit.

The one-time outgoing gather can issue random positional reads of temporary CSR;
the normal round path streams both directions sequentially. The reciprocal test
`u in outgoing(v)` becomes the exactly equivalent `v in incoming(u)`. A large hub
fallback can issue random reads. Scratch payload uses `O_DIRECT` and native libaio with 4 KiB alignment,
up to 64 in-flight requests per thread and a 2 MiB aligned buffer per context.
The preparation gather batches up to 64 rows. Requests are fully reaped before
consumption; this is batched asynchronous I/O, not an inter-batch CPU/I/O pipeline.
Unaligned writes preserve boundary sectors with serialized read/modify/write;
partial submission, interrupted waits, short completion and errors are checked.
Metadata operations (`ftruncate`, `fdatasync`, open/close) remain synchronous.
There is no buffered scratch fallback if direct I/O is unsupported.

`[shuffle-external-io]` counts logical payload calls/bytes.
`[shuffle-external-aio]` separately counts submitted requests, aligned bytes,
completed bytes, depth and errors. Alignment, padding, sector re-reads and RMW
can increase physical request bytes. Neither counter is a device-I/O count or
search I/O. Original graph reads and final export retain their existing paths.

Scratch is under the output transaction on the selected SSD, never pmem. One
group owns its files. Successful consumption releases sorted runs and raw CSR;
finishing the group removes both streams. Exceptions clean owned scratch and
retain the outer failure record. A SIGKILL may leave the job's private staging
folder for diagnosis; never delete unrelated inputs to recover it.

## Measurement and acceptance

Use the existing 1,572,864-node M48/wp7/o4 fixture for the 600 MiB trial. Require
all regression tests, scalar per-round equality, exact RAM/external decisions,
full output readback and byte-identical graph/layout/metadata against the frozen
RAM result. Include codec boundaries/fallback, multi-pass sort, oversized hubs,
truncated files, cleanup and admission rejection tests. Verify memory.events
and oom/oom_kill=0. Observe actual RSS and scratch disk peak while running.

Native `[RSS]` logs label MiB as MB and reset VmHWM after each phase. The final
GNU time high-water value therefore is not the lifetime maximum: retain all
phase peaks and independent process samples. Report cgroup peak (including file
cache), temporary I/O, preparation/round/export/verification time and total wall
time separately. An old RAM run at 4 GiB and an external run at 600 MiB are one
controlled-cap comparison, not replicated proof of a speedup or slowdown.

## Scale limit and follow-up

600 MiB applies to the small fixture, not SIFT100M. External edges remove the
E-sized resident CSR, but the O(N) metadata, maps, filters, offsets and retained
permutations still require multiple GiB at N=100M. The admission model rejects
infeasible fixed working sets before allocating adjacency. The original full
SIFT100M builder/checkpoint remains paused and unchanged during this trial.

Further work may externalize permutations/offsets or improve stream preparation,
after measurement. Those are separate changes; no full SIFT100M shuffle is
implicitly enabled by this implementation.
