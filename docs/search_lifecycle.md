# Search lifecycle and compatibility

Search preparation, query execution and benchmark measurement have separate
owners. The implementation must preserve the selected filter/rerank algorithm,
cache boundaries, serialized formats and measurement protocol when reorganized.

Current file/function ownership is indexed in the
[search source map](workflows/search_structure.md). The query planner
selects the algorithm route; graph layer/entry selection is a distinct stage.
[Search policy](search_policy.md) defines candidate and bound controls.

## Preparation and ownership

The CLI converts tokens into a normalized search configuration. Compatibility
tokens are handled by the parser, not stored as query-runtime state. Preparing
quantization, guided entries, layouts and contexts is outside query timing.
Both native entry points use the same context configuration mapping.

The index owns shared layout mappings, quantization and immutable static cache.
Each context owns its worker, dynamic cache, visited state and I/O queues in
production ML search. Standalone/synchronous compatibility callers can still use
the thread-local worker. An owner/generation change invalidates graph cache and
visited state; retained pipeline/prefilter capacity is a separate lifetime.
I/O observers must outlive the contexts that reference them. Outstanding I/O
must drain before request objects or borrowed buffers are destroyed.

Normal graph traversal, entry scoring and bounded rerank share one prepared
query estimator. A separately called candidate API may own its estimator; a
borrowed estimator must match the query/index and outlive that call. Dynamic
cache clear invalidates occupied slots for sparse occupancy and uses a full
mapping clear for dense occupancy; it releases neither payload capacity nor
the query worker. FIFO/LRU eviction must keep slot ownership valid for both
clear paths. Fine-grained timing and co-block diagnostics remain opt-in and
are excluded from ordinary performance comparisons.

## Query execution

The query entry expresses these stages explicitly:

1. Validate storage and policy; resolve empty, graph, prefilter or legacy route.
2. Construct the query estimator only when required by the selected route.
3. Execute range-code scanning or graph candidate traversal.
4. Merge/deduplicate graph candidates and apply the selected rerank budget.
5. Run ordinary exact-distance rerank or the existing bounded policy.
6. Sort by the existing exact result order and return Top-k.

Graph traversal keeps discovery, stable candidate storage, EF ordering and the
rerank reserve distinct. Pending I/O refers to stable storage indices. Cache
requesters/waiters and co-block prefetch retain their different insertion rules.
Postfilter underfill can run a second constrained traversal, so final candidate
deduplication is required. Bounded rerank uses probabilistic RaBitQ bounds; the
ordinary path remains the exact-distance oracle over the retained candidates.

Successful completion means both a completion event and the full requested byte
count. A short or negative completion must fail before decoding a graph block
or evaluating a vector. Exception guards still drain other outstanding requests.

## Measurement contracts

| Contract | Ordinary ML CLI | Formal native benchmark |
| --- | --- | --- |
| Batch timer | OpenMP team, query calls, label extraction and worker publication | Serial calls plus accepted-AIO observation/source checks |
| Cache allocation | Existing lazy worker allocation | Empty cache/visited prepared before timer |
| Correctness/archive | After batch timer | After batch timer, then independent Python replay |
| Memory | Process/lifetime peak reporting | Explicit loading/search phase or whole-process protocol |
| Input counts | Historical CLI compatibility behavior | Exact expected rows, dimensions and GT validation |

These protocols must not be silently merged. Graph/vector requests retain their
actual page sizes. A 4 KiB equivalent is total requested bytes divided by 4096,
not a physical-device I/O count. Native counters and the independent observer
remain separate checks. Each EF/repeat invalidates dynamic cache; clearing
between queries remains an explicit profile choice. Archives marked as sent or
measured must never be reissued by a structural refactor.

## Retained compatibility and retired functionality

The synchronous/raw asynchronous paths, legacy raw-vector rerank and v2/v3
layout readers remain available for existing consumers and compatibility tests.
Inline rerank remains retired; disabled-only old tokens do not enable it.
Landing histograms are prepared by `alliance_collect_landing` using structural
routing without ANN queries. The old query-executing landing collection mode
is retired; offline histogram input to static-cache preparation is retained.

## Separate performance work

Do not change allocation timing, cache retention, I/O submission order or
candidate tie breaks while moving code. Further performance candidates are:

- Avoid duplicate estimator initialization in raw/synchronous compatibility
  paths, after verifying their independently callable candidate APIs.
- Measure temporary allocation cost before moving containers into workers;
  retained capacity changes search RSS and cold-query behavior.
- Audit O(N) mappings before SIFT100M: each uint64 array of 100M elements costs
  about 0.745 GiB. Narrowing representations needs checked capacity limits and
  format-independent loading; it is not a disk-format cleanup.

SIFT100M planning uses M=48, wp=7. This does not launch an experiment or alter
existing dataset profiles. Performance conclusions require a separate measured
comparison after structural and correctness acceptance.
