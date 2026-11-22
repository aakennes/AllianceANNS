# Filter and final-rerank policy

This is the current policy reference for multi-level disk search. The ordinary
CLI normalizes options before loading inputs; the formal benchmark binds its
policy explicitly. Low-level C++ callers must set their desired context fields
rather than assume that CLI defaults are library defaults.

## Routing and candidate discovery

`query/plan.hpp::PlanQuery` chooses an empty, graph, range-prefilter or legacy
route. `entry_routing.cpp` chooses graph layers and entries; it does not select
the overall search algorithm. `graph`, `prefilter` and `auto` require the
capabilities checked by the query planner. `auto` compares actual range
cardinality with the inclusive positive `prefilter_threshold`.

Graph EF bounds traversal width. The main candidate heap and reserve retain
unique approximate candidates; reserve entries do not expand graph edges.
Highest-layer postfilter is an explicit context/profile setting. It applies
only when the landing is both the highest graph layer and the highest basic
layer of disk level zero, with cardinality at least floor(N/2). It navigates
stored layers without the range constraint but admits only predicate-valid
result candidates. An underfilled pool is retried with constrained traversal
before defensive deduplication. Lower landing layers keep constrained traversal;
small-range prefilter is chosen before graph traversal.

Prefilter scans every matching quantized code without graph expansion. With
`prefilter_rerank=0`, capacity is
`min(range_cardinality, max(k, max(1, candidate_factor) * ef))`; a positive
explicit budget overrides that capacity. Its final stage supports the same
exact or bounded vector rerank. Zero graph reads do not mean zero vector reads.
Forced prefilter requires a complete ML vector region; auto can choose graph
when that capability is absent. Empty/inverted ranges return no results, S<k
returns S unique results, and API calls with k=0 return no results. Quantized-code
enumeration uses 4096-ID chunks and O(4096+R) per-worker scratch.

## Guided entries, packing and candidate identity

The ordinary multi-level CLI defaults to RAM navigation for graph/auto search.
Explicit `--entry_sampling guided` selects guided entry sampling with 256 samples
and four supplemental entries. It reuses query-centroid scores, starts with
the nearest eight nonempty centroid rank lists and adds lists if their combined
eligible capacity is below 256. Quotas are shared equally subject to capacity;
rank sampling and quantized scoring select up to four entries alongside the
original layer entries. The actual sample count is bounded by range cardinality;
EF still bounds traversal admission. `entry_samples=0` disables supplementation.
Non-range predicates use the global-sampling fallback with predicate checks.

`GuidedEntryDirectory` belongs to the shared index. CLI preparation occurs
before timing; lazy API preparation remains possible. Quantization/CID changes
invalidate it and require exclusive index access. Its ID storage costs about
4*N bytes plus list metadata. Query-time rank lookup, quota allocation and
sample scoring remain inside query timing and distance-comparison metrics.
Unselected samples do not claim visited state.

### Default RAM navigation

For ML graph/auto search, both native executables default to `navigation.bin`
beside `graph_file`, or beside `ml_graph_file` for ML-only search without a raw
graph. `--entry_sampling navigation` selects this policy explicitly; guided,
global or range disables implicit navigation. Pure prefilter/legacy routes do
not enable it. An explicit `--navigation_file` still binds the requested asset.
Missing navigation fails before queries. A new graph/layout requires compatible
navigation and cannot directly reuse an old graph's file. This change does not
restore the navigation generator.

`--navigation_file` loads a historical NVGRAPH1 representative graph after the
search layout and quantization are ready, before query timing. The reader checks
the payload checksum, graph structure, population, dimensions, metric, rank/ID
order hash, window ladder and layout ownership. It does not load navigation
vectors: scoring reuses the search RaBitQ estimator. Loading another layout or
refreshing search CIDs invalidates the loaded navigation graph.

Navigation replaces both original range endpoints and guided supplemental
entries. It starts with up to four uniformly spaced representatives inside the
query's attribute-rank interval, then performs bounded best-first navigation in
RAM. Defaults are `navigation_ef=64`, `navigation_budget=256` unique scores and
`navigation_count=16` output entries, independent of disk-search EF. Controls
must satisfy `0 < count <= min(ef, budget)`; EF and budget are each at most 65536.
EF may exceed the scoring budget, preserving the historical CLI contract.
If a coarse group has no representative,
navigation checks its denser groups; a range with no representative fails rather
than silently changing entry policy. Predicates must be attribute ranges.

The native CLI accepts navigation only with the multi-level pipeline and
`graph`/`auto` routing. Auto-prefilter queries bypass entry navigation entirely.
Navigation allocates no guided directory, performs no SSD reads while selecting
entries, and has no endpoint fallback. Query-time navigation scores and CPU
time remain part of the measured search; its graph memory and per-worker scratch
count toward the same search memory limit. Guided-directory bytes and RAM
navigation bytes are reported separately.

The formal binding pins the navigation file SHA-256. Historical protocol
reproduction must still preserve the main graph/layout, cache, quantization and
timer settings; switching only an entry strategy does not recreate a different
historical index. The `test_memory_navigation` fixture independently emits the
legacy file layout, validates its source identity and damage rejection, and
compares search results/counters with frozen pre-refactor output.

Packed search records hold each node's RaBitQ planes and four float factors.
The base allocation is 64-byte aligned and record stride is rounded to 16 bytes.
`PackRaBitQForSearch` precedes static-cache loading, and `PrepareSearchCIDs`
builds the internal-ID lookup. Canonical metadata serialization is unchanged.
Packing and postfilter selection do not choose an exact-rerank policy.

Candidate `visited` means discovered, not expanded. First discovery fixes the
approximate score; traversal and result heaps only tighten their cutoffs, so a
rejected/evicted ID need not be scored again. Evicted traversal storage remains
alive until the query ends for pending I/O. Main/reserve capacities and EF
admission are separate. Co-block candidates obey cached-layer coverage; final
deduplication remains a safeguard. Discovery/duplicate counters count claims,
not duplicate result admissions. `ALLIANCE_VALIDATE_CANDIDATES=ON` adds contract
checks and must remain off for ordinary performance comparisons.

## Graph candidate budgets

| Control | Meaning |
| --- | --- |
| `--rerank_policy error_bound` | Ordinary ML CLI default; resolves candidate and bound controls |
| `--rerank_candidate_factor C` | Capacity `max(k, C*ef)`; does not enlarge graph traversal width |
| `--rerank_num R` | Explicit fixed absolute candidate budget, at least k |
| `--rerank_factor C` | Fixed budget C*k; conflicts with nonzero rerank_num |
| `--prefilter_rerank R` | Separate explicit prefilter capacity |

Explicit compatibility/ablation policies remain: fixed, step (2k when ef<=2k,
3k when ef<=8k, otherwise 4k), and linear
`min(4k, 2k + ceil(max(0, ef-2k)/7))`. Candidate-factor mode cannot combine
with an explicit fixed budget/factor or step/linear policy. With budget R,
main capacity is `min(ef,R)` and reserve capacity is the remainder. A capacity
never guarantees that the graph discovers that many eligible unique IDs.

After traversal, candidates are merged and deduplicated. Ordinary final rerank
scores every retained candidate by its exact vector distance. Candidates sharing
a vector page share one read. This is exact over the retained pool, not an
exhaustive-nearest-neighbor guarantee for undiscovered vectors.

## Error-bound final rerank

The ordinary ML CLI defaults are candidate factor 2, upper-bound prescreen on,
rank factor 1, lazy float refinement on, warm blocks 0 and rerank beam 8. Explicit
profiles/configurations override them. Direct context users keep bounds opt-in.
No query SQ is used: rotated float queries are scored against the database
RaBitQ bit planes using the native library kernel.

For smaller-is-better estimated distance d and model radius E, use L=d-E and
U=d+E. Scheme B first takes the rank-factor*k upper bound over final candidates.
It rejects only L above that threshold, then also uses the kth completed exact
distance after k eligible results exist. Equality retains ties; invalid intervals
force exact reads. Only pages containing surviving candidates are submitted.

The stored model factor includes the bit-dependent error and library
`kConstEpsilon=1.9`. It is multiplied by the appropriate query norm, with a
numerical guard; do not divide by bit count or add a removed alpha multiplier.
Lazy float refinement recomputes ambiguous scores before I/O using the same
kernel. It does not remove a query-SQ error term or guarantee a narrower bound.
For IP, the centroid correction projects the model error onto q-c and the score
convention is aligned with exact `1-IP`; tests use nonzero centroids.

Bounds are probabilistic. Audit loss against exact top-k on the same frozen
candidate pool; asynchronous traversal can otherwise change that pool. The
earlier SQ8-navigation experiments do not establish correctness for the current
float-only path or provide a universal recall guarantee.

`rerank_warm_blocks` caps outstanding vector reads before k exact results exist;
0 uses the full beam immediately. `rerank_batch_blocks` limits requests per
submission syscall, not queue depth. Graph traversal and final vector rerank do
not overlap within a worker: [I/O budgeting](workflows/search_io_budget.md)
uses their maximum, not their sum.

For a same-pool full-exact control, explicitly select fixed policy with the same
candidate factor and disable bound pruning/prescreen; preserve every other flag.
For example, factor 1 compares the retained `max(k,ef)` pool, not all discoveries.

## Compatibility and verification

Inline rerank, `rerank_bound_alpha` and `rerank_sq_refine` are retired. Disabled
inline spellings are stateless compatibility tokens; enabling/tuning inline
fails. Active commands omit them. Search-side quantization, final exact rerank
and v2/v3 readers remain supported.

`test_rerank_contract` covers candidate-pool stream prefixes and bounded-scheduler
contracts. `test_rerank_estimator`, `test_multilevel_search`, `test_prefilter` and
`test_search_options` cover actual L2/IP factors, integrated search, routing and
option conflicts. I/O lifetime/failure and mixed-page tests cover drain and byte
geometry. The [test index](../src/test/README.md) maps these contracts to their
executable regression suites.
