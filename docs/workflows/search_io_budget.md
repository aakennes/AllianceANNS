# Joint search and rerank IO budgets

`alliance_search_disk --io_budget_total B --threads T` partitions an aggregate
outstanding-request cap across this search process's workers. Other processes
are outside the cap. `0` preserves independent beams. This
option requires the multi-level pipeline and `--filter_mode graph`, `prefilter`
or `auto`. Legacy raw-vector fallback uses a different reader and is rejected
when a nonzero budget is requested.

Each worker gets `B / T` slots; the first `B % T` workers get one more. Graph
search and final exact-vector rerank independently use up to that share,
bounded by their requested beams. The phases do not overlap within one worker,
so widths are not added. Rerank starts after graph reads drain. A positive
budget requires `B >= T`; AIO queue allocation follows the effective widths.
Inline rerank has been removed and has no reserved slots or enabled mode.

Examples with requested search 16 / rerank 64:

| Total budget | Threads | Effective search / rerank per worker |
|---|---|---|
| 128 | 8 | 16 / 16 |
| 128 | 16 | 8 / 8 |
| 256 | 16 | 16 / 16 |
| 512 | 16 | 16 / 32 |

Fixed shares require no shared lock or token exchange in the normal search hot
path. Busy workers cannot borrow idle shares. The option is a request-count cap,
not a bandwidth limit or a universally optimal SSD setting. Changing traversal
width can change asynchronous discovery order; check Recall as well as QPS.

The shell entry points forward `--threads`, `--io_budget_total`, and
`--ml_cache_total_bytes`. `--cpu auto` keeps CPU 1 for a single worker and
inherits process affinity for multiple workers; an explicit CPU list is honored.
`--ml_cache_bytes` still means a per-worker payload cap. Use
`--ml_cache_total_bytes 47185920` to split 45 MiB across all workers.

## Profiling

Build separately with `ALLIANCE_PIPELINE_PROFILE=ON`. The `# io_queue` line adds:

- Time-weighted average graph/vector/combined submitted-but-unreaped requests.
- Observed aggregate peak requests, including separate graph and vector peaks.
- Remaining requests at batch end (expected zero).
- Time-weighted mean workers inside traversal and post-rerank phases.

Occupancy is measured from successful submission return to completion polling.
It includes completed-but-unpolled requests, omits device work before submission
returns, and is not hardware queue depth. Phase worker counts include computation
and scheduling time, not just IO.
The separate graph and vector peaks need not occur simultaneously.

The observer uses shared atomics only in profiling builds. Normal throughput
builds have no occupancy probes. Compare normal builds for performance and
measure the profiling overhead separately. Preserve query count, Recall target,
index, candidate budget and cache settings when scanning beam widths.
