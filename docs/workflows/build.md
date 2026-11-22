# Disk build contracts and ownership

Production entry: `alliance_build_disk`, implemented by `src/tools/build`.
Current source organization is described in [module structure](../module_structure.md).
Historical experiment presets, binaries, artifacts and measurements are independent of this source cleanup.

## Lifecycle

```text
build_disk.cpp::main
  -> DiskBuildOptions::ParseAndValidate
  -> RunDiskBuild
       report-only -> ReportDiskBuildMemory (no index construction)
       fresh -> probe shape / attributes / construct / optional search-code initialization / PopulateElements
       resume -> load checkpoint / restore tuning / validate anchor / ResumeStartLayer
       PreparedDiskBuild (owns index and borrowed external partition arrays)
       ExecuteBuild -> resident LeftStream -> ClusterWoW/Vamana
       optional raw-vector export -> Save final metadata
```

`disk_build/stages/execute.cpp` owns stage ordering and the BuildSession lifetime.
`stages/api.cpp` retains the `Build` and void `BuildPostPopulate` compatibility entry points.
The production runner exports optional raw vectors and saves final metadata only
after successful construction. The old IVF preparation-dump branch is retired.

`stages/resident.cpp` sequences resident planning and LeftStream. Its bounded
window storage and internal layer initialization remain part of LeftStream.
`stages/upper.cpp` sequences the post-resident memory boundary, ClusterBuild and
completion state. The old resident two-pass WoW, its separate ring buffer,
independent enumeration and `copy_top` alternative have been removed.
The sole upper-layer path is ClusterWoW with Vamana nearest-two partitioning.

PreparedDiskBuild declares its external cluster arrays before the index owner,
so the index is destroyed first. The arrays are loaded after index preparation.
BuildSession owns transient cluster
workspace release on success or exceptions. Large temporary buffers remain in
their algorithm scopes; orchestration does not copy the data or configuration.

## Checkpoints and completion

The existing persisted fields remain authoritative: completed layer, LeftStream
maximum/batch/completion, window ladder, anchor and optional ID-only extension.
`CaptureBuildProgress` is a short value snapshot, not a second mutable state.
An incomplete LeftStream checkpoint is still rejected for resume. Non-LeftStream
checkpoints are rejected before opening graph output, including report mode.
Completed historical indexes remain readable for shuffle and search; the
metadata reader retains their original resident-method marker. Upper-layer
resume still reruns the whole cluster phase, not an individual spill or node.
Build tuning that is not serialized must be supplied consistently on resume.

Stages keep their graph Sync boundaries before publishing completed progress.
`stages/checkpoint.hpp` centralizes Begin/Commit transitions and optional checkpoint
destinations; `Save` owns
canonical metadata serialization. `storage/atomic_file.cpp` owns file commit:
write a unique sibling temporary, check/flush, fsync and close it, rename, then
fsync the directory. A failure before rename preserves the old metadata. If
parent-directory fsync fails after rename, the new file is already committed;
the error explicitly reports that durability is uncertain. This does not make
graph modifications, raw-vector exports or the whole build transactional.

Payload field order, native sizes/endian, RaBitQ canonical codes/factors and
optional trailing extensions are unchanged. This is disk-format compatibility,
not binary ABI compatibility: clients must relink after facade cleanup.

## Planning and memory observation

`include/build/disk_build_plan.hpp` and `disk_build_geometry.hpp` contain
pure plan/geometry value types. `disk_build/memory/budget.hpp/.cpp` owns the solver;
`memory/index_adapter.hpp` is the only planner bridge to the full index facade.
Planning does not sample proc, allocate graph state, trim or write files.

`disk_build/memory/process.hpp/.cpp` separates:

- `CaptureMemorySnapshot`: current/peak proc sampling;
- `PrintMemorySample` and `PrintTrimResult`: formatting only;
- `TrimReleasedMemory`: glibc allocator trim;
- `ResetPeakWindow`: best-effort clear_refs;
- `CompleteMemoryPhase`: explicit sample/print/trim/sample/print/reset boundary.

The consecutive resident trim/reset boundaries are consolidated into one boundary;
resume still captures its upper-stage baseline. Algorithm buffer lifetimes are preserved. Missing/zero peak
still returns before trim/reset, and reset failure remains best effort. The
post-resident measured floor and post-preparation residue still feed the
existing budget formulas. Index RSS methods remain thin compatibility delegates
for other callers. `report.cpp` retains budget reporting, not process ownership.

## Cluster preparation

| Module | Responsibility |
| --- | --- |
| `stages/cluster_prepare.cpp` | Own partition inputs through RSS sampling; select source, validate, check budget, set up CSR |
| `partition/vamana.cpp`, `partition/kmeans.cpp` | Sample, train centers, evaluate candidate K and assign nearest two |
| `partition/assignment.cpp`, `partition/membership.cpp` | Exact full assignment and rank-major CSR |
| `partition/membership.hpp` | Pure non-template membership state |
| `stages/workspace.cpp` | Transient build-resource lifetime |

Candidate budget queries use `PlanPartition(n,K,balance)` without modifying index
state or computing report-only K tables. The accepted partition publishes its K
and skew; graph passes reuse a model resolved after CSR and prep residue.

External Vamana centers and two-ID membership rows pin K. Otherwise native
partitioning starts from explicit K or its small initial K, checks a random
sample estimate, then verifies actual maximum membership against the budget.
There is no capacity reassignment, subcluster/leaf ordering or build quantizer.
Search RaBitQ remains independent: population can encode it only when search
quantization inputs are explicitly supplied. The graph builder consumes FP32
vectors, not those codes.

## ClusterBuild

`stages/cluster_graph.cpp` owns Phase 2 ordering and the inter-pass memory boundary.
Pass A (`cluster_wow/build.cpp`, `cluster_wow/graph.hpp`) builds cluster-local graphs
using exact FP32 distances and writes rank-ordered ID-only spill records.
Pass B (`cluster_wow/commit.cpp`) deduplicates the membership union, applies the
existing fixed-seed shuffle/truncate rule and writes upper-layer records.
Lower layers, vector bytes and raw-record geometry are preserved. Successful
commit closes streams and removes consumed spills.

There is no `copy_top` bypass. Every layer above the completed resident boundary
is built through ClusterBuild. LeftStream's internal Copy/Activate/FinishLayers
operations are unrelated to that retired option and remain unchanged.

## Input and verification

`input/populate.cpp` owns RAM/streaming population and rank mapping. `input/vectors.cpp`
owns paged `.buildvecs` and rank gathers; dense `.rawvecs` export remains in
storage. Native fbin/fvecs support, optional raw export, post-verification
build-vector cleanup and standalone shuffled-layout loading are unchanged.

CLI input validation is shared by report and execution. Empty, short or non-int32-
aligned attribute files are rejected before opening output; a prefix build accepts
at least its effective vector count, and serial/extra attributes remain supported.
Loading checks again before constructing the writable index.

Tests remain under `src/test` with separate build/storage/integration targets.
The retired disk-partition reference builder is replaced by a focused real
O_DIRECT batch fixture. Its injected poll failure verifies zero outstanding
requests before buffer release and before the IOHandler destructor can mask
cleanup. Atomic-save tests inject real write/fsync/rename failures; build tests
cover LeftStream batches/window eviction/tails, Vamana construction, resume rejection
and output completion. Injected resident write and upper validation failures
verify durable progress; upper retry preserves completed resident records. The CLI still accepts `--resident_builder left_stream`;
`two_pass`, `--copy_top` and `--wow_online_compat` are explicit errors. The old
`--wow_passes 1` spelling is accepted only as a compatibility no-op; other values
are rejected. `--cluster_wow_passes` continues to control upper-layer Cluster WoW.
`enum_thresh` remains a window-geometry input, not an enumeration stage.
Full compatibility validation also exercises shuffle, exact rerank and search.

Retired IVF/direct-inherit flags are explicit errors. The compatibility spellings
`--cluster_builder wow` and `--cluster_partition vamana` select the sole path.
`--cluster_np` and the old capacity/leaf/quantization tuning flags are removed.
The memory planner retains whole-process measured floors, partition skew,
sequential-input buffers and both cluster passes; no search-quantization bytes
are removed from the runtime floor. Read-only auto-K reports remain provisional.

See [the complete three-layer call map](../../src/alliance/disk_build/README.md) for source navigation.
