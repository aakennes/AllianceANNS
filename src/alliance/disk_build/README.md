# Disk build execution boundaries

The build retains LeftStream + Vamana partitioning + ClusterWoW. Search-side
quantization, graph/metadata formats and CLI parameters are unchanged.

## Three execution layers

1. `src/tools/build`: CLI orchestration. `RunDiskBuild` validates inputs before
   opening outputs, obtains the artifact lock, prepares a fresh/checkpoint index,
   invokes the graph builder and saves final artifacts. Private `disk/` helpers
   own input validation, borrowed partition buffers and final export.
2. `stages/`: phase order, scratch lifetime and durable progress publication.
   `ExecuteBuild` runs/reuses resident layers, samples the upper memory baseline,
   and runs remaining upper layers. `checkpoint.hpp` owns completion transitions.
3. `left_stream/`, `partition/`, `cluster_wow/`: record I/O and algorithm kernels.
   `input/` materializes initial records; `memory/` estimates and observes memory.

```text
RunDiskBuild
  report-only -> ReportDiskBuildMemory -> return
  ReadBuildInputShape / ValidateResumeCheckpoint
  BuildArtifactLock
  PrepareDiskIndex
    CreateFreshIndex -> LoadBuildAttributes -> PopulateElements
    or LoadCheckpointIndex -> ResumeStartLayer
    LoadExternalPartition
  ExecuteBuild
    BuildSession
    RunResidentStage
      PlanPhase15
      BeginResidentCheckpoint
      ExecuteLeftStreamRecords -> Batch -> FinishLayers -> Sync/coverage check
      CommitResidentCheckpoint
    PrepareUpperMemoryBoundary
    RunUpperStage
      ClusterPrepDisk
        SynthesizeVamanaClustersDisk or validate external memberships
        PlanPartition -> SetupClusterBuildDisk -> capture prep residue
      ClusterBuildUpperLayersDisk
        Sync -> resolve accepted-partition memory plan
        BuildClusterWoWEdgesDisk
          Distribute -> LoadClusterVectorsAndSeeds -> FinalizeClusterSeeds
          ClusterWowGraph::Build -> WriteClusterSpill
        release Pass A scratch / memory boundary
        CommitClusterEdgesDisk -> fsync
      teardown memberships -> Sync -> CommitUpperCheckpoint
  SaveBuildArtifacts
```

## Contracts that must remain explicit

- Incomplete resident checkpoints cannot resume. A completed resident checkpoint
  resumes the whole upper stage; there is no per-cluster or per-pass checkpoint.
- Completion is published only after graph I/O and validation succeed. Metadata
  saves remain atomic. The compatibility `BuildLeftStreamLayers` entry publishes
  resident record flags but does not advance the staged resume cursor.
- Fixed partition memory evaluation does not mutate the index. Only an accepted
  partition publishes K/skew. Report-only K scans use the same underlying model.
- Generated partition buffers survive CSR RSS sampling. Pass A buffers are dead
  before the Pass B memory window. Sampling logs are not an OS hard RSS limit.
- Lower-layer sync before Direct I/O, two-step seed initialization, neighbor
  ordering, nearest-two memberships and merge RNG/truncation are preserved.
- Population may encode persistent search RaBitQ data. Graph construction uses
  exact FP32 distances and does not consume those codes.
- Shuffle and `.buildvecs` finalization are separate workflow operations.
