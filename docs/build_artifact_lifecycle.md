# Disk build artifact lifecycle

The build, read-only acceptance, and cleanup are separate operations. Builds
retain `.buildvecs` by default, including failures and resumable checkpoints.
Shuffle reads vectors from the graph records, not from `.buildvecs` or `.rawvecs`.

`alliance_build_disk --rawvec_export on|off` defaults to `on`. With `off`, no raw
vector export is made and the saved metadata has an empty vector path. Existing
raw-vector files are preserved. Consumers that require raw vectors must check
that capability; complete multilevel layouts contain their own exact vectors.

The native retention default is distinct from the dataset workflow:
`scripts/defaults.json` selects `rawvec_export=off` and `finalize_on_success=true`.
`run_dataset.py build` invokes full acceptance and explicit finalization after a
successful build, before shuffle. Thus the normal new workflow removes the
accepted `.buildvecs` copy and closes its resume window; direct native callers
retain the compatibility behavior unless they invoke the finalizer themselves.

`scripts/finalize_build.py` is a Linux-only lifecycle operation:

```sh
python3 scripts/finalize_build.py \
  --artifact-root /path/to/job --graph /path/to/job/graph.bin \
  --meta /path/to/job/index.meta --base /path/to/base.fbin --format bin \
  --attrs /path/to/attributes.bin --space l2 \
  --verifier /path/to/alliance_verify_layout
```

The default verifies every original vector, attribute, label and layer's edge
records and retains all data. Add both `--apply --close-resume-window` to remove
only the exact `graph.bin.buildvecs` file. The graph, metadata, original inputs,
raw vectors, and shuffled layout are never deleted by this tool.

The builder and finalizer share a persistent advisory lock. Finalization rejects
unfinished checkpoints, failed verification, symbolic links, multiple hard links,
paths outside the declared artifact root, and identities that change during
acceptance. It durably closes the resume window before unlinking. Subsequent
build/resume at that graph path is rejected. An interrupted cleanup can be retried
with the same inputs and flags; it does not rerun construction or ANN queries.
Each attempt records its verification output, identities, SHA-256 evidence,
reason, and logical/allocated bytes removed in `graph.bin.build.finalization/`.

**Layer-zero limitation:** this finalizer refuses `max_layer=0`, even when
`completed_layer=0`. Zero is also the initial progress value, so equality alone
does not prove layer 0 was built (for example, historical prep-dump artifacts
could be saved before construction). This version has no independent
completion-receipt protocol for that case; it retains the vectors and leaves the
resume window open. Normal multi-layer builds are checked against their final
completed layer.

The artifact contract is covered by `test_build_artifacts.py`: default/off
byte parity, retention, failed/incomplete/unsafe cleanup, lock exclusion,
interruption recovery, closed-resume rejection, and shuffle/search after cleanup.
