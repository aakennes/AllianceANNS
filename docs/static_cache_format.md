# Adaptive static cache offsets

Keep v2 generation, 16-byte persisted/query metadata and budget accounting for
every cache whose actual selected payload fits uint32 byte offsets. Dataset size
alone does not force migration: small caches on SIFT100M also remain v2.

Only actual offset overflow triggers a repeat of node selection using 24-byte
wide metadata and the same landing/in-weight policy. Prepared in-weights are
reused; no graph rebuild or shuffle. Re-selection includes the extra metadata
in the same total static budget. New wide caches use static-section flag3:
uint64 offset, int32 layer_begin/end, uint32 record_bytes, uint32 reserved=0.
Flag2 remains unchanged (16-byte uint32-offset metadata). Graph/routing formats
are unchanged. Reader dispatches both disk and compact query metadata by flag;
legacy caches incur no wider per-node metadata. UINT64_MAX marks absent v3 nodes.
Old executables must not load flag3. Bounds/version/reserved checks are mandatory.

Tests: test_cache_contract preserves old small-cache contracts and verifies v3
offsets across4GiB with sparse virtual memory, v2 loading, corruption rejection.
test_layout_export checks cache generation, graph equivalence and budgets.
