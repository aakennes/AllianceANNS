# Power-of-two disk pages

`alliance_build_disk --node_page_bytes` accepts power-of-two byte lengths
starting at 4096. In particular, 8192 and 16384 are supported; 12288 is rejected
before creating/truncating the graph. A page must contain the entire node record.
The automatic mode retains next-power-of-two sizing.

Build-vector pages are derived from the vector byte length, with a 4096-byte
minimum and next-power-of-two rounding. Thus 1536D float vectors use 8192-byte
build-vector pages, while <=1024D existing buildvecs stay byte-compatible.
Gather offsets, staging buffers and memory planning use the same derived size.
The thread-local gather window is resized when another index has a different
vector page length. IO buffer alignment is 4096 and is independent of page length.

Multi-level export/search uses independent neighbor and vector geometry.
`--block_size` selects neighbor pages; `--vector_block_size 0` defaults vector
pages to the same size. Equal geometry retains uniform-page v2. Different
geometry uses the existing v3 header and readers; each vector must fit in its
vector page and each neighbor record must fit in its neighbor page.

For CaseLaw's 1536D FP32 vectors, build nodes use 8192-byte pages (`wp=6`, M=32,
L2 gives the recorded 7964-byte node). The mixed search layout uses shuffle
`--block_size 4096 --vector_block_size 8192`. The older uniform
`--block_size 8192` layout remains compatible; it is not the mixed-page choice.
This guide does not launch a build, convert or replace historical indexes.

The end-to-end `test_variable_pages` exercises 4/8/16 KiB nodes, 128/256/1536/
2304/3200 dimensions, phased build, vector readback, scattered gather, save/load,
shuffle, both cache policies and async rerank; it checks rejection of undersized
and 12 KiB pages. The memory-planner grid includes the high-dimensional cases.
