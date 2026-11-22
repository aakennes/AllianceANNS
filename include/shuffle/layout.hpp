#pragma once
// alliance/shuffle/layout.hpp -- Multi-level disk layout: point-edge separation.
//
// Disk layout after export (one complete file at out_graph_path):
//   [header block]  -- GraphLayoutHeader (see layout_format.hpp), padded to block_size
//   [level 0 meta blocks]   -- record: internal_id | label | attribute | linklist[level_layers]
//   [level 1 meta blocks]
//   ...
//   [search vector blocks]  -- raw FP32 vectors in the top-level order
//
// Meta block records embed the node's internal_id as the first field so a whole
// block can be linearly scanned for a specific id without a separate CSR index
// (blocks are small: a few dozen records at most). Unused tail slots in the last
// block of each level are filled with 0xFF (id field reads as tableint(-1)).
//
// Each level is independently shuffled -- its own BlockBNF pass, its own
// elements_per_block, its own block-to-node mapping. The runtime mapping
// (node_id -> block_id per level, node_id -> (block_id, slot) for the vector
// region) is persisted to the sidecar file `out_meta_path`, since it doesn't fit
// in a fixed-size header.

#include "shuffle/shuffle.hpp"
#include "shuffle/layout_format.hpp"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace alliance {

// ---------------------------------------------------------------------------
// Per-level disk record sizing
// ---------------------------------------------------------------------------

// Record size and elements_per_block are defined in level_geometry.hpp:
//   ComputeRecordSize(num_layers, M, att_bytes)
//   ComputeElementsPerBlock(block_size, record_size)

// ---------------------------------------------------------------------------
// On-disk format structs (shared by the writer in shuffle/format and any reader,
// e.g. AllianceIndexDisk::LoadMultiLevelLayout in multilevel_traversal.cpp)
// ---------------------------------------------------------------------------

// Persisted POD definitions live in layout_format.hpp, also included here
// to preserve existing writer-side include compatibility.

// ---------------------------------------------------------------------------
// Layout export configuration
// ---------------------------------------------------------------------------

struct LayoutExportConfig {
  // Disk block size in bytes (typically 4096 or 8192 for direct IO alignment).
  // Neighbor blocks; vector_block_size=0 preserves equal-size v2 exports.
  size_t block_size{4096};
  size_t vector_block_size{0};

  // Multi-level config (level_k, level_m, per-level BNF params).
  MultiLevelConfig level_config;

  // Shuffle method used for both meta levels and the search-vector order.
  ShuffleMethod method{ShuffleMethod::BlockBNF};

  // Output file paths.
  std::string out_graph_path; // edge/meta blocks plus exact vectors
  std::string out_meta_path;  // sidecar runtime mappings (node_id -> block_id per level, etc.)
};

// ---------------------------------------------------------------------------
// Layout export result
// ---------------------------------------------------------------------------

struct LayoutExportResult {
  struct LevelInfo {
    LevelRange range;
    size_t num_blocks{0};
    size_t elements_per_block{0};
    size_t record_size{0};
    size_t region_start_block{0}; // first block index (0-based, right after header) for this level
    std::vector<int32_t> disk_blocks; // disk_blocks[node_id] = block_id (absolute, header-relative)
  };
  std::vector<LevelInfo> levels;

  // Search vector region info.
  size_t vec_region_num_blocks{0};
  size_t vec_elements_per_block{0};
  size_t vec_region_start_block{0};  // header-relative v2 block; v3 uses vector_region_offset_bytes
  std::vector<size_t> vec_block_ids; // vec_block_ids[node_id] = block id (region-relative in v3)
  std::vector<size_t> vec_slots;     // vec_slots[node_id] = slot within that block

  // Total size (bytes) written to out_graph_path (header + all meta + vectors).
  size_t total_bytes{0};
  uint64_t layout_version{kLayoutVersion};
  size_t vector_block_size{0};
  uint64_t vector_region_offset_bytes{0};
};

// ---------------------------------------------------------------------------
// Layout export interface
// ---------------------------------------------------------------------------

// Exports the shuffled multi-level layout to disk:
//   1. Generate level ranges from config.level_config
//   2. For each level: compute record size, elements_per_block, run config.method
//      (via ComputeMultiLevelShuffleOrders)
//   3. Write edge/meta blocks per level, sequentially, 0xFF-filled tail
//   4. Write search vector blocks using the top-level order
//   5. Write the sidecar file with node_id -> block_id (per level) and
//      node_id -> (block_id, slot) (vector region) mappings
template <typename att_t, typename vec_t>
auto ExportMultiLevelLayout(AllianceIndexDisk<att_t, vec_t> *idx, const LayoutExportConfig &config)
    -> LayoutExportResult;

// Overload accepting pre-computed shuffle results (avoids recomputing BNF).
template <typename att_t, typename vec_t>
auto ExportMultiLevelLayout(AllianceIndexDisk<att_t, vec_t> *idx, const LayoutExportConfig &config,
                            const std::vector<LevelShuffleResult> &level_results)
    -> LayoutExportResult;

} // namespace alliance
