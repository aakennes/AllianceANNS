// Multi-level layout loading and validation.
#include "index/index_disk.hpp"
#include "io/aio_stream.hpp"
#include "shuffle/layout_format.hpp"
#include "shuffle/level_geometry.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <filesystem>
#include <limits>
#include <memory>
#include <sys/stat.h>
#include <unistd.h>

namespace alliance {

template <typename att_t, typename vec_t>
void AllianceIndexDisk<att_t, vec_t>::LoadMultiLevelLayout(const std::string &graph_path,
                                                           const std::string &meta_path,
                                                           size_t cache_capacity_bytes,
                                                           DynamicBlockCache::EvictPolicy policy) {
  // Navigation ranks and group ownership belong to the loaded layout.
  ClearNavigation();
  ml_layout_loaded_ = false;
  if (ml_fd_ >= 0) {
    ::close(ml_fd_);
    ml_fd_ = -1;
  }

  // The writer of this file (ExportMultiLevelLayout) is O_DIRECT + fdatasync, so
  // there should be nothing dirty left in the page cache -- but an O_DIRECT read
  // that races a dirty page can observe stale block-device contents on ext3/ext4,
  // and that failure mode is silent, so pay one fsync here the way
  // DiskSearchContext already does before opening its own O_DIRECT fd.
  {
    int sfd = ::open(graph_path.c_str(), O_RDONLY);
    if (sfd < 0) {
      throw std::runtime_error("LoadMultiLevelLayout: failed to open graph_path: " + graph_path);
    }
    ::fsync(sfd);
    ::close(sfd);
  }

  // Header + level entries: one O_DIRECT sequential read, in its own scope so
  // the reader's two 4 MB staging buffers are gone before the sidecar arrays
  // below get allocated. The reader rounds an unaligned range out to 512 B
  // itself, so the 96 B header and the 48 B entries need no alignment handling
  // here, and the entries follow the header contiguously so one stream serves
  // both. The range is capped at 64 KB (one device read, room for ~1600 level
  // entries) and at the file size, since the reader treats a short completion
  // inside its logical range as an error -- a tiny layout is legal, a truncated
  // one is not.
  GraphLayoutHeader header{};
  MixedVectorGeometry vector_geometry{};
  VecHeaderEntry vector_entry{};
  uint64_t file_bytes = 0;
  std::vector<LevelHeaderEntry> level_entries;
  size_t off = sizeof(header);
  {
    struct stat mst{};
    if (::stat(graph_path.c_str(), &mst) != 0) {
      throw std::runtime_error("LoadMultiLevelLayout: stat failed for " + graph_path);
    }
    file_bytes = static_cast<uint64_t>(mst.st_size);
    const uint64_t probe_bytes = std::min<uint64_t>(65536, file_bytes);
    if (probe_bytes < sizeof(GraphLayoutHeader)) {
      throw std::runtime_error("LoadMultiLevelLayout: file too small to hold a header: " +
                               graph_path);
    }
    AioSeqStreamReader hr(graph_path, 0, probe_bytes);

    hr.Read(&header, sizeof(header));
    if (header.magic != kLayoutMagic) {
      throw std::runtime_error("LoadMultiLevelLayout: bad header magic in " + graph_path);
    }
    if (header.version != kLayoutVersion && header.version != kMixedLayoutVersion) {
      throw std::runtime_error(
          "LoadMultiLevelLayout: layout version " + std::to_string(header.version) +
          " lacks supported layer ownership; re-export graph and sidecar with alliance_shuffle_disk (layout v2/v3): " +
          graph_path);
    }
    if (header.version == kMixedLayoutVersion) {
      hr.Read(&vector_geometry, sizeof(vector_geometry));
      off += sizeof(vector_geometry);
    }
    if (header.curvec_num != curvec_num_) {
      throw std::runtime_error(
          "LoadMultiLevelLayout: curvec_num mismatch (layout=" + std::to_string(header.curvec_num) +
          ", index=" + std::to_string(curvec_num_) + ")");
    }
    if (header.cur_max_layer != cur_max_layer_ || header.M != M_ || header.vec_d != vec_d_ ||
        header.cur_max_layer > static_cast<uint64_t>(std::numeric_limits<layer_t>::max()) ||
        header.num_levels == 0 || header.num_levels > header.cur_max_layer + 1 ||
        header.layer_stride == 0 ||
        header.layer_stride > static_cast<uint64_t>(std::numeric_limits<layer_t>::max())) {
      throw std::runtime_error("LoadMultiLevelLayout: index geometry or layer grid mismatch in " +
                               graph_path);
    }
    if (header.num_levels > 0) {
      if (off + header.num_levels * sizeof(LevelHeaderEntry) > probe_bytes) {
        throw std::runtime_error("LoadMultiLevelLayout: " + std::to_string(header.num_levels) +
                                 " level entries do not fit the header probe window in " +
                                 graph_path);
      }
      level_entries.resize(header.num_levels);
      hr.Read(level_entries.data(), header.num_levels * sizeof(LevelHeaderEntry));
      off += header.num_levels * sizeof(LevelHeaderEntry);
    }
    if (header.vec_region_enabled) {
      hr.Read(&vector_entry, sizeof(vector_entry));
      off += sizeof(vector_entry);
    }
    if (IsMultiLevelOnly() && (header.vec_region_enabled != 1 || header.wp != wp_ ||
                               header.max_elements != max_elements_))
      throw std::runtime_error(
          "ML-only mode requires matching metadata and a complete colocated vector region");
  }

  ml_block_size_ = header.block_size;
  ml_layer_stride_ = header.layer_stride;
  ml_layer_grid_base_ = static_cast<layer_t>(header.cur_max_layer % ml_layer_stride_);
  const size_t num_levels = header.num_levels;

  if (ml_block_size_ == 0 || ml_block_size_ % kAioAlign != 0 ||
      (ml_block_size_ & (ml_block_size_ - 1)) != 0) {
    throw std::runtime_error("LoadMultiLevelLayout: invalid block_size in " + graph_path);
  }
  const size_t header_padding = (ml_block_size_ - (off % ml_block_size_)) % ml_block_size_;
  ml_header_bytes_ = off + header_padding;
  ml_vec_block_size_ =
      header.version == kMixedLayoutVersion ? vector_geometry.block_size : ml_block_size_;
  ml_vec_base_bytes_ = header.version == kMixedLayoutVersion ? vector_geometry.region_offset_bytes
                                                             : ml_header_bytes_;
  if (!ml_vec_block_size_ || ml_vec_block_size_ % kAioAlign ||
      (ml_vec_block_size_ & (ml_vec_block_size_ - 1)) ||
      (header.vec_region_enabled &&
       (VecBytes() > ml_vec_block_size_ || ml_vec_base_bytes_ % kAioAlign)))
    throw std::runtime_error("LoadMultiLevelLayout: invalid vector block geometry");

  // Every block read below is O_DIRECT, which needs the length and the file
  // offset to be sector multiples. Both follow from the geometry the export
  // wrote (block_size is a page multiple and header_bytes is a block multiple),
  // so this is a guard against a hand-made or future layout rather than a real
  // possibility -- but it has to be checked before ml_fd_ is used, not inside
  // the search hot path.
  if (ml_block_size_ == 0 || ml_block_size_ % kAioAlign != 0 || ml_header_bytes_ % kAioAlign != 0) {
    throw std::runtime_error("LoadMultiLevelLayout: block_size=" + std::to_string(ml_block_size_) +
                             " header_bytes=" + std::to_string(ml_header_bytes_) +
                             " are not O_DIRECT-aligned in " + graph_path);
  }

  // The block reads in ReadEdgesMultiLevel go through libaio on this fd. It is
  // O_DIRECT for the same reason ctx->ml_io_handler is: the page cache would
  // otherwise absorb most of the block reads, which both hides the real device
  // cost and makes the reported ml_block_reads/query incomparable with the
  // pipeline path's, since the two would be measuring different things.
  ml_fd_ = ::open(graph_path.c_str(), O_RDONLY | O_DIRECT | O_LARGEFILE);
  if (ml_fd_ < 0) {
    throw std::runtime_error("LoadMultiLevelLayout: failed to open graph_path O_DIRECT: " +
                             graph_path);
  }

  ml_levels_.resize(num_levels);
  ml_record_size_per_level_.resize(num_levels);
  for (size_t li = 0; li < num_levels; ++li) {
    ml_levels_[li] = {static_cast<layer_t>(level_entries[li].layer_begin),
                      static_cast<layer_t>(level_entries[li].layer_end),
                      static_cast<layer_t>(level_entries[li].owner_begin),
                      static_cast<layer_t>(level_entries[li].owner_end)};
    ml_record_size_per_level_[li] = level_entries[li].record_size;
  }
  ValidateLevelRanges(ml_levels_, static_cast<layer_t>(header.cur_max_layer), ml_layer_stride_);
  for (size_t li = 0; li < num_levels; ++li) {
    const size_t num_layers =
        CountSelectedLayers(ml_levels_[li].layer_begin, ml_levels_[li].layer_end, ml_layer_stride_);
    const size_t expected_record = ComputeRecordSize(num_layers, M_, sizeof(att_t));
    if (ml_record_size_per_level_[li] != expected_record ||
        level_entries[li].elements_per_block != ml_block_size_ / expected_record ||
        level_entries[li].elements_per_block == 0) {
      throw std::runtime_error("LoadMultiLevelLayout: record geometry mismatch at level " +
                               std::to_string(li));
    }
  }

  uint64_t neighbor_blocks = 0;
  for (const auto &e : level_entries) {
    if (e.region_start_block != neighbor_blocks ||
        e.num_blocks != (curvec_num_ + e.elements_per_block - 1) / e.elements_per_block)
      throw std::runtime_error("LoadMultiLevelLayout: noncontiguous/incomplete neighbor region");
    neighbor_blocks += e.num_blocks;
  }
  if (neighbor_blocks >
      (file_bytes - std::min<uint64_t>(file_bytes, ml_header_bytes_)) / ml_block_size_)
    throw std::runtime_error("LoadMultiLevelLayout: truncated neighbor data");
  const uint64_t neighbor_end = ml_header_bytes_ + neighbor_blocks * ml_block_size_;
  if (header.vec_region_enabled) {
    const uint64_t first =
        header.version == kMixedLayoutVersion ? 0 : vector_entry.region_start_block;
    const uint64_t start = MultiLevelVectorOffset(first);
    if (vector_entry.elements_per_block != ml_vec_block_size_ / VecBytes() ||
        !vector_entry.elements_per_block ||
        vector_entry.num_blocks !=
            (curvec_num_ + vector_entry.elements_per_block - 1) / vector_entry.elements_per_block ||
        start < neighbor_end || start % ml_vec_block_size_ || start > file_bytes ||
        vector_entry.num_blocks > (file_bytes - start) / ml_vec_block_size_)
      throw std::runtime_error("LoadMultiLevelLayout: invalid/truncated vector region");
    if (IsMultiLevelOnly() && start + vector_entry.num_blocks * ml_vec_block_size_ != file_bytes)
      throw std::runtime_error("ML-only graph has trailing data outside its declared regions");
  }
  // O(1) LUT routes each selected layer to the replica shuffled for its owned
  // k-layer interval. Stored overlap remains available to cache hits/descent.
  {
    layer_t max_layer = 0;
    for (auto &lr : ml_levels_)
      max_layer = std::max(max_layer, lr.layer_end);
    ml_layer_to_level_.assign(static_cast<size_t>(max_layer) + 1, 0);
    for (layer_t l = 0; l <= max_layer; ++l) {
      ml_layer_to_level_[l] = FindLevelForLayer(l, ml_levels_, ml_layer_stride_);
    }
  }

  // Load sidecar
  std::ifstream mfs(meta_path, std::ios::binary);
  if (!mfs.is_open()) {
    throw std::runtime_error("LoadMultiLevelLayout: failed to open meta_path: " + meta_path);
  }
  uint64_t magic = 0, version = 0, curvec_num64 = 0, nlv = 0;
  mfs.read(reinterpret_cast<char *>(&magic), sizeof(magic));
  mfs.read(reinterpret_cast<char *>(&version), sizeof(version));
  mfs.read(reinterpret_cast<char *>(&curvec_num64), sizeof(curvec_num64));
  mfs.read(reinterpret_cast<char *>(&nlv), sizeof(nlv));
  if (!mfs.good() || magic != kLayoutMagic ||
      (version != kLayoutVersion && version != kMixedLayoutVersion) || version != header.version ||
      curvec_num64 != curvec_num_ || nlv != num_levels) {
    throw std::runtime_error("LoadMultiLevelLayout: sidecar header mismatch: " + meta_path);
  }

  ml_disk_blocks_per_level_.resize(num_levels);
  for (size_t li = 0; li < num_levels; ++li) {
    ml_disk_blocks_per_level_[li].resize(curvec_num_);
    mfs.read(reinterpret_cast<char *>(ml_disk_blocks_per_level_[li].data()),
             static_cast<std::streamsize>(curvec_num_ * sizeof(int32_t)));
    if (!mfs.good()) {
      throw std::runtime_error("LoadMultiLevelLayout: truncated sidecar at level " +
                               std::to_string(li));
    }
    const auto &e = level_entries[li];
    for (int32_t block : ml_disk_blocks_per_level_[li])
      if (block < 0 || static_cast<uint64_t>(block) < e.region_start_block ||
          static_cast<uint64_t>(block) - e.region_start_block >= e.num_blocks)
        throw std::runtime_error("LoadMultiLevelLayout: neighbor mapping out of bounds");
  }

  // Optional search-vector region mappings (node_id -> block, node_id -> slot).
  // Only usable when the region is colocated in graph_path (vec_region_enabled):
  // ml_fd_ / ctx->ml_io_handler read that same file. A layout exported with a
  // separate out_vec_path, or without vectors at all, falls back to the legacy
  // RerankPipeline path.
  ml_vec_region_loaded_ = false;
  ml_vec_block_ids_.clear();
  ml_vec_slots_.clear();
  uint8_t vec_flag = 0;
  mfs.read(reinterpret_cast<char *>(&vec_flag), sizeof(vec_flag));
  if (mfs.good() && vec_flag != 0) {
    std::vector<size_t> vec_blocks(curvec_num_), vec_slots(curvec_num_);
    mfs.read(reinterpret_cast<char *>(vec_blocks.data()),
             static_cast<std::streamsize>(curvec_num_ * sizeof(size_t)));
    mfs.read(reinterpret_cast<char *>(vec_slots.data()),
             static_cast<std::streamsize>(curvec_num_ * sizeof(size_t)));
    if (!mfs.good()) {
      throw std::runtime_error("LoadMultiLevelLayout: truncated sidecar vector mappings: " +
                               meta_path);
    }
    if (header.vec_region_enabled) {
      const size_t first =
          header.version == kMixedLayoutVersion ? 0 : vector_entry.region_start_block;
      for (size_t id = 0; id < curvec_num_; ++id)
        if (vec_blocks[id] < first || vec_blocks[id] - first >= vector_entry.num_blocks ||
            vec_slots[id] >= vector_entry.elements_per_block)
          throw std::runtime_error("LoadMultiLevelLayout: vector mapping out of bounds");
      ml_vec_block_ids_ = std::move(vec_blocks);
      ml_vec_slots_ = std::move(vec_slots);
      ml_vec_region_loaded_ = true;
    }
  }
  if (IsMultiLevelOnly() && (vec_flag != 1 || !ml_vec_region_loaded_))
    throw std::runtime_error("ML-only mode requires complete sidecar vector mappings");

  // ---- Stage 4 load: read the persisted static-layer cache section appended
  // by ExportMultiLevelLayout (after the vec section). Format mirrors the
  // persist: static_flag (0=none, 2=per-node format), then metadata + buffer.
  static_cache_loaded_ = false;
  if (static_cache_buf_) {
    free(static_cache_buf_);
    static_cache_buf_ = nullptr;
  }
  static_cache_bytes_ = 0;
  std::vector<StaticNodeMeta>().swap(static_node_meta_);
  std::vector<StaticNodeMeta64>().swap(static_node_meta64_);
  static_node_lookup_.Clear();
  uint8_t static_flag = 0;
  mfs.read(reinterpret_cast<char *>(&static_flag), sizeof(static_flag));
  if (IsMultiLevelOnly() && (!mfs.good() || (static_flag != 0 && static_flag != kStaticCacheV2 &&
                                             static_flag != kStaticCacheV3)))
    throw std::runtime_error("ML-only mode requires a complete current static-cache section");
  if (mfs.good() && (static_flag == kStaticCacheV2 || static_flag == kStaticCacheV3)) {
    // Per-node format (v2)
    uint64_t scb = 0;
    mfs.read(reinterpret_cast<char *>(&scb), sizeof(scb));
    static_cache_bytes_ = static_cast<size_t>(scb);
    uint64_t num_meta = 0;
    mfs.read(reinterpret_cast<char *>(&num_meta), sizeof(num_meta));
    if (!mfs.good()) {
      throw std::runtime_error("LoadMultiLevelLayout: truncated static cache header: " + meta_path);
    }
    if (num_meta != curvec_num_)
      throw std::runtime_error("LoadMultiLevelLayout: static node count mismatch");
    static_node_lookup_.LoadDense(mfs, static_cast<size_t>(num_meta), static_cache_bytes_,
                                  static_flag);
    if (static_cache_bytes_ > 0) {
      const size_t aligned = ((static_cache_bytes_ + 4095) / 4096) * 4096;
      static_cache_buf_ = static_cast<char *>(aligned_alloc(4096, aligned));
      if (!static_cache_buf_) {
        throw std::runtime_error("LoadMultiLevelLayout: static cache alloc failed");
      }
      mfs.read(static_cache_buf_, static_cast<std::streamsize>(static_cache_bytes_));
      if (!mfs.good()) {
        throw std::runtime_error("LoadMultiLevelLayout: truncated static cache records: " +
                                 meta_path);
      }
    }
    static_cache_loaded_ = true;
  } else if (mfs.good() && static_flag == 1) {
    // Legacy per-level format (v1) -- skip over it without loading
    uint64_t scb = 0;
    mfs.read(reinterpret_cast<char *>(&scb), sizeof(scb));
    for (size_t li = 0; li < num_levels; ++li) {
      layer_t lb, le;
      size_t piece, slots, boff;
      uint64_t nw;
      mfs.read(reinterpret_cast<char *>(&lb), sizeof(lb));
      mfs.read(reinterpret_cast<char *>(&le), sizeof(le));
      mfs.read(reinterpret_cast<char *>(&piece), sizeof(piece));
      mfs.read(reinterpret_cast<char *>(&slots), sizeof(slots));
      mfs.read(reinterpret_cast<char *>(&boff), sizeof(boff));
      mfs.read(reinterpret_cast<char *>(&nw), sizeof(nw));
      if (nw > 0)
        mfs.seekg(static_cast<std::streamoff>(nw * sizeof(uint64_t)), std::ios::cur);
    }
    if (scb > 0)
      mfs.seekg(static_cast<std::streamoff>(scb), std::ios::cur);
    // v1 data skipped; static_cache_loaded_ remains false
  }

  if (IsMultiLevelOnly() &&
      (!mfs.good() || mfs.tellg() < 0 ||
       static_cast<uint64_t>(mfs.tellg()) != std::filesystem::file_size(meta_path)))
    throw std::runtime_error("ML-only sidecar is truncated or has trailing data");

  // Store cache configuration — worker caches are initialized lazily.
  // One unified cache spans all levels (WoW-style): full budget, pieces sized
  // to the largest level record so any level's record fits any slot.
  ml_cache_config_.capacity_bytes = cache_capacity_bytes;
  ml_cache_config_.piece_bytes =
      *std::max_element(ml_record_size_per_level_.begin(), ml_record_size_per_level_.end());
  ml_cache_config_.num_nodes = curvec_num_;
  ml_cache_config_.policy = policy;
  ++ml_cache_config_.generation;

  ml_layout_loaded_ = true;
}

template void AllianceIndexDisk<int, float>::LoadMultiLevelLayout(const std::string &,
                                                                  const std::string &, size_t,
                                                                  DynamicBlockCache::EvictPolicy);

} // namespace alliance
