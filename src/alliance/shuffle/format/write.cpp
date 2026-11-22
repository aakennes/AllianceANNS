#include "shuffle/format/write.hpp"
#include "index/index_disk.hpp"
#include "io/aio_batch.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>

namespace alliance::shuffle_detail {

namespace {
using clk = std::chrono::steady_clock;
double since(clk::time_point start) {
  return std::chrono::duration<double>(clk::now() - start).count();
}
constexpr size_t kExportGatherChunk = 512;
template <typename T> void WritePOD(std::ostream &os, const T &value) {
  os.write(reinterpret_cast<const char *>(&value), sizeof(T));
}
} // namespace

template <typename att_t, typename vec_t>
void WriteLayoutLevels(AllianceIndexDisk<att_t, vec_t> *idx, const LayoutExportConfig &config,
                       const std::vector<LevelShuffleResult> &level_results,
                       LayoutExportResult &result, AioRegionWriter &gw, LayoutIoStats &stats) {
  const size_t n = idx->curvec_num_;
  const size_t num_levels = result.levels.size();
  const size_t per_layer_id_bytes = idx->EdgeIdBytes();
  for (size_t li = 0; li < num_levels; ++li) {
    const auto &order = level_results[li].order;
    const size_t rs = result.levels[li].record_size;
    const layer_t layer_stride = static_cast<layer_t>(LayerStride(config.level_config.shuffle_o));

    // Gather one contiguous source window per node, then copy only selected
    // ID/count slices. Raw distance slots between layers are never exported.
    std::vector<layer_t> sel;
    for (layer_t layer = result.levels[li].range.layer_begin;
         layer <= result.levels[li].range.layer_end; layer += layer_stride) {
      sel.push_back(layer);
    }
    size_t win_off = 0, win_len = 0;
    if (!sel.empty()) {
      size_t lo = idx->EdgeOffsetInRecord(sel.front());
      size_t hi = lo + per_layer_id_bytes;
      for (layer_t layer : sel) {
        const size_t off = idx->EdgeOffsetInRecord(layer);
        lo = std::min(lo, off);
        hi = std::max(hi, off + per_layer_id_bytes);
      }
      win_off = lo;
      win_len = hi - lo;
    }

    std::vector<char> stage(kExportGatherChunk * std::max<size_t>(win_len, 1));
    std::vector<AioSlice> slices(kExportGatherChunk);
    std::vector<char> block_buf(config.block_size, static_cast<char>(0xFF));
    size_t block_offset = 0;
    size_t block_idx_in_level = 0;

    for (size_t base = 0; base < n; base += kExportGatherChunk) {
      const size_t cnt = std::min(kExportGatherChunk, n - base);

      if (win_len > 0) {
        const auto r0 = clk::now();
        for (size_t k = 0; k < cnt; ++k) {
          slices[k].buf = stage.data() + k * win_len;
          slices[k].len = win_len;
          slices[k].off = idx->graph_file_->node_offset(idx->DiskSlot(order[base + k])) + win_off;
        }
        idx->graph_file_->ReadSlices(slices.data(), cnt);
        stats.read_s += since(r0);
        ++stats.read_ops;
        stats.read_slices += cnt;
        stats.read_bytes += cnt * win_len;
      }

      for (size_t k = 0; k < cnt; ++k) {
        tableint node_id = order[base + k];
        if (block_offset + rs > config.block_size) {
          const auto w0 = clk::now();
          gw.Append(block_buf.data(), block_buf.size());
          stats.write_s += since(w0);
          ++stats.write_ops;
          stats.write_bytes += block_buf.size();
          std::fill(block_buf.begin(), block_buf.end(), static_cast<char>(0xFF));
          block_offset = 0;
          ++block_idx_in_level;
        }

        char *dst = block_buf.data() + block_offset;
        std::memcpy(dst, &node_id, sizeof(tableint));
        dst += sizeof(tableint);
        label_t label = idx->labels_[node_id];
        std::memcpy(dst, &label, sizeof(label_t));
        dst += sizeof(label_t);
        att_t att = idx->atts_[node_id];
        std::memcpy(dst, &att, sizeof(att_t));
        dst += sizeof(att_t);
        const char *src = stage.data() + k * win_len;
        for (layer_t layer : sel) {
          std::memcpy(dst, src + (idx->EdgeOffsetInRecord(layer) - win_off), per_layer_id_bytes);
          dst += per_layer_id_bytes;
        }

        result.levels[li].disk_blocks[node_id] =
            static_cast<int32_t>(result.levels[li].region_start_block + block_idx_in_level);
        block_offset += rs;
      }
    }
    if (block_offset > 0) {
      const auto w0 = clk::now();
      gw.Append(block_buf.data(), block_buf.size());
      stats.write_s += since(w0);
      ++stats.write_ops;
      stats.write_bytes += block_buf.size();
    }
  }
}

template <typename att_t, typename vec_t>
void WriteLayoutVectors(AllianceIndexDisk<att_t, vec_t> *idx, const LayoutExportConfig &config,
                        const std::vector<LevelShuffleResult> &level_results,
                        size_t vec_region_start_block, LayoutExportResult &result,
                        AioRegionWriter &writer, LayoutIoStats &stats) {
  const size_t n = idx->curvec_num_;
  const size_t vec_block_bytes = idx->vec_d_ * sizeof(vec_t);
  const auto &vec_order = level_results.front().order;
  result.vec_block_ids.assign(n, 0);
  result.vec_slots.assign(n, 0);

  const size_t vec_off = idx->VecOffsetInRecord();
  std::vector<char> vstage(kExportGatherChunk * vec_block_bytes);
  std::vector<AioSlice> vslices(kExportGatherChunk);
  const size_t vector_block_size =
      config.vector_block_size ? config.vector_block_size : config.block_size;
  std::vector<char> vblock_buf(vector_block_size, 0);
  size_t vblock_offset = 0;
  size_t vblock_idx = 0;

  for (size_t base = 0; base < n; base += kExportGatherChunk) {
    const size_t cnt = std::min(kExportGatherChunk, n - base);

    const auto r0 = clk::now();
    for (size_t k = 0; k < cnt; ++k) {
      vslices[k].buf = vstage.data() + k * vec_block_bytes;
      vslices[k].len = vec_block_bytes;
      vslices[k].off = idx->graph_file_->node_offset(idx->DiskSlot(vec_order[base + k])) + vec_off;
    }
    idx->graph_file_->ReadSlices(vslices.data(), cnt);
    stats.read_s += since(r0);
    ++stats.read_ops;
    stats.read_slices += cnt;
    stats.read_bytes += cnt * vec_block_bytes;

    for (size_t k = 0; k < cnt; ++k) {
      tableint node_id = vec_order[base + k];
      if (vblock_offset + vec_block_bytes > vector_block_size) {
        const auto w0 = clk::now();
        writer.Append(vblock_buf.data(), vblock_buf.size());
        stats.write_s += since(w0);
        ++stats.write_ops;
        stats.write_bytes += vblock_buf.size();
        std::fill(vblock_buf.begin(), vblock_buf.end(), 0);
        vblock_offset = 0;
        ++vblock_idx;
      }
      std::memcpy(vblock_buf.data() + vblock_offset, vstage.data() + k * vec_block_bytes,
                  vec_block_bytes);

      result.vec_block_ids[node_id] = (result.layout_version == kMixedLayoutVersion)
                                          ? vblock_idx
                                          : (vec_region_start_block + vblock_idx);
      result.vec_slots[node_id] = vblock_offset / vec_block_bytes;
      vblock_offset += vec_block_bytes;
    }
  }
  if (vblock_offset > 0) {
    const auto w0 = clk::now();
    writer.Append(vblock_buf.data(), vblock_buf.size());
    stats.write_s += since(w0);
    ++stats.write_ops;
    stats.write_bytes += vblock_buf.size();
  }
}

template <typename att_t, typename vec_t>
size_t WriteLayoutSidecar(AllianceIndexDisk<att_t, vec_t> *idx, const LayoutExportConfig &config,
                          const LayoutExportResult &result) {
  const size_t n = idx->curvec_num_;
  const size_t num_levels = result.levels.size();
  std::ofstream meta_ofs(config.out_meta_path, std::ios::binary | std::ios::trunc);
  if (!meta_ofs.is_open()) {
    throw std::runtime_error("ExportMultiLevelLayout: failed to open out_meta_path: " +
                             config.out_meta_path);
  }
  uint64_t magic = kLayoutMagic, version = result.layout_version, curvec_num64 = n,
           nlv = num_levels;
  WritePOD(meta_ofs, magic);
  WritePOD(meta_ofs, version);
  WritePOD(meta_ofs, curvec_num64);
  WritePOD(meta_ofs, nlv);
  for (size_t li = 0; li < num_levels; ++li) {
    meta_ofs.write(reinterpret_cast<const char *>(result.levels[li].disk_blocks.data()),
                   static_cast<std::streamsize>(n * sizeof(int32_t)));
  }
  uint8_t vec_flag = 1;
  WritePOD(meta_ofs, vec_flag);
  {
    meta_ofs.write(reinterpret_cast<const char *>(result.vec_block_ids.data()),
                   static_cast<std::streamsize>(n * sizeof(size_t)));
    meta_ofs.write(reinterpret_cast<const char *>(result.vec_slots.data()),
                   static_cast<std::streamsize>(n * sizeof(size_t)));
  }

  // Base layouts have no inline cache; independent preparation owns cache policy.
  const uint8_t static_flag = 0;
  WritePOD(meta_ofs, static_flag);
  const size_t meta_bytes = static_cast<size_t>(meta_ofs.tellp());
  meta_ofs.flush();
  if (!meta_ofs)
    throw std::runtime_error("failed writing layout sidecar");
  meta_ofs.close();
  if (!meta_ofs)
    throw std::runtime_error("failed closing layout sidecar");
  return meta_bytes;
}

template void WriteLayoutLevels<int, float>(AllianceIndexDisk<int, float> *idx,
                                            const LayoutExportConfig &config,
                                            const std::vector<LevelShuffleResult> &level_results,
                                            LayoutExportResult &result, AioRegionWriter &gw,
                                            LayoutIoStats &stats);
template void WriteLayoutVectors<int, float>(AllianceIndexDisk<int, float> *idx,
                                             const LayoutExportConfig &config,
                                             const std::vector<LevelShuffleResult> &level_results,
                                             size_t vec_region_start_block,
                                             LayoutExportResult &result, AioRegionWriter &writer,
                                             LayoutIoStats &stats);
template size_t WriteLayoutSidecar<int, float>(AllianceIndexDisk<int, float> *idx,
                                               const LayoutExportConfig &config,
                                               const LayoutExportResult &result);
} // namespace alliance::shuffle_detail
