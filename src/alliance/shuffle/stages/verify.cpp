#include "shuffle/stages/verify.hpp"
#include "index/index_disk.hpp"
#include "io/aio_batch.hpp"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace alliance::shuffle_detail {
namespace {
void Require(bool valid, const char *message) {
  if (!valid)
    throw std::runtime_error(std::string("shuffle readback: ") + message);
}
void Read(std::istream &stream, void *buffer, size_t bytes) {
  stream.read(static_cast<char *>(buffer), static_cast<std::streamsize>(bytes));
  Require(bool(stream), "short read");
}
template <class T> T Read(std::istream &stream) {
  T value{};
  Read(stream, &value, sizeof(value));
  return value;
}
template <class T> void Match(std::istream &stream, const std::vector<T> &values) {
  std::vector<T> buffer(std::min<size_t>(4096, values.size()));
  for (size_t start = 0; start < values.size(); start += buffer.size()) {
    const size_t count = std::min(buffer.size(), values.size() - start);
    Read(stream, buffer.data(), count * sizeof(T));
    Require(std::memcmp(buffer.data(), values.data() + start, count * sizeof(T)) == 0,
            "sidecar mapping mismatch");
  }
}
void VerifySidecar(const LayoutExportConfig &config, const LayoutExportResult &layout,
                   size_t nodes) {
  std::ifstream stream(config.out_meta_path, std::ios::binary);
  Require(Read<uint64_t>(stream) == kLayoutMagic, "sidecar magic");
  Require(Read<uint64_t>(stream) == layout.layout_version, "sidecar version");
  Require(Read<uint64_t>(stream) == nodes, "sidecar node count");
  Require(Read<uint64_t>(stream) == layout.levels.size(), "sidecar level count");
  for (const auto &level : layout.levels)
    Match(stream, level.disk_blocks);
  Require(Read<uint8_t>(stream) == 1, "missing vector mapping");
  Match(stream, layout.vec_block_ids);
  Match(stream, layout.vec_slots);
  Require(Read<uint8_t>(stream) == 0, "base layout must have no inline cache");
  Require(stream.peek() == std::char_traits<char>::eof(), "trailing sidecar bytes");
}
} // namespace

void VerifyLayoutExport(AllianceIndexDisk<int, float> &source, const LayoutExportConfig &config,
                        const LayoutExportResult &layout,
                        const std::vector<LevelShuffleResult> &orders) {
  const size_t nodes = source.curvec_num_, vector_bytes = source.VecBytes();
  Require(nodes > 0 && orders.size() == layout.levels.size(), "invalid export dimensions");
  Require(std::filesystem::file_size(config.out_graph_path) == layout.total_bytes,
          "graph file size");
  VerifySidecar(config, layout, nodes);
  std::ifstream graph(config.out_graph_path, std::ios::binary);
  const auto header = Read<GraphLayoutHeader>(graph);
  Require(header.magic == kLayoutMagic && header.version == layout.layout_version &&
              header.max_elements == source.max_elements_ && header.curvec_num == nodes &&
              header.vec_d == source.vec_d_ && header.wp == source.wp_ && header.M == source.M_ &&
              header.cur_max_layer == size_t(source.cur_max_layer_) &&
              header.block_size == config.block_size && header.num_levels == layout.levels.size() &&
              header.vec_region_enabled == 1 &&
              header.layer_stride == LayerStride(config.level_config.shuffle_o),
          "graph header");
  if (layout.layout_version == kMixedLayoutVersion) {
    const auto geometry = Read<MixedVectorGeometry>(graph);
    Require(geometry.block_size == layout.vector_block_size &&
                geometry.region_offset_bytes == layout.vector_region_offset_bytes,
            "mixed vector geometry");
  }
  for (const auto &level : layout.levels) {
    const auto entry = Read<LevelHeaderEntry>(graph);
    Require(
        entry.layer_begin == level.range.layer_begin && entry.layer_end == level.range.layer_end &&
            entry.owner_begin == level.range.owner_begin &&
            entry.owner_end == level.range.owner_end && entry.record_size == level.record_size &&
            entry.elements_per_block == level.elements_per_block &&
            entry.num_blocks == level.num_blocks &&
            entry.region_start_block == level.region_start_block,
        "level header");
  }
  const auto vector_header = Read<VecHeaderEntry>(graph);
  Require(vector_header.elements_per_block == layout.vec_elements_per_block &&
              vector_header.num_blocks == layout.vec_region_num_blocks &&
              vector_header.region_start_block == layout.vec_region_start_block,
          "vector header");
  const size_t used_header = static_cast<size_t>(graph.tellg());
  const size_t header_bytes =
      (used_header + config.block_size - 1) / config.block_size * config.block_size;

  // Check every stored replica, including overlap layers. Distances are not a
  // shuffled field; each ID/count slot and its neighbor order must be exact.
  std::vector<char> block(config.block_size);
  for (size_t li = 0; li < layout.levels.size(); ++li) {
    const auto &level = layout.levels[li];
    std::vector<layer_t> layers;
    for (layer_t layer = level.range.layer_begin; layer <= level.range.layer_end;
         layer += static_cast<layer_t>(header.layer_stride))
      layers.push_back(layer);
    size_t low = source.EdgeOffsetInRecord(layers.back());
    size_t high = source.EdgeOffsetInRecord(layers.front()) + source.EdgeIdBytes();
    const size_t window = high - low;
    std::vector<char> raw(level.elements_per_block * window);
    std::vector<AioSlice> slices(level.elements_per_block);
    graph.seekg(header_bytes + level.region_start_block * config.block_size);
    for (size_t page = 0, base = 0; base < nodes; ++page) {
      Read(graph, block.data(), block.size());
      const size_t count = std::min(level.elements_per_block, nodes - base);
      for (size_t slot = 0; slot < count; ++slot) {
        const auto id = orders[li].order[base + slot];
        slices[slot] = {raw.data() + slot * window, window,
                        source.graph_file_->node_offset(source.DiskSlot(id)) + low};
      }
      source.graph_file_->ReadSlices(slices.data(), count);
      for (size_t slot = 0; slot < count; ++slot) {
        const char *record = block.data() + slot * level.record_size;
        tableint id;
        label_t label;
        int attribute;
        std::memcpy(&id, record, sizeof(id));
        record += sizeof(id);
        std::memcpy(&label, record, sizeof(label));
        record += sizeof(label);
        std::memcpy(&attribute, record, sizeof(attribute));
        record += sizeof(attribute);
        Require(id == orders[li].order[base + slot], "physical node order");
        Require(label == source.labels_[id] && attribute == source.atts_[id], "label or attribute");
        Require(level.disk_blocks[id] == static_cast<int32_t>(level.region_start_block + page),
                "neighbor block mapping");
        for (auto layer : layers) {
          Require(std::memcmp(record,
                              raw.data() + slot * window + source.EdgeOffsetInRecord(layer) - low,
                              source.EdgeIdBytes()) == 0,
                  "edge IDs/count/order");
          record += source.EdgeIdBytes();
        }
      }
      base += count;
    }
  }

  block.resize(layout.vector_block_size);
  std::vector<char> raw(layout.vec_elements_per_block * vector_bytes);
  std::vector<AioSlice> slices(layout.vec_elements_per_block);
  graph.seekg(layout.vector_region_offset_bytes);
  for (size_t page = 0, base = 0; base < nodes; ++page) {
    Read(graph, block.data(), block.size());
    const size_t count = std::min(layout.vec_elements_per_block, nodes - base);
    for (size_t slot = 0; slot < count; ++slot) {
      const auto id = orders.front().order[base + slot];
      slices[slot] = {raw.data() + slot * vector_bytes, vector_bytes,
                      source.graph_file_->node_offset(source.DiskSlot(id)) +
                          source.VecOffsetInRecord()};
    }
    source.graph_file_->ReadSlices(slices.data(), count);
    for (size_t slot = 0; slot < count; ++slot) {
      const auto id = orders.front().order[base + slot];
      Require(std::memcmp(raw.data() + slot * vector_bytes, block.data() + slot * vector_bytes,
                          vector_bytes) == 0,
              "vector bytes");
      const size_t expected_block = layout.layout_version == kMixedLayoutVersion
                                        ? page
                                        : layout.vec_region_start_block + page;
      Require(layout.vec_block_ids[id] == expected_block && layout.vec_slots[id] == slot,
              "vector block/slot mapping");
    }
    base += count;
  }
}
} // namespace alliance::shuffle_detail
