#include "shuffle/format/plan.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace alliance::shuffle_detail {
namespace {
size_t Add(size_t a, size_t b) {
  if (b > std::numeric_limits<size_t>::max() - a)
    throw std::overflow_error("layout byte offset overflow");
  return a + b;
}
size_t Multiply(size_t a, size_t b) {
  if (a && b > std::numeric_limits<size_t>::max() / a)
    throw std::overflow_error("layout byte count overflow");
  return a * b;
}
size_t Blocks(size_t count, size_t capacity) {
  return count / capacity + (count % capacity != 0);
}
void Page(size_t bytes) {
  if (bytes < 512 || (bytes & (bytes - 1)))
    throw std::invalid_argument("layout page must be an aligned power of two");
}
} // namespace
LayoutPlan PlanLayout(size_t nodes, size_t vector_bytes, size_t degree, size_t attribute_bytes,
                      const std::vector<LevelRange> &levels, const LayoutExportConfig &config) {
  if (!nodes || !vector_bytes || levels.empty())
    throw std::invalid_argument("layout requires nodes, vectors, and levels");
  Page(config.block_size);
  const size_t vector_page =
      config.vector_block_size ? config.vector_block_size : config.block_size;
  Page(vector_page);
  if (vector_bytes > vector_page)
    throw std::invalid_argument("vector does not fit vector_block_size");
  LayoutPlan plan;
  auto &result = plan.layout;
  result.vector_block_size = vector_page;
  result.layout_version = vector_page == config.block_size ? kLayoutVersion : kMixedLayoutVersion;
  size_t neighbor_blocks = 0;
  for (const auto &range : levels) {
    LayoutExportResult::LevelInfo level;
    level.range = range;
    const size_t selected = CountSelectedLayers(range.layer_begin, range.layer_end,
                                                LayerStride(config.level_config.shuffle_o));
    level.record_size = Add(Add(sizeof(tableint), Add(sizeof(label_t), attribute_bytes)),
                            Multiply(selected, Multiply(Add(degree, 1), sizeof(tableint))));
    if (level.record_size > config.block_size)
      throw std::invalid_argument("neighbor record exceeds block_size");
    level.elements_per_block = config.block_size / level.record_size;
    level.num_blocks = Blocks(nodes, level.elements_per_block);
    level.region_start_block = neighbor_blocks;
    neighbor_blocks = Add(neighbor_blocks, level.num_blocks);
    if (neighbor_blocks > static_cast<size_t>(std::numeric_limits<int32_t>::max()))
      throw std::overflow_error("neighbor blocks exceed persisted int32 mapping");
    result.levels.push_back(std::move(level));
  }
  const size_t header_written =
      Add(Add(sizeof(GraphLayoutHeader), sizeof(VecHeaderEntry)),
          Add(Multiply(levels.size(), sizeof(LevelHeaderEntry)),
              result.layout_version == kMixedLayoutVersion ? sizeof(MixedVectorGeometry) : 0));
  plan.header_padding =
      (config.block_size - header_written % config.block_size) % config.block_size;
  const size_t header_bytes = Add(header_written, plan.header_padding);
  plan.neighbor_end = Add(header_bytes, Multiply(neighbor_blocks, config.block_size));
  const size_t alignment = std::max(config.block_size, vector_page);
  const size_t gap = (alignment - plan.neighbor_end % alignment) % alignment;
  result.vector_region_offset_bytes = Add(plan.neighbor_end, gap);
  result.vec_elements_per_block = vector_page / vector_bytes;
  result.vec_region_num_blocks = Blocks(nodes, result.vec_elements_per_block);
  result.vec_region_start_block =
      (result.vector_region_offset_bytes - header_bytes) / config.block_size;
  result.total_bytes =
      Add(result.vector_region_offset_bytes, Multiply(result.vec_region_num_blocks, vector_page));
  if (result.total_bytes > static_cast<size_t>(std::numeric_limits<int64_t>::max()))
    throw std::overflow_error("layout exceeds signed file offsets");
  return plan;
}
} // namespace alliance::shuffle_detail
