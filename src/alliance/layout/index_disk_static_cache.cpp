// Offline static-cache budget allocation, selection and sidecar preparation.
#include "index/index_disk.hpp"
#include "io/aio_stream.hpp"
#include "search/query_visited.hpp"
#include "shuffle/layout.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <unistd.h>

namespace alliance {

template <typename att_t, typename vec_t>
void AllianceIndexDisk<att_t, vec_t>::AllocateStaticGroups(const std::vector<LevelRange> &levels,
                                                           const std::vector<size_t> &record_sizes,
                                                           size_t level_k, size_t level_m,
                                                           size_t stride, size_t budget_bytes,
                                                           size_t num_nodes) {
  static_groups_.clear();
  if (levels.empty() || budget_bytes == 0 || num_nodes == 0)
    return;
  ml_layer_stride_ = (stride > 0) ? stride : 1;

  ValidateLevelRanges(levels, levels.front().layer_end, ml_layer_stride_);
  const bool have_landing = !landing_layer_cnt_.empty();
  static_groups_.resize(levels.size());
  for (size_t li = 0; li < levels.size(); ++li) {
    static_groups_[li].Init(levels[li].layer_begin, levels[li].layer_end, record_sizes[li],
                            num_nodes);
  }

  if (have_landing) {
    // Use the same routing as queries, including landings between selected
    // grid layers. Counting only exact selected layers loses stride>1 traffic.
    const size_t count =
        std::min(landing_layer_cnt_.size(), static_cast<size_t>(levels.front().owner_end) + 1);
    for (size_t l = 0; l < count; ++l) {
      const size_t owner = FindLevelForLayer(static_cast<layer_t>(l), levels, ml_layer_stride_);
      static_groups_[owner].landing_weight += landing_layer_cnt_[l];
    }
  }

  uint64_t total_weight = 0;
  for (const auto &g : static_groups_)
    total_weight += g.landing_weight;
  const bool use_weight = have_landing && total_weight > 0;
  const double equal_weight = 1.0 / static_cast<double>(static_groups_.size());

  for (auto &g : static_groups_) {
    if (g.piece_bytes == 0) {
      g.slot_count = 0;
      continue;
    }
    const double w = use_weight
                         ? static_cast<double>(g.landing_weight) / static_cast<double>(total_weight)
                         : equal_weight;
    g.slot_count = static_cast<size_t>(static_cast<double>(budget_bytes) * w /
                                       static_cast<double>(g.piece_bytes));
  }

  std::vector<size_t> order(static_groups_.size());
  std::iota(order.begin(), order.end(), size_t(0));
  if (use_weight) {
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      if (static_groups_[a].landing_weight != static_groups_[b].landing_weight)
        return static_groups_[a].landing_weight > static_groups_[b].landing_weight;
      return static_groups_[a].layer_end > static_groups_[b].layer_end;
    });
  }
  size_t surplus_bytes = 0;
  for (size_t idx : order) {
    auto &g = static_groups_[idx];
    if (g.piece_bytes == 0)
      continue;
    size_t extra = surplus_bytes / g.piece_bytes;
    g.slot_count += extra;
    surplus_bytes -= extra * g.piece_bytes;
    if (g.slot_count > num_nodes) {
      surplus_bytes += (g.slot_count - num_nodes) * g.piece_bytes;
      g.slot_count = num_nodes;
    }
  }
  size_t used_bytes = 0;
  for (const auto &g : static_groups_)
    used_bytes += g.slot_count * g.piece_bytes;
  size_t remaining = (budget_bytes > used_bytes) ? (budget_bytes - used_bytes) : 0;
  for (size_t idx : order) {
    auto &g = static_groups_[idx];
    if (g.piece_bytes == 0 || g.slot_count >= num_nodes)
      continue;
    size_t extra = std::min(remaining / g.piece_bytes, num_nodes - g.slot_count);
    g.slot_count += extra;
    remaining -= extra * g.piece_bytes;
  }
}

template <typename att_t, typename vec_t>
auto AllianceIndexDisk<att_t, vec_t>::ComputeLevelInWeights(layer_t layer_begin, layer_t layer_end,
                                                            size_t stride) const
    -> std::vector<size_t> {
  const size_t n = curvec_num_;
  std::vector<size_t> in_weights(n, 0);
  std::vector<tableint> buf(M_ + 1);
  const layer_t layer_step = static_cast<layer_t>(stride);
  for (size_t u = 0; u < n; ++u) {
    for (layer_t layer = layer_begin; layer <= layer_end; layer += layer_step) {
      ReadEdges(static_cast<tableint>(u), layer, buf.data());
      const tableint sz = buf[M_];
      for (tableint j = 0; j < sz; ++j) {
        const tableint v = buf[j];
        if (static_cast<size_t>(v) < n) {
          ++in_weights[v];
        }
      }
    }
  }
  return in_weights;
}

template <typename att_t, typename vec_t>
void AllianceIndexDisk<att_t, vec_t>::SelectAndMergeStaticNodes(
    const std::vector<LevelRange> &levels, const std::vector<size_t> &record_sizes, size_t stride,
    const std::vector<std::vector<size_t>> *weights) {
  // Keep the original v2 selection and 16-byte budget accounting first.
  // Retry only the metadata selection if its actual packed payload cannot fit.
  // Sidecar preparation supplies the already-scanned weights to both passes.
  std::vector<StaticNodeMeta64>().swap(static_node_meta64_);
  auto select = [&]<class Meta>(std::vector<Meta> &metadata) {
    const size_t n = curvec_num_;
    const size_t num_levels = levels.size();
    metadata.assign(n, Meta{});

    // Per-level selection: rank hubs using the same owned layers as its shuffle.
    // Records still retain the full range so cached nodes can serve descent.
    // Store per-level selected sets as sorted vectors.
    std::vector<std::vector<tableint>> per_level_selected(num_levels);

    for (size_t li = 0; li < num_levels; ++li) {
      if (static_groups_[li].slot_count == 0)
        continue;
      const auto &g = static_groups_[li];

      fprintf(stderr, "  [static-cache] computing in-weights for level %zu owned=[%d..%d]...\n", li,
              levels[li].owner_begin, levels[li].owner_end);
      std::vector<size_t> computed;
      if (!weights)
        computed = ComputeLevelInWeights(levels[li].owner_begin, levels[li].owner_end, stride);
      if (weights && (weights->size() != num_levels || weights->at(li).size() != n))
        throw std::invalid_argument("static cache weight shape mismatch");
      const auto &in_weights = weights ? weights->at(li) : computed;

      std::vector<tableint> ranked(n);
      std::iota(ranked.begin(), ranked.end(), tableint(0));
      std::sort(ranked.begin(), ranked.end(), [&](tableint a, tableint b) {
        if (in_weights[a] != in_weights[b])
          return in_weights[a] > in_weights[b];
        return a < b;
      });

      size_t count = std::min(g.slot_count, n);
      per_level_selected[li].reserve(count);
      for (size_t i = 0; i < count; ++i) {
        if (in_weights[ranked[i]] == 0)
          break;
        per_level_selected[li].push_back(ranked[i]);
      }
    }

    // Merge: for each selected node, compute its merged layer range across all
    // levels it appears in. The merged range = [min(lb), max(le)].
    const size_t record_header = sizeof(tableint) + sizeof(label_t) + sizeof(att_t);
    const size_t per_layer_bytes = (M_ + 1) * sizeof(tableint);

    for (size_t li = 0; li < num_levels; ++li) {
      for (tableint id : per_level_selected[li]) {
        auto &m = metadata[id];
        if (!m.IsCached()) {
          m.layer_begin = levels[li].layer_begin;
          m.layer_end = levels[li].layer_end;
          m.offset = 0; // placeholder, will be computed later
        } else {
          if (levels[li].layer_begin < m.layer_begin)
            m.layer_begin = levels[li].layer_begin;
          if (levels[li].layer_end > m.layer_end)
            m.layer_end = levels[li].layer_end;
        }
      }
    }

    // Compute per-node record size based on merged layer range.
    size_t total_bytes = 0;
    size_t total_cached = 0;
    for (size_t id = 0; id < n; ++id) {
      auto &m = metadata[id];
      if (!m.IsCached())
        continue;
      const size_t num_layers =
          static_cast<size_t>((m.layer_end - m.layer_begin) / static_cast<layer_t>(stride)) + 1;
      m.record_bytes = static_cast<uint32_t>(record_header + num_layers * per_layer_bytes);
      total_bytes += m.record_bytes;
      ++total_cached;
    }

    // Enforce the query-resident budget, including compact metadata and padding.
    auto resident_bytes = [&]() {
      return StaticNodeLookup::BudgetBytes(n, total_cached, total_bytes,
                                           std::is_same_v<Meta, StaticNodeMeta64>);
    };
    if (resident_bytes() > static_cache_budget_bytes_ && total_cached > 0) {
      // Per-node score = sum of (slot_count - rank) across levels it appears in.
      std::vector<size_t> node_score(n, 0);
      for (size_t li = 0; li < num_levels; ++li) {
        const size_t sc = static_groups_[li].slot_count;
        for (size_t rank = 0; rank < per_level_selected[li].size(); ++rank) {
          tableint id = per_level_selected[li][rank];
          node_score[id] += (rank < sc) ? (sc - rank) : 0;
        }
      }
      // Collect cached nodes sorted by score ascending
      std::vector<std::pair<size_t, tableint>> scored;
      scored.reserve(total_cached);
      for (size_t id = 0; id < n; ++id) {
        if (metadata[id].IsCached())
          scored.push_back({node_score[id], static_cast<tableint>(id)});
      }
      std::sort(scored.begin(), scored.end());
      for (auto &[score, id] : scored) {
        if (resident_bytes() <= static_cache_budget_bytes_)
          break;
        total_bytes -= metadata[id].record_bytes;
        metadata[id] = Meta{};
        --total_cached;
      }
    }

    // Assign byte offsets sequentially by node_id order.
    size_t running_offset = 0;
    for (size_t id = 0; id < n; ++id) {
      auto &m = metadata[id];
      if (!m.IsCached())
        continue;
      if (running_offset >= Meta::kInvalid || m.record_bytes > Meta::kInvalid - running_offset)
        throw StaticCacheOffsetOverflow();
      m.offset = static_cast<decltype(Meta::offset)>(running_offset);
      running_offset += m.record_bytes;
    }
    static_cache_bytes_ = running_offset;

    fprintf(stderr, "  [static-cache] merged: %zu nodes cached, %zu bytes total\n", total_cached,
            static_cache_bytes_);
  };
  try {
    select(static_node_meta_);
  } catch (const StaticCacheOffsetOverflow &) {
    std::vector<StaticNodeMeta>().swap(static_node_meta_);
    select(static_node_meta64_);
  }
}

template <typename att_t, typename vec_t>
void AllianceIndexDisk<att_t, vec_t>::PrepareStaticCacheSidecars(
    const std::string &graph, const std::string &sidecar,
    const std::vector<std::pair<std::string, size_t>> &outputs) {
  std::vector<std::filesystem::path> paths;
  for (const auto &[path, budget] : outputs) {
    auto canonical = std::filesystem::weakly_canonical(path);
    if (std::filesystem::exists(path) || canonical == std::filesystem::weakly_canonical(graph) ||
        canonical == std::filesystem::weakly_canonical(sidecar) ||
        std::find(paths.begin(), paths.end(), canonical) != paths.end())
      throw std::invalid_argument("static output must be a new, distinct path: " + path);
    paths.push_back(canonical);
  }
  LoadMultiLevelLayout(graph, sidecar, 0);
  if (static_cache_buf_) {
    free(static_cache_buf_);
    static_cache_buf_ = nullptr;
  }
  static_cache_bytes_ = 0;
  static_cache_loaded_ = false;
  static_node_lookup_.Clear();
  GraphLayoutHeader h{};
  std::ifstream gh(graph, std::ios::binary);
  gh.read(reinterpret_cast<char *>(&h), sizeof(h));
  if (h.version == kMixedLayoutVersion)
    gh.seekg(sizeof(MixedVectorGeometry), std::ios::cur);
  std::vector<LevelHeaderEntry> entries(h.num_levels);
  gh.read(reinterpret_cast<char *>(entries.data()), entries.size() * sizeof(LevelHeaderEntry));
  if (!gh)
    throw std::runtime_error("cannot read layout entries");
  const size_t n = curvec_num_, prefix = sizeof(tableint) + sizeof(label_t) + sizeof(att_t);
  const size_t edge_bytes = (M_ + 1) * sizeof(tableint);
  std::vector<std::vector<size_t>> weights(entries.size(), std::vector<size_t>(n, 0));
  auto scan = [&](size_t li, auto visit) {
    const auto &e = entries[li];
    AioSeqStreamReader input(graph, ml_header_bytes_ + e.region_start_block * ml_block_size_,
                             ml_header_bytes_ +
                                 (e.region_start_block + e.num_blocks) * ml_block_size_);
    std::vector<char> block(ml_block_size_);
    QueryVisited seen(n);
    size_t count = 0;
    for (size_t b = 0; b < e.num_blocks; ++b) {
      input.Read(block.data(), block.size());
      for (size_t slot = 0; slot < e.elements_per_block; ++slot) {
        const char *record = block.data() + slot * e.record_size;
        tableint id;
        std::memcpy(&id, record, sizeof(id));
        if (id == std::numeric_limits<tableint>::max())
          continue;
        if (id >= n || seen.Test(id))
          throw std::runtime_error("invalid/duplicate layout record");
        seen.Set(id);
        ++count;
        visit(id, record);
      }
    }
    if (count != n)
      throw std::runtime_error("incomplete layout level");
  };
  for (size_t li = 0; li < entries.size(); ++li) {
    const auto &e = entries[li];
    scan(li, [&](tableint, const char *record) {
      for (layer_t l = e.owner_begin; l <= e.owner_end;
           l += static_cast<layer_t>(ml_layer_stride_)) {
        const char *edges = record + prefix + (l - e.layer_begin) / ml_layer_stride_ * edge_bytes;
        tableint degree;
        std::memcpy(&degree, edges + M_ * sizeof(tableint), sizeof(degree));
        if (degree > M_)
          throw std::runtime_error("invalid layout degree");
        for (tableint j = 0; j < degree; ++j) {
          tableint v;
          std::memcpy(&v, edges + j * sizeof(tableint), sizeof(v));
          if (v >= n)
            throw std::runtime_error("invalid layout neighbor");
          ++weights[li][v];
        }
      }
    });
  }
  // The prefix contains only header, per-level maps, and optional vector maps.
  std::ifstream input(sidecar, std::ios::binary);
  size_t prefix_bytes = 4 * sizeof(uint64_t) + entries.size() * n * sizeof(int32_t);
  input.seekg(prefix_bytes);
  uint8_t vec_flag = 0;
  input.read(reinterpret_cast<char *>(&vec_flag), 1);
  if (!input)
    throw std::runtime_error("missing sidecar vector flag");
  ++prefix_bytes;
  if (vec_flag)
    prefix_bytes += 2 * n * sizeof(size_t);
  for (const auto &[path, budget] : outputs) {
    static_cache_budget_bytes_ = budget;
    AllocateStaticGroups(ml_levels_, ml_record_size_per_level_, 0, 0, ml_layer_stride_, budget, n);
    if (budget)
      SelectAndMergeStaticNodes(ml_levels_, ml_record_size_per_level_, ml_layer_stride_, &weights);
    else {
      static_node_meta_.clear();
      static_node_meta64_.clear();
      static_cache_bytes_ = 0;
    }
    AllocateStaticCacheBuffer();
    if (static_cache_bytes_)
      for (size_t li = 0; li < entries.size(); ++li)
        scan(li, [&](tableint id, const char *record) {
          FillStaticCacheRecord(li, id, record, entries[li].record_size);
        });
    // O_EXCL protects existing results even when two preparers race.
    int out_fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (out_fd < 0)
      throw std::runtime_error("cannot create static sidecar: " + path);
    auto write_all = [&](const void *ptr, size_t bytes) {
      const char *p = static_cast<const char *>(ptr);
      while (bytes) {
        ssize_t count = ::write(out_fd, p, bytes);
        if (count <= 0)
          throw std::runtime_error("sidecar write failed");
        p += count;
        bytes -= count;
      }
    };
    try {
      input.clear();
      input.seekg(0);
      std::vector<char> chunk(1024 * 1024);
      for (size_t off = 0; off < prefix_bytes;) {
        size_t bytes = std::min(chunk.size(), prefix_bytes - off);
        input.read(chunk.data(), bytes);
        if (!input)
          throw std::runtime_error("truncated sidecar prefix");
        write_all(chunk.data(), bytes);
        off += bytes;
      }
      const uint8_t flag = !static_cache_bytes_
                               ? 0
                               : (static_node_meta64_.empty() ? kStaticCacheV2 : kStaticCacheV3);
      write_all(&flag, sizeof(flag));
      if (flag) {
        const uint64_t bytes = static_cache_bytes_, nodes = n;
        write_all(&bytes, sizeof(bytes));
        write_all(&nodes, sizeof(nodes));
        if (flag == kStaticCacheV3)
          write_all(static_node_meta64_.data(), n * sizeof(StaticNodeMeta64));
        else
          write_all(static_node_meta_.data(), n * sizeof(StaticNodeMeta));
        write_all(static_cache_buf_, static_cache_bytes_);
      }
      if (::fsync(out_fd) != 0)
        throw std::runtime_error("sidecar sync failed");
      ::close(out_fd);
      out_fd = -1;
    } catch (...) {
      if (out_fd >= 0)
        ::close(out_fd);
      ::unlink(path.c_str());
      throw;
    }
    size_t cached = 0;
    for (const auto &m : static_node_meta_)
      cached += m.IsCached();
    for (const auto &m : static_node_meta64_)
      cached += m.IsCached();
    std::cout << "# static_prepared: path=" << path << " budget_bytes=" << budget
              << " resident_bytes="
              << StaticNodeLookup::BudgetBytes(n, cached, static_cache_bytes_,
                                               !static_node_meta64_.empty())
              << " offset_bits=" << (static_node_meta64_.empty() ? 32 : 64)
              << " payload_bytes=" << static_cache_bytes_ << " cached_nodes=" << cached
              << std::endl;
  }
  // Return to a valid search-ready view of the original sidecar.
  LoadMultiLevelLayout(graph, sidecar, 0);
}

template <typename att_t, typename vec_t>
void AllianceIndexDisk<att_t, vec_t>::AllocateStaticCacheBuffer() {
  if (static_cache_buf_) {
    free(static_cache_buf_);
    static_cache_buf_ = nullptr;
  }
  if (static_cache_bytes_ == 0)
    return;
  const size_t aligned = ((static_cache_bytes_ + 4095) / 4096) * 4096;
  static_cache_buf_ = static_cast<char *>(aligned_alloc(4096, aligned));
  if (!static_cache_buf_) {
    throw std::runtime_error("AllocateStaticCacheBuffer: aligned_alloc failed");
  }
  std::memset(static_cache_buf_, 0, aligned);
}

template <typename att_t, typename vec_t>
void AllianceIndexDisk<att_t, vec_t>::FillStaticCacheRecord(size_t li, tableint node_id,
                                                            const char *record, size_t bytes) {
  if (static_cache_buf_ == nullptr)
    return;
  auto fill = [&](const auto &m) {
    if (!m.IsCached())
      return;
    if (li >= static_groups_.size())
      return;

    // The incoming record covers [level.layer_begin, level.layer_end] at stride.
    // The node's cached record covers [m.layer_begin, m.layer_end] at stride.
    // Copy: header (once, from first call) + each layer's edges into the
    // correct position within the node's merged record.
    const size_t record_header = sizeof(tableint) + sizeof(label_t) + sizeof(att_t);
    const size_t per_layer_bytes = (M_ + 1) * sizeof(tableint);
    const layer_t layer_stride = static_cast<layer_t>(ml_layer_stride_ > 0 ? ml_layer_stride_ : 1);
    const layer_t level_lb = static_groups_[li].layer_begin;
    const layer_t level_le = static_groups_[li].layer_end;

    char *dst_base = static_cache_buf_ + m.offset;

    // Copy header from incoming record (idempotent: same node_id/label/att)
    std::memcpy(dst_base, record, record_header);

    // Copy each layer's edge data from the incoming record to the correct
    // position in the merged record.
    size_t src_layer_idx = 0;
    for (layer_t layer = level_lb; layer <= level_le; layer += layer_stride) {
      if (layer >= m.layer_begin && layer <= m.layer_end) {
        const size_t dst_layer_idx = static_cast<size_t>((layer - m.layer_begin) / layer_stride);
        std::memcpy(dst_base + record_header + dst_layer_idx * per_layer_bytes,
                    record + record_header + src_layer_idx * per_layer_bytes, per_layer_bytes);
      }
      ++src_layer_idx;
    }
    (void)bytes;
  };
  if (!static_node_meta64_.empty()) {
    if (size_t(node_id) < static_node_meta64_.size())
      fill(static_node_meta64_[node_id]);
  } else if (size_t(node_id) < static_node_meta_.size())
    fill(static_node_meta_[node_id]);
}

template void AllianceIndexDisk<int, float>::AllocateStaticGroups(const std::vector<LevelRange> &,
                                                                  const std::vector<size_t> &,
                                                                  size_t, size_t, size_t, size_t,
                                                                  size_t);
template auto AllianceIndexDisk<int, float>::ComputeLevelInWeights(layer_t, layer_t, size_t) const
    -> std::vector<size_t>;
template void
AllianceIndexDisk<int, float>::SelectAndMergeStaticNodes(const std::vector<LevelRange> &,
                                                         const std::vector<size_t> &, size_t,
                                                         const std::vector<std::vector<size_t>> *);
template void AllianceIndexDisk<int, float>::PrepareStaticCacheSidecars(
    const std::string &, const std::string &, const std::vector<std::pair<std::string, size_t>> &);
template void AllianceIndexDisk<int, float>::AllocateStaticCacheBuffer();
template void AllianceIndexDisk<int, float>::FillStaticCacheRecord(size_t, tableint, const char *,
                                                                   size_t);

} // namespace alliance
