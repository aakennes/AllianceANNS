// test_layout_export.cpp -- Verify ExportMultiLevelLayout's on-disk format.
//
// Builds a small multi-layer disk index, exports the multi-level layout, then
// acts as an independent "reader" of the raw output files (out_graph_path +
// out_meta_path) -- replicating the binary format documented in layout.hpp --
// to confirm every node's per-level linklist and raw vector on disk matches
// what AllianceIndexDisk itself reports via ReadEdges()/ReadVec().
//
// Run: ./bin/test_layout_export

#include <cstdio>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <filesystem>
#include <cstdlib>
#include "index/index_disk.hpp"
#include "shuffle/shuffle.hpp"
#include "shuffle/layout.hpp"

using namespace alliance;

namespace {

// Independently mirrors the on-disk POD layout (also now public in
// layout.hpp as GraphLayoutHeader/LevelHeaderEntry/VecHeaderEntry) under a
// distinct name -- this test intentionally avoids depending on those shared
// structs so a mistake in either definition doesn't silently cancel out.
struct TestGraphLayoutHeader {
  uint64_t magic;
  uint64_t version;
  uint64_t max_elements;
  uint64_t vec_d;
  uint64_t wp;
  uint64_t M;
  uint64_t curvec_num;
  uint64_t cur_max_layer;
  uint64_t block_size;
  uint64_t num_levels;
  uint64_t vec_region_enabled;
  uint64_t layer_stride;
};

struct TestLevelHeaderEntry {
  int32_t layer_begin;
  int32_t layer_end;
  int32_t owner_begin;
  int32_t owner_end;
  uint64_t record_size;
  uint64_t elements_per_block;
  uint64_t num_blocks;
  uint64_t region_start_block;
};

struct TestVecHeaderEntry {
  uint64_t elements_per_block;
  uint64_t num_blocks;
  uint64_t region_start_block;
};

template <typename T> void ReadPOD(std::istream &is, T &v) {
  is.read(reinterpret_cast<char *>(&v), sizeof(T));
}

} // namespace

int main(int argc, char **argv) {
  printf("====== ExportMultiLevelLayout Tests ======\n\n");

  const size_t n = 3000;
  const size_t d = 128;
  const size_t M = 16;
  const size_t efc = 64;
  const size_t o = 4;
  const size_t wp = 5;
  const size_t block_size = 4096;
  const size_t shuffle_o = argc > 1 ? std::stoull(argv[1]) : 2;
  const size_t stride = LayerStride(shuffle_o);
  const char *tmp_env = std::getenv("TEST_TMPDIR");
  const std::string tmp_dir = tmp_env ? tmp_env : "/tmp";

  std::mt19937 rng(7);
  std::normal_distribution<float> gauss(0.0f, 1.0f);
  std::vector<float> base_vecs(n * d);
  for (auto &v : base_vecs)
    v = gauss(rng);
  std::vector<int> att_vec(n);
  std::iota(att_vec.begin(), att_vec.end(), 0);

  const std::string graph_path = tmp_dir + "/test_layout_export_graph.bin";
  std::remove(graph_path.c_str());

  AllianceIndexDisk<int, float> idx(n, d, M, efc, "l2", graph_path, o, wp, /*auto_raise_wp=*/false);
  idx.Build(base_vecs.data(), att_vec.data(), n, /*threads=*/1);
  idx.Save(tmp_dir + "/test_layout_export_index.meta");

  printf("Built index: n=%zu M=%zu wp=%zu cur_max_layer=%zu\n\n", n, M, wp, idx.cur_max_layer_);

  const std::string out_graph = tmp_dir + "/test_layout_export_ml_graph.bin";
  const std::string out_meta = tmp_dir + "/test_layout_export_ml_meta.bin";
  std::remove(out_graph.c_str());
  std::remove(out_meta.c_str());

  LayoutExportConfig config;
  config.block_size = block_size;
  config.level_config = MultiLevelConfig{2, 2, shuffle_o, {}};
  config.method = ShuffleMethod::BlockBNF;
  config.out_graph_path = out_graph;
  config.out_meta_path = out_meta;
  idx.landing_layer_cnt_ = {1, 2, 3, 4, 5, 6};

  const auto orders = ComputeMultiLevelShuffleOrders(&idx, config.level_config, config.method);
  auto result = ExportMultiLevelLayout(&idx, config, orders);
  const auto ranges =
      GenerateLevelRanges(static_cast<layer_t>(idx.cur_max_layer_), config.level_config);
  // Production shuffle writes a cache-free layout; prepare a separate sidecar.
  const auto cached_meta = out_meta + ".cached";
  std::remove(cached_meta.c_str());
  idx.PrepareStaticCacheSidecars(out_graph, out_meta, {{cached_meta, 16 * 1024}});

  printf("Export result: %zu levels, vec_region_num_blocks=%zu, total_bytes=%zu\n",
         result.levels.size(), result.vec_region_num_blocks, result.total_bytes);
  for (size_t li = 0; li < result.levels.size(); ++li) {
    auto &lv = result.levels[li];
    printf("  Level %zu [%d..%d]: record=%zuB epb=%zu num_blocks=%zu region_start_block=%zu\n", li,
           lv.range.layer_begin, lv.range.layer_end, lv.record_size, lv.elements_per_block,
           lv.num_blocks, lv.region_start_block);
  }
  printf("\n");

  // --- Read back the header from out_graph_path ---
  std::ifstream gfs(out_graph, std::ios::binary);
  if (!gfs.is_open()) {
    printf("FAIL: could not open out_graph_path for readback\n");
    return 1;
  }
  TestGraphLayoutHeader header{};
  ReadPOD(gfs, header);
  if (header.magic != 0x314D4C4Cull) {
    printf("FAIL: bad magic 0x%lx\n", header.magic);
    return 1;
  }
  if (header.version != 2 || header.layer_stride != stride || header.curvec_num != n ||
      header.M != M || header.wp != wp || header.block_size != block_size) {
    printf("FAIL: header field mismatch (curvec_num=%lu M=%lu wp=%lu block_size=%lu)\n",
           header.curvec_num, header.M, header.wp, header.block_size);
    return 1;
  }
  if (header.num_levels != result.levels.size()) {
    printf("FAIL: num_levels mismatch (header=%lu, result=%zu)\n", header.num_levels,
           result.levels.size());
    return 1;
  }
  printf("Header readback: OK (magic/version/dims all match)\n");

  std::vector<TestLevelHeaderEntry> level_headers(header.num_levels);
  for (auto &e : level_headers)
    ReadPOD(gfs, e);
  TestVecHeaderEntry vec_header{};
  bool vec_colocated = header.vec_region_enabled != 0;
  if (vec_colocated)
    ReadPOD(gfs, vec_header);

  for (size_t li = 0; li < level_headers.size(); ++li) {
    auto &lh = level_headers[li];
    auto &lv = result.levels[li];
    if (lh.layer_begin != lv.range.layer_begin || lh.layer_end != lv.range.layer_end ||
        lh.owner_begin != lv.range.owner_begin || lh.owner_end != lv.range.owner_end ||
        lh.record_size != lv.record_size || lh.elements_per_block != lv.elements_per_block ||
        lh.num_blocks != lv.num_blocks || lh.region_start_block != lv.region_start_block) {
      printf("FAIL: level %zu header entry mismatch vs result\n", li);
      return 1;
    }
  }
  printf("Level header entries: OK (match ExportMultiLevelLayout result)\n\n");

  size_t header_written = sizeof(TestGraphLayoutHeader) +
                          level_headers.size() * sizeof(TestLevelHeaderEntry) +
                          (vec_colocated ? sizeof(TestVecHeaderEntry) : 0);
  size_t header_padding = (block_size - (header_written % block_size)) % block_size;
  size_t header_bytes = header_written + header_padding;

  // --- Verify every node's per-level record matches idx->ReadEdges() ---
  bool all_ok = true;
  std::vector<char> block_buf(block_size);
  std::vector<tableint> expected_edges(M + 1), actual_edges(M + 1);

  for (size_t li = 0; li < level_headers.size(); ++li) {
    auto &lh = level_headers[li];
    auto &lv = result.levels[li];
    const size_t per_layer_id_bytes = (M + 1) * sizeof(tableint);
    size_t checked = 0;

    for (tableint node_id = 0; node_id < static_cast<tableint>(n); ++node_id) {
      int32_t block_id = lv.disk_blocks[node_id];
      if (block_id < 0) {
        printf("FAIL: level %zu node %u has no block assignment\n", li, node_id);
        all_ok = false;
        continue;
      }
      size_t byte_off = header_bytes + static_cast<size_t>(block_id) * block_size;
      gfs.seekg(static_cast<std::streamoff>(byte_off));
      gfs.read(block_buf.data(), static_cast<std::streamsize>(block_size));

      // Linear-scan the block for the record whose embedded node_id matches.
      bool found = false;
      for (size_t off = 0; off + lh.record_size <= block_size; off += lh.record_size) {
        tableint rec_id;
        std::memcpy(&rec_id, block_buf.data() + off, sizeof(tableint));
        if (rec_id != node_id)
          continue;
        found = true;

        label_t rec_label;
        int rec_att;
        std::memcpy(&rec_label, block_buf.data() + off + sizeof(tableint), sizeof(label_t));
        std::memcpy(&rec_att, block_buf.data() + off + sizeof(tableint) + sizeof(label_t),
                    sizeof(int));
        if (rec_label != idx.labels_[node_id] || rec_att != idx.atts_[node_id]) {
          printf("FAIL: level %zu node %u label/att mismatch\n", li, node_id);
          all_ok = false;
        }

        size_t link_off = off + sizeof(tableint) + sizeof(label_t) + sizeof(int);
        for (layer_t layer = lh.layer_begin; layer <= lh.layer_end;
             layer += static_cast<layer_t>(stride)) {
          std::memcpy(actual_edges.data(), block_buf.data() + link_off, per_layer_id_bytes);
          idx.ReadEdges(node_id, layer, expected_edges.data());
          if (std::memcmp(actual_edges.data(), expected_edges.data(), per_layer_id_bytes) != 0) {
            printf("FAIL: level %zu node %u layer %d linklist mismatch\n", li, node_id, layer);
            all_ok = false;
          }
          link_off += per_layer_id_bytes;
        }
        break;
      }
      if (!found) {
        printf("FAIL: level %zu node %u not found in block %d\n", li, node_id, block_id);
        all_ok = false;
      } else {
        ++checked;
      }
    }
    printf("  Level %zu [%d..%d]: verified %zu/%zu node records\n", li, lh.layer_begin,
           lh.layer_end, checked, n);
  }
  printf("\n");

  // --- Verify the vector region (colocated in out_graph_path) ---
  {
    const size_t vec_block_bytes = d * sizeof(float);
    std::vector<float> expected_vec(d), actual_vec(d);
    size_t checked = 0;
    for (tableint node_id = 0; node_id < static_cast<tableint>(n); ++node_id) {
      size_t block_id = result.vec_block_ids[node_id];
      size_t slot = result.vec_slots[node_id];
      size_t byte_off = header_bytes + block_id * block_size + slot * vec_block_bytes;
      gfs.seekg(static_cast<std::streamoff>(byte_off));
      gfs.read(reinterpret_cast<char *>(actual_vec.data()),
               static_cast<std::streamsize>(vec_block_bytes));
      idx.ReadVec(node_id, expected_vec.data());
      if (std::memcmp(actual_vec.data(), expected_vec.data(), vec_block_bytes) != 0) {
        printf("FAIL: node %u raw vector mismatch\n", node_id);
        all_ok = false;
      } else {
        ++checked;
      }
    }
    printf("Vector region: verified %zu/%zu node vectors\n\n", checked, n);
  }
  gfs.close();

  // --- Verify the sidecar file's per-level disk_blocks match the in-memory result ---
  {
    std::ifstream mfs(out_meta, std::ios::binary);
    uint64_t magic = 0, version = 0, curvec_num = 0, num_levels = 0;
    ReadPOD(mfs, magic);
    ReadPOD(mfs, version);
    ReadPOD(mfs, curvec_num);
    ReadPOD(mfs, num_levels);
    if (magic != 0x314D4C4Cull || version != 2 || curvec_num != n ||
        num_levels != result.levels.size()) {
      printf("FAIL: sidecar header mismatch\n");
      all_ok = false;
    } else {
      bool sidecar_ok = true;
      for (size_t li = 0; li < num_levels; ++li) {
        std::vector<int32_t> disk_blocks(n);
        mfs.read(reinterpret_cast<char *>(disk_blocks.data()),
                 static_cast<std::streamsize>(n * sizeof(int32_t)));
        if (disk_blocks != result.levels[li].disk_blocks) {
          printf("FAIL: sidecar level %zu disk_blocks mismatch\n", li);
          sidecar_ok = false;
          all_ok = false;
        }
      }
      uint8_t vec_flag = 0;
      ReadPOD(mfs, vec_flag);
      if (vec_flag) {
        std::vector<size_t> block_ids(n), slots(n);
        mfs.read(reinterpret_cast<char *>(block_ids.data()),
                 static_cast<std::streamsize>(n * sizeof(size_t)));
        mfs.read(reinterpret_cast<char *>(slots.data()),
                 static_cast<std::streamsize>(n * sizeof(size_t)));
        if (block_ids != result.vec_block_ids || slots != result.vec_slots) {
          printf("FAIL: sidecar vec region mapping mismatch\n");
          sidecar_ok = false;
          all_ok = false;
        }
      }
      if (sidecar_ok)
        printf("Sidecar readback: OK (matches in-memory LayoutExportResult)\n\n");
    }
  }

  // Load through the production reader, then check every node/selected layer.
  // Overlap allows correct edges even with the old routing bug, so explicitly
  // check the loaded LUT as well as data bytes and persisted static records.
  idx.LoadMultiLevelLayout(out_graph, cached_meta, 64 * 1024);
  if (idx.StaticResidentBytes() > 16 * 1024 || !idx.static_node_meta_.empty()) {
    printf("FAIL: loaded static cache exceeds total resident budget or retains dense metadata\n");
    all_ok = false;
  }
  if (idx.ml_levels_.size() != ranges.size() || !idx.static_cache_loaded_)
    all_ok = false;
  size_t static_checked = 0;
  for (layer_t layer = 0; layer <= static_cast<layer_t>(idx.cur_max_layer_); ++layer) {
    if (idx.ml_layer_to_level_[layer] != FindLevelForLayer(layer, ranges, stride))
      all_ok = false;
  }
  const layer_t grid_base = static_cast<layer_t>(idx.cur_max_layer_ % stride);
  for (tableint id = 0; id < static_cast<tableint>(n); ++id) {
    for (layer_t layer = grid_base; layer <= static_cast<layer_t>(idx.cur_max_layer_);
         layer += static_cast<layer_t>(stride)) {
      idx.ReadEdges(id, layer, expected_edges.data());
      idx.ReadEdgesMultiLevel(id, layer, actual_edges.data());
      if (actual_edges != expected_edges) {
        printf("FAIL: production loader node %u layer %d linklist mismatch\n", id, layer);
        all_ok = false;
      }
      const auto cached = idx.StaticCacheLookup(id, layer);
      if (cached.found) {
        const size_t off =
            sizeof(tableint) + sizeof(label_t) + sizeof(int) +
            static_cast<size_t>(layer - cached.layer_begin) / stride * (M + 1) * sizeof(tableint);
        if (std::memcmp(cached.data + off, expected_edges.data(), (M + 1) * sizeof(tableint)) != 0)
          all_ok = false;
        ++static_checked;
      }
    }
  }
  if (static_checked == 0)
    all_ok = false;
  printf("Production loader/static-cache roundtrip: checked %zu cached layer records\n",
         static_checked);

  // A cache may not retain the requested record after co-block prefetch. The
  // synchronous reader must work even with no cache or only one record slot.
  const size_t one_slot_bytes =
      *std::max_element(idx.ml_record_size_per_level_.begin(), idx.ml_record_size_per_level_.end());
  for (const size_t cache_bytes : {size_t(0), one_slot_bytes}) {
    idx.LoadMultiLevelLayout(out_graph, cached_meta, cache_bytes);
    for (tableint id = 0; id < static_cast<tableint>(std::min(n, size_t(256))); ++id) {
      for (layer_t layer = grid_base; layer <= static_cast<layer_t>(idx.cur_max_layer_);
           layer += static_cast<layer_t>(stride)) {
        idx.ReadEdges(id, layer, expected_edges.data());
        idx.ReadEdgesMultiLevel(id, layer, actual_edges.data());
        if (actual_edges != expected_edges) {
          printf("FAIL: cache=%zu node=%u layer=%d linklist mismatch\n", cache_bytes, id, layer);
          all_ok = false;
        }
      }
    }
  }
  printf("Production loader zero/one-slot cache roundtrip: completed\n");

  // Build additional caches without changing graph bytes. Check merged records
  // against the original graph, including small budgets that cannot fit metadata.
  std::vector<std::pair<std::string, size_t>> cache_outputs;
  for (size_t budget : {size_t(0), size_t(1), size_t(16384), size_t(131072)}) {
    auto path = out_meta + ".cache_" + std::to_string(budget);
    std::remove(path.c_str());
    cache_outputs.emplace_back(path, budget);
  }
  idx.PrepareStaticCacheSidecars(out_graph, out_meta, cache_outputs);
  bool overwrite_rejected = false;
  try {
    idx.PrepareStaticCacheSidecars(out_graph, out_meta, cache_outputs);
  } catch (const std::invalid_argument &) {
    overwrite_rejected = true;
  }
  if (!overwrite_rejected)
    all_ok = false;
  for (const auto &[path, budget] : cache_outputs) {
    idx.LoadMultiLevelLayout(out_graph, path, 0);
    if (idx.StaticResidentBytes() > budget || !idx.static_node_meta_.empty())
      all_ok = false;
    size_t found = 0;
    for (tableint id = 0; id < n; ++id)
      for (layer_t l = grid_base; l <= idx.cur_max_layer_; l += stride) {
        auto record = idx.StaticCacheLookup(id, l);
        if (!record.found)
          continue;
        ++found;
        idx.ReadEdges(id, l, expected_edges.data());
        size_t off = sizeof(tableint) + sizeof(label_t) + sizeof(int) +
                     (l - record.layer_begin) / stride * (M + 1) * sizeof(tableint);
        if (std::memcmp(record.data + off, expected_edges.data(), (M + 1) * sizeof(tableint)))
          all_ok = false;
      }
    if ((budget >= 16384 && !found) || (budget < 2 && found))
      all_ok = false;
    std::remove(path.c_str());
  }
  printf("Prepared static sidecars: budget/record/overwrite checks completed\n");

  // Independently encode the same small cache as v3, then use the production
  // loader and lookup. Generated small caches must still be flag2 on disk.
  {
    std::ifstream src(cached_meta, std::ios::binary);
    size_t prefix = 4 * sizeof(uint64_t) + result.levels.size() * n * sizeof(int32_t);
    src.seekg(prefix);
    uint8_t vec_flag = 0;
    ReadPOD(src, vec_flag);
    prefix += 1 + (vec_flag ? 2 * n * sizeof(size_t) : 0);
    src.seekg(prefix);
    uint8_t flag = 0;
    uint64_t bytes = 0, nodes = 0;
    ReadPOD(src, flag);
    ReadPOD(src, bytes);
    ReadPOD(src, nodes);
    if (flag != kStaticCacheV2 || nodes != n)
      throw std::runtime_error("small cache format changed");
    std::vector<StaticNodeMeta> old(n);
    src.read(reinterpret_cast<char *>(old.data()), n * sizeof(StaticNodeMeta));
    std::vector<char> payload(bytes), head(prefix);
    src.read(payload.data(), payload.size());
    src.seekg(0);
    src.read(head.data(), head.size());
    std::vector<StaticNodeMeta64> wide(n);
    for (size_t i = 0; i < n; ++i) {
      wide[i].offset = old[i].IsCached() ? uint64_t(old[i].offset) : UINT64_MAX;
      wide[i].layer_begin = old[i].layer_begin;
      wide[i].layer_end = old[i].layer_end;
      wide[i].record_bytes = old[i].record_bytes;
    }
    const auto wide_path = cached_meta + ".v3_fixture";
    std::ofstream out(wide_path, std::ios::binary);
    out.write(head.data(), head.size());
    flag = kStaticCacheV3;
    out.write(reinterpret_cast<char *>(&flag), 1);
    out.write(reinterpret_cast<char *>(&bytes), 8);
    out.write(reinterpret_cast<char *>(&nodes), 8);
    out.write(reinterpret_cast<char *>(wide.data()), n * sizeof(StaticNodeMeta64));
    out.write(payload.data(), payload.size());
    out.close();
    idx.LoadMultiLevelLayout(out_graph, wide_path, 0);
    if (!idx.static_node_lookup_.Wide())
      all_ok = false;
    for (tableint id = 0; id < n; ++id)
      for (layer_t l = grid_base; l <= idx.cur_max_layer_; l += stride) {
        const auto record = idx.StaticCacheLookup(id, l);
        if (record.found != old[id].Covers(l))
          all_ok = false;
        if (!record.found)
          continue;
        idx.ReadEdges(id, l, expected_edges.data());
        const size_t offset = sizeof(tableint) + sizeof(label_t) + sizeof(int) +
                              (l - record.layer_begin) / stride * (M + 1) * sizeof(tableint);
        if (std::memcmp(record.data + offset, expected_edges.data(), (M + 1) * sizeof(tableint)))
          all_ok = false;
      }
    idx.LoadMultiLevelLayout(out_graph, cached_meta, 0);
    if (idx.static_node_lookup_.Wide())
      all_ok = false;
    std::remove(wide_path.c_str());
    printf("Production v2/v3 sidecar decoding and lookup equivalence passed\n");
  }

  // Exercise actual selection overflow with synthetic weights, no giant graph
  // or payload allocation. Small budgets must retain the original v2 metadata.
  {
    const auto saved_n = idx.curvec_num_, saved_m = idx.M_;
    idx.curvec_num_ = 100000;
    idx.M_ = 4096;
    LevelRange range{};
    range.layer_begin = range.owner_begin = 0;
    range.layer_end = range.owner_end = 5;
    const std::vector<LevelRange> levels{range};
    const std::vector<size_t> sizes{16 + 6 * (idx.M_ + 1) * sizeof(tableint)};
    const std::vector<std::vector<size_t>> weights(1, std::vector<size_t>(idx.curvec_num_, 1));
    for (size_t budget : {size_t(64) << 20, size_t(5) << 30, size_t(64) << 20}) {
      idx.static_cache_budget_bytes_ = budget;
      idx.AllocateStaticGroups(levels, sizes, 0, 0, 1, budget, idx.curvec_num_);
      idx.SelectAndMergeStaticNodes(levels, sizes, 1, &weights);
      const bool wide = !idx.static_node_meta64_.empty();
      if (wide != (budget > UINT32_MAX))
        all_ok = false;
      size_t cached = 0;
      for (const auto &m : idx.static_node_meta_)
        cached += m.IsCached();
      for (const auto &m : idx.static_node_meta64_)
        cached += m.IsCached();
      if (StaticNodeLookup::BudgetBytes(idx.curvec_num_, cached, idx.static_cache_bytes_, wide) >
          budget)
        all_ok = false;
      if (wide && idx.static_cache_bytes_ <= UINT32_MAX)
        all_ok = false;
      if (!wide && idx.static_node_meta64_.capacity())
        all_ok = false;
    }
    idx.curvec_num_ = saved_n;
    idx.M_ = saved_m;
    std::vector<StaticNodeMeta>().swap(idx.static_node_meta_);
    std::vector<StaticNodeMeta64>().swap(idx.static_node_meta64_);
    idx.static_cache_bytes_ = 0;
    printf("Adaptive static metadata: small32 / overflow64 / small32 passed\n");
  }

  // Mutate copies only. Version 1 cannot reconstruct the missing bottom
  // replica; a v1 graph or sidecar must fail clearly, never silently route.
  const std::string bad_graph = out_graph + ".bad";
  const std::string bad_meta = out_meta + ".bad";
  auto overwrite = [](const std::string &path, std::streamoff off, const void *value, size_t size) {
    std::fstream fs(path, std::ios::binary | std::ios::in | std::ios::out);
    fs.seekp(off);
    fs.write(static_cast<const char *>(value), size);
    fs.flush();
    if (!fs)
      throw std::runtime_error("test header mutation failed");
  };
  const uint64_t old_version = 1;
  std::filesystem::copy_file(out_graph, bad_graph,
                             std::filesystem::copy_options::overwrite_existing);
  overwrite(bad_graph, sizeof(uint64_t), &old_version, sizeof(old_version));
  bool rejected = false;
  try {
    idx.LoadMultiLevelLayout(bad_graph, out_meta, 64 * 1024);
  } catch (const std::exception &e) {
    rejected = std::string(e.what()).find("re-export") != std::string::npos;
  }
  if (!rejected) {
    printf("FAIL: old graph was not rejected with a re-export diagnostic\n");
    all_ok = false;
  }

  std::filesystem::copy_file(out_meta, bad_meta, std::filesystem::copy_options::overwrite_existing);
  overwrite(bad_meta, sizeof(uint64_t), &old_version, sizeof(old_version));
  rejected = false;
  try {
    idx.LoadMultiLevelLayout(out_graph, bad_meta, 64 * 1024);
  } catch (const std::exception &) {
    rejected = true;
  }
  if (!rejected) {
    printf("FAIL: sidecar version mismatch accepted\n");
    all_ok = false;
  }

  std::filesystem::copy_file(out_graph, bad_graph,
                             std::filesystem::copy_options::overwrite_existing);
  const int32_t invalid_owner = static_cast<int32_t>(wp + 1);
  overwrite(bad_graph, sizeof(TestGraphLayoutHeader) + 2 * sizeof(int32_t), &invalid_owner,
            sizeof(invalid_owner));
  rejected = false;
  try {
    idx.LoadMultiLevelLayout(bad_graph, out_meta, 64 * 1024);
  } catch (const std::exception &) {
    rejected = true;
  }
  if (!rejected) {
    printf("FAIL: malformed ownership accepted\n");
    all_ok = false;
  }
  std::remove(bad_graph.c_str());
  std::remove(bad_meta.c_str());

  if (all_ok) {
    printf("====== ALL TESTS PASSED ======\n");
    return 0;
  } else {
    printf("====== SOME TESTS FAILED ======\n");
    return 1;
  }
}
