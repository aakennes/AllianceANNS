#include "index/index_disk.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
using namespace alliance;
static void Check(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
int main() {
  const auto dir =
      std::filesystem::temp_directory_path() / ("alliance_id_only_" + std::to_string(getpid()));
  std::filesystem::create_directories(dir);
  for (const std::string metric : {"l2", "ip"}) {
    const size_t n = 12, d = 8, m = 4;
    std::vector<float> data(n * d);
    std::vector<int> attr(n);
    for (size_t i = 0; i < n; ++i) {
      attr[i] = n - i;
      for (size_t j = 0; j < d; ++j)
        data[i * d + j] = float(i) * .5f + float(j) * .125f;
    }
    const auto graph = (dir / (metric + ".graph")).string(),
               meta = (dir / (metric + ".meta")).string();
    {
      AllianceIndexDisk<int, float> index(n, d, m, 16, metric, graph, 2, 2, false, 4);
      // Explicit legacy fixture: stored lower distances and no LeftStream tag.
      index.build_config_.left_stream_build_ = false;
      index.PopulateElements(data.data(), attr.data(), n);
      index.id_only_start_layer_ = 1;
      for (tableint id = 0; id < n; ++id)
        for (layer_t l = 0; l < 3; ++l) {
          tableint ids[2] = {tableint((id + 1) % n), tableint((id + 2) % n)};
          dist_t values[2];
          for (size_t k = 0; k < 2; ++k)
            values[k] = index.fstdistfunc_(data.data() + id * d, data.data() + ids[k] * d,
                                           index.dist_func_param_);
          index.WriteEdgesWithDist(id, l, ids, values, 2);
        }
      index.graph_file_->Sync();
      index.Save(meta);
    }
    AllianceIndexDisk<int, float> index(graph, meta, metric);
    Check(index.id_only_start_layer_ == 1, "metadata did not preserve ID-only policy");
    size_t edges = 0;
    for (tableint id = 0; id < n; ++id)
      for (layer_t l = 0; l < 3; ++l) {
        tableint ids[m + 1];
        dist_t distances[m], stored[m];
        index.ReadEdgesWithDist(id, l, ids, distances);
        index.graph_file_->ReadNodeSlice(index.DiskSlot(id),
                                         index.EdgeOffsetInRecord(l) + index.EdgeIdBytes(),
                                         index.EdgeDistBytes(), stored);
        Check(ids[m] == 2, "degree mismatch");
        for (size_t k = 0; k < 2; ++k) {
          const auto exact = index.fstdistfunc_(data.data() + id * d, data.data() + ids[k] * d,
                                                index.dist_func_param_);
          Check(std::abs(distances[k] - exact) < 1e-5, "fallback distance is incorrect");
          Check(l == 0 ? std::abs(stored[k] - exact) < 1e-5 : stored[k] == 0,
                "unexpected SSD distance data");
          ++edges;
        }
      }
    Check(edges == n * 3 * 2, "edge coverage");
    // Old metadata without the optional suffix still has the old semantics.
    auto old = dir / (metric + ".legacy.meta");
    {
      std::ifstream input(meta, std::ios::binary);
      std::string bytes((std::istreambuf_iterator<char>(input)), {});
      bytes.resize(bytes.size() - 12);
      std::ofstream output(old, std::ios::binary);
      output.write(bytes.data(), bytes.size());
    }
    AllianceIndexDisk<int, float> legacy(graph, old.string(), metric);
    Check(legacy.HasStoredEdgeDistances(2), "old metadata compatibility");
  }
  std::filesystem::remove_all(dir);
  std::cout
      << "ID_ONLY_EDGES_PASS: L2/IP, stored lower distances, zero upper SSD distances, exact fallback and old metadata\n";
}
