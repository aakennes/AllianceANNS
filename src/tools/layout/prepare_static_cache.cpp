#include "index/index_disk.hpp"
#include <fstream>
#include <sstream>

int main(int argc, char **argv) {
  try {
    std::string graph, meta, ml_graph, ml_meta, out, space = "l2", budgets, landing;
    for (int i = 1; i < argc; ++i) {
      const std::string key = argv[i];
      if (i + 1 >= argc)
        throw std::invalid_argument("missing value: " + key);
      const std::string value = argv[++i];
      if (key == "--graph_file")
        graph = value;
      else if (key == "--meta_file")
        meta = value;
      else if (key == "--ml_graph_file")
        ml_graph = value;
      else if (key == "--ml_meta_file")
        ml_meta = value;
      else if (key == "--out_prefix")
        out = value;
      else if (key == "--space")
        space = value;
      else if (key == "--budgets")
        budgets = value;
      else if (key == "--static_landing_file")
        landing = value;
      else
        throw std::invalid_argument("unknown argument: " + key);
    }
    if (graph.empty() || meta.empty() || ml_graph.empty() || ml_meta.empty() || out.empty() ||
        budgets.empty())
      throw std::invalid_argument(
          "Usage: alliance_prepare_static_cache --graph_file PATH --meta_file PATH "
          "--ml_graph_file PATH --ml_meta_file PATH --out_prefix PATH --budgets BYTES[,BYTES...] "
          "[--space l2|ip] [--static_landing_file PATH]");
    std::vector<std::pair<std::string, size_t>> outputs;
    std::istringstream parts(budgets);
    std::string part;
    while (std::getline(parts, part, ',')) {
      if (part.empty() || part.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("budgets must be nonnegative integer bytes");
      size_t budget = std::stoull(part);
      outputs.emplace_back(out + "_" + std::to_string(budget) + ".layout", budget);
    }
    alliance::AllianceIndexDisk<int, float> index(graph, meta, space);
    // Cache preparation only needs graph geometry, IDs and layer records.
    // Release search codes before scoring/filling potentially large caches.
    decltype(index.quant_vecs_)().swap(index.quant_vecs_);
    if (!landing.empty()) {
      std::ifstream in(landing, std::ios::binary);
      uint64_t n = 0;
      in.read(reinterpret_cast<char *>(&n), sizeof(n));
      if (!in || n != index.cur_max_layer_ + 1)
        throw std::runtime_error("landing histogram shape mismatch");
      index.landing_layer_cnt_.resize(n);
      in.read(reinterpret_cast<char *>(index.landing_layer_cnt_.data()), n * sizeof(uint64_t));
      if (!in)
        throw std::runtime_error("truncated landing histogram");
    }
    index.PrepareStaticCacheSidecars(ml_graph, ml_meta, outputs);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
