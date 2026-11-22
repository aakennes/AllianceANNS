#include "index/index_disk.hpp"
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// Structural routing only: no query vectors, GT, quantization, traversal or rerank.
int main(int argc, char **argv) {
  try {
    std::string graph, meta, ranges, output, report, space = "l2", mode = "auto";
    size_t threshold = 8192, limit = 0;
    bool can_prefilter = true;
    auto number = [](const std::string &s) {
      if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("expected nonnegative integer: " + s);
      return static_cast<size_t>(std::stoull(s));
    };
    for (int i = 1; i < argc; ++i) {
      const std::string key = argv[i];
      if (key == "--help") {
        std::cout
            << "alliance_collect_landing --graph_file FILE --meta_file FILE --query_rng FILE "
               "--out_file FILE --report_file FILE [--space ip|l2] [--filter_mode auto|graph|prefilter|legacy] "
               "[--prefilter_threshold 8192] [--query_limit 0] [--can_prefilter true|false]\n";
        return 0;
      }
      if (i + 1 >= argc)
        throw std::invalid_argument("missing value: " + key);
      const std::string value = argv[++i];
      if (key == "--graph_file")
        graph = value;
      else if (key == "--meta_file")
        meta = value;
      else if (key == "--query_rng")
        ranges = value;
      else if (key == "--out_file")
        output = value;
      else if (key == "--report_file")
        report = value;
      else if (key == "--space")
        space = value;
      else if (key == "--filter_mode")
        mode = value;
      else if (key == "--prefilter_threshold")
        threshold = number(value);
      else if (key == "--query_limit")
        limit = number(value);
      else if (key == "--can_prefilter") {
        if (value != "true" && value != "false")
          throw std::invalid_argument("can_prefilter must be true or false");
        can_prefilter = value == "true";
      } else
        throw std::invalid_argument("unknown option: " + key);
    }
    if (graph.empty() || meta.empty() || ranges.empty() || output.empty() || report.empty())
      throw std::invalid_argument(
          "graph_file, meta_file, query_rng, out_file and report_file are required");
    if (mode != "auto" && mode != "graph" && mode != "prefilter" && mode != "legacy")
      throw std::invalid_argument("invalid filter_mode");
    if (space != "ip" && space != "l2")
      throw std::invalid_argument("space must be ip or l2");
    if (mode == "auto" && !threshold)
      throw std::invalid_argument("auto threshold must be positive");
    if (mode == "prefilter" && !can_prefilter)
      throw std::invalid_argument("prefilter requires ML vectors");
    if (std::filesystem::exists(output) || std::filesystem::exists(report))
      throw std::invalid_argument("outputs must be new files");
    if (std::filesystem::absolute(output).lexically_normal() ==
        std::filesystem::absolute(report).lexically_normal())
      throw std::invalid_argument("output paths must differ");
    const auto bytes = std::filesystem::file_size(ranges);
    if (!bytes || bytes % (2 * sizeof(int32_t)))
      throw std::invalid_argument("query_rng must be nonempty raw closed int32 [L,U] pairs");
    alliance::AllianceIndexDisk<int, float> index(graph, meta, space);
    decltype(index.quant_vecs_)().swap(index.quant_vecs_);
    std::vector<uint64_t> histogram(static_cast<size_t>(index.cur_max_layer_) + 1, 0);
    const size_t available = bytes / (2 * sizeof(int32_t)),
                 nq = limit ? std::min(limit, available) : available;
    uint64_t empty = 0, prefilter = 0, legacy = 0, graph_count = 0;
    std::ifstream input(ranges, std::ios::binary);
    for (size_t i = 0; i < nq; ++i) {
      int32_t bounds[2];
      input.read(reinterpret_cast<char *>(bounds), sizeof(bounds));
      if (!input)
        throw std::runtime_error("truncated query ranges");
      if (bounds[0] > bounds[1]) {
        ++empty;
        continue;
      }
      const auto ranks = index.order_table_->GetRankRange(
          {bounds[0], 0}, {bounds[1], std::numeric_limits<alliance::label_t>::max()});
      const size_t card = ranks.second - ranks.first;
      if (!card) {
        ++empty;
        continue;
      }
      if (mode == "prefilter" || (mode == "auto" && can_prefilter && card <= threshold)) {
        ++prefilter;
        continue;
      }
      if (mode == "legacy" && index.vec_file_available_ && card < index.window_size_[0]) {
        ++legacy;
        continue;
      }
      std::vector<alliance::tableint> eps;
      auto layers = index.DecideLayerRange({bounds[0], bounds[1]}, eps);
      if (layers.u_ < 0 || static_cast<size_t>(layers.u_) >= histogram.size())
        throw std::runtime_error("invalid landing layer");
      ++histogram[layers.u_];
      ++graph_count;
    }
    std::ofstream out(output, std::ios::binary);
    const uint64_t layers = histogram.size();
    out.write(reinterpret_cast<const char *>(&layers), sizeof(layers));
    out.write(reinterpret_cast<const char *>(histogram.data()),
              histogram.size() * sizeof(uint64_t));
    out.close();
    if (!out)
      throw std::runtime_error("cannot write landing histogram");
    std::ofstream json(report);
    json << "{\"schema_version\":1,\"query_count\":" << nq << ",\"available_queries\":" << available
         << ",\"graph_queries\":" << graph_count << ",\"prefilter_queries\":" << prefilter
         << ",\"empty_queries\":" << empty << ",\"legacy_bruteforce_queries\":" << legacy
         << ",\"nodes\":" << index.curvec_num_ << ",\"histogram\":[";
    for (size_t i = 0; i < histogram.size(); ++i) {
      if (i)
        json << ',';
      json << histogram[i];
    }
    json << "]}\n";
    json.close();
    if (!json)
      throw std::runtime_error("cannot write landing report");
    std::cout << "# landing_collected: queries=" << nq << " graph=" << graph_count
              << " prefilter=" << prefilter << " empty=" << empty << " legacy=" << legacy << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "ERROR: " << e.what() << '\n';
    return 1;
  }
}
