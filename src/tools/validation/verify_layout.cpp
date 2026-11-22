#include "index/index_disk.hpp"
#include "io/float_vector_input.hpp"
#include <fstream>
#include <iostream>
#include <unordered_map>

// Full, read-only layout acceptance. No ANN queries or index writes.
int main(int argc, char **argv) {
  try {
    std::unordered_map<std::string, std::string> args;
    for (int i = 1; i < argc; i += 2) {
      if (i + 1 == argc || !args.emplace(argv[i], argv[i + 1]).second)
        throw std::invalid_argument("missing or duplicate argument");
    }
    for (const auto &[key, value] : args)
      if (key != "--graph" && key != "--meta" && key != "--ml-graph" && key != "--layout" &&
          key != "--base" && key != "--format" && key != "--attrs" && key != "--space" &&
          key != "--mode")
        throw std::invalid_argument("unknown argument: " + key);
    const auto require = [&](const char *key) -> const std::string & {
      const auto it = args.find(key);
      if (it == args.end() || it->second.empty())
        throw std::invalid_argument(key);
      return it->second;
    };
    const auto check = [](bool ok, const char *message) {
      if (!ok)
        throw std::runtime_error(message);
    };
    const auto mode = args.count("--mode") ? args.at("--mode") : "layout";
    check(mode == "build" || mode == "layout", "mode must be build or layout");
    const bool verify_layout = mode == "layout";
    alliance::AllianceIndexDisk<int, float> index(require("--graph"), require("--meta"),
                                                  require("--space"));
    check(index.last_completed_layer_ >= index.cur_max_layer_, "build has incomplete layers");
    check(!index.build_config_.left_stream_build_ || index.left_stream_complete_,
          "left-stream build is incomplete");
    if (verify_layout) {
      index.LoadMultiLevelLayout(require("--ml-graph"), require("--layout"), 0);
      check(index.ml_vec_region_loaded_, "layout has no vector region");
    }
    const auto shape = alliance::ReadFloatVectorFileShape(require("--base"), require("--format"));
    check(shape.count == index.curvec_num_ && shape.dim == index.vec_d_,
          "input/index shape mismatch");
    const size_t n = shape.count, d = shape.dim;
    std::ifstream attrs(require("--attrs"), std::ios::binary | std::ios::ate);
    check(bool(attrs) && attrs.tellg() == std::streamoff(n * sizeof(int)),
          "attribute length mismatch");
    attrs.seekg(0);
    std::vector<int> values(n);
    attrs.read(reinterpret_cast<char *>(values.data()), values.size() * sizeof(int));
    std::ifstream base(require("--base"), std::ios::binary), ml;
    if (verify_layout)
      ml.open(require("--ml-graph"), std::ios::binary);
    std::vector<float> expected(d), raw(d), shuffled(d);
    std::vector<alliance::tableint> before(index.M_ + 1), after(index.M_ + 1);
    std::vector<bool> seen(n, false);
    size_t layers_checked = 0, edges_checked = 0;
    for (size_t id = 0; id < n; ++id) {
      const size_t label = index.labels_[id];
      check(label < n && !seen[label], "label is out of bounds or duplicated");
      seen[label] = true;
      check(values[label] == index.atts_[id], "attribute changed");
      if (require("--format") == "bin")
        base.seekg(8 + label * d * 4);
      else {
        base.seekg(label * (d + 1) * 4);
        uint32_t width{};
        base.read(reinterpret_cast<char *>(&width), 4);
        check(width == d, "fvecs row dimension mismatch");
      }
      base.read(reinterpret_cast<char *>(expected.data()), d * 4);
      index.ReadVec(id, raw.data());
      check(bool(base), "truncated vector payload");
      check(!std::memcmp(expected.data(), raw.data(), d * 4), "build vector changed");
      for (float value : expected)
        check(std::isfinite(value), "nonfinite source vector");
      if (verify_layout) {
        ml.seekg(index.MultiLevelVectorOffset(index.ml_vec_block_ids_[id]) +
                 index.ml_vec_slots_[id] * d * 4);
        ml.read(reinterpret_cast<char *>(shuffled.data()), d * 4);
        check(bool(ml) && !std::memcmp(raw.data(), shuffled.data(), d * 4),
              "shuffle vector changed");
      }
      for (size_t layer = 0; layer <= index.cur_max_layer_; ++layer) {
        index.ReadEdges(id, layer, before.data());
        if (verify_layout) {
          index.ReadEdgesMultiLevel(id, layer, after.data());
          check(before == after, "shuffle changed ordered neighbor slots");
        }
        check(before[index.M_] <= index.M_, "neighbor degree exceeds M");
        for (size_t j = 0; j < before[index.M_]; ++j)
          check(before[j] < n, "neighbor ID out of bounds");
        edges_checked += before[index.M_];
        ++layers_checked;
      }
    }
    std::cout << "{\"status\":\"passed\",\"vectors\":" << n << ",\"dimensions\":" << d
              << ",\"node_layers\":" << layers_checked << ",\"ordered_edges\":" << edges_checked
              << ",\"completed_layer\":" << index.last_completed_layer_
              << ",\"max_layer\":" << index.cur_max_layer_
              << ",\"left_stream\":" << (index.build_config_.left_stream_build_ ? "true" : "false")
              << ",\"left_stream_complete\":" << (index.left_stream_complete_ ? "true" : "false")
              << "}\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
