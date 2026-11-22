#include "layout/shuffle/options.hpp"
#include <cstdint>
#include <cmath>
#include <iostream>
#include <set>
#include <stdexcept>

namespace benchmark::shuffle {
namespace {
size_t Integer(const std::string &key, const std::string &value) {
  if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
    throw std::invalid_argument(key + " needs a nonnegative integer");
  size_t used = 0;
  const auto result = std::stoull(value, &used);
  if (used != value.size() || result > SIZE_MAX)
    throw std::invalid_argument(key + " overflow");
  return static_cast<size_t>(result);
}
void Page(size_t bytes, bool optional) {
  if (optional && bytes == 0)
    return;
  if (bytes < 512 || bytes > 1048576 || (bytes & (bytes - 1)))
    throw std::invalid_argument(
        "page sizes must be O_DIRECT-aligned powers of two in 512..1048576");
}
} // namespace
Options Options::Parse(int argc, char **argv) {
  Options o;
  std::set<std::string> seen;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (!seen.insert(key).second)
      throw std::invalid_argument("duplicate argument: " + key);
    if (key == "--help" || key == "-h") {
      o.help = true;
      continue;
    }
    if (key == "--multi_level")
      continue; // Historical spelling; ML is the sole format.
    if (key == "--diagnostics") {
      o.levels.collect_diagnostics = true;
      continue;
    }
    if (++i == argc || std::string(argv[i]).empty() || std::string(argv[i]).rfind("--", 0) == 0)
      throw std::invalid_argument("missing value: " + key);
    const std::string value = argv[i];
    if (key == "--graph_file")
      o.graph = value;
    else if (key == "--meta_file")
      o.meta = value;
    else if (key == "--out_graph_file")
      o.output = value;
    else if (key == "--out_meta_file")
      o.output_meta = value;
    else if (key == "--space")
      o.space = value;
    else if (key == "--method") {
      if (value == "block_bnf")
        o.method = alliance::ShuffleMethod::BlockBNF;
      else if (value == "identity")
        o.method = alliance::ShuffleMethod::Identity;
      else
        throw std::invalid_argument("retired or unknown shuffle method: " + value);
    } else if (key == "--block_size")
      o.ordering.block_size = Integer(key, value);
    else if (key == "--vector_block_size")
      o.vector_block_size = Integer(key, value);
    else if (key == "--max_iterations")
      o.ordering.max_iterations = Integer(key, value);
    else if (key == "--shuffle_threads") {
      const auto n = Integer(key, value);
      if (n < 1 || n > 256)
        throw std::invalid_argument("shuffle_threads must be 1..256");
      o.ordering.threads = static_cast<int>(n);
    } else if (key == "--affinity_batch_size")
      o.ordering.affinity_batch_size = Integer(key, value);
    else if (key == "--shuffle_storage") {
      if (value == "memory")
        o.ordering.storage = alliance::ShuffleStorage::Memory;
      else if (value == "external")
        o.ordering.storage = alliance::ShuffleStorage::External;
      else if (value == "auto")
        o.ordering.storage = alliance::ShuffleStorage::Auto;
      else
        throw std::invalid_argument("shuffle_storage must be memory, external or auto");
    } else if (key == "--shuffle_memory_mib") {
      const size_t mib = Integer(key, value);
      if (!mib || mib > SIZE_MAX / (1 << 20))
        throw std::invalid_argument("shuffle_memory_mib must be positive and fit size_t");
      o.ordering.memory_limit_bytes = mib * (1 << 20);
    } else if (key == "--level_k")
      o.levels.level_k = Integer(key, value);
    else if (key == "--level_m")
      o.levels.level_m = Integer(key, value);
    else if (key == "--shuffle_o")
      o.levels.shuffle_o = Integer(key, value);
    else if (key == "--shuffle_mem_gb") {
      size_t used = 0;
      o.memory_observation_gib = std::stod(value, &used);
      if (used != value.size() || !std::isfinite(o.memory_observation_gib) ||
          o.memory_observation_gib < 0)
        throw std::invalid_argument(
            "shuffle_mem_gb must be finite and nonnegative; observation only");
    } else if (key == "--static_cache_budget" && Integer(key, value) == 0) {
      // Preserve old cache-free workflow commands. Nonzero budgets are retired.
    } else if (key == "--move_margin" || key == "--min_colocatable_for_bnf" ||
               key == "--static_cache_budget" || key == "--static_landing_file") {
      throw std::invalid_argument("retired option: " + key +
                                  "; BlockBNF uses fixed margin=1; prepare caches separately");
    } else
      throw std::invalid_argument("unknown argument: " + key);
  }
  if (o.help) {
    std::cout
        << "alliance_shuffle_disk --graph_file PATH --meta_file PATH --out_graph_file PATH --out_meta_file PATH\n"
           "  [--method block_bnf|identity] [--block_size 4096] [--vector_block_size 0]\n"
           "  [--max_iterations 32] [--level_k 2] [--level_m 2] [--shuffle_o 2]\n"
           "  [--shuffle_threads 1] [--affinity_batch_size 16384] [--diagnostics]\n"
           "  [--shuffle_storage memory|external|auto] [--shuffle_memory_mib N (selection/admission budget)]\n"
           "  [--space l2|ip] [--shuffle_mem_gb 0 (observation only)]\n"
           "Outputs must be absent and share a directory. Completion: GRAPH.completion.json\n";
    return o;
  }
  if (o.graph.empty() || o.meta.empty() || o.output.empty() || o.output_meta.empty())
    throw std::invalid_argument("four input/output paths are required; see --help");
  if (o.space != "l2" && o.space != "ip")
    throw std::invalid_argument("space must be l2 or ip");
  if (o.ordering.storage != alliance::ShuffleStorage::Memory &&
      (!o.ordering.memory_limit_bytes || o.method != alliance::ShuffleMethod::BlockBNF))
    throw std::invalid_argument(
        "auto/external storage requires block_bnf and --shuffle_memory_mib");
  Page(o.ordering.block_size, false);
  Page(o.vector_block_size, true);
  if (!o.ordering.max_iterations || o.ordering.max_iterations > 1048576 ||
      !o.ordering.affinity_batch_size || o.ordering.affinity_batch_size > 1048576)
    throw std::invalid_argument(
        "iterations must be 1..1048576 and affinity_batch_size must be 1..1048576");
  if (!o.levels.level_k || o.levels.level_k > 65536 || o.levels.level_m > 65536 ||
      o.levels.shuffle_o < 2 || (o.levels.shuffle_o & (o.levels.shuffle_o - 1)))
    throw std::invalid_argument(
        "level_k must be 1..65536 and level_m 0..65536; shuffle_o must be a power of two >=2");
  return o;
}
} // namespace benchmark::shuffle
