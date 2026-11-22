#pragma once
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace benchmark::formal {
inline std::string Text(const char *key) {
  const char *value = std::getenv(key);
  if (!value || !*value)
    throw std::invalid_argument(std::string("missing ") + key);
  return value;
}
inline uint64_t Number(const char *key, bool allow_zero = false) {
  const auto text = Text(key);
  if (text.find_first_not_of("0123456789") != std::string::npos)
    throw std::invalid_argument(std::string("invalid unsigned integer: ") + key);
  const auto value = std::stoull(text);
  if (!value && !allow_zero)
    throw std::invalid_argument(std::string("zero ") + key);
  return value;
}
struct Protocol {
  std::string output = Text("BENCH_OUTPUT");
  std::string mode = std::getenv("BENCH_MODE") ? Text("BENCH_MODE") : "measure";
  std::string quantization = Text("BENCH_QUANTIZATION_SOURCE");
  std::string memory_mode = Text("BENCH_MEMORY_MODE");
  std::string scope_prefix = Text("BENCH_SCOPE_PREFIX");
  std::string order = Text("BENCH_ORDER");
  uint64_t n = Number("BENCH_EXPECTED_N"), dim = Number("BENCH_EXPECTED_D");
  uint64_t wp = Number("BENCH_EXPECTED_WP"), degree = Number("BENCH_EXPECTED_M");
  uint64_t max_layer = Number("BENCH_EXPECTED_MAX_LAYER", true);
  uint64_t centroids = Number("BENCH_EXPECTED_CENTROIDS");
  uint64_t queries = Number("BENCH_EXPECTED_QUERIES"), bits = Number("BENCH_EXPECTED_BITS");
  uint64_t fingerprint = mode == "fingerprint" ? 0 : Number("BENCH_EXPECTED_FP", true);
  uint64_t loading_limit = Number("BENCH_LOADING_RSS_BYTES");
  uint64_t search_limit = Number("BENCH_SEARCH_RSS_BYTES");
  uint64_t static_budget = Number("BENCH_STATIC_BUDGET", true);
  uint64_t dynamic_budget = Number("BENCH_DYNAMIC_BUDGET", true);
  uint64_t repeats = Number("BENCH_REPEATS"), warmups = Number("BENCH_WARMUPS", true);
  Protocol() {
    if ((mode != "measure" && mode != "fingerprint") ||
        (quantization != "meta" && quantization != "checkpoint") ||
        (memory_mode != "phase" && memory_mode != "whole_process") ||
        search_limit > loading_limit || repeats > 100 || warmups > 10 ||
        (order != "repeat_then_ef" && order != "ef_then_repeat") ||
        scope_prefix.rfind("alliance-bench-", 0) != 0 ||
        scope_prefix.find_first_not_of(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.") !=
            std::string::npos)
      throw std::invalid_argument("invalid formal benchmark protocol");
  }
};
struct Batch {
  size_t ef, repeat, warmup_ordinal;
  bool warmup;
};
inline std::vector<Batch> Schedule(const Protocol &p, const std::vector<size_t> &efs) {
  std::vector<Batch> result;
  if (p.order == "ef_then_repeat") {
    for (auto ef : efs) {
      for (size_t warm = 0; warm < p.warmups; ++warm)
        result.push_back({ef, 1, warm, true});
      for (size_t repeat = 1; repeat <= p.repeats; ++repeat)
        result.push_back({ef, repeat, 0, false});
    }
  } else {
    for (size_t repeat = 1; repeat <= p.repeats; ++repeat)
      for (auto ef : efs) {
        for (size_t warm = 0; warm < p.warmups; ++warm)
          result.push_back({ef, repeat, warm, true});
        result.push_back({ef, repeat, 0, false});
      }
  }
  return result;
}
} // namespace benchmark::formal
