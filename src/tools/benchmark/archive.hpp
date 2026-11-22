#pragma once
#include "index/index_disk.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace benchmark::formal {
struct Observation {
  std::vector<std::pair<float, alliance::label_t>> result;
  double latency = 0;
  uint64_t requests = 0, completions = 0, bytes = 0, graph = 0, rerank = 0, hops = 0, matches = 0;
};
template <class T> void Binary(std::ofstream &stream, const T &value) {
  stream.write(reinterpret_cast<const char *>(&value), sizeof(value));
}
inline void WriteResults(const std::filesystem::path &directory,
                         const std::vector<Observation> &observations, uint64_t k, uint64_t ef) {
  std::ofstream raw(directory / "results.bin", std::ios::binary), per(directory / "queries.csv");
  const uint64_t magic = 0x414c4c49414e4331ULL, nq = observations.size();
  Binary(raw, magic);
  Binary(raw, nq);
  Binary(raw, k);
  Binary(raw, ef);
  per << std::setprecision(17)
      << "query_id,latency_seconds,requests,completions,bytes,graph_io,rerank_io,hops,matches\n";
  for (size_t i = 0; i < observations.size(); ++i) {
    const auto &z = observations[i];
    if (z.result.size() != k)
      throw std::runtime_error("raw result count");
    for (const auto &[distance, id] : z.result) {
      uint64_t label = id;
      Binary(raw, label);
      Binary(raw, distance);
    }
    per << i << ',' << z.latency << ',' << z.requests << ',' << z.completions << ',' << z.bytes
        << ',' << z.graph << ',' << z.rerank << ',' << z.hops << ',' << z.matches << '\n';
  }
  raw.close();
  per.close();
  if (!raw || !per)
    throw std::runtime_error("raw result archive write");
}
} // namespace benchmark::formal
