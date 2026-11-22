#include "benchmark/session.hpp"
#include "benchmark/checks.hpp"
#include "benchmark/strict_input.hpp"
#include "search/search_disk_checkpoint.hpp"
#include "search/search_preparation.hpp"
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#include <sys/resource.h>

namespace benchmark::formal {
void CheckLoadingMemory(size_t cap) {
  rusage r{};
  getrusage(RUSAGE_SELF, &r);
  if (cap)
    Require(uint64_t(r.ru_maxrss) * 1024 <= cap, "RSS cap");
}
namespace {
void write_exclusive(const std::string &path, const std::string &text) {
  const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
  Require(fd >= 0, "output already exists or cannot be created");
  size_t written = 0;
  while (written < text.size()) {
    const auto n = ::write(fd, text.data() + written, text.size() - written);
    if (n < 0 && errno == EINTR)
      continue;
    if (n <= 0) {
      ::close(fd);
      throw std::runtime_error("exclusive output write failed");
    }
    written += static_cast<size_t>(n);
  }
  Require(::close(fd) == 0, "exclusive output close failed");
}
} // namespace
void WriteFingerprint(const std::string &path, uint64_t fingerprint) {
  write_exclusive(path, "{\"fingerprint\":\"" + std::to_string(fingerprint) +
                            "\",\"ANN_queries_executed\":0}\n");
}
void ValidateFormalOptions(const DiskSearchConfig &o, const Protocol &protocol) {
  Require(o.ml_pipeline_mode && o.search_threads == 1 && o.packed_rabitq,
          "single-thread packed ML protocol");
  Require(o.rabitq_bits == protocol.bits && (o.space == "ip" || o.space == "l2"),
          "metric/bit width protocol");
  Require(o.ml_cache_bytes == protocol.dynamic_budget && o.ml_cache_policy == "fifo" &&
              o.process_memory_budget_bytes == 0,
          "explicit dynamic cache and RSS protocol");
  Require(!o.recall_stop, "formal driver requires noninterleaved rerank and a complete EF plan");
  Require(o.filter_mode == "graph" || o.filter_mode == "auto" || o.filter_mode == "prefilter",
          "explicit routing required");
}
uint64_t LoadFormalQuantization(Index &index, const DiskSearchConfig &o, const Protocol &protocol) {
  Require(index.curvec_num_ == protocol.n && index.vec_d_ == protocol.dim &&
              index.wp_ == protocol.wp && index.M_ == protocol.degree &&
              index.cur_max_layer_ == protocol.max_layer,
          "index shape differs from protocol");
  uint64_t fp = 0;
  if (protocol.quantization == "checkpoint") {
    for (size_t i = 0; i < index.curvec_num_; ++i)
      Require(index.labels_[i] == i, "checkpoint ID alignment");
    Require(!o.quant_checkpoint.empty() && !o.rabitq_centroids.empty() && !o.rabitq_cids.empty(),
            "checkpoint source requires checkpoint/centroids/CIDs");
    fp = benchmark::LoadSearchCheckpoint(index, o);
  } else {
    Require(o.quant_checkpoint.empty() && o.rabitq_centroids.empty() && o.rabitq_cids.empty(),
            "meta source cannot silently override quantization");
    Require(index.rabitq_initialized_ && index.rabitq_bits_ == protocol.bits && index.centroids_ &&
                index.quant_vecs_.size() == index.curvec_num_ * index.quant_vec_bytes_ &&
                index.factors_.size() == index.curvec_num_ * 4,
            "meta quantization shape");
    fp = benchmark::QuantizationFingerprint(index);
  }
  Require(index.maxC_ == protocol.centroids, "centroid count differs from protocol");
  return fp;
}
Geometry LoadFormalLayout(Index &index, const DiskSearchConfig &o, const Protocol &protocol) {
  benchmark::LoadSearchLayout(index, o);
  const auto geometry = benchmark::formal::ReadGeometry(o.ml_graph_file, index);
  const auto static_resident = index.StaticResidentBytes();
  Require(protocol.static_budget ? index.static_cache_loaded_ && static_resident > 0 &&
                                       static_resident <= protocol.static_budget
                                 : static_resident == 0,
          "verified static-cache budget");
  benchmark::ConfigureSearchCache(index, o, benchmark::CachePreparation::Formal);
  return geometry;
}
FormalInputs LoadFormalInputs(Index &index, const DiskSearchConfig &o, const Protocol &protocol) {
  const size_t d = index.vec_d_, nq = protocol.queries;
  auto q = benchmark::formal::ReadQueries(o.query_vec_path, d, nq);
  auto ranges = benchmark::formal::ReadRanges(o.query_rng_path, nq);
  auto gt = benchmark::formal::ReadGroundTruth(o.gt_file, nq, o.k);
  Require(nq == protocol.queries && (!o.query_limit || o.query_limit == nq),
          "query count differs from protocol");
  for (size_t i = 0; i < nq * d; ++i)
    Require(std::isfinite(q[i]), "nonfinite query");
  auto attrs = benchmark::DenseCheckpointAttributes(index);
  for (size_t i = 0; i < nq; ++i) {
    Require(gt[i].size() == o.k, "Top-k GT row");
    std::unordered_set<size_t> labels;
    for (auto id : gt[i])
      Require(id < attrs.size() && labels.insert(id).second && ranges[i].Test(attrs[id]),
              "invalid, duplicate or out-of-range GT label");
  }
  return {d, nq, std::move(q), std::move(ranges), std::move(gt), std::move(attrs)};
}
std::vector<size_t> FormalEFPlan(const DiskSearchConfig &o) {
  std::vector<size_t> efs = o.efs_list;
  if (o.single_efs)
    efs = {o.single_efs};
  Require(!efs.empty() && efs.size() <= 20 && efs.front() > 0 && efs.back() <= 1024,
          "planned EF bounds");
  return efs;
}
} // namespace benchmark::formal
