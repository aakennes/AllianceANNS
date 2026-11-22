// Cross-path contracts: streaming/RAM population, persisted codes and independent
// graph-source requantization. Queries use the saved codes without re-encoding.
#include "index/index_disk.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <omp.h>
#include <stdexcept>
#include <unistd.h>

using Index = alliance::AllianceIndexDisk<int, float>;
namespace fs = std::filesystem;
static void Require(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
struct Scratch {
  fs::path path;
  Scratch() {
    const char *base = std::getenv("TEST_TMPDIR");
    std::string name =
        ((base ? fs::path(base) : fs::current_path()) / "build-quant-XXXXXX").string();
    std::vector<char> buf(name.begin(), name.end());
    buf.push_back(0);
    Require(mkdtemp(buf.data()) != nullptr, "mkdtemp failed");
    path = buf.data();
  }
  ~Scratch() {
    std::error_code ec;
    fs::remove_all(path, ec);
  }
};
using OwnedIndex = Index;
template <class T>
static void WriteRows(const fs::path &path, const std::vector<T> &values, int dim) {
  std::ofstream file(path, std::ios::binary);
  for (size_t i = 0; i < values.size(); i += dim) {
    file.write(reinterpret_cast<const char *>(&dim), sizeof(dim));
    file.write(reinterpret_cast<const char *>(values.data() + i), dim * sizeof(T));
  }
  Require(bool(file), "write fixture failed");
}
static void SameQuantization(const Index &expected, const Index &actual) {
  Require(expected.rabitq_bits_ == actual.rabitq_bits_ &&
              expected.padded_dim_ == actual.padded_dim_ &&
              expected.quant_vec_bytes_ == actual.quant_vec_bytes_,
          "quantization geometry mismatch");
  Require(expected.quant_vecs_ == actual.quant_vecs_, "packed codes differ");
  Require(expected.factors_.size() == actual.factors_.size() &&
              std::memcmp(expected.factors_.data(), actual.factors_.data(),
                          expected.factors_.size() * sizeof(float)) == 0,
          "quantization factors differ");
  Require(expected.labels_ == actual.labels_ && expected.cids_ == actual.cids_,
          "label/CID mapping differs");
  Require(expected.sorted_ids_ == actual.sorted_ids_ &&
              expected.position_by_id_ == actual.position_by_id_,
          "rank mapping differs");
}
static void CheckDecodedCodes(const Index &index, const std::vector<float> &base) {
  std::vector<float> rotated(index.padded_dim_);
  std::vector<uint8_t> raw(index.padded_dim_);
  const auto metric =
      index.rabitq_space_name_ == "ip" ? rabitqlib::METRIC_IP : rabitqlib::METRIC_L2;
  for (size_t id = 0; id < index.curvec_num_; ++id) {
    std::fill(raw.begin(), raw.end(), uint8_t{0});
    index.rotator_->rotate(base.data() + id * index.vec_d_, rotated.data());
    float add, scale, error;
    rabitqlib::quant::quantize_full_single(
        rotated.data(), index.centroids_ + index.cids_[id] * index.padded_dim_, index.padded_dim_,
        index.rabitq_bits_, raw.data(), add, scale, error, metric, index.rabitq_config_);
    const auto *packed = index.quant_vecs_.data() + id * index.quant_vec_bytes_;
    unsigned sum = 0;
    for (size_t d = 0; d < raw.size(); ++d) {
      unsigned decoded = 0;
      for (size_t bit = 0; bit < index.rabitq_bits_; ++bit)
        decoded |= ((packed[bit * raw.size() / 8 + d / 8] >> (d % 8)) & 1u) << bit;
      Require(decoded == raw[d], "persisted bit planes do not decode to the quantizer's raw codes");
      sum += decoded;
    }
    const float *f = index.factors_.data() + id * 4;
    Require(std::isfinite(f[0]) && std::isfinite(f[1]) && std::isfinite(f[2]) &&
                std::isfinite(f[3]),
            "non-finite quantization factor");
    Require(f[0] == add && f[1] == scale && f[2] == error && f[3] == float(sum),
            "decoded factors mismatch");
  }
}
static void CheckScalarEstimator(Index &actual, const Index &canonical,
                                 const std::vector<float> &base) {
  // Decode the original canonical planes independently of the native packed
  // layout and float-query kernels, both before and after search packing.
  std::vector<float> query(canonical.vec_d_), rotated(canonical.padded_dim_);
  for (size_t qi : {2ul, 23ul, 57ul}) {
    for (size_t d = 0; d < query.size(); ++d)
      query[d] = base[qi * query.size() + d] + .013f * std::sin(float(d + 1));
    canonical.rotator_->rotate(query.data(), rotated.data());
    Index::RaBitQEstimator estimator(query.data(), &actual);
    for (alliance::tableint id = 0; id < canonical.curvec_num_; ++id) {
      const auto cid = canonical.SearchCID(id);
      const auto *code = canonical.quant_vecs_.data() + id * canonical.quant_vec_bytes_;
      const auto *factors = canonical.factors_.data() + id * 4;
      double centered_dot = 0, centroid_distance = canonical.rabitq_space_name_ == "ip" ? 1 : 0;
      for (size_t d = 0; d < rotated.size(); ++d) {
        unsigned decoded = 0;
        for (size_t bit = 0; bit < canonical.rabitq_bits_; ++bit)
          decoded |= ((code[bit * rotated.size() / 8 + d / 8] >> (d % 8)) & 1u) << bit;
        const double q = rotated[d], center = canonical.centroids_[cid * rotated.size() + d];
        centered_dot += (double(decoded) - .5 * double((1u << canonical.rabitq_bits_) - 1)) * q;
        if (canonical.rabitq_space_name_ == "ip")
          centroid_distance -= q * center;
        else {
          const double delta = q - center;
          centroid_distance += delta * delta;
        }
      }
      const float expected = float(factors[0] + centroid_distance + factors[1] * centered_dot);
      const float score = estimator.dist_estimate(cid, id);
      const auto [bound_score, error] = estimator.dist_estimate_with_errorbound(cid, id);
      Require(std::abs(score - expected) <= 1e-3f,
              "native estimate differs from scalar code decode");
      Require(std::memcmp(&score, &bound_score, sizeof(float)) == 0 &&
                  std::memcmp(&error, factors + 2, sizeof(float)) == 0,
              "estimator APIs disagree");
      if (!std::isfinite(factors[2])) {
        const auto bounds = estimator.distance_bounds(cid, id, score);
        Require(std::isinf(bounds.first) && bounds.first < 0 && std::isinf(bounds.second) &&
                    bounds.second > 0,
                "nonfinite factor must force exact IO");
      }
    }
  }
}
static void CheckNonfiniteErrorFallback(Index &index, const std::vector<float> &base) {
  // Guarantee this fallback is exercised, even when synthetic quantization
  // happens to produce finite model errors for every vector.
  const float saved = index.factors_[2];
  Index::RaBitQEstimator estimator(base.data(), &index);
  const auto cid = index.SearchCID(0);
  for (float error :
       {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
    index.factors_[2] = error;
    const auto bounds = estimator.distance_bounds(cid, 0, estimator.dist_estimate(cid, 0));
    Require(std::isinf(bounds.first) && bounds.first < 0 && std::isinf(bounds.second) &&
                bounds.second > 0,
            "nonfinite error did not force exact IO");
  }
  index.factors_[2] = saved;
}
static void DirectQueries(Index &loaded, Index &requantized, const std::vector<float> &base,
                          const std::vector<int> &attrs) {
  Require(!loaded.vec_file_available_,
          "test must exercise graph search, not the small-range brute force path");
  for (size_t qi : {2ul, 23ul, 57ul}) {
    std::vector<float> query(base.begin() + qi * loaded.vec_d_,
                             base.begin() + (qi + 1) * loaded.vec_d_);
    for (auto range : {alliance::attr_range<int>{-1, 100}, alliance::attr_range<int>{5, 15}}) {
      std::vector<std::pair<alliance::dist_t, alliance::label_t>> exact;
      for (size_t id = 0; id < attrs.size(); ++id)
        if (attrs[id] >= range.l_ && attrs[id] <= range.u_)
          exact.emplace_back(loaded.fstdistfunc_(query.data(), base.data() + id * loaded.vec_d_,
                                                 loaded.dist_func_param_),
                             id);
      std::sort(exact.begin(), exact.end());
      exact.resize(5);
      auto direct = loaded.searchKNN(query.data(), 128, 5, range);
      auto reference = requantized.searchKNN(query.data(), 128, 5, range);
      Require(direct == reference, "direct persisted-code search differs from requantized search");
      Require(direct == exact, "complete tiny graph did not return exact top-5");
    }
  }
  Require(loaded.metric_dist_comps_ > 0, "quantized graph traversal was not exercised");
}
static void PackedRoundTrip(Index &loaded, const Index &canonical, const fs::path &dir,
                            const std::vector<float> &base, const std::vector<int> &attrs) {
  CheckScalarEstimator(loaded, canonical, base);
  CheckNonfiniteErrorFallback(loaded, base);
  loaded.PrepareSearchCIDs();
  loaded.PackRaBitQForSearch();
  Require(loaded.quant_vecs_.empty() && loaded.factors_.empty(),
          "packed search retained split arrays");
  CheckScalarEstimator(loaded, canonical, base);
  loaded.Save((dir / "meta.compact").string());
  std::ifstream original(dir / "meta", std::ios::binary),
      compact(dir / "meta.compact", std::ios::binary);
  const std::string original_bytes((std::istreambuf_iterator<char>(original)), {});
  const std::string compact_bytes((std::istreambuf_iterator<char>(compact)), {});
  Require(original_bytes == compact_bytes, "packed Save changed canonical metadata bytes");
  auto roundtrip =
      std::make_unique<OwnedIndex>((dir / "stream.graph").string(), (dir / "meta.compact").string(),
                                   canonical.rabitq_space_name_);
  SameQuantization(canonical, *roundtrip);
  CheckScalarEstimator(*roundtrip, canonical, base);
  DirectQueries(loaded, *roundtrip, base, attrs);
}
static void RunCase(const fs::path &dir, size_t dim, size_t bits, const std::string &space,
                    bool preset_count, bool short_ids = false) {
  // Read only a prefix, use duplicate/non-monotone attributes and varying CIDs.
  constexpr size_t n = 67, input_n = 79, nc = 3;
  // A complete tiny graph needs every other rank inside the L0 half-window.
  constexpr size_t complete_window = 2 * (n - 1);
  std::vector<float> base(input_n * dim), centroids(nc * dim);
  std::vector<int> attrs(n), cids(input_n);
  for (size_t i = 0; i < input_n; ++i) {
    cids[i] = (i * 7 + 1) % nc;
    for (size_t d = 0; d < dim; ++d)
      base[i * dim + d] = std::sin(float(i * 17 + d * 13) * .021f) * .4f +
                          std::cos(float(i * 7 + d * 11) * .019f) * .2f;
    if (i < n)
      attrs[i] = int((i * 37) % n) / 3;
  }
  for (size_t c = 0; c < nc; ++c)
    for (size_t d = 0; d < dim; ++d)
      centroids[c * dim + d] = std::sin(float(c * 11 + d * 3) * .023f) * .15f;
  WriteRows(dir / "base.fvecs", base, int(dim));
  WriteRows(dir / "centroids.fvecs", centroids, int(dim));
  if (short_ids)
    cids.resize(n - 1);
  WriteRows(dir / "cids.ivecs", cids, 1);
  // Each index contains a large mutex array; multiple indices must not share
  // the test thread's limited stack.
  auto ram_owner = std::make_unique<OwnedIndex>(
      n, dim, n - 1, 128, space, (dir / "ram.graph").string(), 2, 0, false, complete_window);
  auto stream_owner = std::make_unique<OwnedIndex>(
      n, dim, n - 1, 128, space, (dir / "stream.graph").string(), 2, 0, false, complete_window);
  auto &ram = *ram_owner;
  auto &stream = *stream_owner;
  if (preset_count)
    ram.curvec_num_ = stream.curvec_num_ = n;
  for (auto *index : {&ram, &stream})
    index->InitRaBitQ((dir / "centroids.fvecs").string(), (dir / "cids.ivecs").string(), space,
                      bits);
  // faster_config samples a random calibration factor for multi-bit codes.
  // Compare input paths under the same factor, without changing production RNG.
  stream.rabitq_config_ = ram.rabitq_config_;
  if (short_ids) {
    int rejected = 0;
    for (bool streaming : {false, true}) {
      try {
        if (streaming)
          stream.PopulateElements((dir / "base.fvecs").string(), attrs.data(), n);
        else
          ram.PopulateElements(base.data(), attrs.data(), n);
      } catch (const std::runtime_error &e) {
        Require(std::string(e.what()).find("cids count") != std::string::npos,
                "unexpected CID error");
        ++rejected;
      }
    }
    Require(rejected == 2, "short CID inputs must fail before encoding");
    return;
  }
  ram.PopulateElements(base.data(), attrs.data(), n);
  stream.PopulateElements((dir / "base.fvecs").string(), attrs.data(), n);
  Require(stream.quant_vecs_.size() == n * stream.quant_vec_bytes_ &&
              stream.factors_.size() == n * 4,
          "population must allocate codes for the actual prefix");
  // Init keeps extra CIDs, but Save persists only n. Compare the active prefix.
  ram.cids_.resize(n);
  stream.cids_.resize(n);
  SameQuantization(ram, stream);
  CheckDecodedCodes(stream, base);
  stream.BuildPostPopulate(1);
  stream.graph_file_->Sync();
  stream.Save((dir / "meta").string());
  auto loaded_owner =
      std::make_unique<OwnedIndex>((dir / "stream.graph").string(), (dir / "meta").string(), space);
  auto &loaded = *loaded_owner;
  SameQuantization(stream, loaded);
  auto reference_owner =
      std::make_unique<OwnedIndex>((dir / "stream.graph").string(), (dir / "meta").string(), space);
  auto &reference = *reference_owner;
  reference.RequantizeRaBitQ((dir / "centroids.fvecs").string(), (dir / "cids.ivecs").string(),
                             space, bits);
  reference.cids_.resize(n);
  CheckDecodedCodes(reference, base);
  // Requantization draws a new calibration. A fresh RAM oracle using that
  // exact factor verifies all persisted-code inputs and output factors.
  auto oracle = std::make_unique<OwnedIndex>(
      n, dim, n - 1, 128, space, (dir / "oracle.graph").string(), 2, 0, false, complete_window);
  oracle->InitRaBitQ((dir / "centroids.fvecs").string(), (dir / "cids.ivecs").string(), space,
                     bits);
  oracle->rabitq_config_ = reference.rabitq_config_;
  oracle->PopulateElements(base.data(), attrs.data(), n);
  oracle->cids_.resize(n);
  SameQuantization(*oracle, reference);
  DirectQueries(loaded, reference, base, attrs);
  PackedRoundTrip(loaded, stream, dir, base, attrs);
  std::cout << "QUANT_CASE_PASS dim=" << dim << " bits=" << bits << " space=" << space
            << " preset_count=" << preset_count
            << " queries=12 scalar_queries=9 packed_roundtrip=1\n";
}
int main(int argc, char **argv) {
  try {
    omp_set_num_threads(2);
    Scratch scratch;
    Require(argc == 1 || (argc == 2 && std::string(argv[1]) == "--regression"),
            "usage: test_disk_build_quantization [--regression]");
    if (argc == 2)
      RunCase(scratch.path, 128, 3, "l2", true);
    else {
      size_t cases = 0;
      for (size_t dim : {128ul, 130ul, 384ul})
        for (const std::string space : {"l2", "ip"})
          for (size_t bits : {1ul, 2ul, 3ul, 4ul, 8ul}) {
            const auto dir = scratch.path / std::to_string(cases);
            fs::create_directory(dir);
            RunCase(dir, dim, bits, space, (cases++ % 2) != 0);
          }
      const auto dir = scratch.path / "short";
      fs::create_directory(dir);
      RunCase(dir, 128, 3, "l2", false, true);
      std::cout << "QUANT_MATRIX_PASS cases=" << cases << " direct_queries=" << cases * 12
                << " short_cids=2\n";
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
}
