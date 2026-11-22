#include "index/index_disk.hpp"
#include "quantization/float_query_dot.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

namespace {

constexpr size_t kDim = 64;
using Index = alliance::AllianceIndexDisk<int, float>;
using Estimator = Index::RaBitQEstimator;

void Check(bool condition, const std::string &message) {
  // These checks execute in Release builds too.
  if (!condition)
    throw std::runtime_error(message);
}

void Near(float actual, float expected, float tolerance, const char *message) {
  Check(std::isfinite(actual) && std::abs(actual - expected) <= tolerance,
        std::string(message) + ": actual=" + std::to_string(actual) +
            ", expected=" + std::to_string(expected));
}

struct TemporaryGraph {
  std::string path;
  TemporaryGraph() {
    std::string pattern =
        (std::filesystem::temp_directory_path() / "alliance_estimator_XXXXXX").string();
    std::vector<char> bytes(pattern.begin(), pattern.end());
    bytes.push_back('\0');
    const int fd = ::mkstemp(bytes.data());
    if (fd < 0)
      throw std::runtime_error("mkstemp failed");
    ::close(fd);
    path = bytes.data();
  }
  ~TemporaryGraph() {
    std::remove(path.c_str());
  }
};

// An orthogonal identity rotation makes the query/code fixture independently
// calculable while still exercising the production estimator constructor.
class IdentityRotator final : public rabitqlib::Rotator<float> {
public:
  IdentityRotator() : Rotator(kDim, kDim) {
  }
  void rotate(const float *src, float *dst) const override {
    std::copy_n(src, kDim, dst);
  }
  void load(std::ifstream &) override {
  }
  void save(std::ofstream &) const override {
  }
};

struct Fixture {
  TemporaryGraph file;
  Index index;
  Fixture() : index(1, kDim, 1, 1, "l2", file.path, 4, 0, false, 64) {
    index.padded_dim_ = kDim;
    index.maxC_ = 2;
    index.rotator_ = new IdentityRotator;
    index.centroids_ = new float[2 * kDim]{};
    std::fill_n(index.centroids_ + kDim, kDim, 0.625f);
    index.rabitq_space_name_ = "l2";
    index.rabitq_space_ = new alliance::L2Space(kDim);
    index.rabitq_distfunc_ = index.rabitq_space_->get_dist_func();
    index.rabitq_dist_func_param_ = index.rabitq_space_->get_dist_func_param();
    SetBits(1);
    index.factors_ = {100.0f, 0.0f, 0.0f, 0.0f};
  }
  void SetBits(unsigned bits) {
    index.packed_rabitq_.Clear();
    index.rabitq_bits_ = bits;
    index.quant_vec_bytes_ = bits * kDim / 8;
    index.quant_vecs_.assign(index.quant_vec_bytes_, 0);
  }
};

void TestQueryCentroidNormAndModelError(Fixture &fixture) {
  auto &index = fixture.index;
  std::vector<float> query(kDim, 1.0f);
  Estimator estimator(query.data(), &index);
  Near(estimator.dist_to_centroids_[0], 64, 0, "origin centroid squared norm");
  Near(estimator.dist_to_centroids_[1], 9, 0, "second centroid squared norm");
  const float no_error0 = estimator.l2_lower_bound(0, 0);
  const float no_error1 = estimator.l2_lower_bound(1, 0);
  index.factors_[2] = 2;
  const float bound0 = estimator.l2_lower_bound(0, 0);
  const float bound1 = estimator.l2_lower_bound(1, 0);
  // F_error is a coefficient: the actual errors are 2*sqrt(64) and 2*sqrt(9).
  // Differencing removes the independent floating-point guard.
  Near(no_error0 - bound0, 16, 0.0001f, "query norm multiplies F_error");
  Near(no_error1 - bound1, 6, 0.0001f, "candidate centroid selects query norm");
  index.factors_[2] = 4;
  Near(bound0 - estimator.l2_lower_bound(0, 0), 16, 0.0001f,
       "stored factor scales the complete model error");
  index.factors_[2] = 0;
}

void TestFloatQueryAgainstExactL2(Fixture &fixture) {
  auto &index = fixture.index;
  std::vector<float> query(kDim, -1.0f), exact_vector(kDim, -0.5f);
  std::fill(query.begin() + kDim / 2, query.end(), 1.0f);
  std::fill(exact_vector.begin() + kDim / 2, exact_vector.end(), 0.5f);
  float exact_distance = 0;
  for (size_t d = 0; d < kDim; ++d) {
    const float delta = query[d] - exact_vector[d];
    exact_distance += delta * delta;
  }
  Near(exact_distance, 16, 0, "independent exact L2 fixture");
  for (unsigned bits : {1u, 2u, 3u, 4u, 8u}) {
    fixture.SetBits(bits);
    const unsigned max_code = (1u << bits) - 1;
    // x[d] = (code[d] - max_code/2) / max_code. All bit planes
    // have their upper 32 bits set; the model itself is exact (F_error=0).
    for (unsigned plane = 0; plane < bits; ++plane) {
      for (size_t d = kDim / 2; d < kDim; ++d) {
        index.quant_vecs_[plane * (kDim / 8) + d / 8] |= uint8_t(1u << (d % 8));
      }
    }
    index.factors_ = {16, -2.0f / max_code, 0, float(max_code * (kDim / 2))};
    Estimator estimator(query.data(), &index);
    const float estimate = estimator.dist_estimate(0, 0);
    Near(estimate, 16.0f, 0.0001f, "float-query packed-code estimate has no SQ endpoint bias");
    const float bound = estimator.l2_lower_bound(0, 0);
    Near(estimator.l2_lower_bound(0, 0, estimate), bound, 0, "reuse estimate is equivalent");
    Near(estimator.l2_lower_bound(0, 0, estimate + 2) - bound, 2, 0.0001f,
         "overload consumes supplied score without decoding code again");
    Check(bound <= exact_distance && bound > exact_distance - 0.1f,
          "float-query bound covers exact L2 for " + std::to_string(bits) + "-bit codes");
  }
}

void TestInvalidBoundsFallBack(Fixture &fixture) {
  auto &index = fixture.index;
  fixture.SetBits(1);
  index.factors_ = {100, 0, 2, 0};
  std::vector<float> query(kDim, 1);
  Estimator estimator(query.data(), &index);
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  auto invalid = [&]() {
    const float bound = estimator.l2_lower_bound(0, 0);
    Check(std::isinf(bound) && bound < 0, "invalid bound must force exact IO");
  };
  index.rabitq_space_name_ = "ip";
  invalid();
  index.rabitq_space_name_ = "l2";
  for (size_t component = 0; component < 4; ++component) {
    const float saved = index.factors_[component];
    for (float value : {nan, inf}) {
      index.factors_[component] = value;
      invalid();
    }
    index.factors_[component] = saved;
  }
  for (size_t component : {size_t(2), size_t(3)}) {
    const float saved = index.factors_[component];
    index.factors_[component] = -1;
    invalid();
    index.factors_[component] = saved;
  }
  for (float value : {-1.0f, nan, inf}) {
    estimator.dist_to_centroids_[0] = value;
    invalid();
  }
}

// Decode generated scalar codes independently of the production bit-plane
// kernel. Both distance APIs must retain the exact same score and model error.
void TestDistanceAPIsAgainstScalarCodes() {
  std::mt19937 random(20260924);
  uint64_t hash = 14695981039346656037ull;
  size_t cases = 0;
  for (const bool ip : {false, true}) {
    Fixture fixture;
    auto &index = fixture.index;
    if (ip) {
      delete index.rabitq_space_;
      index.rabitq_space_ = new alliance::InnerProductSpace(kDim);
      index.rabitq_space_name_ = "ip";
      index.rabitq_distfunc_ = index.rabitq_space_->get_dist_func();
      index.rabitq_dist_func_param_ = index.rabitq_space_->get_dist_func_param();
    }
    for (const unsigned bits : {1u, 2u, 3u, 4u, 8u}) {
      fixture.SetBits(bits);
      for (size_t sample = 0; sample < 32; ++sample) {
        std::vector<uint8_t> codes(kDim);
        std::vector<float> query(kDim);
        for (size_t d = 0; d < kDim; ++d) {
          codes[d] = random() & ((1u << bits) - 1);
          query[d] = sample < 2 ? float(sample) : float(int(random() % 257) - 128) / 16;
        }
        float pop_count = 0;
        index.PackMultiBitCode(codes.data(), index.quant_vecs_.data(), pop_count);
        index.factors_ = {37.5f, sample % 2 ? -0.125f : 0.25f, float(sample) / 8, pop_count};
        Estimator estimator(query.data(), &index);
        double centered_dot = 0;
        const double cb = -0.5 * ((1u << bits) - 1);
        for (size_t d = 0; d < kDim; ++d)
          centered_dot += (static_cast<double>(codes[d]) + cb) * query[d];
        for (uint8_t cid = 0; cid < 2; ++cid) {
          const float expected =
              static_cast<float>(double(index.factors_[0]) + estimator.dist_to_centroids_[cid] +
                                 double(index.factors_[1]) * centered_dot);
          const float plain = estimator.dist_estimate(cid, 0);
          const auto [with_error, error] = estimator.dist_estimate_with_errorbound(cid, 0);
          Near(plain, expected, 1e-5f * std::max(1.0f, std::abs(expected)),
               "packed multibit score matches independent scalar codes");
          Check(std::memcmp(&plain, &with_error, sizeof(float)) == 0,
                "distance APIs disagree at the bit level");
          Check(error == index.factors_[2], "stored model error changed");
          const float split_bound = estimator.l2_lower_bound(cid, 0, plain);
          index.packed_rabitq_.Assign(index.quant_vecs_, index.factors_, 1, index.quant_vec_bytes_);
          const auto [packed_score, packed_error] = estimator.dist_estimate_with_errorbound(cid, 0);
          Check(!std::memcmp(&plain, &packed_score, 4) && !std::memcmp(&error, &packed_error, 4),
                "packed RAM changed score/error");
          const float packed_bound = estimator.l2_lower_bound(cid, 0, plain);
          Check(!std::memcmp(&split_bound, &packed_bound, 4), "packed RAM changed lower bound");
          index.packed_rabitq_.Clear();
          if (bits > 1) {
            index.packed_rabitq_.Assign(index.quant_vecs_, index.factors_, 1,
                                        index.quant_vec_bytes_,
                                        alliance::detail::NativeCodeLayout(bits));
            const auto [native_score, native_error] =
                estimator.dist_estimate_with_errorbound(cid, 0);
            Near(native_score, expected, 1e-5f * std::max(1.0f, std::abs(expected)),
                 "native compact3 estimator score");
            Check(native_error == error, "native compact3 changed factor");
            Check(native_score == estimator.dist_estimate(cid, 0),
                  "native compact3 distance APIs differ");
            index.packed_rabitq_.Clear();
          }
          for (const float value : {plain, with_error, error}) {
            uint32_t raw;
            std::memcpy(&raw, &value, sizeof(raw));
            hash = (hash ^ raw) * 1099511628211ull;
          }
          ++cases;
        }
      }
    }
  }
  std::cout << "distance_api_cases=" << cases << " distance_fixture_hash=" << hash << '\n';
}

void TestIPIntervals() {
  Fixture fixture;
  auto &idx = fixture.index;
  delete idx.rabitq_space_;
  idx.rabitq_space_ = new alliance::InnerProductSpace(kDim);
  idx.rabitq_space_name_ = "ip";
  idx.rabitq_distfunc_ = idx.rabitq_space_->get_dist_func();
  idx.rabitq_dist_func_param_ = idx.rabitq_space_->get_dist_func_param();
  for (unsigned bits : {1u, 2u, 3u, 4u, 8u}) {
    fixture.SetBits(bits);
    std::vector<float> query(kDim, 0.625f), vector(kDim, 0.125f);
    std::fill(vector.begin() + 32, vector.end(), 1.125f);
    std::vector<uint8_t> code(kDim);
    auto &f = idx.factors_;
    rabitqlib::quant::quantize_full_single(vector.data(), idx.centroids_ + kDim, kDim, bits,
                                           code.data(), f[0], f[1], f[2], rabitqlib::METRIC_IP,
                                           idx.rabitq_config_);
    idx.PackMultiBitCode(code.data(), idx.quant_vecs_.data(), f[3]);
    Estimator e(query.data(), &idx);
    const float score = e.dist_estimate(1, 0);
    const float exact =
        idx.rabitq_distfunc_(query.data(), vector.data(), idx.rabitq_dist_func_param_);
    Near(score - 1, exact, 0.01f, "IP estimate offset must align with exact metric");
    auto [lo, hi] = e.distance_bounds(1, 0, score);
    Check(lo <= exact && exact <= hi, "real quantizer IP interval excludes exact distance");
    query.assign(kDim, 1);
    Estimator nonzero(query.data(), &idx);
    f[2] = 0;
    auto zero = nonzero.distance_bounds(1, 0, score);
    f[2] = 2;
    auto model = nonzero.distance_bounds(1, 0, score);
    Near(zero.first - model.first, 6, 0.0001f, "IP model uses Euclidean q-centroid norm");
    Near(model.second - zero.second, 6, 0.0001f, "IP upper bound symmetric model error");
    f[2] = 4;
    auto wider = nonzero.distance_bounds(1, 0, score);
    Check(wider.first < model.first && wider.second > model.second,
          "stored model factor widens IP interval");
  }
}

} // namespace

void TestFloatQueryRefinement() {
  Fixture fixture;
  auto &index = fixture.index;
  std::mt19937 rng(9021);
  std::uniform_real_distribution<float> dist(-3, 3);
  for (const std::string metric : {"l2", "ip"}) {
    index.rabitq_space_name_ = metric;
    delete index.rabitq_space_;
    if (metric == "l2")
      index.rabitq_space_ = new alliance::L2Space(kDim);
    else
      index.rabitq_space_ = new alliance::InnerProductSpace(kDim);
    index.rabitq_distfunc_ = index.rabitq_space_->get_dist_func();
    index.rabitq_dist_func_param_ = index.rabitq_space_->get_dist_func_param();
    for (unsigned bits : {1u, 2u, 3u, 4u})
      for (int trial = 0; trial < 24; ++trial) {
        fixture.SetBits(bits);
        std::vector<float> q(kDim);
        std::vector<unsigned> raw(kDim);
        double centered = 0, pop = 0;
        double cb = -0.5 * ((1u << bits) - 1);
        for (size_t d = 0; d < kDim; ++d) {
          q[d] = trial == 0 ? 0.75f : dist(rng);
          raw[d] = rng() % (1u << bits);
          pop += raw[d];
          centered += (raw[d] + cb) * q[d];
          for (unsigned p = 0; p < bits; ++p)
            index.quant_vecs_[p * kDim / 8 + d / 8] |= ((raw[d] >> p) & 1u) << (d % 8);
        }
        index.factors_ = {100.0f, trial % 2 ? 0.3f : -0.3f, 0.0f, float(pop)};
        if (bits > 1 && trial % 2)
          index.packed_rabitq_.Assign(index.quant_vecs_, index.factors_, 1, index.quant_vec_bytes_,
                                      alliance::detail::NativeCodeLayout(bits));
        Estimator e(q.data(), &index);
        float known = e.dist_estimate(0, 0);
        double expected = 100.0 + e.dist_to_centroids_[0] + double(index.factors_[1]) * centered -
                          (metric == "ip" ? 1.0 : 0.0);
        auto refined = e.distance_bounds(0, 0, known, true);
        auto coarse = e.distance_bounds(0, 0, known);
        Check(refined.first <= expected && refined.second >= expected,
              "float-query interval contains independent centered code score");
        Check(refined.second - refined.first <= coarse.second - coarse.first + 0.001f,
              "float recomputation preserves interval width within rounding");
        Check(known == e.dist_estimate(0, 0), "refinement must not change navigation score");
      }
  }
}
void TestNativeFloatDot() {
  std::mt19937 rng(23891);
  for (size_t dim : {64u, 128u, 192u, 384u, 768u, 1536u})
    for (unsigned bits : {1u, 2u, 3u, 4u, 8u})
      for (unsigned trial = 0; trial < 12; ++trial) {
        std::vector<uint8_t> packed(dim * bits / 8, 0);
        std::vector<float> q(dim);
        double reference = 0, magnitude = 0;
        for (size_t d = 0; d < dim; ++d) {
          unsigned c = rng() & ((1u << bits) - 1);
          q[d] = trial == 0 ? 0.75f : float(int(rng() % 200001) - 100000) / 37.0f;
          for (unsigned p = 0; p < bits; ++p)
            packed[p * dim / 8 + d / 8] |= ((c >> p) & 1u) << (d % 8);
          reference += double(c) * q[d];
          magnitude += std::abs(double(c) * q[d]);
        }
        const double actual =
            alliance::detail::FloatQueryCodeDot(packed.data(), q.data(), dim, bits);
        const double tolerance =
            std::numeric_limits<float>::epsilon() * (dim / 16 + 16) * magnitude + 1e-8;
        Check(std::isfinite(actual) && std::abs(actual - reference) <= tolerance,
              "original float kernel disagrees with independent unpacked dot");
        if (bits > 1) {
          std::vector<uint8_t> compact(packed.size());
          alliance::detail::PackCompact(packed.data(), compact.data(), packed.size(),
                                        alliance::detail::NativeCodeLayout(bits));
          const double native = alliance::detail::FloatQueryCodeDot(
              compact.data(), q.data(), dim, bits, alliance::detail::NativeCodeLayout(bits));
          Check(std::isfinite(native) && std::abs(native - reference) <= tolerance,
                "compact3 kernel scalar oracle");
        }
      }
}

void TestCompactBasis() {
  for (unsigned bits : {2u, 3u, 4u, 8u})
    for (size_t dim : {64u, 128u, 384u})
      for (size_t d = 0; d < dim; ++d)
        for (unsigned c = 0; c < (1u << bits); ++c) {
          std::vector<uint8_t> planes(dim * bits / 8, 0), compact(planes.size());
          std::vector<float> q(dim, 0);
          q[d] = 1;
          for (unsigned p = 0; p < bits; ++p)
            planes[p * dim / 8 + d / 8] |= ((c >> p) & 1) << (d % 8);
          const auto layout = alliance::detail::NativeCodeLayout(bits);
          alliance::detail::PackCompact(planes.data(), compact.data(), planes.size(), layout);
          Check(alliance::detail::FloatQueryCodeDot(compact.data(), q.data(), dim, bits, layout) ==
                    c,
                "compact basis coordinate/bit-order mismatch");
        }
}

void TestRefinementReuse() {
  Fixture fixture;
  auto &idx = fixture.index;
  std::mt19937 rng(908);
  for (const std::string metric : {"l2", "ip"})
    for (unsigned bits : {1u, 2u, 3u, 4u, 8u}) {
      fixture.SetBits(bits);
      idx.rabitq_space_name_ = metric;
      std::vector<float> q(kDim);
      std::vector<uint8_t> code(kDim);
      for (size_t d = 0; d < kDim; ++d) {
        q[d] = float(int(rng() % 20001) - 10000) / 73;
        code[d] = rng() % (1u << bits);
      }
      idx.factors_ = {100, 0.023f, 0.5f, 0};
      idx.PackMultiBitCode(code.data(), idx.quant_vecs_.data(), idx.factors_[3]);
      idx.packed_rabitq_.Assign(idx.quant_vecs_, idx.factors_, 1, idx.quant_vec_bytes_,
                                alliance::detail::NativeCodeLayout(bits));
      Estimator e(q.data(), &idx);
      for (uint8_t cid : {0, 1}) {
        const double precise = e.dist_estimate_precise(cid, 0);
        const float known = static_cast<float>(precise);
        Check(known == e.dist_estimate(cid, 0), "precise API changed navigation score");
        e.refinement_cache_.Reset(0);
        const auto cold = e.distance_bounds(cid, 0, known, true);
        const size_t recomputed = e.refinement_recomputed_, reused = e.refinement_reused_;
        e.refinement_cache_.Reset(1);
        e.refinement_cache_.Insert(0, cid, precise);
        const auto warm = e.distance_bounds(cid, 0, known, true);
        Check(!std::memcmp(&cold.first, &warm.first, 4) &&
                  !std::memcmp(&cold.second, &warm.second, 4),
              "reuse changed refined interval bits");
        Check(e.refinement_reused_ == reused + 1 && e.refinement_recomputed_ == recomputed,
              "reuse did not eliminate dot");
        e.refinement_cache_.Reset(0);
        e.distance_bounds(cid, 0, known, true);
        Check(e.refinement_recomputed_ == recomputed + 1,
              "missing-cache fallback did not recompute");
      }
    }
  alliance::detail::RefinementScoreCache cache;
  cache.Reset(64);
  for (unsigned id = 0; id < 64; ++id)
    cache.Insert(id * 128, 3, double(id) + 0.123456789);
  for (unsigned id = 0; id < 64; ++id) {
    double value = 0;
    const double expected = double(id) + 0.123456789;
    Check(cache.Find(id * 128, 3, float(expected), value) && value == expected,
          "flat cache collision lost value");
    Check(!cache.Find(id * 128, 4, float(expected), value), "cache ignored centroid identity");
    Check(!cache.Find(id * 128, 3, float(expected) + 1, value),
          "cache ignored rounded-score mismatch");
  }
  cache.Reset(1);
  double value = 0;
  Check(!cache.Find(0, 3, 0.123456789f, value), "cache leaked old query");
}

void TestProductionPacking() {
  for (unsigned bits : {1u, 2u, 3u, 4u, 8u}) {
    Fixture fixture;
    auto &idx = fixture.index;
    fixture.SetBits(bits);
    idx.curvec_num_ = 1;
    idx.rabitq_initialized_ = true;
    std::vector<uint8_t> raw(kDim), canonical(kDim * bits / 8);
    std::vector<float> q(kDim);
    double scalar = 0;
    for (size_t d = 0; d < kDim; ++d) {
      raw[d] = (d * 13 + 7) & ((1u << bits) - 1);
      q[d] = float(int(d % 17) - 8) * 0.125f;
      scalar += double(raw[d]) * q[d];
    }
    idx.factors_ = {3.0f, 0.125f, 0.25f, 0.0f};
    idx.PackMultiBitCode(raw.data(), idx.quant_vecs_.data(), idx.factors_[3]);
    canonical = idx.quant_vecs_;
    const auto factors = idx.factors_;
    idx.PackRaBitQForSearch();
    Check(idx.quant_vecs_.empty() && idx.factors_.empty(),
          "production packing retained duplicate arrays");
    const auto layout = idx.RaBitQLayout();
    Check(static_cast<unsigned>(layout) == (bits == 1 ? 0 : bits),
          "production packing did not select original library layout");
    std::vector<uint8_t> restored(canonical.size());
    idx.packed_rabitq_.CopyCanonicalCode(0, restored.data());
    Check(restored == canonical, "production packing changed checkpoint code values");
    Check(!std::memcmp(idx.RaBitQFactors(0), factors.data(), 16),
          "production packing changed factors");
    const double dot =
        alliance::detail::FloatQueryCodeDot(idx.RaBitQCode(0), q.data(), kDim, bits, layout);
    Check(std::abs(dot - scalar) <= 1e-5 * std::max(1.0, std::abs(scalar)),
          "production kernel differs from scalar code dot");
    Estimator e(q.data(), &idx);
    const float score = e.dist_estimate(0, 0);
    const auto bounds = e.distance_bounds(0, 0, score, true);
    Check(std::isfinite(score) && bounds.first <= score && bounds.second >= score,
          "production refinement interval invalid");
    std::cout << "production bits=" << bits
              << " layout=" << alliance::detail::CodeLayoutName(layout) << " passed\n";
  }
}

int main() {
  try {
    TestFloatQueryRefinement();
    TestIPIntervals();
    TestNativeFloatDot();
    TestCompactBasis();
    TestProductionPacking();
    TestRefinementReuse();

    Fixture fixture;
    TestQueryCentroidNormAndModelError(fixture);
    TestFloatQueryAgainstExactL2(fixture);
    TestInvalidBoundsFallBack(fixture);
    TestDistanceAPIsAgainstScalarCodes();
    std::cout << "RaBitQ estimator lower-bound fixtures passed\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "RaBitQ estimator fixture failed: " << e.what() << '\n';
    return 1;
  }
}
