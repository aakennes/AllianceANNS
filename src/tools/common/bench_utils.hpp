#pragma once
#include "search/search_results.hpp"
#include <string>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <omp.h>
#include "core/types.hpp"
#include "space/space.hpp"

namespace benchmark {
auto fvecs_read(const std::string &filename, size_t &d_out, size_t &n_out) -> float * {
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("Cannot open file " + filename);
  }
  int d;
  in.read(reinterpret_cast<char *>(&d), 4);
  d_out = d;
  in.seekg(0, std::ios::beg);
  in.seekg(0, std::ios::end);
  size_t file_size = in.tellg();
  in.seekg(0, std::ios::beg);
  size_t n = file_size / (4 + d * sizeof(float));
  n_out = n;
  float *data = new float[d * n];
  for (size_t i = 0; i < n; ++i) {
    in.read(reinterpret_cast<char *>(&d), 4);
    in.read(reinterpret_cast<char *>(data + i * d), d * sizeof(float));
  }
  in.close();
  return data;
}

auto LoadRange(const std::string &location) -> std::vector<alliance::attr_range<int>> {
  std::vector<alliance::attr_range<int>> query_filters;
  std::ifstream ifs(location, std::ios::binary);
  if (!ifs.is_open()) {
    std::cout << "Fail to open: " << location << std::endl;
    std::abort();
  }
  ifs.seekg(0, std::ios::end);
  size_t file_size = ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  auto n_query = file_size / (2 * sizeof(int));
  query_filters.resize(n_query);
  for (size_t i = 0; i < n_query; ++i) {
    int l, u;
    ifs.read(reinterpret_cast<char *>(&l), sizeof(int));
    ifs.read(reinterpret_cast<char *>(&u), sizeof(int));
    query_filters[i] = alliance::attr_range<int>{l, u};
  }
  ifs.close();
  std::string qf_str;
  for (size_t i = 0; i < std::min<size_t>(10, n_query); ++i) {
    qf_str +=
        "[" + std::to_string(query_filters[i].l_) + "," + std::to_string(query_filters[i].u_) + "]";
  }
  std::cout << "First 10 query filters: " << qf_str << std::endl;
  return query_filters;
}

template <typename att_t, typename filter_t>
auto GenGT(size_t nb, size_t nq, size_t d, size_t k, const std::vector<filter_t> &filter,
           const float *basevec, const float *queryvec, std::vector<att_t> attvec,
           const std::string &space) -> std::vector<std::vector<alliance::label_t>> {
  alliance::SpaceInterface<float> *space_ptr;
  if (space == "l2") {
    space_ptr = new alliance::L2Space(d);
  } else if (space == "ip") {
    space_ptr = new alliance::InnerProductSpace(d);
  } else {
    throw std::runtime_error("unsupported space type " + space + ", supported: l2, ip");
  }
  auto fstdistfunc = space_ptr->get_dist_func();
  auto dist_func_param = space_ptr->get_dist_func_param();
  std::vector<std::vector<alliance::label_t>> gt(nq);
  size_t iq;
  std::cout << "Generating ground truth..." << std::endl;
#pragma omp parallel for num_threads(omp_get_max_threads()) schedule(dynamic)                      \
    shared(gt, queryvec, basevec, attvec, fstdistfunc, dist_func_param, k)
  for (iq = 0; iq < nq; ++iq) {
    std::vector<alliance::dist_id_pair> gt_cand;
    for (int ib = 0; ib < (int)nb; ++ib) {
      if (filter[iq].Test(attvec[ib])) {
        auto dist = fstdistfunc(queryvec + iq * d, basevec + ib * d, dist_func_param);
        PUSH_HEAP(gt_cand, dist, ib);
        if (gt_cand.size() > k) {
          POP_HEAP(gt_cand);
        }
      }
    }

    gt[iq].resize(gt_cand.size());
    for (size_t i = 0; i < gt_cand.size(); ++i) {
      gt[iq][i] = gt_cand[i].id_;
    }
  }
  delete space_ptr;
  return gt;
}

auto LoadGroundTruth(const std::string &gt_file) -> std::vector<std::vector<alliance::label_t>> {
  std::ifstream in(gt_file, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("Cannot open file " + gt_file);
  }
  std::vector<std::vector<alliance::label_t>> all_gt;
  while (!in.eof()) {
    int k;
    in.read(reinterpret_cast<char *>(&k), sizeof(int));
    if (in.eof()) {
      break;
    }
    std::vector<alliance::label_t> gt(k);
    for (int i = 0; i < k; ++i) {
      unsigned int ib;
      in.read(reinterpret_cast<char *>(&ib), sizeof(unsigned int));
      gt[i] = ib;
    }
    all_gt.emplace_back(gt);
  }
  std::cout << "Loaded ground truth: " << all_gt.size() << std::endl;
  std::cout << "Example: first query has " << all_gt[0].size() << std::endl;
  for (auto ib : all_gt[0]) {
    std::cout << ib << ",";
  }
  std::cout << std::endl;
  return all_gt;
}

template <typename att_t> auto LoadAttVec(const std::string att_file) -> std::vector<att_t> {
  std::ifstream in(att_file, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("Cannot open file " + att_file);
  }
  in.seekg(0, std::ios::end);
  size_t file_size = in.tellg();
  in.seekg(0, std::ios::beg);
  size_t n = file_size / sizeof(att_t);
  if (file_size % sizeof(att_t) != 0) {
    throw std::runtime_error("File size is not a multiple of att_t");
  }
  std::vector<att_t> att_vec(n);
  for (size_t i = 0; i < n; ++i) {
    in.read(reinterpret_cast<char *>(&att_vec[i]), sizeof(att_t));
  }
  in.close();
  std::cout << "Loaded att_vec: " << att_file << ", size: " << n << std::endl;
  return att_vec;
}

} // namespace benchmark
