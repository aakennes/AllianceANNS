#pragma once
// alliance/core/io.hpp -- Binary I/O utilities (disk + vector I/O)
// Adapted from StorageAirRFANNS (WoW) wow/disk.hh and wow/vec_io.hh
// Note: libaio dependency removed for pure memory version

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string>

namespace alliance {

template <typename T> static void WriteBinaryPOD(std::ostream &out, const T &podRef) {
  out.write((char *)&podRef, sizeof(T));
}

template <typename T> static void ReadBinaryPOD(std::istream &in, T &podRef) {
  in.read((char *)&podRef, sizeof(T));
}

// FvecsStreamReader used to live here, on a buffered ifstream. Its only callers
// were PopulateElements' two streaming passes, which now use
// AioFvecsStreamReader (alliance/core/aio_stream.hpp) so the fvecs input obeys
// the same all-libaio rule as every other bulk read. This header stays free of
// the libaio dependency on purpose -- the pure in-memory targets include it.

} // namespace alliance

// Vector I/O utilities (fvecs/ivecs format)

inline auto fvecs_read(const std::string &filename, size_t &d_out, size_t &n_out) -> float * {
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

inline auto ivecs_read(const std::string &filename, size_t &d_out, size_t &n_out) -> int * {
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("Cannot open file " + filename);
  }

  int d;
  in.read(reinterpret_cast<char *>(&d), 4);
  if (!in.good()) {
    throw std::runtime_error("Failed to read dimension from " + filename);
  }
  d_out = d;

  in.seekg(0, std::ios::beg);
  in.seekg(0, std::ios::end);
  size_t file_size = in.tellg();
  in.seekg(0, std::ios::beg);

  size_t n = file_size / (4 + d * sizeof(int));
  n_out = n;

  int *data = new int[d * n];

  for (size_t i = 0; i < n; ++i) {
    int dim_check;
    in.read(reinterpret_cast<char *>(&dim_check), 4);
    if (dim_check != d) {
      throw std::runtime_error("Inconsistent vector dimension at entry " + std::to_string(i));
    }
    in.read(reinterpret_cast<char *>(data + i * d), d * sizeof(int));
  }

  in.close();
  return data;
}
