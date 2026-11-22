#pragma once
#include "core/types.hpp"
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace benchmark::formal {
inline uint64_t InputBytes(uint64_t rows, uint64_t width) {
  if (!rows || !width || rows > std::numeric_limits<uint64_t>::max() / width)
    throw std::invalid_argument("invalid or overflowing input shape");
  return rows * width;
}
inline std::ifstream OpenExactInput(const std::string &path, uint64_t bytes) {
  if (std::filesystem::file_size(path) != bytes)
    throw std::runtime_error("input length differs from declared shape: " + path);
  std::ifstream file(path, std::ios::binary);
  if (!file)
    throw std::runtime_error("cannot open input: " + path);
  return file;
}
inline void ReadExactInput(std::ifstream &file, void *data, uint64_t bytes) {
  if (bytes > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max()))
    throw std::invalid_argument("input read exceeds stream limit");
  if (!file.read(static_cast<char *>(data), static_cast<std::streamsize>(bytes)))
    throw std::runtime_error("short input read");
}
inline void FinishInput(std::ifstream &file) {
  if (file.peek() != std::char_traits<char>::eof() || file.bad())
    throw std::runtime_error("input has trailing data or read error");
}
inline std::unique_ptr<float[]> ReadQueries(const std::string &path, size_t dim, size_t rows) {
  if (!dim || dim > static_cast<size_t>(std::numeric_limits<int32_t>::max()))
    throw std::invalid_argument("invalid query dimension");
  const auto elements = InputBytes(rows, dim);
  if (elements > std::numeric_limits<size_t>::max() / sizeof(float))
    throw std::invalid_argument("query allocation overflows");
  auto file = OpenExactInput(path, InputBytes(rows, (uint64_t(dim) + 1) * 4));
  auto data = std::make_unique<float[]>(static_cast<size_t>(elements));
  for (size_t row = 0; row < rows; ++row) {
    int32_t header = 0;
    ReadExactInput(file, &header, sizeof(header));
    if (header != static_cast<int32_t>(dim))
      throw std::runtime_error("query row dimension mismatch");
    ReadExactInput(file, data.get() + row * dim, uint64_t(dim) * sizeof(float));
  }
  FinishInput(file);
  return data;
}
inline std::vector<alliance::attr_range<int>> ReadRanges(const std::string &path, size_t rows) {
  static_assert(sizeof(int) == 4, "formal range format uses int32");
  auto file = OpenExactInput(path, InputBytes(rows, 8));
  std::vector<alliance::attr_range<int>> ranges;
  ranges.reserve(rows);
  for (size_t row = 0; row < rows; ++row) {
    int32_t bounds[2];
    ReadExactInput(file, bounds, sizeof(bounds));
    if (bounds[0] > bounds[1])
      throw std::runtime_error("reversed query range");
    ranges.push_back({bounds[0], bounds[1]});
  }
  FinishInput(file);
  return ranges;
}
inline std::vector<std::vector<alliance::label_t>> ReadGroundTruth(const std::string &path,
                                                                   size_t rows, size_t k) {
  if (!k || k > static_cast<size_t>(std::numeric_limits<int32_t>::max()))
    throw std::invalid_argument("invalid GT width");
  auto file = OpenExactInput(path, InputBytes(rows, (uint64_t(k) + 1) * 4));
  std::vector<std::vector<alliance::label_t>> gt(rows);
  for (auto &row : gt) {
    int32_t header = 0;
    ReadExactInput(file, &header, sizeof(header));
    if (header != static_cast<int32_t>(k))
      throw std::runtime_error("GT row width mismatch");
    row.resize(k);
    for (auto &id : row) {
      uint32_t original_id = 0;
      ReadExactInput(file, &original_id, sizeof(original_id));
      id = original_id;
    }
  }
  FinishInput(file);
  return gt;
}
} // namespace benchmark::formal
