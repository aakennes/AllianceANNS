#pragma once
#include "io/aio_stream.hpp"
#include <fstream>
#include <limits>

namespace alliance {
struct FloatVectorFileShape {
  size_t count, dim;
  uint64_t bytes;
};

// Explicit format avoids guessing between fvecs and an (N,D)+float32 bin file.
// Only the small header and file length are read by the memory-report path.
inline auto ReadFloatVectorFileShape(const std::string &path, const std::string &format)
    -> FloatVectorFileShape {
  if (format != "fvecs" && format != "bin")
    throw std::invalid_argument("basevec_format must be fvecs or bin");
  std::ifstream f(path, std::ios::binary);
  if (!f)
    throw std::runtime_error("Cannot open file: " + path);
  f.seekg(0, std::ios::end);
  const auto end = f.tellg();
  if (end < 0)
    throw std::runtime_error("Cannot determine vector file size: " + path);
  const uint64_t bytes = static_cast<uint64_t>(end);
  f.seekg(0);
  uint32_t first = 0, dim = 0;
  if (!f.read(reinterpret_cast<char *>(&first), 4))
    throw std::runtime_error("Truncated vector header: " + path);
  uint64_t count = 0;
  if (format == "bin") {
    if (!first || !f.read(reinterpret_cast<char *>(&dim), 4) || !dim)
      throw std::runtime_error("Invalid bin (N,D) header: " + path);
    const uint64_t row = uint64_t(dim) * sizeof(float);
    count = first;
    if (bytes < 8 || (bytes - 8) % row || (bytes - 8) / row != count)
      throw std::runtime_error("Bin vector count/length mismatch: " + path);
  } else {
    if (!first || first > uint32_t(std::numeric_limits<int32_t>::max()))
      throw std::runtime_error("Invalid fvecs dimension: " + path);
    dim = first;
    const uint64_t row = 4 + uint64_t(dim) * sizeof(float);
    if (bytes % row || bytes / row == 0)
      throw std::runtime_error("Truncated vector records: " + path);
    count = bytes / row;
  }
  if (count > std::numeric_limits<size_t>::max() ||
      dim > std::numeric_limits<size_t>::max() / sizeof(float))
    throw std::overflow_error("Vector shape exceeds address space");
  return {size_t(count), size_t(dim), bytes};
}

class AioFloatVectorStreamReader {
public:
  explicit AioFloatVectorStreamReader(const std::string &path, const std::string &format = "fvecs")
      : path_(path), shape_(ReadFloatVectorFileShape(path, format)) {
    if (format == "fvecs")
      fvecs_ = std::make_unique<AioFvecsStreamReader>(path);
    Rewind();
  }
  size_t dim() const {
    return shape_.dim;
  }
  size_t count() const {
    return shape_.count;
  }
  bool ReadNext(float *out) {
    if (served_ == shape_.count)
      return false;
    if (fvecs_) {
      if (!fvecs_->ReadNext(out))
        throw std::runtime_error("Unexpected fvecs EOF");
    } else
      stream_->Read(out, shape_.dim * sizeof(float));
    ++served_;
    return true;
  }
  void Rewind() {
    if (fvecs_)
      fvecs_->Rewind();
    else
      stream_ = std::make_unique<AioSeqStreamReader>(path_, 8, shape_.bytes, false,
                                                     AioFvecsStreamReader::kFvecsChunkBytes);
    served_ = 0;
  }

private:
  std::string path_;
  FloatVectorFileShape shape_;
  size_t served_{};
  std::unique_ptr<AioFvecsStreamReader> fvecs_;
  std::unique_ptr<AioSeqStreamReader> stream_;
};
} // namespace alliance
