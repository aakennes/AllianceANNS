#pragma once

#include "core/types.hpp"
#include "storage/window_ladder.hpp"
#include <cstdint>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace alliance::storage {

// Native metadata remains an ordered stream, not a serialized C++ struct.
// These tags are shared by its reader and the existing byte-compatible writer.
inline constexpr uint32_t kIdOnlyMetadataTag = 0x31444945u;
inline constexpr uint32_t kLeftStreamMetadataTag = 0x3157534cu;

class BoundedBinaryInput {
public:
  explicit BoundedBinaryInput(const std::string &path)
      : stream_(path, std::ios::binary), path_(path) {
    if (!stream_)
      throw std::runtime_error("Cannot open file: " + path);
    stream_.seekg(0, std::ios::end);
    const auto end = stream_.tellg();
    if (end < 0 || static_cast<uintmax_t>(end) > std::numeric_limits<size_t>::max())
      Fail();
    remaining_ = static_cast<size_t>(end);
    stream_.seekg(0);
    if (!stream_)
      Fail();
  }

  size_t Remaining() const {
    return remaining_;
  }

  // Check before multiplication or allocation, including skipped payloads.
  size_t Bytes(size_t count, size_t width = 1) const {
    if (width != 0 && count > remaining_ / width)
      Fail();
    const size_t bytes = count * width;
    if (bytes > static_cast<size_t>(std::numeric_limits<std::streamsize>::max()))
      Fail();
    return bytes;
  }

  void ReadBytes(void *destination, size_t count, size_t width = 1) {
    const size_t bytes = Bytes(count, width);
    if (bytes &&
        !stream_.read(static_cast<char *>(destination), static_cast<std::streamsize>(bytes)))
      Fail();
    remaining_ -= bytes;
  }

  template <typename T> T Read() {
    T value{};
    ReadBytes(&value, 1, sizeof(T));
    return value;
  }

  void Skip(size_t count, size_t width = 1) {
    const size_t bytes = Bytes(count, width);
    if (bytes && !stream_.seekg(static_cast<std::streamoff>(bytes), std::ios::cur))
      Fail();
    remaining_ -= bytes;
  }

  template <typename T> void ReadVector(std::vector<T> &values, size_t count, bool load) {
    Bytes(count, sizeof(T));
    if (load) {
      values.resize(count);
      ReadBytes(values.data(), count, sizeof(T));
    } else {
      Skip(count, sizeof(T));
    }
  }

private:
  [[noreturn]] void Fail() const {
    throw std::runtime_error("Truncated or invalid binary input: " + path_);
  }
  std::ifstream stream_;
  std::string path_;
  size_t remaining_{};
};

enum class MetadataReadMode { Load, Inspect };

template <typename att_t> struct DiskMetadata {
  size_t max_elements{}, vec_d{}, wp{}, order{}, degree{}, efc{}, count{}, max_layer{};
  size_t record_bytes{}, node_bytes{};
  std::vector<size_t> windows;
  std::vector<label_t> labels;
  std::vector<att_t> attributes;
  std::vector<tableint> sorted_ids, position_by_id;
  bool quantized{false};
  size_t padded_dim{}, code_bytes{}, centroid_count{};
  std::unique_ptr<float[]> centroids;
  std::vector<uint8_t> cids, codes;
  std::vector<float> factors;
  std::string vector_path;
  size_t completed_layer{};
  WindowAnchor anchor{WindowAnchor::kBottom};
  size_t id_only_start_layer{std::numeric_limits<size_t>::max()};
  bool left_stream{false}, left_complete{false};
  size_t left_max{}, left_batch{256};
};

// Inspect reads only geometry/windows and skips large payloads. It never opens
// the graph, vector file or index, and shares the same format checks as Load.
template <typename att_t>
auto ReadDiskMetadata(const std::string &path, MetadataReadMode mode,
                      const std::function<void(size_t, size_t)> &validate_quantized_header = {})
    -> DiskMetadata<att_t> {
  static_assert(sizeof(bool) == sizeof(uint8_t), "native metadata uses a one-byte bool");
  BoundedBinaryInput input(path);
  const bool load = mode == MetadataReadMode::Load;
  DiskMetadata<att_t> result;
  result.max_elements = input.Read<size_t>();
  result.vec_d = input.Read<size_t>();
  result.wp = input.Read<size_t>();
  result.order = input.Read<size_t>();
  result.degree = input.Read<size_t>();
  result.efc = input.Read<size_t>();
  result.count = input.Read<size_t>();
  result.max_layer = input.Read<size_t>();
  result.record_bytes = input.Read<size_t>();
  result.node_bytes = input.Read<size_t>();
  const size_t window_count = input.Read<size_t>();
  // Inspect must validate the small geometry before allocating its only array.
  // File-size bounds alone allow a corrupt (but large) file to claim huge
  // windows. Keep empty indexes readable here; the build planner rejects them.
  if (!load &&
      (result.count > result.max_elements || result.vec_d == 0 ||
       result.vec_d > std::numeric_limits<size_t>::max() / sizeof(float) || result.order < 2 ||
       result.wp >= static_cast<size_t>(std::numeric_limits<layer_t>::max()) ||
       window_count < result.wp + 1 || window_count > result.wp + 2))
    throw std::runtime_error("Invalid checkpoint geometry");
  input.ReadVector(result.windows, window_count, true);
  input.ReadVector(result.labels, result.count, load);
  input.ReadVector(result.attributes, result.count, load);
  input.ReadVector(result.sorted_ids, result.count, load);
  input.ReadVector(result.position_by_id, result.count, load);
  const uint8_t quantized = input.Read<uint8_t>();
  if (quantized > 1)
    throw std::runtime_error("Invalid checkpoint quantization tag");
  result.quantized = quantized != 0;
  if (result.quantized) {
    result.padded_dim = input.Read<size_t>();
    result.code_bytes = input.Read<size_t>();
    result.centroid_count = input.Read<size_t>();
    // Runtime-dependent validation belongs to the caller, but must precede
    // payload sizing/allocation: a corrupt padded dimension changes its length.
    if (validate_quantized_header)
      validate_quantized_header(result.vec_d, result.padded_dim);
    if (result.padded_dim > std::numeric_limits<size_t>::max() / sizeof(float))
      throw std::runtime_error("Invalid checkpoint padded dimension");
    const size_t centroid_bytes =
        input.Bytes(result.centroid_count, result.padded_dim * sizeof(float));
    if (load) {
      result.centroids.reset(new float[centroid_bytes / sizeof(float)]);
      input.ReadBytes(result.centroids.get(), centroid_bytes);
    } else {
      input.Skip(centroid_bytes);
    }
    input.ReadVector(result.cids, result.count, load);
    const size_t code_size = input.Bytes(result.count, result.code_bytes);
    input.ReadVector(result.codes, code_size, load);
    const size_t factor_bytes = input.Bytes(result.count, 4 * sizeof(float));
    input.ReadVector(result.factors, factor_bytes / sizeof(float), load);
  }

  // Whole optional suffix fields may be absent in older files. Partial fields
  // and partial payloads are truncation, not an older format version.
  if (input.Remaining()) {
    const size_t length = input.Read<size_t>();
    input.Bytes(length);
    if (load) {
      result.vector_path.resize(length);
      input.ReadBytes(result.vector_path.data(), length);
    } else {
      input.Skip(length);
    }
  }
  if (input.Remaining())
    result.completed_layer = input.Read<size_t>();
  if (input.Remaining()) {
    const uint8_t anchor = input.Read<uint8_t>();
    // Preserve the old reader's bottom-anchor fallback for unknown values.
    if (anchor <= static_cast<uint8_t>(WindowAnchor::kTop))
      result.anchor = static_cast<WindowAnchor>(anchor);
  }
  while (input.Remaining()) {
    const uint32_t tag = input.Read<uint32_t>();
    if (tag == kIdOnlyMetadataTag) {
      const uint64_t first = input.Read<uint64_t>();
      if (first > result.wp)
        throw std::runtime_error("invalid ID-only edge metadata extension");
      result.id_only_start_layer = static_cast<size_t>(first);
    } else if (tag == kLeftStreamMetadataTag) {
      const uint64_t maximum = input.Read<uint64_t>();
      const uint64_t batch = input.Read<uint64_t>();
      const uint8_t complete = input.Read<uint8_t>();
      if (maximum > result.wp || !batch || batch > std::numeric_limits<size_t>::max() ||
          complete > 1)
        throw std::runtime_error("invalid left_stream metadata extension");
      result.left_stream = true;
      result.left_max = static_cast<size_t>(maximum);
      result.left_batch = static_cast<size_t>(batch);
      result.left_complete = complete != 0;
    } else {
      throw std::runtime_error("unknown edge metadata extension");
    }
  }
  if (result.left_complete && result.id_only_start_layer != 0)
    throw std::runtime_error("completed left_stream metadata must mark ID-only resident layers");
  return result;
}

} // namespace alliance::storage
