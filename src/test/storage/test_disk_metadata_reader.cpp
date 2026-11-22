#include "index/index_disk.hpp"
#include "storage/disk_metadata.hpp"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <unistd.h>

namespace fs = std::filesystem;
using Index = alliance::AllianceIndexDisk<int, float>;
using alliance::storage::MetadataReadMode;
using alliance::storage::ReadDiskMetadata;

static void Check(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}

struct Scratch {
  fs::path path;
  Scratch() {
    const char *base = std::getenv("TEST_TMPDIR");
    std::string name =
        ((base ? fs::path(base) : fs::current_path()) / "metadata-reader-XXXXXX").string();
    std::vector<char> buffer(name.begin(), name.end());
    buffer.push_back(0);
    Check(mkdtemp(buffer.data()) != nullptr, "mkdtemp failed");
    path = buffer.data();
  }
  ~Scratch() {
    std::error_code error;
    fs::remove_all(path, error);
  }
};

template <typename T> static void Append(std::vector<char> &bytes, const T &value) {
  const char *data = reinterpret_cast<const char *>(&value);
  bytes.insert(bytes.end(), data, data + sizeof(T));
}

template <typename T> static void Replace(std::vector<char> &bytes, size_t offset, T value) {
  Check(offset + sizeof(T) <= bytes.size(), "invalid fixture patch");
  std::memcpy(bytes.data() + offset, &value, sizeof(T));
}

static void Write(const fs::path &path, const std::vector<char> &bytes) {
  std::ofstream file(path, std::ios::binary);
  file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  Check(bool(file), "fixture write failed");
}

static std::vector<char> Read(const fs::path &path) {
  std::ifstream file(path, std::ios::binary);
  Check(bool(file), "fixture open failed");
  return std::vector<char>(std::istreambuf_iterator<char>(file), {});
}

struct Fixture {
  std::vector<char> bytes;
  std::set<size_t> valid_ends;
  size_t required_end{}, quantized_offset{}, padded_offset{}, code_offset{};
  size_t anchor_offset{}, id_extension{}, left_extension{};
  size_t padded{};
};

// Independent native field serialization, including a nonidentity ID/rank map.
// Do not call the production writer: this is also a format compatibility check.
static Fixture MakeFixture(bool quantized) {
  Fixture fixture;
  auto &bytes = fixture.bytes;
  for (size_t value : {size_t(5), size_t(64), size_t(2), size_t(4), size_t(2), size_t(32),
                       size_t(3), size_t(2), size_t(316), size_t(8192)})
    Append(bytes, value);
  Append(bytes, size_t(3));
  for (size_t value : {size_t(2), size_t(4), size_t(8)})
    Append(bytes, value);
  for (alliance::label_t value : {2, 0, 1})
    Append(bytes, value);
  for (int value : {30, 10, 20})
    Append(bytes, value);
  for (alliance::tableint value : {1, 2, 0})
    Append(bytes, value);
  for (alliance::tableint value : {2, 0, 1})
    Append(bytes, value);
  fixture.quantized_offset = bytes.size();
  Append(bytes, uint8_t(quantized));
  if (quantized) {
    std::unique_ptr<rabitqlib::Rotator<float>> rotator(rabitqlib::choose_rotator<float>(64));
    fixture.padded = rotator->size();
    fixture.padded_offset = bytes.size();
    Append(bytes, fixture.padded);
    fixture.code_offset = bytes.size();
    Append(bytes, fixture.padded * 3 / 8);
    Append(bytes, size_t(2));
    for (size_t i = 0; i < fixture.padded * 2; ++i)
      Append(bytes, float(i) / 16);
    for (uint8_t value : {0, 1, 0})
      Append(bytes, value);
    for (size_t i = 0; i < 3 * fixture.padded * 3 / 8; ++i)
      Append(bytes, uint8_t(i));
    for (size_t i = 0; i < 12; ++i)
      Append(bytes, float(i) / 8);
  }
  fixture.required_end = bytes.size();
  fixture.valid_ends.insert(bytes.size());
  const std::string vector_path = "metadata-reader-missing-vectors";
  Append(bytes, vector_path.size());
  bytes.insert(bytes.end(), vector_path.begin(), vector_path.end());
  fixture.valid_ends.insert(bytes.size());
  Append(bytes, size_t(2));
  fixture.valid_ends.insert(bytes.size());
  fixture.anchor_offset = bytes.size();
  Append(bytes, uint8_t(1));
  fixture.valid_ends.insert(bytes.size());
  fixture.id_extension = bytes.size();
  Append(bytes, uint32_t(0x31444945u));
  Append(bytes, uint64_t(0));
  fixture.valid_ends.insert(bytes.size());
  fixture.left_extension = bytes.size();
  Append(bytes, uint32_t(0x3157534cu));
  Append(bytes, uint64_t(2));
  Append(bytes, uint64_t(256));
  Append(bytes, uint8_t(1));
  fixture.valid_ends.insert(bytes.size());
  return fixture;
}

static void ExpectFailure(const fs::path &path, const std::vector<char> &bytes,
                          const char *text = nullptr) {
  Write(path, bytes);
  for (auto mode : {MetadataReadMode::Load, MetadataReadMode::Inspect}) {
    bool failed = false;
    try {
      ReadDiskMetadata<int>(path.string(), mode);
    } catch (const std::runtime_error &error) {
      failed = true;
      if (text)
        Check(std::string(error.what()).find(text) != std::string::npos,
              "unexpected failure reason");
    }
    Check(failed, "invalid metadata was accepted");
  }
}

static void VerifyFixture(const fs::path &root, bool quantized) {
  const auto fixture = MakeFixture(quantized);
  const auto path = root / "metadata";
  Write(path, fixture.bytes);
  const auto loaded = ReadDiskMetadata<int>(path.string(), MetadataReadMode::Load);
  const auto inspected = ReadDiskMetadata<int>(path.string(), MetadataReadMode::Inspect);
  Check(loaded.count == 3 && loaded.vec_d == 64 && loaded.windows == std::vector<size_t>({2, 4, 8}),
        "geometry changed");
  Check(loaded.labels == std::vector<alliance::label_t>({2, 0, 1}) &&
            loaded.sorted_ids == std::vector<alliance::tableint>({1, 2, 0}) &&
            loaded.position_by_id == std::vector<alliance::tableint>({2, 0, 1}),
        "ID/rank order changed");
  Check(inspected.windows == loaded.windows && inspected.count == loaded.count &&
            inspected.completed_layer == loaded.completed_layer &&
            inspected.anchor == loaded.anchor && inspected.left_complete &&
            inspected.left_batch == 256 && inspected.id_only_start_layer == 0,
        "inspect geometry/extensions differ from load");
  Check(inspected.labels.empty() && inspected.attributes.empty() && inspected.sorted_ids.empty() &&
            inspected.position_by_id.empty() && !inspected.centroids && inspected.cids.empty() &&
            inspected.codes.empty() && inspected.factors.empty() && inspected.vector_path.empty(),
        "inspect allocated a metadata payload");
  if (quantized) {
    Check(loaded.padded_dim == fixture.padded && loaded.code_bytes == fixture.padded * 3 / 8 &&
              loaded.cids == std::vector<uint8_t>({0, 1, 0}) && loaded.factors.size() == 12,
          "RaBitQ shape/CID payload changed");
    for (size_t i = 0; i < loaded.codes.size(); ++i)
      Check(loaded.codes[i] == uint8_t(i), "RaBitQ code order changed");
    for (size_t i = 0; i < loaded.factors.size(); ++i)
      Check(loaded.factors[i] == float(i) / 8, "RaBitQ factor order changed");
    for (size_t i = 0; i < fixture.padded * 2; ++i)
      Check(loaded.centroids[i] == float(i) / 16, "centroid changed");
  }

  // No graph traversal is needed for metadata roundtrip; all data is synthetic.
  const auto graph = root / "graph";
  {
    std::ofstream file(graph, std::ios::binary);
    file.seekp(8192 * 4 - 1);
    file.put(0);
  }
  {
    Index index(graph.string(), path.string(), "l2");
    Check(index.labels_ == loaded.labels && index.position_by_id_ == loaded.position_by_id,
          "index ID state changed");
    index.Save((root / "roundtrip").string());
  }
  Check(Read(root / "roundtrip") == fixture.bytes, "Index Load/Save changed native metadata bytes");

  // Every truncated prefix must fail, except complete legacy suffix boundaries.
  for (size_t end = 0; end < fixture.bytes.size(); ++end) {
    std::vector<char> prefix(fixture.bytes.begin(), fixture.bytes.begin() + end);
    if (!fixture.valid_ends.count(end)) {
      ExpectFailure(path, prefix, "Truncated");
    } else {
      Write(path, prefix);
      for (auto mode : {MetadataReadMode::Load, MetadataReadMode::Inspect}) {
        const auto legacy = ReadDiskMetadata<int>(path.string(), mode);
        if (end == fixture.required_end)
          Check(legacy.completed_layer == 0 && legacy.anchor == alliance::WindowAnchor::kBottom &&
                    legacy.vector_path.empty() && !legacy.left_stream && legacy.left_batch == 256 &&
                    legacy.id_only_start_layer == std::numeric_limits<size_t>::max(),
                "legacy defaults changed");
      }
    }
  }
  auto bytes = fixture.bytes;
  // The complete fixture has enough bytes for five windows, but wp=2 only
  // permits three or four; five must fail geometry before allocating/reading
  // the array, rather than eventually failing on a shifted payload field.
  Replace(bytes, 10 * sizeof(size_t), size_t(5));
  Write(path, bytes);
  bool bad_windows_rejected = false;
  try {
    ReadDiskMetadata<int>(path.string(), MetadataReadMode::Inspect);
  } catch (const std::runtime_error &error) {
    bad_windows_rejected = std::string(error.what()) == "Invalid checkpoint geometry";
  }
  Check(bad_windows_rejected, "inspect did not reject window count before payload parsing");
  bytes = fixture.bytes;
  Replace(bytes, fixture.quantized_offset, uint8_t(2));
  ExpectFailure(path, bytes, "quantization tag");
  bytes = fixture.bytes;
  Replace(bytes, fixture.id_extension, uint32_t(0xdeadbeefu));
  ExpectFailure(path, bytes, "unknown edge metadata extension");
  bytes = fixture.bytes;
  Replace(bytes, fixture.id_extension + 4, uint64_t(3));
  ExpectFailure(path, bytes, "invalid ID-only");
  for (size_t offset : {fixture.left_extension + 4, fixture.left_extension + 12}) {
    bytes = fixture.bytes;
    Replace(bytes, offset, uint64_t(offset == fixture.left_extension + 4 ? 3 : 0));
    ExpectFailure(path, bytes, "invalid left_stream");
  }
  bytes = fixture.bytes;
  Replace(bytes, bytes.size() - 1, uint8_t(2));
  ExpectFailure(path, bytes, "invalid left_stream");
  bytes = fixture.bytes;
  Replace(bytes, fixture.id_extension + 4, uint64_t(1));
  ExpectFailure(path, bytes, "must mark ID-only");
  bytes = fixture.bytes;
  Replace(bytes, fixture.anchor_offset, uint8_t(255));
  Write(path, bytes);
  Check(ReadDiskMetadata<int>(path.string(), MetadataReadMode::Load).anchor ==
            alliance::WindowAnchor::kBottom,
        "unknown anchor fallback changed");
  if (quantized) {
    // Only the quantized header remains, with an inconsistent padded dimension.
    // The callback must reject it before any missing payload triggers truncation.
    bytes = fixture.bytes;
    Replace(bytes, fixture.padded_offset, fixture.padded + 64);
    bytes.resize(fixture.code_offset + 2 * sizeof(size_t));
    Write(path, bytes);
    for (auto mode : {MetadataReadMode::Load, MetadataReadMode::Inspect}) {
      bool called = false, rejected = false;
      try {
        ReadDiskMetadata<int>(path.string(), mode, [&](size_t dim, size_t padded) {
          Check(dim == 64 && padded == fixture.padded + 64, "callback received wrong dimensions");
          called = true;
          throw std::runtime_error("quantized header rejected before payload");
        });
      } catch (const std::runtime_error &error) {
        rejected = std::string(error.what()) == "quantized header rejected before payload";
      }
      Check(called && rejected, "quantized header validation ran after payload access");
    }
    bytes = fixture.bytes;
    Replace(bytes, fixture.padded_offset, std::numeric_limits<size_t>::max());
    ExpectFailure(path, bytes, "padded dimension");
    bytes = fixture.bytes;
    Replace(bytes, fixture.code_offset, std::numeric_limits<size_t>::max());
    ExpectFailure(path, bytes, "Truncated");
  }
}

static void VerifyEmptyIndex(const fs::path &root) {
  const auto graph = (root / "empty_graph").string();
  const auto meta = (root / "empty_meta").string();
  const auto saved = (root / "empty_roundtrip").string();
  {
    Index empty(5, 64, 2, 32, "l2", graph, 4, 2, false, 2, 8192);
    empty.Save(meta);
  }
  {
    Index loaded(graph, meta, "l2");
    Check(loaded.curvec_num_ == 0 && loaded.labels_.empty(), "empty index no longer loads");
    loaded.Save(saved);
  }
  Check(Read(meta) == Read(saved), "empty Index Load/Save changed metadata");
  Check(ReadDiskMetadata<int>(meta, MetadataReadMode::Inspect).count == 0,
        "reader applied planner-only nonempty constraint");
}

int main() {
  try {
    Scratch scratch;
    VerifyFixture(scratch.path, false);
    VerifyFixture(scratch.path, true);
    VerifyEmptyIndex(scratch.path);
    std::cout << "metadata reader: native roundtrip, inspect, legacy and malformed cases passed\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
