#pragma once
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>
#include <sys/stat.h>

namespace alliance::shuffle_detail {
// A manifest-last protocol for fixed-name outputs. Readers/workflows must require
// the completion receipt; a crash before it leaves explicitly incomplete files.
class ShuffleArtifacts {
public:
  ShuffleArtifacts(const std::string &input_graph, const std::string &input_meta,
                   const std::string &output_graph, const std::string &output_meta);
  ~ShuffleArtifacts();
  ShuffleArtifacts(const ShuffleArtifacts &) = delete;
  ShuffleArtifacts &operator=(const ShuffleArtifacts &) = delete;
  std::string InputGraph() const;
  std::string InputMetadata() const;
  std::string Graph() const;
  std::string Layout() const;
  std::string Metadata() const;
  void Fail(const std::string &reason) noexcept;
  void Commit(size_t nodes, size_t dimensions, size_t levels, uint64_t version,
              size_t neighbor_page, size_t vector_page);

private:
  struct File {
    std::filesystem::path temporary, final;
    struct stat identity{};
    bool published{false};
  };
  std::filesystem::path directory_, staging_, receipt_;
  std::vector<File> files_;
  std::vector<std::pair<std::filesystem::path, struct stat>> inputs_;
  int lock_{-1}, source_lock_{-1};
  bool complete_{false};
  void Cleanup() noexcept;
};
} // namespace alliance::shuffle_detail
