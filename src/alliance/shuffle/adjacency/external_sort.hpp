#pragma once
#include "shuffle/adjacency/scratch_io.hpp"
#include <functional>

namespace alliance::shuffle_detail {
// Sort (destination processing rank, source ID) keys with bounded runs and fan-in.
// The final merge streams directly into incoming adjacency, without an E-sized array.
class IncomingSorter {
public:
  IncomingSorter(const std::filesystem::path &directory, ScratchIo &io, size_t run_bytes = 32 << 20,
                 size_t fan_in = 32);
  void Add(uint64_t key);
  void Finish(const std::function<void(uint64_t)> &consume);

private:
  std::filesystem::path directory_;
  ScratchIo &io_;
  size_t capacity_, fan_in_, serial_{};
  std::vector<uint64_t> buffer_;
  std::vector<std::unique_ptr<ScratchFile>> runs_;
  std::unique_ptr<ScratchFile> NewRun();
  void Spill();
  void Merge(size_t begin, size_t end, const std::function<void(uint64_t)> &consume);
};
} // namespace alliance::shuffle_detail
