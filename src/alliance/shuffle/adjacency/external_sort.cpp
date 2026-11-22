#include "shuffle/adjacency/external_sort.hpp"
#include <algorithm>
#include <queue>
#include <stdexcept>

namespace alliance::shuffle_detail {
IncomingSorter::IncomingSorter(const std::filesystem::path &directory, ScratchIo &io,
                               size_t run_bytes, size_t fan_in)
    : directory_(directory), io_(io), capacity_(run_bytes / sizeof(uint64_t)), fan_in_(fan_in) {
  if (!capacity_ || fan_in_ < 2 || fan_in_ > 32)
    throw std::invalid_argument("external sort bounds");
  buffer_.reserve(capacity_);
}
std::unique_ptr<ScratchFile> IncomingSorter::NewRun() {
  return std::make_unique<ScratchFile>(directory_ / ("run-" + std::to_string(serial_++)), io_);
}
void IncomingSorter::Add(uint64_t key) {
  buffer_.push_back(key);
  if (buffer_.size() == capacity_)
    Spill();
}
void IncomingSorter::Spill() {
  if (buffer_.empty())
    return;
  std::sort(buffer_.begin(), buffer_.end());
  auto file = NewRun();
  file->Write(0, buffer_.data(), buffer_.size() * sizeof(uint64_t));
  file->Seal(); // Closed runs do not consume one descriptor each.
  runs_.push_back(std::move(file));
  buffer_.clear();
}
void IncomingSorter::Merge(size_t begin, size_t end, const std::function<void(uint64_t)> &consume) {
  struct Reader {
    ScratchFile *file;
    uint64_t offset{}, total;
    size_t pos{}, valid{};
    std::vector<uint64_t> words = std::vector<uint64_t>(8192); // 64 KiB per active run.
    bool Next(uint64_t &key) {
      if (pos == valid) {
        if (offset == total)
          return false;
        valid = std::min<uint64_t>(words.size(), (total - offset) / sizeof(uint64_t));
        file->Read(offset, words.data(), valid * sizeof(uint64_t));
        offset += valid * sizeof(uint64_t);
        pos = 0;
      }
      key = words[pos++];
      return true;
    }
  };
  using Item = std::pair<uint64_t, size_t>;
  std::priority_queue<Item, std::vector<Item>, std::greater<Item>> heap;
  std::vector<Reader> readers;
  readers.reserve(end - begin);
  for (size_t i = begin; i < end; ++i) {
    readers.push_back(Reader{runs_[i].get(), 0, runs_[i]->Size()});
    uint64_t key;
    if (readers.back().Next(key))
      heap.push({key, readers.size() - 1});
  }
  while (!heap.empty()) {
    const auto [key, index] = heap.top();
    heap.pop();
    consume(key);
    uint64_t next;
    if (readers[index].Next(next))
      heap.push({next, index});
  }
  // All inputs have been consumed successfully before their deletion.
  for (size_t i = begin; i < end; ++i)
    runs_[i].reset();
}
void IncomingSorter::Finish(const std::function<void(uint64_t)> &consume) {
  Spill();
  std::vector<uint64_t>().swap(buffer_);
  while (runs_.size() > fan_in_) {
    std::vector<std::unique_ptr<ScratchFile>> next;
    for (size_t first = 0; first < runs_.size(); first += fan_in_) {
      auto file = NewRun();
      ScratchWriter writer(*file);
      Merge(first, std::min(first + fan_in_, runs_.size()),
            [&](uint64_t key) { writer.Append(&key, sizeof(key)); });
      writer.Finish();
      next.push_back(std::move(file));
    }
    runs_ = std::move(next);
  }
  Merge(0, runs_.size(), consume);
  runs_.clear();
}
} // namespace alliance::shuffle_detail
