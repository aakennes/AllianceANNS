#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace alliance {
// Profiling only: submitted requests whose completions have not been reaped.
// These counts include completed-but-unpolled requests; they are NOT hardware
// queue depth. The observer must outlive all trackers attached to it.
struct IOQueueObserver {
  std::atomic<size_t> current{0}, graph{0}, vector{0};
  std::atomic<size_t> peak{0}, graph_peak{0}, vector_peak{0};
  static void Max(std::atomic<size_t> &target, size_t value) {
    auto old = target.load(std::memory_order_relaxed);
    while (old < value && !target.compare_exchange_weak(old, value, std::memory_order_relaxed)) {
    }
  }
  void Submit(size_t n, bool is_vector) {
    auto &phase = is_vector ? vector : graph;
    auto &phase_peak = is_vector ? vector_peak : graph_peak;
    Max(phase_peak, phase.fetch_add(n, std::memory_order_relaxed) + n);
    Max(peak, current.fetch_add(n, std::memory_order_relaxed) + n);
  }
  void Complete(size_t n, bool is_vector) {
    (is_vector ? vector : graph).fetch_sub(n, std::memory_order_relaxed);
    current.fetch_sub(n, std::memory_order_relaxed);
  }
  void Reset() {
    if (current.load() || graph.load() || vector.load())
      throw std::logic_error("cannot reset IO observer with outstanding requests");
    peak = 0;
    graph_peak = 0;
    vector_peak = 0;
  }
};

class IOQueueTracker {
public:
  static uint64_t NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }
  void Reset(IOQueueObserver *observer, uint64_t now = NowNs()) {
    if (graph_ || vector_)
      throw std::logic_error("cannot reset pending IO tracker");
    observer_ = observer;
    graph_ns_ = vector_ns_ = 0;
    last_ = now;
  }
  void Submit(size_t n, bool is_vector, uint64_t now = NowNs()) {
    Advance(now);
    (is_vector ? vector_ : graph_) += n;
    if (observer_)
      observer_->Submit(n, is_vector);
  }
  void Complete(size_t n, bool is_vector, uint64_t now = NowNs()) {
    auto &count = is_vector ? vector_ : graph_;
    if (n > count)
      throw std::logic_error("IO profiling completion underflow");
    Advance(now);
    count -= n;
    if (observer_)
      observer_->Complete(n, is_vector);
  }
  uint64_t GraphRequestNs() const {
    return graph_ns_;
  }
  uint64_t VectorRequestNs() const {
    return vector_ns_;
  }
  size_t Pending() const {
    return graph_ + vector_;
  }

private:
  void Advance(uint64_t now) {
    if (now < last_)
      throw std::logic_error("non-monotonic IO profiling clock");
    graph_ns_ += graph_ * (now - last_);
    vector_ns_ += vector_ * (now - last_);
    last_ = now;
  }
  IOQueueObserver *observer_{nullptr};
  size_t graph_{0}, vector_{0};
  uint64_t graph_ns_{0}, vector_ns_{0}, last_{0};
};
} // namespace alliance
